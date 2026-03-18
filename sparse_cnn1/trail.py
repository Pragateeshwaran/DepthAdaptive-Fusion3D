import argparse
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_loader import get_LiDAR, get_calib, get_images, get_labels, get_radar
from cross_attention import FusionModule
from evaluate import compute_map
from image_model import ImageFeatureExtractor
from lidar_model import LiDARFeatureExtractor
from radar_model import RadarFeatureExtractor
from rpn_refinement import SingleStageDetector, parse_label_line


os.environ.setdefault("SPCONV_ALGO_MODE", "default")
os.environ.setdefault("SPCONV_DISABLE_CONV_CACHE", "0")


def setup_logger(log_file: str = "training.log"):
    """Open fixed log file in append mode."""
    return open(log_file, "a", encoding="utf-8", buffering=1)


def write_epoch_log(log_handle, line: str) -> None:
    print(line)
    log_handle.write(line + "\n")


def transform_boxes_camera_to_lidar(boxes_camera: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    """Transform V2X-Radar-V camera boxes [x,y,z,h,w,l,ry] to LiDAR frame (KITTI-compatible calib format)."""
    if len(boxes_camera) == 0:
        return boxes_camera

    tr_velo_to_cam = calib.get("Tr_velo_to_cam", np.eye(4, dtype=np.float32))
    r0_rect = calib.get("R0_rect", np.eye(3, dtype=np.float32))

    if tr_velo_to_cam.shape == (12,):
        tr_velo_to_cam = tr_velo_to_cam.reshape(3, 4)
    if tr_velo_to_cam.shape == (3, 4):
        tr_velo_to_cam = np.vstack([tr_velo_to_cam, [0, 0, 0, 1]]).astype(np.float32)

    if r0_rect.shape == (9,):
        r0_rect = r0_rect.reshape(3, 3)
    if r0_rect.shape == (3, 3):
        r0_4x4 = np.eye(4, dtype=np.float32)
        r0_4x4[:3, :3] = r0_rect
    elif r0_rect.shape == (4, 4):
        r0_4x4 = r0_rect.astype(np.float32)
    else:
        r0_4x4 = np.eye(4, dtype=np.float32)

    tr_cam_to_velo = np.linalg.inv(r0_4x4 @ tr_velo_to_cam)
    boxes_lidar = np.zeros_like(boxes_camera, dtype=np.float32)

    for i in range(len(boxes_camera)):
        x_cam, y_cam, z_cam = boxes_camera[i, 0:3]
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
        point_lidar = tr_cam_to_velo @ point_cam

        boxes_lidar[i, 0:3] = point_lidar[:3]
        boxes_lidar[i, 3:6] = boxes_camera[i, 3:6]

        rot_cam = boxes_camera[i, 6]
        rot_lidar = -rot_cam - np.pi / 2.0
        while rot_lidar > np.pi:
            rot_lidar -= 2.0 * np.pi
        while rot_lidar < -np.pi:
            rot_lidar += 2.0 * np.pi
        boxes_lidar[i, 6] = rot_lidar

    return boxes_lidar


def voxelize_lidar_proper(
    points_batch: Sequence[np.ndarray],
    voxel_size=(0.1, 0.1, 0.1),
    point_cloud_range=(-50, -50, -3, 50, 50, 5),
):
    """LiDAR voxelization (kept compatible with previous sparse pipeline)."""
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
    ]

    voxel_features_list = []
    voxel_coords_list = []

    pc_range = np.array([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]], dtype=np.float32)
    voxel_size_arr = np.array(voxel_size, dtype=np.float32)
    spatial_shape_arr = np.array(spatial_shape, dtype=np.int32)

    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        if len(points) == 0:
            dummy_feat = np.zeros((1, 3), dtype=np.float32)
            dummy_coords = np.array(
                [[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32
            )
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_coords = ((points[:, :3] - pc_range) / voxel_size_arr).astype(np.int32)
        valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < spatial_shape_arr), axis=1)

        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]

        if len(points_valid) == 0:
            dummy_feat = np.zeros((1, 3), dtype=np.float32)
            dummy_coords = np.array(
                [[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32
            )
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_hash = (
            voxel_coords[:, 0].astype(np.int64) * spatial_shape[1] * spatial_shape[2]
            + voxel_coords[:, 1].astype(np.int64) * spatial_shape[2]
            + voxel_coords[:, 2].astype(np.int64)
        )

        unique_hash, inverse = np.unique(voxel_hash, return_inverse=True)
        num_voxels = len(unique_hash)

        voxel_features = np.zeros((num_voxels, 3), dtype=np.float32)
        for i in range(3):
            voxel_features[:, i] = np.bincount(inverse, weights=points_valid[:, i], minlength=num_voxels)

        voxel_counts = np.bincount(inverse, minlength=num_voxels).astype(np.float32)
        voxel_features = voxel_features / (voxel_counts[:, np.newaxis] + np.float32(1e-8))

        unique_coords = np.zeros((num_voxels, 3), dtype=np.int32)
        for i, h in enumerate(unique_hash):
            idx = np.where(voxel_hash == h)[0][0]
            unique_coords[i] = voxel_coords[idx]

        batch_indices = np.full((num_voxels, 1), batch_idx, dtype=np.int32)
        voxel_coords_with_batch = np.concatenate([batch_indices, unique_coords], axis=1)

        voxel_features_list.append(torch.from_numpy(voxel_features))
        voxel_coords_list.append(torch.from_numpy(voxel_coords_with_batch))

    voxel_features = torch.cat(voxel_features_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    return voxel_features, voxel_coords, spatial_shape, len(points_batch)


def voxelize_radar_proper(
    points_batch: Sequence[np.ndarray],
    voxel_size=(0.2, 0.2, 0.2),
    point_cloud_range=(-50, -50, -3, 50, 50, 5),
):
    """Radar voxelization (kept compatible with previous sparse pipeline)."""
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
    ]

    voxel_features_list = []
    voxel_coords_list = []

    pc_range = np.array([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]], dtype=np.float32)
    voxel_size_arr = np.array(voxel_size, dtype=np.float32)
    spatial_shape_arr = np.array(spatial_shape, dtype=np.int32)

    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        if len(points) == 0:
            dummy_feat = np.zeros((1, 5), dtype=np.float32)
            dummy_coords = np.array(
                [[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32
            )
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_coords = ((points[:, :3] - pc_range) / voxel_size_arr).astype(np.int32)
        valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < spatial_shape_arr), axis=1)

        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]

        if len(points_valid) == 0:
            dummy_feat = np.zeros((1, 5), dtype=np.float32)
            dummy_coords = np.array(
                [[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32
            )
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_hash = (
            voxel_coords[:, 0].astype(np.int64) * spatial_shape[1] * spatial_shape[2]
            + voxel_coords[:, 1].astype(np.int64) * spatial_shape[2]
            + voxel_coords[:, 2].astype(np.int64)
        )

        unique_hash, inverse = np.unique(voxel_hash, return_inverse=True)
        num_voxels = len(unique_hash)

        voxel_features = np.zeros((num_voxels, 5), dtype=np.float32)
        for i in range(min(4, points_valid.shape[1])):
            voxel_features[:, i] = np.bincount(inverse, weights=points_valid[:, i], minlength=num_voxels)

        if points_valid.shape[1] >= 5:
            for i in range(num_voxels):
                mask = inverse == i
                if np.any(mask):
                    voxel_features[i, 4] = points_valid[mask, 4].max()

        voxel_counts = np.bincount(inverse, minlength=num_voxels).astype(np.float32)
        voxel_features[:, :4] = voxel_features[:, :4] / (voxel_counts[:, np.newaxis] + np.float32(1e-8))

        unique_coords = np.zeros((num_voxels, 3), dtype=np.int32)
        for i, h in enumerate(unique_hash):
            idx = np.where(voxel_hash == h)[0][0]
            unique_coords[i] = voxel_coords[idx]

        batch_indices = np.full((num_voxels, 1), batch_idx, dtype=np.int32)
        voxel_coords_with_batch = np.concatenate([batch_indices, unique_coords], axis=1)

        voxel_features_list.append(torch.from_numpy(voxel_features))
        voxel_coords_list.append(torch.from_numpy(voxel_coords_with_batch))

    voxel_features = torch.cat(voxel_features_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    return voxel_features, voxel_coords, spatial_shape, len(points_batch)


voxelize_lidar = voxelize_lidar_proper
voxelize_radar = voxelize_radar_proper


def prepare_images(image_batch: Sequence[np.ndarray]) -> torch.Tensor:
    images_tensor = []
    for img in image_batch:
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_tensor = img.float()
        images_tensor.append(img_tensor)
    return torch.stack(images_tensor, dim=0)


def parse_labels_to_targets(labels_batch, calib_batch=None):
    class_map = {"pedestrian": 0, "cyclist": 1, "car": 2}
    targets = []

    for idx, labels in enumerate(labels_batch):
        boxes_list = []
        labels_list = []
        scores_list = []

        for line in labels:
            if isinstance(line, str) and line.strip():
                parsed = parse_label_line(line)
                obj_type = parsed["type"].lower()
                if obj_type in class_map:
                    box_3d = parsed["location"] + parsed["dimensions"] + [parsed["rotation_y"]]
                    boxes_list.append(box_3d)
                    labels_list.append(class_map[obj_type])
                    scores_list.append(parsed["score"])

        if len(boxes_list) > 0:
            boxes_camera = np.array(boxes_list, dtype=np.float32)
            if calib_batch is not None and idx < len(calib_batch):
                boxes_lidar = transform_boxes_camera_to_lidar(boxes_camera, calib_batch[idx])
            else:
                boxes_lidar = boxes_camera

            boxes_lidar[:, 2] = boxes_lidar[:, 2] + 0.5 * boxes_lidar[:, 3]
            boxes_tensor = torch.tensor(boxes_lidar, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)
            scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 7), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
            scores_tensor = torch.zeros((0,), dtype=torch.float32)

        targets.append(
            {
                "boxes_3d": boxes_tensor,
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "scores": scores_tensor,
            }
        )

    return targets


class V2XRadarDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "training", num_samples: Optional[int] = None):
        self.root_dir = root_dir
        self.split = split

        self.lidar_data = get_LiDAR(root_dir, split, from_idx=0, count=num_samples)
        self.radar_data = get_radar(root_dir, split, from_idx=0, count=num_samples)
        self.image_data = get_images(root_dir, split, from_idx=0, count=num_samples)
        self.labels = get_labels(root_dir, split, from_idx=0, count=num_samples)
        self.calibs = get_calib(root_dir, split, from_idx=0, count=num_samples)

        min_len = min(len(self.lidar_data), len(self.radar_data), len(self.image_data), len(self.labels), len(self.calibs))
        self.lidar_data = self.lidar_data[:min_len]
        self.radar_data = self.radar_data[:min_len]
        self.image_data = self.image_data[:min_len]
        self.labels = self.labels[:min_len]
        self.calibs = self.calibs[:min_len]

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        return {
            "lidar": self.lidar_data[idx],
            "radar": self.radar_data[idx],
            "image": self.image_data[idx],
            "label": self.labels[idx],
            "calib": self.calibs[idx],
            "idx": idx,
        }


def collate_fn(batch):
    return {
        "lidar": [item["lidar"] for item in batch],
        "radar": [item["radar"] for item in batch],
        "image": [item["image"] for item in batch],
        "label": [item["label"] for item in batch],
        "calib": [item["calib"] for item in batch],
        "idx": [item["idx"] for item in batch],
    }


class MultiModalDetectionNetwork(nn.Module):
    def __init__(
        self,
        lidar_dim: int = 128,
        radar_dim: int = 128,
        image_dim: int = 128,
        active_modalities: List[str] = ["lidar", "radar", "image"],
        use_density_threshold: bool = False,
    ):
        super().__init__()

        self.lidar_net = LiDARFeatureExtractor(in_channels=3, feature_dim=lidar_dim)
        self.radar_net = RadarFeatureExtractor(in_channels=5, feature_dim=radar_dim)
        self.image_net = ImageFeatureExtractor(in_channels=3, feature_dim=image_dim)

        self.fusion_module = FusionModule(feature_dim=128, num_heads=8, dropout=0.1)
        self.detector = SingleStageDetector(
            backbone_channels=128,
            num_classes=3,
            num_anchors_per_location=12,
            pos_iou_thresh=0.25,
            neg_iou_thresh=0.10,
        )

        self.active_modalities = set(active_modalities)
        self.use_density_threshold = use_density_threshold

    @staticmethod
    def _zero_sparse_features(sparse_tensor: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return sparse_tensor.replace_feature(torch.zeros_like(sparse_tensor.features))

    def forward(self, lidar_sparse, radar_sparse, images, targets=None, training=True, original_lidar_points=None):
        if self.use_density_threshold and original_lidar_points is not None:
            points_for_density = original_lidar_points
        elif self.use_density_threshold:
            points_for_density = lidar_sparse.indices[:, 1:].float()
        else:
            points_for_density = None

        lidar_feat_sparse, lidar_conf_sparse = self.lidar_net(lidar_sparse)
        radar_feat_sparse, radar_conf_sparse = self.radar_net(radar_sparse)
        image_feat, image_conf = self.image_net(images)

        if "lidar" not in self.active_modalities:
            lidar_feat_sparse = self._zero_sparse_features(lidar_feat_sparse)
        if "radar" not in self.active_modalities:
            radar_feat_sparse = self._zero_sparse_features(radar_feat_sparse)
        if "image" not in self.active_modalities:
            image_feat = torch.zeros_like(image_feat)

        fused_features, depth_threshold = self.fusion_module(
            lidar_feat_sparse,
            radar_feat_sparse,
            image_feat,
            lidar_conf_sparse,
            radar_conf_sparse,
            image_conf,
            original_points=points_for_density,
        )

        detection_outputs = self.detector(fused_features, targets=targets, training=training)

        return {
            "lidar_features": lidar_feat_sparse,
            "lidar_confidence": lidar_conf_sparse,
            "radar_features": radar_feat_sparse,
            "radar_confidence": radar_conf_sparse,
            "image_features": image_feat,
            "image_confidence": image_conf,
            "fused_features": fused_features,
            "depth_threshold": depth_threshold,
            "detections": detection_outputs,
        }


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        detections = predictions["detections"]
        device = predictions["fused_features"].device

        if "losses" not in detections:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "rpn_cls_loss": 0.0,
                "rpn_reg_loss": 0.0,
                "rpn_obj_loss": 0.0,
                "rpn_multi_cls_loss": 0.0,
                "num_pos_anchors": 0.0,
                "total_loss": 0.0,
            }

        losses = detections["losses"]
        obj_loss = losses.get("rpn_cls_loss", torch.tensor(0.0, device=device))
        multi_cls_loss = losses.get("rpn_multi_cls_loss", torch.tensor(0.0, device=device))
        reg_loss = losses.get("rpn_reg_loss", torch.tensor(0.0, device=device))

        cls_loss = obj_loss + multi_cls_loss
        total_loss = 2.0 * cls_loss + 5.0 * reg_loss

        return total_loss, {
            "rpn_cls_loss": float(cls_loss.detach().item()),
            "rpn_reg_loss": float(reg_loss.detach().item()),
            "rpn_obj_loss": float(obj_loss.detach().item()),
            "rpn_multi_cls_loss": float(multi_cls_loss.detach().item()),
            "num_pos_anchors": losses.get("num_pos_anchors", 0.0),
            "total_loss": float(total_loss.detach().item()),
        }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    det_criterion,
    device,
    epoch,
    root_dir=None,
):
    model.train()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    num_batches = 0

    epoch_predictions = []
    epoch_targets = []

    start_t = time.time()

    for batch in dataloader:
        lidar_feat, lidar_coords, lidar_shape, lidar_bs = voxelize_lidar_proper(batch["lidar"])
        radar_feat, radar_coords, radar_shape, radar_bs = voxelize_radar_proper(batch["radar"])

        lidar_sparse = spconv.SparseConvTensor(
            features=lidar_feat.to(device, non_blocking=True),
            indices=lidar_coords.to(device, non_blocking=True).int(),
            spatial_shape=lidar_shape,
            batch_size=lidar_bs,
        )
        radar_sparse = spconv.SparseConvTensor(
            features=radar_feat.to(device, non_blocking=True),
            indices=radar_coords.to(device, non_blocking=True).int(),
            spatial_shape=radar_shape,
            batch_size=radar_bs,
        )

        images = prepare_images(batch["image"]).to(device, non_blocking=True)
        targets = parse_labels_to_targets(batch["label"], batch.get("calib", None))

        for t in targets:
            t["boxes_3d"] = t["boxes_3d"].to(device, non_blocking=True)
            t["boxes"] = t["boxes"].to(device, non_blocking=True)
            t["labels"] = t["labels"].to(device, non_blocking=True)
            t["scores"] = t["scores"].to(device, non_blocking=True)

        original_points_concat = None
        if getattr(model, "use_density_threshold", False):
            original_points = [torch.from_numpy(lidar).float()[:, :3] for lidar in batch["lidar"] if len(lidar) > 0]
            if len(original_points) == 0:
                original_points_concat = torch.zeros((1, 3), dtype=torch.float32, device=device)
            else:
                original_points_concat = torch.cat(original_points, dim=0).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(
                lidar_sparse,
                radar_sparse,
                images,
                targets=targets,
                training=True,
                original_lidar_points=original_points_concat,
            )
            loss, loss_dict = det_criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_cls += float(loss_dict["rpn_cls_loss"])
        total_reg += float(loss_dict["rpn_reg_loss"])
        num_batches += 1

        detections = outputs["detections"]
        batch_proposals = detections.get("proposals", [])
        batch_scores = detections.get("scores", [])
        batch_labels = detections.get("labels", [])

        for i in range(len(batch_proposals)):
            pred_boxes = batch_proposals[i].detach().cpu()
            pred_scores = batch_scores[i].detach().cpu() if i < len(batch_scores) else torch.zeros((0,), dtype=torch.float32)
            pred_labels = batch_labels[i].detach().cpu() if i < len(batch_labels) else torch.zeros((0,), dtype=torch.long)

            epoch_predictions.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
            epoch_targets.append(
                {
                    "boxes": targets[i]["boxes"].detach().cpu(),
                    "labels": targets[i]["labels"].detach().cpu(),
                }
            )

    if num_batches == 0:
        elapsed = time.time() - start_t
        return 0.0, {"rpn_cls_loss": 0.0, "rpn_reg_loss": 0.0, "mAP@0.5": 0.0, "time_s": elapsed}

    map_stats = compute_map(epoch_predictions, epoch_targets, iou_threshold=0.5)
    elapsed = time.time() - start_t

    return total_loss / num_batches, {
        "rpn_cls_loss": total_cls / num_batches,
        "rpn_reg_loss": total_reg / num_batches,
        "mAP@0.5": float(map_stats.get("mAP", 0.0)),
        "time_s": elapsed,
    }


def _parse_modalities(modalities: str) -> List[str]:
    return [m.strip().lower() for m in modalities.split(",") if m.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train V2X-Radar-V multimodal single-stage detector.")
    parser.add_argument("--root-dir", type=str, default=r"F:\Work\DeepLearning\Research\V2X-Radar-V")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--active-modalities", type=str, default="lidar,radar,image")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--use-density-threshold", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    active_modalities = _parse_modalities(args.active_modalities)

    dataset = V2XRadarDataset(args.root_dir, split=args.split, num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = MultiModalDetectionNetwork(
        active_modalities=active_modalities,
        use_density_threshold=args.use_density_threshold,
    ).to(device)
    det_criterion = DetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with setup_logger("training.log") as log_handle:
        for epoch in range(1, args.epochs + 1):
            epoch_loss, stats = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                det_criterion=det_criterion,
                device=device,
                epoch=epoch,
                root_dir=args.root_dir,
            )

            line = (
                f"Epoch {epoch} | Time {stats['time_s']:.2f}s | mAP@0.5 {stats['mAP@0.5']:.4f} | "
                f"Loss {epoch_loss:.4f} | CLS {stats['rpn_cls_loss']:.4f} | REG {stats['rpn_reg_loss']:.4f}"
            )
            write_epoch_log(log_handle, line)

            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "loss_stats": stats,
                    "active_modalities": active_modalities,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()

