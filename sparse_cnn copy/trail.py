import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cross_attention import FusionModule
from evaluate import compute_map
from image_model import ImageFeatureExtractor
from lidar_model import LiDARFeatureExtractor
from radar_model import RadarFeatureExtractor
from rpn_refinement import SingleStageDetector, parse_label_line

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_LiDAR, get_calib, get_images, get_labels, get_radar


# ============ LOGGING SETUP ============
def setup_logger(log_path='training.log'):
    return os.path.abspath(log_path)


LOG_FILE = setup_logger()


def log_epoch_line(line, log_path=LOG_FILE):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)


# ============ CRITICAL FIX #2: Coordinate Transformation ============
def transform_boxes_camera_to_lidar(boxes_camera, calib):
    """Transform 3D boxes from camera to LiDAR coordinates."""
    if len(boxes_camera) == 0:
        return boxes_camera

    Tr_velo_to_cam = calib.get('Tr_velo_to_cam', np.eye(4))
    R0_rect = calib.get('R0_rect', np.eye(3))

    if Tr_velo_to_cam.shape == (12,):
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    elif Tr_velo_to_cam.shape == (3, 4):
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])

    if R0_rect.shape == (9,):
        R0_rect = R0_rect.reshape(3, 3)
    if R0_rect.shape == (3, 3):
        R0_4x4 = np.eye(4, dtype=np.float32)
        R0_4x4[:3, :3] = R0_rect
    elif R0_rect.shape == (4, 4):
        R0_4x4 = R0_rect
    else:
        R0_4x4 = np.eye(4, dtype=np.float32)

    Tr_cam_to_velo = np.linalg.inv(R0_4x4 @ Tr_velo_to_cam)
    boxes_lidar = np.zeros_like(boxes_camera, dtype=np.float32)

    for i in range(len(boxes_camera)):
        x_cam, y_cam, z_cam = boxes_camera[i, 0:3]
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_lidar = Tr_cam_to_velo @ point_cam
        boxes_lidar[i, 0:3] = point_lidar[:3]
        boxes_lidar[i, 3:6] = boxes_camera[i, 3:6]

        rot_cam = boxes_camera[i, 6]
        rot_lidar = -rot_cam - np.pi / 2
        while rot_lidar > np.pi:
            rot_lidar -= 2 * np.pi
        while rot_lidar < -np.pi:
            rot_lidar += 2 * np.pi
        boxes_lidar[i, 6] = rot_lidar

    return boxes_lidar


# ============ ROOT CAUSE FIX: Force spconv to use specific algorithm ============
os.environ['SPCONV_ALGO_MODE'] = 'default'
os.environ['SPCONV_DISABLE_CONV_CACHE'] = '0'
cudnn.benchmark = False
cudnn.deterministic = True


# ============ Checkpoint Management ============
def find_latest_checkpoint(checkpoint_dir='.'):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None

    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_epoch_')[1].split('.pth')[0])
            epochs.append((epoch, f))
        except Exception:
            continue

    if not epochs:
        return None

    return max(epochs, key=lambda x: x[0])


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1


# ============ PROPER VOXELIZATION - NO DOWNSAMPLING, NO SKIPPING ============
def voxelize_lidar_proper(points_batch,
                          voxel_size=(0.1, 0.1, 0.1),
                          point_cloud_range=(-50, -50, -3, 50, 50, 5)):
    """
    PROPER voxelization - NO downsampling, NO limits.
    Uses efficient numpy operations to handle all points.
    """
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
    ]

    voxel_features_list = []
    voxel_coords_list = []

    pc_range = np.array([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]])
    voxel_size_arr = np.array([voxel_size[0], voxel_size[1], voxel_size[2]])
    spatial_shape_arr = np.array(spatial_shape)

    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        if len(points) == 0:
            dummy_feat = np.zeros((1, 3), dtype=np.float32)
            dummy_coords = np.array([[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32)
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_coords = ((points[:, :3] - pc_range) / voxel_size_arr).astype(np.int32)
        valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < spatial_shape_arr), axis=1)

        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]

        if len(points_valid) == 0:
            dummy_feat = np.zeros((1, 3), dtype=np.float32)
            dummy_coords = np.array([[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32)
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_hash = (voxel_coords[:, 0].astype(np.int64) * spatial_shape[1] * spatial_shape[2] +
                      voxel_coords[:, 1].astype(np.int64) * spatial_shape[2] +
                      voxel_coords[:, 2].astype(np.int64))

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


def voxelize_radar_proper(points_batch,
                          voxel_size=(0.2, 0.2, 0.2),
                          point_cloud_range=(-50, -50, -3, 50, 50, 5)):
    """PROPER radar voxelization - NO downsampling, NO limits."""
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
    ]

    voxel_features_list = []
    voxel_coords_list = []

    pc_range = np.array([point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]])
    voxel_size_arr = np.array([voxel_size[0], voxel_size[1], voxel_size[2]])
    spatial_shape_arr = np.array(spatial_shape)

    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        if len(points) == 0:
            dummy_feat = np.zeros((1, 5), dtype=np.float32)
            dummy_coords = np.array([[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32)
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_coords = ((points[:, :3] - pc_range) / voxel_size_arr).astype(np.int32)
        valid_mask = np.all((voxel_coords >= 0) & (voxel_coords < spatial_shape_arr), axis=1)

        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]

        if len(points_valid) == 0:
            dummy_feat = np.zeros((1, 5), dtype=np.float32)
            dummy_coords = np.array([[batch_idx, spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2]], dtype=np.int32)
            voxel_features_list.append(torch.from_numpy(dummy_feat))
            voxel_coords_list.append(torch.from_numpy(dummy_coords))
            continue

        voxel_hash = (voxel_coords[:, 0].astype(np.int64) * spatial_shape[1] * spatial_shape[2] +
                      voxel_coords[:, 1].astype(np.int64) * spatial_shape[2] +
                      voxel_coords[:, 2].astype(np.int64))

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


# ============ Function Aliases for Backward Compatibility ============
voxelize_lidar = voxelize_lidar_proper
voxelize_radar = voxelize_radar_proper


def prepare_images(image_batch):
    images_tensor = []
    for img in image_batch:
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_tensor = img
        images_tensor.append(img_tensor)
    return torch.stack(images_tensor)


def parse_labels_to_targets(labels_batch, calib_batch=None):
    targets = []
    class_map = {'pedestrian': 0, 'cyclist': 1, 'car': 2}

    for idx, labels in enumerate(labels_batch):
        boxes_list, labels_list, scores_list = [], [], []

        for line in labels:
            if isinstance(line, str) and line.strip():
                parsed = parse_label_line(line)
                obj_type = parsed['type'].lower()
                if obj_type in class_map:
                    box_3d = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
                    boxes_list.append(box_3d)
                    labels_list.append(class_map[obj_type])
                    scores_list.append(parsed['score'])

        if len(boxes_list) > 0:
            boxes_camera = np.array(boxes_list)
            if calib_batch and idx < len(calib_batch):
                boxes_lidar = transform_boxes_camera_to_lidar(boxes_camera, calib_batch[idx])
            else:
                boxes_lidar = boxes_camera

            boxes_lidar[:, 2] = boxes_lidar[:, 2] + 0.5 * boxes_lidar[:, 3]

            targets.append({
                'boxes_3d': torch.tensor(boxes_lidar, dtype=torch.float32),
                'labels': torch.tensor(labels_list, dtype=torch.long),
                'scores': torch.tensor(scores_list, dtype=torch.float32)
            })
        else:
            targets.append({
                'boxes_3d': torch.zeros((0, 7), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long),
                'scores': torch.zeros((0,), dtype=torch.float32)
            })

    return targets


# ============ Dataset ============
class V2XRadarDataset(Dataset):
    def __init__(self, root_dir, split='training', num_samples=None):
        self.root_dir = root_dir
        self.split = split

        self.lidar_data = get_LiDAR(root_dir, split, from_idx=0, count=num_samples)
        self.radar_data = get_radar(root_dir, split, from_idx=0, count=num_samples)
        self.image_data = get_images(root_dir, split, from_idx=0, count=num_samples)
        self.labels = get_labels(root_dir, split, from_idx=0, count=num_samples)
        self.calibs = get_calib(root_dir, split, from_idx=0, count=num_samples)

        min_len = min(len(self.lidar_data), len(self.radar_data), len(self.image_data))
        self.lidar_data = self.lidar_data[:min_len]
        self.radar_data = self.radar_data[:min_len]
        self.image_data = self.image_data[:min_len]
        self.labels = self.labels[:min_len]
        self.calibs = self.calibs[:min_len]

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        return {
            'lidar': self.lidar_data[idx],
            'radar': self.radar_data[idx],
            'image': self.image_data[idx],
            'label': self.labels[idx],
            'calib': self.calibs[idx],
            'idx': idx
        }


def collate_fn(batch):
    return {
        'lidar': [item['lidar'] for item in batch],
        'radar': [item['radar'] for item in batch],
        'image': [item['image'] for item in batch],
        'label': [item['label'] for item in batch],
        'calib': [item['calib'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }


# ============ Multi-Modal Detection Network ============
class MultiModalDetectionNetwork(nn.Module):
    def __init__(self, lidar_dim=128, radar_dim=128, image_dim=128, active_modalities=['lidar', 'radar', 'image']):
        super().__init__()

        self.active_modalities = set(active_modalities)

        self.lidar_net = LiDARFeatureExtractor(in_channels=3, feature_dim=lidar_dim)
        self.radar_net = RadarFeatureExtractor(in_channels=5, feature_dim=radar_dim)
        self.image_net = ImageFeatureExtractor(in_channels=3, feature_dim=image_dim)

        self.fusion_module = FusionModule(feature_dim=128, num_heads=8, dropout=0.1)

        self.detector = SingleStageDetector(
            backbone_channels=128,
            num_classes=3,
            num_anchors_per_location=12,
            pos_iou_thresh=0.05,
            neg_iou_thresh=0.01,
        )

    def forward(self, lidar_sparse, radar_sparse, images, targets=None, training=True, original_lidar_points=None):
        if original_lidar_points is not None:
            points_for_density = original_lidar_points
        else:
            points_for_density = lidar_sparse.indices.float()[:, 1:]

        lidar_feat_sparse, lidar_conf_sparse = self.lidar_net(lidar_sparse)
        radar_feat_sparse, radar_conf_sparse = self.radar_net(radar_sparse)
        image_feat, image_conf = self.image_net(images)

        if 'lidar' not in self.active_modalities:
            lidar_feat_sparse = lidar_feat_sparse.replace_feature(torch.zeros_like(lidar_feat_sparse.features))
        if 'radar' not in self.active_modalities:
            radar_feat_sparse = radar_feat_sparse.replace_feature(torch.zeros_like(radar_feat_sparse.features))
        if 'image' not in self.active_modalities:
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
            'lidar_features': lidar_feat_sparse,
            'lidar_confidence': lidar_conf_sparse,
            'radar_features': radar_feat_sparse,
            'radar_confidence': radar_conf_sparse,
            'image_features': image_feat,
            'image_confidence': image_conf,
            'fused_features': fused_features,
            'depth_threshold': depth_threshold,
            'detections': detection_outputs,
        }


# ============ Loss Functions ============
class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        detections = predictions['detections']
        device = predictions['fused_features'].device

        if 'losses' not in detections:
            return torch.tensor(0.0, device=device), {
                'rpn_cls_loss': 0.0,
                'rpn_reg_loss': 0.0,
                'rpn_multi_cls_loss': 0.0,
                'num_pos_anchors': 0.0,
                'total_loss': 0.0,
            }

        rpn_losses = detections['losses']
        rpn_cls_loss = rpn_losses.get('rpn_cls_loss', torch.tensor(0.0, device=device))
        rpn_reg_loss = rpn_losses.get('rpn_reg_loss', torch.tensor(0.0, device=device))
        rpn_multi_cls_loss = rpn_losses.get('rpn_multi_cls_loss', torch.tensor(0.0, device=device))
        num_pos = rpn_losses.get('num_pos_anchors', 0.0)

        total_loss = 5.0 * rpn_cls_loss + 5.0 * rpn_reg_loss + 2.0 * rpn_multi_cls_loss

        return total_loss, {
            'rpn_cls_loss': float(rpn_cls_loss.detach().item() if isinstance(rpn_cls_loss, torch.Tensor) else rpn_cls_loss),
            'rpn_reg_loss': float(rpn_reg_loss.detach().item() if isinstance(rpn_reg_loss, torch.Tensor) else rpn_reg_loss),
            'rpn_multi_cls_loss': float(rpn_multi_cls_loss.detach().item() if isinstance(rpn_multi_cls_loss, torch.Tensor) else rpn_multi_cls_loss),
            'num_pos_anchors': float(num_pos),
            'total_loss': float(total_loss.detach().item() if isinstance(total_loss, torch.Tensor) else total_loss),
        }


# ============ Training Loop ============
def train_one_epoch(model, dataloader, optimizer, det_criterion, device, epoch):
    model.train()
    start_time = time.time()

    total_loss = 0.0
    loss_stats = {'rpn_cls_loss': 0.0, 'rpn_reg_loss': 0.0, 'rpn_multi_cls_loss': 0.0, 'num_pos_anchors': 0.0, 'total_loss': 0.0}
    num_batches = 0

    epoch_predictions = []
    epoch_targets = []

    for batch in dataloader:
        original_points_list = [torch.from_numpy(lidar).float()[:, :3] for lidar in batch['lidar']]

        lidar_result = voxelize_lidar_proper(batch['lidar'])
        radar_result = voxelize_radar_proper(batch['radar'])

        lidar_feat, lidar_coords, lidar_shape, lidar_bs = lidar_result
        radar_feat, radar_coords, radar_shape, radar_bs = radar_result

        lidar_feat = lidar_feat.to(device)
        lidar_coords = lidar_coords.to(device).int()
        radar_feat = radar_feat.to(device)
        radar_coords = radar_coords.to(device).int()

        original_points_concat = torch.cat(original_points_list, dim=0).to(device)

        lidar_sparse = spconv.SparseConvTensor(
            features=lidar_feat,
            indices=lidar_coords,
            spatial_shape=lidar_shape,
            batch_size=lidar_bs,
        )
        radar_sparse = spconv.SparseConvTensor(
            features=radar_feat,
            indices=radar_coords,
            spatial_shape=radar_shape,
            batch_size=radar_bs,
        )

        images = prepare_images(batch['image']).to(device)

        targets = parse_labels_to_targets(batch['label'], batch['calib'])
        for t in targets:
            t['boxes_3d'] = t['boxes_3d'].to(device)
            t['labels'] = t['labels'].to(device)
            t['scores'] = t['scores'].to(device)

        outputs = model(
            lidar_sparse,
            radar_sparse,
            images,
            targets=targets,
            training=True,
            original_lidar_points=original_points_concat,
        )

        det_loss, loss_dict = det_criterion(outputs, targets)

        optimizer.zero_grad()
        det_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += float(det_loss.detach().item())
        for k in loss_stats:
            if k in loss_dict:
                loss_stats[k] += float(loss_dict[k])
        num_batches += 1

        detections = outputs['detections']
        proposals = detections.get('proposals', [])
        scores = detections.get('scores', [])
        labels = detections.get('proposal_labels', [])

        for i in range(len(targets)):
            pred_boxes = proposals[i].detach().cpu() if i < len(proposals) else torch.zeros((0, 7))
            pred_scores = scores[i].detach().cpu() if i < len(scores) else torch.zeros((0,))
            pred_labels = labels[i].detach().cpu().long() if i < len(labels) else torch.zeros((0,), dtype=torch.long)

            epoch_predictions.append({'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels})
            epoch_targets.append({
                'boxes': targets[i]['boxes_3d'].detach().cpu(),
                'labels': targets[i]['labels'].detach().cpu().long(),
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for k in loss_stats:
        loss_stats[k] = loss_stats[k] / num_batches if num_batches > 0 else 0.0

    map_stats = compute_map(epoch_predictions, epoch_targets, iou_threshold=0.5)
    elapsed = int(time.time() - start_time)

    return avg_loss, loss_stats, map_stats, elapsed


def main():
    parser = argparse.ArgumentParser(description='V2X-Radar-V multimodal training')
    parser.add_argument('--root_dir', type=str, default=r'F:\Work\DeepLearning\Research\V2X-Radar-V')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--active_modalities', type=str, default='lidar,radar,image')
    args = parser.parse_args()

    active_modalities = [m.strip() for m in args.active_modalities.split(',') if m.strip()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = V2XRadarDataset(args.root_dir, split='training', num_samples=args.num_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = MultiModalDetectionNetwork(
        lidar_dim=128,
        radar_dim=128,
        image_dim=128,
        active_modalities=active_modalities,
    ).to(device)

    det_criterion = DetectionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.5)

    start_epoch = 1
    checkpoint_info = find_latest_checkpoint()
    if checkpoint_info is not None:
        checkpoint_path, checkpoint_epoch = checkpoint_info
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        for _ in range(checkpoint_epoch):
            scheduler.step()

    train_losses = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, loss_stats, map_stats, epoch_seconds = train_one_epoch(
            model, train_loader, optimizer, det_criterion, device, epoch
        )

        train_losses.append(train_loss)
        log_line = (
            f"Epoch {epoch} | Time {epoch_seconds}s | mAP@0.5 {map_stats['mAP']:.4f} | "
            f"Loss {train_loss:.4f} | CLS {loss_stats['rpn_cls_loss']:.4f} | REG {loss_stats['rpn_reg_loss']:.4f}"
        )
        log_epoch_line(log_line)

        scheduler.step()

        if epoch or epoch == 100:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': train_loss,
                    'loss_stats': loss_stats,
                    'train_losses': train_losses,
                    'map_stats': map_stats,
                },
                f'checkpoint_epoch_{epoch}.pth',
            )

    torch.save(
        {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_losses[-1] if train_losses else 0,
            'train_losses': train_losses,
        },
        'final_model.pth',
    )

    if len(train_losses) > 0:
        plt.figure(figsize=(10, 5))
        epochs_range = range(start_epoch, start_epoch + len(train_losses))
        plt.plot(epochs_range, train_losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (Resumed from Epoch {start_epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')


if __name__ == '__main__':
    main()
