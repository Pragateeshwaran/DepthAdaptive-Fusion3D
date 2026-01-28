import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import spconv.pytorch as spconv

# Import corrected modules
from lidar_model import LiDARFeatureExtractor
from radar_model import RadarFeatureExtractor
from image_model import ImageFeatureExtractor
from rpn_refinement import TwoStageDetector, parse_label_line
from cross_attention import FusionModule

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_LiDAR, get_radar, get_images, get_labels, get_calib

# ============ Voxelization Functions ============

def voxelize_lidar(points_batch, voxel_size=(0.1, 0.1, 0.1), 
                   point_cloud_range=(-50, -50, -3, 50, 50, 5)):
    """Voxelize LiDAR point clouds with SPARSE representation."""
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
    ]
    
    voxel_features_list = []
    voxel_coords_list = []
    
    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        voxel_coords = ((points[:, :3] - [point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]]) / 
                       [voxel_size[0], voxel_size[1], voxel_size[2]]).astype(int)
        
        valid_mask = ((voxel_coords >= 0) & 
                     (voxel_coords < [spatial_shape[0], spatial_shape[1], spatial_shape[2]])).all(axis=1)
        
        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]
        
        if len(points_valid) == 0:
            continue
        
        voxel_coords_unique, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        num_voxels = voxel_coords_unique.shape[0]
        voxel_features = np.zeros((num_voxels, points_valid.shape[1]), dtype=np.float32)
        
        for i in range(num_voxels):
            mask = inverse_indices == i
            voxel_features[i] = points_valid[mask].mean(axis=0)
        
        batch_indices = np.full((num_voxels, 1), batch_idx, dtype=np.int32)
        voxel_coords_with_batch = np.concatenate([batch_indices, voxel_coords_unique], axis=1)
        
        voxel_features_list.append(torch.from_numpy(voxel_features))
        voxel_coords_list.append(torch.from_numpy(voxel_coords_with_batch))
    
    if len(voxel_features_list) == 0:
        return None
    
    voxel_features = torch.cat(voxel_features_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    
    return voxel_features, voxel_coords, spatial_shape, len(points_batch)


def voxelize_radar(points_batch, voxel_size=(0.2, 0.2, 0.2), 
                   point_cloud_range=(-50, -50, -3, 50, 50, 5)):
    """Voxelize Radar point clouds with SPARSE representation."""
    spatial_shape = [
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
    ]
    
    voxel_features_list = []
    voxel_coords_list = []
    
    for batch_idx, points in enumerate(points_batch):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        voxel_coords = ((points[:, :3] - [point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]]) / 
                       [voxel_size[0], voxel_size[1], voxel_size[2]]).astype(int)
        
        valid_mask = ((voxel_coords >= 0) & 
                     (voxel_coords < [spatial_shape[0], spatial_shape[1], spatial_shape[2]])).all(axis=1)
        
        voxel_coords = voxel_coords[valid_mask]
        points_valid = points[valid_mask]
        
        if len(points_valid) == 0:
            continue
        
        voxel_coords_unique, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        num_voxels = voxel_coords_unique.shape[0]
        voxel_features = np.zeros((num_voxels, points_valid.shape[1]), dtype=np.float32)
        
        for i in range(num_voxels):
            mask = inverse_indices == i
            points_in_voxel = points_valid[mask]
            voxel_features[i, :3] = points_in_voxel[:, :3].mean(axis=0)
            if points_valid.shape[1] >= 5:
                voxel_features[i, 3] = points_in_voxel[:, 3].mean(axis=0)
                voxel_features[i, 4] = points_in_voxel[:, 4].max()
        
        batch_indices = np.full((num_voxels, 1), batch_idx, dtype=np.int32)
        voxel_coords_with_batch = np.concatenate([batch_indices, voxel_coords_unique], axis=1)
        
        voxel_features_list.append(torch.from_numpy(voxel_features))
        voxel_coords_list.append(torch.from_numpy(voxel_coords_with_batch))
    
    if len(voxel_features_list) == 0:
        return None
    
    voxel_features = torch.cat(voxel_features_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    
    return voxel_features, voxel_coords, spatial_shape, len(points_batch)


def prepare_images(image_batch):
    """Prepare images for network input."""
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


def transform_boxes_camera_to_lidar(boxes_camera, calib):
    """
    Transform boxes from camera coordinates to LiDAR coordinates.
    
    Args:
        boxes_camera: (N, 7) array [x, y, z, h, w, l, rot] in camera coordinates
        calib: Calibration dictionary
    
    Returns:
        boxes_lidar: (N, 7) array [x, y, z, h, w, l, rot] in LiDAR coordinates
    """
    if len(boxes_camera) == 0:
        return boxes_camera
    
    Tr_velo_to_cam = calib.get('Tr_velo_to_cam', np.eye(4))
    
    if Tr_velo_to_cam.shape == (12,):
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    
    boxes_lidar = boxes_camera.copy()
    
    for i in range(len(boxes_camera)):
        x_cam, y_cam, z_cam = boxes_camera[i, 0:3]
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_lidar = Tr_cam_to_velo @ point_cam
        boxes_lidar[i, 0:3] = point_lidar[:3]
        
        rot_cam = boxes_camera[i, 6]
        rot_lidar = -rot_cam - np.pi / 2
        boxes_lidar[i, 6] = rot_lidar
    
    return boxes_lidar


def parse_labels_to_targets(labels_batch, calib_batch=None):
    """
    ⭐ FIXED: Parse labels and transform to LiDAR coordinates.
    """
    targets = []
    class_map = {'pedestrian': 0, 'cyclist': 1, 'car': 2}
    
    for idx, labels in enumerate(labels_batch):
        boxes_list = []
        labels_list = []
        scores_list = []
        
        for line in labels:
            if isinstance(line, str) and line.strip():
                parsed = parse_label_line(line)
                obj_type = parsed['type'].lower()
                
                if obj_type in class_map:
                    # Box in CAMERA coordinates
                    box_3d = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
                    boxes_list.append(box_3d)
                    labels_list.append(class_map[obj_type])
                    scores_list.append(parsed['score'])
        
        if len(boxes_list) > 0:
            boxes_camera = np.array(boxes_list)
            
            # ⭐ CRITICAL FIX: Transform to LiDAR coordinates
            if calib_batch and idx < len(calib_batch):
                boxes_lidar = transform_boxes_camera_to_lidar(boxes_camera, calib_batch[idx])
            else:
                boxes_lidar = boxes_camera
                print("⚠️ WARNING: No calibration, boxes still in camera coords!")
            
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
        
        print(f"Loading {split} data...")
        
        self.lidar_data = get_LiDAR(root_dir, split, from_idx=0, count=num_samples)
        self.radar_data = get_radar(root_dir, split, from_idx=0, count=num_samples)
        self.image_data = get_images(root_dir, split, from_idx=0, count=num_samples)
        self.labels = get_labels(root_dir, split, from_idx=0, count=num_samples)
        
        min_len = min(len(self.lidar_data), len(self.radar_data), len(self.image_data))
        self.lidar_data = self.lidar_data[:min_len]
        self.radar_data = self.radar_data[:min_len]
        self.image_data = self.image_data[:min_len]
        self.labels = self.labels[:min_len]
        
        print(f"Loaded {len(self)} samples")
    
    def __len__(self):
        return len(self.lidar_data)
    
    def __getitem__(self, idx):
        return {
            'lidar': self.lidar_data[idx],
            'radar': self.radar_data[idx],
            'image': self.image_data[idx],
            'label': self.labels[idx],
            'idx': idx
        }


def collate_fn(batch):
    """Custom collate function."""
    return {
        'lidar': [item['lidar'] for item in batch],
        'radar': [item['radar'] for item in batch],
        'image': [item['image'] for item in batch],
        'label': [item['label'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }


# ============ Multi-Modal Detection Network ============

class MultiModalDetectionNetwork(nn.Module):
    def __init__(self, lidar_dim=128, radar_dim=128, image_dim=128):
        super(MultiModalDetectionNetwork, self).__init__()
        
        self.lidar_net = LiDARFeatureExtractor(in_channels=3, feature_dim=lidar_dim)
        self.radar_net = RadarFeatureExtractor(in_channels=5, feature_dim=radar_dim)
        self.image_net = ImageFeatureExtractor(in_channels=3, feature_dim=image_dim)
        
        self.fusion_module = FusionModule(
            feature_dim=128,
            num_heads=8,
            dropout=0.1
        )
        
        self.detector = TwoStageDetector(
            backbone_channels=128,
            num_classes=3,
            num_anchors_per_location=6,
            pos_iou_thresh=0.3,
            neg_iou_thresh=0.15
        )
    
    def forward(self, lidar_sparse, radar_sparse, images, targets=None, training=True,
                original_lidar_points=None):
        """
        ⭐ FIXED: Accept original LiDAR points for density calculation.
        """
        # Use original points if provided
        if original_lidar_points is not None:
            points_for_density = original_lidar_points
        else:
            # Fallback: use voxel indices (less accurate)
            lidar_indices = lidar_sparse.indices.float()
            points_for_density = lidar_indices[:, 1:]
        
        # Extract features
        lidar_feat_sparse, lidar_conf_sparse = self.lidar_net(lidar_sparse)
        radar_feat_sparse, radar_conf_sparse = self.radar_net(radar_sparse)
        image_feat, image_conf = self.image_net(images)
        
        # Fusion with density-based ATGN
        fused_features, depth_threshold = self.fusion_module(
            lidar_feat_sparse, 
            radar_feat_sparse, 
            image_feat,
            lidar_conf_sparse, 
            radar_conf_sparse, 
            image_conf,
            original_points=points_for_density
        )
        
        # Detection
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
            'detections': detection_outputs
        }


# ============ Loss Functions ============

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
    
    def forward(self, predictions, targets):
        """Compute detection loss."""
        detections = predictions['detections']
        
        if 'losses' in detections:
            rpn_losses = detections['losses']
            
            rpn_cls_loss = rpn_losses.get('rpn_cls_loss', torch.tensor(0.0, device=predictions['fused_features'].device))
            rpn_reg_loss = rpn_losses.get('rpn_reg_loss', torch.tensor(0.0, device=predictions['fused_features'].device))
            num_pos = rpn_losses.get('num_pos_anchors', 0)
            
            # ⭐ INCREASED weight for detection loss
            total_loss = 5.0 * rpn_cls_loss + 5.0 * rpn_reg_loss
            
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss.item() if isinstance(rpn_cls_loss, torch.Tensor) else rpn_cls_loss,
                'rpn_reg_loss': rpn_reg_loss.item() if isinstance(rpn_reg_loss, torch.Tensor) else rpn_reg_loss,
                'num_pos_anchors': num_pos,
                'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            }
            
            return total_loss, loss_dict
        else:
            device = predictions['fused_features'].device
            return torch.tensor(0.0, device=device), {
                'rpn_cls_loss': 0.0,
                'rpn_reg_loss': 0.0,
                'num_pos_anchors': 0,
                'total_loss': 0.0
            }


# ============ Training Loop ============

def train_one_epoch(model, dataloader, optimizer, det_criterion, device, epoch):
    """⭐ FIXED: Load calibration and pass original points."""
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    
    model.train()
    total_loss = 0
    loss_stats = {'rpn_cls_loss': 0, 'rpn_reg_loss': 0, 'num_pos_anchors': 0, 'total_loss': 0, 'conf_loss': 0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # ⭐ SAVE ORIGINAL POINTS before voxelization
        original_points_list = [torch.from_numpy(lidar).float()[:, :3] for lidar in batch['lidar']]
        
        # Voxelization
        lidar_result = voxelize_lidar(batch['lidar'])
        radar_result = voxelize_radar(batch['radar'])
        
        if lidar_result is None or radar_result is None:
            continue
        
        lidar_feat, lidar_coords, lidar_shape, lidar_bs = lidar_result
        radar_feat, radar_coords, radar_shape, radar_bs = radar_result
        
        # Move to device
        lidar_feat = lidar_feat.to(device)
        lidar_coords = lidar_coords.to(device).int()
        radar_feat = radar_feat.to(device)
        radar_coords = radar_coords.to(device).int()
        
        # ⭐ Concatenate and move original points to device
        original_points_concat = torch.cat(original_points_list, dim=0).to(device)
        
        # Create sparse tensors
        lidar_sparse = spconv.SparseConvTensor(
            features=lidar_feat,
            indices=lidar_coords,
            spatial_shape=lidar_shape,
            batch_size=lidar_bs
        )
        
        radar_sparse = spconv.SparseConvTensor(
            features=radar_feat,
            indices=radar_coords,
            spatial_shape=radar_shape,
            batch_size=radar_bs
        )
        
        # Prepare images
        images = prepare_images(batch['image']).to(device)
        
        # ⭐ LOAD CALIBRATION
        calibs = get_calib(ROOT_DIR, 'training', from_idx=batch['idx'][0], count=len(batch['lidar']))
        
        # ⭐ Parse labels WITH calibration (transforms to LiDAR coords)
        targets = parse_labels_to_targets(batch['label'], calibs)
        for t in targets:
            t['boxes_3d'] = t['boxes_3d'].to(device)
            t['labels'] = t['labels'].to(device)
            t['scores'] = t['scores'].to(device)
        
        # ⭐ Forward pass WITH original points
        outputs = model(lidar_sparse, radar_sparse, images, 
                       targets=targets, training=True,
                       original_lidar_points=original_points_concat)
        
        # Detection loss
        det_loss, loss_dict = det_criterion(outputs, targets)
        
        # Confidence regularization
        lidar_conf_features = outputs['lidar_confidence'].features
        radar_conf_features = outputs['radar_confidence'].features
        image_conf = outputs['image_confidence']
        
        conf_loss = (1 - lidar_conf_features.mean()) + \
                   (1 - radar_conf_features.mean()) + \
                   (1 - image_conf.mean())
        
        # Total loss
        total_loss_batch = det_loss + 0.1 * conf_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_batch.item()
        for k in loss_dict:
            loss_stats[k] += loss_dict[k]
        loss_stats['conf_loss'] += conf_loss.item()
        num_batches += 1
        
        # ⭐ Print diagnostic info for first batch
        if batch_idx == 0 and epoch == 1:
            print(f"\n🔍 DIAGNOSIS (Epoch {epoch}, Batch {batch_idx}):")
            if len(targets[0]['boxes_3d']) > 0:
                print(f"  GT Box [0]: {targets[0]['boxes_3d'][0]}")
            print(f"  Num positive anchors: {loss_dict['num_pos_anchors']:.1f}")
            print(f"  Depth threshold: {outputs['depth_threshold']:.2f}m")
        
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'cls': f'{loss_dict["rpn_cls_loss"]:.4f}',
            'reg': f'{loss_dict["rpn_reg_loss"]:.4f}',
            'pos': f'{loss_dict["num_pos_anchors"]:.1f}'
        })
    
    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in loss_stats:
        loss_stats[k] = loss_stats[k] / num_batches if num_batches > 0 else 0
    
    return avg_loss, loss_stats


def main():
    # Config
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-4  # ⭐ REDUCED from 1e-3
    NUM_SAMPLES = 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    train_dataset = V2XRadarDataset(ROOT_DIR, split='training', num_samples=NUM_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, collate_fn=collate_fn)
    
    # Model
    model = MultiModalDetectionNetwork(lidar_dim=128, radar_dim=128, image_dim=128).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✓ FIXES APPLIED:")
    print("  - GT boxes transformed from camera to LiDAR coordinates")
    print("  - Calibration loaded during training")
    print("  - Original LiDAR points passed for density calculation")
    print("  - Learning rate reduced to 5e-4")
    print("  - Detection loss weight increased to 5x\n")
    
    # Loss & Optimizer
    det_criterion = DetectionLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training
    train_losses = []
    
    print("Starting training (Pedestrian, Cyclist, Car detection)...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, loss_stats = train_one_epoch(model, train_loader, optimizer, 
                                                  det_criterion, device, epoch)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {train_loss:.4f}")
        print(f"  RPN CLS: {loss_stats['rpn_cls_loss']:.4f} | "
              f"RPN REG: {loss_stats['rpn_reg_loss']:.4f} | "
              f"Pos Anchors: {loss_stats['num_pos_anchors']:.1f} | "
              f"CONF: {loss_stats['conf_loss']:.4f}")
        
        scheduler.step()
        
        # Save checkpoint and visualize every 10 epochs
        if epoch % 10 == 0:
            print(f"\n{'='*70}")
            print(f"💾 Saving checkpoint at epoch {epoch}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'loss_stats': loss_stats
            }, f'checkpoint_epoch_{epoch}.pth')
            print(f"✓ Saved checkpoint")
            
            # Visualize predictions
            print(f"\n{'='*70}")
            print(f"📊 Visualizing predictions...")
            print(f"  Color Legend: 🟢 GREEN = Ground Truth | 🔴 RED = Predictions")
            print(f"{'='*70}")
            try:
                from visualize_predictions import visualize_epoch_predictions
                visualize_epoch_predictions(model, train_loader, device, epoch, num_samples=5)
                print("✓ Visualization complete")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing training...")
            print(f"{'='*70}\n")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("\n✓ Saved final model to final_model.pth")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Fixed Coordinate System)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    print("✓ Saved training curve to training_curve.png")
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()