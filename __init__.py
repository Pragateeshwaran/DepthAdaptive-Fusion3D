"""
Complete Training Pipeline for Point-GNN on V2X-Radar-V Dataset
Run this file to train the Point-GNN model for 3D object detection

Requirements:
- NaN handling is done in data_loader.py
- V2X-Radar-V dataset at F:\Work\DeepLearning\Research\V2X-Radar-V
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from GNN import PointGNN, intializer
from utils import data_loader
np.random.seed(1808)
torch.manual_seed(1808)

# ========================== Configuration ==========================

class Config:
    """Training configuration"""
    # Dataset
    DATASET_ROOT = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    
    # Model
    NUM_ITERATIONS = 3
    STATE_DIM = 128
    GRAPH_RADIUS = 4.0
    
    # Training
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.125
    LR_DECAY_RATE = 0.1
    LR_DECAY_STEPS = [20, 35]  # Decay at these epochs
    BATCH_SIZE = 1  # Due to variable graph sizes
    NUM_WORKERS = 0
    
    # Loss weights (from paper: α=0.1, β=10, γ=5e-7)
    ALPHA = 0.1  # Classification loss weight
    BETA = 10.0  # Localization loss weight
    GAMMA = 5e-7  # Regularization weight
    
    # Classes: 0=Background, 1=Car, 2=Pedestrian, 3=Cyclist
    CLASS_NAMES = ['Background', 'Car', 'Pedestrian', 'Cyclist']
    NUM_CLASSES = 4
    
    # Scale factors for bbox encoding (from paper for Car)
    SCALE_FACTORS = {
        'l_m': 3.88,   # median length
        'h_m': 1.5,    # median height
        'w_m': 1.63,   # median width
        'theta_0': 0.0,
        'theta_m': 1.57
    }
    
    # Output
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    SAVE_FREQ = 5  # Save every N epochs


# ========================== Label Parser ==========================

def parse_kitti_label(label_lines):
    """
    Parse KITTI format label file.
    
    Label format per line:
    type truncated occluded alpha bbox_2d(4) dimensions(3) location(3) rotation_y [score]
    
    Returns:
        list of dicts with keys: type, bbox_3d(x,y,z,l,h,w,theta)
    """
    objects = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        
        obj_type = parts[0]  # Car, Pedestrian, Cyclist, etc.
        
        # Skip DontCare objects
        if obj_type == 'DontCare':
            continue
        
        # 3D bounding box: h, w, l at indices 8,9,10
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        # Location: x, y, z at indices 11,12,13
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        # Rotation: rotation_y at index 14
        rotation_y = float(parts[14])
        
        objects.append({
            'type': obj_type,
            'bbox_3d': np.array([x, y, z, l, h, w, rotation_y], dtype=np.float32)
        })
    
    return objects


# ========================== Dataset ==========================

class PointGNNDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand"""
    
    def __init__(self, config, split='training', max_samples=None, device=None):
        """
        Args:
            config: Configuration object
            split: 'training' or 'testing'
            max_samples: Limit dataset size for debugging
            device: torch device for GPU acceleration
        """
        self.config = config
        self.split = split
        self.class_mapping = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
        self.device = device if device is not None else torch.device('cpu')
        
        print(f"\n{'='*60}")
        print(f"Loading {split} dataset from {config.DATASET_ROOT}")
        print(f"{'='*60}")
        
        # Get file paths instead of loading all data
        self.dataset_root = config.DATASET_ROOT
        
        # Get list of label files
        label_dir = os.path.join(config.DATASET_ROOT, split, 'label_2')
        lidar_dir = os.path.join(config.DATASET_ROOT, split, 'velodyne')
        
        # Get all available files
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        if max_samples:
            label_files = label_files[:max_samples]
        
        print(f"Found {len(label_files)} samples")
        
        # Filter samples with objects (only load labels, not point clouds)
        if split == 'training':
            valid_files = []
            for label_file in label_files:
                label_path = os.path.join(label_dir, label_file)
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                
                objects = parse_kitti_label(label_lines)
                if any(obj['type'] in self.class_mapping for obj in objects):
                    valid_files.append(label_file)
            
            print(f"Filtered {len(valid_files)}/{len(label_files)} samples with objects")
            self.file_list = valid_files
        else:
            self.file_list = label_files
        
        print(f"Dataset size: {len(self.file_list)} samples")
        print(f"Memory-efficient mode: Loading data on-demand")
        print(f"Graph construction will use device: {self.device}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Load data on-demand to save RAM
        
        Returns:
            dict with keys:
                - graph_data: constructed graph
                - gt_labels: (N,) vertex class labels
                - gt_bboxes: (N, 7) ground truth bboxes
                - is_object_mask: (N,) boolean mask
        """
        # Get file name
        file_name = self.file_list[idx]
        file_id = file_name.replace('.txt', '')
        
        # Load label on-demand
        label_path = os.path.join(self.dataset_root, self.split, 'label_2', file_name)
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
        
        # Load point cloud on-demand
        lidar_path = os.path.join(self.dataset_root, self.split, 'velodyne', f'{file_id}.bin')
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # Clean NaN values
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        
        if len(points) == 0:
            # Return empty graph if no valid points
            return self._create_empty_sample()
        
        # Construct graph (on GPU)
        graph_data = intializer.construct_graph(
            points, 
            radius=self.config.GRAPH_RADIUS, 
            device=self.device,
            verbose=False
        )
        
        # Get vertex coords (already on GPU)
        vertex_coords = graph_data['vertex_coords'].cpu().numpy()
        N = len(vertex_coords)
        
        # Parse labels
        objects = parse_kitti_label(label_lines)
        
        # Initialize ground truth arrays
        gt_labels = np.zeros(N, dtype=np.int64)  # 0 = background
        gt_bboxes = np.zeros((N, 7), dtype=np.float32)
        is_object_mask = np.zeros(N, dtype=np.float32)
        
        # Assign labels to vertices (vertices inside bbox get object label)
        for obj in objects:
            if obj['type'] not in self.class_mapping:
                continue
            
            class_id = self.class_mapping[obj['type']]
            bbox = obj['bbox_3d']
            x, y, z, l, h, w, theta = bbox
            
            # Skip invalid bounding boxes
            if not np.isfinite([x, y, z, l, h, w, theta]).all():
                continue
            
            # Find vertices inside this bounding box (simple AABB check)
            dx = np.abs(vertex_coords[:, 0] - x)
            dy = np.abs(vertex_coords[:, 1] - y)
            dz = np.abs(vertex_coords[:, 2] - z)
            
            inside_mask = (dx < l/2) & (dy < w/2) & (dz < h/2)
            
            # Assign labels to vertices inside this box
            gt_labels[inside_mask] = class_id
            gt_bboxes[inside_mask] = bbox
            is_object_mask[inside_mask] = 1.0
        
        return {
            'graph_data': graph_data,  # Already on GPU
            'gt_labels': torch.tensor(gt_labels, dtype=torch.long),
            'gt_bboxes': torch.tensor(gt_bboxes, dtype=torch.float32),
            'is_object_mask': torch.tensor(is_object_mask, dtype=torch.float32),
            'idx': idx
        }
    
    def _create_empty_sample(self):
        """Create empty sample for edge cases"""
        empty_graph = {
            'vertex_features': torch.zeros((1, self.config.STATE_DIM), device=self.device),
            'vertex_coords': torch.zeros((1, 3), device=self.device),
            'edge_index': torch.zeros((2, 0), dtype=torch.long, device=self.device)
        }
        return {
            'graph_data': empty_graph,
            'gt_labels': torch.zeros(1, dtype=torch.long),
            'gt_bboxes': torch.zeros((1, 7), dtype=torch.float32),
            'is_object_mask': torch.zeros(1, dtype=torch.float32),
            'idx': 0
        }


# ========================== Loss Function ==========================

class PointGNNLoss(nn.Module):
    """
    Multi-task loss from Point-GNN paper:
    L_total = α*L_cls + β*L_loc + γ*L_reg
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.ALPHA
        self.beta = config.BETA
        self.gamma = config.GAMMA
    
    def encode_bbox(self, bbox, vertex_coords):
        """
        Encode bbox relative to vertex coordinates (Equation 7 from paper)
        
        Args:
            bbox: (N, 7) - (x, y, z, l, h, w, θ)
            vertex_coords: (N, 3) - (x_v, y_v, z_v)
        Returns:
            encoded_bbox: (N, 7) - (δx, δy, δz, δl, δh, δw, δθ)
        """
        sf = self.config.SCALE_FACTORS
        
        # Unpack bbox
        x = bbox[:, 0]
        y = bbox[:, 1]
        z = bbox[:, 2]
        l = bbox[:, 3]
        h = bbox[:, 4]
        w = bbox[:, 5]
        theta = bbox[:, 6]
        
        # Unpack vertex coords
        x_v = vertex_coords[:, 0]
        y_v = vertex_coords[:, 1]
        z_v = vertex_coords[:, 2]
        
        # Encode (Equation 7)
        delta_x = (x - x_v) / sf['l_m']
        delta_y = (y - y_v) / sf['h_m']
        delta_z = (z - z_v) / sf['w_m']
        delta_l = torch.log(l / sf['l_m'] + 1e-6)
        delta_h = torch.log(h / sf['h_m'] + 1e-6)
        delta_w = torch.log(w / sf['w_m'] + 1e-6)
        delta_theta = (theta - sf['theta_0']) / sf['theta_m']
        
        encoded = torch.stack([
            delta_x, delta_y, delta_z, 
            delta_l, delta_h, delta_w, 
            delta_theta
        ], dim=1)
        
        return encoded
    
    def forward(self, cls_logits, bbox_pred, gt_labels, gt_bboxes, 
                vertex_coords, is_object_mask, model):
        """
        Compute total loss
        
        Args:
            cls_logits: (N, num_classes) - predicted class logits
            bbox_pred: (N, 7) - predicted bboxes
            gt_labels: (N,) - ground truth class indices
            gt_bboxes: (N, 7) - ground truth bboxes
            vertex_coords: (N, 3) - vertex coordinates
            is_object_mask: (N,) - mask for vertices in objects
            model: model for L1 regularization
        
        Returns:
            total_loss, loss_dict
        """
        device = cls_logits.device
        N = cls_logits.shape[0]
        
        # ===== Classification Loss (Equation 6) =====
        # Cross-entropy loss
        loss_cls = F.cross_entropy(cls_logits, gt_labels, reduction='mean')
        
        # ===== Localization Loss (Equation 8) =====
        # Only compute for vertices inside objects
        if is_object_mask.sum() > 0:
            # Get object vertices
            obj_mask = is_object_mask > 0
            
            # Encode ground truth bboxes
            gt_encoded = self.encode_bbox(gt_bboxes[obj_mask], vertex_coords[obj_mask])
            pred_encoded = bbox_pred[obj_mask]
            
            # Huber loss (smooth L1)
            loss_loc = F.smooth_l1_loss(pred_encoded, gt_encoded, reduction='mean')
        else:
            loss_loc = torch.tensor(0.0, device=device)
        
        # ===== L1 Regularization =====
        loss_reg = 0.0
        for param in model.parameters():
            loss_reg += torch.abs(param).sum()
        
        # ===== Total Loss (Equation 9) =====
        loss_total = (self.alpha * loss_cls + 
                     self.beta * loss_loc + 
                     self.gamma * loss_reg)
        
        loss_dict = {
            'total': loss_total.item(),
            'cls': loss_cls.item(),
            'loc': loss_loc.item() if torch.is_tensor(loss_loc) else 0.0,
            'reg': loss_reg.item() if torch.is_tensor(loss_reg) else loss_reg
        }
        
        return loss_total, loss_dict


# ========================== GPU Diagnostics ==========================

def check_gpu_usage():
    """Check if GPU is actually being used"""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU DIAGNOSTICS")
        print("="*60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print("="*60 + "\n")


def verify_tensors_on_gpu(graph_data, gt_labels, gt_bboxes, is_object_mask, model):
    """Verify all tensors are on GPU"""
    print("\n" + "-"*60)
    print("TENSOR DEVICE CHECK")
    print("-"*60)
    
    # Check graph data
    for key, value in graph_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  graph_data['{key}']: {value.device}")
    
    # Check ground truth
    print(f"  gt_labels: {gt_labels.device}")
    print(f"  gt_bboxes: {gt_bboxes.device}")
    print(f"  is_object_mask: {is_object_mask.device}")
    
    # Check model
    model_device = next(model.parameters()).device
    print(f"  model parameters: {model_device}")
    
    # Check if all on same device
    devices = []
    for key, value in graph_data.items():
        if isinstance(value, torch.Tensor):
            devices.append(value.device)
    devices.extend([gt_labels.device, gt_bboxes.device, is_object_mask.device, model_device])
    
    if len(set(str(d) for d in devices)) == 1:
        print(f"\n  ✓ All tensors on same device: {devices[0]}")
    else:
        print(f"\n  ✗ WARNING: Tensors on different devices!")
        print(f"  Devices found: {set(str(d) for d in devices)}")
    
    print("-"*60 + "\n")


# ========================== Training ==========================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with memory management"""
    model.train()
    
    running_loss = {
        'total': 0.0,
        'cls': 0.0,
        'loc': 0.0,
        'reg': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Graph data is already on GPU from dataset
        graph_data = batch['graph_data']
        
        # Move only ground truth to device (graph_data already on GPU)
        gt_labels = batch['gt_labels'].to(device, non_blocking=True)
        gt_bboxes = batch['gt_bboxes'].to(device, non_blocking=True)
        is_object_mask = batch['is_object_mask'].to(device, non_blocking=True)
        vertex_coords = graph_data['vertex_coords']
        
        # GPU diagnostics for first batch only
        if batch_idx == 0 and epoch == 1:
            verify_tensors_on_gpu(graph_data, gt_labels, gt_bboxes, is_object_mask, model)
        
        # Forward pass
        optimizer.zero_grad()
        cls_logits, bbox_pred = model(graph_data)
        
        # Verify outputs are on GPU (first batch only)
        if batch_idx == 0 and epoch == 1:
            print(f"Model output device - cls_logits: {cls_logits.device}, bbox_pred: {bbox_pred.device}\n")
        
        # Compute loss
        loss, loss_dict = criterion(
            cls_logits, bbox_pred, gt_labels, gt_bboxes,
            vertex_coords, is_object_mask, model
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in running_loss:
            running_loss[key] += loss_dict[key]
        
        # Clear cache periodically to prevent memory buildup
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update progress bar with memory info
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['cls']:.4f}",
                'loc': f"{loss_dict['loc']:.4f}",
                'gpu_mb': f"{mem_allocated:.0f}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['cls']:.4f}",
                'loc': f"{loss_dict['loc']:.4f}"
            })
    
    # Average losses
    num_batches = len(dataloader)
    for key in running_loss:
        running_loss[key] /= num_batches
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return running_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    running_loss = {
        'total': 0.0,
        'cls': 0.0,
        'loc': 0.0,
        'reg': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Graph data already on GPU
            graph_data = batch['graph_data']
            
            gt_labels = batch['gt_labels'].to(device, non_blocking=True)
            gt_bboxes = batch['gt_bboxes'].to(device, non_blocking=True)
            is_object_mask = batch['is_object_mask'].to(device, non_blocking=True)
            vertex_coords = graph_data['vertex_coords']
            
            # Forward pass
            cls_logits, bbox_pred = model(graph_data)
            
            # Compute loss
            loss, loss_dict = criterion(
                cls_logits, bbox_pred, gt_labels, gt_bboxes,
                vertex_coords, is_object_mask, model
            )
            
            # Accumulate losses
            for key in running_loss:
                running_loss[key] += loss_dict[key]
    
    # Average losses
    num_batches = len(dataloader)
    for key in running_loss:
        running_loss[key] /= num_batches
    
    return running_loss


def save_checkpoint(model, optimizer, epoch, loss, config, filename):
    """Save training checkpoint"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': vars(config)
    }, filepath)
    
    print(f"Checkpoint saved: {filepath}")


def main():
    """Main training loop"""
    
    # Configuration
    config = Config()
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run GPU diagnostics
    check_gpu_usage()
    
    # Datasets - PASS DEVICE TO DATASET
    print("Preparing datasets...")
    train_dataset = PointGNNDataset(config, split='training', max_samples=None, device=device)
    
    # DataLoaders with memory optimization
    def collate_fn(batch):
        """Custom collate that returns single item without batching"""
        return batch[0]  # Return first (and only) item without adding batch dimension
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing memory overhead
        pin_memory=False,  # Data already on GPU
        collate_fn=collate_fn,
        prefetch_factor=None  # Disable prefetching to save memory
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    
    # Model
    print("\nInitializing model...")
    model = PointGNN.PointGNN(
        num_iterations=config.NUM_ITERATIONS,
        state_dim=config.STATE_DIM,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    # Verify model is on GPU
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer (SGD as in paper)
    criterion = PointGNNLoss(config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler (staircase decay as in paper)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_DECAY_STEPS,
        gamma=config.LR_DECAY_RATE
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_loc_loss': []
    }
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Print epoch results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss['total']:.4f}")
        print(f"    - Classification: {train_loss['cls']:.4f}")
        print(f"    - Localization: {train_loss['loc']:.4f}")
        print(f"    - Regularization: {train_loss['reg']:.6f}")
        
        # GPU memory stats
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Save history
        history['train_loss'].append(train_loss['total'])
        history['train_cls_loss'].append(train_loss['cls'])
        history['train_loc_loss'].append(train_loss['loc'])
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if epoch % config.SAVE_FREQ == 0 or epoch == config.NUM_EPOCHS:
            save_checkpoint(
                model, optimizer, epoch, train_loss['total'], config,
                f'pointgnn_epoch_{epoch}.pth'
            )
        
        # Save best model
        if epoch == 1 or train_loss['total'] < min(history['train_loss'][:-1]):
            save_checkpoint(
                model, optimizer, epoch, train_loss['total'], config,
                'pointgnn_best.pth'
            )
            print("  ★ Best model saved!")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training completed in {elapsed_time/3600:.2f} hours")
    print("="*60)
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS, 
        history['train_loss'][-1], config,
        'pointgnn_final.pth'
    )
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved: {history_path}")
    
    # Final GPU diagnostics
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("FINAL GPU MEMORY USAGE")
        print("="*60)
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        print("="*60)


if __name__ == '__main__':
    main()