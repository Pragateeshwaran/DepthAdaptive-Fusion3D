import numpy as np
import open3d as o3d
import torch
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_LiDAR, get_labels, get_calib
from rpn_refinement import parse_label_line, nms_bev


def transform_boxes_camera_to_lidar(boxes_camera, calib):
    """
    Transform boxes from camera coordinates to LiDAR coordinates.
    
    Args:
        boxes_camera: (N, 7) array [x, y, z, h, w, l, rot] in camera coordinates
        calib: Calibration dictionary with transformation matrices
    
    Returns:
        boxes_lidar: (N, 7) array [x, y, z, h, w, l, rot] in LiDAR coordinates
    """
    if len(boxes_camera) == 0:
        return boxes_camera
    
    # Get transformation matrix from camera to LiDAR
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


def apply_nms_to_boxes(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: (N, 7) array of boxes [x, y, z, h, w, l, rot]
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for NMS
    
    Returns:
        filtered_boxes: (M, 7) array of non-overlapping boxes
        filtered_scores: (M,) array of scores
    """
    if len(boxes) == 0:
        return boxes, scores
    
    boxes_torch = torch.from_numpy(boxes).float()
    scores_torch = torch.from_numpy(scores).float()
    
    keep_indices = nms_bev(boxes_torch, scores_torch, iou_threshold)
    
    filtered_boxes = boxes[keep_indices.cpu().numpy()]
    filtered_scores = scores[keep_indices.cpu().numpy()]
    
    return filtered_boxes, filtered_scores


def create_bbox_lines():
    """Create line indices for drawing 3D bounding box."""
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]
    return lines


def get_box_corners(center, size, rotation):
    """Get 8 corners of a 3D bounding box."""
    x, y, z = center
    h, w, l = size
    
    corners = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])
    
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    rot_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r,  cos_r, 0],
        [0,      0,     1]
    ])
    
    corners = corners @ rot_matrix.T
    corners += np.array([x, y, z])
    
    return corners


def create_bbox_lineset(boxes, color=[1, 0, 0]):
    """Create Open3D LineSet for multiple bounding boxes."""
    all_points = []
    all_lines = []
    point_offset = 0
    
    line_template = create_bbox_lines()
    
    for box in boxes:
        x, y, z, h, w, l, rot = box
        corners = get_box_corners([x, y, z], [h, w, l], rot)
        
        all_points.append(corners)
        
        for line in line_template:
            all_lines.append([line[0] + point_offset, line[1] + point_offset])
        
        point_offset += 8
    
    if len(all_points) == 0:
        lineset = o3d.geometry.LineSet()
        return lineset
    
    points = np.vstack(all_points)
    lines = np.array(all_lines)
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    
    colors = [color for _ in range(len(lines))]
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset


def visualize_predictions(lidar_points, gt_boxes, pred_boxes, sample_idx=0, pred_scores=None):
    """
    Visualize LiDAR points with ground truth (GREEN) and predicted (RED) boxes.
    
    Args:
        lidar_points: (N, 3) array of LiDAR points
        gt_boxes: (M, 7) array of ground truth boxes [x, y, z, h, w, l, rot]
        pred_boxes: (K, 7) array of predicted boxes [x, y, z, h, w, l, rot]
        sample_idx: Sample index
        pred_scores: (K,) array of prediction scores (optional)
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    
    # Color points by height
    z_values = lidar_points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
    colors = np.zeros((len(lidar_points), 3))
    colors[:, 0] = z_norm * 0.7
    colors[:, 1] = z_norm * 0.7
    colors[:, 2] = z_norm * 0.7
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create ground truth boxes (GREEN)
    gt_lineset = create_bbox_lineset(gt_boxes, color=[0, 1, 0])
    
    # Create predicted boxes (RED)
    pred_lineset = create_bbox_lineset(pred_boxes, color=[1, 0, 0])
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )
    
    # Prepare geometries
    geometries = [pcd, coord_frame]
    
    if len(gt_boxes) > 0:
        geometries.append(gt_lineset)
    
    if len(pred_boxes) > 0:
        geometries.append(pred_lineset)
    
    # Print info
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")
    print(f"LiDAR points: {len(lidar_points):,}")
    print(f"Ground truth boxes (GREEN): {len(gt_boxes)}")
    print(f"Predicted boxes (RED): {len(pred_boxes)}")
    
    if pred_scores is not None and len(pred_scores) > 0:
        print(f"Prediction scores: min={pred_scores.min():.3f}, max={pred_scores.max():.3f}, mean={pred_scores.mean():.3f}")
    
    print(f"{'='*70}")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Press 'Q' or close window to continue")
    print(f"{'='*70}\n")
    
    # Set up visualizer
    vis = o3d.visualization.Visualizer()
    window_name = f"Sample {sample_idx} - GREEN=GT, RED=Predictions"
    vis.create_window(window_name=window_name, width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    vis.run()
    vis.destroy_window()


def visualize_epoch_predictions(model, dataloader, device, epoch, num_samples=5):
    """
    Visualize predictions during training (called every 10 epochs).
    Shows 5 random samples with REAL model predictions.
    
    Args:
        model: Trained model
        dataloader: DataLoader with V2XRadarDataset
        device: Device to run inference on
        epoch: Current epoch number
        num_samples: Number of random samples to visualize (default: 5)
    """
    import spconv.pytorch as spconv
    from trail import voxelize_lidar, voxelize_radar, prepare_images, parse_labels_to_targets
    
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"  VISUALIZATION - EPOCH {epoch}")
    print(f"  Showing {num_samples} random samples with REAL predictions")
    print(f"  Color Legend: 🟢 GREEN = Ground Truth | 🔴 RED = Predictions")
    print(f"{'='*70}\n")
    
    # Collect all samples first
    all_samples = []
    for batch_idx, batch in enumerate(dataloader):
        for i in range(len(batch['lidar'])):
            all_samples.append({
                'lidar': batch['lidar'][i],
                'radar': [batch['radar'][i]],  # Wrap in list for voxelize
                'image': [batch['image'][i]],
                'label': batch['label'][i],
                'idx': batch['idx'][i]
            })
            if len(all_samples) >= 20:  # Collect 20 samples to choose from
                break
        if len(all_samples) >= 20:
            break
    
    # Randomly select samples
    if len(all_samples) > num_samples:
        selected_samples = random.sample(all_samples, num_samples)
    else:
        selected_samples = all_samples[:num_samples]
    
    # Get calibration data
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    
    with torch.no_grad():
        for sample_num, sample in enumerate(selected_samples):
            try:
                # Voxelize
                lidar_result = voxelize_lidar([sample['lidar']])
                radar_result = voxelize_radar(sample['radar'])
                
                if lidar_result is None or radar_result is None:
                    continue
                
                lidar_feat, lidar_coords, lidar_shape, lidar_bs = lidar_result
                radar_feat, radar_coords, radar_shape, radar_bs = radar_result
                
                # Move to device
                lidar_feat = lidar_feat.to(device)
                lidar_coords = lidar_coords.to(device).int()
                radar_feat = radar_feat.to(device)
                radar_coords = radar_coords.to(device).int()
                
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
                images = prepare_images(sample['image']).to(device)
                
                # Forward pass (REAL MODEL PREDICTIONS)
                outputs = model(lidar_sparse, radar_sparse, images, targets=None, training=False)
                
                # Get predictions
                proposals_list = outputs['detections']['proposals']
                scores_list = outputs['detections']['scores']
                
                # Parse ground truth
                targets = parse_labels_to_targets([sample['label']])
                
                # Get LiDAR points
                lidar_points = sample['lidar']  # (N, 3)
                
                # Get ground truth boxes (in camera coords)
                gt_boxes_camera = targets[0]['boxes_3d'].cpu().numpy()
                
                # Load calibration
                calibs = get_calib(ROOT_DIR, 'training', from_idx=sample['idx'], count=1)
                calib = calibs[0] if len(calibs) > 0 else {}
                
                # Transform GT boxes to LiDAR coordinates
                gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calib)
                
                # Get predictions (already in LiDAR coords)
                if len(proposals_list) > 0:
                    pred_boxes = proposals_list[0].cpu().numpy()  # (K, 7)
                    pred_scores = scores_list[0].cpu().numpy()  # (K,)
                    
                    # Filter by score threshold
                    score_thresh = 0.05
                    keep = pred_scores >= score_thresh
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    
                    # Apply NMS
                    if len(pred_boxes) > 0:
                        pred_boxes, pred_scores = apply_nms_to_boxes(
                            pred_boxes, pred_scores, iou_threshold=0.5
                        )
                else:
                    pred_boxes = np.zeros((0, 7))
                    pred_scores = np.array([])
                
                # Visualize
                print(f"\n[{sample_num+1}/{num_samples}] Epoch {epoch} - Sample {sample['idx']}")
                visualize_predictions(
                    lidar_points, 
                    gt_boxes_lidar, 
                    pred_boxes, 
                    sample_idx=sample['idx'],
                    pred_scores=pred_scores
                )
                
            except Exception as e:
                print(f"⚠️  Error visualizing sample {sample['idx']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    model.train()
    print(f"\n{'='*70}")
    print(f"  ✓ Visualization complete for Epoch {epoch}")
    print(f"{'='*70}\n")


def visualize_with_model(model, dataloader, device, num_samples=5, 
                        apply_nms=True, nms_threshold=0.5, score_threshold=0.05):
    """
    Visualize predictions using ACTUAL MODEL outputs.
    
    Args:
        model: Trained model
        dataloader: DataLoader with V2XRadarDataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
        apply_nms: Whether to apply NMS to predictions
        nms_threshold: IoU threshold for NMS
        score_threshold: Minimum confidence score for predictions
    """
    import spconv.pytorch as spconv
    from trail import voxelize_lidar, voxelize_radar, prepare_images, parse_labels_to_targets
    
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"VISUALIZING {num_samples} SAMPLES WITH REAL MODEL PREDICTIONS")
    print(f"Score threshold: {score_threshold}")
    print(f"NMS: {'ENABLED' if apply_nms else 'DISABLED'} (threshold={nms_threshold})")
    print(f"{'='*70}\n")
    
    samples_shown = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_shown >= num_samples:
                break
            
            # Voxelize LiDAR
            lidar_result = voxelize_lidar(batch['lidar'])
            if lidar_result is None:
                continue
            
            lidar_feat, lidar_coords, lidar_shape, lidar_bs = lidar_result
            
            # Voxelize Radar
            radar_result = voxelize_radar(batch['radar'])
            if radar_result is None:
                continue
            
            radar_feat, radar_coords, radar_shape, radar_bs = radar_result
            
            # Move to device
            lidar_feat = lidar_feat.to(device)
            lidar_coords = lidar_coords.to(device).int()
            radar_feat = radar_feat.to(device)
            radar_coords = radar_coords.to(device).int()
            
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
            
            # ===== REAL MODEL FORWARD PASS =====
            outputs = model(lidar_sparse, radar_sparse, images, targets=None, training=False)
            
            # Get REAL predictions from model
            proposals_list = outputs['detections']['proposals']
            scores_list = outputs['detections']['scores']
            
            # Parse ground truth (in camera coordinates)
            targets = parse_labels_to_targets(batch['label'])
            
            # Get calibration data
            ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
            calibs = get_calib(ROOT_DIR, 'training', from_idx=batch['idx'][0], count=len(batch['lidar']))
            
            # Visualize each sample in batch
            for i in range(len(batch['lidar'])):
                if samples_shown >= num_samples:
                    break
                
                # Get LiDAR points
                lidar_points = batch['lidar'][i]  # (N, 3)
                
                # Get ground truth boxes (in camera coords)
                gt_boxes_camera = targets[i]['boxes_3d'].cpu().numpy()
                
                # Transform GT boxes to LiDAR coordinates
                calib = calibs[i] if i < len(calibs) else calibs[0]
                gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calib)
                
                # Get REAL predictions from model (already in LiDAR coords)
                if i < len(proposals_list):
                    pred_boxes = proposals_list[i].cpu().numpy()  # (K, 7)
                    pred_scores = scores_list[i].cpu().numpy()  # (K,)
                    
                    # Filter by score threshold
                    keep = pred_scores >= score_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    
                    print(f"\nSample {batch['idx'][i]}: {len(pred_boxes)} predictions (score >= {score_threshold})")
                    
                    # Apply NMS to remove overlapping predictions
                    if apply_nms and len(pred_boxes) > 0:
                        print(f"  Before NMS: {len(pred_boxes)} boxes")
                        pred_boxes, pred_scores = apply_nms_to_boxes(
                            pred_boxes, pred_scores, nms_threshold
                        )
                        print(f"  After NMS: {len(pred_boxes)} boxes")
                else:
                    pred_boxes = np.zeros((0, 7))
                    pred_scores = np.array([])
                
                # Visualize with REAL predictions
                visualize_predictions(
                    lidar_points, 
                    gt_boxes_lidar, 
                    pred_boxes, 
                    sample_idx=batch['idx'][i],
                    pred_scores=pred_scores
                )
                
                samples_shown += 1
    
    model.train()
    print(f"\n✓ Visualized {samples_shown} samples with REAL model predictions")


if __name__ == "__main__":
    """
    Main script to visualize model predictions.
    Load your trained model and run visualization.
    """
    
    print("\n" + "="*70)
    print("REAL MODEL PREDICTION VISUALIZATION")
    print("="*70 + "\n")
    
    # Setup
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Root directory: {ROOT_DIR}\n")
    
    # Load trained model
    from trail import MultiModalDetectionNetwork, V2XRadarDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # Initialize model
    model = MultiModalDetectionNetwork(
        lidar_dim=128, 
        radar_dim=128, 
        image_dim=128
    ).to(device)
    
    # Load trained weights
    checkpoint_path = 'checkpoint_epoch_10.pth'  # or your checkpoint path
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully\n")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model (predictions will be random)\n")
    
    # Create dataset and dataloader
    dataset = V2XRadarDataset(ROOT_DIR, split='training', num_samples=50)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Visualize with REAL model predictions
    visualize_with_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=5,
        apply_nms=True,
        nms_threshold=0.5,
        score_threshold=0.05
    )
    
    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print("="*70)