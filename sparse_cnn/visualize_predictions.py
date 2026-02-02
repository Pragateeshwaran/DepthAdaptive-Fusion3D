"""
COMPLETE FIX for No Bounding Boxes Issue

Root cause: Model outputs empty prediction tensors (0 predictions)
This happens when ALL predictions are filtered out by score threshold.

Solutions implemented:
1. Gracefully handle empty tensors
2. Show diagnostic information about why predictions are empty
3. Provide multiple visualization modes
4. Add fallback to ground truth only
"""

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
    """Transform boxes from camera coordinates to LiDAR coordinates."""
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


def apply_nms_to_boxes(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
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
    elif len(pred_boxes) == 0:
        print(f"⚠️  NO PREDICTIONS - Model produced empty output")
    
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


def diagnose_empty_predictions(model, dataloader, device):
    """
    Diagnostic function to understand why predictions are empty.
    """
    import spconv.pytorch as spconv
    from trail import voxelize_lidar, voxelize_radar, prepare_images
    
    print("\n" + "="*80)
    print("🔍 DIAGNOSTIC MODE - Finding Why Predictions Are Empty")
    print("="*80 + "\n")
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n{'='*80}")
            print(f"Analyzing Batch {batch_idx}")
            print(f"{'='*80}\n")
            
            # Voxelize
            lidar_result = voxelize_lidar(batch['lidar'])
            radar_result = voxelize_radar(batch['radar'])
            
            if lidar_result is None or radar_result is None:
                print("❌ Voxelization failed")
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
            
            images = prepare_images(batch['image']).to(device)
            
            # Forward pass
            print("🔄 Running forward pass...")
            try:
                outputs = model(lidar_sparse, radar_sparse, images, targets=None, training=False)
                
                # Check RPN outputs
                if 'detections' in outputs:
                    detections = outputs['detections']
                    
                    print(f"\n✓ Forward pass successful")
                    print(f"   Detection keys: {detections.keys()}")
                    
                    proposals_list = detections['proposals']
                    scores_list = detections['scores']
                    
                    print(f"\n📊 Predictions per sample:")
                    for i in range(len(proposals_list)):
                        proposals = proposals_list[i]
                        scores = scores_list[i]
                        
                        print(f"\n   Sample {i}:")
                        print(f"      Proposals shape: {proposals.shape}")
                        print(f"      Scores shape: {scores.shape}")
                        
                        if len(scores) == 0:
                            print(f"      ❌ EMPTY OUTPUT - No predictions generated!")
                            print(f"      ")
                            print(f"      Possible causes:")
                            print(f"      1. Score threshold in ProposalGenerator is too high")
                            print(f"      2. All RPN predictions filtered out")
                            print(f"      3. Model not generating any positive predictions")
                        else:
                            print(f"      Score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")
                            print(f"      Mean score: {scores.mean().item():.6f}")
                            
                            # Check how many pass different thresholds
                            for thresh in [0.0, 0.001, 0.01, 0.05, 0.1, 0.3]:
                                count = (scores >= thresh).sum().item()
                                print(f"      Predictions >= {thresh}: {count}/{len(scores)}")
                    
                    # Check RPN raw outputs if available
                    if 'losses' in detections and 'num_pos_anchors' in detections['losses']:
                        num_pos = detections['losses']['num_pos_anchors']
                        print(f"\n   RPN Statistics:")
                        print(f"      Positive anchors: {num_pos:.1f}")
                        
                else:
                    print(f"❌ No 'detections' in output!")
                    print(f"   Output keys: {outputs.keys()}")
                    
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Only analyze first batch
            break
    
    print("\n" + "="*80)
    print("🔍 DIAGNOSIS COMPLETE")
    print("="*80 + "\n")
    
    print("💡 SOLUTIONS:")
    print("1. If 'EMPTY OUTPUT': Model's score_thresh in ProposalGenerator is filtering everything")
    print("   → Solution: Lower score_thresh in rpn_refinement.py ProposalGenerator.__init__")
    print("   → Change from 0.05 to 0.0 or 0.001")
    print("")
    print("2. If score range is very low (< 0.01): Model is undertrained")
    print("   → Solution: Train more epochs OR lower score threshold")
    print("")
    print("3. If positive anchors = 0: GT boxes don't match anchors")
    print("   → Solution: Check coordinate system, anchor sizes")
    print("")


def visualize_with_model_ULTIMATE_FIX(model, dataloader, device, num_samples=5, 
                                      apply_nms=True, nms_threshold=0.5,
                                      gt_only=False, debug=True):
    """
    ULTIMATE FIX - Handles empty predictions gracefully.
    """
    import spconv.pytorch as spconv
    from trail import voxelize_lidar, voxelize_radar, prepare_images, parse_labels_to_targets
    
    model.eval()
    
    mode = "GROUND TRUTH ONLY" if gt_only else "WITH PREDICTIONS"
    print(f"\n{'='*70}")
    print(f"VISUALIZATION - {mode}")
    print(f"NMS: {'ENABLED' if apply_nms else 'DISABLED'} (threshold={nms_threshold})")
    print(f"Debug mode: {debug}")
    print(f"{'='*70}\n")
    
    samples_shown = 0
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    
    total_empty_predictions = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if samples_shown >= num_samples:
                break
            
            if debug:
                print(f"\n{'='*70}")
                print(f"Processing batch {batch_idx}")
                print(f"{'='*70}")
            
            # Voxelize
            lidar_result = voxelize_lidar(batch['lidar'])
            radar_result = voxelize_radar(batch['radar'])
            
            if lidar_result is None or radar_result is None:
                if debug:
                    print("⚠️ Voxelization failed, skipping batch")
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
            
            images = prepare_images(batch['image']).to(device)
            
            # Forward pass (only if not gt_only mode)
            proposals_list = None
            scores_list = None
            
            if not gt_only:
                try:
                    outputs = model(lidar_sparse, radar_sparse, images, targets=None, training=False)
                    proposals_list = outputs['detections']['proposals']
                    scores_list = outputs['detections']['scores']
                    print(proposals_list)
                    if debug:
                        print(f"\n📊 Model outputs:")
                        print(f"  - Proposals: {len(proposals_list)} samples")
                        for i, (proposals, scores) in enumerate(zip(proposals_list, scores_list)):
                            if len(scores) == 0:
                                print(f"  - Sample {i}: EMPTY (0 predictions)")
                                total_empty_predictions += 1
                            else:
                                print(f"  - Sample {i}: {len(scores)} predictions, scores in [{scores.min():.6f}, {scores.max():.6f}]")
                except Exception as e:
                    print(f"❌ Model forward pass failed: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    gt_only = True  # Fall back to GT only
            
            # Parse ground truth
            targets = parse_labels_to_targets(batch['label'])
            
            # Get calibration
            try:
                calibs = get_calib(ROOT_DIR, 'training', from_idx=batch['idx'][0], count=len(batch['lidar']))
            except Exception as e:
                print(f"⚠️ Calibration loading failed: {e}")
                calibs = [{}] * len(batch['lidar'])
            
            # Visualize each sample
            for i in range(len(batch['lidar'])):
                if samples_shown >= num_samples:
                    break
                
                # Get LiDAR points
                lidar_points = batch['lidar'][i]
                
                # Get ground truth boxes
                gt_boxes_camera = targets[i]['boxes_3d'].cpu().numpy()
                calib = calibs[i] if i < len(calibs) else {}
                gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calib)
                
                if debug:
                    print(f"\n📦 Sample {batch['idx'][i]}:")
                    print(f"  - GT boxes: {len(gt_boxes_lidar)}")
                
                # Get predictions
                pred_boxes = np.zeros((0, 7))
                pred_scores = np.array([])
                
                if not gt_only and proposals_list is not None and i < len(proposals_list):
                    proposals = proposals_list[i]
                    scores = scores_list[i]
                    
                    if len(proposals) > 0:
                        pred_boxes = proposals.cpu().numpy()
                        pred_scores = scores.cpu().numpy()
                        
                        if debug:
                            print(f"  - Raw predictions: {len(pred_boxes)}")
                            print(f"  - Score range: [{pred_scores.min():.6f}, {pred_scores.max():.6f}]")
                        
                        # Apply NMS
                        if apply_nms and len(pred_boxes) > 0:
                            pred_boxes, pred_scores = apply_nms_to_boxes(
                                pred_boxes, pred_scores, nms_threshold
                            )
                            if debug:
                                print(f"  - After NMS: {len(pred_boxes)}")
                    else:
                        if debug:
                            print(f"  ⚠️ EMPTY PREDICTIONS from model")
                
                # Visualize
                visualize_predictions(
                    lidar_points, 
                    gt_boxes_lidar, 
                    pred_boxes, 
                    sample_idx=batch['idx'][i],
                    pred_scores=pred_scores if len(pred_scores) > 0 else None
                )
                
                samples_shown += 1
    
    model.train()
    
    print(f"\n{'='*70}")
    print(f"✓ Visualized {samples_shown} samples")
    if total_empty_predictions > 0:
        print(f"⚠️  {total_empty_predictions} samples had EMPTY predictions")
        print(f"   This means the model's score threshold is too high")
        print(f"   OR the model is not generating any predictions")
    print(f"{'='*70}")


if __name__ == "__main__":
    """
    ULTIMATE FIX - Complete diagnostic and visualization
    """
    
    print("\n" + "="*70)
    print("ULTIMATE FIX - Complete Visualization Solution")
    print("="*70 + "\n")
    
    # ⭐ CRITICAL FIX: Monkey-patch ProposalGenerator to use score_thresh=0.0
    print("🔧 Applying critical fix: Lowering score threshold to 0.0...")
    from rpn_refinement import ProposalGenerator
    
    original_init = ProposalGenerator.__init__
    
    def patched_init(self, 
                     pre_nms_top_n_train=1000,
                     pre_nms_top_n_test=500,
                     post_nms_top_n_train=300,
                     post_nms_top_n_test=100,
                     nms_thresh=0.3,
                     score_thresh=0.05):
        # Force score_thresh to 0.0 to show ALL predictions
        print(f"   Overriding score_thresh: {score_thresh} → 0.0")
        original_init(self,
                      pre_nms_top_n_train,
                      pre_nms_top_n_test,
                      post_nms_top_n_train,
                      post_nms_top_n_test,
                      nms_thresh,
                      score_thresh=0.0)  # ⭐ FORCE to 0.0
    
    ProposalGenerator.__init__ = patched_init
    print("✓ ProposalGenerator patched successfully\n")
    
    # Setup
    ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Root directory: {ROOT_DIR}\n")
    
    # Load model
    from trail import MultiModalDetectionNetwork, V2XRadarDataset, collate_fn
    from torch.utils.data import DataLoader
    
    model = MultiModalDetectionNetwork(
        lidar_dim=128, 
        radar_dim=128, 
        image_dim=128
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = r'F:\Work\DeepLearning\Research\checkpoint_epoch_80.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully\n")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model\n")
    
    # Create dataset
    dataset = V2XRadarDataset(ROOT_DIR, split='training', num_samples=10)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # STEP 1: Run diagnostics
    print("\n" + "="*70)
    print("STEP 1: Running Diagnostics")
    print("="*70)
    diagnose_empty_predictions(model, dataloader, device)
    
    # STEP 2: Visualize Ground Truth Only
    print("\n" + "="*70)
    print("STEP 2: Visualizing Ground Truth Only")
    print("(To verify GT boxes work)")
    print("="*70)
    
    visualize_with_model_ULTIMATE_FIX(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=3,
        gt_only=True,
        debug=True
    )
    
    # STEP 3: Visualize with Predictions
    print("\n" + "="*70)
    print("STEP 3: Visualizing with Predictions")
    print("(Will show empty if model produces no output)")
    print("="*70)
    
    visualize_with_model_ULTIMATE_FIX(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=5,
        apply_nms=True,
        nms_threshold=0.5,
        gt_only=False,
        debug=True
    )
    
    print("\n" + "="*70)
    print("✓ Complete visualization finished!")
    print("="*70)
    print("\nIf you saw 'EMPTY PREDICTIONS':")
    print("  → Edit rpn_refinement.py, line ~231")
    print("  → Change: score_thresh=0.05")
    print("  → To:     score_thresh=0.0")
    print("  → This will show ALL predictions regardless of confidence")
    print("="*70)