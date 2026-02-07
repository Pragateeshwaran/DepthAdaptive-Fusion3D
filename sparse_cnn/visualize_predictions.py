"""
VISUALIZATION WITH FIXED COORDINATE SYSTEM
"""
import numpy as np
import open3d as o3d
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
CHECKPOINT_PATH = r'F:\Work\DeepLearning\Research\checkpoint_epoch_40.pth'  # Use the overfitted model

# Must match training defaults in sparse_cnn/trail.py
LIDAR_VOXEL_SIZE = (0.1, 0.1, 0.1)
RADAR_VOXEL_SIZE = (0.2, 0.2, 0.2)
POINT_CLOUD_RANGE = (-50, -50, -3, 50, 50, 5)

# Visualization-only: avoid dropping all boxes when scores are very low
FORCE_SHOW_PREDICTIONS = True


def transform_boxes_camera_to_lidar(boxes_camera, calib):
    """Transform boxes from camera to LiDAR coordinates."""
    if len(boxes_camera) == 0:
        return boxes_camera
    
    Tr_velo_to_cam = calib.get('Tr_velo_to_cam', np.eye(4))
    
    if Tr_velo_to_cam.shape == (12,):
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    elif Tr_velo_to_cam.shape == (3, 4):
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
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


def create_bbox_lines():
    """Line indices for 3D box."""
    return [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]


def get_box_corners(center, size, rotation):
    """Get 8 corners of 3D box."""
    x, y, z = center
    h, w, l = size
    
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2],
    ])
    
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rot_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    
    corners = corners @ rot_matrix.T + np.array([x, y, z])
    return corners


def create_bbox_lineset(boxes, color=[1, 0, 0], line_width=2.0):
    """Create Open3D LineSet for boxes."""
    if len(boxes) == 0:
        return o3d.geometry.LineSet()
    
    all_points, all_lines = [], []
    line_template = create_bbox_lines()
    point_offset = 0
    
    for box in boxes:
        x, y, z, h, w, l, rot = box
        corners = get_box_corners([x, y, z], [h, w, l], rot)
        all_points.append(corners)
        
        for line in line_template:
            all_lines.append([line[0] + point_offset, line[1] + point_offset])
        point_offset += 8
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    lineset.lines = o3d.utility.Vector2iVector(np.array(all_lines))
    lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(all_lines))])
    
    return lineset


def to_tensor(data, dtype=torch.float32, device='cuda'):
    """Convert data to tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype).to(device)
    else:
        return torch.from_numpy(data).to(dtype).to(device)


def load_model_and_predict(sample_idx, checkpoint_path):
    """Load model and get predictions."""
    print(f"\n{'='*80}")
    print(f"🔄 LOADING MODEL AND RUNNING INFERENCE")
    print(f"{'='*80}")
    
    from data_loader import get_LiDAR, get_radar, get_images, get_labels, get_calib
    from trail import MultiModalDetectionNetwork, voxelize_lidar_proper, voxelize_radar_proper
    from rpn_refinement import parse_label_line
    import spconv.pytorch as spconv
    from PIL import Image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None, None, None
    
    model = MultiModalDetectionNetwork(lidar_dim=128, radar_dim=128, image_dim=128).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    model.eval()
    
    # Load data
    print(f"\n📥 Loading sample {sample_idx}...")
    lidar_data = get_LiDAR(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    radar_data = get_radar(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    image_data = get_images(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    labels = get_labels(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    calibs = get_calib(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    
    if len(lidar_data) == 0:
        print(f"❌ No data found")
        return None, None, None, None
    
    lidar_points = lidar_data[0]
    radar_points = radar_data[0]
    image = image_data[0]
    
    print(f"✅ Data loaded:")
    print(f"   LiDAR points: {len(lidar_points):,}")
    print(f"   LiDAR X range: [{lidar_points[:, 0].min():.1f}, {lidar_points[:, 0].max():.1f}]")
    print(f"   LiDAR Y range: [{lidar_points[:, 1].min():.1f}, {lidar_points[:, 1].max():.1f}]")
    
    # Parse GT boxes
    gt_boxes_camera = []
    for line in labels[0]:
        if isinstance(line, str) and line.strip():
            parsed = parse_label_line(line)
            obj_type = parsed['type'].lower()
            if obj_type in ['car', 'truck', 'pedestrian', 'cyclist']:
                box = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
                gt_boxes_camera.append(box)
    
    gt_boxes_camera = np.array(gt_boxes_camera, dtype=np.float32)
    gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calibs[0])
    
    print(f"✅ Ground truth: {len(gt_boxes_lidar)} boxes")
    print(f"   GT X range: [{gt_boxes_lidar[:, 0].min():.1f}, {gt_boxes_lidar[:, 0].max():.1f}]")
    print(f"   GT Y range: [{gt_boxes_lidar[:, 1].min():.1f}, {gt_boxes_lidar[:, 1].max():.1f}]")
    
    # Prepare input with SAME parameters as training
    print(f"\n🔄 Preparing model input with training parameters...")
    
    # Voxelize with SAME parameters as training
    lidar_feat, lidar_coords, lidar_shape, lidar_bs = voxelize_lidar_proper(
        [lidar_points],
        voxel_size=LIDAR_VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE
    )
    
    radar_feat, radar_coords, radar_shape, radar_bs = voxelize_radar_proper(
        [radar_points],
        voxel_size=RADAR_VOXEL_SIZE,
        point_cloud_range=POINT_CLOUD_RANGE
    )
    
    print(f"   LiDAR voxels: {len(lidar_feat):,}")
    print(f"   Radar voxels: {len(radar_feat):,}")
    
    # Create sparse tensors
    lidar_sparse = spconv.SparseConvTensor(
        features=to_tensor(lidar_feat, torch.float32, device),
        indices=to_tensor(lidar_coords, torch.int32, device),
        spatial_shape=lidar_shape,
        batch_size=lidar_bs
    )
    
    radar_sparse = spconv.SparseConvTensor(
        features=to_tensor(radar_feat, torch.float32, device),
        indices=to_tensor(radar_coords, torch.int32, device),
        spatial_shape=radar_shape,
        batch_size=radar_bs
    )
    
    # Prepare image
    image_resized = np.array(Image.fromarray(image).resize((1242, 375)))
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # Run inference
    print(f"\n🚀 Running inference...")
    try:
        if FORCE_SHOW_PREDICTIONS and hasattr(model, 'detector'):
            model.detector.proposal_generator.score_thresh = 0.0
            print("FORCE_SHOW_PREDICTIONS=True: score_thresh set to 0.0 for visualization")
        with torch.no_grad():
            outputs = model(lidar_sparse, radar_sparse, image_tensor, training=False)
            print("00000000000000000000000000000000000000000000000000000000000000000000000000000")
            print(outputs)
            proposals = outputs['detections']['proposals'][0]
            scores = outputs['detections']['scores'][0]
        
        if len(proposals) > 0:
            pred_boxes = proposals.cpu().numpy()
            pred_scores = scores.cpu().numpy()
            print(f"   Score range: [{pred_scores.min():.6f}, {pred_scores.max():.6f}]")
            
            print(f"\n✅ Predictions: {len(pred_boxes)} boxes")
            print(f"   Pred X range: [{pred_boxes[:, 0].min():.1f}, {pred_boxes[:, 0].max():.1f}]")
            print(f"   Pred Y range: [{pred_boxes[:, 1].min():.1f}, {pred_boxes[:, 1].max():.1f}]")
            
            # 🔍 COORDINATE DIAGNOSTIC
            print(f"\n{'='*80}")
            print(f"🔍 COORDINATE DIAGNOSTIC")
            print(f"{'='*80}")
            
            if len(gt_boxes_lidar) > 0:
                gt_center = gt_boxes_lidar[:, :3].mean(axis=0)
                pred_center = pred_boxes[:, :3].mean(axis=0)
                distance = np.linalg.norm(gt_center - pred_center)
                
                print(f"Ground Truth Center: X={gt_center[0]:.1f}, Y={gt_center[1]:.1f}")
                print(f"Prediction Center:   X={pred_center[0]:.1f}, Y={pred_center[1]:.1f}")
                print(f"Distance between centers: {distance:.1f}m")
                
                if distance > 50:
                    print(f"\n⚠️  CRITICAL: Predictions are {distance:.0f}m away from GT!")
                    print(f"   Coordinate system still mismatched!")
                elif distance < 10:
                    print(f"\n✅ GOOD: Predictions are close to GT ({distance:.1f}m)")
                    print(f"   Coordinate system is CORRECT!")
            
            print(f"{'='*80}\n")
            
        else:
            pred_boxes = np.zeros((0, 7))
            pred_scores = np.array([])
            print(f"⚠️  No predictions")
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        pred_boxes = np.zeros((0, 7))
        pred_scores = np.array([])
    
    return lidar_points, gt_boxes_lidar, pred_boxes, pred_scores


def visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=0):
    """Visualize point cloud with GT (green) and predictions (red)."""
    print(f"\n{'='*80}")
    print(f"🎨 CREATING VISUALIZATION")
    print(f"{'='*80}")
    
    # Filter LiDAR points to reasonable range
    mask_x = (lidar_points[:, 0] >= POINT_CLOUD_RANGE[0]) & (lidar_points[:, 0] <= POINT_CLOUD_RANGE[3])
    mask_y = (lidar_points[:, 1] >= POINT_CLOUD_RANGE[1]) & (lidar_points[:, 1] <= POINT_CLOUD_RANGE[4])
    mask = mask_x & mask_y
    lidar_points_filtered = lidar_points[mask]
    
    print(f"📊 Visualization data:")
    print(f"   LiDAR points: {len(lidar_points_filtered):,} (filtered to POINT_CLOUD_RANGE)")
    print(f"   GT boxes: {len(gt_boxes)}")
    print(f"   Pred boxes: {len(pred_boxes)}")
    
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points_filtered[:, :3])
    
    # Color by height
    z_values = lidar_points_filtered[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    if z_max - z_min > 0:
        z_norm = (z_values - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z_values)
    colors = np.stack([z_norm * 0.7, z_norm * 0.7, z_norm * 0.7], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Boxes
    gt_lineset = create_bbox_lineset(gt_boxes, color=[0, 1, 0])      # GREEN
    pred_lineset = create_bbox_lineset(pred_boxes, color=[1, 0, 0])  # RED
    
    # Coordinate frame
    if len(gt_boxes) > 0:
        center = gt_boxes[:, :3].mean(axis=0)
    else:
        center = np.array([0, 0, 0])
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10.0, origin=center
    )
    
    geometries = [pcd, coord_frame]
    if len(gt_boxes) > 0:
        geometries.append(gt_lineset)
    if len(pred_boxes) > 0:
        geometries.append(pred_lineset)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Sample {sample_idx} | GREEN=GT | RED=Predictions",
        width=1600,
        height=900
    )
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.3)
    ctr.set_front([0.3, 0, -0.95])
    ctr.set_lookat(center.tolist())
    ctr.set_up([0, 0, 1])
    
    print(f"\n🎥 Camera positioned at: {center}")
    print(f"📺 Opening window...")
    print(f"\nLEGEND:")
    print(f"  🟢 GREEN boxes = Ground Truth ({len(gt_boxes)} boxes)")
    print(f"  🔴 RED boxes = Model Predictions ({len(pred_boxes)} boxes)")
    print(f"  ⚫ Gray points = LiDAR point cloud")
    print(f"  📍 RGB axes = Coordinate frame")
    print(f"\nCONTROLS:")
    print(f"  - LEFT CLICK + DRAG: Rotate")
    print(f"  - SCROLL: Zoom")
    print(f"  - RIGHT CLICK + DRAG: Pan")
    print(f"  - Press 'Q': Close")
    print(f"{'='*80}\n")
    
    vis.run()
    vis.destroy_window()
    
    print(f"\n✅ Visualization closed")


if __name__ == "__main__":
    print("="*80)
    print("3D DETECTION VISUALIZATION WITH FIXED COORDINATES")
    print("="*80)
    
    # Load and predict
    lidar_points, gt_boxes, pred_boxes, scores = load_model_and_predict(0, CHECKPOINT_PATH)
    
    if lidar_points is not None:
        # Visualize
        visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=0)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Ground Truth: {len(gt_boxes)} boxes")
        print(f"Predictions: {len(pred_boxes)} boxes")
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            gt_center = gt_boxes[:, :3].mean(axis=0)
            pred_center = pred_boxes[:, :3].mean(axis=0)
            distance = np.linalg.norm(gt_center - pred_center)
            
            print(f"\nCoordinate Check:")
            print(f"  GT center: [{gt_center[0]:.1f}, {gt_center[1]:.1f}]")
            print(f"  Pred center: [{pred_center[0]:.1f}, {pred_center[1]:.1f}]")
            print(f"  Distance: {distance:.1f}m")
            
            if distance < 10:
                print(f"\n✅ SUCCESS: Predictions align with GT!")
                print(f"  The coordinate system is now correct!")
            else:
                print(f"\n⚠️  Still some misalignment: {distance:.1f}m")
                print(f"  May need to adjust anchor sizes or training parameters")
        
        print(f"{'='*80}\n")
    else:
        print(f"\n❌ Failed to load model or data")
