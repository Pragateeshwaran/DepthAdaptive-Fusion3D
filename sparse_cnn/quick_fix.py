"""
Diagnose Coordinate Transformation Issue
This will show you EXACTLY where the boxes are before and after transformation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from data_loader import get_labels, get_calib
from rpn_refinement import parse_label_line, AnchorGenerator, bev_iou

ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'

print("="*80)
print("COORDINATE TRANSFORMATION DIAGNOSTIC")
print("="*80)

# Load first sample
labels = get_labels(ROOT_DIR, 'training', from_idx=0, count=1)
calibs = get_calib(ROOT_DIR, 'training', from_idx=0, count=1)

if len(labels) == 0 or len(calibs) == 0:
    print("ERROR: No data loaded!")
    exit(1)

calib = calibs[0]
label_lines = labels[0]

print(f"\n📋 Calibration Info:")
print(f"   Keys: {list(calib.keys())}")
print(f"   Tr_velo_to_cam shape: {calib['Tr_velo_to_cam'].shape}")
print(f"   Tr_velo_to_cam:\n{calib['Tr_velo_to_cam']}")

# Parse labels
class_map = {'pedestrian': 0, 'cyclist': 1, 'car': 2}
boxes_camera = []
for line in label_lines:
    if isinstance(line, str) and line.strip():
        parsed = parse_label_line(line)
        obj_type = parsed['type'].lower()
        if obj_type in class_map:
            box_3d = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
            boxes_camera.append(box_3d)

boxes_camera = np.array(boxes_camera)

print(f"\n📦 BOXES IN CAMERA COORDINATES:")
print(f"   Number of boxes: {len(boxes_camera)}")
if len(boxes_camera) > 0:
    print(f"\n   Box format: [x, y, z, h, w, l, rot]")
    for i, box in enumerate(boxes_camera):
        print(f"   Box {i}: x={box[0]:6.2f}, y={box[1]:6.2f}, z={box[2]:6.2f}, "
              f"h={box[3]:4.2f}, w={box[4]:4.2f}, l={box[5]:4.2f}, rot={box[6]:5.3f}")
    
    print(f"\n   Camera coordinate ranges:")
    print(f"   X: [{boxes_camera[:, 0].min():6.2f}, {boxes_camera[:, 0].max():6.2f}]")
    print(f"   Y: [{boxes_camera[:, 1].min():6.2f}, {boxes_camera[:, 1].max():6.2f}]")
    print(f"   Z: [{boxes_camera[:, 2].min():6.2f}, {boxes_camera[:, 2].max():6.2f}]")

# Now transform to LiDAR
def transform_boxes_camera_to_lidar(boxes_camera, calib):
    """Transform boxes from camera coordinates to LiDAR coordinates."""
    if len(boxes_camera) == 0:
        return boxes_camera
    
    Tr_velo_to_cam = calib.get('Tr_velo_to_cam', np.eye(4))
    
    if Tr_velo_to_cam.shape == (12,):
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    
    print(f"\n🔄 Transformation Matrix (Cam to Velo):")
    print(Tr_cam_to_velo)
    
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

boxes_lidar = transform_boxes_camera_to_lidar(boxes_camera, calib)

print(f"\n📦 BOXES IN LIDAR COORDINATES (AFTER TRANSFORMATION):")
if len(boxes_lidar) > 0:
    print(f"\n   Box format: [x, y, z, h, w, l, rot]")
    for i, box in enumerate(boxes_lidar):
        print(f"   Box {i}: x={box[0]:6.2f}, y={box[1]:6.2f}, z={box[2]:6.2f}, "
              f"h={box[3]:4.2f}, w={box[4]:4.2f}, l={box[5]:4.2f}, rot={box[6]:5.3f}")
    
    print(f"\n   LiDAR coordinate ranges:")
    print(f"   X: [{boxes_lidar[:, 0].min():6.2f}, {boxes_lidar[:, 0].max():6.2f}]")
    print(f"   Y: [{boxes_lidar[:, 1].min():6.2f}, {boxes_lidar[:, 1].max():6.2f}]")
    print(f"   Z: [{boxes_lidar[:, 2].min():6.2f}, {boxes_lidar[:, 2].max():6.2f}]")

# Check against point cloud range
print(f"\n🎯 ANCHOR GENERATOR CONFIG:")
anchor_gen = AnchorGenerator(
    anchor_sizes=[(1.5, 1.6, 3.9)],
    anchor_rotations=[0, np.pi/2],
    feature_map_size=(200, 200),
    voxel_size=(0.5, 0.5, 0.5),
    point_cloud_range=(-50, -50, -3, 50, 50, 5)
)

pc_range = anchor_gen.point_cloud_range
print(f"   Point cloud range: X=[{pc_range[0]}, {pc_range[3]}], "
      f"Y=[{pc_range[1]}, {pc_range[4]}], Z=[{pc_range[2]}, {pc_range[5]}]")
print(f"   Anchor size (h, w, l): (1.5, 1.6, 3.9)")

# Check if boxes are in range
if len(boxes_lidar) > 0:
    boxes_tensor = torch.from_numpy(boxes_lidar).float()
    
    in_range_x = (boxes_tensor[:, 0] >= pc_range[0]) & (boxes_tensor[:, 0] <= pc_range[3])
    in_range_y = (boxes_tensor[:, 1] >= pc_range[1]) & (boxes_tensor[:, 1] <= pc_range[4])
    in_range_z = (boxes_tensor[:, 2] >= pc_range[2]) & (boxes_tensor[:, 2] <= pc_range[5])
    in_range = in_range_x & in_range_y & in_range_z
    
    print(f"\n✅ BOXES IN POINT CLOUD RANGE CHECK:")
    print(f"   X in range: {in_range_x.sum().item()}/{len(boxes_tensor)}")
    print(f"   Y in range: {in_range_y.sum().item()}/{len(boxes_tensor)}")
    print(f"   Z in range: {in_range_z.sum().item()}/{len(boxes_tensor)}")
    print(f"   ALL in range: {in_range.sum().item()}/{len(boxes_tensor)}")
    
    if in_range.sum() == 0:
        print(f"\n❌ PROBLEM FOUND: ALL boxes are OUTSIDE point cloud range!")
        print(f"\n   Boxes are at:")
        for i, box in enumerate(boxes_lidar):
            print(f"   Box {i}: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})")
        print(f"\n   But point cloud range is:")
        print(f"   X: [{pc_range[0]}, {pc_range[3]}]")
        print(f"   Y: [{pc_range[1]}, {pc_range[4]}]")
        print(f"   Z: [{pc_range[2]}, {pc_range[5]}]")
    else:
        print(f"\n✅ {in_range.sum().item()} boxes are IN range - Good!")
        
        # Check IoU with anchors
        print(f"\n🔍 CHECKING IoU WITH ANCHORS:")
        anchors = anchor_gen.generate_anchors(device='cpu')
        anchors_flat = anchors.view(-1, 7)
        
        ious = bev_iou(anchors_flat, boxes_tensor)
        max_ious, _ = ious.max(dim=1)
        
        print(f"   Total anchors: {len(anchors_flat):,}")
        print(f"   Max IoU: {max_ious.max().item():.4f}")
        print(f"   Mean IoU: {max_ious.mean().item():.6f}")
        print(f"   Anchors with IoU > 0: {(max_ious > 0).sum().item():,}")
        print(f"   Anchors with IoU > 0.3: {(max_ious >= 0.3).sum().item():,}")
        print(f"   Anchors with IoU > 0.5: {(max_ious >= 0.5).sum().item():,}")
        
        if max_ious.max() < 0.3:
            print(f"\n⚠️  WARNING: Max IoU < 0.3!")
            print(f"   This means anchors don't match boxes well")
            print(f"   Possible fixes:")
            print(f"   1. Adjust anchor sizes to match cars better")
            print(f"   2. Lower pos_iou_thresh to 0.1 or 0.15")

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print("="*80)

print(f"\n📊 SUMMARY:")
if len(boxes_lidar) > 0:
    boxes_tensor = torch.from_numpy(boxes_lidar).float()
    in_range_x = (boxes_tensor[:, 0] >= pc_range[0]) & (boxes_tensor[:, 0] <= pc_range[3])
    in_range_y = (boxes_tensor[:, 1] >= pc_range[1]) & (boxes_tensor[:, 1] <= pc_range[4])
    in_range_z = (boxes_tensor[:, 2] >= pc_range[2]) & (boxes_tensor[:, 2] <= pc_range[5])
    in_range = in_range_x & in_range_y & in_range_z
    
    if in_range.sum() == 0:
        print("❌ ISSUE: Boxes are outside point cloud range after transformation")
        print("   → Coordinate transformation might be wrong")
        print("   → Or point cloud range needs adjustment")
    else:
        anchors = anchor_gen.generate_anchors(device='cpu')
        anchors_flat = anchors.view(-1, 7)
        ious = bev_iou(anchors_flat, boxes_tensor)
        max_ious, _ = ious.max(dim=1)
        
        if max_ious.max() < 0.1:
            print("❌ ISSUE: Anchors have near-zero IoU with boxes")
            print("   → Anchor sizes don't match car sizes")
            print("   → Need to adjust anchor_sizes in AnchorGenerator")
        elif max_ious.max() < 0.3:
            print("⚠️  WARNING: Low max IoU (< 0.3)")
            print("   → Model will have very few positive anchors")
            print("   → Lower pos_iou_thresh or adjust anchor sizes")
        else:
            print("✅ Boxes are in range and have good IoU with anchors")
            print(f"   → Max IoU: {max_ious.max().item():.4f}")
            print(f"   → This should work - check training loop")
else:
    print("❌ No boxes found in labels")