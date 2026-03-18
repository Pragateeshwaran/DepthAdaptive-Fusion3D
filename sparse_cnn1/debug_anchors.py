import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_labels
from rpn_refinement import AnchorGenerator, parse_label_line, bev_iou

# Load a few samples
ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
labels = get_labels(ROOT_DIR, 'training', from_idx=0, count=5)

print("="*70)
print("DEBUGGING RPN - Why are cls_loss and reg_loss zero?")
print("="*70)

# Parse labels
for i, label_list in enumerate(labels):
    print(f"\n{'='*70}")
    print(f"Sample {i}:")
    print(f"{'='*70}")
    
    if not label_list:
        print("⚠️  EMPTY LABEL FILE!")
        continue
    
    # Print raw label first
    print("\nRaw label lines:")
    for j, line in enumerate(label_list[:3]):  # First 3 lines
        print(f"  Line {j}: {line.strip()}")
    
    boxes_3d = []
    for line in label_list:
        if isinstance(line, str) and line.strip():
            parsed = parse_label_line(line)
            if parsed['type'].lower() == 'car':
                # CRITICAL: Check label format
                # Format should be: [x, y, z, h, w, l, rot]
                box = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
                boxes_3d.append(box)
                
                print(f"\nCar detected:")
                print(f"  Type: {parsed['type']}")
                print(f"  Location (x, y, z): {parsed['location']}")
                print(f"  Dimensions (h, w, l): {parsed['dimensions']}")
                print(f"  Rotation: {parsed['rotation_y']:.3f} rad ({np.degrees(parsed['rotation_y']):.1f}°)")
                print(f"  Box format [x,y,z,h,w,l,rot]: {box}")
    
    if len(boxes_3d) == 0:
        print("\n⚠️  NO CARS FOUND in this sample!")
        continue
    
    # Convert to tensor
    gt_boxes = torch.tensor(boxes_3d, dtype=torch.float32)
    print(f"\n✓ Total cars: {len(boxes_3d)}")
    
    # Check coordinate ranges
    print(f"\n{'='*70}")
    print("GROUND TRUTH COORDINATE RANGES:")
    print(f"{'='*70}")
    print(f"X range: [{gt_boxes[:, 0].min():.2f}, {gt_boxes[:, 0].max():.2f}]")
    print(f"Y range: [{gt_boxes[:, 1].min():.2f}, {gt_boxes[:, 1].max():.2f}]")
    print(f"Z range: [{gt_boxes[:, 2].min():.2f}, {gt_boxes[:, 2].max():.2f}]")
    print(f"H range: [{gt_boxes[:, 3].min():.2f}, {gt_boxes[:, 3].max():.2f}]")
    print(f"W range: [{gt_boxes[:, 4].min():.2f}, {gt_boxes[:, 4].max():.2f}]")
    print(f"L range: [{gt_boxes[:, 5].min():.2f}, {gt_boxes[:, 5].max():.2f}]")
    
    # Generate anchors
    print(f"\n{'='*70}")
    print("ANCHOR GENERATION:")
    print(f"{'='*70}")
    
    anchor_gen = AnchorGenerator(
        anchor_sizes=[(1.5, 1.6, 3.9)],  # (h, w, l)
        anchor_rotations=[0, np.pi/2],
        feature_map_size=(200, 200),
        voxel_size=(0.5, 0.5, 0.5),
        point_cloud_range=(-50, -50, -3, 50, 50, 5)
    )
    
    anchors = anchor_gen.generate_anchors(device='cpu')
    anchors_flat = anchors.view(-1, 7)
    
    print(f"Point cloud range: {anchor_gen.point_cloud_range}")
    print(f"Anchor size (h,w,l): (1.5, 1.6, 3.9)")
    print(f"Total anchors: {anchors_flat.shape[0]:,}")
    
    # Check if GT boxes are within point cloud range
    pc_range = anchor_gen.point_cloud_range
    in_range_x = (gt_boxes[:, 0] >= pc_range[0]) & (gt_boxes[:, 0] <= pc_range[3])
    in_range_y = (gt_boxes[:, 1] >= pc_range[1]) & (gt_boxes[:, 1] <= pc_range[4])
    in_range_z = (gt_boxes[:, 2] >= pc_range[2]) & (gt_boxes[:, 2] <= pc_range[5])
    in_range = in_range_x & in_range_y & in_range_z
    
    print(f"\nGT boxes within point cloud range:")
    print(f"  X in range: {in_range_x.sum().item()}/{len(gt_boxes)}")
    print(f"  Y in range: {in_range_y.sum().item()}/{len(gt_boxes)}")
    print(f"  Z in range: {in_range_z.sum().item()}/{len(gt_boxes)}")
    print(f"  ALL in range: {in_range.sum().item()}/{len(gt_boxes)}")
    
    if in_range.sum() == 0:
        print("\n⚠️⚠️⚠️  CRITICAL: ALL GT boxes are OUTSIDE point cloud range!")
        print("This is why you have ZERO positive anchors!")
        print("\nPossible causes:")
        print("1. Labels are in CAMERA coordinates, not LiDAR coordinates")
        print("2. Point cloud range is incorrect")
        print("3. Need coordinate transformation")
        continue
    
    # Compute IoU
    print(f"\n{'='*70}")
    print("IoU ANALYSIS:")
    print(f"{'='*70}")
    
    ious = bev_iou(anchors_flat, gt_boxes)
    max_ious, max_indices = ious.max(dim=1)
    
    print(f"IoU statistics:")
    print(f"  Max IoU: {max_ious.max().item():.4f}")
    print(f"  Mean IoU: {max_ious.mean().item():.6f}")
    print(f"  Median IoU: {max_ious.median().item():.6f}")
    print(f"  Non-zero IoUs: {(max_ious > 0).sum().item():,} / {len(max_ious):,}")
    
    # Check with different thresholds
    print(f"\nPositive anchors at different thresholds:")
    for thresh in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
        num_pos = (max_ious >= thresh).sum().item()
        pct = 100 * num_pos / len(max_ious)
        print(f"  IoU >= {thresh:.2f}: {num_pos:,} ({pct:.4f}%)")
    
    # Show best anchors
    if max_ious.max() > 0:
        print(f"\n{'='*70}")
        print("TOP 5 BEST MATCHING ANCHORS:")
        print(f"{'='*70}")
        
        top_k = min(5, (max_ious > 0).sum().item())
        if top_k > 0:
            top_ious, top_indices = max_ious.topk(top_k)
            
            for j in range(top_k):
                idx = top_indices[j].item()
                iou = top_ious[j].item()
                anc = anchors_flat[idx]
                gt_idx = max_indices[idx].item()
                gt = gt_boxes[gt_idx]
                
                print(f"\nRank {j+1}: IoU = {iou:.4f}")
                print(f"  Anchor: x={anc[0]:6.2f}, y={anc[1]:6.2f}, z={anc[2]:5.2f}, "
                      f"h={anc[3]:4.2f}, w={anc[4]:4.2f}, l={anc[5]:4.2f}, rot={anc[6]:5.3f}")
                print(f"  GT Box: x={gt[0]:6.2f}, y={gt[1]:6.2f}, z={gt[2]:5.2f}, "
                      f"h={gt[3]:4.2f}, w={gt[4]:4.2f}, l={gt[5]:4.2f}, rot={gt[6]:5.3f}")
                print(f"  Δx={abs(anc[0]-gt[0]):5.2f}, Δy={abs(anc[1]-gt[1]):5.2f}, "
                      f"Δh={abs(anc[3]-gt[3]):4.2f}, Δw={abs(anc[4]-gt[4]):4.2f}, Δl={abs(anc[5]-gt[5]):4.2f}")
    else:
        print("\n⚠️  NO NON-ZERO IoUs FOUND!")
        print("Anchors and GT boxes have ZERO overlap!")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("\n🔍 ROOT CAUSES:")
print("1. If ALL GT boxes outside range → Coordinate system mismatch")
print("2. If max IoU = 0 → GT boxes and anchors don't overlap at all")
print("3. If max IoU < 0.3 → Anchor sizes don't match GT box sizes")

print("\n💡 SOLUTIONS:")
print("1. Check if labels are in camera coordinates (need Tr_velo_to_cam)")
print("2. Verify point_cloud_range matches your LiDAR data")
print("3. Adjust anchor sizes to match actual car dimensions")
print("4. Lower pos_iou_thresh to 0.1 or even 0.05 temporarily")
print("5. Print first LiDAR point cloud to verify coordinate ranges")