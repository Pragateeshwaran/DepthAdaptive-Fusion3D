"""
QUICK TEST to overfit on ONE sample and verify anchors match GT
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_LiDAR, get_labels, get_calib
from rpn_refinement import AnchorGenerator, parse_label_line

ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'

# Load ONE sample
print("="*80)
print("OVERFIT TEST: Checking anchor-GT alignment")
print("="*80)

lidar_data = get_LiDAR(ROOT_DIR, 'training', from_idx=0, count=1)
labels = get_labels(ROOT_DIR, 'training', from_idx=0, count=1)
calibs = get_calib(ROOT_DIR, 'training', from_idx=0, count=1)

lidar_points = lidar_data[0]
print(f"\n📊 LiDAR data:")
print(f"   Points: {len(lidar_points):,}")
print(f"   X range: [{lidar_points[:, 0].min():.1f}, {lidar_points[:, 0].max():.1f}]")
print(f"   Y range: [{lidar_points[:, 1].min():.1f}, {lidar_points[:, 1].max():.1f}]")

# Parse GT boxes
gt_boxes_camera = []
for line in labels[0]:
    if isinstance(line, str) and line.strip():
        parsed = parse_label_line(line)
        if parsed['type'].lower() == 'car':
            box = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
            gt_boxes_camera.append(box)

gt_boxes_camera = np.array(gt_boxes_camera, dtype=np.float32)

# Transform to LiDAR coordinates
def transform_boxes_camera_to_lidar(boxes_camera, calib):
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

gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calibs[0])

print(f"\n📊 GT boxes (LiDAR coordinates):")
for i, box in enumerate(gt_boxes_lidar):
    print(f"   Box {i}: X={box[0]:.1f}, Y={box[1]:.1f}, Z={box[2]:.1f}")

# Generate anchors with ACTUAL LiDAR range
anchor_gen = AnchorGenerator(
    anchor_sizes=[(1.5, 1.6, 3.9)],
    anchor_rotations=[0, np.pi/2],
    feature_map_size=(200, 200),
    voxel_size=(0.5, 0.5, 0.5),
    point_cloud_range=(-200, -100, -5, 200, 150, 10)  # ACTUAL RANGE
)

anchors = anchor_gen.generate_anchors(device='cpu')
anchors_flat = anchors.view(-1, 7)

print(f"\n📊 Anchors:")
print(f"   Count: {anchors_flat.shape[0]:,}")
print(f"   X range: [{anchors_flat[:, 0].min():.1f}, {anchors_flat[:, 0].max():.1f}]")
print(f"   Y range: [{anchors_flat[:, 1].min():.1f}, {anchors_flat[:, 1].max():.1f}]")

# Check distances between anchors and GT boxes
if len(gt_boxes_lidar) > 0:
    print(f"\n🔍 Distance analysis:")
    for i, gt_box in enumerate(gt_boxes_lidar):
        gt_center = gt_box[:2]  # X, Y
        distances = np.sqrt(
            (anchors_flat[:, 0].numpy() - gt_center[0])**2 + 
            (anchors_flat[:, 1].numpy() - gt_center[1])**2
        )
        
        min_dist = distances.min()
        closest_idx = distances.argmin()
        closest_anchor = anchors_flat[closest_idx]
        
        print(f"   GT Box {i}:")
        print(f"     Center: X={gt_center[0]:.1f}, Y={gt_center[1]:.1f}")
        print(f"     Closest anchor: X={closest_anchor[0]:.1f}, Y={closest_anchor[1]:.1f}")
        print(f"     Distance: {min_dist:.1f}m")
        
        if min_dist < 5:
            print(f"     ✅ GOOD: Anchor within {min_dist:.1f}m of GT")
        else:
            print(f"     ⚠️  WARNING: Anchor {min_dist:.1f}m away from GT")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# LiDAR points and anchors
ax1.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.1, alpha=0.3, c='blue', label='LiDAR')
ax1.scatter(anchors_flat[::1000, 0], anchors_flat[::1000, 1], s=10, c='red', alpha=0.5, label='Anchors')

# GT boxes (as circles)
if len(gt_boxes_lidar) > 0:
    ax1.scatter(gt_boxes_lidar[:, 0], gt_boxes_lidar[:, 1], s=100, c='green', marker='o', 
                edgecolors='black', linewidth=2, label='GT Box Centers')

ax1.set_xlim(-200, 200)
ax1.set_ylim(-100, 150)
ax1.set_xlabel('X (forward)')
ax1.set_ylabel('Y (left)')
ax1.set_title('LiDAR Points with Anchors and GT Boxes')
ax1.legend()
ax1.grid(True)

# Zoomed view
if len(gt_boxes_lidar) > 0:
    gt_center_x = gt_boxes_lidar[:, 0].mean()
    gt_center_y = gt_boxes_lidar[:, 1].mean()
    
    ax2.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.5, alpha=0.3, c='blue')
    ax2.scatter(anchors_flat[:, 0], anchors_flat[:, 1], s=5, c='red', alpha=0.5)
    ax2.scatter(gt_boxes_lidar[:, 0], gt_boxes_lidar[:, 1], s=100, c='green', marker='o', 
                edgecolors='black', linewidth=2)
    
    ax2.set_xlim(gt_center_x - 20, gt_center_x + 20)
    ax2.set_ylim(gt_center_y - 20, gt_center_y + 20)
    ax2.set_xlabel('X (forward)')
    ax2.set_ylabel('Y (left)')
    ax2.set_title('Zoomed View (20m around GT)')
    ax2.grid(True)

plt.tight_layout()
plt.savefig('overfit_test.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to overfit_test.png")

# Final diagnosis
print(f"\n{'='*80}")
print("DIAGNOSIS:")
if len(gt_boxes_lidar) == 0:
    print("❌ NO GT BOXES FOUND! Check label parsing.")
elif anchors_flat.shape[0] == 0:
    print("❌ NO ANCHORS GENERATED! Check anchor generation.")
else:
    print("✅ System ready for overfitting")
    print(f"   GT boxes: {len(gt_boxes_lidar)}")
    print(f"   Anchors: {anchors_flat.shape[0]:,}")
    print(f"   Anchor covers LiDAR: {100*((lidar_points[:, 0] >= -200) & (lidar_points[:, 0] <= 200)).sum()/len(lidar_points):.1f}% X, "
          f"{100*((lidar_points[:, 1] >= -100) & (lidar_points[:, 1] <= 150)).sum()/len(lidar_points):.1f}% Y")
print("="*80)