# """
# VISUALIZATION WITH FIXED COORDINATE SYSTEM
# """
# import numpy as np
# import open3d as o3d
# import torch
# import sys
# import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
# CHECKPOINT_PATH = r'F:\Work\DeepLearning\Research\checkpoint_epoch_52.pth'  # Use the overfitted model

# # Must match training defaults in sparse_cnn/trail.py
# LIDAR_VOXEL_SIZE = (0.1, 0.1, 0.1)
# RADAR_VOXEL_SIZE = (0.2, 0.2, 0.2)
# POINT_CLOUD_RANGE = (-50, -50, -3, 50, 50, 5)

# # Visualization-only filtering knobs
# # Visualization-only filtering knobs (tuned to avoid flooding with low-score proposals)
# FORCE_SHOW_PREDICTIONS = True
# VIS_SCORE_THRESH = 0.5
# VIS_TOPK = 500
# VIS_NMS_THRESH = 0.7
# SHOW_PRED_BOXES = True
# VIS_MATCH_PRED_TO_GT = False
# VIS_MAX_PRED_GT_CENTER_DIST = 30.0  # meters
# VIS_FALLBACK_TOPK = 50
# VIS_CENTER_DEDUP_RADIUS_M = 0.3
# VIS_AUTO_ALIGN_PRED_TO_GT_FRAME = False
# VIS_MIN_ALIGN_IMPROVEMENT = 0.15  # require >=15% center-distance improvement
# VIS_APPLY_CENTER_BIAS_CORRECTION = False
# VIS_CENTER_BIAS_TRIGGER_DIST = 15.0  # meters
# VIS_REPORT_RAW_VS_ALIGNED = True

# # Z convention:
# # - Many models output boxes with z at box CENTER.
# # - Some label formats store z at box BOTTOM (e.g., KITTI camera location is bottom-center).
# # If boxes look like they are "floating", this is often just a z-convention mismatch.
# VIS_Z_CONVENTION = "center"  # "center" or "bottom" for rendering/diagnostics
# GT_Z_CONVENTION = "bottom"   # "center" or "bottom" for parsed GT after cam->lidar transform
# PRED_Z_CONVENTION = "center" # "center" or "bottom" for model outputs
# # Sanity limits for predicted box dimensions [h, w, l] in meters
# VIS_MIN_BOX_SIZE = np.array([0.5, 0.3, 0.5], dtype=np.float32)
# VIS_MAX_BOX_SIZE = np.array([4.0, 4.0, 12.0], dtype=np.float32)
# VIS_MAX_BOX_VOLUME = 80.0

# # Visualization-only: snap boxes to estimated ground from points (helps when z is biased)
# SNAP_TO_GROUND = True
# SNAP_RADIUS_M = 2.5
# SNAP_MIN_POINTS = 20
# SNAP_GROUND_PERCENTILE = 5.0



# def transform_boxes_camera_to_lidar(boxes_camera, calib):
#     """Transform boxes from camera to LiDAR coordinates."""
#     if len(boxes_camera) == 0:
#         return boxes_camera
    
#     Tr_velo_to_cam = calib.get('Tr_velo_to_cam', np.eye(4))
#     R0_rect = calib.get('R0_rect', np.eye(3))
    
#     if Tr_velo_to_cam.shape == (12,):
#         Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
#         Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
#     elif Tr_velo_to_cam.shape == (3, 4):
#         Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])

#     if R0_rect.shape == (9,):
#         R0_rect = R0_rect.reshape(3, 3)
#     if R0_rect.shape == (3, 3):
#         R0_4x4 = np.eye(4, dtype=np.float32)
#         R0_4x4[:3, :3] = R0_rect
#     elif R0_rect.shape == (4, 4):
#         R0_4x4 = R0_rect
#     else:
#         R0_4x4 = np.eye(4, dtype=np.float32)
    
#     # Labels are in rectified camera coordinates, so include R0_rect.
#     Tr_cam_to_velo = np.linalg.inv(R0_4x4 @ Tr_velo_to_cam)
#     boxes_lidar = np.zeros_like(boxes_camera, dtype=np.float32)
    
#     for i in range(len(boxes_camera)):
#         x_cam, y_cam, z_cam = boxes_camera[i, 0:3]
#         point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
#         point_lidar = Tr_cam_to_velo @ point_cam
#         boxes_lidar[i, 0:3] = point_lidar[:3]
#         boxes_lidar[i, 3:6] = boxes_camera[i, 3:6]
        
#         rot_cam = boxes_camera[i, 6]
#         rot_lidar = -rot_cam - np.pi / 2
#         while rot_lidar > np.pi:
#             rot_lidar -= 2 * np.pi
#         while rot_lidar < -np.pi:
#             rot_lidar += 2 * np.pi
#         boxes_lidar[i, 6] = rot_lidar
    
#     return boxes_lidar


# def create_bbox_lines():
#     """Line indices for 3D box."""
#     return [
#         [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
#         [4, 5], [5, 6], [6, 7], [7, 4],  # Top
#         [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
#     ]


# def get_box_corners(center, size, rotation):
#     """Get 8 corners of a 3D box.

#     Expected box format: [x, y, z, h, w, l, rot].
#     This visualization assumes z is at the box CENTER (common for detectors).
#     If you want z-as-bottom, set VIS_Z_CONVENTION/GT_Z_CONVENTION/PRED_Z_CONVENTION accordingly.
#     """
#     x, y, z = center
#     h, w, l = size

#     corners = np.array(
#         [
#             [-l / 2, -w / 2, -h / 2],
#             [l / 2, -w / 2, -h / 2],
#             [l / 2, w / 2, -h / 2],
#             [-l / 2, w / 2, -h / 2],
#             [-l / 2, -w / 2, h / 2],
#             [l / 2, -w / 2, h / 2],
#             [l / 2, w / 2, h / 2],
#             [-l / 2, w / 2, h / 2],
#         ]
#     )
    
#     cos_r, sin_r = np.cos(rotation), np.sin(rotation)
#     rot_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    
#     corners = corners @ rot_matrix.T + np.array([x, y, z])
#     return corners


# def create_bbox_lineset(boxes, color=[1, 0, 0], line_width=2.0):
#     """Create Open3D LineSet for boxes."""
#     if len(boxes) == 0:
#         return o3d.geometry.LineSet()

#     # get_box_corners expects z as CENTER; convert if user is working in z-as-bottom mode.
#     boxes = convert_z_convention(np.asarray(boxes, dtype=np.float32), VIS_Z_CONVENTION, "center")
    
#     all_points, all_lines = [], []
#     line_template = create_bbox_lines()
#     point_offset = 0
    
#     for box in boxes:
#         x, y, z, h, w, l, rot = box
#         corners = get_box_corners([x, y, z], [h, w, l], rot)
#         all_points.append(corners)
        
#         for line in line_template:
#             all_lines.append([line[0] + point_offset, line[1] + point_offset])
#         point_offset += 8
    
#     lineset = o3d.geometry.LineSet()
#     lineset.points = o3d.utility.Vector3dVector(np.vstack(all_points))
#     lineset.lines = o3d.utility.Vector2iVector(np.array(all_lines))
#     lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(all_lines))])
    
#     return lineset


# def snap_boxes_to_ground(boxes_center, points_xyz, radius_m=2.5, min_points=20, ground_percentile=5.0):
#     """
#     Snap boxes to local ground estimated from nearby points.
#     boxes_center: [N,7] with z as CENTER.
#     points_xyz: [M,3]
#     """
#     if len(boxes_center) == 0 or len(points_xyz) == 0:
#         return boxes_center

#     b = np.asarray(boxes_center, dtype=np.float32).copy()
#     pts = np.asarray(points_xyz, dtype=np.float32)

#     global_ground = float(np.percentile(pts[:, 2], ground_percentile))
#     r2 = float(radius_m) ** 2

#     for i in range(len(b)):
#         x, y = float(b[i, 0]), float(b[i, 1])
#         h = float(b[i, 3])
#         dx = pts[:, 0] - x
#         dy = pts[:, 1] - y
#         mask = (dx * dx + dy * dy) <= r2
#         if int(mask.sum()) >= int(min_points):
#             ground_z = float(np.percentile(pts[mask, 2], ground_percentile))
#         else:
#             ground_z = global_ground
#         b[i, 2] = ground_z + 0.5 * h

#     return b


# def convert_z_convention(boxes, src, dst):
#     """Convert boxes between z-as-center and z-as-bottom conventions."""
#     if len(boxes) == 0 or src == dst:
#         return boxes
#     if src not in {"center", "bottom"} or dst not in {"center", "bottom"}:
#         raise ValueError(f"Invalid z convention: src={src}, dst={dst}")

#     b = boxes.copy()
#     h = b[:, 3]
#     if src == "bottom" and dst == "center":
#         b[:, 2] = b[:, 2] + 0.5 * h
#     elif src == "center" and dst == "bottom":
#         b[:, 2] = b[:, 2] - 0.5 * h
#     return b


# def _greedy_match_by_center_xy(pred_boxes, gt_boxes):
#     """Greedy one-to-one matching by XY center distance. Returns (gt_to_pred, chosen_xy_dists)."""
#     if len(pred_boxes) == 0 or len(gt_boxes) == 0:
#         return (
#             np.full((len(gt_boxes),), -1, dtype=np.int64),
#             np.full((len(gt_boxes),), np.inf, dtype=np.float32),
#         )

#     gt_xy = gt_boxes[:, :2]
#     pred_xy = pred_boxes[:, :2]
#     dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))  # (G, P)

#     gt_to_pred = np.full((len(gt_boxes),), -1, dtype=np.int64)
#     used = set()
#     order = np.argsort(dists, axis=1)
#     for gi in range(len(gt_boxes)):
#         chosen = None
#         for pi in order[gi]:
#             pi_int = int(pi)
#             if pi_int not in used:
#                 chosen = pi_int
#                 break
#         if chosen is None:
#             chosen = int(order[gi, 0])
#         gt_to_pred[gi] = chosen
#         used.add(chosen)

#     chosen_d = dists[np.arange(len(gt_boxes)), gt_to_pred]
#     return gt_to_pred, chosen_d.astype(np.float32)


# def alignment_report(pred_boxes, gt_boxes):
#     """Alignment stats between raw preds and GT (no visualization-only transforms)."""
#     if len(pred_boxes) == 0 or len(gt_boxes) == 0:
#         return {"ok": False, "reason": "empty"}

#     gt_center = gt_boxes[:, :3].mean(axis=0)
#     pred_center = pred_boxes[:, :3].mean(axis=0)
#     delta_center = pred_center - gt_center

#     gt_to_pred, d_xy = _greedy_match_by_center_xy(pred_boxes, gt_boxes)
#     matched_pred = pred_boxes[gt_to_pred]
#     deltas = matched_pred[:, :3] - gt_boxes[:, :3]
#     d_xyz = np.linalg.norm(deltas, axis=1)

#     return {
#         "ok": True,
#         "gt_center": gt_center,
#         "pred_center": pred_center,
#         "delta_center": delta_center,
#         "match_mean_xy": float(d_xy.mean()),
#         "match_median_xy": float(np.median(d_xy)),
#         "match_max_xy": float(d_xy.max()),
#         "match_mean_xyz": float(d_xyz.mean()),
#         "match_median_xyz": float(np.median(d_xyz)),
#         "match_max_xyz": float(d_xyz.max()),
#         "delta_mean_xyz": deltas.mean(axis=0),
#         "delta_median_xyz": np.median(deltas, axis=0),
#     }


# def bev_iou_axis_aligned(boxes1, boxes2):
#     """Axis-aligned BEV IoU for [x, y, z, h, w, l, rot] boxes."""
#     if len(boxes1) == 0 or len(boxes2) == 0:
#         return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    
#     x1, y1 = boxes1[:, 0], boxes1[:, 1]
#     w1, l1 = boxes1[:, 4], boxes1[:, 5]
    
#     x2, y2 = boxes2[:, 0], boxes2[:, 1]
#     w2, l2 = boxes2[:, 4], boxes2[:, 5]
    
#     # Expand for broadcasting
#     x1 = x1[:, None]
#     y1 = y1[:, None]
#     w1 = w1[:, None]
#     l1 = l1[:, None]
    
#     x2 = x2[None, :]
#     y2 = y2[None, :]
#     w2 = w2[None, :]
#     l2 = l2[None, :]
    
#     # Axis-aligned overlap
#     # In LiDAR BEV: extent along X is length (l), along Y is width (w).
#     x_overlap = np.maximum(0.0, np.minimum(x1 + l1 / 2, x2 + l2 / 2) - np.maximum(x1 - l1 / 2, x2 - l2 / 2))
#     y_overlap = np.maximum(0.0, np.minimum(y1 + w1 / 2, y2 + w2 / 2) - np.maximum(y1 - w1 / 2, y2 - w2 / 2))
    
#     intersection = x_overlap * y_overlap
#     area1 = (w1 * l1)
#     area2 = (w2 * l2)
#     union = area1 + area2 - intersection
    
#     return (intersection / (union + 1e-6)).astype(np.float32)


# def diagnostics(pred_boxes, pred_scores, gt_boxes):
#     """Print per-GT nearest prediction and BEV IoU stats."""
#     print(f"\n{'='*80}")
#     print("DIAGNOSTICS: GT vs Predictions")
#     print(f"{'='*80}")
    
#     if len(gt_boxes) == 0:
#         print("No GT boxes to compare.")
#         return
#     if len(pred_boxes) == 0:
#         print("No predicted boxes to compare.")
#         return
    
#     # Center distances
#     gt_xy = gt_boxes[:, :2]
#     pred_xy = pred_boxes[:, :2]
#     dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
#     min_idx = dists.argmin(axis=1)
#     min_dist = dists[np.arange(len(gt_boxes)), min_idx]
    
#     # IoU (axis-aligned)
#     ious = bev_iou_axis_aligned(gt_boxes, pred_boxes)
#     best_iou = ious[np.arange(len(gt_boxes)), min_idx]
    
#     for i in range(len(gt_boxes)):
#         gi = gt_boxes[i]
#         pi = pred_boxes[min_idx[i]]
#         ps = pred_scores[min_idx[i]] if len(pred_scores) == len(pred_boxes) else float('nan')
#         print(f"GT[{i}] center=({gi[0]:.1f},{gi[1]:.1f}) size(h,w,l)=({gi[3]:.1f},{gi[4]:.1f},{gi[5]:.1f})")
#         print(f"  Closest pred: center=({pi[0]:.1f},{pi[1]:.1f}) dist={min_dist[i]:.1f}m IoU={best_iou[i]:.4f} score={ps:.6f}")
    
#     print(f"\nSummary:")
#     print(f"  Min/Mean/Max center dist: {min_dist.min():.1f} / {min_dist.mean():.1f} / {min_dist.max():.1f} m")
#     print(f"  Min/Mean/Max best IoU: {best_iou.min():.4f} / {best_iou.mean():.4f} / {best_iou.max():.4f}")
#     print(f"{'='*80}\n")


# def raw_proposal_diagnostics(pred_boxes, pred_scores, gt_boxes, near_xy_m=10.0):
#     """Diagnostic: are there any raw proposals near GT (even if low score)?"""
#     if len(pred_boxes) == 0 or len(gt_boxes) == 0 or len(pred_scores) != len(pred_boxes):
#         return

#     gt_xy = gt_boxes[:, :2]
#     pr_xy = pred_boxes[:, :2]
#     dists = np.sqrt(((gt_xy[:, None, :] - pr_xy[None, :, :]) ** 2).sum(axis=2))  # (G, P)

#     min_per_gt = dists.min(axis=1)
#     argmin_per_gt = dists.argmin(axis=1)
#     scores_at_min = pred_scores[argmin_per_gt]

#     close_any = (dists.min(axis=0) <= near_xy_m)
#     close_count = int(close_any.sum())
#     if close_count > 0:
#         close_scores = pred_scores[close_any]
#         close_d = dists.min(axis=0)[close_any]
#         print(
#             f"Raw proposals near GT (<= {near_xy_m:.1f}m): {close_count}/{len(pred_boxes)} | "
#             f"score min/med/max={close_scores.min():.4f}/{np.median(close_scores):.4f}/{close_scores.max():.4f} | "
#             f"dist min/med/max={close_d.min():.2f}/{np.median(close_d):.2f}/{close_d.max():.2f} m"
#         )
#     else:
#         print(f"Raw proposals near GT (<= {near_xy_m:.1f}m): 0/{len(pred_boxes)}")

#     print(
#         "Per-GT best-by-distance: "
#         f"dist mean/med/max={min_per_gt.mean():.2f}/{np.median(min_per_gt):.2f}/{min_per_gt.max():.2f} m | "
#         f"score at best mean/med/max={scores_at_min.mean():.4f}/{np.median(scores_at_min):.4f}/{scores_at_min.max():.4f}"
#     )


# def summarize_raw_vs_near_gt(pred_boxes, pred_scores, gt_boxes, topk=12):
#     """Return dict with score-topk and near-GT matching stats (no printing)."""
#     out = {"ok": False}
#     if len(pred_boxes) == 0 or len(gt_boxes) == 0 or len(pred_scores) != len(pred_boxes):
#         out["reason"] = "empty"
#         return out

#     order = np.argsort(pred_scores)[::-1]
#     k = min(int(topk), len(order))
#     top_idx = order[:k]
#     top_boxes = pred_boxes[top_idx]
#     top_scores = pred_scores[top_idx]

#     gt_to_pred, d_xy = _greedy_match_by_center_xy(pred_boxes, gt_boxes)
#     near_boxes = pred_boxes[gt_to_pred]
#     near_scores = pred_scores[gt_to_pred]

#     out.update(
#         {
#             "ok": True,
#             "topk": int(k),
#             "top_center": top_boxes[:, :3].mean(axis=0),
#             "top_scores_minmedmax": (float(top_scores.min()), float(np.median(top_scores)), float(top_scores.max())),
#             "near_xy_minmedmax": (float(d_xy.min()), float(np.median(d_xy)), float(d_xy.max())),
#             "near_scores_minmedmax": (float(near_scores.min()), float(np.median(near_scores)), float(near_scores.max())),
#         }
#     )
#     return out


# def nms_bev_axis_aligned(boxes, scores, iou_thresh):
#     """Axis-aligned BEV NMS for [x, y, z, h, w, l, rot] boxes."""
#     if len(boxes) == 0:
#         return np.array([], dtype=np.int64)
    
#     order = scores.argsort()[::-1]
#     keep = []
    
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         if order.size == 1:
#             break
#         ious = bev_iou_axis_aligned(boxes[i:i+1], boxes[order[1:]])[0]
#         order = order[1:][ious <= iou_thresh]
    
#     return np.array(keep, dtype=np.int64)


# def deduplicate_predictions_for_visualization(boxes, scores, iou_thresh=0.95):
#     """Drop near-identical boxes so visible count matches rendered boxes."""
#     if len(boxes) == 0:
#         return boxes, scores
#     if len(scores) != len(boxes):
#         # Best effort: synthesize flat scores to keep deterministic behavior.
#         scores = np.ones((len(boxes),), dtype=np.float32)

#     keep = nms_bev_axis_aligned(boxes, scores, iou_thresh=iou_thresh)
#     return boxes[keep], scores[keep]


# def deduplicate_by_center_distance(boxes, scores, radius_m=1.5):
#     """Keep at most one box within a center radius (highest score wins)."""
#     if len(boxes) == 0:
#         return boxes, scores
#     if len(scores) != len(boxes):
#         scores = np.ones((len(boxes),), dtype=np.float32)

#     order = np.argsort(scores)[::-1]
#     selected = []
#     selected_xy = []
#     r2 = float(radius_m) ** 2

#     for idx in order:
#         xy = boxes[idx, :2]
#         keep = True
#         for sxy in selected_xy:
#             d = xy - sxy
#             if (d[0] * d[0] + d[1] * d[1]) <= r2:
#                 keep = False
#                 break
#         if keep:
#             selected.append(int(idx))
#             selected_xy.append(xy)

#     selected = np.array(selected, dtype=np.int64)
#     return boxes[selected], scores[selected]


# def filter_predictions_for_visualization(pred_boxes, pred_scores, gt_boxes):
#     """Keep predictions in LiDAR range and (optionally) near GT for clean visualization."""
#     initial_count = len(pred_boxes)
#     if len(pred_boxes) == 0:
#         return pred_boxes, pred_scores

#     def normalize_angle(a):
#         return (a + np.pi) % (2 * np.pi) - np.pi

#     def center_error(boxes_a, boxes_b):
#         if len(boxes_a) == 0 or len(boxes_b) == 0:
#             return np.inf
#         d = np.sqrt(((boxes_b[:, None, :2] - boxes_a[None, :, :2]) ** 2).sum(axis=2))
#         return float(d.min(axis=1).mean())

#     def transform_pred_boxes(boxes, mode):
#         b = boxes.copy()
#         if mode == 'identity':
#             return b
#         if mode == 'flip_x':
#             b[:, 0] = -b[:, 0]
#             b[:, 6] = normalize_angle(np.pi - b[:, 6])
#             return b
#         if mode == 'flip_y':
#             b[:, 1] = -b[:, 1]
#             b[:, 6] = normalize_angle(-b[:, 6])
#             return b
#         if mode == 'flip_xy':
#             b[:, 0] = -b[:, 0]
#             b[:, 1] = -b[:, 1]
#             b[:, 6] = normalize_angle(b[:, 6] + np.pi)
#             return b
#         if mode == 'rot_p90':
#             x_old = b[:, 0].copy()
#             y_old = b[:, 1].copy()
#             b[:, 0] = -y_old
#             b[:, 1] = x_old
#             b[:, 6] = normalize_angle(b[:, 6] + np.pi / 2.0)
#             return b
#         if mode == 'rot_m90':
#             x_old = b[:, 0].copy()
#             y_old = b[:, 1].copy()
#             b[:, 0] = y_old
#             b[:, 1] = -x_old
#             b[:, 6] = normalize_angle(b[:, 6] - np.pi / 2.0)
#             return b
#         if mode == 'rot_180':
#             b[:, 0] = -b[:, 0]
#             b[:, 1] = -b[:, 1]
#             b[:, 6] = normalize_angle(b[:, 6] + np.pi)
#             return b
#         return b

#     # Try simple frame fixes if predictions are in a mirrored/rotated XY frame.
#     if VIS_AUTO_ALIGN_PRED_TO_GT_FRAME and len(gt_boxes) > 0:
#         base_err = center_error(pred_boxes, gt_boxes)
#         modes = ['identity', 'flip_x', 'flip_y', 'flip_xy', 'rot_p90', 'rot_m90', 'rot_180']
#         best_mode = 'identity'
#         best_err = base_err
#         best_boxes = pred_boxes
#         for m in modes[1:]:
#             candidate = transform_pred_boxes(pred_boxes, m)
#             err = center_error(candidate, gt_boxes)
#             if err < best_err:
#                 best_err = err
#                 best_mode = m
#                 best_boxes = candidate
#         if np.isfinite(base_err) and base_err > 1e-6:
#             rel_gain = (base_err - best_err) / base_err
#             if best_mode != 'identity' and rel_gain >= VIS_MIN_ALIGN_IMPROVEMENT:
#                 print(f"   Applied pred frame alignment: {best_mode} (dist {base_err:.2f}m -> {best_err:.2f}m)")
#                 pred_boxes = best_boxes

#     # Drop non-finite boxes
#     finite_mask = np.isfinite(pred_boxes).all(axis=1)
#     pred_boxes = pred_boxes[finite_mask]
#     pred_scores = pred_scores[finite_mask]
#     after_finite = len(pred_boxes)
#     if len(pred_boxes) == 0:
#         print(f"   Vis filter: {initial_count} -> 0 (non-finite)")
#         return pred_boxes, pred_scores

#     # Keep only predictions inside configured LiDAR XY range
#     in_x = (pred_boxes[:, 0] >= POINT_CLOUD_RANGE[0]) & (pred_boxes[:, 0] <= POINT_CLOUD_RANGE[3])
#     in_y = (pred_boxes[:, 1] >= POINT_CLOUD_RANGE[1]) & (pred_boxes[:, 1] <= POINT_CLOUD_RANGE[4])
#     range_mask = in_x & in_y
#     pred_boxes = pred_boxes[range_mask]
#     pred_scores = pred_scores[range_mask]
#     after_xy = len(pred_boxes)
#     if len(pred_boxes) == 0:
#         print(f"   Vis filter: {initial_count} -> {after_finite} -> 0 (xy-range)")
#         return pred_boxes, pred_scores

#     # Remove physically implausible boxes (prevents giant boxes covering whole scene).
#     # If this filter removes everything, fall back to the unfiltered set (better to show something than nothing).
#     boxes_before_shape = pred_boxes
#     scores_before_shape = pred_scores
#     dims = pred_boxes[:, 3:6]  # [h, w, l]
#     dims_finite = np.isfinite(dims).all(axis=1)
#     dims_positive = (dims > 0).all(axis=1)
#     dims_min_ok = (dims >= VIS_MIN_BOX_SIZE[None, :]).all(axis=1)
#     dims_max_ok = (dims <= VIS_MAX_BOX_SIZE[None, :]).all(axis=1)
#     dims_volume_ok = (dims[:, 0] * dims[:, 1] * dims[:, 2]) <= VIS_MAX_BOX_VOLUME
#     shape_mask = dims_finite & dims_positive & dims_min_ok & dims_max_ok & dims_volume_ok
#     pred_boxes = pred_boxes[shape_mask]
#     pred_scores = pred_scores[shape_mask]
#     after_shape = len(pred_boxes)
#     if len(pred_boxes) == 0:
#         print(f"   Vis filter: {initial_count} -> {after_finite} -> {after_xy} -> 0 (size/volume) [fallback: keeping pre-shape boxes]")
#         pred_boxes = boxes_before_shape
#         pred_scores = scores_before_shape
#         after_shape = len(pred_boxes)

#     # Keep vertical centers in plausible LiDAR range
#     boxes_before_z = pred_boxes
#     scores_before_z = pred_scores
#     z_mask = (pred_boxes[:, 2] >= POINT_CLOUD_RANGE[2] - 1.0) & (pred_boxes[:, 2] <= POINT_CLOUD_RANGE[5] + 1.0)
#     pred_boxes = pred_boxes[z_mask]
#     pred_scores = pred_scores[z_mask]
#     after_z = len(pred_boxes)
#     if len(pred_boxes) == 0:
#         print(f"   Vis filter: {initial_count} -> {after_finite} -> {after_xy} -> {after_shape} -> 0 (z-range) [fallback: keeping pre-z boxes]")
#         pred_boxes = boxes_before_z
#         pred_scores = scores_before_z
#         after_z = len(pred_boxes)

#     # Optionally keep only predictions that are near GT centers (visualization only)
#     if VIS_MATCH_PRED_TO_GT and len(gt_boxes) > 0:
#         gt_xy = gt_boxes[:, :2]
#         pred_xy = pred_boxes[:, :2]
#         dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))

#         keep_indices = []
#         for gi in range(len(gt_boxes)):
#             near = np.where(dists[gi] <= VIS_MAX_PRED_GT_CENTER_DIST)[0]
#             if near.size == 0:
#                 continue
#             best_local = near[np.argmax(pred_scores[near])]
#             keep_indices.append(int(best_local))

#         if len(keep_indices) > 0:
#             keep_indices = np.array(sorted(set(keep_indices)), dtype=np.int64)
#             pred_boxes = pred_boxes[keep_indices]
#             pred_scores = pred_scores[keep_indices]
#         else:
#             # Fallback: keep the nearest prediction for each GT so red boxes are still visible.
#             # Greedy one-to-one assignment: prefer unique predictions per GT.
#             sorted_idx = np.argsort(dists, axis=1)  # (num_gt, num_pred)
#             selected = []
#             used = set()
#             for gi in range(len(gt_boxes)):
#                 chosen = None
#                 for pi in sorted_idx[gi]:
#                     pi_int = int(pi)
#                     if pi_int not in used:
#                         chosen = pi_int
#                         break
#                 if chosen is None:
#                     chosen = int(sorted_idx[gi, 0])
#                 selected.append(chosen)
#                 used.add(chosen)

#             selected = np.array(selected, dtype=np.int64)
#             nearest_dist = dists[np.arange(len(gt_boxes)), selected]
#             pred_boxes = pred_boxes[selected]
#             pred_scores = pred_scores[selected]
#             print(
#                 f"   No predictions within {VIS_MAX_PRED_GT_CENTER_DIST:.1f}m; "
#                 f"showing nearest-per-GT fallback (mean dist {nearest_dist.mean():.2f}m)"
#             )
#     elif len(pred_scores) > int(VIS_FALLBACK_TOPK):
#         # No GT available: keep a small set of best predictions.
#         topk = min(int(VIS_FALLBACK_TOPK), len(pred_scores))
#         order = np.argsort(pred_scores)[::-1][:topk]
#         pred_boxes = pred_boxes[order]
#         pred_scores = pred_scores[order]

#     print(
#         f"   Vis filter kept: {len(pred_boxes)}/{initial_count} "
#         f"(finite={after_finite}, xy={after_xy}, size={after_shape}, z={after_z})"
#     )
#     if len(pred_boxes) > 0:
#         rounded_centers = np.round(pred_boxes[:, :2], 2)
#         unique_centers = np.unique(rounded_centers, axis=0)
#         print(f"   Visible uniqueness: {len(unique_centers)} unique XY centers out of {len(pred_boxes)} boxes")

#     # Final dedup step for rendering: remove almost identical overlapping boxes.
#     before_dedup = len(pred_boxes)
#     pred_boxes, pred_scores = deduplicate_predictions_for_visualization(
#         pred_boxes, pred_scores, iou_thresh=0.95
#     )
#     if before_dedup != len(pred_boxes):
#         print(f"   Deduped overlapping boxes: {before_dedup} -> {len(pred_boxes)}")

#     # Additional center-based dedup to match what is visually distinguishable.
#     before_center_dedup = len(pred_boxes)
#     pred_boxes, pred_scores = deduplicate_by_center_distance(
#         pred_boxes, pred_scores, radius_m=VIS_CENTER_DEDUP_RADIUS_M
#     )
#     if before_center_dedup != len(pred_boxes):
#         print(f"   Deduped by center radius ({VIS_CENTER_DEDUP_RADIUS_M:.1f}m): {before_center_dedup} -> {len(pred_boxes)}")

#     # Visualization-only correction: remove global XY bias when predictions are consistently shifted.
#     if VIS_APPLY_CENTER_BIAS_CORRECTION and len(pred_boxes) > 0 and len(gt_boxes) > 0:
#         pred_center_xy = pred_boxes[:, :2].mean(axis=0)
#         gt_center_xy = gt_boxes[:, :2].mean(axis=0)
#         delta_xy = gt_center_xy - pred_center_xy
#         center_dist = float(np.linalg.norm(delta_xy))
#         if center_dist >= VIS_CENTER_BIAS_TRIGGER_DIST:
#             pred_boxes = pred_boxes.copy()
#             pred_boxes[:, 0] += delta_xy[0]
#             pred_boxes[:, 1] += delta_xy[1]
#             print(
#                 f"   Applied XY center bias correction: "
#                 f"dx={delta_xy[0]:.2f}, dy={delta_xy[1]:.2f} (dist {center_dist:.2f}m)"
#             )
#     return pred_boxes, pred_scores


# def to_tensor(data, dtype=torch.float32, device='cuda'):
#     """Convert data to tensor."""
#     if isinstance(data, torch.Tensor):
#         return data.to(dtype).to(device)
#     else:
#         return torch.from_numpy(data).to(dtype).to(device)


# def load_model_and_predict(sample_idx, checkpoint_path):
#     """Load model and get predictions."""
#     print(f"\n{'='*80}")
#     print(f"🔄 LOADING MODEL AND RUNNING INFERENCE")
#     print(f"{'='*80}")
    
#     from data_loader import get_LiDAR, get_radar, get_images, get_labels, get_calib
#     from trail import MultiModalDetectionNetwork, voxelize_lidar_proper, voxelize_radar_proper
#     from rpn_refinement import parse_label_line
#     import spconv.pytorch as spconv
#     from PIL import Image
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Device: {device}")
    
#     # Load model
#     print(f"\n📦 Loading checkpoint: {checkpoint_path}")
#     if not os.path.exists(checkpoint_path):
#         print(f"❌ Checkpoint not found: {checkpoint_path}")
#         return None, None, None, None
    
#     model = MultiModalDetectionNetwork(lidar_dim=128, radar_dim=128, image_dim=128).to(device)
    
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
#         print(f"   Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
#     except Exception as e:
#         print(f"❌ Error loading checkpoint: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None, None
    
#     model.eval()
    
#     # Load data
#     print(f"\n📥 Loading sample {sample_idx}...")
#     lidar_data = get_LiDAR(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
#     radar_data = get_radar(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
#     image_data = get_images(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
#     labels = get_labels(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
#     calibs = get_calib(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    
#     if len(lidar_data) == 0:
#         print(f"❌ No data found")
#         return None, None, None, None
    
#     lidar_points = lidar_data[0]
#     radar_points = radar_data[0]
#     image = image_data[0]
    
#     print(f"✅ Data loaded:")
#     print(f"   LiDAR points: {len(lidar_points):,}")
#     print(f"   LiDAR X range: [{lidar_points[:, 0].min():.1f}, {lidar_points[:, 0].max():.1f}]")
#     print(f"   LiDAR Y range: [{lidar_points[:, 1].min():.1f}, {lidar_points[:, 1].max():.1f}]")
#     print(f"   LiDAR Z range: [{lidar_points[:, 2].min():.1f}, {lidar_points[:, 2].max():.1f}]")
    
#     # Parse GT boxes
#     gt_boxes_camera = []
#     for line in labels[0]:
#         if isinstance(line, str) and line.strip():
#             parsed = parse_label_line(line)
#             obj_type = parsed['type'].lower()
#             # Match training classes in sparse_cnn/trail.py (truck is currently not trained)
#             if obj_type in ['car', 'pedestrian', 'cyclist']:
#                 box = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
#                 gt_boxes_camera.append(box)
    
#     gt_boxes_camera = np.array(gt_boxes_camera, dtype=np.float32)
#     gt_boxes_lidar = transform_boxes_camera_to_lidar(gt_boxes_camera, calibs[0])
#     gt_boxes_lidar = convert_z_convention(gt_boxes_lidar, GT_Z_CONVENTION, VIS_Z_CONVENTION)
    
#     print(f"✅ Ground truth (trained classes only): {len(gt_boxes_lidar)} boxes")
#     print(f"   GT X range: [{gt_boxes_lidar[:, 0].min():.1f}, {gt_boxes_lidar[:, 0].max():.1f}]")
#     print(f"   GT Y range: [{gt_boxes_lidar[:, 1].min():.1f}, {gt_boxes_lidar[:, 1].max():.1f}]")
    
#     # Prepare input with SAME parameters as training
#     print(f"\n🔄 Preparing model input with training parameters...")
    
#     # Voxelize with SAME parameters as training
#     lidar_feat, lidar_coords, lidar_shape, lidar_bs = voxelize_lidar_proper(
#         [lidar_points],
#         voxel_size=LIDAR_VOXEL_SIZE,
#         point_cloud_range=POINT_CLOUD_RANGE
#     )
    
#     radar_feat, radar_coords, radar_shape, radar_bs = voxelize_radar_proper(
#         [radar_points],
#         voxel_size=RADAR_VOXEL_SIZE,
#         point_cloud_range=POINT_CLOUD_RANGE
#     )
    
#     print(f"   LiDAR voxels: {len(lidar_feat):,}")
#     print(f"   Radar voxels: {len(radar_feat):,}")
    
#     # Create sparse tensors
#     lidar_sparse = spconv.SparseConvTensor(
#         features=to_tensor(lidar_feat, torch.float32, device),
#         indices=to_tensor(lidar_coords, torch.int32, device),
#         spatial_shape=lidar_shape,
#         batch_size=lidar_bs
#     )
    
#     radar_sparse = spconv.SparseConvTensor(
#         features=to_tensor(radar_feat, torch.float32, device),
#         indices=to_tensor(radar_coords, torch.int32, device),
#         spatial_shape=radar_shape,
#         batch_size=radar_bs
#     )
    
#     # Prepare image
#     image_resized = np.array(Image.fromarray(image).resize((1242, 375)))
#     image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
#     # Run inference
#     debug = {}
#     print(f"\n🚀 Running inference...")
#     try:
#         if FORCE_SHOW_PREDICTIONS and hasattr(model, 'detector'):
#             model.detector.proposal_generator.score_thresh = 0.0
#             print("FORCE_SHOW_PREDICTIONS=True: score_thresh set to 0.0 for visualization")
#         elif hasattr(model, 'detector'):
#             model.detector.proposal_generator.score_thresh = VIS_SCORE_THRESH
#         with torch.no_grad():
#             outputs = model(lidar_sparse, radar_sparse, image_tensor, training=False)
#             print("00000000000000000000000000000000000000000000000000000000000000000000000000000")
#             print(outputs)
#             proposals = outputs['detections']['proposals'][0]
#             scores = outputs['detections']['scores'][0]
#             print(f"Raw proposals: {len(proposals)}")
#             if len(scores) > 0:
#                 print(f"Raw score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")
        
#         if len(proposals) > 0:
#             pred_boxes_raw = proposals.cpu().numpy()
#             pred_scores_raw = scores.cpu().numpy()
#             pred_boxes_raw = convert_z_convention(pred_boxes_raw, PRED_Z_CONVENTION, VIS_Z_CONVENTION)
#             raw_proposal_diagnostics(pred_boxes_raw, pred_scores_raw, gt_boxes_lidar, near_xy_m=10.0)
#             debug["raw_summary"] = summarize_raw_vs_near_gt(
#                 pred_boxes_raw, pred_scores_raw, gt_boxes_lidar, topk=VIS_FALLBACK_TOPK
#             )

#             pred_boxes = pred_boxes_raw
#             pred_scores = pred_scores_raw
            
#             # Visualization-only filtering to reduce clutter
#             if pred_scores.size > 0:
#                 score_mask = pred_scores >= VIS_SCORE_THRESH
#                 pred_boxes = pred_boxes[score_mask]
#                 pred_scores = pred_scores[score_mask]
            
#             if pred_scores.size > VIS_TOPK:
#                 top_idx = np.argsort(pred_scores)[::-1][:VIS_TOPK]
#                 pred_boxes = pred_boxes[top_idx]
#                 pred_scores = pred_scores[top_idx]
            
#             if pred_scores.size > 0:
#                 keep = nms_bev_axis_aligned(pred_boxes, pred_scores, VIS_NMS_THRESH)
#                 pred_boxes = pred_boxes[keep]
#                 pred_scores = pred_scores[keep]

#             if VIS_REPORT_RAW_VS_ALIGNED and len(gt_boxes_lidar) > 0 and len(pred_boxes) > 0:
#                 rep_raw = alignment_report(pred_boxes, gt_boxes_lidar)
#                 dc = rep_raw["delta_center"]
#                 print(
#                     f"Raw alignment (pre-vis-fixes): Δcenter dx={dc[0]:.2f}, dy={dc[1]:.2f}, dz={dc[2]:.2f} | "
#                     f"match XY median={rep_raw['match_median_xy']:.2f}m, XYZ median={rep_raw['match_median_xyz']:.2f}m"
#                 )

#             pred_boxes, pred_scores = filter_predictions_for_visualization(pred_boxes, pred_scores, gt_boxes_lidar)

#             if VIS_REPORT_RAW_VS_ALIGNED and len(gt_boxes_lidar) > 0 and len(pred_boxes) > 0:
#                 rep_aligned = alignment_report(pred_boxes, gt_boxes_lidar)
#                 dc2 = rep_aligned["delta_center"]
#                 print(
#                     f"After vis fixes:               Δcenter dx={dc2[0]:.2f}, dy={dc2[1]:.2f}, dz={dc2[2]:.2f} | "
#                     f"match XY median={rep_aligned['match_median_xy']:.2f}m, XYZ median={rep_aligned['match_median_xyz']:.2f}m"
#                 )

#             if pred_scores.size > 0:
#                 print(f"   Score range: [{pred_scores.min():.6f}, {pred_scores.max():.6f}]")
#             else:
#                 print("   Score range: [N/A, N/A] (no predictions after LiDAR/GT filtering)")
            
#             print(f"\n✅ Predictions: {len(pred_boxes)} boxes")
#             if len(pred_boxes) == 0:
#                 print("   No boxes after visualization filtering")
#             else:
#                 print(f"   Pred X range: [{pred_boxes[:, 0].min():.1f}, {pred_boxes[:, 0].max():.1f}]")
#                 print(f"   Pred Y range: [{pred_boxes[:, 1].min():.1f}, {pred_boxes[:, 1].max():.1f}]")
#                 print(f"   Pred Z range: [{pred_boxes[:, 2].min():.1f}, {pred_boxes[:, 2].max():.1f}]")
            
#             diagnostics(pred_boxes, pred_scores, gt_boxes_lidar)
            
#             # 🔍 COORDINATE DIAGNOSTIC
#             print(f"\n{'='*80}")
#             print(f"🔍 COORDINATE DIAGNOSTIC")
#             print(f"{'='*80}")
            
#             if len(gt_boxes_lidar) > 0:
#                 if len(pred_boxes) > 0:
#                     rep = alignment_report(pred_boxes, gt_boxes_lidar)
#                     gt_center = rep["gt_center"]
#                     pred_center = rep["pred_center"]
#                     dc = rep["delta_center"]

#                     print(
#                         f"GT mean center:   [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]"
#                     )
#                     print(
#                         f"Pred mean center: [{pred_center[0]:.2f}, {pred_center[1]:.2f}, {pred_center[2]:.2f}]"
#                     )
#                     print(f"Δcenter (pred-gt): dx={dc[0]:.2f}, dy={dc[1]:.2f}, dz={dc[2]:.2f}")

#                     print(
#                         "Match dist (greedy by XY) "
#                         f"XY mean/median/max={rep['match_mean_xy']:.2f}/{rep['match_median_xy']:.2f}/{rep['match_max_xy']:.2f} m, "
#                         f"XYZ mean/median/max={rep['match_mean_xyz']:.2f}/{rep['match_median_xyz']:.2f}/{rep['match_max_xyz']:.2f} m"
#                     )

#                     if rep["match_median_xyz"] > 2.0 or rep["match_max_xyz"] > 10.0:
#                         print(f"\n⚠️  CRITICAL: Large pred↔GT mismatch remains.")
#                     elif rep["match_median_xyz"] <= 0.50 and rep["match_max_xyz"] <= 2.0:
#                         print(f"\n✅ GOOD: Predictions are close to GT in 3D.")
#                 else:
#                     print("No predictions available for coordinate diagnostic.")
            
#             print(f"{'='*80}\n")
            
#         else:
#             pred_boxes = np.zeros((0, 7))
#             pred_scores = np.array([])
#             print(f"⚠️  No predictions")
            
#     except Exception as e:
#         print(f"❌ Error during inference: {e}")
#         import traceback
#         traceback.print_exc()
#         pred_boxes = np.zeros((0, 7))
#         pred_scores = np.array([])
    
#     return lidar_points, gt_boxes_lidar, pred_boxes, pred_scores, debug


# def visualize_scene(geometries, window_name):
#     """Render a single Open3D scene."""
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name, width=1600, height=900)
#     for geom in geometries:
#         vis.add_geometry(geom)

#     opt = vis.get_render_option()
#     if opt is not None:
#         opt.point_size = 6.0
#         opt.background_color = np.array([0.0, 0.0, 0.0])
#         opt.line_width = 2.0

#     # Fit view to geometry
#     vis.reset_view_point(True)
#     vis.run()
#     vis.destroy_window()


# def visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=0):
#     """Visualize LiDAR points, GT boxes, and predictions in one frame."""
#     print(f"\n{'='*80}")
#     print(f"🎨 CREATING VISUALIZATION")
#     print(f"{'='*80}")
    
#     # Basic sanity checks
#     if lidar_points is None or len(lidar_points) == 0:
#         print("❌ No LiDAR points provided to visualize()")
#         return
#     finite_mask = np.isfinite(lidar_points[:, :3]).all(axis=1)
#     if not finite_mask.all():
#         lidar_points = lidar_points[finite_mask]
#         print(f"⚠️  Dropped non-finite LiDAR points. Remaining: {len(lidar_points):,}")

#     print(f"LiDAR raw range: X[{lidar_points[:,0].min():.2f}, {lidar_points[:,0].max():.2f}] "
#           f"Y[{lidar_points[:,1].min():.2f}, {lidar_points[:,1].max():.2f}] "
#           f"Z[{lidar_points[:,2].min():.2f}, {lidar_points[:,2].max():.2f}]")

#     # Filter LiDAR points to reasonable range
#     mask_x = (lidar_points[:, 0] >= POINT_CLOUD_RANGE[0]) & (lidar_points[:, 0] <= POINT_CLOUD_RANGE[3])
#     mask_y = (lidar_points[:, 1] >= POINT_CLOUD_RANGE[1]) & (lidar_points[:, 1] <= POINT_CLOUD_RANGE[4])
#     mask = mask_x & mask_y
#     lidar_points_filtered = lidar_points[mask]
#     if len(lidar_points_filtered) == 0:
#         print("⚠️  No LiDAR points after range filter. Using raw points for visualization.")
#         lidar_points_filtered = lidar_points
    
#     print(f"📊 Visualization data:")
#     print(f"   LiDAR points: {len(lidar_points_filtered):,} (filtered to POINT_CLOUD_RANGE)")
#     print(f"   GT boxes: {len(gt_boxes)}")
#     print(f"   Pred boxes: {len(pred_boxes)}")
    
#     # Point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(lidar_points_filtered[:, :3])
    
#     # Use a bright, constant color so points are always visible
#     colors = np.full((len(lidar_points_filtered), 3), 0.9, dtype=np.float32)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # Boxes (optional z-snapping for nicer visuals)
#     gt_boxes_vis = gt_boxes
#     pred_boxes_vis = pred_boxes
#     if SNAP_TO_GROUND and len(lidar_points_filtered) > 0:
#         pts_xyz = lidar_points_filtered[:, :3]
#         if len(gt_boxes_vis) > 0:
#             gt_c = convert_z_convention(gt_boxes_vis, VIS_Z_CONVENTION, "center")
#             gt_c = snap_boxes_to_ground(
#                 gt_c,
#                 pts_xyz,
#                 radius_m=SNAP_RADIUS_M,
#                 min_points=SNAP_MIN_POINTS,
#                 ground_percentile=SNAP_GROUND_PERCENTILE,
#             )
#             gt_boxes_vis = convert_z_convention(gt_c, "center", VIS_Z_CONVENTION)
#         if len(pred_boxes_vis) > 0:
#             pr_c = convert_z_convention(pred_boxes_vis, VIS_Z_CONVENTION, "center")
#             pr_c = snap_boxes_to_ground(
#                 pr_c,
#                 pts_xyz,
#                 radius_m=SNAP_RADIUS_M,
#                 min_points=SNAP_MIN_POINTS,
#                 ground_percentile=SNAP_GROUND_PERCENTILE,
#             )
#             pred_boxes_vis = convert_z_convention(pr_c, "center", VIS_Z_CONVENTION)

#     gt_lineset = create_bbox_lineset(gt_boxes_vis, color=[0, 1, 0])      # GREEN
#     pred_lineset = create_bbox_lineset(pred_boxes_vis, color=[1, 0, 0])  # RED
    
#     # Coordinate frame
#     if len(lidar_points_filtered) > 0:
#         center = lidar_points_filtered[:, :3].mean(axis=0)
#     elif len(gt_boxes) > 0:
#         center = gt_boxes[:, :3].mean(axis=0)
#     else:
#         center = np.array([0, 0, 0])
    
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=5.0, origin=center
#     )
    
#     print(f"\n🎥 Camera positioned at: {center}")
#     print(f"\nLEGEND:")
#     print(f"  🟢 GREEN boxes = Ground Truth ({len(gt_boxes)} boxes)")
#     if SHOW_PRED_BOXES:
#         print(f"  🔴 RED boxes = Model Predictions ({len(pred_boxes)} boxes)")
#     print(f"  ⚫ Gray points = LiDAR point cloud")
#     print(f"  📍 RGB axes = Coordinate frame")
#     print(f"\nCONTROLS:")
#     print(f"  - LEFT CLICK + DRAG: Rotate")
#     print(f"  - SCROLL: Zoom")
#     print(f"  - RIGHT CLICK + DRAG: Pan")
#     print(f"  - Press 'Q': Close window")
#     print(f"{'='*80}\n")

#     geometries = [pcd, coord_frame]
#     if len(gt_boxes) > 0:
#         geometries.append(gt_lineset)
#     if SHOW_PRED_BOXES and len(pred_boxes) > 0:
#         geometries.append(pred_lineset)

#     if SHOW_PRED_BOXES:
#         print("Opening: single combined frame (LiDAR + GT + Predictions)...")
#         window_name = f"Sample {sample_idx} | LiDAR + GT + Predictions"
#     else:
#         print("Opening: single combined frame (LiDAR + GT)...")
#         window_name = f"Sample {sample_idx} | LiDAR + GT"
#     visualize_scene(geometries, window_name)

#     print(f"\n✅ Visualization closed")


# if __name__ == "__main__":
#     print("="*80)
#     print("3D DETECTION VISUALIZATION WITH FIXED COORDINATES")
#     print("="*80)
    
#     # Load and predict
#     lidar_points, gt_boxes, pred_boxes, scores, debug = load_model_and_predict(0, CHECKPOINT_PATH)
    
#     if lidar_points is not None:
#         # Visualize
#         visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=0)
        
#         # Final summary
#         print(f"\n{'='*80}")
#         print(f"📊 FINAL SUMMARY")
#         print(f"{'='*80}")
#         print(f"Ground Truth: {len(gt_boxes)} boxes")
#         print(f"Predictions (after vis filtering): {len(pred_boxes)} boxes")

#         raw_sum = (debug or {}).get("raw_summary")
#         if raw_sum and raw_sum.get("ok"):
#             top_center = raw_sum["top_center"]
#             smin, smed, smax = raw_sum["top_scores_minmedmax"]
#             dmin, dmed, dmax = raw_sum["near_xy_minmedmax"]
#             smin2, smed2, smax2 = raw_sum["near_scores_minmedmax"]
#             print(f"\nRaw proposal summary:")
#             print(f"  Top-{raw_sum['topk']} by score center: [{top_center[0]:.2f}, {top_center[1]:.2f}, {top_center[2]:.2f}]")
#             print(f"  Top-{raw_sum['topk']} score min/med/max: {smin:.4f}/{smed:.4f}/{smax:.4f}")
#             print(f"  Nearest-per-GT XY dist min/med/max: {dmin:.2f}/{dmed:.2f}/{dmax:.2f} m")
#             print(f"  Nearest-per-GT score min/med/max: {smin2:.4f}/{smed2:.4f}/{smax2:.4f}")
        
#         if len(pred_boxes) > 0 and len(gt_boxes) > 0:
#             rep = alignment_report(pred_boxes, gt_boxes)
#             gt_center = rep["gt_center"]
#             pred_center = rep["pred_center"]
#             dc = rep["delta_center"]

#             print(f"\nCoordinate Check (visualized boxes):")
#             print(f"  GT mean center:   [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]")
#             print(f"  Pred mean center: [{pred_center[0]:.2f}, {pred_center[1]:.2f}, {pred_center[2]:.2f}]")
#             print(f"  Δcenter (pred-gt): dx={dc[0]:.2f}, dy={dc[1]:.2f}, dz={dc[2]:.2f}")

#             print(f"\nPer-box matching (greedy by XY center):")
#             print(
#                 f"  XY dist   mean/median/max: {rep['match_mean_xy']:.2f} / {rep['match_median_xy']:.2f} / {rep['match_max_xy']:.2f} m"
#             )
#             print(
#                 f"  XYZ dist  mean/median/max: {rep['match_mean_xyz']:.2f} / {rep['match_median_xyz']:.2f} / {rep['match_max_xyz']:.2f} m"
#             )

#             if rep["match_median_xyz"] <= 0.50 and rep["match_max_xyz"] <= 2.0:
#                 print(f"\n✅ OK: Predictions are close to GT in 3D.")
#             else:
#                 print(f"\n⚠️  Misalignment remains (often dz if XY looks fine).")
        
#         print(f"{'='*80}\n")
#     else:
#         print(f"\n❌ Failed to load model or data")


"""
VISUALIZATION WITH FIXED COORDINATE SYSTEM - MULTI-SAMPLE
"""
import numpy as np
import open3d as o3d
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
CHECKPOINT_PATH = r'F:\Work\DeepLearning\Research\checkpoint_epoch_52.pth'  # Use the overfitted model

# ── Multi-sample settings ─────────────────────────────────────────────────────
# Set SAMPLE_INDICES to a list of ints, e.g. [0, 1, 2, 5], OR set
# SAMPLE_RANGE to (start, stop) to run start..stop-1, OR leave SAMPLE_INDICES
# as None and fill in SAMPLE_RANGE.  SAMPLE_INDICES takes priority.
SAMPLE_INDICES = None          # e.g. [0, 3, 7]  – explicit list
SAMPLE_RANGE   = (55, 60)        # e.g. (0, 10)    – run samples 0-9
# ─────────────────────────────────────────────────────────────────────────────

# Must match training defaults in sparse_cnn/trail.py
LIDAR_VOXEL_SIZE = (0.1, 0.1, 0.1)
RADAR_VOXEL_SIZE = (0.2, 0.2, 0.2)
POINT_CLOUD_RANGE = (-50, -50, -3, 50, 50, 5)

# Visualization-only filtering knobs
FORCE_SHOW_PREDICTIONS = True
VIS_SCORE_THRESH = 0.5
VIS_TOPK = 500
VIS_NMS_THRESH = 0.7
SHOW_PRED_BOXES = True
VIS_MATCH_PRED_TO_GT = False
VIS_MAX_PRED_GT_CENTER_DIST = 30.0  # meters
VIS_FALLBACK_TOPK = 50
VIS_CENTER_DEDUP_RADIUS_M = 0.3
VIS_AUTO_ALIGN_PRED_TO_GT_FRAME = False
VIS_MIN_ALIGN_IMPROVEMENT = 0.15
VIS_APPLY_CENTER_BIAS_CORRECTION = False
VIS_CENTER_BIAS_TRIGGER_DIST = 15.0
VIS_REPORT_RAW_VS_ALIGNED = True

VIS_Z_CONVENTION = "center"
GT_Z_CONVENTION = "bottom"
PRED_Z_CONVENTION = "center"
VIS_MIN_BOX_SIZE = np.array([0.5, 0.3, 0.5], dtype=np.float32)
VIS_MAX_BOX_SIZE = np.array([4.0, 4.0, 12.0], dtype=np.float32)
VIS_MAX_BOX_VOLUME = 80.0

SNAP_TO_GROUND = True
SNAP_RADIUS_M = 2.5
SNAP_MIN_POINTS = 20
SNAP_GROUND_PERCENTILE = 5.0


def transform_boxes_camera_to_lidar(boxes_camera, calib):
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


def create_bbox_lines():
    return [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]


def get_box_corners(center, size, rotation):
    x, y, z = center
    h, w, l = size
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [l/2, -w/2,  h/2],
        [l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rot_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    corners = corners @ rot_matrix.T + np.array([x, y, z])
    return corners


def create_bbox_lineset(boxes, color=[1, 0, 0], line_width=2.0):
    if len(boxes) == 0:
        return o3d.geometry.LineSet()
    boxes = convert_z_convention(np.asarray(boxes, dtype=np.float32), VIS_Z_CONVENTION, "center")
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


def snap_boxes_to_ground(boxes_center, points_xyz, radius_m=2.5, min_points=20, ground_percentile=5.0):
    if len(boxes_center) == 0 or len(points_xyz) == 0:
        return boxes_center
    b = np.asarray(boxes_center, dtype=np.float32).copy()
    pts = np.asarray(points_xyz, dtype=np.float32)
    global_ground = float(np.percentile(pts[:, 2], ground_percentile))
    r2 = float(radius_m) ** 2
    for i in range(len(b)):
        x, y = float(b[i, 0]), float(b[i, 1])
        h = float(b[i, 3])
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        mask = (dx * dx + dy * dy) <= r2
        if int(mask.sum()) >= int(min_points):
            ground_z = float(np.percentile(pts[mask, 2], ground_percentile))
        else:
            ground_z = global_ground
        b[i, 2] = ground_z + 0.5 * h
    return b


def convert_z_convention(boxes, src, dst):
    if len(boxes) == 0 or src == dst:
        return boxes
    if src not in {"center", "bottom"} or dst not in {"center", "bottom"}:
        raise ValueError(f"Invalid z convention: src={src}, dst={dst}")
    b = boxes.copy()
    h = b[:, 3]
    if src == "bottom" and dst == "center":
        b[:, 2] = b[:, 2] + 0.5 * h
    elif src == "center" and dst == "bottom":
        b[:, 2] = b[:, 2] - 0.5 * h
    return b


def _greedy_match_by_center_xy(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return (
            np.full((len(gt_boxes),), -1, dtype=np.int64),
            np.full((len(gt_boxes),), np.inf, dtype=np.float32),
        )
    gt_xy = gt_boxes[:, :2]
    pred_xy = pred_boxes[:, :2]
    dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
    gt_to_pred = np.full((len(gt_boxes),), -1, dtype=np.int64)
    used = set()
    order = np.argsort(dists, axis=1)
    for gi in range(len(gt_boxes)):
        chosen = None
        for pi in order[gi]:
            pi_int = int(pi)
            if pi_int not in used:
                chosen = pi_int
                break
        if chosen is None:
            chosen = int(order[gi, 0])
        gt_to_pred[gi] = chosen
        used.add(chosen)
    chosen_d = dists[np.arange(len(gt_boxes)), gt_to_pred]
    return gt_to_pred, chosen_d.astype(np.float32)


def alignment_report(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {"ok": False, "reason": "empty"}
    gt_center = gt_boxes[:, :3].mean(axis=0)
    pred_center = pred_boxes[:, :3].mean(axis=0)
    delta_center = pred_center - gt_center
    gt_to_pred, d_xy = _greedy_match_by_center_xy(pred_boxes, gt_boxes)
    matched_pred = pred_boxes[gt_to_pred]
    deltas = matched_pred[:, :3] - gt_boxes[:, :3]
    d_xyz = np.linalg.norm(deltas, axis=1)
    return {
        "ok": True,
        "gt_center": gt_center, "pred_center": pred_center, "delta_center": delta_center,
        "match_mean_xy": float(d_xy.mean()), "match_median_xy": float(np.median(d_xy)),
        "match_max_xy": float(d_xy.max()), "match_mean_xyz": float(d_xyz.mean()),
        "match_median_xyz": float(np.median(d_xyz)), "match_max_xyz": float(d_xyz.max()),
        "delta_mean_xyz": deltas.mean(axis=0), "delta_median_xyz": np.median(deltas, axis=0),
    }


def bev_iou_axis_aligned(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    w1, l1 = boxes1[:, 4], boxes1[:, 5]
    x2, y2 = boxes2[:, 0], boxes2[:, 1]
    w2, l2 = boxes2[:, 4], boxes2[:, 5]
    x1 = x1[:, None]; y1 = y1[:, None]; w1 = w1[:, None]; l1 = l1[:, None]
    x2 = x2[None, :]; y2 = y2[None, :]; w2 = w2[None, :]; l2 = l2[None, :]
    x_overlap = np.maximum(0.0, np.minimum(x1+l1/2, x2+l2/2) - np.maximum(x1-l1/2, x2-l2/2))
    y_overlap = np.maximum(0.0, np.minimum(y1+w1/2, y2+w2/2) - np.maximum(y1-w1/2, y2-w2/2))
    intersection = x_overlap * y_overlap
    union = w1*l1 + w2*l2 - intersection
    return (intersection / (union + 1e-6)).astype(np.float32)


def diagnostics(pred_boxes, pred_scores, gt_boxes):
    print(f"\n{'='*80}")
    print("DIAGNOSTICS: GT vs Predictions")
    print(f"{'='*80}")
    if len(gt_boxes) == 0:
        print("No GT boxes to compare."); return
    if len(pred_boxes) == 0:
        print("No predicted boxes to compare."); return
    gt_xy = gt_boxes[:, :2]
    pred_xy = pred_boxes[:, :2]
    dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
    min_idx = dists.argmin(axis=1)
    min_dist = dists[np.arange(len(gt_boxes)), min_idx]
    ious = bev_iou_axis_aligned(gt_boxes, pred_boxes)
    best_iou = ious[np.arange(len(gt_boxes)), min_idx]
    for i in range(len(gt_boxes)):
        gi = gt_boxes[i]; pi = pred_boxes[min_idx[i]]
        ps = pred_scores[min_idx[i]] if len(pred_scores) == len(pred_boxes) else float('nan')
        print(f"GT[{i}] center=({gi[0]:.1f},{gi[1]:.1f}) size(h,w,l)=({gi[3]:.1f},{gi[4]:.1f},{gi[5]:.1f})")
        print(f"  Closest pred: center=({pi[0]:.1f},{pi[1]:.1f}) dist={min_dist[i]:.1f}m IoU={best_iou[i]:.4f} score={ps:.6f}")
    print(f"\nSummary:")
    print(f"  Min/Mean/Max center dist: {min_dist.min():.1f} / {min_dist.mean():.1f} / {min_dist.max():.1f} m")
    print(f"  Min/Mean/Max best IoU: {best_iou.min():.4f} / {best_iou.mean():.4f} / {best_iou.max():.4f}")
    print(f"{'='*80}\n")


def raw_proposal_diagnostics(pred_boxes, pred_scores, gt_boxes, near_xy_m=10.0):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0 or len(pred_scores) != len(pred_boxes):
        return
    gt_xy = gt_boxes[:, :2]; pr_xy = pred_boxes[:, :2]
    dists = np.sqrt(((gt_xy[:, None, :] - pr_xy[None, :, :]) ** 2).sum(axis=2))
    min_per_gt = dists.min(axis=1); argmin_per_gt = dists.argmin(axis=1)
    scores_at_min = pred_scores[argmin_per_gt]
    close_any = (dists.min(axis=0) <= near_xy_m); close_count = int(close_any.sum())
    if close_count > 0:
        close_scores = pred_scores[close_any]; close_d = dists.min(axis=0)[close_any]
        print(f"Raw proposals near GT (<= {near_xy_m:.1f}m): {close_count}/{len(pred_boxes)} | "
              f"score min/med/max={close_scores.min():.4f}/{np.median(close_scores):.4f}/{close_scores.max():.4f} | "
              f"dist min/med/max={close_d.min():.2f}/{np.median(close_d):.2f}/{close_d.max():.2f} m")
    else:
        print(f"Raw proposals near GT (<= {near_xy_m:.1f}m): 0/{len(pred_boxes)}")
    print("Per-GT best-by-distance: "
          f"dist mean/med/max={min_per_gt.mean():.2f}/{np.median(min_per_gt):.2f}/{min_per_gt.max():.2f} m | "
          f"score at best mean/med/max={scores_at_min.mean():.4f}/{np.median(scores_at_min):.4f}/{scores_at_min.max():.4f}")


def summarize_raw_vs_near_gt(pred_boxes, pred_scores, gt_boxes, topk=12):
    out = {"ok": False}
    if len(pred_boxes) == 0 or len(gt_boxes) == 0 or len(pred_scores) != len(pred_boxes):
        out["reason"] = "empty"; return out
    order = np.argsort(pred_scores)[::-1]; k = min(int(topk), len(order))
    top_idx = order[:k]; top_boxes = pred_boxes[top_idx]; top_scores = pred_scores[top_idx]
    gt_to_pred, d_xy = _greedy_match_by_center_xy(pred_boxes, gt_boxes)
    near_boxes = pred_boxes[gt_to_pred]; near_scores = pred_scores[gt_to_pred]
    out.update({"ok": True, "topk": int(k),
        "top_center": top_boxes[:, :3].mean(axis=0),
        "top_scores_minmedmax": (float(top_scores.min()), float(np.median(top_scores)), float(top_scores.max())),
        "near_xy_minmedmax": (float(d_xy.min()), float(np.median(d_xy)), float(d_xy.max())),
        "near_scores_minmedmax": (float(near_scores.min()), float(np.median(near_scores)), float(near_scores.max())),
    })
    return out


def nms_bev_axis_aligned(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        ious = bev_iou_axis_aligned(boxes[i:i+1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def deduplicate_predictions_for_visualization(boxes, scores, iou_thresh=0.95):
    if len(boxes) == 0:
        return boxes, scores
    if len(scores) != len(boxes):
        scores = np.ones((len(boxes),), dtype=np.float32)
    keep = nms_bev_axis_aligned(boxes, scores, iou_thresh=iou_thresh)
    return boxes[keep], scores[keep]


def deduplicate_by_center_distance(boxes, scores, radius_m=1.5):
    if len(boxes) == 0:
        return boxes, scores
    if len(scores) != len(boxes):
        scores = np.ones((len(boxes),), dtype=np.float32)
    order = np.argsort(scores)[::-1]; selected = []; selected_xy = []
    r2 = float(radius_m) ** 2
    for idx in order:
        xy = boxes[idx, :2]; keep = True
        for sxy in selected_xy:
            d = xy - sxy
            if (d[0]*d[0] + d[1]*d[1]) <= r2:
                keep = False; break
        if keep:
            selected.append(int(idx)); selected_xy.append(xy)
    selected = np.array(selected, dtype=np.int64)
    return boxes[selected], scores[selected]


def filter_predictions_for_visualization(pred_boxes, pred_scores, gt_boxes):
    initial_count = len(pred_boxes)
    if len(pred_boxes) == 0:
        return pred_boxes, pred_scores

    def normalize_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def center_error(boxes_a, boxes_b):
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.inf
        d = np.sqrt(((boxes_b[:, None, :2] - boxes_a[None, :, :2]) ** 2).sum(axis=2))
        return float(d.min(axis=1).mean())

    def transform_pred_boxes(boxes, mode):
        b = boxes.copy()
        if mode == 'identity': return b
        if mode == 'flip_x':
            b[:, 0] = -b[:, 0]; b[:, 6] = normalize_angle(np.pi - b[:, 6]); return b
        if mode == 'flip_y':
            b[:, 1] = -b[:, 1]; b[:, 6] = normalize_angle(-b[:, 6]); return b
        if mode == 'flip_xy':
            b[:, 0] = -b[:, 0]; b[:, 1] = -b[:, 1]; b[:, 6] = normalize_angle(b[:, 6] + np.pi); return b
        if mode == 'rot_p90':
            x_old = b[:, 0].copy(); y_old = b[:, 1].copy()
            b[:, 0] = -y_old; b[:, 1] = x_old; b[:, 6] = normalize_angle(b[:, 6] + np.pi/2.0); return b
        if mode == 'rot_m90':
            x_old = b[:, 0].copy(); y_old = b[:, 1].copy()
            b[:, 0] = y_old; b[:, 1] = -x_old; b[:, 6] = normalize_angle(b[:, 6] - np.pi/2.0); return b
        if mode == 'rot_180':
            b[:, 0] = -b[:, 0]; b[:, 1] = -b[:, 1]; b[:, 6] = normalize_angle(b[:, 6] + np.pi); return b
        return b

    if VIS_AUTO_ALIGN_PRED_TO_GT_FRAME and len(gt_boxes) > 0:
        base_err = center_error(pred_boxes, gt_boxes)
        modes = ['identity', 'flip_x', 'flip_y', 'flip_xy', 'rot_p90', 'rot_m90', 'rot_180']
        best_mode = 'identity'; best_err = base_err; best_boxes = pred_boxes
        for m in modes[1:]:
            candidate = transform_pred_boxes(pred_boxes, m)
            err = center_error(candidate, gt_boxes)
            if err < best_err:
                best_err = err; best_mode = m; best_boxes = candidate
        if np.isfinite(base_err) and base_err > 1e-6:
            rel_gain = (base_err - best_err) / base_err
            if best_mode != 'identity' and rel_gain >= VIS_MIN_ALIGN_IMPROVEMENT:
                print(f"   Applied pred frame alignment: {best_mode} (dist {base_err:.2f}m -> {best_err:.2f}m)")
                pred_boxes = best_boxes

    finite_mask = np.isfinite(pred_boxes).all(axis=1)
    pred_boxes = pred_boxes[finite_mask]; pred_scores = pred_scores[finite_mask]
    after_finite = len(pred_boxes)
    if len(pred_boxes) == 0:
        print(f"   Vis filter: {initial_count} -> 0 (non-finite)"); return pred_boxes, pred_scores

    in_x = (pred_boxes[:, 0] >= POINT_CLOUD_RANGE[0]) & (pred_boxes[:, 0] <= POINT_CLOUD_RANGE[3])
    in_y = (pred_boxes[:, 1] >= POINT_CLOUD_RANGE[1]) & (pred_boxes[:, 1] <= POINT_CLOUD_RANGE[4])
    range_mask = in_x & in_y
    pred_boxes = pred_boxes[range_mask]; pred_scores = pred_scores[range_mask]
    after_xy = len(pred_boxes)
    if len(pred_boxes) == 0:
        print(f"   Vis filter: {initial_count} -> {after_finite} -> 0 (xy-range)"); return pred_boxes, pred_scores

    boxes_before_shape = pred_boxes; scores_before_shape = pred_scores
    dims = pred_boxes[:, 3:6]
    shape_mask = (np.isfinite(dims).all(axis=1) & (dims > 0).all(axis=1) &
                  (dims >= VIS_MIN_BOX_SIZE[None, :]).all(axis=1) &
                  (dims <= VIS_MAX_BOX_SIZE[None, :]).all(axis=1) &
                  ((dims[:, 0] * dims[:, 1] * dims[:, 2]) <= VIS_MAX_BOX_VOLUME))
    pred_boxes = pred_boxes[shape_mask]; pred_scores = pred_scores[shape_mask]
    after_shape = len(pred_boxes)
    if len(pred_boxes) == 0:
        print(f"   Vis filter: {initial_count} -> {after_finite} -> {after_xy} -> 0 (size/volume) [fallback]")
        pred_boxes = boxes_before_shape; pred_scores = scores_before_shape; after_shape = len(pred_boxes)

    boxes_before_z = pred_boxes; scores_before_z = pred_scores
    z_mask = (pred_boxes[:, 2] >= POINT_CLOUD_RANGE[2] - 1.0) & (pred_boxes[:, 2] <= POINT_CLOUD_RANGE[5] + 1.0)
    pred_boxes = pred_boxes[z_mask]; pred_scores = pred_scores[z_mask]
    after_z = len(pred_boxes)
    if len(pred_boxes) == 0:
        print(f"   Vis filter: {initial_count} -> {after_finite} -> {after_xy} -> {after_shape} -> 0 (z-range) [fallback]")
        pred_boxes = boxes_before_z; pred_scores = scores_before_z; after_z = len(pred_boxes)

    if VIS_MATCH_PRED_TO_GT and len(gt_boxes) > 0:
        gt_xy = gt_boxes[:, :2]; pred_xy = pred_boxes[:, :2]
        dists = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
        keep_indices = []
        for gi in range(len(gt_boxes)):
            near = np.where(dists[gi] <= VIS_MAX_PRED_GT_CENTER_DIST)[0]
            if near.size == 0: continue
            keep_indices.append(int(near[np.argmax(pred_scores[near])]))
        if len(keep_indices) > 0:
            keep_indices = np.array(sorted(set(keep_indices)), dtype=np.int64)
            pred_boxes = pred_boxes[keep_indices]; pred_scores = pred_scores[keep_indices]
        else:
            sorted_idx = np.argsort(dists, axis=1)
            selected = []; used = set()
            for gi in range(len(gt_boxes)):
                chosen = None
                for pi in sorted_idx[gi]:
                    pi_int = int(pi)
                    if pi_int not in used: chosen = pi_int; break
                if chosen is None: chosen = int(sorted_idx[gi, 0])
                selected.append(chosen); used.add(chosen)
            selected = np.array(selected, dtype=np.int64)
            nearest_dist = dists[np.arange(len(gt_boxes)), selected]
            pred_boxes = pred_boxes[selected]; pred_scores = pred_scores[selected]
            print(f"   No predictions within {VIS_MAX_PRED_GT_CENTER_DIST:.1f}m; "
                  f"showing nearest-per-GT fallback (mean dist {nearest_dist.mean():.2f}m)")
    elif len(pred_scores) > int(VIS_FALLBACK_TOPK):
        topk = min(int(VIS_FALLBACK_TOPK), len(pred_scores))
        order = np.argsort(pred_scores)[::-1][:topk]
        pred_boxes = pred_boxes[order]; pred_scores = pred_scores[order]

    print(f"   Vis filter kept: {len(pred_boxes)}/{initial_count} "
          f"(finite={after_finite}, xy={after_xy}, size={after_shape}, z={after_z})")
    if len(pred_boxes) > 0:
        rounded_centers = np.round(pred_boxes[:, :2], 2)
        unique_centers = np.unique(rounded_centers, axis=0)
        print(f"   Visible uniqueness: {len(unique_centers)} unique XY centers out of {len(pred_boxes)} boxes")

    before_dedup = len(pred_boxes)
    pred_boxes, pred_scores = deduplicate_predictions_for_visualization(pred_boxes, pred_scores, iou_thresh=0.95)
    if before_dedup != len(pred_boxes):
        print(f"   Deduped overlapping boxes: {before_dedup} -> {len(pred_boxes)}")

    before_center_dedup = len(pred_boxes)
    pred_boxes, pred_scores = deduplicate_by_center_distance(pred_boxes, pred_scores, radius_m=VIS_CENTER_DEDUP_RADIUS_M)
    if before_center_dedup != len(pred_boxes):
        print(f"   Deduped by center radius ({VIS_CENTER_DEDUP_RADIUS_M:.1f}m): {before_center_dedup} -> {len(pred_boxes)}")

    if VIS_APPLY_CENTER_BIAS_CORRECTION and len(pred_boxes) > 0 and len(gt_boxes) > 0:
        pred_center_xy = pred_boxes[:, :2].mean(axis=0)
        gt_center_xy = gt_boxes[:, :2].mean(axis=0)
        delta_xy = gt_center_xy - pred_center_xy
        center_dist = float(np.linalg.norm(delta_xy))
        if center_dist >= VIS_CENTER_BIAS_TRIGGER_DIST:
            pred_boxes = pred_boxes.copy()
            pred_boxes[:, 0] += delta_xy[0]; pred_boxes[:, 1] += delta_xy[1]
            print(f"   Applied XY center bias correction: dx={delta_xy[0]:.2f}, dy={delta_xy[1]:.2f} (dist {center_dist:.2f}m)")
    return pred_boxes, pred_scores


def to_tensor(data, dtype=torch.float32, device='cuda'):
    if isinstance(data, torch.Tensor):
        return data.to(dtype).to(device)
    else:
        return torch.from_numpy(data).to(dtype).to(device)


def load_model(checkpoint_path, device):
    """Load model once and return it."""
    from trail import MultiModalDetectionNetwork
    model = MultiModalDetectionNetwork(lidar_dim=128, radar_dim=128, image_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')} | loss={checkpoint.get('loss', 'N/A')}")
    model.eval()
    return model


def run_inference_for_sample(model, sample_idx, device):
    """Run inference for a single sample index; returns (lidar_points, gt_boxes, pred_boxes, pred_scores, debug)."""
    from data_loader import get_LiDAR, get_radar, get_images, get_labels, get_calib
    from trail import voxelize_lidar_proper, voxelize_radar_proper
    from rpn_refinement import parse_label_line
    import spconv.pytorch as spconv
    from PIL import Image

    print(f"\n{'='*80}")
    print(f"📥 SAMPLE {sample_idx}")
    print(f"{'='*80}")

    lidar_data  = get_LiDAR(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    radar_data  = get_radar(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    image_data  = get_images(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    labels      = get_labels(ROOT_DIR, 'training', from_idx=sample_idx, count=1)
    calibs      = get_calib(ROOT_DIR, 'training', from_idx=sample_idx, count=1)

    if len(lidar_data) == 0:
        print(f"❌ No data for sample {sample_idx}"); return None, None, None, None, {}

    lidar_points = lidar_data[0]
    radar_points = radar_data[0]
    image        = image_data[0]

    print(f"   LiDAR: {len(lidar_points):,} pts  "
          f"X[{lidar_points[:,0].min():.1f},{lidar_points[:,0].max():.1f}]  "
          f"Y[{lidar_points[:,1].min():.1f},{lidar_points[:,1].max():.1f}]")

    # Parse GT
    gt_boxes_camera = []
    for line in labels[0]:
        if isinstance(line, str) and line.strip():
            parsed = parse_label_line(line)
            if parsed['type'].lower() in ['car', 'pedestrian', 'cyclist']:
                box = parsed['location'] + parsed['dimensions'] + [parsed['rotation_y']]
                gt_boxes_camera.append(box)
    gt_boxes_camera = np.array(gt_boxes_camera, dtype=np.float32)
    gt_boxes_lidar  = transform_boxes_camera_to_lidar(gt_boxes_camera, calibs[0])
    gt_boxes_lidar  = convert_z_convention(gt_boxes_lidar, GT_Z_CONVENTION, VIS_Z_CONVENTION)
    print(f"   GT boxes (car/ped/cyclist): {len(gt_boxes_lidar)}")

    # Voxelize
    lidar_feat, lidar_coords, lidar_shape, lidar_bs = voxelize_lidar_proper(
        [lidar_points], voxel_size=LIDAR_VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE)
    radar_feat, radar_coords, radar_shape, radar_bs = voxelize_radar_proper(
        [radar_points], voxel_size=RADAR_VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE)

    lidar_sparse = spconv.SparseConvTensor(
        features=to_tensor(lidar_feat, torch.float32, device),
        indices=to_tensor(lidar_coords, torch.int32, device),
        spatial_shape=lidar_shape, batch_size=lidar_bs)
    radar_sparse = spconv.SparseConvTensor(
        features=to_tensor(radar_feat, torch.float32, device),
        indices=to_tensor(radar_coords, torch.int32, device),
        spatial_shape=radar_shape, batch_size=radar_bs)

    image_resized = np.array(Image.fromarray(image).resize((1242, 375)))
    image_tensor  = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Inference
    debug = {}
    try:
        if FORCE_SHOW_PREDICTIONS and hasattr(model, 'detector'):
            model.detector.proposal_generator.score_thresh = 0.0
        elif hasattr(model, 'detector'):
            model.detector.proposal_generator.score_thresh = VIS_SCORE_THRESH

        with torch.no_grad():
            outputs = model(lidar_sparse, radar_sparse, image_tensor, training=False)
            proposals = outputs['detections']['proposals'][0]
            scores    = outputs['detections']['scores'][0]
            print(f"   Raw proposals: {len(proposals)}")
            if len(scores) > 0:
                print(f"   Raw score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")

        if len(proposals) > 0:
            pred_boxes_raw  = proposals.cpu().numpy()
            pred_scores_raw = scores.cpu().numpy()
            pred_boxes_raw  = convert_z_convention(pred_boxes_raw, PRED_Z_CONVENTION, VIS_Z_CONVENTION)
            raw_proposal_diagnostics(pred_boxes_raw, pred_scores_raw, gt_boxes_lidar, near_xy_m=10.0)
            debug["raw_summary"] = summarize_raw_vs_near_gt(pred_boxes_raw, pred_scores_raw, gt_boxes_lidar, topk=VIS_FALLBACK_TOPK)

            pred_boxes  = pred_boxes_raw
            pred_scores = pred_scores_raw

            if pred_scores.size > 0:
                pred_boxes  = pred_boxes[pred_scores >= VIS_SCORE_THRESH]
                pred_scores = pred_scores[pred_scores >= VIS_SCORE_THRESH]
            if pred_scores.size > VIS_TOPK:
                top_idx = np.argsort(pred_scores)[::-1][:VIS_TOPK]
                pred_boxes = pred_boxes[top_idx]; pred_scores = pred_scores[top_idx]
            if pred_scores.size > 0:
                keep = nms_bev_axis_aligned(pred_boxes, pred_scores, VIS_NMS_THRESH)
                pred_boxes = pred_boxes[keep]; pred_scores = pred_scores[keep]

            if VIS_REPORT_RAW_VS_ALIGNED and len(gt_boxes_lidar) > 0 and len(pred_boxes) > 0:
                rep = alignment_report(pred_boxes, gt_boxes_lidar)
                dc = rep["delta_center"]
                print(f"   Raw alignment: Δcenter dx={dc[0]:.2f}, dy={dc[1]:.2f}, dz={dc[2]:.2f} | "
                      f"XY median={rep['match_median_xy']:.2f}m")

            pred_boxes, pred_scores = filter_predictions_for_visualization(pred_boxes, pred_scores, gt_boxes_lidar)
            diagnostics(pred_boxes, pred_scores, gt_boxes_lidar)
        else:
            pred_boxes  = np.zeros((0, 7)); pred_scores = np.array([])
            print("   ⚠️  No raw proposals from model")

    except Exception as e:
        print(f"❌ Inference error on sample {sample_idx}: {e}")
        import traceback; traceback.print_exc()
        pred_boxes  = np.zeros((0, 7)); pred_scores = np.array([])

    return lidar_points, gt_boxes_lidar, pred_boxes, pred_scores, debug


def visualize_scene(geometries, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1600, height=900)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = 6.0
        opt.background_color = np.array([0.0, 0.0, 0.0])
        opt.line_width = 2.0
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


def visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=0):
    print(f"\n{'='*80}")
    print(f"🎨 VISUALIZATION  sample={sample_idx}")
    print(f"{'='*80}")

    if lidar_points is None or len(lidar_points) == 0:
        print("❌ No LiDAR points"); return

    finite_mask = np.isfinite(lidar_points[:, :3]).all(axis=1)
    if not finite_mask.all():
        lidar_points = lidar_points[finite_mask]

    mask_x = (lidar_points[:, 0] >= POINT_CLOUD_RANGE[0]) & (lidar_points[:, 0] <= POINT_CLOUD_RANGE[3])
    mask_y = (lidar_points[:, 1] >= POINT_CLOUD_RANGE[1]) & (lidar_points[:, 1] <= POINT_CLOUD_RANGE[4])
    lidar_filt = lidar_points[mask_x & mask_y]
    if len(lidar_filt) == 0:
        lidar_filt = lidar_points

    print(f"   LiDAR: {len(lidar_filt):,}  GT: {len(gt_boxes)}  Pred: {len(pred_boxes)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_filt[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(lidar_filt), 3), 0.9, dtype=np.float32))

    gt_boxes_vis   = gt_boxes
    pred_boxes_vis = pred_boxes
    if SNAP_TO_GROUND and len(lidar_filt) > 0:
        pts_xyz = lidar_filt[:, :3]
        if len(gt_boxes_vis) > 0:
            gt_c = convert_z_convention(gt_boxes_vis, VIS_Z_CONVENTION, "center")
            gt_c = snap_boxes_to_ground(gt_c, pts_xyz, SNAP_RADIUS_M, SNAP_MIN_POINTS, SNAP_GROUND_PERCENTILE)
            gt_boxes_vis = convert_z_convention(gt_c, "center", VIS_Z_CONVENTION)
        if len(pred_boxes_vis) > 0:
            pr_c = convert_z_convention(pred_boxes_vis, VIS_Z_CONVENTION, "center")
            pr_c = snap_boxes_to_ground(pr_c, pts_xyz, SNAP_RADIUS_M, SNAP_MIN_POINTS, SNAP_GROUND_PERCENTILE)
            pred_boxes_vis = convert_z_convention(pr_c, "center", VIS_Z_CONVENTION)

    gt_lineset   = create_bbox_lineset(gt_boxes_vis,   color=[0, 1, 0])
    pred_lineset = create_bbox_lineset(pred_boxes_vis, color=[1, 0, 0])

    center = lidar_filt[:, :3].mean(axis=0) if len(lidar_filt) > 0 else np.zeros(3)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=center)

    geometries = [pcd, coord_frame]
    if len(gt_boxes) > 0:
        geometries.append(gt_lineset)
    if SHOW_PRED_BOXES and len(pred_boxes) > 0:
        geometries.append(pred_lineset)

    label = "LiDAR + GT + Predictions" if SHOW_PRED_BOXES else "LiDAR + GT"
    visualize_scene(geometries, f"Sample {sample_idx} | {label}")
    print(f"✅ Visualization closed  (sample {sample_idx})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*80)
    print("3D DETECTION VISUALIZATION – MULTI-SAMPLE")
    print("="*80)

    # Resolve sample list
    if SAMPLE_INDICES is not None:
        sample_list = list(SAMPLE_INDICES)
    else:
        start, stop = SAMPLE_RANGE
        sample_list = list(range(start, stop))

    print(f"Samples to process: {sample_list}")
    total = len(sample_list)

    # Load model ONCE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}"); sys.exit(1)
    model = load_model(CHECKPOINT_PATH, device)

    # Per-sample summary collected for final report
    summary_rows = []

    for i, sample_idx in enumerate(sample_list):
        print(f"\n{'#'*80}")
        print(f"# SAMPLE {sample_idx}  ({i+1}/{total})")
        print(f"{'#'*80}")

        lidar_points, gt_boxes, pred_boxes, pred_scores, debug = run_inference_for_sample(
            model, sample_idx, device
        )

        if lidar_points is None:
            summary_rows.append({"idx": sample_idx, "gt": 0, "pred": 0, "status": "SKIP (no data)"})
            continue

        # Open3D window (blocking – closes before next sample opens)
        visualize(lidar_points, gt_boxes, pred_boxes, sample_idx=sample_idx)

        # Collect per-sample stats
        row = {"idx": sample_idx, "gt": len(gt_boxes), "pred": len(pred_boxes), "status": "OK"}
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            rep = alignment_report(pred_boxes, gt_boxes)
            row["xy_median"] = rep["match_median_xy"]
            row["xyz_median"] = rep["match_median_xyz"]
        summary_rows.append(row)

    # ── Final multi-sample report ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"📊 MULTI-SAMPLE SUMMARY  ({total} samples)")
    print(f"{'='*80}")
    print(f"{'Idx':>5}  {'GT':>4}  {'Pred':>5}  {'XY med':>8}  {'XYZ med':>8}  Status")
    print(f"{'-'*5}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in summary_rows:
        xy  = f"{r['xy_median']:.2f}m"  if 'xy_median'  in r else "  n/a  "
        xyz = f"{r['xyz_median']:.2f}m" if 'xyz_median' in r else "  n/a  "
        print(f"{r['idx']:>5}  {r['gt']:>4}  {r['pred']:>5}  {xy:>8}  {xyz:>8}  {r['status']}")
    print(f"{'='*80}\n")
