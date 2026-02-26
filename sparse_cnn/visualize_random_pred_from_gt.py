import os
import sys

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_LiDAR, get_labels, get_calib
from rpn_refinement import parse_label_line


ROOT_DIR = r"F:\Work\DeepLearning\Research\V2X-Radar-V"
SAMPLE_IDX = 0
POINT_CLOUD_RANGE = (-50, -50, -3, 50, 50, 5)
SEED = 42


def transform_boxes_camera_to_lidar(boxes_camera, calib):
    if len(boxes_camera) == 0:
        return boxes_camera

    tr_velo_to_cam = calib.get("Tr_velo_to_cam", np.eye(4))
    r0_rect = calib.get("R0_rect", np.eye(3))

    if tr_velo_to_cam.shape == (12,):
        tr_velo_to_cam = tr_velo_to_cam.reshape(3, 4)
        tr_velo_to_cam = np.vstack([tr_velo_to_cam, [0, 0, 0, 1]])
    elif tr_velo_to_cam.shape == (3, 4):
        tr_velo_to_cam = np.vstack([tr_velo_to_cam, [0, 0, 0, 1]])

    if r0_rect.shape == (9,):
        r0_rect = r0_rect.reshape(3, 3)
    if r0_rect.shape == (3, 3):
        r0_4x4 = np.eye(4, dtype=np.float32)
        r0_4x4[:3, :3] = r0_rect
    elif r0_rect.shape == (4, 4):
        r0_4x4 = r0_rect
    else:
        r0_4x4 = np.eye(4, dtype=np.float32)

    tr_cam_to_velo = np.linalg.inv(r0_4x4 @ tr_velo_to_cam)
    boxes_lidar = np.zeros_like(boxes_camera, dtype=np.float32)

    for i in range(len(boxes_camera)):
        x_cam, y_cam, z_cam = boxes_camera[i, 0:3]
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
        point_lidar = tr_cam_to_velo @ point_cam
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


def get_box_corners(box):
    x, y, z, h, w, l, rot = box
    corners = np.array(
        [
            [-l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(rot), np.sin(rot)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return corners @ r.T + np.array([x, y, z], dtype=np.float32)


def create_lineset(boxes, color):
    if len(boxes) == 0:
        return o3d.geometry.LineSet()

    points = []
    lines = []
    edges = create_bbox_lines()
    offset = 0

    for box in boxes:
        c = get_box_corners(box)
        points.append(c)
        for e in edges:
            lines.append([e[0] + offset, e[1] + offset])
        offset += 8

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(points))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float32), (len(lines), 1)))
    return ls


def load_gt_boxes(sample_idx=0):
    labels = get_labels(ROOT_DIR, "training", from_idx=sample_idx, count=1)
    calibs = get_calib(ROOT_DIR, "training", from_idx=sample_idx, count=1)
    if len(labels) == 0 or len(calibs) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    class_map = {"pedestrian": 0, "cyclist": 1, "car": 2}
    boxes_camera = []
    for line in labels[0]:
        if isinstance(line, str) and line.strip():
            parsed = parse_label_line(line)
            if parsed["type"].lower() in class_map:
                box = parsed["location"] + parsed["dimensions"] + [parsed["rotation_y"]]
                boxes_camera.append(box)

    if not boxes_camera:
        return np.zeros((0, 7), dtype=np.float32)

    boxes_camera = np.asarray(boxes_camera, dtype=np.float32)
    gt_boxes = transform_boxes_camera_to_lidar(boxes_camera, calibs[0])
    gt_boxes[:, 2] += 0.5 * gt_boxes[:, 3]  # bottom-z -> center-z
    return gt_boxes


def main():
    np.random.seed(SEED)

    lidar = get_LiDAR(ROOT_DIR, "training", from_idx=SAMPLE_IDX, count=1)
    if len(lidar) == 0:
        print("No LiDAR sample found.")
        return

    lidar_points = lidar[0]
    gt_boxes = load_gt_boxes(SAMPLE_IDX)
    pred_boxes = gt_boxes + np.random.rand(*gt_boxes.shape).astype(np.float32)

    in_x = (lidar_points[:, 0] >= POINT_CLOUD_RANGE[0]) & (lidar_points[:, 0] <= POINT_CLOUD_RANGE[3])
    in_y = (lidar_points[:, 1] >= POINT_CLOUD_RANGE[1]) & (lidar_points[:, 1] <= POINT_CLOUD_RANGE[4])
    lidar_points = lidar_points[in_x & in_y]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])

    gt_lines = create_lineset(gt_boxes, [0, 1, 0])   # green
    pred_lines = create_lineset(pred_boxes, [1, 0, 0])  # red

    print(f"GT boxes: {len(gt_boxes)}")
    print(f"Pred boxes: {len(pred_boxes)}")
    print("Pred generation: pred_boxes = gt_boxes + np.random.rand(*gt_boxes.shape)")

    o3d.visualization.draw_geometries(
        [pcd, gt_lines, pred_lines],
        window_name=f"Random Pred From GT | sample {SAMPLE_IDX}",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    main()
