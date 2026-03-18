import os
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import visualize_predictions as vp
from data_loader import get_images, get_radar

try:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except Exception as exc:
    raise RuntimeError("Open3D GUI modules are required for this script.") from exc


SAMPLE_IDX = 1
EPOCH = 52
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f"checkpoint_epoch_{EPOCH}.pth"))
ROOT_DIR = vp.ROOT_DIR


class VisualAllApp:
    def __init__(self, image_np, radar_points, lidar_points, pred_boxes):
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window("Visual All | epoch 52 sample 1", 1920, 1080)
        self.window.set_on_layout(self._on_layout)

        self.margin = 10
        self.label_h = 26

        self.title_image = gui.Label("Input: Image")
        self.title_radar = gui.Label("Input: Radar")
        self.title_lidar = gui.Label("Input: LiDAR")
        self.title_output = gui.Label("Output: LiDAR with Object Detection")

        self.img_widget = gui.ImageWidget(self._to_o3d_image(image_np))
        self.radar_scene = gui.SceneWidget()
        self.radar_scene.scene = rendering.Open3DScene(self.window.renderer)
        self.lidar_scene = gui.SceneWidget()
        self.lidar_scene.scene = rendering.Open3DScene(self.window.renderer)
        self.output_scene = gui.SceneWidget()
        self.output_scene.scene = rendering.Open3DScene(self.window.renderer)

        for w in [
            self.title_image,
            self.img_widget,
            self.title_radar,
            self.radar_scene,
            self.title_lidar,
            self.lidar_scene,
            self.title_output,
            self.output_scene,
        ]:
            self.window.add_child(w)

        self._setup_radar_scene(radar_points)
        self._setup_lidar_scene(lidar_points)
        self._setup_output_scene(lidar_points, pred_boxes)

    def _to_o3d_image(self, image_np):
        img = np.asarray(image_np)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return o3d.geometry.Image(img)

    def _pcd_from_points(self, points, color):
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            pts = np.zeros((0, 3), dtype=np.float32)
        pts = pts[:, :3]
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) == 0:
            pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float32), (len(pts), 1)))
        return pcd

    def _setup_scene_common(self, scene_widget):
        scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])

    def _setup_radar_scene(self, radar_points):
        self._setup_scene_common(self.radar_scene)
        radar_pcd = self._pcd_from_points(radar_points, [0.2, 0.8, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        self.radar_scene.scene.add_geometry("radar", radar_pcd, mat)
        bbox = self.radar_scene.scene.bounding_box
        self.radar_scene.setup_camera(60.0, bbox, bbox.get_center())

    def _setup_lidar_scene(self, lidar_points):
        self._setup_scene_common(self.lidar_scene)
        lidar_pcd = self._pcd_from_points(lidar_points, [0.85, 0.85, 0.85])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0
        self.lidar_scene.scene.add_geometry("lidar", lidar_pcd, mat)
        bbox = self.lidar_scene.scene.bounding_box
        self.lidar_scene.setup_camera(60.0, bbox, bbox.get_center())

    def _setup_output_scene(self, lidar_points, pred_boxes):
        self._setup_scene_common(self.output_scene)

        lidar_pcd = self._pcd_from_points(lidar_points, [0.85, 0.85, 0.85])
        mat_points = rendering.MaterialRecord()
        mat_points.shader = "defaultUnlit"
        mat_points.point_size = 2.0
        self.output_scene.scene.add_geometry("output_lidar", lidar_pcd, mat_points)

        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_lines = vp.create_bbox_lineset(pred_boxes, color=[1, 0, 0])
            mat_lines = rendering.MaterialRecord()
            mat_lines.shader = "unlitLine"
            mat_lines.line_width = 2.0
            self.output_scene.scene.add_geometry("pred_boxes", pred_lines, mat_lines)

        bbox = self.output_scene.scene.bounding_box
        self.output_scene.setup_camera(60.0, bbox, bbox.get_center())

    def _on_layout(self, _):
        r = self.window.content_rect
        m = self.margin
        w = max(100, (r.width - (3 * m)) // 2)
        h = max(100, (r.height - (3 * m) - (2 * self.label_h)) // 2)

        x1 = r.x + m
        x2 = x1 + w + m
        y1 = r.y + m
        y2 = y1 + self.label_h + h + m

        self.title_image.frame = gui.Rect(x1, y1, w, self.label_h)
        self.img_widget.frame = gui.Rect(x1, y1 + self.label_h, w, h)

        self.title_radar.frame = gui.Rect(x2, y1, w, self.label_h)
        self.radar_scene.frame = gui.Rect(x2, y1 + self.label_h, w, h)

        self.title_lidar.frame = gui.Rect(x1, y2, w, self.label_h)
        self.lidar_scene.frame = gui.Rect(x1, y2 + self.label_h, w, h)

        self.title_output.frame = gui.Rect(x2, y2, w, self.label_h)
        self.output_scene.frame = gui.Rect(x2, y2 + self.label_h, w, h)

    def run(self):
        self.app.run()


def filter_points_in_xy_range(points, point_cloud_range):
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)

    finite = np.isfinite(pts[:, :3]).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return pts

    mask_x = (pts[:, 0] >= point_cloud_range[0]) & (pts[:, 0] <= point_cloud_range[3])
    mask_y = (pts[:, 1] >= point_cloud_range[1]) & (pts[:, 1] <= point_cloud_range[4])
    pts = pts[mask_x & mask_y]
    return pts


def main():
    print("=" * 80)
    print("VISUAL ALL - SINGLE PAGE")
    print("=" * 80)
    print(f"Using epoch={EPOCH}, sample={SAMPLE_IDX}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = vp.load_model(CHECKPOINT_PATH, device)
    lidar_points, _, pred_boxes, _, _ = vp.run_inference_for_sample(model, SAMPLE_IDX, device)

    if lidar_points is None:
        raise RuntimeError(f"No LiDAR data found for sample {SAMPLE_IDX}")

    images = get_images(ROOT_DIR, "training", from_idx=SAMPLE_IDX, count=1)
    if len(images) == 0:
        raise RuntimeError(f"No image found for sample {SAMPLE_IDX}")
    image_np = images[0]

    radar_list = get_radar(ROOT_DIR, "training", from_idx=SAMPLE_IDX, count=1)
    if len(radar_list) == 0:
        raise RuntimeError(f"No radar data found for sample {SAMPLE_IDX}")
    radar_points = radar_list[0][:, :3]

    lidar_points = filter_points_in_xy_range(lidar_points, vp.POINT_CLOUD_RANGE)
    radar_points = filter_points_in_xy_range(radar_points, vp.POINT_CLOUD_RANGE)

    app = VisualAllApp(
        image_np=image_np,
        radar_points=radar_points,
        lidar_points=lidar_points,
        pred_boxes=pred_boxes,
    )
    app.run()


if __name__ == "__main__":
    main()
