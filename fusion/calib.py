import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import utils.kitti_loader as kitti_loader


def calib_lidar_to_camera(lidar_points, calib_data):
    if isinstance(lidar_points, list):
        lidar_points = lidar_points[0]
        calib_data = calib_data[0]

    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
    R0_rect = calib_data['R0_rect']
    P2 = calib_data['P2']

    if Tr_velo_to_cam.shape == (12,):
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
    if R0_rect.shape == (9,):
        R0_rect = R0_rect.reshape(3, 3)
    if P2.shape == (12,):
        P2 = P2.reshape(3, 4)

    lidar_xyz = lidar_points[:, :3]
    N = lidar_xyz.shape[0]

    lidar_h = np.hstack([lidar_xyz, np.ones((N, 1))])

    cam = (Tr_velo_to_cam @ lidar_h.T).T
    cam_rect = (R0_rect @ cam.T).T

    valid = cam_rect[:, 2] > 0
    cam_rect = cam_rect[valid]

    cam_rect_h = np.hstack([cam_rect, np.ones((cam_rect.shape[0], 1))])

    img = (P2 @ cam_rect_h.T).T
    img[:, 0] /= img[:, 2]
    img[:, 1] /= img[:, 2]

    return img[:, :2]


if __name__ == "__main__":

    lidar_points = kitti_loader.get_LiDAR(count=1)
    calib_data = kitti_loader.get_calib(count=1)
    images = kitti_loader.get_images(count=1)

    img_points_2d = calib_lidar_to_camera(lidar_points, calib_data)
    image = images[0]

    h, w = image.shape[:2]
    mask = (
        (img_points_2d[:, 0] >= 0) &
        (img_points_2d[:, 0] < w) &
        (img_points_2d[:, 1] >= 0) &
        (img_points_2d[:, 1] < h)
    )
    img_points_2d = img_points_2d[mask]

    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    plt.scatter(img_points_2d[:, 0], img_points_2d[:, 1], s=0.3, c='red')
    plt.axis('off')
    plt.show()
