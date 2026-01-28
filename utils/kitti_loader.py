import os
import numpy as np
import matplotlib.pyplot as plt
ROOTDIR = r'F:\Work\DeepLearning\Research\dataset'

def find_kitti_files(root_dir=ROOTDIR, split="train"):
    """
    Find all KITTI LiDAR .bin files (train/test) from your folder structure.
    
    Args:
        root_dir (str): Base dataset folder (e.g., F:\\Work\\DeepLearning\\Research\\dataset)
        split (str): 'train' or 'test'
    
    Returns:
        list: Sorted list of full file paths to .bin files
    """
    if split == 'train':
        lidar_dir = os.path.join(ROOTDIR, 'data_object_velodyne', 'training', 'velodyne')
    else:
        lidar_dir = os.path.join(ROOTDIR, 'data_object_velodyne', 'testing', 'velodyne')
    
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"Path not found: {lidar_dir}")

    kitti_files = [
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith('.bin')
    ]
    return sorted(kitti_files)

def get_LiDAR(root_dir=ROOTDIR, split='train', from_idx=0, count = None):
    """
    Load all KITTI LiDAR .bin files from your dataset into NumPy arrays.
    
    Args:
        root_dir (str): Root KITTI dataset directory
        split (str): 'train' or 'test'
    
    Returns:
        list: List of NumPy arrays, each (N, 4) → (x, y, z, reflectance)
    """
    bin_files = find_kitti_files(root_dir, split)
    if count is not None:
        bin_files = bin_files[from_idx:from_idx+count]

    lidar_arrays = []

    for file in bin_files:
        points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        lidar_arrays.append(points)

    return lidar_arrays


def lidar_fps(points, N=1000):
    """
    Farthest Point Sampling (FPS) on a single LiDAR point cloud in real-time.
    
    Args:
        points (np.ndarray): Point cloud of shape (M, 4)
        N (int): Number of points to sample
    Returns:
        np.ndarray: Sampled point cloud of shape (N, 4)
    """

    M, _ = points.shape
    
    if M == N:
        return points
    
    if M < N:
        choice = np.random.choice(M, N - M, replace=True)
        extra_points = points[choice, :]
        return np.vstack((points, extra_points))
    
    centroids = np.zeros(N, dtype=int)
    distances = np.ones(M) * 1e10 
    farthest = np.random.randint(0, M) 

    for i in range(N): 
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid)**2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return points[centroids, :]


def LiDAR_downsample(points, downsample_rate=2):
    """
    Downsample a LiDAR point cloud using angle-based ring selection.
    
    Args:
        points (np.ndarray): NumPy array of shape (M, 4) where columns are [x, y, z, intensity/reflectance]
        downsample_rate (int): Rate at which to downsample rings (e.g., 2 means keep every other ring)
    
    Returns:
        np.ndarray: Downsampled array of shape (N, 4)
    """
    from sklearn.cluster import KMeans
    
    xyz = points[:, :3]   
    xyz_norm = np.sqrt(np.sum(xyz * xyz, axis=1, keepdims=True))
    z_axis = np.array([[0], [0], [1]])
    cos = xyz.dot(z_axis) / xyz_norm  
    
    kmeans = KMeans(n_clusters=64, n_init=10, random_state=42).fit(cos)
    centers = np.sort(np.squeeze(kmeans.cluster_centers_))
    centers = [-1] + centers.tolist() + [1]
    cos = np.squeeze(cos)
    
    point_total_mask = np.zeros(len(xyz), dtype=bool)
    
    for i in range(0, len(centers) - 2, downsample_rate):
        lower = (centers[i] + centers[i + 1]) / 2
        higher = (centers[i + 1] + centers[i + 2]) / 2
        point_mask = (cos > lower) & (cos < higher)
        point_total_mask = point_total_mask | point_mask   
     
    output = points[point_total_mask, :]
    
    return output

def find_image_files(root_dir=ROOTDIR, split="train"):
    """
    Find all KITTI image files (train/test) from your folder structure.
    
    Args:
        root_dir (str): Base dataset folder (e.g., F:\\Work\\DeepLearning\\Research\\dataset)
        split (str): 'train' or 'test'
    Returns:
        list: Sorted list of full file paths to image files
    """
    if split == 'train':
        image_dir = os.path.join(ROOTDIR, 'image_processed', 'training', 'image_2_resized')
    else:
        image_dir = os.path.join(ROOTDIR, 'image_processed', 'testing', 'image_2_resized')
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Path not found: {image_dir}")

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ]
    return sorted(image_files)

def get_images(root_dir=ROOTDIR, split='train', from_idx = 0, count = None):
    """
    Load all KITTI image files from your dataset into a list of images.
    
    Args:
        root_dir (str): Root KITTI dataset directory
        split (str): 'train' or 'test'
    Returns:
        list: List of images as NumPy arrays 
    """
    image_files = find_image_files(root_dir, split)
    images = []
    if count is not None:
        image_files = image_files[from_idx:from_idx+count]
        
    for file in image_files:
        img = plt.imread(file)
        images.append(img)

    return images

def find_calib_files(filepath = r"F:\Work\DeepLearning\Research\dataset\data_object_calib\training\calib"):
    """
    Read KITTI calibration file and return a dictionary of calibration parameters.
    
    Args:
        filepath (str): Path to the KITTI calibration file.
    Returns:
        dict: Dictionary with calibration parameters.
    """
    files = []
    for f in os.listdir(filepath):
        if f.endswith('.txt'):
            files.append(os.path.join(filepath, f))
    files = sorted(files)
    return files



def get_calib(from_idx=0, count=None):
    """
    Read KITTI calibration files and return a dictionary of calibration parameters.
    Args:
        filepath (str): Path to the KITTI calibration file.
    Returns:
        dict: Dictionary with calibration parameters.
    """
    json_files = []
    data = {}
    files = find_calib_files()
    if count is not None:
        files = files[from_idx:from_idx+count]
    for filepath in files:
        with open(filepath, "r") as f:
            for line in f.readlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values = [float(x) for x in value.strip().split()]
                data[key] = np.array(values)
        json_files.append(data)
        data = {}
    return json_files

def image_resize(img, target_size=(375, 1242)):
    """
    Crop the image to match the target aspect ratio, then resize.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Cropped + resized image.
    """
    from PIL import Image, ImageOps
    pil_img = Image.fromarray(img)

    # Crop to fit target aspect ratio, then resize
    cropped_resized = ImageOps.fit(
        pil_img,
        target_size,
        method=Image.LANCZOS,
        centering=(0.5, 0.5)  # center crop
    )
    return np.array(cropped_resized)

def image_fps():
    ...