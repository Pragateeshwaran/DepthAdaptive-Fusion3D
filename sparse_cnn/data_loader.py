import os
import numpy as np
import matplotlib.pyplot as plt

ROOTDIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'

def find_lidar_files(root_dir=ROOTDIR, split="training"):
    """
    Find all LiDAR .bin files (train/test) from V2X-Radar-V folder structure.
    
    Args:
        root_dir (str): Base dataset folder (e.g., F:\\Work\\DeepLearning\\Research\\V2X-Radar-V)
        split (str): 'train' or 'test'
    
    Returns:
        list: Sorted list of full file paths to .bin files
    """
    if split == 'training':
        lidar_dir = os.path.join(root_dir, 'training', 'velodyne')
        print(lidar_dir)
    else:
        lidar_dir = os.path.join(root_dir, 'testing', 'velodyne')
    
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"Path not found: {lidar_dir}")

    lidar_files = [
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith('.bin')
    ]
    return sorted(lidar_files)

def find_radar_files(root_dir=ROOTDIR, split="training"):
    """
    Find all radar .bin files from V2X-Radar-V folder structure.
    
    Args:
        root_dir (str): Base dataset folder
        split (str): 'train' or 'test'
    
    Returns:
        list: Sorted list of full file paths to radar .bin files
    """
    if split == 'training':
        radar_dir = os.path.join(root_dir, 'training', 'radar')
    else:
        radar_dir = os.path.join(root_dir, 'testing', 'radar')
    
    if not os.path.exists(radar_dir):
        raise FileNotFoundError(f"Path not found: {radar_dir}")

    radar_files = [
        os.path.join(radar_dir, f)
        for f in os.listdir(radar_dir)
        if f.endswith('.bin')
    ]
    return sorted(radar_files)

def get_LiDAR(root_dir=ROOTDIR, split='training', from_idx=0, count=None):
    """
    Load LiDAR .bin files from V2X-Radar-V dataset into NumPy arrays.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
        from_idx (int): Starting index
        count (int): Number of files to load
    
    Returns:
        list: List of NumPy arrays, each (N, 4) → (x, y, z, reflectance)
    """
    bin_files = find_lidar_files(root_dir, split)
    if count is not None:
        bin_files = bin_files[from_idx:from_idx+count]

    lidar_arrays = []

    for file in bin_files:
        # print(file)
        points = np.fromfile(file, dtype=np.float32)
        if points.shape[0] % 4 == 0:
            points = points.reshape(-1, 4)  # LiDAR: x,y,z,intensity
            points = points[:, :3]  # Keep only x,y,z
        else:
            points = points.reshape(-1, 3)  # Some LiDARs may have an extra dimension
        
        # Remove NaN/Inf values (V2X-Radar-V uses NaN padding for fixed-size arrays)
        # Only check coordinates (x, y, z) - intensity NaN is less critical
        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        
        if not valid_mask.all():
            points = points[valid_mask]
        
        # Ensure we have at least some valid points
        if len(points) == 0:
            print(f"Warning: {file} has no valid points, skipping")
            continue
        # points = lidar_fps_gpu(points, N=20000)  # Changed from 16000 to 20000
        lidar_arrays.append(points)

    return lidar_arrays

def get_radar(root_dir=ROOTDIR, split='training', from_idx=0, count=None):
    """
    Load radar .bin files from V2X-Radar-V dataset into NumPy arrays.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
        from_idx (int): Starting index
        count (int): Number of files to load
    
    Returns:
        list: List of NumPy arrays with radar data
    """
    bin_files = find_radar_files(root_dir, split)
    if count is not None:
        bin_files = bin_files[from_idx:from_idx+count]

    radar_arrays = []

    for file in bin_files:
        # Adjust dtype and reshape based on your radar data format
        # print(file)
        points = np.fromfile(file, dtype=np.float32).reshape(-1, 5)  
        
        # Remove NaN/Inf values from radar data too
        valid_mask = np.isfinite(points).all(axis=1)
        if not valid_mask.all():
            points = points[valid_mask]
        
        # Ensure we have at least some valid points
        if len(points) == 0:
            print(f"Warning: {file} has no valid points, skipping")
            continue
        
        # Apply FPS to radar data (20k points like LiDAR)
        # points = radar_fps_gpu(points, N=8000)
        radar_arrays.append(points)

    return radar_arrays

def radar_fps_gpu(points, N=20000):
    """
    Farthest Point Sampling (FPS) on radar point cloud (GPU version).
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        points (torch.Tensor or np.ndarray): (M, 7) radar point cloud
        N (int): Number of points to sample

    Returns:
        torch.Tensor or np.ndarray: (N, 7) sampled points (same type as input)
    """
    import torch
    
    # Convert NumPy to Torch if needed
    was_numpy = isinstance(points, np.ndarray)
    if was_numpy:
        points = torch.from_numpy(points).float()
    
    device = points.device
    M = points.shape[0]

    # Case 1: M == N
    if M == N:
        return points.cpu().numpy() if was_numpy else points

    # Case 2: M < N → duplicate points (with replacement)
    if M < N:
        choice = torch.randint(0, M, (N - M,), device=device)
        extra_points = points[choice]
        result = torch.cat([points, extra_points], dim=0)
        return result.cpu().numpy() if was_numpy else result

    # Case 3: M > N → FPS
    xyz = points[:, :3]  # only XYZ for distance computation

    centroids = torch.zeros(N, dtype=torch.long, device=device)
    distances = torch.full((M,), 1e10, device=device)

    farthest = torch.randint(0, M, (1,), device=device).item()

    for i in range(N):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()

    result = points[centroids]
    return result.cpu().numpy() if was_numpy else result


def lidar_fps_gpu(points, N=20000):
    """
    Farthest Point Sampling (FPS) on a single LiDAR point cloud (GPU version).
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        points (torch.Tensor or np.ndarray): (M, 4) point cloud
        N (int): Number of points to sample

    Returns:
        torch.Tensor or np.ndarray: (N, 4) sampled points (same type as input)
    """
    import torch
    
    # Convert NumPy to Torch if needed
    was_numpy = isinstance(points, np.ndarray)
    if was_numpy:
        points = torch.from_numpy(points).float()
    
    device = points.device
    M = points.shape[0]

    # Case 1: M == N
    if M == N:
        return points.cpu().numpy() if was_numpy else points

    # Case 2: M < N → duplicate points (with replacement)
    if M < N:
        choice = torch.randint(0, M, (N - M,), device=device)
        extra_points = points[choice]
        result = torch.cat([points, extra_points], dim=0)
        return result.cpu().numpy() if was_numpy else result

    # Case 3: M > N → FPS
    xyz = points[:, :3]  # only XYZ for distance computation

    centroids = torch.zeros(N, dtype=torch.long, device=device)
    distances = torch.full((M,), 1e10, device=device)

    farthest = torch.randint(0, M, (1,), device=device).item()

    for i in range(N):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).item()

    result = points[centroids]
    return result.cpu().numpy() if was_numpy else result


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

def find_image_files(root_dir=ROOTDIR, split="training"):
    """
    Find all image files from V2X-Radar-V folder structure.
    
    Args:
        root_dir (str): Base dataset folder
        split (str): 'train' or 'test'
    Returns:
        list: Sorted list of full file paths to image files
    """
    if split == 'training':
        image_dir = os.path.join(root_dir, 'training', 'image_2')
    else:
        image_dir = os.path.join(root_dir, 'testing', 'image_2')
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Path not found: {image_dir}")

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ]
    return sorted(image_files)

def get_images(root_dir=ROOTDIR, split='training', from_idx=0, count=None):
    """
    Load image files from V2X-Radar-V dataset into a list of images.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
        from_idx (int): Starting index
        count (int): Number of files to load
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

def find_calib_files(root_dir=ROOTDIR, split="train"):
    """
    Find calibration files from V2X-Radar-V dataset.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
    Returns:
        list: Sorted list of calibration file paths
    """
    if split == 'training':
        calib_dir = os.path.join(root_dir, 'training', 'calib')
    else:
        calib_dir = os.path.join(root_dir, 'testing', 'calib')
    
    if not os.path.exists(calib_dir):
        raise FileNotFoundError(f"Path not found: {calib_dir}")
    
    files = []
    for f in os.listdir(calib_dir):
        if f.endswith('.txt'):
            files.append(os.path.join(calib_dir, f))
    return sorted(files)

def get_calib(root_dir=ROOTDIR, split='training', from_idx=0, count=None):
    """
    Read calibration files and return a list of dictionaries with calibration parameters.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
        from_idx (int): Starting index
        count (int): Number of files to load
    Returns:
        list: List of dictionaries with calibration parameters
    """
    json_files = []
    data = {}
    files = find_calib_files(root_dir, split)
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

def find_label_files(root_dir=ROOTDIR, split="training"):
    """
    Find label files from V2X-Radar-V dataset.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
    Returns:
        list: Sorted list of label file paths
    """
    if split == 'training':
        label_dir = os.path.join(root_dir, 'training', 'label_2')
    else:
        label_dir = os.path.join(root_dir, 'testing', 'label_2')
    
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Path not found: {label_dir}")
    
    files = []
    for f in os.listdir(label_dir):
        if f.endswith('.txt'):
            files.append(os.path.join(label_dir, f))
    return sorted(files)

def get_labels(root_dir=ROOTDIR, split='training', from_idx=0, count=None):
    """
    Read label files from V2X-Radar-V dataset.
    
    Args:
        root_dir (str): Root dataset directory
        split (str): 'train' or 'test'
        from_idx (int): Starting index
        count (int): Number of files to load
    Returns:
        list: List of label data
    """
    label_files = find_label_files(root_dir, split)
    if count is not None:
        label_files = label_files[from_idx:from_idx+count]
    
    labels = []
    for filepath in label_files:
        with open(filepath, "r") as f:
            label_data = f.readlines()
        labels.append(label_data)
    
    return labels

def image_resize(img, target_size=(375, 1242)):
    """
    Crop the image to match the target aspect ratio, then resize.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        target_size (tuple): Desired output size (height, width).

    Returns:
        np.ndarray: Cropped + resized image.
    """
    from PIL import Image, ImageOps
    pil_img = Image.fromarray(img)

    # Crop to fit target aspect ratio, then resize
    cropped_resized = ImageOps.fit(
        pil_img,
        (target_size[1], target_size[0]),  # PIL uses (width, height)
        method=Image.LANCZOS,
        centering=(0.5, 0.5)  # center crop
    )
    return np.array(cropped_resized)