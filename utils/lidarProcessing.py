import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(r'F:\Work\DeepLearning\Research')
from utils.kitti_loader import get_LiDAR, get_calib, lidar_fps, get_images

ROOTDIR = r'F:\Work\DeepLearning\Research\dataset'

def process_and_save_fps_lidar(split='train', N=16384, output_dir=None):
    """
    Process all LiDAR files: calibrate, filter by image bounds, apply FPS, and save as numpy arrays
    
    Args:
        split (str): 'train' or 'test'
        N (int): Number of points to sample using FPS (default: 16384 as in paper)
        output_dir (str): Output directory to save processed files
    
    Returns:
        None (saves files to disk)
    """
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(ROOTDIR, 'processed_lidar', split)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {split} split...")
    print(f"Target FPS samples: {N}")
    print(f"Output directory: {output_dir}")
    
    # Load all LiDAR data
    print("\n1. Loading LiDAR files...")
    lidar_arrays = get_LiDAR(ROOTDIR, split=split)
    print(f"   Loaded {len(lidar_arrays)} LiDAR scans")
    
    # Load all images to get dimensions
    print("\n2. Loading images...")
    images = get_images(ROOTDIR, split=split)
    print(f"   Loaded {len(images)} images")
    
    # Load all calibration files
    print("\n3. Loading calibration files...")
    if split == 'train':
        calib_path = os.path.join(ROOTDIR, 'data_object_calib', 'training', 'calib')
    else:
        calib_path = os.path.join(ROOTDIR, 'data_object_calib', 'testing', 'calib')
    
    # Get calibration files
    calib_files = []
    for f in sorted(os.listdir(calib_path)):
        if f.endswith('.txt'):
            calib_files.append(os.path.join(calib_path, f))
    
    print(f"   Loaded {len(calib_files)} calibration files")
    
    # Verify matching counts
    assert len(lidar_arrays) == len(calib_files) == len(images), \
        f"Mismatch: {len(lidar_arrays)} LiDAR, {len(calib_files)} calib, {len(images)} images"
    
    # Process each LiDAR scan
    print(f"\n4. Processing and saving {len(lidar_arrays)} files...")
    
    stats = {
        'original_points': [],
        'after_calibration': [],
        'after_image_filter': [],
        'after_fps': []
    }
    
    for idx, (lidar_data, calib_file, image) in enumerate(tqdm(
        zip(lidar_arrays, calib_files, images), 
        total=len(lidar_arrays),
        desc="Processing"
    )):
        
        stats['original_points'].append(lidar_data.shape[0])
        
        # Read calibration data
        calib = {}
        with open(calib_file, "r") as f:
            for line in f.readlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values = [float(x) for x in value.strip().split()]
                    calib[key] = np.array(values)
        
        # Get image dimensions
        H, W, _ = image.shape
        
        # ========================================
        # CALIBRATION: Transform LiDAR to Camera Coordinates
        # ========================================
        # Step 1: Velodyne -> Camera
        p_velo = np.hstack([lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))]).T  # (4, N)
        x_cam = calib["Tr_velo_to_cam"].reshape(3, 4) @ p_velo  # (3, N)
        
        # Step 2: Camera -> Rectified Camera
        x_rect = calib["R0_rect"].reshape(3, 3) @ x_cam  # (3, N)
        
        stats['after_calibration'].append(x_rect.shape[1])
        
        # ========================================
        # PROJECT TO IMAGE AND FILTER
        # ========================================
        # Step 3: Rectified Camera -> Image coordinates
        X_rect = np.vstack([x_rect, np.ones((1, x_rect.shape[1]))])  # (4, N)
        x_img = calib["P2"].reshape(3, 4) @ X_rect  # (3, N)
        x_img /= x_img[2, :]  # Normalize by depth
        x_img = x_img[:2, :].T  # (N, 2) -> [u, v] pixel coordinates
        
        # Step 4: Filter points that project within image bounds
        mask = (x_img[:, 0] >= 0) & (x_img[:, 0] < W) & \
               (x_img[:, 1] >= 0) & (x_img[:, 1] < H) & \
               (x_rect[2, :] > 0)  # Also filter points behind camera (Z > 0)
        
        # Apply mask to get valid points
        valid_x_rect = x_rect[:, mask]  # (3, M)
        valid_reflectance = lidar_data[mask, 3]  # (M,)
        
        # Combine back to (M, 4) format
        calibrated_points = np.vstack([valid_x_rect, valid_reflectance]).T  # (M, 4)
        
        stats['after_image_filter'].append(calibrated_points.shape[0])
        
        # ========================================
        # APPLY FPS (Farthest Point Sampling)
        # ========================================
        fps_points = lidar_fps(calibrated_points, N=N)
        
        stats['after_fps'].append(fps_points.shape[0])
        
        # ========================================
        # SAVE TO DISK
        # ========================================
        # Get original filename (e.g., '000000.bin' -> '000000')
        basename = os.path.splitext(os.path.basename(calib_file))[0]
        output_file = os.path.join(output_dir, f"{basename}.npy")
        
        np.save(output_file, fps_points)
        
        # Optional: Print first file info
        if idx % 10 == 0:
            print(f"\n   Example (first file):")
            print(f"   - Image size: {image.shape} (H={H}, W={W})")
            print(f"   - Original LiDAR points: {lidar_data.shape}")
            print(f"   - After calibration: {x_rect.shape[1]} points")
            print(f"   - After image filtering: {calibrated_points.shape}")
            print(f"   - After FPS: {fps_points.shape}")
            print(f"   - Saved to: {output_file}")
    
    print(f"\n✓ Successfully processed and saved {len(lidar_arrays)} files to:")
    print(f"  {output_dir}")
    
    # Calculate and display statistics
    print(f"\n5. Processing Statistics:")
    print(f"   Average points per file:")
    print(f"   - Original:        {np.mean(stats['original_points']):.0f} ± {np.std(stats['original_points']):.0f}")
    print(f"   - After calib:     {np.mean(stats['after_calibration']):.0f} ± {np.std(stats['after_calibration']):.0f}")
    print(f"   - After filtering: {np.mean(stats['after_image_filter']):.0f} ± {np.std(stats['after_image_filter']):.0f}")
    print(f"   - After FPS:       {np.mean(stats['after_fps']):.0f} ± {np.std(stats['after_fps']):.0f}")
    print(f"   Retention rate: {np.mean(stats['after_image_filter'])/np.mean(stats['original_points'])*100:.1f}%")
    
    # Save metadata
    metadata = {
        'split': split,
        'num_files': len(lidar_arrays),
        'fps_samples': N,
        'calibrated': True,
        'image_filtered': True,
        'shape': (N, 4),
        'columns': ['x_rect', 'y_rect', 'z_rect', 'reflectance'],
        'avg_original_points': np.mean(stats['original_points']),
        'avg_filtered_points': np.mean(stats['after_image_filter']),
        'retention_rate': np.mean(stats['after_image_filter'])/np.mean(stats['original_points'])
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  Metadata saved to: {metadata_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'processing_stats.npy')
    np.save(stats_file, stats)
    print(f"  Statistics saved to: {stats_file}")
    
    return output_dir


def load_processed_lidar(split='train', idx=0, output_dir=None):
    """
    Load a processed LiDAR file
    
    Args:
        split (str): 'train' or 'test'
        idx (int): File index or filename
        output_dir (str): Directory where processed files are saved
    
    Returns:
        np.ndarray: Processed LiDAR points (N, 4)
    """
    if output_dir is None:
        if split == 'train':
            output_dir = os.path.join(ROOTDIR, 'processed_lidar', 'training')
        else:
            output_dir = os.path.join(ROOTDIR, 'processed_lidar', 'testing')
    
    if isinstance(idx, int):
        files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        filepath = os.path.join(output_dir, files[idx])
    else:
        filepath = os.path.join(output_dir, idx)
    
    return np.load(filepath)


def visualize_processing_stats(split='train', output_dir=None):
    """
    Visualize the processing statistics
    """
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        if split == 'train':
            output_dir = os.path.join(ROOTDIR, 'processed_lidar', 'training')
        else:
            output_dir = os.path.join(ROOTDIR, 'processed_lidar', 'testing')
    
    stats_file = os.path.join(output_dir, 'processing_stats.npy')
    stats = np.load(stats_file, allow_pickle=True).item()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Point count through pipeline
    axes[0, 0].hist(stats['original_points'], bins=50, alpha=0.7, label='Original')
    axes[0, 0].hist(stats['after_image_filter'], bins=50, alpha=0.7, label='After Filter')
    axes[0, 0].set_xlabel('Number of Points')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Point Count Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Retention rate
    retention = np.array(stats['after_image_filter']) / np.array(stats['original_points'])
    axes[0, 1].hist(retention * 100, bins=50, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Retention Rate (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Image Filtering Retention Rate\nMean: {retention.mean()*100:.1f}%')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Pipeline progression
    file_indices = range(len(stats['original_points']))
    axes[1, 0].plot(file_indices, stats['original_points'], 'b-', label='Original', alpha=0.5)
    axes[1, 0].plot(file_indices, stats['after_image_filter'], 'g-', label='After Filter', alpha=0.7)
    axes[1, 0].axhline(y=16384, color='r', linestyle='--', label='FPS Target')
    axes[1, 0].set_xlabel('File Index')
    axes[1, 0].set_ylabel('Number of Points')
    axes[1, 0].set_title('Point Count Through Processing Pipeline')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    Processing Summary ({split} split)
    ═══════════════════════════════════
    
    Total files: {len(stats['original_points'])}
    
    Average points per file:
    • Original:        {np.mean(stats['original_points']):.0f}
    • After filtering: {np.mean(stats['after_image_filter']):.0f}
    • After FPS:       {np.mean(stats['after_fps']):.0f}
    
    Retention rate:    {retention.mean()*100:.1f}%
    
    Min/Max retention:
    • Min: {retention.min()*100:.1f}%
    • Max: {retention.max()*100:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'processing_stats.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Statistics visualization saved to: {os.path.join(output_dir, 'processing_stats.png')}")


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    
    print("="*70)
    print("KITTI LiDAR Processing Pipeline with Image Filtering")
    print("="*70)
    
    # Process training set
    print("\n" + "="*70)
    print("PROCESSING TRAINING SET")
    print("="*70)
    train_dir = process_and_save_fps_lidar(
        split='train', 
        N=16384,  # Same as paper
        output_dir=os.path.join(ROOTDIR, 'processed_lidar', 'training', "velodyne")
    )
    
    # Process testing set
    print("\n" + "="*70)
    print("PROCESSING TESTING SET")
    print("="*70)
    test_dir = process_and_save_fps_lidar(
        split='test', 
        N=16384,
        output_dir=os.path.join(ROOTDIR, 'processed_lidar', 'testing')
    )
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Load and verify a sample
    sample_train = load_processed_lidar('train', idx=0, output_dir=train_dir)
    print(f"\nSample training file:")
    print(f"  Shape: {sample_train.shape}")
    print(f"  Data range:")
    print(f"    X: [{sample_train[:, 0].min():.2f}, {sample_train[:, 0].max():.2f}]")
    print(f"    Y: [{sample_train[:, 1].min():.2f}, {sample_train[:, 1].max():.2f}]")
    print(f"    Z: [{sample_train[:, 2].min():.2f}, {sample_train[:, 2].max():.2f}]")
    print(f"    Reflectance: [{sample_train[:, 3].min():.2f}, {sample_train[:, 3].max():.2f}]")
    
    # Visualize statistics
    print("\n" + "="*70)
    print("GENERATING STATISTICS VISUALIZATION")
    print("="*70)
    visualize_processing_stats('train', train_dir)
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print("\nOutput directories:")
    print(f"  Training: {train_dir}")
    print(f"  Testing: {test_dir}")
    print("\nYou can now load these files using:")
    print("  points = np.load('path/to/processed_lidar/training/000000.npy')")
    print("="*70)