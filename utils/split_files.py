import os
import shutil
import numpy as np

ROOTDIR = r'F:\Work\DeepLearning\Research\V2X-Radar-V'

def create_train_test_split_random(root_dir=ROOTDIR, train_ratio=0.8, random_seed=42):
    np.random.seed(random_seed)

    training_dir = os.path.join(root_dir, 'training')
    testing_dir = os.path.join(root_dir, 'testing')

    subdirs = ['calib', 'image_2', 'label_2', 'radar', 'velodyne']

    # Create testing directory structure
    for subdir in subdirs:
        os.makedirs(os.path.join(testing_dir, subdir), exist_ok=True)

    # Use velodyne as reference
    sample_dir = os.path.join(training_dir, 'velodyne')
    all_files = [f for f in os.listdir(sample_dir) if f.endswith('.bin')]

    total_files = len(all_files)
    indices = np.arange(total_files)
    np.random.shuffle(indices)

    train_size = int(total_files * train_ratio)
    test_indices = indices[train_size:]
    test_basenames = [os.path.splitext(all_files[i])[0] for i in test_indices]

    for subdir in subdirs:
        train_subdir = os.path.join(training_dir, subdir)
        test_subdir = os.path.join(testing_dir, subdir)

        if subdir in ['radar', 'velodyne']:
            ext = '.bin'
        elif subdir == 'image_2':
            ext = '.png'
        else:
            ext = '.txt'

        for basename in test_basenames:
            src = os.path.join(train_subdir, basename + ext)

            if subdir == 'image_2' and not os.path.exists(src):
                src = os.path.join(train_subdir, basename + '.jpg')

            if os.path.exists(src):
                shutil.move(src, os.path.join(test_subdir, os.path.basename(src)))

    print("✓ Random train-test split completed")


if __name__ == "__main__":
    create_train_test_split_random(train_ratio=0.8, random_seed=42)
