import os
import shutil
from math import floor

src_dir = 'plantvillage-dataset/color'
train_dir = os.path.join(src_dir, 'train')
test_dir = os.path.join(src_dir, 'test')

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through subdirectories in source directory
for subdir in os.listdir(src_dir):
    subdir_path = os.path.join(src_dir, subdir)
    if os.path.isdir(subdir_path) and subdir not in ['train', 'test']:
        # Create corresponding subdirectories in train and test directories
        train_subdir = os.path.join(train_dir, subdir)
        test_subdir = os.path.join(test_dir, subdir)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)

        # Get list of files in subdirectory
        files = os.listdir(subdir_path)
        num_files = len(files)
        num_train = floor(num_files * 0.8)

        # Move files to train and test directories
        for i, file in enumerate(files):
            src_file = os.path.join(subdir_path, file)
            if i < num_train:
                dst_file = os.path.join(train_subdir, file)
            else:
                dst_file = os.path.join(test_subdir, file)
            shutil.move(src_file, dst_file)

print('Done splitting data into train and test sets.')
