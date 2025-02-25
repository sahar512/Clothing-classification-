import os
import shutil
import random

# Define paths
DATASET_PATH = r"C:\Users\sahar\Downloads\archive (7)\Clothes_Dataset"  # Change to your dataset path
OUTPUT_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split"

# Define split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Ensure output folders exist
for split in ['train', 'val', 'test']:
    split_path = os.path.join(OUTPUT_PATH, split)
    os.makedirs(split_path, exist_ok=True)

# Process each class separately
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip if not a directory

    # Get all images in the class
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle for randomness

    # Compute split sizes
    total_images = len(images)
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)

    # Split images
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Create class subdirectories inside train, val, and test folders
    for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        split_class_dir = os.path.join(OUTPUT_PATH, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)  # Ensure class folder exists

        # Move images
        for img_name in split_images:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(split_class_dir, img_name)
            shutil.move(src_path, dst_path)

    print(f"✅ Processed {class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

print(" Dataset splitting completed successfully!")
