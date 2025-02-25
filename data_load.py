import os
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from glob import glob
from collections import Counter

# Set dataset path
DATASET_PATH = r"C:\Users\sahar\Downloads\archive (7)\Clothes_Dataset"

# Get class names (folder names)
classes = sorted(os.listdir(DATASET_PATH))
print(f"Total Classes: {len(classes)}")
print("Classes:", classes)

# Count images per class
image_counts = {cls: len(glob(os.path.join(DATASET_PATH, cls, '*.jpg'))) for cls in classes}
print("Class Distribution:", image_counts)

# Plot class distribution
plt.figure(figsize=(12, 5))
plt.bar(image_counts.keys(), image_counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution in Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()
