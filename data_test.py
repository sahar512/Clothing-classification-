import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define paths
DATASET_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for CNN models
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for better convergence
])

# Load datasets
train_dataset = ImageFolder(root=f"{DATASET_PATH}/train", transform=transform)
val_dataset = ImageFolder(root=f"{DATASET_PATH}/val", transform=transform)
test_dataset = ImageFolder(root=f"{DATASET_PATH}/test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verify dataset sizes
print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")
print(f"Test set: {len(test_dataset)} images")
print(f"Number of classes: {len(train_dataset.classes)}")

# Show some sample images
def show_sample_images(dataset, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()  # Convert tensor to NumPy
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {train_dataset.classes[label]}")
        axes[i].axis("off")
    plt.show()

# Display sample images from the training dataset
show_sample_images(train_dataset)
