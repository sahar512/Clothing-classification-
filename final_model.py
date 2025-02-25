import os
import timm
import torch
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Define Dataset Path
DATASET_PATH = "C:/Users/sahar/Downloads/Clothes_Dataset_Split"

# Define Classes (Clothing Categories)
classes = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie',
           'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos',
           'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter']

# Define Transformations (Including Data Augmentation)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.RandomRotation(10),  # Rotate images randomly
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Datasets
train_dataset = ImageFolder(root=f"{DATASET_PATH}/train", transform=transform_train)
val_dataset = ImageFolder(root=f"{DATASET_PATH}/val", transform=transform_test)
test_dataset = ImageFolder(root=f"{DATASET_PATH}/test", transform=transform_test)

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model (EfficientNet-B0)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(classes))

# Set up Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss Function & Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=1e-4)

# Learning Rate Scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Define Evaluation Metrics
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)
f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)

# Training Setup
epochs = 15
best_acc = 0
best_loss = float("inf")
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Check for saved model
best_model_path = f"{save_dir}/best_accuracy_model.pth"
if os.path.exists(best_model_path):
    print(f" Found saved model: {best_model_path}. Loading instead of training...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
else:
    print(" No saved model found. Training from scratch...")

    print(" Starting Training...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_acc, train_f1 = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute Metrics
            train_loss += loss.item()
            train_acc += accuracy_metric(outputs, labels)
            train_f1 += f1_metric(outputs, labels)

        # Normalize Metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)

        print(f"📊 Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")

        # Validation Phase
        model.eval()
        val_loss, val_acc, val_f1 = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                val_acc += accuracy_metric(outputs, labels)
                val_f1 += f1_metric(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)

        print(f"📊 Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save Best Models
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" New Best Accuracy Model Saved: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_loss_model.pth")
            print(f" New Best Loss Model Saved: {val_loss:.4f}")

        # Step Learning Rate Scheduler
        scheduler.step()

    print(" Training Completed!")

# **Test Evaluation**
print(" Loading Best Model for Testing...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Run Model on Test Set
test_loss, test_acc, test_f1 = 0, 0, 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🔍 Testing Model"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        test_loss += loss.item()
        test_acc += accuracy_metric(outputs, labels)
        test_f1 += f1_metric(outputs, labels)

# Compute Final Test Metrics
test_loss /= len(test_loader)
test_acc /= len(test_loader)
test_f1 /= len(test_loader)

print(f" Final Test Accuracy: {test_acc:.4f}")
print(f" Final Test F1 Score: {test_f1:.4f}")
print(f" Final Test Loss: {test_loss:.4f}")
