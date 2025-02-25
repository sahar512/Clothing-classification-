import os
import torch
import timm
import torchmetrics
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define paths
BEST_MODEL_PATH = "saved_models/best_accuracy_model.pth"
DATASET_PATH = "C:/Users/sahar/Downloads/Clothes_Dataset_Split"

# Define classes
classes = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie',
           'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos',
           'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter']

# Define image transformations (same as validation)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the test dataset
test_dataset = ImageFolder(root=f"{DATASET_PATH}/test", transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define evaluation metrics
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)
f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()

# Evaluate model on the test set
test_loss, test_acc, test_f1 = 0, 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        test_loss += loss.item()
        test_acc += accuracy_metric(outputs, labels)
        test_f1 += f1_metric(outputs, labels)

        # Store predictions and labels for confusion matrix
        _, predicted_classes = torch.max(outputs, 1)
        all_preds.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute final test metrics
test_loss /= len(test_loader)
test_acc /= len(test_loader)
test_f1 /= len(test_loader)

print(f" Final Test Accuracy: {test_acc:.4f}")
print(f" Final Test F1 Score: {test_f1:.4f}")
print(f" Final Test Loss: {test_loss:.4f}")

# Generate Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Test Dataset")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=45)
plt.show()

# Print Classification Report
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
