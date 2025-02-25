import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib  # For loading the trained model

# Define dataset paths
DATASET_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split"
LOGISTIC_MODEL_PATH = "logistic_model_vgg16.pkl"  # Path to the saved model

# Define image transformations (Same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load class labels from the dataset
dataset = ImageFolder(root=f"{DATASET_PATH}/train")
class_names = dataset.classes  # Get list of class names

# Load VGG16 model (Feature Extractor)
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier = torch.nn.Identity()  # Remove last classification layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)
vgg16.eval()

# Load trained Logistic Regression model
logistic_model = joblib.load(LOGISTIC_MODEL_PATH)

# Load Test Set
test_dataset = ImageFolder(root=f"{DATASET_PATH}/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to extract features and labels for confusion matrix
def extract_features_and_labels(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            feats = vgg16(images)  # Extract feature vectors
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.hstack(labels)

# Extract features and ground-truth labels from the test set
X_test, y_test = extract_features_and_labels(test_loader)

# Predict using Logistic Regression
y_pred = logistic_model.predict(X_test)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for VGG16 + Logistic Regression")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()

# Print Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
