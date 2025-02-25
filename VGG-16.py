import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
import joblib  # Import joblib to save the model

# Define dataset paths
DATASET_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split"
BATCH_SIZE = 32

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load dataset
train_dataset = ImageFolder(root=f"{DATASET_PATH}/train", transform=transform)
test_dataset = ImageFolder(root=f"{DATASET_PATH}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pretrained VGG16 (Feature Extractor)
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier = torch.nn.Identity()  # Remove classification layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)
vgg16.eval()

# Function to extract features
def extract_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            feats = vgg16(images)  # Extract feature vectors
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.hstack(labels)

# Extract features for training & testing
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# Split train data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Train Logistic Regression model on extracted features
print("🔹 Training Logistic Regression on VGG16 Features...")
logistic_model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial', verbose=True)
logistic_model.fit(X_train, y_train)

# Function to evaluate model
def evaluate_model(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    logloss = log_loss(y, model.predict_proba(X))

    print(f"\n {dataset_name} Evaluation:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" Log Loss: {logloss:.4f}")
    return accuracy, f1, logloss

# Evaluate on Training Set
evaluate_model(logistic_model, X_train, y_train, "Training Set")

# Evaluate on Validation Set
evaluate_model(logistic_model, X_val, y_val, "Validation Set")

# Evaluate on Test Set
evaluate_model(logistic_model, X_test, y_test, "Test Set")

# Save trained Logistic Regression model
joblib.dump(logistic_model, "logistic_model_vgg16.pkl")
print("\n Logistic Regression Model Saved as 'logistic_model_vgg16.pkl'")
