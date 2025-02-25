import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from torchvision import models
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
import joblib  # For loading the trained model

# Define paths
DATASET_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split"
TEST_IMAGE_PATH = r"C:\Users\sahar\Downloads\ML_test_images\shop_at_velvet_blazzer_1686723737_ab6804e6_progressive_thumbnail.jpg"  # Change this to a real test image
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

# Function to classify an image and print top 3 predictions
def classify_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_pil = transforms.ToPILImage()(img)  # Convert OpenCV image to PIL format

    # Preprocess image
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Add batch dimension

    # Extract VGG16 features
    with torch.no_grad():
        features = vgg16(img_tensor)

    # Convert features to NumPy array and classify using Logistic Regression
    features = features.cpu().numpy()
    probabilities = logistic_model.predict_proba(features)  # Get probabilities for each class
    prediction = logistic_model.predict(features)[0]  # Get predicted class index

    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities[0])[::-1][:3]  # Get top 3 indices (sorted highest)
    top_3_classes = [class_names[i] for i in top_3_indices]  # Convert indices to class names
    top_3_confidences = [probabilities[0][i] * 100 for i in top_3_indices]  # Convert to percentage

    # Print results
    print(f"\n The model predicts: **{class_names[prediction]}**")
    print("\n **Top 3 Predictions:**")
    for i in range(3):
        print(f"  {i+1}. {top_3_classes[i]} - {top_3_confidences[i]:.2f}% confidence")

# Classify the test image
classify_image(TEST_IMAGE_PATH)
