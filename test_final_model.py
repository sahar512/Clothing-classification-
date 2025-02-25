import torch
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from torchvision.datasets import ImageFolder

# Define path to the saved model and dataset
BEST_MODEL_PATH = "saved_models/best_accuracy_model.pth"
TEST_IMAGE_PATH = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split\test\Celana_Panjang\celana_panjang_wanita_for_all__1696210135_eeedd9a9_progressive_thumbnail.jpg"  # Change to your test image
DATASET_PATH = "C:/Users/sahar/Downloads/Clothes_Dataset_Split"

# Define class names (Clothing categories)
classes = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie',
           'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos',
           'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter']

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Function to predict image class
def classify_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_pil = transforms.ToPILImage()(img)  # Convert OpenCV image to PIL format

    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Preprocess and add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
        top3_probs, top3_classes = torch.topk(probabilities, 3)  # Get top 3 predictions

    # Convert predictions to class labels
    predicted_labels = [classes[idx] for idx in top3_classes[0].cpu().numpy()]
    predicted_probs = top3_probs[0].cpu().numpy()

    # Print results
    print(f" The model predicts: {predicted_labels[0]} ({predicted_probs[0]*100:.2f}%)")
    print(f" Top 3 Predictions:")
    for i in range(3):
        print(f"  {predicted_labels[i]}: {predicted_probs[i]*100:.2f}%")

    return predicted_labels[0], predicted_probs[0]

# Test the model with an image
predicted_label, confidence = classify_image(TEST_IMAGE_PATH)
