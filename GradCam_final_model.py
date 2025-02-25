import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
from PIL import Image

# Define class names (Clothing Categories)
class_names = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie',
               'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos',
               'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter']

# Load Pretrained Model (EfficientNet-B0)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("saved_models/best_accuracy_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define Image Transformations (Same as Training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Image for Testing (Update Path)
image_path = r"C:\Users\sahar\Downloads\Clothes_Dataset_Split\test\Jeans\celana_panjang_jeans_pria_dohc_1694006832_e50c89be_progressive_thumbnail.jpg"
original_image = cv2.imread(image_path)
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Convert NumPy array to PIL image
if isinstance(image, np.ndarray):
    image = Image.fromarray(image)

# Apply transformations
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Select Last Convolutional Layer in EfficientNet for Grad-CAM
target_layer = model.blocks[-1]  # The last convolutional block

# Hook to Capture Gradients
gradients = None
activations = None

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]  # Extract gradients

def save_activation(module, input, output):
    global activations
    activations = output  # Save feature maps

# Register Hooks
grad_handle = target_layer.register_full_backward_hook(save_gradient)
act_handle = target_layer.register_forward_hook(save_activation)

# Forward Pass
output = model(image_tensor)
prediction = torch.argmax(output, dim=1).item()
predicted_class_name = class_names[prediction]  # Get class name

# Backward Pass (Set Gradient for Predicted Class)
model.zero_grad()
output[0, prediction].backward()

# Compute Grad-CAM
grads = gradients.cpu().numpy()
weights = np.mean(grads, axis=(2, 3))  # Global Average Pooling
activation = activations.detach().squeeze().cpu().numpy()
cam = np.zeros(activation.shape[1:], dtype=np.float32)

for i, w in enumerate(weights[0]):  # Iterate over feature maps
    cam += w * activation[i]

# Normalize CAM
cam = np.maximum(cam, 0)
cam = cam / cam.max()

# Resize to Original Image Size
heatmap = cv2.resize(cam, (image.size[0], image.size[1]))

# Overlay Heatmap on Original Image
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

# Display Grad-CAM Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title(f"Grad-CAM Visualization - Predicted: {predicted_class_name}")
plt.axis("off")

plt.show()

# Remove Hooks
grad_handle.remove()
act_handle.remove()
