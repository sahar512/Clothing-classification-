import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.autograd import Function

# Define image path
TEST_IMAGE_PATH =  r"C:\Users\sahar\Downloads\Clothes_Dataset_Split\test\Jeans\celana_panjang_jeans_pria_dohc_1694006832_e50c89be_progressive_thumbnail.jpg"  # Change this

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load VGG16 model
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.eval()

# Select the last convolutional layer of VGG-16
target_layer = vgg16.features[-1]

# Hook function to store gradients
gradients = None


def hook_function(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Register hook to capture gradients in the target layer
target_layer.register_backward_hook(hook_function)


# Function to apply Grad-CAM
def apply_gradcam(image_path, model):
    global gradients

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = transforms.ToPILImage()(img)
    img_tensor = transform(img_pil).unsqueeze(0)

    # Forward pass to get predictions
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backpropagate to get gradients
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, pred_class] = 1
    output.backward(gradient=one_hot_output)

    # Compute Grad-CAM heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = target_layer(img_tensor).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Resize heatmap to match input image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Overlay heatmap on original image
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)

    # Plot images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("Original Image")

    ax[1].imshow(overlayed_image)
    ax[1].axis("off")
    ax[1].set_title("Grad-CAM Heatmap")

    plt.show()


# Run Grad-CAM on test image
apply_gradcam(TEST_IMAGE_PATH, vgg16)
