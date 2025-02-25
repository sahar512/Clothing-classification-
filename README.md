Fashion Image Classification Using Deep Learning
Project Overview
This project focuses on classifying clothing images into 15 categories using machine learning and deep learning models. The goal is to compare the performance of a baseline model (VGG-16 + Logistic Regression) with an advanced deep learning model (EfficientNet-B0) to identify the best approach for clothing classification.

Dataset
The dataset consists of 15 clothing categories, with images collected from an online marketplace. Each category contains diverse styles and colors to enhance model generalization. The dataset is divided into:

Training Set (Used for model learning)
Validation Set (Used to tune hyperparameters)
Test Set (Used to evaluate final performance)
Classes: Blazer, Long Pants, Shorts, Dresses, Hoodie, Jacket, Denim Jacket, Sports Jacket, Jeans, T-shirt, Shirt, Coat, Polo, Skirt, Sweater.

Models Used
🔹 Baseline Model: VGG-16 + Logistic Regression
Feature Extraction: Used VGG-16 pre-trained on ImageNet to extract image features.
Classification: A Logistic Regression model was trained on these extracted features.
📌 Performance:

Test Accuracy: 77.87%
Confusion Matrix: Displayed misclassifications between similar categories.
Limitations: Unable to capture complex patterns in images due to a shallow classifier.
🔹 Final Model: EfficientNet-B0 (Deep Learning Approach)
Architecture: Used EfficientNet-B0, a CNN with MBConv layers, and Compound Scaling for better efficiency.
Training: Trained from scratch on the dataset with Cross-Entropy Loss and AdamW optimizer.
📌 Performance:

Test Accuracy: 80.3%
Confusion Matrix: Showed fewer misclassifications compared to the baseline model.
Grad-CAM Analysis: Visualized which image regions influenced model predictions.
Results & Insights
Baseline Model performed well but struggled with similar-looking categories (e.g., coats vs. jackets).
EfficientNet-B0 achieved better accuracy due to better feature extraction and deeper learning capacity.
Grad-CAM analysis showed that the deep learning model focused on meaningful clothing features, improving classification.
