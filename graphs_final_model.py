import matplotlib.pyplot as plt
import numpy as np

# Define epochs
epochs = np.arange(1, 16)  # Assuming 15 epochs

# Placeholder data (Replace with actual values)
train_loss = [1.4791, 0.5145, 0.2651, 0.1996, 0.1253, 0.0844, 0.0571, 0.0506, 0.0462, 0.0419, 0.0356, 0.0266, 0.0256, 0.0238, 0.0224]
val_loss = [0.7929, 0.8104, 0.7525, 0.7068, 0.7974, 0.7247, 0.7121, 0.7554, 0.7316, 0.7698, 0.7365, 0.7600, 0.7258, 0.7629, 0.7766]

train_acc = [0.5680, 0.8286, 0.9121, 0.9426, 0.9612, 0.9727, 0.9852, 0.9854, 0.9837, 0.9862, 0.9892, 0.9913, 0.9915, 0.9922, 0.9920]
val_acc = [0.7490, 0.7516, 0.7902, 0.7989, 0.7869, 0.8097, 0.8097, 0.8229, 0.8170, 0.8190, 0.8315, 0.8203, 0.8243, 0.8223, 0.8269]

train_f1 = train_acc  # Assuming F1-score is the same as accuracy
val_f1 = val_acc

# Plot Loss Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", color="red")
plt.plot(epochs, val_loss, label="Validation Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss Values")
plt.legend()
plt.title("Train vs Validation Loss Over Epochs")
plt.show()

# Plot Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="Train Accuracy", color="orange")
plt.plot(epochs, val_acc, label="Validation Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy Scores")
plt.legend()
plt.title("Train vs Validation Accuracy Over Epochs")
plt.show()

# Plot F1-Score Graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_f1, label="Train F1 Score", color="purple")
plt.plot(epochs, val_f1, label="Validation F1 Score", color="pink")
plt.xlabel("Epochs")
plt.ylabel("F1 Scores")
plt.legend()
plt.title("Train vs Validation F1 Score Over Epochs")
plt.show()
