import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),
            # Ensure the final output size is (224, 224)
            nn.Upsample(size=(224, 224)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the model architecture again
autoencoder = Autoencoder()
# Move the model to the same device as the input tensor
autoencoder = autoencoder.to(device)
# Ensure the model loads to the correct device
autoencoder.load_state_dict(torch.load(
    'autoencoder_best.pth', map_location=device))
autoencoder.eval()  # Set the model to evaluation mode

# Define Anomaly Detection Function


def detect_anomalies(image_path, autoencoder, threshold=0.01):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = autoencoder(image_tensor)
    reconstruction_error = torch.mean(
        (image_tensor - reconstructed) ** 2).item()
    return reconstruction_error > threshold, reconstruction_error, image_tensor.squeeze().cpu().numpy(), reconstructed.squeeze().cpu().numpy()

# Visualize the Results


def plot_anomalies(original, reconstructed, threshold, error):
    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow((original.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))

    # Reconstructed image
    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.imshow((reconstructed.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))

    # Difference
    difference = np.abs(original - reconstructed)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference.transpose(1, 2, 0).sum(axis=2), cmap='hot')

    plt.suptitle(f'Error: {error:.4f}, Threshold: {threshold}')
    plt.show()


# Test anomaly detection
test_image_path = r"C:\Users\me1elar\data_anomaly_detection\leather\test\cut\003.png"
is_anomalous, reconstruction_error, original_image, reconstructed_image = detect_anomalies(
    test_image_path, autoencoder)
print(
    f'Anomaly detected: {is_anomalous}, Reconstruction error: {reconstruction_error}')
plot_anomalies(original_image, reconstructed_image,
               threshold=0.01, error=reconstruction_error)
