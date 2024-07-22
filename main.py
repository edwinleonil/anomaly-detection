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

# Define the Custom Dataset Class


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = ImageFolder(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img


# Prepare the dataset and data loader
dataset = CustomDataset(
    r"C:\Users\me1elar\data_anomaly_detection\leather\train",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalizing the dataset
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

# Define the Training Function


def train_autoencoder(autoencoder, train_loader, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        for data in train_loader:
            img = data.to(device)
            recon = autoencoder(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

        # Check if the model has improved
        if loss.item() < best_loss:
            best_loss = loss.item()
            # Save the model
            save_model(autoencoder, "autoencoder_best.pth")
            print(
                f"Model improved and saved to autoencoder_best.pth with loss {best_loss}")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(
                f"No improvement in epoch: {epoch+1}, epochs without improvement: {epochs_no_improve}")

        # Early stopping
        if epochs_no_improve == 5:
            print("Early stopping triggered")
            break

# Save the model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Train the autoencoder
autoencoder = Autoencoder().to(device)
train_autoencoder(autoencoder, train_loader, num_epochs=100)

# Save the trained model
model_save_path = "autoencoder.pth"
save_model(autoencoder, model_save_path)
