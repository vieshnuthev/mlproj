import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dessert_dataset"

# 0. DEVICE CONFIGURATION
# This automatically checks if an NVIDIA GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. DATA INGESTION (BULLETPROOF PATHING)
train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "validation"

print(f"Loading images locally from: {DATA_DIR}")

# adding randomness to force the model to see the desserts from different angles so it improves generalisation
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),  # randomness factors
        transforms.RandomRotation(degrees=15),  # randomness factors
        transforms.ColorJitter(brightness=0.2),  # randomness factors
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

try:
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=transform)
    print(
        f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images."
    )
    print(f"Classes found: {train_dataset.classes}")
except FileNotFoundError:
    print(f"Error: Could not find the '{DATA_DIR}' folder. Check your folder structure.")
    exit()

# We can safely increase batch size slightly if using a GPU, but 32 is a safe default
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. MODEL ARCHITECTURE (CNN)
# adding Dropout randomly deactivates neurons during training, forcing the network to find more robust pathways
class OptimisedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimisedCNN, self).__init__()

        # define the layers as instance attributes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Optimization for stability

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Optimization for stability

        # MUST define these here to use self.pool and self.relu in forward()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # regularization to fix the overfitting seen in M3
        self.dropout = nn.Dropout(p=0.5)

        # 128x128 input -> two 2x2 pools -> 32x32 spatial dimension
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        # Layer 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # Layer 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply Dropout before the final layer
        x = self.dropout(x)

        # Final fully connected layer
        x = self.fc1(x)
        return x


num_classes = len(train_dataset.classes)
# Push the model's weights to the selected device (GPU or CPU)
model = OptimisedCNN(num_classes).to(device)

# 3. THE TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
history = {"train_loss": [], "val_loss": []}

print("\nStarting CNN Training Loop...")
for epoch in range(num_epochs):
    start_time = time.time()

    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        # Push the images and labels to the selected device
        inputs, labels = inputs.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_train_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    # --- Validation Phase ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Push the validation images and labels to the selected device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    # --- Metric Logging ---
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Time: {time.time() - start_time:.2f}s"
    )

# 4. VISUAL EVIDENCE GENERATION

print("\nGenerating training progress charts...")
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Training Loss", color="blue")
plt.plot(history["val_loss"], label="Validation Loss", color="red")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
