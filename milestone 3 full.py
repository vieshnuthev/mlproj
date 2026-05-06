import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 0. DEVICE CONFIGURATION
# This automatically checks if an NVIDIA GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. DATA INGESTION (BULLETPROOF PATHING)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_script_dir, 'MLproj', 'full_dataset')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

print(f"Loading images locally from: {data_dir}")

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Classes found: {train_dataset.classes}")
except FileNotFoundError:
    print(f"Error: Could not find the '{data_dir}' folder. Check your folder structure.")
    exit()

# We can safely increase batch size slightly if using a GPU, but 32 is a safe default
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. MODEL ARCHITECTURE (CNN)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

num_classes = len(train_dataset.classes)
# Push the model's weights to the selected device (GPU or CPU)
model = SimpleCNN(num_classes).to(device)

# 3. THE TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 
history = {'train_loss': [], 'val_loss': []}

print("\nStarting CNN Training Loop...")
for epoch in range(num_epochs):
    start_time = time.time()

    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        # Push the images and labels to the selected device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Time: {time.time() - start_time:.2f}s")
        

# 4. VISUAL EVIDENCE GENERATION

print("\nGenerating training progress charts...")
plt.figure(figsize=(8, 5))
plt.plot(history['train_loss'], label='Training Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='red')
plt.title('CNN Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()