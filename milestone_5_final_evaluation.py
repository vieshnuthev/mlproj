# Milestone 5: Final Evaluation — Multi-Seed Runs
# Trains OptimisedCNN (from M4) across multiple random seeds
# Reports mean ± std accuracy for statistical robustness

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ─────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────
SEEDS      = [42, 123, 456]   # 3 runs — ~45 min on CPU. Add more if time allows.
NUM_EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE   = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────
current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir  = os.path.join(current_script_dir, 'dessert_dataset')
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'validation')

# ─────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets once (transforms applied per-batch, not affected by seed)
try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(root=val_dir,   transform=test_transform)
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Classes: {len(train_dataset.classes)}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

class_names  = train_dataset.classes
num_classes  = len(class_names)

# ─────────────────────────────────────────
# 3. MODEL DEFINITION (M4 OptimisedCNN)
# ─────────────────────────────────────────
class OptimisedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimisedCNN, self).__init__()
        self.conv1   = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm2d(16)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm2d(32)
        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

# ─────────────────────────────────────────
# 4. SEED HELPER
# ─────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────
# 5. TRAIN + EVAL FUNCTION (one seed run)
# ─────────────────────────────────────────
def run_experiment(seed):
    set_seed(seed)
    print(f"\n{'='*50}")
    print(f"SEED {seed}")
    print(f"{'='*50}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model     = OptimisedCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # Train
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            run_loss  += loss.item() * inputs.size(0)
            _, pred    = torch.max(outputs, 1)
            correct   += (pred == labels).sum().item()
            total     += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct / total

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss    += loss.item() * inputs.size(0)
                _, pred    = torch.max(outputs, 1)
                v_correct += (pred == labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train {train_acc:.2%} | Val {val_acc:.2%} | "
              f"Time {time.time()-start:.1f}s")

    # Final test pass — collect all predictions
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, pred = torch.max(model(inputs), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"  → Final Test Accuracy: {test_acc:.2%}")

    return test_acc, history, np.array(all_preds), np.array(all_labels)

# ─────────────────────────────────────────
# 6. RUN ALL SEEDS
# ─────────────────────────────────────────
all_accuracies = []
all_histories  = []
best_acc       = -1
best_preds     = None
best_labels    = None

for seed in SEEDS:
    acc, history, preds, labels = run_experiment(seed)
    all_accuracies.append(acc)
    all_histories.append(history)
    if acc > best_acc:
        best_acc    = acc
        best_preds  = preds
        best_labels = labels

mean_acc = np.mean(all_accuracies)
std_acc  = np.std(all_accuracies)

print(f"\n{'='*50}")
print(f"RESULTS ACROSS {len(SEEDS)} SEEDS:")
for i, (s, a) in enumerate(zip(SEEDS, all_accuracies)):
    print(f"  Seed {s}: {a:.2%}")
print(f"\nMean Accuracy : {mean_acc:.2%}")
print(f"Std Deviation : {std_acc:.2%}")
print(f"{'='*50}")

# ─────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────

# Plot 1: Accuracy per seed bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar([f"Seed {s}" for s in SEEDS], [a * 100 for a in all_accuracies], color='steelblue')
plt.axhline(y=mean_acc * 100, color='red', linestyle='--',
            label=f'Mean: {mean_acc:.2%} ± {std_acc:.2%}')
for bar, acc in zip(bars, all_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
plt.title('Test Accuracy Across Multiple Seeds')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig('m5_multi_seed_accuracy.png', dpi=150)
print("Saved: m5_multi_seed_accuracy.png")
plt.show()

# Plot 2: Loss + Accuracy curves (best run)
best_idx = int(np.argmax(all_accuracies))
history  = all_histories[best_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
axes[0].plot(history['val_loss'],   label='Val Loss',   color='red')
axes[0].set_title(f'Loss — Best Run (Seed {SEEDS[best_idx]})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
axes[1].plot(history['val_acc'],   label='Val Acc',   color='red')
axes[1].set_title(f'Accuracy — Best Run (Seed {SEEDS[best_idx]})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('m5_training_curves.png', dpi=150)
print("Saved: m5_training_curves.png")
plt.show()

# Plot 3: Confusion matrix (best run)
cm = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix — Best Run ({best_acc:.2%} accuracy)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('m5_confusion_matrix.png', dpi=150)
print("Saved: m5_confusion_matrix.png")
plt.show()

# Plot 4: Per-class accuracy (best run)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(14, 5))
bars = plt.bar(class_names[:len(per_class_acc)], per_class_acc * 100, color='steelblue')
plt.axhline(y=best_acc * 100, color='red', linestyle='--',
            label=f'Overall: {best_acc:.2%}')
plt.title('Per-Class Accuracy (Best Run)')
plt.xlabel('Dessert Class')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()
plt.savefig('m5_per_class_accuracy.png', dpi=150)
print("Saved: m5_per_class_accuracy.png")
plt.show()

# Print classification report (best run)
print("\nClassification Report (Best Run):")
print(classification_report(best_labels, best_preds, target_names=class_names))

print("\nMilestone 5 Complete.")
print(f"Final Result: {mean_acc:.2%} ± {std_acc:.2%} over {len(SEEDS)} seeds")
