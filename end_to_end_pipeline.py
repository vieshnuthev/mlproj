"""
End-to-end dessert classification pipeline.

This script combines the milestone-style workflow into one local, reproducible
entrypoint:

1. Load the dataset from `dessert_dataset`
2. Preview a batch of images for the data-pipeline milestone
3. Build and summarize the CNN architecture
4. Train the model and save learning curves
5. Run final multi-seed evaluation and export test artifacts

The script keeps the intermediate outputs your lecturer expects, but removes
the need to run separate Colab cells or notebook state.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "dessert_dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
OUTPUT_DIR = ROOT_DIR / "outputs"

IMG_SIZE = 128
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_SAMPLE_PER_CLASS = 0


class OptimisedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full dessert pipeline.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs per seed (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for the final multi-seed evaluation.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=DEFAULT_SAMPLE_PER_CLASS,
        help=(
            "Limit the training set to this many images per class. "
            "Use 0 to keep all training images."
        ),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def ensure_paths() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


def get_class_names(dataset) -> list[str]:
    if isinstance(dataset, Subset):
        return dataset.dataset.classes
    return dataset.classes


def sample_subset(dataset, sample_per_class: int, seed: int) -> torch.utils.data.Dataset:
    if sample_per_class <= 0:
        return dataset

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        by_class[label].append(index)

    rng = random.Random(seed)
    chosen: list[int] = []
    for label, indices in by_class.items():
        if len(indices) <= sample_per_class:
            chosen.extend(indices)
        else:
            chosen.extend(rng.sample(indices, sample_per_class))

    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def make_datasets(sample_per_class: int, seed: int):
    train_transform, eval_transform = build_transforms()
    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)
    train_dataset = sample_subset(train_dataset, sample_per_class, seed)
    return train_dataset, val_dataset


def make_loaders(train_dataset, val_dataset) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def save_sample_grid(train_loader: DataLoader, class_names: list[str], show_plots: bool) -> None:
    for images, labels in train_loader:
        batch_images = images[:9].clone().cpu()
        batch_labels = labels[:9].clone().cpu()
        break
    else:
        raise RuntimeError("Training loader did not yield any batches.")

    def denormalize(image: torch.Tensor) -> np.ndarray:
        image = image * 0.5 + 0.5
        return np.clip(image.permute(1, 2, 0).numpy(), 0.0, 1.0)

    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(zip(batch_images, batch_labels), start=1):
        ax = plt.subplot(3, 3, i)
        ax.imshow(denormalize(image))
        ax.set_title(class_names[int(label)])
        ax.axis("off")

    plt.suptitle("Milestone 1: Sampled Data Preview", fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "milestone_1_sample_preview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")


def save_model_summary(model: nn.Module, show_plots: bool) -> None:
    summary_path = OUTPUT_DIR / "milestone_2_model_summary.txt"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("Milestone 2: CNN Architecture")
        print(model)
        print()
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    summary_path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"Saved: {summary_path}")

    if show_plots:
        print(model)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_one_seed(
    seed: int,
    train_dataset,
    val_dataset,
    num_epochs: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset)
    model = OptimisedCNN(num_classes=len(get_class_names(train_dataset))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'=' * 60}")
    print(f"Seed {seed}")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss, epoch_val_acc, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss {epoch_train_loss:.4f} | Train Acc {epoch_train_acc:.2%} | "
            f"Val Loss {epoch_val_loss:.4f} | Val Acc {epoch_val_acc:.2%} | "
            f"{elapsed:.1f}s"
        )

    final_val_loss, final_val_acc, preds, labels = evaluate_model(
        model, val_loader, criterion, device
    )

    return {
        "seed": seed,
        "model": model,
        "history": history,
        "val_loss": final_val_loss,
        "val_acc": final_val_acc,
        "preds": preds,
        "labels": labels,
    }


def plot_training_curves(history: dict, seed: int, show_plots: bool) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title(f"Milestone 3: Loss Curves (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.title(f"Milestone 3: Accuracy Curves (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "milestone_3_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")


def plot_multi_seed_accuracy(results: list[dict], show_plots: bool) -> tuple[float, float]:
    seeds = [r["seed"] for r in results]
    accuracies = [r["val_acc"] for r in results]
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    plt.figure(figsize=(8, 5))
    bars = plt.bar([f"Seed {seed}" for seed in seeds], [acc * 100 for acc in accuracies])
    plt.axhline(
        y=mean_acc * 100,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_acc:.2%} ± {std_acc:.2%}",
    )
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.title("Milestone 5: Validation Accuracy Across Seeds")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "milestone_5_multi_seed_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")
    return mean_acc, std_acc


def plot_best_run_artifacts(
    best_result: dict,
    class_names: list[str],
    show_plots: bool,
) -> None:
    history = best_result["history"]
    seed = best_result["seed"]
    preds = best_result["preds"]
    labels = best_result["labels"]

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title(f"Milestone 4/5: Loss Curves (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.title(f"Milestone 4/5: Accuracy Curves (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "milestone_5_best_run_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(labels, preds, labels=all_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Milestone 5: Confusion Matrix (Seed {seed})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "milestone_5_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    per_class_acc = np.divide(
        cm.diagonal(),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    plt.figure(figsize=(14, 5))
    plt.bar(class_names[: len(per_class_acc)], per_class_acc * 100, color="steelblue")
    plt.axhline(y=best_result["val_acc"] * 100, color="red", linestyle="--", label=f"Overall: {best_result['val_acc']:.2%}")
    plt.title("Milestone 5: Per-Class Accuracy")
    plt.xlabel("Dessert Class")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / "milestone_5_per_class_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    report = classification_report(
        labels,
        preds,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
    )
    report_path = OUTPUT_DIR / "milestone_5_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")
    print("\nClassification Report (best run):")
    print(report)


def main() -> None:
    args = parse_args()
    ensure_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {DATA_DIR}")

    train_dataset, val_dataset = make_datasets(
        sample_per_class=args.sample_per_class,
        seed=args.seeds[0],
    )
    class_names = get_class_names(train_dataset)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    train_loader, _ = make_loaders(train_dataset, val_dataset)
    save_sample_grid(train_loader, class_names, show_plots=args.show_plots)

    preview_model = OptimisedCNN(num_classes=len(class_names)).to(device)
    save_model_summary(preview_model, show_plots=args.show_plots)

    results: list[dict] = []
    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.epochs,
            device=device,
        )
        results.append(result)

    best_result = max(results, key=lambda item: item["val_acc"])
    best_seed = best_result["seed"]
    print("\nFinal multi-seed summary:")
    for result in results:
        print(f"  Seed {result['seed']}: {result['val_acc']:.2%}")

    mean_acc, std_acc = plot_multi_seed_accuracy(results, show_plots=args.show_plots)
    plot_training_curves(best_result["history"], best_seed, show_plots=args.show_plots)
    plot_best_run_artifacts(best_result, class_names, show_plots=args.show_plots)

    print("\nPipeline complete.")
    print(f"Best seed: {best_seed}")
    print(f"Mean accuracy across seeds: {mean_acc:.2%}")
    print(f"Std deviation across seeds: {std_acc:.2%}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
