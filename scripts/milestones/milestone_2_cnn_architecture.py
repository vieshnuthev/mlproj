import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dessert_dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. Dataset Parameters
# -------------------------------------------------
batch_size = 32
img_height = 128
img_width = 128
data_dir = str(DATA_DIR)

# -------------------------------------------------
# 2. Load Dataset (Train + Validation Split)
# -------------------------------------------------
# Milestone 1: Data Pipeline [cite: 20]
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes identified:", class_names)

# -------------------------------------------------
# 3. Build CNN Model (Milestone 2 Core)
# -------------------------------------------------
model = models.Sequential(
    [
        # Input Layer
        layers.Input(shape=(img_height, img_width, 3)),

        # Normalize Data (0-255 -> 0-1) inside the model for better portability
        layers.Rescaling(1.0 / 255),

        # -------- Convolution Block 1 --------
        # Detects basic features like edges
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # -------- Convolution Block 2 --------
        # Detects complex shapes
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # -------- Convolution Block 3 --------
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # -------- Flatten & Dense Layers --------
        layers.Flatten(),
        layers.Dense(128, activation="relu"),

        # Milestone 4 Preview: Dropout to prevent overfitting
        layers.Dropout(0.5),

        # Output Layer (Softmax for multi-class classification)
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# -------------------------------------------------
# 4. Compile Model
# -------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------------------------------
# 5. Output Visual Evidence [cite: 39]
# -------------------------------------------------
print("\nModel Architecture Summary:")
model.summary()

# Save the diagram
plot_path = OUTPUT_DIR / "milestone_2_model_architecture.png"
try:
    tf.keras.utils.plot_model(
        model,
        to_file=str(plot_path),
        show_shapes=True,
        show_layer_names=True,
    )
    print(f"Model diagram saved as {plot_path}")

    # FORCING THE POP-OUT
    os.startfile(str(plot_path))
except Exception as e:
    print(f"Could not generate diagram: {e}")

print("\nMilestone 2 (Architecture Logic) Completed Successfully! [cite: 22]")
