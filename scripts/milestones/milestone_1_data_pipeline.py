from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dessert_dataset" / "train"

# 1. Define your dataset parameters
batch_size = 32
img_height = 128
img_width = 128
data_dir = str(DATA_DIR)

# 2. Load and Resize the dataset
# This automatically creates a training split and resizes the images
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # Saves 20% of the data for validation later
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Load the validation set
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# 3. Normalize the Pixels (Scaling 0-255 to 0.0-1.0)
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Apply the normalization to the datasets
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print("Data cleaning and preprocessing complete!")

# Retrieve the class names automatically detected by your dataset loader
class_names = train_ds.class_names

# Extract exactly one batch of images and labels from your preprocessed dataset
plt.figure(figsize=(10, 10))
for images, labels in normalized_train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Display the image. Because we already normalized the pixels to fall
        # between 0.0 and 1.0, Matplotlib can read and plot them perfectly.
        plt.imshow(images[i].numpy())

        # Add the class name as the title for each image
        plt.title(class_names[labels[i]])

        # Hide the axis numbers for a cleaner presentation
        plt.axis("off")

# Render the visual output on your screen
print("Displaying images...")
plt.show()
