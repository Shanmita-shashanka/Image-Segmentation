import os
import numpy as np
import cv2
import tensorflow as tf
from unet import build_unet  # Make sure unet.py is in the same folder

# ---------------------------
# Automatically detect dataset paths
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
DATASET_DIR = os.path.join(CURRENT_DIR, "..", "dataset")  # dataset folder one level up

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

# Check if folders exist
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")
if not os.path.exists(MASK_DIR):
    raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

print("Image folder found:", IMAGE_DIR)
print("Mask folder found:", MASK_DIR)

IMG_SIZE = (256, 256)

# ---------------------------
# Load dataset
# ---------------------------
def load_dataset(image_dir, mask_dir, target_size=IMG_SIZE):
    images, masks = [], []
    img_names = sorted(os.listdir(image_dir))
    
    for name in img_names:
        # Load image
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size) / 255.0
        images.append(img)

        # Load mask
        mask_path = os.path.join(mask_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

X, Y = load_dataset(IMAGE_DIR, MASK_DIR)
print("Dataset loaded:", X.shape, Y.shape)

# ---------------------------
# Build and train U-Net
# ---------------------------
model = build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Starting training...")
model.fit(X, Y, batch_size=8, epochs=10, validation_split=0.1)
print("Training finished!")

# ---------------------------
# Save trained model
# ---------------------------
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "unet_model.h5")
model.save(MODEL_SAVE_PATH) 
print(f"Model saved as {MODEL_SAVE_PATH}")