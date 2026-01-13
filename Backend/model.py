import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "unet_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_mask(image):
    img = preprocess(image)
    mask = model.predict(img)[0]
    mask = (mask > 0.5).astype("uint8") * 255
    return mask