"""
predict.py
----------
Classify a single image using the trained RPS model.

Usage:
    python src/predict.py --image PATH_TO_IMAGE [--model MODEL_PATH]

Example:
    python src/predict.py --image my_hand.jpg --model models/rps_model.h5
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import CLASS_NAMES, IMG_SIZE

# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rps_model.h5")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of a single hand-gesture image.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the trained .h5 model.")
    return parser.parse_args()


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk, resize it to the expected input size,
    normalise pixel values to [0, 1], and add a batch dimension.

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.).

    Returns:
        NumPy array with shape (1, H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")  # ensure 3-channel RGB
    # PIL.Image.resize expects (width, height); IMG_SIZE is (height, width),
    # so we reverse it.  For 64×64 this makes no difference, but it is explicit.
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    img_array = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
    return np.expand_dims(img_array, axis=0)               # add batch dimension


def predict(model: tf.keras.Model, image_path: str) -> tuple:
    """
    Run inference on a single image.

    Args:
        model:      Loaded Keras model.
        image_path: Path to the image file.

    Returns:
        Tuple (predicted_class_name, confidence_percentage).
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]   # shape: (num_classes,)

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(predictions[predicted_index]) * 100

    return predicted_class, confidence, predictions


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image file not found – {args.image}")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"Error: model file not found – {args.model}")
        sys.exit(1)

    print(f"Loading model from : {os.path.abspath(args.model)}")
    model = tf.keras.models.load_model(args.model)

    print(f"Classifying image  : {os.path.abspath(args.image)}")
    predicted_class, confidence, all_probs = predict(model, args.image)

    print(f"\nPrediction : {predicted_class.upper()}  ({confidence:.1f}% confidence)")
    print("\nAll class probabilities:")
    for name, prob in zip(CLASS_NAMES, all_probs):
        bar = "█" * int(prob * 30)
        print(f"  {name:<10} {prob * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
