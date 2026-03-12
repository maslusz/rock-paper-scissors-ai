"""
webcam_demo.py
--------------
Real-time Rock-Paper-Scissors hand gesture recognition using your webcam.

Requirements:
    pip install opencv-python tensorflow pillow

Usage:
    python src/webcam_demo.py [--model MODEL_PATH] [--camera CAMERA_ID]

Controls:
    Press  q  to quit the demo.

Example:
    python src/webcam_demo.py --model models/rps_model.h5
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import CLASS_NAMES, IMG_SIZE

# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rps_model.h5")

# Colour per class for the on-screen overlay (BGR format used by OpenCV)
CLASS_COLOURS = {
    "rock":     (0, 128, 255),   # orange
    "paper":    (0, 200,   0),   # green
    "scissors": (0,   0, 255),   # red
}


def parse_args():
    parser = argparse.ArgumentParser(description="Live RPS gesture recognition via webcam.")
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH, help="Path to trained model (.h5).")
    parser.add_argument("--camera", type=int, default=0,        help="Camera device index (default: 0).")
    return parser.parse_args()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to the format expected by the CNN.

    Steps:
        1. Convert BGR → RGB
        2. Resize to IMG_SIZE
        3. Normalise to [0, 1]
        4. Add batch dimension

    Returns:
        NumPy array with shape (1, H, W, 3).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((IMG_SIZE[1], IMG_SIZE[0]))  # PIL wants (width, height)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: model file not found – {args.model}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {os.path.abspath(args.model)}")
    model = tf.keras.models.load_model(args.model)

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}.")
        sys.exit(1)

    print("Webcam demo started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame from camera.")
            break

        # ── Run inference ─────────────────────────────────────────────────────
        img_input   = preprocess_frame(frame)
        predictions = model.predict(img_input, verbose=0)[0]
        pred_index  = int(np.argmax(predictions))
        pred_class  = CLASS_NAMES[pred_index]
        confidence  = float(predictions[pred_index]) * 100

        # ── Overlay text on the frame ─────────────────────────────────────────
        colour = CLASS_COLOURS[pred_class]
        label  = f"{pred_class.upper()}  {confidence:.1f}%"

        # Background rectangle for readability
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

        # Draw a small probability bar for each class at the bottom
        bar_y_start = frame.shape[0] - 20 * len(CLASS_NAMES) - 10
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            y = bar_y_start + i * 20
            bar_len = int(prob * 200)
            c = CLASS_COLOURS[cls]
            cv2.rectangle(frame, (10, y), (10 + bar_len, y + 15), c, -1)
            cv2.putText(frame, f"{cls} {prob * 100:.0f}%", (220, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Rock-Paper-Scissors AI Demo", frame)

        # ── Exit on 'q' ────────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
