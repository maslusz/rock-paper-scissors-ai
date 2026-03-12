"""
evaluate.py
-----------
Evaluates a trained RPS model on the held-out test set and prints:
  • Test loss & accuracy
  • Confusion matrix (with visualisation)
  • Full classification report (precision, recall, F1)

Usage:
    python src/evaluate.py [--dataset DATASET_DIR] [--model MODEL_PATH]
                           [--output-dir OUTPUT_DIR]

Example:
    python src/evaluate.py --dataset dataset --model models/rps_model.h5
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import CLASS_NAMES, get_data_generators

# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DEFAULT_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "rps_model.h5")
DEFAULT_OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained RPS model.")
    parser.add_argument("--dataset",    default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model",      default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Render and save a labelled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to: {save_path}")


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {os.path.abspath(args.model)}")
    model = tf.keras.models.load_model(args.model)

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"Loading test data from: {os.path.abspath(args.dataset)}")
    _, _, test_gen = get_data_generators(args.dataset)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set…")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")

    # ── Predictions for detailed metrics ─────────────────────────────────────
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)   # predicted class indices
    y_true = test_gen.classes                  # true class indices

    # ── Classification report ─────────────────────────────────────────────────
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)


if __name__ == "__main__":
    main()
