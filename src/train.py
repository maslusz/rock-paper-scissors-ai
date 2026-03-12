"""
train.py
--------
Training script for the Rock-Paper-Scissors CNN model.

Usage:
    python src/train.py [--dataset DATASET_DIR] [--epochs EPOCHS]
                        [--model-out MODEL_PATH]

Example:
    python src/train.py --dataset dataset --epochs 20 --model-out models/rps_model.h5
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# Allow importing sibling modules regardless of where the script is run from
sys.path.insert(0, os.path.dirname(__file__))

from model import build_model
from preprocessing import BATCH_SIZE, IMG_SIZE, get_data_generators

# ──────────────────────────────────────────────────────────────────────────────
# Default paths (relative to the project root)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DEFAULT_MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "models", "rps_model.h5")
DEFAULT_EPOCHS      = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RPS CNN model.")
    parser.add_argument("--dataset",   default=DEFAULT_DATASET_DIR,
                        help="Root directory of the dataset (default: dataset/)")
    parser.add_argument("--epochs",    type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT,
                        help="Where to save the trained model (default: models/rps_model.h5)")
    return parser.parse_args()


def plot_history(history, save_path: str):
    """Save a training / validation accuracy & loss curve as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Accuracy ─────────────────────────────────────────────────────────────
    axes[0].plot(history.history["accuracy"],     label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # ── Loss ─────────────────────────────────────────────────────────────────
    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to: {save_path}")


def main():
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {os.path.abspath(args.dataset)}")
    train_gen, val_gen, _ = get_data_generators(args.dataset)

    print(f"  Classes found : {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val   samples : {val_gen.samples}")

    # ── Build model ───────────────────────────────────────────────────────────
    input_shape = IMG_SIZE + (3,)   # e.g. (64, 64, 3)
    model = build_model(input_shape=input_shape)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)

    callbacks = [
        # Stop training early if val_loss stops improving for 5 epochs
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        # Save the best checkpoint during training
        ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True, verbose=1),
        # Halve the learning rate when val_loss plateaus for 3 epochs
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training for up to {args.epochs} epoch(s)…")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    print(f"\nModel saved to: {os.path.abspath(args.model_out)}")

    # ── Save training curves ──────────────────────────────────────────────────
    curves_path = os.path.splitext(args.model_out)[0] + "_training_curves.png"
    plot_history(history, curves_path)


if __name__ == "__main__":
    main()
