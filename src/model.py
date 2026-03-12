"""
model.py
--------
Defines the CNN architecture used for Rock-Paper-Scissors hand gesture recognition.

The model is intentionally kept simple so it is easy to understand for a
university AI project while still achieving solid accuracy on the task.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape: tuple = (64, 64, 3), num_classes: int = 3) -> tf.keras.Model:
    """
    Build and return a CNN model for image classification.

    Architecture overview:
        Conv → ReLU → MaxPool  (×3 blocks, increasing filters)
        Flatten → Dense → Dropout → Dense (softmax output)

    Args:
        input_shape: Height × Width × Channels of input images (default 64×64 RGB).
        num_classes:  Number of output classes (rock, paper, scissors → 3).

    Returns:
        A compiled Keras model ready for training.
    """
    model = models.Sequential(
        [
            # ----------------------------------------------------------------
            # Block 1 – 32 filters, 3×3 kernels
            # ----------------------------------------------------------------
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ----------------------------------------------------------------
            # Block 2 – 64 filters
            # ----------------------------------------------------------------
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ----------------------------------------------------------------
            # Block 3 – 128 filters
            # ----------------------------------------------------------------
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # ----------------------------------------------------------------
            # Classifier head
            # ----------------------------------------------------------------
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),          # regularisation to reduce over-fitting
            layers.Dense(num_classes, activation="softmax"),  # probability per class
        ],
        name="rps_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick sanity-check: print the model summary
    m = build_model()
    m.summary()
