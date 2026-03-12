"""
preprocessing.py
----------------
Utility functions for loading, resizing, normalising, and augmenting images.

All heavy lifting is done via Keras's ImageDataGenerator so there is no need
to write custom data pipelines.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Constants – change these if you want to experiment
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = (64, 64)   # (height, width) fed to the CNN – passed as target_size to Keras
BATCH_SIZE  = 32
CLASS_NAMES = ["rock", "paper", "scissors"]


def get_data_generators(dataset_dir: str):
    """
    Create Keras ImageDataGenerator objects for train, validation, and test sets.

    Data augmentation is applied **only** to the training split to artificially
    increase the variety of training examples and reduce over-fitting.

    Args:
        dataset_dir: Root folder that contains ``train/``, ``val/``, and ``test/``
                     sub-directories, each with one sub-folder per class.

    Returns:
        Tuple (train_gen, val_gen, test_gen) of DirectoryIterator objects.
    """
    # ── Training generator – augmentation turned on ──────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # normalise pixel values to [0, 1]
        rotation_range=15,           # random rotation ± 15°
        width_shift_range=0.1,       # horizontal shift up to 10 %
        height_shift_range=0.1,      # vertical shift up to 10 %
        shear_range=0.1,             # shear transformation
        zoom_range=0.1,              # random zoom
        horizontal_flip=True,        # mirror images left–right
        fill_mode="nearest",         # fill gaps created by transformations
    )

    # ── Validation & test generators – no augmentation, just normalisation ───
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",    # one-hot encoded labels
        classes=CLASS_NAMES,
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,               # keep order for reliable evaluation
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
