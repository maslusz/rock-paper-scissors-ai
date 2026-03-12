# rock-paper-scissors-ai

Hand gesture recognition system for the Rock-Paper-Scissors game using a
Convolutional Neural Network (CNN) trained on image data.

This is a beginner-friendly university AI project built with **Python** and
**TensorFlow / Keras**.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Structure](#dataset-structure)
5. [How to Train the Model](#how-to-train-the-model)
6. [How to Evaluate the Model](#how-to-evaluate-the-model)
7. [How to Run Prediction](#how-to-run-prediction)
8. [Real-Time Webcam Demo](#real-time-webcam-demo)

---

## Project Description

The model classifies hand-gesture images into three classes:

| Class    | Description              |
|----------|--------------------------|
| rock     | Closed fist              |
| paper    | Open hand / flat palm    |
| scissors | Two fingers extended (✌) |

The CNN is trained from scratch using a compact three-block architecture that
is fast to train and easy to understand.  Basic data augmentation (random
rotation, flipping, zooming, etc.) is applied during training to improve
generalisation.

---

## Project Structure

```
rock-paper-scissors-ai/
├── dataset/
│   ├── train/
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   ├── val/
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   └── test/
│       ├── rock/
│       ├── paper/
│       └── scissors/
├── models/                   # saved model files (created during training)
├── src/
│   ├── __init__.py
│   ├── model.py              # CNN architecture definition
│   ├── preprocessing.py      # data loading & augmentation helpers
│   ├── train.py              # training script
│   ├── evaluate.py           # evaluation script
│   ├── predict.py            # single-image prediction script
│   └── webcam_demo.py        # real-time webcam demo (optional)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/maslusz/rock-paper-scissors-ai.git
cd rock-paper-scissors-ai
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Place your images inside the `dataset/` folder, organised by split and class:

```
dataset/
├── train/
│   ├── rock/         ← training images of rock gestures
│   ├── paper/        ← training images of paper gestures
│   └── scissors/     ← training images of scissors gestures
├── val/
│   ├── rock/
│   ├── paper/
│   └── scissors/
└── test/
    ├── rock/
    ├── paper/
    └── scissors/
```

A popular free dataset to use is the
[Rock Paper Scissors dataset on Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).
Download it, then split the images into `train/`, `val/`, and `test/` folders
(e.g. 70 % / 15 % / 15 %).

Images can be **JPEG** or **PNG** in any resolution — the preprocessing script
automatically resizes them to 64 × 64 pixels.

---

## How to Train the Model

```bash
python src/train.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset/` | Root directory of the dataset |
| `--epochs`  | `20`       | Maximum number of training epochs |
| `--model-out` | `models/rps_model.h5` | Where to save the trained model |

Example with custom arguments:

```bash
python src/train.py --dataset dataset --epochs 30 --model-out models/rps_model.h5
```

After training the script saves:
- The best model checkpoint to `models/rps_model.h5`
- A training curve PNG to `models/rps_model_training_curves.png`

---

## How to Evaluate the Model

```bash
python src/evaluate.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset`    | `dataset/`            | Root directory of the dataset  |
| `--model`      | `models/rps_model.h5` | Path to the trained model file |
| `--output-dir` | `models/`             | Where to save evaluation plots |

The script prints:
- Test **loss** and **accuracy**
- **Classification report** (precision, recall, F1 per class)
- **Confusion matrix** (also saved as `models/confusion_matrix.png`)

---

## How to Run Prediction

Classify a single image file:

```bash
python src/predict.py --image path/to/your/image.jpg
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to the image to classify |
| `--model` | `models/rps_model.h5` | Path to the trained model |

Example output:

```
Loading model from : /path/to/models/rps_model.h5
Classifying image  : /path/to/image.jpg

Prediction : ROCK  (97.3% confidence)

All class probabilities:
  rock       97.3%  ██████████████████████████████
  paper       1.9%
  scissors    0.8%
```

---

## Real-Time Webcam Demo

Requires a webcam and OpenCV:

```bash
python src/webcam_demo.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model`  | `models/rps_model.h5` | Path to the trained model |
| `--camera` | `0`                   | Camera device index       |

Press **`q`** to quit the demo.

