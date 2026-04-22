from __future__ import annotations

import sys
from pathlib import Path
from ultralytics import YOLO


def resolve_model_name() -> str:
    pretrained_weights = "yolov8n-cls.pt"
    fallback_architecture = "yolov8n-cls.yaml"
    return pretrained_weights if Path(pretrained_weights).exists() else fallback_architecture


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "data" / "rps_classification"
    model = resolve_model_name()
    epochs = 30
    imgsz = 224

    if not dataset_dir.exists():
        print(
            "Brakuje datasetu klasyfikacyjnego. "
            "Najpierw uruchom: .venv/bin/python scripts/prepare_classification_data.py"
        )
        return 1

    if model.endswith(".yaml"):
        print(
            "Nie znaleziono lokalnych wag yolov8n-cls.pt. "
            "Trening rusza od zera na bazie architektury yolov8n-cls.yaml."
        )

    classifier = YOLO(model)
    classifier.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        project=str(root_dir / "runs" / "classify"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
