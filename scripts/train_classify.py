from __future__ import annotations

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def resolve_model_name() -> str:
    pretrained_weights = "yolov8n-cls.pt"
    fallback_architecture = "yolov8n-cls.yaml"
    return pretrained_weights if Path(pretrained_weights).exists() else fallback_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trenuje model klasyfikacyjny RPS.")
    parser.add_argument("--model", default=None, help="Model bazowy lub plik wag.")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu.")
    parser.add_argument("--imgsz", type=int, default=224, help="Rozmiar obrazu treningowego.")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha.")
    parser.add_argument("--name", default="train", help="Nazwa katalogu run w runs/classify.")
    parser.add_argument("--device", default=None, help="Urzadzenie treningu, np. cpu, mps albo 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "data" / "rps_classification"
    model = args.model or resolve_model_name()

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
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(root_dir / "runs" / "classify"),
        name=args.name,
        device=args.device,
        erasing=0.4,
        mixup=0.15,
        scale=0.6,
        hsv_h=0.025,
        hsv_s=0.8,
        hsv_v=0.6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
