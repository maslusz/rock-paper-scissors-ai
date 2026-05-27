from pathlib import Path
import argparse
import sys
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trenuje model detekcyjny YOLO dla RPS.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model bazowy lub plik wag.")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar obrazu treningowego.")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha.")
    parser.add_argument("--name", default="train", help="Nazwa katalogu run w runs/detect.")
    parser.add_argument("--device", default=None, help="Urzadzenie treningu, np. cpu, mps albo 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    data_yaml = root_dir / "configs" / "dataset-rps.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Nie znaleziono konfiguracji datasetu: {data_yaml}")

    detector = YOLO(args.model)
    detector.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(root_dir / "runs" / "detect"),
        name=args.name,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
