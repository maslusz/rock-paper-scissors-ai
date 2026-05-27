from __future__ import annotations

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def resolve_best_model(root_dir: Path) -> Path:
    candidates = sorted(
        root_dir.glob("runs/classify*/**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("Nie znaleziono wag best.pt w katalogach runs/classify*")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waliduje model klasyfikacyjny RPS.")
    parser.add_argument("--model", type=Path, default=None, help="Sciezka do wag modelu.")
    parser.add_argument("--imgsz", type=int, default=224, help="Rozmiar obrazu walidacji.")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha.")
    parser.add_argument("--split", default="val", choices=("val", "test"), help="Podzial datasetu.")
    parser.add_argument("--device", default=None, help="Urzadzenie walidacji, np. cpu, mps albo 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "data" / "rps_classification"
    model_path = args.model if args.model else resolve_best_model(root_dir)

    classifier = YOLO(model_path)
    classifier.val(
        data=str(dataset_dir),
        project=str(root_dir / "runs" / "classify"),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
