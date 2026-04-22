from __future__ import annotations

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


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "data" / "rps_classification"
    model_path = resolve_best_model(root_dir)

    classifier = YOLO(model_path)
    classifier.val(data=str(dataset_dir), project=str(root_dir / "runs" / "classify"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
