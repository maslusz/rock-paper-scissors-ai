from __future__ import annotations

import sys
from pathlib import Path
from ultralytics import YOLO


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
    model_path = resolve_best_model(root_dir)
    source = root_dir / "data" / "rps_classification" / "test"
    image_files = sorted(
        str(path)
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not image_files:
        raise FileNotFoundError(f"Nie znaleziono obrazow do predykcji w {source}")

    classifier = YOLO(model_path)
    classifier.predict(source=image_files, project=str(root_dir / "runs" / "classify"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
