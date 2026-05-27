from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uruchamia predykcje klasyfikatora RPS i zapisuje wyniki do CSV."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Liczba obrazow przetwarzanych w jednej paczce.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Rozmiar obrazu uzywany podczas predykcji.",
    )
    parser.add_argument("--device", default=None, help="Urzadzenie predykcji, np. cpu, mps albo 0.")
    return parser.parse_args()


def write_predictions_csv(
    results: list,
    image_files: list[str],
    output_path: Path,
    root_dir: Path,
    batch: int,
    imgsz: int,
) -> None:
    if not results:
        raise ValueError("Brak wynikow predykcji do zapisania")

    if len(results) != len(image_files):
        raise ValueError(
            f"Liczba wynikow ({len(results)}) nie zgadza sie z liczba obrazow ({len(image_files)})"
        )

    names = results[0].names
    class_names = [names[index] for index in sorted(names)]
    fieldnames = [
        "image_path",
        "file_name",
        "true_class",
        "predicted_class",
        "confidence",
        "is_correct",
        "batch",
        "imgsz",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
        *[f"prob_{class_name}" for class_name in class_names],
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result, image_file in zip(results, image_files):
            image_path = Path(image_file)
            relative_image_path = image_path.relative_to(root_dir)
            true_class = image_path.parent.name
            predicted_class = names[int(result.probs.top1)]
            scores = result.probs.data.cpu().tolist()

            row = {
                "image_path": str(relative_image_path),
                "file_name": image_path.name,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(float(result.probs.top1conf), 6),
                "is_correct": true_class == predicted_class,
                "batch": batch,
                "imgsz": imgsz,
                "preprocess_ms": round(float(result.speed.get("preprocess", 0.0)), 3),
                "inference_ms": round(float(result.speed.get("inference", 0.0)), 3),
                "postprocess_ms": round(float(result.speed.get("postprocess", 0.0)), 3),
                "total_ms": round(
                    float(result.speed.get("preprocess", 0.0))
                    + float(result.speed.get("inference", 0.0))
                    + float(result.speed.get("postprocess", 0.0)),
                    3,
                ),
            }
            for index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = round(float(scores[index]), 6)

            writer.writerow(row)


def main() -> int:
    args = parse_args()
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
    results = classifier.predict(
        source=image_files,
        project=str(root_dir / "runs" / "classify"),
        verbose=False,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
    output_stem = f"test_predictions_batch{args.batch}_imgsz{args.imgsz}"
    output_path = root_dir / "runs" / "classify" / f"{output_stem}.csv"
    write_predictions_csv(
        results,
        image_files,
        output_path,
        root_dir,
        batch=args.batch,
        imgsz=args.imgsz,
    )
    print(f"\nZapisano eksport CSV: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
