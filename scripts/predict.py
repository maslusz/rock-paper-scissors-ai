from pathlib import Path
import argparse
import csv
import sys
from ultralytics import YOLO


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_best_model(root_dir: Path) -> Path:
    candidates = sorted(
        root_dir.glob("runs/detect*/**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("Nie znaleziono wag best.pt w katalogach runs/detect*")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchamia predykcje detektora RPS i zapisuje CSV.")
    parser.add_argument("--model", type=Path, default=None, help="Sciezka do wag modelu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar obrazu predykcji.")
    parser.add_argument("--device", default=None, help="Urzadzenie predykcji, np. cpu, mps albo 0.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Katalog lub plik z obrazami testowymi.",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[str]:
    if source.is_file():
        return [str(source)] if source.suffix.lower() in SUPPORTED_SUFFIXES else []

    return sorted(
        str(path)
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def write_result_rows(writer: csv.DictWriter, result, image_file: str, root_dir: Path) -> None:
    image_path = Path(image_file)
    relative_image_path = image_path.relative_to(root_dir)
    speed = result.speed
    timing = {
        "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 3),
        "inference_ms": round(float(speed.get("inference", 0.0)), 3),
        "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 3),
    }
    timing["total_ms"] = round(
        timing["preprocess_ms"] + timing["inference_ms"] + timing["postprocess_ms"],
        3,
    )

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        writer.writerow(
            {
                "image_path": str(relative_image_path),
                "file_name": image_path.name,
                "detection_index": "",
                "predicted_class": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "box_width": "",
                "box_height": "",
                **timing,
            }
        )
        return

    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    for detection_index, (box, confidence, class_index) in enumerate(
        zip(xyxy, confidences, classes)
    ):
        x1, y1, x2, y2 = [float(value) for value in box]
        writer.writerow(
            {
                "image_path": str(relative_image_path),
                "file_name": image_path.name,
                "detection_index": detection_index,
                "predicted_class": result.names[int(class_index)],
                "confidence": round(float(confidence), 6),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
                "box_width": round(x2 - x1, 3),
                "box_height": round(y2 - y1, 3),
                **timing,
            }
        )


def write_detections_csv(
    detector: YOLO,
    image_files: list[str],
    output_path: Path,
    root_dir: Path,
    imgsz: int,
    device: str | None,
) -> None:
    fieldnames = [
        "image_path",
        "file_name",
        "detection_index",
        "predicted_class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_width",
        "box_height",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_file in enumerate(image_files, start=1):
            results = detector.predict(source=image_file, imgsz=imgsz, device=device, verbose=False)
            write_result_rows(writer, results[0], image_file, root_dir)
            csv_file.flush()
            if index % 50 == 0 or index == len(image_files):
                print(f"Przetworzono {index}/{len(image_files)} obrazow")


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    model_path = args.model if args.model else resolve_best_model(root_dir)
    source = args.source or (
        root_dir / "data" / "hectorandac_rps_yolo" / "RPS_YOLO_Annotated" / "images" / "test"
    )
    source = source.resolve()
    image_files = collect_images(source)
    if not image_files:
        raise FileNotFoundError(f"Nie znaleziono obrazow testowych w {source}")

    detector = YOLO(model_path)
    output_path = root_dir / "runs" / "detect" / f"test_detections_imgsz{args.imgsz}.csv"
    write_detections_csv(detector, image_files, output_path, root_dir, args.imgsz, args.device)
    print(f"\nZapisano eksport CSV: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
