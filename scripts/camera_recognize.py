from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def resolve_latest_best(root_dir: Path, pattern: str) -> Path:
    candidates = sorted(root_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Nie znaleziono modelu dla wzorca: {pattern}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime rozpoznawanie gestu z kamery: detekcja ROI i klasyfikacja."
    )
    parser.add_argument("--camera-index", type=int, default=1, help="Indeks kamery OpenCV.")
    parser.add_argument("--device", default="mps", help="Urzadzenie YOLO, np. mps albo cpu.")
    parser.add_argument("--detect-imgsz", type=int, default=640, help="Rozmiar obrazu detekcji.")
    parser.add_argument(
        "--classify-imgsz", type=int, default=224, help="Rozmiar obrazu klasyfikacji."
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="Wykonuj detekcje co N klatek, pozostale klatki uzywaja ostatniego ROI.",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=None,
        help="Sciezka do wag modelu detekcyjnego.",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="Sciezka do wag modelu klasyfikacyjnego.",
    )
    parser.add_argument(
        "--min-detect-conf",
        type=float,
        default=0.1,
        help="Minimalny confidence dla zachowania bounding boxa.",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=5,
        help="Liczba ostatnich klasyfikacji uzywana do stabilizacji gestu.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.1,
        help="Dodatkowy margines cropa wzgledem detekcji.",
    )
    parser.add_argument(
        "--no-fallback-roi",
        action="store_true",
        help="Wylacz staly ROI na srodku kadru, gdy detektor nie znajdzie dloni.",
    )
    parser.add_argument(
        "--roi-size",
        type=float,
        default=0.45,
        help="Rozmiar stalego ROI jako czesc krotszego boku kadru.",
    )
    parser.add_argument(
        "--fixed-roi-only",
        action="store_true",
        help="Pomin detektor i zawsze uzywaj stalego ROI na srodku kadru.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pokaz rozszerzone informacje debugowe na obrazie z kamery.",
    )
    return parser.parse_args()


def expand_box(
    box: tuple[int, int, int, int], width: int, height: int, margin_ratio: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def centered_square_roi(width: int, height: int, size_ratio: float) -> tuple[int, int, int, int]:
    side = int(min(width, height) * size_ratio)
    cx = width // 2
    cy = height // 2
    half = side // 2
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(width, cx + half),
        min(height, cy + half),
    )


def choose_best_box(result, min_conf: float) -> tuple[tuple[int, int, int, int] | None, float]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0

    best_index = int(boxes.conf.argmax().item())
    best_conf = float(boxes.conf[best_index].item())
    if best_conf < min_conf:
        return None, best_conf

    x1, y1, x2, y2 = boxes.xyxy[best_index].tolist()
    return (int(x1), int(y1), int(x2), int(y2)), best_conf


def classify_crop(classifier: YOLO, crop, device: str, imgsz: int) -> tuple[str, float]:
    result = classifier.predict(source=crop, imgsz=imgsz, device=device, verbose=False)[0]
    predicted_class = result.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf)
    return predicted_class, confidence


def draw_overlay(
    frame,
    box: tuple[int, int, int, int] | None,
    detect_conf: float,
    raw_label: str | None,
    raw_conf: float | None,
    stable_label: str | None,
    stable_conf: float | None,
    using_fallback_roi: bool,
    fixed_roi_only: bool,
    debug: bool,
    frame_index: int,
    device: str,
    detect_every: int,
    detector_name: str,
    classifier_name: str,
    history_size: int,
    history_maxlen: int,
) -> None:
    if box:
        x1, y1, x2, y2 = box
        color = (0, 180, 255) if using_fallback_roi else (40, 220, 40)
        label = "ROI fixed" if using_fallback_roi else f"ROI conf: {detect_conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    gesture_text = "Gesture: -" if not stable_label else f"Gesture: {stable_label} ({stable_conf:.2f})"
    cv2.putText(frame, gesture_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if debug:
        raw_text = "Raw: -" if not raw_label else f"Raw: {raw_label} ({raw_conf:.2f})"
        cv2.putText(frame, raw_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
        mode_text = (
            "Mode: fixed_roi_only"
            if fixed_roi_only
            else ("Mode: fallback_roi" if using_fallback_roi else "Mode: detector_roi")
        )
        cv2.putText(frame, mode_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        debug_line_1 = f"Frame: {frame_index}  Device: {device}  Detect every: {detect_every}"
        cv2.putText(frame, debug_line_1, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        debug_line_2 = f"History: {history_size}/{history_maxlen}  Detect conf: {detect_conf:.2f}"
        cv2.putText(frame, debug_line_2, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        if box:
            x1, y1, x2, y2 = box
            roi_w = x2 - x1
            roi_h = y2 - y1
            debug_line_3 = f"ROI: x={x1} y={y1} w={roi_w} h={roi_h}"
        else:
            debug_line_3 = "ROI: -"
        cv2.putText(frame, debug_line_3, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        debug_line_4 = f"Detector: {detector_name}"
        cv2.putText(frame, debug_line_4, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)
        debug_line_5 = f"Classifier: {classifier_name}"
        cv2.putText(frame, debug_line_5, (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)
    cv2.putText(
        frame,
        "Q: quit",
        (20, 285 if debug else 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    classifier_path = args.classifier_model or resolve_latest_best(
        root_dir, "runs/classify*/**/weights/best.pt"
    )

    fallback_roi_enabled = not args.no_fallback_roi

    detector = None
    detector_name = "-"
    if not args.fixed_roi_only:
        detector_path = args.detector_model or resolve_latest_best(
            root_dir, "runs/detect*/**/weights/best.pt"
        )
        detector = YOLO(detector_path)
        detector_name = detector_path.name
    classifier = YOLO(classifier_path)
    classifier_name = classifier_path.name

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc kamery o indeksie {args.camera_index}")

    history: deque[tuple[str, float]] = deque(maxlen=args.stability_window)
    frame_index = 0
    current_box: tuple[int, int, int, int] | None = None
    current_detect_conf = 0.0
    using_fallback_roi = False
    raw_label: str | None = None
    raw_conf: float | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Nie udalo sie odczytac klatki z kamery")
            frame = cv2.flip(frame, 1)

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]

            if args.fixed_roi_only:
                current_box = centered_square_roi(frame_w, frame_h, args.roi_size)
                current_detect_conf = 0.0
                using_fallback_roi = True
            elif frame_index % args.detect_every == 1 or current_box is None:
                detect_result = detector.predict(
                    source=frame,
                    imgsz=args.detect_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                box, detect_conf = choose_best_box(detect_result, args.min_detect_conf)
                if box is not None:
                    current_box = expand_box(box, frame_w, frame_h, args.crop_margin)
                    current_detect_conf = detect_conf
                    using_fallback_roi = False
                elif fallback_roi_enabled:
                    current_box = centered_square_roi(frame_w, frame_h, args.roi_size)
                    current_detect_conf = 0.0
                    using_fallback_roi = True
                else:
                    current_box = None
                    current_detect_conf = detect_conf
                    using_fallback_roi = False

            stable_label = None
            stable_conf = None
            raw_label = None
            raw_conf = None
            if current_box is not None:
                x1, y1, x2, y2 = current_box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    label, confidence = classify_crop(
                        classifier, crop, args.device, args.classify_imgsz
                    )
                    raw_label = label
                    raw_conf = confidence
                    history.append((label, confidence))
                else:
                    history.clear()
            else:
                history.clear()

            if history:
                counts = Counter(label for label, _ in history)
                stable_label, _ = counts.most_common(1)[0]
                label_confidences = [conf for label, conf in history if label == stable_label]
                stable_conf = sum(label_confidences) / len(label_confidences)

            draw_overlay(
                frame,
                current_box,
                current_detect_conf,
                raw_label,
                raw_conf,
                stable_label,
                stable_conf,
                using_fallback_roi,
                args.fixed_roi_only,
                args.debug,
                frame_index,
                args.device,
                args.detect_every,
                detector_name,
                classifier_name,
                len(history),
                history.maxlen,
            )
            cv2.imshow("Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
