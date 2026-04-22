from pathlib import Path
import sys
from ultralytics import YOLO


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    model = "yolov8n.pt"
    epochs = 30
    imgsz = 640

    detector = YOLO(model)
    detector.train(
        data=str(root_dir / "configs" / "dataset-rps.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        project=str(root_dir / "runs" / "detect"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
