from pathlib import Path
import sys
from ultralytics import YOLO


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "runs" / "detect" / "train" / "weights" / "best.pt"

    detector = YOLO(model_path)
    detector.val(data=str(root_dir / "configs" / "dataset-rps.yaml"), project=str(root_dir / "runs" / "detect"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
