from pathlib import Path
import subprocess
import sys


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    model = "yolov8n.pt"
    epochs = 30
    imgsz = 640

    cmd = [
        "yolo",
        "detect",
        "train",
        f"data={root_dir / 'configs' / 'dataset-rps.yaml'}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
    ]
    return subprocess.run(cmd, cwd=root_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
