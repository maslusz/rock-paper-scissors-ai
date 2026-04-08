from pathlib import Path
import subprocess
import sys


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    model = "runs/detect/train/weights/best.pt"
    source = "data/hectorandac_rps_yolo/RPS_YOLO_Annotated/images/test"

    cmd = [
        "yolo",
        "detect",
        "predict",
        f"model={model}",
        f"source={source}",
    ]
    return subprocess.run(cmd, cwd=root_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
