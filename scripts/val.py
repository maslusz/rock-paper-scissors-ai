from pathlib import Path
import subprocess
import sys


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    model = "runs/detect/train/weights/best.pt"

    cmd = [
        "yolo",
        "detect",
        "val",
        f"model={model}",
        f"data={root_dir / 'configs' / 'dataset-rps.yaml'}",
    ]
    return subprocess.run(cmd, cwd=root_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
