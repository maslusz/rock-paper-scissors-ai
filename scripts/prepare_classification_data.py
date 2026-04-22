from __future__ import annotations

import random
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "data" / "hectorandac_rps_yolo" / "RPS_Raw_Images"
TARGET_DIR = ROOT_DIR / "data" / "rps_classification"
SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}
SEED = 42
CLASS_NAMES = ("paper", "rock", "scissors")


def ensure_clean_target_dir(target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)

    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def split_files(files: list[Path]) -> dict[str, list[Path]]:
    shuffled = files[:]
    random.Random(SEED).shuffle(shuffled)

    train_end = int(len(shuffled) * SPLITS["train"])
    val_end = train_end + int(len(shuffled) * SPLITS["val"])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_split_files(class_name: str, split_files_map: dict[str, list[Path]]) -> None:
    for split_name, files in split_files_map.items():
        destination_dir = TARGET_DIR / split_name / class_name
        for source_file in files:
            shutil.copy2(source_file, destination_dir / source_file.name)


def validate_source_dir() -> None:
    missing = [class_name for class_name in CLASS_NAMES if not (SOURCE_DIR / class_name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Brakuje katalogow klas w {SOURCE_DIR}: {missing_str}")


def main() -> int:
    validate_source_dir()
    ensure_clean_target_dir(TARGET_DIR)

    print("Przygotowuje dataset klasyfikacyjny:")
    print(f"- zrodlo: {SOURCE_DIR}")
    print(f"- cel:    {TARGET_DIR}")
    print(f"- seed:   {SEED}")

    for class_name in CLASS_NAMES:
        source_files = sorted(
            file_path
            for file_path in (SOURCE_DIR / class_name).iterdir()
            if file_path.is_file()
        )
        split_files_map = split_files(source_files)
        copy_split_files(class_name, split_files_map)

        counts = ", ".join(f"{split_name}={len(files)}" for split_name, files in split_files_map.items())
        print(f"- {class_name}: {counts}")

    print("Dataset klasyfikacyjny gotowy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
