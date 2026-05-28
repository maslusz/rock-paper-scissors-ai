from __future__ import annotations

import random
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
    ROOT_DIR / "data" / "hectorandac_rps_yolo" / "RPS_Raw_Images",
    ROOT_DIR / "data" / "google_rps" / "rps",
]
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


def split_files(files: list[tuple[Path, str]]) -> dict[str, list[tuple[Path, str]]]:
    shuffled = files[:]
    random.Random(SEED).shuffle(shuffled)

    train_end = int(len(shuffled) * SPLITS["train"])
    val_end = train_end + int(len(shuffled) * SPLITS["val"])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_split_files(class_name: str, split_files_map: dict[str, list[tuple[Path, str]]]) -> None:
    for split_name, items in split_files_map.items():
        destination_dir = TARGET_DIR / split_name / class_name
        for source_file, target_name in items:
            shutil.copy2(source_file, destination_dir / target_name)


def validate_source_dirs() -> None:
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            raise FileNotFoundError(f"Katalog zrodlowy nie istnieje: {source_dir}")
        missing = [class_name for class_name in CLASS_NAMES if not (source_dir / class_name).exists()]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(f"Brakuje katalogow klas w {source_dir}: {missing_str}")


def main() -> int:
    validate_source_dirs()
    ensure_clean_target_dir(TARGET_DIR)

    print("Przygotowuje polaczony dataset klasyfikacyjny:")
    for i, source_dir in enumerate(SOURCE_DIRS, 1):
        print(f"- zrodlo {i}: {source_dir}")
    print(f"- cel:    {TARGET_DIR}")
    print(f"- seed:   {SEED}")

    for class_name in CLASS_NAMES:
        source_files = []
        for source_dir in SOURCE_DIRS:
            prefix = "kaggle" if "hectorandac" in str(source_dir) else "google"
            dir_path = source_dir / class_name
            for file_path in dir_path.iterdir():
                if file_path.is_file() and not file_path.name.startswith("."):
                    target_name = f"{prefix}_{file_path.name}"
                    source_files.append((file_path, target_name))
        
        source_files.sort(key=lambda x: x[1])  # Sort by target name for determinism
        split_files_map = split_files(source_files)
        copy_split_files(class_name, split_files_map)

        counts = ", ".join(f"{split_name}={len(files)}" for split_name, files in split_files_map.items())
        print(f"- {class_name} (razem {len(source_files)}): {counts}")

    print("Dataset klasyfikacyjny gotowy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
