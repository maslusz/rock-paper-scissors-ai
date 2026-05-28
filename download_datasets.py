from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = {
    # "hectorandac/rock-paper-scissors-yolo-annotated": "data/hectorandac_rps_yolo",
    "cubeai/rock-paper-scissors-detection-for-yolov8": "data/cubeai_rps_yolov8",
    # "adilshamim8/rock-paper-scissors": "data/adilshamim8_rps_xml",
}

def main():
    api = KaggleApi()
    api.authenticate()

    for dataset_name, output_dir in DATASETS.items():
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Pobieram: {dataset_name}")
        print(f"Do folderu: {out_path}")

        api.dataset_download_files(
            dataset_name,
            path=str(out_path),
            unzip=True
        )

        print("Gotowe\n")

if __name__ == "__main__":
    main()