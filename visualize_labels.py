from pathlib import Path

import cv2
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "rock",
    1: "paper",
    2: "scissors",
}


def draw_yolo_boxes(image_path: str, label_path: str) -> None:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Nie udało się wczytać obrazu: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    label_file = Path(label_path)
    if not label_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku etykiet: {label_path}")

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(float, parts)

            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            class_id = int(class_id)
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(Path(image_path).name)
    plt.show()


if __name__ == "__main__":
    image_path = "data/hectorandac_rps_yolo/RPS_YOLO_Annotated/images/test/paper_back (44).jpg"
    label_path = "data/hectorandac_rps_yolo/RPS_YOLO_Annotated/labels/test/paper_back (44).txt"

    draw_yolo_boxes(image_path, label_path)