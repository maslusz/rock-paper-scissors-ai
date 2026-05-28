from __future__ import annotations

import cv2
import random
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["paper", "rock", "scissors"]

# Inicjalizacja słownika ścieżek
DIRS = {
    c: {
        "train": ROOT_DIR / "data" / "rps_classification" / "train" / c,
        "val": ROOT_DIR / "data" / "rps_classification" / "val" / c,
        "test": ROOT_DIR / "data" / "rps_classification" / "test" / c,
    }
    for c in CLASS_NAMES
}


def centered_square_roi(width: int, height: int, size_ratio: float, x_center_ratio: float = 0.75) -> tuple[int, int, int, int]:
    side = int(min(width, height) * size_ratio)
    cx = int(width * x_center_ratio)
    cy = height // 2
    half = side // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Zabezpieczenie przed wyjściem poza ekran
    if x1 < 0:
        diff = -x1
        x1 += diff
        x2 += diff
    if x2 > width:
        diff = x2 - width
        x1 -= diff
        x2 -= diff
    if y1 < 0:
        diff = -y1
        y1 += diff
        y2 += diff
    if y2 > height:
        diff = y2 - height
        y1 -= diff
        y2 -= diff

    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def main() -> int:
    # Upewnij się, że wszystkie foldery docelowe istnieją
    for class_dirs in DIRS.values():
        for d in class_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(0)  # Użyj kamery 0
    if not capture.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return 1

    print("\n=== PROGRAM DO ZBIERANIA WŁASNYCH ZDJĘĆ ===")
    print("Przełączaj klasy klawiszami na klawiaturze:")
    print(" - 'P' -> Papier (złożone palce)")
    print(" - 'R' -> Kamień (pięść)")
    print(" - 'S' -> Nożyce")
    print("Następnie:")
    print(" 1. Umieść dłoń w kwadracie.")
    print(" 2. Naciskaj SPACJĘ, aby zapisać zdjęcie.")
    print(" 3. Naciśnij 'Q', aby zakończyć program.\n")

    active_class = "paper"
    counts = {c: 0 for c in CLASS_NAMES}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Nie udało się odczytać obrazu.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Wyznacz kwadrat identyczny jak w camera_recognize.py
            x1, y1, x2, y2 = centered_square_roi(w, h, 0.55, 0.75)

            # Rysuj kwadrat ROI i instrukcje
            display_frame = frame.copy()
            
            # Kolor ramki w zależności od aktywnej klasy
            if active_class == "paper":
                color = (40, 220, 40)      # Zielony dla papieru
            elif active_class == "rock":
                color = (40, 40, 220)      # Czerwony dla kamienia
            else:
                color = (220, 220, 40)     # Niebieski dla nożyc

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Paski informacyjne
            cv2.putText(
                display_frame,
                f"AKTYWNA KLASA: {active_class.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            cv2.putText(
                display_frame,
                f"SPACJA: zapisz | Q: wyjscie | P, R, S: zmiana klasy",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )
            cv2.putText(
                display_frame,
                f"Zapisano - P:{counts['paper']} R:{counts['rock']} S:{counts['scissors']}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                2,
            )

            cv2.imshow("Zbieranie danych - RPS", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                # Wytnij dłoń z ROI
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    # Losowy podział: 75% train, 15% val, 10% test
                    rand = random.random()
                    if rand < 0.75:
                        split = "train"
                    elif rand < 0.90:
                        split = "val"
                    else:
                        split = "test"

                    target_dir = DIRS[active_class][split]
                    timestamp = int(time.time() * 1000)
                    filename = f"custom_{active_class}_{timestamp}.jpg"
                    filepath = target_dir / filename

                    # Przeskaluj crop do 224x224 (rozmiar wejściowy klasyfikacji) dla spójności
                    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(filepath), crop_resized)

                    counts[active_class] += 1
                    print(f"[{counts[active_class]}] Zapisano {active_class.upper()} do {split}: {filename}")
            elif key == ord("p"):
                active_class = "paper"
                print("Zmieniono aktywną klasę na: PAPIER (paper)")
            elif key == ord("r"):
                active_class = "rock"
                print("Zmieniono aktywną klasę na: KAMIEŃ (rock)")
            elif key == ord("s"):
                active_class = "scissors"
                print("Zmieniono aktywną klasę na: NOŻYCE (scissors)")
            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("\n=== PODSUMOWANIE ZBIERANIA ===")
    print(f" - Papier (paper): {counts['paper']} zdjeć")
    print(f" - Kamień (rock):  {counts['rock']} zdjeć")
    print(f" - Nożyce (scissors): {counts['scissors']} zdjeć")
    print("Zbiór danych został zbalansowany. Uruchom ponownie trening klasyfikatora!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
