# rock-paper-scissors-ai

Hand gesture recognition system for the rock-paper-scissors game using a YOLO model for real-time detection and classification of hand gestures from camera input.

---

## Przygotowanie środowiska

### Instalacja Pythona (w wersji 3.13) na macOS

Najlepiej przez `homebrew`:

```bash
brew install python@3.13
```

### Utworzenie środowiska .venv

Jeśli środowisko nie istnieje, utwórz je poleceniem:

```bash
python3.13 -m venv .venv
```

### Wybór interpretera w VS Code

- `Cmd + Shift + P`
- `Python: Select Interpreter`
- wybierz interpreter z folderu `.venv`

### Uruchomienie środowiska .venv

Dzięki temu pakiety typu `ultralytics`, `opencv`, `kaggle` instalują się tylko do tego projektu, a nie do całego systemu.

```bash
source .venv/bin/activate
```

Po aktywacji w terminalu powinno pojawić się `(.venv)`.

### Sprawdzenie właściwego środowiska

```bash
which python
```

Ścieżka powinna wskazywać na:

```bash
.../.venv/bin/python
```

### Instalacja pakietów wewnątrz środowiska

Najpierw zaktualizuj `pip`:

```bash
python -m pip install --upgrade pip
```

Następnie zainstaluj potrzebne pakiety z pliku `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### Uruchamianie `main.py` wewnątrz środowiska

Skrypt sprawdza czy środowisko Python jest poprawnie przygotowane.

```bash
python main.py
```

### Deaktywacja środowiska

```bash
deactivate
```

### Usuwanie środowiska

Wykonaj poza środowiskiem:

```bash
rm -rf .venv
```

---

## Workflow modelu detekcyjnego

Model detekcyjny sluzy do wykrywania polozenia gestu na obrazie. Jest trenowany na danych YOLO z katalogu:

```bash
data/hectorandac_rps_yolo/RPS_YOLO_Annotated
```

Konfiguracja datasetu znajduje sie w:

```bash
configs/dataset-rps.yaml
```

### Trening modelu detekcyjnego

```bash
python scripts/train.py --epochs 30 --imgsz 640 --batch 16 --name train --device mps
```

Wyniki treningu zostana zapisane do:

```bash
runs/detect/train/
```

Najwazniejsze pliki po treningu:

```bash
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
runs/detect/train/results.csv
```

### Walidacja modelu detekcyjnego

Walidacja na zbiorze `val`:

```bash
python scripts/val.py --model runs/detect/train/weights/best.pt --imgsz 640 --batch 16 --split val --device mps
```

Ewaluacja na zbiorze `test`:

```bash
python scripts/val.py --model runs/detect/train/weights/best.pt --imgsz 640 --batch 16 --split test --device mps
```

### Predykcja detekcyjna na zbiorze testowym

```bash
python scripts/predict.py --model runs/detect/train/weights/best.pt --imgsz 640 --device mps
```

Skrypt uruchamia detekcje na obrazach z:

```bash
data/hectorandac_rps_yolo/RPS_YOLO_Annotated/images/test
```

Wyniki sa zapisywane do pliku CSV:

```bash
runs/detect/test_detections_imgsz640.csv
```

Plik CSV zawiera kolumny:

```csv
image_path,file_name,detection_index,predicted_class,confidence,x1,y1,x2,y2,box_width,box_height,preprocess_ms,inference_ms,postprocess_ms,total_ms
```

---

## Workflow modelu klasyfikacyjnego (Zintegrowany & Ulepszony)

Wdrożyliśmy zaawansowany dwuetapowy proces klasyfikacji z użyciem modelu **YOLOv8s-cls** (Small - 5.07M parametrów), osiągając **100% dokładności (Top-1 Accuracy)** na zbiorze walidacyjnym oraz oszałamiającą prędkość inferencji **0.2 ms** na klatkę dzięki optymalizacji pod Apple Silicon GPU (MPS).

### 1. Przygotowanie zintegrowanego zbioru danych

Skrypt łączy i przygotowuje dane z dwóch niezależnych źródeł:
- **Kaggle dataset** (`hectorandac_rps_yolo`): Zdjęcia dłoni od strony zewnętrznej (backs of hands).
- **Google RPS dataset** (`google_rps`): Zdjęcia dłoni z obu stron (palms i backs of hands).

Dzięki połączeniu zbiorów model bezbłędnie klasyfikuje dłonie pokazane z dowolnej strony (zarówno wierzch, jak i pełne wnętrze dłoni - palm view). Skrypt automatycznie zapobiega konfliktom nazw plików, dodając unikalne prefiksy `google_` i `kaggle_`.

Podział zbioru (train 70%, val 15%, test 15%) uruchomisz poleceniem:
```bash
python scripts/prepare_classification_data.py
```
Wynik trafi do: `data/rps_classification/`

### 2. Zaawansowany trening modelu (YOLOv8s-cls)

Trening korzysta z modelu **YOLOv8s-cls** (Small), który ma 4-krotnie większą pojemność sieciową niż Nano. Wdrożyliśmy w nim profesjonalne techniki augmentacji tła i kształtu w `scripts/train_classify.py`, aby uodpornić sieć na cienie i zróżnicowane tło w pokoju:
- **Random Erasing (erasing=0.4):** Losowe wymazywanie części dłoni w trakcie nauki.
- **Mixup (mixup=0.15):** Losowe miksowanie ze sobą obrazów różnych klas.
- **HSV Hue/Saturation/Value (hsv_h=0.025, hsv_s=0.8, hsv_v=0.6):** Dynamiczna zmiana barwy, jasności i nasycenia pikseli.
- **Scale (scale=0.6):** Losowe przeskalowania obrazu.

Uruchomienie zaawansowanego treningu na Apple Silicon GPU (MPS) trwa zaledwie **12-14 minut** na 12 epokach:
```bash
python scripts/train_classify.py --model yolov8s-cls.pt --epochs 12 --imgsz 224 --batch 16 --name merged_classify --device mps
```
Najlepsze wagi zostaną zapisane w: `runs/classify/merged_classify/weights/best.pt`.

### 3. Walidacja modelu klasyfikacyjnego
```bash
python scripts/val_classify.py --model runs/classify/merged_classify/weights/best.pt --imgsz 224 --device mps
```

### 4. Predykcja na zbiorze testowym
```bash
python scripts/predict_classify.py --batch 32 --imgsz 224 --device mps
```

---

## Integracja z kamerą (Zoptymalizowana ergonomia)

Skrypt `camera_recognize.py` korzysta z dwuetapowego pipeline: detektor lokalizuje dłoń (ROI), a ultra-precyzyjny klasyfikator określa gest. 

### Wdrożone ulepszenia ergonomii i stabilności:
1. **Większy i wygodniejszy ROI (`--roi-size 0.55`):** Powiększyliśmy ramkę ROI do 55% boku ekranu.
2. **Przesunięcie w prawo (`--roi-x-center 0.75`):** Domyślna pozycja ROI została przesunięta w prawą stronę ekranu, aby Twoja twarz nie blokowała i nie zakłócała kadru dłoni.
3. **Zapobieganie ucinaniu palców (`--crop-margin 0.2`):** Margines wycinania dłoni z detektora został zwiększony do 20%, dzięki czemu szeroko rozczapierzony gest **papieru** nie jest ucinany na krawędziach klatki.
4. **Filtrowanie twarzy i tła (`--min-detect-conf 0.45`):** Zwiększyliśmy próg pewności detektora do 0.45. Kiedy pokazujesz dłoń od wewnątrz (której detektor nie zna), słabe wykrycia twarzy lub tła są natychmiast odrzucane, a system płynnie przełącza się na fallback ROI (gdzie klasyfikator bezbłędnie rozpoznaje gest).
5. **Zwiększona płynność (`--stability-window 7`):** Dłuższe okno uśredniania eliminuje chwilowe "mignięcia" gestów.

### Uruchomienie standardowe (Rekomendowane):
```bash
python scripts/camera_recognize.py --use-classifier --device mps
```

### Uruchomienie na stałym, przesuniętym ROI (Idealne przy złym świetle):
```bash
python scripts/camera_recognize.py --use-classifier --fixed-roi-only --device mps
```

Uruchomienie w trybie rozszerzonego debugowania (pokazuje wszystkie klatki, nazwy modeli i stabilizację):
```bash
python scripts/camera_recognize.py --use-classifier --debug --device mps
```

---

## Instrukcja dla Zespołu (Kolega pracujący na innym komputerze)

Dzięki profesjonalnej konfiguracji `.gitignore`, **Twoi koledzy z zespołu NIE muszą uruchamiać procesu uczenia ani pobierać danych na nowo!** 

Czysty plik `.gitignore` filtruje gigantyczne dane treningowe i surowe zbiory danych, ale **automatycznie zachowuje i śledzi wagi modeli `best.pt`** wewnątrz folderu `runs/`.

### Szybki start dla kolegi (Plug & Play):
1. **Sklonować repozytorium** z Twojego gita (będzie od razu zawierać najlepsze, wyuczone wagi w folderze `runs/`).
2. **Przygotować środowisko lokalne**:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. **Od razu odpalić grę** w ultrapłynnym trybie:
   ```bash
   python scripts/camera_recognize.py --use-classifier
   ```
Wszystko zadziała bezbłędnie zoptymalizowane pod Apple Silicon / CPU "out-of-the-box"!
