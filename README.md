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

## Workflow modelu klasyfikacyjnego

### Przygotowanie danych do klasyfikacji

Skrypt dzieli surowe obrazy z `data/hectorandac_rps_yolo/RPS_Raw_Images` na zbiory `train`, `val`, `test` i kopiuje je do struktury wymaganej przez YOLO Classification.

```bash
python scripts/prepare_classification_data.py
```

Wynik trafi do:

```bash
data/rps_classification/
```

### Trening modelu klasyfikacyjnego

```bash
python scripts/train_classify.py
```

Domyslnie trening uzywa:

- modelu bazowego `yolov8n-cls.pt`, jesli plik wag jest dostepny lokalnie
- w trybie offline fallbacku do `yolov8n-cls.yaml`, czyli treningu od zera
- `30` epok
- rozmiaru obrazu `224`
- batcha `16`

Najlepsze wagi powinny pojawić się w:

```bash
runs/classify/.../weights/best.pt
```

Parametry treningu mozna ustawic jawnie:

```bash
python scripts/train_classify.py --epochs 30 --imgsz 224 --batch 16 --name train --device mps
```

### Walidacja modelu klasyfikacyjnego

```bash
python scripts/val_classify.py
```

Przyklad walidacji wskazanego modelu:

```bash
python scripts/val_classify.py --model runs/classify/train/weights/best.pt --imgsz 224 --batch 16 --split val --device mps
```

### Predykcja na zbiorze testowym

```bash
python scripts/predict_classify.py --batch 32 --imgsz 224 --device cpu
```

Skrypt uruchamia najlepszy znaleziony model `runs/classify*/**/weights/best.pt` na obrazach z:

```bash
data/rps_classification/test
```

Wyniki sa zapisywane do pliku CSV w katalogu `runs/classify/`. Nazwa pliku zawiera parametry predykcji, np.:

```bash
runs/classify/test_predictions_batch32_imgsz224.csv
```

Jesli plik o takiej nazwie juz istnieje, zostanie nadpisany nowym wynikiem dla tych samych parametrow.

Plik CSV zawiera kolumny:

```csv
image_path,file_name,true_class,predicted_class,confidence,is_correct,batch,imgsz,preprocess_ms,inference_ms,postprocess_ms,total_ms,prob_paper,prob_rock,prob_scissors
```

Parametry predykcji mozna ustawic jawnie:

```bash
python scripts/predict_classify.py --batch 32 --imgsz 224 --device mps
```

---

## Integracja z kamerą

Realtime z kamera korzysta z modelu detekcyjnego do wyznaczenia ROI dloni i z modelu klasyfikacyjnego do rozpoznania gestu na wycinku obrazu.

### Samo rozpoznawanie gestu

Uruchomienie prostego podgladu z kamery, bez elementow gry:

```bash
python scripts/camera_recognize.py --device mps
```

Tylko staly kwadrat ROI na srodku kadru, bez detektora:

```bash
python scripts/camera_recognize.py --device mps --fixed-roi-only
```

Skrypt pokazuje:

- bounding box wykrytego ROI
- przewidziany gest
- confidence klasyfikacji

Rozszerzone informacje debugowe na ekranie:

```bash
python scripts/camera_recognize.py --device mps --debug
```

W trybie `--debug` pokazywane sa dodatkowo: tryb ROI, surowa klasyfikacja,
confidence detekcji, rozmiar i polozenie ROI, licznik stabilizacji oraz nazwy
zaladowanych wag.

Jesli detektor nie znajdzie dloni, skrypt domyslnie przechodzi na staly ROI na srodku kadru.

---
