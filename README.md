# rock-paper-scissors-ai

Hand gesture recognition system for the rock-paper-scissors game using a YOLO model for real-time detection and classification of hand gestures from camera input.

## Manual do projektu

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

```bash
python main.py
```

## Milestone 3: model klasyfikacyjny AI

Na potrzeby punktu 3 warto pracowac na klasyfikacji obrazow, bo celem jest rozpoznanie jednej z trzech klas: `paper`, `rock`, `scissors`.

### 1. Przygotowanie danych do klasyfikacji

Skrypt dzieli surowe obrazy z `data/hectorandac_rps_yolo/RPS_Raw_Images` na zbiory `train`, `val`, `test` i kopiuje je do struktury wymaganej przez YOLO Classification.

```bash
.venv/bin/python scripts/prepare_classification_data.py
```

Wynik trafi do:

```bash
data/rps_classification/
```

### 2. Trening modelu

```bash
.venv/bin/python scripts/train_classify.py
```

Domyslnie trening uzywa:

- modelu bazowego `yolov8n-cls.pt`, jesli plik wag jest dostepny lokalnie
- w trybie offline fallbacku do `yolov8n-cls.yaml`, czyli treningu od zera
- `30` epok
- rozmiaru obrazu `224`

Najlepsze wagi powinny pojawic sie w:

```bash
runs/classify/.../weights/best.pt
```

### 3. Walidacja modelu

```bash
.venv/bin/python scripts/val_classify.py
```

### 4. Predykcja na zbiorze testowym

```bash
.venv/bin/python scripts/predict_classify.py
```

Ten workflow odpowiada wprost kamieniowi milowemu 3: wybor modelu, implementacja, trening i uzyskanie modelu klasyfikujacego trzy gesty.

### Deaktywacja środowiska

```bash
deactivate
```

### Usuwanie środowiska

Wykonaj poza środowiskiem:

```bash
rm -rf .venv
```
