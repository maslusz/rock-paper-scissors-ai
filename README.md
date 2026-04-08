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

### Deaktywacja środowiska

```bash
deactivate
```

### Usuwanie środowiska

Wykonaj poza środowiskiem:

```bash
rm -rf .venv
```
