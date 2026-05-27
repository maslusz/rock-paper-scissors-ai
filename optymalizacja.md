# Kamień milowy 4 — Ocena wyników modelu i optymalizacja

## 1. Cel

Celem kamienia milowego była ocena skuteczności modelu klasyfikującego gesty oraz sprawdzenie, jak parametry `batch` i `imgsz` wpływają na czas działania modelu.

Model rozpoznaje trzy klasy: `paper`, `rock` oraz `scissors`. Wszystkie próby wykonano na tym samym zbiorze testowym, aby wyniki można było bezpośrednio porównać.

W testach używany był model klasyfikacyjny zapisany jako `runs/classify/train/weights/best.pt`. Jest to model wybierający jedną z trzech klas dla całego obrazu wejściowego. Nie jest to model detekcyjny wyznaczający położenie dłoni na zdjęciu.

## 2. Zbiór testowy

| Klasa      | Liczba obrazów |
| ---------- | -------------: |
| `paper`    |             93 |
| `rock`     |            104 |
| `scissors` |            104 |
| **Razem**  |        **301** |

## 3. Znaczenie mierzonych czasów

Podczas predykcji YOLO raportuje trzy etapy przetwarzania.

- `Preprocess` to przygotowanie obrazu przed przekazaniem go do modelu, np. wczytanie, przeskalowanie i konwersja do tensora (zamianę obrazu na tablicę liczb w formacie batch x kanały x wysokość x szerokość).

- `Inference` to właściwa predykcja modelu, czyli obliczenie prawdopodobieństw dla klas `paper`, `rock` i `scissors`.

- `Postprocess` to obróbka wyniku po predykcji, np. wybór klasy o najwyższym prawdopodobieństwie.

W kontekście pracy z kamerą najważniejszy jest całkowity czas obsługi jednej klatki. W przeprowadzonych testach sama inferencja była krótka, natomiast największy udział w czasie całkowitym miał preprocessing.

Obrazy wejściowe w zbiorze testowym mają rozmiar `3024x3024 px`. Przed predykcją każdy z nich musi zostać wczytany i przeskalowany do rozmiaru używanego przez model, np. `224x224 px`. Dlatego też preprocessing ma największy udział w czasie całkowitym.

## 4. Próba 1 — `batch=1`, `imgsz=224`

Pierwsza próba została wykonana z `batch=1` oraz rozmiarem wejściowym `224x224 px`. Jest to wariant najbardziej zbliżony do pracy z kamerą, gdzie zwykle klasyfikowana jest pojedyncza klatka.

| Metryka            |    Wartość |
| ------------------ | ---------: |
| Poprawne predykcje |  290 / 301 |
| Błędne predykcje   |         11 |
| Accuracy           |     96,35% |
| Preprocess         | 341,578 ms |
| Inference          |   9,759 ms |
| Postprocess        |   0,085 ms |
| Total              | 351,423 ms |

Wariant `batch=1` zachował taką samą skuteczność jak większe batche dla `imgsz=224`, ale miał większy całkowity czas przetwarzania niż `batch=32`.

## 5. Próba 2 — `batch=32`, `imgsz=224`

Druga próba została wykonana z rozmiarem wejściowym `224x224 px`, czyli takim samym jak przy treningu modelu. Predykcja była wykonywana w partiach po `32` obrazy.

| Metryka            |    Wartość |
| ------------------ | ---------: |
| Poprawne predykcje |  290 / 301 |
| Błędne predykcje   |         11 |
| Accuracy           |     96,35% |
| Preprocess         | 332,045 ms |
| Inference          |   8,563 ms |
| Postprocess        |   0,070 ms |
| Total              | 340,677 ms |

Ten wariant uzyskał wysoką skuteczność i krótki czas inferencji. Był to najlepszy wariant jakościowy spośród testów z `batch=32`.

## 6. Próba 3 — `batch=64`, `imgsz=224`

Trzecia próba zachowała rozmiar obrazu `224x224 px`, ale zwiększono batch z `32` do `64`. Cel - czy większa partia obrazów poprawi szybkość przetwarzania.

| Metryka            |    Wartość |
| ------------------ | ---------: |
| Poprawne predykcje |  290 / 301 |
| Błędne predykcje   |         11 |
| Accuracy           |     96,35% |
| Preprocess         | 333,382 ms |
| Inference          |  11,644 ms |
| Postprocess        |   0,273 ms |
| Total              | 345,300 ms |

Zwiększenie batcha nie poprawiło skuteczności ani czasu działania. Accuracy pozostało takie samo, ale czas inferencji i czas całkowity wzrosły.

## 7. Próba 4 — `batch=32`, `imgsz=192`

Czwarta próba sprawdzała wariant pośredni. Rozmiar wejściowy zmniejszono z `224x224 px` do `192x192 px`, pozostawiając `batch=32`.

| Metryka            |    Wartość |
| ------------------ | ---------: |
| Poprawne predykcje |  278 / 301 |
| Błędne predykcje   |         23 |
| Accuracy           |     92,36% |
| Preprocess         | 318,520 ms |
| Inference          |   7,160 ms |
| Postprocess        |   0,096 ms |
| Total              | 325,776 ms |

Ten wariant dał najniższy czas całkowity, ale skuteczność spadła do `92,36%`. Może być traktowany jako kompromis między szybkością i jakością, jednak nie zachowuje tak dobrej dokładności jak `imgsz=224`.

## 8. Próba 5 — `batch=32`, `imgsz=160`

Piąta próba sprawdzała mocniejsze zmniejszenie rozmiaru wejściowego do `160x160 px`. Batch pozostał ustawiony na `32`.

| Metryka            |    Wartość |
| ------------------ | ---------: |
| Poprawne predykcje |  261 / 301 |
| Błędne predykcje   |         40 |
| Accuracy           |     86,71% |
| Preprocess         | 328,028 ms |
| Inference          |   6,421 ms |
| Postprocess        |   0,118 ms |
| Total              | 334,567 ms |

Zmniejszenie obrazu do `160x160 px` dało najniższy czas inferencji, ale spowodowało największy spadek skuteczności. Taki wariant nie nadaje się jako finalne ustawienie dla obecnie wytrenowanego modelu.

## 9. Porównanie prób

| Próba | Parametry               | Accuracy | Błędy | Inference |      Total |
| ----- | ----------------------- | -------: | ----: | --------: | ---------: |
| 1     | `batch=1`, `imgsz=224`  |   96,35% |    11 |  9,759 ms | 351,423 ms |
| 2     | `batch=32`, `imgsz=224` |   96,35% |    11 |  8,563 ms | 340,677 ms |
| 3     | `batch=64`, `imgsz=224` |   96,35% |    11 | 11,644 ms | 345,300 ms |
| 4     | `batch=32`, `imgsz=192` |   92,36% |    23 |  7,160 ms | 325,776 ms |
| 5     | `batch=32`, `imgsz=160` |   86,71% |    40 |  6,421 ms | 334,567 ms |

Najwyższą skuteczność uzyskały próby z `imgsz=224`. Wariant `batch=32`, `imgsz=224` był lepszy od `batch=1` i `batch=64` - zachował tę samą skuteczność przy krótszym czasie inferencji oraz niższym czasie całkowitym.

Zmniejszenie `imgsz` przyspieszało samą inferencję, ale obniżało skuteczność.

Dla najlepszego wariantu `batch=32`, `imgsz=224` model popełnił 11 błędów na 301 obrazach. Najwięcej błędów dotyczyło klasy `paper`, która była mylona z `rock` i `scissors`.

## 10. Wnioski

Sam model nie wydaje się głównym ograniczeniem, ponieważ problemem jest całkowity czas przetwarzania obrazu, który wynosi na ogół ok. `340 ms`. Powodem może być to, że test był wykonywany na obrazach zapisanych na dysku, a nie na klatkach pobieranych bezpośrednio z kamery.

Aktualnie model klasyfikuje cały obraz po przeskalowaniu. Dalsza optymalizacja powinna więc dotyczyć głównie pipeline'u obrazu z kamery: ograniczenia analizowanego obszaru do fragmentu z dłonią, zmniejszenia kosztu preprocessingu oraz wykonywania predykcji tylko wtedy, gdy jest to potrzebne. Najważniejszym kolejnym krokiem powinna być optymalizacja sposobu pobierania i przygotowywania klatek obrazu z kamery.

---
