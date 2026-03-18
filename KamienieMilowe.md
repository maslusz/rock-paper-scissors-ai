### 1. Określenie tematu i celu projektu, analiza wymagań

**Opis:** Zdefiniowanie celu projektu, którym jest stworzenie systemu rozpoznawania gestów dłoni w grze *papier–kamień–nożyce* z wykorzystaniem sieci neuronowej. Określenie wymagań funkcjonalnych aplikacji, w szczególności działania w trybie *człowiek vs komputer*, z możliwością dalszego rozszerzenia o tryb *dwóch graczy*. Analiza założeń technicznych, w tym wykorzystania obrazu z kamery oraz klasyfikacji trzech podstawowych gestów: *papier*, *kamień*, *nożyce*.

**Oczekiwany wynik:** Dokumentacja wstępna projektu zawierająca cel, zakres, wymagania funkcjonalne i ogólną koncepcję działania aplikacji.

### 2. Zbiór danych i ich przygotowanie

**Opis:** Wybór i analiza publicznego zbioru danych *Rock Paper Scissors* z platformy Kaggle. Przygotowanie danych do trenowania modelu, obejmujące weryfikację jakości obrazów, podział na zbiory treningowe, walidacyjne i testowe oraz ewentualne przeskalowanie i normalizację obrazów. Dostosowanie danych do klasyfikacji trzech gestów dłoni wykorzystywanych w grze.

**Oczekiwany wynik:** Przygotowany i uporządkowany zbiór danych gotowy do trenowania modelu sieci neuronowej.

### 3. Wybór i implementacja modelu AI

**Opis:** Zaprojektowanie i implementacja modelu sieci neuronowej odpowiedzialnego za rozpoznawanie gestów dłoni na podstawie obrazu wejściowego. Przeprowadzenie procesu trenowania modelu na przygotowanym zbiorze danych oraz zapisanie wytrenowanych wag. Na tym etapie celem jest uzyskanie modelu poprawnie klasyfikującego gesty *papier*, *kamień* i *nożyce*.

**Oczekiwany wynik:** Wytrenowany model AI zdolny do rozpoznawania trzech gestów dłoni z satysfakcjonującą skutecznością.

### 4. Ocena wyników modelu i optymalizacja

**Opis:** Przetestowanie modelu na zbiorze testowym oraz analiza skuteczności klasyfikacji. Identyfikacja najczęstszych błędów rozpoznawania i optymalizacja modelu pod kątem dokładności oraz szybkości działania. Szczególny nacisk zostanie położony na skrócenie czasu reakcji systemu, aby komputer mógł możliwie szybko wykrywać gest gracza i podejmować decyzję w czasie rzeczywistym.

**Oczekiwany wynik:** Zoptymalizowany model o wysokiej skuteczności i krótkim czasie predykcji, gotowy do integracji z aplikacją.

### 5. Implementacja aplikacji i integracja z kamerą

**Opis:** Stworzenie działającej aplikacji gry *papier–kamień–nożyce* w trybie *człowiek vs komputer* z wykorzystaniem obrazu z kamery. Integracja modelu z modułem przechwytywania obrazu oraz logiką gry odpowiedzialną za wybór ruchu komputera, porównanie wyników i zliczanie punktów. Przygotowanie podstaw do dalszego rozszerzenia projektu o tryb rozgrywki dla dwóch osób obserwowanych przez kamerę.

**Oczekiwany wynik:** Działająca aplikacja umożliwiająca rozgrywkę *człowiek vs komputer* z automatycznym rozpoznawaniem gestów dłoni w czasie rzeczywistym.

