### 1. Określenie tematu i celu projektu, analiza wymagań

**Opis:** Zdefiniowanie celu projektu, którym jest stworzenie systemu rozpoznawania gestów dłoni w grze _papier–kamień–nożyce_ z wykorzystaniem sieci neuronowej. Określenie wymagań funkcjonalnych i niefunkcjonalnych aplikacji, w szczególności działania w trybie _człowiek vs komputer_, z możliwością dalszego rozszerzenia o tryb _dwóch graczy_. Analiza założeń technicznych, w tym wykorzystania obrazu z kamery oraz klasyfikacji trzech podstawowych gestów: _papier_, _kamień_, _nożyce_.

**Oczekiwany wynik:** Dokumentacja wstępna projektu zawierająca cel, zakres, wymagania funkcjonalne i ogólną koncepcję działania aplikacji.

### 2. Zbiór danych i ich przygotowanie

**Opis:** Wybór i analiza publicznie dostepnych zbiorów danych. Przygotowanie danych do trenowania modelu, obejmujące weryfikację jakości obrazów, podział na zbiory treningowe, walidacyjne i testowe oraz ewentualne przeskalowanie i normalizację obrazów. Dostosowanie danych do klasyfikacji trzech gestów dłoni wykorzystywanych w grze.

**Oczekiwany wynik:** Przygotowany i uporządkowany zbiór danych gotowy do trenowania modelu sieci neuronowej.

### 3. Wybór i implementacja modelu AI

**Opis:** Zaprojektowanie i implementacja modelu sieci neuronowej odpowiedzialnego za rozpoznawanie gestów dłoni na podstawie obrazu wejściowego. Przeprowadzenie procesu trenowania modelu na przygotowanym zbiorze danych. Na tym etapie celem jest uzyskanie modelu poprawnie klasyfikującego gesty _papier_, _kamień_ i _nożyce_.

**Oczekiwany wynik:** Wytrenowany model AI zdolny do rozpoznawania trzech gestów dłoni z satysfakcjonującą skutecznością.

### 4. Ocena wyników modelu i optymalizacja

**Opis:** Przetestowanie modelu na zbiorze testowym oraz analiza skuteczności klasyfikacji. Identyfikacja najczęstszych błędów rozpoznawania i optymalizacja modelu pod kątem dokładności oraz szybkości działania. Szczególny nacisk zostanie położony na skrócenie czasu reakcji systemu, aby komputer mógł możliwie szybko wykrywać gest gracza i podejmować decyzję w czasie rzeczywistym.

**Oczekiwany wynik:** Zoptymalizowany model o wysokiej skuteczności i krótkim czasie predykcji, gotowy do integracji z aplikacją.

### 5. Implementacja aplikacji i integracja z kamerą

**Opis:** Stworzenie działającej aplikacji gry _papier–kamień–nożyce_ w trybie _człowiek vs komputer_ z wykorzystaniem obrazu z kamery. Integracja modelu z modułem przechwytywania obrazu oraz logiką gry odpowiedzialną za wybór ruchu komputera, porównanie wyników i zliczanie punktów. Przygotowanie podstaw do dalszego rozszerzenia projektu o tryb rozgrywki dla dwóch osób obserwowanych przez kamerę.

**Oczekiwany wynik:** Działająca aplikacja umożliwiająca rozgrywkę _człowiek vs komputer_ z automatycznym rozpoznawaniem gestów dłoni w czasie rzeczywistym.
