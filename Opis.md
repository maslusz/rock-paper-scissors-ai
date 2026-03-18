# Dokumentacja wstępna projektu

## 1. Temat projektu

**Rozpoznawanie gestów dłoni w grze papier–kamień–nożyce z wykorzystaniem sieci neuronowej**

## 2. Cel projektu

Celem projektu jest stworzenie aplikacji umożliwiającej rozpoznawanie gestów dłoni w grze _papier–kamień–nożyce_ na podstawie obrazu z kamery, z wykorzystaniem modelu opartego na sieci neuronowej. System ma działać w czasie jak najbardziej zbliżonym do rzeczywistego i umożliwiać rozgrywkę w trybie _człowiek vs komputer_. Dodatkowym celem projektu jest przygotowanie rozwiązania, które w dalszym etapie będzie można rozszerzyć o tryb gry dla dwóch osób obserwowanych przez kamerę.

Istotnym założeniem jest uzyskanie możliwie szybkiego i skutecznego rozpoznawania gestów, tak aby komputer mógł reagować natychmiast po wykryciu ruchu gracza.

## 3. Zakres projektu

Zakres projektu obejmuje przygotowanie kompletnego rozwiązania realizującego podstawową wersję gry _papier–kamień–nożyce_ z automatycznym rozpoznawaniem gestów dłoni. W ramach projektu planowane jest wykorzystanie publicznego zbioru danych do trenowania modelu klasyfikacyjnego rozpoznającego trzy podstawowe gesty: _papier_, _kamień_ oraz _nożyce_.

Projekt obejmuje:

- analizę problemu i określenie wymagań systemowych,
- przygotowanie i wstępne przetworzenie zbioru danych,
- wybór oraz wytrenowanie konwolucyjnej sieci neuronowej (CNN) do klasyfikacji obrazów,
- testowanie i optymalizację modelu pod kątem skuteczności oraz szybkości działania,
- implementację aplikacji działającej z użyciem kamery,
- integrację modelu AI z logiką gry i systemem zliczania punktów.

Projekt w pierwszej kolejności będzie skupiał się na trybie _człowiek vs komputer_. Tryb _dwóch graczy_ traktowany jest jako możliwe rozszerzenie w dalszym etapie prac.

## 4. Wymagania funkcjonalne

Aplikacja powinna umożliwiać:

- pobieranie obrazu z kamery w czasie rzeczywistym,
- wykrywanie i analizę obrazu dłoni użytkownika,
- rozpoznawanie jednego z trzech gestów: _papier_, _kamień_, _nożyce_,
- generowanie ruchu komputera,
- porównanie gestu użytkownika z ruchem komputera zgodnie z zasadami gry,
- wyświetlanie wyniku pojedynczej rundy,
- zliczanie punktów w trakcie rozgrywki,
- rozpoczęcie nowej gry lub kolejnej rundy,
- działanie w podstawowym trybie _człowiek vs komputer_.

Dodatkowo aplikacja powinna być przygotowana do możliwej rozbudowy o:

- tryb gry dla dwóch osób,
- rozszerzenie logiki wykrywania o więcej niż jedną dłoń,
- bardziej rozbudowany interfejs użytkownika.

## 5. Wymagania niefunkcjonalne

System powinien:

- działać w czasie zbliżonym do rzeczywistego,
- zapewniać możliwie wysoką skuteczność klasyfikacji gestów,
- cechować się stabilnym działaniem podczas korzystania z kamery,
- umożliwiać dalszy rozwój i modyfikację rozwiązania,
- działać lokalnie na komputerze użytkownika.

## 6. Dane wejściowe i wyjściowe

Danymi wejściowymi systemu będą:

- obrazy przedstawiające gesty dłoni ze zbioru danych treningowych,
- obraz z kamery przechwytywany w czasie rzeczywistym podczas działania aplikacji.

Natomiast danymi wyjściowymi systemu będą:

- rozpoznana klasa gestu użytkownika,
- ruch wygenerowany przez komputer,
- wynik rundy,
- aktualny stan punktacji.

## 7. Ogólna koncepcja działania aplikacji

Działanie aplikacji będzie opierało się na modelu sieci neuronowej wytrenowanym na zbiorze danych zawierającym obrazy gestów _papier_, _kamień_ i _nożyce_. W pierwszym etapie model zostanie nauczony rozpoznawania trzech klas na podstawie przygotowanego datasetu. Następnie model zostanie zintegrowany z aplikacją wykorzystującą kamerę.

W czasie działania programu kamera będzie przechwytywać obraz użytkownika, a następnie wybrana klatka lub fragment obrazu zawierający dłoń zostanie przekazany do modelu. Model dokona klasyfikacji gestu i zwróci przewidywaną klasę. Na tej podstawie aplikacja wygeneruje ruch komputera, porówna oba gesty według zasad gry i wyświetli rezultat rundy. Wynik będzie zapisywany w liczniku punktów, co umożliwi prowadzenie pełnej rozgrywki.

Docelowo aplikacja ma działać szybko i płynnie, tak aby odpowiedź systemu była natychmiastowa z perspektywy użytkownika.
