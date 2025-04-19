# Plan Poprawy Semantycznego Dzielenia Tekstu

Aby usprawnić semantyczne dzielenie tekstu w aplikacji `semantic_splitter_app.py` i sprawić, aby była ona bardziej inteligentna w rozpoznawaniu granic zdań (np. ignorując kropki w skrótach takich jak "asp."), należy wykorzystać istniejącą strategię "Semantyczny".

## Kroki do wykonania:

1.  **Wybierz strategię "Semantyczny" w interfejsie aplikacji.** Ta strategia, w przeciwieństwie do "Na zdania", wykorzystuje osadzenia językowe do analizy znaczenia tekstu i identyfikacji naturalnych punktów podziału, co pozwala na lepsze rozpoznawanie granic logicznych fragmentów, niezależnie od interpunkcji.
2.  **Wybierz odpowiednią Metodę Semantyczną.** W ramach strategii "Semantyczny" dostępne są różne metody (np. "percentile", "standard_deviation", "interquartile", "gradient"), które wpływają na sposób identyfikacji punktów podziału na podstawie odległości semantycznych między fragmentami tekstu. Domyślna metoda "percentile" (z progiem 95%) jest często skutecznym punktem wyjścia, ale można eksperymentować z innymi w celu uzyskania optymalnych rezultatów dla konkretnych danych.
3.  **Przetestuj dzielenie na przykładzie.** Użyj tekstu zawierającego problematyczne przypadki (jak skróty z kropkami) i zastosuj strategię "Semantyczny", aby zweryfikować, czy podział jest bardziej zgodny z logicznymi granicami zdań.
4.  **Upewnij się, że serwer osadzeń jest dostępny.** Strategia "Semantyczny" wymaga działającego serwera LM Studio (lub kompatybilnego API) dostarczającego osadzenia, dostępnego pod adresem skonfigurowanym w kodzie aplikacji (domyślnie `http://localhost:1234/v1`). Upewnij się również, że wybrany model osadzeń ("nomic-embed-text" domyślnie) jest dostępny na serwerze.

Przestrzeganie powyższych kroków powinno znacząco poprawić jakość semantycznego dzielenia tekstu w aplikacji.