import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import pypdf
import docx
import requests # Dodajemy import requests
import re # Potrzebne dla _extract_toc_and_remainder

# Dodajemy RecursiveCharacterTextSplitter i NLTK
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# Importujemy interfejs Embeddings, aby stworzyć własną klasę
# Upewniamy się, że potrzebne importy są obecne
from langchain_core.embeddings import Embeddings
from typing import List

# Importy dla PyQt/PySide (użyjemy PySide6 jako przykład)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit,
                                 QComboBox, QTextEdit, QFileDialog, QSizePolicy,
                                 QGroupBox, QMessageBox, QSpinBox) # Dodano QMessageBox i QSpinBox (lepsze dla liczb)
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtCore import Qt, Signal

# --- Własna klasa Embeddings dla LM Studio ---

class LMStudioEmbeddings(Embeddings):
    """Niestandardowa klasa Embeddings do komunikacji z LM Studio API /v1/embeddings."""
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:1234/v1"):
        self.model_name = model_name
        # Upewniamy się, że base_url kończy się na /v1 i dodajemy /embeddings
        if not base_url.endswith("/v1"):
             # Spróbuj dodać /v1 jeśli go brakuje, lub obsłuż błąd jeśli jest niepoprawny
             if base_url.endswith("/"):
                 base_url = base_url[:-1] # Usuń ostatni slash
             # Zakładamy, że użytkownik podał bazowy URL np. http://localhost:1234
             # Jeśli nie, ta logika może wymagać dostosowania
             if not base_url.endswith(":1234"): # Prosta heurystyka
                 print(f"Ostrzeżenie: base_url '{base_url}' nie wygląda jak standardowy adres LM Studio. Próba użycia domyślnego http://localhost:1234/v1")
                 base_url = "http://localhost:1234/v1"
             else:
                  base_url += "/v1"

        self.api_url = base_url + "/embeddings"
        print(f"LMStudioEmbeddings zainicjalizowano. API URL: {self.api_url}, Model: {self.model_name}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Wysyła żądanie do LM Studio API i zwraca osadzenia."""
        payload = {
            "input": texts,
            "model": self.model_name
        }
        headers = {"Content-Type": "application/json"}

        try:
            # Dodano timeout do żądania
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status() # Rzuci wyjątkiem dla błędów HTTP (4xx, 5xx)

            response_data = response.json()

            # Sprawdź strukturę odpowiedzi - oczekujemy listy w 'data'
            if "data" not in response_data or not isinstance(response_data["data"], list):
                raise ValueError(f"Nieoczekiwana struktura odpowiedzi z LM Studio: brak klucza 'data' lub nie jest listą. Odpowiedź: {response_data}")

            # Wyciągnij osadzenia
            embeddings = [item["embedding"] for item in response_data["data"]]
            return embeddings

        except requests.exceptions.RequestException as e:
            # Obsługa błędów połączenia, timeoutów itp.
            print(f"Błąd połączenia z LM Studio API ({self.api_url}): {e}")
            raise ConnectionError(f"Nie można połączyć się z LM Studio API: {e}") from e
        except Exception as e:
            # Obsługa innych błędów (np. parsowanie JSON, błędy statusu HTTP)
            print(f"Błąd podczas przetwarzania odpowiedzi z LM Studio API: {e}")
            # Dodajemy więcej kontekstu do błędu
            error_context = f"URL: {self.api_url}, Payload: {payload}"
            try:
                 error_context += f", Response Status: {response.status_code}, Response Body: {response.text}"
            except NameError: # jeśli response nie został przypisany
                 pass
            raise RuntimeError(f"Błąd przetwarzania odpowiedzi LM Studio: {e}. {error_context}") from e


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Osadza listę dokumentów."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Osadza pojedyncze zapytanie."""
        result = self._embed([text])
        return result[0]


# --- Funkcje pomocnicze ---

def read_file_content(file_path):
    """Odczytuje zawartość tekstową z plików .txt, .pdf, .docx."""
    _, file_extension = os.path.splitext(file_path)
    content = ""
    try:
        if file_extension.lower() == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_extension.lower() == ".pdf":
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() or "" # Dodaj pusty string jeśli extract_text zwróci None
        elif file_extension.lower() == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                content += para.text + "\n"
        else:
            return None, f"Nieobsługiwany format pliku: {file_extension}"
        return content, None
    except Exception as e:
        return None, f"Błąd podczas odczytu pliku: {e}"

# --- Funkcja do pobierania listy modeli z LM Studio ---
def _fetch_lm_studio_models(base_url: str = "http://localhost:1234/v1") -> List[str] or None:
    """Pobiera listę modeli dostępnych w LM Studio."""
    api_url = base_url.rstrip('/') + "/models" # Upewnij się, że nie ma podwójnego slasha
    try:
        response = requests.get(api_url, timeout=10) # Dodano timeout
        response.raise_for_status() # Rzuci wyjątkiem dla błędów HTTP
        data = response.json()
        # Zakładamy, że lista modeli jest w 'data' i każdy model ma 'id'
        if "data" in data and isinstance(data["data"], list):
            model_ids = [model.get('id') for model in data['data'] if model.get('id')]
            print(f"Pomyślnie pobrano listę modeli z LM Studio: {model_ids}")
            return model_ids if model_ids else None # Zwróć None, jeśli lista jest pusta
        else:
            print(f"Nieoczekiwana struktura odpowiedzi z {api_url}: {data}")
            return None
    except requests.exceptions.Timeout:
         print(f"Timeout podczas próby połączenia z LM Studio API ({api_url}) w celu pobrania modeli.")
         return None
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z LM Studio API ({api_url}) podczas pobierania modeli: {e}")
        return None
    except Exception as e:
        print(f"Błąd podczas przetwarzania listy modeli z LM Studio API: {e}")
        return None


# --- Funkcje pomocnicze dla strategii LLM ---

def _extract_toc_and_remainder(text: str, max_toc_chars: int = 5000) -> tuple[str | None, str, int]:
    """
    Próbuje heurystycznie wykryć i wyodrębnić blok Spisu Treści (ToC).

    Args:
        text: Pełny tekst dokumentu.
        max_toc_chars: Maksymalna liczba znaków przeszukiwanych na potrzeby ToC.

    Returns:
        Krotka: (toc_block, remainder_text, toc_start_index).
        Jeśli ToC nie zostanie znaleziony, toc_block będzie None, remainder_text
        będzie całym tekstem, a toc_start_index będzie -1.
    """
    toc_keywords = ["spis treści", "table of contents", "zawartość"]
    lines = text.splitlines()
    toc_start_line_index = -1
    toc_start_char_index = -1
    toc_end_char_index = -1

    # Znajdź linię rozpoczynającą ToC
    current_char_index = 0
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if any(stripped_line.lower().startswith(keyword) for keyword in toc_keywords):
            # Sprawdźmy, czy to nie jest tylko wzmianka w tekście (prosta heurystyka)
            # - Sprawdź, czy linia jest stosunkowo krótka
            # - Sprawdź, czy w pobliżu są linie z numerami stron (np. ... 123)
            # Na razie zakładamy, że znaleziona linia rozpoczyna ToC
            toc_start_line_index = i
            toc_start_char_index = current_char_index
            print(f"Znaleziono potencjalny początek ToC w linii {i}: '{line}'")
            break
        # Używamy oryginalnej długości linii +1 dla znaku nowej linii, aby poprawnie liczyć indeksy znaków
        current_char_index += len(line) + 1

    if toc_start_line_index == -1:
        print("Nie znaleziono słowa kluczowego rozpoczynającego ToC.")
        return None, text, -1

    # Znajdź koniec ToC (heurystyka: szukaj 2+ pustych linii lub końca sekcji ToC)
    search_end_char_index = min(toc_start_char_index + max_toc_chars, len(text))
    toc_end_search_text = text[toc_start_char_index:search_end_char_index]
    lines_in_search_area = toc_end_search_text.splitlines()

    consecutive_blank_lines = 0
    current_char_offset = 0 # Offset względem toc_start_char_index

    # Szukamy końca ToC iterując po liniach *od początku potencjalnego ToC*
    for i, line in enumerate(lines_in_search_area):
        line_length_with_newline = len(line) + 1
        if not line.strip():
            consecutive_blank_lines += 1
        else:
            # Prosta heurystyka końca ToC: linia nie wygląda jak wpis ToC
            # (np. nie zawiera "..." ani cyfr na końcu po serii kropek/spacji)
            # To jest bardzo uproszczone i może wymagać dopracowania.
            # Na razie użyjemy tylko pustych linii jako wskaźnika końca.
            consecutive_blank_lines = 0

        # Uznajemy koniec ToC po znalezieniu 2 pustych linii z rzędu
        if consecutive_blank_lines >= 2:
            # Koniec ToC to pozycja znaku *przed* pierwszą z tych pustych linii
            # Cofamy się o długość ostatniej pustej linii i poprzedzającej ją niepustej linii
            # To wymaga ostrożności, aby poprawnie obliczyć indeks
            # Spróbujmy inaczej: koniec to początek *pierwszej* pustej linii z sekwencji
            toc_end_char_index = toc_start_char_index + current_char_offset
            # Musimy cofnąć się o długość ostatniej niepustej linii i jej znaku nowej linii,
            # a następnie o długość pierwszej pustej linii i jej znaku nowej linii.
            if i > 0: # Upewnij się, że jest poprzednia linia
                 # Indeks końca to początek pierwszej pustej linii
                 toc_end_char_index = toc_start_char_index + current_char_offset - len(lines_in_search_area[i]) -1
                 # Sprawdźmy czy poprzednia linia była pusta, jeśli tak cofnijmy jeszcze raz
                 if i>1 and not lines_in_search_area[i-1].strip():
                      toc_end_char_index -= (len(lines_in_search_area[i-1])+1)


            print(f"Znaleziono potencjalny koniec ToC (2 puste linie) na pozycji znaku: {toc_end_char_index}")
            break

        current_char_offset += line_length_with_newline

    # Jeśli nie znaleziono końca przez puste linie w ramach limitu, spróbuj innej heurystyki
    # (np. pierwszy nagłówek po ToC - na razie pomijamy dla prostoty)
    if toc_end_char_index == -1:
         # Jeśli nie znaleźliśmy końca (np. brak pustych linii), załóżmy, że ToC zajmuje
         # tylko kilka pierwszych linii lub nie udało się go wyodrębnić poprawnie.
         # W takim przypadku lepiej nie wyodrębniać niczego.
         print("Nie udało się jednoznacznie zidentyfikować końca ToC w przeszukiwanym obszarze. Traktowanie jakby ToC nie został znaleziony.")
         return None, text, -1


    # Wyodrębnij blok ToC i resztę tekstu
    toc_block = text[toc_start_char_index:toc_end_char_index].strip()
    remainder_text = text[toc_end_char_index:].strip() # Tekst po ToC
    # Można by też zachować tekst przed ToC, jeśli toc_start_char_index > 0

    print(f"Wyodrębniono blok ToC (długość: {len(toc_block)}), reszta tekstu (długość: {len(remainder_text)})")
    return toc_block, remainder_text, toc_start_char_index


def _apply_heuristic_filters(phrase: str, min_length: int = 5, forbidden_endings: tuple = ('.', ',', ';', ':'), forbidden_keywords: list = None, required_keyword: str = None) -> bool:
    """Stosuje filtry heurystyczne do potencjalnej frazy rozdziału."""
    if forbidden_keywords is None:
        # Rozszerzona lista słów kluczowych często wskazujących na podrozdziały lub inne elementy
        forbidden_keywords = [
            "punkt", "sekcja", "podpunkt", "rys.", "tab.", "uwaga", "przypis",
            "indeks", "bibliografia", "spis treści", "załącznik", "aneks",
            "streszczenie", "podsumowanie", "wstęp", "wprowadzenie" # Można dostosować
        ]

    # Normalizacja
    normalized_phrase = phrase.strip()
    lower_phrase = normalized_phrase.lower()

    # 1. Filtr długości
    if len(normalized_phrase) < min_length:
        # print(f"Odrzucono (za krótka): '{phrase}'")
        return False

    # 2. Filtr niedozwolonych zakończeń
    if normalized_phrase.endswith(forbidden_endings):
        # print(f"Odrzucono (niedozwolone zakończenie): '{phrase}'")
        return False

    # 3. Filtr zakazanych słów kluczowych
    if any(keyword in lower_phrase for keyword in forbidden_keywords):
        # print(f"Odrzucono (zakazane słowo kluczowe): '{phrase}'")
        return False

    # 4. Filtr wymaganego słowa kluczowego (jeśli aktywowany)
    if required_keyword and required_keyword.lower() not in lower_phrase:
        # print(f"Odrzucono (brak wymaganego słowa kluczowego '{required_keyword}'): '{phrase}'")
        return False

    # 5. Filtr wielkości liter (opcjonalny - na razie pominięty dla większej elastyczności)
    # if not normalized_phrase.istitle() and not normalized_phrase.isupper():
    #     return False

    return True # Fraza przeszła wszystkie filtry

def _find_and_verify_occurrences(phrases: List[str], full_text: str) -> List[tuple[int, str]]:
    """Znajduje wszystkie wystąpienia fraz i weryfikuje ich kontekst (początek linii)."""
    verified_occurrences = []
    seen_indices = set() # Aby unikać duplikatów tej samej pozycji startowej

    for phrase in phrases:
        start_index = 0
        while True:
            # Znajdź kolejne wystąpienie frazy
            found_index = full_text.find(phrase, start_index)
            if found_index == -1:
                break # Nie ma więcej wystąpień tej frazy

            # Weryfikacja kontekstowa: czy zaczyna się na początku linii?
            # Sprawdź, czy indeks to 0 LUB poprzedni znak to nowa linia
            is_start_of_line = (found_index == 0) or (found_index > 0 and full_text[found_index - 1] == '\n')

            if is_start_of_line and found_index not in seen_indices:
                 verified_occurrences.append((found_index, phrase))
                 seen_indices.add(found_index)
                 # print(f"Zweryfikowano kontekstowo: '{phrase}' na pozycji {found_index}")


            # Przesuń start_index, aby szukać dalej
            start_index = found_index + 1 # Szukaj od następnego znaku

    # Sortowanie znalezionych i zweryfikowanych wystąpień według indeksu
    verified_occurrences.sort(key=lambda item: item[0])
    return verified_occurrences


# --- Główna funkcja dzielenia tekstu ---

def split_text(
    text,
    strategy="Semantyczny",
    semantic_method="percentile",
    threshold=None,
    base_url="http://localhost:1234/v1",
    embedding_model_name="nomic-embed-text", # Nazwa modelu osadzeń
    chat_model_name="gemma-3-4b-it",       # Nazwa modelu czatu LLM
    phrase=None,
    min_chunk_chars: int = 500 # Minimalna długość fragmentu
):
    """
    Dzieli tekst na fragmenty używając wybranej strategii.

    Args:
        text: Tekst do podziału.
        strategy: Wybrana strategia ('Semantyczny', 'Na akapity', 'Na zdania', 'Po konkretnej frazie', 'Po rozdziałach (LLM)').
        semantic_method: Metoda progowania dla strategii semantycznej.
        threshold: Wartość progu dla strategii semantycznej.
        base_url: Bazowy URL dla LM Studio (dla osadzeń i LLM).
        embedding_model_name: Nazwa modelu osadzeń do użycia (dla strategii Semantycznej).
        chat_model_name: Nazwa modelu czatu LLM do użycia (dla strategii Po rozdziałach (LLM)).
        phrase: Fraza do podziału dla strategii 'Po konkretnej frazie'.
        min_chunk_chars: Minimalna liczba znaków dla fragmentu (stosowane jako post-processing).
                         Fragmenty krótsze niż ta wartość zostaną połączone z poprzednimi.
                         Ustaw na 0 lub mniej, aby wyłączyć.

    Returns:
        Krotka (List[str], str | None): Lista fragmentów tekstu i ewentualny błąd.
    """
    if not text:
        return [], None

    # Parametry połączenia z LM Studio (przekazywane z GUI)
    llm_base_url = base_url # Używamy tego samego adresu bazowego

    # Szablon promptu dla LLM (do dopracowania)
    LLM_PROMPT_TEMPLATE = """Zidentyfikuj *główne rozdziały* w poniższym tekście (ignorując podrozdziały, sekcje i pozycje ze spisu treści).

Tekst:
---
{text_chunk}
---

Wypisz tylko i wyłącznie pierwsze 5-10 słów (frazę) *każdego głównego rozdziału*, po jednej frazie w każdej linii. Nie dodawaj żadnego dodatkowego tekstu ani wyjaśnień, tylko listę fraz. Zachowaj oryginalną pisownię i wielkość liter. Jeśli nie znaleziono głównych rozdziałów, zwróć pustą odpowiedź.

Przykład formatu odpowiedzi:
Pierwsza fraza rozdziału A...
Pierwsza fraza rozdziału B...
...
"""

    # Funkcja do interakcji z LLM (używa teraz parametru model_name)
    def call_llm_for_chapters(text_chunk: str, model_name: str) -> List[str] or None:
        """Wysyła fragment tekstu do LLM (LM Studio) i zwraca listę fraz rozpoczynających rozdziały."""
        api_url = llm_base_url.rstrip('/') + "/chat/completions" # Endpoint dla czatu w LM Studio
        headers = {"Content-Type": "application/json"}

        # Formatowanie promptu do formatu czatu
        messages = [
             {"role": "system", "content": "Jesteś asystentem pomagającym w identyfikacji rozdziałów w tekście."},
             {"role": "user", "content": LLM_PROMPT_TEMPLATE.format(text_chunk=text_chunk)}
        ]

        payload = {
            "model": model_name, # Użyj nazwy modelu przekazanej jako argument
            "messages": messages,
            "temperature": 0.1, # Niska temperatura dla bardziej deterministycznych wyników
            "max_tokens": 200, # Ogranicz liczbę tokenów w odpowiedzi (lista fraz nie powinna być długa)
             # Dodatkowe parametry API LM Studio, jeśli potrzebne
        }

        try:
            # Dodano timeout do żądania
            response = requests.post(api_url, json=payload, headers=headers, timeout=120) # Dłuższy timeout dla LLM
            response.raise_for_status() # Rzuci wyjątkiem dla błędów HTTP

            response_data = response.json()

            # Parsuj odpowiedź - szukaj tekstu wygenerowanego przez model
            # Zakładamy, że odpowiedź czatu ma strukturę: choices -> [0] -> message -> content
            if "choices" in response_data and len(response_data["choices"]) > 0 and "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                llm_response_text = response_data["choices"][0]["message"]["content"]
                # Podziel odpowiedź na linie i usuń puste
                chapter_phrases = [line.strip() for line in llm_response_text.splitlines() if line.strip()]
                print(f"LLM zidentyfikował frazy: {chapter_phrases}")
                return chapter_phrases
            else:
                 print(f"Nieoczekiwana struktura odpowiedzi LLM: {response_data}")
                 return None # Zwróć None w przypadku problemu z parsowaniem
        except requests.exceptions.Timeout:
             print(f"Timeout podczas wywołania LLM API ({api_url}).")
             return None
        except requests.exceptions.RequestException as e:
            print(f"Błąd połączenia z LLM API ({api_url}): {e}")
            return None # Zwróć None w przypadku błędu połączenia/HTTP
        except Exception as e:
            print(f"Błąd podczas przetwarzania odpowiedzi LLM API: {e}")
            return None # Zwróć None w przypadku innego błędu

    # Koniec zagnieżdżonej funkcji call_llm_for_chapters

    try:
        initial_chunks = [] # Zmienna na wyniki pośrednie z każdej strategii
        error_message = None

        if strategy == "Semantyczny":
            # Logika dla podziału semantycznego
            embeddings = LMStudioEmbeddings(model_name=embedding_model_name, base_url=base_url) # Użyj embedding_model_name z parametrów
            threshold_kwargs = {}
            if threshold is not None:
                threshold_kwargs['breakpoint_threshold_amount'] = threshold
            elif semantic_method == "percentile":
                 threshold_kwargs.setdefault('breakpoint_threshold_amount', 95.0)
            elif semantic_method == "standard_deviation":
                 threshold_kwargs.setdefault('breakpoint_threshold_amount', 3.0)
            elif semantic_method == "interquartile":
                 threshold_kwargs.setdefault('breakpoint_threshold_amount', 1.5)
            elif semantic_method == "gradient":
                 threshold_kwargs.setdefault('breakpoint_threshold_amount', 95.0)
            # usunięto zduplikowane threshold_kwargs

            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=semantic_method,
                **threshold_kwargs
            )
            docs = text_splitter.create_documents([text])
            initial_chunks = [doc.page_content for doc in docs] # Przypisz do initial_chunks

        elif strategy == "Na akapity":
            # Logika dla podziału na akapity
            # Ustawiamy chunk_size na dużą wartość, aby upewnić się, że dzieli głównie po separatorach
            # chunk_overlap=0, aby nie było nakładania
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, # Rozsądnie duży rozmiar, aby nie dzielić wewnątrz akapitów
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""] # Domyślne, dobre dla akapitów
            )
            chunks = text_splitter.split_text(text)
            # Usuńmy puste fragmenty, które mogą powstać przez wielokrotne puste linie
            chunks = [chunk for chunk in chunks if chunk.strip()]
            initial_chunks = chunks # Przypisz do initial_chunks

        elif strategy == "Na zdania":
            # Logika dla podziału na zdania przy użyciu NLTK
            sentences = None # Inicjalizujemy jako None
            try:
                # Pierwsza próba tokenizacji
                print("Próba tokenizacji NLTK...")
                sentences = nltk.sent_tokenize(text)
                print("Pierwsza próba tokenizacji NLTK powiodła się.")
            except LookupError:
                # Brak danych 'punkt', próba pobrania
                print("Nie znaleziono danych NLTK 'punkt'. Próba pobrania...")
                try:
                    nltk.download('punkt', quiet=False)
                    print("Pobieranie 'punkt' zakończone lub pakiet jest aktualny. Ponowna próba tokenizacji...")
                    # Druga próba tokenizacji po pobraniu/sprawdzeniu
                    sentences = nltk.sent_tokenize(text)
                    print("Druga próba tokenizacji NLTK powiodła się.")
                except Exception as download_or_retry_e:
                    # Błąd podczas pobierania LUB podczas drugiej próby tokenizacji
                    print(f"Błąd podczas pobierania NLTK 'punkt' lub ponownej tokenizacji: {download_or_retry_e}")
                    manual_instructions = "Spróbuj ręcznie pobrać dane 'punkt' uruchamiając interpreter Python i wpisując:\nimport nltk\nnltk.download('punkt')"
                    error_message = f"Nie udało się użyć tokenizatora NLTK. {manual_instructions}"
                    initial_chunks = None
            except Exception as initial_nltk_e:
                 # Inne błędy NLTK podczas pierwszej próby
                 error_message = f"Błąd podczas pierwszej próby tokenizacji NLTK: {initial_nltk_e}."
                 initial_chunks = None

            # Jeśli po wszystkich próbach 'sentences' jest None (co nie powinno się zdarzyć przy tej logice)
            if sentences is None and initial_chunks is not None: # Sprawdź czy nie ma już błędu
                 error_message = "Nie udało się uzyskać zdań za pomocą NLTK (niespodziewany błąd)."
                 initial_chunks = None

            # Przetwarzanie pomyślnie uzyskanych zdań
            if initial_chunks is not None: # Kontynuuj tylko jeśli nie było błędu
                try:
                    # Usuń puste zdania
                    processed_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                    print(f"Pomyślnie przetworzono {len(processed_sentences)} zdań.")
                    initial_chunks = processed_sentences # Przypisz do initial_chunks
                except Exception as process_e:
                     error_message = f"Błąd podczas przetwarzania zdań po tokenizacji NLTK: {process_e}"
                     initial_chunks = None # Sygnalizuj błąd

        elif strategy == "Po konkretnej frazie":
             # Logika dla podziału po konkretnej frazie
            if not phrase: # Sprawdź czy fraza została podana
                 error_message = "Nie podano frazy do podziału."
                 initial_chunks = None
            else:
                chunks = []
                current_pos = 0
                while True:
                    # Znajdź kolejne wystąpienie frazy z uwzględnieniem wielkości liter
                    find_pos = text.find(phrase, current_pos)

                    if find_pos == -1:
                        # Nie znaleziono więcej wystąpień frazy
                        # Dodaj pozostały tekst jako ostatni fragment, jeśli nie jest pusty
                        remaining_text = text[current_pos:].strip()
                        if remaining_text:
                             chunks.append(remaining_text)
                        break
                    else:
                        # Znaleziono frazę
                        # Dodaj tekst od aktualnej pozycji do początku znalezionej frazy jako fragment
                        chunk_before_phrase = text[current_pos:find_pos].strip()
                        if chunk_before_phrase: # Dodaj tylko jeśli nie jest pusty
                             chunks.append(chunk_before_phrase)

                        # Teraz dodaj fragment zaczynający się od znalezionej frazy
                        # Znajdź koniec tego fragmentu - będzie to początek następnej frazy
                        next_find_pos = text.find(phrase, find_pos + len(phrase))

                        if next_find_pos == -1:
                             # To ostatnie wystąpienie frazy, dodaj tekst od frazy do końca dokumentu
                             chunk_from_phrase = text[find_pos:].strip()
                             if chunk_from_phrase: # Dodaj tylko jeśli nie jest pusty
                                  chunks.append(chunk_from_phrase)
                             break # Zakończ pętlę

                        else:
                             # Znaleziono następną frazę, dodaj tekst od obecnej frazy do początku następnej frazy
                             chunk_from_phrase = text[find_pos:next_find_pos].strip()
                             if chunk_from_phrase: # Dodaj tylko jeśli nie jest pusty
                                  chunks.append(chunk_from_phrase)
                             current_pos = next_find_pos # Przesuń pozycję do początku następnej frazy

                # Usuń ewentualne puste fragmenty, które mogły powstać na początku lub końcu
                chunks = [chunk for chunk in chunks if chunk]
                initial_chunks = chunks # Przypisz do initial_chunks


        elif strategy == "Po rozdziałach (LLM)":
            # Krok 0: Spróbuj wyodrębnić Spis Treści (ToC)
            toc_block, process_text, toc_start_index = _extract_toc_and_remainder(text)

            if toc_block is None:
                # Nie znaleziono ToC, przetwarzaj cały tekst
                print("Nie wykryto Spisu Treści, przetwarzanie całego tekstu przez LLM.")
                process_text = text # Użyj oryginalnego tekstu
            else:
                print("Wykryto Spis Treści, przetwarzanie reszty tekstu przez LLM.")
                # process_text już zawiera tekst *po* ToC

            # Logika dla podziału na rozdziały za pomocą LLM (działa na process_text)
            # Podziel tekst (bez ToC) na mniejsze fragmenty dla LLM
            # Parametry chunk_size i chunk_overlap zależą od okna kontekstowego LLM
            # Gemma 3 4B IT ma okno kontekstowe 8192 tokenów.
            # Użyjemy mniejszych chunków, żeby zostawić miejsce na prompt i odpowiedź.
            # 10000 znaków to przybliżenie, lepsze byłoby liczenie tokenów.
            # Chunk overlap, żeby nie przegapić granic rozdziałów na brzegach chunków.
            temp_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, # Przybliżony rozmiar fragmentu w znakach
                chunk_overlap=400, # Nakładanie się fragmentów
                length_function=len,
            )
            # Dzielimy process_text (tekst bez ToC lub cały tekst)
            text_chunks_for_llm = temp_splitter.split_text(process_text)
            print(f"Podzielono tekst do przetworzenia na {len(text_chunks_for_llm)} fragmentów dla LLM.")

            all_candidate_phrases: List[str] = []
            # Przetwarzaj każdy fragment tekstu przez LLM
            for i, chunk in enumerate(text_chunks_for_llm):
                print(f"Przetwarzanie fragmentu {i+1}/{len(text_chunks_for_llm)} przez LLM...")
                # Przekaż nazwę modelu czatu LLM
                phrases_in_chunk = call_llm_for_chapters(chunk, chat_model_name)
                if phrases_in_chunk:
                    # Krok 1: Zastosuj filtry heurystyczne do odpowiedzi LLM
                    for phrase in phrases_in_chunk:
                        if _apply_heuristic_filters(phrase): # Użycie nowej funkcji
                            all_candidate_phrases.append(phrase.strip()) # Dodaj oczyszczoną frazę
                        # else:
                        #     print(f"Odfiltrowano heurystycznie: '{phrase}'")


            # Krok 2: Usuń duplikaty z listy kandydatów (zachowując kolejność dla stabilności)
            unique_candidate_phrases = list(dict.fromkeys(all_candidate_phrases))
            print(f"Unikalne frazy kandydackie (po filtrach heurystycznych): {len(unique_candidate_phrases)}")
            # print(unique_candidate_phrases) # Opcjonalnie do debugowania


            # Krok 3: Znajdź wszystkie wystąpienia i zweryfikuj kontekstowo W process_text
            # Ważne: Indeksy będą odnosić się do początku process_text
            verified_split_points_relative = _find_and_verify_occurrences(unique_candidate_phrases, process_text) # Użycie nowej funkcji
            print(f"Zweryfikowane punkty podziału (względne, po weryfikacji kontekstowej): {len(verified_split_points_relative)}")
            # print(verified_split_points_relative) # Opcjonalnie do debugowania


            # Krok 4: Podziel process_text na podstawie zweryfikowanych punktów podziału
            processed_chunks = []
            if not verified_split_points_relative:
                # Jeśli nie znaleziono punktów podziału w reszcie tekstu
                if process_text.strip(): # Dodaj resztę tekstu jako jeden chunk, jeśli nie jest pusty
                    processed_chunks.append(process_text.strip())
                print("Nie znaleziono zweryfikowanych punktów podziału w tekście po ToC.")
            else:
                # Podziel process_text
                last_split_pos_relative = 0
                # Obsłuż tekst PRZED pierwszym punktem podziału w process_text
                if verified_split_points_relative[0][0] > 0:
                    chunk_before_first = process_text[0:verified_split_points_relative[0][0]].strip()
                    if chunk_before_first:
                        processed_chunks.append(chunk_before_first)
                    # Nie ustawiamy last_split_pos_relative tutaj

                # Iteruj przez zweryfikowane punkty podziału (względne)
                for i, (idx_relative, _) in enumerate(verified_split_points_relative):
                    # Określ koniec bieżącego fragmentu w process_text
                    end_idx_relative = len(process_text) # Domyślnie koniec process_text
                    if i + 1 < len(verified_split_points_relative):
                        end_idx_relative = verified_split_points_relative[i + 1][0]

                    # Dodaj fragment od bieżącego punktu (idx_relative) do początku następnego (end_idx_relative)
                    chunk_content = process_text[idx_relative:end_idx_relative].strip()
                    if chunk_content:
                        processed_chunks.append(chunk_content)

            # Krok 5: Połącz wyniki - wstaw ToC na początek (uproszczenie)
            final_chunks = []
            if toc_block:
                final_chunks.append(toc_block) # Dodaj ToC jako pierwszy fragment
            final_chunks.extend(processed_chunks) # Dodaj fragmenty z reszty tekstu

            # Usuń puste fragmenty na wszelki wypadek
            final_chunks = [chunk for chunk in final_chunks if chunk]

            # Zwróć ostateczne fragmenty
            if not final_chunks and toc_block is None: # Tylko jeśli nie było ToC i nic nie znaleziono
                 error_message = "Nie udało się wygenerować żadnych fragmentów po przetworzeniu."
                 initial_chunks = None
            else:
                 initial_chunks = final_chunks # Przypisz do initial_chunks

        else:
             error_message = f"Nieznana strategia podziału: {strategy}"
             initial_chunks = None # Sygnalizuj błąd

        # --- Post-processing: Łączenie małych fragmentów ---
        if initial_chunks is not None and min_chunk_chars > 0:
            print(f"Stosowanie filtra minimalnej długości fragmentu (min_chars={min_chunk_chars})...")
            merged_chunks = []
            for i, chunk in enumerate(initial_chunks):
                stripped_chunk = chunk.strip()
                if not stripped_chunk: # Pomiń puste fragmenty
                    continue

                if len(stripped_chunk) < min_chunk_chars:
                    if merged_chunks:
                        # Dołącz do poprzedniego fragmentu
                        separator = "\n\n" # Domyślny separator
                        # Zachowaj oryginalny chunk, aby nie stracić formatowania
                        if not merged_chunks[-1].endswith(('\n', '\r')): separator = "\n\n"
                        elif not merged_chunks[-1].endswith(('\n\n', '\r\n\r\n')): separator = "\n"
                        else: separator = ""

                        merged_chunks[-1] += separator + chunk
                        print(f"Połączono fragment {i} (dł. {len(stripped_chunk)}) z poprzednim.")
                    else:
                        # Pierwszy fragment jest za mały, zachowaj go
                        merged_chunks.append(chunk)
                        print(f"Pierwszy fragment {i} jest za mały (dł. {len(stripped_chunk)}), zachowano.")
                else:
                    # Fragment jest wystarczająco duży
                    merged_chunks.append(chunk)

            # Zwróć połączone fragmenty, jeśli były jakieś zmiany
            if len(merged_chunks) != len(initial_chunks) or any(merged_chunks[i] != initial_chunks[i] for i in range(len(merged_chunks))):
                 print(f"Zakończono łączenie małych fragmentów. Wynik: {len(merged_chunks)} fragmentów.")
                 initial_chunks = merged_chunks # Zaktualizuj listę do zwrócenia
            else:
                 print("Nie dokonano żadnych zmian podczas łączenia małych fragmentów.")

        # --- Ostateczne zwrócenie wyników ---
        if initial_chunks is None:
             # Jeśli wystąpił błąd wcześniej
             return None, error_message
        else:
             # Jeśli wszystko poszło dobrze (lub nie było potrzeby łączyć)
             return initial_chunks, None


    except ImportError as e:
         # Błąd importu może dotyczyć różnych bibliotek w zależności od strategii
         missing_lib = str(e).split("'")[-2] # Próba wyciągnięcia nazwy biblioteki
         return None, f"Brak wymaganej biblioteki: '{missing_lib}'. Sprawdź instalację."
    except Exception as e:
        # Ogólny błąd
        return None, f"Błąd podczas dzielenia tekstu (Strategia: {strategy}): {e}"


# --- Klasa aplikacji GUI (PyQt/PySide) ---
class PyQtSplitterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantic Text Splitter (PyQt/PySide)")
        self.setGeometry(100, 100, 800, 600) # x, y, szerokość, wysokość

        # Centralny widget i layout główny
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) # Dodaj marginesy zewnętrzne
        main_layout.setSpacing(10) # Dodaj odstępy między elementami

        # Ustawienie czcionki Roboto (jeśli dostępna)
        font = QFont("Roboto")
        # Usunięto sprawdzanie isValid(), które może nie być dostępne w PySide6
        QApplication.setFont(font)
        # Komunikat o niedostępności czcionki nie jest już wyświetlany automatycznie przez aplikację
        # System operacyjny użyje domyślnej czcionki, jeśli Roboto nie będzie dostępna

        # Grupa dla wyboru pliku
        file_group = QGroupBox("Wybór Pliku Wejściowego")
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(10, 10, 10, 10) # Marginesy wewnętrzne grupy
        file_layout.setSpacing(10) # Odstępy wewnątrz grupy

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Wybierz plik (.txt, .pdf, .docx)")
        self.select_file_button = QPushButton("Wybierz plik")
        self.select_file_button.clicked.connect(self.select_file) # Połącz przycisk z metodą

        file_layout.addWidget(self.file_path_edit, 1) # Rozciągnij pole tekstowe
        file_layout.addWidget(self.select_file_button)
        main_layout.addWidget(file_group)

        # Grupa dla strategii podziału i opcji
        options_group = QGroupBox("Opcje Dzielenia")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(10, 10, 10, 10) # Marginesy wewnętrzne grupy
        options_layout.setSpacing(10) # Odstępy wewnątrz grupy

        # Strategia podziału
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategia podziału:"))
        self.split_strategy_combo = QComboBox()
        strategies = ["Semantyczny", "Na akapity", "Na zdania", "Po konkretnej frazie", "Po rozdziałach (LLM)"]
        self.split_strategy_combo.addItems(strategies)
        self.split_strategy_combo.currentTextChanged.connect(self.update_options_visibility) # Połącz z metodą

        strategy_layout.addWidget(self.split_strategy_combo, 1) # Rozciągnij combobox
        strategy_layout.addStretch() # Rozciągnij przestrzeń po prawej
        options_layout.addLayout(strategy_layout)

        # Opcje semantyczne (początkowo ukryte lub wyłączone)
        self.semantic_options_layout = QHBoxLayout() # Użyjemy layoutu do grupowania pól
        self.semantic_options_layout.setSpacing(10)

        self.semantic_method_label = QLabel("Metoda semantyczna:")
        self.semantic_method_combo = QComboBox()
        semantic_methods = ["percentile", "standard_deviation", "interquartile", "gradient"]
        self.semantic_method_combo.addItems(semantic_methods)
        self.semantic_threshold_label = QLabel("Próg (Threshold):")
        self.semantic_threshold_edit = QLineEdit()
        self.semantic_threshold_edit.setPlaceholderText("np. 95.0") # Placeholder
        self.semantic_threshold_edit.setMaximumWidth(80) # Ogranicz szerokość pola progu

        self.semantic_options_layout.addWidget(self.semantic_method_label)
        self.semantic_options_layout.addWidget(self.semantic_method_combo)
        self.semantic_options_layout.addWidget(self.semantic_threshold_label)
        self.semantic_options_layout.addWidget(self.semantic_threshold_edit)
        self.semantic_options_layout.addStretch()
        options_layout.addLayout(self.semantic_options_layout)


        # Pole frazy (początkowo ukryte lub wyłączone)
        self.phrase_layout = QHBoxLayout()
        self.phrase_layout.setSpacing(10)
        self.phrase_label = QLabel("Fraza do podziału:")
        self.phrase_edit = QLineEdit()
        self.phrase_layout.addWidget(self.phrase_label)
        self.phrase_layout.addWidget(self.phrase_edit, 1) # Rozciągnij pole frazy
        self.phrase_layout.addStretch()
        options_layout.addLayout(self.phrase_layout)

        # Pole wyboru modelu osadzeń
        self.embedding_model_layout = QHBoxLayout()
        self.embedding_model_layout.setSpacing(10)
        self.embedding_model_label = QLabel("Model do osadzeń:")
        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.setToolTip("Model używany do generowania osadzeń (strategia Semantyczna).")
        self.embedding_model_layout.addWidget(self.embedding_model_label)
        self.embedding_model_layout.addWidget(self.embedding_model_combo, 1)
        self.embedding_model_layout.addStretch()
        options_layout.addLayout(self.embedding_model_layout)

        # Pole wyboru modelu LLM do czatu (dla strategii LLM)
        self.chat_model_layout = QHBoxLayout()
        self.chat_model_layout.setSpacing(10)
        self.chat_model_label = QLabel("Model LLM do czatu:")
        self.chat_model_combo = QComboBox()
        self.chat_model_combo.setToolTip("Model LLM używany do identyfikacji rozdziałów (strategia Po rozdziałach (LLM)).")
        self.chat_model_layout.addWidget(self.chat_model_label)
        self.chat_model_layout.addWidget(self.chat_model_combo, 1)
        self.chat_model_layout.addStretch()
        options_layout.addLayout(self.chat_model_layout)

        # Pole wprowadzania minimalnej długości fragmentu
        self.min_chunk_layout = QHBoxLayout()
        self.min_chunk_layout.setSpacing(10)
        self.min_chunk_label = QLabel("Min. dł. fragmentu (znaki):")
        # Użyjemy QSpinBox dla lepszej walidacji liczb całkowitych
        self.min_chunk_spinbox = QSpinBox()
        self.min_chunk_spinbox.setRange(0, 99999) # Ustaw zakres od 0 (wyłączone) do dużej wartości
        self.min_chunk_spinbox.setValue(500) # Domyślna wartość
        self.min_chunk_spinbox.setToolTip("Minimalna liczba znaków dla fragmentu. Krótsze fragmenty zostaną połączone z poprzednimi (0 = wyłączone).")
        # self.min_chunk_edit = QLineEdit() # Zastąpione przez QSpinBox
        # self.min_chunk_edit.setPlaceholderText("np. 500")
        # self.min_chunk_edit.setMaximumWidth(80)
        # self.min_chunk_edit.setText("500") # Domyślna wartość
        self.min_chunk_layout.addWidget(self.min_chunk_label)
        self.min_chunk_layout.addWidget(self.min_chunk_spinbox) # Dodaj QSpinBox
        self.min_chunk_layout.addStretch()
        options_layout.addLayout(self.min_chunk_layout)


        main_layout.addWidget(options_group)

        # Przycisk uruchomienia
        self.run_button = QPushButton("Podziel tekst")
        self.run_button.clicked.connect(self.run_split) # Połącz przycisk z metodą
        main_layout.addWidget(self.run_button)


        # Grupa dla wyników
        results_group = QGroupBox("Wynikowe fragmenty tekstu")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(10)

        self.results_text_edit = QTextEdit()
        self.results_text_edit.setReadOnly(True) # Ustaw tylko do odczytu
        self.results_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Pozwól na rozszerzanie

        results_layout.addWidget(self.results_text_edit)
        main_layout.addWidget(results_group, 1) # Rozciągnij grupę wyników

        # Ustaw początkową widoczność opcji (wywołaj po stworzeniu wszystkich widgetów)
        self.update_options_visibility(self.split_strategy_combo.currentText())

        # Pobierz i wypełnij listę modeli LM Studio
        self.fetch_lm_studio_models_and_populate_combos()


    # Metoda do pobierania i wypełniania modeli LM Studio
    def fetch_lm_studio_models_and_populate_combos(self):
        """Pobiera listę modeli z LM Studio i wypełnia pola wyboru."""
        # Tutaj zakładamy, że LM Studio działa na domyślnym adresie
        lm_studio_base_url = "http://localhost:1234/v1"
        models = _fetch_lm_studio_models(lm_studio_base_url)

        # Wyczyść comboboxy przed dodaniem nowych
        self.embedding_model_combo.clear()
        self.chat_model_combo.clear()

        if models:
            self.embedding_model_combo.addItems(models)
            self.chat_model_combo.addItems(models)
            # Spróbuj ustawić domyślne modele, jeśli znane nazwy istnieją w liście
            default_embed_model = "nomic-embed-text"
            default_chat_model = "gemma-3-4b-it" # lub inna często używana nazwa

            # Sprawdź, czy domyślne modele są na liście przed ich ustawieniem
            if default_embed_model in models:
                 self.embedding_model_combo.setCurrentText(default_embed_model)
            elif models: # Jeśli jest jakikolwiek model, ustaw pierwszy jako domyślny
                 self.embedding_model_combo.setCurrentIndex(0)

            if default_chat_model in models:
                 self.chat_model_combo.setCurrentText(default_chat_model)
            elif models: # Jeśli jest jakikolwiek model, ustaw pierwszy jako domyślny
                 self.chat_model_combo.setCurrentIndex(0)

        else:
            # Wyświetl błąd lub ostrzeżenie, jeśli nie udało się pobrać modeli
            QMessageBox.warning(self, "Błąd LM Studio", "Nie udało się pobrać listy modeli z LM Studio. Sprawdź, czy LM Studio jest uruchomione i API jest dostępne (domyślnie http://localhost:1234). Pola wyboru modeli będą puste.")
            # Dodaj pustą opcję, aby uniknąć błędów
            self.embedding_model_combo.addItem("")
            self.chat_model_combo.addItem("")


    # Metoda do dynamicznego ukrywania/pokazywania opcji (PyQt/PySide)
    def update_options_visibility(self, selected_strategy):
        is_semantic = (selected_strategy == "Semantyczny")
        is_phrase = (selected_strategy == "Po konkretnej frazie")
        is_llm_strategy = (selected_strategy == "Po rozdziałach (LLM)") # Nowa flaga dla strategii LLM

        # Zarządzaj widocznością opcji semantycznych
        self.semantic_method_label.setVisible(is_semantic)
        self.semantic_method_combo.setVisible(is_semantic)
        self.semantic_threshold_label.setVisible(is_semantic)
        self.semantic_threshold_edit.setVisible(is_semantic)

        # Zarządzaj widocznością pola frazy
        self.phrase_label.setVisible(is_phrase)
        self.phrase_edit.setVisible(is_phrase)

        # Zarządzaj widocznością pól wyboru modeli LM Studio
        # Model osadzeń widoczny tylko dla strategii Semantycznej
        self.embedding_model_label.setVisible(is_semantic)
        self.embedding_model_combo.setVisible(is_semantic)
        # Model czatu widoczny tylko dla strategii LLM
        self.chat_model_label.setVisible(is_llm_strategy)
        self.chat_model_combo.setVisible(is_llm_strategy)

        # Zarządzaj widocznością pola minimalnej długości fragmentu
        # Widoczne dla wszystkich strategii
        self.min_chunk_label.setVisible(True)
        self.min_chunk_spinbox.setVisible(True)


    def select_file(self):
        """Otwiera okno dialogowe do wyboru pliku."""
        # Użyj QFileDialog z PyQt/PySide
        filepath, _ = QFileDialog.getOpenFileName(
            self, # Rodzic okna (QMainWindow)
            "Wybierz plik", # Tytuł okna
            "", # Domyślny katalog (pusty oznacza bieżący)
            "Pliki tekstowe (*.txt);;Pliki PDF (*.pdf);;Pliki Word (*.docx);;Wszystkie pliki (*.*)" # Filtry plików
        )
        if filepath:
            self.file_path_edit.setText(os.path.basename(filepath)) # Wyświetl tylko nazwę pliku w QLineEdit
            self._selected_full_path = filepath # Zapisz pełną ścieżkę wewnętrznie
        else:
            self.file_path_edit.setText("") # Wyczyść pole tekstowe w GUI
            self._selected_full_path = None

    def run_split(self):
        """Uruchamia proces odczytu pliku i dzielenia tekstu."""
        # Sprawdź, czy wybrano plik
        if not hasattr(self, '_selected_full_path') or not self._selected_full_path:
            QMessageBox.warning(self, "Błąd", "Nie wybrano pliku.")
            return

        # Pobierz wybrane opcje z widżetów PyQt/PySide
        strategy = self.split_strategy_combo.currentText()
        semantic_method = self.semantic_method_combo.currentText()
        phrase = self.phrase_edit.text()

        # Pobierz wybrane modele LM Studio
        embedding_model = self.embedding_model_combo.currentText()
        chat_model = self.chat_model_combo.currentText()

        # Pobierz wartość minimalnej długości fragmentu z QSpinBox
        min_chunk_chars_value = self.min_chunk_spinbox.value()


        # Wyczyść poprzednie wyniki
        self.results_text_edit.clear()
        self.results_text_edit.append("Przetwarzanie...\n")
        QApplication.processEvents() # Wymuś aktualizację GUI

        # Odczytaj plik
        content, error = read_file_content(self._selected_full_path)
        if error:
            QMessageBox.critical(self, "Błąd odczytu pliku", error)
            self.results_text_edit.clear() # Wyczyść "Przetwarzanie..."
            return
        if not content:
             QMessageBox.warning(self, "Informacja", "Plik jest pusty lub nie udało się odczytać tekstu.")
             self.results_text_edit.clear()
             return


        # Pobierz wartość progu semantycznego, jeśli strategia to "Semantyczny"
        threshold_value = None
        if strategy == "Semantyczny":
            try:
                threshold_str = self.semantic_threshold_edit.text() # Pobierz tekst z QLineEdit
                if threshold_str: # Sprawdź, czy pole nie jest puste
                    threshold_value = float(threshold_str) # Spróbuj skonwertować na float
                else:
                    threshold_value = None # Puste pole -> brak wartości progu

            except ValueError: # Błąd konwersji na float
                QMessageBox.warning(self, "Błąd wartości progu", "Wprowadzona wartość progu semantycznego jest nieprawidłowa. Użyję wartości domyślnej dla wybranej metody.")
                threshold_value = None # Użyj wartości domyślnej z funkcji split_text


        # Sprawdź, czy fraza jest podana, jeśli strategia to "Po konkretnej frazie"
        if strategy == "Po konkretnej frazie" and not phrase:
             QMessageBox.critical(self, "Błąd", "Proszę podać frazę do podziału.")
             self.results_text_edit.clear()
             return

        # Sprawdź, czy wybrano modele LM Studio dla strategii, które ich potrzebują
        if strategy == "Semantyczny" and not embedding_model:
             QMessageBox.critical(self, "Błąd konfiguracji", "Proszę wybrać model do osadzeń dla strategii Semantycznej.")
             self.results_text_edit.clear()
             return
        if strategy == "Po rozdziałach (LLM)" and not chat_model:
             QMessageBox.critical(self, "Błąd konfiguracji", "Proszę wybrać model LLM do czatu dla strategii Po rozdziałach (LLM).")
             self.results_text_edit.clear()
             return


        # Wywołaj funkcję dzielenia tekstu z odpowiednimi parametrami.
        # Funkcja split_text zawiera logikę dla wszystkich strategii.
        # Adres bazowy LM Studio - można by go też dodać do GUI w przyszłości
        lm_studio_base_url = "http://localhost:1234/v1"

        try:
            chunks, error = split_text(
                content,
                strategy=strategy,
                semantic_method=semantic_method, # Przekaż metodę semantyczną
                threshold=threshold_value, # Przekaż wartość progu
                base_url=lm_studio_base_url, # Przekaż adres bazowy
                embedding_model_name=embedding_model, # Przekaż nazwę modelu osadzeń z GUI
                chat_model_name=chat_model,         # Przekaż nazwę modelu czatu LLM z GUI
                phrase=phrase, # Przekaż frazę
                min_chunk_chars=min_chunk_chars_value # Przekaż minimalną długość fragmentu z GUI
            )
        except ConnectionError as e:
             # Obsługa błędu połączenia z LM Studio zgłoszonego przez LMStudioEmbeddings
             QMessageBox.critical(self, "Błąd połączenia z LM Studio", str(e))
             self.results_text_edit.clear()
             return
        except Exception as e:
             # Inne nieoczekiwane błędy podczas dzielenia
             QMessageBox.critical(self, "Błąd podczas dzielenia tekstu", f"Wystąpił nieoczekiwany błąd: {e}")
             self.results_text_edit.clear()
             return


        # Wyświetl wyniki
        self.results_text_edit.clear() # Wyczyść "Przetwarzanie..."
        if error:
            self.results_text_edit.append(f"BŁĄD: {error}\n")
        elif chunks:
            self.results_text_edit.append(f"Podzielono tekst na {len(chunks)} fragmentów:\n")
            self.results_text_edit.append("="*40 + "\n")
            for i, chunk in enumerate(chunks):
                self.results_text_edit.append(f"--- Fragment {i+1} (długość: {len(chunk)}) ---\n")
                self.results_text_edit.append(chunk)
                self.results_text_edit.append("\n" + "="*40 + "\n")
        else:
            self.results_text_edit.append("Nie wygenerowano żadnych fragmentów.")


# --- Uruchomienie aplikacji ---
if __name__ == "__main__":
    app = QApplication([]) # Inicjalizacja aplikacji PyQt/PySide
    window = PyQtSplitterApp() # Utworzenie głównego okna
    window.show() # Pokazanie okna
    app.exec() # Uruchomienie pętli zdarzeń aplikacji
