import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import pypdf
import docx
import requests # Dodajemy import requests
# Dodajemy RecursiveCharacterTextSplitter i NLTK
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# Importujemy interfejs Embeddings, aby stworzyć własną klasę
# Upewniamy się, że potrzebne importy są obecne
from langchain_core.embeddings import Embeddings
from typing import List

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
            response = requests.post(self.api_url, json=payload, headers=headers)
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

# Zmieniamy nazwę funkcji i dodajemy parametr 'strategy'
def split_text(text, strategy="Semantyczny", semantic_method="percentile", threshold=None, base_url="http://localhost:1234/v1", model_name="nomic-embed-text"):
    """Dzieli tekst na fragmenty używając wybranej strategii."""
    if not text:
        return [], None

    try:
        if strategy == "Semantyczny":
            # Logika dla podziału semantycznego (przeniesiona tutaj)
            embeddings = LMStudioEmbeddings(model_name=model_name, base_url=base_url)
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

            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=semantic_method,
                **threshold_kwargs
            )
            docs = text_splitter.create_documents([text])
            return [doc.page_content for doc in docs], None

        elif strategy == "Na akapity":
            # Logika dla podziału na akapity
            # Domyślne separatory RecursiveCharacterTextSplitter zaczynają od "\n\n"
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
            return chunks, None

        elif strategy == "Na zdania":
            # Logika dla podziału na zdania przy użyciu NLTK
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
                    return None, f"Nie udało się użyć tokenizatora NLTK. {manual_instructions}"
            except Exception as initial_nltk_e:
                 # Inne błędy NLTK podczas pierwszej próby
                 return None, f"Błąd podczas pierwszej próby tokenizacji NLTK: {initial_nltk_e}."

            # Jeśli po wszystkich próbach 'sentences' jest None (co nie powinno się zdarzyć przy tej logice)
            if sentences is None:
                 return None, "Nie udało się uzyskać zdań za pomocą NLTK (niespodziewany błąd)."

            # Przetwarzanie pomyślnie uzyskanych zdań
            try:
                # Usuń puste zdania
                processed_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                print(f"Pomyślnie przetworzono {len(processed_sentences)} zdań.")
                return processed_sentences, None
            except Exception as process_e:
                 return None, f"Błąd podczas przetwarzania zdań po tokenizacji NLTK: {process_e}"

        else:
            return None, f"Nieznana strategia podziału: {strategy}"

    except ImportError as e:
         # Błąd importu może dotyczyć różnych bibliotek w zależności od strategii
         missing_lib = str(e).split("'")[-2] # Próba wyciągnięcia nazwy biblioteki
         return None, f"Brak wymaganej biblioteki: '{missing_lib}'. Sprawdź instalację."
    except Exception as e:
        # Ogólny błąd
        return None, f"Błąd podczas dzielenia tekstu (Strategia: {strategy}): {e}"


# --- Klasa aplikacji GUI ---
        threshold_kwargs = {}
        if threshold is not None:
            threshold_kwargs['breakpoint_threshold_amount'] = threshold
        elif method == "percentile":
             threshold_kwargs.setdefault('breakpoint_threshold_amount', 95.0)
        elif method == "standard_deviation":
             threshold_kwargs.setdefault('breakpoint_threshold_amount', 3.0)
        elif method == "interquartile":
             threshold_kwargs.setdefault('breakpoint_threshold_amount', 1.5)
        elif method == "gradient":
             threshold_kwargs.setdefault('breakpoint_threshold_amount', 95.0) # Gradient używa też percentyla

        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=method,
            **threshold_kwargs
        )
        docs = text_splitter.create_documents([text])
        return [doc.page_content for doc in docs], None
    except ImportError as e:
         return None, f"Brak wymaganej biblioteki: {e}. Zainstaluj: pip install langchain_experimental langchain_openai openai tiktoken"
    except Exception as e:
        # Przechwytywanie błędów związanych z API key itp.
        return None, f"Błąd podczas dzielenia tekstu: {e}"

# --- Klasa aplikacji GUI ---

class SemanticSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Text Splitter")
        self.root.geometry("700x600")

        # Zmienne
        self.file_path = tk.StringVar()
        # Nowa zmienna dla strategii podziału
        self.split_strategy = tk.StringVar(value="Semantyczny")
        self.semantic_method = tk.StringVar(value="percentile") # Zmieniona nazwa zmiennej dla metody semantycznej

        # --- Górna ramka (Wybór pliku) ---
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        # Wybór pliku
        tk.Button(top_frame, text="Wybierz plik (.txt, .pdf, .docx)", command=self.select_file).pack(side=tk.LEFT, padx=5)
        tk.Label(top_frame, textvariable=self.file_path, relief=tk.SUNKEN, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5) # Zwiększono szerokość

        # --- Ramka Strategii Podziału ---
        strategy_frame = tk.Frame(root, padx=10, pady=5)
        strategy_frame.pack(fill=tk.X)

        tk.Label(strategy_frame, text="Strategia podziału:").pack(side=tk.LEFT, padx=5)
        strategies = ["Semantyczny", "Na akapity", "Na zdania"]
        # Usuwamy 'command' i dodajemy powiązanie zdarzenia poniżej
        strategy_menu = ttk.Combobox(strategy_frame, textvariable=self.split_strategy, values=strategies, state="readonly", width=15)
        strategy_menu.pack(side=tk.LEFT, padx=5)
        # Powiązanie zdarzenia <<ComboboxSelected>> z funkcją aktualizującą
        strategy_menu.bind("<<ComboboxSelected>>", self.update_semantic_options_state)

        # --- Ramka Metody Semantycznej (warunkowo aktywna) ---
        self.semantic_options_frame = tk.Frame(root, padx=10, pady=5)
        self.semantic_options_frame.pack(fill=tk.X)

        self.semantic_method_label = tk.Label(self.semantic_options_frame, text="Metoda semantyczna:")
        self.semantic_method_label.pack(side=tk.LEFT, padx=5)
        semantic_methods = ["percentile", "standard_deviation", "interquartile", "gradient"]
        # Zmieniamy textvariable na self.semantic_method
        self.semantic_method_menu = ttk.Combobox(self.semantic_options_frame, textvariable=self.semantic_method, values=semantic_methods, state="readonly", width=20)
        self.semantic_method_menu.pack(side=tk.LEFT, padx=5)

        # --- Ramka Przycisku ---
        button_frame = tk.Frame(root, padx=10, pady=10)
        button_frame.pack(fill=tk.X)
        # Przycisk uruchomienia
        tk.Button(button_frame, text="Podziel tekst", command=self.run_split).pack(side=tk.LEFT, padx=5)
        # Usunięto zduplikowany i błędny przycisk odnoszący się do 'options_frame'

        # --- Dolna ramka (Wyniki) ---
        result_frame = tk.Frame(root, padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(result_frame, text="Wynikowe fragmenty tekstu:").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=25)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED) # Tylko do odczytu na początku

    def update_semantic_options_state(self, event=None):
        """Włącza/wyłącza opcje metody semantycznej w zależności od wybranej strategii."""
        if self.split_strategy.get() == "Semantyczny":
            self.semantic_method_label.config(state=tk.NORMAL)
            self.semantic_method_menu.config(state="readonly")
        else:
            self.semantic_method_label.config(state=tk.DISABLED)
            self.semantic_method_menu.config(state=tk.DISABLED)

    def select_file(self):
        """Otwiera okno dialogowe do wyboru pliku."""
        filepath = filedialog.askopenfilename(
            title="Wybierz plik",
            filetypes=(("Pliki tekstowe", "*.txt"),
                       ("Pliki PDF", "*.pdf"),
                       ("Pliki Word", "*.docx"),
                       ("Wszystkie pliki", "*.*"))
        )
        if filepath:
            self.file_path.set(os.path.basename(filepath)) # Wyświetl tylko nazwę pliku
            self._selected_full_path = filepath # Zapisz pełną ścieżkę wewnętrznie
        else:
            self.file_path.set("")
            self._selected_full_path = None

    def run_split(self):
        """Uruchamia proces odczytu pliku i dzielenia tekstu."""
        if not hasattr(self, '_selected_full_path') or not self._selected_full_path:
            messagebox.showerror("Błąd", "Nie wybrano pliku.")
            return

        strategy = self.split_strategy.get() # Pobierz wybraną strategię
        semantic_method = self.semantic_method.get() # Pobierz metodę semantyczną (jeśli potrzebna)

        # Wyczyść poprzednie wyniki
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Przetwarzanie...\n")
        self.result_text.update() # Odśwież interfejs

        # Odczytaj plik
        content, error = read_file_content(self._selected_full_path)
        if error:
            messagebox.showerror("Błąd odczytu pliku", error)
            self.result_text.delete(1.0, tk.END) # Wyczyść "Przetwarzanie..."
            self.result_text.config(state=tk.DISABLED)
            return
        if not content:
             messagebox.showwarning("Informacja", "Plik jest pusty lub nie udało się odczytać tekstu.")
             self.result_text.delete(1.0, tk.END)
             self.result_text.config(state=tk.DISABLED)
             return


        # Podziel tekst (bez klucza API)
        # Możesz zmienić base_url jeśli Twój LM Studio działa na innym adresie/porcie
        # Podajemy też model_name - domyślnie "nomic-embed-text"
        # Wywołujemy funkcję z poprawnymi parametrami dla naszej nowej implementacji
        # Wywołaj nową funkcję dzielenia tekstu
        chunks, error = split_text(
            content,
            strategy=strategy,
            semantic_method=semantic_method, # Przekaż metodę semantyczną
            base_url="http://localhost:1234/v1",
            model_name="nomic-embed-text"
        )
        if error:
            # Poprawiony komunikat błędu, dodajemy informację o strategii
            messagebox.showerror(f"Błąd dzielenia tekstu (Strategia: {strategy})", f"{error}\n\nSprawdź ustawienia i logi serwera LM Studio (jeśli używasz strategii semantycznej).")
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            return

        # Wyświetl wyniki i zapisz do plików
        self.result_text.delete(1.0, tk.END) # Wyczyść "Przetwarzanie..."

        output_dir = "" # Zmienna do przechowywania ścieżki folderu wyjściowego
        if chunks:
            try:
                # Utwórz folder wyjściowy w tym samym katalogu co plik wejściowy
                input_dir = os.path.dirname(self._selected_full_path)
                input_filename_base = os.path.splitext(os.path.basename(self._selected_full_path))[0]
                output_dir = os.path.join(input_dir, f"{input_filename_base}_chunks")
                os.makedirs(output_dir, exist_ok=True)

                # Wyświetl fragmenty w GUI i zapisz każdy do pliku
                for i, chunk in enumerate(chunks):
                    # Wyświetlanie w GUI
                    self.result_text.insert(tk.END, f"--- Fragment {i+1} ---\n")
                    self.result_text.insert(tk.END, chunk)
                    self.result_text.insert(tk.END, "\n\n")

                    # Zapisywanie do pliku
                    chunk_filename = f"chunk_{i+1}.txt"
                    chunk_filepath = os.path.join(output_dir, chunk_filename)
                    with open(chunk_filepath, 'w', encoding='utf-8') as f:
                        f.write(chunk)

                # Dodaj informację o zapisaniu plików na końcu (poprawione przecinki)
                self.result_text.insert(tk.END, f"\n--- ZAPISANO PLIKI ---\n")
                self.result_text.insert(tk.END, f"Zapisano {len(chunks)} fragmentów jako pliki .txt w folderze:\n{output_dir}\n")
                messagebox.showinfo("Zapisano pliki", f"Zapisano {len(chunks)} fragmentów w folderze:\n{output_dir}")

            except Exception as save_e: # Dodano brakujący blok except
                 messagebox.showerror("Błąd zapisu plików", f"Wystąpił błąd podczas zapisywania fragmentów do plików: {save_e}")
                 # Poprawiony przecinek
                 self.result_text.insert(tk.END, f"\n--- BŁĄD ZAPISU PLIKÓW ---\nWystąpił błąd: {save_e}\n")

        else:
            self.result_text.insert(tk.END, "Nie znaleziono fragmentów do wyświetlenia ani zapisania.")

        self.result_text.config(state=tk.DISABLED) # Ustaw z powrotem na tylko do odczytu


if __name__ == "__main__":
    # Sprawdzenie i instalacja brakujących bibliotek (opcjonalne, lepsze jest poinstruowanie użytkownika)
    try:
        import langchain_experimental
        # Usunięto niepotrzebne już importy openai, langchain_openai, tiktoken
        import pypdf
        import docx
        # Dodajemy 'requests' do sprawdzanych bibliotek
        import requests
    except ImportError as e:
         # Dodajemy nltk do listy sprawdzanych bibliotek
         missing_libs = ["langchain_experimental", "pypdf", "python-docx", "requests", "langchain-core", "nltk"]
         print(f"Brakująca biblioteka: {e}. Upewnij się, że zainstalowano: pip install {' '.join(missing_libs)}")
         # Można dodać `sys.exit()` jeśli chcesz zatrzymać wykonanie bez bibliotek


    root = tk.Tk()
    app = SemanticSplitterApp(root)
    # Wywołaj raz na początku, aby ustawić poprawny stan początkowy
    app.update_semantic_options_state()
    root.mainloop()