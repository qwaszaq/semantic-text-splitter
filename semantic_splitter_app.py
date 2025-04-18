import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import pypdf
import docx
import requests # Dodajemy import requests
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

def split_text_semantically(text, method="percentile", threshold=None, base_url="http://localhost:1234/v1", model_name="nomic-embed-text"):
    """Dzieli tekst semantycznie używając LangChain z własną klasą LMStudioEmbeddings."""
    # api_key nie jest już potrzebny
    if not text:
        return [], None # Zwraca pustą listę, jeśli tekst jest pusty

    try:
        # Użyj naszej niestandardowej klasy Embeddings
        # Przekazujemy model_name i base_url (klasa sama doda /embeddings)
        embeddings = LMStudioEmbeddings(model_name=model_name, base_url=base_url)

        # Usunięto krok diagnostyczny, bo teraz mamy własną implementację

        # Ustawienie progu na podstawie typu metody, jeśli nie podano konkretnego
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
        self.split_method = tk.StringVar(value="percentile") # Wartość domyślna

        # --- Górna ramka (Wybór pliku) ---
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        # Wybór pliku
        tk.Button(top_frame, text="Wybierz plik (.txt, .pdf, .docx)", command=self.select_file).pack(side=tk.LEFT, padx=5)
        tk.Label(top_frame, textvariable=self.file_path, relief=tk.SUNKEN, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5) # Zwiększono szerokość

        # --- Środkowa ramka (Opcje podziału) ---
        options_frame = tk.Frame(root, padx=10, pady=5)
        options_frame.pack(fill=tk.X)

        tk.Label(options_frame, text="Metoda podziału:").pack(side=tk.LEFT, padx=5)
        methods = ["percentile", "standard_deviation", "interquartile", "gradient"]
        method_menu = ttk.Combobox(options_frame, textvariable=self.split_method, values=methods, state="readonly", width=20)
        method_menu.pack(side=tk.LEFT, padx=5)

        # Przycisk uruchomienia
        tk.Button(options_frame, text="Podziel tekst", command=self.run_split).pack(side=tk.LEFT, padx=20)

        # --- Dolna ramka (Wyniki) ---
        result_frame = tk.Frame(root, padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(result_frame, text="Wynikowe fragmenty tekstu:").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=25)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED) # Tylko do odczytu na początku

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

        method = self.split_method.get()
        # Klucz API nie jest już potrzebny dla LM Studio

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
        chunks, error = split_text_semantically(
            content,
            method=method,
            base_url="http://localhost:1234/v1", # Upewnij się, że to poprawny adres serwera LM Studio (bez /embeddings)
            model_name="nomic-embed-text" # Upewnij się, że to poprawna nazwa dla API embeddings (jeśli jest wymagana)
        )
        if error:
            # Poprawiony komunikat błędu
            messagebox.showerror("Błąd dzielenia tekstu", f"{error}\n\nSprawdź, czy serwer LM Studio działa pod adresem http://localhost:1234, czy załadowany model ('{model_name}') obsługuje osadzenia i czy endpoint /v1/embeddings jest aktywny. Sprawdź też logi serwera LM Studio.")
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            return

        # Wyświetl wyniki
        self.result_text.delete(1.0, tk.END) # Wyczyść "Przetwarzanie..."
        if chunks:
            for i, chunk in enumerate(chunks):
                self.result_text.insert(tk.END, f"--- Fragment {i+1} ---\n")
                self.result_text.insert(tk.END, chunk)
                self.result_text.insert(tk.END, "\n\n")
        else:
            self.result_text.insert(tk.END, "Nie znaleziono fragmentów do wyświetlenia.")

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
         missing_libs = ["langchain_experimental", "pypdf", "python-docx", "requests", "langchain-core"]
         print(f"Brakująca biblioteka: {e}. Upewnij się, że zainstalowano: pip install {' '.join(missing_libs)}")
         # Można dodać `sys.exit()` jeśli chcesz zatrzymać wykonanie bez bibliotek


    root = tk.Tk()
    app = SemanticSplitterApp(root)
    root.mainloop()