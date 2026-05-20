# 🏢 Melbourne Housing ML & Market Dashboard

Interaktywny dashboard analityczno-predykcyjny zbudowany w oparciu o bibliotekę **Streamlit** oraz model Machine Learning (Regresja) do szacowania wartości nieruchomości w Melbourne. Projekt został zaimplementowany obiektowo (zgodnie z zasadami czystego kodu) i zawiera zaawansowane filtrowanie oraz interaktywne wykresy rynkowe.

---

## 📊 Funkcje aplikacji
* **Kalkulator Ceny (ML Predykcja):** Prognozowanie wartości domu w czasie rzeczywistym na podstawie cech takich jak liczba pokoi, odległość od centrum, wskaźniki ekonomiczne (CPI, inflacja) oraz czas sprzedaży.
* **Interaktywny Wykres Trendów:** Dynamiczna wizualizacja zależności ceny od odległości z uwzględnieniem liczby pokoi (wykorzystująca bibliotekę Plotly).
* **Zaawansowane Filtrowanie:** Możliwość selekcji danych po dzielnicach (`Suburb`) oraz typie zabudowy (`Type`) bezpośrednio z panelu bocznego.
* **Wydajność (Caching):** Wykorzystanie mechanizmów pamięci podręcznej Streamlit (`@st.cache_resource`) do błyskawicznego ładowania bazy danych i wag modelu.

---

## 📁 Struktura Projektu
```text
DataScience/
│
├── app/
│   └── streamlit_app.py      # Główny kod aplikacji webowej (Klasa MelbourneHousingApp)
│
├── data/
│   └── Melbourne_housing.csv # Oczyszczona baza danych nieruchomości
│
└── models/
    └── melbourne_model.pkl   # Zapisany, wytrenowany model Machine Learning (Joblib)
```

---

## 🛠️ Instrukcja uruchomienia lokalnego

Wykonaj poniższe kroki w terminalu swojego środowiska (np. PyCharm Terminal), aby uruchomić projekt na własnym komputerze.

### 1. Pobranie i przygotowanie projektu
Upewnij się, że znajdujesz się w głównym katalogu projektu:
```bash
cd C:\Users\Asus\PycharmProjects\DataScience
```

### 2. Instalacja wymaganych bibliotek
Zainstaluj pakiety niezbędne do uruchomienia interfejsu oraz obsługi danych i wykresów:
```bash
pip install streamlit pandas joblib plotly scikit-learn
```

### 3. Uruchomienie serwera Streamlit
Przejdź do folderu `app/` i uruchom dedykowany serwer aplikacji:
```bash
cd app
streamlit run streamlit_app.py
```
*Po wpisaniu tej komendy aplikacja automatycznie otworzy się w Twojej przeglądarce pod adresem `http://localhost:8501`.*

---

## 🚀 Instrukcja dla deweloperów (Git Workflow)

Poniżej znajduje się ściągawka z komend Gita użytych do synchronizacji repozytorium lokalnego z serwerem GitHub.

### Pierwsza konfiguracja i wymuszenie synchronizacji (w przypadku braku spójności historii):
```bash
# Pobranie zmian ze zdalnego repozytorium zezwalając na niepowiązane historie
git pull origin master --allow-unrelated-histories

# Wymuszenie wypchnięcia (Overriding zdalnego mastera wersją lokalną)
git push origin master --force
```

### Codzienna praca z projektem (Zapisywanie stabilnych wersji):
```bash
# 1. Sprawdzenie statusu zmodyfikowanych plików
git status

# 2. Dodanie wszystkich zmian do poczekalni (Staging area)
git add .

# 3. Zatwierdzenie zmian czytelnym komunikatem
git commit -m "Add production-ready Melbourne Housing ML Dashboard with Plotly and dynamic filters"

# 4. Wypchnięcie kodu na GitHub
git push origin master
```

---
*Projekt rozwijany w celach demonstracyjnych i portfolio z zakresu Data Science & MLOps.*
