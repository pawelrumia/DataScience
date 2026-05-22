# 🏢 Melbourne Housing ML & Market Dashboard

Interaktywny dashboard analityczno-predykcyjny zbudowany w oparciu o bibliotekę **Streamlit** oraz model Machine Learning (Regresja) do szacowania wartości nieruchomości w Melbourne. Projekt został zaimplementowany obiektowo (zgodnie z zasadami czystego kodu) i zawiera zaawansowane filtrowanie oraz interaktywne wykresy rynkowe.

---

## 📊 Funkcje aplikacji
* **Kalkulator Ceny (ML Predykcja):** Prognozowanie wartości domu w czasie rzeczywistym na podstawie cech takich jak liczba pokoi, odległość od centrum, wskaźniki ekonomiczne (CPI, inflacja) oraz czas sprzedaży.
* **Interaktywny Wykres Trendów:** Dynamiczna wizualizacja zależności ceny od odległości z uwzględnieniem liczby pokoi (wykorzystująca bibliotekę Plotly).
* **Zaawansowane Filtrowanie:** Możliwość selekcji danych po dzielnicach (`Suburb`) oraz typie zabudowy (`Type`) bezpośrednio z panelu bocznego.
* **Wydajność (Caching):** Wykorzystanie mechanizmów pamięci podręcznej Streamlit (`@st.cache_resource`) do błyskawicznego ładowania bazy danych i wag modelu.

---

## 📁 Struktura Projektu (Repository Architecture)

Repozytorium zostało ustrukturyzowane w sposób modułowy, oddzielając warstwę prezentacji (Web UI) od niezależnych potoków inżynierii danych, automatyzacji (MLOps) oraz uczenia maszynowego:

```text
DataScience/
│
├── .venv/                      # Lokalne środowisko wirtualne Pythona
│
├── app/                        # WARSTWA PREZENTACJI (WEB INTERFACE)
│   └── streamlit_app.py        # Interaktywny dashboard Streamlit (Wycena ML & K-Means)
│
├── data/                       # REPOZYTORIUM REKORDÓW (DATA HUB)
│   ├── Bank_Loan.csv           # Dane bankowości detalicznej pod Spark SQL
│   ├── Customers_Data.csv      # Dane zachowań konsumenckich pod Scikit-Learn Pipeline
│   └── Melbourne_housing.csv   # Surowa baza cen nieruchomości rynkowych
│
├── ecommerce_pipeline/         # MODUŁ PRZETWARZANIA STRUMIENINGOWEGO (BIG DATA)
│   ├── README_ecommerce.md     # Specyfikacja architektury chmurowej Databricks/Snowflake
│   ├── kafka_clickstream.py    # Producent Kafki symulujący ruch użytkowników live
│   └── spark_warehouse_etl.py  # Spark Structured Streaming przetwarzający logi w locie
│
├── ml_benchmark_suite/         # POTĘŻNY POTOK PORÓWNAWCZY (BENCHMARK SUITE)
│   ├── benchmark_results/      # Zrzuty plików .json zawierające zunifikowane metryki
│   ├── data/
│   │   └── census_income.csv   # Zbiór danych U.S. Census Income (>32k wierszy)
│   ├── pytorch/
│   │   └── train_pytorch.py    # Niskopoziomowy trening sieci neuronowej MLP w PyTorch
│   ├── scikit_learn/
│   │   └── train_sklearn.py    # Klasyczny trening modeli Random Forest & KNN (Sklearn)
│   ├── tensorflow/
│   │   └── train_tf.py         # Trening głębokiej sieci neuronowej MLP w TensorFlow/Keras
│   └── run_benchmark.py        # Skrypt agregujący wyniki JSON w zbiorczą tabelę rankingową
│
├── models/                     # ZAMROŻONE WAGI MODELI (ARTIFACT STORAGE)
│   ├── bank_loan_xgb_model.pkl # Klasyfikator XGBoost dla danych bankowych (98.22% acc)
│   ├── melbourne_nn_model.keras# Sieć neuronowa Keras dla rynku nieruchomości
│   ├── melbourne_xgb_model.pkl # Regresor XGBoost dla rynku nieruchomości
│   └── nn_scaler.pkl           # Skaler dla prawidłowej normalizacji cech sieci neuronowej
│
├── src/                        # SAMODZIELNE MODUŁY I SKRYPTY ANALITYCZNE
│   ├── anomaly_detector.py     # Detekcja anomalii rynkowych przez Isolation Forest
│   ├── bank_loan_xgb.py        # Potok treningowy klasyfikatora bankowego XGBoost
│   ├── bank_spark_analysis.py  # Zaawansowane agregacje biznesowe w chmurze Spark SQL
│   ├── customer_analysis_ml.py # Klasyfikacja zachowań klientów z użyciem Sklearn Pipeline
│   ├── customer_spark_analytics.py # 25 całościowych przykładów transformacji w PySpark SQL
│   ├── dl_nn_model.py          # Potok uczenia sieci Keras z callbackiem TensorBoard
│   ├── geo_analysis.py         # Segmentacja przestrzenna rynku nieruchomości (K-Means)
│   └── ml_orchestrator_dag.py  # DAG Apache Airflow zarządzający automatyzacją retreningu
│
├── logs/                       # Logi treningowe i zdarzenia systemowe pod TensorBoard
├── dockerfile                  # Plik budowania obrazu Docker (Python 3.12 + Java 21)
├── requirements.txt            # Spis produkcyjnych zależności bibliotecznych
└── readme.md                   # Główna dokumentacja techniczna projektu
```


### 🐳 Zarządzanie aplikacją przez Docker

Poniżej znajduje się kompletna sekwencja komend używanych do budowania, uruchamiania i czyszczenia środowiska kontenerowego:

```bash
# 1. Budowanie obrazu kontenera (na podstawie Dockerfile)
docker build -t melbourne-housing-ml .

# 2. Uruchomienie kontenera z przekierowaniem portu na lokalną maszynę
docker run -p 8501:8501 melbourne-housing-ml

# 3. Podgląd aktualnie uruchomionych kontenerów i ich identyfikatorów (Container ID)
docker ps

# 4. Zatrzymanie działającego kontenera (zamiast skrótu Ctrl+C)
docker stop <CONTAINER_ID>

# 5. Usunięcie nieużywanych, wiszących warstw i obrazów w celu zwolnienia miejsca na dysku
docker system prune -f
```

## 🏆 Wielki Turniej Frameworków (ML/DL Benchmark Suite)

Moduł `ml_benchmark_suite/` realizuje niezależny proces masowego uczenia maszynowego (Model Benchmarking) na zbiorze danych *U.S. Census Income* (>32 000 wierszy). System trenuje i unifikuje metryki dla trzech potężnych frameworków rynkowych: **Scikit-Learn, TensorFlow/Keras** oraz **PyTorch**, zapisując wyniki do ujednoliconych plików tekstowych `.json`.

### 🚀 Instrukcja uruchomienia turnieju krok po kroku

Ponieważ środowisko jest w pełni skonteneryzowane, cały turniej możesz wywołać wewnątrz kontenera Docker za pomocą poniższych komend PowerShell/Bash (pamiętaj o flagach `-v`, które montują folder wyników, oraz `--rm`, która chroni dysk przed zapychaniem):

1. **Uruchomienie modułu Scikit-Learn (Trening Random Forest & KNN):**
   ```bash
   docker run --rm -it -v "\${PWD}/ml_benchmark_suite/benchmark_results:/workspace/ml_benchmark_suite/benchmark_results" -w /workspace/ml_benchmark_suite/scikit_learn melbourne-housing-ml python train_sklearn.py
   ```

2. **Uruchomienie modułu TensorFlow / Keras (Trening sieci neuronowej MLP):**
   ```bash
   docker run --rm -it -v "\${PWD}/ml_benchmark_suite/benchmark_results:/workspace/ml_benchmark_suite/benchmark_results" -w /workspace/ml_benchmark_suite/tensorflow melbourne-housing-ml python train_tf.py
   ```

3. **Uruchomienie modułu PyTorch (Niskopoziomowy trening sieci neuronowej MLP):**
   ```bash
   docker run --rm -it -v "\${PWD}/ml_benchmark_suite/benchmark_results:/workspace/ml_benchmark_suite/benchmark_results" -w /workspace/ml_benchmark_suite/pytorch melbourne-housing-ml python train_pytorch.py
   ```

4. **Wyświetlenie zbiorczego raportu (Tabela i ranking końcowy turnieju):**
   ```bash
   docker run --rm -it -v "\${PWD}/ml_benchmark_suite/benchmark_results:/workspace/ml_benchmark_suite/benchmark_results" -w /workspace/ml_benchmark_suite melbourne-housing-ml python run_benchmark.py
   ```
---

# Screenshots

## TensorBoard

![TensorBoard](images/dashboard.png)

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

## 🎛️ Monitorowanie Eksperymentów (MLflow Model Tracking)

Projekt wykorzystuje system **MLflow** do centralnego logowania parametrów, metryk (Accuracy, Precision, Recall) oraz zamrażania wag modeli (Artefaktów) dla wszystkich trzech użytych frameworków [2.1].

### 🛠️ 1. Uruchamianie procesów treningowych w Dockerze

Aby przeprowadzić trening modeli i zapisać dane eksperymentalne bezpośrednio do współdzielonego katalogu `mlruns/`, uruchom poniższe komendy w terminalu PowerShell (będąc w głównym folderze projektu `DataScience`) [1.110, 2.1]:

```powershell
# Przebudowanie obrazu po aktualizacji kodów źródłowych
docker build -t melbourne-housing-ml .

# Trening Scikit-Learn (Random Forest & KNN)
docker run --rm -it -v "\${PWD}/ml_benchmark_suite:/workspace/ml_benchmark_suite" -w /workspace/ml_benchmark_suite/scikit_learn melbourne-housing-ml python train_sklearn.py

# Trening TensorFlow / Keras (Głęboka sieć neuronowa MLP)
docker run --rm -it -v "\${PWD}/ml_benchmark_suite:/workspace/ml_benchmark_suite" -w /workspace/ml_benchmark_suite/tensorflow melbourne-housing-ml python train_tf.py

# Trening PyTorch (Niskopoziomowy trening sieci neuronowej MLP)
docker run --rm -it -v "\${PWD}/ml_benchmark_suite:/workspace/ml_benchmark_suite" -w /workspace/ml_benchmark_suite/pytorch melbourne-housing-ml python train_pytorch.py
```

### 📊 2. Start Panelu Graficznego MLflow UI

Po zakończeniu procesów obliczeniowych baza danych eksperymentów znajduje się fizycznie na Twoim dysku w folderze `ml_benchmark_suite/mlruns/` [2.1]. Aby uruchomić interaktywny pulpit w przeglądarce, wykonaj poniższe kroki w lokalnym terminalu systemu Windows [2.1]:

```bash
# 1. Przejdź do katalogu z modułem benchmarku
cd ml_benchmark_suite

# 2. Uruchom serwer wizualizacji MLflow
mlflow ui
```

*   **Adres panelu:** Otwórz przeglądarkę i przejdź pod adres **`http://localhost:5000`** [2.1].
*   **Porównywanie modeli:** W menu po lewej stronie wybierz eksperyment `Census_Income_Benchmark`, zaznacz checkboxy przy interesujących Cię modelach i kliknij **`Compare`**, aby wygenerować zbiorcze wykresy wydajności [2.1].


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

### 📡 Testowanie architektury strumieniowej (Apache Kafka & TensorFlow Live)

Projekt umożliwia symulację przetwarzania danych w czasie rzeczywistym. Aby uruchomić potok streamingowy:

1. **Uruchom lokalny serwer Apache Kafka:**
   ```bash
   docker run -d --name kafka-server -p 9092:9092 -e KAFKA_NODE_ID=1 -e KAFKA_PROCESS_ROLES=broker,controller -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 apache/kafka:latest
   ```

2. **W osobnym terminalu uruchom Konsumenta (Silnik Predykcji Deep Learning):**
   ```bash
   cd src
   python kafka_consumer_dl.py
   ```

3. **W kolejnym terminalu uruchom Producenta (Generator Strumienia Ofert):**
   ```bash
   cd src
   python kafka_producer.py
   ```
## ☁️ Enterprise Cloud Mapping (Databricks & Snowflake)

Mimo że projekt został uruchomiony i przetestowany lokalnie w kontenerze Docker (przy użyciu zapisu do formatu JSON/Console), architektura kodu została przygotowana pod bezpośrednie wdrożenie w chmurze korporacyjnej:

1. **Zamiast lokalnego Sparka -> Databricks:** Kod zawarty w `spark_warehouse_etl.py` wykorzystuje natywną składnię PySpark Structured Streaming, dzięki czemu może zostać bezpośrednio wklejony jako Job/Notebook w środowisku **Databricks** (np. na AWS lub Azure).
2. **Zamiast lokalnego zapisu -> Snowflake:** Odkomentowanie sekcji `.format("snowflake")` w połączeniu z dostarczonym słownikiem konfiguracji `snowflake_options` umożliwia bezpośrednie przesyłanie mikro-paczek (Micro-batches) do tabeli typu Fact Table wewnątrz chmurowej hurtowni **Snowflake**, implementując architekturę typu Kappa/Lambda.
3. **Zamiast Hadoopa -> Cloud Storage:** W środowisku produkcyjnym punkty kontrolne (`checkpointLocation`) oraz dane tymczasowe są kierowane na AWS S3 lub Azure Blob Storage przy użyciu mechanizmów, które lokalnie emulują biblioteki Apache Hadoop.

### 🧠 Wizualizacja procesu uczenia (TensorBoard)

Projekt umożliwia śledzenie metryk treningowych głębokiej sieci neuronowej Keras w czasie rzeczywistym. 

1. **Generowanie logów podczas treningu:**
   ```bash
   docker run -it -v \${PWD}/logs:/workspace/logs -w /workspace/src melbourne-housing-ml python dl_nn_model.py
   ```

2. **Uruchomienie serwera TensorBoard (Docker):**
   ```bash
   docker run -it -p 6006:6006 -v \${PWD}/logs:/workspace/logs -w /workspace melbourne-housing-ml tensorboard --logdir=logs --bind_all
   ```
   *Panel interaktywnych wykresów sieci neuronowej dostępny jest pod adresem: `http://localhost:6006`.*
lub zwykłe             tensorboard --logdir=logs

---
*Projekt rozwijany w celach demonstracyjnych i portfolio z zakresu Data Science & MLOps.*
