# 1. Wybieramy oficjalny obraz Pythona jako bazę systemu
FROM python:3.12-slim

# 2. Instalujemy Javę (OpenJDK-17), która jest niezbędna do działania Apache Spark
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Ustawiamy zmienną środowiskową dla Javy (Spark automatycznie ją wykryje)
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# 4. Ustalamy katalog roboczy wewnątrz kontenera kontenera
WORKDIR /workspace

# 5. Kopiujemy plik z listą bibliotek i instalujemy je
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Kopiujemy całą zawartość naszego projektu do kontenera
COPY . .

# 7. Informujemy, że aplikacja będzie działać na porcie 8501
EXPOSE 8501

# Zmieniamy katalog roboczy bezpośrednio na folder app przed startem
WORKDIR /workspace/app

# Uruchamiamy aplikację tak, jak robiłeś to lokalnie w terminalu
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

