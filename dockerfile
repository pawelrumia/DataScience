FROM python:3.12

# Instalacja Javy (pod Sparka) oraz bibliotek kompilacji C pod klienta Kafki (librdkafka)
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    librdkafka-dev build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy pandas scikit-learn
RUN pip install --no-cache-dir xgboost pyspark==3.5.0 streamlit plotly joblib tensorflow confluent-kafka apache-airflow torch


COPY . .

EXPOSE 8501

WORKDIR /workspace/app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
