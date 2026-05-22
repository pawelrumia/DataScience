import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import mlflow
import mlflow.keras


class TensorflowBenchmark:
    def __init__(self, data_path="../data/census_income.csv"):
        self.data_path = data_path
        self.columns = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
            "hours.per.week", "native.country", "income"
        ]

    def prepare_data(self):
        print("📊 [TensorFlow] Ładowanie i kodowanie danych Census...")
        df = pd.read_csv(self.data_path, names=self.columns, sep=",", engine="python", header=0)
        df = df.map(lambda x: pd.NA if str(x).strip() == '?' else x).dropna()

        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

        categorical_cols = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex",
                            "native.country"]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df.drop(columns=["income"])
        y = df["income"]

        X_scaled = StandardScaler().fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def save_results(self, model_name, accuracy, precision, recall, duration):
        os.makedirs("../benchmark_results", exist_ok=True)
        results = {
            "framework": "TensorFlow / Keras",
            "model_name": model_name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "training_time_seconds": round(float(duration), 2)
        }
        file_path = f"../benchmark_results/tf_{model_name.lower().replace(' ', '_')}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"💾 Metryki modelu {model_name} zapisane w pliku JSON: {file_path}")

    def run_benchmark(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        y_train = y_train.values.astype(np.float32)
        y_test = y_test.values.astype(np.float32)

        # Włączenie pełnego autologowania dla TensorFlow (przechwytuje epoki, loss i architekturę)
        mlflow.set_tracking_uri("file:../mlruns")
        mlflow.set_experiment("Census_Income_Benchmark")
        mlflow.tensorflow.autolog()

        with mlflow.start_run(run_name="TensorFlow_Keras_MLP"):
            print("🤖 [TensorFlow] Budowanie architektury sieci neuronowej...")
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )

            print("🧠 [TensorFlow] Rozpoczynanie uczenia głębokiego (Deep Learning)...")
            start_time = time.time()
            # Zwiększamy liczbę epok dla dłuższego, dokładniejszego treningu
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
            duration_tf = time.time() - start_time

            print("🔬 [TensorFlow] Ewaluacja sieci na zbiorze testowym...")
            loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

            # Manualne zalogowanie czasu do bazy eksperymentu MLflow
            mlflow.log_metric("training_time_seconds", duration_tf)

            # Tradycyjny zapis do pliku JSON
            self.save_results(
                model_name="Neural Network MLP",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                duration=duration_tf
            )

if __name__ == "__main__":
    bench = TensorflowBenchmark()
    bench.run_benchmark()
