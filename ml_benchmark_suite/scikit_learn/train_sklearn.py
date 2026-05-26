import os
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib


class SklearnBenchmark:
    def __init__(self, data_path="../data/census_income.csv"):
        self.data_path = data_path
        self.columns = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
            "hours.per.week", "native.country", "income"
        ]

    def prepare_data(self):
        print("📊 [Skikit-Learn] Ładowanie i kodowanie danych Census...")
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
            "framework": "Scikit-Learn",
            "model_name": model_name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "training_time_seconds": round(float(duration), 2)
        }
        file_path = f"../benchmark_results/sklearn_{model_name.lower().replace(' ', '_').replace('-', '_')}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"💾 Metryki modelu {model_name} zapisane w pliku JSON: {file_path}")

    def run_benchmark(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        # WYMUSZENIE ZAPISU DO KATALOGU TEKSTOWEGO MLRUNS
        mlflow.set_tracking_uri("file:../mlruns")
        mlflow.set_experiment("Census_Income_Benchmark")

        # --- MODEL 1: RANDOM FOREST ---
        with mlflow.start_run(run_name="Scikit-Learn_Random_Forest"):
            print("🤖 [Skikit-Learn] Trenowanie modelu Random Forest...")
            start_time = time.time()

            n_est = 300
            max_d = 15
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", max_d)

            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # zapisywanie modeli Random Forest do katalogu `models` w formacie pickle
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "..", "models")
            os.makedirs(models_dir, exist_ok=True)

            joblib.dump(rf, os.path.join(models_dir, "census_rf_model.pkl"))
            print(
                f"💾 Model Random Forest został pomyślnie zapisany w: {os.path.join(models_dir, 'census_rf_model.pkl')}")
            duration_rf = time.time() - start_time

            preds_rf = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, preds_rf)
            prec_rf = precision_score(y_test, preds_rf, zero_division=0)
            rec_rf = recall_score(y_test, preds_rf, zero_division=0)

            # Logowanie metryk i modelu do serwera MLflow
            mlflow.log_metric("accuracy", acc_rf)
            mlflow.log_metric("precision", prec_rf)
            mlflow.log_metric("recall", rec_rf)
            mlflow.log_metric("training_time", duration_rf)
            mlflow.sklearn.log_model(rf, "random_forest_model")

            # Tradycyjny zapis do pliku JSON
            self.save_results("Random Forest", acc_rf, prec_rf, rec_rf, duration_rf)

        # --- MODEL 2: K-NEAREST NEIGHBORS (KNN) ---
        with mlflow.start_run(run_name="Scikit-Learn_K-Nearest_Neighbors"):
            print("🤖 [Skikit-Learn] Trenowanie modelu K-Nearest Neighbors...")
            start_time = time.time()

            k_neighbors = 7
            mlflow.log_param("n_neighbors", k_neighbors)

            knn = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1)
            knn.fit(X_train, y_train)
            duration_knn = time.time() - start_time

            preds_knn = knn.predict(X_test)
            acc_knn = accuracy_score(y_test, preds_knn)
            prec_knn = precision_score(y_test, preds_knn, zero_division=0)
            rec_knn = recall_score(y_test, preds_knn, zero_division=0)

            # Logowanie do serwera MLflow
            mlflow.log_metric("accuracy", acc_knn)
            mlflow.log_metric("precision", prec_knn)
            mlflow.log_metric("recall", rec_knn)
            mlflow.log_metric("training_time", duration_knn)
            mlflow.sklearn.log_model(knn, "knn_model")

            # Tradycyjny zapis do pliku JSON
            self.save_results("K-Nearest Neighbors", acc_knn, prec_knn, rec_knn, duration_knn)


if __name__ == "__main__":
    bench = SklearnBenchmark()
    bench.run_benchmark()
