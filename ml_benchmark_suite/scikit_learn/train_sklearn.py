import os
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SklearnBenchmark:
    def __init__(self, data_path="../data/census_income.csv"):
        self.data_path = data_path
        # Zmieniamy podkreślenia na kropki, aby idealnie pasowało do Twojego pliku CSV:
        self.columns = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
            "hours.per.week", "native.country", "income"
        ]

    def prepare_data(self):
        print("📊 [Skikit-Learn] Ładowanie i kodowanie danych Census...")
        # Dodajemy header=0, aby Pandas wiedział, że pierwszy wiersz to nazwy kolumn i należy go pominąć
        df = pd.read_csv(self.data_path, names=self.columns, sep=",", engine="python", header=0)

        # Pancerne czyszczenie braków danych: usuwamy wiersze ze znakami zapytania i spacjami wokół nich
        df = df.map(lambda x: pd.NA if str(x).strip() == '?' else x).dropna()

        # Zamiana targetu na format binarny
        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

        # Kodowanie kolumn tekstowych na numeryczne
        categorical_cols = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex",
                            "native.country"]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df.drop(columns=["income"])
        y = df["income"]

        # Skalowanie cech numerycznych
        X_scaled = StandardScaler().fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def save_results(self, model_name, accuracy, precision, recall, duration):
        """Zapisuje metryki do ujednoliconego pliku JSON w folderze wyników."""
        os.makedirs("../benchmark_results", exist_ok=True)
        results = {
            "framework": "Scikit-Learn",
            "model_name": model_name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "training_time_seconds": round(float(duration), 2)
        }
        file_path = f"../benchmark_results/sklearn_{model_name.lower().replace(' ', '_')}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"💾 Metryki modelu {model_name} zapisane w: {file_path}")

    def run_benchmark(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        # --- MODEL 1: RANDOM FOREST ---
        print("🤖 [Skikit-Learn] Trenowanie modelu Random Forest...")
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        duration_rf = time.time() - start_time

        preds_rf = rf.predict(X_test)
        self.save_results(
            model_name="Random Forest",
            accuracy=accuracy_score(y_test, preds_rf),
            precision=precision_score(y_test, preds_rf, zero_division=0),
            recall=recall_score(y_test, preds_rf, zero_division=0),
            duration=duration_rf
        )

        # --- MODEL 2: K-NEAREST NEIGHBORS (KNN) ---
        print("🤖 [Skikit-Learn] Trenowanie modelu K-Nearest Neighbors...")
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        knn.fit(X_train, y_train)
        duration_knn = time.time() - start_time

        preds_knn = knn.predict(X_test)
        self.save_results(
            model_name="K-Nearest Neighbors",
            accuracy=accuracy_score(y_test, preds_knn),
            precision=precision_score(y_test, preds_knn, zero_division=0),
            recall=recall_score(y_test, preds_knn, zero_division=0),
            duration=duration_knn
        )


if __name__ == "__main__":
    bench = SklearnBenchmark()
    bench.run_benchmark()
