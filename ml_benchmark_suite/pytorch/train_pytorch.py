import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score


# 1. Definicja architektury sieci w PyTorch (odpowiednik modelu Keras)
class CensusMLP(nn.Module):
    def __init__(self, input_dim):
        super(CensusMLP, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Wyjście 0-1 do klasyfikacji binarnej
        )

    def forward(self, x):
        return self.pipeline(x)


class PyTorchBenchmark:
    def __init__(self, data_path="../data/census_income.csv"):
        self.data_path = data_path
        self.columns = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
            "hours.per.week", "native.country", "income"
        ]

    def prepare_data(self):
        print("📊 [PyTorch] Ładowanie i kodowanie danych Census...")
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
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Konwersja na Tensory PyTorch (wymóg konieczny)
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)

        return X_train_t, X_test_t, y_train_t, y_test_t

    def save_results(self, model_name, accuracy, precision, recall, duration):
        os.makedirs("../benchmark_results", exist_ok=True)
        results = {
            "framework": "PyTorch",
            "model_name": model_name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "training_time_seconds": round(float(duration), 2)
        }
        file_path = f"../benchmark_results/pytorch_{model_name.lower().replace(' ', '_')}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"💾 Metryki modelu {model_name} zapisane w: {file_path}")

    def run_benchmark(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Inicjalizacja modelu, funkcji straty i optymalizatora Adam
        model = CensusMLP(input_dim=X_train.shape[1])
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("🧠 [PyTorch] Rozpoczynanie niskopoziomowej pętli uczenia (50 epok)...")
        start_time = time.time()

        # Jawna pętla treningowa PyTorch
        epochs = 200
        batch_size = 32

        for epoch in range(epochs):
            model.train()
            # Prosty mechanizm dzielenia na paczki (batching) w pamięci
            permutation = torch.randperm(X_train.size()[0])
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                optimizer.zero_grad()  # Wyczyszczenie starych gradientów
                outputs = model(batch_x)  # Krok w przód (Forward pass)
                loss = criterion(outputs, batch_y)  # Obliczenie błędu
                loss.backward()  # Wsteczna propagacja (Backward pass)
                optimizer.step()  # Aktualizacja wag wag sieci

        duration_pt = time.time() - start_time
        print("🔬 [PyTorch] Ewaluacja sieci na zbiorze testowym...")

        model.eval()
        with torch.no_grad():
            raw_preds = model(X_test)
            preds = raw_preds.round().numpy()  # Zaokrąglamy prawdopodobieństwa do 0 lub 1
            y_true = y_test.numpy()

        self.save_results(
            model_name="Neural Network MLP",
            accuracy=accuracy_score(y_true, preds),
            precision=precision_score(y_true, preds, zero_division=0),
            recall=recall_score(y_true, preds, zero_division=0),
            duration=duration_pt
        )


if __name__ == "__main__":
    bench = PyTorchBenchmark()
    bench.run_benchmark()
