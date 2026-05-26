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
import mlflow
import mlflow.pytorch


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
            nn.Sigmoid()
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
        file_path = f"../benchmark_results/pytorch_{model_name.lower().replace(' ', '_').replace('-', '_')}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"💾 Metryki modelu {model_name} zapisane w pliku JSON: {file_path}")

    def run_benchmark(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        mlflow.set_tracking_uri("file:../mlruns")
        mlflow.set_experiment("Census_Income_Benchmark")

        with mlflow.start_run(run_name="PyTorch_Natywny_MLP"):
            model = CensusMLP(input_dim=X_train.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Inicjalizacja writera TensorBoard dla PyTorcha
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir="../logs/pytorch_mlp")

            # Podkręcone parametry pętli uczącej
            epochs = 100
            batch_size = 32
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            print("🧠 [PyTorch] Rozpoczynanie niskopoziomowej pętli uczenia...")
            start_time = time.time()

            for epoch in range(epochs):
                model.train()
                permutation = torch.randperm(X_train.size()[0])
                epoch_loss = 0.0

                for i in range(0, X_train.size()[0], batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x, batch_y = X_train[indices], y_train[indices]

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                #Po każdej epoce jawnie wysyłamy błąd do TensorBoarda
                current_epoch_loss = epoch_loss / (X_train.size()[0] / batch_size)
                writer.add_scalar("Loss/train", current_epoch_loss, epoch)

                # Wysyłanie błędu epoki do wykresu liniowego w MLflow
                mlflow.log_metric("epoch_loss", epoch_loss / (X_train.size()[0] / batch_size), step=epoch)

            duration_pt = time.time() - start_time
            writer.close()  # Zamykamy writer po treningu
            print("🔬 [PyTorch] Ewaluacja sieci na zbiorze testowym...")

            model.eval()
            with torch.no_grad():
                raw_preds = model(X_test)
                preds = raw_preds.round().numpy()
                y_true = y_test.numpy()

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)

            # Rejestracja końcowych wskaźników jakości i struktury modelu w MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("training_time_seconds", duration_pt)
            mlflow.pytorch.log_model(model, "pytorch_model")

            # Tradycyjny zapis do pliku JSON
            self.save_results("Neural Network MLP", acc, prec, rec, duration_pt)


if __name__ == "__main__":
    bench = PyTorchBenchmark()
    bench.run_benchmark()
