import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ==============================================================================
# PRZYGOTOWANIE DANYCH POD ZAAWANSOWANE MECHANIZMY PYTORCH
# ==============================================================================
class CensusDataset(Dataset):
    """
    [PRZYKŁAD 1: Custom Dataset & DataLoader]
    Produkcyjny sposób przekazywania danych w PyTorch. Zamiast operować na wielkich
    macierzach w RAM, pakujemy dane w obiekt Dataset, co pozwala na automatyczne
    dzielenie na paczki (batches), tasowanie (shuffling) i wielowątkowe ładowanie.
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_census_data():
    path = "../data/census_income.csv"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country", "income"
    ]
    df = pd.read_csv(path, names=columns, sep=",", engine="python", header=0)
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


# ==============================================================================
# ARCHITEKTURY SIECI NEURONOWYCH W PYTORCH (METODY I ERY)
# ==============================================================================

class LogisticRegressionPyTorch(nn.Module):
    """
    [PRZYKŁAD 2: Regresja Logistyczna jako Sieć Neuronowa]
    Najprostsza możliwa struktura. Jedna warstwa liniowa z aktywacją Sigmoid.
    Pokazuje matematyczne podstawy uczenia maszynowego zaimplementowane w PyTorch.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class DeepWideMLP(nn.Module):
    """
    [PRZYKŁAD 3: Głęboka i Szeroka Sieć z Normalizacją i Dropoutem]
    Zaawansowana architektura z technikami stabilizacyjnymi:
    - BatchNorm1d: Normalizuje sygnał między warstwami (przyspiesza naukę).
    - Dropout: Losowo wyłącza neurony (zapobiega przeuczeniu / overfittingowi).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = torch.relu(self.layer3(x))
        return torch.sigmoid(self.out(x))


class ResidualBlockMLP(nn.Module):
    """
    [PRZYKŁAD 4: Sieć ResNet / Połączenia Skrócone (Skip Connections)]
    Architektura inspirowana najnowocześniejszymi sieciami wizyjnymi.
    Sygnał wejściowy omija warstwę transformacji i jest dodawany bezpośrednio do wyjścia
    (x + residual). Zapobiega to zanikaniu gradientu w bardzo głębokich sieciach.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, 32)
        # Blok rezydualny (wejście i wyjście mają ten sam wymiar: 32)
        self.res_layer1 = nn.Linear(32, 32)
        self.res_layer2 = nn.Linear(32, 32)
        self.linear_out = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linear_in(x))

        # Zapamiętujemy tożsamość sygnału wejściowego do bloku
        identity = x

        residual = torch.relu(self.res_layer1(x))
        residual = self.res_layer2(residual)

        # MATEMATYCZNE POŁĄCZENIE REZYDUALNE: Dodajemy sygnał początkowy do przetworzonego
        x = torch.relu(residual + identity)

        return torch.sigmoid(self.linear_out(x))


# ==============================================================================
# SPERSONALIZOWANE FUNKCJE STRATY I SILNIKI OPTYMALIZACJI
# ==============================================================================

class FocalLoss(nn.Module):
    """
    [PRZYKŁAD 5: Custom Loss Function (Funkcja Straty Focal Loss)]
    PyTorch pozwala na pisanie własnych funkcji matematycznych do liczenia błędu.
    Focal Loss to zaawansowana modyfikacja błędu Cross-Entropy. Zmusza ona sieć
    do skupienia się na "trudnych" i rzadkich wierszach, ignorując przykłady łatwe.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ==============================================================================
# PĘTLE TRENINGOWE I STRATEGIE NAUKI
# ==============================================================================

def train_and_evaluate_model(model_name, model_instance, train_loader, X_test_t, y_test, criterion,
                             optimizer_type="Adam"):
    """
    [PRZYKŁAD 6: Elastyczna Pętla Treningowa z Weight Decay (Regularyzacja L2)]
    """
    if optimizer_type == "Adam":
        # Weight Decay dodaje karę za zbyt duże wagi (regularyzacja L2)
        optimizer = optim.Adam(model_instance.parameters(), lr=0.002, weight_decay=1e-4)
    else:
        # [PRZYKŁAD 7: Optymalizator SGD z pędem (Momentum)]
        # Klasyczny stochastyczny spadek wzdłuż gradientu z fizycznym pędem ułatwiającym wyjście z minimów lokalnych
        optimizer = optim.SGD(model_instance.parameters(), lr=0.01, momentum=0.9)

    print(f"\n🏃 Start treningu dla architektury: {model_name}...")
    start_time = time.time()

    for epoch in range(20):
        model_instance.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_instance(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    duration = time.time() - start_time

    # Ewaluacja
    model_instance.eval()
    with torch.no_grad():
        preds = model_instance(X_test_t).round().numpy()
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, preds)
        print(f"✅ [{model_name}] Ukończono w {duration:.2f}s | Test Accuracy: {acc:.2%}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_census_data()

    # Konwersja na zestawy danych PyTorch DataLoader
    train_dataset = CensusDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    X_test_t = torch.FloatTensor(X_test)

    # --- TURNIEJ METOD WEWNĄTRZ PYTORCH ---

    # Test 1: Prosta Regresja Logistyczna + Klasyczny optymalizator SGD
    model_lr = LogisticRegressionPyTorch(input_dim=X_train.shape[1])
    train_and_evaluate_model(
        "Regresja Logistyczna (SGD)", model_lr, train_loader,
        X_test_t, y_test, nn.BCELoss(), optimizer_type="SGD"
    )

    # Test 2: Głęboki Perceptron (Deep Wide MLP) + Custom Focal Loss + Adam
    model_mlp = DeepWideMLP(input_dim=X_train.shape[1])
    train_and_evaluate_model(
        "Deep Wide MLP (BatchNorm + Dropout + FocalLoss)", model_mlp, train_loader,
        X_test_t, y_test, FocalLoss(), optimizer_type="Adam"
    )

    # Test 3: Zaawansowana sieć z połączeniami rezydualnymi (ResNet MLP) + Adam
    model_res = ResidualBlockMLP(input_dim=X_train.shape[1])
    train_and_evaluate_model(
        "Residual MLP (Skip Connections)", model_res, train_loader,
        X_test_t, y_test, nn.BCELoss(), optimizer_type="Adam"
    )
