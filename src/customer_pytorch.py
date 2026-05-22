import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Przygotowanie danych pod PyTorch
df = pd.read_csv("../data/Customers_Data.csv")
df = df[df['Wyksztalcenie'] != 'Wyksztalcenie'].dropna()
X = StandardScaler().fit_transform(df[["Dochod", "Wiek", "Spent", "WydatkiWino", "WydatkiMieso"]].values)
y = df["CzyRodzic"].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja na Tensory PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)


# Definicja architektury sieci w PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = Net()
criterion = nn.BCELoss()  # Binary Cross Entropy dla klasyfikacji binarnej
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Jawna pętla treningowa PyTorch
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()  # Wsteczna propagacja błędu
    optimizer.step()

# Wyłączenie gradientów i przejście w tryb ewaluacji
model.eval()
with torch.no_grad():
    # Pobranie surowych prawdopodobieństw z sieci i zaokrąglenie do 0 lub 1
    raw_outputs = model(X_test_t)
    preds = raw_outputs.round().numpy()

    # Konwersja rzeczywistych etykiet testowych na tablicę Numpy
    y_true = y_test_t.numpy()

    # Import metryk do szczegółowego raportu
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("\n🎯 --- ZAAWANSOWANE STATYSTYKI MODELU PYTORCH ---")
    print(f"Ogólna dokładność (Accuracy Score): {accuracy_score(y_true, preds):.2%}")

    print("\n📋 Raport klasyfikacji sieci neuronowej:")
    print(classification_report(y_true, preds, target_names=["Nie jest rodzicem", "Jest rodzicem"]))

    print("\n🧩 Macierz konfuzji (Confusion Matrix):")
    print(confusion_matrix(y_true, preds))
