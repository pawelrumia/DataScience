import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Union, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn import svm, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as skm

# 0) Wczytanie danych
# Najpierw spróbuj pobrać "dataset" z titanic.py; jeśli się nie uda, fallback do CSV.
try:
    from titanic import dataset  # w pliku titanic.py powinna istnieć zmienna "dataset"
except Exception:
    dataset = pd.read_csv(
        "C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\titanic_train.csv")  # podmień ścieżkę, jeśli potrzebujesz

# 1) Oddzielenie kolumny docelowej 'Survived' od reszty
dataset_x = dataset.drop(columns=["Survived"])
dataset_y = dataset["Survived"]

# 2) Usunięcie mniej przydatnych kolumn
cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
dataset_x = dataset_x.drop(columns=cols_to_drop, errors="ignore")

# 3) Uzupełnianie braków
# - Age: średnia
if "Age" in dataset_x.columns:
    age_mean = dataset_x["Age"].mean(skipna=True)
    dataset_x["Age"] = dataset_x["Age"].fillna(age_mean)

# - Embarked: najczęstsza wartość (mode)
if "Embarked" in dataset_x.columns:
    embarked_mode = dataset_x["Embarked"].mode(dropna=True).iloc[0]
    dataset_x["Embarked"] = dataset_x["Embarked"].fillna(embarked_mode)

# 4) LabelEncoder dla 'Sex' i 'Embarked'
for col in ["Sex", "Embarked"]:
    if col in dataset_x.columns:
        le = LabelEncoder()
        dataset_x[col] = le.fit_transform(dataset_x[col].astype(str))

# 5) Podział na zbiory treningowy i testowy (80% : 20%) z random_state=42 (bez stratify)
train_x, test_x, train_y, test_y = train_test_split(
    dataset_x,
    dataset_y,
    train_size=0.8,
    random_state=42
)

# 6) Standaryzacja cech: fit na train, transform na train i test (z zachowaniem indeksów/kolumn)
scaler = StandardScaler()
train_index, test_index = train_x.index, test_x.index
cols = train_x.columns

train_x = pd.DataFrame(scaler.fit_transform(train_x), index=train_index, columns=cols)
test_x = pd.DataFrame(scaler.transform(test_x), index=test_index, columns=cols)

# 7) Wykres słupkowy liczności klas w train_y
counts = train_y.value_counts().reindex([0, 1]).fillna(0).astype(int)
plt.figure()
plt.bar([0, 1], counts.values)
plt.xticks([0, 1], ['not survived', 'survived'])
plt.suptitle('Survivors')
plt.show()

# 8) Modele i trening
# - SGDClassifier (hinge, l2)
model_sgd = SGDClassifier(loss="hinge", penalty="l2")
model_sgd.fit(train_x, train_y)

# - SVC (domyślnie)
model_svc = svm.SVC()
model_svc.fit(train_x, train_y)

# - DecisionTreeClassifier (domyślnie)
model_tree = DecisionTreeClassifier()
model_tree.fit(train_x, train_y)


# 9) Definicja klasy Metric i słownik metryk
@dataclass
class Metric:
    model: Union[SGDClassifier, svm.SVC, tree.DecisionTreeClassifier]
    accuracy: Any
    f1: Any
    mean_abs_error: Any
    r2: Any


metrics = {key: None for key in ["SGDClassifier", "SVC", "DecisionTreeClassifier"]}

# 10) Predykcje na zbiorze testowym
y_pred_sgd = model_sgd.predict(test_x)
y_pred_svc = model_svc.predict(test_x)
y_pred_tree = model_tree.predict(test_x)

# 11) Wypełnienie słownika 'metrics' obiektami Metric
metrics["SGDClassifier"] = Metric(
    model=model_sgd,
    accuracy=skm.accuracy_score(test_y, y_pred_sgd),
    f1=skm.f1_score(test_y, y_pred_sgd, zero_division=0),
    mean_abs_error=skm.mean_absolute_error(test_y, y_pred_sgd),
    r2=skm.r2_score(test_y, y_pred_sgd),
)

metrics["SVC"] = Metric(
    model=model_svc,
    accuracy=skm.accuracy_score(test_y, y_pred_svc),
    f1=skm.f1_score(test_y, y_pred_svc, zero_division=0),
    mean_abs_error=skm.mean_absolute_error(test_y, y_pred_svc),
    r2=skm.r2_score(test_y, y_pred_svc),
)

metrics["DecisionTreeClassifier"] = Metric(
    model=model_tree,
    accuracy=skm.accuracy_score(test_y, y_pred_tree),
    f1=skm.f1_score(test_y, y_pred_tree, zero_division=0),
    mean_abs_error=skm.mean_absolute_error(test_y, y_pred_tree),
    r2=skm.r2_score(test_y, y_pred_tree),
)

# (opcjonalnie) podgląd wyników
for name, m in metrics.items():
    print(f"{name}: "
          f"accuracy={m.accuracy:.4f}, "
          f"f1={m.f1:.4f}, "
          f"MAE={m.mean_abs_error:.4f}, "
          f"R2={m.r2:.4f}")

# dodatkowy AI
# 6) Standaryzacja cech (Dokończenie)
scaler = StandardScaler()

# Dopasowanie (fit) i transformacja na zbiorze treningowym
train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns, index=train_x.index)

# Tylko transformacja na zbiorze testowym (używając średniej i odchylenia z train!)
test_x_scaled = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns, index=test_x.index)

# 7) Przykładowe trenowanie modelu (np. Drzewo Decyzyjne)
model = DecisionTreeClassifier(random_state=42)
model.fit(train_x_scaled, train_y)

# 8) Predykcja i ocena
predictions = model.predict(test_x_scaled)
print("Dokładność modelu:", skm.accuracy_score(test_y, predictions))
