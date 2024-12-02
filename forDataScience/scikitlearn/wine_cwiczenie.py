import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

wine = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\wine.csv", delimiter=';')
wine_nas = wine.isnull().sum()
plt.figure(figsize=(10, 6))
sns.histplot(wine["quality"], bins=7, color='red', kde=False)
plt.title("Rozklad jakosci wina")
plt.xlabel("Jakosc")
plt.ylabel("Ilosc win")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(wine.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Macierz korelacji cech wine")
plt.savefig("macierz korelacji.png", dpi=150)
plt.show()

# wine["quality"] = (wine["quality"] >= 6).astype(int)

X = wine.drop(columns=["quality"])
y = wine['quality']

label_encoder = LabelEncoder()
y = pd.Series(label_encoder.fit_transform(y), name='quality')  # <-- KLUCZOWE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_value_counts = y_train.value_counts(normalize=True)
test_value_counts  = y_test.value_counts(normalize=True)

scaler = StandardScaler()
num_features = wine.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features.remove('quality')

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"model": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

predictions = {name: None for name in models}


# Funkcja oceny modeli
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return accuracy, precision, recall, f1


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    metrics["model"].append(model_name)
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)

result_df = pd.DataFrame(metrics)

fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))

if len(predictions) == 1:
    axes = [axes]

for i, (model_name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Macierz pomyłek - {model_name}")
    axes[i].set_xlabel("Predykcja")
    axes[i].set_ylabel("Rzeczywistość")

plt.tight_layout()
plt.savefig('macierz_pomylek.png', dpi=150)
plt.show()


def plot_roc_curve(model, X_test, Y_test, label):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{label}, AUC= {auc_score:.2f}')

plt.figure(figsize=(8,6))
for model_name, model in models.items():
    plot_roc_curve(model, X_test, y_test, model_name)

plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Krzywa ROC dla roznych modeli")
plt.savefig('krzywa ROC.png', dpi=150)
plt.legend()
plt.show()
