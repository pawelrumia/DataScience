import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class BankLoanMLPipeline:
    def __init__(self, data_path="../data/Bank_Loan.csv"):
        self.data_path = data_path
        # Wybieramy wszystkie istotne cechy klienta banku
        self.feature_cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family',
                             'CCAvg', 'Education', 'Mortgage', 'Securities Account',
                             'CD Account', 'Online', 'CreditCard']
        self.target_col = 'Personal Loan'

    def prepare_data(self):
        print("📊 Wczytywanie i przygotowywanie danych bankowych do modelowania...")
        df = pd.read_csv(self.data_path)

        # Konwersja typów danych i usunięcie ewentualnych braków
        for col in self.feature_cols + [self.target_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=self.feature_cols + [self.target_col])

        X = df[self.feature_cols]
        y = df[self.target_col]

        # Podział danych na zbiór treningowy (80%) i testowy (20%) z zachowaniem proporcji klas
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Obliczamy wagę dla klasy pozytywnej, ponieważ pożyczki bierze mniejszość klientów
        # Zapobiega to sytuacji, w której model uczy się tylko ignorować ofertę pożyczki
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print("🤖 Inicjalizacja i trening klasyfikatora XGBoost...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_weight,  # Balansowanie klas
            random_state=42,
            n_jobs=-1
        )

        # Trening modelu
        model.fit(X_train, y_train)

        # Predykcja na zbiorze testowym
        predictions = model.predict(X_test)

        print("\n🎯 --- WYNIKI EWALUACJI MODELU BANKOWEGO ---")
        print(f"Dokładność ogólna (Accuracy): {accuracy_score(y_test, predictions):.2%}")

        print("\n📋 Szczegółowy raport klasyfikacji (Precision, Recall, F1-Score):")
        print(classification_report(y_test, predictions, target_names=["Odrzucił pożyczkę", "Zaakceptował pożyczkę"]))

        print("\n🧩 Macierz konfuzji (Confusion Matrix):")
        print(confusion_matrix(y_test, predictions))

        # Zapisanie gotowego modelu oraz listy cech do ponownego użycia
        joblib.dump(model, "../models/bank_loan_xgb_model.pkl")
        print("\n💾 Model klasyfikacji bankowej został bezpiecznie zapisany w `models/bank_loan_xgb_model.pkl`.")


if __name__ == "__main__":
    pipeline = BankLoanMLPipeline()
    pipeline.train_and_evaluate()
