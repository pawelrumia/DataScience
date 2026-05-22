import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class CustomerMLPipeline:
    def __init__(self, data_path="../data/Customers_Data.csv"):
        self.data_path = data_path
        self.target_col = "CzyRodzic"

        # Definiujemy cechy wejściowe (Features) na podstawie struktury Twoich danych
        self.categorical_features = ["Wyksztalcenie", "MieszkaZ"]
        self.numeric_features = [
            "Dochod", "LDniOdOstWizyty", "WydatkiWino", "WydatkiOwoce",
            "WydatkiMieso", "WydatkiRyby", "WydatkiSlodycze", "IloscZakupowInternet",
            "IloscZakupowKatalog", "IloscZakupowWSklepie", "LOdwiedzinStrony",
            "OdKiedyKlient", "Wiek", "Spent"
        ]

    def load_and_preprocess(self):
        print("📥 Ładowanie danych klientów...")
        df = pd.read_csv(self.data_path)

        # Oczyszczanie: konwersja kolumn numerycznych na poprawne typy
        for col in self.numeric_features + [self.target_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Pozbywamy się ewentualnych wierszy z brakami danych (NaN)
        df = df.dropna(subset=self.numeric_features + [self.target_col])

        X = df[self.categorical_features + self.numeric_features]
        y = df[self.target_col].astype(int)

        # Podział na zbiór treningowy (80%) i testowy (20%)
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def build_pipeline(self):
        print("🛠️ Budowanie zautomatyzowanego potoku przetwarzania (Transformer Pipeline)...")

        # Przetwarzanie cech numerycznych: Standaryzacja (średnia=0, wariancja=1)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Przetwarzanie cech kategorycznych: One-Hot Encoding (zamiana tekstu na kolumny 0/1)
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Składamy transformatory kolumn w jeden blok (ColumnTransformer)
        # Zabezpiecza to przed wyciekiem danych (Data Leakage) podczas walidacji
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Łączymy przetwarzanie danych z ostatecznym klasyfikatorem Scikit-Learn
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])

        return full_pipeline

    def run_full_ml_cycle(self):
        # 1. Przygotowanie zestawów danych
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        print(f"📊 Dane przygotowane. Trening: {X_train.shape[0]} wierszy, Test: {X_test.shape[0]} wierszy.")

        # 2. Inicjalizacja Pipeline
        model_pipeline = self.build_pipeline()

        # 3. Nauka modelu (Trening)
        print("🤖 Uruchamianie algorytmu Random Forest na danych wejściowych...")
        model_pipeline.fit(X_train, y_train)

        # 4. Predykcja i Ocena Skuteczności
        predictions = model_pipeline.predict(X_test)

        print("\n🎯 --- WYNIKI KLASYFIKACJI SCIKIT-LEARN ---")
        print(f"Dokładność modelu (Accuracy Score): {accuracy_score(y_test, predictions):.2%}")

        print("\n📋 Raport klasyfikacji:")
        print(classification_report(y_test, predictions, target_names=["Nie jest rodzicem", "Jest rodzicem"]))

        print("\n🧩 Macierz konfuzji (Confusion Matrix):")
        print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    pipeline = CustomerMLPipeline()
    pipeline.run_full_ml_cycle()
