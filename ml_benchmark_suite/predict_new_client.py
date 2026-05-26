import os
import joblib
import numpy as np
import pandas as pd

# Wyłączamy zbędne logi startowe TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class CensusMultiPredictor:
    def __init__(self):
        # Dynamicznie znajdujemy główny folder projektu (DataScience) niezależnie od środowiska
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Pancerne ścieżki bezwzględne
        self.rf_path = os.path.join(base_dir, "models", "census_rf_model.pkl")
        self.tf_path = os.path.join(base_dir, "models", "census_tf_model.keras")
        self.scaler_path = os.path.join(base_dir, "models", "census_tf_scaler.pkl")

        # Weryfikacja plików
        for path in [self.rf_path, self.tf_path, self.scaler_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Brak pliku: {path}. Uruchom najpierw pełny retrening turnieju!")

        print("📥 Ładowanie modeli i skalerów z dysku...")
        # Ładowanie modeli
        self.rf_model = joblib.load(self.rf_path)
        self.tf_model = tf.keras.models.load_model(self.tf_path)
        self.scaler = joblib.load(self.scaler_path)

        self.feature_names = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
            "hours.per.week", "native.country"
        ]

    def predict_all(self, client_data):
        print("\n🔮 Analiza profilu nowego obywatela...")
        df_client = pd.DataFrame([client_data])[self.feature_names]

        # --- 1. PREDYKCJA SCIKIT-LEARN (Drzewa nie wymagają skalowania) ---
        rf_pred = self.rf_model.predict(df_client)[0]
        rf_prob = self.rf_model.predict_proba(df_client)[0][1]

        # --- 2. PREDYKCJA TENSORFLOW (Wymaga identycznego skalowania cech!) ---
        client_scaled = self.scaler.transform(df_client)
        tf_prob = self.tf_model.predict(client_scaled, verbose=0)[0][0]
        tf_pred = 1 if tf_prob >= 0.5 else 0

        # --- WYŚWIETLENIE WYNIKÓW BIZNESOWYCH ---
        print("\n🏆 ======================================================= 🏆")
        print("                 PORÓWNANIE WERDYKTÓW MODELI                 ")
        print("🏆 ======================================================= 🏆")

        print(f"🌲 [Scikit-Learn Random Forest]:")
        status_rf = "POWYŻEJ $50k" if rf_pred == 1 else "PONIŻEJ lub RÓWNE $50k"
        print(f"   -> Klasa: {status_rf} (Prawdopodobieństwo klasy >50k: {rf_prob:.2%})")

        print(f"\n🧠 [TensorFlow Deep Learning NN]:")
        status_tf = "POWYŻEJ $50k" if tf_pred == 1 else "PONIŻEJ lub RÓWNE $50k"
        print(f"   -> Klasa: {status_tf} (Prawdopodobieństwo klasy >50k: {tf_prob:.2%})")
        print("=============================================================\n")


if __name__ == "__main__":
    # Testowy profil: 45 lat, wykształcenie wyższe (Bachelors=13), wysokie zyski kapitałowe, 45h pracy/tydz.
    new_client = {
        "age": 45, "workclass": 4, "fnlwgt": 150000, "education": 9, "education.num": 13,
        "marital.status": 2, "occupation": 3, "relationship": 0, "race": 4, "sex": 1,
        "capital.gain": 5000, "capital.loss": 0, "hours.per.week": 45, "native.country": 39
    }

    predictor = CensusMultiPredictor()
    predictor.predict_all(new_client)
