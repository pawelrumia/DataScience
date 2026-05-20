import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class MelbourneDLModel:
    def __init__(self, data_path="../data/Melbourne_housing.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        self.target_col = 'Price'

    def prepare_data(self):
        print("📊 Wczytywanie i czyszczenie danych dla sieci neuronowej...")
        df = pd.read_csv(self.data_path)
        df = df[df['Date'] != 'Date']

        # Konwersja kolumn na typ numeryczny
        for col in self.feature_cols + [self.target_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=self.feature_cols + [self.target_col])

        X = df[self.feature_cols].values
        y = df[self.target_col].values

        # Podział na zbiór treningowy i testowy (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standaryzacja cech (Skalowanie do średniej = 0 i wariancji = 1)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_and_train_nn(self, epochs=40, batch_size=32):
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Budowanie architektury sieci Deep Learning (MLP)
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.1),  # Zapobieganie przeuczeniu (overfitting)
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Jeden neuron na wyjściu dla wartości ciągłej (cena)
        ])

        # Kompilacja modelu z optymalizatorem Adam
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
            loss='mse',
            metrics=['mae']
        )

        print("🧠 Rozpoczynanie treningu głębokiej sieci neuronowej Keras...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

        # Ewaluacja i obliczenie błędu RMSE na zbiorze testowym
        predictions = model.predict(X_test).flatten()
        rmse = root_mean_squared_error(y_test, predictions)
        print(f"\n✅ Sukces! Trening sieci neuronowej zakończony.")
        print(f"📉 Keras NN - Błąd walidacji RMSE: ${rmse:,.2f}")

        # Tworzenie katalogu na modele jeśli nie istnieje i zapis wag
        os.makedirs("../models", exist_ok=True)
        model.save("../models/melbourne_nn_model.keras")  # Zapis w nowym formacie .keras
        joblib.dump(self.scaler, "../models/nn_scaler.pkl")
        print("💾 Model sieci oraz skaler zostały pomyślnie zapisane w folderze `models/`.")


if __name__ == "__main__":
    nn_pipeline = MelbourneDLModel()
    nn_pipeline.build_and_train_nn(epochs=40, batch_size=32)
