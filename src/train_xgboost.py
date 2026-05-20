import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def train_xgb_model():
    print("🚀 Ładowanie danych do treningu XGBoost...")
    # Wczytanie danych z poziomu katalogu src/
    df = pd.read_csv("../data/Melbourne_housing.csv")

    # Oczyszczanie danych (czyszczenie nagłówków tekstowych z wnętrza pliku)
    df = df[df['Date'] != 'Date']

    # Definicja cech numerycznych i zmiennej celu
    feature_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
    target_col = 'Price'

    # Konwersja typów danych na numeryczne i usunięcie braków danych
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=feature_cols + [target_col])
    print(f"📊 Liczba oczyszczonych rekordów do analizy: {len(df)}")

    X = df[feature_cols]
    y = df[target_col]

    # Podział na zbiór treningowy i testowy (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Rozpoczynanie treningu modelu XGBoost (Gradient Boosting)...")
    # Konfiguracja zaawansowanego modelu z parametrami optymalizacyjnymi
    xgb_model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1  # Wykorzystaj wszystkie rdzenie procesora
    )

    # Trening modelu
    xgb_model.fit(X_train, y_train)

    # Ewaluacja za pomocą metryki RMSE
    predictions = xgb_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"✅ Sukces! Model XGBoost został przeszkolony.")
    print(f"📉 Błąd walidacji RMSE: ${rmse:,.2f}")

    # Zapisanie gotowego modelu do katalogu models/
    model_save_path = "../models/melbourne_xgb_model.pkl"
    joblib.dump(xgb_model, model_save_path)
    print(f"💾 Model został bezpiecznie zapisany w: {model_save_path}\n")


if __name__ == "__main__":
    train_xgb_model()
