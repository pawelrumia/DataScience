import pandas as pd
from sklearn.ensemble import IsolationForest


class MarketAnomalyDetector:
    def __init__(self, data_path="../data/Melbourne_housing.csv"):
        self.df = pd.read_csv(data_path)
        self.df = self.df[self.df['Date'] != 'Date']
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df['Rooms'] = pd.to_numeric(self.df['Rooms'], errors='coerce')
        self.df = self.df.dropna(subset=['Price', 'Rooms'])

    def detect_outliers(self):
        """Wykrywa nienaturalne oferty cenowe za pomocą Isolation Forest."""
        detector = IsolationForest(contamination=0.03, random_state=42)  # Szukamy 3% skrajnych anomalii

        # Analiza relacji ceny do liczby pokoi
        data_to_analyze = self.df[['Price', 'Rooms']]
        self.df['Anomaly_Score'] = detector.fit_predict(data_to_analyze)

        # -1 oznacza anomalię, 1 oznacza normalną ofertę
        anomalies = self.df[self.df['Anomaly_Score'] == -1]
        print(f"🚨 Wykryto {len(anomalies)} podejrzanych ofert na rynku nieruchomości.")
        return anomalies


if __name__ == "__main__":
    detector = MarketAnomalyDetector()
    outliers = detector.detect_outliers()
    print(outliers[['Suburb', 'Address', 'Rooms', 'Price']].head())
