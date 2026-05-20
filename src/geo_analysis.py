import pandas as pd
from sklearn.cluster import KMeans


class MelbourneGeoAnalyzer:
    def __init__(self, data_path="../data/Melbourne_housing.csv"):
        self.data_path = data_path

    def segment_suburbs(self, n_clusters=4):
        """Grupuje dzielnice na podstawie odległości i średniej ceny za pokój."""
        df = pd.read_csv(self.data_path)
        df = df[df['Date'] != 'Date']

        for col in ['Price', 'Distance', 'Rooms']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Price', 'Distance', 'Suburb', 'Rooms'])
        df['Price_per_Room'] = df['Price'] / df['Rooms']

        # Agregacja statystyk dla każdej dzielnicy
        suburb_stats = df.groupby('Suburb').agg({
            'Distance': 'mean',
            'Price_per_Room': 'mean',
            'Price': 'count'
        }).rename(columns={'Price': 'Offer_Count'}).dropna()

        # Trening algorytmu K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        suburb_stats['Market_Cluster'] = kmeans.fit_predict(suburb_stats[['Distance', 'Price_per_Room']])

        # Mapowanie klastrów na czytelne nazwy stref
        suburb_stats['Market_Cluster'] = suburb_stats['Market_Cluster'].map({
            0: "Strefa A (Premium / Blisko)",
            1: "Strefa B (Średnia / Przedmieścia)",
            2: "Strefa C (Ekonomiczna / Daleko)",
            3: "Strefa D (Okazyjna)"
        })

        return suburb_stats.reset_index()


if __name__ == "__main__":
    analyzer = MelbourneGeoAnalyzer()
    res = analyzer.segment_suburbs()
    print(res.head())
