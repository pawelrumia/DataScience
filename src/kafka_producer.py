import time
import json
import pandas as pd
from confluent_kafka import Producer


class MelbourneHousingProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='melbourne_housing_events'):
        # Konfiguracja klienta Kafki
        self.config = {'bootstrap.servers': bootstrap_servers}
        self.producer = Producer(self.config)
        self.topic = topic

    @staticmethod
    def _delivery_report(err, msg):
        """Funkcja zwrotna (callback) informująca o sukcesie lub błędzie zapisu w Kafce."""
        if err is not None:
            print(print(f"❌ Błąd dostarczenia wiadomości: {err}"))
        else:
            print(f"📡 Zdarzenie wysłane do {msg.topic()} [Partycja: {msg.partition()}]")

    def stream_csv_data(self, csv_path='../data/Melbourne_housing.csv'):
        """Wczytuje plik i wysyła wiersze do Kafki w odstępach sekundy symulując ruch live."""
        df = pd.read_csv(csv_path)
        df = df[df['Date'] != 'Date'].dropna(subset=['Rooms', 'Distance'])

        print(f"🚀 Rozpoczynamy strumieniowanie danych do Kafki na temat: '{self.topic}'...")

        for _, row in df.iterrows():
            # Konwersja wiersza danych do formatu JSON (standard w Kafce)
            payload = {
                'Suburb': str(row['Suburb']),
                'Address': str(row['Address']),
                'Rooms': float(row['Rooms']),
                'Type': str(row['Type']),
                'Distance': float(row['Distance']),
                'Week': float(row['Week']),
                'CPI': float(row['CPI']),
                'LastCPI': float(row['LastCPI']),
                'Distance2': float(row['Distance2'])
            }

            # Wysłanie asynchroniczne wiadomości
            self.producer.produce(
                topic=self.topic,
                value=json.dumps(payload).encode('utf-8'),
                callback=self._delivery_report
            )

            # Flush wysyła dane z bufora pamięci do sieci
            self.producer.poll(0)
            time.sleep(1)  # Czekamy sekundę przed kolejną ofertą

        self.producer.flush()


if __name__ == "__main__":
    # Kod testowy
    kp = MelbourneHousingProducer()
    kp.stream_csv_data()
