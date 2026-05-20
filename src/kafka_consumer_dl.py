import json
import joblib
import numpy as np
from confluent_kafka import Consumer, KafkaError
from tensorflow.keras.models import load_model


class MelbourneHousingConsumerDL:
    def __init__(self, bootstrap_servers='localhost:9092', group_id='dl_prediction_group',
                 topic='melbourne_housing_events'):
        # Konfiguracja Konsumenta Kafki
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'  # Czytaj od początku kanału w przypadku nowego konsumenta
        }
        self.consumer = Consumer(self.config)
        self.topic = topic
        self.feature_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        self.load_ml_assets()

    def load_ml_assets(self):
        """Ładowanie wytrenowanych wcześniej zasobów Deep Learning."""
        print("🧠 Ładowanie modelu TensorFlow (Keras) oraz Skalera...")
        self.nn_model = load_model('../models/melbourne_nn_model.keras')
        self.scaler = joblib.load('../models/nn_scaler.pkl')
        print("✅ Zasoby ML załadowane pomyślnie!")

    def start_consuming(self):
        """Nieskończona pętla nasłuchująca wiadomości rynkowe w czasie rzeczywistym."""
        self.consumer.subscribe([self.topic])
        print(f"📥 Konsument TensorFlow uruchomiony. Nasłuchiwanie na kanale '{self.topic}'...\n")

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"❌ Błąd Kafki: {msg.error()}")
                        break

                # Dekodowanie JSON-a z Kafki
                data = json.loads(msg.value().decode('utf-8'))

                # Przygotowanie wektora pod TensorFlow
                raw_features = [[
                    data['Rooms'], data['Distance'], data['Week'],
                    data['CPI'], data['LastCPI'], data['Distance2']
                ]]

                # Normalizacja danych wejściowych
                scaled_features = self.scaler.transform(raw_features)

                # Predykcja za pomocą Sieci Neuronowej TensorFlow
                prediction = self.nn_model.predict(scaled_features, verbose=0)[0][0]

                # Wynik przetwarzania strumieniowego
                print(
                    f"🏠 NOWA OFERTA: Dzielnica: {data['Suburb']} | Adres: {data['Address']} | Pokoje: {int(data['Rooms'])}")
                print(f"🔮 PREDYKCJA TENSORFLOW LIVE: ${prediction:,.2f}")
                print("-" * 60)

        except KeyboardInterrupt:
            print("🔒 Zatrzymywanie konsumenta...")
        finally:
            self.consumer.close()


if __name__ == "__main__":
    kc = MelbourneHousingConsumerDL()
    kc.start_consuming()
