import time
import json
import random
from datetime import datetime
from confluent_kafka import Producer


class ECommerceClickstreamProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='ecommerce_clickstream'):
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})
        self.topic = topic
        self.categories = ['Electronics', 'Fashion', 'Home', 'Books', 'Sports']
        self.event_types = ['view_item', 'view_item', 'add_to_cart', 'purchase']  # Więcej wyświetleń niż zakupów

    @staticmethod
    def _delivery_report(err, msg):
        if err is not None:
            print(f"❌ Błąd: {err}")
        else:
            print(f"📡 Event wysłany -> {msg.topic()} [Offset: {msg.offset()}]")

    def start_simulation(self, duration_seconds=60):
        print(f"🚀 Uruchamiam symulator ruchu E-commerce na topicu: {self.topic}...")
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Generowanie losowego zdarzenia użytkownika
            payload = {
                'session_id': f"sess_{random.randint(10000, 99999)}",
                'user_id': f"user_{random.randint(100, 999)}",
                'event_type': random.choice(self.event_types),
                'product_id': f"prod_{random.randint(1, 50)}",
                'category': random.choice(self.categories),
                'price': round(random.uniform(5.0, 1500.0), 2),
                'device': random.choice(['mobile', 'desktop', 'tablet']),
                'country': random.choice(['PL', 'US', 'DE', 'UK', 'FR']),
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Wysłanie do Kafki
            self.producer.produce(
                topic=self.topic,
                value=json.dumps(payload).encode('utf-8'),
                callback=self._delivery_report
            )
            self.producer.poll(0)

            # Losowy odstęp między kliknięciami (np. od 0.1 do 0.5 sekundy)
            time.sleep(random.uniform(0.1, 0.5))

        self.producer.flush()
        print("🔒 Symulacja zakończona.")


if __name__ == "__main__":
    simulator = ECommerceClickstreamProducer()
    # Uruchamiamy symulację na 60 sekund testowo
    simulator.start_simulation(duration_seconds=60)
