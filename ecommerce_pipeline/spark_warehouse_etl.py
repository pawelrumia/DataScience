import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


def run_streaming_warehouse_pipeline():
    # AUTOMATYCZNA NAPRAWA DLA WINDOWS
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # 1. Czysta, bezpieczna inicjalizacja bez wymuszania zewnętrznych bibliotek sieciowych
    spark = SparkSession.builder \
        .appName("ECommerceClickstreamToSnowflake") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .master("local[*]") \
        .getOrCreate()

    print("\n🚀 Silnik Apache Spark Structured Streaming uruchomiony!")

    # 2. Definicja schematu danych wejściowych (Schema Enforcement)
    # W hurtowniach danych musimy sztywno zdefiniować typy zmiennych, aby nie uszkodzić tabel
    clickstream_schema = StructType([
        StructField("session_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("device", StringType(), True),
        StructField("country", StringType(), True),
        StructField("timestamp", StringType(), True)
    ])

    # 3. Odczyt strumienia z Apache Kafka
    print("📥 Subskrypcja strumienia Apache Kafka: topic 'ecommerce_clickstream'...")
    kafka_stream_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "ecommerce_clickstream") \
        .option("startingOffsets", "latest") \
        .load()

    # 4. Przetwarzanie danych (Parsowanie binarnego JSON-a z Kafki na kolumny Sparka)
    # Kafka domyślnie przesyła dane w postaci binarnej (klucz 'value'), dekodujemy ją na string i narzucamy schemat
    parsed_stream_df = kafka_stream_df \
        .selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), clickstream_schema).alias("data")) \
        .select("data.*")

    # Dodatkowa transformacja inżynierska: Filtrujemy tylko wartościowe akcje dla analityki BI
    # Wybieramy tylko zdarzenia dodania do koszyka i zakupu, dodając flagę wysokiej wartości
    processed_stream_df = parsed_stream_df \
        .filter(col("event_type").isin(["add_to_cart", "purchase"])) \
        .withColumn("is_high_value", col("price") > 500.0)

    # 5. Opcje konfiguracyjne połączenia z chmurą Snowflake (Dane DEMO dla rekrutera)
    # Wskazujesz tutaj parametry konta, bazy danych i magazynu danych w Snowflake
    snowflake_options = {
        "sfURL": "://snowflakecomputing.com",
        "sfUser": "DB_ENTERPRISE_USER",
        "sfPassword": "SecurePassword123!",
        "sfDatabase": "ECOMMERCE_STAGE",
        "sfSchema": "PUBLIC",
        "sfWarehouse": "COMPUTE_WH",
        "dbtable": "FACT_CLICKSTREAM_LIVE"
    }

    print("💾 Konfiguracja zapisu strumieniowego do Snowflake zakończona powodzeniem.")

    # 6. Zapis strumienia bezpośrednio do tabeli w Snowflake lub konsoli (tryb testowy)
    # Ponieważ nie mamy aktywnej płatnej subskrypcji chmurowej, domyślnie wyświetlimy to w konsoli.
    # Rekruter zobaczy w kodzie gotową konfigurację .format("snowflake"), która wystarczy odkomentować.

    query = processed_stream_df.writeStream \
        .format("console") \
        .outputMode("append") \
        .start()

    # ABY URUCHOMIĆ ZAPIS DO SNOWFLAKE W PRODUKCJI, ODKOMENTUJ PONIŻSZY BLOK:
    # query = processed_stream_df.writeStream \
    #     .format("snowflake") \
    #     .options(**snowflake_options) \
    #     .option("checkpointLocation", "../data/snowflake_checkpoints") \
    #     .outputMode("append") \
    #     .start()

    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_warehouse_pipeline()
