import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, round


class BankLoanSparkAnalyzer:
    def __init__(self, data_path="../data/Bank_Loan.csv"):
        # Automatyczna naprawa ścieżek Pythona dla systemu Windows
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        # 1. Inicjalizacja sesji Apache Spark
        self.spark = SparkSession.builder \
            .appName("BankLoanDataAnalysis") \
            .master("local[*]") \
            .getOrCreate()

        self.data_path = data_path

    def load_and_explore(self):
        print("\n📥 Wczytywanie danych bankowych przez Apache Spark...")
        # Wczytujemy plik CSV wymuszając automatyczne wykrywanie typów (Schema Inference)
        df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)

        print("📋 Struktura tabeli (Schema) w hurtowni danych:")
        df.printSchema()
        return df

    def run_business_analytics(self):
        df = self.load_and_explore()

        print("\n📊 --- ANALIZA BIZNESOWA SPARK SQL ---")

        # 1. Profil klienta: Jaki jest średni dochód (Income) i wiek osób, które biorą pożyczkę (Personal Loan = 1)?
        print("\n💡 Profil finansowy klientów w zależności od posiadania pożyczki:")
        df.groupby("Personal Loan") \
            .agg(
            round(avg("Income"), 2).alias("Sredni_Dochod_k$"),
            round(avg("Age"), 2).alias("Sredni_Wiek"),
            round(avg("CCAvg"), 2).alias("Sredni_Obrót_Karty"),
            count("ID").alias("Liczba_Klientów")
        ).show()

        # 2. Wpływ wykształcenia (Education) na decyzję o pożyczce
        # Mapujemy numery na czytelne poziomy: 1: Undergrad, 2: Graduate, 3: Advanced
        print("💡 Analiza wpływu wykształcenia na akceptację pożyczki:")
        df.withColumn("Poziom_Edukacji",
                      when(col("Education") == 1, "Undergrad")
                      .when(col("Education") == 2, "Graduate")
                      .otherwise("Advanced")
                      ).groupby("Poziom_Edukacji", "Personal Loan") \
            .count() \
            .show()

        # 3. Analiza ryzyka: Czy osoby z pożyczką hipoteczną (Mortgage) częściej biorą kredyt osobisty?
        print("💡 Statystyki hipoteczne (Mortgage) dla pożyczkobiorców:")
        df.filter(col("Mortgage") > 0) \
            .groupby("Personal Loan") \
            .agg(round(avg("Mortgage"), 2).alias("Srednia_Hipoteka_k$")) \
            .show()

        # Zamknięcie sesji Sparka
        self.spark.stop()
        print("🔒 Sesja Apache Spark zamknięta.")


if __name__ == "__main__":
    analyzer = BankLoanSparkAnalyzer()
    analyzer.run_business_analytics()
