import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, min, max, count, round, when, desc, asc, concat, lit, length, corr


class CustomerSparkAnalyzer:
    def __init__(self, data_path="../data/Customers_Data.csv"):
        # Automatyczna naprawa ścieżek Pythona dla systemu Windows
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        # Inicjalizacja rozproszonej sesji Apache Spark
        self.spark = SparkSession.builder \
            .appName("AdvancedCustomerAnalytics") \
            .master("local[*]") \
            .getOrCreate()
        self.data_path = data_path

    def run_all_analytics(self):
        # Wczytanie pliku CSV z automatycznym wykrywaniem typów (Schema Inference)
        df_raw = self.spark.read.csv(self.data_path, header=True, inferSchema=True)

        # Oczyszczanie bazy z błędnych wierszy nagłówkowych i rzutowanie typów
        df = df_raw.filter(df_raw.Wyksztalcenie != "Wyksztalcenie")
        numeric_cols = ["Dochod", "Wiek", "Spent", "WydatkiWino", "WydatkiMieso", "LDzieci", "CzyRodzic",
                        "LOdwiedzinStrony", "IloscZakupowInternet"]
        for c in numeric_cols:
            df = df.withColumn(c, col(c).cast("double"))
        df = df.dropna(subset=["Dochod", "Wiek", "CzyRodzic"])

        print("🚀 Rozpoczynam zaawansowany proces analityczny w Apache Spark (25 przykładów)...")

        # ==============================================================================
        # SEKCJA 1: PODSTAWOWA EKSPLORACJA I FILTROWANIE (PRZYKŁADY 1-5)
        # ==============================================================================
        print("\n📌 [1] Podgląd struktury tabeli (printSchema):")
        df.printSchema()

        print("\n📌 [2] Filtrowanie: Klienci z dochodem powyżej 60 000, sortowani od najwyższego:")
        df.filter(col("Dochod") > 60000).select("Wyksztalcenie", "Dochod", "Wiek").sort(desc("Dochod")).show(3)

        print("\n📌 [3] Multi-filtrowanie: Młodzi klienci (poniżej 45 lat) będący rodzicami:")
        df.filter((col("Wiek") < 45) & (col("CzyRodzic") == 1)).select("Wiek", "Dochod", "MieszkaZ").show(3)

        print("\n📌 [4] Usunięcie duplikatów i pobranie unikalnych kombinacji wykształcenia i statusu mieszkaniowego:")
        df.select("Wyksztalcenie", "MieszkaZ").distinct().show(3)

        print("\n📌 [5] Tworzenie nowej kolumny: Łączenie tekstu z wartością statyczną (concat, lit):")
        df.withColumn("Profil_Edukacyjny", concat(lit("Status: "), col("Wyksztalcenie"))).select(
            "Profil_Edukacyjny").show(3)

        # ==============================================================================
        # SEKCJA 2: PROSTE I ZŁOŻONE AGREGACJE BIZNESOWE (PRZYKŁADY 6-10)
        # ==============================================================================
        print("\n📌 [6] Ogólne podsumowanie statystyczne rynku (agregacja globalna):")
        df.agg(
            count("Dochod").alias("Suma_Klientow"),
            round(avg("Dochod"), 2).alias("Sredni_Dochod"),
            round(avg("Wiek"), 2).alias("Sredni_Wiek")
        ).show()

        print("\n📌 [7] Grupowanie (GroupBy): Średnie wydatki na wino i mięso w zależności od wykształcenia:")
        df.groupby("Wyksztalcenie").agg(
            round(avg("WydatkiWino"), 2).alias("Srednio_Wino"),
            round(avg("WydatkiMieso"), 2).alias("Srednio_Mieso")
        ).show()

        print("\n📌 [8] Segmentacja rynkowa: Suma wszystkich wydatków (Spent) per status relacji (MieszkaZ):")
        df.groupby("MieszkaZ").sum("Spent").withColumnRenamed("sum(Spent)", "Total_Spent").sort(
            desc("Total_Spent")).show()

        print("\n📌 [9] Ekstrema rynkowe: Minimalny i maksymalny wiek klienta w zależności od tego, czy ma dzieci:")
        df.groupby("CzyRodzic").agg(min("Wiek").alias("Min_Wiek"), max("Wiek").alias("Max_Wiek")).show()

        print("\n📌 [10] Filtrowanie po agregacji: Grupy wykształcenia, gdzie średni dochód przekracza 50 000:")
        df.groupby("Wyksztalcenie").agg(avg("Dochod").alias("Avg_Inc")).filter(col("Avg_Inc") > 50000).show()

        # ==============================================================================
        # SEKCJA 3: LOGIKA WARUNKOWA I TRANSFORMACJE CECH (PRZYKŁADY 11-15)
        # ==============================================================================
        print("\n📌 [11] Instrukcja warunkowa (when/otherwise): Podział dochodów na klasy ekonomiczne:")
        df.withColumn("Klasa_Zarobkowa",
                      when(col("Dochod") < 40000, "Niska")
                      .when((col("Dochod") >= 40000) & (col("Dochod") <= 70000), "Srednia")
                      .otherwise("Wysoka")
                      ).select("Dochod", "Klasa_Zarobkowa").show(3)

        print(
            "\n📌 [12] Obliczanie wskaźników pochodnych: Procentowy udział wydatków na wino w całkowitym budżecie (Spent):")
        df.withColumn("Procent_Wino", round((col("WydatkiWino") / col("Spent")) * 100, 2)).select("Spent",
                                                                                                  "WydatkiWino",
                                                                                                  "Procent_Wino").show(
            3)

        print(
            "\n📌 [13] Operacje na tekście (length): Znajdowanie wierszy, gdzie nazwa statusu mieszkaniowego jest dłuższa niż 6 znaków:")
        df.filter(length(col("MieszkaZ")) > 6).select("MieszkaZ").distinct().show(3)

        print("\n📌 [14] Rzutowanie typów i operacje matematyczne w locie: Zaokrąglanie zmiennych zmiennoprzecinkowych:")
        df.withColumn("Wiek_Dekada", round(col("Wiek") / 10, 0) * 10).select("Wiek", "Wiek_Dekada").show(3)

        print("\n📌 [15] Złożone filtrowanie tekstowe: Wykluczanie klientów o statusie 'Alone':")
        df.filter(~col("MieszkaZ").isin(["Alone"])).select("MieszkaZ", "Dochod").show(3)

        # ==============================================================================
        # SEKCJA 4: ANALIZA STATYSTYCZNA I KORELACJE (PRZYKŁADY 16-20)
        # ==============================================================================
        print("\n📌 [16] Weryfikacja korelacji liniowej (corr): Czy wyższy dochód oznacza większe całkowite wydatki?")
        korelacja_dochod_wydatki = df.stat.corr("Dochod", "Spent")
        print(f"Współczynnik korelacji Pearsona: {korelacja_dochod_wydatki:.4f}")

        print("\n📌 [17] Korelacja behawioralna: Zależność między ilością odwiedzin strony a zakupami w internecie:")
        corr_web_clicks = df.stat.corr("LOdwiedzinStrony", "IloscZakupowInternet")
        print(f"Współczynnik korelacji (Wizyty vs Zakupy): {corr_web_clicks:.4f}")

        print("\n📌 [18] Tabela krzyżowa (crosstab): Powiązanie poziomu edukacji ze statusem rodzica:")
        df.stat.crosstab("Wyksztalcenie", "CzyRodzic").show()

        print("\n📌 [19] Obliczanie kwantyli rynkowych (approxQuantile): Mediana oraz 90. percentyl dochodów klientów:")
        quantiles = df.approxQuantile("Dochod", [0.5, 0.9], 0.01)
        print(f"Mediana zarobków: ${quantiles[0]:,.2f} | 90. percentyl: ${quantiles[1]:,.2f}")

        print("\n📌 [20] Częstotliwość występowania (freqItems): Najpopularniejsze statusy mieszkaniowe:")
        df.stat.freqItems(["MieszkaZ"], 0.3).show()

        # ==============================================================================
        # SEKCJA 5: ZAAWANSOWANE MANIPULACJE I AGREGACJE MULTI-LEVEL (PRZYKŁADY 21-25)
        # ==============================================================================
        print("\n📌 [21] Multi-GroupBy: Analiza nawyków zakupowych per wykształcenie ORAZ status rodzica:")
        df.groupby("Wyksztalcenie", "CzyRodzic").agg(
            round(avg("IloscZakupowInternet"), 1).alias("Srednia_Zakupow_Net")).show()

        print("\n📌 [22] Sortowanie wielopoziomowe: Najpierw po wykształceniu (rosnąco), potem po dochodzie (malejąco):")
        df.select("Wyksztalcenie", "Dochod").sort(asc("Wyksztalcenie"), desc("Dochod")).show(3)

        print(
            "\n📌 [23] Zliczanie warunkowe wewnątrz agregacji: Ile osób w danej grupie edukacyjnej ma więcej niż 2 dzieci:")
        df.groupby("Wyksztalcenie").agg(
            count(when(col("LDzieci") > 1, True)).alias("Liczba_Rodzin_Wielodzietnych")
        ).show()

        print("\n📌 [24] Tworzenie flag binarnych na podstawie progów statystycznych (Zamożność):")
        df.withColumn("Is_VIP", col("Dochod") > quantiles[1]).select("Dochod", "Is_VIP").filter(
            col("Is_VIP") == True).show(3)

        print("\n📌 [25] Próbkowanie losowe zbioru danych (Sample) - przydatne do testów statystycznych:")
        df_sample = df.sample(withReplacement=False, fraction=0.05, seed=42)
        print(f"Pobrana losowa próbka 5% danych zawiera: {df_sample.count()} rekordów.")

        # Zamknięcie wirtualnego klastra Sparka
        self.spark.stop()
        print("\n🔒 Sesja Apache Spark została bezpiecznie zamknięta.")


if __name__ == "__main__":
    analyzer = CustomerSparkAnalyzer()
    analyzer.run_all_analytics()
