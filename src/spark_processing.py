import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def run_spark_pipeline():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    # 1. Inicjalizacja sesji Apache Spark (Big Data Engine)
    spark = SparkSession.builder \
        .appName("MelbourneHousingSparkPipeline") \
        .config("spark.sql.shuffle.partitions", "2") \
        .master("local[*]") \
        .getOrCreate()

    print("\n🚀 Pomyślnie uruchomiono rozproszoną sesję Apache Spark!")

    # 2. Ładowanie danych przez silnik Sparka
    # Ścieżka wychodzi z katalogu src/ do data/
    df = spark.read.csv("../data/Melbourne_housing.csv", header=True, inferSchema=True)

    # 3. Oczyszczanie danych (ETL): Odfiltrowanie uszkodzonych wierszy nagłówkowych
    df_cleaned = df.filter(df.Date != "Date")

    # Rzutowanie typów kolumn na numeryczne (Double)
    numeric_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2', 'Price']
    for c in numeric_cols:
        df_cleaned = df_cleaned.withColumn(c, col(c).cast("double"))

    # Usunięcie brakujących danych (NaN)
    df_cleaned = df_cleaned.dropna(subset=numeric_cols)
    print(f"📊 Przetworzono rekordy w Sparku. Liczba czystych wierszy: {df_cleaned.count()}")

    # 4. Przygotowanie wektora cech dla algorytmu MLlib
    feature_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    spark_ml_data = assembler.transform(df_cleaned).select("features", "Price")

    # Podział na zbiór treningowy (80%) i testowy (20%)
    train_data, test_data = spark_ml_data.randomSplit([0.8, 0.2], seed=42)

    # 5. Trening modelu Big Data (Random Forest Regressor z Apache Spark MLlib)
    print("🤖 Rozpoczynanie treningu modelu Random Forest na klastrze Sparka...")
    rf = RandomForestRegressor(featuresCol="features", labelCol="Price", numTrees=20, seed=42)
    rf_model = rf.fit(train_data)

    # 6. Prognoza i ewaluacja modelu
    predictions = rf_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print(f"\n✅ Sukces! Model Spark MLlib został przeszkolony.")
    print(f"📉 Błąd walidacji RMSE (Root Mean Squared Error): ${rmse:,.2f}")

    # ... (pod koniec funkcji run_spark_pipeline, po wyliczeniu rmse) ...
    print(f"📉 Błąd walidacji RMSE (Root Mean Squared Error): ${rmse:,.2f}")

    # NOWOŚĆ: Zapisujemy wytrenowany model Sparka do folderu models
    model_save_path = "../models/spark_rf_model"
    rf_model.write().overwrite().save(model_save_path)
    print(f"💾 Model Spark został zapisany w: {model_save_path}")

    # 7. Zamknięcie sesji Sparka
    spark.stop()
    print("🔒 Sesja Apache Spark została bezpiecznie zamknięta.\n")


if __name__ == "__main__":
    run_spark_pipeline()
