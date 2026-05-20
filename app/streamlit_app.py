import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import sys


# --- BEZPIECZNE FUNKCJE CACHE (IZOLACJA SILNIKÓW) ---

@st.cache_resource
def get_spark_session_and_model(model_features):
    """Izolowane uruchamianie sesji Apache Spark i błyskawiczny trening."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor

    # Konfiguracja środowiska pod Windows
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    try:
        spark = SparkSession.builder \
            .appName("StreamlitSparkBackend") \
            .config("spark.sql.shuffle.partitions", "2") \
            .master("local[*]") \
            .getOrCreate()

        spark_df = spark.read.csv('../data/Melbourne_housing.csv', header=True, inferSchema=True)
        spark_df = spark_df.filter(spark_df.Date != "Date")

        numeric_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2', 'Price']
        for c in numeric_cols:
            spark_df = spark_df.withColumn(c, col(c).cast("double"))
        spark_df = spark_df.dropna(subset=numeric_cols)

        assembler = VectorAssembler(inputCols=model_features, outputCol="features")
        train_data = assembler.transform(spark_df).select("features", "Price")

        rf = RandomForestRegressor(featuresCol="features", labelCol="Price", numTrees=20, seed=42)
        model = rf.fit(train_data)
        return spark, model, True
    except Exception as e:
        return None, None, False


@st.cache_resource
def get_xgb_model():
    """Niezależne ładowanie modelu XGBoost."""
    try:
        model = joblib.load('../models/melbourne_xgb_model.pkl')
        return model, True
    except Exception:
        return None, False


@st.cache_data
def get_raw_data():
    """Niezależne ładowanie danych rynkowych przez Pandas."""
    df = pd.read_csv('../data/Melbourne_housing.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed', dayfirst=True)
    df = df.dropna(subset=['Date'])
    return df


class MelbourneHousingApp:
    def __init__(self):
        self.model_features = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        # Wywołanie niezależnych modułów
        self.spark, self.spark_model, self.spark_active = get_spark_session_and_model(self.model_features)
        self.xgb_model, self.xgb_active = get_xgb_model()
        self.df = get_raw_data()

    def render_ui(self):
        st.set_page_config(page_title="Melbourne Housing ML", layout="wide")
        st.title('🏢 Melbourne Housing ML & Market Dashboard')

        # Panel boczny
        st.sidebar.header("⚙️ Ustawienia i Filtry")

        model_choice = st.sidebar.selectbox(
            "Wybierz model ML:",
            ["XGBoost Regressor", "Apache Spark (MLlib)"]
        )

        if model_choice == "Apache Spark (MLlib)":
            if self.spark_active:
                st.sidebar.success("🟢 Serwer Apache Spark: AKTYWNY")
            else:
                st.sidebar.error("🔴 Serwer Apache Spark: BŁĄD INICJALIZACJI")
            st.sidebar.info("📉 RMSE Modelu: ~$497,040.95")
        else:
            if self.xgb_active:
                st.sidebar.success("🟢 Model XGBoost: ZAŁADOWANY")
            else:
                st.sidebar.error("🔴 Model XGBoost: BRAK PLIKU .PKL")
            st.sidebar.info("📉 RMSE Modelu: ~$351,684.20 (Zoptymalizowany)")

        st.sidebar.markdown("---")

        selected_suburb = st.sidebar.multiselect('Wybierz dzielnicę (Suburb):',
                                                 options=sorted(self.df['Suburb'].unique()), default=[])
        selected_type = st.sidebar.multiselect('Typ nieruchomości:', options=self.df['Type'].unique(),
                                               default=self.df['Type'].unique())

        filtered_df = self.df[self.df['Type'].isin(selected_type)]
        if selected_suburb:
            filtered_df = filtered_df[filtered_df['Suburb'].isin(selected_suburb)]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔮 Kalkulator Ceny (Prognoza ML)")
            rooms = st.slider('Liczba pokoi (Rooms)', int(self.df['Rooms'].min()), int(self.df['Rooms'].max()), 3)
            distance = st.slider('Odległość od centrum (Distance)', float(self.df['Distance'].min()),
                                 float(self.df['Distance'].max()), 10.0)
            week = st.slider('Tydzień roku (Week)', int(self.df['Week'].min()), int(self.df['Week'].max()), 25)
            cpi = st.slider('Bieżący indeks CPI', float(self.df['CPI'].min()), float(self.df['CPI'].max()), 108.5)

            last_cpi = cpi - 0.4
            distance2 = distance * 1.2

            if st.button('🚀 Przewiduj cenę nieruchomości', use_container_width=True):
                self.predict_price(model_choice, rooms, distance, week, cpi, last_cpi, distance2)

        with col2:
            st.subheader("📊 Analiza trendów rynkowych")
            fig = px.scatter(filtered_df, x="Distance", y="Price", color="Rooms", hover_data=['Suburb', 'Address'],
                             title="Stosunek ceny do odległości od centrum")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("📈 Zaawansowana Analiza Statystyczna")
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.subheader("📊 Rozkład cen nieruchomości (Histogram)")
            fig_hist = px.histogram(filtered_df, x="Price", nbins=50, title="Gęstość cenowa ofert w Melbourne",
                                    color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_hist, use_container_width=True)

        with vis_col2:
            st.subheader("🌡️ Macierz Korelacji Cech (Heatmap)")
            numerical_cols = ['Price', 'Rooms', 'Distance', 'Week', 'CPI', 'Distance2']
            numerical_df = filtered_df[numerical_cols].corr()
            fig_heat = px.imshow(numerical_df, text_auto=".2f", aspect="auto",
                                 title="Wpływ poszczególnych zmiennych na cenę", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Podgląd przefiltrowanych nieruchomości")
        st.dataframe(filtered_df.sort_values(by='Price'), use_container_width=True)

    def predict_price(self, model_choice, rooms, distance, week, cpi, last_cpi, distance2):
        input_data_dict = {
            'Rooms': float(rooms), 'Distance': float(distance), 'Week': float(week),
            'CPI': float(cpi), 'LastCPI': float(last_cpi), 'Distance2': float(distance2)
        }

        if model_choice == "Apache Spark (MLlib)":
            if not self.spark_active:
                st.error("Silnik Apache Spark nie został zainicjalizowany poprawnie.")
                return
            from pyspark.ml.feature import VectorAssembler

            spark_input = self.spark.createDataFrame([input_data_dict])
            assembler = VectorAssembler(inputCols=self.model_features, outputCol="features")
            spark_input = assembler.transform(spark_input)

            prediction_df = self.spark_model.transform(spark_input)
            prediction = prediction_df.select("prediction").collect()
            st.success(f'### 💰 Wycena (Apache Spark MLlib): ${prediction[0][0]:,.2f}')

        else:
            if not self.xgb_active:
                st.error("Model XGBoost jest niedostępny.")
                return
            pandas_input = pd.DataFrame([input_data_dict])[self.model_features]
            prediction = self.xgb_model.predict(pandas_input)
            st.success(f'### 💰 Wycena (XGBoost Regressor): ${prediction[0]:,.2f}')


if __name__ == '__main__':
    app = MelbourneHousingApp()
    app.render_ui()
