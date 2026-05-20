import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys


class MelbourneHousingApp:
    def __init__(self):
        self.model_features = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        self.load_assets()

    @st.cache_resource
    def load_assets(_self):
        """Uruchomienie Apache Spark i trening modelu bezpośrednio w pamięci RAM."""
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.regression import RandomForestRegressor

        # Zabezpieczenie ścieżek Pythona pod Windows
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        # 1. Start lokalnej sesji Spark
        _self.spark = SparkSession.builder \
            .appName("StreamlitSparkBackend") \
            .config("spark.sql.shuffle.partitions", "2") \
            .master("local[*]") \
            .getOrCreate()

        # 2. Ładowanie i czyszczenie danych w Sparku do treningu
        spark_df = _self.spark.read.csv('../data/Melbourne_housing.csv', header=True, inferSchema=True)
        spark_df = spark_df.filter(spark_df.Date != "Date")

        numeric_cols = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2', 'Price']
        for c in numeric_cols:
            spark_df = spark_df.withColumn(c, col(c).cast("double"))
        spark_df = spark_df.dropna(subset=numeric_cols)

        # 3. Trening błyskawiczny modelu w pamięci
        assembler = VectorAssembler(inputCols=_self.model_features, outputCol="features")
        train_data = assembler.transform(spark_df).select("features", "Price")

        rf = RandomForestRegressor(featuresCol="features", labelCol="Price", numTrees=20, seed=42)
        _self.model = rf.fit(train_data)

        # 4. Tradycyjne ładowanie danych do wykresów przez Pandas
        _self.df = pd.read_csv('../data/Melbourne_housing.csv')
        _self.df['Date'] = pd.to_datetime(_self.df['Date'], errors='coerce', format='mixed', dayfirst=True)
        _self.df = _self.df.dropna(subset=['Date'])

        st.sidebar.success("🤖 Apache Spark MLlib Aktywny (In-Memory)!")

    def render_ui(self):
        st.set_page_config(page_title="Melbourne Housing ML", layout="wide")
        st.title('🏢 Melbourne Housing ML & Market Dashboard')

        st.sidebar.header("🔍 Filtry bazy danych")
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
                self.predict_price(rooms, distance, week, cpi, last_cpi, distance2)

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

    def predict_price(self, rooms, distance, week, cpi, last_cpi, distance2):
        from pyspark.ml.feature import VectorAssembler

        input_data = self.spark.createDataFrame([{
            'Rooms': float(rooms), 'Distance': float(distance), 'Week': float(week),
            'CPI': float(cpi), 'LastCPI': float(last_cpi), 'Distance2': float(distance2)
        }])

        assembler = VectorAssembler(inputCols=self.model_features, outputCol="features")
        spark_input = assembler.transform(input_data)

        prediction_df = self.model.transform(spark_input)
        prediction = prediction_df.select("prediction").collect()

        st.success(f'### 💰 Estymowana wartość (Spark ML): ${prediction[0][0]:,.2f}')


if __name__ == '__main__':
    app = MelbourneHousingApp()
    app.render_ui()
