import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import sys


# --- FUNKCJE CACHE (IZOLACJA MODUŁÓW) ---

@st.cache_resource
def get_spark_session_and_model(model_features):
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
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
    except Exception:
        return None, None, False


@st.cache_resource
def get_xgb_model():
    try:
        return joblib.load('../models/melbourne_xgb_model.pkl'), True
    except Exception:
        return None, False


@st.cache_resource
def get_keras_nn_model():
    from tensorflow.keras.models import load_model
    try:
        return load_model('../models/melbourne_nn_model.keras'), joblib.load('../models/nn_scaler.pkl'), True
    except Exception:
        return None, None, False


@st.cache_data
def get_geo_clusters():
    """Niezależne wywołanie segmentacji K-Means z modułu src."""
    sys.path.append('../src')
    from geo_analysis import MelbourneGeoAnalyzer
    analyzer = MelbourneGeoAnalyzer(data_path='../data/Melbourne_housing.csv')
    return analyzer.segment_suburbs()


@st.cache_data
def get_raw_data():
    df = pd.read_csv('../data/Melbourne_housing.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed', dayfirst=True)
    df = df.dropna(subset=['Date'])
    return df


class MelbourneHousingApp:
    def __init__(self):
        self.model_features = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        self.spark, self.spark_model, self.spark_active = get_spark_session_and_model(self.model_features)
        self.xgb_model, self.xgb_active = get_xgb_model()
        self.nn_model, self.nn_scaler, self.nn_active = get_keras_nn_model()
        self.df = get_raw_data()
        self.geo_df = get_geo_clusters()

    def render_ui(self):
        st.set_page_config(page_title="Melbourne Housing ML", layout="wide")
        st.title('🏢 Melbourne Housing Enterprise Dashboard')

        # Panel boczny
        st.sidebar.header("⚙️ Ustawienia i Filtry")
        model_choice = st.sidebar.selectbox(
            "Wybierz model ML:",
            ["Keras Deep Learning NN", "XGBoost Regressor", "Apache Spark (MLlib)"]
        )

        if model_choice == "Apache Spark (MLlib)":
            st.sidebar.info("📉 RMSE: ~$497,040.95")
        elif model_choice == "Keras Deep Learning NN":
            st.sidebar.info("📉 RMSE: ~$355,000.00")
        else:
            st.sidebar.info("📉 RMSE: ~$351,684.20 (Najlepszy)")

        st.sidebar.markdown("---")
        selected_suburb = st.sidebar.multiselect('Wybierz dzielnicę (Suburb):',
                                                 options=sorted(self.df['Suburb'].unique()), default=[])
        selected_type = st.sidebar.multiselect('Typ nieruchomości:', options=self.df['Type'].unique(),
                                               default=self.df['Type'].unique())

        filtered_df = self.df[self.df['Type'].isin(selected_type)]
        if selected_suburb:
            filtered_df = filtered_df[filtered_df['Suburb'].isin(selected_suburb)]

        # PODZIAŁ NA ZAKŁADKI
        tab1, tab2 = st.tabs(["🔮 Kalkulator Cenowy & Analityka", "📍 Segmentacja Rynku (K-Means)"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔮 Kalkulator Ceny")
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
                fig = px.scatter(filtered_df, x="Distance", y="Price", color="Rooms", hover_data=['Suburb', 'Address'])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.header("📈 Zaawansowana Analiza Statystyczna")
            vis_col1, vis_col2 = st.columns(2)
            with vis_col1:
                fig_hist = px.histogram(filtered_df, x="Price", nbins=50, title="Gęstość cenowa ofert",
                                        color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_hist, use_container_width=True)
            with vis_col2:
                numerical_cols = ['Price', 'Rooms', 'Distance', 'Week', 'CPI', 'Distance2']
                numerical_df = filtered_df[numerical_cols].corr()
                fig_heat = px.imshow(numerical_df, text_auto=".2f", aspect="auto", title="Macierz Korelacji Cech",
                                     color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Podgląd przefiltrowanych nieruchomości")
            st.dataframe(filtered_df.sort_values(by='Price'), use_container_width=True)

        with tab2:
            st.subheader("📍 Analiza klastrów dzielnic (Machine Learning)")
            st.write("Algorytm K-Means dokonał automatycznego podziału rynku na strefy cenowe:")

            fig_clusters = px.scatter(
                self.geo_df,
                x="Distance",
                y="Price_per_Room",
                color="Market_Cluster",
                size="Offer_Count",
                hover_data=['Suburb'],
                title="Wynik klastrowania dzielnic Melbourne (K-Means)"
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            st.dataframe(self.geo_df, use_container_width=True)

    def predict_price(self, model_choice, rooms, distance, week, cpi, last_cpi, distance2):
        input_data_dict = {
            'Rooms': float(rooms), 'Distance': float(distance), 'Week': float(week),
            'CPI': float(cpi), 'LastCPI': float(last_cpi), 'Distance2': float(distance2)
        }
        if model_choice == "Apache Spark (MLlib)":
            if not self.spark_active: st.error("Spark nieaktywny."); return
            from pyspark.ml.feature import VectorAssembler
            spark_input = self.spark.createDataFrame([input_data_dict])
            assembler = VectorAssembler(inputCols=self.model_features, outputCol="features")
            spark_input = assembler.transform(spark_input)
            prediction_df = self.spark_model.transform(spark_input)
            prediction = prediction_df.select("prediction").collect()
            st.success(f'### 💰 Wycena (Apache Spark MLlib): ${prediction:,.2f}')
        elif model_choice == "Keras Deep Learning NN":
            if not self.nn_active: st.error("Keras nieaktywny."); return
            raw_features = [[float(rooms), float(distance), float(week), float(cpi), float(last_cpi), float(distance2)]]
            scaled_features = self.nn_scaler.transform(raw_features)
            prediction = self.nn_model.predict(scaled_features)
            st.success(f'### 💰 Wycena (Keras Deep Learning NN): ${prediction:,.2f}')
        else:
            if not self.xgb_active: st.error("XGBoost nieaktywny."); return
            pandas_input = pd.DataFrame([input_data_dict])[self.model_features]
            prediction = self.xgb_model.predict(pandas_input)
            st.success(f'### 💰 Wycena (XGBoost Regressor): ${prediction:,.2f}')


if __name__ == '__main__':
    app = MelbourneHousingApp()
    app.render_ui()
