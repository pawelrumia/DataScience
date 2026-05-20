import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


class MelbourneHousingApp:
    def __init__(self):
        # Dokładne kolumny numeryczne, na których uczył się Twój model
        self.model_features = ['Rooms', 'Distance', 'Week', 'CPI', 'LastCPI', 'Distance2']
        self.load_assets()

    @st.cache_resource
    def load_assets(_self):
        """Ładowanie zasobów z automatycznym czyszczeniem uszkodzonych wierszy."""
        try:
            _self.model = joblib.load('../models/melbourne_model.pkl')
        except FileNotFoundError:
            _self.model = None

        _self.df = pd.read_csv('../data/Melbourne_housing.csv')

        # 1. Wymuszamy konwersję daty. Błędne wiersze (tekstowe) zamienią się w wartości NaT (puste)
        _self.df['Date'] = pd.to_datetime(_self.df['Date'], errors='coerce', format='mixed', dayfirst=True)

        # 2. Usuwamy z całej tabeli wiersze, które miały uszkodzoną datę
        _self.df = _self.df.dropna(subset=['Date'])

    def render_ui(self):
        st.set_page_config(page_title="Melbourne Housing ML", layout="wide")
        st.title('🏢 Melbourne Housing ML & Market Dashboard')

        # Panel boczny z filtrami
        st.sidebar.header("🔍 Filtry bazy danych")
        selected_suburb = st.sidebar.multiselect(
            'Wybierz dzielnicę (Suburb):',
            options=sorted(self.df['Suburb'].unique()),
            default=[]
        )

        selected_type = st.sidebar.multiselect(
            'Typ nieruchomości:',
            options=self.df['Type'].unique(),
            default=self.df['Type'].unique()
        )

        # Filtrowanie danych do tabeli i wykresu
        filtered_df = self.df[self.df['Type'].isin(selected_type)]
        if selected_suburb:
            filtered_df = filtered_df[filtered_df['Suburb'].isin(selected_suburb)]

        # Układ kolumn w aplikacji
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔮 Kalkulator Ceny (Prognoza ML)")

            rooms = st.slider('Liczba pokoi (Rooms)', int(self.df['Rooms'].min()), int(self.df['Rooms'].max()), 3)
            distance = st.slider('Odległość od centrum (Distance)', float(self.df['Distance'].min()),
                                 float(self.df['Distance'].max()), 10.0)
            week = st.slider('Tydzień roku (Week)', int(self.df['Week'].min()), int(self.df['Week'].max()), 25)
            cpi = st.slider('Bieżący indeks CPI', float(self.df['CPI'].min()), float(self.df['CPI'].max()), 108.5)

            # Wyliczenia cech pochodnych
            last_cpi = cpi - 0.4
            distance2 = distance * 1.2

            if st.button('🚀 Przewiduj cenę nieruchomości', use_container_width=True):
                self.predict_price(rooms, distance, week, cpi, last_cpi, distance2)

        with col2:
            st.subheader("📊 Analiza trendów rynkowych")
            fig = px.scatter(
                filtered_df,
                x="Distance",
                y="Price",
                color="Rooms",
                hover_data=['Suburb', 'Address'],
                title="Stosunek ceny do odległości od centrum"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Podgląd przefiltrowanych nieruchomości")
        st.dataframe(filtered_df.sort_values(by='Price'), use_container_width=True)

    def predict_price(self, rooms, distance, week, cpi, last_cpi, distance2):
        if self.model is None:
            st.error("Nie znaleziono pliku modelu w `./models/melbourne_model.pkl`. Sprawdź strukturę folderów.")
            return

        input_data = pd.DataFrame([{
            'Rooms': rooms,
            'Distance': distance,
            'Week': week,
            'CPI': cpi,
            'LastCPI': last_cpi,
            'Distance2': distance2
        }])

        # Wymuszenie kolejności kolumn
        input_data = input_data[self.model_features]
        prediction = self.model.predict(input_data)

        st.success(f'### 💰 Estymowana wartość: ${prediction[0]:,.2f}')


if __name__ == '__main__':
    app = MelbourneHousingApp()
    app.render_ui()
