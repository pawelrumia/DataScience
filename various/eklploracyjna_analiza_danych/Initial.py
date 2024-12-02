import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\mazurp2\\PycharmProjects\\DataScience\\resources\\cars.csv")
print(df.describe())
print(df.sample())
print(df.columns)
print(df.head(7))
print(df.shape)
print(df.dtypes)
print("------------------------------")
print(df['color'].unique())
print(df.isna())
print(df.isna().sum())
print(df.describe())
print(df.info())
print("Feature engineering")
print(pd.to_datetime(df['offer_timestamp']))
df['offer_timestamp'] = pd.to_datetime(df['offer_timestamp'])
df['rok'] = df['offer_timestamp'].dt.year
df['godzina'] = df['offer_timestamp'].dt.hour

df['wiek_auta'] = df['rok'] - df['prod_year']

print("wykresyyyyyyyyyyyyyyyyyyyyy")
# plt.show()

prawdziwa_cena = df[df["price"]<200000]["price"]
# plt.hist(prawdziwa_cena, edgecolor='pink', bins=50);
# plt.show()

currency = df['currency'].value_counts()
# plt.bar(currency.index, currency.values, log=True)
# plt.show()

print("brands==============================")
brands = df['brand'].value_counts()
others = len(df) - sum(brands.iloc[:10])
brand_popular = brands.iloc[:10]
brand_popular['inne'] = others
# print(brand_popular)
# plt.figure(figsize=(12,5))
# plt.xticks(rotation=15)
# plt.title('Wykres')
# plt.ylabel('Liczebnosc marek')
# plt.bar(brand_popular.index, brand_popular.values)
# plt.show()

# mileage = df[df['mileage'] < 0.5e6]['mileage']
# plt.hist(mileage[(mileage>8e4) & (mileage<18e4)], bins=np.arange(80000, 180000, 1000), edgecolor='k')
# plt.show()


print("PALIWO")
paliwo = df['fuel'].value_counts()# plt.pie(paliwo, labels=paliwo.index, explode=[0,0,0.1,0.2,0.3,0.4], startangle=80)
# plt.show()

print('Analiza zaleznosci miedzy zmiennymi')
print(df.groupby('fuel')['price'].mean().sort_values(ascending=False))

print("rok produkcji vs power")
wynik = df[(df['prod_year']>=2000) & (df['prod_year']<=2018)]
print(wynik.head(10))
pogrupowane = wynik.groupby('prod_year')['power'].mean()
plt.bar(pogrupowane.index, pogrupowane.values)
plt.xticks(list(range(2000, 2019, 2)), rotation=12)
# plt.show()


print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{[   ILOC")
