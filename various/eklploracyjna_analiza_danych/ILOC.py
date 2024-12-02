import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\mazurp2\\PycharmProjects\\DataScience\\resources\\cars.csv")
dfDrugie = pd.read_csv("C:\\Users\\mazurp2\\PycharmProjects\\DataScience\\resources\\cars.csv", index_col='title')
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
print("ILOC  :")
print(df.iloc[26:30])
print("LOC:   ")
# print(df.loc[9900])

print(dfDrugie.loc[['Ferrari Dino GT4', 'BMW Seria 5 523i']])
print(dfDrugie.get('dynks', default='Nie istnieje!'))