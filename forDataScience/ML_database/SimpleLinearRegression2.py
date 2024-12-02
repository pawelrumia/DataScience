from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

marketing = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\tvmarketing.csv')

x = marketing["TV"].values.reshape(-1, 1)
y = marketing['Sales'].values.reshape(-1, 1)

print(f"Nulls: {marketing.isnull().sum()}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(regressor.intercept_)

print(regressor.coef_)

plt.scatter(x_test, y_test, color="green")
plt.plot(x_test, y_pred, color="orange")
plt.show()
