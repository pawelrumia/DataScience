import pandas as pd
from sklearn.model_selection import train_test_split

housing = pd.read_csv("C:\\Users\\mazurp2\\PycharmProjects\\Data Science\\various\\resources\\Melbourne_housing.csv")

print(housing.head())
print(housing.describe())
print(housing.columns)

y = housing.Distance
x = housing.drop('Distance', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print("shape of original dataset :", housing.shape)
print("shape of input - training set :", x_train.shape)
print("shape of output - training set :", y_train.shape)
print("shape of input - testing set :" , x_test.shape)
print("shape of output - testing set :", y_test.shape)