import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import warnings
import warnings

warnings.filterwarnings("ignore")

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\DATA.csv')
print(df.head())

x = df[['Weight', 'Volume']]
y = df['CO2']

fig, axis = plt.subplots(2, figsize=(5, 5))
plt1=sns.boxplot(df['Weight'], ax=axis[0])
plt2=sns.boxplot(df['Volume'], ax=axis[1])
plt.tight_layout()
# plt.show()

# sns.displot(df['CO2'])
# plt.show()

sns.pairplot(df, x_vars=['Weight', 'Volume'], y_vars='CO2', height=4, aspect=1, kind='scatter')
plt.show()


X_train,X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 100)
print(y_train.shape)

reg_model = linear_model.LinearRegression().fit(X_train,y_train)
print('Intercept:',reg_model.intercept_)
#pair the feature names with the coefficients
list(zip(x,reg_model.coef_))

y_pred = reg_model.predict(X_test)
x_pred = reg_model.predict(X_train)

print("Prediction for test set: {}".format(y_pred))

#actual value and the predicted value
reg_model_diff = pd.DataFrame({'Actual value':y_test,'Predicted value':y_pred})
print(reg_model_diff)

mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test,y_pred))


print('Mean Absolute Error:' , mae)
print('Mean Sqaure Error:',mse)
print('Root Mean Sqaure Error:',r2)