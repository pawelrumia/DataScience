import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\Social_Network_Ads.csv')

print(df)
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.columns)
df = df.drop(columns=['Gender'])

plt.figure(figsize=(10, 4), dpi=96)
plt.title("Histogram for Estimated Salary values")
plt.hist(df['EstimatedSalary'], bins='sturges', edgecolor='black')
plt.show()


X = df.drop(columns = 'Purchased')
y = df['Purchased']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print(lr.score(X_train,y_train))
y_train_pred = lr.predict(X_train)


print("\n Confusion Matrix \n")
print(confusion_matrix(y_train,y_train_pred))

print("\n Classification Report\n")
print(classification_report(y_train,y_train_pred))