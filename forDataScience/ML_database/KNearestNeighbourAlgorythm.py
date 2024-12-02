import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#apply SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\train.csv')
print(df.head())
print(f"Nulls: {df.isnull().sum()}")
print(df.describe())

print("==============================")
print(df.head())
