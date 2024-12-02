import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

df = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\sp500.txt")
df = df.sort_values("Date")
data = df[["Close"]].values
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)