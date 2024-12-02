import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report


df = pd.read_csv("creditcard.csv")

X = df.drop(["Time", "Class"], axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[y==0]
X_test = X_scaled

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)

encoded = Dense(encoding_dim, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)

decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True, validation_split=0.1)

X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

threshold = np.percentile(mse[y==0], 95)
y_pred = (mse > threshold).astype(int)

roc_auc = roc_auc_score(y, mse)
cr = classification_report(y, y_pred)