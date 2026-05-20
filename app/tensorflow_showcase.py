import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs("../images", exist_ok=True)
os.makedirs("../models", exist_ok=True)
os.makedirs("../logs", exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "../data/wine.csv",
    sep=";"
)

print("\nHEAD:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nNULLS:")
print(df.isnull().sum())

# ==========================================
# TARGET
# ==========================================

df['quality_label'] = (
    df['quality'] >= 6
).astype(int)

# ==========================================
# FEATURES / TARGET
# ==========================================

X = df.drop(columns=['quality', 'quality_label'])

y = df['quality_label']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("../images/correlation_heatmap.png")

plt.close()

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)

checkpoint = ModelCheckpoint(
    '../models/best_model.keras',
    save_best_only=True
)

tensorboard = TensorBoard(
    log_dir='../logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# ==========================================
# MODEL
# ==========================================

model = keras.Sequential([

    layers.Dense(
        128,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.3),

    layers.Dense(
        64,
        activation='relu'
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.3),

    layers.Dense(
        32,
        activation='relu'
    ),

    layers.Dense(
        1,
        activation='sigmoid'
    )
])

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(),
        keras.metrics.Recall()
    ]
)

# ==========================================
# MODEL SUMMARY
# ==========================================

print("\nMODEL SUMMARY:")
model.summary()

# ==========================================
# TRAIN
# ==========================================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint,
        tensorboard
    ],

    verbose=1
)

# ==========================================
# EVALUATION
# ==========================================

results = model.evaluate(
    X_test,
    y_test
)

print("\nTEST RESULTS:")
print(results)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

predictions = (
    predictions > 0.5
).astype(int)

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nACCURACY:")
print(accuracy)

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../images/confusion_matrix.png")

plt.close()

# ==========================================
# LOSS PLOT
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history['loss'],
    label='train_loss'
)

plt.plot(
    history.history['val_loss'],
    label='val_loss'
)

plt.title("Loss During Training")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("../images/loss_plot.png")

plt.close()

# ==========================================
# ACCURACY PLOT
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history['accuracy'],
    label='train_accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='val_accuracy'
)

plt.title("Accuracy During Training")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("../images/accuracy_plot.png")

plt.close()

# ==========================================
# SAVE MODEL
# ==========================================

model.save("../models/wine_tensorflow_model.keras")

print("\nMODEL SAVED!")

# ==========================================
# LOAD MODEL
# ==========================================

loaded_model = keras.models.load_model(
    "../models/wine_tensorflow_model.keras"
)

print("\nMODEL LOADED!")

# ==========================================
# SINGLE PREDICTION
# ==========================================

single_prediction = loaded_model.predict(
    X_test[:1]
)

print("\nSINGLE PREDICTION:")
print(single_prediction)

# ==========================================
# FEATURE IMPORTANCE APPROXIMATION
# ==========================================

weights = model.layers[0].get_weights()[0]

importance = np.mean(
    np.abs(weights),
    axis=1
)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
})

importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(importance_df)

# ==========================================
# FEATURE IMPORTANCE PLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x='importance',
    y='feature'
)

plt.title("TensorFlow Feature Importance")

plt.savefig("../images/tensorflow_feature_importance.png")

plt.close()

print("\nALL DONE!")