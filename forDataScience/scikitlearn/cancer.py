import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

dataset = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\cancer.csv")

X = dataset.drop(["diagnosis", "id"], axis=1)
y = dataset["diagnosis"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model = keras.Sequential([
#     keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(1, activation="sigmoid")
# ])

model = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_split=0.1,
    verbose=1
)

preds = model.predict(X_test).reshape(-1)
preds_binary = (preds > 0.5).astype(int)

cr = classification_report(y_test, preds_binary, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test, preds_binary)

plt.plot(history.history['accuracy'], label="Train accuracy")
plt.plot(history.history['val_accuracy'], label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model training accuracy")
plt.savefig("Model training accuracy")
plt.show()
