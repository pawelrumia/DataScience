import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers, models
import numpy as np

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("result.png")
plt.show()

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="leaky_relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="leaky_relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="leaky_relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"\nTest loss: {test_loss:.4f}")

predictions = model.predict(X_test)

def plot_predictions_grid(X, y_true, preds, class_names, indices, rows=3, cols=3, fname=None):
    """Rysuje siatkę obrazków 3x3 z predykcjami i zapisuje do pliku (opcjonalnie)."""
    n = rows * cols
    indices = np.array(indices)[:n]
    y_pred = np.argmax(preds[indices], axis=1)
    conf = preds[indices, y_pred]

    plt.figure(figsize=(cols*3, rows*3))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx].squeeze(), cmap="gray")
        ok = (y_pred[i] == y_true[idx])
        title = f"pred: {class_names[y_pred[i]]} ({conf[i]*100:.1f}%)\ntrue: {class_names[y_true[idx]]}"
        plt.title(title, color=("green" if ok else "red"), fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

# --- przykładowe trzy różne siatki 3x3 ---
rng = np.random.default_rng(43)  # reprodukowalne losowania
idx1 = rng.choice(len(X_test), 9, replace=False)
idx2 = rng.choice(len(X_test), 9, replace=False)
idx3 = rng.choice(len(X_test), 9, replace=False)

plot_predictions_grid(X_test, y_test, predictions, class_names, indices=idx1, fname="preds_grid_1.png")
plot_predictions_grid(X_test, y_test, predictions, class_names, indices=idx2, fname="preds_grid_2.png")
plot_predictions_grid(X_test, y_test, predictions, class_names, indices=idx3, fname="preds_grid_3.png")

def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            filters=hp.Choice
        )
    )