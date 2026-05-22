import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target.reshape(-1, 1), random_state=42
)

# One-hot encode target values
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_train_onehot = enc.fit_transform(y_train)
y_test_onehot = enc.transform(y_test)

# Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    X_train, y_train_onehot,
    epochs=10,
    validation_data=(X_test, y_test_onehot)
)

# Save as a TensorFlow SavedModel by saving to a directory (no file extension)
model.export("my_model")