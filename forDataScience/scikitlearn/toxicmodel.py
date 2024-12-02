import os
import random
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel

# 1) Stała wartość seed
seed = 42

# 2) Ustawienie seedów (uwaga: poprawna nazwa zmiennej środowiskowej)
os.environ["PYTHONHASHSEED"] = str(seed)  # <-- poprawka
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# 3) Determinizm TF (może spowolnić trening)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

# Ścieżka lokalna na model/tokenizer
bert_mini_path = "./bert-tiny"
hf_model_id = "prajjwal1/bert-tiny"  # sprawdzony BERT-tiny

# -- Funkcja, która zapewnia komplet plików lokalnie --
def ensure_local_bert(path: str, model_id: str):
    needed_tok = {"vocab.txt", "tokenizer_config.json", "special_tokens_map.json"}
    needed_model_any = {"tf_model.h5", "tf_model.h5.index", "pytorch_model.bin"}  # wystarczy jedno z nich
    have = set(os.listdir(path)) if os.path.isdir(path) else set()
    ok_tokenizer = needed_tok.issubset(have)
    ok_weights = len(needed_model_any.intersection(have)) > 0
    ok_config = "config.json" in have

    if not (ok_tokenizer and ok_weights and ok_config):
        # Jednorazowe pobranie i zapis kompletu do katalogu
        tok = AutoTokenizer.from_pretrained(model_id)
        try:
            mdl = TFAutoModel.from_pretrained(model_id, from_pt=False)
        except Exception:
            # Jeśli brak wag TF, przekonwertuj z PyTorch
            mdl = TFAutoModel.from_pretrained(model_id, from_pt=True)
        os.makedirs(path, exist_ok=True)
        tok.save_pretrained(path)
        mdl.save_pretrained(path)

# Upewnij się, że mamy komplet lokalnie
ensure_local_bert(bert_mini_path, hf_model_id)

# Teraz ładujemy WYŁĄCZNIE lokalnie (bez internetu)
tokenizer = AutoTokenizer.from_pretrained(bert_mini_path, local_files_only=True)
bert_base = TFAutoModel.from_pretrained(bert_mini_path, local_files_only=True)

# --- Twoje dane ---
df = pd.read_csv(r"/forDataScience\resources\toxic_subset.csv")
texts = df["comment_text"].astype(str).tolist()
labels = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

# Podziały
texts_train_val, texts_test, y_train_val, y_test = train_test_split(
    texts, labels, test_size=0.10, random_state=seed
)
texts_train, texts_val, y_train, y_val = train_test_split(
    texts_train_val, y_train_val, test_size=0.20, random_state=seed
)

# Tokenizacja
max_len = 128
def tok(batch_texts):
    return tokenizer(
        batch_texts,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
    )

X_train = tok(texts_train)
X_val = tok(texts_val)
X_test = tok(texts_test)

# --- Model: BERT + head ---
def build_model():
    input_ids = keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    outputs = bert_base(input_ids, attention_mask=attention_mask)
    pooled = outputs.pooler_output  # bezpieczniejsze niż indeks [1]

    x = keras.layers.Dropout(0.3, seed=seed)(pooled)
    # Uwaga: w Keras 3 zalecane jest użycie warstwy aktywacji zamiast stringu "leaky_relu"
    x = keras.layers.Dense(
        64,
        activation=None,
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
    )(x)
    x = keras.layers.LeakyReLU()(x)
    logits = keras.layers.Dense(
        6,
        activation="sigmoid",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
    )(x)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.5, name="bin_acc")],
    )
    return model

model = build_model()

history = model.fit(
    {"input_ids": X_train["input_ids"], "attention_mask": X_train["attention_mask"]},
    y_train,
    validation_data=(
        {"input_ids": X_val["input_ids"], "attention_mask": X_val["attention_mask"]},
        y_val
    ),
    epochs=10,
    batch_size=32
)

# Zapis modelu w formacie .keras (zalecane w Keras 3)
model.save("toxic_model.keras", include_optimizer=False)
print("Zapisano model do pliku: toxic_model.keras")
