import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# KONFIGURASI DASAR
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "data" / "dataset_lstm_vigenere_playfair.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "lstm_cipher_model.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation_metrics.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"

FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LABEL_NAMES = {
    0: "Plaintext / Teks Biasa",
    1: "Ciphertext Vigenere",
    2: "Ciphertext Playfair"
}

CHAR_TO_INT = {c: i + 1 for i, c in enumerate(FULL_ALPHABET)}


# =========================================================
# FUNGSI PREPROCESSING
# =========================================================
def clean_text(text):
    text = str(text).upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_cipher(text):
    return clean_text(text).replace(" ", "")


def encode_text(text):
    text = clean_for_cipher(text).replace("J", "I")
    return [CHAR_TO_INT[c] for c in text if c in CHAR_TO_INT]


def to_builtin(obj):
    """
    Mengubah tipe numpy agar bisa disimpan ke JSON.
    """
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Kolom dataset kurang: {missing}")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


# =========================================================
# BUILD MODEL
# =========================================================
def build_model(max_len):
    model = Sequential([
        Embedding(input_dim=27, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(96)),
        Dropout(0.35),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# MAIN TRAINING
# =========================================================
def main():
    print("Membaca dataset...")
    df = load_dataset()

    print("Jumlah data:", len(df))
    print("Distribusi label:")
    print(df["label"].value_counts().sort_index())

    print("Melakukan encoding karakter...")
    encoded = df["text"].apply(encode_text).tolist()

    max_len = max(len(x) for x in encoded)

    X = pad_sequences(
        encoded,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    y = df["label"].values

    print("Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    print("Membangun model LSTM...")
    model = build_model(max_len)

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    )

    print("Training model dimulai...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluasi model...")
    pred_probs = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    accuracy = accuracy_score(y_test, pred_labels)
    cm = confusion_matrix(y_test, pred_labels)

    report = classification_report(
        y_test,
        pred_labels,
        target_names=["Plaintext", "Vigenere", "Playfair"],
        output_dict=True,
        zero_division=0
    )

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)

    print("Menyimpan model dan hasil evaluasi...")
    MODEL_DIR.mkdir(exist_ok=True)

    model.save(MODEL_PATH)

    metadata = {
        "max_len": int(max_len),
        "label_names": {
            "0": LABEL_NAMES[0],
            "1": LABEL_NAMES[1],
            "2": LABEL_NAMES[2]
        },
        "alphabet": FULL_ALPHABET,
        "encoding": "A=1 sampai Z=26",
        "padding": "post",
        "model_file": "lstm_cipher_model.keras"
    }

    evaluation = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(to_builtin(metadata), f, indent=4, ensure_ascii=False)

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(to_builtin(evaluation), f, indent=4, ensure_ascii=False)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(to_builtin(history.history), f, indent=4, ensure_ascii=False)

    print("Selesai.")
    print(f"Model disimpan di: {MODEL_PATH}")
    print(f"Metadata disimpan di: {METADATA_PATH}")
    print(f"Evaluasi disimpan di: {EVALUATION_PATH}")
    print(f"History training disimpan di: {HISTORY_PATH}")


if __name__ == "__main__":
    main()