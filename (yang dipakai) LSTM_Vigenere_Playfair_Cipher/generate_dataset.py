import re
import random
from pathlib import Path
import pandas as pd

random.seed(42)

FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLAYFAIR_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"


def clean_text(text):
    text = str(text).upper()
    text = re.sub(r"[^A-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_cipher(text):
    return clean_text(text).replace(" ", "")


def vigenere_encrypt(text, key="LSTM"):
    text = clean_for_cipher(text)
    key = clean_for_cipher(key)
    if not key:
        key = "LSTM"
    result = []
    for i, char in enumerate(text):
        shift = ord(key[i % len(key)]) - 65
        encrypted = chr((ord(char) - 65 + shift) % 26 + 65)
        result.append(encrypted)
    return "".join(result)


def create_playfair_matrix(key):
    key = clean_for_cipher(key).replace("J", "I")
    if not key:
        key = "LSTM"
    seen = set()
    seq = []
    for char in key + PLAYFAIR_ALPHABET:
        if char not in seen:
            seen.add(char)
            seq.append(char)
    return [seq[i:i+5] for i in range(0, 25, 5)]


def find_position(matrix, char):
    if char == "J":
        char = "I"
    for row in range(5):
        for col in range(5):
            if matrix[row][col] == char:
                return row, col
    raise ValueError(f"Karakter {char} tidak ditemukan dalam matriks Playfair")


def prepare_playfair_text(text):
    text = clean_for_cipher(text).replace("J", "I")
    pairs = []
    i = 0
    while i < len(text):
        a = text[i]
        if i + 1 < len(text):
            b = text[i + 1]
            if a == b:
                pairs.append(a + "X")
                i += 1
            else:
                pairs.append(a + b)
                i += 2
        else:
            pairs.append(a + "X")
            i += 1
    return pairs


def playfair_encrypt(text, key="LSTM"):
    matrix = create_playfair_matrix(key)
    pairs = prepare_playfair_text(text)
    result = []
    for pair in pairs:
        a, b = pair[0], pair[1]
        r1, c1 = find_position(matrix, a)
        r2, c2 = find_position(matrix, b)
        if r1 == r2:
            result.append(matrix[r1][(c1 + 1) % 5])
            result.append(matrix[r2][(c2 + 1) % 5])
        elif c1 == c2:
            result.append(matrix[(r1 + 1) % 5][c1])
            result.append(matrix[(r2 + 1) % 5][c2])
        else:
            result.append(matrix[r1][c2])
            result.append(matrix[r2][c1])
    return "".join(result)


def build_plaintexts():
    subjects = [
        "keamanan data", "kriptografi klasik", "model lstm", "neural network",
        "kecerdasan buatan", "sistem informasi", "analisis teks",
        "pola enkripsi", "komunikasi digital", "perlindungan informasi",
        "vigenere cipher", "playfair cipher", "plaintext", "ciphertext",
        "data rahasia", "pesan digital", "sistem keamanan", "klasifikasi teks",
        "deep learning", "machine learning", "pengujian model", "dataset penelitian",
        "karakter teks", "proses enkripsi", "hasil klasifikasi"
    ]
    verbs = [
        "digunakan untuk", "membantu", "bertujuan untuk", "menganalisis",
        "menguji", "mengenali", "mengklasifikasikan", "mempelajari",
        "mengevaluasi", "mengukur", "menjelaskan", "mendukung",
        "membandingkan", "memproses"
    ]
    objects = [
        "pola teks", "pola enkripsi", "keamanan informasi",
        "jenis cipher", "hasil enkripsi", "karakter ciphertext",
        "perbedaan plaintext dan ciphertext", "performa model",
        "akurasi klasifikasi", "keterbatasan model", "urutan karakter",
        "hubungan antara plaintext dan ciphertext", "data sekuensial"
    ]
    contexts = [
        "dalam penelitian ini", "dengan pendekatan deep learning",
        "menggunakan model lstm", "pada kriptografi klasik",
        "melalui evaluasi akurasi", "dalam sistem analisis sederhana",
        "dengan variasi panjang teks", "menggunakan beberapa kunci enkripsi",
        "berdasarkan pola karakter", "dengan metode eksperimen",
        "untuk mengetahui kemampuan model", "dalam proses klasifikasi"
    ]
    sentences = []
    for s in subjects:
        for v in verbs:
            for o in objects:
                for c in contexts:
                    text = f"{s} {v} {o} {c}"
                    text = clean_text(text)
                    length = len(clean_for_cipher(text))
                    if 18 <= length <= 120:
                        sentences.append(text)

    prefixes = [
        "berdasarkan hasil pengamatan", "secara umum", "dalam konteks penelitian",
        "pada tahap pengujian", "melalui proses simulasi", "dalam pembahasan ini",
        "berdasarkan data penelitian"
    ]
    suffixes = [
        "secara sistematis", "secara bertahap", "dengan hasil yang dapat diukur",
        "untuk mendukung analisis", "sebagai bagian dari penelitian",
        "dalam bentuk data teks", "menggunakan pendekatan kuantitatif"
    ]
    extra = []
    sample_size = min(2000, len(sentences))
    for text in random.sample(sentences, sample_size):
        extra.append(clean_text(f"{random.choice(prefixes)} {text}"))
        extra.append(clean_text(f"{text} {random.choice(suffixes)}"))

    all_sentences = list(set(sentences + extra))
    random.shuffle(all_sentences)
    return all_sentences[:2500]


def build_dataset():
    keys = [
        "DATA", "AMAN", "KRIPTO", "MODEL", "POLA",
        "NEURAL", "ENKRIPSI", "SISTEM", "TEKS", "RAHASIA"
    ]
    rows = []
    plaintexts = build_plaintexts()
    for i, text in enumerate(plaintexts):
        key = keys[i % len(keys)]
        plain = clean_for_cipher(text)
        if len(plain) < 10:
            continue
        rows.append({"text": plain, "label": 0, "jenis": "Plaintext / Teks Biasa"})
        rows.append({"text": vigenere_encrypt(plain, key), "label": 1, "jenis": "Ciphertext Vigenere"})
        rows.append({"text": playfair_encrypt(plain, key), "label": 2, "jenis": "Ciphertext Playfair"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "dataset_lstm_vigenere_playfair.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(out, index=False)
    print(f"Dataset disimpan ke: {out}")
    print(df["jenis"].value_counts())
    print("Total:", len(df))
