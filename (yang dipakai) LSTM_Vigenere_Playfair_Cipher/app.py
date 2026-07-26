import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# KONFIGURASI DASAR
# =========================================================
st.set_page_config(
    page_title="Analisis Pola Enkripsi dengan LSTM",
    layout="wide"
)

LABEL_NAMES = {
    0: "Plaintext / Teks Biasa",
    1: "Ciphertext Vigenère",
    2: "Ciphertext Playfair"
}

FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLAYFAIR_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "data" / "dataset_lstm_vigenere_playfair.csv"

MODEL_PATH = BASE_DIR / "model" / "lstm_cipher_model.keras"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
EVALUATION_PATH = BASE_DIR / "model" / "evaluation_metrics.json"
HISTORY_PATH = BASE_DIR / "model" / "training_history.json"

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


def encode_input(text, char_to_int):
    text = clean_for_cipher(text).replace("J", "I")
    return [char_to_int[c] for c in text if c in char_to_int]


def get_removed_characters(text):
    """
    Mengambil karakter yang tidak termasuk alfabet A-Z dan spasi.
    Digunakan untuk menjelaskan uji batas model.
    """
    return re.findall(r"[^A-Za-z\s]", str(text))


def explain_pattern(pred):
    """
    Penjelasan pola berdasarkan hasil prediksi model.
    Bagian ini digunakan agar web tidak hanya menampilkan hasil prediksi,
    tetapi juga menjelaskan pola karakter yang dikenali oleh model.
    """
    if pred == 0:
        return (
            "Model mengenali teks ini sebagai plaintext karena pola karakternya "
            "lebih mendekati pola bahasa alami. Plaintext umumnya masih memiliki "
            "susunan huruf yang lebih teratur dan belum mengalami perubahan pola "
            "akibat proses enkripsi."
        )

    if pred == 1:
        return (
            "Model mengenali teks ini sebagai ciphertext Vigenère karena pola "
            "karakternya lebih mendekati hasil pergeseran huruf berdasarkan kunci. "
            "Pada Vigenère Cipher, setiap karakter plaintext mengalami pergeseran "
            "sesuai karakter kunci, sehingga membentuk pola substitusi polialfabetik."
        )

    return (
        "Model mengenali teks ini sebagai ciphertext Playfair karena pola karakternya "
        "lebih mendekati pola pasangan huruf atau digraph. Pada Playfair Cipher, "
        "teks diproses dua huruf sekaligus menggunakan matriks 5x5, sehingga pola "
        "karakter yang terbentuk berbeda dari plaintext dan Vigenère Cipher."
    )


# =========================================================
# VIGENÈRE CIPHER
# =========================================================
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


# =========================================================
# PLAYFAIR CIPHER
# =========================================================
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

    return [seq[i:i + 5] for i in range(0, 25, 5)]


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


# =========================================================
# LOAD DATASET, MODEL, METADATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATASET_PATH.exists():
        st.error(f"File dataset tidak ditemukan: {DATASET_PATH}")
        st.stop()

    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        st.error("Dataset harus memiliki kolom `text` dan `label`.")
        st.stop()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if "jenis" not in df.columns:
        df["jenis"] = df["label"].map(LABEL_NAMES)

    return df


@st.cache_resource(show_spinner=False)
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error(
            "File model belum ditemukan.\n\n"
            "Pastikan file model sudah ada di:\n"
            f"`{MODEL_PATH}`\n\n"
            "Jalankan dulu `python train_export_model.py` untuk membuat file `.keras`."
        )
        st.stop()

    model = load_model(MODEL_PATH)
    return model


def load_json(path, default=None):
    if default is None:
        default = {}

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_max_len(df, metadata):
    if "max_len" in metadata:
        return int(metadata["max_len"])

    encoded = df["text"].apply(lambda x: encode_input(x, CHAR_TO_INT)).tolist()
    return max(len(x) for x in encoded)


def compute_evaluation_if_needed(model, df, max_len):
    """
    Fallback jika evaluation_metrics.json belum ada.
    Evaluasi ini memakai seluruh dataset sebagai pengecekan tampilan.
    Lebih baik tetap gunakan evaluation_metrics.json dari train_export_model.py.
    """
    encoded = df["text"].apply(lambda x: encode_input(x, CHAR_TO_INT)).tolist()

    X = pad_sequences(
        encoded,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    y = df["label"].values

    pred_probs = model.predict(X, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    accuracy = accuracy_score(y, pred_labels)
    cm = confusion_matrix(y, pred_labels)

    report = classification_report(
        y,
        pred_labels,
        target_names=["Plaintext", "Vigenère", "Playfair"],
        output_dict=True,
        zero_division=0
    )

    return accuracy, cm, report


def load_evaluation(model, df, max_len):
    evaluation = load_json(EVALUATION_PATH, default={})

    if evaluation:
        accuracy = float(evaluation.get("accuracy", 0))
        cm = np.array(evaluation.get("confusion_matrix", []))
        report = evaluation.get("classification_report", {})

        if cm.size == 0 or not report:
            accuracy, cm, report = compute_evaluation_if_needed(model, df, max_len)

        return accuracy, cm, report

    st.warning(
        "File evaluation_metrics.json belum ditemukan. "
        "Aplikasi menghitung evaluasi sementara dari dataset. "
        "Sebaiknya jalankan train_export_model.py agar hasil evaluasi tersimpan resmi."
    )

    return compute_evaluation_if_needed(model, df, max_len)


def predict_text(model, text, char_to_int, max_len):
    cleaned = clean_for_cipher(text).replace("J", "I")

    seq = encode_input(cleaned, char_to_int)

    seq = pad_sequences(
        [seq],
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    probs = model.predict(seq, verbose=0)[0]
    pred = int(np.argmax(probs))

    return cleaned, pred, probs


# =========================================================
# LOAD APP DATA
# =========================================================
st.title("Analisis Pola Enkripsi Menggunakan LSTM")

st.caption(
    "Aplikasi web untuk menganalisis pola karakter pada Plaintext, "
    "Ciphertext Vigenère, dan Ciphertext Playfair."
)

st.info(
    "Tujuan sistem ini bukan sebagai encryption generator utama dan bukan untuk membobol "
    "enkripsi modern. Sistem ini digunakan untuk menganalisis apakah model LSTM dapat "
    "mengenali pola karakter pada teks biasa dan teks hasil enkripsi cipher klasik."
)

with st.spinner("Memuat model LSTM yang sudah dilatih..."):
    df = load_dataset()
    model = load_trained_model()

    metadata = load_json(METADATA_PATH, default={})
    max_len = get_max_len(df, metadata)

    accuracy, cm, report = load_evaluation(model, df, max_len)
    hist = load_json(HISTORY_PATH, default={})

    char_to_int = CHAR_TO_INT


# =========================================================
# STREAMLIT UI
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analisis Teks",
    "Hasil Model",
    "Dataset",
    "Uji Kasus",
    "Keterangan"
])


# =========================================================
# TAB 1: ANALISIS TEKS
# =========================================================
with tab1:
    st.subheader("Analisis Pola Teks Baru")

    st.write(
        "Halaman ini digunakan untuk menguji teks baru. Teks yang dimasukkan akan "
        "diproses oleh model LSTM untuk melihat apakah pola karakternya lebih mirip "
        "plaintext, ciphertext Vigenère, atau ciphertext Playfair."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        user_text = st.text_input(
            "Masukkan teks",
            value="Dokumen rahasia berisi data kredensial pengguna sistem"
        )

    with col2:
        user_key = st.text_input("Kunci untuk skenario uji", value="LSTM")

    mode = st.radio(
        "Pilih skenario input",
        [
            "Uji teks apa adanya",
            "Bentuk ciphertext Vigenère untuk skenario uji",
            "Bentuk ciphertext Playfair untuk skenario uji"
        ]
    )

    if mode == "Uji teks apa adanya":
        input_text = user_text
        proses = (
            "Teks diuji apa adanya. Sistem akan melakukan preprocessing dan model LSTM "
            "akan menganalisis pola karakter dari teks tersebut."
        )
    elif mode == "Bentuk ciphertext Vigenère untuk skenario uji":
        input_text = vigenere_encrypt(user_text, user_key)
        proses = (
            f"Teks dibentuk menjadi ciphertext Vigenère menggunakan kunci '{user_key}' "
            "sebagai skenario uji. Ciphertext ini kemudian dianalisis oleh model LSTM."
        )
    else:
        input_text = playfair_encrypt(user_text, user_key)
        proses = (
            f"Teks dibentuk menjadi ciphertext Playfair menggunakan kunci '{user_key}' "
            "sebagai skenario uji. Ciphertext ini kemudian dianalisis oleh model LSTM."
        )

    st.write("### Teks yang Dianalisis Model")
    st.code(input_text)

    removed_chars = get_removed_characters(user_text)

    if removed_chars:
        st.warning(
            "Input mengandung karakter di luar alfabet A-Z, seperti angka, simbol, "
            "atau tanda baca. Pada tahap preprocessing, karakter tersebut tidak digunakan "
            "karena model hanya dilatih pada karakter alfabet A-Z. Kondisi ini dapat "
            "memengaruhi hasil prediksi model."
        )

        st.write("Karakter non-alfabet yang terdeteksi:")
        st.code(" ".join(sorted(set(removed_chars))))

    cleaned_preview = clean_for_cipher(input_text).replace("J", "I")

    if len(cleaned_preview) == 0:
        st.error(
            "Teks tidak memiliki karakter alfabet A-Z setelah preprocessing. "
            "Masukkan teks yang mengandung huruf agar dapat dianalisis."
        )
    else:
        cleaned, pred, probs = predict_text(model, input_text, char_to_int, max_len)

        st.write("### Hasil Prediksi Model")
        st.success(f"Model memprediksi teks ini sebagai: **{LABEL_NAMES[pred]}**")

        prob_df = pd.DataFrame({
            "Kategori": [LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            "Nilai Kemungkinan": [
                float(probs[0]),
                float(probs[1]),
                float(probs[2])
            ]
        })

        st.dataframe(prob_df, use_container_width=True)

        st.write("### Analisis Pola Karakter")
        st.write(f"**Teks setelah preprocessing:** `{cleaned}`")
        st.write(f"**Panjang teks setelah preprocessing:** {len(cleaned)} karakter")
        st.write(f"**Panjang maksimum input model:** {max_len} karakter")

        if len(cleaned) < 18:
            st.warning(
                "Panjang teks berada di bawah rentang utama dataset penelitian. "
                "Karena pola karakter yang tersedia lebih sedikit, hasil prediksi "
                "dapat menjadi kurang stabil."
            )

        if len(cleaned) > max_len:
            st.warning(
                "Panjang teks melebihi panjang maksimum input model. Sistem akan "
                "melakukan pemotongan sequence sesuai batas panjang input model, "
                "sehingga sebagian karakter di akhir teks tidak dianalisis."
            )

        st.info(explain_pattern(pred))

        st.write("### Proses Pengujian")
        st.write(proses)

        st.warning(
            "Catatan: hasil prediksi tidak selalu benar. Model hanya belajar dari pola "
            "data latih dan tidak memahami makna teks seperti manusia."
        )


# =========================================================
# TAB 2: HASIL MODEL
# =========================================================
with tab2:
    st.subheader("Hasil Evaluasi Model")

    st.write("""
    Bagian ini menampilkan hasil belajar model LSTM setelah dilatih menggunakan data
    Plaintext, Vigenère Cipher, dan Playfair Cipher.
    """)

    st.metric("Tingkat Akurasi Model", f"{accuracy:.2%}")

    if accuracy >= 0.80:
        st.success(
            "Model sudah cukup baik karena tingkat akurasinya berada di atas 80%. "
            "Artinya, sebagian besar teks berhasil diklasifikasikan dengan benar."
        )
    elif accuracy >= 0.60:
        st.warning(
            "Model sudah dapat mengenali pola, tetapi hasilnya masih sedang. "
            "Masih ada beberapa teks yang salah diklasifikasikan."
        )
    else:
        st.error(
            "Akurasi model masih rendah. Model belum cukup stabil dalam mengenali pola teks."
        )

    st.write("""
    Nilai akurasi digunakan untuk melihat seberapa besar kemampuan model dalam
    mengklasifikasikan teks secara benar. Namun, akurasi tidak cukup untuk menjelaskan
    seluruh performa model. Oleh karena itu, hasil lain seperti confusion matrix,
    precision, recall, dan F1-score juga ditampilkan.
    """)

    st.divider()

    st.subheader("Penjelasan Epoch")

    st.info("""
    Epoch adalah jumlah putaran belajar model terhadap seluruh data training.
    Misalnya epoch 25 berarti model diberi kesempatan belajar sampai 25 kali.
    Semakin banyak epoch, model bisa belajar lebih lama, tetapi jika terlalu banyak
    model dapat terlalu menghafal data training.
    """)

    hist_df = pd.DataFrame(hist)

    if "accuracy" in hist_df.columns and "val_accuracy" in hist_df.columns:
        st.subheader("Grafik Perkembangan Akurasi")

        st.line_chart(hist_df[["accuracy", "val_accuracy"]])

        st.write("""
        Grafik ini menunjukkan perkembangan kemampuan model selama proses training.
        Garis **accuracy** menunjukkan kemampuan model pada data latihan.
        Garis **val_accuracy** menunjukkan kemampuan model pada data validasi atau pengujian.
        Jika keduanya meningkat dan stabil, berarti model belajar dengan baik.
        """)
    else:
        st.warning(
            "File training_history.json belum tersedia, sehingga grafik training belum dapat ditampilkan."
        )

    st.divider()

    st.subheader("Confusion Matrix")

    if cm is not None and len(cm) > 0:
        cm_df = pd.DataFrame(
            cm,
            index=[
                "Data Asli: Plaintext",
                "Data Asli: Vigenère",
                "Data Asli: Playfair"
            ],
            columns=[
                "Prediksi: Plaintext",
                "Prediksi: Vigenère",
                "Prediksi: Playfair"
            ]
        )

        st.dataframe(cm_df, use_container_width=True)

        st.write("""
        Confusion matrix digunakan untuk melihat bagian mana yang berhasil diprediksi
        dengan benar dan bagian mana yang masih salah. Angka pada posisi diagonal
        menunjukkan prediksi yang benar. Jika terdapat angka di luar diagonal, berarti
        model masih mengalami kesalahan dalam membedakan kelas tertentu.
        """)
    else:
        st.warning("Confusion matrix belum tersedia.")

    st.divider()

    st.subheader("Precision, Recall, dan F1-Score")

    if report:
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("Classification report belum tersedia.")

    st.markdown("""
    Penjelasan sederhana:

    - **Precision**: mengukur seberapa tepat prediksi model pada suatu kelas.
    - **Recall**: mengukur seberapa baik model menemukan semua data dari kelas tertentu.
    - **F1-score**: gabungan antara precision dan recall.
    - **Support**: jumlah data yang diuji pada masing-masing kelas.
    """)

    st.write("""
    Dengan melihat accuracy, confusion matrix, precision, recall, dan F1-score,
    performa model dapat dianalisis secara lebih lengkap. Bagian ini menjadi dasar
    pembahasan pada Bab IV.
    """)


# =========================================================
# TAB 3: DATASET
# =========================================================
with tab3:
    st.subheader("Dataset yang Digunakan")

    st.write(
        "Dataset terdiri dari tiga kelas yang digunakan untuk melatih model LSTM "
        "dalam mengenali pola karakter."
    )

    st.markdown("""
    1. **Plaintext**: teks asli yang belum dienkripsi.
    2. **Vigenère**: teks hasil enkripsi menggunakan Vigenère Cipher.
    3. **Playfair**: teks hasil enkripsi menggunakan Playfair Cipher.
    """)

    st.write("Jumlah data:", len(df))

    st.write("### Contoh Data")
    st.dataframe(df.sample(20, random_state=7), use_container_width=True)

    st.write("### Distribusi Data")
    st.bar_chart(df["jenis"].value_counts())

    st.info(
        "Distribusi data yang seimbang membantu model agar tidak terlalu condong "
        "mempelajari salah satu kelas saja."
    )


# =========================================================
# TAB 4: UJI KASUS
# =========================================================
with tab4:
    st.subheader("Uji Kasus: Pemeriksaan Teks Dokumen Rahasia")

    st.write("""
    Studi kasus ini mensimulasikan penggunaan aplikasi web oleh pengguna lain untuk
    melakukan pemeriksaan pola teks pada dokumen rahasia atau kredensial pengguna.
    Aplikasi tidak digunakan untuk membuka isi pesan, mencari kunci enkripsi, atau
    melakukan dekripsi otomatis. Aplikasi digunakan untuk melihat apakah model LSTM
    dapat mengenali pola karakter dari teks yang diuji.
    """)

    case_text = st.text_area(
        "Teks studi kasus",
        value="Dokumen rahasia berisi data kredensial pengguna sistem",
        height=100
    )

    case_key = st.text_input("Kunci studi kasus", value="LSTM", key="case_key")

    case_plain = clean_for_cipher(case_text)
    case_vigenere = vigenere_encrypt(case_text, case_key)
    case_playfair = playfair_encrypt(case_text, case_key)

    st.write("### Bentuk Teks pada Studi Kasus")

    case_source_df = pd.DataFrame({
        "Jenis Teks": [
            "Plaintext",
            "Ciphertext Vigenère",
            "Ciphertext Playfair"
        ],
        "Teks yang Diuji": [
            case_plain,
            case_vigenere,
            case_playfair
        ]
    })

    st.dataframe(case_source_df, use_container_width=True)

    st.write("### Hasil Prediksi Studi Kasus")

    case_rows = []

    for jenis, teks in [
        ("Plaintext", case_plain),
        ("Ciphertext Vigenère", case_vigenere),
        ("Ciphertext Playfair", case_playfair)
    ]:
        if len(teks) == 0:
            case_rows.append({
                "Jenis Data": jenis,
                "Teks yang Diuji": teks,
                "Prediksi Model": "Tidak dapat diprediksi",
                "Plaintext": 0.0,
                "Vigenère": 0.0,
                "Playfair": 0.0
            })
        else:
            cleaned_case, pred_case, probs_case = predict_text(
                model,
                teks,
                char_to_int,
                max_len
            )

            case_rows.append({
                "Jenis Data": jenis,
                "Teks yang Diuji": teks,
                "Prediksi Model": LABEL_NAMES[pred_case],
                "Plaintext": float(probs_case[0]),
                "Vigenère": float(probs_case[1]),
                "Playfair": float(probs_case[2])
            })

    case_result_df = pd.DataFrame(case_rows)
    st.dataframe(case_result_df, use_container_width=True)

    st.write("### Analisis Studi Kasus")

    st.info("""
    Studi kasus ini menunjukkan bahwa aplikasi web tidak hanya digunakan untuk
    menghasilkan ciphertext, tetapi juga untuk menguji apakah model LSTM mampu
    mengenali pola karakter dari tiga bentuk teks. Plaintext memiliki pola bahasa
    alami, Vigenère memiliki pola pergeseran karakter berdasarkan kunci, sedangkan
    Playfair memiliki pola pasangan huruf atau digraph. Perbedaan pola tersebut
    menjadi dasar model dalam melakukan klasifikasi.
    """)

    st.warning("""
    Hasil studi kasus ini tetap dipengaruhi oleh ruang lingkup penelitian. Model
    hanya dilatih pada karakter alfabet A-Z dan cipher klasik, sehingga hasil
    prediksi dapat berubah jika input mengandung angka, simbol, tanda baca, atau
    pola teks yang berbeda dari data latihan.
    """)

    st.divider()

    st.subheader("Uji Batas Model")

    st.write("""
    Bagian ini digunakan untuk mencoba input yang berada di luar kondisi dataset utama,
    misalnya teks dengan angka, simbol, apostrophe, atau teks yang sangat pendek.
    Pengujian ini membantu menunjukkan keterbatasan teknis model.
    """)

    borderline_examples = [
        "DATA RAHASIA USER",
        "PASSWORD123",
        "USER@EMAIL.COM",
        "USER'S SECRET",
        "DATA",
        "DOKUMEN RAHASIA BERISI DATA KREDENSIAL PENGGUNA SISTEM YANG HARUS DIJAGA DENGAN BAIK"
    ]

    selected_borderline = st.selectbox(
        "Pilih contoh uji batas",
        borderline_examples
    )

    custom_borderline = st.text_input(
        "Atau masukkan teks uji batas sendiri",
        value=selected_borderline
    )

    borderline_cleaned_preview = clean_for_cipher(custom_borderline).replace("J", "I")
    borderline_removed = get_removed_characters(custom_borderline)

    st.write("### Hasil Preprocessing Uji Batas")
    st.write(f"**Input asli:** `{custom_borderline}`")
    st.write(f"**Setelah preprocessing:** `{borderline_cleaned_preview}`")
    st.write(f"**Panjang setelah preprocessing:** {len(borderline_cleaned_preview)} karakter")

    if borderline_removed:
        st.warning(
            "Teks mengandung karakter di luar alfabet A-Z. Karakter tersebut akan "
            "dihilangkan pada tahap preprocessing, sehingga sebagian informasi pada "
            "input asli tidak masuk ke model."
        )

        st.code(" ".join(sorted(set(borderline_removed))))

    if len(borderline_cleaned_preview) == 0:
        st.error("Teks tidak dapat dianalisis karena tidak memiliki karakter alfabet A-Z.")
    else:
        cleaned_borderline, pred_borderline, probs_borderline = predict_text(
            model,
            custom_borderline,
            char_to_int,
            max_len
        )

        st.success(
            f"Prediksi model untuk uji batas: **{LABEL_NAMES[pred_borderline]}**"
        )

        borderline_prob_df = pd.DataFrame({
            "Kategori": [LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            "Nilai Kemungkinan": [
                float(probs_borderline[0]),
                float(probs_borderline[1]),
                float(probs_borderline[2])
            ]
        })

        st.dataframe(borderline_prob_df, use_container_width=True)

        st.info(explain_pattern(pred_borderline))


# =========================================================
# TAB 5: KETERANGAN
# =========================================================
with tab5:
    st.subheader("Penjelasan Project")

    st.write("""
    Project ini dibuat untuk menganalisis kemampuan model LSTM dalam mengenali pola
    karakter pada tiga jenis teks, yaitu plaintext, ciphertext hasil Vigenère Cipher,
    dan ciphertext hasil Playfair Cipher.
    """)

    with st.expander("1. Apa tujuan project ini?"):
        st.write("""
        Tujuan project ini adalah menganalisis kemampuan model LSTM dalam mengenali
        pola enkripsi klasik. Model tidak digunakan untuk membobol enkripsi, mencari
        kunci, atau melakukan dekripsi otomatis, tetapi untuk mengklasifikasikan teks
        berdasarkan pola karakter yang dipelajari.
        """)

    with st.expander("2. Apakah project ini hanya encryption generator?"):
        st.write("""
        Tidak. Fitur pembentukan ciphertext hanya digunakan sebagai skenario uji dan
        bagian dari pembentukan dataset. Fokus utama project adalah klasifikasi dan
        analisis pola karakter menggunakan model LSTM.
        """)

    with st.expander("3. Apa saja kelas yang diprediksi model?"):
        st.markdown("""
        Model memprediksi teks ke dalam tiga kelas:

        1. **Plaintext**: teks asli yang belum dienkripsi.
        2. **Vigenère**: teks hasil enkripsi menggunakan Vigenère Cipher.
        3. **Playfair**: teks hasil enkripsi menggunakan Playfair Cipher.
        """)

    with st.expander("4. Apa pola yang dianalisis?"):
        st.markdown("""
        Pola yang dianalisis adalah pola urutan karakter.

        - **Plaintext** memiliki pola bahasa alami.
        - **Vigenère Cipher** memiliki pola pergeseran karakter berdasarkan kunci.
        - **Playfair Cipher** memiliki pola pasangan huruf atau digraph.
        """)

    with st.expander("5. Apa yang dilakukan model LSTM?"):
        st.write("""
        Model LSTM membaca urutan karakter dalam teks. Dari urutan tersebut,
        model mencoba mengenali apakah teks lebih mirip plaintext, ciphertext
        Vigenère, atau ciphertext Playfair berdasarkan pola yang dipelajari dari
        data training.
        """)

    with st.expander("6. Apa arti accuracy?"):
        st.write("""
        Accuracy menunjukkan seberapa sering model menjawab dengan benar.
        Jika akurasi 80%, artinya dari 100 data uji, sekitar 80 data berhasil
        diklasifikasikan dengan benar.
        """)

    with st.expander("7. Apa arti loss?"):
        st.write("""
        Loss menunjukkan tingkat kesalahan model saat belajar. Semakin kecil nilai
        loss, semakin baik model dalam menyesuaikan prediksinya terhadap label yang benar.
        """)

    with st.expander("8. Apa arti confusion matrix?"):
        st.write("""
        Confusion matrix adalah tabel yang menunjukkan hasil prediksi benar dan salah.
        Tabel ini membantu melihat kelas mana yang paling sering benar ditebak dan
        kelas mana yang masih sering tertukar.
        """)

    with st.expander("9. Apa arti precision, recall, dan F1-score?"):
        st.markdown("""
        - **Precision** menunjukkan ketepatan prediksi model.
        - **Recall** menunjukkan kemampuan model menemukan data dari suatu kelas.
        - **F1-score** menunjukkan keseimbangan antara precision dan recall.
        """)

    with st.expander("10. Apa keterbatasan project ini?"):
        st.write("""
        Project ini masih memiliki keterbatasan. Model hanya belajar dari dataset
        yang dibuat secara sintetis dan dibatasi pada karakter alfabet A-Z.
        Oleh karena itu, hasil prediksi dapat berubah apabila input mengandung angka,
        simbol, tanda baca, atau pola teks yang berbeda dari data latihan.
        Project ini juga hanya digunakan untuk cipher klasik, bukan untuk algoritma
        modern seperti AES atau RSA.
        """)

    st.success("""
    Kesimpulannya, aplikasi ini digunakan sebagai media analisis untuk melihat
    kemampuan LSTM dalam mengenali pola karakter pada Plaintext, Vigenère Cipher,
    dan Playfair Cipher.
    """)