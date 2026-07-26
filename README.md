# Klasifikasi Pola Karakter Plaintext dan Ciphertext Menggunakan BiLSTM

Repository ini berisi implementasi model **Bidirectional Long Short-Term Memory (BiLSTM)** untuk mengklasifikasikan pola karakter pada tiga kelas teks, yaitu **plaintext**, **ciphertext hasil Vigenère Cipher**, dan **ciphertext hasil Playfair Cipher**. Proyek ini berfokus pada analisis pola karakter, bukan pada proses dekripsi, pencarian kunci enkripsi, atau pemecahan algoritma kriptografi.

## Deskripsi Proyek

Kriptografi klasik merupakan salah satu metode yang digunakan untuk mengubah pesan asli menjadi bentuk yang sulit dipahami. Pada penelitian ini digunakan dua algoritma kriptografi klasik, yaitu **Vigenère Cipher** dan **Playfair Cipher**. Vigenère Cipher memiliki pola enkripsi berupa pergeseran karakter berdasarkan kunci, sedangkan Playfair Cipher memiliki pola enkripsi berbasis pasangan huruf atau **digraph** melalui matriks 5 × 5.

Perbedaan pola tersebut menjadi dasar dalam proses klasifikasi menggunakan model **BiLSTM**. Model BiLSTM digunakan karena mampu memproses data sekuensial dari dua arah, yaitu arah maju dan arah mundur. Dengan pendekatan berbasis karakter, model dilatih untuk mengenali perbedaan pola antara plaintext, ciphertext Vigenère Cipher, dan ciphertext Playfair Cipher.

Repository ini juga dilengkapi dengan aplikasi web sederhana berbasis **Streamlit** yang digunakan untuk menampilkan hasil klasifikasi, probabilitas prediksi, hasil evaluasi model, grafik pelatihan, dan confusion matrix.

## Kelas Klasifikasi

| Label | Kelas                      |
| ----- | -------------------------- |
| 0     | Plaintext                  |
| 1     | Ciphertext Vigenère Cipher |
| 2     | Ciphertext Playfair Cipher |

## Dataset

Dataset yang digunakan merupakan dataset sintetis yang dibuat secara mandiri dan terkontrol. Dataset terdiri dari 7.500 data dengan pembagian sebagai berikut:

| Kelas                      | Jumlah Data |
| -------------------------- | ----------: |
| Plaintext                  |       2.500 |
| Ciphertext Vigenère Cipher |       2.500 |
| Ciphertext Playfair Cipher |       2.500 |
| **Total**                  |   **7.500** |

Dataset melalui beberapa tahapan, yaitu pembentukan plaintext sintetis, normalisasi teks, enkripsi menggunakan Vigenère Cipher, enkripsi menggunakan Playfair Cipher, pelabelan data, validasi dataset, encoding karakter, dan padding sequence.

Karakter yang digunakan dibatasi pada alfabet **A–Z**. Rentang panjang teks utama berada pada **18 sampai 120 karakter** setelah proses preprocessing. Dataset kemudian dibagi menjadi 80% data training dan 20% data testing, sehingga terdapat 6.000 data training dan 1.500 data testing.

## Arsitektur Model

Arsitektur model yang digunakan adalah **Bidirectional Long Short-Term Memory (BiLSTM)** berbasis karakter. Tahapan arsitektur model terdiri dari input teks, normalisasi teks, encoding karakter, padding sequence, embedding layer, Bidirectional LSTM layer, dropout layer, dense layer, dropout layer, softmax layer, dan output kelas.

Konfigurasi utama model:

| Komponen            | Nilai                           |
| ------------------- | ------------------------------- |
| Model               | BiLSTM                          |
| Embedding Dimension | 64                              |
| Unit BiLSTM         | 96                              |
| Dropout pertama     | 0.35                            |
| Dense Layer         | 64 unit, ReLU                   |
| Dropout kedua       | 0.25                            |
| Output Layer        | 3 unit, Softmax                 |
| Optimizer           | Adam                            |
| Loss Function       | Sparse Categorical Crossentropy |
| Epoch maksimum      | 25                              |
| Batch Size          | 32                              |

## Hasil Evaluasi

Model dievaluasi menggunakan data testing sebanyak 1.500 data. Hasil evaluasi menunjukkan bahwa model BiLSTM memperoleh accuracy sebesar **99%**, dengan 1.485 data berhasil diklasifikasikan dengan benar dan 15 data mengalami kesalahan klasifikasi.

| Metrik         | Nilai |
| -------------- | ----: |
| Accuracy       |   99% |
| Data Testing   | 1.500 |
| Prediksi Benar | 1.485 |
| Prediksi Salah |    15 |

Hasil confusion matrix pada data testing:

| Data Asli       | Prediksi Plaintext | Prediksi Vigenère | Prediksi Playfair |
| --------------- | -----------------: | ----------------: | ----------------: |
| Plaintext       |                500 |                 0 |                 0 |
| Vigenère Cipher |                  1 |               486 |                13 |
| Playfair Cipher |                  0 |                 1 |               499 |

Kesalahan klasifikasi paling banyak terjadi antara kelas **Vigenère Cipher** dan **Playfair Cipher**. Hal ini menunjukkan bahwa membedakan dua jenis ciphertext merupakan bagian yang lebih sulit dibandingkan membedakan plaintext dan ciphertext.

## Fitur Aplikasi Web

Aplikasi web dibuat menggunakan **Streamlit** dan memiliki beberapa fitur, yaitu:

* Input teks untuk klasifikasi.
* Prediksi kelas teks.
* Tampilan probabilitas setiap kelas.
* Informasi hasil model.
* Visualisasi grafik akurasi training dan validasi.
* Visualisasi grafik loss training dan validasi.
* Tampilan confusion matrix.
* Penjelasan kelas plaintext, Vigenère Cipher, dan Playfair Cipher.
* Uji kasus dan uji batas input teks.

## Teknologi yang Digunakan

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

## Struktur Folder

Struktur folder pada repository ini dapat disusun sebagai berikut:

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset_bilstm_vigenere_playfair.csv
├── model/
│   └── bilstm_cipher_model.keras
├── output/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── training_loss.png
└── notebooks/
    └── training_model_bilstm.ipynb
```

## Cara Instalasi dan Penggunaan Aplikasi

Clone repository terlebih dahulu:

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

Ganti `username` dan `nama-repository` sesuai dengan akun GitHub dan nama repository yang digunakan.

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment.

Untuk Windows:

```bash
venv\Scripts\activate
```

Untuk macOS atau Linux:

```bash
source venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

Setelah perintah dijalankan, Streamlit akan menampilkan alamat lokal seperti berikut:

```text
Local URL: http://localhost:8501
```

Buka alamat tersebut melalui browser untuk menggunakan aplikasi.

Setelah aplikasi terbuka, pengguna dapat memasukkan teks ke dalam kolom input yang tersedia. Contoh input plaintext:

```text
DOKUMEN RAHASIA BERISI DATA USER
```

Contoh input ciphertext:

```text
NSVEOXVRKJXWJWSYFE
```

Aplikasi akan melakukan preprocessing terhadap teks yang dimasukkan. Proses preprocessing meliputi mengubah huruf menjadi kapital, menghapus spasi, menghapus karakter selain alfabet A–Z, mengubah teks menjadi urutan karakter, mengubah karakter menjadi angka, dan melakukan padding sequence agar panjang input sesuai dengan model.

Setelah teks dimasukkan, pengguna dapat menekan tombol prediksi atau klasifikasi. Model BiLSTM akan memproses input dan menghasilkan salah satu dari tiga kelas, yaitu **Plaintext**, **Ciphertext Vigenère Cipher**, atau **Ciphertext Playfair Cipher**.

Selain hasil prediksi, aplikasi juga dapat menampilkan probabilitas dari masing-masing kelas. Kelas dengan nilai probabilitas tertinggi akan dipilih sebagai hasil prediksi akhir model.

Contoh hasil probabilitas:

| Kelas           | Probabilitas |
| --------------- | -----------: |
| Plaintext       |         0.02 |
| Vigenère Cipher |         0.95 |
| Playfair Cipher |         0.03 |

Berdasarkan contoh tersebut, model memilih kelas **Vigenère Cipher** karena memiliki nilai probabilitas paling tinggi.

Pengguna juga dapat melihat hasil evaluasi model, seperti accuracy, precision, recall, F1-score, confusion matrix, grafik akurasi training dan validasi, serta grafik loss training dan validasi. Informasi tersebut digunakan untuk menunjukkan performa model selama proses pelatihan dan pengujian.

Aplikasi juga dapat digunakan untuk melakukan uji kasus terhadap tiga bentuk teks, yaitu plaintext, ciphertext hasil Vigenère Cipher, dan ciphertext hasil Playfair Cipher. Selain itu, pengguna dapat melakukan uji batas dengan memasukkan teks yang sangat pendek, teks yang terlalu panjang, teks yang mengandung angka, simbol, tanda baca, atau spasi.

Namun, model hanya memproses karakter alfabet A–Z. Angka, simbol, tanda baca, dan spasi akan dihapus pada tahap preprocessing.

Contoh hasil preprocessing:

| Input Awal                              | Setelah Preprocessing |
| --------------------------------------- | --------------------- |
| DATA RAHASIA                            | DATARAHASIA           |
| PASSWORD123                             | PASSWORD              |
| [USER@EMAIL.COM](mailto:USER@EMAIL.COM) | USEREMAILCOM          |

## Batasan Proyek

Model pada repository ini memiliki beberapa batasan, yaitu:

1. Model hanya digunakan untuk klasifikasi pola karakter.
2. Model tidak digunakan untuk dekripsi atau mencari kunci enkripsi.
3. Dataset yang digunakan merupakan dataset sintetis.
4. Karakter yang digunakan dibatasi pada alfabet A–Z.
5. Rentang utama panjang teks adalah 18 sampai 120 karakter.
6. Model hanya mengklasifikasikan tiga kelas, yaitu plaintext, Vigenère Cipher, dan Playfair Cipher.
7. Hasil akurasi tinggi berlaku dalam ruang lingkup dataset penelitian yang terkontrol.

## Catatan

Hasil akurasi yang tinggi perlu dipahami dalam ruang lingkup dataset penelitian. Dataset yang digunakan merupakan dataset sintetis yang dibuat secara terkontrol, sehingga performa model belum tentu sama apabila diuji menggunakan data eksternal dengan variasi teks, karakter, atau pola bahasa yang berbeda.

Aplikasi ini dibuat sebagai bagian dari penelitian Tugas Akhir dan digunakan untuk mendukung analisis klasifikasi pola karakter pada plaintext dan ciphertext hasil kriptografi klasik.

## Author

**Hafizhan Noor Amril**
Program Studi Informatika
Fakultas Teknologi Industri
Universitas Gunadarma
