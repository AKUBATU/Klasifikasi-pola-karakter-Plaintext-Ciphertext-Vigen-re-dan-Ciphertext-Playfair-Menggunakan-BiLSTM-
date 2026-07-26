# LSTM Cipher Web App - Dataset Terpisah

Versi ini mempertahankan tampilan tab seperti versi awal:

- Uji Teks
- Hasil Model
- Dataset
- Keterangan

Perubahan hanya pada dataset: dataset tidak lagi dibuat langsung di `app.py`, tetapi dibaca dari file CSV:

```text
data/dataset_lstm_vigenere_playfair.csv
```

Jumlah dataset tetap 7.500 data:

- Plaintext / Teks Biasa: 2.500
- Ciphertext Vigenere: 2.500
- Ciphertext Playfair: 2.500

## Cara menjalankan

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Catatan

Pada saat pertama kali dibuka, aplikasi tetap melakukan training model seperti versi awal. Setelah itu hasil training tersimpan di cache Streamlit selama sesi berjalan.
