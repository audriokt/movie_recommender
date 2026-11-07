# Movie Recommendation System

Sistem rekomendasi film berbasis **Content-Based Filtering** yang memberikan rekomendasi film berdasarkan kemiripan karakteristik dan informasi konten film.

Sistem memanfaatkan:
- **TF-IDF** untuk fitur teks (`genres` dan `plot_keywords`)
- **Word2Vec** untuk representasi actor dan director
- **MinMaxScaler** untuk fitur numerik
- **One-Hot Encoding** untuk fitur kategorikal
- **Frequency Encoding** untuk country
- **Cosine Similarity** untuk menghitung kemiripan antarfilm
- **Streamlit** sebagai frontend aplikasi

---

## Daftar Isi

- [Tentang Sistem](#-tentang-sistem)
- [Tujuan](#-tujuan)
- [Dataset](#-dataset)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Preprocessing Dataset](#-preprocessing-dataset)
- [Feature Engineering](#-feature-engineering)
- [Model dan Metode](#-model-dan-metode)
- [Combined Embedding](#-combined-embedding)
- [Sistem Rekomendasi](#-sistem-rekomendasi)
- [Penyimpanan Model](#-penyimpanan-model)
- [Struktur Project](#-struktur-project)
- [Instalasi dari GitHub](#-instalasi-dari-github)
- [Training Model](#-training-model)
- [Menjalankan Streamlit](#-menjalankan-streamlit)
- [Training vs Inference](#-training-vs-inference)
- [Konsistensi Embedding dan Metadata](#-konsistensi-embedding-dan-metadata)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Kesimpulan](#-kesimpulan)

---

# Tentang Sistem

Movie Recommendation System menggunakan pendekatan **Content-Based Filtering**.

Sistem menentukan rekomendasi berdasarkan karakteristik film, bukan berdasarkan pola interaksi pengguna.

Ketika pengguna memilih film, sistem mengambil representasi fitur film tersebut lalu membandingkannya dengan representasi film lain.

Alurnya:

```text
Film yang dipilih
       ↓
Ambil embedding film
       ↓
Bandingkan dengan embedding film lain
       ↓
Cosine Similarity
       ↓
Urutkan berdasarkan similarity
       ↓
Top-N rekomendasi
```

Semakin tinggi nilai cosine similarity, semakin mirip karakteristik kedua film.

---

# Tujuan

Sistem ini dibuat untuk:

1. Mengimplementasikan sistem rekomendasi berbasis konten.
2. Melakukan preprocessing dataset film secara sistematis.
3. Menggabungkan berbagai jenis fitur film.
4. Menggunakan TF-IDF untuk representasi fitur teks.
5. Menggunakan Word2Vec untuk representasi actor dan director.
6. Menggabungkan fitur numerik, kategorikal, dan embedding.
7. Menggunakan Cosine Similarity untuk menentukan film yang paling mirip.
8. Menyediakan antarmuka pengguna menggunakan Streamlit.
9. Memisahkan proses training dengan aplikasi frontend.
10. Menyimpan model dan embedding agar tidak perlu retraining setiap aplikasi dijalankan.

---

# Dataset

Dataset merupakan dataset metadata film yang berisi informasi seperti:

| Fitur | Deskripsi |
|---|---|
| `movie_title` | Judul film |
| `director_name` | Nama sutradara |
| `actor_1_name` | Aktor utama |
| `actor_2_name` | Aktor kedua |
| `actor_3_name` | Aktor ketiga |
| `genres` | Genre film |
| `plot_keywords` | Kata kunci plot |
| `language` | Bahasa film |
| `country` | Negara produksi |
| `content_rating` | Rating film |
| `duration` | Durasi film |
| `title_year` | Tahun rilis |

Dataset mentah ditempatkan di:

```text
data/
└── movie_metadata.csv
```

Dataset mentah tidak langsung digunakan oleh frontend. Dataset melewati preprocessing dan feature engineering terlebih dahulu.

> Pastikan penggunaan dataset mengikuti lisensi dan ketentuan sumber dataset.

---

# Arsitektur Sistem

```text
                         RAW DATASET
                              │
                              ▼
                    ┌──────────────────┐
                    │  PREPROCESSING   │
                    │ preprocessing.py │
                    └────────┬─────────┘
                             │
                             ▼
                     Processed Dataset
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      Text Features      Entity Features    Other Features
          │                  │                  │
          ▼                  ▼             ┌────┴─────┐
        TF-IDF            Word2Vec       Numeric  Categorical
          │                  │             │          │
          │                  │          Scaling       │
          │                  │                       OHE
          └──────────────────┼───────────────────────┘
                             │
                             ▼
                  Combined Embedding
                             │
                             ▼
                     embeddings.npy
                             │
                             ▼
                  Cosine Similarity
                             │
                             ▼
                    Top-N Recommendation
                             │
                             ▼
                     Streamlit Frontend
```

---

# Preprocessing Dataset

Pipeline preprocessing:

```text
Raw Dataset
     ↓
Drop Unnecessary Columns
     ↓
Remove Duplicate
     ↓
Missing Value Handling
     ↓
Content Rating Normalization
     ↓
Numeric Processing
     ↓
String Normalization
     ↓
Text Processing
     ↓
Language Grouping
     ↓
Country Frequency Encoding
     ↓
One-Hot Encoding
     ↓
Processed Dataset
```

## 1. Menghapus Kolom Tidak Relevan

Beberapa kolom yang tidak digunakan dalam model dihapus:

```text
num_critic_for_reviews
director_facebook_likes
actor_3_facebook_likes
actor_1_facebook_likes
actor_2_facebook_likes
gross
num_voted_users
cast_total_facebook_likes
facenumber_in_poster
movie_imdb_link
num_user_for_reviews
budget
imdb_score
movie_facebook_likes
aspect_ratio
```

Tujuannya mengurangi fitur yang tidak diperlukan dalam pembentukan representasi film.

## 2. Menghapus Duplikat

Data duplikat dihapus menggunakan:

```python
df.drop_duplicates()
```

## 3. Missing Value

Penanganan missing value disesuaikan dengan tipe data:

```text
color            → mode
director_name    → unknown
actor_*          → unknown
duration         → median
plot_keywords    → imputasi berbasis informasi film / unknown
country          → estimasi berbasis language / unknown
language         → estimasi berbasis country / unknown
content_rating   → unknown
```

## 4. Normalisasi Content Rating

Contoh normalisasi:

```text
Not Rated → Unrated
Passed    → Legacy_Approved
Approved  → Legacy_Approved

TV-Y      → G
TV-Y7     → G
TV-G      → G

TV-MA     → R

GP        → PG
M         → PG
TV-PG     → PG

TV-14     → PG-13

X         → NC-17
```

## 5. Penanganan Tahun

Dibuat fitur:

```text
year_unknown
```

dengan:

```text
1 → tahun asli tidak tersedia
0 → tahun tersedia
```

Tahun yang kosong kemudian diimputasi menggunakan median.

## 6. Normalisasi String

String dinormalisasi dengan:

```python
.lower()
.strip()
```

Contoh:

```text
" Action "
"ACTION"
"action"
```

menjadi:

```text
"action"
```

---

# Feature Engineering

Fitur dibagi menjadi:

```text
1. Text Features
2. Entity Features
3. Numeric Features
4. Categorical Features
```

## 1. Text Features

Terdiri dari:

```text
plot_keywords
genres
```

Data list dikonversi menjadi string, lalu digabungkan:

```text
plot_keywords + genres
```

sebagai dokumen untuk TF-IDF.

## 2. Language Grouping

Bahasa yang paling sering muncul dipertahankan sebagai kategori utama.

Bahasa yang jarang muncul dikelompokkan menjadi:

```text
other
```

Tujuannya mengurangi dimensi fitur kategorikal.

## 3. Country Frequency Encoding

Country direpresentasikan berdasarkan frekuensi kemunculannya.

Contoh:

```text
USA       → 3700
UK        → 450
France    → 120
India     → 80
```

Disimpan sebagai:

```text
country_freq
```

## 4. Numeric Features

Fitur numerik:

```text
duration
title_year
country_freq
```

Fitur numerik dinormalisasi menggunakan:

```python
MinMaxScaler()
```

## 5. Categorical Features

Fitur:

```text
content_rating
language
```

diubah dengan:

```python
OneHotEncoder(handle_unknown="ignore")
```

Pengaturan `handle_unknown="ignore"` memungkinkan data baru memiliki kategori yang tidak ditemukan saat training tanpa menyebabkan error.

---

# Model dan Metode

| Metode | Digunakan Untuk |
|---|---|
| TF-IDF | `genres` + `plot_keywords` |
| Word2Vec | actor + director |
| MinMaxScaler | fitur numerik |
| One-Hot Encoding | `content_rating` + `language` |
| Frequency Encoding | `country` |
| Cosine Similarity | kemiripan film |

---

# TF-IDF

TF-IDF digunakan untuk merepresentasikan informasi teks:

```text
plot_keywords
genres
```

Implementasi:

```python
TfidfVectorizer()
```

Alur:

```text
Genre + Plot Keywords
        ↓
     Text Data
        ↓
 TfidfVectorizer
        ↓
  TF-IDF Matrix
```

Setiap film memperoleh vector TF-IDF berdasarkan kata-kata yang terdapat pada genre dan plot keywords.

---

# Word2Vec

Word2Vec digunakan untuk:

```text
director_name
actor_1_name
actor_2_name
actor_3_name
```

Konfigurasi:

```python
Word2Vec(
    films_entities,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1
)
```

Parameter:

| Parameter | Nilai |
|---|---:|
| `vector_size` | 50 |
| `window` | 3 |
| `min_count` | 1 |
| `sg` | 1 |

`sg=1` menggunakan metode Skip-Gram.

Untuk setiap film, vector actor dan director yang tersedia dirata-ratakan sehingga menghasilkan entity embedding berdimensi 50.

---

# Combined Embedding

Seluruh fitur digabungkan menjadi satu vector:

```text
Combined Embedding
│
├── TF-IDF
├── Word2Vec
├── duration
├── title_year
├── country_freq
├── content_rating
└── language
```

Implementasi:

```python
combined_embeddings = np.hstack([
    tfidf_normalized,
    entity_embeddings,
    numeric_features,
    categorical_features
])
```

TF-IDF dinormalisasi sebelum digabungkan.

Hasil akhirnya adalah satu vector representasi untuk setiap film.

---

# Cosine Similarity

Kemiripan dihitung menggunakan Cosine Similarity:

```text
                  A · B
cosine(A,B) = -------------
               ||A|| ||B||
```

Secara umum:

```text
1   → sangat mirip
0   → semakin tidak mirip
```

Contoh:

```text
Film yang dipilih:
Avatar

Hasil similarity:

Avatar                    1.000
John Carter                0.821
Guardians of the Galaxy    0.798
Star Trek                  0.764
Interstellar               0.731
```

Film dengan nilai similarity tertinggi dipilih sebagai rekomendasi.

---

# Penyimpanan Model

Agar aplikasi tidak perlu melakukan training setiap kali dijalankan:

```text
backend/artifacts/
├── preprocessing.joblib
├── tfidf.joblib
├── word2vec.model
├── embeddings.npy
├── movies.csv
└── feature_metadata.joblib
```

### `preprocessing.joblib`

Menyimpan parameter preprocessing yang telah di-fit, misalnya:

```text
median_duration
median_year
mode_color
top_languages
country_freq_map
duration_scaler
year_scaler
content_rating_encoder
language_encoder
```

### `tfidf.joblib`

Model TF-IDF yang telah di-fit pada dataset training.

Pada inference digunakan:

```python
tfidf.transform(...)
```

bukan melakukan fit ulang.

### `word2vec.model`

Model Word2Vec yang telah dilatih.

### `embeddings.npy`

Combined embedding seluruh film. File ini mempercepat proses rekomendasi karena embedding tidak perlu dihitung ulang saat aplikasi dijalankan.

### `movies.csv`

Metadata film dengan urutan baris yang harus sama dengan `embeddings.npy`.

### `feature_metadata.joblib`

Informasi dimensi dan konfigurasi fitur untuk pemeriksaan dan debugging.

---

# Struktur Project

```text
movie-recommendation-system/
│
├── data/
│   └── movie_metadata.csv
│
├── backend/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── recommender.py
│   │
│   └── artifacts/
│       ├── preprocessing.joblib
│       ├── tfidf.joblib
│       ├── word2vec.model
│       ├── embeddings.npy
│       ├── movies.csv
│       └── feature_metadata.joblib
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   └── recommended-sys-01.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Instalasi dari GitHub

## 1. Clone Repository

```bash
https://github.com/audriokt/movie_recommender.git
```

Kemudian:

```bash
cd movie_recommender
```

## 2. Membuat Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Jika `requirements.txt` belum tersedia:

```bash
pip install numpy pandas scikit-learn gensim joblib streamlit
```

---

# Training Model

Training diperlukan jika artifacts belum tersedia atau dataset berubah.

Pastikan:

```text
data/movie_metadata.csv
```

tersedia.

Kemudian:

```bash
python backend/train.py
```

Pipeline:

```text
movie_metadata.csv
        ↓
preprocessing.py
        ↓
TF-IDF
        ↓
Word2Vec
        ↓
Numeric Features
        ↓
Categorical Features
        ↓
Combined Embedding
        ↓
Save Artifacts
```

Output:

```text
backend/artifacts/
├── preprocessing.joblib
├── tfidf.joblib
├── word2vec.model
├── embeddings.npy
├── movies.csv
└── feature_metadata.joblib
```

Jika artifacts sudah tersedia dan berasal dari dataset yang sama, training tidak perlu diulang.

---

# ▶️ Menjalankan Streamlit

Setelah artifacts tersedia:

```bash
streamlit run frontend/app.py
```

Aplikasi biasanya dapat diakses melalui:

```text
http://localhost:8501
```

Buka alamat tersebut di browser.

---

# 🎬 Alur Penggunaan Aplikasi

```text
Buka aplikasi
     ↓
Pilih film
     ↓
Pilih jumlah rekomendasi
     ↓
Ambil embedding film
     ↓
Hitung Cosine Similarity
     ↓
Urutkan hasil
     ↓
Tampilkan Top-N rekomendasi
```

Contoh:

```text
Film:
Interstellar

Jumlah rekomendasi:
5
```

Aplikasi menampilkan film dengan tingkat kemiripan tertinggi.

---

# 🔄 Training vs Inference

## Training

Dilakukan ketika:

- Dataset berubah
- Dataset diperbarui
- Preprocessing berubah
- Konfigurasi TF-IDF berubah
- Konfigurasi Word2Vec berubah
- Model ingin diperbarui

Perintah:

```bash
python backend/train.py
```

## Inference

Saat aplikasi digunakan, model tidak dilatih ulang.

Aplikasi hanya:

```text
Load Artifacts
      ↓
Pilih Film
      ↓
Ambil Embedding
      ↓
Cosine Similarity
      ↓
Recommendation
```

Tidak dilakukan:

```text
TF-IDF.fit()
Word2Vec training
Scaler.fit()
Encoder.fit()
```

Dengan demikian aplikasi dapat berjalan lebih cepat.

---

# 🆕 Jika Dataset Diubah

Jika:

```text
data/movie_metadata.csv
```

diganti atau diperbarui, jalankan:

```bash
python backend/train.py
```

Jangan hanya mengganti `movies.csv`.

Seluruh artifacts harus dibuat dari dataset dan pipeline yang sama:

```text
Dataset Baru
     ↓
Preprocessing Baru
     ↓
TF-IDF Baru
     ↓
Word2Vec Baru
     ↓
Embedding Baru
     ↓
movies.csv Baru
```

---

# ⚠️ Konsistensi `embeddings.npy` dan `movies.csv`

Urutan film harus identik.

Contoh:

```text
movies.csv

index 0 → Avatar
index 1 → Titanic
index 2 → Interstellar
```

maka:

```text
embeddings.npy

index 0 → embedding Avatar
index 1 → embedding Titanic
index 2 → embedding Interstellar
```

Jangan melakukan sorting atau filtering `movies.csv` secara manual setelah training.

Jika metadata berubah, jalankan kembali:

```bash
python backend/train.py
```

---

# 🔒 `.gitignore`

Contoh `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environment
venv/
.venv/

# Environment
.env

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Dataset besar
data/*.csv

# Model artifacts besar
backend/artifacts/*.npy
backend/artifacts/*.model
backend/artifacts/*.joblib
backend/artifacts/*.pkl

# Temporary files
*.tmp
```

Jika dataset atau artifacts diperlukan saat deployment, gunakan penyimpanan yang sesuai seperti Git LFS atau model/data storage.

---

# 🌐 Deployment

Aplikasi dapat dijalankan secara lokal maupun dideploy pada platform yang mendukung Streamlit.

Arsitektur deployment:

```text
GitHub Repository
        ↓
Streamlit Hosting / Server
        ↓
frontend/app.py
        ↓
backend/recommender.py
        ↓
backend/artifacts/
```

Pastikan repository memiliki:

```text
requirements.txt
```

agar dependency dapat di-install pada environment deployment.

Jika artifacts tidak disimpan di repository, deployment harus memiliki mekanisme untuk mengambil artifacts sebelum aplikasi dijalankan.

---

# 🧩 Pemisahan Frontend dan Backend

## Frontend

Lokasi:

```text
frontend/app.py
```

Bertanggung jawab terhadap:

- Tampilan aplikasi
- Input pengguna
- Pemilihan film
- Jumlah rekomendasi
- Menampilkan hasil

## Backend

Lokasi:

```text
backend/
├── preprocessing.py
├── train.py
└── recommender.py
```

Bertanggung jawab terhadap:

- Preprocessing
- Loading model
- Transformasi fitur
- Embedding
- Similarity
- Recommendation

Pemisahan ini membuat project lebih terstruktur dan memudahkan pengembangan.

---

# 🧪 Workflow Developer

```bash
# 1. Clone repository
https://github.com/audriokt/movie_recommender.git

# 2. Masuk project
cd movie_recommender

# 3. Buat virtual environment
python -m venv venv

# 4. Aktifkan environment - Windows
venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Pastikan dataset tersedia
# data/movie_metadata.csv

# 7. Training model
python backend/train.py

# 8. Jalankan aplikasi
streamlit run frontend/app.py
```

---

# 🧠 Ringkasan Pipeline

```text
                    RAW DATASET
                         │
                         ▼
                ┌─────────────────┐
                │  PREPROCESSING  │
                └────────┬────────┘
                         │
                         ▼
                Processed Dataset
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
        Text           Entity        Numeric/
       Features        Features     Categorical
          │              │              │
          ▼              ▼              ▼
       TF-IDF         Word2Vec       Scaling/OHE
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                Combined Embedding
                         │
                         ▼
                  embeddings.npy
                         │
                         ▼
                Cosine Similarity
                         │
                         ▼
                  Top-N Movies
                         │
                         ▼
                Streamlit Frontend
```

---

# 📌 Kesimpulan

Movie Recommendation System ini menggunakan pendekatan **Content-Based Filtering** dengan menggabungkan berbagai karakteristik film.

Representasi film dibentuk dari:

```text
TF-IDF
   +
Word2Vec
   +
Numeric Features
   +
Categorical Features
```

Seluruh fitur digabungkan menjadi:

```text
Combined Embedding
```

Kemudian sistem menggunakan:

```text
Cosine Similarity
```

untuk menentukan film yang paling mirip.

Pipeline utama:

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
TF-IDF + Word2Vec
   ↓
Combined Embedding
   ↓
Cosine Similarity
   ↓
Movie Recommendation
   ↓
Streamlit
```

Model dan parameter preprocessing disimpan sebagai artifacts sehingga aplikasi dapat melakukan inference tanpa melakukan training ulang setiap kali dijalankan.

---

# 👤 Author

**Nama:** Audrio

**Project:** Movie Recommendation System

**Method:** Content-Based Filtering

**Machine Learning:** TF-IDF + Word2Vec + Cosine Similarity

**Framework:** Streamlit

**Programming Language:** Python

---

# 📄 License

```text
MIT License
```
