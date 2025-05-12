# 🎬 Sentiment Classification on Movie Reviews

Proyek ini membandingkan performa tiga algoritma machine learning dalam mengklasifikasikan sentimen dari ulasan film:

- Random Forest
- Support Vector Machine (SVM)
- Neural Network (Keras)

## 📦 Prasyarat

- Python 3.8 atau lebih baru

## ⚙️ Instalasi

1. Buat virtual environment:

   ```bash
   python -m venv eda_env
   ```

2. Aktifkan environment:

   ```bash
   eda_env\Scripts\activate  # Windows
   source eda_env/bin/activate  # Linux/Mac
   ```

3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset

Pastikan file dataset berada di:

```
dataset/movie.csv
```

## 🔎 Exploratory Data Analysis (EDA)

- Menampilkan distribusi label sentimen:
  - `0`: Negatif
  - `1`: Positif
- Visualisasi dengan:
  - Bar Chart
  - Pie Chart
- WordCloud dan fitur penting berdasarkan TF-IDF.

## 🧹 Preprocessing

- Menghapus HTML, emoji, karakter non-ASCII
- Normalisasi teks: lowercase, hilangkan tanda baca dan spasi ganda

## 🔀 Split Dataset

- Dataset dibagi menjadi 70% data latih dan 30% data uji
- Menggunakan stratifikasi untuk menjaga proporsi label

## 🔠 TF-IDF Vectorization

- TF-IDF digunakan untuk mengubah teks menjadi fitur numerik
- Parameter:
  ```python
  max_features=15000, min_df=2, max_df=0.85, ngram_range=(1, 2)
  ```

## 🧠 Model Training

### 1. Random Forest

- 300 estimators, depth = 20

### 2. Support Vector Machine (LinearSVC)

- Cocok untuk dataset besar, digunakan `LinearSVC` untuk efisiensi

### 3. Neural Network (Keras)

- 2 hidden layers + dropout & batch normalization
- Softmax output layer (3 kelas)
- Adam optimizer dengan `learning_rate=0.0005`
- Epoch = 15, Batch size = 64

## 📊 Visualisasi

- Confusion matrix (dalam bentuk heatmap dan pie chart)
- Top 20 TF-IDF words per class
- WordCloud untuk masing-masing label
- Grafik akurasi dan loss selama training NN

## ▶️ Menjalankan Script

```bash
python main.py
```

## ✅ Evaluasi dan Hasil Akurasi

### Setup 1

- Epoch = 10, Batch Size = 128, LR = 0.001
- TF-IDF: `max_features=10000, min_df=3, max_df=0.9`

| Model          | Akurasi |
| -------------- | ------- |
| Random Forest  | 0.8503  |
| SVM            | 0.8922  |
| Neural Network | 0.8840  |

### Setup 2

- Epoch = 15, Batch Size = 64, LR = 0.0005
- TF-IDF: `max_features=15000, min_df=2, max_df=0.85`

| Model          | Akurasi |
| -------------- | ------- |
| Random Forest  | 0.8449  |
| SVM            | 0.9006  |
| Neural Network | 0.8862  |

🏆 **Best Model: SVM**

---

## 🗂 Struktur Folder Output

- `plots/label_distribution.png`
- `plots/pie_label_distribution.png`
- `plots/top20_tfidf_[label].png`
- `plots/wordcloud_[label].png`
- `plots/confusion_matrix_[model].png`
- `plots/confusion_pie_[model].png`
- `plots/nn_training_progress.png`

---

## ✍️ Catatan

Pastikan folder `plots/` sudah dibuat sebelum menjalankan script, atau tambahkan perintah `os.makedirs('plots', exist_ok=True)` di awal script untuk otomatis membuatnya.

## 📘 Dokumentasi Tambahan

👉 [Lihat penjelasan teknis lengkap di EXPLAINATION.MD](docs/EXPLAINATION.MD)
