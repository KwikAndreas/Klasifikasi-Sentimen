# 📘 EXPLAINATION.md

Dokumentasi ini menjelaskan secara mendalam seluruh bagian script Python dalam proyek klasifikasi sentimen ulasan film. Fokus utama juga diberikan pada algoritma **Random Forest, SVM, dan Neural Network**, termasuk alasan pemilihan parameter dan pendekatan yang digunakan.

---

## 1. 📦 Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, os
from bs4 import BeautifulSoup
import emoji
```
- `pandas`, `numpy`: manipulasi data dan array.
- `matplotlib`, `seaborn`: visualisasi data.
- `re`, `string`: cleaning text.
- `os`: manajemen file (digunakan untuk menyimpan plot).
- `BeautifulSoup`: menghapus tag HTML dari teks.
- `emoji`: menghapus emoji dari teks.

```python
from sklearn... , from tensorflow... , from wordcloud import WordCloud
```
- `sklearn`: untuk preprocessing, modeling, evaluasi.
- `tensorflow.keras`: membangun dan melatih neural network.
- `wordcloud`: visualisasi kata penting.

---

## 2. 📥 Load Dataset

```python
df = pd.read_csv('dataset/movie.csv')
```
- Memuat dataset CSV ke dalam DataFrame.
- Diasumsikan memiliki kolom `text` dan `label`.

---

## 3. 📊 Distribusi Label

```python
sns.countplot(x='label', ...)
plt.pie(...)
```
- Menampilkan distribusi label untuk mengetahui keseimbangan data.

---

## 4. 🧼 Preprocessing Teks

```python
def clean_text(text): ...
```
- Menghapus HTML, emoji, simbol, dan karakter non-ASCII.
- Case folding dan normalisasi spasi.
- Ini penting untuk hasil vektorisasi TF-IDF yang konsisten.

---

## 5. ✂️ Train-Test Split

```python
train_test_split(..., stratify=df['label'])
```
- Membagi data secara seimbang berdasarkan label.
- Test size = 30% untuk pengujian model.

---

## 6. 🔡 TF-IDF Vectorization

```python
tfidf = TfidfVectorizer(...)
```
- `max_features=15000`: membatasi jumlah fitur untuk efisiensi.
- `min_df=2`: hanya term yang muncul di minimal 2 dokumen.
- `max_df=0.85`: menghindari term yang terlalu umum.
- `ngram_range=(1,2)`: mempertimbangkan unigram dan bigram.

---

## 7. 📈 Top 20 Kata Berdasarkan TF-IDF

```python
get_top_tfidf_words(...)
```
- Mengambil kata-kata yang paling bernilai untuk setiap kelas.
- Digunakan untuk interpretabilitas model.

---

## 8. 🌥 WordCloud

```python
WordCloud().generate(text)
```
- Membuat visualisasi term yang paling sering muncul per kelas.

---

## 9. 🔢 One-Hot Encoding untuk Neural Network

```python
to_categorical(...)
```
- Model NN menggunakan `categorical_crossentropy`, maka label harus dalam format one-hot.

---

# 🔍 Penjelasan Algoritma

## 🐾 Random Forest Classifier

```python
RandomForestClassifier(n_estimators=300, max_depth=20)
```
- **n_estimators=300**: jumlah pohon yang digunakan. Lebih banyak pohon = stabilitas lebih tinggi, tapi lebih lambat.
- **max_depth=20**: membatasi kedalaman pohon agar tidak overfitting.
- Cocok untuk dataset besar dan tidak perlu normalisasi fitur.

### ➕ Kelebihan:
- Robust terhadap noise dan outlier.
- Tidak sensitif terhadap distribusi data.

### ➖ Kekurangan:
- Lebih lambat dibanding SVM untuk inference.
- Interpretasi lebih kompleks.

---

## ⚙️ Support Vector Machine (LinearSVC)

```python
LinearSVC(C=0.5, max_iter=2000)
```
- **LinearSVC** dipilih daripada `SVC` karena lebih efisien untuk dataset besar.
- **C=0.5**: regularisasi. Semakin kecil nilai C, semakin kuat regularisasi.
- **max_iter=2000**: menghindari tidak konvergen saat training.

### ➕ Kelebihan:
- Performa sangat baik untuk data teks (high dimensional).
- Cepat untuk training dan prediksi pada data besar.

### ➖ Kekurangan:
- Tidak memberikan probabilitas (kecuali dikombinasikan dengan `CalibratedClassifierCV`).
- Tidak cocok jika hubungan antara fitur dan target sangat non-linear.

---

## 🧠 Neural Network (Keras Sequential)

```python
Sequential([...])
```
- 3 layer utama: 512 → 256 → 3.
- Aktivasi: `relu` untuk hidden layer, `softmax` untuk output.
- Dropout 0.4 dan BatchNormalization untuk mencegah overfitting.
- Loss function: `categorical_crossentropy`.

### Hyperparameter:
- **Epoch = 15**: jumlah iterasi penuh ke seluruh data.
- **Batch size = 64**: jumlah sample tiap mini-batch.
- **Learning rate = 0.0005**: digunakan oleh Adam optimizer.

### ➕ Kelebihan:
- Dapat menangkap pola kompleks dan non-linear.
- Arsitektur dapat dikustomisasi lebih lanjut.

### ➖ Kekurangan:
- Butuh preprocessing (konversi sparse matrix → dense array).
- Training lebih lama.
- Rentan overfitting jika data tidak cukup besar atau dropout tidak disetel dengan benar.

---

## 📊 Evaluasi Model

```python
accuracy_score, classification_report, confusion_matrix
```
- Metrik utama: akurasi.
- Disertai juga visualisasi confusion matrix dan pie chart untuk tiap model.

---

## 📉 Visualisasi Proses Training NN

```python
plt.plot(plot_callback.history[...])
```
- Menampilkan akurasi dan loss selama epoch.
- Digunakan untuk memantau konvergensi model.

---

# 🏁 Kesimpulan
- SVM adalah model terbaik dengan akurasi tertinggi pada dua setup.
- Neural Network mendekati performa SVM namun membutuhkan lebih banyak resource.
- Random Forest stabil tapi kurang kompetitif dalam dimensi tinggi seperti TF-IDF.

---


---

# 🧠 Pengertian & Cara Kerja Model

## 🌲 Random Forest - Konsep & Cara Kerja

**Random Forest** adalah ensemble learning method berbasis decision tree. Model ini membangun banyak decision tree dan menggabungkan hasilnya (via majority voting untuk klasifikasi) untuk meningkatkan akurasi dan mengurangi overfitting.

### Cara Kerja:
1. Setiap pohon dilatih pada subset data secara acak (bootstrap sampling).
2. Di setiap split, hanya subset acak dari fitur yang dipertimbangkan (menciptakan variasi).
3. Output akhir diambil dari voting mayoritas seluruh pohon.

### Sifat:
- Bersifat non-linear.
- Mampu menangani missing data dan fitur kategorik.
- Cenderung robust terhadap overfitting dibanding single decision tree.

---

## 📐 Support Vector Machine (SVM) - Konsep & Cara Kerja

**SVM** mencari hyperplane terbaik untuk memisahkan data berdasarkan label. Untuk data teks (dimensi tinggi), SVM sangat efektif karena memfokuskan pada margin optimal antar kelas.

### Cara Kerja:
1. SVM memproyeksikan data ke ruang dimensi tinggi (high-dimensional space).
2. Mencari hyperplane (garis pemisah) dengan margin terbesar antar kelas.
3. Data di luar margin disebut support vectors.

### Tipe yang Digunakan: `LinearSVC`
- Khusus linear kernel untuk efisiensi pada data teks besar.
- Tidak menghitung kernel transform eksplisit — cocok untuk TF-IDF.

---

## 🔮 Neural Network - Konsep & Cara Kerja

Neural Network adalah model non-linear yang terinspirasi dari neuron biologis. Ia belajar dari data melalui propagasi bobot (weights) dan aktivasi.

### Cara Kerja (Feedforward NN):
1. Input layer menerima fitur (TF-IDF vector).
2. Hidden layer memproses melalui aktivasi (`ReLU`) dan dropout.
3. Output layer (3 neuron) menggunakan `softmax` untuk menghasilkan probabilitas 3 kelas.
4. Backpropagation digunakan untuk menyesuaikan bobot berdasar loss `categorical_crossentropy`.

### Arsitektur dalam Script:
- 512 neuron → 256 neuron → 3 kelas output.
- Dropout = 0.4 untuk regularisasi.
- BatchNormalization membantu stabilisasi training.

---

