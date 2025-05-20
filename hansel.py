import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM

df = pd.read_csv('dataset\movie.csv')

print(df.columns)
print(df.head())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label')
plt.title('Distribusi Label Sentimen')
plt.savefig('eda_label_distribution.png')
plt.show()

df['text_length'] = df['text'].astype(str).apply(len)
print(f"Rata-rata panjang teks: {df['text_length'].mean():.2f}")
plt.hist(df['text_length'], bins=50)
plt.title('Distribusi Panjang Teks')
plt.xlabel('Jumlah Karakter')
plt.ylabel('Jumlah Dialog')
plt.savefig('eda_text_length.png')
plt.show()

def general_data(text):
    text=text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['general_data'] = df['text'].astype(str).apply(general_data)

# Gunakan kolom general_data yang sudah dibersihkan
texts = df['general_data'].astype(str).tolist()
labels = df['label'].tolist()

# Tokenisasi
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)  # Batasi panjang sequence

# Encode label
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, one_hot_labels, test_size=0.3, random_state=42
)


model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Latih model dengan EarlyStopping
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=64,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
)

model.save('models/model.h5')

# Evaluasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi: {accuracy:.2f}")

plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Model')
plt.ylabel('Akurasi')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('model_accuracy.png')
plt.show()