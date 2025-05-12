import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from wordcloud import WordCloud

# === Load Data ===
print("[READ] Checking...")
df = pd.read_csv('dataset/movie.csv')
print("[INFO] Data loaded.")

# === EDA ===
os.makedirs("plots", exist_ok=True)
print("Label distribution:")
print(df['label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Label Distribution")
plt.savefig("plots/label_distribution.png")
plt.close()

# === Pie Chart Distribution ===
label_counts = df['label'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Sentiment Distribution (Pie)")
plt.savefig("plots/pie_label_distribution.png")
plt.close()

# === Preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(clean_text)

# === Train-test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.3, stratify=df['label'], random_state=42
)

# === TF-IDF ===
print("[STARTING] TF-IDF Vectorization.")
tfidf = TfidfVectorizer(max_features=10000, min_df=3, max_df=0.9, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print("[INFO] Finished TF-IDF vectorization.")

# === Top 20 TF-IDF Words (Positive & Neutral) ===
def get_top_tfidf_words(class_label, top_n=20):
    texts = df[df['label'] == class_label]['clean_text']
    tfidf_subset = tfidf.transform(texts)
    mean_tfidf = np.asarray(tfidf_subset.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[-top_n:][::-1]
    top_features = [tfidf.get_feature_names_out()[i] for i in top_indices]
    top_scores = mean_tfidf[top_indices]
    return top_features, top_scores

for sentiment in [0, 1]:  # 0 = positif, 1 = netral
    label_name = 'positive' if sentiment == 0 else 'neutral'
    words, scores = get_top_tfidf_words(sentiment)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=words, palette='viridis')
    plt.title(f"Top 20 TF-IDF Words - {label_name.capitalize()}")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig(f"plots/top20_tfidf_{label_name}.png")
    plt.close()

# === WordCloud (Positive & Neutral) ===
for sentiment in [0, 1]:  # 0 = positif, 1 = netral
    label_name = 'positive' if sentiment == 0 else 'neutral'
    text = " ".join(df[df['label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - {label_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"plots/wordcloud_{label_name}.png")
    plt.close()

# === One-hot for Neural Network ===
y_train_cat = to_categorical([int(l) + 1 for l in y_train])
y_test_cat = to_categorical([int(l) + 1 for l in y_test])

results = {}

# === Random Forest ===
print("[STARTING] Random Forest.")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)
y_pred_rf = rf_model.predict(X_test_vec)
results['Random Forest'] = accuracy_score(y_test, y_pred_rf)
print("[INFO] Finished Random Forest.")

# === SVM ===
print("[STARTING] SVM.")
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
results['SVM'] = accuracy_score(y_test, y_pred_svm)
print("[INFO] Finished SVM.")

# === Neural Network ===
class PlotMetrics(Callback):
    def __init__(self):
        self.history = {'acc': [], 'loss': []}

    def on_epoch_end(self, epoch, logs=None):
        acc = logs['accuracy']
        loss = logs['loss']
        print(f"[EPOCH {epoch+1}] Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        self.history['acc'].append(logs['accuracy'])
        self.history['loss'].append(logs['loss'])

plot_callback = PlotMetrics()

nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_vec.toarray(), y_train_cat, epochs=10, batch_size=128, verbose=1, callbacks=[plot_callback])

y_pred_nn = np.argmax(nn_model.predict(X_test_vec.toarray()), axis=1) - 1
results['Neural Network'] = accuracy_score(y_test, y_pred_nn)

# === Evaluation ===
best_model = max(results, key=results.get)
print("\nModel Accuracies:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
print(f"\nBest Model: {best_model}")

# === Confusion Matrix & Report ===
predictions = {
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'Neural Network': y_pred_nn
}

for model_name, y_pred in predictions.items():
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plots/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

# === Neural Network Training Plot ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(plot_callback.history['acc'], label='Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(plot_callback.history['loss'], label='Loss', color='orange')
plt.title('Training Loss')
plt.legend()

plt.savefig("plots/nn_training_progress.png")
plt.close()
