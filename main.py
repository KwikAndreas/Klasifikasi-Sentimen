import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
from bs4 import BeautifulSoup
import emoji
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from wordcloud import WordCloud
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# === Load Dataset ===
print("[READ] Loading dataset...")
df = pd.read_csv('dataset/movie.csv')
print("[INFO] Dataset loaded.")

# === Label Distribution ===
print("Label distribution:")
print(df['label'].value_counts())
print(df.head(10))

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, hue='label', palette='pastel', legend=False)
plt.title("Label Distribution")
plt.savefig("plots/label_distribution.png")
plt.close()

# === Pie Chart ===
label_counts = df['label'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    label_counts, 
    labels=['Negative', 'Positive'], 
    autopct='%1.2f%%', 
    startangle=140, 
    colors=sns.color_palette("pastel")
)
plt.title("Sentiment Distribution (Pie)")
plt.savefig("plots/pie_label_distribution.png")
plt.close()

# === Text Preprocessing ===
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Remove non-ascii
    return text.strip()

df['clean_text'] = df['text'].astype(str).apply(clean_text)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.3, stratify=df['label'], random_state=42
)

# === TF-IDF Vectorization (Tokenisasi) ===
print("[STARTING] TF-IDF Vectorization.")
# max_features=10000, min_df=3, max_df=0.9
tfidf = TfidfVectorizer(max_features=15000, min_df=2, max_df=0.85, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print("[INFO] TF-IDF complete.")

# === Top 20 TF-IDF Words per Class ===
def get_top_tfidf_words(class_label, top_n=20):
    texts = df[df['label'] == class_label]['clean_text']
    tfidf_subset = tfidf.transform(texts)
    mean_tfidf = np.asarray(tfidf_subset.mean(axis=0)).flatten()

    # Get all feature names (tokens)
    feature_names = tfidf.get_feature_names_out()
    
    # POS tagging: get POS for each unigram only (ignore bigrams)
    unigram_features = [w for w in feature_names if ' ' not in w]
    pos_tags = dict(pos_tag(unigram_features))

    # Select adjectives only
    adj_indices = [i for i, word in enumerate(feature_names)
                   if word in pos_tags and pos_tags[word] in ('JJ', 'JJR', 'JJS')]
    
    # Filter TF-IDF values
    adj_scores = mean_tfidf[adj_indices]
    sorted_idx = np.argsort(adj_scores)[-top_n:][::-1]

    top_adj_indices = [adj_indices[i] for i in sorted_idx]
    top_features = [feature_names[i] for i in top_adj_indices]
    top_scores = mean_tfidf[top_adj_indices]

    return top_features, top_scores

for sentiment in [0, 1]:
    label_name = 'Negative' if sentiment == 0 else 'Positive'
    words, scores = get_top_tfidf_words(sentiment)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=words, palette='viridis', hue=words)
    plt.title(f"Top 20 TF-IDF Words - {label_name}")
    plt.xlabel("TF-IDF Score")
    plt.tight_layout()
    plt.savefig(f"plots/top20_tfidf_{label_name.lower()}.png")
    plt.close()

# === WordCloud ===
for sentiment in [0, 1]:
    label_name = 'Negative' if sentiment == 0 else 'Positive'
    text = " ".join(df[df['label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - {label_name}")
    plt.tight_layout()
    plt.savefig(f"plots/wordcloud_{label_name.lower()}.png")
    plt.close()

# === One-hot Encoding NN ===
y_train_cat = to_categorical([int(l) + 1 for l in y_train])
y_test_cat = to_categorical([int(l) + 1 for l in y_test])

results = {}

# === Random Forest ===
print("[STARTING] Random Forest")
rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rf_model.fit(X_train_vec, y_train)
y_pred_rf = rf_model.predict(X_test_vec)
results['Random Forest'] = accuracy_score(y_test, y_pred_rf)

# === SVM ===
print("[STARTING] SVM")
svm_model = LinearSVC(C=0.5, max_iter=2000, random_state=42, verbose=1)
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
results['SVM'] = accuracy_score(y_test, y_pred_svm)

# === Neural Network ===
class PlotMetrics(Callback):
    def __init__(self):
        self.history = {'acc': [], 'loss': []}

    def on_epoch_end(self, epoch, logs=None):
        acc = logs['accuracy']
        loss = logs['loss']
        print(f"[EPOCH {epoch+1}] Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        self.history['acc'].append(acc)
        self.history['loss'].append(loss)

plot_callback = PlotMetrics()

nn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_vec.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

nn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train_vec.toarray(), y_train_cat, epochs=15, batch_size=64, verbose=1, callbacks=[plot_callback])

y_pred_nn = np.argmax(nn_model.predict(X_test_vec.toarray()), axis=1) - 1
results['Neural Network'] = accuracy_score(y_test, y_pred_nn)

# === Model Accuracies ===
print("\nModel Accuracies:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model}")

# === Evaluation: Confusion Matrix & Report ===
predictions = {
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'Neural Network': y_pred_nn
}

# === Save the Best Model and Vectorizer ===
print("[SAVING] Saving best model (SVM) and vectorizer...")
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("[DONE] Model and vectorizer saved.")

for model_name, y_pred in predictions.items():
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)

    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plots/confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

    # Pie chart
    cm_labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    cm_pie_data = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    plt.figure(figsize=(6, 6))
    plt.pie(cm_pie_data, labels=cm_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f"Confusion Matrix Pie Chart: {model_name}")
    plt.savefig(f"plots/confusion_pie_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

# === Training History Plot ===
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
