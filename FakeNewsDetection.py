import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and label data
df_fake = pd.read_csv(r"C:\Users\MUBASHAR\PycharmProjects\PythonProject\News _dataset\Fake.csv")[["title", "text"]].fillna(" ")
df_true = pd.read_csv(r"C:\Users\MUBASHAR\PycharmProjects\PythonProject\News _dataset\True.csv")[["title", "text"]].fillna(" ")

df_fake["label"] = 1  # Fake
df_true["label"] = 0  # Real

df = pd.concat([df_fake, df_true], ignore_index=True)
df["content"] = df["title"] + " " + df["text"]

# 2. Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    words = re.sub("[^a-zA-Z]", " ", text).lower().split()
    return " ".join(ps.stem(w) for w in words if w not in stop_words)

df["content"] = df["content"].apply(clean_text)
X = df["content"].values
y = df["label"].values

# 3. Split dataset
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
)

# 4. Tokenization and Padding
MAX_WORDS = 20000
MAX_LEN = 300
EMBED_DIM = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_txt)

X_train_seq = tokenizer.texts_to_sequences(X_train_txt)
X_test_seq  = tokenizer.texts_to_sequences(X_test_txt)

X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 5. Load GloVe embeddings
embeddings_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((MAX_WORDS, EMBED_DIM))
for word, i in tokenizer.word_index.items():
    if i < MAX_WORDS:
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

# 6. Build or load model
model_path = "lstm_glove_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM,
                  weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=5, batch_size=64,
              validation_split=0.1, callbacks=[es], verbose=1)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")

# 7. Evaluation
train_pred = (model.predict(X_train) > 0.5).astype(int).flatten()
test_pred  = (model.predict(X_test) > 0.5).astype(int).flatten()

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test  Accuracy:", accuracy_score(y_test, test_pred))
print("ROC AUC Score:", roc_auc_score(y_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))

# 8. Inference
def predict_fake_news(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(pad_seq)[0][0]
    return "Fake News" if prob > 0.5 else "True News"

# Interactive test
while True:
    input_text = input("Enter news text (or 'exit'): ").strip()
    if input_text.lower() == 'exit':
        break
    print("Prediction:", predict_fake_news(input_text))
