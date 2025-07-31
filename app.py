import streamlit as st
import tensorflow as tf
import pickle
import re
import os
import string
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Constants
MODEL_PATH = "lstm_glove_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 300

# Preprocessing (must match training)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    words = re.sub("[^a-zA-Z]", " ", text).lower().split()
    return " ".join(ps.stem(w) for w in words if w not in stop_words)

# Load tokenizer
if not os.path.exists(TOKENIZER_PATH):
    st.error(f"❌ Tokenizer file not found at {TOKENIZER_PATH}")
    st.stop()
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}")
    st.stop()
model = tf.keras.models.load_model(MODEL_PATH)

# Prediction
def predict_news(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(pad_seq)[0][0]
    label = "🟢 Real News" if prob < 0.5 else "🔴 Fake News"
    confidence = round((1 - prob) * 100, 2) if prob < 0.5 else round(prob * 100, 2)
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

user_input = st.text_area("📝 Enter News Text:", height=200)

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        label, confidence = predict_news(user_input)
        st.markdown(f"### Prediction: {label}")
        st.write(f"🧠 Confidence: **{confidence:.2f}%**")
