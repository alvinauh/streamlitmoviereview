import streamlit as st
import joblib
import numpy as np
import re

classifier = joblib.load("sentiment_model.pkl")
word2vec = joblib.load("word2vec_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def get_review_vector(words):
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    if not vectors:
        return np.zeros(word2vec.vector_size)
    return np.mean(vectors, axis=0)

def predict_sentiment(review):
    tokens = clean_text(review)
    vector = get_review_vector(tokens).reshape(1, -1)
    pred = classifier.predict(vector)[0]
    conf = classifier.predict_proba(vector).max()
    return ("😊 Positive" if pred == 1 else "😞 Negative"), conf

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")
st.title("🎬 Sentiment Analysis using Word2Vec")

review = st.text_area("Enter your movie review")

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment, confidence = predict_sentiment(review)
        if "Positive" in sentiment:
            st.success(sentiment)
        else:
            st.error(sentiment)
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.warning("Please enter a review.")

st.caption("Model: Word2Vec + Logistic Regression")
