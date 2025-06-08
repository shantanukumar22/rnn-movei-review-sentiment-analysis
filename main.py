# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('final-model.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title('🎬 IMDB Movie Review Sentiment Classifier')
st.markdown('''
This app uses a pre-trained Recurrent Neural Network to classify whether a movie review is **positive** or **negative**.
''')

st.sidebar.header("About")
st.sidebar.markdown("""
- **Dataset:** IMDB Movie Reviews  
- **Model:** Trained RNN with ReLU activation  
- **Max Review Length:** 500 words  
""")

st.subheader("📝 Enter a movie review")
user_input = st.text_area('Your Review', height=150)

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        with st.spinner("Analyzing..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            score = prediction[0][0]
            sentiment = 'Positive' if score > 0.7 else 'Negative'

        if sentiment == 'Positive':
            st.success(f"✅ Sentiment: {sentiment} ({score:.2f})")
        else:
            st.error(f"❌ Sentiment: {sentiment} ({score:.2f})")

