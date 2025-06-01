import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load stopwords
stop_words = set(stopwords.words("english"))
negations = {'not', 'no', 'nor', "didn't", "wasn't", "isn't", "aren't", "doesn't"}
custom_stopwords = stop_words - negations

# Load resources
MODEL_PATH = (r"C:\Users\katre\Downloads\Amazon Alexa Review -sentiment Analysis using NLP\Model\model_xgb.pkl")
SCALER_PATH = (r"C:\Users\katre\Downloads\Amazon Alexa Review -sentiment Analysis using NLP\Model\scaler.pkl")
VECTORIZER_PATH = (r"C:\Users\katre\Downloads\Amazon Alexa Review -sentiment Analysis using NLP\Model\countVectorizer.pkl")

@st.cache_resource
def load_resources():
    predictor = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    cv = pickle.load(open(VECTORIZER_PATH, "rb"))
    return predictor, scaler, cv

predictor, scaler, cv = load_resources()

# Preprocess function
def preprocess_text(text_input):
    lemmatizer = WordNetLemmatizer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in custom_stopwords]
    return " ".join(review)

# Prediction function
def single_prediction(text_input):
    text_processed = preprocess_text(text_input)
    X_prediction = cv.transform([text_processed]).toarray()
    X_prediction_scaled = scaler.transform(X_prediction)
    y_prediction = predictor.predict_proba(X_prediction_scaled)
    threshold = 0.90
    positive_probability = y_prediction[0][1]
    return "Positive" if positive_probability > threshold else "Negative"

# Apply custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4B8BBE;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.5em 2em;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #3a6f9e;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">🗣️ Alexa Reviews Sentiment Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This app uses an XGBoost model to classify reviews as Positive or Negative</div>', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter your Alexa review here:", height=150)

# Predict button
if st.button('🔍 Predict Sentiment'):
    if user_input.strip():
        result = single_prediction(user_input)
        if result == "Positive":
            st.success(f"✅ Predicted Sentiment: **{result}** 😊")
        else:
            st.error(f"❌ Predicted Sentiment: **{result}** 😞")
    else:
        st.warning("⚠️ Please enter some text to analyze.")