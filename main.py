import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit App UI Enhancements
st.set_page_config(page_title="Spam Classifier", page_icon="🚀", layout="wide")

# Custom Dark Mode Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #0F2027, #203A43, #2C5364);
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #1E1E1E;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
    }
    .stTextInput, .stTextArea {
        background-color: #333;
        color: white;
        border-radius: 5px;
        border: 2px solid #00FFFF;
    }
    .stButton>button {
        background: linear-gradient(to right, #00C9FF, #92FE9D);
        color: black;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #11998E, #38EF7D);
        color: white;
    }
    .title-text {
        text-align: center;
        color: #00FFFF;
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .spam-text {
        color: #FF4C4C;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .not-spam-text {
        color: #00FF7F;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🚀 About This App")
st.sidebar.info(
    "This AI-powered Email/SMS Spam Classifier detects spam messages using "
    "Natural Language Processing (NLP) and Machine Learning (ML). "
    "Simply enter a message, and our model will predict if it's spam or not!"
)

st.sidebar.markdown("🔹 Built With:")
st.sidebar.markdown("- Python & Streamlit")
st.sidebar.markdown("- Scikit-Learn (ML)")
st.sidebar.markdown("- NLTK (NLP Processing)")
st.sidebar.markdown("🔹 Model: Trained on 5000 SMS dataset.")

# Main App
st.markdown("<h1 class='title-text'>🚀 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("✍ Enter the message", placeholder="Type your message here...")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠ Please enter a valid message!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown("<h2 class='spam-text'>🚨 Spam Message! 🚨</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='not-spam-text'>✅ Not Spam ✅</h2>", unsafe_allow_html=True)