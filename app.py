import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Streamlit App ---
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♂️", layout="centered")

st.markdown("<h1 style='text-align: center;'>🕵️‍♂️ Fake Job Posting Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a job description below and we'll analyze whether it's real or fake using Machine Learning.</p>", unsafe_allow_html=True)

st.markdown("---")

user_input = st.text_area("✍️ Enter Job Description Here", height=200, placeholder="e.g., Work from home and earn $5000/week...")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][prediction] * 100

        if prediction == 1:
            st.error(f"🚨 This job posting is likely **FAKE** ({prob:.2f}% confidence)")
        else:
            st.success(f"✅ This job posting is likely **REAL** ({prob:.2f}% confidence)")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Made with ❤️ using Streamlit & scikit-learn</p>", unsafe_allow_html=True)
