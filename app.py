import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# --- Load saved model and vectorizer ---
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Preprocessing setup (same as training) ---
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'never', 'none', "n't", 'cannot',
                   "don't", "doesn't", "didn't", "won't", "can't", "shouldn't",
                   "wouldn't", "isn't", "aren't", "wasn't", "weren't"}
stop_words = stop_words - negation_words
stemmer = PorterStemmer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)                    # remove HTML tags like <br />
    text = re.sub(r'http\S+|www\S+', ' ', text)            # remove URLs

    # Expand contractions BEFORE stripping punctuation,
    # so "wasn't" -> "was not" instead of being mangled into "wasn t"
    text = text.replace('’', "'").replace('‘', "'")
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for pat, repl in contractions.items():
        text = text.replace(pat, repl)

    text = re.sub(r'[^a-z\s]', ' ', text)                  # remove punctuation/numbers, keep letters
    text = re.sub(r'\s+', ' ', text).strip()               # remove extra whitespace
    return text


def preprocess_text(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)


def predict_sentiment(text):
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob


# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Classical ML (TF-IDF + Logistic Regression) — no deep learning used.")

user_input = st.text_area("Enter a movie review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        pred, prob = predict_sentiment(user_input)
        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
        confidence = prob[1] if pred == 1 else prob[0]
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: **{confidence*100:.1f}%**")
        st.progress(float(confidence))
        with st.expander("See probability breakdown"):
            st.write(f"Negative: {prob[0]*100:.1f}%")
            st.write(f"Positive: {prob[1]*100:.1f}%")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF (unigrams + bigrams) | Trained on IMDB 50K dataset")
