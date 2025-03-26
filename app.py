import pandas as pd
import nltk
import re
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data (done once at startup)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('train.csv')
    df['context_clean'] = df['Context'].apply(preprocess_text)
    return df

# Cache TF-IDF and LSA model fitting
@st.cache_resource
def fit_models(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['context_clean'])
    svd_model = TruncatedSVD(n_components=100, random_state=42)
    lsa_matrix = svd_model.fit_transform(tfidf_matrix)
    return tfidf_vectorizer, svd_model, lsa_matrix

# Load and preprocess data once
df = load_and_preprocess_data()

# Fit models once
tfidf_vectorizer, svd_model, lsa_matrix = fit_models(df)

# Response function (runs per query, but uses cached objects)
def get_random_response(query):
    query_clean = preprocess_text(query)
    query_vec = tfidf_vectorizer.transform([query_clean])
    query_lsa = svd_model.transform(query_vec)
    sims = cosine_similarity(query_lsa, lsa_matrix).flatten()
    top_indices = sims.argsort()[-5:][::-1]
    top_indices = [idx for idx in top_indices if sims[idx] > 0.5]
    if not top_indices:
        return "No matching response found.", 0.0
    random_idx = random.choice(top_indices)
    similarity_score = sims[random_idx]
    return df.iloc[random_idx]['Response'], similarity_score

# Streamlit App Layout
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Therapist Suggestion App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555555;'>Enter your sentence below and receive a suggestion based on patient context.</p>", unsafe_allow_html=True)

with st.form(key='input_form'):
    user_input = st.text_input("Your Input:", placeholder="e.g., 'I feel anxious about my job.'")
    submit_button = st.form_submit_button(label='Get Suggestion')

if submit_button:
    if user_input:
        with st.spinner('Analyzing your input and finding the best suggestion...'):
            response, similarity_score = get_random_response(user_input)
        if response != "No matching response found.":
            st.markdown(
                "<div style='padding: 10px; border: 1px solid #4A90E2; border-radius: 5px;'><h3>Suggested Response</h3><p>{}</p><p><strong>Similarity Score:</strong> {:.4f}</p></div>".format(response, similarity_score),
                unsafe_allow_html=True
            )
        else:
            st.warning("No matching response found. Try rephrasing your input.")
    else:
        st.error("Please enter a valid input.")