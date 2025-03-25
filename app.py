# Import required libraries
import pandas as pd
# Removed unused imports
import seaborn as sns
import nltk
import re
import streamlit as st
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Set plotting style
sns.set_theme(style='whitegrid')

# Importing Dataframe
df = pd.read_csv('train.csv')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)

    return ' '.join(tokens)

df['context_clean'] = df['Context'].apply(preprocess_text)

# Initialize TF-IDF vectorizer and compute matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['context_clean'])

# LSA: Reduce dimensionality using TruncatedSVD
n_components = 100  # adjust number of LSA components if necessary
svd_model = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd_model.fit_transform(tfidf_matrix)
print('LSA matrix shape:', lsa_matrix.shape)

# Compute cosine similarity for LSA model
cosine_sim_lsa = cosine_similarity(lsa_matrix, lsa_matrix)

def get_random_response(query):
    # Pre-process the query
    query_clean = preprocess_text(query)

    query_vec = tfidf_vectorizer.transform([query_clean])
    query_lsa = svd_model.transform(query_vec)
    sims = cosine_similarity(query_lsa, lsa_matrix).flatten()
    
    # Get indices of top 5 matching responses sorted by similarity
    top_indices = sims.argsort()[-5:][::-1]
    
    # Filter out responses with similarity score <= 0.1
    top_indices = [idx for idx in top_indices if sims[idx] > 0.5]
    
    if not top_indices:
        return "No matching response found.", 0.0
    
    # Select a random index from the top 8 responses
    random_idx = random.choice(top_indices)
    similarity_score = sims[random_idx]
    
    # Return the random matching response and its similarity score
    return df.iloc[random_idx]['Response'], similarity_score

# Streamlit App Layout
# Styled title and description
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Therapist Suggestion App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555555;'>Enter your sentence below and receive a suggestion based on patient context.</p>", unsafe_allow_html=True)

# Form for input and submission
with st.form(key='input_form'):
    user_input = st.text_input("Your Input:", placeholder="e.g., 'I feel anxious about my job.'")
    submit_button = st.form_submit_button(label='Get Suggestion')

# Handle form submission
if submit_button:
    if user_input:
        with st.spinner('Fetching suggestion...'):
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