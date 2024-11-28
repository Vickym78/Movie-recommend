import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load and preprocess dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)[['title', 'overview', 'vote_average', 'vote_count']].dropna()
    return df

# Compute features using TF-IDF and normalize vote_average & vote_count
def compute_features(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df[['vote_average', 'vote_count']])
    return np.hstack([tfidf_matrix.toarray(), numeric_features])

# Train KNN model
def train_knn(features):
    knn = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
    knn.fit(features)
    return knn

# Get recommendations based on KNN
def get_recommendations(title, df, knn, features):
    idx = df[df['title'] == title].index[0]
    distances, indices = knn.kneighbors(features[idx].reshape(1, -1))
    movie_indices = indices.flatten()[1:]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit app
def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎥")
    
    # Header section with styling
    st.markdown("""
    <style>
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #2B2B2B;
        margin-bottom: 30px;
    }
    .recommendation {
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Movie Recommendation System 🎥</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find movies similar to your favorites!</div>', unsafe_allow_html=True)

    # Load and preprocess data
    with st.spinner("Loading data..."):
        df = load_data('tmdb_5000_movies.csv')
        features = compute_features(df)
        knn = train_knn(features)

    # Movie selection
    st.write("### Select a Movie")
    selected_movie = st.selectbox("Choose a movie to get recommendations:", df['title'].tolist())

    # Button for recommendations
    if st.button("🎬 Recommend Movies"):
        recommendations = get_recommendations(selected_movie, df, knn, features)

        # Display recommendations in a styled format
        st.write(f"### Recommendations for **{selected_movie}**:")
        for i, movie in enumerate(recommendations, 1):  # Start enumeration from 1
            st.markdown(f'<div class="recommendation">{i}. {movie}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
