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
    st.title("Movie Recommendation System")

    # Load and preprocess data
    df = load_data('tmdb_5000_movies.csv')
    features = compute_features(df)
    knn = train_knn(features)

    # Movie selection
    selected_movie = st.selectbox("Select a movie", df['title'].tolist())

    # Recommend button
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie, df, knn, features)
        st.write(f"**Recommended Movies (Similar to {selected_movie}):**")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

if __name__ == "__main__":
    main()
