import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tmdb_5000_movies.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Preprocess data
def preprocess_data(df):
    try:
        df = df[['title', 'overview', 'vote_average', 'vote_count', 'genres']].dropna()
        df['overview'] = df['overview'].fillna('')
        df['genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')  # Extract first genre
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Compute the feature matrix using TF-IDF and other features
def compute_features(df):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['overview'])

        # Normalize vote_average and vote_count
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[['vote_average', 'vote_count']])

        # Combine features
        features = np.hstack([tfidf_matrix.toarray(), scaled_features])
        return features
    except Exception as e:
        st.error(f"Error computing features: {e}")
        return np.array([])  # Return empty array on error

# Train the KNN model
def train_knn(features):
    try:
        knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
        knn.fit(features)
        return knn
    except Exception as e:
        st.error(f"Error training KNN model: {e}")
        return None

# Get movie recommendations based on KNN
def get_recommendations(title, df, knn, features):
    try:
        idx = df[df['title'] == title].index[0]
        distances, indices = knn.kneighbors(features[idx].reshape(1, -1), n_neighbors=11)
        movie_indices = indices.flatten()[1:]  # Exclude the first one as it is the selected movie itself
        recommended_movies = df['title'].iloc[movie_indices].tolist()
        return recommended_movies
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []  # Return empty list on error

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Load and preprocess data
    df = load_data()
    if df.empty:
        st.stop()  # Stop execution if data loading failed

    df = preprocess_data(df)
    if df.empty:
        st.stop()  # Stop execution if preprocessing failed

    features = compute_features(df)
    if features.size == 0:
        st.stop()  # Stop execution if feature computation failed

    knn = train_knn(features)
    if knn is None:
        st.stop()  # Stop execution if model training failed
    
    # Movie selection
    st.header("Select a Movie")
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)
    
    # Show recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie, df, knn, features)
        if recommendations:
            st.write(f"**Recommended Movies (Similar to {selected_movie}):**")
            # Display only the top 10 recommendations
            for i, movie in enumerate(recommendations):
                if i >= 10:  # Use '>=', not '>'
                    break
                st.write(f"{i+1}. {movie}")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
