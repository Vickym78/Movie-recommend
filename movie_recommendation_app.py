import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Preprocess data
def preprocess_data(df):
    try:
        # Select relevant numerical features and movie titles
        df = df[['title', 'vote_average', 'vote_count']].dropna()
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Compute features
def compute_features(df):
    try:
        # Normalize vote_average and vote_count
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[['vote_average', 'vote_count']])
        return features
    except Exception as e:
        st.error(f"Error computing features: {e}")
        return np.array([])  # Return empty array on error

# Train the KNN model
def train_knn(features):
    try:
        knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
        knn.fit(features)
        return knn
    except Exception as e:
        st.error(f"Error training KNN model: {e}")
        return None

# Get movie recommendations based on KNN
def get_recommendations(title, df, knn, features):
    try:
        if title not in df['title'].values:
            return ["Movie not found in dataset."]
        
        # Get the index of the selected movie
        idx = df[df['title'] == title].index[0]
        
        # Find similar movies
        distances, indices = knn.kneighbors([features[idx]], n_neighbors=11)
        movie_indices = indices.flatten()[1:]  # Exclude the selected movie itself
        
        # Get movie titles
        recommended_movies = df['title'].iloc[movie_indices].tolist()
        return recommended_movies
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []  # Return empty list on error

# Streamlit App
def main():
    st.title("Movie Recommendation System")
    
    # Load and preprocess data
    file_path = 'tmdb_5000_movies.csv'  # Replace with your file path
    df = load_data(file_path)
    
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
    selected_movie = st.selectbox("Choose a movie:", movie_list)
    
    # Show recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie, df, knn, features)
        if recommendations:
            st.write(f"**Recommended Movies (Similar to {selected_movie}):**")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
