# movie_recommendation_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('tmdb_5000_movies.csv')

# Preprocess data
def preprocess_data(df):
    df = df[['title', 'overview', 'vote_average', 'vote_count']].dropna()
    df['overview'] = df['overview'].fillna('')
    return df

# Compute the feature matrix using TF-IDF and other features
def compute_features(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Normalize vote_average and vote_count
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['vote_average', 'vote_count']])

    # Combine features
    features = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(scaled_features)], axis=1)
    return features

# Train the KNN model
def train_knn(features):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    knn.fit(features)
    return knn

# Get movie recommendations using KNN
def get_recommendations(title, df, knn, features):
    idx = df[df['title'] == title].index[0]
    distances, indices = knn.kneighbors(features.iloc[idx, :].values.reshape(1, -1), n_neighbors=11)
    movie_indices = indices.flatten()[1:]  # Exclude the first one as it is the selected movie itself
    return df['title'].iloc[movie_indices]

# Streamlit app
def main():
    st.title("Movie Recommendation System using KNN")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    features = compute_features(df)
    knn = train_knn(features)
    
    # Movie selection
    st.header("Select a Movie")
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)
    
    # Show recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie, df, knn, features)
        st.write("**Recommended Movies:**")
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")

if __name__ == "__main__":
    main()
