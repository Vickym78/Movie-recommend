import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('tmdb_5000_movies.csv')

# Preprocess data
def preprocess_data(df):
    df = df[['title', 'overview', 'vote_average', 'vote_count', 'genres']].dropna()
    df['overview'] = df['overview'].fillna('')
    df['genre'] = df['genres'].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')  # Extract first genre
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

# Train the Random Forest model
def train_random_forest(features, labels):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    return rf

# Get movie recommendations based on predicted genre
def get_recommendations_by_genre(title, df, rf, features, label_encoder):
    idx = df[df['title'] == title].index[0]
    predicted_genre_idx = rf.predict(features.iloc[idx, :].values.reshape(1, -1))
    predicted_genre = label_encoder.inverse_transform(predicted_genre_idx)[0]
    recommended_movies = df[df['genre'] == predicted_genre]['title'].tolist()
    return recommended_movies

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    features = compute_features(df)

    # Encode genres
    label_encoder = LabelEncoder()
    df['genre_encoded'] = label_encoder.fit_transform(df['genre'])

    rf = train_random_forest(features, df['genre_encoded'])
    
    # Movie selection
    st.header("Select a Movie")
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)
    
    # Show recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations_by_genre(selected_movie, df, rf, features, label_encoder)
        st.write(f"**Recommended Movies (Same Genre as {selected_movie}):**")
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")

if __name__ == "__main__":
    main()
