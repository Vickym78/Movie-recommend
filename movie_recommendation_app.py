import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('streamlit')
install('pandas')
install('scikit-learn')

# Create the Streamlit app script
streamlit_app_code = '''
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('tmdb_5000_movies.csv')

# Preprocess data
def preprocess_data(df):
    df = df[['title', 'overview']].dropna()
    df['overview'] = df['overview'].fillna('')
    return df

# Compute the cosine similarity matrix
def compute_cosine_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get movie recommendations
def get_recommendations(title, df, cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Streamlit app
def main():
    st.title("Movie Recommendation System")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    cosine_sim = compute_cosine_similarity(df)
    
    # Movie selection
    st.header("Select a Movie")
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)
    
    # Show recommendations
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_movie, df, cosine_sim)
        st.write("**Recommended Movies:**")
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")

if __name__ == "__main__":
    main()
'''

with open('movie_recommendation_app.py', 'w') as f:
    f.write(streamlit_app_code)

# Run the Streamlit app
subprocess.run(["streamlit", "run", "movie_recommendation_app.py"])
