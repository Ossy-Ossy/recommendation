import numpy as np
import pandas as pd
import joblib
import difflib
import streamlit as st

st.write("""
# Welcome to this AI-Based Movie Recommendation System

This system recommends for you similar movies based on your favorite movie
""")

movies_data = pd.read_csv("movies.csv")
movie_vectorizer = joblib.load('vectorizer_movie_recommend.pkl')
similarity = joblib.load('similarity_movie_recommend.pkl')

def get_movie_recommendations(movie_name):
    list_of_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    st.write("Movies suggested for you:\n")
    i = 1
    
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 21:
            st.write(f"{i} - {title_from_index}")
            i += 1
st.sidebar.header("🔍 Recommendation Engine")
st.sidebar.write("""
**We analyze these 5 key elements:**  
- **Genres**: Primary + secondary categories (e.g., "Sci-Fi + Mystery")  
- **Keywords**: Plot tags like "time travel" or "dystopian future"  
- **Taglines**: Thematic hooks ("In space no one hears you scream")  
- **Cast**: Lead actor/actress combinations  
- **Director**: Filmmaking style fingerprints  
""")

st.sidebar.header("⚙️ Behind the Scenes")
st.sidebar.write("""
Your top 20 are ranked by:  
- **85%** Content similarity (genre/director/plot)  
- **10%** Popularity balance  
- **5%** Surprise factor (expand your horizons)  

*Based on films in our database*  
""")
# First, show the movie selection dropdown
movie_name = st.selectbox("Select your favorite movie name", movies_data['title'].unique())

# Then show the recommendations button
if st.button("Get Recommendations"):
    get_movie_recommendations(movie_name)
