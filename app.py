import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    return movies

def convert(obj):
    items = []
    for i in ast.literal_eval(obj):
        items.append(i["name"])
    return items

def convert_cast(obj):
    items = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            items.append(i["name"])
            count += 1
        else:
            break
    return items

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []

@st.cache_data
def prepare_data():
    movies = load_data()

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].fillna("")
    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity


def fetch_poster(title, movies):
    idx = movies[movies["title"] == title].index[0]
    movie_id = movies.iloc[idx].movie_id
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=No+Image"

    return "https://via.placeholder.com/500x750?text=No+Image"
def recommend(movie, movies, similarity):
    movie = movie.lower()

    if movie not in movies["title"].str.lower().values:
        return [], []

    idx = movies[movies["title"].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    posters = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        posters.append(fetch_poster(title, movies))

    return recommended_movies, posters

st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get 5 similar movie recommendations.")

movies, similarity = prepare_data()
movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    titles, posters = recommend(movie_name, movies, similarity)

    if titles:
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(posters[i], use_container_width=True)
                st.caption(titles[i])
    else:
        st.write("Sorry, we don't have related movies or information for this title.")