import streamlit as st
import pickle
import pandas as pd
import requests

# PAGE CONFIG
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #E50914;
    color: white;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #ff1e1e;
    color: white;
}

.stSelectbox label {
    font-size: 20px;
    font-weight: bold;
    color: white;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 10px;
    border-radius: 12px;
    margin-top: 8px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# LOAD PICKLE FILES
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# TMDB API KEY
API_KEY = st.secrets["TMDB_API_KEY"]

# SIDEBAR
with st.sidebar:

    st.title("🎬 Movie Recommendation System")

    st.markdown("---")

    st.subheader("📌 About")

    st.write("""
    Select Movies & Discover Similar Movies You'll Love Instantly
    """)

    st.markdown("---")

    st.subheader("🧠 How It Works")

    st.write("""
    It recommends movies using:
    - Cosine Similarity
    - Count Vectorizer
    - Genre Matching
    - Cast & Keywords Analysis
    """)

    st.markdown("---")

    st.success("Developed by Vamsi Andavarapu")

# FETCH MOVIE DETAILS
def fetch_movie_details(title, movies):

    idx = movies[movies["title"] == title].index[0]
    movie_id = movies.iloc[idx].movie_id

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        poster = (
            "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
            if data.get("poster_path")
            else "https://via.placeholder.com/500x750?text=No+Image"
        )

        rating = data.get("vote_average", "N/A")
        release_date = data.get("release_date", "N/A")

        overview = data.get(
            "overview",
            "No overview available."
        )

        return poster, rating, release_date, overview

    except:
         return "https://via.placeholder.com/300x450?text=Error"


# RECOMMEND FUNCTION
def recommend(movie):

    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:

        title = movies.iloc[i[0]].title

        poster, rating, release_date, overview = fetch_movie_details(
            title,
            movies
        )

        movie_data = {
            "title": title,
            "poster": poster,
            "rating": rating,
            "release_date": release_date,
            "overview": overview
        }

        recommended_movies.append(movie_data)

    return recommended_movies

# MAIN UI
st.title("🎥 Movie Recommendation System")

st.markdown("""
### Discover Movies Similar to Your Favorites 🍿
Select a movie from the dropdown and get recommendations instantly.
""")

movie_list = sorted(movies["title"].values)

selected_movie = st.selectbox(
    "🎬 Select a Movie",
    movie_list
)

# SELECTED MOVIE DETAILS
poster, rating, release_date, overview = fetch_movie_details(
    selected_movie,
    movies
)

st.markdown("---")

st.subheader("🎞️ Selected Movie")

col1, col2 = st.columns([1, 2])

with col1:
    st.image(
        poster,
        width="stretch"
    )

with col2:
    st.markdown(f"## {selected_movie}")

    st.markdown(f"⭐ Rating: **{rating}**")

    st.markdown(f"📅 Release Date: **{release_date}**")

    st.markdown("### Overview")

    st.write(overview)

# RECOMMEND BUTTON
if st.button("Recommend Movies"):

    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie)

    st.success("Recommendations Generated Successfully!")

    st.markdown("---")

    st.subheader("✨ Recommended Movies")

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):

        with cols[idx]:

            st.image(
                movie["poster"],
                width="stretch"
            )

            st.markdown(
                f"<div class='card'><h4>{movie['title']}</h4></div>",
                unsafe_allow_html=True
            )

            st.markdown(f"⭐ Rating: **{movie['rating']}**")

            st.markdown(f"📅 {movie['release_date']}")

# FOOTER
st.markdown("---")

st.markdown("""
<center>
Made with ❤️ using Python, Streamlit and Machine Learning
</center>
""", unsafe_allow_html=True)
