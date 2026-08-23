# frontend/app.py

import streamlit as st
import sys
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

sys.path.append(BASE_DIR)

from backend.recommender import (
    MovieRecommender
)

# ====================
# Config
# ====================

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# ====================
# Load Model
# ====================

@st.cache_resource
def load_model():

    return MovieRecommender()

recommender = load_model()

# ====================
# UI
# ====================

st.title(
    "Movie Recommendation System"
)

st.write(
    "Temukan film yang memiliki "
    "katrakteristik serupa"
)

movies = (
    recommender.movies[
        'movie_title'
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_movie = st.selectbox(
    "Pilih film:",
    sorted(movies)
)

top_n = st.slider(
    "Jumlah rekomendasi",
    3,
    15,
    5
)

if st.button(
    "Cari remodasi:"
):

    result = recommender.recommend(
        selected_movie,
        top_n
    )

    st.subheader(
        f"Rekomendasi untuk "
        f"'{selected_movie}'"
    )

    for _, movie in result.iterrows():

        st.markdown(
            f"### {movie['movie_title']}"
        )

        st.write(
            f"**Similarity:** "
            f"{movie['similarity_score']:.4f}"
        )

        st.write(
            f"**Genre:** "
            f"{movie['genres']}"
        )

        st.write(
            f"**Director:** "
            f"{movie['director_name']}"
        )

        st.divider()