# backend/recommender.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "artifacts"
)

class MovieRecommender:

    def __init__(self):

        self.preprocessing = joblib.load(
            os.path.join(
                ARTIFACT_DIR,
                "preprocessing.joblib"
            )
        )

        self.tfidf = joblib.load(
            os.path.join(
                ARTIFACT_DIR,
                "tfidf.joblib"
            )
        )

        self.word2vec = Word2Vec.load(
            os.path.join(
                ARTIFACT_DIR,
                "word2vec.model"
            )
        )

        self.embeddings = np.load(
            os.path.join(
                ARTIFACT_DIR,
                "embeddings.npy"
            )
        )

        self.movies = pd.read_csv(
            os.path.join(
                ARTIFACT_DIR,
                "movies.csv"
            )
        )


    def recommend(
        self,
        movie_title,
        top_n=10
    ):

        titles = (
            self.movies['movie_title']
            .fillna('')
            .astype(str)
            .str.lower()
            .str.strip()
        )

        target = (
            movie_title
            .lower()
            .strip()
        )

        matches = titles == target

        if not matches.any():
            return pd.DataFrame()

        index = np.where(
            matches
        )[0][0]

        query = (
            self.embeddings[index]
            .reshape(1, -1)
        )

        database = self.embeddings

        query = normalize(query)

        database = normalize(
            database
        )

        similarity = (
            database @ query.T
        ).flatten()

        similarity[index] = -1


        indices = np.argsort(
            similarity
        )[::-1][:top_n]

        result = self.movies.iloc[
            indices
        ].copy()

        result[
            'similarity_score'
        ] = similarity[indices]

        return result.reset_index(
            drop=True
        )