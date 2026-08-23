# backend/train.py

"""

TF-IDF
Word2Vec
Combined Embedding
embeddings.npy
movies.csv

"""

import pandas as pd
import numpy as np
import joblib
import os

from preprocessing import fit_preprocessing

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from gensim.models import Word2Vec

from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'movie_metadata.csv'
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    'backend',
    'artifacts'
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

def train_tfidf(df):
    """
    
    plot_keywords:
        list -> string

    genres:
        list -> string

    Kemudian keduanya digabungkan menjadi 
    satu dikumen untuk setiap film
    
    """

    # ======================
    # plot_keywords + string
    # ======================

    df['plot_keywords_str'] = (
        df['plot_keywords']
        .apply(
            lambda x:
                ' '.join(x)
                if isinstance(x, list)
                else ''
        )
    )

    # =======================
    # genres + string
    # =======================

    df['genres_str'] = (
        df['genres']
        .apply(
            lambda x:
                " ".join(x)
                if isinstance(x, list)
                else ''
        )
    )

    # ===================
    # Gabingkan text
    # ===================

    text_docs = (
        df['plot_keywords_str']
        + " "
        + df['genres_str']
    )

    # ===================
    # TF-IDF
    # ===================

    vectorizer = TfidfVectorizer()

    tfidf_matrix = (
        vectorizer.fit_transform(
            text_docs
        )
    )

    return (
        df,
        vectorizer,
        tfidf_matrix
    )

# =======================
# WORD2VEC
# =======================

def train_word2vec(df):
    """
    
    Entitas:
        director_name,
        actor_1_name.
        actor_2_name,
        actor_3_name

    Word2Vec:
        vector_size=50
        window=3
        min_count=1
        sg=1
    
    """

    entity_cols = [
        'director_name',
        'actor_1_name',
        'actor_2_name',
        'actor_3_name'
    ]

    films_entities = (
        df[
            entity_cols
        ]
        .values
        .tolist()
    )

    # ==================
    # Word2Vec
    # ==================

    model = Word2Vec(
        films_entities,
        vector_size=50,
        window=3, 
        min_count=1,
        sg=1
    )

    def film_vector(entities):

        vectors = [
            model.wv[e]
            for e in entities
            if e in model.wv
        ]

        return np.mean(
            vectors, 
            axis=0
        )

    entity_embeddings = np.array([
        film_vector(f)
        for f in films_entities
    ])

    return (
        model,
        entity_embeddings
    )

def get_categorical_features(
    df,
    preprocessing_artifacts
):
    """
    
    Mengambil fitur kategorikal hasil OHE
    dari output preprocessing 

    Tidak melakukan fit ulang

    Urutan:
        content_rating
        language
    
    """

    content_rating_encoder = (
        preprocessing_artifacts[
            'content_rating_encoder'
        ]
    )

    content_rating_encoded = (
        content_rating_encoder
        .transform(
            df[
                ['content_rating']
            ]
        )
    )

    content_rating_columns = (
        content_rating_encoder
        .get_feature_names_out(
            ['content_rating']
        )
    )

    content_rating_df = (
        pd.DataFrame(
            content_rating_encoded,
            columns=content_rating_columns,
            index=df.index
        )
    )

    language_encoder = (
        preprocessing_artifacts[
            'language_encoder'
        ]
    )

    language_encoded = (
        language_encoder
        .transform(
            df[
                ['language']
            ]
        )
    )

    language_columns = (
        language_encoder
        .get_feature_names_out(
            ['language']
        )
    )

    language_df = (
        pd.DataFrame(
            language_encoded,
            columns=language_columns,
            index=df.index
        )
    )

    # =========================
    # COMBINE
    # =========================

    categorical_features = np.hstack([
        content_rating_df.values,
        language_df.values
    ])

    return (
        categorical_features,
        content_rating_columns,
        language_columns
    )

# ========================
# NUMERIC FEATURES
# ========================

def get_numeric_features(df):
    """
    
    duration
    title_year
    country_freq
    
    """

    numeric_features = (
        df[
            [
                'duration',
                'title_year',
                'country_freq'
            ]
        ]
        .values
    )

    return numeric_features

# =========================
# COMBINE FEATURES
# =========================

def build_combined_embeddings(
        tfidf_matrix,
        entity_embeddings,
        numeric_features,
        categorical_features
):
    """
    
    combined_embedding = 
    TF-IDF
    +
    Word2Vec
    +
    numeric
    + 
    categorical
    
    """

    tfidf_normalized = (
        normalize(
            tfidf_matrix
        )
        .toarray()
    )

    combined_embeddings = np.hstack([
        tfidf_normalized,
        entity_embeddings,
        numeric_features,
        categorical_features
    ])

    return combined_embeddings

# ==========================
# SAVE MOVIES.CSV
# ==========================

def save_movies(
        df,
        output_path
):
    """
    
    Menyimpan metadata film yang index/barisnya
    HARUS sama dengan embeddings.npy
    
    """

    movie_columns = [
        'movie_title',
        'genres',
        'plot_keywords',
        'director_name',
        'actor_1_name',
        'actor_2_name',
        'actor_3_name',
        'language',
        'country',
        'content_rating',
        'duration',
        'title_year',
        'year_unknown',
        'country_freq'
    ]

    movie_columns = [
        col
        for col in movie_columns
        if col in df.columns
    ]

    movies = (
        df[
            movie_columns
        ]
        .copy()
    )

    for col in [
        'genres',
        'plot_keywords'
    ]:
        if col in movies.columns:
            movies[col] = (
                movies[col]
                .apply(
                    lambda x:
                        '|'.join(x)
                        if isinstance(x, list)
                        else x
                )
            )

    movies.to_csv(
        output_path,
        index=False
    )


# =============================
# SAVE FEATURE INFORMATION
# =============================

def save_feature_metadata(
        tfidf_matrix,
        entity_enbeddings,
        numeric_features,
        categorical_features,
        content_rating_columns,
        language_columns
    ):
        """
        
        Menyimpan informasi simensi embedding

        Berguna untuk debugging dan dokumentasi
        
        """

        metadata = {
            'tfidf_dimension':
                tfidf_matrix.shape[1],

            'word2vec_dimension':
                entity_enbeddings.shape[1],

            'numeric_dimension':
                numeric_features.shape[1],

            'categorical_dimension':
                categorical_features.shape[1],

            'combined_dimension':
                (
                    tfidf_matrix.shape[1]
                    +
                    entity_enbeddings.shape[1]
                    +
                    numeric_features.shape[1]
                    +
                    categorical_features.shape[1]
                ),

            'content_rating_columns':
                list(content_rating_columns),

            'language_columns':
                list(language_columns)
        }

        joblib.dump(
            metadata,
            os.path.join(
                ARTIFACT_DIR,
                'feature_metadata.joblib'
            )
        )

        return metadata

# =================================
# MAIN TRAINING PIPELINE
# =================================

def train():
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 60)

    # =============================
    # PREPROCESSING
    # =============================

    print("\n[1/7] Preprocessing dataset...")

    (
        df,
        preprocessing_artifacts,
    ) = fit_preprocessing(
        DATA_PATH
    )

    print(
        f"      Data shape: {df.shape}"
    )

    # ============================
    # TF-IDF
    # ============================

    print("\n[2/7] Training TF-IDF...")

    (
        df,
        tfidf_vectorizer,
        tfidf_matrix
    ) = train_tfidf(
        df
    )

    print(
        f"      TF-IDF shape: "
        f"{tfidf_matrix.shape}"
    )

    # ============================
    # WORD2VEC
    # ============================
    
    print("\n[3/7] Training Word2Vec...")

    (
        word2vec_model,
        entity_embeddings
    ) = train_word2vec(
        df
    )

    print(
        f"      Word2Vec embedding: "
        f"{entity_embeddings.shape}"
    )

    # ============================
    # NUMERIC
    # ============================

    print("\n[4/7] Preparing numeric features...")

    numeric_features = (
        get_numeric_features(
            df
        )
    )

    print(
        f"      Numeric shape: "
        f"{numeric_features.shape}"
    )

    # ============================
    # CATEGORICAL
    # ============================

    print(
        "\n[5/7] Preparing categorical features..."
    )

    (
        categorical_features,
        content_rating_columns,
        language_columns
    ) = get_categorical_features(
        df,
        preprocessing_artifacts
    )

    print(
        f"      Categorical shape: "
        f"{categorical_features.shape}"
    )

    # ============================
    # COMBINED EMBEDDINGS
    # ============================

    print(
        "\n[6/7] Building combined embeddings..."
    )

    combined_embeddings = (
        build_combined_embeddings(
            tfidf_matrix,
            entity_embeddings,
            numeric_features,
            categorical_features
        )
    )

    print(
        f"      Combined embedding shape: "
        f"{combined_embeddings.shape}"
    )

    # ============================
    # SAVE
    # ============================

    print(
        "\m[7/7] Saving artifacts..."
    )

    # ============================
    # TF-IDF
    # ============================

    tfidf_path = os.path.join(
        ARTIFACT_DIR,
        'tfidf.joblib'
    )

    joblib.dump(
        tfidf_vectorizer,
        tfidf_path
    )

    # ============================
    # Word2Vec
    # ============================

    word2vec_path = os.path.join(
        ARTIFACT_DIR,
        'word2vec.model'
    )

    word2vec_model.save(
        word2vec_path
    )

    # ============================
    # Combined embeddings
    # ============================

    embeddings_path = os.path.join(
        ARTIFACT_DIR,
        'embeddings.npy'
    )

    np.save(
        embeddings_path,
        combined_embeddings
    )

    # ============================
    # Movies metadata
    # ============================

    movies_path = os.path.join(
        ARTIFACT_DIR,
        'movies.csv'
    )

    movies = save_movies(
        df,
        movies_path
    )

    # ============================
    # Features metadata
    # ============================

    feature_metadata = (
        save_feature_metadata(
            tfidf_matrix,
            entity_embeddings,
            numeric_features,
            categorical_features,
            content_rating_columns,
            language_columns
        )
    )

    # ============================
    # Print result
    # ============================

    print("\n" + "=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)

    print(
        f"\nJumlah film        : {len(df)}"
    )

    print(
        f"TF-IDF dimension   : "
        f"{tfidf_matrix.shape[1]}"
    )

    print(
        f"Word2Vec dimension : "
        f"{entity_embeddings.shape[1]}"
    )

    print(
        f"Numeric dimension  : "
        f"{numeric_features.shape[1]}"
    )

    print(
        f"Categorical dim.   : "
        f"{categorical_features.shape[1]}"
    )

    print(
        f"Combined dimension : "
        f"{combined_embeddings.shape[1]}"
    )

    print("\nArtifacts: ")

    print(
        f"{tfidf_path}"
    )

    print(
        f"{word2vec_path}"
    )

    print(
        f"{embeddings_path}"
    )

    print(
        f"{movies_path}"
    )

    print(
        "preprocessing.joblib Completed"
    )

    print(
        "feature_metadata.joblib Completed"
    )

    return {
        'df': df,
        'tfidf_vectorizer':
            tfidf_vectorizer,
        'word2vec_model':
            word2vec_model,
        'embeddings':
            combined_embeddings,
        'movies':
            movies,
        'feature_metadata':
            feature_metadata
    }


if __name__ == "__main__":
    train()
