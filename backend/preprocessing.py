# backend/preprocessing.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec


# ============================
# PATH
# ============================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "artifacts"
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)

DROP_COLS = [
    'num_critic_for_reviews',
    'director_facebook_likes',
    'actor_3_facebook_likes',
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'gross',
    'num_voted_users',
    'cast_total_facebook_likes',
    'facenumber_in_poster',
    'movie_imdb_link',
    'num_user_for_reviews',
    'budget',
    'imdb_score',
    'movie_facebook_likes',
    'aspect_ratio'
]

STRING_COLS = [
    'color',
    'director_name',
    'actor_1_name',
    'actor_2_name',
    'genres',
    'actor_3_name',
    'movie_title',
    'plot_keywords',
    'language',
    'country',
    'content_rating'
]

# =======================
# LOAD + DROP + DUPLICATE
# =======================

def load_and_basic_cleaning(
        csv_path
):
    """

    1. load dataset
    2. Drop inrelevant columns
    3. Remove duplicate

    """

    df = pd.read_csv(csv_path)

    df.drop(
        columns=DROP_COLS
    )

    df = df.drop_duplicates()

    return df

# ===========================
# COLOR
# ===========================

def handle_color(df):
    """
    Missing color -> mode

    """

    mode_color = (
        df['color']
        .mode()[0]
    )

    df['color'] = (
        df['color']
        .fillna(mode_color)
    )

    return df, mode_color

# =========================
# DIRECTOR NAME
# =========================

def handle_director(df):
    """
    Missing director_name -> unknown
    
    """

    df['director_name'] = (
        df['director_name']
        .fillna('unknown')
    )

    return df

# =========================
# DURATION
# =========================

def handle_duration(df):
    """
    Missing duration -> median 
    
    """

    median_duration = (
        df['duration']
        .median()
    )

    df['duration'] = (
        df['duration']
        .fillna(median_duration)
    )

    return df, median_duration

# =========================
# ACTORS
# =========================

def handle_actors(df):
    """
    Missing actor -> 'unknown
    
    """

    df['actor_1_name'] = (
        df['actor_1_name']
        .fillna('unknown')
    )

    df['actor_2_name'] = (
        df['actor_2_name']
        .fillna('unknown')
    )

    df['actor_3_name'] = (
        df['actor_3_name']
        .fillna('unknown')
    )

    return df


# ======================
# PLOT KEYWORDS
# ======================

def handle_plot_keywords(df):
    """
    jika plot_keywords kosong:
    cari film dengan director yang sama dan genre
     yang sama

     kemudian gunakan plot_keywords film tersebut

     jika tidak ditemukan -> unknown 
    
    """

    def impute_keywords(row):

        if pd.isna(
            row['plot_keywords']
        ):

            similar = df[
                (df['director_name'] == row['director_name'])
                &
                (df['genres'] == row['genres'])
                &
                (df['plot_keywords']
                 .notna())
            ]

            if not similar.empty:
                return(
                    similar['plot_keywords'].iloc[0]
                )
            else:
                return 'unknown'

        else:
            return row['plot_keywords']

    df['plot_keywords'] = (
        df.apply(
            impute_keywords,
            axis = 1
        )
    )

    return df

# ====================
# COUNTRY
# ====================

def handle_country(df):
    """
    jika country kosong dan language tersedia:
    gunakan mode country berdasarkan language

    jika tidah tersedia:
    unknown

    """

    def impute_country(row):

        if pd.isna(
            row['country']
        ):
            if not pd.isna(
                row['language']
            ):

                countries = (
                    df['country'][
                        df['language']
                        == row['language']
                    ]
                    .dropna()
                )

                if not countries.empty:
                    return (
                        countries
                        .mode()[0]
                    )

                return 'unknown'

            else:
                return 'unknown'

        else:
            return row['country']

    df['country'] = (
        df.apply(
            impute_country,
            axis=1
            )
    )

    return df

# =======================
# LANGUAGE
# =======================

def handle_language(df):
    """
    
    jika language kosong:
    gunakan mode langugae berdasarkan country

    jika tidak ditemukan:
    unknown
    
    """

    def impute_lan(row):

        if pd.isna(
            row['language']
        ):
            if row['country'] != 'unknown':
                langs = (
                    df.loc[
                        df['country']
                        == row['country'],
                        'language'
                    ]
                    .dropna()
                )

                if not langs.empty:
                    return (
                        langs.mode()[0]
                    )
                else:
                    return 'unknown'
            else:
                return 'unknown'
        else:
            return row['language']

    df['language'] = (
        df.apply(
            impute_lan,
            axis=1
        )
    )

    return df

# ======================
# CONTENT RATING
# ======================

def normalize_content_rating(df):
    """
    
    Normalisasi content_rating sesuai
    
    """

    def impute_con_rat(row):

        value = row[
            'content_rating'
        ]

        if not pd.isna(value):
            if value == 'Not Rated':
                return 'unrated'
            elif(
                value == 'Passed'
                or
                value == 'Approved'
            ):
                return 'legacy_approved'
            elif(
                value == 'TV-Y'
                or
                value == 'TV-Y7'
                or
                value == 'TV-G'
            ):
                return 'G'
            elif(
                value == 'R'
                or
                value == 'TV-MA'
            ):
                return 'R'
            elif(
                value == 'GP'
                or 
                value == 'M'
                or
                value == 'TV-PG'
            ):
                return 'PG'
            elif(
                value == 'TV-14'
            ):
                return 'PG-13'
            elif value == 'X':
                return 'NC-17'
            else:
                return value

        else:
            return 'unknown'

    df['content_rating'] = (
        df.apply(
            impute_con_rat,
            axis=1
        )
    )

    return df

# ======================
# TITLE YEAR
# ======================

def handle_title_year(df):
    """
    
    1. year_unknown = 1 jika NaN
    2. Nan title_year -> median
    
    """

    df['year_unknown'] = (
        df['title_year']
        .isna()
        .astype(int)
    )

    median_year = (
        df['title_year']
        .median()
    )

    df['title_year'] = (
        df['title_year']
        .fillna(median_year)
    )

    return df, median_year

# ===========================
# Min Max Scaling
# ===========================

def scale_numeric_features(df):
    # scaler duration 
    duration_scaler = (
        MinMaxScaler()
    )

    df['duration'] = (
        duration_scaler
        .fit_transform(
            df[['duration']]
        )
    )

    # scaler title_year
    year_scaler = (
        MinMaxScaler()
    )

    df['title_year'] = (
        year_scaler
        .fit_transform(
            df[['title_year']]
        )
    )

    return (
        df,
        duration_scaler,
        year_scaler
    )

# =========================
# STRING NORMALIZATION
# =========================

def normalize_strings(df):
    """
    
    Semua string:
     - lowercase
     - strip whitespace
    
    """

    df[STRING_COLS] = (
        df[STRING_COLS]
        .apply(
            lambda x:
                x.str.lower()
        )
    )

    df[STRING_COLS] = (
        df[STRING_COLS]
        .apply(
            lambda x:
                x.str.strip()
        )
    )

    return df

# ==============================
# SPLIT GENRES + PLOT KEYWORDS
# ==============================

def split_text_features(df):
    """
    
    genres:
        Action|Adventure|Fantasy
        ->
        ['Action', 'Adventure', 'Fantasy']

    plot_keywords:
        hero|space|future
        ->
        ['hero', 'space', 'future']
    
    """

    df['plot_keywords'] = (
        df['plot_keywords']
        .apply(
            lambda x:
                x.lower()
        )
    )

    df['plot_keywords'] = (
        df['plot_keywords']
        .apply(
            lambda x:
                x.split('|')
                if (
                    isinstance(x, str)
                    and
                    x != 'unknown'
                )
                else []
        )
    )

    df['genres'] = (
        df['genres']
        .apply(
            lambda x:
                x.lower()
        )
    )

    df['genres'] = (
        df['genres']
        .apply(
            lambda x:
                x.split('|')
                if(
                    isinstance(x, str)
                    and
                    x != 'unknown'
                )
                else []
        )
    )

    return df

# =======================
# LANGUAGE GROUPING
# =======================

def group_languages(df):
    """
    
    Pertahankan 10 bahasa paling populer
    bahasa lainnya -> other
    
    """

    top_lan = (
        df['language']
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    def grouping_lan(row):
        if(
            row['language']
            not in top_lan
        ):
            return 'other'
        else:
            return row['language']

    df['language'] = (
        df.apply(
            grouping_lan,
            axis=1
        )
    )

    return df, top_lan

# ===========================
# COUNTRY FREQUENCY ENCODING
# ===========================

def frequency_encode_country(df):
    """
    
    country -> frequency count
    
    """

    freq_map = (
        df['country']
        .value_counts()
        .to_dict()
    )

    df['country_freq'] = (
        df['country']
        .map(freq_map)
    )

    return df, freq_map

# =========================
# ONE HOT ENCODING
# =========================

def ohe(df):
    """
    
    One-Hot:
        content_rating
        language

    
    """

    # =======================
    # Content Rating Encoder
    # =======================

    content_rating_encoder = (
        OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
    )

    content_rating_encoded = (
        content_rating_encoder
        .fit_transform(
            df[['content_rating']]
        )
    )

    content_rating_df = (
        pd.DataFrame(
            content_rating_encoded,
            columns=(
                content_rating_encoder
                .get_feature_names_out(
                    ['content_rating']
                )
            )
        )
    )

    # ==========================
    # Language Encoder
    # ==========================

    language_encoder = (
        OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
    )

    language_encoded = (
        language_encoder
        .fit_transform(
            df[['language']]
        )
    )

    language_df = (
        pd.DataFrame(
            language_encoded,
            columns=(
                language_encoder
                .get_feature_names_out(
                    ['language']
                )
            )
        )
    )

    # ========================
    # Combine
    # ========================

    df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            content_rating_df.reset_index(
                drop=True
            ),
            language_df.reset_index(
                drop=True
            )
        ],
        axis=1 
    )

    return (
        df,
        content_rating_encoder,
        language_encoder
    )

# ======================================
# MAIN TRAINING PREPROCESSING PIPELINE
# ======================================

def fit_preprocessing(
        csv_path
):
    """
    
    Menjalankan preprocessing
    
    return: 
        df
        preprocessing_artifacts

    """

    df = load_and_basic_cleaning(
        csv_path
    )

    df, mode_color = (
        handle_color(df)
    )

    df = handle_director(
        df
    )

    (
        df,
        median_duration,
    ) = handle_duration(df)

    df = handle_actors(
        df
    )

    df = handle_plot_keywords(
        df
    )

    df = handle_country(
        df
    )

    df = handle_language(
        df
    )

    df = normalize_content_rating(
        df
    )

    (
        df,
        median_year
    ) = handle_title_year(
        df
    )

    (
        df,
        duration_scaler,
        year_scaler
    ) = scale_numeric_features(
        df
    )

    df = normalize_strings(
        df
    )

    df = split_text_features(
        df
    )


    (
        df,
        top_lans
    ) = group_languages(
        df
    )

    (
        df,
        country_freq_map
    ) = frequency_encode_country(
        df
    )

    (
        df,
        content_rating_encoder,
        language_encoder
    ) = ohe(
        df
    )

    # =====================
    # SAVE PREPROCESSING
    # =====================

    preprocessing_artifacts = {

        # Missing value parameters
        'mode_color':
            mode_color,
        'median_duration':
            median_duration,
        'median_year':
            median_year,
        'top_lans':
            top_lans,
        'country_freq_map':
            country_freq_map,
        'duration_scaler':
            duration_scaler,
        'year_scaler':
            year_scaler,
        'content_rating_encoder':
            content_rating_encoder,
        'language_encoder':
            language_encoder,
        'drop_cols':
            DROP_COLS,
        'string_cols':
            STRING_COLS
    }

    joblib.dump(
        preprocessing_artifacts,
        os.path.join(
            ARTIFACT_DIR,
            'preprocessing.joblib'
        )
    )

    return (
        df,
        preprocessing_artifacts
    )

# =============================
# TEST
# =============================

if __name__ == "__main__":

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "movie_metadata.csv"
    )

    print('==================================================================================')

    print(
        "Loading dataset"
    )

    print(DATA_PATH)

    df_processed, artifacts = (
        fit_preprocessing(
            DATA_PATH
        )
    )

    print(
        "\nPreprocessing Selesai."
    )

    print(
        "\nShape:"
    )

    print(
        df_processed.shape
    )

    print(
        "\nColumns:"
    )

    print(
        df_processed.columns.tolist()
    )

    print(
        "\nArtifacts disimpan:"
    )

    print(
        os.path.join(
            ARTIFACT_DIR,
            'preprocessing.joblib'
        )
    )

    print('==================================================================================')

