import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_PATH = os.path.join(DATA_DIR, "enriched_data.csv")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cache model at module load time — fixes the slow reload on every request
_model = SentenceTransformer(EMBEDDING_MODEL)


def build_index():
    print("Loading enriched data...")
    df = pd.read_csv(ENRICHED_PATH).fillna("")

    print("Computing TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content"])

    print("Computing sentence embeddings (this takes a few minutes)...")
    embeddings = _model.encode(df["content"].tolist(), show_progress_bar=True)

    index = {
        "df": df,
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "embeddings": embeddings,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Index saved to {INDEX_PATH}")
    return index


def load_index():
    if not os.path.exists(INDEX_PATH):
        return build_index()
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def recommend(
    query: str,
    index: dict,
    top_k: int = 5,
    content_type: str = "Both",
    min_imdb: float = 0.0,
) -> list[dict]:
    df = index["df"].copy()
    tfidf = index["tfidf"]
    tfidf_matrix = index["tfidf_matrix"]
    embeddings = index["embeddings"]

    # Apply filters before scoring
    mask = np.ones(len(df), dtype=bool)

    if content_type in ("Movie", "TV Show"):
        mask &= df["type"].values == content_type

    if min_imdb > 0:
        ratings = pd.to_numeric(df["averageRating"], errors="coerce").fillna(0)
        mask &= ratings.values >= min_imdb

    if not mask.any():
        mask = np.ones(len(df), dtype=bool)  # fallback: ignore filters

    filtered_df = df[mask].reset_index(drop=True)
    filtered_embeddings = embeddings[mask]
    filtered_tfidf = tfidf_matrix[mask]

    # Semantic similarity
    query_embedding = _model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, filtered_embeddings)[0]

    # TF-IDF similarity
    query_tfidf = tfidf.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, filtered_tfidf)[0]

    # IMDB rating signal
    rating_scores = (
        pd.to_numeric(filtered_df["averageRating"], errors="coerce").fillna(0) / 10.0
    ).values

    # Weighted hybrid score
    hybrid_scores = (
        0.5 * semantic_scores +
        0.3 * tfidf_scores +
        0.2 * rating_scores
    )

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = filtered_df.iloc[idx]
        imdb_raw = str(row.get("averageRating", ""))
        results.append({
            "title": row.get("title", ""),
            "type": row.get("type", ""),
            "release_year": str(row.get("release_year", "")),
            "rating": row.get("rating", ""),
            "imdb_score": round(float(imdb_raw), 1) if imdb_raw not in ["", "nan"] else None,
            "genres": row.get("listed_in", ""),
            "description": row.get("description", ""),
            "score": round(float(hybrid_scores[idx]), 4),
        })

    return results