import pandas as pd
import requests
import gzip
import io
import os

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NETFLIX_PATH = os.path.join(DATA_DIR, "netflix_data.csv")
ENRICHED_PATH = os.path.join(DATA_DIR, "enriched_data.csv")


def load_imdb_dataframe(url: str) -> pd.DataFrame:
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    compressed = gzip.GzipFile(fileobj=io.BytesIO(response.content))
    return pd.read_csv(compressed, sep="\t", low_memory=False, na_values="\\N")


def build_enriched_dataset():
    # Netflix data loaded
    netflix = pd.read_csv(NETFLIX_PATH)
    print(f"Loaded Netflix data: {len(netflix)} rows")

    # Load IMDB basics 
    basics = load_imdb_dataframe(IMDB_BASICS_URL)
    basics = basics[basics["titleType"].isin(["movie", "tvSeries", "tvMiniSeries"])]
    basics = basics[["tconst", "primaryTitle", "startYear", "genres"]]
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")

    # Load IMDB ratings
    ratings = load_imdb_dataframe(IMDB_RATINGS_URL)
    ratings = ratings[["tconst", "averageRating", "numVotes"]]

    # Merge IMDB basics + ratings
    imdb = basics.merge(ratings, on="tconst", how="left")

    # Normalize titles for matching
    netflix["title_lower"] = netflix["title"].str.lower().str.strip()
    netflix["release_year"] = pd.to_numeric(netflix["release_year"], errors="coerce")
    imdb["title_lower"] = imdb["primaryTitle"].str.lower().str.strip()

    # Match
    merged = netflix.merge(
        imdb,
        left_on=["title_lower", "release_year"],
        right_on=["title_lower", "startYear"],
        how="left"
    )

    # Drop duplicates 
    merged = merged.sort_values("averageRating", ascending=False)
    merged = merged.drop_duplicates(subset=["show_id"])

    # Build enriched content field for embeddings
    def build_content(row):
        parts = [
            str(row.get("title", "")),
            str(row.get("description", "")),
            str(row.get("listed_in", "")),
            str(row.get("cast", "")),
            str(row.get("director", "")),
            str(row.get("genres", "")),  # IMDB genres
        ]
        return " ".join(p for p in parts if p and p != "nan")

    merged["content"] = merged.apply(build_content, axis=1)

    # saves
    merged.to_csv(ENRICHED_PATH, index=False)
    print(f"Enriched dataset saved: {len(merged)} rows → {ENRICHED_PATH}")
    return merged


if __name__ == "__main__":
    build_enriched_dataset()