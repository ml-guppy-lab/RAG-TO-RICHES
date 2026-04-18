import pandas as pd


def merge_data():
    # read files
    movies = pd.read_csv("data/movies-names.txt", sep="~", header=None, names=["movie_id", "movie_name"], encoding="utf-8")
    quotes = pd.read_csv("data/movies-quotes.txt", sep="~", header=None, names=["movie_id", "dialogue_number", "dialogue"], encoding="utf-8")

    # merge on movie_id
    merged = pd.merge(quotes, movies, on="movie_id", how="left")
    # reorder columns
    merged = merged[["movie_id", "movie_name", "dialogue_number", "dialogue"]]

    return merged


def clean_data(df):
    print(f"Before cleaning: {len(df)} rows")

    # drop rows with null movie names or dialogues
    df = df.dropna(subset=["movie_name", "dialogue"])
    print(f"After removing null movie names/dialogues: {len(df)} rows")

    # drop rows with dialogues under 6 words
    df = df[df["dialogue"].str.split().str.len() >= 6]
    print(f"After removing dialogues under 6 words: {len(df)} rows")

    # drop duplicate dialogues
    df = df.drop_duplicates(subset=["dialogue"])
    print(f"After removing duplicate dialogues: {len(df)} rows")

    # write to csv
    df.to_csv("data/movies-dialogues.csv", index=False, encoding="utf-8")
    print(f"Done. {len(df)} rows written to data/movies-dialogues.csv")


merged = merge_data()
clean_data(merged)

