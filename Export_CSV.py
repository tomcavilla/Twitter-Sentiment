import argparse
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CITIES_SHAPEFILE = "Utilities/Shapefiles/CityBoundaries.shp"


def get_city_names():
    return gpd.read_file(CITIES_SHAPEFILE)["NAME"].tolist()


def check_city(city: str) -> bool:
    return city in get_city_names()


def city_tracts(state_tracts, city_name):
    tracts = gpd.read_file(state_tracts)

    city_bounds = gpd.read_file(CITIES_SHAPEFILE)
    chosen_bounds = city_bounds[city_bounds["NAME"] == city_name]

    tracts = tracts.to_crs(epsg=3857)
    tracts = gpd.sjoin(tracts, chosen_bounds, how="inner", predicate="intersects")
    tracts = tracts.reset_index(drop=True)
    tracts = tracts.drop(columns=[col for col in tracts.columns if col.startswith("index_")])
    tracts = tracts.to_crs(epsg=4326)

    return tracts


def load_tweets(city, tweet_folder):
    """Same parsing/caching logic as Generate.py / Generate_User_Agg.py — reuses their
    pickle if present. Includes the 'user' column so both tweet- and user-level
    aggregations below can be derived from the same cached data."""
    pickle_path = Path(__file__).parent.parent / "Pickles" / f"{city}_tweets.pkl"
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists():
        return pd.read_pickle(pickle_path)

    def parse_file(fp):
        rows = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    tweet = json.loads(line)
                    centroid = tweet.get("centroid")
                    if centroid is None:
                        continue
                    lat, lon = centroid
                    rows.append((lon, lat, tweet.get("timestamp"), tweet.get("sentiment_score"), tweet.get("user")))
                except json.JSONDecodeError:
                    continue
        return rows

    all_rows = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(parse_file, f): f for f in tweet_folder.glob("*.jsonl")}
        for future in as_completed(futures):
            all_rows.extend(future.result())

    tweets_df = pd.DataFrame(all_rows, columns=["lon", "lat", "timestamp", "sentiment", "user"])
    tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"], errors="coerce")
    tweets_df.to_pickle(pickle_path)
    return tweets_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, help="Chosen city to analyse.")
    parser.add_argument("--tracts", type=str, required=True, help="Shapefile of chosen city's state.")
    parser.add_argument("--out", type=str, required=False, help="Output directory (defaults to Out/).")
    parser.add_argument(
        "--min-tweets", type=int, default=1,
        help="Minimum total tweets a tract must have to be included (default: 1)."
    )
    args = parser.parse_args()

    if not check_city(args.city):
        raise ValueError(f"City '{args.city}' not found in analysed cities.")

    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "Out"
    data_dir = out_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tract geometry ----

    city_gdf = city_tracts(args.tracts, args.city)
    city_gdf["GEOID"] = city_gdf["GEOID"].astype(str).str.zfill(11)

    # ---- Census data ----

    census = pd.read_csv("Utilities/ICPSR-CENSUS.tsv", delimiter="\t", low_memory=False)
    census["TRACT_FIPS10"] = census["TRACT_FIPS10"].astype(str).str.zfill(11)
    census = census[census["TRACT_FIPS10"].isin(city_gdf["GEOID"].unique())].copy()

    # Income score (derived — weighted sum of income-bracket midpoints, not a NaNDA codebook variable)
    income_cols = ["PIN1B_13_17", "PIN2B_13_17", "PIN3B_13_17", "PIN4B_13_17", "PIN5B_13_17"]
    midpoints = [7.5, 22.5, 40, 75, 125]

    census_income = census[["TRACT_FIPS10"] + income_cols].copy()
    census_income[income_cols] = census_income[income_cols].apply(pd.to_numeric, errors="coerce")
    census_income = census_income.dropna()
    census_income["INCOME_SCORE"] = sum(census_income[col] * mid for col, mid in zip(income_cols, midpoints))

    all_tracts = gpd.read_file(args.tracts)
    all_tracts["GEOID"] = all_tracts["GEOID"].astype(str).str.zfill(11)

    gdf = all_tracts.merge(
        census_income[["TRACT_FIPS10", "INCOME_SCORE"]], left_on="GEOID", right_on="TRACT_FIPS10"
    )

    # All NaNDA Socioeconomic Status & Demographic Characteristics variables, ACS 2013-2017 wave
    # (openICPSR-119451 codebook). ALAND10 has no year suffix in the source dataset.
    CENSUS_VAR_COLS = [
        "ALAND10",
        "TOTPOP13_17",
        "POPDEN13_17",
        "PHISPANIC13_17",
        "PNHWHITE13_17",
        "PNHBLACK13_17",
        "PFBORN13_17",
        "PED1_13_17",
        "PED2_13_17",
        "PED3_13_17",
        "PIN1B_13_17",
        "PIN2B_13_17",
        "PIN3B_13_17",
        "PIN4B_13_17",
        "PIN5B_13_17",
        "PINCGT75K13_17",
        "PNVMAR13_17",
        "P18YR_13_17",
        "P18_2913_17",
        "P30_3913_17",
        "P40_4913_17",
        "P50_6913_17",
        "PGE7013_17",
        "PUNEMP13_17",
        "PPROF13_17",
        "PPOV13_17",
        "PPUBAS13_17",
        "PFHFAM13_17",
        "POWNOC13_17",
        "DISADVANTAGE13_17",
        "DISADVANTAGE2_13_17",
        "AFFLUENCE13_17",
        "ETHNICIMMIGRANT13_17",
    ]

    census_vars = census[["TRACT_FIPS10"] + CENSUS_VAR_COLS].copy()
    census_vars[CENSUS_VAR_COLS] = census_vars[CENSUS_VAR_COLS].apply(pd.to_numeric, errors="coerce")
    census_vars = census_vars.dropna(subset=CENSUS_VAR_COLS)

    # ---- Tweet-derived sentiment (tweet- and user-level) ----

    tweet_folder = Path(__file__).parent.parent / "CityTweets" / args.city

    if not tweet_folder.exists():
        raise FileNotFoundError(
            f"No tweet data found at {tweet_folder} — scatter data requires sentiment scores from Sentiment.py."
        )

    tweets_df = load_tweets(args.city, tweet_folder)

    tweets_gdf = gpd.GeoDataFrame(
        tweets_df,
        geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")
    joined["date"] = joined["timestamp"].dt.date

    # -- Tweet-level aggregation (mirrors Generate.py): every tweet counted once --

    tweet_data = (
        joined.dropna(subset=["sentiment", "date"])
        .groupby(["GEOID", "date"])
        .agg(mean_sentiment=("sentiment", "mean"), tweet_count=("sentiment", "size"))
        .reset_index()
        .sort_values(["GEOID", "date"])
    )

    tract_tweet_sentiment = (
        tweet_data.groupby("GEOID")
        .agg(mean_sentiment_by_tweet=("mean_sentiment", "mean"), total_tweets=("tweet_count", "sum"))
        .reset_index()
    )
    tract_tweet_sentiment["GEOID"] = tract_tweet_sentiment["GEOID"].astype(str).str.zfill(11)

    # -- User-level aggregation (mirrors Generate_User_Agg.py): mean of each
    # user's own mean sentiment, so a prolific tweeter can't outweigh someone
    # who only tweeted once in that tract --

    user_tract_sentiment = (
        joined.dropna(subset=["sentiment", "user"])
        .groupby(["GEOID", "user"], as_index=False)
        .agg(mean_sentiment=("sentiment", "mean"))
    )

    tract_user_sentiment = (
        user_tract_sentiment.groupby("GEOID", as_index=False)
        .agg(mean_sentiment_by_user=("mean_sentiment", "mean"), total_users=("user", "nunique"))
    )
    tract_user_sentiment["GEOID"] = tract_user_sentiment["GEOID"].astype(str).str.zfill(11)

    # -- Combine both sentiment sources on GEOID --

    combined_sentiment = tract_tweet_sentiment.merge(tract_user_sentiment, on="GEOID", how="outer")

    # ---- sentiment_census.csv ----

    sentiment_census = (
        combined_sentiment
        .merge(
            gdf[["GEOID", "INCOME_SCORE"]],
            on="GEOID",
            how="left"
        )
        .merge(
            census_vars,
            left_on="GEOID",
            right_on="TRACT_FIPS10",
            how="left"
        )
        .pipe(lambda d: d[d["total_tweets"] >= args.min_tweets])
        .drop(columns="TRACT_FIPS10")
        .dropna()
    )

    sentiment_census.to_csv(
        data_dir / f"{args.city}-Sentiment_Census.csv",
        index=False
    )

    print(f"Done — CSV written to {data_dir}")