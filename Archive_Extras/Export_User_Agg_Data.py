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
    """
    Same parsing/caching logic as Generate_User_Agg.py, including the
    "user" field, so per-user sentiment can be computed. Note this pickle
    schema (lon, lat, timestamp, sentiment, user) differs from the one
    written by Export_Scatter_Data.py / Generate.py (no "user" column),
    even though both cache to the same Pickles/{city}_tweets.pkl path.
    If that file already exists from the tweet-volume pipeline, delete it
    first so it gets rebuilt with the "user" column included.
    """
    pickle_path = Path(__file__).parent.parent / "Pickles" / f"{city}_tweets.pkl"
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists():
        cached = pd.read_pickle(pickle_path)
        if "user" in cached.columns:
            return cached
        # Stale cache from the tweet-volume pipeline - rebuild with "user".

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
    args = parser.parse_args()

    if not check_city(args.city):
        raise ValueError(f"City '{args.city}' not found in analysed cities.")

    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "Out"
    data_dir = out_dir / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tract geometry ----

    city_gdf = city_tracts(args.tracts, args.city)
    city_gdf["GEOID"] = city_gdf["GEOID"].astype(str).str.zfill(11)

    # ---- Census data ----

    census = pd.read_csv("Utilities/ICPSR-CENSUS.tsv", delimiter="\t", low_memory=False)
    census["TRACT_FIPS10"] = census["TRACT_FIPS10"].astype(str).str.zfill(11)
    census = census[census["TRACT_FIPS10"].isin(city_gdf["GEOID"].unique())].copy()

    # Income score (derived - weighted sum of income-bracket midpoints, not a NaNDA codebook variable)
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

    # ---- Tweet-derived sentiment (user-aggregated) ----

    tweet_folder = Path(__file__).parent.parent / "CityTweets" / args.city

    if not tweet_folder.exists():
        raise FileNotFoundError(
            f"No tweet data found at {tweet_folder} - scatter data requires sentiment scores from Sentiment.py."
        )

    tweets_df = load_tweets(args.city, tweet_folder)

    tweets_gdf = gpd.GeoDataFrame(
        tweets_df,
        geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")

    # Collapse to one sentiment value per user per tract first, so a single
    # prolific tweeter in a tract can't outweigh a user who tweeted once
    # there. Everything downstream is built from this per-user base, not
    # raw tweet counts.
    user_tract_sentiment = (
        joined.dropna(subset=["sentiment", "user"])
        .groupby(["GEOID", "user"], as_index=False)
        .agg(mean_sentiment=("sentiment", "mean"))
    )

    tract_sentiment = (
        user_tract_sentiment.groupby("GEOID")
        .agg(mean_sentiment=("mean_sentiment", "mean"), total_users=("user", "nunique"))
        .reset_index()
    )
    tract_sentiment["GEOID"] = tract_sentiment["GEOID"].astype(str).str.zfill(11)

    # ---- user_agg_sentiment_census.csv ----

    sentiment_census = (
        tract_sentiment
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
        .pipe(lambda d: d[d["total_users"] >= 10])
        .drop(columns="TRACT_FIPS10")
        .dropna()
    )

    sentiment_census.to_csv(
        data_dir / f"{args.city}-User_Agg_Sentiment_Census.csv",
        index=False
    )

    print(f"Done - CSV written to {data_dir}")