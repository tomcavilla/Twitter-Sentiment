import argparse
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from timezonefinder import TimezoneFinder

CITIES_SHAPEFILE = "Utilities/Shapefiles/CityBoundaries.shp"

POS_THRESHOLD = 0.65
NEG_THRESHOLD = 0.35


def to_local(series, tz):
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC").dt.tz_convert(tz)
    return series.dt.tz_convert(tz)


def check_city(city: str) -> bool:
    return city in gpd.read_file(CITIES_SHAPEFILE)["NAME"].tolist()


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


def load_tweets(city, tweet_folder, pickle_path):
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
    pickle_path.parent.mkdir(exist_ok=True)
    tweets_df.to_pickle(pickle_path)
    return tweets_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, help="Chosen city to analyse.")
    parser.add_argument("--tracts", type=str, required=True, help="Shapefile of chosen city's state.")
    parser.add_argument("--out", type=str, required=False, help="The output directory for generated CSVs.")
    args = parser.parse_args()

    if not check_city(args.city):
        raise ValueError(f"City '{args.city}' not found in analysed cities.")

    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "Out" / "Extremes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build city tract geometry and detect local timezone, matching Generate.py
    city_gdf = city_tracts(args.tracts, args.city)
    minx, miny, maxx, maxy = city_gdf.total_bounds
    local_tz = TimezoneFinder().timezone_at(lng=(minx + maxx) / 2, lat=(miny + maxy) / 2) or "UTC"

    tweet_folder = Path(__file__).parent.parent / "CityTweets" / args.city
    if not tweet_folder.exists():
        raise FileNotFoundError(f"No tweet folder found for '{args.city}' at {tweet_folder}")

    pickle_path = Path(__file__).parent.parent / "Pickles" / f"{args.city}_tweets.pkl"
    tweets_df = load_tweets(args.city, tweet_folder, pickle_path)

    tweets_gdf = gpd.GeoDataFrame(
        tweets_df,
        geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")

    # Rebuild sent_temporal exactly as in Generate.py
    sent_temporal = joined[["timestamp", "sentiment"]].dropna().copy()
    sent_temporal["timestamp"] = to_local(sent_temporal["timestamp"], local_tz)
    sent_temporal["hour"] = sent_temporal["timestamp"].dt.hour
    sent_temporal["dayofweek"] = sent_temporal["timestamp"].dt.dayofweek
    sent_temporal["month"] = sent_temporal["timestamp"].dt.month
    sent_temporal["date"] = sent_temporal["timestamp"].dt.date

    sent_temporal["positive"] = sent_temporal["sentiment"] > POS_THRESHOLD
    sent_temporal["negative"] = sent_temporal["sentiment"] < NEG_THRESHOLD

    day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # --- Extremes by Hour ---
    hour_grp = sent_temporal.groupby("hour")[["positive", "negative"]].mean().reindex(range(24)) * 100
    hour_grp = hour_grp.rename(columns={"positive": "pct_positive", "negative": "pct_negative"})
    hour_grp.index.name = "hour"
    hour_grp.reset_index().to_csv(out_dir / f"{args.city}-Sentiment_Extremes_Hour.csv", index=False)

    # --- Extremes by Day of Week ---
    day_grp = sent_temporal.groupby("dayofweek")[["positive", "negative"]].mean().reindex(range(7)) * 100
    day_grp = day_grp.rename(columns={"positive": "pct_positive", "negative": "pct_negative"})
    day_grp.index = day_labels
    day_grp.index.name = "day_of_week"
    day_grp.reset_index().to_csv(out_dir / f"{args.city}-Sentiment_Extremes_Day.csv", index=False)

    # --- Extremes by Month ---
    month_grp = sent_temporal.groupby("month")[["positive", "negative"]].mean().reindex(range(1, 13)) * 100
    month_grp = month_grp.rename(columns={"positive": "pct_positive", "negative": "pct_negative"})
    month_grp.index = month_labels
    month_grp.index.name = "month"
    month_grp.reset_index().to_csv(out_dir / f"{args.city}-Sentiment_Extremes_Month.csv", index=False)

    # --- Extremes Time Series (raw daily + 7-day rolling, as plotted) ---
    daily_pos = sent_temporal.groupby("date")["positive"].mean().sort_index() * 100
    daily_neg = sent_temporal.groupby("date")["negative"].mean().sort_index() * 100
    daily_pos.index = pd.to_datetime(daily_pos.index)
    daily_neg.index = pd.to_datetime(daily_neg.index)

    roll_pos = daily_pos.rolling(7, center=True, min_periods=1).mean()
    roll_neg = daily_neg.rolling(7, center=True, min_periods=1).mean()

    timeseries = pd.DataFrame({
        "date": daily_pos.index,
        "pct_positive": daily_pos.values,
        "pct_negative": daily_neg.values,
        "pct_positive_7day_rolling": roll_pos.values,
        "pct_negative_7day_rolling": roll_neg.values,
    })
    timeseries.to_csv(out_dir / f"{args.city}-Sentiment_Extremes_Timeseries.csv", index=False)

    print(f"Exported extremes CSVs for {args.city} to {out_dir}")