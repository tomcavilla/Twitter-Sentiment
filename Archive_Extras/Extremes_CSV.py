"""
export_extreme_sentiment_timeseries.py

Exports the data behind the "Extreme Sentiment Time Series"
(<city>-Sentiment_Extremes_Timeseries.png) figure from Generate.py as a CSV
file, for a chosen city.

It reproduces exactly the pipeline used by Generate.py to reach that figure:
    1. Load the city's census tracts (city_tracts) and detect its local timezone.
    2. Load / spatially join the city's tweets against those tracts.
    3. Convert tweet timestamps to local time.
    4. Flag each tweet as "positive" (sentiment > POS_THRESHOLD) or
       "negative" (sentiment < NEG_THRESHOLD).
    5. Aggregate the daily % positive / % negative, plus their 7-day
       centered rolling means (matching the plotted lines exactly).

Usage:
    python export_extreme_sentiment_timeseries.py --city Chicago --tracts path/to/state_tracts.shp --out output.csv

Notes:
    - This script must be run from the same location Generate.py expects
      (i.e. it looks for tweets in "<repo_root>/CityTweets/<city>" and a
      pickle cache in "<repo_root>/Pickles/<city>_tweets.pkl", exactly like
      Generate.py does), so place this file alongside Generate.py.
    - Requires the same dependencies as Generate.py: geopandas, pandas,
      numpy, timezonefinder.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

# Import the tract-loading / city-checking helpers from Generate.py so the
# pipeline stays identical to the report generator.
from Generate import city_tracts, check_city, to_local

POS_THRESHOLD = 0.65
NEG_THRESHOLD = 0.35


def load_city_tweets(city: str) -> pd.DataFrame:
    """Load (and cache) raw tweets for a city, exactly as Generate.py does."""
    tweet_folder = Path(__file__).parent.parent / "CityTweets" / city

    if not tweet_folder.exists():
        raise FileNotFoundError(f"No tweet folder found for city '{city}' at {tweet_folder}")

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


def build_extreme_sentiment_timeseries(city: str, tracts_shapefile: str) -> pd.DataFrame:
    """Reproduce sent_temporal and the daily positive/negative % + 7-day rolling means."""

    # 1. Tracts + local timezone (identical to Generate.py)
    city_gdf = city_tracts(tracts_shapefile, city)
    minx, miny, maxx, maxy = city_gdf.total_bounds
    local_tz = TimezoneFinder().timezone_at(lng=(minx + maxx) / 2, lat=(miny + maxy) / 2) or "UTC"

    # 2. Tweets + spatial join (identical to Generate.py)
    tweets_df = load_city_tweets(city)
    tweets_gdf = gpd.GeoDataFrame(
        tweets_df,
        geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")

    # 3. sent_temporal (identical to Generate.py)
    sent_temporal = joined[["timestamp", "sentiment"]].dropna().copy()
    sent_temporal["timestamp"] = to_local(sent_temporal["timestamp"], local_tz)
    sent_temporal["date"] = sent_temporal["timestamp"].dt.date

    # 4. Extreme sentiment flags (identical to Generate.py)
    sent_temporal["positive"] = sent_temporal["sentiment"] > POS_THRESHOLD
    sent_temporal["negative"] = sent_temporal["sentiment"] < NEG_THRESHOLD

    # 5. Daily % positive / negative + 7-day centered rolling means
    daily_pos = sent_temporal.groupby("date")["positive"].mean().sort_index() * 100
    daily_neg = sent_temporal.groupby("date")["negative"].mean().sort_index() * 100
    daily_pos.index = pd.to_datetime(daily_pos.index)
    daily_neg.index = pd.to_datetime(daily_neg.index)

    roll_pos = daily_pos.rolling(7, center=True, min_periods=1).mean()
    roll_neg = daily_neg.rolling(7, center=True, min_periods=1).mean()

    out = pd.DataFrame({
        "date": daily_pos.index,
        "pct_positive": daily_pos.values,
        "pct_negative": daily_neg.values,
        "pct_positive_7d_rolling": roll_pos.values,
        "pct_negative_7d_rolling": roll_neg.values,
    })

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the Extreme Sentiment Time Series figure's data as a CSV for a chosen city."
    )
    parser.add_argument("--city", type=str, required=True, help="Chosen city to analyse.")
    parser.add_argument("--tracts", type=str, required=True, help="Shapefile of chosen city's state.")
    parser.add_argument("--out", type=str, required=False, default=None,
                         help="Output CSV file path (default: <city>-Sentiment_Extremes_Timeseries.csv)")
    args = parser.parse_args()

    if check_city(args.city) is False:
        raise ValueError(f"City '{args.city}' not found in analysed cities.")

    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")

    df = build_extreme_sentiment_timeseries(args.city, args.tracts)

    out_path = Path(args.out) if args.out else Path(f"{args.city}-Sentiment_Extremes_Timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")