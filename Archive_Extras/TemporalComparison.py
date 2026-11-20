"""
TemporalComparison.py

Standalone companion to Generate.py. Produces a single grouped, colour-coded
bar chart figure comparing tweet-frequency temporal patterns (Hour of Day,
Day of Week, Month) across New York, Los Angeles, Chicago, Houston, and San
Antonio on the same axes, with one shared legend outside to the right.

Reuses the exact same loading / spatial-join / timezone-localisation
pipeline as Generate.py (imported directly from it), so results are
consistent with the per-city figures already generated.

Usage:
python TemporalComparison.py --new-york-tracts 'Utilities/Shapefiles/New York Tracts/tl_2017_36_tract.shp' --los-angeles-tracts 'Utilities/Shapefiles/California Tracts/tl_2017_06_tract.shp' --chicago-tracts 'Utilities/Shapefiles/Illinois Tracts/tl_2017_17_tract.shp' --houston-tracts 'Utilities/Shapefiles/Texas Tracts/tl_2017_48_tract.shp' --san-antonio-tracts 'Utilities/Shapefiles/Texas Tracts/tl_2017_48_tract.shp' --out ../Out/Figures/Temporal5
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

# ---- Reuse existing pipeline helpers from Generate.py ----
sys.path.insert(0, str(Path(__file__).parent))
from Archive_Code.Generate import check_city, city_tracts, to_local  # noqa: E402

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]

# Distinct, colour-blind-friendly palette, one colour per city
CITY_COLORS = {
    "New York":    "#06402B",
    "Los Angeles": "#ff7f0e",
    "Chicago":     "#1f77b4",
    "Houston":     "#d62728",
    "San Antonio": "#9467bd",  
}


def parse_tweet_file(fp):
    """Parse a single .jsonl tweet file into (lon, lat, timestamp, sentiment) rows."""
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                tweet = json.loads(line)
                centroid = tweet.get("centroid")
                if centroid is None:
                    continue
                lat, lon = centroid
                rows.append((lon, lat, tweet.get("timestamp"), tweet.get("sentiment_score")))
            except json.JSONDecodeError:
                continue
    return rows


def load_city_timestamps(city, tracts_path):
    """
    Load (and cache) a city's tweets, spatially join to its census tracts,
    localise timestamps to the city's local timezone, and return the
    localised timestamp Series.

    Mirrors the loading logic in Generate.py so results match the
    per-city figures already produced there, including reuse of the same
    Pickles/{city}_tweets.pkl cache.
    """
    project_root = Path(__file__).parent.parent

    tweet_folder = project_root / "CityTweets" / city
    if not tweet_folder.exists():
        raise FileNotFoundError(f"No tweet folder found for '{city}' at {tweet_folder}")

    pickle_path = project_root / "Pickles" / f"{city}_tweets.pkl"
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists():
        tweets_df = pd.read_pickle(pickle_path)
    else:
        all_rows = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(parse_tweet_file, f): f for f in tweet_folder.glob("*.jsonl")}
            for future in as_completed(futures):
                all_rows.extend(future.result())

        tweets_df = pd.DataFrame(all_rows, columns=["lon", "lat", "timestamp", "sentiment"])
        tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"], errors="coerce")
        tweets_df.to_pickle(pickle_path)

    city_gdf = city_tracts(tracts_path, city)

    minx, miny, maxx, maxy = city_gdf.total_bounds
    local_tz = TimezoneFinder().timezone_at(lng=(minx + maxx) / 2, lat=(miny + maxy) / 2) or "UTC"

    tweets_gdf = gpd.GeoDataFrame(
        tweets_df,
        geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")

    return to_local(joined["timestamp"].dropna(), local_tz)


def bucket_counts(ts, freq_attr, idx, as_percent):
    """Count tweets per bucket (hour / dayofweek / month), reindexed to idx."""
    counts = getattr(ts.dt, freq_attr).value_counts().reindex(idx, fill_value=0)
    if as_percent:
        total = counts.sum()
        return counts / total * 100 if total else counts
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-york-tracts", required=True, help="Tracts shapefile covering New York.")
    parser.add_argument("--los-angeles-tracts", required=True, help="Tracts shapefile covering Los Angeles.")
    parser.add_argument("--chicago-tracts", required=True, help="Tracts shapefile covering Chicago.")
    parser.add_argument("--houston-tracts", required=True, help="Tracts shapefile covering Houston.")
    parser.add_argument("--san-antonio-tracts", required=True, help="Tracts shapefile covering San Antonio.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for figures.")
    args = parser.parse_args()

    cities = {
        "New York":    args.new_york_tracts,
        "Los Angeles": args.los_angeles_tracts,
        "Houston":     args.houston_tracts,
        "Chicago":     args.chicago_tracts,
        "San Antonio": args.san_antonio_tracts,
    }

    for city, tracts_path in cities.items():
        if not check_city(city):
            raise ValueError(f"City '{city}' not found in the city boundaries shapefile.")
        if not Path(tracts_path).exists():
            raise FileNotFoundError(f"Tracts shapefile not found for '{city}': {tracts_path}")

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "Out" / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    city_ts = {}
    for city, tracts_path in cities.items():
        print(f"Loading and localising tweets for {city}...")
        city_ts[city] = load_city_timestamps(city, tracts_path)

    as_percent = False
    ylabel = "Tweet Count"

    breakdowns = [
        ("hour",      range(24),    None,        "Hour of Day",  "Tweet Frequency by Hour of Day"),
        ("dayofweek", range(7),     DAY_LABELS,  "Day of Week",  "Tweet Frequency by Day of Week"),
        ("month",     range(1, 13), MONTH_LABELS,"Month",        "Tweet Frequency by Month"),
    ]

    n_cities = len(cities)
    group_width = 0.8
    bar_w = group_width / n_cities

    fig, axes = plt.subplots(len(breakdowns), 1, figsize=(14, 15))

    for ax, (attr, idx, labels, xlabel, title) in zip(axes, breakdowns):
        idx_list = list(idx)
        x = np.arange(len(idx_list))

        for i, (city, ts) in enumerate(city_ts.items()):
            values = bucket_counts(ts, attr, idx_list, as_percent)
            offset = (i - (n_cities - 1) / 2) * bar_w
            ax.bar(x + offset, values.values, bar_w, label=city, color=CITY_COLORS[city])

        ax.set_xticks(x)
        ax.set_xticklabels(labels if labels else idx_list)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Single shared legend, built from the last axes' handles, placed outside to the right
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="City", loc="center left",
               bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("Tweet Temporal Patterns — City Comparison", fontsize=14, y=1.02)
    fig.subplots_adjust(right=0.85)
    plt.tight_layout()

    out_path = out_dir / "Compare_Tweet_Temporal_All.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()