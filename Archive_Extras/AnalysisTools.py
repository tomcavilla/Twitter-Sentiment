import folium
import json
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from tqdm import tqdm

EARTH_METRES_PER_DEG = 111320


def bbox_area_km2(points):
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]

    dlon = max(lons) - min(lons)
    dlat = max(lats) - min(lats)

    lat_centre = sum(lats) / len(lats)

    width_m = dlon * EARTH_METRES_PER_DEG * np.cos(np.radians(lat_centre))
    height_m = dlat * EARTH_METRES_PER_DEG

    area_m2 = width_m * height_m
    return area_m2 / 1000000


def median_tract_size(tracts):
    if tracts.crs.is_geographic:
        tracts = tracts.to_crs(epsg=3857)

    tracts['area_km2'] = tracts.geometry.area / 1000000
    return tracts['area_km2'].median()

def standard_dev(devs, tracts):
    tracts['area_km2'] = tracts.geometry.area / 1000000
    return tracts['area_km2'].mean() + (devs * tracts['area_km2'].std())


def map_city_tweets(city, shapefile, outfile, map_start):

    tracts = gpd.read_file(shapefile)

    city_bounds = gpd.read_file("Utilities/Shapefiles/CityBoundaries.shp")
    chosen_bounds = city_bounds[city_bounds["NAME"] == city]
    
    tracts = tracts.to_crs(epsg=3857)

    tracts = gpd.sjoin(tracts, chosen_bounds, how="inner", predicate="intersects")
    tracts = tracts.reset_index(drop=True)
    tracts = tracts.drop(columns=[col for col in tracts.columns if col.startswith("index_")])

    tracts = tracts.to_crs(epsg=4326)

    max_size_bbox = standard_dev(1, tracts)

    unused_tweets = 0
    points = []

    folder_path = Path(f'../CityTweets/{city}/')

    print("Reading tweets...")

    for file_path in folder_path.glob('*.jsonl'):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    tweet = json.loads(line)

                    bbox = tweet.get('bounding_box')
                    #if bbox_area_km2(bbox) > max_size_bbox:
                        #unused_tweets += 1
                        #continue

                    centroid = tweet.get('centroid')
                    lat, lon = centroid

                    points.append((lon, lat))

                except json.JSONDecodeError:
                    continue

    print(f"Valid tweet points: {len(points)}")

    # Convert to GeoDataFrame
    df = pd.DataFrame(points, columns=["lon", "lat"])

    tweets_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    print("Running spatial join...")

    joined = gpd.sjoin(tweets_gdf, tracts, predicate="within")

    print("Counting tweets per tract...")

    counts = joined.groupby("index_right").size()

    tracts["tweet_count"] = 0
    tracts.loc[counts.index, "tweet_count"] = counts.values

    # Remove empty tracts, except those that touch any non-zero tract.
    non_zero = tracts[tracts["tweet_count"] > 0]
    touching = tracts[tracts.geometry.apply(
        lambda geom: non_zero.geometry.touches(geom).any()
    )]
    tracts = pd.concat([non_zero, touching]).drop_duplicates()

    print("Creating map...")

    m = folium.Map(location=map_start, zoom_start=10, tiles='CartoDB positron')

    max_count = tracts["tweet_count"].max()
    log_max = np.log1p(max_count)

    def style_function(feature):
        count = feature["properties"]["tweet_count"]

        if count == 0:
            return {
                "fillColor": "#d3d3d3",  # Light grey for zero-tweet tracts
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7
            }

        log_val = np.log1p(count)
        intensity = int(255 * (log_val / log_max))

        return {
            "fillColor": f"#00{intensity:02x}00",
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        }

    folium.GeoJson(
        tracts,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["tweet_count"],
            aliases=["Tweets:"]
        )
    ).add_to(m)

    m.save(outfile)

    print(f"Map saved to {outfile}")
    print(f"Unused tweets skipped: {unused_tweets}")

    if tracts.crs.is_geographic:
        tracts = tracts.to_crs(epsg=3857)

    tracts["area_km2"] = tracts.geometry.area / 1_000_000
    tracts["tweet_density"] = tracts["tweet_count"] / tracts["area_km2"]

    return tracts[["GEOID", "tweet_count", "tweet_density"]]



def load_tweets(file):
    rows = []
    with open(file) as f:
        for line in f:
            t = json.loads(line)

            if "centroid" not in t or t["centroid"] is None:
                continue

            lat, lon = t["centroid"]

            rows.append({
                "time": pd.to_datetime(t["timestamp"]),
                "lat": lat,
                "lon": lon,
                "sentiment": t["sentiment_score"]
            })

    return pd.DataFrame(rows)

def tweets_to_gdf(df):
    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    return gdf

def join_to_tracts(tweets_gdf, tracts):
    return gpd.sjoin(
        tweets_gdf,
        tracts,
        how="left",
        predicate="within"
    )

def aggregate_daily_sentiment(joined):
    joined["date"] = joined["time"].dt.date
    agg = (
        joined
        .groupby(["GEOID", "date"])
        .agg(
            mean_sentiment=("sentiment", "mean"),
            tweet_count=("sentiment", "size")
        )
        .reset_index()
    )
    return agg

def process_tweet_folder(folder, tracts):
    results = []
    files = sorted(Path(folder).glob("*.jsonl"))

    for file in tqdm(files):
        df = load_tweets(file)
        if df.empty:
            continue
        gdf = tweets_to_gdf(df)
        joined = join_to_tracts(gdf, tracts)
        agg = aggregate_daily_sentiment(joined)
        results.append(agg)

    return pd.concat(results)

def compute_sentiment_change(df):
    df = df.sort_values(["GEOID", "date"])
    df["sentiment_change"] = (
        df.groupby("GEOID")["mean_sentiment"].diff()
    )
    return df
    