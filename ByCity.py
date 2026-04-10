import argparse, json, gzip, os, ijson
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

city_geom = None
city_bbox = None


def init_worker(chosen_city):
    """Load city geometry once per worker."""
    global city_geom, city_bbox

    cities = gpd.read_file("Utilities/Shapefiles/CityBoundaries.shp")[["NAME","geometry"]]

    if cities.crs != "EPSG:4326":
        cities = cities.to_crs("EPSG:4326")

    row = cities[cities["NAME"] == chosen_city].iloc[0]

    city_geom = row.geometry
    city_bbox = city_geom.bounds  # (minx, miny, maxx, maxy)


def process_file(path, chosen_city):

    day = Path(path).name[:10]

    out_dir = Path(f"../CityTweets/{chosen_city}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{day}.jsonl"

    minx, miny, maxx, maxy = city_bbox

    with open(out_file, "a", encoding="utf-8") as out:

        with gzip.open(path, "rb") as f:
            try:
                for tweet in ijson.items(f, "item", use_float=True):

                    coords = tweet.get("centroid")
                    if not coords:
                        continue

                    lat, lon = coords

                    # FAST bounding box filter
                    if not (minx <= lon <= maxx and miny <= lat <= maxy):
                        continue

                    # precise polygon check
                    if not city_geom.contains(Point(lon, lat)):
                        continue

                    out.write(json.dumps(tweet) + "\n")

            except Exception as e:
                print(f"Error parsing {path}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="Name of the city to extract tweets for.")
    args = parser.parse_args()

    chosen_city = args.city

    all_files = []

    for month in range(1,13):
        directory = f"../Extracted/{month}"

        if Path(directory).exists():
            for fname in os.listdir(directory):
                if fname.endswith(".json.gz") and not fname.startswith("._"):
                    all_files.append(os.path.join(directory, fname))

    print(f"Found {len(all_files)} files.")

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker,
        initargs=(chosen_city,)
    ) as executor:

        futures = [executor.submit(process_file, f, chosen_city) for f in all_files]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass

    print(f"{chosen_city} Extraction Completed.")