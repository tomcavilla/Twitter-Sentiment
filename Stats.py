import json, gzip, os, ijson
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

class CityLookup:
    def __init__(self, shp_path, name_col="NAME"):
        # Load city polygons.
        self.cities = gpd.read_file(shp_path)[[name_col, "geometry"]]
        self.name_col = name_col

        # Ensure lat/lon coordinate system.
        if self.cities.crs != "EPSG:4326":
            self.cities = self.cities.to_crs("EPSG:4326")

        # Spatial index for faster lookups.
        self.sindex = self.cities.sindex

    def get_city(self, lon, lat):
        point = Point(lon, lat)

        # Only check nearby polygons.
        possible = self.cities.iloc[
            list(self.sindex.intersection(point.bounds))
        ]

        match = possible[possible.intersects(point)]

        if len(match):
            return match.iloc[0][self.name_col]

        return None
    

# Worker for parallelism.
def process_file(args):
    path, month = args

    lookup = CityLookup("Utilities/Shapefiles/CityBoundaries.shp")

    count = 0
    cities = Counter()
    users = set()

    with gzip.open(path, "rb") as f:
        try:
            for tweet in ijson.items(f, "item"):
                count += 1
                coords = tweet.get("centroid")
                if coords:
                    city = lookup.get_city(coords[1], coords[0])
                    if city:
                        cities[city] += 1
                user = tweet.get("user")
                if user:
                    users.add(user)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return month, 0, Counter(), set()

    return month, count, cities, users


# Main
if __name__ == "__main__":

    stats = {
        "count": 0,
        "cities": {},
        "different_users": 0,
        "monthly": {}
    }

    # Collect files.
    all_files = []

    # Store file path with its month.
    for month in range(1, 13):
        directory = f"../Extracted/{month}"

        if Path(directory).exists():
            for fname in os.listdir(directory):
                if fname.endswith(".json.gz") and not fname.startswith("._"):
                    all_files.append((os.path.join(directory, fname), month))

    print(f"Found {len(all_files)} files")

    # Global totals.
    total_cities = Counter()
    all_users = set()

    # Monthly totals.
    monthly = defaultdict(lambda: {
        "count": 0,
        "cities": Counter(),
        "users": set()
    })

    # Parallel processing.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        futures = [
            executor.submit(process_file, args)
            for args in all_files
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing files",
            unit="file"
        ):
            month, count, cities, users = future.result()

            stats["count"] += count
            total_cities.update(cities)
            all_users |= users

            monthly[month]["count"] += count
            monthly[month]["cities"].update(cities)
            monthly[month]["users"] |= users

    # Final stats.
    stats["cities"] = dict(total_cities)
    stats["different_users"] = len(all_users)

    # Convert monthly sets/counters for JSON.
    stats["monthly"] = {
        str(m): {
            "count": d["count"],
            "cities": dict(d["cities"]),
            "different_users": len(d["users"])
        }
        for m, d in sorted(monthly.items())
    }

    # Save to stats file.
    with open("../Stats.json", "wt", encoding="utf-8") as out:
        json.dump(stats, out, ensure_ascii=False, indent=2)

    print("Stats file saved.")
