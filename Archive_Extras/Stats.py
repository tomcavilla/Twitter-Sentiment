import json
import gzip
import os
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
from shapely.geometry import Point


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

    def get_city(self, lat, lon):
        point = Point(lon, lat)

        # Only check nearby polygons.
        possible = self.cities.iloc[
            list(self.sindex.intersection(point.bounds))
        ]

        match = possible[possible.contains(point)]

        if len(match):
            return match.iloc[0][self.name_col]

        return None


# Worker for parallelism.
def process_file(args):
    path, month = args

    lookup = CityLookup("Shapefiles/CityBoundaries.shp")

    # Stats for this file.
    count = 0
    cities = Counter()
    users = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            tweets = json.loads(line)

            # Overall count for file.
            for tweet in tweets:
                count += 1

                try:
                    # Match the tweet with its city of origin.
                    coords = tweet.get("coordinates")
                    if not coords:
                        continue

                    city = lookup.get_city(coords[1], coords[0])

                    if city:
                        cities[city] += 1

                    users.add(tweet["user"])

                except Exception:
                    pass

    # Return stats along with month.
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
        results = executor.map(process_file, all_files)

        for month, count, cities, users in results:

            # Update global stats.
            stats["count"] += count
            total_cities.update(cities)
            all_users |= users

            # Update monthly stats.
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
