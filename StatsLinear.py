import json, gzip, os
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path

class CityLookup:
    def __init__(self, shp_path, name_col="NAME"):
        self.cities = gpd.read_file(shp_path)[[name_col, "geometry"]]
        self.name_col = name_col

        if self.cities.crs != "EPSG:4326":
            self.cities = self.cities.to_crs ("EPSG:4326")

        self.sindex = self.cities.sindex

    def get_city(self, lat, lon):
        point = Point(lon,lat)
        possible = self.cities.iloc[list(self.sindex.intersection(point.bounds))]
        match = possible[possible.contains(point)]
        if len(match):
            return match.iloc[0][self.name_col]
        return None

if __name__ == "__main__":
    stats = {
        'count': 0,
        'cities': {},
        'different_users': 0,
    }

    users = []

    lookup = CityLookup("Shapefiles/CityBoundaries.shp")

    for i in range (1,13):
        directory = '../Extracted/' + str(i)
        if Path(directory).exists() and Path(directory).is_dir():
            for fname in os.listdir(directory):
                if not fname.endswith('.json.gz') or fname.startswith('._'):
                    continue

                path = os.path.join(directory, fname)
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        tweets = json.loads(line)
                        for tweet in tweets:
                            stats['count'] += 1
                            try:
                                city = lookup.get_city(tweet['coordinates'][1], tweet['coordinates'][0])
                                print(city)
                                if city not in stats['cities'].keys():
                                    stats['cities'][city] = 1
                                else:
                                    stats['cities'][city] += 1

                                if tweet['user'] not in users:
                                    stats['different_users'] += 1
                                    users.append(tweet['user'])
                            except:
                                print(tweet)


    with open('../Stats.json', 'wt', encoding='utf-8') as out:
            json.dump(stats, out, ensure_ascii=False)