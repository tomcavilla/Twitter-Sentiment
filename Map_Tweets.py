import folium, os, json, gzip
from folium.plugins import MarkerCluster
from pathlib import Path

m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
marker_cluster = MarkerCluster.add_to(m)


for i in range (1,13):
    directory = f'../Extracted/{i}'
    if Path(directory).exists() and Path(directory).is_dir():
        for fname in os.listdir(directory):
            if not fname.endswith('.json.gz') or fname.startswith('._'):
                continue

            path = os.path.join(directory, fname)
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    tweets = json.loads(line)
                    for tweet in tweets:
                        coords = tweet.get("coordinates")
                        if coords:
                            folium.Marker(
                                location=[coords[0], coords[1]],
                                icon=folium.Icon(color="red")
                            ).add_to(marker_cluster)

m.save("Clustered_Tweets.html")

#Lempel-ziv
