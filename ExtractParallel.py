import json
import gzip
import os
from multiprocessing import Pool, cpu_count

directory = '../2017_Original'

with open("US_BOUNDING.json") as f:
    # State bounding boxes.
    STATES = json.load(f)


def in_us(lat, lon):
    # Check coordinates reside within the US, using the STATES bounding boxes.
    for s in STATES:
        (min_lon, min_lat), (max_lon, max_lat) = s["bounds"]
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return True
    return False


def process_file(fname):
    # Process one .json.gz file and return number of tweets kept from the file (i.e US-tagged Tweets)
    if not fname.endswith('.json.gz') or fname.startswith('._'):
        return 0

    print(f"Processing {fname}")

    path = os.path.join(directory, fname)
    tweets = []
    count = 0

    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    json_l = json.loads(line)
                except json.JSONDecodeError:
                    continue

                coords = json_l.get('coordinates')
                if not coords:
                    continue

                lon, lat = coords['coordinates']

                if in_us(lat, lon):
                    tweet = {
                        'id': json_l['id'],
                        'text': json_l['text'],
                        'date': json_l['created_at'],
                        'user': json_l['user']['id'],
                        'coordinates': coords['coordinates']
                    }
                    tweets.append(tweet)
                    count += 1
        
    except (EOFError, OSError, gzip.BadGzipFile) as e:
        print(f"Exception {e} on file {fname}")
        return 0

    # Create output directory.
    # File name yyyy-mm-dd.world.json.gz
    month = int(fname[5:7])
    out_dir = f"../Extracted/{month}"
    # Seperate all data by month so that it can be used individually if needed. [1-12]
    os.makedirs(out_dir, exist_ok=True)

    save_loc = os.path.join(out_dir, fname)

    with gzip.open(save_loc, 'wt', encoding='utf-8') as out:
        json.dump(tweets, out, ensure_ascii=False)

    return count


if __name__ == "__main__":
    files = os.listdir(directory)

    # Use all CPU cores
    workers = cpu_count()

    with Pool(workers) as pool:
        results = pool.map(process_file, files)

    # Completion message:
    print("Total kept tweets:", sum(results))
