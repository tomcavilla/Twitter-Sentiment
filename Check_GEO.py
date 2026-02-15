import json, gzip, os
from multiprocessing import Pool, cpu_count

directory = '../2017_Original'

def process_file(fname):
    # Process one .json.gz file and return number of tweets kept from the file (i.e US-tagged Tweets)
    if not fname.endswith('.json.gz') or fname.startswith('._'):
        return 0, [0,0], [0,0], [0,0]

    print(f"Processing {fname}")

    path = os.path.join(directory, fname)
    overall_count = 0
    geos = [0,0] # With / Without
    coordinates = [0,0]
    places = [0,0]


    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    json_l = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                tweet = json_l.get('text')
                coords = json_l.get('coordinates')
                geo = json_l.get('geo')
                place = json_l.get('place')

                if not tweet:
                    continue
                else:
                    overall_count += 1

                if not coords:
                    coordinates[1] += 1
                else:
                    coordinates[0] += 1
                
                if not geo:
                    geos[1] += 1
                else:
                    geos[0] += 1

                if not place:
                    places[1] += 1
                else:
                    places[0] += 1

    except (EOFError, OSError, gzip.BadGzipFile) as e:
        print(f"Exception {e} on file {fname}")
        return 0, [0,0], [0,0], [0,0]
    
    return overall_count, geos, coordinates, places
    
if __name__ == "__main__":
    files = os.listdir(directory)

    # Use all CPU cores
    workers = cpu_count()

    with Pool(workers) as pool:
        results = pool.map(process_file, files)

    total_count = 0
    total_geo = [0, 0]
    total_coordinates = [0, 0]
    total_places = [0, 0]

    for count, geos, coords, places in results:
        total_count += count
        total_geo[0] += geos[0]
        total_geo[1] += geos[1]
        total_coordinates[0] += coords[0]
        total_coordinates[1] += coords[1]
        total_places[0] += places[0]
        total_places[1] += places[1]

    print(f"Total Tweets: {total_count}")
    print(f"Coordinates   -> with: {total_coordinates[0]}, without: {total_coordinates[1]}")
    print(f"Geo           -> with: {total_geo[0]}, without: {total_geo[1]}")
    print(f"Places        -> with: {total_places[0]}, without: {total_places[1]}")
    
