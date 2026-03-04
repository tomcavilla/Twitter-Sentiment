import json, gzip, os
from multiprocessing import Pool, cpu_count

directory = '../2017_Original'

# ----- HELPER -----

def centroid(points):
    """
    Compute centroid of bounding box points.
    points format: [[lon, lat], [lon, lat], [lon, lat], [lon, lat]]

    Mapping uses lat, lon. 
    Return lat, lon.
    """
    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)
    return lat, lon

# ----- WORKER -----

def process_file(fname):
    # Process one .json.gz file and return number of tweets kept from the file (i.e US-tagged Tweets)
    if not fname.endswith('.json.gz') or fname.startswith('._'):
        return 0, 0, 0

    print(f"Processing {fname}")

    path = os.path.join(directory, fname)
    processed = 0
    removed = 0
    kept = 0

    try:
        month = int(fname[5:7])
        out_dir = f"../Extracted/{month}"
        os.makedirs(out_dir, exist_ok=True)
        save_loc = os.path.join(out_dir, fname)

        with gzip.open(path, 'rt', encoding='utf-8') as f, \
             gzip.open(save_loc, 'wt', encoding='utf-8') as out:

            # stream-write JSON array
            out.write('[\n')
            first = True

            for line in f:
                try:
                    json_l = json.loads(line)
                except json.JSONDecodeError:
                    continue

                place = json_l.get('place')
                if not place:
                    continue

                processed += 1

                bbox = place.get('bounding_box')
                if not bbox:
                    continue

                points = bbox.get('coordinates', [[]])[0]
                if not points:
                    continue

                country = place.get('country')
                if not country:
                    continue
                elif country != "United States":
                    removed += 1
                    continue

                kept += 1

                lat, lon = centroid(points)

                tweet = {
                    'id': json_l.get('id'),
                    'lang': json_l.get('lang'),
                    'text': json_l.get('text'),
                    'timestamp': json_l.get('created_at'),
                    'user': json_l.get('user', {}).get('id'),
                    'country': country,
                    'bounding_box': points,
                    'centroid': [lat, lon]
                }

                if not first:
                    out.write(',\n')
                json.dump(tweet, out, ensure_ascii=False)
                first = False

                

            out.write('\n]')

    except (EOFError, OSError, gzip.BadGzipFile) as e:
        print(f"Exception {e} on file {fname}")
        return 0, 0, 0

    return processed, kept, removed


if __name__ == "__main__":
    files = os.listdir(directory)

    # Use all CPU cores
    workers = cpu_count()

    with Pool(workers) as pool:
        results = pool.map(process_file, files)

    total_processed = sum(r[0] for r in results)
    total_kept = sum(r[1] for r in results)
    total_removed = sum(r[2] for r in results)

    output_path = "../Extraction_Stats.txt"

    pct_kept = round(100 * total_kept / total_processed, 2) if total_processed else 0

    lines = [
        f"{len(files)} Files Processed",
        f"Total Tweets Processed:          {total_processed}",
        f"Total U.S. Tweets Kept:          {total_kept}",
        f"Total Non U.S. Tweets Discarded: {total_removed}",
        f"Percentage of Tweets Kept:       {pct_kept}%",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))