import json, gzip, os

directory = '../2017_Original'
with open("US_BOUNDING.json") as f:
    STATES = json.load(f)

def in_us(lat, lon):
    for s in STATES:
        (min_lon, min_lat), (max_lon, max_lat) = s["bounds"]
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return True
    return False


count = 0
for fname in os.listdir(directory):
    if not fname.endswith('.json.gz') or fname.startswith('._'):
        continue

    tweets = []
    print(fname)

    path = os.path.join(directory, fname)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                json_l = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed JSON

            
            try:
                if json_l['coordinates'] is not None:
                    tweet = {
                        'id': json_l['id'],
                        'text': json_l['text'],
                        'date': json_l['created_at'],
                        'user': json_l['user']['id'],
                        'coordinates': json_l['coordinates']['coordinates']
                    }
                
                    
                    if in_us(tweet['coordinates'][1], tweet['coordinates'][0]):
                        count += 1
                        print(count)
                        tweets.append(tweet)

            except KeyError:
                continue

    save_loc = f"../Extracted/{str(int(fname[5:7]))}/fname"

    with gzip.open(save_loc, 'wt', encoding='utf-8') as out:
        json.dump(tweets, out, ensure_ascii=False)
