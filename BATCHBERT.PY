import pandas as pd
import torch, gzip, json, os
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import langid

# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# -- Load Model --

MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

if device == "cuda":
    model = model.half()

# -- Batch --
BATCH_SIZE = 256
MAX_LENGTH = 64

def score_batch(tweets):
    inputs = tokenizer(
        tweets,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH, # Max no. of tokens
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1)

        negative = probs[:,0]
        positive = probs[:,2]

        score = (positive - negative + 1)/2
        return score.cpu().numpy()

# -- Main --

base = "../Extracted/"
output = "../Sentiment-Scored"
os.makedirs(output, exist_ok=True)

en_count = 0
non_en_count = 0

for month in sorted(os.listdir(base)):
    month_path = os.path.join(base, month)
    if not os.path.isdir(month_path):
        continue

    for fname in sorted(os.listdir(month_path)):
        if not fname.endswith(".world.json.gz") or fname.startswith('._'):
            continue

        input_file = os.path.join(month_path, fname)
        day_name = fname.replace(".world.json.gz", ".csv")
        output_file = os.path.join(output, day_name)

        batch_tweets = []
        batch_coords = []

        with open(output_file, "w", encoding="utf-8") as out_csv:
            out_csv.write("tweet,sentiment_score,coordinates\n")

        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            tweets = json.load(f)   # loads entire array safely

        for data in tqdm(tweets, desc=f"Processing {fname}", leave=False):
                tweet = data.get("text", "")
                coords = data.get("coordinates")

                lang, conf = langid.classify(tweet)
                if lang != "en":
                    non_en_count += 1
                    continue

                en_count += 1
                batch_tweets.append(tweet)
                batch_coords.append(coords)

                if len(batch_tweets) >= BATCH_SIZE:
                    scores = score_batch(batch_tweets)
                    with open(output_file, "a", encoding="utf-8") as out_csv:
                        for t, s, c in zip(batch_tweets, scores, batch_coords):
                            tweet_clean = t.replace('"', '""')
                            if c is None:
                                coord_str = ""
                            else:
                                coord_str = f"{c[0]},{c[1]}"  # lon,lat
                            out_csv.write(f'"{tweet_clean}",{s},{coord_str}\n')
                    batch_tweets, batch_coords = [], []

        if batch_tweets:
            scores = score_batch(batch_tweets)
            with open(output_file, "a", encoding="utf-8") as out_csv:
                for t, s, c in zip(batch_tweets, scores, batch_coords):
                    tweet_clean = t.replace('"', '""')
                    if c is None:
                        coord_str = ""
                    else:
                        coord_str = f"{c[0]},{c[1]}"
                    out_csv.write(f'"{tweet_clean}",{s},{coord_str}\n')


print(f"Total Tweets   : {en_count+non_en_count:,}")
print(f"English Tweets : {en_count:,}")
print(f"Non-English Tweets : {non_en_count:,}")
