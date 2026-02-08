import pandas as pd
import torch, gzip, json, os
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


base = "../Extracted/"
output = "../Processed"
os.makedirs(output, exist_ok=True)

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

        print(f"Processing {input_file}")

        batch_tweets = []
        batch_coords = []

        with open(output_file, "w", encoding="utf-8") as out_csv:
            out_csv.write("tweet,sentiment_score,coordinates\n")

        with gzip.open(input_file, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {fname}", leave=False):
                data = json.loads(line[0])
                tweet = data.get("text", "")
                coords = data.get("coordinates")

                batch_tweets.append(tweet)
                batch_coords.append(coords)

                if len(batch_texts) >= BATCH_SIZE:
                    scores = score_batch(batch_texts)
                    with open(output_file, "a", encoding="utf-8") as out_csv:
                        for t, s, c in zip(batch_texts, scores, batch_coords):
                            tweet_clean = t.replace('"', '""')
                            if c is None:
                                coord_str = ""
                            else:
                                coord_str = f"{c[0]},{c[1]}"  # lon,lat
                            out_csv.write(f'"{tweet_clean}",{s},{coord_str}\n')
                    batch_texts, batch_coords = [], []

        if batch_texts:
            scores = score_batch(batch_texts)
            with open(output_file, "a", encoding="utf-8") as out_csv:
                for t, s, c in zip(batch_texts, scores, batch_coords):
                    tweet_clean = t.replace('"', '""')
                    if c is None:
                        coord_str = ""
                    else:
                        coord_str = f"{c[0]},{c[1]}"
                    out_csv.write(f'"{tweet_clean}",{s},{coord_str}\n')