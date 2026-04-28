import os
import gzip
import json
from tqdm import tqdm 
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# BERTweet fine-tuned on sentiment — outputs NEG / NEU / POS
MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# fp16 for faster GPU inference
if device == "cuda":
    model = model.half()

BATCH_SIZE = 256
MAX_LENGTH = 64

def score_batch(tweets):
    inputs = tokenizer(
        tweets,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1)

        negative = probs[:, 0]
        positive = probs[:, 2]

        # scale to [0, 1] where 0.5 is neutral
        score = (positive - negative + 1) / 2
        return score.cpu().numpy()
    
base = "../CityTweets/"

for city in sorted(os.listdir(base)):
    city_path = os.path.join(base, city)
    if not os.path.isdir(city_path):
        continue

    print(f"Processing city: {city}")

    for fname in sorted(os.listdir(city_path)):
        if not fname.endswith(".jsonl") or fname.startswith('._'):
            continue

        input_file = os.path.join(city_path, fname)
        temp_file = os.path.join(city_path, fname + ".tmp")

        batch_tweets = []
        batch_lines = []

        # write to temp file then replace so we don't corrupt on crash
        with open(input_file, "r", encoding="utf-8") as f_in, \
             open(temp_file, "w", encoding="utf-8") as f_out:

            for line in tqdm(f_in, desc=f"Processing {city}/{fname}", leave=False):
                data = json.loads(line)
                tweet_text = data.get("text", "")
                batch_tweets.append(tweet_text)
                batch_lines.append(data)

                if len(batch_tweets) >= BATCH_SIZE:
                    scores = score_batch(batch_tweets)
                    for obj, s in zip(batch_lines, scores):
                        obj["sentiment_score"] = float(s)
                        f_out.write(json.dumps(obj) + "\n")
                    batch_tweets, batch_lines = [], []

            # flush remainder
            if batch_tweets:
                scores = score_batch(batch_tweets)
                for obj, s in zip(batch_lines, scores):
                    obj["sentiment_score"] = float(s)
                    f_out.write(json.dumps(obj) + "\n")


        os.replace(temp_file, input_file)
        print(f"Updated {input_file}")