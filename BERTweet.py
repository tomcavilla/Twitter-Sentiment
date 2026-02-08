from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis', use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis").to(device)
model.eval()
tweet = "Tryin' something new, I almost forgot I have this in my drawer. 👊👍👏… https://t.co/o4HsGa8CZD"

inputs = tokenizer(tweet, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = softmax(logits, dim=1)

negative, neutral, positive = probs[0]
# Combine to a single score. 
score = round(float(((positive - negative)+1)/2),3)

print(float(score))