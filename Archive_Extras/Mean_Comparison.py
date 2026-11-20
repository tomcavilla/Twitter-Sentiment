import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot(file_path, out_file):
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(df.index, df['mean_sentiment_by_tweet'], label='Mean Sentiment (Tweet)', color='#1f77b4', linewidth=2)
    ax.plot(df.index, df['mean_sentiment_by_user'], label='Mean Sentiment (User)', color='#ff7f0e', linewidth=1.2, alpha=0.85)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('GEOIDs')
    ax.set_ylabel('Mean Sentiment')
    ax.set_title('Two Line Plot Comparison Across GEOIDs')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="CSV Filepath")
    parser.add_argument("--out", type=str, required=True, help="Figure out filepath")
    args = parser.parse_args()

    plot(args.file, args.out)