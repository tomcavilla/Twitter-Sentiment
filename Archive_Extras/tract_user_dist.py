import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_columns(csv_path: str | Path) -> pd.DataFrame:
	"""Read a CSV file and return its GEOID and total_users columns."""
	data = pd.read_csv(csv_path, dtype={"GEOID": "string"})
	required_columns = {"GEOID", "total_users"}
	missing_columns = required_columns - set(data.columns)

	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"CSV is missing required column(s): {missing}")

	return data[["GEOID", "total_users"]]


data = extract_columns("../CSVFiltered/San Antonio-Sentiment_Census.csv")

plt.figure(figsize=(12, 6))
plt.bar(data["GEOID"], data["total_users"])
plt.xlabel("")
plt.ylabel("Total Users")
plt.xticks([])
plt.tight_layout()
plt.show()

print(max(data["total_users"]))