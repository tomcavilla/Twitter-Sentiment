# Twitter Sentiment Pipeline

> All data is stored in the parent directory (`../`) alongside this `Code/` folder: `2017_Original/`, `Extracted/`, `CityTweets/`, `Pickles/`, `CSV/`, `Stats.json`, etc.

The pipeline processes raw 2017 tweet archives into census-joined datasets, user- and tweet-level aggregations, and publication figures.

```
Extract.py → Stats.py → ByCity.py → Sentiment.py ─┬─→ Generate_Filtered.py
                                                  ├─→ Generate_User_Agg.py
                                                  └─→ Export_CSV.py / Export_CSV_Filter.py → Generate_From_CSV.py
```

The analysis investigates U.S. cities using boundaries from `Utilities/Shapefiles/CityBoundaries.shp` and census tract shapefiles (`--tracts`). Additional 2017 TIGER/Line state tract shapefiles can be downloaded here: <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>

The NaNDA census dataset and documentation can be found here: <https://www.icpsr.umich.edu/sites/nanda/view/studies/38528/datasets/2>

The city boundary shapefile source can be found here: <https://catalog.data.gov/dataset/500-cities-city-boundaries>

---

## Data Extraction & Corpus Statistics

### Extract.py

Reads raw `.json.gz` tweet archives from `../2017_Original/`, filters to `"United States"` tweets, computes `[lat, lon]` centroids from each tweet's bounding box, and writes JSON array records to `../Extracted/{month}/`.

```bash
python Extract.py
```

No CLI arguments — edit the `directory` variable at the top of the script to change the source path.

| Output Path | Description |
|---|---|
| `../Extracted/{month}/*.json.gz` | Filtered U.S. tweets grouped by month, one file per source archive |
| `../Extraction_Stats.txt` | File count, total tweets processed, U.S. tweets kept, non-U.S. tweets discarded, and percentage kept |

Each output record contains: `id`, `lang`, `text`, `timestamp`, `user`, `country`, `bounding_box`, and `centroid` (`[lat, lon]`). Uses all available CPU cores via `multiprocessing.Pool`.

---

### Stats.py

Streams all `../Extracted/{month}/` archives with `ijson`, performs spatial lookups against `Utilities/Shapefiles/CityBoundaries.shp` (using an `EPSG:4326` spatial index), and aggregates tweet and user counts to `../Stats.json`.

```bash
python Stats.py
```

No CLI arguments. Must be run from the `Code/` directory (uses `Utilities/Shapefiles/CityBoundaries.shp` and `../Extracted/` / `../Stats.json`).

| Output Path | Description |
|---|---|
| `../Stats.json` | Global and monthly total tweet counts, per-city tweet counts, and unique user counts |

Uses `ProcessPoolExecutor(max_workers=os.cpu_count())`.

---

### ByCity.py

Filters the extracted tweet corpus to a single city and writes daily `.jsonl` files to `../CityTweets/{city}/`.

```bash
python ByCity.py --city "Chosen City"
```

| Argument | Required | Description |
|---|---|---|
| `--city` | Yes | City name matching an entry in `CityBoundaries.shp` |

| Output Path | Description |
|---|---|
| `../CityTweets/{city}/{YYYY-MM-DD}.jsonl` | One tweet JSON object per line whose centroid falls within the city boundary |

---

### Sentiment.py

Scores tweets in `../CityTweets/{city}/*.jsonl` with [`finiteautomata/bertweet-base-sentiment-analysis`](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) and writes `sentiment_score` back to each `.jsonl` file via a `.tmp` file and `os.replace`.

```bash
python Sentiment.py
```

No CLI arguments. Iterates over all city subdirectories in `../CityTweets/`.

| Setting | Value |
|---|---|
| Model | `finiteautomata/bertweet-base-sentiment-analysis` |
| Batch size | `256` |
| Max token length | `64` |
| Device / Precision | CUDA if available (FP16 on GPU), otherwise CPU |

`sentiment_score` is computed as `(P(positive) - P(negative) + 1) / 2` in `[0, 1]` (`0.5` neutral, `1.0` maximally positive).

---

## CSV Export Scripts (`Export_CSV.py` & `Export_CSV_Filter.py`)

Both scripts join census tract geometries, derive `INCOME_SCORE` (weighted sum of ACS 2013–2017 income brackets with midpoints `[7.5, 22.5, 40, 75, 125]`), extract 33 NaNDA ACS 2013–2017 variables (`ALAND10`, `TOTPOP13_17`, `POPDEN13_17`, demographics, education, income brackets, age brackets, unemployment, poverty, affluence/disadvantage indices, etc.), aggregate tweet sentiment both **by tweet** (`mean_sentiment_by_tweet`, `total_tweets`) and **by user** (`mean_sentiment_by_user`, `total_users`), and write `{city}-Sentiment_Census.csv`.

### Export_CSV.py

```bash
python Export_CSV.py --city "Chosen City" --tracts "path/to/state_tracts.shp" [--out "path/to/out"] [--min-tweets 1]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--city` | Yes | — | Target city in `Utilities/Shapefiles/CityBoundaries.shp` |
| `--tracts` | Yes | — | Path to state census tract shapefile |
| `--out` | No | `Code/Out/` | Output directory for `{city}-Sentiment_Census.csv` |
| `--min-tweets` | No | `1` | Minimum `total_tweets` required for a tract to be included |

Caches parsed tweet data to `../Pickles/{city}_tweets.pkl` (`lon`, `lat`, `timestamp`, `sentiment`, `user`).

### Export_CSV_Filter.py

Adds optional bounding-box area filtering to exclude coarse, city-wide geotags before spatial join and aggregation.

```bash
python Export_CSV_Filter.py --city "Chosen City" --tracts "path/to/state_tracts.shp" \
  [--out "path/to/out"] [--min-tweets 1] [--filter-citywide-bbox] [--bbox-area-threshold 50.0]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--city` | Yes | — | Target city in `Utilities/Shapefiles/CityBoundaries.shp` |
| `--tracts` | Yes | — | Path to state census tract shapefile |
| `--out` | No | `Code/Out/` | Output directory for `{city}-Sentiment_Census.csv` |
| `--min-tweets` | No | `1` | Minimum `total_tweets` required for a tract to be included |
| `--filter-citywide-bbox` | No | `False` | Exclude tweets whose `bounding_box` area (`km^2`) is `>= --bbox-area-threshold` |
| `--bbox-area-threshold` | No | `50.0` | Area threshold in `km^2` when `--filter-citywide-bbox` is enabled |

When `--filter-citywide-bbox` is set, results are cached separately at `../Pickles/{city}_tweets_filtered.pkl` so unfiltered pickles (`{city}_tweets.pkl`) are never read or overwritten.

---

## Analysis & Figure Generation Scripts

### Generate_From_CSV.py

Generates general `Stats.json` figures (if `Stats.json` exists and the three general figures are not already present), census maps/heatmaps, and **both tweet-aggregated (`_Tweets`) and user-aggregated (`_Users`)** sentiment figures from a pre-exported `{city}-Sentiment_Census.csv` file.

```bash
python Generate_From_CSV.py --city "Chosen City" --tracts "path/to/state_tracts.shp" \
  [--out "path/to/out"] [--csv-dir "path/to/csv_dir"]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--city` | Yes | — | Target city in `Utilities/Shapefiles/CityBoundaries.shp` |
| `--tracts` | Yes | — | Path to state census tract shapefile |
| `--out` | No | `Code/Out/` | Base output directory (`Figures/` and `General_Stats.txt`) |
| `--csv-dir` | No | `../CSV` | Directory containing `{city}-Sentiment_Census.csv` |

Note: `load_sentiment_census()` raises `FileNotFoundError` if `{csv_dir}/{city}-Sentiment_Census.csv` is missing.

---

### Generate_Filtered.py

Generates general `Stats.json` figures, census figures, tweet-level statistics, tweet-level sentiment figures, daily tract sentiment trend slopes (for tracts with `>= 14` daily observations), and temporal sentiment charts directly from `../CityTweets/{city}/*.jsonl` (with optional bounding-box area filtering and pickle caching).

```bash
python Generate_Filtered.py --city "Chosen City" --tracts "path/to/state_tracts.shp" \
  [--out "path/to/out"] [--filter-citywide-bbox] [--bbox-area-threshold 100.0]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--city` | Yes | — | Target city in `Utilities/Shapefiles/CityBoundaries.shp` |
| `--tracts` | Yes | — | Path to state census tract shapefile |
| `--out` | No | `Code/Out/` | Base output directory |
| `--filter-citywide-bbox` | No | `False` | Exclude coarse city-wide bounding boxes (`area >= threshold`) |
| `--bbox-area-threshold` | No | `100.0` | Area threshold in `km^2` when `--filter-citywide-bbox` is set (help text notes `50.0`) |

Caches parsed tweets to `../Pickles/{city}_tweets.pkl` or `../Pickles/{city}_tweets_filtered.pkl`.

---

### Generate_User_Agg.py

Aggregates activity and sentiment **by unique user** rather than raw tweet volume so prolific accounts do not dominate tract, temporal, or extreme-sentiment metrics.

```bash
python Generate_User_Agg.py --city "Chosen City" --tracts "path/to/state_tracts.shp" [--out "path/to/out"]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--city` | Yes | — | Target city in `Utilities/Shapefiles/CityBoundaries.shp` |
| `--tracts` | Yes | — | Path to state census tract shapefile |
| `--out` | No | `Code/Out/` | Base output directory |

Key differences from tweet-volume aggregation:
- `Stats.json` general stats remain tweet/user volume summaries from `Stats.json`.
- Tract choropleths and hour/day/month bar charts report **unique users** (`{city}-User_Count.png`, `{city}-User_Hour.png`, `{city}-User_Day.png`, `{city}-User_Month.png`).
- Sentiment maps, variance across user means, census scatters/boxplots, daily tract slopes (per-user daily means averaged per tract-day across `>= 14` tract-days), temporal means (`aggregate_by_user`), and extreme sentiment rates (`user_extreme_rates`: `% of users` with mean sentiment `> 0.65` or `< 0.35`) are computed per user.

---

## Output Structure & Figures

All figure scripts write to `{out}/Figures/` (general figures) and `{out}/Figures/{city}/` (city-specific figures), and write `{out}/General_Stats.txt` when generating general stats from `Stats.json`.

### General Stats (`Figures/` & `General_Stats.txt`)
- `Tweet_Count_Top10.png` — Top 10 cities by 2017 tweet count
- `Tweet_Count_Top5.png` — Top 5 cities by 2017 tweet count
- `Tweet_Count_Months.png` — Monthly U.S. tweet counts in 2017

### Census Demographics (`Figures/{city}/`)
- `{city}-Wealth_Distribution.png` — Weighted income score (`INCOME_SCORE`, `$k`)
- `{city}-Age_Distribution.png`, `{city}-Dominant_Age.png`, `{city}-Income_vs_Age.png`
- `{city}-Population_Density.png` (`POPDEN13_17`)
- `{city}-Education_Levels.png`, `{city}-Dominant_Education.png`, `{city}-Income_vs_Education.png`
- `{city}-Unemployment.png` (`PUNEMP13_17`)
- `{city}-Owner_Occupied_Housing.png` (`POWNOC13_17`)
- `{city}-Affluence_Disadvantage.png` (`AFFLUENCE13_17` and `DISADVANTAGE13_17`)
- `{city}-Race_Ethnicity.png`, `{city}-Dominant_Ethnicity.png`, `{city}-Ethnicity_vs_Income.png`, `{city}-Ethnicity_vs_Education.png`

### Tweet / User Activity & Sentiment (`Figures/{city}/`)
- **Activity maps & temporal bars:** `{city}-Tweet_Count.png`, `{city}-Tweet_Hour.png`, `{city}-Tweet_Day.png`, `{city}-Tweet_Month.png` (in `Generate_Filtered.py`) or `{city}-User_Count.png`, `{city}-User_Hour.png`, `{city}-User_Day.png`, `{city}-User_Month.png` (in `Generate_User_Agg.py`).
- **Sentiment maps & income comparisons:**
  - `Generate_Filtered.py` & `Generate_User_Agg.py`: `{city}-Sentiment_Map.png`, `{city}-Sentiment_Variance.png`, `{city}-Sentiment_vs_Income.png` (with Pearson `r`, `R^2`, and `p`-value), `{city}-Sentiment_Residual.png`, `{city}-Sentiment_Income_Boxplot.png`.
  - `Generate_From_CSV.py`: Produces separate `_Tweets` and `_Users` variants for choropleths (`Sentiment_Map_Tweets.png` / `Sentiment_Map_Users.png`), income scatters (`Sentiment_vs_Income_Tweets.png` / `Sentiment_vs_Income_Users.png`), residuals, income boxplots, census scatters, disadvantage/education/age/ethnicity boxplots, and ethnicity scatters.
- **Additional `Generate_Filtered.py` & `Generate_User_Agg.py` charts:** Census scatters, disadvantage/education/age/ethnicity boxplots, ethnicity scatters, trend slopes (`{city}-Sentiment_Slope_Map.png`, `{city}-Sentiment_Slope_Census.png`), temporal means (`Sentiment_Hour.png`, `Sentiment_Day.png`, `Sentiment_Month.png`, `Sentiment_Timeseries.png`, `Sentiment_Heatmap.png`, `Sentiment_Weekday_Weekend.png`), and extremes (`Sentiment_Extremes_Hour.png`, `Sentiment_Extremes_Day.png`, `Sentiment_Extremes_Month.png`, `Sentiment_Extremes_Timeseries.png`, using thresholds `> 0.65` and `< 0.35`).

---

## Dependencies

Install dependencies from `requirements.txt` (`geopandas`, `ijson`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `shapely`, `timezonefinder`, `torch`, `tqdm`, `transformers`) plus `scipy` (used for `scipy.stats.pearsonr` in `Generate_Filtered.py`, `Generate_From_CSV.py`, and `Generate_User_Agg.py`).