# Twitter Sentiment Pipeline

> All data is stored in the parent directory (`../`) alongside this `Code/` folder: `2017_Original/`, `Extracted/`, `CityTweets/`, `Stats.json`, etc.

Five scripts run in sequence to go from raw tweet archives to the final figures and HTML report.

```
Extract.py → Stats.py → ByCity.py → Sentiment.py → Generate.py
```

The analysis used in the report used this pipeline to investigate: New York, Los Angeles, Chicago, Houston, and San Antonio. The 2017 shapefiles needed for these are included within the `Utilities/` folder, for any other states download them here: `https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html`

The census data and documentation can be found here: `https://www.icpsr.umich.edu/sites/nanda/view/studies/38528/datasets/2`

The shapefile used to identify U.S cities in the filtering stage can be found here: `https://catalog.data.gov/dataset/500-cities-city-boundaries`

---

## Extract.py

Reads raw `.json.gz` tweet archives from `../2017_Original/`, filters to US-only tweets, computes a centroid from each tweet's bounding box, and writes slim records to `../Extracted/{month}/`.

```
python Extract.py
```

No arguments — edit the `directory` variable at the top of the script to change the source path.

**Output**

| Path | Description |
|---|---|
| `../Extracted/{month}/*.json.gz` | Filtered tweets grouped by month, one file per source archive |
| `../Extraction_Stats.txt` | File count, total tweets processed, tweets kept/discarded, and percentage kept |

Each output record contains: `id`, `lang`, `text`, `timestamp`, `user`, `country`, `bounding_box`, `centroid` (`[lat, lon]`).

Uses all available CPU cores via `multiprocessing.Pool`.

---

## Stats.py

Reads all `../Extracted/{month}/` files, performs a spatial join against `CityBoundaries.shp` to assign each tweet to a city, and aggregates counts. Saves the result to `../Stats.json` (consumed by `Generate.py` for the general stats figures).

```
python Stats.py
```

No arguments. Must be run from the `Code/` directory.

**Output**

| Path | Description |
|---|---|
| `../Stats.json` | Total tweet count, per-city counts, unique user count, and all three metrics broken down by month |

Uses all available CPU cores via `ProcessPoolExecutor`.

---

## ByCity.py

Filters the extracted tweet corpus to a single city and writes one `.jsonl` file per day to `../CityTweets/{city}/`. This per-city folder is the input for both `Sentiment.py` and `Generate.py`.

```
python ByCity.py --city "Chosen City"
```

| Argument | Required | Description |
|---|---|---|
| `--city` | Yes | City name — must match an entry in `CityBoundaries.shp` |

**Output**

| Path | Description |
|---|---|
| `../CityTweets/{city}/{YYYY-MM-DD}.jsonl` | One tweet per line for tweets whose centroid falls within the city polygon |

Applies a fast bounding-box pre-filter followed by a precise polygon containment check. Uses all available CPU cores via `ProcessPoolExecutor`.

---

## Sentiment.py

Scores every tweet in `../CityTweets/` with the [BERTweet](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) sentiment model and writes the score back into each `.jsonl` file in-place.

```
python Sentiment.py
```

No arguments. Processes all cities found under `../CityTweets/`.

**Model details**

| Setting | Value |
|---|---|
| Model | `finiteautomata/bertweet-base-sentiment-analysis` |
| Batch size | 256 |
| Max token length | 64 |
| Device | CUDA if available, otherwise CPU; uses FP16 on GPU |

The `sentiment_score` field is added to each record using the formula `(P(positive) − P(negative) + 1) / 2`, producing a value in `[0, 1]` where 1 is maximally positive. Files are written to a `.tmp` sibling and then atomically replaced.

---

## Generate.py — Output Reference

Generates census demographic maps, tweet statistics, and sentiment analysis figures for a given US city.

```
python Generate.py --city "Chosen City" --tracts "path/to/state_tracts.shp" [--out "path/to/output"]
```

| Argument | Required | Description |
|---|---|---|
| `--city` | Yes | City name — must match an entry in `CityBoundaries.shp` |
| `--tracts` | Yes | Path to the state-level census tract shapefile |
| `--out` | No | Output directory (defaults to `Out/`) |

## Output Structure

```
Out/
  General_Stats.txt
  Figures/
    Tweet_Count_Top10.png
    Tweet_Count_Top5.png
    Tweet_Count_Months.png
    {city}/
      (all city-specific figures)
  Pages/
    {city}.html
```

## HTML Report

After all figures are generated, an HTML report is written to `Out/Pages/{city}.html`. It includes every figure that was produced for that run, organised into labelled sections with explanatory text describing how to read each chart. Sections that require tweet data (sentiment, temporal) are omitted automatically if that data was not available.

The report uses relative paths to the `Figures/` directory, so the entire `Out/` folder can be moved or shared as a unit without breaking image references.

---

## Figures

### General Stats
*Saved directly in `Figures/`. Generated once from `Stats.json` regardless of which city is run. Skipped if all three files already exist.*

| File | Type | Description |
|---|---|---|
| `Tweet_Count_Top10.png` | Horizontal bar | Top 10 US cities by total tweet count in 2017 |
| `Tweet_Count_Top5.png` | Horizontal bar | Top 5 US cities by total tweet count in 2017 |
| `Tweet_Count_Months.png` | Bar | US-wide tweet volume by month across 2017 |

---

### Census Demographics
*Saved in `Figures/{city}/`. Sourced from the NaNDA ICPSR dataset (ACS 2013–2017 five-year estimates). All choropleth maps show state-level tracts with the city boundary overlaid.*

#### Wealth

| File | Type | Description |
|---|---|---|
| `{city}-Wealth_Distribution.png` | Choropleth | Weighted income score per tract, derived from the proportion of families in each income bracket (<$15k, $15–30k, $30–50k, $50–100k, >$100k) multiplied by bracket midpoints |

#### Age

| File | Type | Description |
|---|---|---|
| `{city}-Age_Distribution.png` | 2×3 choropleth grid | Proportion of population in each age group: 18–29, 30–39, 40–49, 50–69, 70+. Shared colour scale clipped to 5th–95th percentile |
| `{city}-Dominant_Age.png` | Choropleth | The age group with the highest proportion in each census tract |
| `{city}-Income_vs_Age.png` | Heatmap | Mean age group percentage broken down by dominant income bracket — rows = income brackets, columns = age groups |

#### Population

| File | Type | Description |
|---|---|---|
| `{city}-Population_Density.png` | Choropleth | Persons per square mile (`POPDEN13_17`), colour scale clipped to 5th–95th percentile |

#### Education

| File | Type | Description |
|---|---|---|
| `{city}-Education_Levels.png` | 1×3 choropleth grid | Proportion of population by education level: < High School, High School / Some College, Bachelor's+. Shared colour scale |
| `{city}-Dominant_Education.png` | Choropleth | The education level with the highest proportion in each census tract |

#### Labour Market

| File | Type | Description |
|---|---|---|
| `{city}-Unemployment.png` | Choropleth | Proportion of the 16+ civilian labour force that is unemployed (`PUNEMP13_17`) |

#### Housing

| File | Type | Description |
|---|---|---|
| `{city}-Owner_Occupied_Housing.png` | Choropleth | Proportion of housing units that are owner-occupied (`POWNOC13_17`) |

#### Race & Ethnicity

| File | Type | Description |
|---|---|---|
| `{city}-Race_Ethnicity.png` | 2×2 choropleth grid | Proportion of population that is Non-Hispanic White, Non-Hispanic Black, Hispanic, and Born Outside the U.S. Each panel uses an independent 5th–95th percentile colour scale |
| `{city}-Dominant_Ethnicity.png` | Choropleth | The race/ethnicity group with the highest proportion in each census tract |

#### Cross-Census Relationships

| File | Type | Description |
|---|---|---|
| `{city}-Income_vs_Education.png` | Heatmap | Mean proportion of population at each education level, broken down by dominant income bracket — rows = income brackets, columns = education levels. Annotated cell values |
| `{city}-Ethnicity_vs_Income.png` | Heatmap | Mean proportion of families in each income bracket, broken down by dominant ethnicity group per tract. Annotated cell values |
| `{city}-Ethnicity_vs_Education.png` | Heatmap | Mean proportion of population at each education level, broken down by dominant ethnicity group per tract. Annotated cell values |

#### Neighbourhood Indices

| File | Type | Description |
|---|---|---|
| `{city}-Affluence_Disadvantage.png` | Side-by-side choropleth | Left: neighbourhood affluence index (mean of % income >$75k, % Bachelor's+, % professional occupations). Right: neighbourhood disadvantage index (mean of % non-Hispanic Black, % female-headed families, % public assistance, % below poverty, % unemployed). Both scored 0–1 |

---

### Tweet Statistics
*Requires tweet data in `CityTweets/{city}/`. Timestamps are converted to the city's local timezone before any temporal aggregation.*

| File | Type | Description |
|---|---|---|
| `{city}-Tweet_Count.png` | Choropleth | Total tweet count per census tract |
| `{city}-Tweet_Hour.png` | Bar | Tweet volume by hour of day (local time) |
| `{city}-Tweet_Day.png` | Bar | Tweet volume by day of week (local time) |
| `{city}-Tweet_Month.png` | Bar | Tweet volume by month |

---

### Sentiment Analysis
*Sentiment scores are aggregated to tract level (minimum 10 tweets per tract). Timestamps are converted to local timezone.*

#### Spatial

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Map.png` | Choropleth | Mean tweet sentiment score per census tract (0–1 scale, RdYlGn) |
| `{city}-Sentiment_Variance.png` | Choropleth | Standard deviation of tweet sentiment per census tract — highlights areas with inconsistent or polarised sentiment vs stable sentiment |

#### Sentiment vs Income

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_vs_Income.png` | Scatter | Mean tract sentiment vs weighted income score. Dot colour = tweet count (log scale). Includes linear trend line |
| `{city}-Sentiment_Residual.png` | Scatter | Residual sentiment after removing the income trend, plotted against income score. Dot colour = log tweet count |
| `{city}-Sentiment_Income_Boxplot.png` | Boxplot + strip | Sentiment distribution across five income quintiles |

#### Sentiment vs Other Census Variables

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Census_Scatter.png` | 2×3 scatter grid | Mean tract sentiment vs population density, unemployment, % Bachelor's+, owner-occupied housing, affluence score, and disadvantage score. Dot size scaled by tweet count. Each panel includes a linear trend line |
| `{city}-Sentiment_Disadvantage_Boxplot.png` | Boxplot + strip | Sentiment distribution across five neighbourhood disadvantage quintiles |
| `{city}-Sentiment_Education_Boxplot.png` | Boxplot + strip | Sentiment distribution by dominant education level per tract (< High School, High School / Some College, Bachelor's+) |
| `{city}-Sentiment_Age_Boxplot.png` | Boxplot + strip | Sentiment distribution by dominant age group per tract (18–29, 30–39, 40–49, 50–69, 70+) |

#### Sentiment Trend Slopes
*Tracts with fewer than 14 daily observations are excluded. Slope is in sentiment units per day.*

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Slope_Map.png` | Choropleth | Linear trend slope of daily mean sentiment per tract across 2017. Coolwarm diverging scale: blue = improving sentiment over the year, red = declining. Centred at zero, clipped to 95th percentile |
| `{city}-Sentiment_Slope_Census.png` | 2×3 scatter grid | Sentiment trend slope vs income, population density, unemployment, education, affluence, and disadvantage. The dashed zero line distinguishes tracts where sentiment improved vs declined. Trend lines show the direction of association between census variables and trajectory |

#### Sentiment vs Race & Ethnicity

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Ethnicity_Scatter.png` | 2×2 scatter grid | Mean tract sentiment vs proportion Non-Hispanic White, Non-Hispanic Black, Hispanic, and Born Outside the U.S. Dot size scaled by tweet count. Each panel includes a linear trend line |
| `{city}-Sentiment_Ethnicity_Boxplot.png` | Boxplot + strip | Sentiment distribution by dominant race/ethnicity group per tract |

#### Sentiment vs Time (Weekday vs Weekend)

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Weekday_Weekend.png` | Line | Mean sentiment by hour of day split into two overlaid lines — weekday (Mon–Fri) vs weekend (Sat–Sun) |

#### Sentiment vs Time (Mean)

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Hour.png` | Bar + error bars | Mean sentiment by hour of day (local time) with 95% confidence intervals |
| `{city}-Sentiment_Day.png` | Bar + error bars | Mean sentiment by day of week (local time) with 95% confidence intervals |
| `{city}-Sentiment_Month.png` | Bar + error bars | Mean sentiment by month with 95% confidence intervals |
| `{city}-Sentiment_Timeseries.png` | Line | Daily mean sentiment across the year with 7-day rolling average overlay |
| `{city}-Sentiment_Heatmap.png` | Heatmap | Mean sentiment broken down by both hour of day and day of week simultaneously |

#### Sentiment vs Time (Extremes)
*Positive threshold: > 0.65. Negative threshold: < 0.35.*

| File | Type | Description |
|---|---|---|
| `{city}-Sentiment_Extremes_Hour.png` | Grouped bar | % of tweets classed as positive and % classed as negative, by hour of day |
| `{city}-Sentiment_Extremes_Day.png` | Grouped bar | % positive and % negative tweets by day of week |
| `{city}-Sentiment_Extremes_Month.png` | Grouped bar | % positive and % negative tweets by month |
| `{city}-Sentiment_Extremes_Timeseries.png` | Line | Daily % positive and % negative tweets over the year with 7-day rolling average overlays |

---

## Data Sources

| Data | Source |
|---|---|
| Tweet data | `CityTweets/{city}/*.jsonl` — one tweet per line with `centroid`, `timestamp`, and `sentiment_score` fields |
| Census demographics | `Utilities/ICPSR-CENSUS.tsv` — NaNDA dataset (openICPSR-119451), ACS 2013–2017 five-year estimates |
| City boundaries | `Utilities/Shapefiles/CityBoundaries.shp` |
| State tract shapefiles | Provided via `--tracts` argument (TIGER/Line format) |
| Overall tweet stats | `Stats.json` in the project root |
