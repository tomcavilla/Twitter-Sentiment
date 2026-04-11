
import argparse
import geopandas as gpd
import json
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from timezonefinder import TimezoneFinder


CITIES_SHAPEFILE = "Utilities/Shapefiles/CityBoundaries.shp"

GENERATED_FIGURES = []

def save_figure(path, **kwargs):
    plt.savefig(path, **kwargs)
    GENERATED_FIGURES.append(path)

def to_local(series, tz):
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC").dt.tz_convert(tz)
    return series.dt.tz_convert(tz)

def get_city_names():
    return gpd.read_file(CITIES_SHAPEFILE)["NAME"].tolist()

def check_city(city: str) -> bool:
    return city in get_city_names()

def city_tracts(state_tracts, city_name):
    tracts = gpd.read_file(state_tracts)

    city_bounds = gpd.read_file(CITIES_SHAPEFILE)
    chosen_bounds = city_bounds[city_bounds["NAME"] == city_name]
    
    tracts = tracts.to_crs(epsg=3857)

    tracts = gpd.sjoin(tracts, chosen_bounds, how="inner", predicate="intersects")
    tracts = tracts.reset_index(drop=True)
    tracts = tracts.drop(columns=[col for col in tracts.columns if col.startswith("index_")])

    tracts = tracts.to_crs(epsg=4326)

    return tracts

if __name__ == "__main__":
    # ---- Add Arguments ----

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, help="Chosen city to analyse.")
    parser.add_argument("--tracts", type=str, required=True, help="Shapefile of chosen city's state.")
    parser.add_argument("--out", type=str, required=False, help="The output directory for generated pages, figures etc.")
    args = parser.parse_args()

    # ---- Check Arguments ----

    # Check city exists in the city shapefile used.
    if check_city(args.city) == False:
        raise ValueError(f"City '{args.city}' not found in analysed cities.")
    
    # Check the tracts shapefile path given exists.
    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")
    
    # Create / Check Output Directory
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).parent / "Out"

    out_dir.mkdir(parents=True, exist_ok=True)
    general_figures_dir = out_dir / "Figures"
    general_figures_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "Figures" / args.city
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Create gdf of tracts overlapping city boundaries.

    city_gdf = city_tracts(args.tracts, args.city)

    # Detect local timezone from city bounding box centre
    minx, miny, maxx, maxy = city_gdf.total_bounds
    local_tz = TimezoneFinder().timezone_at(lng=(minx + maxx) / 2, lat=(miny + maxy) / 2) or "UTC"

    # ---- General Tweet Stats (Stats.json) ----

    stats_path = Path(__file__).parent.parent / "Stats.json"
    tweet_figures = [
        general_figures_dir / "Tweet_Count_Top10.png",
        general_figures_dir / "Tweet_Count_Top5.png",
        general_figures_dir / "Tweet_Count_Months.png",
    ]

    if stats_path.exists() and not all(f.exists() for f in tweet_figures):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        top10 = sorted(stats['cities'].items(), key=lambda x: x[1], reverse=True)[:10]
        cities = [c for c, _ in top10]
        counts = [n for _, n in top10]

        millions_fmt = ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(cities, counts, color='#06402B')
        ax.invert_yaxis()
        ax.set_xlabel("Count (Millions)")
        ax.xaxis.set_major_formatter(millions_fmt)
        ax.set_title("Top 10 Cities by Tweet Count in 2017")
        plt.tight_layout()
        save_figure(general_figures_dir / "Tweet_Count_Top10.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(cities[:5], counts[:5], color='#06402B')
        ax.invert_yaxis()
        ax.set_xlabel("Count (Millions)")
        ax.xaxis.set_major_formatter(millions_fmt)
        ax.set_title("Top 5 Cities by Tweet Count in 2017")
        plt.tight_layout()
        save_figure(general_figures_dir / "Tweet_Count_Top5.png", dpi=300, bbox_inches="tight")
        plt.close()

        month_key = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
        tweet_count = []

        with open(out_dir / "General_Stats.txt", "w") as f:
            f.write(f"Total Count: {stats['count']:,}\n\nDifferent Users: {stats['different_users']:,}\n\n")
            f.write("Overall Top 10 Cities:\n\n")
            for city, count in top10:
                f.write(f"{city}: {count:,}\n")

            for month, data in stats["monthly"].items():
                month_name = month_key[int(month) - 1]
                tweet_count.append(data['count'])
                f.write(f"\n--- {month_name} ---\n\n")
                f.write(f"Total Tweets: {data['count']:,}\n\n")
                f.write(f"Total Different Users: {data['different_users']:,}\n")
                f.write(f"Top 10 Cities in {month_name}\n\n")
                for city, count in sorted(data['cities'].items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"{city}: {count:,}\n")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(month_key, tweet_count, color='#06402B')
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.set_xticklabels(month_key, rotation=90)
        ax.set_title("U.S. Tweet Fluctuations by Month in 2017")
        plt.tight_layout()
        save_figure(general_figures_dir / "Tweet_Count_Months.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---- General Census Data ----

    census = pd.read_csv("Utilities/ICPSR-CENSUS.tsv", delimiter="\t", low_memory=False)

    # Match tract code datatypes.
    census['TRACT_FIPS10'] = census['TRACT_FIPS10'].astype(str)
    city_gdf['GEOID'] = city_gdf['GEOID'].astype(str)
    # Pad in case of lost leading zeros.
    census['TRACT_FIPS10'] = census['TRACT_FIPS10'].str.zfill(11)
    city_gdf['GEOID'] = city_gdf['GEOID'].str.zfill(11)

    census = census[census['TRACT_FIPS10'].isin(city_gdf['GEOID'].unique())].copy()

    # - Wealth Distribution -

    income_cols =  ['PIN1B_13_17', 'PIN2B_13_17', 'PIN3B_13_17', 'PIN4B_13_17', 'PIN5B_13_17']
    income_labels = ['<$15k', '$15k–30k', '$30k–50k', '$50k–100k', '>$100k']

    census_income = census[['TRACT_FIPS10'] + income_cols].copy()
    census_income[income_cols] = census_income[income_cols].apply(pd.to_numeric, errors='coerce')
    census_income = census_income.dropna()

    midpoints = [7.5, 22.5, 40, 75, 125]

    census_income['INCOME_SCORE'] = sum(
        census_income[col] * mid for col, mid in zip(income_cols, midpoints)
    )

    all_tracts = gpd.read_file(args.tracts)
    all_tracts['GEOID'] = all_tracts['GEOID'].astype(str).str.zfill(11)

    gdf = all_tracts.merge(census_income[['TRACT_FIPS10', 'INCOME_SCORE']],
                           left_on='GEOID', right_on='TRACT_FIPS10')

    gdf.plot(
        column='INCOME_SCORE',
        cmap='RdYlGn',
        legend=True,
        legend_kwds={'label': 'Weighted Income Score ($k)'},
        figsize=(10, 12)
    )
    plt.title(f'Wealth Distribution across {args.city} Census Tracts')
    plt.axis('off')
    save_figure(figures_dir / f"{args.city}-Wealth_Distribution.png", dpi=300)
    plt.close()

    # - Age Distribution -

    age_cols = ['P18_2913_17', 'P30_3913_17', 'P40_4913_17', 'P50_6913_17', 'PGE7013_17']
    age_labels = ['18–29', '30–39', '40–49', '50–69', '70+']

    census_age = census[['TRACT_FIPS10'] + age_cols].copy()
    census_age[age_cols] = census_age[age_cols].apply(pd.to_numeric, errors='coerce')
    census_age = census_age.dropna(subset=age_cols)

    age_gdf = all_tracts.merge(census_age, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')
    
    city_bounds = gpd.read_file(CITIES_SHAPEFILE)
    city_bounds = city_bounds[city_bounds["NAME"] == args.city].to_crs(age_gdf.crs)

    blues = LinearSegmentedColormap.from_list('blues_clipped', plt.colormaps['Blues'](np.linspace(0.35, 1.0, 256)))

    all_values = age_gdf[age_cols].stack().dropna()
    vmin = all_values.quantile(0.05)
    vmax = all_values.quantile(0.95)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for ax, col, label in zip(axes.flatten(), age_cols, age_labels):

        age_gdf.plot(column=col, cmap=blues, legend=True, ax=ax,
                     vmin=vmin, vmax=vmax,
                     legend_kwds={'shrink': 0.6, 'label': '% of population'})
        city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        ax.set_title(f'Age {label}')
        ax.axis('off')

    axes.flatten()[-1].set_visible(False)

    plt.suptitle(f'Age Distribution across {args.city} Census Tracts', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(figures_dir / f"{args.city}-Age_Distribution.png", dpi=300)
    plt.close()

    # - Dominant Age Group -

    age_gdf['dominant_age'] = age_gdf[age_cols].idxmax(axis=1).map(dict(zip(age_cols, age_labels)))

    fig, ax = plt.subplots(figsize=(10, 12))
    age_gdf.plot(column='dominant_age', cmap=blues, legend=True, ax=ax,
                 legend_kwds={'title': 'Dominant Age Group', 'loc': 'upper left',
                              'bbox_to_anchor': (1, 1), 'frameon': True})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'Dominant Age Group across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    save_figure(figures_dir / f"{args.city}-Dominant_Age.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Income vs Age Heatmap -

    income_age = census_income[['TRACT_FIPS10'] + income_cols].merge(
        census_age[['TRACT_FIPS10'] + age_cols], on='TRACT_FIPS10'
    )

    # Bin each tract into an income bracket by its dominant income group
    income_age['income_bracket'] = income_age[income_cols].idxmax(axis=1).map(
        dict(zip(income_cols, income_labels))
    )

    heatmap_data = (
        income_age.groupby('income_bracket')[age_cols]
        .mean()
        .rename(columns=dict(zip(age_cols, age_labels)))
        .reindex(income_labels)  # ensure correct income order
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heatmap_data.values, cmap=blues, aspect='auto')

    ax.set_xticks(range(len(age_labels)))
    ax.set_xticklabels(age_labels)
    ax.set_yticks(range(len(income_labels)))
    ax.set_yticklabels(income_labels)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Dominant Income Bracket')

    for i in range(len(income_labels)):
        for j in range(len(age_labels)):
            ax.text(j, i, f"{heatmap_data.values[i, j]:.1f}",
                    ha='center', va='center', fontsize=9, color='white')

    plt.colorbar(im, ax=ax, label='Mean % of tract population')
    plt.title(f'Income vs Age Group across {args.city} Census Tracts')
    plt.tight_layout()
    save_figure(figures_dir / f"{args.city}-Income_vs_Age.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Population Density -

    density_col = 'POPDEN13_17'

    census_density = census[['TRACT_FIPS10', density_col]].copy()
    census_density[density_col] = pd.to_numeric(census_density[density_col], errors='coerce')
    census_density = census_density.dropna(subset=[density_col])

    density_gdf = all_tracts.merge(census_density, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')

    vmin = density_gdf[density_col].quantile(0.05)
    vmax = density_gdf[density_col].quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 12))
    density_gdf.plot(column=density_col, cmap='plasma', legend=True, ax=ax,
                     vmin=vmin, vmax=vmax,
                     legend_kwds={'label': 'Persons per square mile', 'shrink': 0.6})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1.5)
    ax.set_title(f'Population Density across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    save_figure(figures_dir / f"{args.city}-Population_Density.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Education Levels -

    edu_cols = ['PED1_13_17', 'PED2_13_17', 'PED3_13_17']
    edu_labels = ['< High School', 'High School / Some College', "Bachelor's+"]

    census_edu = census[['TRACT_FIPS10'] + edu_cols].copy()
    census_edu[edu_cols] = census_edu[edu_cols].apply(pd.to_numeric, errors='coerce')
    census_edu = census_edu.dropna(subset=edu_cols)

    edu_gdf = all_tracts.merge(census_edu, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')

    all_edu_values = edu_gdf[edu_cols].stack().dropna()
    edu_vmin = all_edu_values.quantile(0.05)
    edu_vmax = all_edu_values.quantile(0.95)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax, col, label in zip(axes, edu_cols, edu_labels):
        edu_gdf.plot(column=col, cmap='YlOrRd', legend=True, ax=ax,
                     vmin=edu_vmin, vmax=edu_vmax,
                     legend_kwds={'shrink': 0.6, 'label': '% of population'})
        city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(f'Education Levels across {args.city} Census Tracts', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(figures_dir / f"{args.city}-Education_Levels.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Dominant Education Level -

    edu_gdf['dominant_edu'] = pd.Categorical(
        edu_gdf[edu_cols].idxmax(axis=1).map(dict(zip(edu_cols, edu_labels))),
        categories=edu_labels,
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    edu_gdf.plot(column='dominant_edu', cmap='YlOrRd', legend=True, ax=ax,
                 legend_kwds={'title': 'Dominant Education Level', 'loc': 'upper left',
                              'bbox_to_anchor': (1, 1), 'frameon': True})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'Dominant Education Level across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    save_figure(figures_dir / f"{args.city}-Dominant_Education.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Unemployment -

    unemp_col = 'PUNEMP13_17'

    census_unemp = census[['TRACT_FIPS10', unemp_col]].copy()
    census_unemp[unemp_col] = pd.to_numeric(census_unemp[unemp_col], errors='coerce')
    census_unemp = census_unemp.dropna(subset=[unemp_col])

    unemp_gdf = all_tracts.merge(census_unemp, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')

    vmin = unemp_gdf[unemp_col].quantile(0.05)
    vmax = unemp_gdf[unemp_col].quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 12))
    unemp_gdf.plot(column=unemp_col, cmap='Reds', legend=True, ax=ax,
                   vmin=vmin, vmax=vmax,
                   legend_kwds={'label': 'Proportion unemployed (16+ civilian labour force)', 'shrink': 0.6})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'Unemployment across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    save_figure(figures_dir / f"{args.city}-Unemployment.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Owner-Occupied Housing -

    owned_col = 'POWNOC13_17'

    census_owned = census[['TRACT_FIPS10', owned_col]].copy()
    census_owned[owned_col] = pd.to_numeric(census_owned[owned_col], errors='coerce')
    census_owned = census_owned.dropna(subset=[owned_col])

    owned_gdf = all_tracts.merge(census_owned, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')

    vmin = owned_gdf[owned_col].quantile(0.05)
    vmax = owned_gdf[owned_col].quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 12))
    greens = LinearSegmentedColormap.from_list('greens_clipped', plt.colormaps['Greens'](np.linspace(0.35, 1.0, 256)))

    owned_gdf.plot(column=owned_col, cmap=greens, legend=True, ax=ax,
                   vmin=vmin, vmax=vmax,
                   legend_kwds={'label': 'Proportion owner-occupied housing', 'shrink': 0.6})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'Owner-Occupied Housing across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    save_figure(figures_dir / f"{args.city}-Owner_Occupied_Housing.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Affluence & Disadvantage Scores -

    score_cols = ['AFFLUENCE13_17', 'DISADVANTAGE13_17']
    score_labels = ['Neighbourhood Affluence', 'Neighbourhood Disadvantage']
    score_cmaps = ['Purples', 'YlOrBr']

    census_scores = census[['TRACT_FIPS10'] + score_cols].copy()
    census_scores[score_cols] = census_scores[score_cols].apply(pd.to_numeric, errors='coerce')
    census_scores = census_scores.dropna(subset=score_cols)

    scores_gdf = all_tracts.merge(census_scores, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    for ax, col, label, cmap in zip(axes, score_cols, score_labels, score_cmaps):
        clipped = LinearSegmentedColormap.from_list(cmap, plt.colormaps[cmap](np.linspace(0.35, 1.0, 256)))
        vmin = scores_gdf[col].quantile(0.05)
        vmax = scores_gdf[col].quantile(0.95)
        scores_gdf.plot(column=col, cmap=clipped, legend=True, ax=ax,
                        vmin=vmin, vmax=vmax,
                        legend_kwds={'label': 'Index score (0–1)', 'shrink': 0.6})
        city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        ax.set_title(label, fontsize=14)
        ax.axis('off')

    plt.suptitle(f'Neighbourhood Scores across {args.city} Census Tracts', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(figures_dir / f"{args.city}-Affluence_Disadvantage.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Tweet Statistics ----

    # - Tweet Count per Tract -

    tweet_folder = Path(__file__).parent.parent / "CityTweets" / args.city

    if tweet_folder.exists():
        pickle_path = Path(__file__).parent.parent / "Pickles" / f"{args.city}_tweets.pkl"
        pickle_path.parent.mkdir(exist_ok=True)

        if pickle_path.exists():
            tweets_df = pd.read_pickle(pickle_path)
        else:
            def parse_file(fp):
                rows = []
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            tweet = json.loads(line)
                            centroid = tweet.get("centroid")
                            if centroid is None:
                                continue
                            lat, lon = centroid
                            rows.append((lon, lat, tweet.get("timestamp"), tweet.get("sentiment_score")))
                        except json.JSONDecodeError:
                            continue
                return rows

            all_rows = []
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(parse_file, f): f for f in tweet_folder.glob("*.jsonl")}
                for future in as_completed(futures):
                    all_rows.extend(future.result())

            tweets_df = pd.DataFrame(all_rows, columns=["lon", "lat", "timestamp", "sentiment"])
            tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"], errors="coerce")
            tweets_df.to_pickle(pickle_path)

        # Single spatial join — used for all tweet-based figures
        tweets_gdf = gpd.GeoDataFrame(
            tweets_df,
            geometry=gpd.points_from_xy(tweets_df["lon"], tweets_df["lat"]),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(tweets_gdf, city_gdf[["GEOID", "geometry"]], predicate="within")

        # - Tweet Count Choropleth -

        counts = joined.groupby("index_right").size()
        tweet_count_gdf = city_gdf.copy()
        tweet_count_gdf["tweet_count"] = 0
        tweet_count_gdf.loc[counts.index, "tweet_count"] = counts.values

        vmin = tweet_count_gdf["tweet_count"].quantile(0.05)
        vmax = tweet_count_gdf["tweet_count"].quantile(0.95)

        fig, ax = plt.subplots(figsize=(10, 12))
        tweet_count_gdf.plot(column="tweet_count", cmap="viridis", legend=True, ax=ax,
                             vmin=vmin, vmax=vmax,
                             legend_kwds={"label": "Tweet count", "shrink": 0.6})
        city_bounds.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=1.5)
        ax.set_title(f"Tweet Count per Census Tract — {args.city}", fontsize=14)
        ax.axis("off")
        save_figure(figures_dir / f"{args.city}-Tweet_Count.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Temporal Breakdowns -

        ts = to_local(joined["timestamp"].dropna(), local_tz)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        temporal = [
            (ts.dt.hour.value_counts().sort_index(),  range(24),    None,       "Hour of Day",  "Tweet Frequency by Hour of Day"),
            (ts.dt.dayofweek.value_counts().sort_index(), range(7), day_labels, "Day of Week",  "Tweet Frequency by Day of Week"),
            (ts.dt.month.value_counts().sort_index(), range(1, 13), month_labels, "Month",      "Tweet Frequency by Month"),
        ]
        out_names = ["Tweet_Hour", "Tweet_Day", "Tweet_Month"]
        fig_widths = [12, 10, 12]

        for (counts_s, idx, labels, xlabel, title), name, width in zip(temporal, out_names, fig_widths):
            values = [counts_s.get(i, 0) for i in idx]
            x_ticks = labels if labels else list(idx)
            fig, ax = plt.subplots(figsize=(width, 5))
            ax.bar(x_ticks, values, color="#06402B")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Tweet Count")
            ax.set_title(f"{title} — {args.city}")
            plt.tight_layout()
            save_figure(figures_dir / f"{args.city}-{name}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # ---- Sentiment Analysis ----

        # Aggregate daily sentiment per tract from the already-joined data
        joined["date"] = joined["timestamp"].dt.date
        tweet_data = (
            joined.dropna(subset=["sentiment", "date"])
            .groupby(["GEOID", "date"])
            .agg(mean_sentiment=("sentiment", "mean"), tweet_count=("sentiment", "size"))
            .reset_index()
            .sort_values(["GEOID", "date"])
        )
        tweet_data["sentiment_change"] = tweet_data.groupby("GEOID")["mean_sentiment"].diff()

        tract_sentiment = (
            tweet_data.groupby("GEOID").agg(
                mean_sentiment=("mean_sentiment", "mean"),
                total_tweets=("tweet_count", "sum")
            ).reset_index()
        )
        tract_sentiment["GEOID"] = tract_sentiment["GEOID"].astype(str).str.zfill(11)

        # - Mean Sentiment Choropleth -

        sentiment_map = city_gdf.merge(tract_sentiment, on="GEOID", how="left")

        fig, ax = plt.subplots(figsize=(10, 12))
        sentiment_map.plot(column="mean_sentiment", cmap="RdYlGn", legend=True, ax=ax,
                           vmin=0, vmax=1,
                           legend_kwds={"label": "Mean Sentiment Score", "shrink": 0.6})
        city_bounds.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
        ax.set_title(f"Mean Tweet Sentiment — {args.city}", fontsize=14)
        ax.axis("off")
        save_figure(figures_dir / f"{args.city}-Sentiment_Map.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Merge with income — filter low-sample tracts
        sentiment_income = (
            tract_sentiment
            .merge(gdf[["GEOID", "INCOME_SCORE"]], on="GEOID", how="inner")
            .pipe(lambda d: d[d["total_tweets"] >= 10])
            .dropna()
        )

        # - Sentiment vs Income Scatter -

        z = np.polyfit(sentiment_income["INCOME_SCORE"], sentiment_income["mean_sentiment"], 1)
        x_line = np.linspace(sentiment_income["INCOME_SCORE"].min(), sentiment_income["INCOME_SCORE"].max(), 200)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            sentiment_income["INCOME_SCORE"], sentiment_income["mean_sentiment"],
            s=50, alpha=0.5, c=sentiment_income["total_tweets"],
            cmap="plasma", norm=mpl.colors.LogNorm(),
            edgecolors="grey", linewidths=0.3
        )
        ax.plot(x_line, np.polyval(z, x_line), "k--", linewidth=1.2, label="Trend")
        plt.colorbar(scatter, ax=ax, label="Tweet Count")
        ax.set_xlabel("Estimated Mean Income ($k)")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title(f"Tweet Sentiment vs Tract Income — {args.city}")
        ax.legend()
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_vs_Income.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Residual Plot -

        sentiment_income = sentiment_income.copy()
        sentiment_income["residual"] = sentiment_income["mean_sentiment"] - np.polyval(z, sentiment_income["INCOME_SCORE"])
        sentiment_income["log_tweets"] = np.log1p(sentiment_income["total_tweets"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            sentiment_income["INCOME_SCORE"], sentiment_income["residual"],
            c=sentiment_income["log_tweets"], cmap="viridis", alpha=0.7
        )
        ax.axhline(0, linestyle="--", linewidth=1)
        plt.colorbar(sc, ax=ax, label="Log Tweet Count")
        ax.set_xlabel("Income Score")
        ax.set_ylabel("Residual Sentiment")
        ax.set_title(f"Residual Sentiment vs Income — {args.city}")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Residual.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Sentiment by Income Quintile Boxplot -

        sentiment_income["income_bin"] = pd.qcut(
            sentiment_income["INCOME_SCORE"], 5,
            labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=sentiment_income, x="income_bin", y="mean_sentiment", color="#595358", ax=ax)
        sns.stripplot(data=sentiment_income, x="income_bin", y="mean_sentiment",
                      color="black", alpha=0.3, size=2, ax=ax)
        ax.set_xlabel("Income Group (Quintiles)")
        ax.set_ylabel("Mean Sentiment")
        ax.set_title(f"Sentiment Distribution Across Income Groups — {args.city}")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Income_Boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Sentiment vs Census Variables ----

        # Merge tract sentiment with all relevant census variables
        sentiment_census = (
            tract_sentiment
            .merge(census_unemp,                              left_on="GEOID", right_on="TRACT_FIPS10")
            .merge(census_density,                            on="TRACT_FIPS10")
            .merge(census_edu[["TRACT_FIPS10", "PED3_13_17"]], on="TRACT_FIPS10")
            .merge(census_owned,                              on="TRACT_FIPS10")
            .merge(census_scores,                             on="TRACT_FIPS10")
            .pipe(lambda d: d[d["total_tweets"] >= 10])
            .dropna()
        )

        # - Sentiment vs Census Variables Scatter Grid -

        census_scatter_vars = [
            ("POPDEN13_17",       "Population Density (persons/sq mi)",   "Sentiment vs Population Density"),
            ("PUNEMP13_17",       "Proportion Unemployed",                 "Sentiment vs Unemployment"),
            ("PED3_13_17",        "Proportion with Bachelor's+",           "Sentiment vs Education"),
            ("POWNOC13_17",       "Proportion Owner-Occupied Housing",     "Sentiment vs Owner-Occupied Housing"),
            ("AFFLUENCE13_17",    "Affluence Score",                       "Sentiment vs Affluence"),
            ("DISADVANTAGE13_17", "Disadvantage Score",                    "Sentiment vs Disadvantage"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for ax, (col, xlabel, title) in zip(axes.flatten(), census_scatter_vars):
            x = sentiment_census[col]
            y = sentiment_census["mean_sentiment"]
            sizes = sentiment_census["total_tweets"] / sentiment_census["total_tweets"].max() * 35 + 5
            ax.scatter(x, y, s=sizes, alpha=0.4, color="#06402B")
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, np.polyval(z, x_line), color="crimson", linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean Sentiment")
            ax.set_title(title)

        plt.suptitle(f"Tweet Sentiment vs Census Variables — {args.city}", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_figure(figures_dir / f"{args.city}-Sentiment_Census_Scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Sentiment by Disadvantage Quintile Boxplot -

        sentiment_census["disadvantage_bin"] = pd.qcut(
            sentiment_census["DISADVANTAGE13_17"], 5,
            labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=sentiment_census, x="disadvantage_bin", y="mean_sentiment", color="#595358", ax=ax)
        sns.stripplot(data=sentiment_census, x="disadvantage_bin", y="mean_sentiment",
                      color="black", alpha=0.3, size=2, ax=ax)
        ax.set_xlabel("Disadvantage Group (Quintiles)")
        ax.set_ylabel("Mean Sentiment")
        ax.set_title(f"Sentiment Distribution Across Disadvantage Groups — {args.city}")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Disadvantage_Boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Sentiment by Dominant Education Level -

        census_edu_dom = census_edu.copy()
        census_edu_dom["dominant_edu"] = (
            census_edu_dom[edu_cols].idxmax(axis=1).map(dict(zip(edu_cols, edu_labels)))
        )

        sentiment_edu = (
            tract_sentiment
            .merge(census_edu_dom[["TRACT_FIPS10", "dominant_edu"]], left_on="GEOID", right_on="TRACT_FIPS10")
            .pipe(lambda d: d[d["total_tweets"] >= 10])
            .dropna()
        )
        sentiment_edu["dominant_edu"] = pd.Categorical(
            sentiment_edu["dominant_edu"], categories=edu_labels, ordered=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=sentiment_edu, x="dominant_edu", y="mean_sentiment", color="#595358", ax=ax)
        sns.stripplot(data=sentiment_edu, x="dominant_edu", y="mean_sentiment",
                      color="black", alpha=0.3, size=2, ax=ax)
        ax.set_xlabel("Dominant Education Level")
        ax.set_ylabel("Mean Sentiment")
        ax.set_title(f"Sentiment by Dominant Education Level — {args.city}")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Education_Boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Sentiment by Dominant Age Group -

        census_age_dom = census_age.copy()
        census_age_dom["dominant_age"] = (
            census_age_dom[age_cols].idxmax(axis=1).map(dict(zip(age_cols, age_labels)))
        )

        sentiment_age = (
            tract_sentiment
            .merge(census_age_dom[["TRACT_FIPS10", "dominant_age"]], left_on="GEOID", right_on="TRACT_FIPS10")
            .pipe(lambda d: d[d["total_tweets"] >= 10])
            .dropna()
        )
        sentiment_age["dominant_age"] = pd.Categorical(
            sentiment_age["dominant_age"], categories=age_labels, ordered=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=sentiment_age, x="dominant_age", y="mean_sentiment", color="#595358", ax=ax)
        sns.stripplot(data=sentiment_age, x="dominant_age", y="mean_sentiment",
                      color="black", alpha=0.3, size=2, ax=ax)
        ax.set_xlabel("Dominant Age Group")
        ax.set_ylabel("Mean Sentiment")
        ax.set_title(f"Sentiment by Dominant Age Group — {args.city}")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Age_Boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Sentiment vs Temporal Factors ----

        sent_temporal = joined[["timestamp", "sentiment"]].dropna().copy()
        sent_temporal["timestamp"] = to_local(sent_temporal["timestamp"], local_tz)
        sent_temporal["hour"] = sent_temporal["timestamp"].dt.hour
        sent_temporal["dayofweek"] = sent_temporal["timestamp"].dt.dayofweek
        sent_temporal["month"] = sent_temporal["timestamp"].dt.month
        sent_temporal["date"] = sent_temporal["timestamp"].dt.date

        # - Mean Sentiment by Hour of Day -

        hourly = sent_temporal.groupby("hour")["sentiment"].agg(["mean", "sem"]).reindex(range(24))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(24), hourly["mean"], color="#06402B", alpha=0.8)
        ax.errorbar(range(24), hourly["mean"], yerr=hourly["sem"] * 1.96,
                    fmt="none", color="black", linewidth=0.8, capsize=3)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title(f"Mean Tweet Sentiment by Hour of Day — {args.city}")
        ax.set_xticks(range(24))
        ax.set_ylim(0, 1)
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Hour.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Mean Sentiment by Day of Week -

        dow_sent = sent_temporal.groupby("dayofweek")["sentiment"].agg(["mean", "sem"]).reindex(range(7))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(day_labels, dow_sent["mean"], color="#06402B", alpha=0.8)
        ax.errorbar(range(7), dow_sent["mean"], yerr=dow_sent["sem"] * 1.96,
                    fmt="none", color="black", linewidth=0.8, capsize=3)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title(f"Mean Tweet Sentiment by Day of Week — {args.city}")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Day.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Mean Sentiment by Month -

        monthly_sent = sent_temporal.groupby("month")["sentiment"].agg(["mean", "sem"]).reindex(range(1, 13))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(month_labels, monthly_sent["mean"], color="#06402B", alpha=0.8)
        ax.errorbar(range(12), monthly_sent["mean"], yerr=monthly_sent["sem"] * 1.96,
                    fmt="none", color="black", linewidth=0.8, capsize=3)
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title(f"Mean Tweet Sentiment by Month — {args.city}")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Month.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Daily Sentiment Time Series -

        daily_ts = sent_temporal.groupby("date")["sentiment"].mean().sort_index()
        daily_ts.index = pd.to_datetime(daily_ts.index)
        rolling_avg = daily_ts.rolling(7, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily_ts.index, daily_ts.values, alpha=0.25, color="#06402B", linewidth=0.7, label="Daily mean")
        ax.plot(rolling_avg.index, rolling_avg.values, color="#06402B", linewidth=2, label="7-day rolling mean")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Sentiment Score")
        ax.set_title(f"Daily Tweet Sentiment Over Time — {args.city}")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Timeseries.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Sentiment Heatmap: Hour x Day of Week -

        pivot = sent_temporal.groupby(["dayofweek", "hour"])["sentiment"].mean().unstack(fill_value=np.nan)

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_labels)
        ax.set_xlabel("Hour of Day")
        ax.set_title(f"Mean Sentiment Heatmap (Hour × Day of Week) — {args.city}")
        plt.colorbar(im, ax=ax, label="Mean Sentiment Score")
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---- Extreme Sentiment vs Temporal Factors ----

        POS_THRESHOLD = 0.65
        NEG_THRESHOLD = 0.35

        sent_temporal["positive"] = sent_temporal["sentiment"] > POS_THRESHOLD
        sent_temporal["negative"] = sent_temporal["sentiment"] < NEG_THRESHOLD

        temporal_extremes = [
            ("hour",      range(24),    None,         "Hour of Day", "Extreme Sentiment by Hour of Day", "Sentiment_Extremes_Hour",  12),
            ("dayofweek", range(7),     day_labels,   "Day of Week", "Extreme Sentiment by Day of Week", "Sentiment_Extremes_Day",   10),
            ("month",     range(1, 13), month_labels, "Month",       "Extreme Sentiment by Month",       "Sentiment_Extremes_Month", 12),
        ]

        for col, idx, labels, xlabel, title, fname, width in temporal_extremes:
            grp = sent_temporal.groupby(col)[["positive", "negative"]].mean().reindex(idx) * 100
            x = np.arange(len(list(idx)))
            bar_w = 0.35
            x_ticks = labels if labels else list(idx)

            fig, ax = plt.subplots(figsize=(width, 5))
            ax.bar(x - bar_w / 2, grp["positive"], bar_w, label=f"Positive (>{POS_THRESHOLD})", color="#2ecc71")
            ax.bar(x + bar_w / 2, grp["negative"], bar_w, label=f"Negative (<{NEG_THRESHOLD})", color="#e74c3c")
            ax.set_xticks(x)
            ax.set_xticklabels(x_ticks)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("% of Tweets")
            ax.set_title(f"{title} — {args.city}")
            ax.legend()
            plt.tight_layout()
            save_figure(figures_dir / f"{args.city}-{fname}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # - Extreme Sentiment Time Series -

        daily_pos = sent_temporal.groupby("date")["positive"].mean().sort_index() * 100
        daily_neg = sent_temporal.groupby("date")["negative"].mean().sort_index() * 100
        daily_pos.index = pd.to_datetime(daily_pos.index)
        daily_neg.index = pd.to_datetime(daily_neg.index)

        roll_pos = daily_pos.rolling(7, center=True, min_periods=1).mean()
        roll_neg = daily_neg.rolling(7, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily_pos.index, daily_pos.values, alpha=0.2, color="#2ecc71", linewidth=0.7)
        ax.plot(daily_neg.index, daily_neg.values, alpha=0.2, color="#e74c3c", linewidth=0.7)
        ax.plot(roll_pos.index, roll_pos.values, color="#2ecc71", linewidth=2, label=f"% Positive (>{POS_THRESHOLD})")
        ax.plot(roll_neg.index, roll_neg.values, color="#e74c3c", linewidth=2, label=f"% Negative (<{NEG_THRESHOLD})")
        ax.set_xlabel("Date")
        ax.set_ylabel("% of Tweets")
        ax.set_title(f"Extreme Tweet Sentiment Over Time — {args.city}")
        ax.legend()
        plt.tight_layout()
        save_figure(figures_dir / f"{args.city}-Sentiment_Extremes_Timeseries.png", dpi=300, bbox_inches="tight")
        plt.close()