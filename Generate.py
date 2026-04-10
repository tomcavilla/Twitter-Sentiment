
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


CITIES_SHAPEFILE = "Utilities/Shapefiles/CityBoundaries.shp"

GENERATED_FIGURES = []

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
    figures_dir = out_dir / "Figures"
    figures_dir.mkdir(exist_ok=True)

    # Create gdf of tracts overlapping city boundaries.

    city_gdf = city_tracts(args.tracts, args.city)

    # ---- General Tweet Stats (Stats.json) ----

    stats_path = Path(__file__).parent.parent / "Stats.json"
    tweet_figures = [
        figures_dir / "Tweet_Count_Top10.png",
        figures_dir / "Tweet_Count_Top5.png",
        figures_dir / "Tweet_Count_Months.png",
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
        plt.savefig(figures_dir / "Tweet_Count_Top10.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(cities[:5], counts[:5], color='#06402B')
        ax.invert_yaxis()
        ax.set_xlabel("Count (Millions)")
        ax.xaxis.set_major_formatter(millions_fmt)
        ax.set_title("Top 5 Cities by Tweet Count in 2017")
        plt.tight_layout()
        plt.savefig(figures_dir / "Tweet_Count_Top5.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(figures_dir / "Tweet_Count_Months.png", dpi=300, bbox_inches="tight")
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

    gdf = gpd.read_file("Utilities/Shapefiles/New York Tracts/tl_2017_36_tract.shp")  
    gdf = gdf.merge(census_income[['TRACT_FIPS10', 'INCOME_SCORE']], 
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
    plt.savefig(figures_dir / f"{args.city}-Wealth_Distribution.png", dpi=300)
    plt.close()

    # - Age Distribution -

    age_cols = ['P18_2913_17', 'P30_3913_17', 'P40_4913_17', 'P50_6913_17', 'PGE7013_17']
    age_labels = ['18–29', '30–39', '40–49', '50–69', '70+']

    all_tracts = gpd.read_file(args.tracts)
    all_tracts['GEOID'] = all_tracts['GEOID'].astype(str).str.zfill(11)

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
    plt.savefig(figures_dir / f"{args.city}-Age_Distribution.png", dpi=300)
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
    plt.savefig(figures_dir / f"{args.city}-Dominant_Age.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Income_vs_Age.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Population_Density.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Education_Levels.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Dominant_Education.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Unemployment.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Owner_Occupied_Housing.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{args.city}-Affluence_Disadvantage.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(figures_dir / f"{args.city}-Tweet_Count.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Tweet Count Scatter Plots -

        scatter_census = census[['TRACT_FIPS10', 'POPDEN13_17', 'PUNEMP13_17', 'PED3_13_17']].copy()
        scatter_census = scatter_census.apply(pd.to_numeric, errors='coerce')
        scatter_census['TRACT_FIPS10'] = census['TRACT_FIPS10']

        scatter_df = (
            tweet_count_gdf[["GEOID", "tweet_count"]]
            .merge(census_income[["TRACT_FIPS10", "INCOME_SCORE"]], left_on="GEOID", right_on="TRACT_FIPS10")
            .merge(scatter_census, on="TRACT_FIPS10")
            .dropna()
        )
        scatter_df = scatter_df[scatter_df["tweet_count"] > 0]

        scatter_vars = [
            ("INCOME_SCORE", "Weighted Income Score ($k)", "Tweet Count vs Income"),
            ("PED3_13_17",   "Proportion with Bachelor's+", "Tweet Count vs Education"),
            ("POPDEN13_17",  "Persons per sq. mile",        "Tweet Count vs Population Density"),
            ("PUNEMP13_17",  "Proportion unemployed",       "Tweet Count vs Unemployment"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        for ax, (col, xlabel, title) in zip(axes.flatten(), scatter_vars):
            x = scatter_df[col]
            y = scatter_df["tweet_count"]
            ax.scatter(x, y, alpha=0.35, s=10, color="#06402B")
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Tweet count")
            ax.set_title(title)

        plt.suptitle(f"Tweet Count vs Census Variables — {args.city}", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(figures_dir / f"{args.city}-Tweet_Scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

        # - Temporal Breakdowns -

        ts = joined["timestamp"].dropna()
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
            plt.savefig(figures_dir / f"{args.city}-{name}.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(figures_dir / f"{args.city}-Sentiment_Map.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(figures_dir / f"{args.city}-Sentiment_vs_Income.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(figures_dir / f"{args.city}-Sentiment_Residual.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(figures_dir / f"{args.city}-Sentiment_Income_Boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

