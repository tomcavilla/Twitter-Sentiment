import argparse
import json
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats


CITIES_SHAPEFILE = "Utilities/Shapefiles/CityBoundaries.shp"


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


def load_sentiment_census(city, csv_dir):
    """Loads the pre-aggregated tract-level CSV produced by Export_Scatter_Data.py.
    Columns include: GEOID, total_tweets, mean_sentiment_by_tweet, total_users,
    mean_sentiment_by_user, INCOME_SCORE, and the NaNDA census variables."""
    csv_path = Path(csv_dir) / f"{city}-Sentiment_Census.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No exported CSV found at {csv_path} — run Export_Scatter_Data.py for this city first."
        )

    df = pd.read_csv(csv_path, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.zfill(11)
    return df


if __name__ == "__main__":
    # ---- Add Arguments ----

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, help="Chosen city to analyse.")
    parser.add_argument("--tracts", type=str, required=True, help="Shapefile of chosen city's state.")
    parser.add_argument("--out", type=str, required=False, help="The output directory for generated figures.")
    parser.add_argument(
        "--csv-dir", type=str, required=False,
        help="Directory containing the exported <city>-Sentiment_Census.csv files (defaults to ../CSV)."
    )
    args = parser.parse_args()

    # ---- Check Arguments ----

    if check_city(args.city) == False:
        raise ValueError(f"City '{args.city}' not found in analysed cities.")

    if not Path(args.tracts).exists():
        raise FileNotFoundError("Tracts shapefile not found.")

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).parent / "Out"

    out_dir.mkdir(parents=True, exist_ok=True)
    general_figures_dir = out_dir / "Figures"
    general_figures_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "Figures" / args.city
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(__file__).parent.parent / "CSV"

    sentiment_census = load_sentiment_census(args.city, csv_dir)

    city_gdf = city_tracts(args.tracts, args.city)
    city_gdf["GEOID"] = city_gdf["GEOID"].astype(str).str.zfill(11)

    # ---- General Tweet Stats (Stats.json) ----

    stats_path = Path(__file__).parent.parent / "Stats.json"
    tweet_figures = [
        general_figures_dir / "Tweet_Count_Top10.png",
        general_figures_dir / "Tweet_Count_Top5.png",
        general_figures_dir / "Tweet_Count_Months.png",
    ]

    if stats_path.exists() and not all(f.exists() for f in tweet_figures):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

        top10 = sorted(stats_data['cities'].items(), key=lambda x: x[1], reverse=True)[:10]
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
        plt.savefig(general_figures_dir / "Tweet_Count_Top10.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(cities[:5], counts[:5], color='#06402B')
        ax.invert_yaxis()
        ax.set_xlabel("Count (Millions)")
        ax.xaxis.set_major_formatter(millions_fmt)
        ax.set_title("Top 5 Cities by Tweet Count in 2017")
        plt.tight_layout()
        plt.savefig(general_figures_dir / "Tweet_Count_Top5.png", dpi=300, bbox_inches="tight")
        plt.close()

        month_key = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
        tweet_count = []

        with open(out_dir / "General_Stats.txt", "w") as f:
            f.write(f"Total Count: {stats_data['count']:,}\n\nDifferent Users: {stats_data['different_users']:,}\n\n")
            f.write("Overall Top 10 Cities:\n\n")
            for city, count in top10:
                f.write(f"{city}: {count:,}\n")

            for month, data in stats_data["monthly"].items():
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
        plt.savefig(general_figures_dir / "Tweet_Count_Months.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---- General Census Data ----

    census = pd.read_csv("Utilities/ICPSR-CENSUS.tsv", delimiter="\t", low_memory=False)

    census['TRACT_FIPS10'] = census['TRACT_FIPS10'].astype(str)
    city_gdf['GEOID'] = city_gdf['GEOID'].astype(str)
    census['TRACT_FIPS10'] = census['TRACT_FIPS10'].str.zfill(11)
    city_gdf['GEOID'] = city_gdf['GEOID'].str.zfill(11)

    census = census[census['TRACT_FIPS10'].isin(city_gdf['GEOID'].unique())].copy()

    # - Wealth Distribution -

    income_cols = ['PIN1B_13_17', 'PIN2B_13_17', 'PIN3B_13_17', 'PIN4B_13_17', 'PIN5B_13_17']
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

    city_bounds = gpd.read_file(CITIES_SHAPEFILE)
    city_bounds = city_bounds[city_bounds["NAME"] == args.city].to_crs(all_tracts.crs)
    city_boundary = city_bounds.dissolve()

    city_bounds_geo = city_bounds.to_crs(city_gdf.crs)
    city_boundary_geo = city_boundary.to_crs(city_gdf.crs)

    gdf = all_tracts.merge(census_income[['TRACT_FIPS10', 'INCOME_SCORE']],
                           left_on='GEOID', right_on='TRACT_FIPS10')
    gdf = gpd.clip(gdf, city_boundary)

    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(
        column='INCOME_SCORE',
        cmap='RdYlGn',
        legend=True,
        legend_kwds={'label': 'Weighted Income Score ($k)'},
        ax=ax
    )
    city_bounds.boundary.plot(ax=ax, color='black', linewidth=0.5)
    plt.title(f'Wealth Distribution across {args.city} Census Tracts')
    plt.axis('off')
    plt.savefig(figures_dir / f"{args.city}-Wealth_Distribution.png", dpi=300)
    plt.close()

    # - Age Distribution -

    age_cols = ['P18_2913_17', 'P30_3913_17', 'P40_4913_17', 'P50_6913_17', 'PGE7013_17']
    age_labels = ['18–29', '30–39', '40–49', '50–69', '70+']

    census_age = census[['TRACT_FIPS10'] + age_cols].copy()
    census_age[age_cols] = census_age[age_cols].apply(pd.to_numeric, errors='coerce')
    census_age = census_age.dropna(subset=age_cols)

    age_gdf = all_tracts.merge(census_age, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')
    age_gdf = gpd.clip(age_gdf, city_boundary)

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

    income_age['income_bracket'] = income_age[income_cols].idxmax(axis=1).map(
        dict(zip(income_cols, income_labels))
    )

    heatmap_data = (
        income_age.groupby('income_bracket')[age_cols]
        .mean()
        .rename(columns=dict(zip(age_cols, age_labels)))
        .reindex(income_labels)
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
    density_gdf = gpd.clip(density_gdf, city_boundary)

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
    edu_gdf = gpd.clip(edu_gdf, city_boundary)

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

    # - Income vs Education Heatmap -

    income_edu = census_income[['TRACT_FIPS10'] + income_cols].merge(
        census_edu[['TRACT_FIPS10'] + edu_cols], on='TRACT_FIPS10'
    )
    income_edu['income_bracket'] = income_edu[income_cols].idxmax(axis=1).map(
        dict(zip(income_cols, income_labels))
    )

    inc_edu_heatmap = (
        income_edu.groupby('income_bracket')[edu_cols]
        .mean()
        .rename(columns=dict(zip(edu_cols, edu_labels)))
        .reindex(income_labels)
    )

    yg = LinearSegmentedColormap.from_list('ylgn', plt.colormaps['YlGn'](np.linspace(0.2, 1.0, 256)))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(inc_edu_heatmap.values, cmap=yg, aspect='auto')
    ax.set_xticks(range(len(edu_labels)))
    ax.set_xticklabels(edu_labels)
    ax.set_yticks(range(len(income_labels)))
    ax.set_yticklabels(income_labels)
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Dominant Income Bracket')
    for i in range(len(income_labels)):
        for j in range(len(edu_labels)):
            ax.text(j, i, f"{inc_edu_heatmap.values[i, j]:.2f}",
                    ha='center', va='center', fontsize=9, color='white')
    plt.colorbar(im, ax=ax, label='Mean proportion of tract population')
    plt.title(f'Education Distribution by Income Bracket — {args.city}')
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Income_vs_Education.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Unemployment -

    unemp_col = 'PUNEMP13_17'

    census_unemp = census[['TRACT_FIPS10', unemp_col]].copy()
    census_unemp[unemp_col] = pd.to_numeric(census_unemp[unemp_col], errors='coerce')
    census_unemp = census_unemp.dropna(subset=[unemp_col])

    unemp_gdf = all_tracts.merge(census_unemp, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')
    unemp_gdf = gpd.clip(unemp_gdf, city_boundary)

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
    owned_gdf = gpd.clip(owned_gdf, city_boundary)

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
    scores_gdf = gpd.clip(scores_gdf, city_boundary)

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

    # - Race & Ethnicity -

    ethnicity_cols   = ['PNHWHITE13_17', 'PNHBLACK13_17', 'PHISPANIC13_17', 'PFBORN13_17']
    ethnicity_labels = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Born Outside the U.S.']

    census_ethnicity = census[['TRACT_FIPS10'] + ethnicity_cols].copy()
    census_ethnicity[ethnicity_cols] = census_ethnicity[ethnicity_cols].apply(pd.to_numeric, errors='coerce')
    census_ethnicity = census_ethnicity.dropna(subset=ethnicity_cols)

    ethnicity_gdf = all_tracts.merge(census_ethnicity, left_on='GEOID', right_on='TRACT_FIPS10', how='inner')
    ethnicity_gdf = gpd.clip(ethnicity_gdf, city_boundary)

    oranges_clipped = LinearSegmentedColormap.from_list(
        'oranges_eth', plt.colormaps['Oranges'](np.linspace(0.2, 1.0, 256))
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for ax, col, label in zip(axes.flatten(), ethnicity_cols, ethnicity_labels):
        vmin = ethnicity_gdf[col].quantile(0.05)
        vmax = ethnicity_gdf[col].quantile(0.95)
        ethnicity_gdf.plot(column=col, cmap=oranges_clipped, legend=True, ax=ax,
                           vmin=vmin, vmax=vmax,
                           legend_kwds={'shrink': 0.6, 'label': '% of population'})
        city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(f'Race & Ethnicity across {args.city} Census Tracts', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures_dir / f"{args.city}-Race_Ethnicity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Dominant Ethnicity Group -

    ethnicity_gdf['dominant_ethnicity'] = (
        ethnicity_gdf[ethnicity_cols].idxmax(axis=1).map(dict(zip(ethnicity_cols, ethnicity_labels)))
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    ethnicity_gdf.plot(column='dominant_ethnicity', cmap='tab10', legend=True, ax=ax,
                       legend_kwds={'title': 'Dominant Group', 'loc': 'upper left',
                                    'bbox_to_anchor': (1, 1), 'frameon': True})
    city_bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.set_title(f'Dominant Race/Ethnicity Group across {args.city} Census Tracts', fontsize=14)
    ax.axis('off')
    plt.savefig(figures_dir / f"{args.city}-Dominant_Ethnicity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Ethnicity vs Income Heatmap -

    ethnicity_income = census_ethnicity[['TRACT_FIPS10'] + ethnicity_cols].merge(
        census_income[['TRACT_FIPS10'] + income_cols], on='TRACT_FIPS10'
    )
    ethnicity_income['dominant_ethnicity'] = (
        ethnicity_income[ethnicity_cols].idxmax(axis=1).map(dict(zip(ethnicity_cols, ethnicity_labels)))
    )

    eth_income_heatmap = (
        ethnicity_income.groupby('dominant_ethnicity')[income_cols]
        .mean()
        .rename(columns=dict(zip(income_cols, income_labels)))
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(eth_income_heatmap.values, cmap=oranges_clipped, aspect='auto')
    ax.set_xticks(range(len(income_labels)))
    ax.set_xticklabels(income_labels)
    ax.set_yticks(range(len(eth_income_heatmap)))
    ax.set_yticklabels(eth_income_heatmap.index)
    ax.set_xlabel('Income Bracket')
    ax.set_ylabel('Dominant Ethnicity Group')
    for i in range(len(eth_income_heatmap)):
        for j in range(len(income_labels)):
            ax.text(j, i, f"{eth_income_heatmap.values[i, j]:.2f}",
                    ha='center', va='center', fontsize=9, color='white')
    plt.colorbar(im, ax=ax, label='Mean proportion of tract population')
    plt.title(f'Income Distribution by Dominant Ethnicity — {args.city}')
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Ethnicity_vs_Income.png", dpi=300, bbox_inches='tight')
    plt.close()

    # - Ethnicity vs Education Heatmap -

    ethnicity_edu = census_ethnicity[['TRACT_FIPS10'] + ethnicity_cols].merge(
        census_edu[['TRACT_FIPS10'] + edu_cols], on='TRACT_FIPS10'
    )
    ethnicity_edu['dominant_ethnicity'] = (
        ethnicity_edu[ethnicity_cols].idxmax(axis=1).map(dict(zip(ethnicity_cols, ethnicity_labels)))
    )

    eth_edu_heatmap = (
        ethnicity_edu.groupby('dominant_ethnicity')[edu_cols]
        .mean()
        .rename(columns=dict(zip(edu_cols, edu_labels)))
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(eth_edu_heatmap.values, cmap=oranges_clipped, aspect='auto')
    ax.set_xticks(range(len(edu_labels)))
    ax.set_xticklabels(edu_labels)
    ax.set_yticks(range(len(eth_edu_heatmap)))
    ax.set_yticklabels(eth_edu_heatmap.index)
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Dominant Ethnicity Group')
    for i in range(len(eth_edu_heatmap)):
        for j in range(len(edu_labels)):
            ax.text(j, i, f"{eth_edu_heatmap.values[i, j]:.2f}",
                    ha='center', va='center', fontsize=9, color='white')
    plt.colorbar(im, ax=ax, label='Mean proportion of tract population')
    plt.title(f'Education Distribution by Dominant Ethnicity — {args.city}')
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Ethnicity_vs_Education.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Tweet & Sentiment Statistics ----

    # - Tweet Count per Tract Choropleth -

    tweet_count_gdf = city_gdf.merge(sentiment_census[["GEOID", "total_tweets"]], on="GEOID", how="left")
    tweet_count_gdf["total_tweets"] = tweet_count_gdf["total_tweets"].fillna(0)
    tweet_count_gdf = gpd.clip(tweet_count_gdf, city_boundary_geo)

    vmin = tweet_count_gdf["total_tweets"].quantile(0.05)
    vmax = tweet_count_gdf["total_tweets"].quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 12))
    tweet_count_gdf.plot(column="total_tweets", cmap="viridis", legend=True, ax=ax,
                         vmin=vmin, vmax=vmax,
                         legend_kwds={"label": "Tweet count", "shrink": 0.6})
    city_bounds_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
    ax.set_title(f"Tweet Count per Census Tract — {args.city}", fontsize=14)
    ax.axis("off")
    plt.savefig(figures_dir / f"{args.city}-Tweet_Count.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Mean Sentiment Choropleths (Tweet vs User) -

    sentiment_map = city_gdf.merge(
        sentiment_census[["GEOID", "mean_sentiment_by_tweet", "mean_sentiment_by_user"]],
        on="GEOID", how="left"
    )
    sentiment_map = gpd.clip(sentiment_map, city_boundary_geo)

    # 1. By Tweet
    fig, ax = plt.subplots(figsize=(10, 12))
    sentiment_map.plot(column="mean_sentiment_by_tweet", cmap="RdYlGn", legend=True, ax=ax,
                       vmin=0, vmax=1, missing_kwds={"color": "white", "label": "No data"},
                       legend_kwds={"label": "Mean Sentiment Score (by tweet)", "shrink": 0.6})
    city_bounds_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
    ax.set_title(f"Mean Tweet Sentiment — {args.city}", fontsize=14)
    ax.axis("off")
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Map_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    fig, ax = plt.subplots(figsize=(10, 12))
    sentiment_map.plot(column="mean_sentiment_by_user", cmap="RdYlGn", legend=True, ax=ax,
                       vmin=0, vmax=1, missing_kwds={"color": "white", "label": "No data"},
                       legend_kwds={"label": "Mean Sentiment Score (by user)", "shrink": 0.6})
    city_bounds_geo.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
    ax.set_title(f"Mean User Sentiment — {args.city}", fontsize=14)
    ax.axis("off")
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Map_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment vs Income Scatters (Tweet vs User) -

    # 1. By Tweet
    tweet_income = sentiment_census.dropna(subset=["INCOME_SCORE", "mean_sentiment_by_tweet", "total_tweets"])
    z_t = np.polyfit(tweet_income["INCOME_SCORE"], tweet_income["mean_sentiment_by_tweet"], 1)
    x_line_t = np.linspace(tweet_income["INCOME_SCORE"].min(), tweet_income["INCOME_SCORE"].max(), 200)
    r_val_t, p_val_t = stats.pearsonr(tweet_income["INCOME_SCORE"], tweet_income["mean_sentiment_by_tweet"])
    r_squared_t = r_val_t ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        tweet_income["INCOME_SCORE"], tweet_income["mean_sentiment_by_tweet"],
        s=50, alpha=0.5, c=tweet_income["total_tweets"],
        cmap="plasma", norm=mpl.colors.LogNorm(),
        edgecolors="grey", linewidths=0.3
    )
    ax.plot(x_line_t, np.polyval(z_t, x_line_t), "k--", linewidth=1.2)
    plt.colorbar(scatter, ax=ax, label="Tweet Count")
    ax.set_xlabel("Estimated Mean Income ($k)")
    ax.set_ylabel("Mean Tweet Sentiment")
    ax.set_title(
        f"Tweet-Aggregated Sentiment vs Tract Income — {args.city}\n"
        f"r = {r_val_t:.3f}, R² = {r_squared_t:.3f} (p = {p_val_t:.2e})"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_vs_Income_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    user_income = sentiment_census.dropna(subset=["INCOME_SCORE", "mean_sentiment_by_user", "total_users"])
    z_u = np.polyfit(user_income["INCOME_SCORE"], user_income["mean_sentiment_by_user"], 1)
    x_line_u = np.linspace(user_income["INCOME_SCORE"].min(), user_income["INCOME_SCORE"].max(), 200)
    r_val_u, p_val_u = stats.pearsonr(user_income["INCOME_SCORE"], user_income["mean_sentiment_by_user"])
    r_squared_u = r_val_u ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        user_income["INCOME_SCORE"], user_income["mean_sentiment_by_user"],
        s=50, alpha=0.5, c=user_income["total_users"],
        cmap="plasma", norm=mpl.colors.LogNorm(),
        edgecolors="grey", linewidths=0.3
    )
    ax.plot(x_line_u, np.polyval(z_u, x_line_u), "k--", linewidth=1.2)
    plt.colorbar(scatter, ax=ax, label="Unique Users")
    ax.set_xlabel("Estimated Mean Income ($k)")
    ax.set_ylabel("Mean User Sentiment")
    ax.set_title(
        f"User-Aggregated Sentiment vs Tract Income — {args.city}\n"
        f"r = {r_val_u:.3f}, R² = {r_squared_u:.3f} (p = {p_val_u:.2e})"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_vs_Income_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Residual Plots (Tweet vs User) -

    # 1. By Tweet
    sentiment_income_t = tweet_income.copy()
    sentiment_income_t["residual"] = sentiment_income_t["mean_sentiment_by_tweet"] - np.polyval(z_t, sentiment_income_t["INCOME_SCORE"])
    sentiment_income_t["log_tweets"] = np.log1p(sentiment_income_t["total_tweets"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        sentiment_income_t["INCOME_SCORE"], sentiment_income_t["residual"],
        c=sentiment_income_t["log_tweets"], cmap="viridis", alpha=0.7
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    plt.colorbar(sc, ax=ax, label="Log Tweet Count")
    ax.set_xlabel("Income Score")
    ax.set_ylabel("Residual Sentiment (by tweet)")
    ax.set_title(f"Residual Sentiment vs Income (by Tweet) — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Residual_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    sentiment_income_u = user_income.copy()
    sentiment_income_u["residual"] = sentiment_income_u["mean_sentiment_by_user"] - np.polyval(z_u, sentiment_income_u["INCOME_SCORE"])
    sentiment_income_u["log_users"] = np.log1p(sentiment_income_u["total_users"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        sentiment_income_u["INCOME_SCORE"], sentiment_income_u["residual"],
        c=sentiment_income_u["log_users"], cmap="viridis", alpha=0.7
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    plt.colorbar(sc, ax=ax, label="Log User Count")
    ax.set_xlabel("Income Score")
    ax.set_ylabel("Residual Sentiment (by user)")
    ax.set_title(f"Residual Sentiment vs Income (by User) — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Residual_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment by Income Quintile Boxplots (Tweet vs User) -

    # 1. By Tweet
    sentiment_income_t["income_bin"] = pd.qcut(
        sentiment_income_t["INCOME_SCORE"], 5,
        labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=sentiment_income_t, x="income_bin", y="mean_sentiment_by_tweet", color="#595358", ax=ax)
    sns.stripplot(data=sentiment_income_t, x="income_bin", y="mean_sentiment_by_tweet",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Income Group (Quintiles)")
    ax.set_ylabel("Mean Sentiment (by tweet)")
    ax.set_title(f"Tweet Sentiment Distribution Across Income Groups — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Income_Boxplot_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    sentiment_income_u["income_bin"] = pd.qcut(
        sentiment_income_u["INCOME_SCORE"], 5,
        labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=sentiment_income_u, x="income_bin", y="mean_sentiment_by_user", color="#595358", ax=ax)
    sns.stripplot(data=sentiment_income_u, x="income_bin", y="mean_sentiment_by_user",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Income Group (Quintiles)")
    ax.set_ylabel("Mean Sentiment (by user)")
    ax.set_title(f"User Sentiment Distribution Across Income Groups — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Income_Boxplot_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Sentiment vs Census Variables ----

    census_scatter_vars = [
        ("POPDEN13_17",       "Population Density (persons/sq mi)",   "Population Density"),
        ("PUNEMP13_17",       "Proportion Unemployed",                 "Unemployment"),
        ("PED3_13_17",        "Proportion with Bachelor's+",           "Education"),
        ("POWNOC13_17",       "Proportion Owner-Occupied Housing",     "Owner-Occupied Housing"),
        ("AFFLUENCE13_17",    "Affluence Score",                       "Affluence"),
        ("DISADVANTAGE13_17", "Disadvantage Score",                    "Disadvantage"),
    ]

    # - Sentiment vs Census Variables Scatter Grids (Tweet vs User) -

    # 1. By Tweet
    census_scatter_t = sentiment_census.dropna(
        subset=[col for col, _, _ in census_scatter_vars] + ["mean_sentiment_by_tweet", "total_tweets"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (col, xlabel, title) in zip(axes.flatten(), census_scatter_vars):
        x = census_scatter_t[col]
        y = census_scatter_t["mean_sentiment_by_tweet"]
        sizes = census_scatter_t["total_tweets"] / census_scatter_t["total_tweets"].max() * 35 + 5
        ax.scatter(x, y, s=sizes, alpha=0.4, color="#06402B")
        z_c = np.polyfit(x, y, 1)
        x_line_c = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line_c, np.polyval(z_c, x_line_c), color="crimson", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Sentiment (by tweet)")
        ax.set_title(f"Sentiment vs {title}")

    plt.suptitle(f"Tweet Sentiment (by Tweet) vs Census Variables — {args.city}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Census_Scatter_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    census_scatter_u = sentiment_census.dropna(
        subset=[col for col, _, _ in census_scatter_vars] + ["mean_sentiment_by_user", "total_users"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (col, xlabel, title) in zip(axes.flatten(), census_scatter_vars):
        x = census_scatter_u[col]
        y = census_scatter_u["mean_sentiment_by_user"]
        sizes = census_scatter_u["total_users"] / census_scatter_u["total_users"].max() * 35 + 5
        ax.scatter(x, y, s=sizes, alpha=0.4, color="#06402B")
        z_c = np.polyfit(x, y, 1)
        x_line_c = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line_c, np.polyval(z_c, x_line_c), color="crimson", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Sentiment (by user)")
        ax.set_title(f"Sentiment vs {title}")

    plt.suptitle(f"User Sentiment vs Census Variables — {args.city}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Census_Scatter_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment by Disadvantage Quintile Boxplots (Tweet vs User) -

    # 1. By Tweet
    disadv_t = sentiment_census.dropna(subset=["DISADVANTAGE13_17", "mean_sentiment_by_tweet"]).copy()
    disadv_t["disadvantage_bin"] = pd.qcut(
        disadv_t["DISADVANTAGE13_17"], 5,
        labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=disadv_t, x="disadvantage_bin", y="mean_sentiment_by_tweet", color="#595358", ax=ax)
    sns.stripplot(data=disadv_t, x="disadvantage_bin", y="mean_sentiment_by_tweet",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Disadvantage Group (Quintiles)")
    ax.set_ylabel("Mean Sentiment (by tweet)")
    ax.set_title(f"Tweet Sentiment Distribution Across Disadvantage Groups — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Disadvantage_Boxplot_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    disadv_u = sentiment_census.dropna(subset=["DISADVANTAGE13_17", "mean_sentiment_by_user"]).copy()
    disadv_u["disadvantage_bin"] = pd.qcut(
        disadv_u["DISADVANTAGE13_17"], 5,
        labels=["0–20th\npct", "20–40th\npct", "40–60th\npct", "60–80th\npct", "80–100th\npct"]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=disadv_u, x="disadvantage_bin", y="mean_sentiment_by_user", color="#595358", ax=ax)
    sns.stripplot(data=disadv_u, x="disadvantage_bin", y="mean_sentiment_by_user",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Disadvantage Group (Quintiles)")
    ax.set_ylabel("Mean Sentiment (by user)")
    ax.set_title(f"User Sentiment Distribution Across Disadvantage Groups — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Disadvantage_Boxplot_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment by Dominant Education Level (Tweet vs User) -

    # 1. By Tweet
    edu_t = sentiment_census.dropna(subset=edu_cols + ["mean_sentiment_by_tweet"]).copy()
    edu_t["dominant_edu"] = pd.Categorical(
        edu_t[edu_cols].idxmax(axis=1).map(dict(zip(edu_cols, edu_labels))),
        categories=edu_labels, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=edu_t, x="dominant_edu", y="mean_sentiment_by_tweet", color="#595358", ax=ax)
    sns.stripplot(data=edu_t, x="dominant_edu", y="mean_sentiment_by_tweet",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Education Level")
    ax.set_ylabel("Mean Sentiment (by tweet)")
    ax.set_title(f"Tweet Sentiment by Dominant Education Level — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Education_Boxplot_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    edu_u = sentiment_census.dropna(subset=edu_cols + ["mean_sentiment_by_user"]).copy()
    edu_u["dominant_edu"] = pd.Categorical(
        edu_u[edu_cols].idxmax(axis=1).map(dict(zip(edu_cols, edu_labels))),
        categories=edu_labels, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=edu_u, x="dominant_edu", y="mean_sentiment_by_user", color="#595358", ax=ax)
    sns.stripplot(data=edu_u, x="dominant_edu", y="mean_sentiment_by_user",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Education Level")
    ax.set_ylabel("Mean Sentiment (by user)")
    ax.set_title(f"User Sentiment by Dominant Education Level — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Education_Boxplot_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment by Dominant Age Group (Tweet vs User) -

    # 1. By Tweet
    age_t = sentiment_census.dropna(subset=age_cols + ["mean_sentiment_by_tweet"]).copy()
    age_t["dominant_age"] = pd.Categorical(
        age_t[age_cols].idxmax(axis=1).map(dict(zip(age_cols, age_labels))),
        categories=age_labels, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=age_t, x="dominant_age", y="mean_sentiment_by_tweet", color="#595358", ax=ax)
    sns.stripplot(data=age_t, x="dominant_age", y="mean_sentiment_by_tweet",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Age Group")
    ax.set_ylabel("Mean Sentiment (by tweet)")
    ax.set_title(f"Tweet Sentiment by Dominant Age Group — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Age_Boxplot_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    age_u = sentiment_census.dropna(subset=age_cols + ["mean_sentiment_by_user"]).copy()
    age_u["dominant_age"] = pd.Categorical(
        age_u[age_cols].idxmax(axis=1).map(dict(zip(age_cols, age_labels))),
        categories=age_labels, ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=age_u, x="dominant_age", y="mean_sentiment_by_user", color="#595358", ax=ax)
    sns.stripplot(data=age_u, x="dominant_age", y="mean_sentiment_by_user",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Age Group")
    ax.set_ylabel("Mean Sentiment (by user)")
    ax.set_title(f"User Sentiment by Dominant Age Group — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Age_Boxplot_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment vs Ethnicity Scatter Grid (Tweet vs User) -

    ethnicity_scatter_vars = [
        ("PNHWHITE13_17", "Proportion Non-Hispanic White", "% Non-Hispanic White"),
        ("PNHBLACK13_17", "Proportion Non-Hispanic Black", "% Non-Hispanic Black"),
        ("PHISPANIC13_17","Proportion Hispanic",           "% Hispanic"),
        ("PFBORN13_17",   "Proportion Born Outside U.S.",   "% Born Outside U.S."),
    ]

    # 1. By Tweet
    eth_scatter_t = sentiment_census.dropna(
        subset=ethnicity_cols + ["mean_sentiment_by_tweet", "total_tweets"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, xlabel, title) in zip(axes.flatten(), ethnicity_scatter_vars):
        x = eth_scatter_t[col]
        y = eth_scatter_t["mean_sentiment_by_tweet"]
        sizes = eth_scatter_t["total_tweets"] / eth_scatter_t["total_tweets"].max() * 35 + 5
        ax.scatter(x, y, s=sizes, alpha=0.4, color="#06402B")
        z_e = np.polyfit(x, y, 1)
        x_line_e = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line_e, np.polyval(z_e, x_line_e), color="crimson", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Sentiment (by tweet)")
        ax.set_title(f"Sentiment vs {title}")

    plt.suptitle(f"Tweet Sentiment (by Tweet) vs Race & Ethnicity — {args.city}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Ethnicity_Scatter_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    eth_scatter_u = sentiment_census.dropna(
        subset=ethnicity_cols + ["mean_sentiment_by_user", "total_users"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, xlabel, title) in zip(axes.flatten(), ethnicity_scatter_vars):
        x = eth_scatter_u[col]
        y = eth_scatter_u["mean_sentiment_by_user"]
        sizes = eth_scatter_u["total_users"] / eth_scatter_u["total_users"].max() * 35 + 5
        ax.scatter(x, y, s=sizes, alpha=0.4, color="#06402B")
        z_e = np.polyfit(x, y, 1)
        x_line_e = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line_e, np.polyval(z_e, x_line_e), color="crimson", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Sentiment (by user)")
        ax.set_title(f"Sentiment vs {title}")

    plt.suptitle(f"User Sentiment vs Race & Ethnicity — {args.city}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Ethnicity_Scatter_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    # - Sentiment by Dominant Ethnicity Group (Tweet vs User) -

    # 1. By Tweet
    eth_t = sentiment_census.dropna(subset=ethnicity_cols + ["mean_sentiment_by_tweet"]).copy()
    eth_t["dominant_ethnicity"] = (
        eth_t[ethnicity_cols].idxmax(axis=1).map(dict(zip(ethnicity_cols, ethnicity_labels)))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=eth_t, x="dominant_ethnicity", y="mean_sentiment_by_tweet", color="#595358", ax=ax)
    sns.stripplot(data=eth_t, x="dominant_ethnicity", y="mean_sentiment_by_tweet",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Race/Ethnicity Group")
    ax.set_ylabel("Mean Sentiment (by tweet)")
    ax.set_title(f"Tweet Sentiment by Dominant Race/Ethnicity Group — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Ethnicity_Boxplot_Tweets.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. By User
    eth_u = sentiment_census.dropna(subset=ethnicity_cols + ["mean_sentiment_by_user"]).copy()
    eth_u["dominant_ethnicity"] = (
        eth_u[ethnicity_cols].idxmax(axis=1).map(dict(zip(ethnicity_cols, ethnicity_labels)))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=eth_u, x="dominant_ethnicity", y="mean_sentiment_by_user", color="#595358", ax=ax)
    sns.stripplot(data=eth_u, x="dominant_ethnicity", y="mean_sentiment_by_user",
                  color="black", alpha=0.3, size=2, ax=ax)
    ax.set_xlabel("Dominant Race/Ethnicity Group")
    ax.set_ylabel("Mean Sentiment (by user)")
    ax.set_title(f"User Sentiment by Dominant Race/Ethnicity Group — {args.city}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{args.city}-Sentiment_Ethnicity_Boxplot_Users.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Done — figures written to {figures_dir}")