#!/usr/bin/env python3
"""
Reads JSONL tweet files from a directory, pulls out each tweet's geotagged
BOUNDING BOX (not just its centroid), computes the area of that box, and
plots the distribution of box sizes.

Why this matters: a lot of geotagged tweets don't carry a precise point --
Twitter/X attaches a "place" bounding box (e.g. the whole box for "Chicago,
IL" or a neighborhood polygon), and the SAME box gets reused across
thousands of unrelated tweets. If you only ever look at the centroid, that
reuse is invisible. This script surfaces it by:

  1. Plotting a histogram of bbox areas (log-scaled, since areas span many
     orders of magnitude -- from a few square meters to hundreds of km^2).
  2. Plotting a bar chart of the N most frequently-occurring EXACT area
     values, with counts, so you can see how many tweets share the exact
     same box.
  3. (Optional, --folium-output) Dropping the single most-common exact
     bounding box -- or the top --folium-top-n most common -- onto an
     interactive folium map centered on Chicago, so you can see exactly
     what area is being reused.
    4. Writing the printed summary to a .txt file beside the output image.

Expected JSONL line format (one JSON object per line). This script will
try, in order, to find a bounding box under any of these shapes:

    {"bbox": [min_lon, min_lat, max_lon, max_lat]}

    {"bounding_box": {"coordinates": [[[lon, lat], [lon, lat],
                                        [lon, lat], [lon, lat]]]}}

    {"place": {"bounding_box": {"coordinates": [[[lon, lat], ...]]}}}

If your data uses a different key/shape, pass --bbox-field to point at the
top-level key, or edit `extract_bbox()` below.

Usage:
    python chicago_bbox_area_analysis.py /path/to/jsonl_dir \
        --output chicago_bbox_areas.png --top-n 20

    # also drop the most common bbox on an interactive Chicago map:
    python chicago_bbox_area_analysis.py /path/to/jsonl_dir \
        --output chicago_bbox_areas.png \
        --folium-output chicago_top_bbox_map.html --folium-top-n 5
"""

import argparse
import contextlib
import io
import json
import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# Rough bounding box for Chicago (lon_min, lon_max, lat_min, lat_max).
# Used to sanity-filter boxes whose corners fall wildly outside the city,
# same idea as the bounding-box filter in the original heatmap script.
CHICAGO_BOUNDS = {
    "lon_min": -87.95,
    "lon_max": -87.50,
    "lat_min": 41.60,
    "lat_max": 42.05,
}
CHICAGO_CENTER = [41.8781, -87.6298]

# A small, high-contrast palette for drawing the top-N boxes on the folium map.
FOLIUM_PALETTE = ["#d1495b", "#2e86ab", "#f4a259", "#5b8c5a", "#8e7dbe", "#e8a0bf"]

# Meters per degree of latitude is ~constant; meters per degree of
# longitude shrinks with cos(latitude). Chicago sits at ~41.85 N.
CHICAGO_LAT_FOR_SCALE = 41.85
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LON = METERS_PER_DEG_LAT * np.cos(np.radians(CHICAGO_LAT_FOR_SCALE))


def extract_bbox(record: dict, bbox_field: str = None):
    """
    Try to pull a [min_lon, min_lat, max_lon, max_lat] box out of a record.
    Returns None if nothing usable is found.
    """
    def corners_to_extent(coords):
        # coords: iterable of [lon, lat] pairs (any nesting gets flattened
        # by the callers below before reaching here)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]

    candidates = []
    if bbox_field:
        candidates.append(record.get(bbox_field))
    candidates.append(record.get("bbox"))
    candidates.append(record.get("bounding_box"))
    if isinstance(record.get("place"), dict):
        candidates.append(record["place"].get("bounding_box"))

    for cand in candidates:
        if cand is None:
            continue

        # Shape 1: flat [min_lon, min_lat, max_lon, max_lat]
        if isinstance(cand, (list, tuple)) and len(cand) == 4 and all(
            isinstance(v, (int, float)) for v in cand
        ):
            return list(cand)

        # Shape 2: Twitter-style {"coordinates": [[[lon,lat], ...]]}
        if isinstance(cand, dict) and "coordinates" in cand:
            coords = cand["coordinates"]
            # unwrap nesting until we hit a list of [lon, lat] pairs
            while (
                isinstance(coords, list)
                and len(coords) > 0
                and isinstance(coords[0], list)
                and len(coords[0]) > 0
                and isinstance(coords[0][0], list)
            ):
                coords = coords[0]
            if isinstance(coords, list) and len(coords) >= 3:
                return corners_to_extent(coords)

        # Shape 3: bare list of [lon, lat] corner pairs
        if (
            isinstance(cand, list)
            and len(cand) >= 3
            and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in cand)
        ):
            return corners_to_extent(cand)

    return None


def load_bboxes(directory: str, pattern: str = "*.jsonl", bbox_field: str = None) -> pd.DataFrame:
    """
    Walk a directory of .jsonl files, parse each line, and pull out a
    bounding box extent. Returns a DataFrame with columns:
    min_lon, min_lat, max_lon, max_lat
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"No files matching '{pattern}' found in {directory}", file=sys.stderr)
        sys.exit(1)

    rows = []
    n_lines = 0
    n_bad = 0

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue

                extent = extract_bbox(record, bbox_field=bbox_field)
                if extent is None:
                    n_bad += 1
                    continue

                min_lon, min_lat, max_lon, max_lat = extent
                if max_lon < min_lon or max_lat < min_lat:
                    n_bad += 1
                    continue

                rows.append((min_lon, min_lat, max_lon, max_lat))

    print(f"Parsed {n_lines} lines from {len(files)} file(s); {n_bad} skipped (no usable bbox)")

    df = pd.DataFrame(rows, columns=["min_lon", "min_lat", "max_lon", "max_lat"])

    # Sanity-filter to boxes that at least overlap the Chicago area
    before = len(df)
    df = df[
        (df["max_lon"] >= CHICAGO_BOUNDS["lon_min"]) & (df["min_lon"] <= CHICAGO_BOUNDS["lon_max"]) &
        (df["max_lat"] >= CHICAGO_BOUNDS["lat_min"]) & (df["min_lat"] <= CHICAGO_BOUNDS["lat_max"])
    ]
    after = len(df)
    print(f"Kept {after}/{before} bounding boxes overlapping the Chicago area")

    if after == 0:
        print("No valid bounding boxes found inside the Chicago area.", file=sys.stderr)
        sys.exit(1)

    return df


def add_area_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add width/height (meters) and area (m^2 and km^2) columns using a flat
    equirectangular approximation scaled for Chicago's latitude. That's
    plenty accurate for a city-sized area and much simpler than reprojecting.
    """
    df = df.copy()
    df["width_deg"] = df["max_lon"] - df["min_lon"]
    df["height_deg"] = df["max_lat"] - df["min_lat"]
    df["width_m"] = df["width_deg"] * METERS_PER_DEG_LON
    df["height_m"] = df["height_deg"] * METERS_PER_DEG_LAT
    df["area_m2"] = df["width_m"] * df["height_m"]
    df["area_km2"] = df["area_m2"] / 1_000_000.0
    return df


def plot_area_distribution(df: pd.DataFrame, output_path: str, top_n: int = 20):
    """
    Two-panel figure:
      Left:  histogram of bbox areas (log-scaled x-axis) -- overall shape
             of how big these boxes tend to be.
      Right: bar chart of the `top_n` most frequently repeated EXACT area
             values, with their counts -- reveals duplicate/reused boxes.

    Returns the Counter of exact (min_lon, min_lat, max_lon, max_lat) tuples
    -> count, so callers (e.g. the folium map step) don't have to recompute it.
    """
    # Areas of zero (degenerate boxes / single points) can't go on a log
    # axis; keep them out of the histogram but note how many there were.
    nonzero = df[df["area_km2"] > 0]
    n_zero = len(df) - len(nonzero)
    if n_zero:
        print(f"Note: {n_zero} bounding box(es) had zero area (point-like) and are excluded from the log histogram")

    # Round area to a stable precision so floating-point noise doesn't
    # split what is really "the same box" into separate bins. A small
    # precision is enough for km² values in city-scale bbox analysis.
    area_rounded = df["area_km2"].round(6)
    area_counts = Counter(area_rounded)
    most_common = area_counts.most_common(top_n)

    bbox_counts = Counter(
        tuple(row)
        for row in df[["min_lon", "min_lat", "max_lon", "max_lat"]].itertuples(index=False, name=None)
    )
    most_common_bboxes = bbox_counts.most_common(top_n)
    largest_bboxes = (
        df.groupby(["min_lon", "min_lat", "max_lon", "max_lat"], as_index=False)
        .agg(area_km2=("area_km2", "first"), count=("area_km2", "size"))
        .sort_values(["area_km2", "count"], ascending=[False, False])
        .head(top_n)
    )

    fig, (ax_hist, ax_top) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: overall distribution ---
    if len(nonzero) > 0:
        bins = np.logspace(
            np.log10(nonzero["area_km2"].min()),
            np.log10(nonzero["area_km2"].max()),
            60,
        )
        ax_hist.hist(nonzero["area_km2"], bins=bins, color="#d1495b", edgecolor="white", linewidth=0.3)
        ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Bounding box area (km², log scale)")
    ax_hist.set_ylabel("Number of tweets")
    ax_hist.set_title(f"Distribution of bbox areas (n={len(df):,})")

    # --- Right: most common exact areas ---
    labels = [f"{area:,.1f} km²" for area, _ in most_common]
    counts = [count for _, count in most_common]
    y_pos = np.arange(len(most_common))

    ax_top.barh(y_pos, counts, color="#2e86ab")
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(labels, fontsize=8)
    ax_top.invert_yaxis()  # largest count at top
    ax_top.set_xlabel("Number of tweets sharing this exact area")
    ax_top.set_title(f"Top {len(most_common)} most repeated exact bbox areas")

    for y, c in zip(y_pos, counts):
        ax_top.text(c, y, f"  {c:,}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")

    # Also print a quick summary to stdout
    dup_total = sum(c for _, c in most_common if c > 1)
    print(f"\nTop {len(most_common)} exact areas account for {dup_total:,} of {len(df):,} tweets "
          f"({dup_total / len(df) * 100:.1f}%)")
    print("\narea_km2".ljust(16), "count")
    for area, count in most_common:
        print(f"{area:>15,.1f}  {count:,}")

    print("\nMost common bbox coordinates")
    print("count    min_lon      min_lat      max_lon      max_lat")
    for (min_lon, min_lat, max_lon, max_lat), count in most_common_bboxes:
        print(f"{count:>5,}   {min_lon:>10.6f}   {min_lat:>10.6f}   {max_lon:>10.6f}   {max_lat:>10.6f}")

    print("\nLargest bounding boxes")
    print("area_km2       count    min_lon      min_lat      max_lon      max_lat")
    for row in largest_bboxes.itertuples(index=False):
        print(
            f"{row.area_km2:>10,.3f}   {row.count:>7,}   "
            f"{row.min_lon:>10.6f}   {row.min_lat:>10.6f}   "
            f"{row.max_lon:>10.6f}   {row.max_lat:>10.6f}"
        )

    return bbox_counts


def plot_top_bboxes_folium(bbox_counts: Counter, output_path: str, top_n: int = 1):
    """
    Draw the top_n most frequently-occurring exact bounding boxes on an
    interactive folium map centered on Chicago. bbox_counts maps
    (min_lon, min_lat, max_lon, max_lat) -> count, e.g. as returned by
    plot_area_distribution().
    """
    if not HAS_FOLIUM:
        print(
            "folium is not installed -- skipping map output. Install it with "
            "'pip install folium' and re-run with --folium-output.",
            file=sys.stderr,
        )
        return

    most_common = bbox_counts.most_common(top_n)
    if not most_common:
        print("No bounding boxes available to plot.", file=sys.stderr)
        return

    m = folium.Map(location=CHICAGO_CENTER, zoom_start=11, tiles="cartodbpositron")
    total = sum(c for _, c in most_common)

    # Draw from least common to most common so the #1 box ends up on top.
    for rank, ((min_lon, min_lat, max_lon, max_lat), count) in enumerate(reversed(most_common)):
        true_rank = len(most_common) - rank  # 1 = most common
        color = FOLIUM_PALETTE[(true_rank - 1) % len(FOLIUM_PALETTE)]
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        folium.Rectangle(
            bounds=bounds,
            color=color,
            weight=3 if true_rank == 1 else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.25 if true_rank == 1 else 0.08,
            popup=folium.Popup(
                f"<b>Rank #{true_rank}</b><br>"
                f"Tweets in this exact box: {count:,} ({count / total * 100:.1f}% of top {len(most_common)})<br>"
                f"min_lon={min_lon:.6f}, min_lat={min_lat:.6f}<br>"
                f"max_lon={max_lon:.6f}, max_lat={max_lat:.6f}",
                max_width=320,
            ),
            tooltip=f"#{true_rank}: {count:,} tweets",
        ).add_to(m)

    # Mark the center of the #1 box for orientation.
    (top_min_lon, top_min_lat, top_max_lon, top_max_lat), top_count = most_common[0]
    center_lat = (top_min_lat + top_max_lat) / 2
    center_lon = (top_min_lon + top_max_lon) / 2
    folium.Marker(
        [center_lat, center_lon],
        icon=folium.Icon(color="red", icon="star"),
        popup=f"Most common bbox center<br>{top_count:,} tweets",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Saved interactive folium map to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze how large tweet bounding boxes are and whether many share the exact same area."
    )
    parser.add_argument("directory", help="Directory containing .jsonl tweet files")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for input files (default: *.jsonl)")
    parser.add_argument(
        "--output",
        default="chicago_bbox_areas.png",
        help="Output image path; a matching .txt summary is written beside it",
    )
    parser.add_argument("--top-n", type=int, default=20, help="How many top repeated areas to show (default: 20)")
    parser.add_argument(
        "--bbox-field",
        default=None,
        help="Top-level JSON key holding the bounding box, if it's not "
             "'bbox', 'bounding_box', or 'place.bounding_box'",
    )
    parser.add_argument(
        "--folium-output",
        default=None,
        help="If set, also save an interactive HTML map (via folium) showing the most common "
             "bounding box(es) over Chicago, e.g. chicago_top_bbox_map.html",
    )
    parser.add_argument(
        "--folium-top-n",
        type=int,
        default=1,
        help="How many of the most-common exact bounding boxes to draw on the folium map (default: 1)",
    )
    args = parser.parse_args()

    text_output = os.path.splitext(args.output)[0] + ".txt"
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        df = load_bboxes(args.directory, args.pattern, bbox_field=args.bbox_field)
        df = add_area_columns(df)
        bbox_counts = plot_area_distribution(df, args.output, top_n=args.top_n)

        if args.folium_output:
            plot_top_bboxes_folium(bbox_counts, args.folium_output, top_n=args.folium_top_n)

    report = output_buffer.getvalue()
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, end="")
    print(f"Saved text summary to {text_output}")