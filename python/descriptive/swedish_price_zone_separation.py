"""
swedish_price_zone_separation.py

Print yearly Swedish price-zone separation metrics for 2015-2025 using the
hourly spot-price master file.

Output:
- One terminal table with yearly values.
- Connection metrics are reported for the three adjacent Swedish links:
  SE1-SE2, SE2-SE3, SE3-SE4.
- Percentage columns show the share of total yearly hours with a price
  difference on that link.
- Percentage-difference columns are computed from yearly average zone prices:
  ((mean right zone price - mean left zone price) / mean left zone price) * 100.

Usage:
    python python/descriptive/swedish_price_zone_separation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "master data files/2015-2025/Spot_Prices_2015_2025.xlsx"
PRICE_COLS = {
    "SE1": "SE1_Price (EUR)",
    "SE2": "SE2_Price (EUR)",
    "SE3": "SE3_Price (EUR)",
    "SE4": "SE4_Price (EUR)",
}
CONNECTIONS = [("SE1", "SE2"), ("SE2", "SE3"), ("SE3", "SE4")]


def load_data() -> pd.DataFrame:
    usecols = ["Timestamp", *PRICE_COLS.values()]
    df = pd.read_excel(DATA_PATH, usecols=usecols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Year"] = df["Timestamp"].dt.year
    df["duration_hours"] = 1.0
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in df.groupby("Year", sort=True):
        year_row = {
            "Year": year,
        }

        for left, right in CONNECTIONS:
            left_col = PRICE_COLS[left]
            right_col = PRICE_COLS[right]
            gap = (group[left_col] - group[right_col]).abs()
            separated = gap > 0
            link_name = f"{left}_{right}"
            left_mean = np.average(group[left_col], weights=group["duration_hours"])
            right_mean = np.average(group[right_col], weights=group["duration_hours"])

            year_row[f"{link_name}_Hours"] = int(separated.sum())
            year_row[f"{link_name}_Pct"] = 100 * separated.mean()
            year_row[f"{link_name}_AvgPctDiff"] = (
                ((right_mean - left_mean) / left_mean) * 100 if left_mean != 0 else 0.0
            )

        rows.append(year_row)

    result = pd.DataFrame(rows)

    ordered_cols = [
        "Year",
    ]
    for left, right in CONNECTIONS:
        link_name = f"{left}_{right}"
        ordered_cols.extend(
            [
                f"{link_name}_Hours",
                f"{link_name}_Pct",
                f"{link_name}_AvgPctDiff",
            ]
        )

    return result[ordered_cols]


def format_table(result: pd.DataFrame) -> pd.DataFrame:
    formatted = result.copy()

    for left, right in CONNECTIONS:
        link_name = f"{left}_{right}"
        formatted[f"{link_name}_Pct"] = formatted[f"{link_name}_Pct"].map(lambda x: f"{x:.2f}")
        formatted[f"{link_name}_AvgPctDiff"] = formatted[f"{link_name}_AvgPctDiff"].map(
            lambda x: "0.00" if pd.isna(x) else f"{x:.2f}"
        )

    return formatted


def print_compact_table(result: pd.DataFrame) -> None:
    labels = [f"{left}-{right}" for left, right in CONNECTIONS]
    pct_cols = [f"{left}_{right}_Pct" for left, right in CONNECTIONS]
    diff_cols = [f"{left}_{right}_AvgPctDiff" for left, right in CONNECTIONS]

    widths = {"Year": 6, "link": 10}
    group_width = widths["link"] * len(labels) + (len(labels) - 1)
    total_width = widths["Year"] + 1 + group_width + 3 + group_width

    header_1 = (
        f"{'Year':<{widths['Year']}} "
        f"{'% hours with price difference':^{group_width}}   "
        f"{'Average price difference (%)':^{group_width}}"
    )
    header_2 = (
        f"{'':<{widths['Year']}} "
        + " ".join(f"{label:>{widths['link']}}" for label in labels)
        + "   "
        + " ".join(f"{label:>{widths['link']}}" for label in labels)
    )

    print()
    print("Swedish price-zone separation by year (2015-2025)")
    print(header_1)
    print(header_2)
    print("-" * total_width)

    for _, row in result.iterrows():
        pct_values = " ".join(
            f"{row[col]:>{widths['link']}.2f}" for col in pct_cols
        )
        diff_values = " ".join(
            f"{row[col]:>{widths['link']}.2f}" if pd.notna(row[col]) else f"{0:>{widths['link']}.2f}"
            for col in diff_cols
        )
        print(f"{int(row['Year']):<{widths['Year']}} {pct_values}   {diff_values}")
    print()


def main() -> None:
    df = load_data()
    result = build_table(df)
    print_compact_table(result)


if __name__ == "__main__":
    main()
