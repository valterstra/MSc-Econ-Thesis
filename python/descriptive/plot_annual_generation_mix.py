"""
plot_annual_generation_mix.py

Build annual generation-mix figures for Swedish bidding zones using:
- official annual data for 2015-2024
- annualized monthly data for 2025

Outputs:
- one stacked bar chart per bidding zone
- one combined line chart of wind share over time

Usage:
    python plot_annual_generation_mix.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANNUAL_DATA_PATH = ROOT / "data" / "annual production" / "sweden_annual_generation_by_source_2015_2024.csv"
MONTHLY_DATA_PATH = ROOT / "data" / "annual production" / "sweden_monthly_production_usage_by_zone_2021_2026.csv"
OUTPUT_DIR = ROOT / "results" / "descriptive_overview"

ZONES = ["SE1", "SE2", "SE3", "SE4"]
ZONE_COLORS = {
    "SE1": "#163a70",
    "SE2": "#2f5aa8",
    "SE3": "#5c92cf",
    "SE4": "#9fc0e6",
}

ANNUAL_CATEGORY_MAP = {
    "vattenkraft": "Hydro",
    "vindkraft": "Wind",
    "solkraft": "Solar",
    "kärnkraft": "Nuclear",
    "konventionell värmekraft (industriellt mottryck, kraftvärme, kondenskraft, gasturbiner, annan)": "Thermal",
}

MONTHLY_CATEGORY_MAP = {
    "hydro power (including pump power), net": "Hydro",
    "wind power": "Wind",
    "solar power (on-grid)": "Solar",
    "nuclear power (condensing), net": "Nuclear",
    "conventional thermal power, net": "Thermal",
}

CATEGORY_ORDER = ["Nuclear", "Hydro", "Wind", "Solar", "Thermal"]
CATEGORY_COLORS = {
    "Nuclear": "#0f172a",
    "Hydro": "#1d4e89",
    "Wind": "#4f8cc9",
    "Solar": "#f4b942",
    "Thermal": "#b8bcc2",
}


def load_annual_2015_2024() -> pd.DataFrame:
    df = pd.read_csv(ANNUAL_DATA_PATH, encoding="cp1252", skiprows=2)
    df = df.rename(columns={df.columns[0]: "source", df.columns[1]: "year"})
    df = df[df["source"].isin(ANNUAL_CATEGORY_MAP)].copy()
    df["source"] = df["source"].map(ANNUAL_CATEGORY_MAP)

    df = df.melt(
        id_vars=["source", "year"],
        value_vars=ZONES,
        var_name="zone",
        value_name="gwh",
    )
    df["gwh"] = pd.to_numeric(df["gwh"].replace("..", 0), errors="coerce").fillna(0)
    df["year"] = df["year"].astype(int)
    return df[["year", "zone", "source", "gwh"]]


def load_monthly_2025() -> pd.DataFrame:
    df = pd.read_csv(MONTHLY_DATA_PATH, encoding="utf-8-sig", skiprows=2)
    df = df[df["Production and usage"].isin(MONTHLY_CATEGORY_MAP)].copy()
    df["source"] = df["Production and usage"].map(MONTHLY_CATEGORY_MAP)

    period_cols = [
        col
        for col in df.columns
        if len(str(col)) == 7 and str(col)[:4].isdigit() and str(col)[4] == "M"
    ]

    df = df.melt(
        id_vars=["source", "bidding zone"],
        value_vars=period_cols,
        var_name="period",
        value_name="gwh",
    )
    df["gwh"] = pd.to_numeric(df["gwh"], errors="coerce").fillna(0)
    df["year"] = df["period"].str[:4].astype(int)
    df = df[df["year"] == 2025].copy()
    df = df.rename(columns={"bidding zone": "zone"})

    df = (
        df.groupby(["year", "zone", "source"], as_index=False)["gwh"]
        .sum()
        .sort_values(["zone", "source"])
    )
    return df[["year", "zone", "source", "gwh"]]


def load_combined_generation_mix() -> pd.DataFrame:
    annual_df = load_annual_2015_2024()
    monthly_2025_df = load_monthly_2025()
    df = pd.concat([annual_df, monthly_2025_df], ignore_index=True)
    df["twh"] = df["gwh"] / 1000.0
    return df.sort_values(["zone", "year", "source"]).reset_index(drop=True)


def plot_zone_generation_mix(df: pd.DataFrame, zone: str) -> Path:
    zone_df = df[df["zone"] == zone].copy()
    years = sorted(zone_df["year"].unique())
    pivot = (
        zone_df.pivot_table(index="year", columns="source", values="twh", aggfunc="sum")
        .reindex(index=years, columns=CATEGORY_ORDER, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = pd.Series(0.0, index=pivot.index)
    handles = []

    for source in CATEGORY_ORDER:
        bars = ax.bar(
            pivot.index.astype(str),
            pivot[source],
            bottom=bottom,
            color=CATEGORY_COLORS[source],
            edgecolor="white",
            linewidth=0.45,
            width=0.72,
            label=source,
        )
        bottom = bottom + pivot[source]
        handles.append(bars[0])

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Annual Production (TWh)", fontsize=10)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles,
        CATEGORY_ORDER,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        fontsize=12,
        borderpad=0.7,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = OUTPUT_DIR / f"annual_generation_mix_{zone}_2015_2025.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_wind_share(df: pd.DataFrame) -> Path:
    totals = (
        df.groupby(["zone", "year"], as_index=False)["gwh"]
        .sum()
        .rename(columns={"gwh": "total_gwh"})
    )
    wind = (
        df[df["source"] == "Wind"]
        .groupby(["zone", "year"], as_index=False)["gwh"]
        .sum()
        .rename(columns={"gwh": "wind_gwh"})
    )
    share_df = totals.merge(wind, on=["zone", "year"], how="left")
    share_df["wind_gwh"] = share_df["wind_gwh"].fillna(0)
    share_df["wind_share_pct"] = 100 * share_df["wind_gwh"] / share_df["total_gwh"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for zone in ZONES:
        zone_df = share_df[share_df["zone"] == zone].sort_values("year")
        ax.plot(
            zone_df["year"],
            zone_df["wind_share_pct"],
            color=ZONE_COLORS[zone],
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=4.5,
            label=zone,
        )

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Wind Share of Annual Production (%)", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(sorted(share_df["year"].unique()))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        fontsize=12,
        borderpad=0.7,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = OUTPUT_DIR / "wind_share_by_zone_2015_2025.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_combined_generation_mix()

    saved_paths = []
    for zone in ZONES:
        saved_paths.append(plot_zone_generation_mix(df, zone))
    saved_paths.append(plot_wind_share(df))

    for path in saved_paths:
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
