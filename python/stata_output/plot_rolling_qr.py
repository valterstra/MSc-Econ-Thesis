"""
plot_rolling_qr.py

Plots rolling-window QARX wind coefficients from rolling_window_qarx.do.

This replaces the old QR plotting helper with a simpler output contract:

  - one plot per quantile: tau=0.10, 0.50, 0.90
  - one line per zone (SE1-SE4)
  - ±1.96*SE ribbon
  - zero reference line

Output is saved next to the QARX results under:
  output from stata/rolling_window_results/qarx/<quantile_tag>_ar<AR_LAGS>/plots_v2/

Usage:
    python plot_rolling_qr.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ZONES = ["SE1", "SE2", "SE3", "SE4"]
QUANTILES = [0.10, 0.50, 0.90]
PIPELINE = "log"
AR_LAGS = 3
TARGET_VAR = "wind_log"

ZONE_COLORS = {
    "SE1": "#00193d",
    "SE2": "#034d8e",
    "SE3": "#0b8dd8",
    "SE4": "#6cb9e9",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _quantile_tag() -> str:
    parts = []
    for tau in QUANTILES:
        parts.append(f"q{int(round(tau * 100)):03d}")
    return "_".join(parts)


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "output from stata",
    "rolling_window_results",
    "qarx",
    f"{_quantile_tag()}_ar{AR_LAGS}",
)
PLOT_DIR = os.path.join(RESULTS_DIR, "plots_v2")


def load_zone(zone: str) -> pd.DataFrame:
    """
    Load the long-format QARX results for one zone and return the rows for the
    target wind coefficient only.
    """
    path = os.path.join(RESULTS_DIR, f"rolling_qarx_{zone}_1yr_{PIPELINE}.csv")
    raw = pd.read_csv(path)
    wind = raw[(raw["type"] == "coef") & (raw["variable"] == TARGET_VAR)].copy()
    wind = wind[["start_year", "quantile", "value", "se", "converged"]].copy()
    wind = wind.sort_values(["quantile", "start_year"]).reset_index(drop=True)
    wind.loc[wind["converged"] == 0, ["value", "se"]] = np.nan
    return wind


def plot_quantile(tau: float) -> None:
    """
    Plot one quantile across all zones and years.
    """
    tau_label = f"{int(round(tau * 100)):03d}"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for zone in ZONES:
        try:
            df = load_zone(zone)
        except FileNotFoundError:
            print(f"  {zone}: results file not found - skipping")
            continue

        sub = df[np.isclose(df["quantile"], tau)].copy()
        if sub.empty:
            print(f"  {zone}: no data for tau={tau:.2f} - skipping")
            continue

        years = sub["start_year"].values + 0.5
        b = sub["value"].values
        se = sub["se"].values
        color = ZONE_COLORS[zone]

        ax.plot(years, b, color=color, linewidth=2, marker="o", markersize=5, label=zone)
        ax.fill_between(years, b - 1.96 * se, b + 1.96 * se, color=color, alpha=0.08)

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"beta_wind(tau={tau:.2f})  [log-price per log-wind unit]", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, color="#aaaaaa")
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, f"rolling_qarx_wind_tau{tau_label}_1yr.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Plot dir:    {PLOT_DIR}")
    for tau in QUANTILES:
        plot_quantile(tau)
