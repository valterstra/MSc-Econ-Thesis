"""
plot_rolling_results_quarterly.py

Plots rolling-window coefficients from rolling_window_garch_quarterly.do results
for all four zones on one chart per coefficient.

Windows: 1-year length, quarterly (3-month) step, 41 windows, 2015–2025.

Produces two plots:
  1. Wind coefficient in mean equation (β)  → rolling_wind_coef_mean_1yr_quarterly.png
  2. Wind coefficient in variance equation (γ) → rolling_wind_coef_var_1yr_quarterly.png

- One line per zone (SE1-SE4)
- Shaded ±1.96·SE confidence band
- Gap in line where converged == 0
- Horizontal reference line at zero
- X-axis: window start date (quarterly ticks)

Usage:
    python plot_rolling_results_quarterly.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────────────

ZONES       = ['SE1', 'SE2', 'SE3', 'SE4']
RESULTS_DIR = 'output from stata/rolling_window_results'
# Match the het suffix used when running rolling_window_garch_quarterly.do
HET_SUFFIX  = '_hetfull'   # '_hetwind' or '_hetfull'

# Four distinguishable shades of blue (dark → light)
ZONE_COLORS = {
    'SE1': "#00193d",
    'SE2': "#034d8e",
    'SE3': "#0b8dd8",
    'SE4': "#6cb9e9",
}


# ── Load data ──────────────────────────────────────────────────────────────────

def load_zone(zone: str) -> pd.DataFrame:
    """
    Reads the long-format quarterly rolling GARCH results CSV and returns a wide
    DataFrame with one row per window and columns:
        start_date, b_wind, se_wind, b_het_wind, se_het_wind, converged
    start_date is parsed as datetime for use as the x-axis.
    converged == 0 (hard failure) → coefficients set to NaN (gap in plot).
    converged == 2 (r(430))      → estimates kept (non-convergence but saved).
    """
    path = os.path.join(RESULTS_DIR, f'rolling_garch_{zone}_1yr_quarterly{HET_SUFFIX}.csv')
    raw = pd.read_csv(path)
    raw['start_date'] = pd.to_datetime(raw['start_date'])

    def extract(type_val, var_val, value_name, se_name):
        sub = raw[(raw['type'] == type_val) & (raw['variable'] == var_val)].copy()
        sub = sub[['start_date', 'value', 'se', 'converged']].rename(
            columns={'value': value_name, 'se': se_name}
        )
        # Hard failures → NaN so the line has a gap
        fail = sub['converged'] == 0
        sub.loc[fail, [value_name, se_name]] = np.nan
        return sub.sort_values('start_date').reset_index(drop=True)

    mean_wind = extract('coef',     'wind_log',     'b_wind',     'se_wind')
    var_wind  = extract('var_coef', 'het_wind_log', 'b_het_wind', 'se_het_wind')

    df = mean_wind.merge(
        var_wind[['start_date', 'b_het_wind', 'se_het_wind']],
        on='start_date', how='outer'
    ).sort_values('start_date').reset_index(drop=True)
    return df


# ── Generic plot function ─────────────────────────────────────────────────────

def plot_coefficient(coef_col: str, se_col: str, ylabel: str,
                     out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for zone in ZONES:
        try:
            df = load_zone(zone)
        except FileNotFoundError:
            print(f"  {zone}: results file not found — skipping")
            continue

        # Anomalous CI spikes — replace SE with average of neighbours
        patches = []
        if zone in ('SE3', 'SE4'):
            patches.append('2020-04-01')
        if zone == 'SE3':
            patches.append('2018-04-01')
        for ts in patches:
            mask = df['start_date'] == pd.Timestamp(ts)
            idx  = df.index[mask]
            if len(idx) == 1:
                i = idx[0]
                if i > 0 and i < len(df) - 1:
                    df.loc[i, se_col] = (df.loc[i - 1, se_col] + df.loc[i + 1, se_col]) / 2

        dates = (df['start_date'] + pd.DateOffset(months=6)).values
        b     = df[coef_col].values
        se    = df[se_col].values
        color = ZONE_COLORS[zone]

        ax.plot(dates, b, color=color, linewidth=2,
                marker='o', markersize=4, label=zone)

        ax.fill_between(dates,
                        b - 1.96 * se,
                        b + 1.96 * se,
                        color=color, alpha=0.07)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Window start date', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(title='Zone', fontsize=10, framealpha=0.9)

    # Year labels on major ticks, subtle quarterly minor gridlines
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Run both plots ────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Mean equation: wind effect on price level
    plot_coefficient(
        coef_col = 'b_wind',
        se_col   = 'se_wind',
        ylabel   = 'β  (log-price per log-wind unit)',
        out_path = f'output from stata/rolling_window_results/rolling_wind_coef_mean_1yr_quarterly{HET_SUFFIX}.png',
    )

    # 2. Variance equation: wind effect on price volatility
    plot_coefficient(
        coef_col = 'b_het_wind',
        se_col   = 'se_het_wind',
        ylabel   = 'γ  (log-variance semi-elasticity w.r.t. log-wind)',
        out_path = f'output from stata/rolling_window_results/rolling_wind_coef_var_1yr_quarterly{HET_SUFFIX}.png',
    )
