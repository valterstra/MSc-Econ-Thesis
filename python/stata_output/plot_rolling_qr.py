"""
plot_rolling_qr.py

Plots rolling-window wind coefficients from rolling_window_qr.do results.

Produces three sets of plots (53 PNGs total):

  Set 1 — one plot per quantile (5 plots):
    Each shows one line per zone (SE1–SE4) with ±1.96·SE confidence band.
        rolling_qr_wind_tau010_1yr.png  ...  rolling_qr_wind_tau090_1yr.png

  Set 2 — one plot per zone (4 plots):
    Each shows one line per quantile (τ = 0.10 … 0.90) with ±1.96·SE band.
        rolling_qr_wind_SE1_1yr.png  ...  rolling_qr_wind_SE4_1yr.png

  Set 3 — quantile process plots (44 plots, one per zone × year):
    X-axis: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    Y-axis: β_wind(τ) with ±1.96·SE band
    Saved to: quantile_process/{ZONE}/qp_{ZONE}_{YEAR}.png

Usage:
    python plot_rolling_qr.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ──────────────────────────────────────────────────────────────

ZONES       = ['SE1', 'SE2', 'SE3', 'SE4']
QUANTILES   = [0.10, 0.25, 0.50, 0.75, 0.90]
RESULTS_DIR = 'output from stata/rolling_window_results/qr'
PIPELINE    = 'log'

# Four distinguishable shades of blue (dark → light), matching plot_rolling_results.py
ZONE_COLORS = {
    'SE1': "#00193d",
    'SE2': "#034d8e",
    'SE3': "#0b8dd8",
    'SE4': "#6cb9e9",
}


# ── Load data ──────────────────────────────────────────────────────────────────

def load_zone(zone: str) -> pd.DataFrame:
    """
    Reads the long-format rolling QR results CSV for one zone and returns
    the wind_log coefficients across all quantiles and windows.

    Returns a DataFrame with columns:
        start_year, quantile, value, se
    """
    path = os.path.join(RESULTS_DIR, f'rolling_qr_{zone}_1yr_{PIPELINE}.csv')
    raw = pd.read_csv(path)

    wind = raw[(raw['type'] == 'coef') & (raw['variable'] == 'wind_log')].copy()
    wind = wind[['start_year', 'quantile', 'value', 'se']].copy()
    wind = wind.sort_values(['quantile', 'start_year']).reset_index(drop=True)
    return wind


# ── Plot function ──────────────────────────────────────────────────────────────

def plot_quantile(tau: float) -> None:
    """
    One plot for a single quantile τ: four zone lines, ±1.96·SE band, zero line.
    """
    tau_label = f'{int(round(tau * 100)):02d}'   # e.g. "10", "25", "50"
    tau_pct   = f'{int(round(tau * 100))}th'      # e.g. "10th"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for zone in ZONES:
        try:
            df = load_zone(zone)
        except FileNotFoundError:
            print(f"  {zone}: results file not found — skipping")
            continue

        sub = df[np.isclose(df['quantile'], tau)].copy()
        if sub.empty:
            print(f"  {zone}: no data for τ={tau} — skipping")
            continue

        years = sub['start_year'].values + 0.5
        b     = sub['value'].values
        se    = sub['se'].values
        color = ZONE_COLORS[zone]

        ax.plot(years, b, color=color, linewidth=2,
                marker='o', markersize=5, label=zone)
        ax.fill_between(years,
                        b - 1.96 * se,
                        b + 1.96 * se,
                        color=color, alpha=0.07)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(f'β_wind(τ={tau:.2f})  [log-price per log-wind unit]', fontsize=11)
    ax.legend(title='Zone', fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'rolling_qr_wind_tau{tau_label}_1yr.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Colors for quantiles (dark → light, low → high τ) ─────────────────────────

# Five shades of blue (dark → light), low → high τ
QUANTILE_COLORS = {
    0.10: "#00193d",
    0.25: "#034d8e",
    0.50: "#0b8dd8",
    0.75: "#6cb9e9",
    0.90: "#888888",
}
QUANTILE_BAND_ALPHA = 0.10


# ── Plot per zone ──────────────────────────────────────────────────────────────

def plot_zone(zone: str) -> None:
    """
    One plot for a single zone: five quantile lines, ±1.96·SE band, zero line.
    """
    try:
        df = load_zone(zone)
    except FileNotFoundError:
        print(f"  {zone}: results file not found — skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for tau in QUANTILES:
        sub = df[np.isclose(df['quantile'], tau)].copy()
        if sub.empty:
            continue

        years = sub['start_year'].values + 0.5
        b     = sub['value'].values
        se    = sub['se'].values
        color = QUANTILE_COLORS[tau]

        ax.plot(years, b, color=color, linewidth=2,
                marker='o', markersize=5, label=f'τ = {tau:.2f}')
        ax.fill_between(years,
                        b - 1.96 * se,
                        b + 1.96 * se,
                        color=color, alpha=QUANTILE_BAND_ALPHA)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('β_wind(τ)  [log-price per log-wind unit]', fontsize=11)
    ax.legend(title='Quantile', fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'rolling_qr_wind_{zone}_1yr.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Quantile process plots (per zone × year) ───────────────────────────────────

YEARS = list(range(2015, 2026))


def compute_global_ylim(pad: float = 0.10):
    """
    Scans all zone rolling-QR results and returns (ymin, ymax) based on
    the range of β_wind point estimates (not CIs) plus fractional padding.
    CIs that exceed these limits will simply be clipped.
    """
    all_b = []
    for zone in ZONES:
        try:
            df = load_zone(zone)
        except FileNotFoundError:
            continue
        df = df[df['quantile'].apply(lambda q: any(np.isclose(q, t) for t in QUANTILES))]
        all_b.append(df['value'].values)
    combined = np.concatenate(all_b)
    rng  = combined.max() - combined.min()
    ymin = combined.min() - pad * rng
    ymax = combined.max() + pad * rng
    return ymin, ymax


def plot_quantile_process(zone: str, year: int,
                          ylim: tuple | None = None) -> None:
    """
    Quantile process plot for one zone × year.

    X-axis: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    Y-axis: β_wind(τ) with ±1.96·SE ribbon and zero reference line.

    ylim: optional (ymin, ymax) tuple for a fixed y-axis across plots.
          CIs that exceed the limits are clipped (clip_on=True by default).
    """
    try:
        df = load_zone(zone)
    except FileNotFoundError:
        print(f"  {zone}: results file not found — skipping")
        return

    sub = df[df['start_year'] == year].copy()
    sub = sub[sub['quantile'].apply(lambda q: any(np.isclose(q, t) for t in QUANTILES))]
    if sub.empty:
        print(f"  {zone} {year}: no data — skipping")
        return

    sub = sub.sort_values('quantile').reset_index(drop=True)
    taus = sub['quantile'].values
    b    = sub['value'].values
    se   = sub['se'].values

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    color = ZONE_COLORS[zone]
    ax.plot(taus, b, color=color, linewidth=2, marker='o', markersize=6, zorder=3)
    ax.fill_between(taus, b - 1.96 * se, b + 1.96 * se, color=color, alpha=0.15, zorder=2)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Quantile (τ)', fontsize=11)
    ax.set_ylabel('β_wind(τ)  [log-price per log-wind unit]', fontsize=11)
    ax.set_xticks(QUANTILES)
    ax.set_xticklabels([f'{t:.2f}' for t in QUANTILES])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    out_dir  = os.path.join(RESULTS_DIR, 'quantile_process', zone)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'qp_{zone}_{year}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Full-period quantile process plots (one per zone, 2015–2025) ───────────────

STATIC_DIR    = 'output from stata/qr_results'
STATIC_STUB   = '2015-01-01_2025-12-31_log'


def load_static(zone: str) -> pd.DataFrame:
    """
    Reads the full-period QR results CSV for one zone and returns
    the wind_log coefficients across all quantiles.

    Returns a DataFrame with columns:
        quantile, value, se
    """
    path = os.path.join(STATIC_DIR, f'qr_results_{zone}_{STATIC_STUB}.csv')
    raw  = pd.read_csv(path)

    wind = raw[(raw['type'] == 'coef') & (raw['variable'] == 'wind_log')].copy()
    wind = wind[['quantile', 'value', 'se']].copy()
    wind = wind.sort_values('quantile').reset_index(drop=True)
    return wind


def plot_quantile_process_static(zone: str) -> None:
    """
    Full-period quantile process plot for one zone (2015–2025).

    X-axis: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    Y-axis: β_wind(τ) with ±1.96·SE ribbon and zero reference line.
    """
    try:
        df = load_static(zone)
    except FileNotFoundError:
        print(f"  {zone}: static results file not found — skipping")
        return

    df   = df[df['quantile'].apply(lambda q: any(np.isclose(q, t) for t in QUANTILES))]
    taus  = df['quantile'].values
    b     = df['value'].values
    se    = df['se'].values
    color = ZONE_COLORS[zone]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(taus, b, color=color, linewidth=2, marker='o', markersize=6, zorder=3)
    ax.fill_between(taus, b - 1.96 * se, b + 1.96 * se, color=color, alpha=0.15, zorder=2)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Quantile (τ)', fontsize=11)
    ax.set_ylabel('β_wind(τ)  [log-price per log-wind unit]', fontsize=11)
    ax.set_xticks(QUANTILES)
    ax.set_xticklabels([f'{t:.2f}' for t in QUANTILES])
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    out_dir  = os.path.join(RESULTS_DIR, 'quantile_process', 'full_period')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'qp_{zone}_full_period.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Combined full-period quantile process plot (all zones) ────────────────────

def plot_quantile_process_static_combined() -> None:
    """
    Full-period quantile process plot with all four zones on one axes.

    X-axis: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    Y-axis: β_wind(τ) with ±1.96·SE ribbon per zone and zero reference line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for zone in ZONES:
        try:
            df = load_static(zone)
        except FileNotFoundError:
            print(f"  {zone}: static results file not found — skipping")
            continue

        df    = df[df['quantile'].apply(lambda q: any(np.isclose(q, t) for t in QUANTILES))]
        taus  = df['quantile'].values
        b     = df['value'].values
        se    = df['se'].values
        color = ZONE_COLORS[zone]

        ax.plot(taus, b, color=color, linewidth=2, marker='o', markersize=6,
                label=zone, zorder=3)
        ax.fill_between(taus, b - 1.96 * se, b + 1.96 * se,
                        color=color, alpha=0.10, zorder=2)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Quantile (τ)', fontsize=11)
    ax.set_ylabel('β_wind(τ)  [log-price per log-wind unit]', fontsize=11)
    ax.legend(title='Zone', fontsize=10, framealpha=0.9)
    ax.set_xticks(QUANTILES)
    ax.set_xticklabels([f'{t:.2f}' for t in QUANTILES])
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    out_dir  = os.path.join(RESULTS_DIR, 'quantile_process', 'full_period')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'qp_all_zones_full_period.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Terminal coefficient table (zone × year × quantile) ───────────────────────

def print_qr_table(zone: str) -> None:
    """
    Prints a coefficient + stars table to the terminal for one zone.

    Rows:    2015 … 2025 (rolling 1-year windows) + Overall (full period)
    Columns: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    Values:  β_wind(τ) with significance stars
    """
    # Load rolling results with stars
    roll_path = os.path.join(RESULTS_DIR, f'rolling_qr_{zone}_1yr_{PIPELINE}.csv')
    roll_raw  = pd.read_csv(roll_path)
    roll_wind = roll_raw[
        (roll_raw['type'] == 'coef') & (roll_raw['variable'] == 'wind_log')
    ][['start_year', 'quantile', 'value', 'stars']].copy()
    roll_wind['stars'] = roll_wind['stars'].fillna('')

    # Load static full-period results with stars
    stat_path = os.path.join(STATIC_DIR, f'qr_results_{zone}_{STATIC_STUB}.csv')
    stat_raw  = pd.read_csv(stat_path)
    stat_wind = stat_raw[
        (stat_raw['type'] == 'coef') & (stat_raw['variable'] == 'wind_log')
    ][['quantile', 'value', 'stars']].copy()
    stat_wind['stars'] = stat_wind['stars'].fillna('')

    tau_labels = [f'τ={t:.2f}' for t in QUANTILES]
    col_w = 14

    print(f'\n{"="*75}')
    print(f'  {zone}  —  β_wind(τ)     *** p<0.01   ** p<0.05   * p<0.10')
    print(f'{"="*75}')
    print(f'{"Year":<10}' + ''.join(f'{lbl:>{col_w}}' for lbl in tau_labels))
    print(f'{"-"*75}')

    for year in YEARS:
        row = f'{year:<10}'
        for tau in QUANTILES:
            sub = roll_wind[
                (roll_wind['start_year'] == year) &
                roll_wind['quantile'].apply(lambda q: np.isclose(q, tau))
            ]
            if sub.empty:
                row += f'{"—":>{col_w}}'
            else:
                coef = sub['value'].values[0]
                s    = sub['stars'].values[0]
                row += f'{coef:.4f}{s:>{col_w - 7}}'
        print(row)

    print(f'{"-"*75}')

    # Overall row
    row = f'{"Overall":<10}'
    for tau in QUANTILES:
        sub = stat_wind[stat_wind['quantile'].apply(lambda q: np.isclose(q, tau))]
        if sub.empty:
            row += f'{"—":>{col_w}}'
        else:
            coef = sub['value'].values[0]
            s    = sub['stars'].values[0]
            row += f'{coef:.4f}{s:>{col_w - 7}}'
    print(row)
    print(f'{"="*75}')


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Set 1: one plot per quantile")
    for tau in QUANTILES:
        plot_quantile(tau)

    print("\nSet 2: one plot per zone")
    for zone in ZONES:
        plot_zone(zone)

    print("\nSet 3: quantile process plots (zone × year)")
    global_ylim = compute_global_ylim()
    print(f"  Global y-axis: [{global_ylim[0]:.4f}, {global_ylim[1]:.4f}]")
    for zone in ZONES:
        for year in YEARS:
            plot_quantile_process(zone, year, ylim=global_ylim)

    print("\nSet 4: full-period quantile process plots (one per zone)")
    for zone in ZONES:
        plot_quantile_process_static(zone)
    plot_quantile_process_static_combined()

    print("\nCoefficient tables")
    for zone in ZONES:
        print_qr_table(zone)
