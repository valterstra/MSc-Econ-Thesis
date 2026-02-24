"""
outlier_comparison.py  –  Full-Spectrum vs. Rolling-Window Outlier Detection
════════════════════════════════════════════════════════════════════════════════

Compares how Fredriksson (2016) and Gianfreda (2010) outlier detection behave
when applied on the full price series vs. locally within overlapping rolling
windows.

Produces two figures (one per method), each with 4 panels:

  Panel 1 – Full-spectrum: points outside global thresholds, thresholds shown
  Panel 2 – Rolling-window density: how many windows flagged each point (heat)
  Panel 3 – Overlap/difference map:
               ● Purple = flagged by BOTH full-spectrum AND rolling window
               ● Red    = flagged by full-spectrum ONLY
               ● Blue   = flagged by rolling window ONLY
  Panel 4 – Window consistency: fraction of containing windows that flag each
             point  (green = always an outlier, red = marginal / context-dependent)
             Points flagged in ≥50% of windows are circled as "robust" outliers.

Usage:
    python outlier_comparison.py          (from MSc-Econ-Thesis root)

Requires data at the paths configured in PATHS below (same layout as main.py).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')

# ── add package root to sys.path ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from full_regression.data_loading import load_data
from full_regression.preprocessing import (
    handle_negative_prices,
    apply_log_transform,
    deseasonalize_logged_variables,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  edit here only
# ══════════════════════════════════════════════════════════════════════════════
ACTIVE_ZONE      = 'SE1'
PATHS = {
    'combined':    f'master data files/2015-2025/Combined_{ACTIVE_ZONE}_Data_2015_2025.xlsx',
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}
START_DATE       = '2015-01-01'
END_DATE         = '2017-12-31'
NEG_PRICE_METHOD = 'shift'          # 'shift' or 'clip'

# Rolling window: overlapping so each point appears in multiple windows.
# A shorter step → more overlap → richer consistency information.
WINDOW_HOURS     = 365 * 24        # 1-year window
STEP_HOURS       = 30  * 24        # ~1-month step  (≈12× overlap per point)

OUTPUT_DIR       = 'results/outlier_comparison'
# ══════════════════════════════════════════════════════════════════════════════


# ─── DETECTION HELPERS  (detect only – no replacement) ───────────────────────

def detect_fredriksson(series: pd.Series) -> pd.Series:
    """
    Boolean mask: True where point exceeds Fredriksson (2016) thresholds.
    Uses the global mean/std of `series` (+6σ upper, −3.7σ lower).
    """
    mu    = series.mean()
    sigma = series.std()
    return (series > mu + 6.0 * sigma) | (series < mu - 3.7 * sigma)


def detect_gianfreda(series: pd.Series) -> pd.Series:
    """
    Boolean mask: True where point exceeds Gianfreda (2010) thresholds.
    Computes ±3σ separately for each weekday using `series.index`.
    """
    mask = pd.Series(False, index=series.index)
    dow  = series.index.dayofweek
    for day in range(7):
        sel = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma = day_data.mean(), day_data.std()
        mask[sel] = (series[sel] > mu + 3.0 * sigma) | (series[sel] < mu - 3.0 * sigma)
    return mask


# ─── ROLLING-WINDOW OUTLIER COUNTS ───────────────────────────────────────────

def rolling_outlier_counts(series: pd.Series, method: str) -> pd.DataFrame:
    """
    Slide an overlapping window over `series` and for each data point track:
      - seen    : number of windows that contained this point
      - flagged : number of those windows that marked it as an outlier

    Returns a DataFrame with columns ['seen', 'flagged'] aligned to series.index.
    """
    n       = len(series)
    seen    = np.zeros(n, dtype=np.int32)
    flagged = np.zeros(n, dtype=np.int32)

    starts = range(0, max(1, n - WINDOW_HOURS + 1), STEP_HOURS)
    n_windows = len(list(starts))

    for i, start in enumerate(range(0, max(1, n - WINDOW_HOURS + 1), STEP_HOURS)):
        end    = min(start + WINDOW_HOURS, n)
        window = series.iloc[start:end]

        if method == 'fredriksson':
            out_mask = detect_fredriksson(window)
        else:
            out_mask = detect_gianfreda(window)

        seen[start:end]    += 1
        flagged[start:end] += out_mask.values.astype(np.int32)

        if (i + 1) % max(1, n_windows // 5) == 0:
            print(f"      window {i+1}/{n_windows}  ({start} → {end})")

    return pd.DataFrame({'seen': seen, 'flagged': flagged}, index=series.index)


# ─── FIGURE BUILDER ───────────────────────────────────────────────────────────

METHOD_LABELS = {
    'fredriksson': 'Fredriksson (2016)  [+6σ / −3.7σ, global]',
    'gianfreda':   'Gianfreda (2010)  [±3σ per weekday, global]',
}

# Colour palette (colour-blind friendly)
C_PRICE       = '#4477AA'
C_BOTH        = '#7B2D8B'   # purple
C_FULL_ONLY   = '#D62728'   # red
C_ROLLING_ONLY = '#1F77B4'  # blue
C_THRESHOLD   = '#AA3333'


def make_figure(series: pd.Series, method: str) -> None:
    label = METHOD_LABELS[method]
    print(f"\n  ── {label} ──")

    # ── global detection ─────────────────────────────────────────────────────
    if method == 'fredriksson':
        full_mask = detect_fredriksson(series)
    else:
        full_mask = detect_gianfreda(series)

    # ── rolling window detection ──────────────────────────────────────────────
    print(f"    Running rolling-window detection "
          f"(window={WINDOW_HOURS//24}d, step={STEP_HOURS//24}d) …")
    counts = rolling_outlier_counts(series, method)

    rolling_mask = counts['flagged'] > 0          # flagged in at least one window

    with np.errstate(divide='ignore', invalid='ignore'):
        consistency = np.where(
            counts['seen'] > 0,
            counts['flagged'] / counts['seen'],
            0.0,
        )
    consistency = pd.Series(consistency, index=series.index)

    # ── overlap classification ────────────────────────────────────────────────
    both         = full_mask  &  rolling_mask
    full_only    = full_mask  & ~rolling_mask
    rolling_only = ~full_mask &  rolling_mask

    print(f"    Full-spectrum  : {full_mask.sum():5d} outliers")
    print(f"    Rolling (any)  : {rolling_mask.sum():5d} points flagged in ≥1 window")
    print(f"    Both           : {both.sum():5d}")
    print(f"    Full-only      : {full_only.sum():5d}")
    print(f"    Rolling-only   : {rolling_only.sum():5d}")

    n_windows_approx = len(range(0, max(1, len(series) - WINDOW_HOURS + 1), STEP_HOURS))

    # ── layout ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=(18, 24), sharex=True,
        gridspec_kw={'hspace': 0.08, 'top': 0.965, 'bottom': 0.04,
                     'left': 0.055, 'right': 0.93},
    )
    fig.suptitle(
        f'Outlier Detection Comparison — {label}\n'
        f'Zone: {ACTIVE_ZONE}  |  {START_DATE} – {END_DATE}  |  '
        f'{len(series):,} hourly observations',
        fontsize=13, fontweight='bold',
    )

    scatter_kw = dict(zorder=5, linewidths=0, s=10)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 – Full-spectrum outliers
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.70, label='Price_Log_Deseas.')

    if full_mask.sum() > 0:
        ax.scatter(series.index[full_mask], series.values[full_mask],
                   color=C_FULL_ONLY, label=f'Outliers  (n={full_mask.sum()})',
                   **scatter_kw)

    mu, sigma = series.mean(), series.std()
    if method == 'fredriksson':
        ax.axhline(mu + 6.0 * sigma,  color=C_THRESHOLD, ls='--', lw=0.9, alpha=0.8,
                   label=f'+6σ = {mu+6*sigma:.2f}')
        ax.axhline(mu - 3.7 * sigma, color=C_THRESHOLD, ls=':',  lw=0.9, alpha=0.8,
                   label=f'−3.7σ = {mu-3.7*sigma:.2f}')
    else:
        ax.axhline(mu + 3.0 * sigma,  color=C_THRESHOLD, ls='--', lw=0.9, alpha=0.8,
                   label=f'+3σ (overall) = {mu+3*sigma:.2f}')
        ax.axhline(mu - 3.0 * sigma, color=C_THRESHOLD, ls=':',  lw=0.9, alpha=0.8,
                   label=f'−3σ (overall) = {mu-3*sigma:.2f}')

    ax.set_title('Panel 1 — Full-Spectrum Outliers  (thresholds computed on entire series)',
                 fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price')
    ax.legend(fontsize=8, ncol=4, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.20)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 – Rolling-window flagging density (heat colouring)
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.45)

    n_max = int(counts['flagged'].max())
    if rolling_mask.sum() > 0 and n_max > 0:
        cmap1 = plt.cm.YlOrRd
        norm1 = Normalize(vmin=1, vmax=max(n_max, 2))
        sc1   = ax.scatter(
            series.index[rolling_mask],
            series.values[rolling_mask],
            c=counts['flagged'][rolling_mask],
            cmap=cmap1, norm=norm1,
            **scatter_kw,
        )
        cb1 = fig.colorbar(sc1, ax=ax, orientation='vertical',
                           pad=0.005, fraction=0.018,
                           label='# windows that\nflagged this point')
        cb1.ax.tick_params(labelsize=7)

    ax.set_title(
        f'Panel 2 — Rolling-Window Flagging Density  '
        f'(window={WINDOW_HOURS//24}d  |  step={STEP_HOURS//24}d  |  '
        f'≈{n_windows_approx} windows)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )
    ax.set_ylabel('Deseas. Log Price')
    ax.grid(True, alpha=0.20)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3 – Overlap / difference map
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.45)

    layer_order = [
        (both,         C_BOTH,         f'Both (n={both.sum()})'),
        (full_only,    C_FULL_ONLY,    f'Full-spectrum only (n={full_only.sum()})'),
        (rolling_only, C_ROLLING_ONLY, f'Rolling window only (n={rolling_only.sum()})'),
    ]
    for mask, colour, lbl in layer_order:
        if mask.sum() > 0:
            ax.scatter(series.index[mask], series.values[mask],
                       color=colour, label=lbl, **scatter_kw)

    ax.set_title('Panel 3 — Overlap / Difference Map',
                 fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price')
    ax.legend(fontsize=9, ncol=3, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.20)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 4 – Window consistency
    # For every point ever flagged by a rolling window, colour it by the fraction
    # of windows that contained AND flagged it.
    #   green  → flagged in almost every window  (robust, context-independent outlier)
    #   red    → flagged in very few windows      (marginal, context-dependent)
    # Points circled in black are "robust" outliers: consistency ≥ 50 %.
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.35)

    ever_flagged = rolling_mask
    if ever_flagged.sum() > 0:
        cmap2 = plt.cm.RdYlGn
        norm2 = Normalize(vmin=0.0, vmax=1.0)
        sc2   = ax.scatter(
            series.index[ever_flagged],
            series.values[ever_flagged],
            c=consistency[ever_flagged],
            cmap=cmap2, norm=norm2,
            s=12, zorder=5, linewidths=0,
        )
        cb2 = fig.colorbar(sc2, ax=ax, orientation='vertical',
                           pad=0.005, fraction=0.018,
                           label='Fraction of containing\nwindows that flagged point')
        cb2.ax.tick_params(labelsize=7)

        # Circle robust outliers (flagged by ≥50 % of windows that contain them)
        robust = (consistency >= 0.5) & ever_flagged
        if robust.sum() > 0:
            ax.scatter(
                series.index[robust], series.values[robust],
                edgecolors='black', facecolors='none',
                s=30, linewidths=0.7, zorder=6,
                label=f'Robust outliers  (≥50% of windows,  n={robust.sum()})',
            )

        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)

    ax.set_title(
        'Panel 4 — Window Consistency  '
        '(green = outlier in almost every window  |  red = outlier in very few windows)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )
    ax.set_ylabel('Deseas. Log Price')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.20)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'outlier_comparison_{method}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("═" * 72)
    print("  OUTLIER COMPARISON: Full-Spectrum vs. Rolling-Window")
    print("═" * 72)

    # 1. Load data
    print("\n[1/3] Loading data …")
    raw = load_data(
        PATHS,
        target_region=ACTIVE_ZONE,
        zone_hydro=ACTIVE_ZONE,
        use_interpolation=True,
        start_date=START_DATE,
        end_date=END_DATE,
        lag_commodity_hours=24,
    )
    print(f"      {len(raw):,} hourly observations loaded.")

    # 2. Standard preprocessing: negative handling → log → deseasonalize
    print("\n[2/3] Preprocessing (negative handling → log → deseasonalize) …")
    data = handle_negative_prices(raw.copy(), method=NEG_PRICE_METHOD)
    data = apply_log_transform(data, save_temp_plots=False)
    data = deseasonalize_logged_variables(data, save_temp_plots=False)

    price_series = data['Price_Log_Deseasonalized'].dropna()
    print(f"      Price_Log_Deseasonalized ready: {len(price_series):,} observations.")

    # 3. Build one figure per method
    print("\n[3/3] Generating comparison figures …")
    for method in ('fredriksson', 'gianfreda'):
        make_figure(price_series, method)

    print(f"\nDone. Figures saved to:  {OUTPUT_DIR}/")
