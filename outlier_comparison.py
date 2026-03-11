"""
outlier_comparison.py  –  Gianfreda ±3σ Outlier Detection Across All SE Zones
════════════════════════════════════════════════════════════════════════════════

Applies the Gianfreda (2010) per-weekday ±3σ outlier detection with a manual
structural break at 2021-10-01 (energy-crisis onset) to each of the four
Swedish electricity price zones (SE1 – SE4).

Every figure shows
  · Price_Log_Deseasonalized time series
  · Per-weekday ±3σ bound ribbon (per regime), faint shaded fill + bound lines
  · Structural break vertical line at 2021-10-01
  · Outlier points scattered in red (count of points outside bounds)
  · Annotation box:
      – Total N; pre-break N; post-break N
      – Outlier count & percentage (total, pre-break, post-break)
      – Min and Max of the series before and after outlier removal

Figures produced  (4 files)
────────────────
  outlier_gianfreda_3sd_SE1.png
  outlier_gianfreda_3sd_SE2.png
  outlier_gianfreda_3sd_SE3.png
  outlier_gianfreda_3sd_SE4.png

Usage
─────
    python outlier_comparison.py      (from MSc-Econ-Thesis root)
"""

import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
ZONES            = ['SE1', 'SE2', 'SE3', 'SE4']
START_DATE       = '2015-01-01'
END_DATE         = '2025-12-31'
NEG_PRICE_METHOD = 'shift'          # 'shift' or 'clip'
N_SIGMA          = 3.0              # Gianfreda threshold multiplier
MANUAL_BREAK     = pd.Timestamp('2021-10-01')   # energy-crisis onset

PATHS_STATIC = {
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}

OUTPUT_DIR = 'results/outlier_comparison/zonal'
# ══════════════════════════════════════════════════════════════════════════════


def _paths_for_zone(zone: str) -> dict:
    return {
        'combined': f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
        **PATHS_STATIC,
    }


# ── colour palette ────────────────────────────────────────────────────────────
C_PRICE      = '#4477AA'   # steel blue   – price series
C_THRESHOLD  = '#AA3333'   # dark red     – bound lines
C_OUTLIER    = '#D62728'   # red          – outlier scatter
C_PRE_FILL   = '#0072B2'   # blue         – pre-break safe zone
C_POST_FILL  = '#E69F00'   # amber        – post-break safe zone
C_BREAK_LINE = '#222222'   # near-black   – structural break vertical line


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def detect_gianfreda_sb(series:     pd.Series,
                         n_sigma:    float,
                         break_date: pd.Timestamp) -> pd.Series:
    """
    Gianfreda per-weekday ±n_sigma × σ outlier detection with thresholds
    estimated independently on each sub-period defined by break_date.
    Uses positional indexing to tolerate DST duplicate timestamps.
    """
    mask_arr  = np.zeros(len(series), dtype=bool)
    break_pos = series.index.searchsorted(break_date)
    vals      = series.values
    dow_all   = series.index.dayofweek

    for start, end in [(0, break_pos), (break_pos, len(series))]:
        sub_vals = vals[start:end]
        dow      = dow_all[start:end]
        for day in range(7):
            sel      = dow == day
            day_vals = sub_vals[sel]
            if len(day_vals) < 2:
                continue
            mu, sigma = day_vals.mean(), day_vals.std()
            flagged   = (
                (day_vals > mu + n_sigma * sigma) |
                (day_vals < mu - n_sigma * sigma)
            )
            sub_pos              = np.arange(start, end)
            mask_arr[sub_pos[sel][flagged]] = True

    return pd.Series(mask_arr, index=series.index)


# ══════════════════════════════════════════════════════════════════════════════
#  REPLACEMENT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def apply_gianfreda_sb_replacement(series:     pd.Series,
                                    mask:       pd.Series,
                                    n_sigma:    float,
                                    break_date: pd.Timestamp) -> pd.Series:
    """
    Cap outliers at the sub-period per-weekday ±n_sigma × σ boundary.
    Thresholds re-estimated per regime; positional indexing handles DST dupes.
    """
    replaced  = series.copy()
    mask_arr  = mask.values
    vals      = series.values
    break_pos = series.index.searchsorted(break_date)
    dow_all   = series.index.dayofweek

    for start, end in [(0, break_pos), (break_pos, len(series))]:
        sub_vals = vals[start:end]
        dow      = dow_all[start:end]
        for day in range(7):
            sel      = dow == day
            day_vals = sub_vals[sel]
            if len(day_vals) < 2:
                continue
            mu, sigma    = day_vals.mean(), day_vals.std()
            upper, lower = mu + n_sigma * sigma, mu - n_sigma * sigma
            sub_pos      = np.arange(start, end)
            day_pos      = sub_pos[sel]
            sub_mask     = mask_arr[day_pos]
            replaced.iloc[day_pos[sub_mask & (day_vals > upper)]] = upper
            replaced.iloc[day_pos[sub_mask & (day_vals < lower)]] = lower

    return replaced


# ══════════════════════════════════════════════════════════════════════════════
#  BOUND-RIBBON HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _bounds_series_sb(series:     pd.Series,
                       n_sigma:    float,
                       break_date: pd.Timestamp):
    """
    Return two Series (upper, lower) aligned to series.index, each holding the
    per-weekday ±n_sigma × σ threshold computed within its sub-period.
    """
    upper     = pd.Series(np.nan, index=series.index)
    lower     = pd.Series(np.nan, index=series.index)
    vals      = series.values
    dow_all   = series.index.dayofweek
    break_pos = series.index.searchsorted(break_date)

    for start, end in [(0, break_pos), (break_pos, len(series))]:
        sub_vals = vals[start:end]
        dow      = dow_all[start:end]
        for day in range(7):
            sel      = dow == day
            day_vals = sub_vals[sel]
            if len(day_vals) < 2:
                continue
            mu, sigma = day_vals.mean(), day_vals.std()
            sub_pos   = np.arange(start, end)
            day_pos   = sub_pos[sel]
            upper.iloc[day_pos] = mu + n_sigma * sigma
            lower.iloc[day_pos] = mu - n_sigma * sigma

    return upper, lower


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_zone(zone:        str,
                     series:      pd.Series,
                     break_date:  pd.Timestamp,
                     n_sigma:     float) -> None:
    """
    Build and save one figure for a single zone:
      Gianfreda ±n_sigma per weekday, structural break at break_date.
    Output: outlier_gianfreda_{int(n_sigma)}sd_{zone}.png
    """
    print(f"\n  ── Zone {zone} ───────────────────────────────────────────────────")

    # ── Detection & replacement ───────────────────────────────────────────────
    mask     = detect_gianfreda_sb(series, n_sigma, break_date)
    replaced = apply_gianfreda_sb_replacement(series, mask, n_sigma, break_date)

    bp      = series.index.searchsorted(break_date)
    n_total = len(series)
    n_pre_t = bp
    n_post_t = n_total - bp
    n_out   = int(mask.sum())
    n_pre   = int(mask.values[:bp].sum())
    n_post  = int(mask.values[bp:].sum())

    print(f"    Total outliers : {n_out}  ({100*n_out/n_total:.3f}%)")
    print(f"    Pre  {break_date.date()} : {n_pre}  ({100*n_pre/n_pre_t:.3f}%)")
    print(f"    Post {break_date.date()} : {n_post}  ({100*n_post/n_post_t:.3f}%)")

    # ── Bound ribbon ──────────────────────────────────────────────────────────
    upper_b, lower_b = _bounds_series_sb(series, n_sigma, break_date)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        1, 1, figsize=(18, 8),
        gridspec_kw={'top': 0.88, 'bottom': 0.09, 'left': 0.055, 'right': 0.98},
    )
       # 1. Base series
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.70,
            label='Price_Log_Deseas.', zorder=2)

    # 2. Bound ribbons (pre / post)
    ax.fill_between(series.index[:bp],
                    lower_b.values[:bp], upper_b.values[:bp],
                    color=C_PRE_FILL, alpha=0.05,
                    label=f'Pre-break ±{n_sigma}σ safe zone', zorder=1)
    ax.fill_between(series.index[bp:],
                    lower_b.values[bp:], upper_b.values[bp:],
                    color=C_POST_FILL, alpha=0.05,
                    label=f'Post-break ±{n_sigma}σ safe zone', zorder=1)

    # 3. Bound lines
    ax.plot(series.index, upper_b,
            color=C_THRESHOLD, lw=0.5, ls='--', alpha=0.35,
            label=f'+{n_sigma}σ bound (per regime/weekday)', zorder=3)
    ax.plot(series.index, lower_b,
            color=C_THRESHOLD, lw=0.5, ls=':', alpha=0.35,
            label=f'−{n_sigma}σ bound (per regime/weekday)', zorder=3)

    # 4. Break line
    ax.axvline(break_date, color=C_BREAK_LINE, lw=1.4, ls='-', alpha=0.85,
               zorder=4, label=f'Break: {break_date.strftime("%Y-%m-%d")}')

    # 5. Outlier scatter
    if n_out > 0:
        ax.scatter(series.index[mask], series.values[mask],
                   color=C_OUTLIER, s=10, zorder=5, linewidths=0,
                   label=f'Outliers  (n = {n_out})')

    # 6. Annotation box
    pct      = 100.0 * n_out  / n_total
    pct_pre  = 100.0 * n_pre  / n_pre_t  if n_pre_t  > 0 else 0.0
    pct_post = 100.0 * n_post / n_post_t if n_post_t > 0 else 0.0
    min_bef  = series.min()
    max_bef  = series.max()
    min_aft  = replaced.min()
    max_aft  = replaced.max()

    ann = (
        f'N = {n_total:,}   '
        f'(pre-break: {n_pre_t:,}  |  post-break: {n_post_t:,})\n'
        f'Outliers outside ±{n_sigma}σ bounds: {n_out}  ({pct:.3f}%)   '
        f'[pre: {n_pre}  ({pct_pre:.3f}%)   |   post: {n_post}  ({pct_post:.3f}%)]\n'
        f'Min (before / after removal):  {min_bef:.4f}  →  {min_aft:.4f}'
        f'   (Δ = {min_aft - min_bef:+.4f})\n'
        f'Max (before / after removal):  {max_bef:.4f}  →  {max_aft:.4f}'
        f'   (Δ = {max_aft - max_bef:+.4f})'
    )
    ax.annotate(
        ann,
        xy=(0.01, 0.97), xycoords='axes fraction',
        va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.40', fc='white', ec='#BBBBBB', alpha=0.92),
    )

    # 7. Cosmetics
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Deseas. Log Price', fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.20)

    _save(fig, f'outlier_gianfreda_{int(n_sigma)}sd_{zone}.png')


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 72)
    print("  GIANFREDA ±3σ OUTLIER DETECTION  –  ALL SE ZONES")
    print(f"  Break: {MANUAL_BREAK.date()}  |  Zones: {', '.join(ZONES)}")
    print("═" * 72)

    for zone in ZONES:
        print(f"\n{'─'*72}")
        print(f"  Processing zone {zone} …")
        print(f"{'─'*72}")

        # ── Load ──────────────────────────────────────────────────────────────
        print(f"  [1/3] Loading data for {zone} …")
        raw = load_data(
            _paths_for_zone(zone),
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=START_DATE,
            end_date=END_DATE,
            lag_commodity_hours=24,
        )
        print(f"        {len(raw):,} hourly observations loaded.")

        # ── Preprocess ────────────────────────────────────────────────────────
        print(f"  [2/3] Preprocessing …")
        data = handle_negative_prices(raw, method=NEG_PRICE_METHOD)
        del raw
        data = apply_log_transform(data, save_temp_plots=False)
        data = deseasonalize_logged_variables(data, save_temp_plots=False)
        series = data['Price_Log_Deseasonalized'].dropna()
        del data
        print(f"        Price_Log_Deseasonalized ready: {len(series):,} observations.")

        # ── Figure ────────────────────────────────────────────────────────────
        print(f"  [3/3] Building figure …")
        make_figure_zone(zone, series, MANUAL_BREAK, N_SIGMA)

        del series
        gc.collect()

    print(f"\nDone.  All 4 figures saved to:  {OUTPUT_DIR}")
    for zone in ZONES:
        print(f"  outlier_gianfreda_{int(N_SIGMA)}sd_{zone}.png")
