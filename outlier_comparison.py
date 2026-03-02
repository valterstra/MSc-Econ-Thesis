"""
outlier_comparison.py  –  Gianfreda Outlier Detection: SD Threshold Comparison
════════════════════════════════════════════════════════════════════════════════

Produces three figures (pages), one per standard-deviation multiplier (3σ, 4σ,
5σ).  Each figure contains three panels stacked vertically:

  Panel 1 – Gianfreda per-weekday thresholds, NO structural break (global)
  Panel 2 – Gianfreda per-weekday thresholds, MANUAL break at 2021-10-01
  Panel 3 – Gianfreda per-weekday thresholds, QLR-detected break (~2022)

Every panel shows
  · Price_Log_Deseasonalized time series
  · Per-weekday ±Nσ bound ribbon (shaded) with explicit bound lines
  · Outlier points scattered in red (count of points outside bounds)
  · Annotation box:
      – Total N; outlier count & percentage
      – Pre / post-break outlier counts (break panels only)
      – Min and Max of the series before and after outlier removal

Figures produced
────────────────
  outlier_gianfreda_3sd.png   3σ per-weekday  ×  {no break / 2021 / QLR}
  outlier_gianfreda_4sd.png   4σ per-weekday  ×  {no break / 2021 / QLR}
  outlier_gianfreda_5sd.png   5σ per-weekday  ×  {no break / 2021 / QLR}

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
ACTIVE_ZONE      = 'SE1'
PATHS = {
    'combined':    f'master data files/2015-2025/Combined_{ACTIVE_ZONE}_Data_2015_2025.xlsx',
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}
START_DATE       = '2015-01-01'
END_DATE         = '2025-12-31'
NEG_PRICE_METHOD = 'shift'          # 'shift' or 'clip'

OUTPUT_DIR       = 'results/outlier_comparison'
MANUAL_BREAK     = pd.Timestamp('2021-10-01')   # energy-crisis onset
# ══════════════════════════════════════════════════════════════════════════════


# ── colour palette (colour-blind friendly, Okabe–Ito) ────────────────────────
C_PRICE      = '#4477AA'   # steel blue   – price series
C_THRESHOLD  = '#AA3333'   # dark red     – bound lines
C_OUTLIER    = '#D62728'   # red          – outlier scatter
C_BOUND_FILL = '#4477AA'   # blue         – safe-zone fill (no break)
C_PRE_FILL   = '#0072B2'   # blue         – pre-break safe zone
C_POST_FILL  = '#E69F00'   # amber        – post-break safe zone
C_BREAK_LINE = '#222222'   # near-black   – structural break vertical line


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def find_structural_break(series: pd.Series) -> pd.Timestamp:
    """
    Quandt/QLR structural break: scan the central 15%–85% of the sample and
    return the date that minimises the pooled within-period sum of squared
    deviations (equivalent to the maximum Chow F-statistic for a mean shift).
    Uses O(n) cumulative-sum vectorisation.
    """
    n    = len(series)
    lo   = int(0.15 * n)
    hi   = int(0.85 * n)
    vals = series.values

    cumsum  = np.cumsum(vals)
    cumsum2 = np.cumsum(vals ** 2)
    total_sum  = cumsum[-1]
    total_sum2 = cumsum2[-1]

    t_range = np.arange(lo, hi)
    n1  = t_range
    n2  = n - t_range
    s1  = cumsum[t_range - 1]
    s21 = cumsum2[t_range - 1]
    s2  = total_sum  - s1
    s22 = total_sum2 - s21

    ssr    = (s21 - s1 ** 2 / n1) + (s22 - s2 ** 2 / n2)
    best_t = t_range[np.argmin(ssr)]
    return series.index[best_t]


def detect_gianfreda_nsigma(series: pd.Series, n_sigma: float) -> pd.Series:
    """
    Flag observations that exceed ±n_sigma × σ of their own weekday's
    distribution (mean + std computed globally across the full series).
    """
    mask = pd.Series(False, index=series.index)
    dow  = series.index.dayofweek
    for day in range(7):
        sel      = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma = day_data.mean(), day_data.std()
        mask[sel] = (
            (series[sel] > mu + n_sigma * sigma) |
            (series[sel] < mu - n_sigma * sigma)
        )
    return mask


def detect_gianfreda_nsigma_sb(series:     pd.Series,
                                n_sigma:    float,
                                break_date: pd.Timestamp) -> pd.Series:
    """
    Apply Gianfreda ±n_sigma × σ detection per weekday, but with thresholds
    estimated independently on each sub-period defined by break_date.
    Uses positional (numpy) indexing to tolerate DST duplicate timestamps.
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
#  REPLACEMENT HELPERS  (cap flagged points at the threshold boundary)
# ══════════════════════════════════════════════════════════════════════════════

def apply_gianfreda_replacement(series:  pd.Series,
                                 mask:    pd.Series,
                                 n_sigma: float) -> pd.Series:
    """Cap outliers at per-weekday ±n_sigma × σ (global thresholds)."""
    replaced = series.copy()
    dow = series.index.dayofweek
    for day in range(7):
        sel      = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma       = day_data.mean(), day_data.std()
        upper, lower    = mu + n_sigma * sigma, mu - n_sigma * sigma
        replaced[sel & mask & (series > upper)] = upper
        replaced[sel & mask & (series < lower)] = lower
    return replaced


def apply_gianfreda_sb_replacement(series:     pd.Series,
                                    mask:       pd.Series,
                                    n_sigma:    float,
                                    break_date: pd.Timestamp) -> pd.Series:
    """
    Cap outliers at sub-period per-weekday ±n_sigma × σ.
    Thresholds are re-estimated on each regime so the cap value matches the
    local scale.  Positional indexing tolerates DST duplicate timestamps.
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
#  BOUND-RIBBON HELPERS  (return per-observation threshold Series for plotting)
# ══════════════════════════════════════════════════════════════════════════════

def _bounds_series(series: pd.Series, n_sigma: float):
    """
    Build two Series (upper, lower) aligned to series.index, each containing
    the per-weekday ±n_sigma × σ threshold for that observation.
    """
    upper = pd.Series(np.nan, index=series.index)
    lower = pd.Series(np.nan, index=series.index)
    dow   = series.index.dayofweek
    for day in range(7):
        sel      = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma   = day_data.mean(), day_data.std()
        upper[sel]  = mu + n_sigma * sigma
        lower[sel]  = mu - n_sigma * sigma
    return upper, lower


def _bounds_series_sb(series:     pd.Series,
                       n_sigma:    float,
                       break_date: pd.Timestamp):
    """
    Build per-weekday ±n_sigma × σ threshold Series, with thresholds
    estimated independently on each sub-period.
    Positional indexing used to tolerate DST duplicate timestamps.
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
#  PANEL DRAWING  (inner helper called once per subplot)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panel(ax,
                series:     pd.Series,
                mask:       pd.Series,
                replaced:   pd.Series,
                n_sigma:    float,
                break_date,           # pd.Timestamp or None
                title:      str) -> None:
    """
    Populate a single Axes with:
      · Price time series
      · Per-weekday ±n_sigma bound ribbon (shaded + explicit lines)
      · Outlier scatter (points outside the bounds, in red)
      · Statistics annotation box
    """
    n_total = len(series)

    # ── 1. Base series ────────────────────────────────────────────────────────
    ax.plot(series.index, series.values,
            color=C_PRICE, lw=0.4, alpha=0.70,
            label='Price_Log_Deseas.', zorder=2)

    # ── 2. Bound ribbon ───────────────────────────────────────────────────────
    if break_date is None:
        upper_b, lower_b = _bounds_series(series, n_sigma)

        ax.fill_between(series.index, lower_b, upper_b,
                        color=C_BOUND_FILL, alpha=0.07,
                        label=f'±{n_sigma}σ safe zone (per weekday)', zorder=1)
        ax.plot(series.index, upper_b,
                color=C_THRESHOLD, lw=0.9, ls='--', alpha=0.80,
                label=f'+{n_sigma}σ bound', zorder=3)
        ax.plot(series.index, lower_b,
                color=C_THRESHOLD, lw=0.9, ls=':', alpha=0.80,
                label=f'−{n_sigma}σ bound', zorder=3)

    else:
        bp = series.index.searchsorted(break_date)
        upper_b, lower_b = _bounds_series_sb(series, n_sigma, break_date)

        # pre-break fill
        ax.fill_between(series.index[:bp],
                        lower_b.values[:bp], upper_b.values[:bp],
                        color=C_PRE_FILL, alpha=0.10,
                        label=f'Pre-break ±{n_sigma}σ safe zone', zorder=1)
        # post-break fill
        ax.fill_between(series.index[bp:],
                        lower_b.values[bp:], upper_b.values[bp:],
                        color=C_POST_FILL, alpha=0.10,
                        label=f'Post-break ±{n_sigma}σ safe zone', zorder=1)
        # bound lines
        ax.plot(series.index, upper_b,
                color=C_THRESHOLD, lw=0.9, ls='--', alpha=0.75,
                label=f'+{n_sigma}σ bound (per regime/weekday)', zorder=3)
        ax.plot(series.index, lower_b,
                color=C_THRESHOLD, lw=0.9, ls=':', alpha=0.75,
                label=f'−{n_sigma}σ bound (per regime/weekday)', zorder=3)
        # break vertical line
        ax.axvline(break_date, color=C_BREAK_LINE, lw=1.4, ls='-', alpha=0.85,
                   zorder=4, label=f'Break: {break_date.strftime("%Y-%m-%d")}')

    # ── 3. Outlier scatter ────────────────────────────────────────────────────
    n_out = int(mask.sum())
    if n_out > 0:
        ax.scatter(series.index[mask], series.values[mask],
                   color=C_OUTLIER, s=10, zorder=5, linewidths=0,
                   label=f'Outliers  (n = {n_out})')

    # ── 4. Annotation box ─────────────────────────────────────────────────────
    pct     = 100.0 * n_out / n_total
    min_bef = series.min()
    max_bef = series.max()
    min_aft = replaced.min()
    max_aft = replaced.max()

    if break_date is None:
        ann = (
            f'N = {n_total:,}   |   Outliers outside ±{n_sigma}σ bounds: '
            f'{n_out}  ({pct:.3f}%)\n'
            f'Min (before / after removal):  {min_bef:.4f}  →  {min_aft:.4f}'
            f'   (Δ = {min_aft - min_bef:+.4f})\n'
            f'Max (before / after removal):  {max_bef:.4f}  →  {max_aft:.4f}'
            f'   (Δ = {max_aft - max_bef:+.4f})'
        )
    else:
        bp       = series.index.searchsorted(break_date)
        n_pre    = int(mask.values[:bp].sum())
        n_post   = int(mask.values[bp:].sum())
        n_pre_t  = bp
        n_post_t = n_total - bp
        pct_pre  = 100.0 * n_pre  / n_pre_t  if n_pre_t  > 0 else 0.0
        pct_post = 100.0 * n_post / n_post_t if n_post_t > 0 else 0.0
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

    # ── 5. Cosmetics ──────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price', fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.20)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FIGURE FUNCTION  (one page = one file = 3 panels)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_nsigma(series:       pd.Series,
                        n_sigma:      float,
                        manual_break: pd.Timestamp,
                        qlr_break:    pd.Timestamp) -> None:
    """
    Build and save a 3-panel figure for Gianfreda ±n_sigma detection.

      Panel 1 – no structural break (global per-weekday thresholds)
      Panel 2 – manual structural break at manual_break
      Panel 3 – QLR-detected structural break at qlr_break

    Output file: outlier_gianfreda_{int(n_sigma)}sd.png
    """
    ns = int(n_sigma)
    print(f"\n  ── Gianfreda ±{n_sigma}σ ──────────────────────────────────────────")

    # ── Pre-compute detection masks ───────────────────────────────────────────
    print(f"    Detecting outliers (global) …")
    mask_global = detect_gianfreda_nsigma(series, n_sigma)

    print(f"    Detecting outliers (manual break {manual_break.date()}) …")
    mask_manual = detect_gianfreda_nsigma_sb(series, n_sigma, manual_break)

    print(f"    Detecting outliers (QLR break {qlr_break.date()}) …")
    mask_qlr    = detect_gianfreda_nsigma_sb(series, n_sigma, qlr_break)

    # ── Pre-compute replaced series ───────────────────────────────────────────
    rep_global = apply_gianfreda_replacement(series, mask_global, n_sigma)
    rep_manual = apply_gianfreda_sb_replacement(series, mask_manual, n_sigma, manual_break)
    rep_qlr    = apply_gianfreda_sb_replacement(series, mask_qlr,    n_sigma, qlr_break)

    # ── Console summary ───────────────────────────────────────────────────────
    bp_m = series.index.searchsorted(manual_break)
    bp_q = series.index.searchsorted(qlr_break)
    print(f"    Global   : {mask_global.sum()} outliers  "
          f"({100*mask_global.sum()/len(series):.3f}%)")
    print(f"    Manual SB: {mask_manual.sum()} outliers  "
          f"(pre {manual_break.date()}: {mask_manual.values[:bp_m].sum()}  "
          f"| post: {mask_manual.values[bp_m:].sum()})")
    print(f"    QLR SB   : {mask_qlr.sum()} outliers  "
          f"(pre {qlr_break.date()}: {mask_qlr.values[:bp_q].sum()}  "
          f"| post: {mask_qlr.values[bp_q:].sum()})")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(18, 24), sharex=True,
        gridspec_kw={'hspace': 0.12, 'top': 0.958, 'bottom': 0.035,
                     'left': 0.055, 'right': 0.98},
    )
    fig.suptitle(
        f'Gianfreda (2010) Outlier Detection — ±{n_sigma}σ per Weekday\n'
        f'Zone: {ACTIVE_ZONE}  |  {START_DATE} – {END_DATE}  |  '
        f'{len(series):,} hourly observations',
        fontsize=13, fontweight='bold',
    )

    panels = [
        (axes[0], mask_global, rep_global, None,
         f'Panel 1 — No Structural Break  '
         f'(global per-weekday ±{n_sigma}σ thresholds)'),
        (axes[1], mask_manual, rep_manual, manual_break,
         f'Panel 2 — Manual Structural Break  '
         f'[{manual_break.strftime("%Y-%m-%d")}, energy-crisis onset]  |  '
         f'per-regime per-weekday ±{n_sigma}σ'),
        (axes[2], mask_qlr, rep_qlr, qlr_break,
         f'Panel 3 — QLR-Detected Structural Break  '
         f'[{qlr_break.strftime("%Y-%m-%d")}]  |  '
         f'per-regime per-weekday ±{n_sigma}σ'),
    ]

    for ax, mask, replaced, break_date, title in panels:
        _draw_panel(ax, series, mask, replaced, n_sigma, break_date, title)

    axes[-1].set_xlabel('Date', fontsize=10)
    _save(fig, f'outlier_gianfreda_{ns}sd.png')


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
    print("  GIANFREDA OUTLIER DETECTION – SD THRESHOLD COMPARISON")
    print("  3σ / 4σ / 5σ  ×  No break / Manual 2021 break / QLR break")
    print("═" * 72)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
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

    print("\n[2/3] Preprocessing (negative handling → log → deseasonalize) …")
    data = handle_negative_prices(raw, method=NEG_PRICE_METHOD)
    del raw
    data = apply_log_transform(data, save_temp_plots=False)
    data = deseasonalize_logged_variables(data, save_temp_plots=False)
    series = data['Price_Log_Deseasonalized'].dropna()
    del data
    print(f"      Price_Log_Deseasonalized ready: {len(series):,} observations.")

    # ── 2. Detect QLR structural break ────────────────────────────────────────
    print("\n[3/3] Generating figures …")
    print("  Detecting QLR structural break …")
    qlr_break = find_structural_break(series)
    print(f"    QLR break detected : {qlr_break.date()}")
    print(f"    Manual break (fixed): {MANUAL_BREAK.date()}")

    # ── 3. Produce one figure per SD multiplier ───────────────────────────────
    for n_sd in (3, 4, 5):
        make_figure_nsigma(series, float(n_sd), MANUAL_BREAK, qlr_break)

    print(f"\nDone.  All figures saved to:  {OUTPUT_DIR}/")
    print("  outlier_gianfreda_3sd.png")
    print("  outlier_gianfreda_4sd.png")
    print("  outlier_gianfreda_5sd.png")
