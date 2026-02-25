"""
outlier_comparison.py  –  Full-Spectrum vs. Rolling-Window Outlier Detection
════════════════════════════════════════════════════════════════════════════════

Compares Fredriksson (2016) and Gianfreda (2010) outlier methods across two
detection scopes (full-series and overlapping rolling windows).

Figures produced
────────────────
  Figure A  [per method × 2]  –  outlier_comparison_{method}.png
    Panel 1 – Full-spectrum: global thresholds with flagged points
    Panel 2 – Rolling-window density: heat map of how many windows flagged each point
    Panel 3 – Full vs. rolling overlap: both / full-only / rolling-only
    Panel 4 – Window consistency: fraction of containing windows that flagged each point

  Figure B  –  outlier_replacement.png
    Panel 1 – Original Price_Log_Deseasonalized (reference)
    Panel 2 – After Fredriksson replacement: original in grey, replaced series
              overlaid, connector lines show the magnitude of each correction
    Panel 3 – After Gianfreda replacement: same layout

  Figure C  –  outlier_cross_method.png
    Panel 1 – Full-window: Both / Fredriksson-only / Gianfreda-only
    Panel 2 – Rolling-window: same three-way split on "flagged in ≥1 window"

Usage
─────
    python outlier_comparison.py      (from MSc-Econ-Thesis root)

Requires data files at the PATHS configured below (same layout as main.py).
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
NEG_PRICE_METHOD = 'shift'           # 'shift' or 'clip'

# Rolling window – overlapping so each point appears in multiple windows.
WINDOW_HOURS     = 365 * 24         # 1-year window
STEP_HOURS       = 30  * 24         # ~1-month step  (≈12× overlap per point)

OUTPUT_DIR       = 'results/outlier_comparison'
# ══════════════════════════════════════════════════════════════════════════════


# ── colour palette (colour-blind friendly) ────────────────────────────────────
C_PRICE        = '#4477AA'   # steel blue – price series
C_BOTH         = '#7B2D8B'   # purple     – flagged by both
C_FULL_ONLY    = '#D62728'   # red        – full-spectrum only
C_ROLLING_ONLY = '#1F77B4'   # blue       – rolling-window only
C_THRESHOLD    = '#AA3333'   # dark red   – threshold lines
C_FRED         = '#E55C00'   # orange     – Fredriksson only  (cross-method)
C_GIAN         = '#009E73'   # teal       – Gianfreda only    (cross-method)
C_ORIG         = '#AAAAAA'   # grey       – original (background in replacement plots)

METHOD_LABELS = {
    'fredriksson': 'Fredriksson (2016)  [+6σ / −3.7σ, global]',
    'gianfreda':   'Gianfreda (2010)  [±3σ per weekday, global]',
}


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION HELPERS  (boolean masks, no replacement)
# ══════════════════════════════════════════════════════════════════════════════

def detect_fredriksson(series: pd.Series) -> pd.Series:
    """True where point exceeds +6σ above or −3.7σ below the global mean."""
    mu, sigma = series.mean(), series.std()
    return (series > mu + 6.0 * sigma) | (series < mu - 3.7 * sigma)


def detect_gianfreda(series: pd.Series) -> pd.Series:
    """True where point exceeds ±3σ of its own weekday's distribution."""
    mask = pd.Series(False, index=series.index)
    dow  = series.index.dayofweek
    for day in range(7):
        sel      = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma = day_data.mean(), day_data.std()
        mask[sel] = (series[sel] > mu + 3.0 * sigma) | (series[sel] < mu - 3.0 * sigma)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  REPLACEMENT HELPERS  (return replaced series + boolean mask of changed pts)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_fred_replacement(series: pd.Series, mask: pd.Series) -> pd.Series:
    """
    Apply Fredriksson replacement logic to all True positions in `mask`.
    Each flagged point → mean of the values at ±24 h and ±48 h (whichever exist).
    """
    replaced = series.copy()
    for pos in np.where(mask)[0]:
        neighbours = []
        for offset in [24, 48]:
            if pos - offset >= 0:
                neighbours.append(series.iloc[pos - offset])
            if pos + offset < len(series):
                neighbours.append(series.iloc[pos + offset])
        if neighbours:
            replaced.iloc[pos] = np.mean(neighbours)
    return replaced


def _apply_gian_replacement(series: pd.Series, mask: pd.Series) -> pd.Series:
    """
    Apply Gianfreda replacement logic to all True positions in `mask`.
    Each flagged point → capped at the ±3σ boundary of its own weekday
    (thresholds computed on the full series).
    """
    replaced = series.copy()
    dow = series.index.dayofweek
    for day in range(7):
        sel      = dow == day
        day_data = series[sel]
        if len(day_data) < 2:
            continue
        mu, sigma = day_data.mean(), day_data.std()
        upper, lower = mu + 3.0 * sigma, mu - 3.0 * sigma
        replaced[sel & mask & (series > upper)] = upper
        replaced[sel & mask & (series < lower)] = lower
    return replaced


def replace_fredriksson(series: pd.Series) -> tuple:
    """
    Fredriksson (2016): detect outliers globally, then replace via ±24h/±48h mean.
    Returns (replaced_series, outlier_boolean_mask).
    """
    mask = detect_fredriksson(series)
    return _apply_fred_replacement(series, mask), mask


def replace_gianfreda(series: pd.Series) -> tuple:
    """
    Gianfreda (2010): detect outliers per weekday, then cap at weekday ±3σ.
    Returns (replaced_series, outlier_boolean_mask).
    """
    mask = detect_gianfreda(series)
    return _apply_gian_replacement(series, mask), mask


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING-WINDOW OUTLIER COUNTS
# ══════════════════════════════════════════════════════════════════════════════

def rolling_outlier_counts(series: pd.Series, method: str) -> pd.DataFrame:
    """
    Slide overlapping windows over `series`.  For every data point record:
      seen    – how many windows contained this point
      flagged – how many of those windows flagged it as an outlier

    Returns DataFrame['seen', 'flagged'] aligned to series.index.
    """
    n       = len(series)
    seen    = np.zeros(n, dtype=np.int32)
    flagged = np.zeros(n, dtype=np.int32)

    window_starts = list(range(0, max(1, n - WINDOW_HOURS + 1), STEP_HOURS))
    n_win         = len(window_starts)

    detect_fn = detect_fredriksson if method == 'fredriksson' else detect_gianfreda

    for i, start in enumerate(window_starts):
        end    = min(start + WINDOW_HOURS, n)
        window = series.iloc[start:end]
        out    = detect_fn(window).values.astype(np.int32)
        seen[start:end]    += 1
        flagged[start:end] += out
        if (i + 1) % max(1, n_win // 5) == 0:
            print(f"      window {i+1:3d}/{n_win}  (obs {start:6d}–{end:6d})")

    return pd.DataFrame({'seen': seen, 'flagged': flagged}, index=series.index)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE A – per-method 4-panel (full vs. rolling within one method)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_A(series:       pd.Series,
                  method:       str,
                  full_mask:    pd.Series,
                  counts:       pd.DataFrame) -> None:
    """4-panel per-method figure (unchanged logic, now uses pre-computed data)."""
    label = METHOD_LABELS[method]
    print(f"  Building Figure A – {method} …")

    rolling_mask = counts['flagged'] > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        consistency = pd.Series(
            np.where(counts['seen'] > 0, counts['flagged'] / counts['seen'], 0.0),
            index=series.index,
        )

    both         = full_mask  &  rolling_mask
    full_only    = full_mask  & ~rolling_mask
    rolling_only = ~full_mask &  rolling_mask
    n_win        = len(range(0, max(1, len(series) - WINDOW_HOURS + 1), STEP_HOURS))

    print(f"    Full-spectrum  : {full_mask.sum():5d} outliers")
    print(f"    Rolling (any)  : {rolling_mask.sum():5d} flagged in ≥1 window")
    print(f"    Both / F-only / R-only : {both.sum()} / {full_only.sum()} / {rolling_only.sum()}")

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
    sk = dict(zorder=5, linewidths=0, s=10)

    # Panel 1 – full-spectrum
    ax = axes[0]
    ax.plot(series.index, series.values, color=C_PRICE, lw=0.4, alpha=0.70,
            label='Price_Log_Deseas.')
    if full_mask.sum() > 0:
        ax.scatter(series.index[full_mask], series.values[full_mask],
                   color=C_FULL_ONLY, label=f'Outliers  (n={full_mask.sum()})', **sk)
    mu, sigma = series.mean(), series.std()
    if method == 'fredriksson':
        ax.axhline(mu + 6.0 * sigma, color=C_THRESHOLD, ls='--', lw=0.9, alpha=0.8,
                   label=f'+6σ = {mu+6*sigma:.2f}')
        ax.axhline(mu - 3.7 * sigma, color=C_THRESHOLD, ls=':', lw=0.9, alpha=0.8,
                   label=f'−3.7σ = {mu-3.7*sigma:.2f}')
    else:
        ax.axhline(mu + 3.0 * sigma, color=C_THRESHOLD, ls='--', lw=0.9, alpha=0.8,
                   label=f'+3σ (overall) = {mu+3*sigma:.2f}')
        ax.axhline(mu - 3.0 * sigma, color=C_THRESHOLD, ls=':', lw=0.9, alpha=0.8,
                   label=f'−3σ (overall) = {mu-3*sigma:.2f}')
    ax.set_title('Panel 1 — Full-Spectrum Outliers  (thresholds on entire series)',
                 fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price')
    ax.legend(fontsize=8, ncol=4, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.20)

    # Panel 2 – rolling density heat map
    ax = axes[1]
    ax.plot(series.index, series.values, color=C_PRICE, lw=0.4, alpha=0.45)
    n_max = int(counts['flagged'].max())
    if rolling_mask.sum() > 0 and n_max > 0:
        sc = ax.scatter(
            series.index[rolling_mask], series.values[rolling_mask],
            c=counts['flagged'][rolling_mask],
            cmap=plt.cm.YlOrRd, norm=Normalize(vmin=1, vmax=max(n_max, 2)),
            **sk,
        )
        cb = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.005,
                          fraction=0.018, label='# windows that\nflagged this point')
        cb.ax.tick_params(labelsize=7)
    ax.set_title(
        f'Panel 2 — Rolling-Window Flagging Density  '
        f'(window={WINDOW_HOURS//24}d  |  step={STEP_HOURS//24}d  |  ≈{n_win} windows)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )
    ax.set_ylabel('Deseas. Log Price')
    ax.grid(True, alpha=0.20)

    # Panel 3 – overlap map
    ax = axes[2]
    ax.plot(series.index, series.values, color=C_PRICE, lw=0.4, alpha=0.45)
    for mask, col, lbl in [
        (both,         C_BOTH,         f'Both (n={both.sum()})'),
        (full_only,    C_FULL_ONLY,    f'Full-spectrum only (n={full_only.sum()})'),
        (rolling_only, C_ROLLING_ONLY, f'Rolling window only (n={rolling_only.sum()})'),
    ]:
        if mask.sum() > 0:
            ax.scatter(series.index[mask], series.values[mask],
                       color=col, label=lbl, **sk)
    ax.set_title('Panel 3 — Overlap / Difference Map  (full-spectrum vs. rolling)',
                 fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price')
    ax.legend(fontsize=9, ncol=3, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.20)

    # Panel 4 – window consistency
    ax = axes[3]
    ax.plot(series.index, series.values, color=C_PRICE, lw=0.4, alpha=0.35)
    if rolling_mask.sum() > 0:
        sc2 = ax.scatter(
            series.index[rolling_mask], series.values[rolling_mask],
            c=consistency[rolling_mask],
            cmap=plt.cm.RdYlGn, norm=Normalize(vmin=0.0, vmax=1.0),
            s=12, zorder=5, linewidths=0,
        )
        cb2 = fig.colorbar(sc2, ax=ax, orientation='vertical', pad=0.005,
                           fraction=0.018,
                           label='Fraction of containing\nwindows that flagged point')
        cb2.ax.tick_params(labelsize=7)
        robust = (consistency >= 0.5) & rolling_mask
        if robust.sum() > 0:
            ax.scatter(series.index[robust], series.values[robust],
                       edgecolors='black', facecolors='none', s=30,
                       linewidths=0.7, zorder=6,
                       label=f'Robust outliers  (≥50% of windows,  n={robust.sum()})')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.set_title(
        'Panel 4 — Window Consistency  '
        '(green = outlier in almost every window  |  red = outlier in very few)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )
    ax.set_ylabel('Deseas. Log Price')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.20)

    _save(fig, f'outlier_comparison_{method}.png')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE B – replacement effect (original vs. each method's cleaned series)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_B(series:            pd.Series,
                  fred_full_mask:    pd.Series,
                  gian_full_mask:    pd.Series,
                  fred_rolling_mask: pd.Series,
                  gian_rolling_mask: pd.Series) -> None:
    """
    5-panel figure comparing full-window vs. rolling-window replacement for each method.

    Panel 1 – Original series (reference baseline)
    Panel 2 – Fredriksson replacement using full-window flags
    Panel 3 – Fredriksson replacement using rolling-window flags
    Panel 4 – Gianfreda replacement using full-window flags
    Panel 5 – Gianfreda replacement using rolling-window flags

    For every replacement panel:
      · Thin grey line  = original series (background context)
      · Coloured line   = series after replacement
      · Grey ×          = original value of each replaced point
      · Coloured ★      = replacement value
      · Vertical lines  = connector showing the magnitude of each correction
      · Annotation box  = median and max |correction|, count of replaced points
    """
    print("  Building Figure B – replacement effect (full-window vs. rolling-window) …")

    # Compute the four replaced series from the pre-detected masks
    fred_full_rep    = _apply_fred_replacement(series, fred_full_mask)
    fred_rolling_rep = _apply_fred_replacement(series, fred_rolling_mask)
    gian_full_rep    = _apply_gian_replacement(series, gian_full_mask)
    gian_rolling_rep = _apply_gian_replacement(series, gian_rolling_mask)

    fig, axes = plt.subplots(
        5, 1, figsize=(18, 28), sharex=True,
        gridspec_kw={'hspace': 0.08, 'top': 0.965, 'bottom': 0.03,
                     'left': 0.06, 'right': 0.98},
    )
    fig.suptitle(
        f'Outlier Replacement Effect: Full-Window vs. Rolling-Window Flags\n'
        f'Zone: {ACTIVE_ZONE}  |  {START_DATE} – {END_DATE}  |  '
        f'Rolling: window={WINDOW_HOURS//24}d, step={STEP_HOURS//24}d',
        fontsize=13, fontweight='bold',
    )

    # ── Panel 1: original (reference) ────────────────────────────────────────
    ax = axes[0]
    ax.plot(series.index, series.values, color=C_PRICE, lw=0.5, alpha=0.85,
            label='Price_Log_Deseasonalized (original, no replacement)')
    ax.set_title('Panel 1 — Original Series  (reference baseline – no outlier replacement)',
                 fontsize=10, fontweight='bold', loc='left', pad=4)
    ax.set_ylabel('Deseas. Log Price')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.20)

    def _draw_replacement_panel(ax, replaced, mask, line_color, scope_label):
        """Shared drawing logic for all replacement panels."""
        ax.plot(series.index, series.values,
                color=C_ORIG, lw=0.35, alpha=0.50, label='Original (grey)')
        ax.plot(replaced.index, replaced.values,
                color=line_color, lw=0.55, alpha=0.85,
                label=f'After replacement  ({scope_label})')

        if mask.sum() > 0:
            orig_vals = series[mask].values
            repl_vals = replaced[mask].values
            dates     = series.index[mask]

            # vertical connectors: original → replacement
            for dt, ov, rv in zip(dates, orig_vals, repl_vals):
                ax.plot([dt, dt], [ov, rv],
                        color='#888888', lw=0.75, alpha=0.50, zorder=3)

            # original outlier positions (grey ×)
            ax.scatter(dates, orig_vals,
                       color=C_ORIG, marker='x', s=25, linewidths=0.9,
                       zorder=5, label=f'Original outlier value  (n={mask.sum()})')

            # replacement values (coloured ★)
            ax.scatter(dates, repl_vals,
                       color=line_color, marker='*', s=55, linewidths=0,
                       zorder=6, label='Replacement value')

            corrections = np.abs(repl_vals - orig_vals)
            ax.annotate(
                f'Points replaced : {mask.sum()}\n'
                f'Median |Δ|      : {np.median(corrections):.3f}\n'
                f'Max    |Δ|      : {corrections.max():.3f}',
                xy=(0.01, 0.97), xycoords='axes fraction',
                va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.80),
            )
        else:
            ax.annotate('No points replaced', xy=(0.01, 0.97),
                        xycoords='axes fraction', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.80))

        ax.set_ylabel('Deseas. Log Price')
        ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.20)

    # ── Panel 2: Fredriksson – full-window flags ──────────────────────────────
    ax = axes[1]
    _draw_replacement_panel(ax, fred_full_rep, fred_full_mask, C_FRED, 'full-window flags')
    ax.set_title(
        f'Panel 2 — Fredriksson  |  Full-Window Flags  '
        f'[+6σ/−3.7σ global → mean ±24h/±48h]  '
        f'({fred_full_mask.sum()} replaced)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )

    # ── Panel 3: Fredriksson – rolling-window flags ───────────────────────────
    ax = axes[2]
    _draw_replacement_panel(ax, fred_rolling_rep, fred_rolling_mask, C_FRED, 'rolling-window flags')
    ax.set_title(
        f'Panel 3 — Fredriksson  |  Rolling-Window Flags  '
        f'[flagged in ≥1 window → same ±24h/±48h replacement]  '
        f'({fred_rolling_mask.sum()} replaced)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )

    # ── Panel 4: Gianfreda – full-window flags ────────────────────────────────
    ax = axes[3]
    _draw_replacement_panel(ax, gian_full_rep, gian_full_mask, C_GIAN, 'full-window flags')
    ax.set_title(
        f'Panel 4 — Gianfreda  |  Full-Window Flags  '
        f'[±3σ per weekday global → capped at threshold]  '
        f'({gian_full_mask.sum()} replaced)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )

    # ── Panel 5: Gianfreda – rolling-window flags ─────────────────────────────
    ax = axes[4]
    _draw_replacement_panel(ax, gian_rolling_rep, gian_rolling_mask, C_GIAN, 'rolling-window flags')
    ax.set_title(
        f'Panel 5 — Gianfreda  |  Rolling-Window Flags  '
        f'[flagged in ≥1 window → same weekday-cap replacement]  '
        f'({gian_rolling_mask.sum()} replaced)',
        fontsize=10, fontweight='bold', loc='left', pad=4,
    )
    ax.set_xlabel('Date')

    _save(fig, 'outlier_replacement.png')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE C – cross-method comparison (Fredriksson vs. Gianfreda)
# ══════════════════════════════════════════════════════════════════════════════

def make_figure_C(series:         pd.Series,
                  fred_full:      pd.Series,
                  gian_full:      pd.Series,
                  fred_rolling:   pd.Series,
                  gian_rolling:   pd.Series) -> None:
    """
    2-panel cross-method figure.

    Each panel shows the price series with outlier points coloured by which
    method(s) flagged them:
      ● Purple  – flagged by BOTH Fredriksson and Gianfreda
      ● Orange  – flagged by Fredriksson ONLY
      ● Teal    – flagged by Gianfreda ONLY

    Panel 1 uses full-window (global) thresholds.
    Panel 2 uses rolling-window flags ("flagged by ≥1 window").
    """
    print("  Building Figure C – cross-method comparison …")

    fig, axes = plt.subplots(
        2, 1, figsize=(18, 13), sharex=True,
        gridspec_kw={'hspace': 0.10, 'top': 0.950, 'bottom': 0.05,
                     'left': 0.06, 'right': 0.98},
    )
    fig.suptitle(
        f'Cross-Method Outlier Comparison: Fredriksson vs. Gianfreda\n'
        f'Zone: {ACTIVE_ZONE}  |  {START_DATE} – {END_DATE}  |  '
        f'{len(series):,} hourly observations',
        fontsize=13, fontweight='bold',
    )

    sk = dict(zorder=5, linewidths=0, s=12)

    panels = [
        (axes[0], fred_full,    gian_full,    'Panel 1 — Full-Window  (global thresholds applied to entire series)'),
        (axes[1], fred_rolling, gian_rolling, 'Panel 2 — Rolling-Window  (flagged by ≥1 overlapping window)'),
    ]

    for ax, f_mask, g_mask, title in panels:
        both      = f_mask  &  g_mask
        fred_only = f_mask  & ~g_mask
        gian_only = ~f_mask &  g_mask

        print(f"    {title[:20]}…  both={both.sum()}  fred-only={fred_only.sum()}  gian-only={gian_only.sum()}")

        ax.plot(series.index, series.values,
                color=C_PRICE, lw=0.4, alpha=0.45, label='Price_Log_Deseas.')

        for mask, col, lbl in [
            (both,      C_BOTH, f'Both methods  (n={both.sum()})'),
            (fred_only, C_FRED, f'Fredriksson only  (n={fred_only.sum()})'),
            (gian_only, C_GIAN, f'Gianfreda only  (n={gian_only.sum()})'),
        ]:
            # Always scatter (even if empty) so every category appears in the legend.
            # A zero-count entry makes it explicit that the method flagged no unique points,
            # rather than looking like the category was omitted by mistake.
            ax.scatter(
                series.index[mask] if mask.sum() > 0 else [],
                series.values[mask] if mask.sum() > 0 else [],
                color=col, label=lbl, **sk,
            )

        # summary box
        total_flagged = (f_mask | g_mask).sum()
        ax.annotate(
            f'Total flagged (either method): {total_flagged}\n'
            f'Both: {both.sum()}  |  Fred-only: {fred_only.sum()}  |  Gian-only: {gian_only.sum()}',
            xy=(0.01, 0.97), xycoords='axes fraction',
            va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.80),
        )

        ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.set_ylabel('Deseas. Log Price')
        ax.legend(fontsize=9, ncol=4, loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.20)

    axes[1].set_xlabel('Date')
    _save(fig, 'outlier_cross_method.png')


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 72)
    print("  OUTLIER COMPARISON: Full-Spectrum vs. Rolling-Window")
    print("  Methods: Fredriksson (2016) vs. Gianfreda (2010)")
    print("═" * 72)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
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

    print("\n[2/4] Preprocessing (negative handling → log → deseasonalize) …")
    data = handle_negative_prices(raw.copy(), method=NEG_PRICE_METHOD)
    data = apply_log_transform(data, save_temp_plots=False)
    data = deseasonalize_logged_variables(data, save_temp_plots=False)
    series = data['Price_Log_Deseasonalized'].dropna()
    print(f"      Price_Log_Deseasonalized ready: {len(series):,} observations.")

    # ── 2. Pre-compute everything (avoid running rolling detection twice) ──────
    print("\n[3/4] Running outlier detection …")

    print("  Full-spectrum – Fredriksson …")
    fred_full_mask = detect_fredriksson(series)

    print("  Full-spectrum – Gianfreda …")
    gian_full_mask = detect_gianfreda(series)

    print(f"  Rolling-window – Fredriksson  "
          f"(window={WINDOW_HOURS//24}d, step={STEP_HOURS//24}d) …")
    fred_counts       = rolling_outlier_counts(series, 'fredriksson')
    fred_rolling_mask = fred_counts['flagged'] > 0

    print(f"  Rolling-window – Gianfreda …")
    gian_counts       = rolling_outlier_counts(series, 'gianfreda')
    gian_rolling_mask = gian_counts['flagged'] > 0

    # (replacements are computed inside make_figure_B from the pre-detected masks)

    # ── 3. Generate all figures ───────────────────────────────────────────────
    print("\n[4/4] Generating figures …")

    # Figure A: per-method 4-panel (full vs. rolling within each method)
    make_figure_A(series, 'fredriksson', fred_full_mask, fred_counts)
    make_figure_A(series, 'gianfreda',   gian_full_mask, gian_counts)

    # Figure B: replacement effect (full-window vs. rolling-window flags)
    make_figure_B(series,
                  fred_full_mask,    gian_full_mask,
                  fred_rolling_mask, gian_rolling_mask)

    # Figure C: cross-method (Fredriksson vs. Gianfreda, full + rolling)
    make_figure_C(series,
                  fred_full_mask,    gian_full_mask,
                  fred_rolling_mask, gian_rolling_mask)

    print(f"\nDone. All figures saved to:  {OUTPUT_DIR}/")
    print("  outlier_comparison_fredriksson.png")
    print("  outlier_comparison_gianfreda.png")
    print("  outlier_replacement.png")
    print("  outlier_cross_method.png")
