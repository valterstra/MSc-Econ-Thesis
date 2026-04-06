"""
plot_structural_break_monthly.py

Plots the rolling wind coefficient (mean equation, beta) from the monthly
GARCH-X results with piecewise-linear structural breaks overlaid.

Source files:  output from stata/rolling_window_results/
                   rolling_garch_{zone}_1yr_monthly_hetfull.csv
Output files:  output from stata/structural_break/
                   sb_monthly_{zone}.png   (one per zone)

Break detection
---------------
Piecewise-linear trend break (Bai-Perron style):
  For each candidate number of breaks (0 ... MAX_BREAKS), dynamic programming
  finds the split that minimises total RSS when a separate OLS line
  (intercept + slope on time index) is fitted in every segment.
  BIC selects the optimal number of breaks.  Sequential F-tests (m vs m+1)
  are also reported to the console.
  Minimum segment length: MIN_SEG windows on each side.

Usage:
    python plot_structural_break_monthly.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -- Configuration -------------------------------------------------------------

ZONES       = ['SE4']   #'SE1', 'SE2', 'SE3', 
RESULTS_DIR = 'output from stata/rolling_window_results'
OUT_DIR     = 'output from stata/structural_break'

MAX_BREAKS   = 6     # upper limit; BIC chooses 0 ... MAX_BREAKS
FORCE_BREAKS = None     # set to an int to force exactly N breaks; None = BIC
MIN_SEG      = 12    # minimum windows per segment (12 months ~ 1 year)

ZONE_COLORS = {
    'SE1': "#00193d",
    'SE2': "#034d8e",
    'SE3': "#0b8dd8",
    'SE4': "#6cb9e9",
}


# -- Data loading --------------------------------------------------------------

def load_zone(zone: str, row_type: str, variable: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        start_date, b_wind, se_wind, converged
    Hard failures (converged == 0) -> NaN.
    """
    path = os.path.join(RESULTS_DIR,
                        f'rolling_garch_{zone}_1yr_monthly_hetfull.csv')
    raw  = pd.read_csv(path)
    raw['start_date'] = pd.to_datetime(raw['start_date'])

    sub = raw[(raw['type'] == row_type) & (raw['variable'] == variable)].copy()
    sub = (sub[['start_date', 'value', 'se', 'converged']]
           .rename(columns={'value': 'b_wind', 'se': 'se_wind'})
           .sort_values('start_date')
           .reset_index(drop=True))

    fail = sub['converged'] == 0
    sub.loc[fail, ['b_wind', 'se_wind']] = np.nan

    return sub


# -- Piecewise-linear break detection ------------------------------------------

def fit_linear(t: np.ndarray, y: np.ndarray):
    """
    OLS fit of y ~ 1 + t on non-NaN observations.
    Returns (rss, intercept, slope) or (inf, nan, nan) if too few points.
    """
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.inf, np.nan, np.nan
    t_ok, y_ok = t[ok], y[ok]
    A = np.column_stack([np.ones(len(t_ok)), t_ok])
    coef, rss_arr, _, _ = np.linalg.lstsq(A, y_ok, rcond=None)
    rss = float(rss_arr[0]) if len(rss_arr) == 1 \
          else float(np.sum((y_ok - A @ coef) ** 2))
    return rss, coef[0], coef[1]


def find_breaks(t: np.ndarray, b: np.ndarray,
                n_breaks: int, min_seg: int = MIN_SEG):
    """
    Dynamic-programming search for the best `n_breaks` split points.

    Returns (break_indices, total_rss, list_of_(intercept, slope) per segment).
    break_indices are positions in [0, n) where the new segment begins
    (i.e. segment boundaries are [0, k0), [k0, k1), ...).
    Returns ([], inf, []) if no valid partition exists.
    """
    n = len(b)

    if n_breaks == 0:
        rss, a, s = fit_linear(t, b)
        return [], rss, [(a, s)]

    best_breaks = []
    best_rss    = np.inf
    best_params = []

    def search(remaining, start, chosen):
        nonlocal best_breaks, best_rss, best_params

        if remaining == 0:
            rss_last, a_last, s_last = fit_linear(t[start:], b[start:])
            if not np.isfinite(rss_last):
                return
            total = 0.0
            params = []
            prev = 0
            for k in chosen:
                r, a, s = fit_linear(t[prev:k], b[prev:k])
                if not np.isfinite(r):
                    return
                total += r
                params.append((a, s))
                prev = k
            total += rss_last
            params.append((a_last, s_last))
            if total < best_rss:
                best_rss    = total
                best_breaks = chosen.copy()
                best_params = params.copy()
            return

        earliest = start + min_seg
        latest   = n - remaining * min_seg - min_seg
        if earliest > latest:
            return
        for k in range(earliest, latest + 1):
            chosen.append(k)
            search(remaining - 1, k, chosen)
            chosen.pop()

    search(n_breaks, 0, [])
    return best_breaks, best_rss, best_params


def select_breaks(t: np.ndarray, b: np.ndarray,
                  max_breaks: int = MAX_BREAKS,
                  min_seg: int = MIN_SEG) -> dict:
    """
    Fit piecewise-linear models for 0 ... max_breaks breaks.
    Select optimal number by BIC.  Also run sequential F-tests.

    Returns a dict with keys:
        n_breaks, break_indices, params, bic_table, f_tests
    """
    n = len(b)
    models = []

    for m in range(max_breaks + 1):
        bk, rss, params = find_breaks(t, b, m, min_seg)
        if not np.isfinite(rss):
            break
        k_params = 2 * (m + 1)   # intercept + slope per segment
        bic = n * np.log(rss / n + 1e-15) + k_params * np.log(n)
        models.append({'n': m, 'breaks': bk, 'rss': rss,
                       'params': params, 'bic': bic})

    if not models:
        return {'n_breaks': 0, 'break_indices': [], 'params': [],
                'bic_table': [], 'f_tests': []}

    best = min(models, key=lambda x: x['bic'])

    f_tests = []
    for i in range(len(models) - 1):
        m0, m1 = models[i], models[i + 1]
        df_num = 2
        df_den = n - 2 * (m1['n'] + 1)
        if df_den > 0 and m1['rss'] > 0:
            f  = ((m0['rss'] - m1['rss']) / df_num) / (m1['rss'] / df_den)
            pv = 1 - stats.f.cdf(f, df_num, df_den)
        else:
            f, pv = np.nan, np.nan
        f_tests.append({'test': f'{m0["n"]} vs {m1["n"]}',
                        'f': f, 'p': pv})

    return {
        'n_breaks':      best['n'],
        'break_indices': best['breaks'],
        'params':        best['params'],
        'bic_table':     [{'n': m['n'], 'bic': m['bic']} for m in models],
        'f_tests':       f_tests,
    }


# -- Plot ----------------------------------------------------------------------

def plot_zone(zone: str, row_type: str, variable: str,
             ylabel: str, tag: str, coef_label: str) -> None:
    df    = load_zone(zone, row_type, variable)
    color = ZONE_COLORS[zone]
    n     = len(df)

    # Window midpoint: start + 6 months (centre of the 1-year window)
    dates = (df['start_date'] + pd.DateOffset(months=6)).values
    b     = df['b_wind'].values
    se    = df['se_wind'].values
    t     = np.arange(n, dtype=float)

    if FORCE_BREAKS is not None:
        bk, rss, params = find_breaks(t, b, FORCE_BREAKS)
        result = {
            'n_breaks':      FORCE_BREAKS,
            'break_indices': bk,
            'params':        params,
            'bic_table':     [],
            'f_tests':       [],
        }
    else:
        result = select_breaks(t, b)

    # Console report
    print(f'\n{zone}  [{coef_label}]')
    print('  BIC table:')
    for row in result['bic_table']:
        marker = ' <-- selected' if row['n'] == result['n_breaks'] else ''
        print(f"    {row['n']} break(s): BIC = {row['bic']:.2f}{marker}")
    print('  Sequential F-tests:')
    for ft in result['f_tests']:
        sig = '***' if ft['p'] < 0.01 else ('**' if ft['p'] < 0.05
              else ('*' if ft['p'] < 0.10 else ''))
        print(f"    {ft['test']}:  F = {ft['f']:.2f},  p = {ft['p']:.4f}  {sig}")

    # Figure
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(dates, b, color=color, linewidth=2, marker='o', markersize=3,
            label=zone, zorder=3)
    ax.fill_between(dates,
                    b - 1.96 * se,
                    b + 1.96 * se,
                    color=color, alpha=0.12, zorder=2)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)

    boundaries = [0] + result['break_indices'] + [n]

    legend_trend = True
    for seg_idx, (start, end, (a, s)) in enumerate(
            zip(boundaries[:-1], boundaries[1:], result['params'])):
        if not (np.isfinite(a) and np.isfinite(s)):
            continue
        seg_dates = dates[start:end]
        seg_t     = t[start:end]
        y_fit     = a + s * seg_t
        label     = 'Trend segment' if legend_trend else None
        ax.plot(seg_dates, y_fit, color='#c0392b', linewidth=2.0,
                linestyle='-', zorder=4, label=label)
        legend_trend = False

    ylo, yhi = ax.get_ylim()
    for i, brk in enumerate(result['break_indices']):
        bd  = dates[brk]
        ts  = pd.Timestamp(bd)
        label = 'Structural break' if i == 0 else None
        ax.axvline(bd, color='#c0392b', linewidth=1.6,
                   linestyle='--', zorder=4, label=label)
        ax.text(bd, ylo + 0.92 * (yhi - ylo),
                f'  {ts.strftime("%b %Y")}',
                color='#c0392b', fontsize=9, va='center', zorder=5)

    n_brk = result['n_breaks']
    ax.set_xlabel('Window midpoint', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1, 13)))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.4,
            alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'sb_monthly_{zone}_{tag}_{n_brk}brk.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# -- Run -----------------------------------------------------------------------

SPECS = [
    # (row_type,   variable,       ylabel,                                          tag,   coef_label)
    ('coef',      'wind_log',     r'$\hat{\beta}_{wind}$  (mean equation)',         'mean', 'wind coefficient (mean eq.)'),
    ('var_coef',  'het_wind_log', r'$\hat{\gamma}_{wind}$  (variance equation)',    'var',  'wind coefficient (variance eq.)'),
]

if __name__ == '__main__':
    for zone in ZONES:
        for row_type, variable, ylabel, tag, coef_label in SPECS:
            try:
                plot_zone(zone, row_type, variable, ylabel, tag, coef_label)
            except FileNotFoundError as e:
                print(f'  {zone}: skipped -- {e}')
