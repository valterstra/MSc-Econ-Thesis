"""
preprocessing_justification.py

Methodological justification for preprocessing decisions.
For each variable we examine:
  1. Distribution (histogram, log-scale check)
  2. Seasonal patterns by hour-of-day, day-of-week, month-of-year (boxplots + means)
  3. Formal F-tests (one-way ANOVA) for each grouping dimension
  4. Variance decomposition via OLS R² on seasonal dummies

Run one variable at a time, or call run_all_variables() for a full sweep.

Usage:
    python preprocessing_justification.py
"""

import os
import io
import sys
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from preprocessing import (
    load_data,
    handle_negative_prices,
    apply_log_transform,
    deseasonalize_logged_variables,
)


# -- Configuration --------------------------------------------------------------

ZONE       = 'SE4'
START_DATE = '2025-01-01'
END_DATE   = '2025-12-31'

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    'preprocessing justification'
)
os.makedirs(OUT_DIR, exist_ok=True)

PATHS = {
    'combined':    f'master data files/2015-2025/Combined_{ZONE}_Data_2015_2025.xlsx',
    'hydro':        'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':    'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities':  'master data files/Master_Commodities.xlsx',
}

DOW_LABELS   = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# -- Data loading --------------------------------------------------------------─

def load_raw(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    """Load raw (untransformed) data for the given zone and date range."""
    paths = {
        'combined':   f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
        'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
        'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
        'commodities': 'master data files/Master_Commodities.xlsx',
    }
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_data(paths, target_region=zone, zone_hydro=zone,
                       use_interpolation=True,
                       start_date=start_date, end_date=end_date,
                       lag_commodity_hours=24,
                       use_bilateral_exchange=True)
    return df


# -- Core analysis functions ----------------------------------------------------

def add_time_features(series: pd.Series) -> pd.DataFrame:
    """Add hour, DOW, month columns to a series with DatetimeIndex."""
    df = series.to_frame(name='value')
    df['hour']  = df.index.hour
    df['dow']   = df.index.dayofweek
    df['month'] = df.index.month
    return df


def anova_test(series: pd.Series, grouper: pd.Series, label: str) -> dict:
    """
    One-way ANOVA: test whether group means of `series` differ across `grouper`.
    Returns dict with F-statistic, p-value, group count, eta-squared (effect size).
    """
    groups = [series[grouper == g].dropna().values for g in sorted(grouper.unique())]
    groups = [g for g in groups if len(g) > 1]
    F, p = stats.f_oneway(*groups)

    # Eta-squared: proportion of variance explained by grouping
    grand_mean = series.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = ((series - grand_mean)**2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    return {'grouping': label, 'F': F, 'p': p, 'eta_squared': eta2,
            'n_groups': len(groups)}


def variance_decomposition(series: pd.Series) -> pd.DataFrame:
    """
    OLS R² of series regressed on each seasonal dummy set separately,
    then jointly, including interaction specs (hour×DOW, hour×month).
    Uses pure numpy to avoid any index alignment issues.
    Returns a DataFrame of results.
    """
    df = add_time_features(series.dropna())
    y = df['value'].values  # 1-D numpy array

    # Main effect dummies
    h = pd.get_dummies(df['hour'],  prefix='h', drop_first=True, dtype=float).values
    d = pd.get_dummies(df['dow'],   prefix='d', drop_first=True, dtype=float).values
    m = pd.get_dummies(df['month'], prefix='m', drop_first=True, dtype=float).values

    # Interaction dummies: combined cell labels (e.g. "3_1" = hour 3, Monday)
    hd_key = df['hour'].astype(str) + '_' + df['dow'].astype(str)
    hm_key = df['hour'].astype(str) + '_' + df['month'].astype(str)
    hd = pd.get_dummies(hd_key, prefix='hd', drop_first=True, dtype=float).values  # 167 dummies
    hm = pd.get_dummies(hm_key, prefix='hm', drop_first=True, dtype=float).values  # 287 dummies

    ones = np.ones((len(y), 1))

    specs = {
        'Hour dummies (24)':                    np.hstack([ones, h]),
        'DOW dummies (7)':                      np.hstack([ones, d]),
        'Month dummies (12)':                   np.hstack([ones, m]),
        'Hour + DOW (main effects)':            np.hstack([ones, h, d]),
        'Hour + DOW + Month (main effects)':    np.hstack([ones, h, d, m]),
        'Hour×DOW interactions (168 cells)':    np.hstack([ones, hd]),
        'Hour×Month interactions (288 cells)':  np.hstack([ones, hm]),
        'Hour×DOW + Hour×Month (full spec)':    np.hstack([ones, hd, hm]),
    }

    results = []
    for label, X in specs.items():
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results.append({'Specification': label, 'R2': r2})

    return pd.DataFrame(results)


# -- Plotting functions --------------------------------------------------------─

def plot_seasonal_patterns(series: pd.Series, var_name: str,
                           log_series: pd.Series = None,
                           zone: str = ZONE,
                           start_date: str = START_DATE,
                           end_date: str = END_DATE) -> str:
    """
    3×2 grid of boxplots: rows = hour / DOW / month, cols = raw / log.
    Returns the path to the saved figure.
    """
    df = add_time_features(series)
    has_log = log_series is not None
    if has_log:
        df_log = add_time_features(log_series)

    ncols = 2 if has_log else 1
    fig, axes = plt.subplots(3, ncols, figsize=(7 * ncols, 14))
    if ncols == 1:
        axes = axes[:, np.newaxis]

    col_titles = ['Raw', 'Log'] if has_log else ['Raw']
    row_groups = [
        ('hour',  list(range(24)),     [str(h) for h in range(24)], 'Hour of Day'),
        ('dow',   list(range(7)),      DOW_LABELS,                  'Day of Week'),
        ('month', list(range(1, 13)),  MONTH_LABELS,                'Month'),
    ]

    for col_idx, (col_title, src_df) in enumerate(
            zip(col_titles, [df, df_log] if has_log else [df])):

        for row_idx, (groupcol, group_vals, group_labels, xlabel) in enumerate(row_groups):
            ax = axes[row_idx, col_idx]

            data_by_group = [
                src_df.loc[src_df[groupcol] == g, 'value'].dropna().values
                for g in group_vals
            ]
            bp = ax.boxplot(data_by_group, tick_labels=group_labels,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.5))
            for patch in bp['boxes']:
                patch.set_facecolor('#4C8CBF')
                patch.set_alpha(0.6)

            # Overlay mean line
            means = [np.mean(d) if len(d) > 0 else np.nan for d in data_by_group]
            ax.plot(range(1, len(group_vals) + 1), means,
                    'o-', color='crimson', linewidth=1.2, markersize=3, label='Mean')

            if groupcol == 'hour':
                ax.set_xticks(range(1, 25, 3))
                ax.set_xticklabels([str(h) for h in range(0, 24, 3)], fontsize=8)
            else:
                ax.set_xticklabels(group_labels, fontsize=8)

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(col_title, fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            if row_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()

    var_dir = os.path.join(OUT_DIR, 'variable analysis')
    os.makedirs(var_dir, exist_ok=True)
    fname = f'{var_name.lower().replace(" ", "_")}_{zone}_seasonal_patterns.png'
    fpath = os.path.join(var_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_distribution(series: pd.Series, var_name: str,
                      log_series: pd.Series = None,
                      zone: str = ZONE) -> str:
    """
    Histogram of raw series and (optionally) log series side by side.
    Returns path to saved figure.
    """
    ncols = 2 if log_series is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, s, title in zip(axes,
                             [series] + ([log_series] if log_series is not None else []),
                             ['Raw', 'Log']):
        ax.hist(s.dropna(), bins=80, color='#4C8CBF', edgecolor='white', linewidth=0.3)
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Annotate skewness
        sk = s.dropna().skew()
        ax.text(0.97, 0.95, f'Skew = {sk:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.tight_layout()
    var_dir = os.path.join(OUT_DIR, 'variable analysis')
    os.makedirs(var_dir, exist_ok=True)
    fname = f'{var_name.lower().replace(" ", "_")}_{zone}_distribution.png'
    fpath = os.path.join(var_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


# -- Top-level variable analysis ------------------------------------------------

def analyze_variable(series: pd.Series, var_name: str,
                     apply_log: bool = True,
                     zone: str = ZONE,
                     start_date: str = START_DATE,
                     end_date: str = END_DATE) -> None:
    """
    Full analysis for one variable:
      - Summary statistics
      - Distribution plot (raw + log if apply_log)
      - Seasonal pattern plots (raw + log if apply_log)
      - ANOVA F-tests for hour, DOW, month
      - Variance decomposition R²
    Prints a report to console.
    """
    series = series.dropna()
    log_series = np.log(series.clip(lower=0.01)) if apply_log else None

    print('\n' + '='*70)
    print(f'  VARIABLE: {var_name}  |  {zone}  {start_date}–{end_date}')
    print('='*70)

    # --- Summary statistics ---
    print('\n-- Summary statistics (raw) --')
    desc = series.describe()
    print(f"  N        : {int(desc['count']):,}")
    print(f"  Mean     : {desc['mean']:.4f}")
    print(f"  Std      : {desc['std']:.4f}")
    print(f"  Min      : {desc['min']:.4f}")
    print(f"  Max      : {desc['max']:.4f}")
    print(f"  Skewness : {series.skew():.4f}")
    print(f"  Negatives: {(series < 0).sum():,}  ({(series < 0).mean()*100:.2f}%)")
    print(f"  Zeros    : {(series == 0).sum():,}")

    # --- Distribution plots ---
    dist_path = plot_distribution(series, var_name, log_series, zone)
    print(f'\n-- Distribution plot saved: {os.path.basename(dist_path)}')

    # --- Seasonal pattern plots ---
    seas_path = plot_seasonal_patterns(series, var_name, log_series,
                                        zone, start_date, end_date)
    print(f'-- Seasonal pattern plot saved: {os.path.basename(seas_path)}')

    # --- ANOVA F-tests ---
    print('\n-- ANOVA F-tests (raw series) --')
    df_feat = add_time_features(series)
    print(f"  {'Grouping':<25} {'F':>10} {'p-value':>12} {'Eta²':>8}  Significant?")
    print(f"  {'-'*60}")
    for gcol, glabel in [('hour', 'Hour of day'), ('dow', 'Day of week'), ('month', 'Month of year')]:
        res = anova_test(df_feat['value'], df_feat[gcol], glabel)
        sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else ('*' if res['p'] < 0.05 else 'n.s.'))
        print(f"  {glabel:<25} {res['F']:>10.2f} {res['p']:>12.2e} {res['eta_squared']:>8.4f}  {sig}")

    if apply_log:
        print('\n-- ANOVA F-tests (log series) --')
        df_log_feat = add_time_features(log_series)
        print(f"  {'Grouping':<25} {'F':>10} {'p-value':>12} {'Eta²':>8}  Significant?")
        print(f"  {'-'*60}")
        for gcol, glabel in [('hour', 'Hour of day'), ('dow', 'Day of week'), ('month', 'Month of year')]:
            res = anova_test(df_log_feat['value'], df_log_feat[gcol], glabel)
            sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else ('*' if res['p'] < 0.05 else 'n.s.'))
            print(f"  {glabel:<25} {res['F']:>10.2f} {res['p']:>12.2e} {res['eta_squared']:>8.4f}  {sig}")

    # --- Variance decomposition ---
    target = log_series if apply_log else series
    label  = 'log series' if apply_log else 'raw series'
    print(f'\n-- Variance decomposition R² ({label}) --')
    vd = variance_decomposition(target)
    for _, row in vd.iterrows():
        print(f"  {row['Specification']:<35} R2 = {row['R2']:.4f}  ({row['R2']*100:.1f}% of variance)")

    print('\n' + '='*70)


# -- Price-specific helpers ----------------------------------------------------

def _make_holiday_dummy(index):
    """Return a float array (0/1) for Swedish public holidays."""
    import holidays as hol_lib
    se_holidays = hol_lib.Sweden(years=list(index.year.unique()))
    return np.array([1.0 if d in se_holidays else 0.0 for d in index.date])


def deseasonalize_price_spec(log_price: pd.Series, spec: str) -> pd.Series:
    """
    Deseasonalize log price with one of three progressively richer specs.
      'basic'   : hour + DOW + holiday + month + year
      'plus_hd' : hour×DOW + holiday + month + year
      'full'    : hour×DOW + hour×month + holiday + year
    Returns residuals as pd.Series with the same index.
    """
    s  = log_price.dropna()
    df = add_time_features(s)
    y  = df['value'].values

    ones = np.ones((len(y), 1))
    hol  = _make_holiday_dummy(df.index).reshape(-1, 1)
    yr   = pd.get_dummies(pd.Series(df.index.year, index=df.index),
                          prefix='yr', drop_first=True, dtype=float).values
    m    = pd.get_dummies(df['month'], prefix='m',  drop_first=True, dtype=float).values
    h    = pd.get_dummies(df['hour'],  prefix='h',  drop_first=True, dtype=float).values
    d    = pd.get_dummies(df['dow'],   prefix='d',  drop_first=True, dtype=float).values

    hd_key = df['hour'].astype(str) + '_' + df['dow'].astype(str)
    hm_key = df['hour'].astype(str) + '_' + df['month'].astype(str)
    hd = pd.get_dummies(hd_key, prefix='hd', drop_first=True, dtype=float).values
    hm = pd.get_dummies(hm_key, prefix='hm', drop_first=True, dtype=float).values

    if spec == 'basic':
        X = np.hstack([ones, h, d, hol, m, yr])
    elif spec == 'plus_hd':
        X = np.hstack([ones, hd, hol, m, yr])
    elif spec == 'full':
        X = np.hstack([ones, hd, hm, hol, yr])
    else:
        raise ValueError(f'Unknown spec: {spec}')

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ beta, index=s.index, name='price_ds')


def price_spec_r2(log_price: pd.Series) -> None:
    """Print R² for each pipeline-faithful deseasonalization spec."""
    s = log_price.dropna()
    ss_tot = np.sum((s.values - s.values.mean()) ** 2)
    print('\n-- R² by deseasonalization spec (pipeline-faithful, incl. holidays & year) --')
    for spec, label in [
        ('basic',   'Basic   (hour + DOW + holiday + month + year)'),
        ('plus_hd', '+hxDOW  (hour×DOW + holiday + month + year) '),
        ('full',    'Full    (hour×DOW + hour×month + holiday + year)'),
    ]:
        resid = deseasonalize_price_spec(s, spec).values
        ss_res = np.sum(resid ** 2)
        r2 = 1 - ss_res / ss_tot
        n_params = {'basic': 1+23+6+1+11+1, 'plus_hd': 1+167+1+11+1, 'full': 1+167+287+1+1}[spec]
        print(f'  {label}  R2 = {r2:.4f}  ({r2*100:.1f}%)  [{n_params} params]')


def plot_price_acf_raw(log_price: pd.Series,
                       zone: str = ZONE,
                       start_date: str = START_DATE,
                       end_date: str = END_DATE) -> str:
    """
    Single-panel ACF of the basic-deseasonalized log price series (no AR correction).
    Shows the raw autocorrelation structure before any ARMA modelling.
    """
    from statsmodels.tsa.stattools import acf as sm_acf

    resid_ds = deseasonalize_price_spec(log_price, 'basic')
    nlags    = 200
    conf     = 1.96 / np.sqrt(len(resid_ds))
    acf_vals = sm_acf(resid_ds.values, nlags=nlags, fft=True)
    lags     = np.arange(len(acf_vals))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags[1:], acf_vals[1:], width=0.8, color='#4C8CBF', alpha=0.7)
    ax.axhline( conf, color='crimson', linestyle='--', linewidth=0.9, label='95% CI')
    ax.axhline(-conf, color='crimson', linestyle='--', linewidth=0.9)
    ax.axhline(0, color='black', linewidth=0.4)
    ax.axvline(24,  color='green',  linestyle='--', linewidth=1.2, label='Lag 24 (daily)')
    ax.axvline(168, color='orange', linestyle='--', linewidth=1.2, label='Lag 168 (weekly)')
    ax.set_xlabel('Lag (hours)', fontsize=9)
    ax.set_ylabel('ACF', fontsize=9)
    ax.set_xlim(0, nlags)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    var_dir = os.path.join(OUT_DIR, 'variable analysis')
    os.makedirs(var_dir, exist_ok=True)
    fpath = os.path.join(var_dir, f'price_acf_raw_deseason_{zone}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def plot_price_acf_comparison(log_price: pd.Series,
                               zone: str = ZONE,
                               start_date: str = START_DATE,
                               end_date: str = END_DATE) -> str:
    """
    Three-panel ACF plot of AR(1) residuals under progressively richer
    deseasonalization specs. Highlights lag 168 (weekly cycle).
    Returns path to saved figure.
    """
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.stattools import acf as sm_acf

    specs = [
        ('basic',   'Basic\n(hour + DOW + holiday\n+ month + year)'),
        ('plus_hd', '+ hour\u00d7DOW\ninteractions'),
        ('full',    'Full spec\n(+ hour\u00d7month\ninteractions)'),
    ]

    nlags = 200
    conf  = 1.96 / np.sqrt(len(log_price.dropna()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    def _draw_acf_panel(ax, acf_vals, conf, spec_label, show_ylabel=False):
        lags = np.arange(len(acf_vals))
        ax.bar(lags[1:], acf_vals[1:], width=0.8, color='#4C8CBF', alpha=0.7)
        ax.axhline( conf, color='crimson', linestyle='--', linewidth=0.9, label='95% CI')
        ax.axhline(-conf, color='crimson', linestyle='--', linewidth=0.9)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.axvline(24,  color='green',  linestyle='--', linewidth=1.2, label='Lag 24 (daily)')
        ax.axvline(168, color='orange', linestyle='--', linewidth=1.2, label='Lag 168 (weekly)')
        ax.set_xlabel('Lag (hours)', fontsize=8)
        ax.set_xlim(0, nlags)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7)
        if show_ylabel:
            ax.set_ylabel('ACF', fontsize=8)

    # Pre-compute ACF values for all specs
    acf_data = []
    for spec_key, spec_label in specs:
        resid_ds  = deseasonalize_price_spec(log_price, spec_key)
        ar1_resid = AutoReg(resid_ds.values, lags=1, old_names=False).fit().resid
        acf_vals  = sm_acf(ar1_resid, nlags=nlags, fft=True)
        acf_data.append((spec_key, spec_label, acf_vals))

    # Combined 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (spec_key, spec_label, acf_vals) in zip(axes, acf_data):
        _draw_acf_panel(ax, acf_vals, conf, spec_label, show_ylabel=(ax is axes[0]))
    fig.tight_layout()
    var_dir = os.path.join(OUT_DIR, 'variable analysis')
    os.makedirs(var_dir, exist_ok=True)
    combined_path = os.path.join(var_dir, f'price_acf_comparison_{zone}.png')
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Individual panel figures
    for spec_key, spec_label, acf_vals in acf_data:
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        _draw_acf_panel(ax_s, acf_vals, conf, spec_label, show_ylabel=True)
        fig_s.tight_layout()
        ind_path = os.path.join(var_dir, f'price_acf_{spec_key}_{zone}.png')
        fig_s.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close(fig_s)
        print(f'  Saved individual panel: {os.path.basename(ind_path)}')

    return combined_path


# -- Predefined variable analyses ----------------------------------------------─

def analyze_wind(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    analyze_variable(df['Wind_Forecast'], var_name='Wind Forecast',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)


def analyze_price(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    # Handle negatives before log (same as pipeline)
    price = df['Price'].copy()
    min_p = price.min()
    if min_p < 0.01:
        price = price + (0.01 - min_p)
    analyze_variable(price, var_name='Price',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)

    # Pipeline-faithful R² across deseasonalization specs
    log_price = np.log(price.dropna())
    price_spec_r2(log_price)

    # ACF of basic-deseasonalized series (no AR)
    raw_acf_path = plot_price_acf_raw(log_price, zone, start_date, end_date)
    print(f'\n-- ACF raw deseason plot saved: {os.path.basename(raw_acf_path)}')

    # ACF comparison across deseasonalization specs (AR(1) residuals)
    acf_path = plot_price_acf_comparison(log_price, zone, start_date, end_date)
    print(f'-- ACF comparison plot saved: {os.path.basename(acf_path)}')


def analyze_hydro(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    analyze_variable(df['Hydro_Reserves'], var_name='Hydro Reserves',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)


def analyze_consumption(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    analyze_variable(df['Consumption'], var_name='Consumption',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)


def analyze_net_exchange(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    # Bilateral exchange columns: not logged (can be negative)
    netexch_cols = sorted([c for c in df.columns if c.startswith('NetExch_')])
    for col in netexch_cols:
        partner = col.replace('NetExch_', '')
        analyze_variable(df[col], var_name=f'Net Exchange ({partner})',
                         apply_log=False, zone=zone,
                         start_date=start_date, end_date=end_date)


def analyze_oil(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    # Already lagged 24h by load_data
    analyze_variable(df['Oil_Price'], var_name='Oil Price (lagged 24h)',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)


def analyze_gas(zone=ZONE, start_date=START_DATE, end_date=END_DATE):
    df = load_raw(zone, start_date, end_date)
    # Already lagged 24h by load_data
    analyze_variable(df['Gas_Price'], var_name='Gas Price (lagged 24h)',
                     apply_log=True, zone=zone,
                     start_date=start_date, end_date=end_date)


# -- Outlier bounds plots -------------------------------------------------------

ZONES_ALL = ['SE1', 'SE2', 'SE3', 'SE4']
YEARS_ALL = list(range(2015, 2026))
SELECTED_YEARS = [2015, 2020, 2025]
SEASONAL_INTERACTIONS = 'hour_dow_month'


def plot_outlier_bounds(zone: str, year: int, sigma_lower: int = 3) -> str:
    """
    Run preprocessing steps 1-3 on a 1-year window and plot the deseasonalized
    price series with the outlier thresholds overlaid.
    Highlights observations that would be capped by step 4.

    sigma_lower: multiplier for the lower bound (default 3 = mean - 3σ).
                 Upper bound is always mean + 4σ.
    """
    start = f'{year}-01-01'
    end   = f'{year}-12-31'

    paths = {
        'combined':    f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
        'hydro':        'master data files/Master_Hydro_Reservoir.xlsx',
        'crude_oil':    'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
        'commodities':  'master data files/Master_Commodities.xlsx',
    }

    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths, target_region=zone, zone_hydro=zone,
            use_interpolation=True, start_date=start, end_date=end,
            lag_commodity_hours=24, use_bilateral_exchange=True,
        )
        df = handle_negative_prices(df_raw)
        df = apply_log_transform(df)
        df = deseasonalize_logged_variables(df, seasonal_interactions=SEASONAL_INTERACTIONS)

    price_ds = df['Price_DS']
    mean = price_ds.mean()
    std  = price_ds.std()
    upper = mean + 4 * std
    lower = mean - sigma_lower * std

    is_upper = price_ds > upper
    is_lower = price_ds < lower
    is_outlier = is_upper | is_lower
    n_upper = is_upper.sum()
    n_lower = is_lower.sum()
    n_total = is_outlier.sum()
    pct = (n_total / len(price_ds)) * 100

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(price_ds.index, price_ds.values, color='#4C8CBF', linewidth=0.4,
            alpha=0.7, label='Price (log, deseasonalized)')

    if n_total > 0:
        ax.scatter(price_ds.index[is_outlier], price_ds.values[is_outlier],
                   color='red', s=12, alpha=0.8, zorder=5,
                   label=f'Outliers ({n_total}: {n_upper} upper, {n_lower} lower)')

    ax.axhline(upper, color='orange', linestyle='--', linewidth=1.2,
               label=f'Upper bound (mean + 4$\\sigma$ = {upper:.3f})')
    ax.axhline(lower, color='orange', linestyle='--', linewidth=1.2,
               label=f'Lower bound (mean $-$ {sigma_lower}$\\sigma$ = {lower:.3f})')
    ax.axhline(mean, color='green', linestyle='-', linewidth=0.8, alpha=0.5,
               label=f'Mean = {mean:.3f}')

    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Price_DS', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    outlier_dir = os.path.join(OUT_DIR, 'outlier visualization')
    os.makedirs(outlier_dir, exist_ok=True)
    sigma_tag = f'4u_{sigma_lower}l'
    fname = f'outlier_bounds_{zone}_{year}_{sigma_tag}.png'
    fpath = os.path.join(outlier_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')

    if year in SELECTED_YEARS:
        sel_dir = os.path.join(outlier_dir, 'selected years')
        os.makedirs(sel_dir, exist_ok=True)
        fig.savefig(os.path.join(sel_dir, fname), dpi=150, bbox_inches='tight')

    plt.close(fig)

    print(f'  [{zone} {year}] {n_total} outliers ({pct:.2f}%): '
          f'{n_upper} upper, {n_lower} lower  ->  {fname}')
    return fpath


def run_outlier_bounds(sigma_lower: int = 3):
    """Generate outlier bounds plots for all zones and years."""
    print(f'\nOutlier bounds plots: {len(ZONES_ALL)} zones x {len(YEARS_ALL)} years'
          f'  (upper=4σ, lower={sigma_lower}σ)')
    for zone in ZONES_ALL:
        print(f'\n{zone}')
        for year in YEARS_ALL:
            try:
                plot_outlier_bounds(zone, year, sigma_lower=sigma_lower)
            except Exception as exc:
                print(f'  [{zone} {year}] ERROR: {exc}')


# -- Entry point ----------------------------------------------------------------

RUN_VARIABLE_ANALYSIS = False
RUN_OUTLIER_BOUNDS    = True
OUTLIER_SIGMA_LOWER   = 4

if __name__ == '__main__':
    print(f"Zone: {ZONE}  |  Period: {START_DATE} to {END_DATE}")
    print(f"Output directory: {OUT_DIR}\n")

    if RUN_VARIABLE_ANALYSIS:
        analyze_wind()
        analyze_price()
        analyze_hydro()
        analyze_consumption()
        analyze_net_exchange()
        analyze_oil()
        analyze_gas()

    if RUN_OUTLIER_BOUNDS:
        run_outlier_bounds(sigma_lower=OUTLIER_SIGMA_LOWER)
