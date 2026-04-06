"""
plot_rolling_results.py

Plots rolling-window coefficients from rolling_window_garch.do results
for all four zones on one chart per coefficient.

For the joint_t5_interact spec, produces:
  1. Normal-regime wind elasticity (β₁)           → rolling_wind_coef_normal_1yr_{suffix}.png
  2. Spike-regime wind elasticity  (β₁+β₃)        → rolling_wind_coef_spike_1yr_{suffix}.png
  3. Low-regime wind elasticity    (β₁+β₄)        → rolling_wind_coef_low_1yr_{suffix}.png
  4. Normal-regime variance effect (γ₁)           → rolling_wind_coef_var_normal_1yr_{suffix}.png
  5. Spike-regime variance effect  (γ₁+γ₃)        → rolling_wind_coef_var_spike_1yr_{suffix}.png
  6. Low-regime variance effect    (γ₁+γ₄)        → rolling_wind_coef_var_low_1yr_{suffix}.png

For other specs, produces:
  1. Wind coefficient in mean equation (β)         → rolling_wind_coef_mean_1yr_{suffix}.png
  2. Wind coefficient in variance equation (γ)     → rolling_wind_coef_var_1yr_{suffix}.png

- One line per zone (SE1-SE4)
- Shaded ±1.96·SE confidence band (lincom SEs for spike/low regimes)
- Gap in line where converged == 0
- Horizontal reference line at zero

Usage:
    python plot_rolling_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────

ZONES       = ['SE1', 'SE2', 'SE3', 'SE4']
RESULTS_DIR = 'output from stata/rolling_window_results'

# Choose specification to plot:
#   'joint_gaussian'    — original joint ARMAX-GARCH with Gaussian errors
#   'joint_t5'          — joint ARMAX-GARCH with Student-t(5) errors
#   'twostep_t5'        — two-step ARMAX then GARCH-X with Student-t(5) errors
#   'joint_t5_interact' — joint t(5) with spike/low regime interactions (dual regime)
SPEC = 'joint_t5_interact'

_SPEC_CONFIG = {
    'joint_gaussian': {
        'results_subdir': '',
        'file_pattern':   'rolling_garch_{zone}_1yr_hetfull.csv',
        'type_mean':      'coef',
        'type_var':       'var_coef',
        'file_suffix':    'joint_gaussian',
        'label':          'Joint Gaussian',
        'has_regimes':    False,
    },
    'joint_t5': {
        'results_subdir': 'joint_t',
        'file_pattern':   'rolling_garch_joint_t_{zone}_1yr_tdf5.csv',
        'type_mean':      'coef',
        'type_var':       'var_coef',
        'file_suffix':    'joint_t5',
        'label':          'Joint t(5)',
        'has_regimes':    False,
    },
    'twostep_t5': {
        'results_subdir': '2step',
        'file_pattern':   'rolling_garch_2step_{zone}_1yr_tdf5.csv',
        'type_mean':      'coef',
        'type_var':       'var_coef',
        'file_suffix':    'twostep_t5',
        'label':          'Two-step t(5)',
        'has_regimes':    False,
    },
    'joint_t5_interact': {
        'results_subdir': 'joint_t_interact',
        'file_pattern':   'rolling_garch_joint_t_interact_{zone}_1yr_tdf5.csv',
        'type_mean':      'elasticity',
        'type_var':       'var_coef',
        'file_suffix':    'joint_t5_interact',
        'label':          'Joint t(5) with Regime Interactions',
        'has_regimes':    True,
    },
}
_CFG = _SPEC_CONFIG[SPEC]

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
    Reads the long-format rolling GARCH results CSV and returns a wide
    DataFrame with one row per window.

    For has_regimes=True, columns:
        start_year, b_wind_normal, se_wind_normal,
        b_wind_spike, se_wind_spike,
        b_wind_low,   se_wind_low,
        b_het_wind_normal, se_het_wind_normal,
        b_het_wind_spike,  se_het_wind_spike,
        b_het_wind_low,    se_het_wind_low, converged

    For has_regimes=False, columns:
        start_year, b_wind, se_wind, b_het_wind, se_het_wind, converged

    converged == 0 (hard failure) → coefficients set to NaN (gap in plot).
    """
    subdir = _CFG['results_subdir']
    fname  = _CFG['file_pattern'].format(zone=zone)
    path   = os.path.join(RESULTS_DIR, subdir, fname) if subdir else os.path.join(RESULTS_DIR, fname)
    raw    = pd.read_csv(path)

    def extract(type_val, var_val, value_name, se_name):
        sub = raw[(raw['type'] == type_val) & (raw['variable'] == var_val)].copy()
        sub = sub[['start_year', 'value', 'se', 'converged']].rename(
            columns={'value': value_name, 'se': se_name}
        )
        fail = sub['converged'] == 0
        sub.loc[fail, [value_name, se_name]] = np.nan
        return sub.sort_values('start_year').reset_index(drop=True)

    if _CFG.get('has_regimes'):
        normal     = extract('elasticity',     'wind_normal', 'b_wind_normal',     'se_wind_normal')
        spike      = extract('elasticity',     'wind_spike',  'b_wind_spike',      'se_wind_spike')
        low        = extract('elasticity',     'wind_low',    'b_wind_low',        'se_wind_low')
        var_normal = extract('var_elasticity', 'wind_normal', 'b_het_wind_normal', 'se_het_wind_normal')
        var_spike  = extract('var_elasticity', 'wind_spike',  'b_het_wind_spike',  'se_het_wind_spike')
        var_low    = extract('var_elasticity', 'wind_low',    'b_het_wind_low',    'se_het_wind_low')
        df = normal
        for other in [
            spike[['start_year',     'b_wind_spike',      'se_wind_spike']],
            low[['start_year',       'b_wind_low',        'se_wind_low']],
            var_normal[['start_year','b_het_wind_normal', 'se_het_wind_normal']],
            var_spike[['start_year', 'b_het_wind_spike',  'se_het_wind_spike']],
            var_low[['start_year',   'b_het_wind_low',    'se_het_wind_low']],
        ]:
            df = df.merge(other, on='start_year', how='outer')
    else:
        var_wind  = extract(_CFG['type_var'], 'het_wind_log', 'b_het_wind', 'se_het_wind')
        mean_wind = extract(_CFG['type_mean'], 'wind_log', 'b_wind', 'se_wind')
        df = mean_wind.merge(
            var_wind[['start_year', 'b_het_wind', 'se_het_wind']],
            on='start_year', how='outer'
        )

    return df.sort_values('start_year').reset_index(drop=True)


# ── Generic plot function ─────────────────────────────────────────────────────

def plot_coefficient(coef_col: str, se_col: str, ylabel: str,
                     out_path: str, title_suffix: str = '') -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for zone in ZONES:
        try:
            df = load_zone(zone)
        except FileNotFoundError:
            print(f"  {zone}: results file not found — skipping")
            continue

        years = df['start_year'].values + 0.5
        b     = df[coef_col].values
        se    = df[se_col].values
        color = ZONE_COLORS[zone]

        ax.plot(years, b, color=color, linewidth=2,
                marker='o', markersize=5, label=zone)
        ax.fill_between(years,
                        b - 1.96 * se,
                        b + 1.96 * se,
                        color=color, alpha=0.07)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('')
    ax.legend(title='Zone', fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5, color='#aaaaaa')
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Terminal coefficient table ─────────────────────────────────────────────────

GARCH_RESULTS_DIR = 'output from stata/garch_results'
FULL_PERIOD_STUB  = '2015-01-01_2025-12-31_log'
YEARS             = list(range(2015, 2026))


def _load_zone_data(zone: str) -> dict:
    """
    Returns a dict of rolling result DataFrames (indexed by start_year,
    columns [value, stars]) plus a 'stat' dict for full-period estimates.

    has_regimes=True  keys: 'normal', 'spike', 'low', 'var_normal', 'var_spike', 'var_low', 'stat'
    has_regimes=False keys: 'mean', 'var', 'stat'
    """
    subdir    = _CFG['results_subdir']
    fname     = _CFG['file_pattern'].format(zone=zone)
    roll_path = os.path.join(RESULTS_DIR, subdir, fname) if subdir else os.path.join(RESULTS_DIR, fname)
    try:
        roll_raw = pd.read_csv(roll_path)
    except FileNotFoundError:
        print(f"  {zone}: results file not found — skipping table")
        empty = pd.DataFrame(columns=['value', 'stars'])
        if _CFG.get('has_regimes'):
            regime_keys = ('normal', 'spike', 'low', 'var_normal', 'var_spike', 'var_low')
            return {k: empty for k in regime_keys} | \
                   {'stat': {k: (None, '') for k in regime_keys}}
        else:
            return {'mean': empty, 'var': empty,
                    'stat': {'mean': (None, ''), 'var': (None, '')}}
    roll_raw['stars'] = roll_raw['stars'].fillna('')

    def get_roll(type_val, var_val):
        sub = roll_raw[(roll_raw['type'] == type_val) & (roll_raw['variable'] == var_val)]
        return sub.set_index('start_year')[['value', 'stars']]

    if _CFG.get('has_regimes'):
        regime_keys = ('normal', 'spike', 'low', 'var_normal', 'var_spike', 'var_low')
        result = {
            'normal':     get_roll('elasticity',     'wind_normal'),
            'spike':      get_roll('elasticity',     'wind_spike'),
            'low':        get_roll('elasticity',     'wind_low'),
            'var_normal': get_roll('var_elasticity', 'wind_normal'),
            'var_spike':  get_roll('var_elasticity', 'wind_spike'),
            'var_low':    get_roll('var_elasticity', 'wind_low'),
            'stat':       {k: (None, '') for k in regime_keys},
        }
    else:
        roll_mean = get_roll(_CFG['type_mean'], 'wind_log')
        roll_var  = get_roll(_CFG['type_var'],  'het_wind_log')

        # Overall row: placeholder until full-period joint_t5 estimates are available.
        # To populate: update garch_results_{zone}_{FULL_PERIOD_STUB}_hetfull.csv in
        # GARCH_RESULTS_DIR with a joint t(5) full-period run, then remove these two lines
        # and uncomment the block below.
        stat_mean = (None, '')
        stat_var  = (None, '')
        # stat_path = os.path.join(GARCH_RESULTS_DIR,
        #                          f'garch_results_{zone}_{FULL_PERIOD_STUB}_hetfull.csv')
        # try:
        #     stat_raw = pd.read_csv(stat_path)
        #     stat_raw['stars'] = stat_raw['stars'].fillna('')
        #     def get_stat(type_val, var_val):
        #         sub = stat_raw[(stat_raw['type'] == type_val) & (stat_raw['variable'] == var_val)]
        #         if sub.empty:
        #             return None, ''
        #         return sub['value'].values[0], sub['stars'].values[0]
        #     stat_mean = get_stat('coef',     'wind_log')
        #     stat_var  = get_stat('var_coef', 'het_wind_log')
        # except FileNotFoundError:
        #     stat_mean = (None, '')
        #     stat_var  = (None, '')

        result = {
            'mean': roll_mean,
            'var':  roll_var,
            'stat': {'mean': stat_mean, 'var': stat_var},
        }

    return result


def print_garch_tables() -> None:
    """
    Prints cross-zone coefficient tables to the terminal.
    For has_regimes=True: four panels (normal, spike, low, variance).
    For has_regimes=False: two panels (mean, variance).
    """
    data    = {zone: _load_zone_data(zone) for zone in ZONES}
    col_w   = 16
    total_w = 10 + col_w * len(ZONES)

    if _CFG.get('has_regimes'):
        panels = [
            ('normal',     'beta_wind   normal regime  (beta_1)'),
            ('spike',      'beta_wind   spike  regime  (beta_1 + beta_3, top 10%)'),
            ('low',        'beta_wind   low    regime  (beta_1 + beta_4, bottom 10%)'),
            ('var_normal', 'gamma_wind  normal regime  (gamma_1)'),
            ('var_spike',  'gamma_wind  spike  regime  (gamma_1 + gamma_3, top 10%)'),
            ('var_low',    'gamma_wind  low    regime  (gamma_1 + gamma_4, bottom 10%)'),
        ]
    else:
        panels = [
            ('mean', 'b_wind  --  mean equation'),
            ('var',  'g_wind  --  variance equation'),
        ]

    for eq, label in panels:
        print(f'\n{"="*total_w}')
        print(f'  {label}     *** p<0.01   ** p<0.05   * p<0.10')
        print(f'{"="*total_w}')
        print(f'{"Year":<10}' + ''.join(f'{z:>{col_w}}' for z in ZONES))
        print(f'{"-"*total_w}')

        for year in YEARS:
            row = f'{year:<10}'
            for zone in ZONES:
                src = data[zone].get(eq, pd.DataFrame())
                if not src.empty and year in src.index:
                    v, s = src.loc[year, 'value'], src.loc[year, 'stars']
                    cell = f'{v:.4f}{s}'
                else:
                    cell = '—'
                row += f'{cell:>{col_w}}'
            print(row)

        print(f'{"-"*total_w}')
        row = f'{"Overall":<10}'
        for zone in ZONES:
            val, stars = data[zone]['stat'].get(eq, (None, ''))
            cell = f'{val:.4f}{stars}' if val is not None else '—'
            row += f'{cell:>{col_w}}'
        print(row)
        print(f'{"="*total_w}')


# ── Run plots and tables ───────────────────────────────────────────────────────

if __name__ == '__main__':
    suffix  = _CFG['file_suffix']
    _subdir = _CFG.get('results_subdir', '')
    out_dir = os.path.join(RESULTS_DIR, _subdir) if _subdir else RESULTS_DIR
    plot_dir = os.path.join(out_dir, 'plots_v2')

    if _CFG.get('has_regimes'):
        for regime, coef_col, se_col, regime_label in [
            ('normal', 'b_wind_normal', 'se_wind_normal', 'normal regime (\u03b2\u2081)'),
            ('spike',  'b_wind_spike',  'se_wind_spike',  'spike regime (\u03b2\u2081+\u03b2\u2083, top 10%)'),
            ('low',    'b_wind_low',    'se_wind_low',    'low regime (\u03b2\u2081+\u03b2\u2084, bottom 10%)'),
        ]:
            plot_coefficient(
                coef_col     = coef_col,
                se_col       = se_col,
                ylabel       = '\u03b2  (log-price per log-wind unit)',
                out_path     = f'{plot_dir}/rolling_wind_coef_{regime}_1yr_{suffix}.png',
                title_suffix = regime_label,
            )
    else:
        plot_coefficient(
            coef_col = 'b_wind',
            se_col   = 'se_wind',
            ylabel   = '\u03b2  (log-price per log-wind unit)',
            out_path = f'{plot_dir}/rolling_wind_coef_mean_1yr_{suffix}.png',
        )

    # Variance equation
    if _CFG.get('has_regimes'):
        for regime, coef_col, se_col, regime_label in [
            ('var_normal', 'b_het_wind_normal', 'se_het_wind_normal', 'normal regime (\u03b3\u2081)'),
            ('var_spike',  'b_het_wind_spike',  'se_het_wind_spike',  'spike regime (\u03b3\u2081+\u03b3\u2083, top 10%)'),
            ('var_low',    'b_het_wind_low',    'se_het_wind_low',    'low regime (\u03b3\u2081+\u03b3\u2084, bottom 10%)'),
        ]:
            plot_coefficient(
                coef_col     = coef_col,
                se_col       = se_col,
                ylabel       = '\u03b3  (log-variance semi-elasticity w.r.t. log-wind)',
                out_path     = f'{plot_dir}/rolling_wind_coef_{regime}_1yr_{suffix}.png',
                title_suffix = regime_label,
            )
    else:
        plot_coefficient(
            coef_col = 'b_het_wind',
            se_col   = 'se_het_wind',
            ylabel   = '\u03b3  (log-variance semi-elasticity w.r.t. log-wind)',
            out_path = f'{plot_dir}/rolling_wind_coef_var_1yr_{suffix}.png',
        )

    # Coefficient tables printed to terminal
    print_garch_tables()
