"""
descriptive_stats.py

Prints descriptive statistics for the raw (untransformed, unpreprocessed)
2015-2025 dataset across all four Swedish price zones.

Layout
------
- Zone-specific variables: Spot_Price, Wind_Forecast, Consumption_Forecast,
  Net_Exchange, Hydro_Reserves  →  one row per variable, zones as column groups
- Common variables (bottom block): Oil_Price, Gas_Price  →  computed once from
  the raw source files (daily non-null observations)

Stats per variable: N, Mean, SD, Min, Max

Output: LaTeX tabular block printed to terminal.

Usage:
    python descriptive_stats.py
"""

import warnings
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

ZONES = ['SE1', 'SE2', 'SE3', 'SE4']

COMBINED_PATH = 'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx'
OIL_PATH      = 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx'
GAS_PATH      = 'master data files/Master_Commodities.xlsx'

# ── Variable config ────────────────────────────────────────────────────────────

ZONE_VARS = [
    ('Spot_Price',            r'Spot price (EUR/MWh)'),
    ('Wind_Forecast',         r'Wind forecast (MW)'),
    ('Consumption_Forecast',  r'Consumption forecast (MW)'),
    ('Net_Exchange',          r'Net exchange (MW)'),
    ('Hydro_Reserves',        r'Hydro reserves (GWh)'),
]

COMMON_VARS = [
    ('Oil_Price', r'Oil price (USD/bbl)'),
    ('Gas_Price', r'Gas price (EUR/MWh)'),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_zone(zone: str) -> pd.DataFrame:
    path = COMBINED_PATH.format(zone=zone)
    df = pd.read_excel(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    cols = ['Timestamp'] + [v for v, _ in ZONE_VARS if v in df.columns]
    return df[cols].set_index('Timestamp')


def load_oil() -> pd.Series:
    df = pd.read_excel(OIL_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Oil_Price'] = pd.to_numeric(df['Close'], errors='coerce')
    # Use unique daily close prices (drop NaN weekend/holiday hours)
    daily = (df.set_index('Timestamp')['Oil_Price']
               .resample('D').last()
               .dropna())
    daily = daily[(daily.index >= '2015-01-01') & (daily.index <= '2025-12-31')]
    return daily


def load_gas() -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_excel(GAS_PATH, header=None, skiprows=5)
    df.columns = ['Date', 'TTF_Gas', 'WTI_Oil', 'Brent_Oil', 'MT1', 'LUA1', 'CP1']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Gas_Price'] = pd.to_numeric(df['TTF_Gas'], errors='coerce')
    daily = df.set_index('Date')['Gas_Price'].dropna()
    daily = daily[(daily.index >= '2015-01-01') & (daily.index <= '2025-12-31')]
    return daily


# ── Stats ──────────────────────────────────────────────────────────────────────

def stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        'N':    len(s),
        'Mean': s.mean(),
        'SD':   s.std(),
        'Skew': s.skew(),
        'Kurt': s.kurtosis(),
        'Min':  s.min(),
        'Max':  s.max(),
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt(x, decimals=2) -> str:
    if pd.isna(x):
        return ''
    if abs(x) >= 1000:
        return f'{x:,.0f}'
    return f'{x:.{decimals}f}'

def fmt_n(n: int) -> str:
    return f'{n:,}'


# ── LaTeX output ───────────────────────────────────────────────────────────────

def print_latex(zone_stats: dict, common_stats: dict) -> None:
    n_zones = len(ZONES)
    n_cols  = n_zones * 7  # Mean SD Skew Kurt Min Max N per zone

    # Column spec: l + 7 r per zone
    col_spec = 'l' + ('rrrrrrr' * n_zones)

    print()
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Descriptive statistics, 2015--2025}')
    print(r'\label{tab:descriptive_stats}')
    print(r'\resizebox{\textwidth}{!}{%')
    print(r'\begin{tabular}{' + col_spec + r'}')
    print(r'\toprule')

    # Zone headers
    zone_header = ' & ' + ' & '.join(
        r'\multicolumn{7}{c}{' + z + '}' for z in ZONES
    ) + r' \\'
    print(zone_header)

    # Cmidrules
    cmidrules = ''
    for i, _ in enumerate(ZONES):
        start = 2 + i * 7
        end   = start + 6
        cmidrules += r'\cmidrule(lr){' + str(start) + '-' + str(end) + '}'
    print(cmidrules)

    # Sub-header
    sub = ' & ' + ' & '.join(['Mean & SD & Skew & Kurt & Min & Max & $N$'] * n_zones) + r' \\'
    print(sub)
    print(r'\midrule')

    # Zone-specific variables
    for col, label in ZONE_VARS:
        row = label
        for zone in ZONES:
            st = zone_stats[zone].get(col, {})
            if st:
                row += ' & ' + ' & '.join([
                    fmt(st['Mean']),
                    fmt(st['SD']),
                    fmt(st['Skew']),
                    fmt(st['Kurt']),
                    fmt(st['Min']),
                    fmt(st['Max']),
                    fmt_n(st['N']),
                ])
            else:
                row += ' & -- & -- & -- & -- & -- & -- & --'
        row += r' \\'
        print(row)

    # Common variables block
    print(r'\midrule')
    print(r'\multicolumn{' + str(1 + n_cols) + r'}{l}{\textit{Common across zones}} \\')

    for col, label in COMMON_VARS:
        st = common_stats.get(col, {})
        if not st:
            continue
        first_cell = ' & '.join([fmt(st['Mean']), fmt(st['SD']),
                                  fmt(st['Skew']), fmt(st['Kurt']),
                                  fmt(st['Min']), fmt(st['Max']),
                                  fmt_n(st['N'])])
        blank_cells = ' & '.join([''] * 7)
        row = label + ' & ' + first_cell
        for _ in ZONES[1:]:
            row += ' & ' + blank_cells
        row += r' \\'
        print(row)

    print(r'\bottomrule')
    print(r'\end{tabular}%')
    print(r'}')
    print(r'\end{table}')
    print()


# ── Plain-text summary ─────────────────────────────────────────────────────────

def print_plain(zone_stats: dict, common_stats: dict) -> None:
    col_w  = 12
    lbl_w  = 30
    header = f"{'Variable':<{lbl_w}}" + ''.join(
        f"{'':>{col_w}}{'SE'+z[2:]:<{col_w*4}}" for z in ZONES
    )
    sub = f"{'':>{lbl_w}}" + ''.join(
        f"{'Mean':>{col_w}}{'SD':>{col_w}}{'Skew':>{col_w}}{'Kurt':>{col_w}}{'Min':>{col_w}}{'Max':>{col_w}}"
        for _ in ZONES
    )
    sep = '-' * (lbl_w + col_w * 6 * len(ZONES))
    print()
    print(sep)
    print(header)
    print(sub)
    print(sep)

    for col, label in ZONE_VARS:
        row = f'{label:<{lbl_w}}'
        for zone in ZONES:
            st = zone_stats[zone].get(col, {})
            if st:
                row += (f"{fmt(st['Mean']):>{col_w}}"
                        f"{fmt(st['SD']):>{col_w}}"
                        f"{fmt(st['Skew']):>{col_w}}"
                        f"{fmt(st['Kurt']):>{col_w}}"
                        f"{fmt(st['Min']):>{col_w}}"
                        f"{fmt(st['Max']):>{col_w}}")
            else:
                row += f"{'--':>{col_w}}" * 6
        print(row)

    n_row = f"{'N (hourly obs.)':<{lbl_w}}"
    for zone in ZONES:
        n = zone_stats[zone].get('Spot_Price', {}).get('N', '')
        cell = fmt_n(n) if n else '--'
        n_row += f"{cell:>{col_w*6}}"
    print(n_row)

    print(sep)
    print(f"{'Common across zones':<{lbl_w}}")
    for col, label in COMMON_VARS:
        st = common_stats.get(col, {})
        if not st:
            continue
        freq = 'daily' if col in ('Oil_Price', 'Gas_Price') else 'hourly'
        row = f'{label:<{lbl_w}}'
        row += (f"{fmt(st['Mean']):>{col_w}}"
                f"{fmt(st['SD']):>{col_w}}"
                f"{fmt(st['Skew']):>{col_w}}"
                f"{fmt(st['Kurt']):>{col_w}}"
                f"{fmt(st['Min']):>{col_w}}"
                f"{fmt(st['Max']):>{col_w}}")
        print(row)
        print(f"{'  N (' + freq + ' obs.)':<{lbl_w}}{fmt_n(st['N']):>{col_w}}")
    print(sep)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')

    zone_stats = {}
    for zone in ZONES:
        print(f'  {zone}')
        df = load_zone(zone)
        zone_stats[zone] = {col: stats(df[col]) for col, _ in ZONE_VARS if col in df.columns}

    print('  Oil & Gas')
    oil = load_oil()
    gas = load_gas()
    common_stats = {
        'Oil_Price': stats(oil),
        'Gas_Price': stats(gas),
    }

    print_plain(zone_stats, common_stats)
    print_latex(zone_stats, common_stats)
