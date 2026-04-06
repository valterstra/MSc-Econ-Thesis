"""
ljungbox_table.py

Runs Ljung-Box test on the log-deseasonalized spot price (price_ds) for SE1-SE4,
using the 2024-2025 baseline window (annual CSV files concatenated).
Lags: 1, 4, 6, 8, 12. Prints results in tabular format.
"""

import os
from pathlib import Path
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE  = os.path.join(PROJECT_ROOT,
                     'stata_input', 'rolling_windows')
ZONES = ['SE1', 'SE2', 'SE3', 'SE4']
LAGS  = [1, 2, 3, 4, 5, 10, 15, 20]


def stars(pval):
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''


def load_series(zone):
    dfs = []
    for year in [2024, 2025]:
        fname = f'armax_input_{zone}_{year}-01-01_{year}-12-31_log.csv'
        path  = os.path.join(BASE, fname)
        df    = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        dfs.append(df[['price_ds']])
    return pd.concat(dfs).squeeze()


# Collect results: results[zone][lag] = (Q, pval)
results = {}
for zone in ZONES:
    s   = load_series(zone).dropna()
    lb  = acorr_ljungbox(s, lags=LAGS, return_df=True)
    results[zone] = {
        lag: (lb.loc[lag, 'lb_stat'], lb.loc[lag, 'lb_pvalue'])
        for lag in LAGS
    }

# Print table
col_w = 22
header = f"{'Lag':<6}" + "".join(f"{z:<{col_w}}" for z in ZONES)
print(header)
print("-" * (6 + col_w * 4))
for lag in LAGS:
    row = f"{lag:<6}"
    for zone in ZONES:
        q, p = results[zone][lag]
        cell = f"{q:,.2f}{stars(p)}"
        row += f"{cell:<{col_w}}"
    print(row)

print()
print("Notes: Q = Ljung-Box statistic. *** p<0.01, ** p<0.05, * p<0.10.")
print("Series: log-deseasonalized spot price (price_ds). Sample: 2024-2025.")
