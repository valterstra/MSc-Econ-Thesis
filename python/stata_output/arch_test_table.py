"""
arch_test_table.py

Fits AR(1) to log-deseasonalized spot price (price_ds) for SE1-SE4,
then runs Engle's ARCH LM test on the residuals at multiple lag lengths.
Sample: 2024-2025.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

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
    return pd.concat(dfs).squeeze().dropna()


def fit_ar1_residuals(s):
    """OLS AR(1): regress s_t on s_{t-1} + constant, return residuals."""
    y = s.iloc[1:].values
    x = sm.add_constant(s.iloc[:-1].values)
    model = sm.OLS(y, x).fit()
    return model.resid


# Collect results: results[zone][lag] = (LM_stat, LM_pval)
results = {}
for zone in ZONES:
    s    = load_series(zone)
    resid = fit_ar1_residuals(s)
    results[zone] = {}
    for lag in LAGS:
        lm_stat, lm_pval, _, _ = het_arch(resid, nlags=lag)
        results[zone][lag] = (lm_stat, lm_pval)

# Print table
col_w = 22
header = f"{'Lag':<6}" + "".join(f"{z:<{col_w}}" for z in ZONES)
print(header)
print("-" * (6 + col_w * 4))
for lag in LAGS:
    row = f"{lag:<6}"
    for zone in ZONES:
        lm, p = results[zone][lag]
        cell  = f"{lm:,.2f}{stars(p)}"
        row  += f"{cell:<{col_w}}"
    print(row)

print()
print("Notes: LM = Engle ARCH LM statistic. H0: no ARCH effects.")
print("*** p<0.01, ** p<0.05, * p<0.10.")
print("AR(1) fitted by OLS; residuals tested. Sample: 2024-2025.")
