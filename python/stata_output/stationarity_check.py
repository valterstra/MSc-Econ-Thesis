"""
stationarity_check.py

Reads armax_input_*_log.csv files exported by rolling_window_export.py or
rolling_window_export_monthly.py and runs ADF + DF-GLS stationarity tests
on every column.

Usage:
    python stationarity_check.py           # annual windows
    python stationarity_check.py monthly   # monthly windows
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS

BASE_DIR = Path(__file__).resolve().parents[2]

DIRS = {
    'annual':  os.path.join(BASE_DIR, 'stata_input', 'rolling_windows'),
    'monthly': os.path.join(BASE_DIR, 'stata_input', 'rolling_windows_monthly'),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_adf(series):
    res = adfuller(series.dropna(), autolag='AIC')
    return {
        'adf_stat':       round(res[0], 4),
        'adf_pval':       round(res[1], 4),
        'adf_lags':       res[2],
        'adf_stationary': res[1] < 0.05,
    }

def run_dfgls(series):
    try:
        res = DFGLS(series.dropna())
        return {
            'dfgls_stat':       round(float(res.stat), 4),
            'dfgls_pval':       round(float(res.pvalue), 4),
            'dfgls_stationary': float(res.pvalue) < 0.05,
        }
    except Exception:
        return {'dfgls_stat': None, 'dfgls_pval': None, 'dfgls_stationary': None}

# ── Main ───────────────────────────────────────────────────────────────────────

def main(mode='annual'):
    in_dir  = DIRS[mode]
    out_csv = os.path.join(in_dir, 'stationarity_summary.csv')

    # Matches both annual (2015-01-01_2015-12-31) and monthly (2015-02-01_2016-01-31)
    pattern = re.compile(r'armax_input_(SE\d)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_log\.csv$')

    files = sorted([f for f in os.listdir(in_dir) if pattern.match(f)])

    if not files:
        print(f"No matching files found in {in_dir}")
        return

    print(f"Mode: {mode} | Found {len(files)} files. Running stationarity tests...\n")

    rows = []
    for i, fname in enumerate(files, 1):
        m = pattern.match(fname)
        zone       = m.group(1)
        start_date = m.group(2)
        end_date   = m.group(3)

        df = pd.read_csv(os.path.join(in_dir, fname), index_col='timestamp', parse_dates=True)

        for col in df.columns:
            adf   = run_adf(df[col])
            dfgls = run_dfgls(df[col])
            rows.append({
                'zone': zone, 'start': start_date, 'end': end_date,
                'variable': col, **adf, **dfgls
            })

        print(f"  [{i}/{len(files)}] {zone} {start_date}->{end_date} ({len(df.columns)} vars)")

    summary = pd.DataFrame(rows, columns=[
        'zone', 'start', 'end', 'variable',
        'adf_stat', 'adf_pval', 'adf_lags', 'adf_stationary',
        'dfgls_stat', 'dfgls_pval', 'dfgls_stationary',
    ])

    summary.to_csv(out_csv, index=False)
    print(f"\nSummary written to {out_csv}")
    print(f"Total tests: {len(summary)}")

    flagged = summary[~summary['adf_stationary'] | ~summary['dfgls_stationary']]
    non_commodity = flagged[~flagged['variable'].isin(['oil_log', 'gas_log'])]

    print(f"\nNon-stationary cases (at least one test fails): {len(flagged)}")
    print(f"  of which oil_log/gas_log: {len(flagged) - len(non_commodity)}")
    print(f"  of which other variables: {len(non_commodity)}")

    if not non_commodity.empty:
        print(f"\nNon-oil/gas flagged cases:")
        print(non_commodity[['zone', 'start', 'end', 'variable', 'adf_pval', 'dfgls_pval']].to_string(index=False))


# ── Toggle ─────────────────────────────────────────────────────────────────────
# Set to 'annual' or 'monthly'
MODE = 'monthly'
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode not in DIRS:
        print(f"Unknown mode '{mode}'. Use 'annual' or 'monthly'.")
        sys.exit(1)
    main(mode)
