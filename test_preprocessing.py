"""
Smoke test for preprocessing.py

Loads SE1 data for 2022, runs the full preprocessing pipeline, and asserts
column correctness and value ranges. Run before and after any refactor to
confirm identical behaviour.

Usage:
    python test_preprocessing.py
"""

import sys
import traceback
import pandas as pd
import numpy as np

from preprocessing import load_data, preprocess_data_for_regression

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATHS = {
    'combined':   'master data files/2015-2025/Combined_SE1_Data_2015_2025.xlsx',
    'hydro':      'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':  'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}

REQUIRED_COLUMNS = [
    'Price_Log_Deseasonalized',
    'Wind_Forecast_Log',
    'Hydro_Reserves_Log_Deseasonalized',
    'Consumption_Log_Deseasonalized',
    'Oil_Price_Log_Deseasonalized',
    'Gas_Price_Log_Deseasonalized',
    'Net_Exchange',
]

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok  : {message}")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
print("=" * 70)
print("test_preprocessing.py — smoke test")
print("=" * 70)

try:
    print("\n[1] Loading SE1 data for 2022...")
    raw = load_data(
        PATHS,
        target_region='SE1',
        zone_hydro='SE1',
        start_date='2022-01-01',
        end_date='2022-12-31',
    )
    print(f"    Raw data loaded: {len(raw)} rows, {len(raw.columns)} columns")
except Exception as e:
    print(f"\nFATAL: load_data raised an exception:\n{traceback.format_exc()}")
    sys.exit(1)

try:
    print("\n[2] Running preprocess_data_for_regression (suppress_output=True)...")
    df = preprocess_data_for_regression(raw, suppress_output=True)
    print(f"    Preprocessed: {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"\nFATAL: preprocess_data_for_regression raised an exception:\n{traceback.format_exc()}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
print("\n[3] Assertions:")

check(isinstance(df.index, pd.DatetimeIndex),
      "Output has DatetimeIndex")

check(len(df) > 100,
      f"Row count > 100 (got {len(df)})")

for col in REQUIRED_COLUMNS:
    check(col in df.columns,
          f"Column '{col}' exists")

missing_counts = {col: df[col].isna().sum() for col in REQUIRED_COLUMNS if col in df.columns}
for col, n in missing_counts.items():
    check(n == 0,
          f"No NaN in '{col}' (found {n})")

if 'Price_Log_Deseasonalized' in df.columns:
    mean_val = df['Price_Log_Deseasonalized'].mean()
    check(abs(mean_val) < 0.5,
          f"Price_Log_Deseasonalized mean ≈ 0 (got {mean_val:.4f}, |mean| < 0.5 confirms deseasonalization)")

for col in REQUIRED_COLUMNS:
    if col in df.columns:
        s = df[col].std()
        check(s > 0,
              f"'{col}' is not constant (std = {s:.4f})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILURES:
    print(f"RESULT: FAIL ({len(FAILURES)} assertion(s) failed)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"RESULT: PASS (all {3 + len(REQUIRED_COLUMNS) * 3} checks passed)")
    sys.exit(0)
