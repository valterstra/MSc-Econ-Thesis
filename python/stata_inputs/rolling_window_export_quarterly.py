"""
rolling_window_export_quarterly.py

Preprocesses overlapping 1-year windows stepped by 1 quarter for each of the
four Swedish electricity price zones (SE1–SE4) and exports one CSV per window
per zone for use in rolling-window GARCH-X Stata estimation.

Window design:
    Length : 1 year  (via dateutil.relativedelta, handles leap years)
    Step   : 1 quarter (3 months)
    First  : 2015-01-01 → 2015-12-31
    Last   : 2025-01-01 → 2025-12-31  (next step would exceed data end)
    Total  : 41 windows per zone

Each window is preprocessed independently using the same pipeline as
the shared preprocessing pipeline:
    1. handle_negative_prices   (shift based on window minimum)
    2. apply_log_transform
    3. deseasonalize_logged_variables  with SEASONAL_INTERACTIONS='hour_dow_month'
    4. cap_price_outliers               (global ±4σ Winsorization, window-specific σ)

Note on Year dummies: windows starting on Jan 1 (Q1 starts) fall entirely
within one calendar year → 0 year dummies. Windows starting in Q2/Q3/Q4 span
two calendar years → 1 year dummy column. Both cases are handled automatically
by pd.get_dummies(..., drop_first=True) inside deseasonalize_logged_variables.

Output (per zone × window):
    stata_input/rolling_windows_quarterly/armax_input_{ZONE}_{start}_{end}_{pipeline}.csv
    stata_input/rolling_windows_quarterly/armax_meta_{ZONE}_{start}_{end}_{pipeline}.csv

Usage:
    python rolling_window_export_quarterly.py
"""

import os
import sys
import io
import contextlib
from datetime import date, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Mirror the shared LOG_PRICE toggle used for preprocessing
# True  → log pipeline (preprocessing.py)
# False → levels pipeline (preprocessing2.py)
LOG_PRICE = True

if LOG_PRICE:
    from preprocessing import load_data, preprocess_data_for_regression
else:
    from preprocessing.preprocessing2 import load_data, preprocess_data_for_regression


# ── Configuration ─────────────────────────────────────────────────────────────

ZONES     = ['SE1', 'SE2', 'SE3', 'SE4']
DATA_END  = date(2025, 12, 31)          # last day with data
ARMA_ORDER = (1, 0, 1)                  # fixed p=1, d=0, q=1 for all windows
SEASONAL_INTERACTIONS = 'hour_dow_month'

# Generate windows: 1-year length, quarterly step, starting 2015-01-01
_start = date(2015, 1, 1)
WINDOWS = []
_current = _start
while True:
    _end = _current + relativedelta(years=1) - timedelta(days=1)
    if _end > DATA_END:
        break
    WINDOWS.append((_current, _end))
    _current += relativedelta(months=3)

# Stata-friendly column rename (bilateral NetExch_* handled dynamically)
STATA_RENAME = {
    'Price_DS':                            'price_ds',
    'Wind_Forecast_Log':                   'wind_log',
    'Hydro_Reserves_Log_Deseasonalized':   'hydro_log_ds',
    'Consumption_Log_Deseasonalized':      'consump_log_ds',
    'Oil_Price_Log':                       'oil_log',
    'Gas_Price_Log':                       'gas_log',
}

EXOG_VARS = [
    'Wind_Forecast_Log',
    'Hydro_Reserves_Log_Deseasonalized',
    'Consumption_Log_Deseasonalized',
    'Oil_Price_Log',
    'Gas_Price_Log',
]


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = str(PROJECT_ROOT)
OUT_DIR  = os.path.join(BASE_DIR, 'stata_input', 'rolling_windows_quarterly')
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}


# ── Main export function ───────────────────────────────────────────────────────

def export_window(zone: str, start: date, end: date) -> None:
    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')
    pipeline  = 'log' if LOG_PRICE else 'levels'

    csv_path  = os.path.join(OUT_DIR, f'armax_input_{zone}_{start_str}_{end_str}_{pipeline}.csv')
    meta_path = os.path.join(OUT_DIR, f'armax_meta_{zone}_{start_str}_{end_str}_{pipeline}.csv')

    if os.path.exists(csv_path) and os.path.exists(meta_path):
        print(f"  [{zone} {start_str}→{end_str}] already exists — skipping")
        return

    paths = {
        **SHARED_PATHS,
        'combined': f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
    }

    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths,
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=start_str,
            end_date=end_str,
            lag_commodity_hours=24,
            use_bilateral_exchange=True,
        )

    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    # Standard controls + bilateral exchange columns (NetExch_* → netexch_*)
    standard_exog = [v for v in EXOG_VARS if v in df_proc.columns]
    netexch_cols  = sorted([c for c in df_proc.columns if c.startswith('NetExch_')])
    available_exog = standard_exog + netexch_cols

    armax_input = pd.concat([df_proc['Price_DS'], df_proc[available_exog]], axis=1)
    armax_input.index.name = 'timestamp'

    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in armax_input.columns}
    rename_dict.update({col: col.lower() for col in netexch_cols})
    armax_input = armax_input.rename(columns=rename_dict)

    armax_input.to_csv(csv_path)

    p, d, q = ARMA_ORDER
    pd.DataFrame({
        'arma_p': [p], 'arma_d': [d], 'arma_q': [q], 'extra_ar_lags': ['']
    }).to_csv(meta_path, index=False)

    print(f"  [{zone} {start_str}→{end_str}] {len(armax_input)} obs → {os.path.basename(csv_path)}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    total  = len(ZONES) * len(WINDOWS)
    done   = 0
    errors = []

    print(f"Quarterly rolling window export: {len(ZONES)} zones × {len(WINDOWS)} windows = {total} exports")
    print(f"Window length: 1 year  |  Step: 1 quarter (3 months)")
    print(f"First window: {WINDOWS[0][0]} → {WINDOWS[0][1]}")
    print(f"Last window:  {WINDOWS[-1][0]} → {WINDOWS[-1][1]}")
    print(f"Seasonal interactions: {SEASONAL_INTERACTIONS}")
    print(f"Fixed ARMA order: {ARMA_ORDER}")
    print(f"Output directory: {OUT_DIR}\n")

    for zone in ZONES:
        print(f"\n{'='*60}")
        print(f"  Zone: {zone}")
        print(f"{'='*60}")
        for start, end in WINDOWS:
            try:
                export_window(zone, start, end)
                done += 1
            except Exception as exc:
                msg = f"[{zone} {start}→{end}] ERROR: {exc}"
                print(f"  {msg}")
                errors.append(msg)

    print(f"\n{'='*60}")
    print(f"Done: {done}/{total} windows exported successfully")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"{'='*60}")
