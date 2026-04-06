"""
rolling_window_export.py

Preprocesses 11 non-overlapping 1-year windows (2015–2025) for each of the
four Swedish electricity price zones (SE1–SE4) and exports one CSV per window
per zone for use in the rolling-window GARCH-X Stata estimation.

Each window is preprocessed independently using the same pipeline as
the shared preprocessing pipeline:
    1. handle_negative_prices   (shift based on window minimum)
    2. apply_log_transform
    3. deseasonalize_logged_variables  with SEASONAL_INTERACTIONS='hour_dow_month'
    4. cap_price_outliers               (global ±4σ Winsorization, window-specific σ)

Note on Year dummies: with a 1-year window, pd.get_dummies(..., drop_first=True)
on Year produces zero columns → Year dummies are automatically absent, which is
correct (no cross-year variation to remove).

Output:
    output/rolling_windows/armax_input_{ZONE}_{YEAR}-01-01_{YEAR}-12-31.csv
    output/rolling_windows/armax_meta_{ZONE}_{YEAR}-01-01_{YEAR}-12-31.csv

Usage:
    python rolling_window_export.py
"""

import os
import sys
import io
import contextlib
from pathlib import Path

import pandas as pd

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

ZONES        = ['SE1', 'SE2', 'SE3', 'SE4']
WINDOW_YEARS = 1                          # length of each window in years
STEP_YEARS   = 1                          # slide step in years
START_YEAR   = 2015
END_YEAR     = 2025
# Windows: start years 2015 … 2025, each spanning 1 year → 11 windows
WINDOWS = [
    (y, y + WINDOW_YEARS - 1)
    for y in range(START_YEAR, END_YEAR - WINDOW_YEARS + 2, STEP_YEARS)
]
ARMA_ORDER = (1, 0, 1)                    # fixed p=1, d=0, q=1 for all windows
SEASONAL_INTERACTIONS = 'hour_dow_month'

# --- Exchange & control specification ----------------------------------------
# True  → individual bilateral flows (NetExch_*) replace total Net_Exchange
# False → total Net_Exchange (single regressor); appends '_netexch' to filenames
USE_BILATERAL_EXCHANGE = False

CONTROLS = {
    'Wind':        True,
    'Hydro':       False,
    'Consumption': True,
    'Oil':         False,
    'Gas':         False,
}
# -----------------------------------------------------------------------------

# Stata-friendly column names (≤32 chars, no ambiguity)
# NetExch_* bilateral columns are renamed dynamically to lowercase (e.g. NetExch_DK2 → netexch_dk2)
STATA_RENAME = {
    'Price_DS':                            'price_ds',
    'Wind_Forecast_Log':                   'wind_log',
    'Hydro_Reserves_Log_Deseasonalized':   'hydro_log_ds',
    'Consumption_Log_Deseasonalized':      'consump_log_ds',
    'Oil_Price_Log':                       'oil_log',
    'Gas_Price_Log':                       'gas_log',
}

# Map CONTROLS keys to column names in the preprocessed dataframe
_CONTROL_COLS = {
    'Wind':        'Wind_Forecast_Log',
    'Hydro':       'Hydro_Reserves_Log_Deseasonalized',
    'Consumption': 'Consumption_Log_Deseasonalized',
    'Oil':         'Oil_Price_Log',
    'Gas':         'Gas_Price_Log',
}

# Standard exogenous variables derived from CONTROLS (bilateral exchange added dynamically)
EXOG_VARS = [_CONTROL_COLS[k] for k, v in CONTROLS.items() if v]

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = str(PROJECT_ROOT)
OUT_DIR  = os.path.join(BASE_DIR, 'stata_input', 'rolling_windows')
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}


# ── Main loop ──────────────────────────────────────────────────────────────────

def export_window(zone: str, start_year: int, end_year: int) -> None:
    start = f'{start_year}-01-01'
    end   = f'{end_year}-12-31'

    pipeline     = 'log' if LOG_PRICE else 'levels'
    exch_suffix  = '' if USE_BILATERAL_EXCHANGE else '_netexch'
    file_suffix  = f'{pipeline}{exch_suffix}'
    csv_path  = os.path.join(OUT_DIR, f'armax_input_{zone}_{start}_{end}_{file_suffix}.csv')
    meta_path = os.path.join(OUT_DIR, f'armax_meta_{zone}_{start}_{end}_{file_suffix}.csv')

    # Skip if already exported
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        print(f"  [{zone} {start_year}–{end_year}] already exists — skipping")
        return

    paths = {
        **SHARED_PATHS,
        'combined': f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
    }

    # Load raw data for this window (suppress verbose load_data output)
    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths,
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=start,
            end_date=end,
            lag_commodity_hours=24,
            use_bilateral_exchange=USE_BILATERAL_EXCHANGE,
        )

    # Preprocess independently on this window
    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    # Keep only the columns needed for Stata regression
    standard_exog = [v for v in EXOG_VARS if v in df_proc.columns]
    if USE_BILATERAL_EXCHANGE:
        netexch_cols = sorted([c for c in df_proc.columns if c.startswith('NetExch_')])
    else:
        netexch_cols = ['Net_Exchange'] if 'Net_Exchange' in df_proc.columns else []
    available_exog = standard_exog + netexch_cols

    armax_input = pd.concat([df_proc['Price_DS'], df_proc[available_exog]], axis=1)
    armax_input.index.name = 'timestamp'

    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in armax_input.columns}
    if USE_BILATERAL_EXCHANGE:
        rename_dict.update({col: col.lower() for col in netexch_cols})
    else:
        rename_dict['Net_Exchange'] = 'net_exchange'
    armax_input = armax_input.rename(columns=rename_dict)

    # Export data CSV
    armax_input.to_csv(csv_path)

    # Export metadata CSV (fixed ARMA order)
    p, d, q = ARMA_ORDER
    pd.DataFrame({
        'arma_p': [p], 'arma_d': [d], 'arma_q': [q], 'extra_ar_lags': ['']
    }).to_csv(meta_path, index=False)

    print(f"  [{zone} {start_year}–{end_year}] {len(armax_input)} obs → {os.path.basename(csv_path)}")


if __name__ == '__main__':
    total = len(ZONES) * len(WINDOWS)
    done  = 0
    errors = []

    _pipeline    = 'log' if LOG_PRICE else 'levels'
    _exch_suffix = '' if USE_BILATERAL_EXCHANGE else '_netexch'
    print(f"Rolling window export: {len(ZONES)} zones × {len(WINDOWS)} windows = {total} exports")
    print(f"Window size: {WINDOW_YEARS} years  |  Step: {STEP_YEARS} year")
    print(f"Windows: {WINDOWS[0][0]}–{WINDOWS[0][1]} … {WINDOWS[-1][0]}–{WINDOWS[-1][1]}")
    print(f"Seasonal interactions: {SEASONAL_INTERACTIONS}")
    print(f"Fixed ARMA order: {ARMA_ORDER}")
    print(f"Controls: {[k for k, v in CONTROLS.items() if v]}  |  Exchange: {'bilateral' if USE_BILATERAL_EXCHANGE else 'net_exchange'}")
    print(f"File suffix: _{_pipeline}{_exch_suffix}")
    print(f"Output directory: {OUT_DIR}\n")

    for zone in ZONES:
        print(f"\n{'='*60}")
        print(f"  Zone: {zone}")
        print(f"{'='*60}")
        for start_year, end_year in WINDOWS:
            try:
                export_window(zone, start_year, end_year)
                done += 1
            except Exception as exc:
                msg = f"[{zone} {start_year}–{end_year}] ERROR: {exc}"
                print(f"  {msg}")
                errors.append(msg)

    print(f"\n{'='*60}")
    print(f"Done: {done}/{total} windows exported successfully")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"{'='*60}")
