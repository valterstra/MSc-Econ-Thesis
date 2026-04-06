"""
full_period_export.py

Exports a single full-period (2015-01-01 → 2025-12-31) armax_input CSV
for each of the four Swedish electricity price zones (SE1–SE4).

To handle the structural break in price volatility around 2021-10-01, the
preprocessing is applied independently to each sub-period:

    Period A: 2015-01-01 → 2021-09-30  (own σ, own deseasonalization fit)
    Period B: 2021-10-01 → 2025-12-31  (own σ, own deseasonalization fit)

The two preprocessed halves are concatenated into one dataset and exported
as a single CSV for use in Stata (qr_from_python.do).

Output (per zone):
    stata_input/armax/armax_input_{ZONE}_2015-01-01_2025-12-31_log.csv
    stata_input/armax/armax_meta_{ZONE}_2015-01-01_2025-12-31_log.csv

Usage:
    python full_period_export.py
"""

import os
import sys
import io
import contextlib
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from preprocessing import load_data, preprocess_data_for_regression


# ── Configuration ──────────────────────────────────────────────────────────────

ZONES      = ['SE1', 'SE2', 'SE3', 'SE4']
START_DATE = date(2019, 1, 1)
END_DATE   = date(2019, 12, 31)
BREAK_DATE = None   # set to date(2021, 10, 1) for full 2015-2025 split; None = single preprocessing pass

ARMA_ORDER            = (1, 0, 1)           # stored in metadata only; actual lag order set in Stata
SEASONAL_INTERACTIONS = 'hour_dow_month'
PIPELINE              = 'log'


# ── Column configuration ───────────────────────────────────────────────────────

STATA_RENAME = {
    'Price_DS':                          'price_ds',
    'Wind_Forecast_Log':                 'wind_log',
    'Hydro_Reserves_Log_Deseasonalized': 'hydro_log_ds',
    'Consumption_Log_Deseasonalized':    'consump_log_ds',
    'Oil_Price_Log':                     'oil_log',
    'Gas_Price_Log':                     'gas_log',
}

EXOG_VARS = [
    'Wind_Forecast_Log',
    'Hydro_Reserves_Log_Deseasonalized',
    'Consumption_Log_Deseasonalized',
    'Oil_Price_Log',
    'Gas_Price_Log',
]


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = str(PROJECT_ROOT)
OUT_DIR  = os.path.join(BASE_DIR, 'stata_input', 'armax_garch')
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    'hydro':       'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil':   'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx',
}


# ── Export function ────────────────────────────────────────────────────────────

def export_zone(zone: str) -> None:
    start_str = START_DATE.strftime('%Y-%m-%d')
    end_str   = END_DATE.strftime('%Y-%m-%d')

    csv_path  = os.path.join(OUT_DIR, f'input_{zone}_{start_str}_{end_str}_{PIPELINE}.csv')
    meta_path = os.path.join(OUT_DIR, f'meta_{zone}_{start_str}_{end_str}_{PIPELINE}.csv')

    if os.path.exists(csv_path) and os.path.exists(meta_path):
        print(f'  [{zone}] already exists — skipping')
        return

    paths = {
        **SHARED_PATHS,
        'combined': f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx',
    }

    # Load full raw data in one call
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

    if BREAK_DATE is not None:
        # Split at break date and preprocess each half independently.
        # This ensures handle_negative_prices, deseasonalization, and
        # cap_price_outliers (±4σ) all use period-specific statistics.
        break_ts = pd.Timestamp(BREAK_DATE)
        df_a_raw = df_raw[df_raw.index < break_ts]
        df_b_raw = df_raw[df_raw.index >= break_ts]

        print(f'  [{zone}] Period A: {df_a_raw.index[0].date()} -> {df_a_raw.index[-1].date()}  ({len(df_a_raw)} obs)')
        print(f'  [{zone}] Period B: {df_b_raw.index[0].date()} -> {df_b_raw.index[-1].date()}  ({len(df_b_raw)} obs)')

        df_a = preprocess_data_for_regression(
            df_a_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )
        df_b = preprocess_data_for_regression(
            df_b_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )
        df_proc = pd.concat([df_a, df_b]).sort_index()
    else:
        print(f'  [{zone}] Single period: {df_raw.index[0].date()} -> {df_raw.index[-1].date()}  ({len(df_raw)} obs)')
        df_proc = preprocess_data_for_regression(
            df_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )

    # Build output columns: standard controls + any bilateral exchange columns
    standard_exog  = [v for v in EXOG_VARS if v in df_proc.columns]
    netexch_cols   = sorted([c for c in df_proc.columns if c.startswith('NetExch_')])
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

    print(f'  [{zone}] {len(armax_input)} total obs -> {os.path.basename(csv_path)}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Full-period export: {len(ZONES)} zones')
    print(f'Date range : {START_DATE} -> {END_DATE}')
    print(f'Break date : {BREAK_DATE if BREAK_DATE else "None (single preprocessing pass)"}')
    print(f'Output dir : {OUT_DIR}\n')

    errors = []
    for zone in ZONES:
        print(f'\n{zone}')
        try:
            export_zone(zone)
        except Exception as exc:
            msg = f'[{zone}] ERROR: {exc}'
            print(f'  {msg}')
            errors.append(msg)

    print('\nDone.')
    if errors:
        print(f'\nErrors ({len(errors)}):')
        for e in errors:
            print(f'  {e}')
