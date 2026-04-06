"""
preprocessing2.py — No-log preprocessing pipeline.

Toggle: set LOG_PRICE = False in the relevant export script to use this module.

Difference from preprocessing.py:
  Stage 1  handle_negative_prices  — report only, NO shift applied
  Stage 2  apply_log_transform     — Price only: raw alias (no log)
                                     All other variables logged identically
  Stage 3  deseasonalize_logged_variables — identical (re-exported from preprocessing.py)
  Stage 4  cap_price_outliers            — no-op (skipped for now)

Price_DS is a levels-deseasonalized series (not log-deseasonalized).
Wind_Forecast_Log is log(Wind_Forecast), identical to preprocessing.py.
All other column names and transformations are identical to preprocessing.py.
"""

import io
import contextlib
import numpy as np

# Re-export shared objects — load_data and deseasonalize_logged_variables are identical
from preprocessing import (
    load_data,
    deseasonalize_logged_variables,
)


# ---------------------------------------------------------------------------
# Stage 1 — no-op: report only, never shift
# ---------------------------------------------------------------------------

def handle_negative_prices(df):
    """
    Report negative prices but do NOT shift the series.

    The shift in preprocessing.py distorts windows containing large negative
    prices (every observation gets lifted by the same large constant, breaking
    the level relationships used in deseasonalization).  In the no-log pipeline
    deseasonalization runs on raw levels, so negatives are valid inputs.
    """
    print("\n" + "="*80)
    print("NEGATIVE VALUE CHECK (NO-LOG PIPELINE — NO SHIFT APPLIED)")
    print("="*80)

    variables_to_check = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Consumption',
                          'Oil_Price', 'Gas_Price']
    found_negatives = False

    for var in variables_to_check:
        if var in df.columns:
            neg_count = (df[var] < 0).sum()
            if neg_count > 0:
                found_negatives = True
                pct = neg_count / len(df) * 100
                print(f"  {var}: {neg_count} negative values ({pct:.2f}%) — reported only, NOT modified")
        else:
            print(f"  {var}: NOT FOUND in dataframe")

    if not found_negatives:
        print("  [OK] No negative values found in any variable")

    print("  Exchange variables: NOT CHECKED (can have negative values)")
    print("  Price: NO SHIFT applied — raw levels preserved")
    print("="*80 + "\n")

    return df.copy()


# ---------------------------------------------------------------------------
# Stage 2 — partial alias: only Price and Wind skip the log
# ---------------------------------------------------------------------------

def apply_log_transform(df):
    """
    No-log pipeline: Price and Wind_Forecast are NOT logged (raw aliases).
    All other variables are logged identically to preprocessing.py.

    Price_Log and Wind_Forecast_Log are set to raw values so that
    deseasonalize_logged_variables (re-used unchanged) still finds the expected
    column names.  Hydro, Consumption, Oil, Gas are logged as normal.
    """
    print("\n--- APPLYING LOG TRANSFORM (NO-LOG PIPELINE) ---")
    print("Price: raw alias (no log); all other variables logged identically to log pipeline")
    print("Note: Oil & Gas prices are already lagged by 24h (automatically in load_data)")

    # Price — raw alias (no log; negatives are valid in levels pipeline)
    df['Price_Log'] = df['Price'].copy()
    print("Price_Log = Price  (raw level, no log)")

    # Wind — logged identically to preprocessing.py
    df['Wind_Forecast_Log'] = np.log(df['Wind_Forecast'].clip(lower=0.01))
    print("Wind_Forecast: log(raw) applied")

    # Everything else — actual log, identical to preprocessing.py
    df['Hydro_Reserves_Log'] = np.log(df['Hydro_Reserves'].clip(lower=0.01))
    print("Hydro_Reserves: log(raw) applied")

    df['Consumption_Log'] = np.log(df['Consumption'].clip(lower=0.01))
    print("Consumption: log(raw) applied")

    df['Oil_Price_Log'] = np.log(df['Oil_Price'].clip(lower=0.01))
    print("Oil_Price: log(raw) applied [USD/barrel]")

    df['Gas_Price_Log'] = np.log(df['Gas_Price'].clip(lower=0.01))
    print("Gas_Price: log(raw) applied [EUR/MWh]")

    print("Exchange variables: NOT logged (can contain negative values)")
    return df


# ---------------------------------------------------------------------------
# Stage 4 — no-op: outlier capping skipped for now
# ---------------------------------------------------------------------------

def cap_price_outliers(df):
    """
    Outlier capping is skipped in the no-log pipeline for now.
    Returns (df, {}) to match the signature of the log-pipeline version.
    """
    print("\n--- OUTLIER HANDLING SKIPPED (NO-LOG PIPELINE) ---")
    return df.copy(), {}


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def preprocess_data_for_regression(df_raw, suppress_output=False, seasonal_interactions='none'):
    """
    No-log preprocessing pipeline:
        1. handle_negative_prices        — report only, no shift
        2. apply_log_transform           — raw aliases, no actual log
        3. deseasonalize_logged_variables — identical to log pipeline
        4. cap_price_outliers            — no-op

    seasonal_interactions: 'none' | 'hour_dow' | 'hour_dow_month'
    """
    if suppress_output:
        stream = io.StringIO()
        context = contextlib.redirect_stdout(stream)
    else:
        context = contextlib.nullcontext()

    with context:
        data = handle_negative_prices(df_raw.copy())
        data = apply_log_transform(data)
        data = deseasonalize_logged_variables(data, seasonal_interactions=seasonal_interactions)
        data, _ = cap_price_outliers(data)

    return data
