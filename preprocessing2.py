import pandas as pd
import numpy as np
import io
import contextlib
import statsmodels.api as sm
import holidays

# Re-export unchanged functions from preprocessing so full_regression.py
# can import the same names from either module.
from preprocessing import (
    TRADING_PARTNERS,
    load_data,
    handle_outliers_gianfreda,
)


# --- 1. NO-OP NEGATIVE PRICE HANDLER ---

def handle_negative_prices(df):
    """
    Semi-log mode: price is used in levels, so no shift is needed.
    Negative prices are valid and are kept as-is.
    This function is a no-op — it exists only so full_regression.py
    can call the same step sequence regardless of LOG_PRICE mode.
    """
    print("\nSEMI-LOG MODE: handle_negative_prices skipped (price kept in levels)")
    return df


# --- 2. LOG TRANSFORMATION (price excluded) ---

def apply_log_transform(df):
    """
    Semi-log approach: log all variables EXCEPT Price.
    Price is kept in levels for deseasonalization.

    Logs applied to: Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    NOT logged: Price (semi-log mode), Net_Exchange (can be negative)
    """
    print("\n--- APPLYING LOGARITHMIC TRANSFORMATION (SEMI-LOG APPROACH) ---")
    print("Price is NOT logged — deseasonalization runs on raw price levels")
    print("Note: Oil & Gas prices are already lagged by 24h (automatically in load_data)")

    print(f"Price: NOT logged (semi-log mode)")

    df['Wind_Forecast_Log'] = np.log(df['Wind_Forecast'].clip(lower=0.01))
    print(f"Wind_Forecast: log(raw) applied")

    df['Hydro_Reserves_Log'] = np.log(df['Hydro_Reserves'].clip(lower=0.01))
    print(f"Hydro_Reserves: log(raw) applied")

    df['Consumption_Log'] = np.log(df['Consumption'].clip(lower=0.01))
    print(f"Consumption: log(raw) applied")

    print(f"Net_Exchange: NOT logged (contains negative values)")

    df['Oil_Price_Log'] = np.log(df['Oil_Price'].clip(lower=0.01))
    print(f"Oil_Price: log(raw) applied [USD/barrel]")

    df['Gas_Price_Log'] = np.log(df['Gas_Price'].clip(lower=0.01))
    print(f"Gas_Price: log(raw) applied [EUR/MWh]")

    return df


# --- 3. DESEASONALIZATION (price in levels, others in log space) ---

def deseasonalize_logged_variables(df):
    """
    Semi-log deseasonalization.

    Price is deseasonalized in levels (raw Price, not Price_Log) using FULL
    seasonal dummies. The residual is stored as Price_DS — the same column
    name used by the standard pipeline so all downstream code is unaffected.

    All other variables follow the standard approach (logged before deseasonalization).

    Seasonal dummies:
    - Price (levels) & Consumption_Log: FULL (Year + Month + DOW + Hour + Holiday)
    - Hydro_Log, Oil_Log, Gas_Log: PARTIAL (Year + Month only)

    Wind_Forecast_Log and Net_Exchange: NOT deseasonalized (following Fredriksson).
    """
    print("\n--- DESEASONALIZING VARIABLES (SEMI-LOG APPROACH) ---")
    print("Price deseasonalized in LEVELS; all other variables in log space")
    print("\nDeseasonalization strategy:")
    print("  - Price (levels) & Consumption_Log: Year + Month + DOW + Hour + Holiday (FULL)")
    print("  - Hydro_Log, Oil_Log, Gas_Log: Year + Month ONLY (PARTIAL)")

    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Hour'] = df.index.hour

    swedish_holidays = holidays.Sweden(years=range(df.index.year.min(), df.index.year.max() + 1))
    df['Holiday'] = df.index.to_series().apply(lambda x: 1 if x.date() in swedish_holidays else 0).values
    print("\nHoliday dummies created using Swedish holiday calendar")

    year_dummies  = pd.get_dummies(df['Year'],      prefix='Year',  drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['Month'],     prefix='Month', drop_first=True, dtype=float)
    dow_dummies   = pd.get_dummies(df['DayOfWeek'], prefix='DOW',   drop_first=True, dtype=float)
    hour_dummies  = pd.get_dummies(df['Hour'],      prefix='Hour',  drop_first=True, dtype=float)
    holiday_dummy = df[['Holiday']].astype(float)

    seasonal_dummies_full = pd.concat(
        [year_dummies, month_dummies, dow_dummies, hour_dummies, holiday_dummy], axis=1
    )
    seasonal_dummies_full = sm.add_constant(seasonal_dummies_full).astype(float)

    seasonal_dummies_partial = pd.concat([year_dummies, month_dummies], axis=1)
    seasonal_dummies_partial = sm.add_constant(seasonal_dummies_partial).astype(float)

    print("\n--- Deseasonalizing with FULL seasonal controls (Year+Month+DOW+Hour+Holiday) ---")

    # Price in LEVELS (semi-log: no log applied to price)
    price_model = sm.OLS(df['Price'], seasonal_dummies_full).fit()
    df['Price_DS'] = price_model.resid
    print(f"Price (levels): Seasonal R² = {price_model.rsquared:.4f}")
    print(f"  Original std: {df['Price'].std():.4f}, Deseasonalized std: {df['Price_DS'].std():.4f}")

    # Consumption_Log (FULL)
    consumption_log_model = sm.OLS(df['Consumption_Log'], seasonal_dummies_full).fit()
    df['Consumption_Log_Deseasonalized'] = consumption_log_model.resid
    print(f"Consumption_Log: Seasonal R² = {consumption_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Consumption_Log'].std():.4f}, Deseasonalized std: {df['Consumption_Log_Deseasonalized'].std():.4f}")

    print("\n--- Deseasonalizing with PARTIAL seasonal controls (Year+Month ONLY) ---")

    # Hydro_Reserves_Log (PARTIAL)
    hydro_log_model = sm.OLS(df['Hydro_Reserves_Log'], seasonal_dummies_partial).fit()
    df['Hydro_Reserves_Log_Deseasonalized'] = hydro_log_model.resid
    print(f"Hydro_Reserves_Log: Seasonal R² = {hydro_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Hydro_Reserves_Log'].std():.4f}, Deseasonalized std: {df['Hydro_Reserves_Log_Deseasonalized'].std():.4f}")

    # Oil_Price_Log (PARTIAL)
    oil_log_model = sm.OLS(df['Oil_Price_Log'], seasonal_dummies_partial).fit()
    df['Oil_Price_Log_Deseasonalized'] = oil_log_model.resid
    print(f"Oil_Price_Log: Seasonal R² = {oil_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Oil_Price_Log'].std():.4f}, Deseasonalized std: {df['Oil_Price_Log_Deseasonalized'].std():.4f}")

    # Gas_Price_Log (PARTIAL)
    gas_log_model = sm.OLS(df['Gas_Price_Log'], seasonal_dummies_partial).fit()
    df['Gas_Price_Log_Deseasonalized'] = gas_log_model.resid
    print(f"Gas_Price_Log: Seasonal R² = {gas_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Gas_Price_Log'].std():.4f}, Deseasonalized std: {df['Gas_Price_Log_Deseasonalized'].std():.4f}")

    df = df.drop(columns=['Year', 'Month', 'DayOfWeek', 'Hour', 'Holiday'])

    print("\nNote: Wind_Forecast_Log and Net_Exchange are NOT deseasonalized (following Fredriksson)")

    return df


# --- 4. PIPELINE WRAPPER ---

def preprocess_data_for_regression(df_raw, suppress_output=False):
    """
    Semi-log preprocessing sequence:
      1. handle_negative_prices — no-op (price stays in levels)
      2. apply_log_transform    — logs all variables except Price
      3. deseasonalize_logged_variables — deseasonalizes Price in levels → Price_DS
      4. handle_outliers_gianfreda     — weekday ±3σ cap on Price_DS
    """
    if suppress_output:
        stream = io.StringIO()
        context = contextlib.redirect_stdout(stream)
    else:
        context = contextlib.nullcontext()

    with context:
        data = handle_negative_prices(df_raw.copy())
        data = apply_log_transform(data)
        data = deseasonalize_logged_variables(data)
        data, _ = handle_outliers_gianfreda(data)

    return data
