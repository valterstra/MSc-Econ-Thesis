"""
preprocessing3.py — Monthly-window levels preprocessing pipeline.

Designed for short (~720 obs) monthly rolling windows where year/month
dummies are meaningless (single value per window).

Differences from preprocessing2.py:
  Stage 3  deseasonalize_logged_variables — monthly spec:
             Price & Consumption: DOW + Hour + Holiday  (30 regressors)
             Hydro, Oil, Gas:     constant only          (subtract mean)
           No Year, Month, or interaction terms.

Everything else is identical to preprocessing2.py:
  Stage 1  handle_negative_prices  — report only, no shift
  Stage 2  apply_log_transform     — Price raw alias; everything else logged
  Stage 4  cap_price_outliers — no-op
"""

import io
import contextlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import holidays

# Re-export shared objects
from preprocessing import (
    TRADING_PARTNERS,
    load_data,
)
from preprocessing.preprocessing2 import (
    handle_negative_prices,
    apply_log_transform,
    cap_price_outliers,
)


# ---------------------------------------------------------------------------
# Stage 3 — monthly deseasonalization
# ---------------------------------------------------------------------------

def deseasonalize_logged_variables(df, seasonal_interactions=None):
    """
    Monthly-window deseasonalization.

    seasonal_interactions is accepted but ignored — the spec is fixed:
      Price & Consumption : DOW (6) + Hour (23) + Holiday (1) + constant
      Hydro, Oil, Gas     : constant only (subtract window mean)

    No Year, Month, or interaction dummies — all meaningless within a
    single calendar month.
    """
    print("\n--- DESEASONALIZING (MONTHLY PIPELINE) ---")
    print("Price & Consumption: DOW + Hour + Holiday")
    print("Hydro, Oil, Gas:     constant only (subtract mean)")

    df['DayOfWeek'] = df.index.dayofweek
    df['Hour']      = df.index.hour

    swedish_holidays = holidays.Sweden(
        years=range(df.index.year.min(), df.index.year.max() + 1)
    )
    df['Holiday'] = df.index.to_series().apply(
        lambda x: 1 if x.date() in swedish_holidays else 0
    ).values

    dow_dummies  = pd.get_dummies(df['DayOfWeek'], prefix='DOW',  drop_first=True, dtype=float)
    hour_dummies = pd.get_dummies(df['Hour'],      prefix='Hour', drop_first=True, dtype=float)
    holiday_col  = df[['Holiday']].astype(float)

    # FULL monthly spec: DOW + Hour + Holiday + constant
    full_dummies = pd.concat([dow_dummies, hour_dummies, holiday_col], axis=1)
    full_dummies = sm.add_constant(full_dummies).astype(float)

    # PARTIAL monthly spec: constant only
    const_only = sm.add_constant(pd.DataFrame(index=df.index)).astype(float)

    # Price_Log (raw alias of Price) → FULL
    price_model = sm.OLS(df['Price_Log'], full_dummies).fit()
    df['Price_DS'] = price_model.resid
    print(f"Price (levels): R² = {price_model.rsquared:.4f}  "
          f"std {df['Price_Log'].std():.3f} → {df['Price_DS'].std():.3f}")

    # Consumption_Log → FULL
    consump_model = sm.OLS(df['Consumption_Log'], full_dummies).fit()
    df['Consumption_Log_Deseasonalized'] = consump_model.resid
    print(f"Consumption_Log: R² = {consump_model.rsquared:.4f}")

    # Hydro_Reserves_Log → constant only
    hydro_model = sm.OLS(df['Hydro_Reserves_Log'], const_only).fit()
    df['Hydro_Reserves_Log_Deseasonalized'] = hydro_model.resid
    print(f"Hydro_Reserves_Log: constant only (subtract mean)")

    # Oil_Price_Log → constant only
    oil_model = sm.OLS(df['Oil_Price_Log'], const_only).fit()
    df['Oil_Price_Log_Deseasonalized'] = oil_model.resid
    print(f"Oil_Price_Log: constant only (subtract mean)")

    # Gas_Price_Log → constant only
    gas_model = sm.OLS(df['Gas_Price_Log'], const_only).fit()
    df['Gas_Price_Log_Deseasonalized'] = gas_model.resid
    print(f"Gas_Price_Log: constant only (subtract mean)")

    df = df.drop(columns=['DayOfWeek', 'Hour', 'Holiday'])

    print("Wind_Forecast_Log, Net_Exchange: not deseasonalized")
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def preprocess_data_for_regression(df_raw, suppress_output=False,
                                   seasonal_interactions=None):
    """
    Monthly levels preprocessing pipeline:
        1. handle_negative_prices        — report only, no shift
        2. apply_log_transform           — Price raw alias, others logged
        3. deseasonalize_logged_variables — DOW + Hour + Holiday only
        4. cap_price_outliers            — no-op
    """
    if suppress_output:
        context = contextlib.redirect_stdout(io.StringIO())
    else:
        context = contextlib.nullcontext()

    with context:
        data = handle_negative_prices(df_raw.copy())
        data = apply_log_transform(data)
        data = deseasonalize_logged_variables(data)
        data, _ = cap_price_outliers(data)

    return data
