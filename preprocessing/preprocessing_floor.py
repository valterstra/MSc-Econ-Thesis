"""
preprocessing_floor.py - Capped-price log robustness pipeline.

This variant is for the negative-price sensitivity check. It keeps the hourly
panel intact, does not shift the full price series, and logs price after
flooring only the non-positive/very-low price tail at 0.01.

Differences from preprocessing/__init__.py:
  Stage 1  handle_negative_prices  - report only, no shift
  Stage 2  apply_log_transform     - Price_Log = log(Price clipped at 0.01)
                                     Other variables logged as in baseline
  Stage 3  deseasonalize_logged_variables - baseline implementation
  Stage 4  cap_price_outliers      - baseline implementation
"""

import contextlib
import io

import numpy as np

from preprocessing import (
    load_data,
    deseasonalize_logged_variables,
    cap_price_outliers,
)


PRICE_FLOOR = 0.01


def handle_negative_prices(df):
    """
    Report negative/non-positive prices but do not modify the raw Price column.

    The floor is applied only when constructing Price_Log. Keeping Price raw here
    makes the transformation explicit and avoids changing any level variable
    before the log step.
    """
    print("\n" + "=" * 80)
    print("NEGATIVE VALUE CHECK (PRICE-FLOOR ROBUSTNESS - NO SHIFT)")
    print("=" * 80)

    variables_to_check = [
        "Price",
        "Wind_Forecast",
        "Hydro_Reserves",
        "Consumption",
        "Oil_Price",
        "Gas_Price",
    ]
    found_negatives = False

    for var in variables_to_check:
        if var not in df.columns:
            print(f"  {var}: NOT FOUND in dataframe")
            continue

        min_val = df[var].min()
        neg_count = (df[var] < 0).sum()
        zero_count = (df[var] == 0).sum()
        floor_count = (df[var] < PRICE_FLOOR).sum()

        if neg_count > 0 or zero_count > 0 or floor_count > 0:
            found_negatives = True
            print(f"  {var}:")
            print(f"    Min value: {min_val:.4f}")
            print(f"    Negative values: {neg_count} ({(neg_count / len(df)) * 100:.2f}%)")
            print(f"    Zero values: {zero_count} ({(zero_count / len(df)) * 100:.2f}%)")
            if var == "Price":
                print(
                    f"    Values below {PRICE_FLOOR}: {floor_count} "
                    f"({(floor_count / len(df)) * 100:.2f}%)"
                )

    if not found_negatives:
        print("  [OK] No negative, zero, or below-floor values found")

    print("  Exchange variables: NOT CHECKED (can have negative values)")
    print(f"  Price: NO SHIFT applied; log step floors Price at {PRICE_FLOOR}")
    print("=" * 80 + "\n")

    return df.copy()


def apply_log_transform(df):
    """
    Apply log transforms for the capped-price robustness pipeline.

    Price is logged as log(max(Price, 0.01)). Other logged variables follow the
    baseline pipeline's clipping behavior.
    """
    print("\n--- APPLYING LOG TRANSFORM (PRICE-FLOOR ROBUSTNESS) ---")
    print(f"Price: log(Price clipped at {PRICE_FLOOR}) applied")
    print("Other logged variables follow the baseline log pipeline")
    print("Exchange variables: NOT logged (can contain negative values)")

    df["Price_Log"] = np.log(df["Price"].clip(lower=PRICE_FLOOR))

    df["Wind_Forecast_Log"] = np.log(df["Wind_Forecast"].clip(lower=PRICE_FLOOR))
    print("Wind_Forecast: log(raw) applied")

    df["Hydro_Reserves_Log"] = np.log(df["Hydro_Reserves"].clip(lower=PRICE_FLOOR))
    print("Hydro_Reserves: log(raw) applied")

    df["Consumption_Log"] = np.log(df["Consumption"].clip(lower=PRICE_FLOOR))
    print("Consumption: log(raw) applied")

    df["Oil_Price_Log"] = np.log(df["Oil_Price"].clip(lower=PRICE_FLOOR))
    print("Oil_Price: log(raw) applied [USD/barrel]")

    df["Gas_Price_Log"] = np.log(df["Gas_Price"].clip(lower=PRICE_FLOOR))
    print("Gas_Price: log(raw) applied [EUR/MWh]")

    return df


def preprocess_data_for_regression(df_raw, suppress_output=False, seasonal_interactions="none"):
    """
    Capped-price robustness preprocessing pipeline:
        1. report negative prices, no shift
        2. Price_Log = log(Price clipped at 0.01); other logs as baseline
        3. deseasonalize logged variables
        4. cap Price_DS outliers as in baseline

    seasonal_interactions: 'none' | 'hour_dow' | 'hour_dow_month'
    """
    if suppress_output:
        context = contextlib.redirect_stdout(io.StringIO())
    else:
        context = contextlib.nullcontext()

    with context:
        data = handle_negative_prices(df_raw.copy())
        data = apply_log_transform(data)
        data = deseasonalize_logged_variables(
            data,
            seasonal_interactions=seasonal_interactions,
        )
        data, _ = cap_price_outliers(data)

    return data
