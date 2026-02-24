"""
################################################################################
#  [Module 02/10]  data_loading.py  –  Data Loading
#
#  Contains:
#    - load_data()   : loads combined regional data and merges hydro/commodity
#                      files, applies 24h commodity lag for day-ahead alignment
#
#  Dependencies: config
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings

from .config import TRADING_PARTNERS


# --- 1. DATA LOADING FUNCTIONS ---


def load_data(paths, target_region='SE1', zone_hydro='SE1', use_interpolation=False,
                                 start_date=None, end_date=None, lag_commodity_hours=24):
    """
    Load pre-combined regional data file and merge with hydro reserves and commodities.

    This function loads the Combined_{region}_Data file which already contains:
    - Timestamp, Spot_Price, Wind_Forecast, Net_Exchange, Consumption_Forecast
    - Bottleneck dummies (BNECK_{region}_{partner})

    It then separately merges:
    - Hydro reserves from Master_Hydro_Reservoir.xlsx
    - Light Crude Oil (hourly) from Light_Crude_Oil_2015_2025.xlsx
    - TTF Gas prices (daily) from Master_Commodities.xlsx
    - Automatically lags commodity prices to align with day-ahead market timing

    Parameters:
    - paths: dict with keys 'combined', 'hydro', 'crude_oil', 'commodities'
    - target_region: Target region for analysis (default 'SE1')
    - zone_hydro: Zone for hydro reserves (default 'SE1')
    - use_interpolation: If True, interpolate missing values; if False, drop rows with NaN
    - start_date: Optional start date filter (e.g., '2021-01-01')
    - end_date: Optional end date filter (e.g., '2024-12-31')
    - lag_commodity_hours: Hours to lag commodity prices (default 24 for day-ahead market)

    Returns:
    - DataFrame indexed by Datetime with columns:
      Price, Wind_Forecast, Hydro_Reserves, Net_Exchange, Consumption, Oil_Price, Gas_Price,
      plus bottleneck dummies (BNECK_{region}_{partner})
      Note: Oil_Price (Light Crude Close) and Gas_Price (TTF) are automatically lagged by lag_commodity_hours
    """
    print("\n--- LOADING COMBINED REGIONAL DATA ---")

    # Step 1: Load combined file
    print(f"Loading combined data from: {paths['combined']}")
    df_combined = pd.read_excel(paths['combined'])

    # Map columns to standard names
    final_df = pd.DataFrame({
        'Datetime': pd.to_datetime(df_combined['Timestamp']),
        'Price': pd.to_numeric(df_combined['Spot_Price'], errors='coerce'),
        'Wind_Forecast': pd.to_numeric(df_combined['Wind_Forecast'], errors='coerce'),
        'Net_Exchange': pd.to_numeric(df_combined['Net_Exchange'], errors='coerce'),
        'Consumption': pd.to_numeric(df_combined['Consumption_Forecast'], errors='coerce')
    })

    # Load bottleneck dummy variables (if present in combined file)
    trading_partners = TRADING_PARTNERS.get(target_region, [])
    bneck_cols = []

    for partner in trading_partners:
        bneck_col = f'BNECK_{target_region}_{partner}'
        if bneck_col in df_combined.columns:
            final_df[bneck_col] = pd.to_numeric(df_combined[bneck_col], errors='coerce')
            bneck_cols.append(bneck_col)
            print(f"    Loaded bottleneck dummy: {bneck_col}")
        else:
            print(f"    WARNING: {bneck_col} not found in combined data")

    if bneck_cols:
        print(f"  Loaded {len(bneck_cols)} bottleneck dummies for {target_region}")
    else:
        print(f"  No bottleneck dummies found for {target_region}")

    print(f"  Combined data: {len(final_df)} observations")
    print(f"  Date range: {final_df['Datetime'].min()} to {final_df['Datetime'].max()}")

    # Step 2: Load and merge hydro reserves
    if 'hydro' in paths:
        print(f"\nLoading hydro reserves from: {paths['hydro']}")
        df_hydro = pd.read_excel(paths['hydro'])

        # Select the appropriate zone column
        hydro_col = f'{zone_hydro}_Hydro_Reserves'
        if hydro_col not in df_hydro.columns:
            raise ValueError(f"Column '{hydro_col}' not found in hydro file. "
                           f"Available: {df_hydro.columns.tolist()}")

        df_hydro_subset = pd.DataFrame({
            'Datetime': pd.to_datetime(df_hydro['Timestamp']),
            'Hydro_Reserves': pd.to_numeric(df_hydro[hydro_col], errors='coerce')
        })

        print(f"  Hydro data: {len(df_hydro_subset)} observations")
        print(f"  Date range: {df_hydro_subset['Datetime'].min()} to {df_hydro_subset['Datetime'].max()}")

        # Merge on Datetime
        final_df = pd.merge(final_df, df_hydro_subset, on='Datetime', how='left')
        print(f"  After hydro merge: {len(final_df)} observations")

    # Step 3: Load and merge commodity prices (always required)

    # Step 3a: Load Light Crude Oil (hourly data)
    print(f"\nLoading Light Crude Oil from: {paths['crude_oil']}")
    df_crude = pd.read_excel(paths['crude_oil'])

    # Process crude oil data
    df_crude['Datetime'] = pd.to_datetime(df_crude['Timestamp'])
    df_crude['Oil_Price'] = pd.to_numeric(df_crude['Close'], errors='coerce')  # Use Close price

    df_crude_subset = df_crude[['Datetime', 'Oil_Price']].copy()

    print(f"  Light Crude Oil data: {len(df_crude_subset)} hourly observations")
    print(f"  Date range: {df_crude_subset['Datetime'].min()} to {df_crude_subset['Datetime'].max()}")
    print(f"  Using Close price (USD/barrel)")

    # Merge on Datetime (hourly to hourly)
    final_df = pd.merge(final_df, df_crude_subset, on='Datetime', how='left')
    print(f"  After crude oil merge: {len(final_df)} observations")

    # Step 3b: Load TTF Gas (daily data from Bloomberg)
    print(f"\nLoading TTF Gas from: {paths['commodities']}")
    df_comm = pd.read_excel(paths['commodities'], header=None, skiprows=5)
    df_comm.columns = ['Date', 'TTF_Gas', 'WTI_Oil', 'Brent_Oil', 'MT1', 'LUA1', 'CP1']

    # Process gas data only
    df_comm['Date'] = pd.to_datetime(df_comm['Date'], errors='coerce')
    df_comm = df_comm.dropna(subset=['Date'])
    df_comm['TTF_Gas'] = pd.to_numeric(df_comm['TTF_Gas'], errors='coerce')

    df_gas = df_comm[['Date', 'TTF_Gas']].copy()
    df_gas.columns = ['Date', 'Gas_Price']

    print(f"  TTF Gas data: {len(df_gas)} daily observations")
    print(f"  Date range: {df_gas['Date'].min().date()} to {df_gas['Date'].max().date()}")

    # Create date column for merging (extract date from hourly Datetime)
    final_df['Date'] = final_df['Datetime'].dt.date
    df_gas['Date'] = df_gas['Date'].dt.date

    # Merge gas on date (each hour gets the daily gas price)
    final_df = pd.merge(final_df, df_gas, on='Date', how='left')
    final_df = final_df.drop(columns=['Date'])

    print(f"  Merged commodity prices: Oil_Price (Light Crude hourly, USD/barrel), Gas_Price (TTF daily, EUR/MWh)")

    # Step 3c: Lag commodity prices for day-ahead market alignment
    print(f"\n--- LAGGING COMMODITY PRICES BY {lag_commodity_hours} HOURS ---")
    print("Rationale: Day-ahead market pricing uses commodity prices from bidding time (D-1)")

    rows_before_lag = len(final_df)

    # Lag Oil Price
    oil_before = final_df['Oil_Price'].notna().sum()
    final_df['Oil_Price'] = final_df['Oil_Price'].shift(lag_commodity_hours)
    oil_after = final_df['Oil_Price'].notna().sum()
    oil_lost = oil_before - oil_after
    print(f"  Oil_Price: Lagged by {lag_commodity_hours}h ({oil_lost} observations lost at start)")

    # Lag Gas Price
    gas_before = final_df['Gas_Price'].notna().sum()
    final_df['Gas_Price'] = final_df['Gas_Price'].shift(lag_commodity_hours)
    gas_after = final_df['Gas_Price'].notna().sum()
    gas_lost = gas_before - gas_after
    print(f"  Gas_Price: Lagged by {lag_commodity_hours}h ({gas_lost} observations lost at start)")

    print(f"All subsequent transformations (log, deseasonalization) will use lagged commodity prices")

    # Step 4: Apply date filter if specified
    if start_date is not None or end_date is not None:
        rows_before = len(final_df)
        if start_date is not None:
            final_df = final_df[final_df['Datetime'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            final_df = final_df[final_df['Datetime'] <= pd.to_datetime(end_date)]
        rows_after = len(final_df)
        print(f"\nDate filter applied: {rows_before} -> {rows_after} observations")
        print(f"  Filtered range: {start_date or 'start'} to {end_date or 'end'}")

    # Step 5: Handle missing values
    if use_interpolation:
        print("\n--- APPLYING LINEAR INTERPOLATION FOR MISSING VALUES ---")

        missing_by_var = final_df.isna().sum()
        missing_before = missing_by_var.sum()

        variables_with_missing = [var for var, count in missing_by_var.items() if count > 0]

        if variables_with_missing:
            print(f"\nVariables with missing values: {', '.join(variables_with_missing)}")
            print(f"\nDetailed breakdown:")
            for var, count in missing_by_var.items():
                if count > 0:
                    pct = (count / len(final_df)) * 100
                    print(f"  {var}: {count} ({pct:.2f}%)")
        else:
            print("\nNo missing values detected in any variable.")

        print(f"\nTotal missing values: {missing_before}")

        final_df = final_df.interpolate(method='linear', limit_direction='both')
        final_df = final_df.dropna()
        missing_after = final_df.isna().sum().sum()
        print(f"Missing values after interpolation: {missing_after}")
        print(f"Rows retained: {len(final_df)}")
    else:
        rows_before = len(final_df)

        missing_by_var = final_df.isna().sum()
        if missing_by_var.sum() > 0:
            print(f"\nMissing values by variable (before dropping rows):")
            for var, count in missing_by_var.items():
                if count > 0:
                    pct = (count / len(final_df)) * 100
                    print(f"  {var}: {count} ({pct:.2f}%)")

        final_df = final_df.dropna()
        rows_dropped = rows_before - len(final_df)
        if rows_dropped > 0:
            print(f"\nDropped {rows_dropped} rows with missing values")

    print(f"\n--- FINAL DATASET ---")
    print(f"Total observations: {len(final_df)}")
    print(f"Columns: {final_df.columns.tolist()}")

    # Set index and infer hourly frequency
    final_df = final_df.set_index('Datetime')
    # Infer frequency from data (should be hourly 'H')
    inferred_freq = pd.infer_freq(final_df.index)
    if inferred_freq:
        final_df = final_df.asfreq(inferred_freq)
        print(f"Inferred frequency: {inferred_freq}")
    else:
        print("Warning: Could not infer frequency from datetime index")

    return final_df
