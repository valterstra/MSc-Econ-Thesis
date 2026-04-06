import pandas as pd
import numpy as np
import os
import warnings
import io
import contextlib
import statsmodels.api as sm
import holidays



# --- 1. DATA LOADING FUNCTIONS ---


def load_data(paths, target_region='SE1', zone_hydro='SE1', use_interpolation=False,
                                 start_date=None, end_date=None, lag_commodity_hours=24,
                                 use_bilateral_exchange=False):
    """
    Load pre-combined regional data file and merge with hydro reserves and commodities.

    This function loads the Combined_{region}_Data file which already contains:
    - Timestamp, Spot_Price, Wind_Forecast, Net_Exchange, NetExch_* (bilateral), Consumption_Forecast

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
    - use_bilateral_exchange: If True, load per-partner NetExch_* cols instead of total Net_Exchange

    Returns:
    - DataFrame indexed by Datetime with columns:
      Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price,
      plus Net_Exchange (if use_bilateral_exchange=False) or NetExch_* cols (if True)
      Note: Oil_Price (Light Crude Close) and Gas_Price (TTF) are automatically lagged by lag_commodity_hours
    """
    print("\n--- LOADING COMBINED REGIONAL DATA ---")

    # Step 1: Load combined file
    df_combined = pd.read_excel(paths['combined'])

    # Map columns to standard names
    final_df = pd.DataFrame({
        'Datetime': pd.to_datetime(df_combined['Timestamp']),
        'Price': pd.to_numeric(df_combined['Spot_Price'], errors='coerce'),
        'Wind_Forecast': pd.to_numeric(df_combined['Wind_Forecast'], errors='coerce'),
        'Consumption': pd.to_numeric(df_combined['Consumption_Forecast'], errors='coerce')
    })

    if use_bilateral_exchange:
        netexch_cols = [c for c in df_combined.columns if c.startswith('NetExch_')]
        for col in netexch_cols:
            final_df[col] = pd.to_numeric(df_combined[col], errors='coerce')
        print(f"  [Step 1] Combined data: {len(final_df)} obs, bilateral exchange: {netexch_cols}")
    else:
        final_df['Net_Exchange'] = pd.to_numeric(df_combined['Net_Exchange'], errors='coerce')
        print(f"  [Step 1] Combined data: {len(final_df)} obs, total Net_Exchange loaded")

    # Step 2: Load and merge hydro reserves
    if 'hydro' in paths:
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

        # Merge on Datetime
        final_df = pd.merge(final_df, df_hydro_subset, on='Datetime', how='left')
        print(f"  [Step 2] Hydro merged: {len(final_df)} obs")

    # Step 3: Load and merge commodity prices (always required)

    # Step 3a: Load Light Crude Oil (hourly data)
    df_crude = pd.read_excel(paths['crude_oil'])

    # Process crude oil data
    df_crude['Datetime'] = pd.to_datetime(df_crude['Timestamp'])
    df_crude['Oil_Price'] = pd.to_numeric(df_crude['Close'], errors='coerce')  # Use Close price

    df_crude_subset = df_crude[['Datetime', 'Oil_Price']].copy()

    # Merge on Datetime (hourly to hourly)
    final_df = pd.merge(final_df, df_crude_subset, on='Datetime', how='left')
    print(f"  [Step 3a] Light Crude Oil merged (Close, USD/barrel)")

    # Step 3b: Load TTF Gas (daily data from Bloomberg)
    df_comm = pd.read_excel(paths['commodities'], header=None, skiprows=5)
    df_comm.columns = ['Date', 'TTF_Gas', 'WTI_Oil', 'Brent_Oil', 'MT1', 'LUA1', 'CP1']

    # Process gas data only
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Could not infer format, so each element will be parsed individually.*"
        )
        df_comm['Date'] = pd.to_datetime(df_comm['Date'], errors='coerce')
    df_comm = df_comm.dropna(subset=['Date'])
    df_comm['TTF_Gas'] = pd.to_numeric(df_comm['TTF_Gas'], errors='coerce')

    df_gas = df_comm[['Date', 'TTF_Gas']].copy()
    df_gas.columns = ['Date', 'Gas_Price']

    # Create date column for merging (extract date from hourly Datetime)
    final_df['Date'] = final_df['Datetime'].dt.date
    df_gas['Date'] = df_gas['Date'].dt.date

    # Merge gas on date (each hour gets the daily gas price)
    final_df = pd.merge(final_df, df_gas, on='Date', how='left')
    final_df = final_df.drop(columns=['Date'])
    print(f"  [Step 3b] TTF Gas merged (EUR/MWh)")

    # Step 3c: Lag commodity prices for day-ahead market alignment
    final_df['Oil_Price'] = final_df['Oil_Price'].shift(lag_commodity_hours)
    final_df['Gas_Price'] = final_df['Gas_Price'].shift(lag_commodity_hours)
    print(f"  [Step 3c] Commodity prices lagged by {lag_commodity_hours}h for day-ahead market")

    # Step 4: Apply date filter if specified
    if start_date is not None or end_date is not None:
        rows_before = len(final_df)
        if start_date is not None:
            final_df = final_df[final_df['Datetime'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            final_df = final_df[final_df['Datetime'] <= pd.to_datetime(end_date)]
        print(f"  [Step 4] Date filter: {rows_before} -> {len(final_df)} obs ({start_date or 'start'} to {end_date or 'end'})")

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


# --- 2. DATA PREPROCESSING FUNCTIONS ---

def handle_negative_prices(df):
    """
    Check all variables for negative values and handle Price column using the shift method.

    First checks all variables (except Net_Exchange) and reports negative values.
    Then shifts the entire Price series upward so the minimum becomes 0.01,
    preserving relative differences across all observations.

    Parameters:
    - df: DataFrame with variables

    Returns:
    - DataFrame with adjusted Price column (if needed)
    """
    print("\n" + "="*80)
    print("NEGATIVE VALUE CHECK & PRICE HANDLING")
    print("="*80)

    # --- PART 1: Check all variables for negative values ---
    print("\nChecking all variables for negative values (except exchange variables)...\n")

    variables_to_check = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Consumption',
                         'Oil_Price', 'Gas_Price']

    found_negatives = False

    for var in variables_to_check:
        if var in df.columns:
            min_val = df[var].min()
            negative_count = (df[var] < 0).sum()
            negative_pct = (negative_count / len(df)) * 100
            zero_count = (df[var] == 0).sum()

            if negative_count > 0:
                found_negatives = True
                print(f"  {var}:")
                print(f"    Min value: {min_val:.4f}")
                print(f"    Negative values: {negative_count} ({negative_pct:.2f}%)")
                print(f"    Zero values: {zero_count}")
        else:
            print(f"  {var}: NOT FOUND in dataframe")

    if not found_negatives:
        print("  [OK] No negative values found in any variable")

    print("\n  Exchange variables: NOT CHECKED (expected to have negative values)")

    # --- PART 2: Handle Price variable ---
    print("\n" + "-"*80)
    print("HANDLING PRICE VARIABLE")
    print("-"*80)

    if 'Price' not in df.columns:
        print("Warning: Price column not found. Skipping price handling.")
        print("="*80 + "\n")
        return df

    min_price = df['Price'].min()
    negative_count = (df['Price'] < 0).sum()
    zero_count = (df['Price'] == 0).sum()

    print(f"\nPrice statistics:")
    print(f"  Minimum value: {min_price:.4f}")
    print(f"  Negative values: {negative_count} ({(negative_count/len(df))*100:.2f}%)")
    print(f"  Zero values: {zero_count} ({(zero_count/len(df))*100:.2f}%)")

    df_clean = df.copy()

    if min_price < 0.01:
        shift_amount = 0.01 - min_price
        df_clean['Price'] = df_clean['Price'] + shift_amount
        new_min = df_clean['Price'].min()
        print(f"\nSHIFT METHOD: Shifted entire series upward by {shift_amount:.4f}")
        print(f"  Old minimum: {min_price:.4f}")
        print(f"  New minimum: {new_min:.4f}")
        print(f"  All {len(df)} observations shifted by the same amount")
    else:
        print("\nSHIFT METHOD: Minimum value already >= 0.01, no shift needed")

    print("="*80 + "\n")

    return df_clean


def apply_log_transform(df):
    """
    Apply logarithmic transformation to variables following Fredriksson (2016).
    STANDARD APPROACH: Log transformation is applied FIRST, then deseasonalization.

    Logs applied to: Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    NOT logged: Net_Exchange (can contain negative values)

    Note:
    - Oil_Price and Gas_Price are already lagged by 24h (automatically in load_data)
    - Price negative values are already handled by handle_negative_prices()
    - Other variables are clipped to 0.01 to handle zeros and edge cases

    Returns df with logged columns: Price_Log, Wind_Forecast_Log, Hydro_Reserves_Log,
    Consumption_Log, Oil_Price_Log, Gas_Price_Log
    """
    print("\n--- APPLYING LOGARITHMIC TRANSFORMATION (STANDARD APPROACH) ---")
    print("Log transformation applied BEFORE deseasonalization")
    print("Note: Oil & Gas prices are already lagged by 24h (automatically in load_data)")
    print("Note: Price negative values already handled by handle_negative_prices()")

    # Log Price (no clipping - negative values already handled)
    df['Price_Log'] = np.log(df['Price'])
    print(f"Price: log(raw) applied [no clipping - negatives already handled]")

    # Log Wind Forecast
    df['Wind_Forecast_Log'] = np.log(df['Wind_Forecast'].clip(lower=0.01))
    print(f"Wind_Forecast: log(raw) applied")

    # Log Hydro Reserves
    df['Hydro_Reserves_Log'] = np.log(df['Hydro_Reserves'].clip(lower=0.01))
    print(f"Hydro_Reserves: log(raw) applied")

    # Log Consumption
    df['Consumption_Log'] = np.log(df['Consumption'].clip(lower=0.01))
    print(f"Consumption: log(raw) applied")

    # Exchange variables (Net_Exchange or NetExch_*) - NOT logged (can be negative)

    # Log Oil Price
    df['Oil_Price_Log'] = np.log(df['Oil_Price'].clip(lower=0.01))
    print(f"Oil_Price: log(raw) applied [USD/barrel]")

    # Log Gas Price
    df['Gas_Price_Log'] = np.log(df['Gas_Price'].clip(lower=0.01))
    print(f"Gas_Price: log(raw) applied [EUR/MWh]")

    return df


def deseasonalize_logged_variables(df, seasonal_interactions='none'):
    """
    Remove seasonal patterns from LOGGED variables using dummy variable regression.
    Based on Fredriksson (2016) methodology.

    STANDARD APPROACH: Deseasonalization is applied to LOGGED series (after log transformation).

    seasonal_interactions controls the intraday seasonality spec for Price & Consumption:
    - 'none':             Year + Month + DOW + Hour + Holiday  (original additive)
    - 'hour_dow':         Year + Month + Hour×DOW (167) + Holiday
    - 'hour_dow_month':   Year + Hour×DOW (167) + Hour×Month (287) + Holiday
                          (Month main effect subsumed by Hour×Month)

    Hydro_Reserves_Log, Oil_Price_Log, Gas_Price_Log always use: Year + Month ONLY (PARTIAL).

    Note: Wind_Forecast_Log and Net_Exchange are NOT deseasonalized (following Fredriksson).
    """
    print("\n--- DESEASONALIZING LOGGED VARIABLES (Fredriksson 2016 methodology) ---")
    print("Deseasonalization applied to LOGGED series (standard approach)")
    print(f"\nSeasonal interactions setting: '{seasonal_interactions}'")
    print("\nDeseasonalization strategy:")
    if seasonal_interactions == 'none':
        print("  - Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL additive)")
    elif seasonal_interactions == 'hour_dow':
        print("  - Price & Consumption: Year + Month + Hour×DOW (167 dummies) + Holiday")
    elif seasonal_interactions == 'hour_dow_month':
        print("  - Price & Consumption: Year + Hour×DOW (167) + Hour×Month (287) + Holiday")
    print("  - Hydro, Oil, Gas: Year + Month ONLY (PARTIAL)")

    # Extract time components from Datetime index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['Hour'] = df.index.hour

    # Create holiday indicator for Swedish/Nordic holidays
    swedish_holidays = holidays.Sweden(years=range(df.index.year.min(), df.index.year.max() + 1))
    df['Holiday'] = df.index.to_series().apply(lambda x: 1 if x.date() in swedish_holidays else 0).values
    print("\nHoliday dummies created using Swedish holiday calendar")

    # Base dummies always created
    year_dummies  = pd.get_dummies(df['Year'],  prefix='Year',  drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True, dtype=float)
    holiday_dummy = df[['Holiday']].astype(float)

    # Build FULL seasonal dummies based on selected interaction mode
    if seasonal_interactions == 'none':
        # Original additive spec
        dow_dummies  = pd.get_dummies(df['DayOfWeek'], prefix='DOW',  drop_first=True, dtype=float)
        hour_dummies = pd.get_dummies(df['Hour'],      prefix='Hour', drop_first=True, dtype=float)
        seasonal_dummies_full = pd.concat(
            [year_dummies, month_dummies, dow_dummies, hour_dummies, holiday_dummy], axis=1)

    elif seasonal_interactions == 'hour_dow':
        # Hour×DOW interaction: 24×7=168 cells → 167 dummies (absorbs Hour and DOW main effects)
        hour_dow_cat = df['Hour'].astype(str) + '_' + df['DayOfWeek'].astype(str)
        hour_dow_dummies = pd.get_dummies(hour_dow_cat, prefix='HourDOW', drop_first=True, dtype=float)
        seasonal_dummies_full = pd.concat(
            [year_dummies, month_dummies, hour_dow_dummies, holiday_dummy], axis=1)

    elif seasonal_interactions == 'hour_dow_month':
        # Hour×DOW (167) + Hour×Month (287); Month main effect subsumed by Hour×Month
        hour_dow_cat   = df['Hour'].astype(str) + '_' + df['DayOfWeek'].astype(str)
        hour_month_cat = df['Hour'].astype(str) + '_' + df['Month'].astype(str)
        hour_dow_dummies   = pd.get_dummies(hour_dow_cat,   prefix='HourDOW',   drop_first=True, dtype=float)
        hour_month_dummies = pd.get_dummies(hour_month_cat, prefix='HourMonth', drop_first=True, dtype=float)
        seasonal_dummies_full = pd.concat(
            [year_dummies, hour_dow_dummies, hour_month_dummies, holiday_dummy], axis=1)

    else:
        raise ValueError(f"seasonal_interactions must be 'none', 'hour_dow', or 'hour_dow_month'. Got: '{seasonal_interactions}'")

    seasonal_dummies_full = sm.add_constant(seasonal_dummies_full).astype(float)

    # PARTIAL seasonal dummies (Year + Month only, for Hydro, Oil, Gas)
    seasonal_dummies_partial = pd.concat([year_dummies, month_dummies], axis=1)
    seasonal_dummies_partial = sm.add_constant(seasonal_dummies_partial).astype(float)

    print("\n--- Deseasonalizing with FULL seasonal controls (Year+Month+Hour×DOW+Holiday) ---")

    # Deseasonalize Price_Log (FULL)
    price_log_model = sm.OLS(df['Price_Log'], seasonal_dummies_full).fit()
    df['Price_DS'] = price_log_model.resid
    print(f"Price_Log: Seasonal R² = {price_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Price_Log'].std():.4f}, Deseasonalized std: {df['Price_DS'].std():.4f}")

    # Deseasonalize Consumption_Log (FULL)
    consumption_log_model = sm.OLS(df['Consumption_Log'], seasonal_dummies_full).fit()
    df['Consumption_Log_Deseasonalized'] = consumption_log_model.resid
    print(f"Consumption_Log: Seasonal R² = {consumption_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Consumption_Log'].std():.4f}, Deseasonalized std: {df['Consumption_Log_Deseasonalized'].std():.4f}")

    print("\n--- Deseasonalizing with PARTIAL seasonal controls (Year+Month ONLY) ---")

    # Deseasonalize Hydro_Reserves_Log (PARTIAL - Year + Month only)
    hydro_log_model = sm.OLS(df['Hydro_Reserves_Log'], seasonal_dummies_partial).fit()
    df['Hydro_Reserves_Log_Deseasonalized'] = hydro_log_model.resid
    print(f"Hydro_Reserves_Log: Seasonal R² = {hydro_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Hydro_Reserves_Log'].std():.4f}, Deseasonalized std: {df['Hydro_Reserves_Log_Deseasonalized'].std():.4f}")

    # Oil_Price_Log and Gas_Price_Log: NOT deseasonalized.
    # FWL guarantees their seasonal component is orthogonalized at estimation
    # since Price is fully deseasonalized. Consistent with Net_Exchange treatment.
    print("Oil_Price_Log: not deseasonalized (FWL / consistency with Net_Exchange)")
    print("Gas_Price_Log: not deseasonalized (FWL / consistency with Net_Exchange)")

    # Clean up temporary columns
    df = df.drop(columns=['Year', 'Month', 'DayOfWeek', 'Hour', 'Holiday'])

    print("\nNote: Wind_Forecast_Log and exchange variables are NOT deseasonalized (following Fredriksson)")

    return df


def cap_price_outliers(df):
    """
    Winsorize Price_DS at ±4σ (global mean and std over the full window).

    Always operates on 'Price_DS' (after log transform and deseasonalization).

    Returns:
    - DataFrame with outliers replaced in Price_DS
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - SYMMETRIC ±4σ WINSORIZATION")
    print("="*80)

    if 'Price_DS' not in df.columns:
        print("Warning: Price_DS not found. Cannot apply outlier handling.")
        return df, {}
    price_col = 'Price_DS'
    print("Applying to: Logged and Deseasonalized Price")

    print("Replacing outliers with ±3*std threshold for respective weekday")
    print("Note: Outlier handling only applied to Price, not explanatory variables\n")

    df_clean = df.copy()
    outlier_stats = {}

    print(f"\nProcessing: {price_col}")

    data = df_clean[price_col].copy()

    # Compute single global threshold from full sample (asymmetric: -3σ lower, +4σ upper)
    overall_mean = data.mean()
    overall_std = data.std()
    upper_threshold = overall_mean + 4 * overall_std
    lower_threshold = overall_mean - 4 * overall_std

    print(f"  Mean: {overall_mean:.4f}")
    print(f"  Std:  {overall_std:.4f}")
    print(f"  Threshold: [{lower_threshold:.4f} (mean−4σ), {upper_threshold:.4f} (mean+4σ)]")

    # Winsorise: cap at threshold value
    upper_outliers = data > upper_threshold
    lower_outliers = data < lower_threshold
    n_upper = upper_outliers.sum()
    n_lower = lower_outliers.sum()
    total_outliers = n_upper + n_lower

    data[upper_outliers] = upper_threshold
    data[lower_outliers] = lower_threshold

    df_clean[price_col] = data

    pct_outliers = (total_outliers / len(data)) * 100
    new_mean = data.mean()
    new_std = data.std()

    print(f"\n  {'='*70}")
    print(f"  OUTLIER SUMMARY")
    print(f"  {'='*70}")
    print(f"  Upper outliers capped (> +4σ): {n_upper}")
    print(f"  Lower outliers capped (< -4σ): {n_lower}")
    print(f"  Total replaced: {total_outliers} ({pct_outliers:.2f}% of data)")
    print(f"\n  Statistics before vs. after:")
    print(f"    Mean: {overall_mean:.4f} -> {new_mean:.4f}")
    print(f"    Std:  {overall_std:.4f} -> {new_std:.4f}")
    print(f"  {'='*70}")

    outlier_stats[price_col] = {
        'n_outliers': total_outliers,
        'n_upper': n_upper,
        'n_lower': n_lower,
        'original_mean': overall_mean,
        'original_std': overall_std,
        'new_mean': new_mean,
        'new_std': new_std,
    }

    return df_clean, outlier_stats


def preprocess_data_for_regression(df_raw, suppress_output=False, seasonal_interactions='none'):
    """
    Apply preprocessing sequence and return transformed data for regression.

    Pipeline:
        1. handle_negative_prices
        2. apply_log_transform
        3. deseasonalize_logged_variables  (seasonal_interactions controls intraday spec)
        4. cap_price_outliers               (global ±4σ Winsorization on Price_DS)

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
