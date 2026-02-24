import pandas as pd
import numpy as np
import os
import warnings
import time
import io
import contextlib
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from arch.unitroot import DFGLS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (save plots only, no display)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import holidays
import ruptures as rpt
from scipy import stats


# Trading partners for congestion dummies (must match regional_data_combiner.py)
TRADING_PARTNERS = {
    'SE1': ['FI', 'NO4', 'SE2'],
    'SE2': ['NO3', 'NO4', 'SE1', 'SE3'],
    'SE3': ['DK1', 'FI', 'NO1', 'SE2', 'SE4'],
    'SE4': []   # To be defined
}

# ARMAX fitting defaults (overridden in __main__ config block if needed)
ARMAX_ALLOW_NONCONVERGED = False
ARMAX_MAXITER = 300
ARMAX_SOLVER = 'statespace'
ARMAX_USE_WARM_START = True
ARMAX_ENABLE_FALLBACK_ORDERS = True
ARMAX_FALLBACK_ORDERS = [(1, 0, 1), (2, 0, 2), (3, 0, 3)]
ARMAX_BASELINE_SPEC = {
    'enabled': True,
    'order': (1, 0, 1),
    'extra_ar_lags': [],
    'drop_initial_nan': True,
    'label': 'ARMAX(1,0,1)'
}

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


# --- 2. DATA PREPROCESSING FUNCTIONS ---

def handle_negative_prices(df, method='clip'):
    """
    Check all variables for negative values and handle Price column using specified method.

    First checks all variables (except Net_Exchange) and reports negative values.
    Then handles negative Price values using the chosen method.

    Two methods available for Price:
    1. 'clip': Replace negative/zero values with 0.01 (current default)
    2. 'shift': Shift entire series upward so minimum value becomes 0.01

    Parameters:
    - df: DataFrame with variables
    - method: 'clip' or 'shift' (for Price handling only)

    Returns:
    - DataFrame with adjusted Price column (if needed)
    """
    print("\n" + "="*80)
    print("NEGATIVE VALUE CHECK & PRICE HANDLING")
    print("="*80)

    # --- PART 1: Check all variables for negative values ---
    print("\nChecking all variables for negative values (except Net_Exchange)...\n")

    variables_to_check = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Consumption',
                         'Oil_Price', 'Gas_Price']

    negative_stats = {}
    found_negatives = False

    for var in variables_to_check:
        if var in df.columns:
            min_val = df[var].min()
            negative_count = (df[var] < 0).sum()
            negative_pct = (negative_count / len(df)) * 100
            zero_count = (df[var] == 0).sum()

            negative_stats[var] = {
                'min': min_val,
                'negative_count': negative_count,
                'negative_pct': negative_pct,
                'zero_count': zero_count
            }

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

    print("\n  Net_Exchange: NOT CHECKED (expected to have negative values)")

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

    print(f"Method: {method.upper()}")
    print(f"\nPrice statistics:")
    print(f"  Minimum value: {min_price:.4f}")
    print(f"  Negative values: {negative_count} ({(negative_count/len(df))*100:.2f}%)")
    print(f"  Zero values: {zero_count} ({(zero_count/len(df))*100:.2f}%)")

    df_clean = df.copy()

    if method == 'clip':
        # Current method: clip to 0.01
        if min_price < 0.01:
            below_threshold = (df['Price'] < 0.01).sum()
            df_clean['Price'] = df_clean['Price'].clip(lower=0.01)
            print(f"\nCLIP METHOD: Replaced {below_threshold} values below 0.01 with 0.01")
        else:
            print("\nCLIP METHOD: No values below 0.01, no clipping needed")

    elif method == 'shift':
        # New method: shift entire series upward
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

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'clip' or 'shift'")

    print("="*80 + "\n")

    return df_clean


def handle_outliers_fredriksson(df, apply_to_raw=False):
    """
    Replace outliers using Fredriksson (2016) methodology.

    Outlier definition:
    - Exceeds 6x standard deviation above the mean, OR
    - Lower than 3.7x standard deviation below the mean

    Replacement method:
    - Replace outlier with mean of 24 and 48 hours before and after the outlier
    - Only applied to PRICE series, not explanatory variables

    Parameters:
    - apply_to_raw: If True, applies to raw 'Price' column (early handling before log transform)
                    If False, applies to 'Price_Log_Deseasonalized' column (late handling)

    Returns:
    - DataFrame with outliers replaced in Price
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - FREDRIKSSON (2016) METHODOLOGY")
    print("="*80)

    # Determine which Price column to use
    if apply_to_raw:
        price_col = 'Price'
        print("Applying to: Raw Price (before log transformation)")
    else:
        if 'Price_Log_Deseasonalized' not in df.columns:
            print("Warning: Price_Log_Deseasonalized not found. Cannot apply outlier handling.")
            return df, {}
        price_col = 'Price_Log_Deseasonalized'
        print("Applying to: Logged and Deseasonalized Price")

    print("Replacing outliers with mean of 24 and 48 hours before/after")
    print("Note: Outlier handling only applied to Price, not explanatory variables\n")

    df_clean = df.copy()
    outlier_stats = {}

    print(f"\nProcessing: {price_col}")

    data = df_clean[price_col].copy()
    mean_val = data.mean()
    std_val = data.std()

    # Fredriksson thresholds
    upper_threshold = mean_val + 6 * std_val
    lower_threshold = mean_val - 3.7 * std_val

    print(f"  Mean: {mean_val:.2f}")
    print(f"  Std Dev: {std_val:.2f}")
    print(f"  Upper threshold (+6*std): {upper_threshold:.2f}")
    print(f"  Lower threshold (-3.7*std): {lower_threshold:.2f}")

    # Identify outliers
    outliers = (data > upper_threshold) | (data < lower_threshold)
    n_outliers = outliers.sum()
    pct_outliers = (n_outliers / len(data)) * 100

    if n_outliers > 0:
        # Count upper vs lower outliers
        upper_outliers = (data > upper_threshold).sum()
        lower_outliers = (data < lower_threshold).sum()

        print(f"\n  {'='*70}")
        print(f"  OUTLIER SUMMARY")
        print(f"  {'='*70}")
        print(f"  Total outliers found: {n_outliers} ({pct_outliers:.2f}% of data)")
        print(f"  Total observations: {len(data)}")
        print(f"    Upper outliers (>{upper_threshold:.2f}): {upper_outliers}")
        print(f"    Lower outliers (<{lower_threshold:.2f}): {lower_outliers}")
        print(f"\n  Replacing with mean of ±24h and ±48h surrounding values...")

        # Replace each outlier with mean of surrounding hours
        outlier_positions = np.where(outliers)[0]

        for pos in outlier_positions:
            # Calculate mean of 24 and 48 hours before and after
            surrounding_values = []

            # 24 hours before
            if pos >= 24:
                surrounding_values.append(data.iloc[pos - 24])
            # 48 hours before
            if pos >= 48:
                surrounding_values.append(data.iloc[pos - 48])
            # 24 hours after
            if pos + 24 < len(data):
                surrounding_values.append(data.iloc[pos + 24])
            # 48 hours after
            if pos + 48 < len(data):
                surrounding_values.append(data.iloc[pos + 48])

            # Replace with mean of surrounding values
            if surrounding_values:
                replacement_value = np.mean(surrounding_values)
                original_value = data.iloc[pos]
                data.iloc[pos] = replacement_value
                # Individual outlier replacements not printed (too verbose)

        # Update the dataframe
        df_clean[price_col] = data

        # Recalculate statistics after replacement
        new_mean = data.mean()
        new_std = data.std()

        print(f"\n  Statistics before vs. after replacement:")
        print(f"    Mean: {mean_val:.4f} -> {new_mean:.4f} (change: {new_mean - mean_val:.4f})")
        print(f"    Std:  {std_val:.4f} -> {new_std:.4f} (change: {new_std - std_val:.4f})")
        print(f"  {'='*70}")

        outlier_stats[price_col] = {
            'n_outliers': n_outliers,
            'n_upper': upper_outliers,
            'n_lower': lower_outliers,
            'original_mean': mean_val,
            'original_std': std_val,
            'new_mean': new_mean,
            'new_std': new_std
        }
    else:
        print(f"\n  No outliers found")
        outlier_stats[price_col] = {
            'n_outliers': 0,
            'n_upper': 0,
            'n_lower': 0,
            'original_mean': mean_val,
            'original_std': std_val,
            'new_mean': mean_val,
            'new_std': std_val
        }

    return df_clean, outlier_stats


def handle_outliers_gianfreda(df, apply_to_raw=False):
    """
    Replace outliers using Gianfreda (2010) / Mugele et al. (2005) methodology.

    Outlier definition:
    - Exceeds 3x standard deviation above or below the mean (symmetric threshold)

    Replacement method:
    - Replace outlier with 3*std value for the respective weekday
    - Each weekday has its own 3σ threshold (Monday outliers capped at Monday's 3σ, etc.)

    Parameters:
    - apply_to_raw: If True, applies to raw 'Price' column (early handling before log transform)
                    If False, applies to 'Price_Log_Deseasonalized' column (late handling)

    Returns:
    - DataFrame with outliers replaced in Price
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - GIANFREDA (2010) / MUGELE ET AL. (2005) METHODOLOGY")
    print("="*80)

    # Determine which Price column to use
    if apply_to_raw:
        price_col = 'Price'
        print("Applying to: Raw Price (before log transformation)")
    else:
        if 'Price_Log_Deseasonalized' not in df.columns:
            print("Warning: Price_Log_Deseasonalized not found. Cannot apply outlier handling.")
            return df, {}
        price_col = 'Price_Log_Deseasonalized'
        print("Applying to: Logged and Deseasonalized Price")

    print("Replacing outliers with ±3*std threshold for respective weekday")
    print("Note: Outlier handling only applied to Price, not explanatory variables\n")

    df_clean = df.copy()
    outlier_stats = {}

    print(f"\nProcessing: {price_col}")

    data = df_clean[price_col].copy()

    # Extract day of week (0=Monday, 6=Sunday)
    df_clean['DayOfWeek'] = df_clean.index.dayofweek

    # Calculate overall statistics
    overall_mean = data.mean()
    overall_std = data.std()

    print(f"  Overall Mean: {overall_mean:.2f}")
    print(f"  Overall Std Dev: {overall_std:.2f}")
    print(f"  Overall ±3*std threshold: [{overall_mean - 3*overall_std:.2f}, {overall_mean + 3*overall_std:.2f}]")

    # Calculate weekday-specific statistics and thresholds
    print("\n  Weekday-specific thresholds:")
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_stats = {}

    for day in range(7):
        day_mask = df_clean['DayOfWeek'] == day
        day_data = data[day_mask]
        day_mean = day_data.mean()
        day_std = day_data.std()

        weekday_stats[day] = {
            'mean': day_mean,
            'std': day_std,
            'upper_threshold': day_mean + 3 * day_std,
            'lower_threshold': day_mean - 3 * day_std,
            'cap_value': 3 * day_std  # The replacement value (3σ for this weekday)
        }

        print(f"    {weekday_names[day]}: mean={day_mean:.2f}, std={day_std:.2f}, "
              f"threshold=[{weekday_stats[day]['lower_threshold']:.2f}, {weekday_stats[day]['upper_threshold']:.2f}]")

    # Identify and replace outliers by weekday (using vectorized operations)
    total_outliers = 0
    outliers_by_day = {day: 0 for day in range(7)}

    for day in range(7):
        day_mask = df_clean['DayOfWeek'] == day

        upper_threshold = weekday_stats[day]['upper_threshold']
        lower_threshold = weekday_stats[day]['lower_threshold']

        # Identify upper outliers for this weekday
        upper_outliers = day_mask & (data > upper_threshold)
        n_upper = upper_outliers.sum()
        if n_upper > 0:
            data[upper_outliers] = upper_threshold
            outliers_by_day[day] += n_upper
            total_outliers += n_upper

        # Identify lower outliers for this weekday
        lower_outliers = day_mask & (data < lower_threshold)
        n_lower = lower_outliers.sum()
        if n_lower > 0:
            data[lower_outliers] = lower_threshold
            outliers_by_day[day] += n_lower
            total_outliers += n_lower

    # Update the dataframe
    df_clean[price_col] = data

    # Calculate outlier percentage
    pct_outliers = (total_outliers / len(data)) * 100

    print(f"\n  {'='*70}")
    print(f"  OUTLIER SUMMARY")
    print(f"  {'='*70}")
    print(f"  Total outliers found and replaced: {total_outliers} ({pct_outliers:.2f}% of data)")
    print(f"  Total observations: {len(data)}")
    print(f"\n  Outliers by weekday:")
    for day in range(7):
        if outliers_by_day[day] > 0:
            pct_day = (outliers_by_day[day] / total_outliers) * 100 if total_outliers > 0 else 0
            print(f"    {weekday_names[day]}: {outliers_by_day[day]} ({pct_day:.1f}% of outliers)")
        else:
            print(f"    {weekday_names[day]}: 0")

    # Recalculate statistics after replacement
    new_mean = data.mean()
    new_std = data.std()

    print(f"\n  Statistics before vs. after replacement:")
    print(f"    Mean: {overall_mean:.4f} -> {new_mean:.4f} (change: {new_mean - overall_mean:.4f})")
    print(f"    Std:  {overall_std:.4f} -> {new_std:.4f} (change: {new_std - overall_std:.4f})")
    print(f"  {'='*70}")

    outlier_stats[price_col] = {
        'n_outliers': total_outliers,
        'outliers_by_weekday': outliers_by_day,
        'original_mean': overall_mean,
        'original_std': overall_std,
        'new_mean': new_mean,
        'new_std': new_std,
        'weekday_stats': weekday_stats
    }

    # Clean up temporary column
    df_clean = df_clean.drop(columns=['DayOfWeek'])

    return df_clean, outlier_stats


def deseasonalize_logged_variables(df, save_temp_plots=True):
    """
    Remove seasonal patterns from LOGGED variables using dummy variable regression.
    Based on Fredriksson (2016) methodology.

    STANDARD APPROACH: Deseasonalization is applied to LOGGED series (after log transformation).

    Seasonal dummies applied:
    - Price_Log & Consumption_Log: Year, Month, Day-of-Week, Hour, Holidays (FULL deseasonalization)
    - Hydro_Reserves_Log, Oil_Price_Log, Gas_Price_Log: Year, Month ONLY (PARTIAL deseasonalization)

    Deseasonalizes: Price_Log, Consumption_Log, Hydro_Reserves_Log, Oil_Price_Log, Gas_Price_Log
    Creates: Price_Log_Deseasonalized, Consumption_Log_Deseasonalized, etc.

    Note: Wind_Forecast_Log is NOT deseasonalized (Fredriksson does not deseasonalize wind).
    Note: Net_Exchange is NOT deseasonalized (Fredriksson does not deseasonalize it).
    """
    print("\n--- DESEASONALIZING LOGGED VARIABLES (Fredriksson 2016 methodology) ---")
    print("Deseasonalization applied to LOGGED series (standard approach)")
    print("\nDeseasonalization strategy:")
    print("  - Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL)")
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

    # Create dummy variables (drop_first=True avoids multicollinearity)
    year_dummies = pd.get_dummies(df['Year'], prefix='Year', drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True, dtype=float)
    dow_dummies = pd.get_dummies(df['DayOfWeek'], prefix='DOW', drop_first=True, dtype=float)
    hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour', drop_first=True, dtype=float)
    holiday_dummy = df[['Holiday']].astype(float)

    # FULL seasonal dummies (for Price and Consumption)
    seasonal_dummies_full = pd.concat([year_dummies, month_dummies, dow_dummies, hour_dummies, holiday_dummy], axis=1)
    seasonal_dummies_full = sm.add_constant(seasonal_dummies_full).astype(float)

    # PARTIAL seasonal dummies (Year + Month only, for Hydro, Oil, Gas)
    seasonal_dummies_partial = pd.concat([year_dummies, month_dummies], axis=1)
    seasonal_dummies_partial = sm.add_constant(seasonal_dummies_partial).astype(float)

    print("\n--- Deseasonalizing with FULL seasonal controls (Year+Month+DOW+Hour+Holiday) ---")

    # Deseasonalize Price_Log (FULL)
    price_log_model = sm.OLS(df['Price_Log'], seasonal_dummies_full).fit()
    df['Price_Log_Deseasonalized'] = price_log_model.resid
    print(f"Price_Log: Seasonal R² = {price_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Price_Log'].std():.4f}, Deseasonalized std: {df['Price_Log_Deseasonalized'].std():.4f}")

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

    if save_temp_plots:
        # TEMPORARY: Visual check of deseasonalization
        # Create temporary plots folder if it doesn't exist
        temp_plots_dir = 'temporary plots'
        if not os.path.exists(temp_plots_dir):
            os.makedirs(temp_plots_dir)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Plot original logged hydro
        ax1.plot(df.index, df['Hydro_Reserves_Log'], color='blue', linewidth=0.5, alpha=0.7)
        ax1.set_title('Hydro_Reserves_Log (Original - with seasonal patterns)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Log(Hydro Reserves)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot deseasonalized hydro
        ax2.plot(df.index, df['Hydro_Reserves_Log_Deseasonalized'], color='green', linewidth=0.5, alpha=0.7)
        ax2.set_title('Hydro_Reserves_Log_Deseasonalized (Year+Month patterns removed)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Deseasonalized Log(Hydro)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        temp_plot_path = os.path.join(temp_plots_dir, 'TEMP_hydro_deseasonalization.png')
        plt.savefig(temp_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  TEMP: Saved hydro deseasonalization plot to {temp_plot_path}")
        # END TEMPORARY

    # Deseasonalize Oil_Price_Log (PARTIAL - Year + Month only)
    oil_log_model = sm.OLS(df['Oil_Price_Log'], seasonal_dummies_partial).fit()
    df['Oil_Price_Log_Deseasonalized'] = oil_log_model.resid
    print(f"Oil_Price_Log: Seasonal R² = {oil_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Oil_Price_Log'].std():.4f}, Deseasonalized std: {df['Oil_Price_Log_Deseasonalized'].std():.4f}")

    # Deseasonalize Gas_Price_Log (PARTIAL - Year + Month only)
    gas_log_model = sm.OLS(df['Gas_Price_Log'], seasonal_dummies_partial).fit()
    df['Gas_Price_Log_Deseasonalized'] = gas_log_model.resid
    print(f"Gas_Price_Log: Seasonal R² = {gas_log_model.rsquared:.4f}")
    print(f"  Original std: {df['Gas_Price_Log'].std():.4f}, Deseasonalized std: {df['Gas_Price_Log_Deseasonalized'].std():.4f}")

    # Clean up temporary columns
    df = df.drop(columns=['Year', 'Month', 'DayOfWeek', 'Hour', 'Holiday'])

    print("\nNote: Wind_Forecast_Log and Net_Exchange are NOT deseasonalized (following Fredriksson)")

    return df


def apply_log_transform(df, save_temp_plots=True):
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

    if save_temp_plots:
        # TEMPORARY: Visual check of log transformation
        # Create temporary plots folder if it doesn't exist
        temp_plots_dir = 'temporary plots'
        if not os.path.exists(temp_plots_dir):
            os.makedirs(temp_plots_dir)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Plot raw hydro reserves
        ax1.plot(df.index, df['Hydro_Reserves'], color='blue', linewidth=0.5, alpha=0.7)
        ax1.set_title('Hydro_Reserves (Raw - original scale)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Hydro Reserves (MWh)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot logged hydro reserves
        ax2.plot(df.index, df['Hydro_Reserves_Log'], color='orange', linewidth=0.5, alpha=0.7)
        ax2.set_title('Hydro_Reserves_Log (Log transformed)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log(Hydro Reserves)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        temp_plot_path = os.path.join(temp_plots_dir, 'TEMP_hydro_log_transformation.png')
        plt.savefig(temp_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  TEMP: Saved hydro log transformation plot to {temp_plot_path}")
        # END TEMPORARY

    # Log Consumption
    df['Consumption_Log'] = np.log(df['Consumption'].clip(lower=0.01))
    print(f"Consumption: log(raw) applied")

    # Net_Exchange - NOT logged (can be negative)
    print(f"Net_Exchange: NOT logged (contains negative values)")

    # Log Oil Price
    df['Oil_Price_Log'] = np.log(df['Oil_Price'].clip(lower=0.01))
    print(f"Oil_Price: log(raw) applied [USD/barrel]")

    # Log Gas Price
    df['Gas_Price_Log'] = np.log(df['Gas_Price'].clip(lower=0.01))
    print(f"Gas_Price: log(raw) applied [EUR/MWh]")

    return df


# --- 3. DIAGNOSTIC TEST FUNCTIONS ---

def run_ljungbox_test(residuals, lags=[5, 10, 15, 20], return_results=False, print_output=True):
    """
    Ljung-Box test for autocorrelation in residuals.

    Tests the null hypothesis that residuals are independently distributed (no autocorrelation).
    Low p-values (< 0.05) indicate significant autocorrelation.

    Following Fredriksson (2016), tests at multiple lag lengths.

    Parameters:
    - residuals: Residual series to test
    - lags: List of lag values to test
    - return_results: If True, return DataFrame with results
    - print_output: If True, print formatted table (default behavior)

    Returns:
    - If return_results=True: DataFrame with columns [lag, test_stat, p_value, reject_h0]
    - If return_results=False: None (backward compatible)
    """
    if print_output:
        print("\n--- LJUNG-BOX TEST FOR AUTOCORRELATION ---")
        print("H0: Residuals are independently distributed (no autocorrelation)")
        print("Reject H0 if p-value < 0.05\n")

    # Run test at multiple lags
    lb_results = acorr_ljungbox(residuals, lags=lags, return_df=True)

    if print_output:
        print(f"{'Lag':<10} {'Test Statistic':<20} {'P-value':<15} {'Result'}")
        print("-" * 60)

    results_data = []
    for lag in lags:
        if lag in lb_results.index:
            stat = lb_results.loc[lag, 'lb_stat']
            pval = lb_results.loc[lag, 'lb_pvalue']
            reject_h0 = pval < 0.05

            if print_output:
                result = "REJECT H0 (autocorr present)" if reject_h0 else "Fail to reject H0"
                print(f"{lag:<10} {stat:<20.4f} {pval:<15.4f} {result}")

            results_data.append({
                'lag': lag,
                'test_stat': stat,
                'p_value': pval,
                'reject_h0': reject_h0
            })

    if return_results:
        return pd.DataFrame(results_data)
    return None


def run_heteroskedasticity_tests(residuals, nlags=10):
    """
    Tests for heteroskedasticity and ARCH effects.

    1. Engle's ARCH test (Lagrange Multiplier test)
    2. Ljung-Box Q test on squared residuals

    Following Fredriksson (2016) Table 2.
    """
    print("\n--- HETEROSKEDASTICITY AND ARCH EFFECTS TESTS ---")

    # 1. Engle's ARCH Test (Lagrange Multiplier)
    print("\n1. ENGLE'S ARCH TEST (Lagrange Multiplier)")
    print("   H0: No ARCH effects (homoskedastic residuals)")
    print("   Reject H0 if p-value < 0.05\n")

    try:
        # ARCH test with specified lags
        lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=nlags)

        print(f"   LM Statistic: {lm_stat:.4f}")
        print(f"   LM P-value:   {lm_pval:.4f}")
        print(f"   F-Statistic:  {f_stat:.4f}")
        print(f"   F P-value:    {f_pval:.4f}")

        if lm_pval < 0.05:
            print(f"   Result: REJECT H0 - ARCH effects detected (use GARCH model)")
        else:
            print(f"   Result: Fail to reject H0 - No significant ARCH effects")

    except Exception as e:
        print(f"   Error running ARCH test: {e}")

    # 2. Ljung-Box Q test on squared residuals
    print("\n2. LJUNG-BOX Q TEST ON SQUARED RESIDUALS")
    print("   H0: No autocorrelation in squared residuals")
    print("   Reject H0 if p-value < 0.05\n")

    try:
        squared_resid = residuals ** 2
        lb_squared = acorr_ljungbox(squared_resid, lags=[5, 10, 15, 20], return_df=True)

        print(f"   {'Lag':<10} {'Q-Statistic':<20} {'P-value':<15} {'Result'}")
        print("   " + "-" * 60)

        for lag in [5, 10, 15, 20]:
            if lag in lb_squared.index:
                stat = lb_squared.loc[lag, 'lb_stat']
                pval = lb_squared.loc[lag, 'lb_pvalue']
                result = "REJECT H0 (heteroskedasticity)" if pval < 0.05 else "Fail to reject H0"
                print(f"   {lag:<10} {stat:<20.4f} {pval:<15.4f} {result}")

    except Exception as e:
        print(f"   Error running Ljung-Box on squared residuals: {e}")


def run_stationarity_tests(series, series_name="Series"):
    """
    Stationarity tests: Augmented Dickey-Fuller (ADF) and Dickey-Fuller GLS (DF-GLS).

    H0: Series has a unit root (non-stationary)
    Reject H0 if p-value < 0.05 (series is stationary)

    Following Fredriksson (2016) Table G in appendix.
    """
    print(f"\n--- STATIONARITY TESTS: {series_name} ---")
    print("H0: Series has a unit root (non-stationary)")
    print("Reject H0 if p-value < 0.05 (series is stationary)\n")

    # 1. Augmented Dickey-Fuller (ADF) Test
    print("1. AUGMENTED DICKEY-FULLER (ADF) TEST")
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_stat, adf_pval = adf_result[0], adf_result[1]
        adf_lags = adf_result[2]

        print(f"   ADF Statistic: {adf_stat:.4f}")
        print(f"   P-value:       {adf_pval:.4f}")
        print(f"   Lags used:     {adf_lags}")
        print(f"   Critical values: 1%={adf_result[4]['1%']:.3f}, 5%={adf_result[4]['5%']:.3f}, 10%={adf_result[4]['10%']:.3f}")

        if adf_pval < 0.05:
            print(f"   Result: REJECT H0 - Series is STATIONARY")
        else:
            print(f"   Result: Fail to reject H0 - Series is NON-STATIONARY")

    except Exception as e:
        print(f"   Error running ADF test: {e}")

    # 2. Dickey-Fuller GLS (DF-GLS) Test
    print("\n2. DICKEY-FULLER GLS (DF-GLS) TEST")
    try:
        dfgls = DFGLS(series.dropna())
        dfgls_stat = dfgls.stat
        dfgls_pval = dfgls.pvalue

        print(f"   DF-GLS Statistic: {dfgls_stat:.4f}")
        print(f"   P-value:          {dfgls_pval:.4f}")
        print(f"   Critical values:  1%={dfgls.critical_values['1%']:.3f}, 5%={dfgls.critical_values['5%']:.3f}, 10%={dfgls.critical_values['10%']:.3f}")

        if dfgls_pval < 0.05:
            print(f"   Result: REJECT H0 - Series is STATIONARY")
        else:
            print(f"   Result: Fail to reject H0 - Series is NON-STATIONARY")

    except Exception as e:
        print(f"   Error running DF-GLS test: {e}")


# --- 4. MODELING FUNCTIONS ---

def run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, plots_dir="plots"):
    """
    Estimate time-varying parameter (TVP) model for the wind coefficient using state-space Kalman filter.

    Model specification:
        Observation: y_t = beta_t * w_t + controls_t' * gamma + e_t
        State:       beta_t = beta_{t-1} + u_t

    Uses Frisch-Waugh-Lovell partialling out to control for other regressors,
    then estimates a random-walk state-space model for the wind coefficient.

    Note: Always uses Wind_Forecast_Log (logged wind variable).

    **IMPORTANT - NEEDS FURTHER INVESTIGATION:**
    Current implementation on hourly data shows excessive high-frequency volatility,
    suggesting the model is overfitting to noise rather than capturing genuine
    structural variation in the wind coefficient. The random walk state process
    with unconstrained variance allows β_t to change hour-to-hour, which is
    economically implausible.

    Potential improvements to explore:
    1. Aggregate to daily data (daily averages or peak prices) to reduce noise
    2. Constrain state variance to force smoother evolution
    3. Use AR(1) state process instead of random walk for mean reversion
    4. Consider fixed-coefficient models with structural breaks instead
    5. Estimate on rolling windows rather than full state-space approach

    NOTE: Kalman filters may not be the optimal approach for analyzing coefficient
    evolution over time in this context. Alternative methods (rolling regressions,
    regime-switching models, or dummy variable interactions) may be more appropriate.

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier (e.g., 'SE1')
    - Y: Dependent variable (price series)
    - exog_vars: List of exogenous variable names
    - plots_dir: Directory to save output plots
    """
    print("\n" + "="*80)
    print(f"TVP KALMAN FILTER ANALYSIS - TIME-VARYING WIND COEFFICIENT ({zone})")
    print("="*80)
    print("\nModel: y_t = beta_t * w_t + controls' * gamma + e_t")
    print("State: beta_t = beta_{t-1} + u_t (random walk)")

    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Always use logged wind variable
    wind_col = 'Wind_Forecast_Log'

    print(f"\nWind variable: {wind_col}")

    # Step 2: Create control columns (exog_vars minus wind)
    control_cols = [col for col in exog_vars if col != wind_col]
    print(f"Control variables: {control_cols}")

    # Extract data
    y = Y.copy()
    w = df[wind_col].copy()
    controls = df[control_cols].copy()

    # Align indices and drop any NaN
    combined = pd.concat([y, w, controls], axis=1).dropna()
    y = combined.iloc[:, 0]
    w = combined.iloc[:, 1]
    controls = combined.iloc[:, 2:]

    print(f"\nObservations after alignment: {len(y)}")

    # Step 3: Frisch-Waugh-Lovell partialling out
    print("\n--- Frisch-Waugh-Lovell Partialling Out ---")
    print("Removing control variable effects from both Y and Wind...")

    # Add constant to controls
    controls_with_const = sm.add_constant(controls)

    # Regress Y on controls and get residuals
    y_on_controls = sm.OLS(y, controls_with_const).fit()
    y_star = y_on_controls.resid
    print(f"  y* = residuals from OLS(Y ~ const + controls), R²={y_on_controls.rsquared:.4f}")

    # Regress wind on controls and get residuals
    w_on_controls = sm.OLS(w, controls_with_const).fit()
    w_star = w_on_controls.resid
    print(f"  w* = residuals from OLS(Wind ~ const + controls), R²={w_on_controls.rsquared:.4f}")

    # Step 4: Define custom TVP state-space model
    print("\n--- Fitting State-Space Model ---")
    print("Estimating time-varying wind coefficient via Kalman filter...")

    class TVPWind(sm.tsa.statespace.MLEModel):
        """
        Time-varying parameter model for wind coefficient.
        State equation: beta_t = beta_{t-1} + u_t (random walk)
        Observation equation: y*_t = beta_t * w*_t + e_t
        """
        def __init__(self, y_star, w_star):
            super().__init__(y_star, k_states=1)
            # Store w_star for use in design matrix
            self._w_star = w_star.values.reshape(1, 1, -1)
            # Design matrix: (1, k_states, nobs) - contains w_star
            self.ssm['design'] = self._w_star
            # Transition matrix: [[1.0]] (random walk)
            self.ssm['transition'] = np.array([[1.0]])
            # Selection matrix: [[1.0]]
            self.ssm['selection'] = np.array([[1.0]])
            # Initialize with approximate diffuse prior
            self.initialize_approximate_diffuse()

        @property
        def param_names(self):
            return ['log_obs_var', 'log_state_var']

        @property
        def start_params(self):
            # Starting values: r=1 (log=0), q≈0.14 (log=-2)
            return np.array([0.0, -2.0])

        def update(self, params, **kwargs):
            # Observation variance (r)
            r = np.exp(params[0])
            # State variance (q)
            q = np.exp(params[1])
            self.ssm['obs_cov'] = np.array([[r]])
            self.ssm['state_cov'] = np.array([[q]])

    # Fit the model
    tvp_model = TVPWind(y_star, w_star)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        tvp_results = tvp_model.fit(disp=False)

    # Step 5: Extract results
    beta_t = tvp_results.smoothed_state[0]
    se_t = np.sqrt(tvp_results.smoothed_state_cov[0, 0, :])
    upper_95 = beta_t + 1.96 * se_t
    lower_95 = beta_t - 1.96 * se_t

    # Get estimated variances
    r_hat = np.exp(tvp_results.params.iloc[0])  # Observation variance
    q_hat = np.exp(tvp_results.params.iloc[1])  # State variance

    # Step 6: Print summary statistics
    print("\n" + "="*80)
    print("TVP WIND COEFFICIENT - SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTime-varying beta_t (wind coefficient):")
    print(f"  Mean:   {beta_t.mean():.6f}")
    print(f"  Std:    {beta_t.std():.6f}")
    print(f"  Min:    {beta_t.min():.6f}")
    print(f"  Max:    {beta_t.max():.6f}")
    print(f"  Final:  {beta_t[-1]:.6f}")

    print(f"\nEstimated variances:")
    print(f"  Observation variance (r): {r_hat:.6f}")
    print(f"  State variance (q):       {q_hat:.6f}")
    print(f"  Signal-to-noise ratio:    {q_hat/r_hat:.6f}")

    print(f"\nModel fit:")
    print(f"  Log-likelihood: {tvp_results.llf:.2f}")
    print(f"  AIC:            {tvp_results.aic:.2f}")
    print(f"  BIC:            {tvp_results.bic:.2f}")

    # Step 7: Create and save plot
    print(f"\n--- Creating TVP Wind Coefficient Plot ---")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create time index
    time_index = y.index

    # Plot beta_t with confidence bands
    ax.plot(time_index, beta_t, color='blue', linewidth=1.5, label=r'$\beta_t$ (Wind coefficient)')
    ax.fill_between(time_index, lower_95, upper_95, color='blue', alpha=0.2, label='95% CI')

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add horizontal line at mean
    ax.axhline(y=beta_t.mean(), color='red', linestyle=':', linewidth=1.5,
               label=f'Mean = {beta_t.mean():.4f}')

    ax.set_title(f'Time-Varying Wind Coefficient - {zone}\n(Kalman Filter State-Space Estimation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(r'$\beta_t$ (Wind Coefficient)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation with key statistics
    stats_text = (f'Mean: {beta_t.mean():.4f}\n'
                  f'Std: {beta_t.std():.4f}\n'
                  f'Min: {beta_t.min():.4f}\n'
                  f'Max: {beta_t.max():.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot to zone-specific subfolder
    zone_plots_dir = os.path.join(plots_dir, zone)
    os.makedirs(zone_plots_dir, exist_ok=True)
    plot_path = os.path.join(zone_plots_dir, f'tvp_beta_wind_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("TVP KALMAN FILTER ANALYSIS COMPLETE")
    print("="*80)

    return beta_t, se_t, tvp_results


def get_regression_variable_names(df, target_region='SE1'):
    """Return dependent variable name and exogenous variable names used in regression."""
    y_name = 'Price_Log_Deseasonalized'
    exog_vars = [
        'Wind_Forecast_Log',
        'Hydro_Reserves_Log_Deseasonalized',
        'Net_Exchange',
        'Consumption_Log_Deseasonalized',
        'Oil_Price_Log_Deseasonalized',
        'Gas_Price_Log_Deseasonalized'
    ]

    trading_partners = TRADING_PARTNERS.get(target_region, [])
    for partner in trading_partners:
        bneck_col = f'BNECK_{target_region}_{partner}'
        if bneck_col in df.columns:
            exog_vars.append(bneck_col)

    return y_name, exog_vars


def preprocess_data_for_regression(df_raw,
                                   negative_price_handling='clip',
                                   outlier_method='fredriksson',
                                   handle_outliers_before_log=False,
                                   suppress_output=False,
                                   save_temp_plots=True):
    """
    Apply standard preprocessing sequence and return transformed data for regression.
    """
    if suppress_output:
        stream = io.StringIO()
        context = contextlib.redirect_stdout(stream)
    else:
        context = contextlib.nullcontext()

    with context:
        data = handle_negative_prices(df_raw.copy(), method=negative_price_handling)

        if handle_outliers_before_log:
            if outlier_method == 'fredriksson':
                data, _ = handle_outliers_fredriksson(data, apply_to_raw=True)
            elif outlier_method == 'gianfreda':
                data, _ = handle_outliers_gianfreda(data, apply_to_raw=True)
            else:
                raise ValueError(
                    f"Unknown outlier method: {outlier_method}. Choose 'fredriksson' or 'gianfreda'."
                )

        data = apply_log_transform(data, save_temp_plots=save_temp_plots)
        data = deseasonalize_logged_variables(data, save_temp_plots=save_temp_plots)

        if not handle_outliers_before_log:
            if outlier_method == 'fredriksson':
                data, _ = handle_outliers_fredriksson(data, apply_to_raw=False)
            elif outlier_method == 'gianfreda':
                data, _ = handle_outliers_gianfreda(data, apply_to_raw=False)
            else:
                raise ValueError(
                    f"Unknown outlier method: {outlier_method}. Choose 'fredriksson' or 'gianfreda'."
                )

    return data


def run_rolling_window_analysis(df, zone, Y=None, exog_vars=None,
                                window_years=3, step_years=1, min_obs=24*180,
                                plots_dir="plots", results_dir="results",
                                use_window_local_preprocessing=False,
                                target_region='SE1',
                                negative_price_handling='clip',
                                outlier_method='fredriksson',
                                handle_outliers_before_log=False):
    """
    Estimate wind coefficient using overlapping rolling windows with OLS.

    use_window_local_preprocessing=False:
      expects preprocessed df + Y/exog_vars (legacy behavior).
    use_window_local_preprocessing=True:
      applies full preprocessing independently inside each window.
    """
    from dateutil.relativedelta import relativedelta

    print("\n" + "="*80)
    print("ROLLING-WINDOW WIND COEFFICIENT ESTIMATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Window size: {window_years} years")
    step_months = int(round(step_years * 12)) if step_years < 1 else None
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")

    if use_window_local_preprocessing:
        y_name, exog_names = get_regression_variable_names(df, target_region=target_region)
        tmp = df.sort_index().copy()
        print("  Preprocessing mode: window-local")
        print(f"  Raw data range: {tmp.index.min()} to {tmp.index.max()}")
        print(f"  Total raw observations: {len(tmp):,}")
    else:
        if Y is None or exog_vars is None:
            raise ValueError("Legacy rolling mode requires Y and exog_vars.")
        y_name = Y.name
        exog_names = exog_vars
        cols_needed = [y_name] + exog_names
        tmp = df[cols_needed].dropna().copy()
        tmp = tmp.sort_index()
        print("  Preprocessing mode: full-sample (legacy)")
        print(f"  Data range: {tmp.index.min()} to {tmp.index.max()}")
        print(f"  Total observations after cleaning: {len(tmp):,}")

    wind_col = [col for col in exog_names if 'Wind' in col and 'Forecast' in col][0]
    control_cols = [col for col in exog_names if col != wind_col]
    print(f"\nTarget variable: {wind_col}")
    print(f"Control variables: {control_cols}")

    results = []
    start_date = tmp.index.min()
    end_of_data = tmp.index.max()

    total_windows = 0
    temp_start = start_date
    while temp_start <= end_of_data:
        total_windows += 1
        if step_months:
            temp_start = temp_start + relativedelta(months=step_months)
        else:
            temp_start = temp_start + relativedelta(years=int(step_years))

    print(f"\n--- Estimating Rolling Windows ---")
    print(f"Total candidate windows: {total_windows}\n")

    processed_count = 0
    valid_count = 0
    current_start = start_date

    while current_start <= end_of_data:
        processed_count += 1
        window_end = current_start + relativedelta(years=window_years)
        raw_window = tmp[(tmp.index >= current_start) & (tmp.index < window_end)]

        if len(raw_window) < min_obs:
            if step_months:
                current_start = current_start + relativedelta(months=step_months)
            else:
                current_start = current_start + relativedelta(years=int(step_years))
            continue

        if use_window_local_preprocessing:
            try:
                window_processed = preprocess_data_for_regression(
                    raw_window,
                    negative_price_handling=negative_price_handling,
                    outlier_method=outlier_method,
                    handle_outliers_before_log=handle_outliers_before_log,
                    suppress_output=True,
                    save_temp_plots=False
                )
            except Exception as e:
                print(f"[{processed_count}/{total_windows}] Window {current_start.date()} skipped (preprocessing failed: {e})")
                if step_months:
                    current_start = current_start + relativedelta(months=step_months)
                else:
                    current_start = current_start + relativedelta(years=int(step_years))
                continue

            cols_needed = [y_name] + exog_names
            window_data = window_processed[cols_needed].dropna().copy()
            window_data = window_data.sort_index()
        else:
            window_data = raw_window

        if len(window_data) >= min_obs:
            valid_count += 1
            actual_start_year = window_data.index.min().year
            actual_end_year = window_data.index.max().year
            print(f"[{processed_count}/{total_windows}] Window {actual_start_year} to {actual_end_year}... ", end="")

            X = sm.add_constant(window_data[exog_names])
            y = window_data[y_name]
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})

            window_midpoint = current_start + relativedelta(months=window_years * 6)

            results.append({
                'window_start': current_start,
                'window_end': window_data.index.max(),
                'window_midpoint': window_midpoint,
                'beta_wind': model.params[wind_col],
                'se_wind': model.bse[wind_col],
                't_stat': model.tvalues[wind_col],
                'pvalue': model.pvalues[wind_col],
                'n_obs': len(window_data),
                'r_squared': model.rsquared
            })

            print(f"beta_wind={model.params[wind_col]:.4f}, p={model.pvalues[wind_col]:.4f}")

        if step_months:
            current_start = current_start + relativedelta(months=step_months)
        else:
            current_start = current_start + relativedelta(years=int(step_years))

    if not results:
        print("\nWARNING: No valid windows found. Check data range and window parameters.")
        return

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("ROLLING-WINDOW SUMMARY STATISTICS")
    print("="*80)
    print(f"\nNumber of windows analyzed: {len(results_df)}")
    print(f"Valid windows / candidates: {valid_count}/{total_windows}")
    print(f"\nWind coefficient (beta):")
    print(f"  Mean:   {results_df['beta_wind'].mean():.6f}")
    print(f"  Std:    {results_df['beta_wind'].std():.6f}")
    print(f"  Min:    {results_df['beta_wind'].min():.6f}")
    print(f"  Max:    {results_df['beta_wind'].max():.6f}")

    sig_count = (results_df['pvalue'] < 0.05).sum()
    print(f"\nSignificance at 5% level: {sig_count}/{len(results_df)} windows "
          f"({100*sig_count/len(results_df):.1f}%)")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'rolling_wind_coef_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    midpoints = pd.to_datetime(results_df['window_midpoint'])
    beta_values = results_df['beta_wind'].values
    se_values = results_df['se_wind'].values

    upper_95 = beta_values + 1.96 * se_values
    lower_95 = beta_values - 1.96 * se_values

    ax.plot(midpoints, beta_values, color='blue', linewidth=2, marker='o',
            markersize=6, label=r'$\beta_{wind}$ coefficient')
    ax.fill_between(midpoints, lower_95, upper_95, color='blue', alpha=0.2,
                    label='95% CI')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
               label='Zero')

    mean_beta = results_df['beta_wind'].mean()
    ax.axhline(y=mean_beta, color='red', linestyle=':', linewidth=1.5,
               label=f'Mean = {mean_beta:.4f}')

    ax.set_title(f'Rolling-Window Wind Coefficient - {zone}\n'
                 f'({window_years}-year windows, {step_years}-year steps, Newey-West SE)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Window Midpoint', fontsize=12)
    ax.set_ylabel(r'$\beta_{wind}$ (Wind Coefficient)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = (f'Windows: {len(results_df)}\n'
                  f'Mean: {mean_beta:.4f}\n'
                  f'Std: {results_df["beta_wind"].std():.4f}\n'
                  f'Sig (p<0.05): {sig_count}/{len(results_df)}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    zone_plots_dir = os.path.join(plots_dir, zone)
    os.makedirs(zone_plots_dir, exist_ok=True)
    plot_path = os.path.join(zone_plots_dir, f'rolling_wind_coef_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("ROLLING-WINDOW ANALYSIS COMPLETE")
    print("="*80)

def _get_break_model_tag(estimation_model, dynamic_armax_order=(3, 0, 3)):
    """Return filesystem-safe tag used in break-analysis outputs."""
    if estimation_model == 'ols':
        return 'ols'
    if estimation_model == 'dynamic_armax':
        p, d, q = dynamic_armax_order
        return f'dynamic_armax_{p}{d}{q}'
    raise ValueError(
        f"Unknown structural-break estimation model: '{estimation_model}'. "
        "Allowed values: 'ols', 'dynamic_armax'."
    )


def _get_break_model_label(estimation_model, dynamic_armax_order=(3, 0, 3)):
    """Return human-readable label for plot titles and summaries."""
    if estimation_model == 'ols':
        return 'OLS'
    if estimation_model == 'dynamic_armax':
        return f'Dynamic ARMAX{dynamic_armax_order}'
    raise ValueError(
        f"Unknown structural-break estimation model: '{estimation_model}'. "
        "Allowed values: 'ols', 'dynamic_armax'."
    )


def _extract_armax_wind_coef(armax_model, wind_col):
    """Robustly extract wind coefficient and standard error from ARIMAResults."""
    param_names = list(armax_model.param_names)
    params = np.asarray(armax_model.params)
    bse = np.asarray(armax_model.bse)
    pvalues = np.asarray(armax_model.pvalues)

    if wind_col in param_names:
        idx = param_names.index(wind_col)
        return params[idx], bse[idx], pvalues[idx]

    normalized = [name.replace('x', '').replace('.', '').strip().lower() for name in param_names]
    target_norm = wind_col.lower()
    for idx, name_norm in enumerate(normalized):
        if target_norm in name_norm:
            return params[idx], bse[idx], pvalues[idx]

    raise ValueError(
        f"Could not locate wind coefficient '{wind_col}' in ARMAX parameter names: {param_names}"
    )


def _extract_armax_exog_anchor(armax_model, exog_names, scale_means=None, scale_stds=None):
    """
    Extract exogenous coefficient anchor vector (const + exog terms) from ARMAX results.
    Excludes AR/MA/sigma parameters by explicit term selection.
    """
    param_names = list(armax_model.param_names)
    params = np.asarray(armax_model.params)
    param_map = dict(zip(param_names, params))

    target_terms = ['const'] + list(exog_names)
    missing = [term for term in target_terms if term not in param_map]
    if missing:
        raise ValueError(
            f"Missing exogenous terms in ARMAX parameter map: {missing}. "
            f"Available terms: {param_names}"
        )

    anchor = {term: float(param_map[term]) for term in target_terms}

    # If model was fit with scaled exogenous variables, convert coefficients back to raw units.
    if scale_means is not None and scale_stds is not None:
        scale_means = pd.Series(scale_means)
        scale_stds = pd.Series(scale_stds).replace(0, 1.0).fillna(1.0)
        for col in exog_names:
            if col in anchor and col in scale_stds.index:
                anchor[col] = float(anchor[col]) / float(scale_stds[col])
        if 'const' in anchor:
            const_raw = float(anchor['const'])
            for col in exog_names:
                if col in anchor and col in scale_means.index:
                    const_raw -= float(anchor[col]) * float(scale_means[col])
            anchor['const'] = const_raw

    return anchor


def _save_armax_exog_anchor_csv(anchor_map, zone, source_order, results_dir='results'):
    """Save exogenous anchor coefficients as one-row-per-term CSV."""
    os.makedirs(results_dir, exist_ok=True)
    p, d, q = source_order
    timestamp = pd.Timestamp.now().isoformat()
    rows = []
    for term, coef in anchor_map.items():
        rows.append({
            'zone': zone,
            'source_order': str(tuple(source_order)),
            'term': term,
            'coef': float(coef),
            'timestamp': timestamp
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, f'armax_exog_anchor_{zone}_order_{p}_{d}_{q}.csv')
    out_df.to_csv(out_path, index=False)
    return out_path


def _load_armax_exog_anchor_csv(zone, source_order=(1, 0, 1), results_dir='results'):
    """
    Load exogenous anchor coefficients from CSV (one row per term).
    Returns dict(term -> coef) or None if file is unavailable/invalid.
    """
    p, d, q = source_order
    path = os.path.join(results_dir, f'armax_exog_anchor_{zone}_order_{p}_{d}_{q}.csv')
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required = {'term', 'coef'}
    if not required.issubset(df.columns):
        return None

    anchor_map = {}
    for _, row in df.iterrows():
        term = str(row['term'])
        coef = row['coef']
        if pd.isna(term) or pd.isna(coef):
            continue
        anchor_map[term] = float(coef)

    return anchor_map if anchor_map else None


def _zscore_exog(X_exog):
    """Z-score exogenous variables column-wise; keep zero-variance cols unchanged."""
    means = X_exog.mean(axis=0)
    stds = X_exog.std(axis=0, ddof=0)
    stds_safe = stds.replace(0, 1.0).fillna(1.0)
    X_scaled = (X_exog - means) / stds_safe
    return X_scaled, means, stds_safe


def _transform_anchor_raw_to_scaled(exog_anchor_map, means, stds):
    """
    Convert raw-scale anchor map (const + exog betas) to scaled-x parameterization.
    If const is missing, only exog coefficients are transformed.
    """
    if not exog_anchor_map:
        return exog_anchor_map

    out = {}
    for col in means.index:
        if col in exog_anchor_map:
            out[col] = float(exog_anchor_map[col]) * float(stds[col])

    if 'const' in exog_anchor_map:
        const_scaled = float(exog_anchor_map['const'])
        for col in means.index:
            if col in exog_anchor_map:
                const_scaled += float(exog_anchor_map[col]) * float(means[col])
        out['const'] = const_scaled

    return out


def _build_exog_backtransform_table(armax_model, exog_names, scale_means=None, scale_stds=None):
    """
    Build a comparison table for scaled vs raw exogenous coefficients.
    If scale stats are not provided, raw columns equal scaled columns.
    """
    param_names = list(armax_model.param_names)
    params = np.asarray(armax_model.params)
    bse = np.asarray(armax_model.bse)
    pvalues = np.asarray(armax_model.pvalues)

    means = pd.Series(scale_means) if scale_means is not None else pd.Series(dtype=float)
    stds = pd.Series(scale_stds) if scale_stds is not None else pd.Series(dtype=float)
    stds = stds.replace(0, 1.0).fillna(1.0)

    rows = []
    for term in exog_names:
        if term not in param_names:
            continue
        idx = param_names.index(term)
        coef_scaled = float(params[idx])
        se_scaled = float(bse[idx])
        pval = float(pvalues[idx])

        if term in stds.index:
            coef_raw = coef_scaled / float(stds[term])
            se_raw = se_scaled / float(stds[term])
        else:
            coef_raw = coef_scaled
            se_raw = se_scaled

        rows.append({
            'term': term,
            'coef_scaled': coef_scaled,
            'se_scaled': se_scaled,
            'coef_raw': coef_raw,
            'se_raw': se_raw,
            'p_value': pval
        })

    return pd.DataFrame(rows)


def _attach_inferred_frequency(y, X_exog):
    """
    Attach inferred frequency metadata to datetime index when possible.
    This avoids noisy statsmodels date-index warnings without changing values.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        return y, X_exog, None

    inferred = y.index.freqstr or pd.infer_freq(y.index)
    if not inferred:
        return y, X_exog, None

    try:
        new_idx = pd.DatetimeIndex(y.index.values, freq=inferred)
        y_adj = y.copy()
        x_adj = X_exog.copy()
        y_adj.index = new_idx
        x_adj.index = new_idx
        return y_adj, x_adj, inferred
    except Exception:
        return y, X_exog, None


def _validate_armax_baseline_spec(spec):
    """
    Validate and normalize baseline ARMAX specification.
    Returns a normalized spec dict.
    """
    default_spec = {
        'enabled': True,
        'order': (1, 0, 1),
        'extra_ar_lags': [],
        'drop_initial_nan': True,
        'label': 'ARMAX(1,0,1)'
    }
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("ARMAX baseline spec must be a dictionary.")

    normalized = default_spec.copy()
    normalized.update(spec)

    order = normalized.get('order')
    if (not isinstance(order, (tuple, list))) or len(order) != 3:
        raise ValueError("ARMAX baseline spec 'order' must be a tuple/list of length 3, e.g. (1, 0, 1).")
    try:
        order = (int(order[0]), int(order[1]), int(order[2]))
    except Exception:
        raise ValueError("ARMAX baseline spec 'order' values must be integers.")
    if order[0] < 0 or order[2] < 0:
        raise ValueError("ARMAX baseline spec requires non-negative AR/MA orders.")
    normalized['order'] = order

    raw_lags = normalized.get('extra_ar_lags', [])
    if raw_lags is None:
        raw_lags = []
    if not isinstance(raw_lags, (list, tuple)):
        raise ValueError("ARMAX baseline spec 'extra_ar_lags' must be a list/tuple of positive integers.")
    lag_list = []
    for lag in raw_lags:
        if isinstance(lag, bool):
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' cannot contain boolean values.")
        try:
            lag_i = int(lag)
        except Exception:
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' must contain integers.")
        if lag_i <= 0:
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' must contain positive integers.")
        lag_list.append(lag_i)
    if len(set(lag_list)) != len(lag_list):
        raise ValueError("ARMAX baseline spec 'extra_ar_lags' contains duplicates.")
    normalized['extra_ar_lags'] = sorted(lag_list)
    normalized['drop_initial_nan'] = bool(normalized.get('drop_initial_nan', True))

    label = normalized.get('label')
    normalized['label'] = str(label) if label is not None else f"ARMAX{normalized['order']}"
    normalized['enabled'] = bool(normalized.get('enabled', True))

    return normalized


def _prepare_baseline_armax_design(y, X_exog, extra_ar_lags=None, drop_initial_nan=True):
    """
    Build ARMAX design matrix for baseline run, optionally augmenting exogenous set
    with sparse lagged dependent terms (e.g., y_lag_23, y_lag_24, y_lag_25).
    """
    y_aligned = y.copy()
    x_aug = X_exog.copy()
    added_cols = []

    lag_list = [] if extra_ar_lags is None else list(extra_ar_lags)
    for lag in lag_list:
        col = f"y_lag_{int(lag)}"
        x_aug[col] = y_aligned.shift(int(lag))
        added_cols.append(col)

    if drop_initial_nan and len(added_cols) > 0:
        joined = pd.concat([y_aligned.rename('_y'), x_aug], axis=1).dropna()
        y_aligned = joined['_y']
        x_aug = joined.drop(columns=['_y'])

    return y_aligned, x_aug, added_cols


def _build_warm_start_params(y_fit, x_fit, order, exog_anchor_map=None):
    """
    Build start params by mapping ARMAX(0,0,0)-X estimates into target order params.
    Falls back to statsmodels defaults if mapping fails.
    """
    model = sm.tsa.ARIMA(y_fit, exog=x_fit, order=order)
    start_params = np.asarray(model.start_params, dtype=float).copy()
    param_names = list(model.param_names)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=ValueWarning)
            base_fit = sm.tsa.ARIMA(y_fit, exog=x_fit, order=(0, 0, 0)).fit(
                method='statespace',
                method_kwargs={'maxiter': 100}
            )
        base_map = dict(zip(base_fit.param_names, np.asarray(base_fit.params)))
        for i, name in enumerate(param_names):
            if name in base_map:
                start_params[i] = base_map[name]
    except Exception:
        # Keep default start params if warm-start fitting fails.
        pass

    if exog_anchor_map:
        try:
            for i, name in enumerate(param_names):
                if name in exog_anchor_map:
                    start_params[i] = float(exog_anchor_map[name])
        except Exception:
            # Keep existing starts if anchor mapping fails.
            pass

    return start_params


def _fit_armax_with_controls(y, X_exog, order=(3, 0, 3), context_label="",
                             accept_nonconverged=True, maxiter=300, solver='statespace',
                             use_warm_start=True, exog_anchor_map=None, scale_exog=False):
    """
    Fit ARMAX quietly and enforce convergence acceptance.
    Returns dict with keys: ok, model, fail_reason, error, freq, converged, diagnostics.
    """
    y_fit, x_fit, freq = _attach_inferred_frequency(y, X_exog)

    diagnostics = {
        'order': order,
        'converged': False,
        'iterations': None,
        'warnflag': None,
        'fopt': None,
        'gradient_max_abs': None,
        'exog_scaled': bool(scale_exog),
        'context': context_label
    }

    try:
        x_fit_local = x_fit
        anchor_local = exog_anchor_map
        if scale_exog:
            x_fit_local, means, stds = _zscore_exog(x_fit)
            diagnostics['scale_means'] = {k: float(v) for k, v in means.items()}
            diagnostics['scale_stds'] = {k: float(v) for k, v in stds.items()}
            if exog_anchor_map:
                anchor_local = _transform_anchor_raw_to_scaled(exog_anchor_map, means, stds)

        fit_kwargs = {
            'method': solver,
            'method_kwargs': {'maxiter': maxiter}
        }
        if use_warm_start:
            fit_kwargs['start_params'] = _build_warm_start_params(
                y_fit, x_fit_local, order, exog_anchor_map=anchor_local
            )

        with warnings.catch_warnings():
            # Keep console noise minimal while still handling failures explicitly.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=ValueWarning)
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"statsmodels\.tsa\.statespace\.sarimax"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Non-stationary starting autoregressive parameters found.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Non-invertible starting MA parameters found.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="A date index has been provided, but it has no associated frequency information.*"
            )
            fitted = sm.tsa.ARIMA(y_fit, exog=x_fit_local, order=order).fit(**fit_kwargs)

        mle_retvals = getattr(fitted, 'mle_retvals', {})
        converged = bool(mle_retvals.get('converged', False))
        gopt = mle_retvals.get('gopt', None)
        diagnostics.update({
            'converged': converged,
            'iterations': mle_retvals.get('iterations', None),
            'warnflag': mle_retvals.get('warnflag', None),
            'fopt': mle_retvals.get('fopt', None),
            'gradient_max_abs': float(np.max(np.abs(gopt))) if gopt is not None else None
        })

        if not converged and not accept_nonconverged:
            return {
                'ok': False,
                'model': None,
                'fail_reason': 'non_converged',
                'error': f"Non-converged MLE fit ({context_label})",
                'freq': freq,
                'converged': False,
                'diagnostics': diagnostics
            }

        return {
            'ok': True,
            'model': fitted,
            'fail_reason': None,
            'error': None,
            'freq': freq,
            'converged': converged,
            'diagnostics': diagnostics
        }
    except Exception as e:
        return {
            'ok': False,
            'model': None,
            'fail_reason': 'exception',
            'error': f"{type(e).__name__}: {e}",
            'freq': freq,
            'converged': False,
            'diagnostics': diagnostics
        }


def _fit_armax_with_fallback(y, X_exog, primary_order=(3, 0, 3), context_label="",
                             allow_nonconverged=False, maxiter=300, solver='statespace',
                             use_warm_start=True, scale_exog=False, enable_fallback_orders=True,
                             fallback_orders=None):
    """
    Fit ARMAX using a fallback ladder. Prefers converged models.
    """
    orders = [primary_order]
    if enable_fallback_orders:
        if fallback_orders is None:
            fallback_orders = []
        for order in fallback_orders:
            if order not in orders:
                orders.append(order)

    attempts = []
    first_nonconverged_ok = None

    for order in orders:
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog,
            order=order,
            context_label=context_label,
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start,
            scale_exog=scale_exog
        )
        attempts.append({
            'order': order,
            'ok': fit['ok'],
            'converged': fit.get('converged', False),
            'fail_reason': fit.get('fail_reason'),
            'error': fit.get('error'),
            'diagnostics': fit.get('diagnostics', {})
        })

        if fit['ok'] and fit.get('converged', False):
            fit['selected_order'] = order
            fit['attempts'] = attempts
            fit['used_nonconverged'] = False
            return fit

        if fit['ok'] and first_nonconverged_ok is None:
            first_nonconverged_ok = fit
            first_nonconverged_ok['selected_order'] = order

    if first_nonconverged_ok is not None and allow_nonconverged:
        first_nonconverged_ok['attempts'] = attempts
        first_nonconverged_ok['used_nonconverged'] = True
        return first_nonconverged_ok

    return {
        'ok': False,
        'model': None,
        'fail_reason': 'non_converged',
        'error': f"No converged ARMAX fit found for {context_label}",
        'freq': None,
        'converged': False,
        'diagnostics': None,
        'selected_order': None,
        'attempts': attempts,
        'used_nonconverged': False
    }


def _diagnose_nonconvergence_simple(attempt, configured_maxiter=None):
    """
    Classify non-convergence reason using lightweight diagnostics from one attempt.
    """
    if attempt is None:
        return 'unknown', "No diagnostics available for this failed fit."

    if attempt.get('ok') is False and attempt.get('fail_reason') == 'exception':
        err = attempt.get('error', 'Unknown exception')
        return 'unknown', f"Fit failed with exception before optimizer convergence checks: {err}"

    diag = attempt.get('diagnostics', {}) or {}
    iterations = diag.get('iterations', None)
    warnflag = diag.get('warnflag', None)
    grad = diag.get('gradient_max_abs', None)

    if iterations == 0:
        return (
            'immediate_stop',
            "Optimizer stopped at iteration 0: it could not take a valid first improvement step "
            "(often due to hard AR/MA constraints or poor local geometry)."
        )

    if configured_maxiter is not None and iterations is not None:
        if iterations >= int(round(0.95 * configured_maxiter)):
            return (
                'hit_iteration_limit',
                "Optimizer was still working but reached the iteration limit before satisfying convergence tolerance."
            )

    if warnflag not in (0, None):
        return (
            'unstable_or_numerical',
            f"Optimizer ended with warnflag={warnflag}; convergence likely blocked by numerical instability "
            "or an ill-conditioned likelihood surface."
        )

    if grad is not None and grad > 1e-3:
        return (
            'unstable_or_numerical',
            f"Gradient remained materially non-zero ({grad:.3e}), suggesting it did not settle at an optimum."
        )

    return (
        'unknown',
        "Non-convergence occurred, but diagnostics are inconclusive from this run."
    )


def _estimate_rolling_wind_coefficients(tmp, Y_name, exog_vars, wind_col,
                                        window_years=1, step_years=1/12,
                                        min_obs=24*365 - 24*30,
                                        estimation_model='ols',
                                        dynamic_armax_order=(3, 0, 3)):
    """
    Estimate rolling-window wind coefficients with either OLS or dynamic ARMAX.
    Returns: rolling_df, total_candidate_windows, failure_stats.
    """
    from dateutil.relativedelta import relativedelta

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    rolling_results = []
    failure_stats = {
        'failed_windows_total': 0,
        'failed_windows_exception': 0,
        'failed_windows_nonconvergence': 0,
        'used_nonconverged_windows': 0
    }

    # First pass: build candidate windows so we can report progress/ETA accurately.
    candidate_windows = []
    start_date = tmp.index.min()
    end_of_data = tmp.index.max()
    current_start = start_date

    while current_start <= end_of_data:
        window_end = current_start + relativedelta(years=window_years)
        window_data = tmp[(tmp.index >= current_start) & (tmp.index < window_end)]

        if len(window_data) >= min_obs:
            midpoint = current_start + relativedelta(months=window_years * 6)
            candidate_windows.append((current_start, window_end, midpoint, window_data))

        if step_months:
            current_start = current_start + relativedelta(months=step_months)
        else:
            current_start = current_start + relativedelta(years=int(step_years))

    total_candidate_windows = len(candidate_windows)
    batch_start_time = time.perf_counter()

    # Second pass: estimate each candidate and print progress after each attempt.
    for i, (current_start, window_end, midpoint, window_data) in enumerate(candidate_windows, start=1):
        status_label = "ok"
        try:
            if estimation_model == 'ols':
                X = sm.add_constant(window_data[exog_vars])
                y = window_data[Y_name]
                model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
                rolling_results.append({
                    'midpoint': midpoint,
                    'beta_wind': model.params[wind_col],
                    'se_wind': model.bse[wind_col],
                    'pvalue': model.pvalues[wind_col],
                    'r_squared': model.rsquared,
                    'n_obs_window': len(window_data),
                    'model_used': 'ols',
                    'fit_status': 'ok'
                })
            elif estimation_model == 'dynamic_armax':
                y = window_data[Y_name]
                X_exog = window_data[exog_vars]
                fit_result = _fit_armax_with_fallback(
                    y=y,
                    X_exog=X_exog,
                    primary_order=dynamic_armax_order,
                    context_label=f"rolling window {current_start} -> {window_end}",
                    allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                    maxiter=ARMAX_MAXITER,
                    solver=ARMAX_SOLVER,
                    use_warm_start=ARMAX_USE_WARM_START,
                    enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                    fallback_orders=ARMAX_FALLBACK_ORDERS
                )
                if not fit_result['ok']:
                    failure_stats['failed_windows_total'] += 1
                    if fit_result['fail_reason'] == 'non_converged':
                        failure_stats['failed_windows_nonconvergence'] += 1
                        status_label = "failed_nonconverged"
                    else:
                        failure_stats['failed_windows_exception'] += 1
                        status_label = "failed_exception"
                else:
                    armax_model = fit_result['model']
                    beta_wind, se_wind, pvalue = _extract_armax_wind_coef(armax_model, wind_col)
                    fit_status = 'ok'
                    if not fit_result.get('converged', True):
                        failure_stats['used_nonconverged_windows'] += 1
                        fit_status = 'nonconverged_used'
                        status_label = "used_nonconverged"
                    diag = fit_result.get('diagnostics', {})
                    status_label = (
                        f"{status_label}:order={fit_result.get('selected_order', dynamic_armax_order)}"
                        f",iter={diag.get('iterations')},warn={diag.get('warnflag')}"
                    )
                    rolling_results.append({
                        'midpoint': midpoint,
                        'beta_wind': beta_wind,
                        'se_wind': se_wind,
                        'pvalue': pvalue,
                        'r_squared': np.nan,
                        'n_obs_window': len(window_data),
                        'model_used': 'dynamic_armax',
                        'fit_status': fit_status
                    })
            else:
                raise ValueError(
                    f"Unknown structural-break estimation model: '{estimation_model}'. "
                    "Allowed values: 'ols', 'dynamic_armax'."
                )
        except Exception:
            failure_stats['failed_windows_total'] += 1
            failure_stats['failed_windows_exception'] += 1
            status_label = "failed_exception"

        elapsed = time.perf_counter() - batch_start_time
        avg_per_window = elapsed / i
        remaining = total_candidate_windows - i
        eta_sec = avg_per_window * remaining
        print(f"[Window {i}/{total_candidate_windows}] status={status_label} "
              f"remaining={remaining} eta={eta_sec/60:.1f}m")

    rolling_df = pd.DataFrame(rolling_results)
    return rolling_df, total_candidate_windows, failure_stats


def _run_dynamic_break_lr_tests(tmp, Y_name, exog_vars, wind_col, test_dates, dynamic_armax_order=(3, 0, 3)):
    """
    Run ARMAX likelihood-ratio split tests at break dates.
    H0: one ARMAX model for full sample. H1: separate ARMAX models pre/post break.
    """
    results = []
    test_stats = {
        'total_test_dates': len(test_dates),
        'skipped_insufficient_obs': 0,
        'skipped_fit_fail_total': 0,
        'skipped_fit_fail_nonconvergence': 0,
        'skipped_fit_fail_exception': 0,
        'used_nonconverged_test_dates': 0
    }

    for test_info in test_dates:
        break_date = test_info['date']
        source = test_info['source']
        print(f"\nTesting break at {break_date.strftime('%Y-%m-%d')} ({source}) [ARMAX LR]:")

        pre_break = tmp[tmp.index < break_date]
        post_break = tmp[tmp.index >= break_date]

        if len(pre_break) < 100 or len(post_break) < 100:
            print(f"  Skipped: Insufficient observations (pre={len(pre_break)}, post={len(post_break)})")
            test_stats['skipped_insufficient_obs'] += 1
            continue

        try:
            fit_full = _fit_armax_with_fallback(
                y=tmp[Y_name],
                X_exog=tmp[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR full sample ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )
            fit_pre = _fit_armax_with_fallback(
                y=pre_break[Y_name],
                X_exog=pre_break[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR pre-break ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )
            fit_post = _fit_armax_with_fallback(
                y=post_break[Y_name],
                X_exog=post_break[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR post-break ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )

            fit_bundle = [fit_full, fit_pre, fit_post]
            if not all(fr['ok'] for fr in fit_bundle):
                test_stats['skipped_fit_fail_total'] += 1
                if any(fr['fail_reason'] == 'non_converged' for fr in fit_bundle):
                    test_stats['skipped_fit_fail_nonconvergence'] += 1
                    print("  Skipped: Dynamic model fit non-converged")
                else:
                    test_stats['skipped_fit_fail_exception'] += 1
                    first_err = next((fr['error'] for fr in fit_bundle if not fr['ok']), "Unknown dynamic fit error")
                    print(f"  Skipped: Dynamic model fit failed ({first_err})")
                continue

            if any(not fr.get('converged', True) for fr in fit_bundle):
                test_stats['used_nonconverged_test_dates'] += 1

            model_full = fit_full['model']
            model_pre = fit_pre['model']
            model_post = fit_post['model']

            ll_full = float(model_full.llf)
            ll_pre = float(model_pre.llf)
            ll_post = float(model_post.llf)
            lr_df = int(max(len(model_full.params), 1))
            lr_stat = 2.0 * ((ll_pre + ll_post) - ll_full)
            p_value = 1.0 - stats.chi2.cdf(lr_stat, lr_df)

            beta_pre, se_pre, _ = _extract_armax_wind_coef(model_pre, wind_col)
            beta_post, se_post, _ = _extract_armax_wind_coef(model_post, wind_col)
            beta_change = beta_post - beta_pre
            beta_change_pct = (beta_change / abs(beta_pre) * 100.0) if beta_pre != 0 else np.nan

            print(f"  Pre-break:  n={len(pre_break):,}, beta_wind={beta_pre:.4f} (SE={se_pre:.4f})")
            print(f"  Post-break: n={len(post_break):,}, beta_wind={beta_post:.4f} (SE={se_post:.4f})")
            print(f"  Change in beta_wind: {beta_change:.4f} ({beta_change_pct:.1f}%)")
            print(f"  LR statistic: {lr_stat:.2f} (df={lr_df})")
            print(f"  p-value: {p_value:.4e}")
            print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
            print(f"  Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")

            results.append({
                'break_date': break_date,
                'source': source,
                'n_pre': len(pre_break),
                'n_post': len(post_break),
                'beta_wind_pre': beta_pre,
                'beta_wind_post': beta_post,
                'se_pre': se_pre,
                'se_post': se_post,
                'beta_change': beta_change,
                'beta_change_pct': beta_change_pct,
                'lr_statistic': lr_stat,
                'lr_df': lr_df,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'test_type': 'ARMAX_LR'
            })
        except Exception as e:
            print(f"  Skipped: Dynamic model fit failed ({e})")
            test_stats['skipped_fit_fail_total'] += 1
            test_stats['skipped_fit_fail_exception'] += 1
            continue

    return results, test_stats


def run_structural_break_analysis(df, zone, Y, exog_vars,
                                  max_breaks=5, min_segment_length=None,
                                  known_break_dates=None, trimming=0.15,
                                  window_years=1, step_years=1/12,
                                  min_obs=24*365 - 24*30,
                                  config_label=None,
                                  plots_dir="plots", results_dir="results",
                                  estimation_model='ols',
                                  dynamic_armax_order=(3, 0, 3)):
    """
    Detect structural breaks in the wind coefficient using Bai-Perron methodology.

    This function:
    1. Estimates rolling window coefficients to visualize coefficient evolution
    2. Applies Bai-Perron change point detection to identify structural breaks
    3. Runs Chow tests at detected break points and known event dates
    4. Generates CUSUM plots for parameter stability diagnostics
    5. Outputs comprehensive results and visualizations

    Note: Always uses logged variables (Wind_Forecast_Log).

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - Y: Dependent variable (Series)
    - exog_vars: List of exogenous variable column names
    - max_breaks: Maximum number of breaks to detect (default 5)
    - min_segment_length: Minimum observations between breaks (default: 10% of data)
    - known_break_dates: List of dates to test with Chow test (e.g., ['2022-02-24'] for Ukraine invasion).
                          Set to None or [] to only use Bai-Perron detected breaks.
    - trimming: Fraction of data to trim from endpoints for break detection (default 0.15)
    - window_years: Rolling window size in years (default 1)
    - step_years: Step size between windows in years (default 1/12 = 1 month)
    - min_obs: Minimum observations required per window (default 24*365 - 24*30)
    - config_label: Custom label for this configuration (e.g., '1y_window_1m_step').
                    If None, auto-generated from window_years and step_years.
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns:
    - Dictionary with break detection results
    """
    from dateutil.relativedelta import relativedelta

    # Generate config label for file naming if not provided
    if config_label is None:
        step_months = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months:
            step_label = f"{step_months}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"

        config_label = f"{window_label}_window_{step_label}_step"

    print("\n" + "="*80)
    print("STRUCTURAL BREAK ANALYSIS - BAI-PERRON METHODOLOGY")
    print("="*80)

    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Identify wind column from exog_vars
    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    print(f"\nConfiguration:")
    print(f"  Zone: {zone}")
    print(f"  Config label: {config_label}")
    print(f"  Target coefficient: {wind_col}")
    print(f"  Rolling window: {window_years} year(s)")
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")
    print(f"  Maximum breaks to detect: {max_breaks}")
    print(f"  Trimming (endpoints): {trimming*100:.0f}%")
    print(f"  Estimation model: {model_label}")
    if known_break_dates:
        print(f"  Known event dates to test: {known_break_dates}")
    else:
        print(f"  Known event dates: None (using only Bai-Perron detection)")

    # Prepare clean data
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy()
    tmp = tmp.sort_index()

    n_obs = len(tmp)
    print(f"\nData range: {tmp.index.min()} to {tmp.index.max()}")
    print(f"Total observations: {n_obs:,}")

    # Set minimum segment length (default: 10% of data or ~1 year of hourly data)
    if min_segment_length is None:
        min_segment_length = max(int(n_obs * 0.10), 24 * 365)  # At least 1 year
    print(f"Minimum segment length: {min_segment_length:,} observations (~{min_segment_length/(24*365):.1f} years)")

    # Create output directories with structural break subdirectories
    zone_plots_dir = os.path.join(plots_dir, zone, "structural_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)

    zone_results_dir = os.path.join(results_dir, "structural_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    results = {
        'zone': zone,
        'config_label': config_label,
        'window_years': window_years,
        'step_years': step_years,
        'min_obs': min_obs,
        'n_obs': n_obs,
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'detected_breaks': [],
        'chow_tests': [],
        'bic_scores': {}
    }

    # =========================================================================
    # STEP 1: ROLLING WINDOW ESTIMATION (for coefficient time series)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)
    print(f"Window settings: {window_years} year(s) window, "
          f"{f'{step_months} month(s)' if step_months else f'{step_years} year(s)'} step")
    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )

    if rolling_df.empty:
        print("No valid rolling windows available for break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'estimation_model': estimation_model,
            'model_tag': model_tag,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {len(rolling_df)} rolling window coefficients")
    print(f"Successful windows: {len(rolling_df)}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Coefficient range: [{rolling_df['beta_wind'].min():.4f}, {rolling_df['beta_wind'].max():.4f}]")
    results['rolling_fit_stats'] = {
        'total_candidate_windows': int(total_candidate_windows),
        'successful_windows': int(len(rolling_df)),
        **failure_stats
    }

    # =========================================================================
    # STEP 2: BAI-PERRON CHANGE POINT DETECTION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: BAI-PERRON CHANGE POINT DETECTION")
    print("-"*80)

    # Prepare signal for change point detection
    beta_signal = rolling_df['beta_wind'].values.reshape(-1, 1)

    # Method 1: PELT (Pruned Exact Linear Time) - optimal for multiple breaks
    print("\nMethod 1: PELT algorithm (optimal partitioning)")
    algo_pelt = rpt.Pelt(model="rbf", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    # Use BIC-like penalty: pen = log(n) * dim * sigma^2
    # Adjust penalty to control number of breaks
    sigma = np.std(beta_signal)
    pen = np.log(len(beta_signal)) * sigma**2 * 2  # Moderate penalty

    try:
        breaks_pelt = algo_pelt.predict(pen=pen)
        # Remove the last element (which is always n)
        breaks_pelt = [b for b in breaks_pelt if b < len(beta_signal)]
        print(f"  PELT detected {len(breaks_pelt)} break(s) at indices: {breaks_pelt}")
    except Exception as e:
        print(f"  PELT failed: {e}")
        breaks_pelt = []

    # Method 2: Binary Segmentation (faster, good approximation)
    print("\nMethod 2: Binary Segmentation algorithm")
    algo_binseg = rpt.Binseg(model="l2", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    try:
        breaks_binseg = algo_binseg.predict(n_bkps=max_breaks)
        breaks_binseg = [b for b in breaks_binseg if b < len(beta_signal)]
        print(f"  BinSeg detected {len(breaks_binseg)} break(s) at indices: {breaks_binseg}")
    except Exception as e:
        print(f"  BinSeg failed: {e}")
        breaks_binseg = []

    # Method 3: Dynamic Programming (exact solution)
    print("\nMethod 3: Dynamic Programming (exact, slower)")
    algo_dynp = rpt.Dynp(model="l2", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    # Test different numbers of breaks and compute BIC
    print("\n  Testing different numbers of breaks (BIC selection):")
    bic_results = []

    for n_breaks in range(0, max_breaks + 1):
        try:
            if n_breaks == 0:
                # No breaks: cost is total variance
                cost = np.sum((beta_signal - np.mean(beta_signal))**2)
                n_params = 1
            else:
                breaks = algo_dynp.predict(n_bkps=n_breaks)
                breaks = [0] + [b for b in breaks if b < len(beta_signal)] + [len(beta_signal)]

                # Calculate cost (sum of squared residuals within segments)
                cost = 0
                for i in range(len(breaks) - 1):
                    segment = beta_signal[breaks[i]:breaks[i+1]]
                    if len(segment) > 0:
                        cost += np.sum((segment - np.mean(segment))**2)
                n_params = n_breaks + 1  # n_breaks + 1 segment means

            # BIC = n*log(RSS/n) + k*log(n)
            n = len(beta_signal)
            bic = n * np.log(cost / n + 1e-10) + n_params * np.log(n)
            bic_results.append({'n_breaks': n_breaks, 'bic': bic, 'cost': cost})
            print(f"    {n_breaks} breaks: BIC = {bic:.2f}")

        except Exception as e:
            print(f"    {n_breaks} breaks: Failed ({e})")

    # Select optimal number of breaks by BIC
    if bic_results:
        bic_df = pd.DataFrame(bic_results)
        optimal_n_breaks = bic_df.loc[bic_df['bic'].idxmin(), 'n_breaks']
        print(f"\n  Optimal number of breaks (BIC): {int(optimal_n_breaks)}")
        results['bic_scores'] = bic_results

        # Get break points for optimal model
        if optimal_n_breaks > 0:
            optimal_breaks = algo_dynp.predict(n_bkps=int(optimal_n_breaks))
            optimal_breaks = [b for b in optimal_breaks if b < len(beta_signal)]
        else:
            optimal_breaks = []
    else:
        optimal_n_breaks = 0
        optimal_breaks = []

    # Convert break indices to dates
    detected_break_dates = []
    for brk_idx in optimal_breaks:
        if brk_idx < len(rolling_df):
            break_date = rolling_df.iloc[brk_idx]['midpoint']
            detected_break_dates.append(break_date)
            print(f"\n  Break detected at: {break_date.strftime('%Y-%m-%d')}")

    results['detected_breaks'] = detected_break_dates
    results['optimal_n_breaks'] = int(optimal_n_breaks)

    # =========================================================================
    # STEP 3: BREAK TESTS AT DETECTED AND KNOWN BREAK DATES
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: BREAK TESTS FOR STRUCTURAL BREAKS")
    print("-"*80)

    # Combine detected breaks with known event dates
    test_dates = []

    # Add detected break dates
    for bd in detected_break_dates:
        test_dates.append({'date': bd, 'source': 'Bai-Perron detected'})

    # Add known event dates
    if known_break_dates:
        for date_str in known_break_dates:
            test_dates.append({'date': pd.to_datetime(date_str), 'source': 'Known event'})

    # Run break tests
    chow_results = []
    dynamic_lr_test_stats = None
    if estimation_model == 'dynamic_armax':
        chow_results, dynamic_lr_test_stats = _run_dynamic_break_lr_tests(
            tmp=tmp,
            Y_name=Y.name,
            exog_vars=exog_vars,
            wind_col=wind_col,
            test_dates=test_dates,
            dynamic_armax_order=dynamic_armax_order
        )
        results['dynamic_lr_test_stats'] = dynamic_lr_test_stats
        print(f"Dynamic LR test summary: total={dynamic_lr_test_stats['total_test_dates']}, "
              f"skipped_insufficient={dynamic_lr_test_stats['skipped_insufficient_obs']}, "
              f"skipped_fit_fail={dynamic_lr_test_stats['skipped_fit_fail_total']}, "
              f"used_nonconverged={dynamic_lr_test_stats['used_nonconverged_test_dates']}")
    else:
        for test_info in test_dates:
            break_date = test_info['date']
            source = test_info['source']

            print(f"\nTesting break at {break_date.strftime('%Y-%m-%d')} ({source}):")

            pre_break = tmp[tmp.index < break_date]
            post_break = tmp[tmp.index >= break_date]

            if len(pre_break) < 100 or len(post_break) < 100:
                print(f"  Skipped: Insufficient observations (pre={len(pre_break)}, post={len(post_break)})")
                continue

            X_full = sm.add_constant(tmp[exog_vars])
            y_full = tmp[Y.name]
            model_full = sm.OLS(y_full, X_full).fit()
            rss_full = model_full.ssr
            k = len(model_full.params)

            X_pre = sm.add_constant(pre_break[exog_vars])
            y_pre = pre_break[Y.name]
            model_pre = sm.OLS(y_pre, X_pre).fit()
            rss_pre = model_pre.ssr

            X_post = sm.add_constant(post_break[exog_vars])
            y_post = post_break[Y.name]
            model_post = sm.OLS(y_post, X_post).fit()
            rss_post = model_post.ssr

            rss_unrestricted = rss_pre + rss_post
            n = len(tmp)
            f_stat = ((rss_full - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

            beta_pre = model_pre.params[wind_col]
            beta_post = model_post.params[wind_col]
            se_pre = model_pre.bse[wind_col]
            se_post = model_post.bse[wind_col]

            print(f"  Pre-break:  n={len(pre_break):,}, beta_wind={beta_pre:.4f} (SE={se_pre:.4f})")
            print(f"  Post-break: n={len(post_break):,}, beta_wind={beta_post:.4f} (SE={se_post:.4f})")
            print(f"  Change in beta_wind: {beta_post - beta_pre:.4f} ({((beta_post - beta_pre)/abs(beta_pre))*100:.1f}%)")
            print(f"  Chow F-statistic: {f_stat:.2f}")
            print(f"  p-value: {p_value:.4e}")
            print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
            print(f"  Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")

            chow_results.append({
                'break_date': break_date,
                'source': source,
                'n_pre': len(pre_break),
                'n_post': len(post_break),
                'beta_wind_pre': beta_pre,
                'beta_wind_post': beta_post,
                'se_pre': se_pre,
                'se_post': se_post,
                'beta_change': beta_post - beta_pre,
                'beta_change_pct': ((beta_post - beta_pre)/abs(beta_pre))*100,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'test_type': 'CHOW_OLS'
            })

    results['chow_tests'] = chow_results

    # =========================================================================
    # STEP 4: CUSUM TEST FOR PARAMETER STABILITY
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: CUSUM TEST FOR PARAMETER STABILITY")
    print("-"*80)

    # Run recursive OLS and compute CUSUM
    X_full = sm.add_constant(tmp[exog_vars])
    y_full = tmp[Y.name].values

    # For CUSUM, we need recursive residuals
    # Simplified approach: compute rolling prediction errors
    print("\nComputing CUSUM statistics...")

    # Start recursive estimation after initial window
    init_window = max(len(exog_vars) * 10, 24 * 30)  # At least 1 month
    recursive_residuals = []
    recursive_dates = []

    for t in range(init_window, n_obs, 24 * 7):  # Weekly steps for speed
        # Estimate on data up to t
        X_t = sm.add_constant(tmp[exog_vars].iloc[:t])
        y_t = tmp[Y.name].iloc[:t]
        model_t = sm.OLS(y_t, X_t).fit()

        # One-step-ahead prediction error
        if t < n_obs:
            # Get the next observation's exog values and manually add constant
            # (sm.add_constant behaves inconsistently with single-row DataFrames)
            X_next_raw = tmp[exog_vars].iloc[t].values
            X_next = np.concatenate([[1.0], X_next_raw])  # Prepend constant
            y_next = tmp[Y.name].iloc[t]
            pred = model_t.predict(X_next.reshape(1, -1))[0]
            resid = y_next - pred
            recursive_residuals.append(resid)
            recursive_dates.append(tmp.index[t])

    recursive_residuals = np.array(recursive_residuals)
    sigma_resid = np.std(recursive_residuals)

    # Standardized cumulative sum
    cusum = np.cumsum(recursive_residuals) / (sigma_resid * np.sqrt(len(recursive_residuals)))

    # Critical values (5% significance): ±0.948 * sqrt(n) at endpoints
    # Linear boundaries that start at 0 and reach ±0.948*sqrt(n)
    n_cusum = len(cusum)
    t_values = np.arange(1, n_cusum + 1)
    upper_bound = 0.948 * np.sqrt(n_cusum) * (t_values / n_cusum)
    lower_bound = -upper_bound

    # Check for boundary violations
    violations = (cusum > upper_bound) | (cusum < lower_bound)
    n_violations = np.sum(violations)

    print(f"CUSUM observations: {n_cusum}")
    print(f"Boundary violations: {n_violations} ({100*n_violations/n_cusum:.1f}%)")
    if n_violations > 0:
        first_violation_idx = np.where(violations)[0][0]
        first_violation_date = recursive_dates[first_violation_idx]
        print(f"First violation at: {first_violation_date}")
        results['cusum_first_violation'] = first_violation_date
    else:
        print("No boundary violations detected (parameters appear stable)")
        results['cusum_first_violation'] = None

    results['cusum_violations'] = n_violations

    # =========================================================================
    # STEP 5: GENERATE PLOTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: GENERATING DIAGNOSTIC PLOTS")
    print("-"*80)

    # Plot 1: Coefficient evolution with detected breaks
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel A: Rolling coefficient with breaks
    ax1 = axes[0]
    midpoints = pd.to_datetime(rolling_df['midpoint'])
    beta_values = rolling_df['beta_wind'].values
    se_values = rolling_df['se_wind'].values

    ax1.plot(midpoints, beta_values, color='blue', linewidth=2, label=r'$\beta_{wind}$')
    ax1.fill_between(midpoints, beta_values - 1.96*se_values, beta_values + 1.96*se_values,
                     color='blue', alpha=0.2, label='95% CI')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Mark detected breaks
    for i, break_date in enumerate(detected_break_dates):
        ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                    label='Detected break' if i == 0 else None)

    # Mark known event dates
    if known_break_dates:
        for i, date_str in enumerate(known_break_dates):
            ax1.axvline(x=pd.to_datetime(date_str), color='orange', linestyle=':',
                        linewidth=2, label='Known event' if i == 0 else None)

    ax1.set_title(
        f'Wind Coefficient Evolution with Structural Breaks - {zone}\n(Model: {model_label})',
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Date')
    ax1.set_ylabel(r'$\beta_{wind}$')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel B: BIC by number of breaks
    ax2 = axes[1]
    if results['bic_scores']:
        bic_df = pd.DataFrame(results['bic_scores'])
        ax2.bar(bic_df['n_breaks'], bic_df['bic'], color='steelblue', edgecolor='black')
        ax2.axvline(x=results['optimal_n_breaks'], color='red', linestyle='--',
                    linewidth=2, label=f'Optimal: {results["optimal_n_breaks"]} breaks')
        ax2.set_xlabel('Number of Breaks')
        ax2.set_ylabel('BIC')
        ax2.set_title('Model Selection: BIC by Number of Breaks', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'BIC analysis not available\n(ruptures package not installed)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Model Selection: BIC by Number of Breaks', fontsize=12, fontweight='bold')

    # Panel C: CUSUM plot
    ax3 = axes[2]
    ax3.plot(recursive_dates, cusum, color='blue', linewidth=1.5, label='CUSUM')
    ax3.plot(recursive_dates, upper_bound, color='red', linestyle='--', linewidth=1.5, label='5% bounds')
    ax3.plot(recursive_dates, lower_bound, color='red', linestyle='--', linewidth=1.5)
    ax3.fill_between(recursive_dates, lower_bound, upper_bound, color='red', alpha=0.1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('CUSUM')
    ax3.set_title('CUSUM Test for Parameter Stability', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(zone_plots_dir, f'sb_{config_label}_{model_tag}_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6: SAVING RESULTS")
    print("-"*80)

    # Save rolling coefficients
    rolling_csv = os.path.join(zone_results_dir, f'sb_rolling_coef_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    # Save break-test results
    if chow_results:
        chow_df = pd.DataFrame(chow_results)
        if estimation_model == 'dynamic_armax':
            chow_csv = os.path.join(zone_results_dir, f'sb_dynamic_lr_tests_{config_label}_{model_tag}_{zone}.csv')
        else:
            chow_csv = os.path.join(zone_results_dir, f'sb_chow_tests_{config_label}_{model_tag}_{zone}.csv')
        chow_df.to_csv(chow_csv, index=False)
        print(f"Saved break-test results: {chow_csv}")

    # Save summary
    summary_path = os.path.join(zone_results_dir, f'sb_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"STRUCTURAL BREAK ANALYSIS SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Config label: {config_label}\n")
        f.write(f"  Estimation model: {model_label}\n")
        f.write(f"  Rolling window: {window_years} year(s)\n")
        if step_months:
            f.write(f"  Step size: {step_months} month(s)\n")
        else:
            f.write(f"  Step size: {step_years} year(s)\n")
        f.write(f"  Minimum observations per window: {min_obs:,}\n\n")
        if 'rolling_fit_stats' in results:
            rfs = results['rolling_fit_stats']
            f.write("  Rolling fit diagnostics:\n")
            f.write(f"    Candidate windows: {rfs['total_candidate_windows']}\n")
            f.write(f"    Successful windows: {rfs['successful_windows']}\n")
            f.write(f"    Failed windows: {rfs['failed_windows_total']}\n")
            f.write(f"      Non-converged: {rfs['failed_windows_nonconvergence']}\n")
            f.write(f"      Exceptions: {rfs['failed_windows_exception']}\n")
            f.write(f"    Used non-converged fits: {rfs.get('used_nonconverged_windows', 0)}\n\n")

        f.write(f"Data range: {tmp.index.min()} to {tmp.index.max()}\n")
        f.write(f"Total observations: {n_obs:,}\n\n")

        f.write("-"*80 + "\n")
        f.write("BAI-PERRON BREAK DETECTION\n")
        f.write("-"*80 + "\n")
        if results['optimal_n_breaks'] is not None:
            f.write(f"Optimal number of breaks (BIC): {results['optimal_n_breaks']}\n")
            if detected_break_dates:
                f.write("Detected break dates:\n")
                for bd in detected_break_dates:
                    f.write(f"  - {bd.strftime('%Y-%m-%d')}\n")
            else:
                f.write("No breaks detected.\n")
        else:
            f.write("Bai-Perron analysis not available (ruptures not installed)\n")

        f.write("\n" + "-"*80 + "\n")
        if estimation_model == 'dynamic_armax':
            f.write("ARMAX LR TEST RESULTS\n")
            if dynamic_lr_test_stats is not None:
                f.write(f"  Test dates evaluated: {dynamic_lr_test_stats['total_test_dates']}\n")
                f.write(f"  Skipped (insufficient obs): {dynamic_lr_test_stats['skipped_insufficient_obs']}\n")
                f.write(f"  Skipped (fit failures): {dynamic_lr_test_stats['skipped_fit_fail_total']}\n")
                f.write(f"    Non-converged: {dynamic_lr_test_stats['skipped_fit_fail_nonconvergence']}\n")
                f.write(f"    Exceptions: {dynamic_lr_test_stats['skipped_fit_fail_exception']}\n")
                f.write(f"  Used non-converged test dates: {dynamic_lr_test_stats['used_nonconverged_test_dates']}\n\n")
        else:
            f.write("CHOW TEST RESULTS\n")
        f.write("-"*80 + "\n")
        for cr in chow_results:
            f.write(f"\nBreak date: {cr['break_date'].strftime('%Y-%m-%d')} ({cr['source']})\n")
            f.write(f"  Pre-break beta_wind:  {cr['beta_wind_pre']:.4f} (SE={cr['se_pre']:.4f})\n")
            f.write(f"  Post-break beta_wind: {cr['beta_wind_post']:.4f} (SE={cr['se_post']:.4f})\n")
            f.write(f"  Change: {cr['beta_change']:.4f} ({cr['beta_change_pct']:.1f}%)\n")
            if 'lr_statistic' in cr:
                f.write(f"  LR-statistic: {cr['lr_statistic']:.2f} (df={cr['lr_df']}), p-value: {cr['p_value']:.4e}\n")
            else:
                f.write(f"  F-statistic: {cr['f_statistic']:.2f}, p-value: {cr['p_value']:.4e}\n")
            f.write(f"  Significant at 5%: {'YES' if cr['significant_5pct'] else 'NO'}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("CUSUM TEST\n")
        f.write("-"*80 + "\n")
        f.write(f"Boundary violations: {results['cusum_violations']}\n")
        if results['cusum_first_violation']:
            f.write(f"First violation: {results['cusum_first_violation']}\n")
        else:
            f.write("No violations (parameters appear stable)\n")

    print(f"Saved summary: {summary_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("STRUCTURAL BREAK ANALYSIS COMPLETE")
    print("="*80)

    print(f"\nKey findings for {zone}:")
    if results['optimal_n_breaks'] is not None:
        print(f"  - Detected {results['optimal_n_breaks']} structural break(s) via Bai-Perron")
    if chow_results:
        sig_chow = sum(1 for cr in chow_results if cr['significant_5pct'])
        if estimation_model == 'dynamic_armax':
            print(f"  - {sig_chow}/{len(chow_results)} break dates significant at 5% (ARMAX LR)")
        else:
            print(f"  - {sig_chow}/{len(chow_results)} break dates significant at 5% (Chow test)")
    print(f"  - CUSUM violations: {results['cusum_violations']}")

    return results


def run_trend_break_analysis_legacy(df, zone, Y, exog_vars,
                             max_breaks=5, min_segment_pct=0.10,
                             trimming=0.15,
                             window_years=1, step_years=1/12,
                             min_obs=24*365 - 24*30,
                             config_label=None,
                             plots_dir="plots", results_dir="results", show_progress=True,
                             estimation_model='ols',
                             dynamic_armax_order=(3, 0, 3)):
    """
    Detect structural breaks in the TREND of wind coefficient using SEQUENTIAL TESTING.

    This function tests for changes in the SLOPE of coefficient evolution over time,
    using sequential hypothesis testing (0 vs 1 break, 1 vs 2 breaks, etc.).

    Methodology:
    1. Estimate rolling window coefficients
    2. For m = 0, 1, 2, ..., max_breaks:
       - Find optimal break locations using dynamic programming
       - Calculate BIC and F-statistic for m vs m-1 breaks
    3. Select optimal number of breaks via BIC
    4. Perform sequential F-tests for significance

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - Y: Dependent variable (Series)
    - exog_vars: List of exogenous variable column names
    - max_breaks: Maximum number of trend breaks to test (default: 5)
    - min_segment_pct: Minimum segment length as fraction of total windows (default: 0.10)
    - trimming: Fraction of data to trim from endpoints (default 0.15)
    - window_years: Rolling window size in years (default 1)
    - step_years: Step size between windows in years (default 1/12 = 1 month)
    - min_obs: Minimum observations required per window (default 24*365 - 24*30)
    - config_label: Custom label for this configuration (auto-generated if None)
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving results
    - show_progress: Whether to show progress indicators

    Returns:
    - Dictionary with trend break detection results
    """
    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Generate config label if not provided
    if config_label is None:
        step_months_label = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months_label:
            step_label = f"{step_months_label}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"

        config_label = f"{window_label}_window_{step_label}_step_trend"

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    print("\n" + "="*80)
    print("SEQUENTIAL TREND BREAK ANALYSIS")
    print("="*80)

    # Identify wind column
    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]

    print(f"\nConfiguration:")
    print(f"  Zone: {zone}")
    print(f"  Config label: {config_label}")
    print(f"  Target coefficient: {wind_col}")
    print(f"  Rolling window: {window_years} year(s)")
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")
    print(f"  Maximum breaks to test: {max_breaks}")
    print(f"  Minimum segment: {min_segment_pct*100:.0f}% of windows")
    print(f"  Trimming (endpoints): {trimming*100:.0f}%")
    print(f"  Estimation model: {model_label}")

    # Prepare clean data
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy()
    tmp = tmp.sort_index()

    n_obs = len(tmp)
    print(f"\nData range: {tmp.index.min()} to {tmp.index.max()}")
    print(f"Total observations: {n_obs:,}")

    # Create output directories
    zone_plots_dir = os.path.join(plots_dir, zone, "trend_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)

    zone_results_dir = os.path.join(results_dir, "trend_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: ROLLING WINDOW ESTIMATION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)

    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )
    if rolling_df.empty:
        print("No valid rolling windows available for trend break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'method': 'legacy',
            'model_tag': model_tag,
            'estimation_model': estimation_model,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }
    rolling_df['time_idx'] = np.arange(len(rolling_df))
    n_windows = len(rolling_df)

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {n_windows} rolling window coefficients")
    print(f"Successful windows: {n_windows}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Coefficient range: [{rolling_df['beta_wind'].min():.4f}, {rolling_df['beta_wind'].max():.4f}]")

    # Minimum segment length
    min_segment = max(int(n_windows * min_segment_pct), 5)
    print(f"Minimum segment length: {min_segment} windows")

    # =========================================================================
    # STEP 2: HELPER FUNCTIONS FOR SEGMENTED REGRESSION
    # =========================================================================

    def fit_segment(data):
        """Fit linear trend to a segment, return RSS and model"""
        if len(data) < 3:
            return np.inf, None
        X = sm.add_constant(data['time_idx'])
        y = data['beta_wind']
        model = sm.OLS(y, X).fit()
        return model.ssr, model

    def find_optimal_breaks_dp(n_breaks):
        """
        Find optimal break locations for exactly n_breaks using dynamic programming.
        Returns: (break_indices, total_rss, segment_models)
        """
        if n_breaks == 0:
            rss, model = fit_segment(rolling_df)
            return [], rss, [model]

        # Trim indices
        trim_n = int(n_windows * trimming)

        # Dynamic programming approach
        # Cost[i][k] = minimum RSS for first i observations with k breaks
        # We need to find k break points that partition [0, n) into k+1 segments

        # For simplicity with small max_breaks, use recursive search with memoization
        best_breaks = None
        best_rss = np.inf
        best_models = None

        def get_segment_rss(start, end):
            """Get RSS for segment [start, end)"""
            if end - start < 3:
                return np.inf, None
            segment_data = rolling_df.iloc[start:end]
            return fit_segment(segment_data)

        def search_breaks(remaining_breaks, start_idx, current_breaks):
            """Recursive search for optimal break locations"""
            nonlocal best_breaks, best_rss, best_models

            if remaining_breaks == 0:
                # No more breaks to place, fit final segment
                rss_final, model_final = get_segment_rss(start_idx, n_windows)
                if model_final is None:
                    return

                # Calculate total RSS
                total_rss = 0
                models = []
                prev_idx = 0
                for brk in current_breaks:
                    rss_seg, model_seg = get_segment_rss(prev_idx, brk)
                    if model_seg is None:
                        return
                    total_rss += rss_seg
                    models.append(model_seg)
                    prev_idx = brk
                total_rss += rss_final
                models.append(model_final)

                if total_rss < best_rss:
                    best_rss = total_rss
                    best_breaks = current_breaks.copy()
                    best_models = models
                return

            # Try placing next break
            # Break can be placed from (start_idx + min_segment) to (n_windows - remaining_breaks*min_segment - min_segment)
            earliest = max(start_idx + min_segment, trim_n)
            latest = n_windows - remaining_breaks * min_segment - min_segment
            latest = min(latest, n_windows - trim_n)

            for brk in range(earliest, latest + 1):
                current_breaks.append(brk)
                search_breaks(remaining_breaks - 1, brk, current_breaks)
                current_breaks.pop()

        search_breaks(n_breaks, 0, [])

        return best_breaks if best_breaks else [], best_rss, best_models

    # =========================================================================
    # STEP 3: SEQUENTIAL TESTING
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: SEQUENTIAL TREND BREAK TESTING")
    print("-"*80)

    model_results = []

    for m in range(0, max_breaks + 1):
        print(f"\n--- Testing {m} break(s) ---")

        breaks, rss, models = find_optimal_breaks_dp(m)

        if models is None or rss == np.inf:
            print(f"  Could not fit model with {m} breaks (insufficient segment length)")
            break

        # Calculate BIC
        k = 2 * (m + 1)  # Each segment has intercept + slope
        bic = n_windows * np.log(rss / n_windows) + k * np.log(n_windows)

        # Get break dates
        break_dates = [rolling_df.iloc[b]['midpoint'] for b in breaks] if breaks else []

        # Extract slopes for each segment
        slopes = []
        for i, model in enumerate(models):
            slope = model.params['time_idx']
            se = model.bse['time_idx']
            slopes.append({'slope': slope, 'se': se})

        print(f"  Break locations: {breaks if breaks else 'None'}")
        if break_dates:
            print(f"  Break dates: {[d.strftime('%Y-%m-%d') for d in break_dates]}")
        print(f"  RSS: {rss:.4f}")
        print(f"  BIC: {bic:.2f}")
        slopes_str = [f"{s['slope']:.6f}" for s in slopes]
        print(f"  Segment slopes: {slopes_str}")

        model_results.append({
            'n_breaks': m,
            'breaks': breaks,
            'break_dates': break_dates,
            'rss': rss,
            'bic': bic,
            'models': models,
            'slopes': slopes
        })

    # =========================================================================
    # STEP 4: SEQUENTIAL F-TESTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: SEQUENTIAL F-TESTS (m vs m+1 breaks)")
    print("-"*80)

    f_test_results = []

    for i in range(len(model_results) - 1):
        m0 = model_results[i]
        m1 = model_results[i + 1]

        rss0 = m0['rss']
        rss1 = m1['rss']
        k0 = 2 * (m0['n_breaks'] + 1)
        k1 = 2 * (m1['n_breaks'] + 1)

        # F-statistic: (RSS0 - RSS1) / (k1 - k0) / (RSS1 / (n - k1))
        if rss1 > 0 and (k1 - k0) > 0:
            f_stat = ((rss0 - rss1) / (k1 - k0)) / (rss1 / (n_windows - k1))
            p_value = 1 - stats.f.cdf(f_stat, k1 - k0, n_windows - k1)
        else:
            f_stat = np.nan
            p_value = np.nan

        result = {
            'test': f"{m0['n_breaks']} vs {m1['n_breaks']} breaks",
            'f_stat': f_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05 if not np.isnan(p_value) else False,
            'significant_1pct': p_value < 0.01 if not np.isnan(p_value) else False
        }
        f_test_results.append(result)

        sig_5 = "YES" if result['significant_5pct'] else "NO"
        sig_1 = "YES" if result['significant_1pct'] else "NO"
        print(f"\n  {m0['n_breaks']} vs {m1['n_breaks']} breaks:")
        print(f"    F-statistic: {f_stat:.2f}")
        print(f"    p-value: {p_value:.4e}")
        print(f"    Significant at 5%: {sig_5}")
        print(f"    Significant at 1%: {sig_1}")

    # =========================================================================
    # STEP 5: MODEL SELECTION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: MODEL SELECTION SUMMARY")
    print("-"*80)

    # BIC-based selection
    bic_values = [m['bic'] for m in model_results]
    optimal_by_bic = np.argmin(bic_values)
    optimal_model = model_results[optimal_by_bic]

    print(f"\n  BIC comparison:")
    for m in model_results:
        marker = " <-- OPTIMAL" if m['n_breaks'] == optimal_model['n_breaks'] else ""
        print(f"    {m['n_breaks']} breaks: BIC = {m['bic']:.2f}{marker}")

    # Sequential testing selection (stop when F-test is not significant)
    optimal_by_seq = 0
    for i, f_result in enumerate(f_test_results):
        if f_result['significant_5pct']:
            optimal_by_seq = i + 1
        else:
            break

    print(f"\n  Optimal by BIC: {optimal_model['n_breaks']} break(s)")
    print(f"  Optimal by sequential F-test (5%): {optimal_by_seq} break(s)")

    # Use BIC-selected model
    selected_model = optimal_model

    print(f"\n  SELECTED MODEL: {selected_model['n_breaks']} break(s)")
    if selected_model['break_dates']:
        print(f"  Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}")

    # Print segment details
    print(f"\n  Segment details:")
    segment_starts = [0] + selected_model['breaks']
    segment_ends = selected_model['breaks'] + [n_windows]

    for i, (start, end, slope_info) in enumerate(zip(segment_starts, segment_ends, selected_model['slopes'])):
        start_date = rolling_df.iloc[start]['midpoint']
        end_date = rolling_df.iloc[min(end-1, n_windows-1)]['midpoint']
        print(f"    Segment {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"      Slope: {slope_info['slope']:.6f} (SE: {slope_info['se']:.6f})")

    # =========================================================================
    # STEP 6: GENERATE PLOTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: GENERATING DIAGNOSTIC PLOTS")
    print("-"*80)

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    # Coefficient evolution with segmented trend lines
    midpoints = pd.to_datetime(rolling_df['midpoint'])
    beta_values = rolling_df['beta_wind'].values

    ax1.plot(midpoints, beta_values, 'o', color='steelblue', alpha=0.5, markersize=4, label='Rolling coefficients')

    # Plot fitted trend lines for each segment
    colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
    segment_starts = [0] + selected_model['breaks']
    segment_ends = selected_model['breaks'] + [n_windows]

    for i, (start, end, model) in enumerate(zip(segment_starts, segment_ends, selected_model['models'])):
        segment_data = rolling_df.iloc[start:end]
        X_plot = sm.add_constant(segment_data['time_idx'])
        y_pred = model.predict(X_plot)
        segment_midpoints = pd.to_datetime(segment_data['midpoint'])
        color = colors[i % len(colors)]
        slope = selected_model['slopes'][i]['slope']
        ax1.plot(segment_midpoints, y_pred, '-', color=color, linewidth=2.5,
                 label=f'Segment {i+1} (slope={slope:.4f})')

    # Mark break points
    for i, break_date in enumerate(selected_model['break_dates']):
        ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                    label='Break' if i == 0 else None)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_title(f'Wind Coefficient Evolution with {selected_model["n_breaks"]} Trend Break(s) - {zone}\n'
                  f'(Model: {model_label})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(r'$\beta_{wind}$')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(zone_plots_dir, f'tb_{config_label}_{model_tag}_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6: SAVING RESULTS")
    print("-"*80)

    # Save rolling coefficients
    rolling_csv = os.path.join(zone_results_dir, f'tb_rolling_coef_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    # Save model comparison
    model_comparison = []
    for m in model_results:
        row = {
            'n_breaks': m['n_breaks'],
            'bic': m['bic'],
            'rss': m['rss'],
            'break_dates': ';'.join([d.strftime('%Y-%m-%d') for d in m['break_dates']]) if m['break_dates'] else '',
            'slopes': ';'.join([f"{s['slope']:.6f}" for s in m['slopes']])
        }
        model_comparison.append(row)

    comparison_df = pd.DataFrame(model_comparison)
    comparison_csv = os.path.join(zone_results_dir, f'tb_model_comparison_{config_label}_{model_tag}_{zone}.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved model comparison: {comparison_csv}")

    # Save F-test results
    ftest_df = pd.DataFrame(f_test_results)
    ftest_csv = os.path.join(zone_results_dir, f'tb_ftest_results_{config_label}_{model_tag}_{zone}.csv')
    ftest_df.to_csv(ftest_csv, index=False)
    print(f"Saved F-test results: {ftest_csv}")

    # Save summary
    summary_path = os.path.join(zone_results_dir, f'tb_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SEQUENTIAL TREND BREAK ANALYSIS SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Config label: {config_label}\n")
        f.write(f"  Estimation model: {model_label}\n")
        f.write(f"  Rolling window: {window_years} year(s)\n")
        if step_months:
            f.write(f"  Step size: {step_months} month(s)\n")
        else:
            f.write(f"  Step size: {step_years} year(s)\n")
        f.write(f"  Minimum observations per window: {min_obs:,}\n")
        f.write(f"  Maximum breaks tested: {max_breaks}\n\n")
        f.write("  Rolling fit diagnostics:\n")
        f.write(f"    Candidate windows: {total_candidate_windows}\n")
        f.write(f"    Successful windows: {n_windows}\n")
        f.write(f"    Failed windows: {failure_stats['failed_windows_total']}\n")
        f.write(f"      Non-converged: {failure_stats['failed_windows_nonconvergence']}\n")
        f.write(f"      Exceptions: {failure_stats['failed_windows_exception']}\n")
        f.write(f"    Used non-converged fits: {failure_stats.get('used_nonconverged_windows', 0)}\n\n")

        f.write(f"Data: {n_windows} rolling windows\n\n")

        f.write("-"*80 + "\n")
        f.write("MODEL COMPARISON (BIC)\n")
        f.write("-"*80 + "\n")
        for m in model_results:
            marker = " <-- SELECTED" if m['n_breaks'] == selected_model['n_breaks'] else ""
            f.write(f"  {m['n_breaks']} breaks: BIC = {m['bic']:.2f}{marker}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("SEQUENTIAL F-TESTS\n")
        f.write("-"*80 + "\n")
        for r in f_test_results:
            f.write(f"  {r['test']}: F = {r['f_stat']:.2f}, p = {r['p_value']:.4e}\n")
            f.write(f"    Significant at 5%: {'YES' if r['significant_5pct'] else 'NO'}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("SELECTED MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of breaks: {selected_model['n_breaks']}\n")
        if selected_model['break_dates']:
            f.write(f"Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}\n")

        f.write("\nSegment details:\n")
        for i, slope_info in enumerate(selected_model['slopes']):
            f.write(f"  Segment {i+1}: slope = {slope_info['slope']:.6f} (SE: {slope_info['se']:.6f})\n")

    print(f"Saved summary: {summary_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SEQUENTIAL TREND BREAK ANALYSIS COMPLETE")
    print("="*80)

    print(f"\nKey findings for {zone}:")
    print(f"  - Optimal breaks (BIC): {selected_model['n_breaks']}")
    if selected_model['break_dates']:
        print(f"  - Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}")
    slopes_final = [f"{s['slope']:.6f}" for s in selected_model['slopes']]
    print(f"  - Segment slopes: {slopes_final}")

    # Return results
    results = {
        'zone': zone,
        'config_label': config_label,
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'rolling_fit_stats': {
            'total_candidate_windows': int(total_candidate_windows),
            'successful_windows': int(n_windows),
            **failure_stats
        },
        'n_windows': n_windows,
        'model_results': model_results,
        'f_test_results': f_test_results,
        'selected_model': selected_model,
        'optimal_by_bic': optimal_by_bic,
        'optimal_by_seq': optimal_by_seq,
        'rolling_df': rolling_df
    }

    return results


def run_trend_break_analysis_bp_supf(df, zone, Y, exog_vars,
                                     max_breaks=5, min_segment_pct=0.10,
                                     trimming=0.15,
                                     window_years=1, step_years=1/12,
                                     min_obs=24*365 - 24*30,
                                     config_label=None,
                                     plots_dir="plots", results_dir="results",
                                     bp_inference_mode='both',
                                     bp_significance_level=0.05,
                                     bp_bootstrap_reps=999,
                                     bp_bootstrap_block_length=8,
                                     bp_random_seed=42,
                                     estimation_model='ols',
                                     dynamic_armax_order=(3, 0, 3)):
    """
    Bai-Perron style sequential supF trend-break analysis.

    This applies supF(l+1|l) testing to the rolling wind-coefficient trend series,
    with optional tabulated critical values and bootstrap inference.
    """
    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Bai-Perron (2003) Table 2b: Asymptotic critical values for sequential test
    # F_T(l+1|l), epsilon = 0.10. Index: q -> alpha -> [ell=0..9].
    BP_TABLE2B_EPS_010 = {
        2: {
            0.10: [10.37, 13.20, 13.79, 14.37, 14.66, 15.07, 15.42, 15.81, 16.09, 16.09],
            0.05: [12.25, 13.83, 14.73, 15.46, 16.13, 16.55, 16.82, 17.07, 17.34, 17.58],
            0.025: [13.86, 15.51, 16.55, 17.07, 17.58, 17.98, 18.19, 18.55, 18.92, 19.02],
            0.01: [16.19, 17.88, 18.31, 18.96, 19.63, 20.09, 20.30, 20.87, 20.97, 21.13]
        }
    }

    if config_label is None:
        step_months_label = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months_label:
            step_label = f"{step_months_label}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"
        config_label = f"{window_label}_window_{step_label}_step_trend_bp"

    step_months = int(round(step_years * 12)) if step_years < 1 else None
    rng = np.random.default_rng(bp_random_seed)

    print("\n" + "="*80)
    print("BAI-PERRON STYLE SEQUENTIAL SUPF TREND BREAK ANALYSIS")
    print("="*80)
    print(f"Inference mode: {bp_inference_mode}")
    print(f"Significance level: {bp_significance_level}")
    print(f"Bootstrap reps: {bp_bootstrap_reps}")
    print(f"Bootstrap block length: {bp_bootstrap_block_length}")
    print(f"Estimation model: {model_label}")

    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy().sort_index()
    n_obs = len(tmp)

    zone_plots_dir = os.path.join(plots_dir, zone, "trend_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)
    zone_results_dir = os.path.join(results_dir, "trend_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)
    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )
    if rolling_df.empty:
        print("No rolling windows available for trend break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'method': 'bp_supf',
            'model_tag': model_tag,
            'estimation_model': estimation_model,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }

    rolling_df['time_idx'] = np.arange(len(rolling_df))
    y_arr = rolling_df['beta_wind'].to_numpy(dtype=float)
    time_idx = rolling_df['time_idx'].to_numpy(dtype=float)
    n = len(y_arr)
    min_segment = max(int(n * min_segment_pct), 5)
    trim_n = int(n * trimming)
    q = 2  # Intercept + slope restrictions for an added break
    max_l_table = 9  # Table 2b provides ell columns 0..9

    if bp_inference_mode in ('tables', 'both') and abs(float(trimming) - 0.10) > 1e-9:
        raise ValueError(
            f"BP table mode requires trimming=0.10 (epsilon=0.10), got {trimming}. "
            "Set STRUCTURAL_BREAK_TRIMMING=0.10 or use bp_inference_mode='bootstrap'."
        )

    if bp_significance_level not in (0.10, 0.05, 0.025, 0.01):
        raise ValueError(
            f"Unsupported bp_significance_level={bp_significance_level}. "
            "Choose one of: 0.10, 0.05, 0.025, 0.01"
        )

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {n} rolling window coefficients")
    print(f"Successful windows: {n}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Minimum segment length: {min_segment}")

    def fit_segment_range(y_values, start, end):
        if end - start < 3:
            return np.inf, None, None, None
        x_seg = time_idx[start:end]
        X = np.column_stack([np.ones(end - start), x_seg])
        y_seg = y_values[start:end]
        beta, _, _, _ = np.linalg.lstsq(X, y_seg, rcond=None)
        fitted = X @ beta
        resid = y_seg - fitted
        rss = float(np.sum(resid ** 2))
        return rss, beta, fitted, resid

    def evaluate_break_list(y_values, breaks):
        starts = [0] + list(breaks)
        ends = list(breaks) + [n]
        total_rss = 0.0
        models = []
        for s, e in zip(starts, ends):
            rss, beta, fitted, resid = fit_segment_range(y_values, s, e)
            if beta is None:
                return np.inf, None
            total_rss += rss
            models.append({'start': s, 'end': e, 'beta': beta, 'rss': rss, 'fitted': fitted, 'resid': resid})
        return total_rss, models

    def find_optimal_breaks_dp(y_values, n_breaks):
        if n_breaks == 0:
            rss, models = evaluate_break_list(y_values, [])
            return [], rss, models

        best_breaks = None
        best_rss = np.inf
        best_models = None

        def search(remaining_breaks, start_idx, current_breaks):
            nonlocal best_breaks, best_rss, best_models
            if remaining_breaks == 0:
                rss, models = evaluate_break_list(y_values, current_breaks)
                if models is not None and rss < best_rss:
                    best_rss = rss
                    best_breaks = current_breaks.copy()
                    best_models = models
                return

            earliest = max(start_idx + min_segment, trim_n)
            latest = n - remaining_breaks * min_segment - min_segment
            latest = min(latest, n - trim_n)
            if earliest > latest:
                return

            for brk in range(earliest, latest + 1):
                current_breaks.append(brk)
                search(remaining_breaks - 1, brk, current_breaks)
                current_breaks.pop()

        search(n_breaks, 0, [])
        return best_breaks if best_breaks else [], best_rss, best_models

    def supf_lplus1_given_l(y_values, breaks_l, rss_l):
        starts = [0] + list(breaks_l)
        ends = list(breaks_l) + [n]
        k_u = 2 * (len(breaks_l) + 2)
        best_f = -np.inf
        best_break = None
        best_rss_u = None

        for s, e in zip(starts, ends):
            c_start = s + min_segment
            c_end = e - min_segment
            if c_start > c_end:
                continue
            for c in range(c_start, c_end + 1):
                candidate_breaks = sorted(list(breaks_l) + [c])
                rss_u, _ = evaluate_break_list(y_values, candidate_breaks)
                if np.isinf(rss_u) or (n - k_u) <= 0 or rss_u <= 0:
                    continue
                f_stat = ((rss_l - rss_u) / q) / (rss_u / (n - k_u))
                if f_stat > best_f:
                    best_f = f_stat
                    best_break = c
                    best_rss_u = rss_u

        if best_break is None:
            return np.nan, None, np.nan
        return float(best_f), int(best_break), float(best_rss_u)

    def moving_block_sample(resid_values, block_len):
        T = len(resid_values)
        if T <= 1:
            return resid_values.copy()
        block_len = max(2, min(block_len, T))
        starts = np.arange(0, T - block_len + 1)
        draws = []
        while len(draws) < T:
            st = rng.choice(starts)
            draws.extend(resid_values[st:st + block_len].tolist())
        return np.array(draws[:T], dtype=float)

    print("\n" + "-"*80)
    print("STEP 2: MODEL GRID (0..M breaks)")
    print("-"*80)
    model_results = []
    for m in range(0, max_breaks + 1):
        brks, rss, models = find_optimal_breaks_dp(y_arr, m)
        if models is None or np.isinf(rss):
            break
        k = 2 * (m + 1)
        bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)
        break_dates = [rolling_df.iloc[b]['midpoint'] for b in brks] if brks else []
        slopes = [{'slope': mdl['beta'][1], 'se': np.nan} for mdl in models]
        model_results.append({
            'n_breaks': m,
            'breaks': brks,
            'break_dates': break_dates,
            'rss': rss,
            'bic': bic,
            'models': models,
            'slopes': slopes
        })
        print(f"{m} breaks: RSS={rss:.4f}, BIC={bic:.2f}")

    print("\n" + "-"*80)
    print("STEP 3: SEQUENTIAL SUPF(l+1|l)")
    print("-"*80)
    supf_results = []
    optimal_by_supf = 0

    for l in range(0, min(max_breaks, len(model_results) - 1)):
        if bp_inference_mode in ('tables', 'both') and l > max_l_table:
            print(f"Stopping sequential test at l={l-1}: Table 2b supports ell up to {max_l_table}.")
            break

        model_l = model_results[l]
        rss_l = model_l['rss']
        breaks_l = model_l['breaks']
        supf_obs, best_split, rss_u = supf_lplus1_given_l(y_arr, breaks_l, rss_l)

        cv_table = np.nan
        p_boot = np.nan
        cv_boot = np.nan

        # Table-based critical value (Bai-Perron Table 2b, epsilon=0.10)
        if q in BP_TABLE2B_EPS_010 and bp_significance_level in BP_TABLE2B_EPS_010[q]:
            cv_table = BP_TABLE2B_EPS_010[q][bp_significance_level][l]

        if bp_inference_mode in ('bootstrap', 'both'):
            boot_stats = []
            fitted_l = np.empty(n)
            resid_l = np.empty(n)
            for mdl in model_l['models']:
                s, e = mdl['start'], mdl['end']
                fitted_l[s:e] = mdl['fitted']
                resid_l[s:e] = mdl['resid']
            resid_l = resid_l - np.mean(resid_l)

            for _ in range(bp_bootstrap_reps):
                e_star = moving_block_sample(resid_l, bp_bootstrap_block_length)
                y_star = fitted_l + e_star
                rss_l_star, _ = evaluate_break_list(y_star, breaks_l)
                supf_star, _, _ = supf_lplus1_given_l(y_star, breaks_l, rss_l_star)
                if not np.isnan(supf_star):
                    boot_stats.append(supf_star)

            if boot_stats:
                boot_stats = np.array(boot_stats, dtype=float)
                cv_boot = float(np.quantile(boot_stats, 1 - bp_significance_level))
                p_boot = float(np.mean(boot_stats >= supf_obs))

        # Decision rule
        reject_5 = False
        decision_source = 'none'
        if bp_inference_mode == 'tables' and not np.isnan(cv_table):
            reject_5 = supf_obs > cv_table
            decision_source = 'table'
        elif bp_inference_mode == 'tables' and np.isnan(cv_table):
            raise ValueError(
                f"No BP Table 2b critical value for q={q}, alpha={bp_significance_level}, l={l}."
            )
        elif bp_inference_mode == 'bootstrap':
            reject_5 = (p_boot < bp_significance_level) if not np.isnan(p_boot) else False
            decision_source = 'bootstrap'
        else:  # both
            # In 'both', table is primary for decision; bootstrap is reported as robustness.
            if not np.isnan(cv_table):
                reject_5 = supf_obs > cv_table
                decision_source = 'table_primary'
            elif not np.isnan(p_boot):
                reject_5 = p_boot < bp_significance_level
                decision_source = 'bootstrap_fallback'

        supf_results.append({
            'l': l,
            'test': f"supF({l+1}|{l})",
            'supf_stat': supf_obs,
            'best_split_idx': best_split,
            'best_split_date': rolling_df.iloc[best_split]['midpoint'] if best_split is not None else pd.NaT,
            'rss_l': rss_l,
            'rss_l_plus_1_best': rss_u,
            'cv_table': cv_table,
            'cv_boot': cv_boot,
            'p_boot': p_boot,
            'alpha': bp_significance_level,
            'reject_alpha': bool(reject_5),
            'reject_5pct': bool(reject_5),
            'decision_source': decision_source
        })

        print(f"supF({l+1}|{l})={supf_obs:.3f}, reject@{int((1-bp_significance_level)*100)}%={reject_5} ({decision_source})")
        if reject_5:
            optimal_by_supf = l + 1
        else:
            break

    optimal_by_bic = int(np.argmin([m['bic'] for m in model_results])) if model_results else 0
    selected_model = model_results[optimal_by_supf] if model_results else None

    # Plot selected model
    if selected_model is not None:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        midpoints = pd.to_datetime(rolling_df['midpoint'])
        beta_values = rolling_df['beta_wind'].values
        ax1.plot(midpoints, beta_values, 'o', color='steelblue', alpha=0.5, markersize=4, label='Rolling coefficients')

        colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
        segment_starts = [0] + selected_model['breaks']
        segment_ends = selected_model['breaks'] + [n]
        for i, (s, e, mdl) in enumerate(zip(segment_starts, segment_ends, selected_model['models'])):
            x_seg = time_idx[s:e]
            y_pred = mdl['beta'][0] + mdl['beta'][1] * x_seg
            ax1.plot(midpoints[s:e], y_pred, '-', color=colors[i % len(colors)], linewidth=2.5,
                     label=f"Segment {i+1} (slope={mdl['beta'][1]:.4f})")
        for i, break_date in enumerate(selected_model['break_dates']):
            ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                        label='Break' if i == 0 else None)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_title(
            f'BP-supF Trend Breaks ({selected_model["n_breaks"]}) - {zone}\n(Model: {model_label})',
            fontsize=12, fontweight='bold'
        )
        ax1.set_xlabel('Date')
        ax1.set_ylabel(r'$\beta_{wind}$')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(zone_plots_dir, f'tb_bp_{config_label}_{model_tag}_{zone}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    # Save outputs
    rolling_csv = os.path.join(zone_results_dir, f'tb_rolling_coef_bp_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    model_comp_rows = []
    for m in model_results:
        model_comp_rows.append({
            'n_breaks': m['n_breaks'],
            'bic': m['bic'],
            'rss': m['rss'],
            'break_dates': ';'.join([d.strftime('%Y-%m-%d') for d in m['break_dates']]) if m['break_dates'] else ''
        })
    model_comp_df = pd.DataFrame(model_comp_rows)
    model_comp_csv = os.path.join(zone_results_dir, f'tb_bp_model_comparison_{config_label}_{model_tag}_{zone}.csv')
    model_comp_df.to_csv(model_comp_csv, index=False)
    print(f"Saved model comparison: {model_comp_csv}")

    supf_df = pd.DataFrame(supf_results)
    supf_csv = os.path.join(zone_results_dir, f'tb_bp_supf_seq_{config_label}_{model_tag}_{zone}.csv')
    supf_df.to_csv(supf_csv, index=False)
    print(f"Saved supF results: {supf_csv}")

    summary_path = os.path.join(zone_results_dir, f'tb_bp_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BP STYLE SEQUENTIAL SUPF TREND BREAK SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Config label: {config_label}\n")
        f.write(f"Estimation model: {model_label}\n")
        f.write(f"Inference mode: {bp_inference_mode}\n")
        f.write(f"Epsilon (trimming): {trimming}\n")
        f.write(f"q restrictions per added break: {q}\n")
        f.write(f"Significance level: {bp_significance_level}\n")
        f.write(f"Bootstrap reps: {bp_bootstrap_reps}\n")
        f.write(f"Bootstrap block length: {bp_bootstrap_block_length}\n")
        f.write(f"Random seed: {bp_random_seed}\n\n")
        f.write(f"Data windows: {n}\n")
        f.write("Rolling fit diagnostics:\n")
        f.write(f"  Candidate windows: {total_candidate_windows}\n")
        f.write(f"  Successful windows: {n}\n")
        f.write(f"  Failed windows: {failure_stats['failed_windows_total']}\n")
        f.write(f"    Non-converged: {failure_stats['failed_windows_nonconvergence']}\n")
        f.write(f"    Exceptions: {failure_stats['failed_windows_exception']}\n")
        f.write(f"  Used non-converged fits: {failure_stats.get('used_nonconverged_windows', 0)}\n")
        f.write(f"Optimal breaks by supF: {optimal_by_supf}\n")
        f.write(f"Optimal breaks by BIC: {optimal_by_bic}\n")
        if selected_model and selected_model['break_dates']:
            f.write(f"Selected break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}\n")
    print(f"Saved summary: {summary_path}")

    return {
        'zone': zone,
        'config_label': config_label,
        'method': 'bp_supf',
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'rolling_fit_stats': {
            'total_candidate_windows': int(total_candidate_windows),
            'successful_windows': int(n),
            **failure_stats
        },
        'rolling_df': rolling_df,
        'model_results': model_results,
        'supf_results': supf_results,
        'selected_model': selected_model,
        'optimal_by_supf': optimal_by_supf,
        'optimal_by_bic': optimal_by_bic
    }


def run_trend_break_analysis(df, zone, Y, exog_vars,
                             max_breaks=5, min_segment_pct=0.10,
                             trimming=0.15,
                             window_years=1, step_years=1/12,
                             min_obs=24*365 - 24*30,
                             config_label=None,
                             plots_dir="plots", results_dir="results", show_progress=True,
                             trend_break_test_method='legacy',
                             bp_inference_mode='both',
                             bp_significance_level=0.05,
                             bp_bootstrap_reps=999,
                             bp_bootstrap_block_length=8,
                             bp_random_seed=42,
                             bp_use_hac_se=True,
                             estimation_model='ols',
                             dynamic_armax_order=(3, 0, 3)):
    """
    Dispatcher for trend-break analysis.
    - legacy: existing BIC + sequential F implementation
    - bp_supf: Bai-Perron style sequential supF implementation
    """
    if trend_break_test_method == 'legacy':
        return run_trend_break_analysis_legacy(
            df, zone, Y, exog_vars,
            max_breaks=max_breaks,
            min_segment_pct=min_segment_pct,
            trimming=trimming,
            window_years=window_years,
            step_years=step_years,
            min_obs=min_obs,
            config_label=config_label,
            plots_dir=plots_dir,
            results_dir=results_dir,
            show_progress=show_progress,
            estimation_model=estimation_model,
            dynamic_armax_order=dynamic_armax_order
        )

    if trend_break_test_method == 'bp_supf':
        return run_trend_break_analysis_bp_supf(
            df, zone, Y, exog_vars,
            max_breaks=max_breaks,
            min_segment_pct=min_segment_pct,
            trimming=trimming,
            window_years=window_years,
            step_years=step_years,
            min_obs=min_obs,
            config_label=config_label,
            plots_dir=plots_dir,
            results_dir=results_dir,
            bp_inference_mode=bp_inference_mode,
            bp_significance_level=bp_significance_level,
            bp_bootstrap_reps=bp_bootstrap_reps,
            bp_bootstrap_block_length=bp_bootstrap_block_length,
            bp_random_seed=bp_random_seed,
            estimation_model=estimation_model,
            dynamic_armax_order=dynamic_armax_order
        )

    raise ValueError(
        f"Unknown trend_break_test_method='{trend_break_test_method}'. "
        "Choose 'legacy' or 'bp_supf'."
    )


def run_quantile_regression_analysis(df, zone,
                                     plots_dir="plots", results_dir="results"):
    """
    Estimate wind coefficient across quantiles of the price distribution.

    Uses logged variables (NOT deseasonalized) with calendar dummies included directly
    in the regression to control for seasonality (FULL basis: Year+Month+DOW+Hour+Holiday).

    Note: Always uses logged variables.

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns: None (saves outputs to files)
    """
    # Hardcoded quantiles and seasonality settings (not user-configurable)
    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    print("\n" + "="*80)
    print("QUANTILE REGRESSION ANALYSIS")
    print("="*80)

    # --- Step 1: Determine dependent variable (logged, NOT deseasonalized) ---
    y_col = 'Price_Log'

    print(f"\nDependent variable: {y_col}")
    print("  (Using logged price, NOT deseasonalized - seasonality handled via dummies)")

    # --- Step 2: Determine economic regressors (logged, NOT deseasonalized) ---
    econ_vars = [
        'Wind_Forecast_Log',
        'Consumption_Log',
        'Hydro_Reserves_Log',
        'Net_Exchange',  # NOT logged
        'Oil_Price_Log',
        'Gas_Price_Log'
    ]

    print(f"\nEconomic regressors: {econ_vars}")

    # --- Step 3: Build calendar/seasonal dummies (FULL basis) ---
    print("\nBuilding seasonality controls (FULL basis: Year+Month+DOW+Hour+Holiday)...")

    # Create a working copy to avoid modifying original
    tmp = df.copy()

    # Extract time components from datetime index
    tmp['Year'] = tmp.index.year
    tmp['Month'] = tmp.index.month
    tmp['DayOfWeek'] = tmp.index.dayofweek  # 0=Monday, 6=Sunday
    tmp['Hour'] = tmp.index.hour

    # Create holiday indicator for Swedish holidays
    try:
        import holidays
        swedish_holidays = holidays.Sweden(years=range(tmp.index.year.min(), tmp.index.year.max() + 1))
        tmp['Holiday'] = tmp.index.to_series().apply(lambda x: 1 if x.date() in swedish_holidays else 0).values
        print("  Holiday dummies created using Swedish holiday calendar")
    except ImportError:
        tmp['Holiday'] = 0
        print("  WARNING: 'holidays' package not installed; Holiday set to 0 (no crash)")

    # Create dummy variables with drop_first=True to avoid multicollinearity
    year_dummies = pd.get_dummies(tmp['Year'], prefix='Year', drop_first=True)
    month_dummies = pd.get_dummies(tmp['Month'], prefix='Month', drop_first=True)
    dow_dummies = pd.get_dummies(tmp['DayOfWeek'], prefix='DOW', drop_first=True)
    hour_dummies = pd.get_dummies(tmp['Hour'], prefix='Hour', drop_first=True)

    print(f"  Year dummies: {len(year_dummies.columns)} columns")
    print(f"  Month dummies: {len(month_dummies.columns)} columns")
    print(f"  DOW dummies: {len(dow_dummies.columns)} columns")
    print(f"  Hour dummies: {len(hour_dummies.columns)} columns")
    print(f"  Holiday: 1 column (binary indicator)")

    # --- Step 4: Assemble data matrix ---
    # Combine all regressors
    cols_needed = [y_col] + econ_vars
    data_subset = tmp[cols_needed].copy()

    # Add seasonal dummies
    data_subset = pd.concat([data_subset, year_dummies, month_dummies, dow_dummies, hour_dummies], axis=1)
    data_subset['Holiday'] = tmp['Holiday'].values

    # Drop rows with NA and sort by index
    data_subset = data_subset.dropna()
    data_subset = data_subset.sort_index()

    print(f"\nData range: {data_subset.index.min()} to {data_subset.index.max()}")
    print(f"Observations after cleaning: {len(data_subset):,}")

    # Build y and X
    y = data_subset[y_col].astype(float)

    # X includes: constant + economic vars + seasonal dummies + holiday
    seasonal_cols = list(year_dummies.columns) + list(month_dummies.columns) + \
                    list(dow_dummies.columns) + list(hour_dummies.columns) + ['Holiday']
    X_cols = econ_vars + seasonal_cols

    # Ensure all columns are numeric (convert to float64)
    X_data = data_subset[X_cols].astype(float)
    X = sm.add_constant(X_data)

    print(f"\nTotal regressors (incl. const): {X.shape[1]}")
    print(f"  Economic controls: {len(econ_vars)}")
    print(f"  Seasonal controls: {len(seasonal_cols)}")

    # --- Step 5: Run quantile regressions ---
    print(f"\n--- Estimating Quantile Regressions ---")
    print(f"Quantiles: {QUANTILES}")
    # TODO: Block bootstrap can be added later for time-series-robust inference

    results = []
    print(f"\nEstimating quantile regressions for {len(QUANTILES)} quantiles...\n")

    for idx, q in enumerate(QUANTILES, 1):
        print(f"[{idx}/{len(QUANTILES)}] Quantile q={q:.2f}... ", end="")

        model = sm.QuantReg(y, X)
        res = model.fit(q=q)

        # Extract coefficients for key variables
        result_row = {
            'quantile': q,
            'beta_wind': res.params[wind_col],
            'se_wind': res.bse[wind_col] if wind_col in res.bse.index else np.nan,
            'p_wind': res.pvalues[wind_col] if wind_col in res.pvalues.index else np.nan,
            'beta_demand': res.params[demand_col],
            'se_demand': res.bse[demand_col] if demand_col in res.bse.index else np.nan,
            'p_demand': res.pvalues[demand_col] if demand_col in res.pvalues.index else np.nan,
            'beta_hydro': res.params[hydro_col],
            'se_hydro': res.bse[hydro_col] if hydro_col in res.bse.index else np.nan,
            'p_hydro': res.pvalues[hydro_col] if hydro_col in res.pvalues.index else np.nan,
            'n_obs': int(res.nobs)
        }

        # Add oil/gas if available
        if oil_col and oil_col in res.params.index:
            result_row['beta_oil'] = res.params[oil_col]
            result_row['se_oil'] = res.bse[oil_col] if oil_col in res.bse.index else np.nan
            result_row['p_oil'] = res.pvalues[oil_col] if oil_col in res.pvalues.index else np.nan

        if gas_col and gas_col in res.params.index:
            result_row['beta_gas'] = res.params[gas_col]
            result_row['se_gas'] = res.bse[gas_col] if gas_col in res.bse.index else np.nan
            result_row['p_gas'] = res.pvalues[gas_col] if gas_col in res.pvalues.index else np.nan

        results.append(result_row)
        print(f"β_wind={res.params[wind_col]:.4f}, p={res.pvalues[wind_col] if wind_col in res.pvalues.index else np.nan:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # --- Step 6: Print summary ---
    print("\n" + "="*80)
    print("QUANTILE REGRESSION RESULTS SUMMARY")
    print("="*80)
    print(f"\nDependent variable: {y_col}")
    print(f"Seasonality basis: FULL (Year+Month+DOW+Hour+Holiday)")
    print(f"Observations: {results_df['n_obs'].iloc[0]:,}")

    print(f"\nWind coefficient by quantile:")
    print(f"{'Quantile':<10} {'Beta':<12} {'SE':<12} {'p-value':<10}")
    print("-" * 44)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_wind'] < 0.01 else "**" if row['p_wind'] < 0.05 else "*" if row['p_wind'] < 0.1 else ""
        print(f"{row['quantile']:<10.2f} {row['beta_wind']:<12.6f} {row['se_wind']:<12.6f} {row['p_wind']:<10.4f} {sig}")

    print(f"\nDemand coefficient by quantile:")
    print(f"{'Quantile':<10} {'Beta':<12} {'SE':<12} {'p-value':<10}")
    print("-" * 44)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_demand'] < 0.01 else "**" if row['p_demand'] < 0.05 else "*" if row['p_demand'] < 0.1 else ""
        print(f"{row['quantile']:<10.2f} {row['beta_demand']:<12.6f} {row['se_demand']:<12.6f} {row['p_demand']:<10.4f} {sig}")

    # --- Step 7: Save CSV output ---
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'quantreg_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # --- Step 8: Create plots ---
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Wind coefficient across quantiles
    fig, ax = plt.subplots(figsize=(10, 6))

    quantiles = results_df['quantile'].values
    beta_wind = results_df['beta_wind'].values
    se_wind = results_df['se_wind'].values

    # 95% CI
    upper_95 = beta_wind + 1.96 * se_wind
    lower_95 = beta_wind - 1.96 * se_wind

    ax.plot(quantiles, beta_wind, 'o-', linewidth=2, markersize=8, label=r'$\beta_{wind}$')
    ax.fill_between(quantiles, lower_95, upper_95, alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel(r'$\beta_{wind}$ (Wind Coefficient)', fontsize=12)
    ax.set_title(f'Quantile Regression: Wind Coefficient - {zone}\n(FULL seasonality: Year+Month+DOW+Hour+Holiday)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(quantiles)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'quantreg_beta_wind_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: Demand coefficient across quantiles
    fig, ax = plt.subplots(figsize=(10, 6))

    beta_demand = results_df['beta_demand'].values
    se_demand = results_df['se_demand'].values

    upper_95 = beta_demand + 1.96 * se_demand
    lower_95 = beta_demand - 1.96 * se_demand

    ax.plot(quantiles, beta_demand, 'o-', linewidth=2, markersize=8, label=r'$\beta_{demand}$')
    ax.fill_between(quantiles, lower_95, upper_95, alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel(r'$\beta_{demand}$ (Demand Coefficient)', fontsize=12)
    ax.set_title(f'Quantile Regression: Demand Coefficient - {zone}\n(FULL seasonality: Year+Month+DOW+Hour+Holiday)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(quantiles)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'quantreg_beta_demand_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("QUANTILE REGRESSION ANALYSIS COMPLETE")
    print("="*80)


def _evaluate_armax_candidate(Y, exog_vars, p, q,
                            ljungbox_lags=(5, 10, 15, 20),
                            maxiter=300,
                            solver='statespace',
                            use_warm_start=True,
                            exog_anchor_map=None,
                            scale_exog=False,
                            max_p_for_schema=10,
                            max_q_for_schema=10):
    """Evaluate a single ARMAX(p,0,q) candidate without fallback."""
    order = (p, 0, q)
    timestamp = pd.Timestamp.now().isoformat()

    def _sig_stars(p_value):
        if p_value is None or pd.isna(p_value):
            return ""
        if p_value < 0.01:
            return "***"
        if p_value < 0.05:
            return "**"
        if p_value < 0.10:
            return "*"
        return ""

    result = {
        'p': p,
        'q': q,
        'status': 'exception',
        'converged': False,
        'iterations': np.nan,
        'warnflag': np.nan,
        'fopt': np.nan,
        'gradient_max_abs': np.nan,
        'aic': np.nan,
        'bic': np.nan,
        'hqic': np.nan,
        'beta_wind': np.nan,
        'wind_se': np.nan,
        'wind_stars': '',
        'passes_ljungbox': False,
        'diagnosis_label': 'unknown',
        'diagnosis_why': '',
        'error_message': '',
        'timestamp': timestamp
    }

    for lag in ljungbox_lags:
        result[f'ljungbox_lag_{lag}_stat'] = np.nan
        result[f'ljungbox_lag_{lag}_pval'] = np.nan

    max_p_for_schema = int(max(0, max_p_for_schema))
    max_q_for_schema = int(max(0, max_q_for_schema))
    for i in range(1, max_p_for_schema + 1):
        result[f'ar{i}_coef'] = np.nan
        result[f'ar{i}_se'] = np.nan
        result[f'ar{i}_stars'] = ''
    for i in range(1, max_q_for_schema + 1):
        result[f'ma{i}_coef'] = np.nan
        result[f'ma{i}_se'] = np.nan
        result[f'ma{i}_stars'] = ''

    fit = _fit_armax_with_controls(
        y=Y,
        X_exog=exog_vars,
        order=order,
        context_label=f"lag_search ARMAX({p},{q})",
        accept_nonconverged=True,
        maxiter=maxiter,
        solver=solver,
        use_warm_start=use_warm_start,
        exog_anchor_map=exog_anchor_map,
        scale_exog=scale_exog
    )

    if not fit['ok']:
        result['status'] = 'exception'
        result['error_message'] = fit.get('error', 'Unknown fit failure')
        label, why = _diagnose_nonconvergence_simple(
            {'ok': False, 'fail_reason': 'exception', 'error': result['error_message'], 'diagnostics': {}},
            configured_maxiter=maxiter
        )
        result['diagnosis_label'] = label
        result['diagnosis_why'] = why
        return result

    model = fit['model']
    diag = fit.get('diagnostics', {}) or {}
    converged = bool(fit.get('converged', False))

    result['converged'] = converged
    result['status'] = 'converged' if converged else 'nonconverged'
    result['iterations'] = diag.get('iterations', np.nan)
    result['warnflag'] = diag.get('warnflag', np.nan)
    result['fopt'] = diag.get('fopt', np.nan)
    result['gradient_max_abs'] = diag.get('gradient_max_abs', np.nan)
    result['aic'] = float(getattr(model, 'aic', np.nan))
    result['bic'] = float(getattr(model, 'bic', np.nan))
    result['hqic'] = float(getattr(model, 'hqic', np.nan))
    if converged:
        result['diagnosis_label'] = 'converged'
        result['diagnosis_why'] = 'Converged successfully.'
        try:
            beta_wind, se_wind, p_wind = _extract_armax_wind_coef(model, 'Wind_Forecast_Log')
            if bool(diag.get('exog_scaled', False)):
                stds = pd.Series(diag.get('scale_stds', {}))
                wind_std = float(stds.get('Wind_Forecast_Log', 1.0))
                if wind_std != 0:
                    beta_wind = float(beta_wind) / wind_std
                    se_wind = float(se_wind) / wind_std
            result['beta_wind'] = float(beta_wind)
            result['wind_se'] = float(se_wind)
            result['wind_stars'] = _sig_stars(p_wind)
        except Exception as e:
            msg = f"Wind coefficient extraction failed: {e}"
            if result['error_message']:
                result['error_message'] += f" | {msg}"
            else:
                result['error_message'] = msg

        try:
            param_names = list(model.param_names)
            params = np.asarray(model.params)
            bse = np.asarray(model.bse)
            pvals = np.asarray(model.pvalues)
            for idx, name in enumerate(param_names):
                if name.startswith('ar.L'):
                    lag_str = name.split('ar.L')[-1]
                    lag = int(lag_str)
                    if 1 <= lag <= max_p_for_schema:
                        result[f'ar{lag}_coef'] = float(params[idx])
                        result[f'ar{lag}_se'] = float(bse[idx])
                        result[f'ar{lag}_stars'] = _sig_stars(float(pvals[idx]))
                elif name.startswith('ma.L'):
                    lag_str = name.split('ma.L')[-1]
                    lag = int(lag_str)
                    if 1 <= lag <= max_q_for_schema:
                        result[f'ma{lag}_coef'] = float(params[idx])
                        result[f'ma{lag}_se'] = float(bse[idx])
                        result[f'ma{lag}_stars'] = _sig_stars(float(pvals[idx]))
        except Exception as e:
            msg = f"AR/MA coefficient extraction failed: {e}"
            if result['error_message']:
                result['error_message'] += f" | {msg}"
            else:
                result['error_message'] = msg
    else:
        label, why = _diagnose_nonconvergence_simple(
            {'ok': True, 'converged': False, 'diagnostics': diag, 'fail_reason': None, 'error': None},
            configured_maxiter=maxiter
        )
        result['diagnosis_label'] = label
        result['diagnosis_why'] = why

    try:
        lb_results = run_ljungbox_test(
            model.resid,
            lags=list(ljungbox_lags),
            return_results=True,
            print_output=False
        )
        for _, row in lb_results.iterrows():
            lag = int(row['lag'])
            result[f'ljungbox_lag_{lag}_stat'] = row['test_stat']
            result[f'ljungbox_lag_{lag}_pval'] = row['p_value']
        result['passes_ljungbox'] = bool((lb_results['p_value'] > 0.05).all())
    except Exception as e:
        result['passes_ljungbox'] = False
        if not result['error_message']:
            result['error_message'] = f"Ljung-Box failed: {e}"

    return result


def _select_best_armax_candidate(results_df,
                                require_convergence=True,
                                require_ljungbox_pass=True,
                                criterion='bic'):
    """Select best candidate from evaluated grid based on eligibility and criterion."""
    if results_df.empty:
        return None, pd.DataFrame(), {
            'tested': 0,
            'converged': 0,
            'eligible': 0,
            'exceptions': 0,
            'nonconverged': 0
        }

    criterion_col = criterion.lower()
    if criterion_col not in {'aic', 'bic'}:
        raise ValueError(f"Invalid criterion '{criterion}'. Use 'aic' or 'bic'.")

    df = results_df.copy()
    mask = pd.Series(True, index=df.index)
    if require_convergence:
        mask &= (df['converged'] == True)
    if require_ljungbox_pass:
        mask &= (df['passes_ljungbox'] == True)
    mask &= df[criterion_col].notna()

    df['eligible_for_selection'] = mask
    eligible = df[df['eligible_for_selection']].copy()

    if eligible.empty:
        best_order = None
    else:
        best_idx = eligible[criterion_col].idxmin()
        best_row = eligible.loc[best_idx]
        best_order = (int(best_row['p']), int(best_row['q']))

    stats = {
        'tested': int(len(df)),
        'converged': int((df['converged'] == True).sum()),
        'eligible': int(len(eligible)),
        'exceptions': int((df['status'] == 'exception').sum()),
        'nonconverged': int((df['status'] == 'nonconverged').sum())
    }

    return best_order, df, stats


def _build_armax_search_grid(p_min, p_max, q_min, q_max, exclude_00=True):
    """Build grid list [(p,q), ...] for ARMAX search."""
    grid = []
    for p in range(int(p_min), int(p_max) + 1):
        for q in range(int(q_min), int(q_max) + 1):
            if exclude_00 and p == 0 and q == 0:
                continue
            grid.append((p, q))
    return grid


def _save_armax_search_reports(results_df, zone='SE1', top_n=20,
                              criterion='bic', results_dir='results'):
    """Persist full and top-N search reports to CSV and Excel."""
    os.makedirs(results_dir, exist_ok=True)

    all_path = os.path.join(results_dir, f'armax_search_all_{zone}.csv')
    results_df.to_csv(all_path, index=False)
    all_xlsx_path = os.path.join(results_dir, f'armax_search_all_{zone}.xlsx')
    try:
        results_df.to_excel(all_xlsx_path, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write Excel full report: {e}")

    criterion_col = criterion.lower()
    eligible = results_df[results_df.get('eligible_for_selection', False) == True].copy()
    if not eligible.empty and criterion_col in eligible.columns:
        top_df = eligible.sort_values(criterion_col).head(int(top_n))
    else:
        top_df = pd.DataFrame(columns=results_df.columns)

    top_path = os.path.join(results_dir, f'armax_search_top_{zone}.csv')
    top_df.to_csv(top_path, index=False)
    top_xlsx_path = os.path.join(results_dir, f'armax_search_top_{zone}.xlsx')
    try:
        top_df.to_excel(top_xlsx_path, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write Excel top report: {e}")

    return all_path, top_path, all_xlsx_path, top_xlsx_path


def select_armax_lags_aic(Y, exog_vars, zone='SE1',
                          p_min=0, p_max=10, q_min=0, q_max=10,
                          exclude_00=True,
                          require_convergence=True,
                          require_ljungbox_pass=True,
                          selection_criterion='bic',
                          ljungbox_lags=(5, 10, 15, 20),
                          maxiter=300,
                          solver='statespace',
                          use_warm_start=True,
                          scale_exog=False,
                          save_top_n=20,
                          results_dir='results'):
    """Strict ARMAX lag search without fallback; returns best order and full results."""
    requested_p_min = int(p_min)
    requested_q_min = int(q_min)
    p_min_eff = max(1, requested_p_min)
    q_min_eff = max(1, requested_q_min)
    p_max_eff = int(p_max)
    q_max_eff = int(q_max)
    if p_min_eff != requested_p_min or q_min_eff != requested_q_min:
        print(f"NOTE: Enforcing ARMAX lag-search floor p>=1, q>=1 (requested p_min={p_min}, q_min={q_min}).")
    if p_min_eff > p_max_eff or q_min_eff > q_max_eff:
        raise ValueError(
            f"Invalid ARMAX grid after enforcing p,q>=1: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )

    grid = _build_armax_search_grid(p_min_eff, p_max_eff, q_min_eff, q_max_eff, exclude_00=exclude_00)
    max_p_schema = int(max(0, p_max_eff))
    max_q_schema = int(max(0, q_max_eff))
    exog_anchor_map = _load_armax_exog_anchor_csv(zone=zone, source_order=(1, 0, 1), results_dir=results_dir)

    print("\n--- ARMAX LAG SELECTION (STRICT GRID SEARCH) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}, exclude_00={exclude_00}")
    print(f"Eligibility: require_convergence={require_convergence}, require_ljungbox_pass={require_ljungbox_pass}")
    print(f"Selection criterion: {selection_criterion.upper()}")
    print(f"Models to test: {len(grid)}")
    if exog_anchor_map is None:
        print(f"Anchor starts: no anchor file found for {zone}; using default warm-start logic.")
    else:
        print(f"Anchor starts: loaded {len(exog_anchor_map)} terms from armax_exog_anchor_{zone}_order_1_0_1.csv")

    rows = []
    total = len(grid)
    for i, (p, q) in enumerate(grid, start=1):
        print(f"[{i}/{total}] Testing ARMAX({p},{q})... ", end="")
        row = _evaluate_armax_candidate(
            Y=Y,
            exog_vars=exog_vars,
            p=p,
            q=q,
            ljungbox_lags=ljungbox_lags,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start,
            exog_anchor_map=exog_anchor_map,
            scale_exog=scale_exog,
            max_p_for_schema=max_p_schema,
            max_q_for_schema=max_q_schema
        )
        rows.append(row)
        diag_short = row.get('diagnosis_label', '')
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
            f"diag={diag_short}"
        )

    results_df = pd.DataFrame(rows)
    best_order, results_df, stats = _select_best_armax_candidate(
        results_df,
        require_convergence=require_convergence,
        require_ljungbox_pass=require_ljungbox_pass,
        criterion=selection_criterion
    )
    all_path, top_path, all_xlsx_path, top_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        top_n=save_top_n,
        criterion=selection_criterion,
        results_dir=results_dir
    )

    print("\n" + "="*80)
    print("ARMAX SEARCH SUMMARY")
    print("="*80)
    print(f"Tested: {stats['tested']} | Converged: {stats['converged']} | Eligible: {stats['eligible']} | "
          f"Nonconverged: {stats['nonconverged']} | Exceptions: {stats['exceptions']}")
    print(f"Saved full report: {all_path}")
    print(f"Saved full Excel:  {all_xlsx_path}")
    print(f"Saved top report:  {top_path}")
    print(f"Saved top Excel:   {top_xlsx_path}")

    if best_order is None:
        print("No admissible ARMAX order found under current eligibility constraints.")
    else:
        print(f"Selected order: ARMAX{best_order} by {selection_criterion.upper()}")

    return best_order, results_df


def select_armax_lags_aic_checkpointed(Y, exog_vars, zone='SE1',
                                       p_min=0, p_max=10, q_min=0, q_max=10,
                                       exclude_00=True,
                                       require_convergence=True,
                                       require_ljungbox_pass=True,
                                       selection_criterion='bic',
                                       checkpoint_file=None,
                                       ljungbox_lags=(5, 10, 15, 20),
                                       save_interval=1,
                                       maxiter=300,
                                       solver='statespace',
                                       use_warm_start=True,
                                       scale_exog=False,
                                       save_top_n=20,
                                       results_dir='results'):
    """Checkpointed strict ARMAX lag search (no fallback)."""
    os.makedirs(results_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(results_dir, f'armax_lag_selection_checkpoint_{zone}.csv')

    requested_p_min = int(p_min)
    requested_q_min = int(q_min)
    p_min_eff = max(1, requested_p_min)
    q_min_eff = max(1, requested_q_min)
    p_max_eff = int(p_max)
    q_max_eff = int(q_max)
    if p_min_eff != requested_p_min or q_min_eff != requested_q_min:
        print(f"NOTE: Enforcing ARMAX lag-search floor p>=1, q>=1 (requested p_min={p_min}, q_min={q_min}).")
    if p_min_eff > p_max_eff or q_min_eff > q_max_eff:
        raise ValueError(
            f"Invalid ARMAX grid after enforcing p,q>=1: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )

    grid = _build_armax_search_grid(p_min_eff, p_max_eff, q_min_eff, q_max_eff, exclude_00=exclude_00)
    max_p_schema = int(max(0, p_max_eff))
    max_q_schema = int(max(0, q_max_eff))
    exog_anchor_map = _load_armax_exog_anchor_csv(zone=zone, source_order=(1, 0, 1), results_dir=results_dir)
    print("\n--- ARMAX LAG SELECTION (CHECKPOINTED, STRICT) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}, exclude_00={exclude_00}")
    print(f"Selection criterion: {selection_criterion.upper()}")
    print(f"Checkpoint file: {checkpoint_file}")
    if exog_anchor_map is None:
        print(f"Anchor starts: no anchor file found for {zone}; using default warm-start logic.")
    else:
        print(f"Anchor starts: loaded {len(exog_anchor_map)} terms from armax_exog_anchor_{zone}_order_1_0_1.csv")

    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        completed_specs = set()
        if {'p', 'q', 'status'}.issubset(checkpoint_df.columns):
            for _, row in checkpoint_df.iterrows():
                status = str(row.get('status', ''))
                if status in {'converged', 'nonconverged', 'exception', 'completed', 'failed'}:
                    completed_specs.add((int(row['p']), int(row['q'])))
        print(f"Resuming: found {len(completed_specs)} tested specifications")
    else:
        checkpoint_df = pd.DataFrame()
        completed_specs = set()
        print("Starting fresh (no existing checkpoint)")

    total = len(grid)
    tested_before = len(completed_specs)
    print(f"Models to test: {total - tested_before} (Total: {total}, Already tested: {tested_before})")

    tested_counter = tested_before
    for p, q in grid:
        if (p, q) in completed_specs:
            continue

        tested_counter += 1
        print(f"[{tested_counter}/{total}] Testing ARMAX({p},{q})... ", end="")
        row = _evaluate_armax_candidate(
            Y=Y,
            exog_vars=exog_vars,
            p=p,
            q=q,
            ljungbox_lags=ljungbox_lags,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start,
            exog_anchor_map=exog_anchor_map,
            scale_exog=scale_exog,
            max_p_for_schema=max_p_schema,
            max_q_for_schema=max_q_schema
        )
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
            f"diag={row.get('diagnosis_label', '')}"
        )

        row_df = pd.DataFrame([row])
        if checkpoint_df.empty:
            checkpoint_df = row_df
        else:
            checkpoint_df = pd.concat([checkpoint_df, row_df], ignore_index=True)

        if len(checkpoint_df) % max(int(save_interval), 1) == 0:
            checkpoint_df.to_csv(checkpoint_file, index=False)
            checkpoint_xlsx = os.path.splitext(checkpoint_file)[0] + '.xlsx'
            try:
                checkpoint_df.to_excel(checkpoint_xlsx, index=False)
            except Exception as e:
                print(f"WARNING: Failed to write checkpoint Excel report: {e}")

    checkpoint_df.to_csv(checkpoint_file, index=False)
    checkpoint_xlsx = os.path.splitext(checkpoint_file)[0] + '.xlsx'
    try:
        checkpoint_df.to_excel(checkpoint_xlsx, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write checkpoint Excel report: {e}")

    best_order, results_df, stats = _select_best_armax_candidate(
        checkpoint_df.copy(),
        require_convergence=require_convergence,
        require_ljungbox_pass=require_ljungbox_pass,
        criterion=selection_criterion
    )
    all_path, top_path, all_xlsx_path, top_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        top_n=save_top_n,
        criterion=selection_criterion,
        results_dir=results_dir
    )

    print("\n" + "="*80)
    print("ARMAX CHECKPOINT SEARCH SUMMARY")
    print("="*80)
    print(f"Tested: {stats['tested']} | Converged: {stats['converged']} | Eligible: {stats['eligible']} | "
          f"Nonconverged: {stats['nonconverged']} | Exceptions: {stats['exceptions']}")
    print(f"Checkpoint:       {checkpoint_file}")
    print(f"Checkpoint Excel: {checkpoint_xlsx}")
    print(f"Saved full report: {all_path}")
    print(f"Saved full Excel:  {all_xlsx_path}")
    print(f"Saved top report:  {top_path}")
    print(f"Saved top Excel:   {top_xlsx_path}")

    if best_order is None:
        print("No admissible ARMAX order found under current eligibility constraints.")
    else:
        print(f"Selected order: ARMAX{best_order} by {selection_criterion.upper()}")

    return best_order, results_df

def fit_garchx_model(armax_residuals, df, wind_var='Wind_Forecast_Log',
                     p=1, q=1, show_diagnostics=True):
    """
    Fits GARCH(p,q)-X model on ARMAX residuals with Wind as exogenous variable.

    Parameters:
    - armax_residuals: Residuals from ARMAX mean equation (pd.Series)
    - df: DataFrame with all variables (must include wind_var)
    - wind_var: Name of wind variable for variance equation
    - p: GARCH lag order (default 1)
    - q: ARCH lag order (default 1)
    - show_diagnostics: Run tests on standardized residuals

    Returns:
    - garch_res: Fitted ARCH model object (or None if fitting fails)
    - diagnostics: Dict with test results (or None)
    """
    print(f"\n--- Fitting GARCH({p},{q})-X model with {wind_var} in variance equation ---")

    try:
        # Align wind variable with residuals index
        wind_series = df.loc[armax_residuals.index, wind_var]

        # Check for NaNs
        if wind_series.isna().any():
            print(f"WARNING: {wind_var} contains NaN values. Dropping NaNs...")
            valid_idx = ~(armax_residuals.isna() | wind_series.isna())
            armax_residuals = armax_residuals[valid_idx]
            wind_series = wind_series[valid_idx]

        # Specify GARCH(p,q)-X model
        garch_spec = arch_model(
            armax_residuals,
            vol='GARCH',
            p=p, q=q,
            x=wind_series.values.reshape(-1, 1),  # Must be 2D array
            rescale=False
        )

        # Fit with MLE
        garch_res = garch_spec.fit(disp='off', show_warning=False)

        # Display results
        print(f"\nVARIANCE EQUATION - GARCH({p},{q})-X:")
        print(garch_res.summary())

        # Extract standardized residuals
        std_resid = garch_res.std_resid

        # Run diagnostics if requested
        diagnostics = None
        if show_diagnostics:
            print("\n" + "="*80)
            print("DIAGNOSTIC TESTS ON GARCH STANDARDIZED RESIDUALS")
            print("="*80)
            print("(These should show NO autocorrelation and NO ARCH effects)")

            # Ljung-Box test
            run_ljungbox_test(std_resid, lags=[5, 10, 15, 20])

            # Heteroskedasticity tests
            run_heteroskedasticity_tests(std_resid, nlags=10)

            diagnostics = {
                'std_resid_mean': std_resid.mean(),
                'std_resid_std': std_resid.std(),
                'aic': garch_res.aic,
                'bic': garch_res.bic
            }

        return garch_res, diagnostics

    except Exception as e:
        print(f"ERROR: GARCH fitting failed - {str(e)}")
        print("Continuing with ARMAX-only results...")
        return None, None


def perform_multivariate_analysis(df, zone, target_region='SE1',
                                 run_ljungbox=False, run_hetero_tests=False, run_stationarity=False,
                                 optimize_armax_lags=False, use_checkpointed_lag_selection=True,
                                 armax_search_p_min=0, armax_search_p_max=10,
                                 armax_search_q_min=0, armax_search_q_max=10,
                                 armax_search_exclude_00=True,
                                 armax_search_require_convergence=True,
                                 armax_search_require_ljungbox_pass=True,
                                 armax_search_selection_criterion='bic',
                                 armax_search_save_top_n=20,
                                 run_tvp_wind_kalman=False,
                                 run_rolling_window=False, rolling_window_years=3,
                                 rolling_step_years=1, rolling_min_obs=24*180,
                                 run_quantile_regression=False,
                                 run_structural_break=False, structural_break_type='level',
                                 structural_break_max_breaks=5,
                                 structural_break_trimming=0.15, structural_break_known_dates=None,
                                 structural_break_window_years=1, structural_break_step_years=1/12,
                                 structural_break_min_obs=24*365 - 24*30,
                                 trend_break_test_method='legacy',
                                 bp_inference_mode='both',
                                 bp_significance_level=0.05,
                                 bp_bootstrap_reps=999,
                                 bp_bootstrap_block_length=8,
                                 bp_random_seed=42,
                                 bp_use_hac_se=True,
                                 structural_break_estimation_model='ols',
                                 armax_baseline_spec=None,
                                 save_armax_exog_anchor=False,
                                 armax_scale_exog=False):
    """
    Runs OLS, ARMAX, and conditionally GARCH-X with full control variables.

    GARCH-X is fitted only if ARCH effects detected in ARMAX residuals (p < 0.05).

    Note: Always uses logged and deseasonalized variables (standard approach).

    Parameters:
    - df: DataFrame with all variables
    - zone: Zone identifier for display purposes
    - target_region: Target region for bottleneck dummies (default 'SE1')

    Returns:
    - ols_model: OLS regression results
    - armax_res: ARMAX model results
    - garch_res: GARCH-X results (None if not fitted)
    """
    print(f"\n--- RUNNING MULTIVARIATE ANALYSIS ({zone}) ---")
    print("Using: Logged and Deseasonalized variables (Standard approach)")

    y_name, exog_vars = get_regression_variable_names(df, target_region=target_region)
    Y = df[y_name]

    print(f"Dependent variable: {Y.name}")
    print(f"Exogenous variables: {exog_vars}")

    # TVP Kalman Filter mode: run time-varying parameter analysis and return early
    if run_tvp_wind_kalman:
        run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, plots_dir="plots")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Rolling-window mode: run rolling window analysis and return early
    if run_rolling_window:
        run_rolling_window_analysis(df, zone, Y, exog_vars,
                                    window_years=rolling_window_years,
                                    step_years=rolling_step_years,
                                    min_obs=rolling_min_obs,
                                    plots_dir="plots",
                                    results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Quantile regression mode: run quantile regression analysis and return early
    if run_quantile_regression:
        run_quantile_regression_analysis(df, zone,
                                         plots_dir="plots",
                                         results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Structural break mode: run analysis and return early
    if run_structural_break:
        if structural_break_type == 'trend':
            # Trend break analysis: detects changes in coefficient slope (sequential testing)
            run_trend_break_analysis(df, zone, Y, exog_vars,
                                    max_breaks=structural_break_max_breaks,
                                    trimming=structural_break_trimming,
                                    window_years=structural_break_window_years,
                                    step_years=structural_break_step_years,
                                    min_obs=structural_break_min_obs,
                                    trend_break_test_method=trend_break_test_method,
                                    bp_inference_mode=bp_inference_mode,
                                    bp_significance_level=bp_significance_level,
                                    bp_bootstrap_reps=bp_bootstrap_reps,
                                    bp_bootstrap_block_length=bp_bootstrap_block_length,
                                    bp_random_seed=bp_random_seed,
                                    bp_use_hac_se=bp_use_hac_se,
                                    estimation_model=structural_break_estimation_model,
                                    dynamic_armax_order=(3, 0, 3),
                                    plots_dir="plots",
                                    results_dir="results")
        else:  # 'level' or default
            # Level break analysis: detects step changes in coefficient mean (Bai-Perron)
            run_structural_break_analysis(df, zone, Y, exog_vars,
                                          max_breaks=structural_break_max_breaks,
                                          trimming=structural_break_trimming,
                                          known_break_dates=structural_break_known_dates,
                                          window_years=structural_break_window_years,
                                          step_years=structural_break_step_years,
                                          min_obs=structural_break_min_obs,
                                          estimation_model=structural_break_estimation_model,
                                          dynamic_armax_order=(3, 0, 3),
                                          plots_dir="plots",
                                          results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    X = sm.add_constant(df[exog_vars])

    # 1. Standard OLS Regression
    ols_model = sm.OLS(Y, X).fit()
    print("\n--- OLS RESULTS ---")
    print(ols_model.summary())

    # Optional: Diagnostic tests on OLS residuals
    if run_stationarity:
        # Test stationarity of ALL variables used in the regression
        print("\n" + "="*80)
        print("STATIONARITY TESTS FOR ALL REGRESSION VARIABLES")
        print("="*80)

        # Test dependent variable (Price)
        run_stationarity_tests(Y, series_name=f"{zone} {Y.name} (Dependent Variable)")

        # Test all independent variables
        for var in exog_vars:
            run_stationarity_tests(df[var], series_name=f"{zone} {var} (Independent Variable)")

    if run_ljungbox:
        # Test for autocorrelation in OLS residuals
        run_ljungbox_test(ols_model.resid, lags=[5, 10, 15, 20])

    if run_hetero_tests:
        # Test for heteroskedasticity and ARCH effects in OLS residuals
        run_heteroskedasticity_tests(ols_model.resid, nlags=10)

    # 2. ARMAX(3,3)-GARCHX(1,1) Framework
    print(f"\n--- ARMAX-GARCHX RESULTS ---")
    baseline_spec = _validate_armax_baseline_spec(armax_baseline_spec)

    # Determine optimal lags if enabled, otherwise use default (3,3)
    if optimize_armax_lags:
        if len(baseline_spec.get('extra_ar_lags', [])) > 0:
            print(
                f"Note: baseline sparse AR lags {baseline_spec['extra_ar_lags']} are ignored when OPTIMIZE_ARMAX_LAGS=True "
                "(search uses contiguous ARMA(p,q) only)."
            )
        if use_checkpointed_lag_selection:
            # Use new checkpointed version with Ljung-Box diagnostics
            optimal_order, search_df = select_armax_lags_aic_checkpointed(
                Y, df[exog_vars],
                zone=zone,
                p_min=armax_search_p_min,
                p_max=armax_search_p_max,
                q_min=armax_search_q_min,
                q_max=armax_search_q_max,
                exclude_00=armax_search_exclude_00,
                require_convergence=armax_search_require_convergence,
                require_ljungbox_pass=armax_search_require_ljungbox_pass,
                selection_criterion=armax_search_selection_criterion,
                checkpoint_file=None,  # Auto-generate based on zone
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                scale_exog=armax_scale_exog,
                save_top_n=armax_search_save_top_n
            )
        else:
            # Use strict version (no checkpointing)
            optimal_order, search_df = select_armax_lags_aic(
                Y, df[exog_vars],
                zone=zone,
                p_min=armax_search_p_min,
                p_max=armax_search_p_max,
                q_min=armax_search_q_min,
                q_max=armax_search_q_max,
                exclude_00=armax_search_exclude_00,
                require_convergence=armax_search_require_convergence,
                require_ljungbox_pass=armax_search_require_ljungbox_pass,
                selection_criterion=armax_search_selection_criterion,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                scale_exog=armax_scale_exog,
                save_top_n=armax_search_save_top_n
            )
        if optimal_order is None:
            raise RuntimeError(
                "No admissible ARMAX order found in lag search. "
                "Check results/armax_search_all_<ZONE>.csv and relax search constraints or widen p/q range."
            )
        armax_order = (optimal_order[0], 0, optimal_order[1])
        Y_armax = Y
        X_armax = df[exog_vars]
        added_lag_cols = []
        dropped_rows = 0
    else:
        if baseline_spec.get('enabled', True):
            armax_order = baseline_spec['order']
            extra_lags = baseline_spec.get('extra_ar_lags', [])
            spec_label = baseline_spec.get('label', f'ARMAX{armax_order}')
        else:
            armax_order = (1, 0, 1)
            extra_lags = []
            spec_label = "ARMAX baseline disabled -> fallback ARMAX(1,0,1)"

        Y_armax, X_armax, added_lag_cols = _prepare_baseline_armax_design(
            Y,
            df[exog_vars],
            extra_ar_lags=extra_lags,
            drop_initial_nan=baseline_spec.get('drop_initial_nan', True)
        )
        dropped_rows = int(len(Y) - len(Y_armax))
        print(
            f"Using baseline spec: {spec_label}, "
            f"order={armax_order}, extra_ar_lags={extra_lags}"
        )
        if len(added_lag_cols) > 0:
            print(f"Added lagged dependent regressors: {added_lag_cols}")
        print(f"ARMAX estimation sample: n={len(Y_armax)} (dropped {dropped_rows} rows due to lagging/alignment)")
        print(f"Set OPTIMIZE_ARMAX_LAGS=True for AIC/BIC-based contiguous lag search.")

    # Mean Equation (Price Level)
    print(f"\n--- Fitting ARMAX{armax_order} model ---")

    armax_fit = _fit_armax_with_fallback(
        y=Y_armax,
        X_exog=X_armax,
        primary_order=armax_order,
        context_label=f"main ARMAX ({zone})",
        allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
        maxiter=ARMAX_MAXITER,
        solver=ARMAX_SOLVER,
        use_warm_start=ARMAX_USE_WARM_START,
        scale_exog=armax_scale_exog,
        enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
        fallback_orders=ARMAX_FALLBACK_ORDERS
    )
    if not armax_fit['ok']:
        attempts = armax_fit.get('attempts', [])
        print("ARMAX fitting failed. Attempt diagnostics:")
        for a in attempts:
            diag = a.get('diagnostics', {})
            print(f"  order={a.get('order')} ok={a.get('ok')} converged={a.get('converged')} "
                  f"iter={diag.get('iterations')} warnflag={diag.get('warnflag')} err={a.get('error')}")
        raise RuntimeError(
            "No converged ARMAX model found. "
            "You can relax ARMAX_ALLOW_NONCONVERGED=True to use non-converged estimates."
        )
    armax_res = armax_fit['model']
    selected_order = armax_fit.get('selected_order', armax_order)
    diag = armax_fit.get('diagnostics', {})
    print(f"ARMAX fit status: converged={armax_fit.get('converged')} "
          f"order_used={selected_order} iterations={diag.get('iterations')} "
          f"warnflag={diag.get('warnflag')} fopt={diag.get('fopt')}")
    attempts = armax_fit.get('attempts', [])
    primary_attempt = None
    for a in attempts:
        if tuple(a.get('order', ())) == tuple(armax_order):
            primary_attempt = a
            break

    if primary_attempt is not None and not primary_attempt.get('converged', False):
        pdiag = primary_attempt.get('diagnostics', {}) or {}
        label, explanation = _diagnose_nonconvergence_simple(
            primary_attempt,
            configured_maxiter=ARMAX_MAXITER
        )
        print("\n" + "="*80)
        print("ARMAX NON-CONVERGENCE DIAGNOSIS (PRIMARY ORDER)")
        print("="*80)
        print(f"Attempted order: ARMAX{armax_order}")
        print(f"Converged:       {primary_attempt.get('converged')}")
        print(f"Iterations:      {pdiag.get('iterations')}")
        print(f"Warnflag:        {pdiag.get('warnflag')}")
        print(f"Gradient max abs:{pdiag.get('gradient_max_abs')}")
        print(f"Diagnosis:       {label}")
        print(f"Why:             {explanation}")
        if tuple(selected_order) != tuple(armax_order):
            selected_attempt = next(
                (a for a in attempts if tuple(a.get('order', ())) == tuple(selected_order)),
                None
            )
            sdiag = (selected_attempt or {}).get('diagnostics', {}) or {}
            print(f"Fallback succeeded with ARMAX{selected_order} "
                  f"(iterations={sdiag.get('iterations')}, warnflag={sdiag.get('warnflag')}).")
        print("="*80)
    if armax_fit.get('used_nonconverged', False):
        print("WARNING: Using non-converged ARMAX fit due to ARMAX_ALLOW_NONCONVERGED=True.")

    print(f"\nMEAN EQUATION (Price Level) - ARMAX{selected_order}:")
    print(armax_res.summary())
    if bool(diag.get('exog_scaled', False)):
        try:
            back_df = _build_exog_backtransform_table(
                armax_model=armax_res,
                exog_names=exog_vars,
                scale_means=diag.get('scale_means', {}),
                scale_stds=diag.get('scale_stds', {})
            )
            if not back_df.empty:
                print("\n--- EXOGENOUS COEFFICIENTS BACK-TRANSFORMED TO ORIGINAL UNITS ---")
                print(f"{'Term':<38} {'Raw Coef':>12} {'Raw SE':>12} {'p-value':>12}")
                print("-" * 78)
                for _, row in back_df.iterrows():
                    print(
                        f"{row['term']:<38} "
                        f"{row['coef_raw']:>12.6f} {row['se_raw']:>12.6f} {row['p_value']:>12.4g}"
                    )
        except Exception as e:
            print(f"WARNING: Failed to print back-transformed exogenous coefficients: {e}")

    if save_armax_exog_anchor:
        if zone == 'SE1' and tuple(selected_order) == (1, 0, 1):
            try:
                anchor_map = _extract_armax_exog_anchor(
                    armax_res,
                    exog_vars,
                    scale_means=diag.get('scale_means', {}) if bool(diag.get('exog_scaled', False)) else None,
                    scale_stds=diag.get('scale_stds', {}) if bool(diag.get('exog_scaled', False)) else None
                )
                anchor_path = _save_armax_exog_anchor_csv(
                    anchor_map=anchor_map,
                    zone=zone,
                    source_order=tuple(selected_order),
                    results_dir='results'
                )
                print(f"Saved ARMAX exogenous anchor vector: {anchor_path} ({len(anchor_map)} terms)")
            except Exception as e:
                print(f"WARNING: Failed to save ARMAX exogenous anchor vector: {e}")
        else:
            print(
                f"Anchor extraction skipped: requires zone='SE1' and selected_order=(1, 0, 1), "
                f"got zone='{zone}', selected_order={tuple(selected_order)}."
            )

    # Optional: Diagnostic tests on ARMAX residuals
    arch_detected = False
    if run_ljungbox:
        print("\n" + "="*70)
        print("DIAGNOSTIC TESTS ON ARMAX RESIDUALS")
        print("="*70)
        run_ljungbox_test(armax_res.resid, lags=[5, 10, 15, 20])

    if run_hetero_tests:
        # Run tests and check if ARCH effects detected
        run_heteroskedasticity_tests(armax_res.resid, nlags=10)

        # Check ARCH test result
        lm_stat, lm_pval, f_stat, f_pval = het_arch(armax_res.resid, nlags=10)
        if lm_pval < 0.05:
            arch_detected = True
            print(f"\n{'='*70}")
            print(f"ARCH EFFECTS DETECTED (p={lm_pval:.4f} < 0.05)")
            print(f"Proceeding with GARCH-X modeling...")
            print(f"{'='*70}")

    # GARCH-X component: Fit if ARCH effects confirmed
    garch_res = None
    if arch_detected and FIT_GARCH_IF_ARCH:
        # Always use logged wind variable
        wind_var = 'Wind_Forecast_Log'

        garch_res, garch_diagnostics = fit_garchx_model(
            armax_res.resid,
            df,
            wind_var=wind_var,
            p=GARCH_ORDER[0],
            q=GARCH_ORDER[1],
            show_diagnostics=True
        )

        # Compare AIC/BIC
        if garch_res is not None:
            print(f"\n{'='*70}")
            print("MODEL COMPARISON")
            print(f"{'='*70}")
            print(f"ARMAX({selected_order[0]},{selected_order[2]}) AIC: {armax_res.aic:.2f}")
            print(f"ARMAX-GARCH({GARCH_ORDER[0]},{GARCH_ORDER[1]})-X AIC: {garch_res.aic:.2f}")
            improvement = armax_res.aic - garch_res.aic
            print(f"AIC Improvement: {improvement:.2f} {'(better)' if improvement > 0 else '(worse)'}")
            print(f"{'='*70}")
    elif not arch_detected:
        print(f"\n{'='*70}")
        print("NO ARCH EFFECTS DETECTED - GARCH modeling not necessary")
        print("ARMAX model is sufficient (constant variance assumption holds)")
        print(f"{'='*70}")

    return ols_model, armax_res, garch_res


# --- 5. VISUALIZATION FUNCTIONS ---

def plot_zone_comparisons(start_date=None, end_date=None, plots_dir='plots'):
    """
    Create comparison plots overlaying SE1-SE4 on the same figure.

    Produces 4 plots:
    1. Energy price level (Spot_Price)
    2. Energy price level log (log of Spot_Price)
    3. Energy price volatility (24h rolling std of Spot_Price)
    4. Wind production share of total production (Wind_Forecast / Total_Forecast)

    Reads directly from Combined_SE{N}_Data files (no need for full load_data pipeline).
    """
    print("\n" + "="*80)
    print("ZONE COMPARISON PLOTS (SE1-SE4)")
    print("="*80)

    zones = ['SE1', 'SE2', 'SE3', 'SE4']
    zone_colors = {'SE1': '#e74c3c', 'SE2': '#3498db', 'SE3': '#2ecc71', 'SE4': '#f39c12'}
    zone_data = {}

    # Load data for all zones
    for zone in zones:
        filepath = f'master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx'
        print(f"  Loading {zone} from {filepath}...")
        df = pd.read_excel(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp').sort_index()

        # Apply date filter
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        zone_data[zone] = df
        print(f"    {len(df)} observations ({df.index.min()} to {df.index.max()})")

    # Create output directory
    comparison_dir = os.path.join(plots_dir, 'zone_comparisons')
    os.makedirs(comparison_dir, exist_ok=True)

    sns.set_style("whitegrid")
    date_range_str = f"{start_date or 'start'} to {end_date or 'end'}"

    # Resample all zones to monthly means for readable comparisons
    monthly = {}
    for zone in zones:
        df = zone_data[zone]
        m = pd.DataFrame()
        m['Spot_Price'] = df['Spot_Price'].resample('MS').mean()
        m['Spot_Price_Log'] = np.log(df['Spot_Price'].clip(lower=0.01)).resample('MS').mean()
        m['Volatility'] = df['Spot_Price'].rolling(window=24).std().resample('MS').mean()
        monthly[zone] = m

    # Resample wind share to yearly means for cleaner trend visibility
    yearly = {}
    for zone in zones:
        df = zone_data[zone]
        wind_share = df['Wind_Forecast'] / df['Total_Forecast'].replace(0, np.nan) * 100
        yearly[zone] = wind_share.resample('YS').mean()

    # --- Combined 4-panel figure ---
    print("\n  Creating: Combined zone comparison plot (4 panels)...")
    fig, axes = plt.subplots(4, 1, figsize=(16, 24))
    fig.suptitle(f'Zone Comparison — SE1-SE4 ({date_range_str})', fontsize=16, fontweight='bold')

    # Panel 1: Energy price level
    ax = axes[0]
    for zone in zones:
        ax.plot(monthly[zone].index, monthly[zone]['Spot_Price'],
                color=zone_colors[zone], linewidth=1.5, label=zone, marker='o', markersize=2)
    ax.set_title('Energy Price Level — Monthly Mean', fontsize=14, fontweight='bold')
    ax.set_ylabel('EUR/MWh', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    stats_lines = [f'{z}: Mean={zone_data[z]["Spot_Price"].mean():.2f}, Std={zone_data[z]["Spot_Price"].std():.2f}' for z in zones]
    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Energy price level (log)
    ax = axes[1]
    for zone in zones:
        ax.plot(monthly[zone].index, monthly[zone]['Spot_Price_Log'],
                color=zone_colors[zone], linewidth=1.5, label=zone, marker='o', markersize=2)
    ax.set_title('Energy Price Level (Log) — Monthly Mean', fontsize=14, fontweight='bold')
    ax.set_ylabel('log(EUR/MWh)', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    stats_lines = [f'{z}: Mean={np.log(zone_data[z]["Spot_Price"].clip(lower=0.01)).mean():.2f}, Std={np.log(zone_data[z]["Spot_Price"].clip(lower=0.01)).std():.2f}' for z in zones]
    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Energy price volatility
    ax = axes[2]
    for zone in zones:
        ax.plot(monthly[zone].index, monthly[zone]['Volatility'],
                color=zone_colors[zone], linewidth=1.5, label=zone, marker='o', markersize=2)
    ax.set_title('Energy Price Volatility (24h Rolling Std) — Monthly Mean', fontsize=14, fontweight='bold')
    ax.set_ylabel('EUR/MWh', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    stats_lines = [f'{z}: Mean={zone_data[z]["Spot_Price"].rolling(24).std().dropna().mean():.2f}, Std={zone_data[z]["Spot_Price"].rolling(24).std().dropna().std():.2f}' for z in zones]
    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 4: Wind production share (yearly mean)
    ax = axes[3]
    for zone in zones:
        ax.plot(yearly[zone].index, yearly[zone],
                color=zone_colors[zone], linewidth=2, label=zone, marker='o', markersize=5)
    ax.set_title('Wind Production Share of Total Production — Yearly Mean (Forecasted)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wind Share (%)', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    stats_lines = [f'{z}: Mean={(zone_data[z]["Wind_Forecast"]/zone_data[z]["Total_Forecast"].replace(0,np.nan)*100).dropna().mean():.1f}%, Std={(zone_data[z]["Wind_Forecast"]/zone_data[z]["Total_Forecast"].replace(0,np.nan)*100).dropna().std():.1f}%' for z in zones]
    ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = os.path.join(comparison_dir, 'zone_comparison_all.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fpath}")

    print(f"\nAll zone comparison plots saved to: {comparison_dir}/")


def plot_time_series(df, zone, stage='raw', plots_dir='plots'):
    """Create time series plots for all variables."""

    print("\n--- Creating Time Series Plots ---")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(6, 1, figsize=(16, 24))
    stage_title = stage.replace('_', ' ').title()
    fig.suptitle(f'{stage_title} Time Series Data - {zone} (2021-2024)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#8B4513']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'EUR/MWh']

    for i, (var, color, unit) in enumerate(zip(variables, colors, units)):
        ax = axes[i]
        ax.plot(df.index, df[var], color=color, alpha=0.7, linewidth=0.5)
        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{unit}', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics as text
        stats_text = f'Mean: {df[var].mean():.2f}\nStd: {df[var].std():.2f}\nMin: {df[var].min():.2f}\nMax: {df[var].max():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'{stage}_time_series_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_distributions(df, zone, method='fredriksson', stage='raw', plots_dir='plots'):
    """Create distribution plots (histograms + KDE) for all variables."""

    print("\n--- Creating Distribution Plots ---")

    # Set thresholds based on method
    if method == 'fredriksson':
        upper_mult = 6.0
        lower_mult = -3.7
        method_label = 'Fredriksson'
    elif method == 'gianfreda':
        upper_mult = 3.0
        lower_mult = -3.0
        method_label = 'Gianfreda'
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    stage_title = stage.replace('_', ' ').title()
    fig.suptitle(f'Distribution of {stage_title} Variables - {zone} (with outlier detection)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#8B4513']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'EUR/MWh']

    axes = axes.flatten()

    for i, (var, color, unit) in enumerate(zip(variables, colors, units)):
        ax = axes[i]
        data = df[var].dropna()

        # Plot histogram with KDE
        ax.hist(data, bins=50, color=color, alpha=0.6, edgecolor='black', density=True, label='Histogram')

        # Add KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')

        # Calculate statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()

        # Mark mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        # Mark outlier thresholds based on method
        upper_threshold = mean_val + upper_mult * std_val
        lower_threshold = mean_val + lower_mult * std_val
        ax.axvline(upper_threshold, color='orange', linestyle=':', linewidth=2, label=f'{upper_mult:+.1f}*std: {upper_threshold:.2f}')
        ax.axvline(lower_threshold, color='orange', linestyle=':', linewidth=2, label=f'{lower_mult:+.1f}*std: {lower_threshold:.2f}')

        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{unit}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Count outliers using specified method
        outliers_upper = (data > upper_threshold).sum()
        outliers_lower = (data < lower_threshold).sum()
        total_outliers = outliers_upper + outliers_lower

        # Add outlier count
        outlier_text = f'Outliers ({method_label}):\nUpper (>{upper_mult:+.1f}*std): {outliers_upper}\nLower (<{lower_mult:+.1f}*std): {outliers_lower}\nTotal: {total_outliers}'
        ax.text(0.98, 0.98, outlier_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'{stage}_distributions_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_boxplots(df, zone, stage='raw', plots_dir='plots'):
    """Create box plots to visualize outliers."""

    print("\n--- Creating Box Plots ---")

    fig, axes = plt.subplots(1, 6, figsize=(24, 6))
    stage_title = stage.replace('_', ' ').title()
    fig.suptitle(f'Box Plots for Outlier Detection - {zone} ({stage_title} Data)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#8B4513']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'EUR/MWh']

    for i, (var, color, unit) in enumerate(zip(variables, colors, units)):
        ax = axes[i]
        data = df[var].dropna()

        # Create box plot
        bp = ax.boxplot(data, patch_artist=True, vert=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{unit}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Calculate IQR-based outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

        # Add statistics
        stats_text = f'Q1: {Q1:.2f}\nMedian: {data.median():.2f}\nQ3: {Q3:.2f}\nIQR Outliers: {outliers_iqr}'
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'{stage}_boxplots_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def detect_outliers(df, zone, method='fredriksson', plots_dir='plots'):
    """
    Detect outliers using specified methodology.

    Parameters:
    - df: DataFrame with variables to check
    - zone: Region name (for reporting)
    - method: 'fredriksson' or 'gianfreda'
    - plots_dir: Directory for saving results

    Fredriksson (2016):
    - Upper: +6σ, Lower: -3.7σ (asymmetric)

    Gianfreda (2010) / Mugele et al. (2005):
    - Upper: +3σ, Lower: -3σ (symmetric)
    """

    if method == 'fredriksson':
        print("\n" + "="*80)
        print(f"OUTLIER DETECTION - FREDRIKSSON (2016) METHODOLOGY - {zone}")
        print("="*80)
        print("Definition: Outliers exceed +6*std above mean OR fall below -3.7*std below mean\n")
        upper_multiplier = 6.0
        lower_multiplier = -3.7
    elif method == 'gianfreda':
        print("\n" + "="*80)
        print(f"OUTLIER DETECTION - GIANFREDA (2010) / MUGELE ET AL. (2005) METHODOLOGY - {zone}")
        print("="*80)
        print("Definition: Outliers exceed ±3*std (symmetric threshold)\n")
        upper_multiplier = 3.0
        lower_multiplier = -3.0
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'fredriksson' or 'gianfreda'.")

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']

    outlier_summary = []

    for var in variables:
        data = df[var].dropna()
        mean_val = data.mean()
        std_val = data.std()

        # Calculate thresholds based on method
        upper_threshold = mean_val + upper_multiplier * std_val
        lower_threshold = mean_val + lower_multiplier * std_val

        # Identify outliers
        outliers_upper = data > upper_threshold
        outliers_lower = data < lower_threshold
        outliers_total = outliers_upper | outliers_lower

        n_outliers_upper = outliers_upper.sum()
        n_outliers_lower = outliers_lower.sum()
        n_outliers_total = outliers_total.sum()

        pct_outliers = (n_outliers_total / len(data)) * 100

        print(f"\n{var}:")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Std Dev: {std_val:.2f}")
        print(f"  Upper threshold ({upper_multiplier:+.1f}*std): {upper_threshold:.2f}")
        print(f"  Lower threshold ({lower_multiplier:+.1f}*std): {lower_threshold:.2f}")
        print(f"  Outliers above threshold: {n_outliers_upper}")
        print(f"  Outliers below threshold: {n_outliers_lower}")
        print(f"  Total outliers: {n_outliers_total} ({pct_outliers:.2f}% of data)")

        if n_outliers_total > 0:
            print(f"  Min outlier value: {data[outliers_total].min():.2f}")
            print(f"  Max outlier value: {data[outliers_total].max():.2f}")

        outlier_summary.append({
            'Variable': var,
            'Mean': mean_val,
            'Std': std_val,
            'Upper_Threshold': upper_threshold,
            'Lower_Threshold': lower_threshold,
            'Upper_Multiplier': upper_multiplier,
            'Lower_Multiplier': lower_multiplier,
            'N_Outliers_Upper': n_outliers_upper,
            'N_Outliers_Lower': n_outliers_lower,
            'N_Outliers_Total': n_outliers_total,
            'Pct_Outliers': pct_outliers
        })

    return pd.DataFrame(outlier_summary)


def plot_outliers_timeline(df, zone, method='fredriksson', stage='raw', plots_dir='plots'):
    """Plot time series highlighting detected outliers."""

    print("\n--- Creating Outlier Timeline Visualization ---")

    # Set thresholds based on method
    if method == 'fredriksson':
        upper_mult = 6.0
        lower_mult = -3.7
        method_label = 'Fredriksson'
    elif method == 'gianfreda':
        upper_mult = 3.0
        lower_mult = -3.0
        method_label = 'Gianfreda'
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, axes = plt.subplots(6, 1, figsize=(16, 24))
    stage_title = stage.replace('_', ' ').title()
    fig.suptitle(f'Time Series with Detected Outliers - {zone} ({method_label} Methodology, {stage_title})',
                 fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#8B4513']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh', 'EUR/MWh']

    for i, (var, color, unit) in enumerate(zip(variables, colors, units)):
        ax = axes[i]
        data = df[var]

        # Calculate thresholds
        mean_val = data.mean()
        std_val = data.std()
        upper_threshold = mean_val + upper_mult * std_val
        lower_threshold = mean_val + lower_mult * std_val

        # Identify outliers
        outliers = (data > upper_threshold) | (data < lower_threshold)

        # Plot normal data
        ax.plot(df.index, data, color=color, alpha=0.5, linewidth=0.5, label='Normal data')

        # Highlight outliers
        ax.scatter(df.index[outliers], data[outliers], color='red', s=20,
                  alpha=0.8, label=f'Outliers (n={outliers.sum()})', zorder=5)

        # Add threshold lines
        ax.axhline(upper_threshold, color='orange', linestyle='--',
                  linewidth=1.5, label=f'{upper_mult:+.1f}*std: {upper_threshold:.2f}')
        ax.axhline(lower_threshold, color='orange', linestyle='--',
                  linewidth=1.5, label=f'{lower_mult:+.1f}*std: {lower_threshold:.2f}')
        ax.axhline(mean_val, color='green', linestyle='-',
                  linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.2f}')

        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{unit}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'{stage}_outliers_timeline_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_scatter_matrix(df, zone, stage='raw', plots_dir='plots'):
    """Create scatter plot matrix to see relationships between variables."""

    print("\n--- Creating Scatter Plot Matrix ---")

    # Select variables for scatter matrix
    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']

    # Sample data if too large (for performance)
    if len(df) > 5000:
        df_sample = df[variables].sample(n=5000, random_state=42)
        print(f"  Sampling 5000 points from {len(df)} for visualization performance")
    else:
        df_sample = df[variables]

    # Create scatter matrix
    fig = plt.figure(figsize=(16, 16))
    axes = pd.plotting.scatter_matrix(df_sample, alpha=0.3, figsize=(16, 16),
                                      diagonal='kde', color='#3498db')

    # Adjust appearance
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    stage_title = stage.replace('_', ' ').title()
    plt.suptitle(f'Scatter Plot Matrix - Relationships Between Variables - {zone} ({stage_title})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'{stage}_scatter_matrix_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def run_visualizations(data, zone, method='fredriksson', stage='raw', plots_dir='plots'):
    """
    Run all visualization functions.

    Parameters:
    - data: DataFrame with variables to visualize
    - zone: Region name (SE1, SE2, etc.)
    - method: Outlier detection method ('fredriksson' or 'gianfreda')
    - stage: Data transformation stage ('raw', 'logged', etc.)
    - plots_dir: Base directory for plots (zone-specific subfolder will be created)
    """

    stage_title = stage.replace('_', ' ').title()
    print("\n" + "="*80)
    print(f"RUNNING {stage_title.upper()} DATA VISUALIZATIONS - {zone}")
    print("="*80)

    # Create zone-specific subdirectory
    zone_plots_dir = os.path.join(plots_dir, zone)
    if not os.path.exists(zone_plots_dir):
        os.makedirs(zone_plots_dir)
        print(f"\nCreated directory: {zone_plots_dir}/")

    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 8)

    # Basic statistics
    print("\n" + "="*80)
    print(f"DESCRIPTIVE STATISTICS ({stage_title.upper()} DATA)")
    print("="*80)
    print(data[['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption', 'Oil_Price']].describe())

    # Detect outliers using specified methodology
    outlier_summary = detect_outliers(data, zone, method=method, plots_dir=zone_plots_dir)

    # Save outlier summary to CSV in zone-specific plots directory
    csv_path = os.path.join(zone_plots_dir, f'outlier_summary_{method}_{stage}_{zone}.csv')
    outlier_summary.to_csv(csv_path, index=False)
    print(f"\n  Saved outlier summary to: {csv_path}")

    # Create visualizations
    print("\n--- Generating Visualizations ---")
    plot_time_series(data, zone, stage=stage, plots_dir=zone_plots_dir)
    plot_distributions(data, zone, method=method, stage=stage, plots_dir=zone_plots_dir)
    plot_boxplots(data, zone, stage=stage, plots_dir=zone_plots_dir)
    plot_outliers_timeline(data, zone, method=method, stage=stage, plots_dir=zone_plots_dir)
    plot_scatter_matrix(data, zone, stage=stage, plots_dir=zone_plots_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {zone_plots_dir}/")
    print("\nGenerated files:")
    print(f"  1. {stage}_time_series_{zone}.png - Time series plots for all variables")
    print(f"  2. {stage}_distributions_{zone}.png - Distribution histograms with outlier thresholds")
    print(f"  3. {stage}_boxplots_{zone}.png - Box plots for outlier detection")
    print(f"  4. {stage}_outliers_timeline_{zone}.png - Time series with outliers highlighted")
    print(f"  5. {stage}_scatter_matrix_{zone}.png - Relationships between variables")
    print(f"  6. outlier_summary_{method}_{stage}_{zone}.csv - Outlier detection statistics")


def export_data_for_R(data, zone, use_log_transform=False, use_deseasonalized=False,
                      handle_outliers=False, outlier_method='fredriksson',
                      negative_price_handling='clip', use_interpolation=False,
                      export_dir='data_for_R'):
    """
    Export fully processed regression data to CSV for R structural break analysis.

    This function exports the cleaned, transformed, and regression-ready data for use
    with R's strucchange package (Bai-Perron structural break tests).

    Parameters:
    - data: DataFrame with all processed variables (indexed by Datetime)
    - zone: Zone identifier (e.g., 'SE1', 'SE2')
    - use_log_transform: Whether log transformation was applied
    - use_deseasonalized: Whether deseasonalization was applied
    - handle_outliers: Whether outlier handling was applied
    - outlier_method: Method used for outlier handling
    - negative_price_handling: Method used for negative price handling
    - use_interpolation: Whether interpolation was used for missing values
    - export_dir: Directory to save exported files (default: 'data_for_R')

    Returns:
    - export_path: Path to the exported CSV file

    Outputs:
    - CSV file with regression-ready data
    - Metadata text file with configuration and starter R code
    """
    print("\n" + "="*80)
    print("EXPORTING DATA FOR R ANALYSIS (BAI-PERRON STRUCTURAL BREAKS)")
    print("="*80)

    # Determine which variables to export based on transformation flags
    if use_log_transform and use_deseasonalized:
        print("Exporting: Logged and Deseasonalized variables")

        # Create export dataframe with regression-ready variables
        export_data = pd.DataFrame({
            'Datetime': data.index,
            'Price': data['Price_Log_Deseasonalized'],
            'Wind_Forecast': data['Wind_Forecast_Log'],  # NOT deseasonalized per Fredriksson
            'Hydro_Reserves': data['Hydro_Reserves_Log_Deseasonalized'],
            'Net_Exchange': data['Net_Exchange'],  # NOT logged or deseasonalized
            'Consumption': data['Consumption_Log_Deseasonalized'],
            'Oil_Price': data['Oil_Price_Log_Deseasonalized'],
            'Gas_Price': data['Gas_Price_Log_Deseasonalized']
        })

        # Add bottleneck dummies if present
        trading_partners = TRADING_PARTNERS.get(zone, [])
        for partner in trading_partners:
            bneck_col = f'BNECK_{zone}_{partner}'
            if bneck_col in data.columns:
                export_data[bneck_col] = data[bneck_col]
                print(f"  Added bottleneck dummy: {bneck_col}")

    elif use_log_transform:
        print("Exporting: Logged variables (not deseasonalized)")

        export_data = pd.DataFrame({
            'Datetime': data.index,
            'Price': data['Price_Log'],
            'Wind_Forecast': data['Wind_Forecast_Log'],
            'Hydro_Reserves': data['Hydro_Reserves_Log'],
            'Net_Exchange': data['Net_Exchange'],
            'Consumption': data['Consumption_Log'],
            'Oil_Price': data['Oil_Price_Log'],
            'Gas_Price': data['Gas_Price_Log']
        })

        # Add bottleneck dummies
        trading_partners = TRADING_PARTNERS.get(zone, [])
        for partner in trading_partners:
            bneck_col = f'BNECK_{zone}_{partner}'
            if bneck_col in data.columns:
                export_data[bneck_col] = data[bneck_col]

    else:
        print("Exporting: Raw variables (no transformations)")

        export_data = pd.DataFrame({
            'Datetime': data.index,
            'Price': data['Price'],
            'Wind_Forecast': data['Wind_Forecast'],
            'Hydro_Reserves': data['Hydro_Reserves'],
            'Net_Exchange': data['Net_Exchange'],
            'Consumption': data['Consumption'],
            'Oil_Price': data['Oil_Price'],
            'Gas_Price': data['Gas_Price']
        })

        # Add bottleneck dummies
        trading_partners = TRADING_PARTNERS.get(zone, [])
        for partner in trading_partners:
            bneck_col = f'BNECK_{zone}_{partner}'
            if bneck_col in data.columns:
                export_data[bneck_col] = data[bneck_col]

    # Remove any remaining NaN values
    rows_before = len(export_data)
    export_data = export_data.dropna()
    rows_after = len(export_data)

    if rows_before > rows_after:
        print(f"\nDropped {rows_before - rows_after} rows with missing values")

    # Create export directory if needed
    os.makedirs(export_dir, exist_ok=True)

    # Save to CSV
    export_path = os.path.join(export_dir, f'regression_data_{zone}_for_R.csv')
    export_data.to_csv(export_path, index=False)

    print(f"\n[OK] Exported {len(export_data):,} observations")
    print(f"[OK] Variables: {list(export_data.columns)}")
    print(f"[OK] Date range: {export_data['Datetime'].min()} to {export_data['Datetime'].max()}")
    print(f"[OK] Saved to: {export_path}")

    # Also save a metadata file with configuration info
    metadata_path = os.path.join(export_dir, f'metadata_{zone}.txt')
    with open(metadata_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"DATA EXPORT FOR R ANALYSIS - {zone}\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Zone: {zone}\n")
        f.write(f"  Log transform: {use_log_transform}\n")
        f.write(f"  Deseasonalized: {use_deseasonalized}\n")
        f.write(f"  Outlier handling: {handle_outliers} ({outlier_method})\n")
        f.write(f"  Negative price handling: {negative_price_handling}\n")
        f.write(f"  Interpolation: {use_interpolation}\n\n")

        f.write("Regression specification for R:\n")
        f.write("  Dependent variable: Price\n")
        f.write("  Independent variables:\n")
        for col in export_data.columns:
            if col != 'Datetime' and col != 'Price':
                f.write(f"    - {col}\n")

        f.write(f"\nData characteristics:\n")
        f.write(f"  Observations: {len(export_data):,}\n")
        f.write(f"  Start date: {export_data['Datetime'].min()}\n")
        f.write(f"  End date: {export_data['Datetime'].max()}\n")
        f.write(f"  Frequency: Hourly\n\n")

        f.write("Suggested R code:\n")
        f.write("  library(strucchange)\n")
        f.write(f"  data <- read.csv('data_for_R/regression_data_{zone}_for_R.csv')\n")
        f.write("  data$Datetime <- as.POSIXct(data$Datetime)\n")
        f.write("  \n")
        f.write("  # Bai-Perron test\n")
        f.write("  formula <- Price ~ Wind_Forecast + Hydro_Reserves + Net_Exchange + \n")
        f.write("                     Consumption + Oil_Price + Gas_Price")

        # Add bottleneck dummies to formula if present
        if any(col.startswith('BNECK_') for col in export_data.columns):
            f.write(" + \n")
            bneck_cols = [col for col in export_data.columns if col.startswith('BNECK_')]
            f.write("                     " + " + ".join(bneck_cols))

        f.write("\n  bp <- breakpoints(formula, data=data, h=0.15)\n")
        f.write("  summary(bp)\n")
        f.write("  plot(bp)\n")

    print(f"[OK] Saved metadata to: {metadata_path}")
    print("="*80 + "\n")

    return export_path


# --- 6. EXECUTION BLOCK ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ACTIVE_ZONE = 'SE1'

    # --- ZONE COMPARISON PLOTS ---
    # When True: generates comparison plots overlaying SE1-SE4 for price, log price,
    # volatility (24h rolling std), and wind production share
    RUN_ZONE_COMPARISONS = False

    # --- VISUALIZATION TOGGLE ---
    # Toggle for data visualization and outlier detection
    # When True: generates comprehensive visualizations of raw data and outlier detection
    # When False: skips visualization and proceeds directly to regression analysis
    RUN_VISUALIZATIONS = False

    # --- TRANSFORMATION SETTINGS (ALWAYS APPLIED) ---
    # Log transformation and deseasonalization are ALWAYS applied (Fredriksson 2016 methodology)
    # - Log: applies log() to Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    # - Net_Exchange is NOT logged (can contain negative values)
    # - Deseasonalization: applied to LOGGED series using dummy variable regression
    # These transformations are hardcoded and cannot be toggled off.

    # --- NEGATIVE VALUE HANDLING ---
    # Method for handling negative price values (before log transformation)
    # 'clip': Replace values below 0.01 with 0.01 (current default, affects only negative/zero values)
    # 'shift': Shift entire price series upward so minimum becomes 0.01 (preserves relative differences)
    # Note: Net_Exchange is never modified (expected to have negative values)
    NEGATIVE_PRICE_HANDLING = 'shift'  # Options: 'clip' or 'shift'

    # --- OUTLIER HANDLING SETTINGS ---
    # Outlier handling is ALWAYS applied (cannot be toggled off)
    #
    # Outlier handling method selection
    # 'fredriksson': Fredriksson (2016) methodology
    #   - Threshold: +6σ / -3.7σ (asymmetric)
    #   - Replacement: Mean of 24h and 48h before/after outlier
    # 'gianfreda': Gianfreda (2010) / Mugele et al. (2005) methodology
    #   - Threshold: ±3σ (symmetric)
    #   - Replacement: Capped at ±3σ for respective weekday
    OUTLIER_METHOD = 'fredriksson'  # Options: 'fredriksson' or 'gianfreda'

    # METHODOLOGICAL NOTE:
    # Fredriksson (2016) applies outlier filter TWICE:
    #   1st: On original price series (found 31 outliers)
    #   2nd: On deseasonalized price series (found 42 outliers)
    #
    # OUR DEFAULT APPROACH: Apply outlier filter ONCE, on logged-deseasonalized series
    # Rationale:
    #   - Seasonal patterns mask true outliers (e.g., high winter prices vs low summer)
    #   - Log transformation stabilizes variance
    #   - Deseasonalized mean ≈ 0 makes threshold more meaningful
    #   - Cleaner single-pass approach with stronger statistical justification
    #   - Fredriksson provides no theoretical justification for double application
    #
    # ALTERNATIVE APPROACH (via HANDLE_OUTLIERS_BEFORE_LOG=True):
    #   Apply outlier filter on raw price series before log transformation
    #   - Suitable when outliers are data quality issues rather than market events
    #   - Prevents near-zero prices from creating excessive negative outliers in log space
    #
    # TODO: Future sensitivity analysis could compare single vs. double application

    # Toggle for linear interpolation of missing values
    # When True: fills missing values by linear interpolation between surrounding values
    # When False: drops all rows with missing values (original behavior)
    USE_LINEAR_INTERPOLATION = True

    # --- OUTLIER TIMING CONFIGURATION ---
    # Control WHEN outlier handling is applied in the transformation pipeline
    #
    # When True: Apply outlier detection/replacement BEFORE log transformation
    #   - Applied to raw Price series (after negative value handling)
    #   - Statistical rationale: Remove extreme values that could distort log transformation
    #   - Suitable when outliers are data quality issues (recording errors, system failures)
    #
    # When False: Apply outlier detection/replacement AFTER log + deseasonalization (DEFAULT)
    #   - Applied to logged-deseasonalized Price series (current behavior)
    #   - Statistical rationale: Remove outliers in transformed space with stabilized variance
    #   - Suitable when outliers are legitimate extreme market events
    #
    # Note: Cannot apply BOTH early and late outlier handling in single run
    HANDLE_OUTLIERS_BEFORE_LOG = False  # Default: False (preserves current behavior)

    # --- COMMODITY PRICE LAGGING ---
    # Commodity prices (oil & gas) are ALWAYS lagged by 24 hours (hardcoded in pipeline)
    # Rationale: Day-ahead electricity market uses commodity prices from bidding time (D-1)
    # This aligns with standard literature (Weron, Huisman, etc.)
    LAG_COMMODITY_HOURS = 24  # Applied automatically in load_data()

    # --- DIAGNOSTIC TEST TOGGLES (Fredriksson 2016 methodology) ---
    # Toggle for Ljung-Box test for autocorrelation
    # Tests whether residuals exhibit autocorrelation at various lag lengths
    RUN_LJUNGBOX_TEST = False

    # Toggle for heteroskedasticity and ARCH effects tests
    # Includes Engle's ARCH test and Ljung-Box Q test on squared residuals
    # If ARCH effects detected, consider implementing GARCHX model
    RUN_HETEROSKEDASTICITY_TESTS = False

    # Toggle for stationarity tests (ADF and DF-GLS)
    # Tests whether price series has a unit root (non-stationary)
    RUN_STATIONARITY_TESTS = False

    # --- MODEL SPECIFICATION TOGGLES ---
    # Toggle for automated ARMAX lag selection via AIC minimization
    # When True: Tests AR lags 1-10 and MA lags 1-10, selects optimal model
    # When False: Uses default ARMAX(1,1) specification
    # WARNING: This can take several minutes to run (tests 100 model combinations)
    OPTIMIZE_ARMAX_LAGS = False

    # Toggle for checkpointed lag selection with Ljung-Box diagnostics
    # When True: Uses checkpointed version that saves progress and includes diagnostics
    # When False: Uses original version (no checkpointing)
    # Only applies if OPTIMIZE_ARMAX_LAGS = True
    USE_CHECKPOINTED_LAG_SELECTION = False

    # --- ARMAX SEARCH GRID & SELECTION POLICY ---
    # Search ranges for p and q (inclusive)
    ARMAX_SEARCH_P_MIN = 0
    ARMAX_SEARCH_P_MAX = 3
    ARMAX_SEARCH_Q_MIN = 0
    ARMAX_SEARCH_Q_MAX = 3
    ARMAX_SEARCH_EXCLUDE_00 = True
    # Eligibility rules for model selection in grid search
    ARMAX_SEARCH_REQUIRE_CONVERGENCE = True
    ARMAX_SEARCH_REQUIRE_LJUNGBOX_PASS = True
    # Ranking criterion among eligible models: 'bic' (recommended) or 'aic'
    ARMAX_SEARCH_SELECTION_CRITERION = 'bic'
    # Number of top eligible models to save to results/armax_search_top_<ZONE>.csv
    ARMAX_SEARCH_SAVE_TOP_N = 20

    # --- ARMAX CONVERGENCE POLICY ---
    # When False (recommended): reject non-converged ARMAX fits and fail loudly
    # When True: allow using non-converged coefficients (can mimic OLS with zero AR/MA)
    ARMAX_ALLOW_NONCONVERGED = False
    # Optimizer controls
    ARMAX_MAXITER = 300
    ARMAX_SOLVER = 'statespace'
    ARMAX_USE_WARM_START = False
    # Scale exogenous variables to z-scores for ARMAX fitting (improves numerical conditioning)
    ARMAX_SCALE_EXOG = True
    # Fallback ladder if primary order does not converge
    ARMAX_ENABLE_FALLBACK_ORDERS = True
    ARMAX_FALLBACK_ORDERS = [(1, 0, 1), (2, 0, 2), (3, 0, 3)]
    # Save SE1 ARMAX(1,0,1) exogenous anchor vector (const + wind + controls) to results CSV
    SAVE_ARMAX_EXOG_ANCHOR = True
    # Baseline ARMAX spec (used when OPTIMIZE_ARMAX_LAGS=False):
    # - order: contiguous ARMA(p,q) component
    # - extra_ar_lags: sparse lagged dependent terms added to exogenous set for baseline run only
    ARMAX_BASELINE_SPEC = {
        'enabled': True,
        'order': (3, 0, 3),
        'extra_ar_lags': [],  # Example: [23, 24, 25]
        'drop_initial_nan': True,
        'label': 'ARMAX(3,0,3)'
    }

    # --- GARCH CONFIGURATION ---
    # Fit GARCH only if ARCH effects detected (p < 0.05)
    FIT_GARCH_IF_ARCH = True
    # GARCH order: (p, q) for GARCH(p,q)
    GARCH_ORDER = (1, 1)
    # Note: Variance equation uses Wind_Forecast_Log (hardcoded, following Fredriksson 2016)

    # --- TVP KALMAN FILTER TOGGLE ---
    # When True: estimates time-varying wind coefficient using state-space model
    # When False: runs standard OLS + ARMAX analysis
    RUN_TVP_WIND_KALMAN = False

    # --- ROLLING-WINDOW ESTIMATION TOGGLE ---
    # When True: estimates wind coefficient using overlapping rolling windows (skips OLS/ARMAX)
    # When False: runs standard full-sample analysis
    RUN_ROLLING_WINDOW = False

    # Rolling window configuration
    ROLLING_WINDOW_YEARS = 1          # Window size in years
    ROLLING_STEP_YEARS = 1            # Step size between windows in years
    ROLLING_MIN_OBS = 24 * 365 - 24 * 30        # ~3 years minus 1 month tolerance, 24 * 365 * 3 - 24 * 30

    # --- QUANTILE REGRESSION TOGGLE ---
    # When True: estimates wind coefficient across quantiles of price distribution (skips OLS/ARMAX)
    # When False: runs standard analysis
    RUN_QUANTILE_REGRESSION = False

    # --- STRUCTURAL BREAK ANALYSIS TOGGLE ---
    # When True: detects structural breaks in wind coefficient
    # When False: runs standard analysis
    # NOTE: Requires 'ruptures' package for level break analysis (pip install ruptures)
    RUN_STRUCTURAL_BREAK = False

    # Structural break TYPE:
    # 'level' - Tests for step changes in coefficient mean (Bai-Perron methodology)
    # 'trend' - Tests for changes in coefficient slope over time (segmented linear regression)
    STRUCTURAL_BREAK_TYPE = 'trend'  # 'level' or 'trend'
    # Break estimation model:
    # 'ols' - Current baseline rolling OLS coefficient estimation
    # 'dynamic_armax' - Rolling ARMAX(3,0,3) coefficient estimation
    STRUCTURAL_BREAK_ESTIMATION_MODEL = 'ols'  # 'ols' or 'dynamic_armax'

    # Structural break configuration
    STRUCTURAL_BREAK_MAX_BREAKS = 3           # Maximum number of breaks to test (for both 'level' and 'trend')
    STRUCTURAL_BREAK_TRIMMING = 0.1          # Fraction of data to trim from endpoints (0.15 = 15%)
    # Known event dates to test with Chow test (list of 'YYYY-MM-DD' strings) - only for 'level' type
    # Examples: Russia-Ukraine invasion, COVID lockdowns, policy changes
    STRUCTURAL_BREAK_KNOWN_DATES = None #['2022-02-24', '2020-03-11'] # Russia invades Ukraine, # WHO declares COVID-19 pandemic

    # Structural break rolling window configuration (independent from standalone rolling window)
    STRUCTURAL_BREAK_WINDOW_YEARS = 1       # Window size in years for coefficient estimation
    STRUCTURAL_BREAK_STEP_YEARS = 3/12      # Step size between windows in years (2 months)
    STRUCTURAL_BREAK_MIN_OBS = 24 * 365 - 24 * 30  # Minimum observations per window

    # --- TREND BREAK TEST METHOD (trend mode only) ---
    # 'legacy': current implementation (BIC + standard sequential F-tests)
    # 'bp_supf': Bai-Perron style sequential supF(l+1|l) with bootstrap/table inference
    # Default keeps existing behavior for immediate rollback safety.
    TREND_BREAK_TEST_METHOD = 'bp_supf'  # 'legacy' or 'bp_supf'

    # BP trend inference configuration (used when TREND_BREAK_TEST_METHOD='bp_supf')
    BP_INFERENCE_MODE = 'tables'           # 'both', 'tables', or 'bootstrap'
    BP_SIGNIFICANCE_LEVEL = 0.05         # One of: 0.10, 0.05, 0.025, 0.01
    BP_BOOTSTRAP_REPS = 999
    BP_BOOTSTRAP_BLOCK_LENGTH = 8
    BP_RANDOM_SEED = 42
    BP_USE_HAC_SE = True

    # --- R DATA EXPORT TOGGLE ---
    # When True: exports fully processed data to CSV for R's strucchange package (Bai-Perron)
    # When False: skips data export
    # Output: data_for_R/regression_data_{zone}_for_R.csv + metadata file
    EXPORT_DATA_FOR_R = False

    # Data file paths - dynamically set based on ACTIVE_ZONE
    PATHS = {
        'combined': f'master data files/2015-2025/Combined_{ACTIVE_ZONE}_Data_2015_2025.xlsx',
        'hydro': 'master data files/Master_Hydro_Reservoir.xlsx',
        'crude_oil': 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
        'commodities': 'master data files/Master_Commodities.xlsx'  # Still used for TTF Gas
    }

    # --- ZONE COMPARISON PLOTS (runs before zone-specific pipeline) ---
    if RUN_ZONE_COMPARISONS:
        plot_zone_comparisons(start_date='2015-01-01', end_date='2025-12-31', plots_dir='plots')

    try:
        # Load and clean full dataset (filtered to 2021-2024 for hydro/commodity availability)
        # Note: Commodity prices are automatically lagged within load_data()
        data = load_data(
            PATHS,
            target_region=ACTIVE_ZONE,
            zone_hydro=ACTIVE_ZONE,
            use_interpolation=USE_LINEAR_INTERPOLATION,
            start_date='2015-01-01',
            end_date='2026-12-31',
            lag_commodity_hours=LAG_COMMODITY_HOURS
        )
        print(f"Merge successful. Total hourly observations: {len(data)}")

        # --- ROLLING WINDOW MODE (WINDOW-LOCAL PREPROCESSING) ---
        # Run rolling-window analysis directly on raw data so each window is transformed locally.
        # This avoids full-sample transformation leakage into local coefficient estimates.
        if RUN_ROLLING_WINDOW:
            run_rolling_window_analysis(
                data,
                ACTIVE_ZONE,
                window_years=ROLLING_WINDOW_YEARS,
                step_years=ROLLING_STEP_YEARS,
                min_obs=ROLLING_MIN_OBS,
                plots_dir="plots",
                results_dir="results",
                use_window_local_preprocessing=True,
                target_region=ACTIVE_ZONE,
                negative_price_handling=NEGATIVE_PRICE_HANDLING,
                outlier_method=OUTLIER_METHOD,
                handle_outliers_before_log=HANDLE_OUTLIERS_BEFORE_LOG
            )
            raise SystemExit(0)

        # --- STEP 1: VISUALIZATION OF RAW DATA (if enabled) ---
        # Visualize pure, untouched raw data BEFORE any transformations
        # This shows the true state of the data including any negative values or quality issues
        if RUN_VISUALIZATIONS:
            run_visualizations(data, ACTIVE_ZONE, method=OUTLIER_METHOD, stage='raw')

        # --- STEP 2: CHECK NEGATIVE VALUES & HANDLE PRICE ---
        # Check all variables for negative values and handle Price if needed
        # Net_Exchange is NOT checked or modified (expected to have negative values)
        # NOTE: This happens AFTER raw visualization so we can see the original data quality issues
        data = handle_negative_prices(data, method=NEGATIVE_PRICE_HANDLING)

        # --- STEP 2.5: EARLY OUTLIER HANDLING (if configured) ---
        # Applies if HANDLE_OUTLIERS_BEFORE_LOG=True
        #
        # EARLY OUTLIER DETECTION RATIONALE:
        #   - Applied to RAW price series (after negative value handling)
        #   - Suitable when outliers are data quality issues (recording errors, sensor failures)
        #   - Prevents extreme values from distorting log transformation
        #   - Trade-off: Less statistically rigorous (non-stabilized variance, seasonal patterns present)
        if HANDLE_OUTLIERS_BEFORE_LOG:
            print("\n" + "="*80)
            print("STEP 2.5: EARLY OUTLIER HANDLING (BEFORE LOG TRANSFORMATION)")
            print("="*80)
            print("Applying outlier detection and replacement to raw Price series")
            print("This occurs AFTER negative value handling but BEFORE log transformation\n")

            if OUTLIER_METHOD == 'fredriksson':
                data, outlier_stats_early = handle_outliers_fredriksson(data, apply_to_raw=True)
            elif OUTLIER_METHOD == 'gianfreda':
                data, outlier_stats_early = handle_outliers_gianfreda(data, apply_to_raw=True)
            else:
                raise ValueError(f"Unknown outlier method: {OUTLIER_METHOD}. Choose 'fredriksson' or 'gianfreda'.")

        # --- STEP 3: LOG TRANSFORMATION (always applied) ---
        # STANDARD APPROACH: Apply log transformation FIRST, before deseasonalization
        # This is the standard econometric approach for handling multiplicative seasonality
        # Note: Commodity prices are already lagged at this point
        # Note: Price negative values handled in STEP 2 (before log transformation)
        data = apply_log_transform(data)

        # Visualize logged data (if enabled)
        # This shows data AFTER negative handling and log transformation
        if RUN_VISUALIZATIONS:
            # Create temporary dataframe with logged variables mapped to base names for visualization
            data_logged_viz = data.copy()
            data_logged_viz['Price'] = data_logged_viz['Price_Log']
            data_logged_viz['Wind_Forecast'] = data_logged_viz['Wind_Forecast_Log']
            data_logged_viz['Hydro_Reserves'] = data_logged_viz['Hydro_Reserves_Log']
            data_logged_viz['Consumption'] = data_logged_viz['Consumption_Log']
            data_logged_viz['Oil_Price'] = data_logged_viz['Oil_Price_Log']
            # Net_Exchange stays the same (not logged)
            run_visualizations(data_logged_viz, ACTIVE_ZONE, method=OUTLIER_METHOD, stage='logged')

        # --- STEP 4: DESEASONALIZATION (always applied) ---
        # STANDARD APPROACH: Deseasonalize the LOGGED variables (after log transformation)
        # Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL deseasonalization)
        # Hydro, Oil, Gas: Year + Month ONLY (PARTIAL - no intraday patterns)
        data = deseasonalize_logged_variables(data)

        # --- STEP 5: LATE OUTLIER HANDLING (if not applied early) ---
        # Applies if HANDLE_OUTLIERS_BEFORE_LOG=False
        #
        # LATE OUTLIER DETECTION RATIONALE (RECOMMENDED APPROACH):
        #   - Applied to LOGGED-DESEASONALIZED series
        #   - More statistically rigorous:
        #       * Log transformation stabilizes variance
        #       * Deseasonalization removes seasonal patterns (high winter vs low summer)
        #       * Threshold more meaningful on zero-centered deseasonalized data
        #   - Suitable when outliers are extreme market events (not recording errors)
        #
        # MUTUALLY EXCLUSIVE WITH STEP 2.5:
        #   - Cannot apply both early and late outlier handling in same run
        #   - Prevents double-replacement of outliers
        if not HANDLE_OUTLIERS_BEFORE_LOG:
            print("\n" + "="*80)
            print("STEP 5: LATE OUTLIER HANDLING (AFTER TRANSFORMATIONS)")
            print("="*80)
            print("Applying outlier detection and replacement to transformed Price series")
            print("This occurs AFTER log transformation and deseasonalization\n")

            if OUTLIER_METHOD == 'fredriksson':
                data, outlier_stats = handle_outliers_fredriksson(data, apply_to_raw=False)
            elif OUTLIER_METHOD == 'gianfreda':
                data, outlier_stats = handle_outliers_gianfreda(data, apply_to_raw=False)
            else:
                raise ValueError(f"Unknown outlier method: {OUTLIER_METHOD}. Choose 'fredriksson' or 'gianfreda'.")

        # --- EXPORT DATA FOR R ANALYSIS (BAI-PERRON) ---
        # Export fully processed data for structural break testing in R
        # Note: Always exports logged and deseasonalized data (standard pipeline)
        if EXPORT_DATA_FOR_R:
            export_data_for_R(
                data=data,
                zone=ACTIVE_ZONE,
                use_log_transform=True,  # Always applied in standard pipeline
                use_deseasonalized=True,  # Always applied in standard pipeline
                handle_outliers=True,     # Always applied in standard pipeline
                outlier_method=OUTLIER_METHOD,
                negative_price_handling=NEGATIVE_PRICE_HANDLING,
                use_interpolation=USE_LINEAR_INTERPOLATION
            )

        # --- STEP 6: REGRESSION ANALYSIS ---
        # Run regression models with optional diagnostic tests
        # Commodity prices used in regression are lagged by 24h (from load_data)
        ols_model, armax_res, garch_res = perform_multivariate_analysis(data, ACTIVE_ZONE,
                                      target_region=ACTIVE_ZONE,
                                      run_ljungbox=RUN_LJUNGBOX_TEST,
                                      run_hetero_tests=RUN_HETEROSKEDASTICITY_TESTS,
                                      run_stationarity=RUN_STATIONARITY_TESTS,
                                      optimize_armax_lags=OPTIMIZE_ARMAX_LAGS,
                                      use_checkpointed_lag_selection=USE_CHECKPOINTED_LAG_SELECTION,
                                      armax_search_p_min=ARMAX_SEARCH_P_MIN,
                                      armax_search_p_max=ARMAX_SEARCH_P_MAX,
                                      armax_search_q_min=ARMAX_SEARCH_Q_MIN,
                                      armax_search_q_max=ARMAX_SEARCH_Q_MAX,
                                      armax_search_exclude_00=ARMAX_SEARCH_EXCLUDE_00,
                                      armax_search_require_convergence=ARMAX_SEARCH_REQUIRE_CONVERGENCE,
                                      armax_search_require_ljungbox_pass=ARMAX_SEARCH_REQUIRE_LJUNGBOX_PASS,
                                      armax_search_selection_criterion=ARMAX_SEARCH_SELECTION_CRITERION,
                                      armax_search_save_top_n=ARMAX_SEARCH_SAVE_TOP_N,
                                      run_tvp_wind_kalman=RUN_TVP_WIND_KALMAN,
                                      run_rolling_window=RUN_ROLLING_WINDOW,
                                      rolling_window_years=ROLLING_WINDOW_YEARS,
                                      rolling_step_years=ROLLING_STEP_YEARS,
                                      rolling_min_obs=ROLLING_MIN_OBS,
                                      run_quantile_regression=RUN_QUANTILE_REGRESSION,
                                      run_structural_break=RUN_STRUCTURAL_BREAK,
                                      structural_break_type=STRUCTURAL_BREAK_TYPE,
                                      structural_break_max_breaks=STRUCTURAL_BREAK_MAX_BREAKS,
                                      structural_break_trimming=STRUCTURAL_BREAK_TRIMMING,
                                      structural_break_known_dates=STRUCTURAL_BREAK_KNOWN_DATES,
                                      structural_break_window_years=STRUCTURAL_BREAK_WINDOW_YEARS,
                                      structural_break_step_years=STRUCTURAL_BREAK_STEP_YEARS,
                                      structural_break_min_obs=STRUCTURAL_BREAK_MIN_OBS,
                                      trend_break_test_method=TREND_BREAK_TEST_METHOD,
                                      bp_inference_mode=BP_INFERENCE_MODE,
                                      bp_significance_level=BP_SIGNIFICANCE_LEVEL,
                                      bp_bootstrap_reps=BP_BOOTSTRAP_REPS,
                                      bp_bootstrap_block_length=BP_BOOTSTRAP_BLOCK_LENGTH,
                                      bp_random_seed=BP_RANDOM_SEED,
                                      bp_use_hac_se=BP_USE_HAC_SE,
                                      structural_break_estimation_model=STRUCTURAL_BREAK_ESTIMATION_MODEL,
                                      armax_baseline_spec=ARMAX_BASELINE_SPEC,
                                      save_armax_exog_anchor=SAVE_ARMAX_EXOG_ANCHOR,
                                      armax_scale_exog=ARMAX_SCALE_EXOG)

    except Exception as e:
        print(f"Critical error during execution: {e}")
