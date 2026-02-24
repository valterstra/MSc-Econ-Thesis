"""
################################################################################
#  [Module 03/10]  preprocessing.py  –  Data Transformation Pipeline
#
#  Contains (in pipeline order):
#    1. handle_negative_prices()           : clip or shift Price below 0.01
#    2. handle_outliers_fredriksson()      : asymmetric ±6σ / -3.7σ thresholds
#    3. handle_outliers_gianfreda()        : symmetric ±3σ weekday-specific caps
#    4. deseasonalize_logged_variables()   : dummy-variable regression on logs
#    5. apply_log_transform()              : log() on all positive variables
#    6. preprocess_data_for_regression()   : orchestrates steps 1-5 in sequence
#
#  Pipeline order: handle_negative_prices → apply_log_transform
#                  → deseasonalize_logged_variables → handle_outliers_*
#
#  Dependencies: none (self-contained)
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings
import io
import contextlib
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import holidays

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


