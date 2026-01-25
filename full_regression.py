import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (save plots only, no display)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# --- 1. DATA LOADING FUNCTIONS ---

def load_commodities(path):
    """
    Load commodity prices (Brent crude oil and TTF natural gas) from Bloomberg export file.

    The Master_Commodities.xlsx file has Bloomberg-style headers:
    - Rows 0-5: Header metadata (skip)
    - Row 6+: Data with columns: Date, TTF_Gas, WTI_Oil, Brent_Oil, MT1, LUA1, CP1

    Returns DataFrame with columns: Date, Oil_Price (Brent), Gas_Price (TTF)
    Daily frequency - will be merged with hourly data by date.
    """
    print("\n--- LOADING COMMODITY PRICES (Brent Oil & TTF Gas) ---")

    # Read with proper header skipping (Bloomberg format)
    df = pd.read_excel(path, header=None, skiprows=5)

    # Assign column names based on Bloomberg structure
    # Column 0: Dates, Column 1: TTF Gas, Column 3: Brent Oil
    df.columns = ['Date', 'TTF_Gas', 'WTI_Oil', 'Brent_Oil', 'MT1', 'LUA1', 'CP1']

    # Convert date column and filter valid dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Convert price columns to numeric
    df['Brent_Oil'] = pd.to_numeric(df['Brent_Oil'], errors='coerce')
    df['TTF_Gas'] = pd.to_numeric(df['TTF_Gas'], errors='coerce')

    # Select and rename columns we need
    df_commodities = df[['Date', 'Brent_Oil', 'TTF_Gas']].copy()
    df_commodities.columns = ['Date', 'Oil_Price', 'Gas_Price']

    # Filter to 2021-2024 range (matching electricity data)
    df_commodities = df_commodities[
        (df_commodities['Date'] >= '2021-01-01') &
        (df_commodities['Date'] <= '2024-12-31')
    ]

    print(f"  Loaded {len(df_commodities)} daily commodity observations")
    print(f"  Date range: {df_commodities['Date'].min().date()} to {df_commodities['Date'].max().date()}")
    print(f"  Oil Price (Brent): mean={df_commodities['Oil_Price'].mean():.2f}, "
          f"min={df_commodities['Oil_Price'].min():.2f}, max={df_commodities['Oil_Price'].max():.2f}")
    print(f"  Gas Price (TTF): mean={df_commodities['Gas_Price'].mean():.2f}, "
          f"min={df_commodities['Gas_Price'].min():.2f}, max={df_commodities['Gas_Price'].max():.2f}")

    # Report missing values (handled later in load_all_thesis_data with other variables)
    oil_missing = df_commodities['Oil_Price'].isna().sum()
    gas_missing = df_commodities['Gas_Price'].isna().sum()
    if oil_missing > 0 or gas_missing > 0:
        print(f"  Missing values - Oil: {oil_missing}, Gas: {gas_missing}")
        print(f"  (Will be handled with other variables via interpolation or row dropping)")

    return df_commodities


def load_all_thesis_data(paths, zone_price='SE1', zone_wind='S1', zone_hydro='SE1', zone_exch='SE1', zone_cons='SE1', use_interpolation=False):
    """Loads and merges all master files based on standardized Timestamps."""

    # Load individual master files
    df_price = pd.read_excel(paths['price'])
    df_wind = pd.read_excel(paths['wind'])
    df_hydro = pd.read_excel(paths['hydro'])
    df_exch = pd.read_excel(paths['exch'])
    df_cons = pd.read_excel(paths['cons'])

    # Sequential merge on standardized Timestamp
    merged = pd.merge(df_price, df_wind, on='Timestamp')
    merged = pd.merge(merged, df_hydro, on='Timestamp')
    merged = pd.merge(merged, df_exch, on='Timestamp')
    merged = pd.merge(merged, df_cons, on='Timestamp')

    # Select specific zone columns for analysis
    # Note: Column names follow your master file headers exactly
    final_df = pd.DataFrame({
        'Datetime': pd.to_datetime(merged['Timestamp']),
        'Price': pd.to_numeric(merged[f'{zone_price}_Price (EUR)'], errors='coerce'),
        'Wind_Forecast': pd.to_numeric(merged[f'{zone_wind}_Wind'], errors='coerce'),
        'Hydro_Reserves': pd.to_numeric(merged[f'{zone_hydro}_Hydro_Reserves'], errors='coerce'),
        'Net_Exchange': pd.to_numeric(merged[f'{zone_exch}_Total_Net_Exchange'], errors='coerce'),
        'Consumption': pd.to_numeric(merged[zone_cons], errors='coerce')
    })

    # Load and merge commodity prices (daily -> hourly) if path provided
    if 'commodities' in paths:
        df_commodities = load_commodities(paths['commodities'])

        # Create date column for merging (extract date from hourly Datetime)
        final_df['Date'] = final_df['Datetime'].dt.date
        df_commodities['Date'] = df_commodities['Date'].dt.date

        # Merge commodities on date (each hour gets the daily commodity price)
        final_df = pd.merge(final_df, df_commodities, on='Date', how='left')

        # Drop the temporary Date column
        final_df = final_df.drop(columns=['Date'])

        print(f"  Merged commodity prices: Oil (USD/barrel), Gas (EUR/MWh)")

    # Handle missing values based on configuration
    if use_interpolation:
        print("\n--- APPLYING LINEAR INTERPOLATION FOR MISSING VALUES ---")

        # Show detailed breakdown of missing values by variable
        missing_by_var = final_df.isna().sum()
        missing_before = missing_by_var.sum()

        print(f"\nMissing values by variable (before interpolation):")
        for var, count in missing_by_var.items():
            if count > 0:
                pct = (count / len(final_df)) * 100
                print(f"  {var}: {count} ({pct:.2f}%)")

        print(f"\nTotal missing values: {missing_before}")

        # Apply linear interpolation to fill missing values
        final_df = final_df.interpolate(method='linear', limit_direction='both')
        # Drop any remaining NaN values (e.g., at edges if interpolation couldn't fill)
        final_df = final_df.dropna()
        missing_after = final_df.isna().sum().sum()
        print(f"Missing values after interpolation: {missing_after}")
        print(f"Rows retained: {len(final_df)}")
    else:
        # Original behavior: drop all rows with missing values
        rows_before = len(final_df)

        # Show detailed breakdown of missing values by variable
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
            print(f"\nDropped {rows_dropped} rows with missing values (original behavior)")

    return final_df.set_index('Datetime')


# --- 2. DATA PREPROCESSING FUNCTIONS ---

def lag_commodity_prices(df, lag_hours=24):
    """
    Lag commodity prices (oil and gas) by specified hours (default: 24 hours).

    Rationale:
    - Electricity spot prices are determined in day-ahead auctions (day D-1 for delivery on day D)
    - Oil and gas prices should reflect information available at the time of bidding (24h before delivery)
    - This aligns commodity prices with the information set used when electricity prices were set

    Following standard practice in electricity price modeling literature (Weron, Huisman, etc.)

    Parameters:
    - df: DataFrame with Oil_Price and/or Gas_Price columns
    - lag_hours: Number of hours to lag (default 24 for day-ahead market)

    Returns:
    - DataFrame with lagged commodity prices (NaN values from lagging are dropped)
    """
    print(f"\n--- LAGGING COMMODITY PRICES BY {lag_hours} HOURS ---")
    print("Rationale: Day-ahead market pricing uses commodity prices from bidding time (D-1)")

    rows_before_lag = len(df)

    if 'Oil_Price' in df.columns:
        oil_before = df['Oil_Price'].notna().sum()
        df['Oil_Price'] = df['Oil_Price'].shift(lag_hours)
        oil_after = df['Oil_Price'].notna().sum()
        oil_lost = oil_before - oil_after
        print(f"  Oil_Price: Lagged by {lag_hours}h ({oil_lost} observations lost at start)")

    if 'Gas_Price' in df.columns:
        gas_before = df['Gas_Price'].notna().sum()
        df['Gas_Price'] = df['Gas_Price'].shift(lag_hours)
        gas_after = df['Gas_Price'].notna().sum()
        gas_lost = gas_before - gas_after
        print(f"  Gas_Price: Lagged by {lag_hours}h ({gas_lost} observations lost at start)")

    # Drop rows with NaN values created by lagging
    df = df.dropna()
    rows_after_drop = len(df)
    rows_dropped = rows_before_lag - rows_after_drop

    print(f"\nDropped {rows_dropped} rows with NaN values created by lagging")
    print(f"Rows retained: {rows_after_drop}")
    print("All subsequent transformations (log, deseasonalization) will use lagged commodity prices")

    return df


def handle_outliers_fredriksson(df, use_log_transform=False, use_deseasonalized=False):
    """
    Replace outliers using Fredriksson (2016) methodology.

    Outlier definition:
    - Exceeds 6x standard deviation above the mean, OR
    - Lower than 3.7x standard deviation below the mean

    Replacement method:
    - Replace outlier with mean of 24 and 48 hours before and after the outlier
    - Only applied to PRICE series, not explanatory variables

    Parameters:
    - use_log_transform: If True, works on logged price
    - use_deseasonalized: If True, works on deseasonalized logged price

    Returns:
    - DataFrame with outliers replaced in Price
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - FREDRIKSSON (2016) METHODOLOGY")
    print("="*80)

    # Determine which Price column to use based on flags
    if use_log_transform and use_deseasonalized:
        if 'Price_Log_Deseasonalized' in df.columns:
            price_col = 'Price_Log_Deseasonalized'
            print("Applying to: Logged and Deseasonalized Price")
        else:
            print("Warning: Price_Log_Deseasonalized not found. Cannot apply outlier handling.")
            return df, {}
    elif use_log_transform:
        if 'Price_Log' in df.columns:
            price_col = 'Price_Log'
            print("Applying to: Logged Price")
        else:
            print("Warning: Price_Log not found. Cannot apply outlier handling.")
            return df, {}
    else:
        price_col = 'Price'
        print("Applying to: Raw Price")

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
        outlier_indices = data[outliers].index

        for idx in outlier_indices:
            # Get position in the series
            pos = df_clean.index.get_loc(idx)

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


def handle_outliers_gianfreda(df, use_log_transform=False, use_deseasonalized=False):
    """
    Replace outliers using Gianfreda (2010) / Mugele et al. (2005) methodology.

    Outlier definition:
    - Exceeds 3x standard deviation above or below the mean (symmetric threshold)

    Replacement method:
    - Replace outlier with 3*std value for the respective weekday
    - Each weekday has its own 3σ threshold (Monday outliers capped at Monday's 3σ, etc.)

    Parameters:
    - use_log_transform: If True, works on logged price
    - use_deseasonalized: If True, works on deseasonalized logged price

    Returns:
    - DataFrame with outliers replaced in Price
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - GIANFREDA (2010) / MUGELE ET AL. (2005) METHODOLOGY")
    print("="*80)

    # Determine which Price column to use based on flags
    if use_log_transform and use_deseasonalized:
        if 'Price_Log_Deseasonalized' in df.columns:
            price_col = 'Price_Log_Deseasonalized'
            print("Applying to: Logged and Deseasonalized Price")
        else:
            print("Warning: Price_Log_Deseasonalized not found. Cannot apply outlier handling.")
            return df, {}
    elif use_log_transform:
        if 'Price_Log' in df.columns:
            price_col = 'Price_Log'
            print("Applying to: Logged Price")
        else:
            print("Warning: Price_Log not found. Cannot apply outlier handling.")
            return df, {}
    else:
        price_col = 'Price'
        print("Applying to: Raw Price")

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


def deseasonalize_logged_variables(df):
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
    # Following Fredriksson (2016), we include major Swedish holidays
    try:
        import holidays
        swedish_holidays = holidays.Sweden(years=range(df.index.year.min(), df.index.year.max() + 1))
        df['Holiday'] = df.index.to_series().apply(lambda x: 1 if x.date() in swedish_holidays else 0).values
        print("\nHoliday dummies created using Swedish holiday calendar")
    except ImportError:
        # Fallback: simple holiday definition if holidays package not available
        print("\nWarning: 'holidays' package not found. Using basic holiday definition.")
        print("Install with: pip install holidays")
        # Define basic holidays manually (New Year, Christmas, Midsummer, etc.)
        df['Holiday'] = 0
        for date in df.index:
            # New Year's Day
            if (date.month == 1 and date.day == 1) or \
               (date.month == 12 and date.day in [24, 25, 26, 31]) or \
               (date.month == 6 and date.day in [19, 20, 21, 22, 23, 24, 25, 26]) or \
               (date.month == 5 and date.day == 1):  # Labor Day, Midsummer, Christmas
                df.loc[date, 'Holiday'] = 1

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
    if 'Oil_Price_Log' in df.columns:
        oil_log_model = sm.OLS(df['Oil_Price_Log'], seasonal_dummies_partial).fit()
        df['Oil_Price_Log_Deseasonalized'] = oil_log_model.resid
        print(f"Oil_Price_Log: Seasonal R² = {oil_log_model.rsquared:.4f}")
        print(f"  Original std: {df['Oil_Price_Log'].std():.4f}, Deseasonalized std: {df['Oil_Price_Log_Deseasonalized'].std():.4f}")

    # Deseasonalize Gas_Price_Log (PARTIAL - Year + Month only)
    if 'Gas_Price_Log' in df.columns:
        gas_log_model = sm.OLS(df['Gas_Price_Log'], seasonal_dummies_partial).fit()
        df['Gas_Price_Log_Deseasonalized'] = gas_log_model.resid
        print(f"Gas_Price_Log: Seasonal R² = {gas_log_model.rsquared:.4f}")
        print(f"  Original std: {df['Gas_Price_Log'].std():.4f}, Deseasonalized std: {df['Gas_Price_Log_Deseasonalized'].std():.4f}")

    # Clean up temporary columns
    df = df.drop(columns=['Year', 'Month', 'DayOfWeek', 'Hour', 'Holiday'])

    print("\nNote: Wind_Forecast_Log and Net_Exchange are NOT deseasonalized (following Fredriksson)")

    return df


def apply_log_transform(df):
    """
    Apply logarithmic transformation to variables following Fredriksson (2016).
    STANDARD APPROACH: Log transformation is applied FIRST, then deseasonalization.

    Logs applied to: Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    NOT logged: Net_Exchange (can contain negative values)

    Note: Oil_Price and Gas_Price are already lagged by 24h (from lag_commodity_prices step)

    Returns df with logged columns: Price_Log, Wind_Forecast_Log, Hydro_Reserves_Log,
    Consumption_Log, Oil_Price_Log, Gas_Price_Log
    """
    print("\n--- APPLYING LOGARITHMIC TRANSFORMATION (STANDARD APPROACH) ---")
    print("Log transformation applied BEFORE deseasonalization")
    print("Note: Oil & Gas prices are already lagged by 24h (day-ahead market alignment)")

    # Log Price
    df['Price_Log'] = np.log(df['Price'].clip(lower=0.01))
    print(f"Price: log(raw) applied")

    # Log Wind Forecast
    df['Wind_Forecast_Log'] = np.log(df['Wind_Forecast'].clip(lower=0.01))
    print(f"Wind_Forecast: log(raw) applied")

    # Log Hydro Reserves
    df['Hydro_Reserves_Log'] = np.log(df['Hydro_Reserves'].clip(lower=0.01))
    print(f"Hydro_Reserves: log(raw) applied")

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

    # Log Oil Price (if available)
    if 'Oil_Price' in df.columns:
        df['Oil_Price_Log'] = np.log(df['Oil_Price'].clip(lower=0.01))
        print(f"Oil_Price: log(raw) applied [USD/barrel]")

    # Log Gas Price (if available)
    if 'Gas_Price' in df.columns:
        df['Gas_Price_Log'] = np.log(df['Gas_Price'].clip(lower=0.01))
        print(f"Gas_Price: log(raw) applied [EUR/MWh]")

    return df


# --- 3. DIAGNOSTIC TEST FUNCTIONS ---

def run_ljungbox_test(residuals, lags=[5, 10, 15, 20]):
    """
    Ljung-Box test for autocorrelation in residuals.

    Tests the null hypothesis that residuals are independently distributed (no autocorrelation).
    Low p-values (< 0.05) indicate significant autocorrelation.

    Following Fredriksson (2016), tests at multiple lag lengths.
    """
    print("\n--- LJUNG-BOX TEST FOR AUTOCORRELATION ---")
    print("H0: Residuals are independently distributed (no autocorrelation)")
    print("Reject H0 if p-value < 0.05\n")

    # Run test at multiple lags
    lb_results = acorr_ljungbox(residuals, lags=lags, return_df=True)

    print(f"{'Lag':<10} {'Test Statistic':<20} {'P-value':<15} {'Result'}")
    print("-" * 60)

    for lag in lags:
        if lag in lb_results.index:
            stat = lb_results.loc[lag, 'lb_stat']
            pval = lb_results.loc[lag, 'lb_pvalue']
            result = "REJECT H0 (autocorr present)" if pval < 0.05 else "Fail to reject H0"
            print(f"{lag:<10} {stat:<20.4f} {pval:<15.4f} {result}")


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

def run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, use_log_transform=True, plots_dir="plots"):
    """
    Estimate time-varying parameter (TVP) model for the wind coefficient using state-space Kalman filter.

    Model specification:
        Observation: y_t = beta_t * w_t + controls_t' * gamma + e_t
        State:       beta_t = beta_{t-1} + u_t

    Uses Frisch-Waugh-Lovell partialling out to control for other regressors,
    then estimates a random-walk state-space model for the wind coefficient.

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
    - use_log_transform: If True, uses Wind_Forecast_Log; otherwise Wind_Forecast
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

    # Step 1: Identify wind column based on mode
    if use_log_transform:
        wind_col = 'Wind_Forecast_Log'
    else:
        wind_col = 'Wind_Forecast'

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

    # Save plot
    plot_path = os.path.join(plots_dir, f'tvp_beta_wind_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("TVP KALMAN FILTER ANALYSIS COMPLETE")
    print("="*80)

    return beta_t, se_t, tvp_results


def run_rolling_window_analysis(df, zone, Y, exog_vars, use_log_transform,
                                window_years=3, step_years=1, min_obs=24*180,
                                plots_dir="plots", results_dir="results"):
    """
    Estimate wind coefficient using overlapping rolling windows with OLS.

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - Y: Dependent variable (Series)
    - exog_vars: List of exogenous variable column names
    - use_log_transform: Whether log transformation was applied
    - window_years: Size of each rolling window in years
    - step_years: Step size between windows in years
    - min_obs: Minimum observations required per window
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns: None (saves outputs to files)
    """
    from dateutil.relativedelta import relativedelta

    print("\n" + "="*80)
    print("ROLLING-WINDOW WIND COEFFICIENT ESTIMATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Window size: {window_years} years")
    print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")

    # Identify wind column from exog_vars
    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]
    control_cols = [col for col in exog_vars if col != wind_col]

    print(f"\nTarget variable: {wind_col}")
    print(f"Control variables: {control_cols}")

    # Prepare clean data
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy()
    tmp = tmp.sort_index()

    print(f"\nData range: {tmp.index.min()} to {tmp.index.max()}")
    print(f"Total observations after cleaning: {len(tmp):,}")

    # Define rolling windows by calendar time
    results = []
    start_date = tmp.index.min()
    end_of_data = tmp.index.max()

    window_count = 0
    print(f"\n--- Estimating Rolling Windows ---")

    while True:
        window_end = start_date + relativedelta(years=window_years)
        if window_end > end_of_data:
            break

        window_data = tmp[(tmp.index >= start_date) & (tmp.index < window_end)]

        if len(window_data) >= min_obs:
            window_count += 1

            # Run OLS regression with Newey-West (HAC) standard errors
            X = sm.add_constant(window_data[exog_vars])
            y = window_data[Y.name]
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})

            # Calculate window midpoint
            window_midpoint = start_date + relativedelta(months=window_years * 6)

            # Record results
            results.append({
                'window_start': start_date,
                'window_end': window_end,
                'window_midpoint': window_midpoint,
                'beta_wind': model.params[wind_col],
                'se_wind': model.bse[wind_col],
                't_stat': model.tvalues[wind_col],
                'pvalue': model.pvalues[wind_col],
                'n_obs': len(window_data),
                'r_squared': model.rsquared
            })

            print(f"  Window {window_count}: {start_date.strftime('%Y-%m-%d')} to "
                  f"{window_end.strftime('%Y-%m-%d')} | n={len(window_data):,} | "
                  f"beta={model.params[wind_col]:.6f} | p={model.pvalues[wind_col]:.4f}")

        start_date = start_date + relativedelta(years=step_years)

    if not results:
        print("\nWARNING: No valid windows found. Check data range and window parameters.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*80)
    print("ROLLING-WINDOW SUMMARY STATISTICS")
    print("="*80)
    print(f"\nNumber of windows analyzed: {len(results_df)}")
    print(f"\nWind coefficient (beta):")
    print(f"  Mean:   {results_df['beta_wind'].mean():.6f}")
    print(f"  Std:    {results_df['beta_wind'].std():.6f}")
    print(f"  Min:    {results_df['beta_wind'].min():.6f}")
    print(f"  Max:    {results_df['beta_wind'].max():.6f}")

    sig_count = (results_df['pvalue'] < 0.05).sum()
    print(f"\nSignificance at 5% level: {sig_count}/{len(results_df)} windows "
          f"({100*sig_count/len(results_df):.1f}%)")

    # Create results directory if needed
    os.makedirs(results_dir, exist_ok=True)

    # Save CSV output
    csv_path = os.path.join(results_dir, f'rolling_wind_coef_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # Create and save plot
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert midpoints to datetime for plotting
    midpoints = pd.to_datetime(results_df['window_midpoint'])
    beta_values = results_df['beta_wind'].values
    se_values = results_df['se_wind'].values

    # Calculate 95% confidence intervals
    upper_95 = beta_values + 1.96 * se_values
    lower_95 = beta_values - 1.96 * se_values

    # Plot coefficient with confidence bands
    ax.plot(midpoints, beta_values, color='blue', linewidth=2, marker='o',
            markersize=6, label=r'$\beta_{wind}$ coefficient')
    ax.fill_between(midpoints, lower_95, upper_95, color='blue', alpha=0.2,
                    label='95% CI')

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
               label='Zero')

    # Add horizontal line at mean
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

    # Add annotation with key statistics
    stats_text = (f'Windows: {len(results_df)}\n'
                  f'Mean: {mean_beta:.4f}\n'
                  f'Std: {results_df["beta_wind"].std():.4f}\n'
                  f'Sig (p<0.05): {sig_count}/{len(results_df)}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plots_dir, f'rolling_wind_coef_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("ROLLING-WINDOW ANALYSIS COMPLETE")
    print("="*80)


def run_quantile_regression_analysis(df, zone, use_log_transform,
                                     plots_dir="plots", results_dir="results"):
    """
    Estimate wind coefficient across quantiles of the price distribution.

    Uses raw/log variables (NOT deseasonalized) with calendar dummies included directly
    in the regression to control for seasonality (FULL basis: Year+Month+DOW+Hour+Holiday).

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - use_log_transform: Whether log transformation was applied
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns: None (saves outputs to files)
    """
    # Hardcoded quantiles and seasonality settings (not user-configurable)
    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    print("\n" + "="*80)
    print("QUANTILE REGRESSION ANALYSIS")
    print("="*80)

    # --- Step 1: Determine dependent variable (raw/log, NOT deseasonalized) ---
    if use_log_transform and 'Price_Log' in df.columns:
        y_col = 'Price_Log'
    else:
        y_col = 'Price'

    print(f"\nDependent variable: {y_col}")
    print("  (Using raw/log price, NOT deseasonalized - seasonality handled via dummies)")

    # --- Step 2: Determine economic regressors (raw/log, NOT deseasonalized) ---
    econ_vars = []

    # Wind (required)
    if use_log_transform and 'Wind_Forecast_Log' in df.columns:
        wind_col = 'Wind_Forecast_Log'
    else:
        wind_col = 'Wind_Forecast'
    econ_vars.append(wind_col)

    # Consumption/Demand (required)
    if use_log_transform and 'Consumption_Log' in df.columns:
        demand_col = 'Consumption_Log'
    else:
        demand_col = 'Consumption'
    econ_vars.append(demand_col)

    # Hydro reserves (required)
    if use_log_transform and 'Hydro_Reserves_Log' in df.columns:
        hydro_col = 'Hydro_Reserves_Log'
    else:
        hydro_col = 'Hydro_Reserves'
    econ_vars.append(hydro_col)

    # Net exchange (optional)
    if 'Net_Exchange' in df.columns:
        econ_vars.append('Net_Exchange')
    else:
        print("  Note: 'Net_Exchange' not found, skipping")

    # Oil price (optional)
    oil_col = None
    if use_log_transform and 'Oil_Price_Log' in df.columns:
        oil_col = 'Oil_Price_Log'
        econ_vars.append(oil_col)
    elif 'Oil_Price' in df.columns:
        oil_col = 'Oil_Price'
        econ_vars.append(oil_col)
    else:
        print("  Note: Oil price not found, skipping")

    # Gas price (optional)
    gas_col = None
    if use_log_transform and 'Gas_Price_Log' in df.columns:
        gas_col = 'Gas_Price_Log'
        econ_vars.append(gas_col)
    elif 'Gas_Price' in df.columns:
        gas_col = 'Gas_Price'
        econ_vars.append(gas_col)
    else:
        print("  Note: Gas price not found, skipping")

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
    for q in QUANTILES:
        print(f"  Estimating q={q:.2f}...", end='')

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
        print(f" beta_wind={result_row['beta_wind']:.6f}, p={result_row['p_wind']:.4f}")

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


def select_armax_lags_aic(Y, exog_vars, max_p=10, max_q=10):
    """
    Automated lag selection for ARMAX model using AIC minimization.

    Tests all combinations of AR lags (p) and MA lags (q) from 1 to max values.
    Returns the (p, q) combination with the lowest AIC.

    Following Fredriksson (2016) methodology for optimal lag selection.
    """
    print("\n--- ARMAX LAG SELECTION VIA AIC MINIMIZATION ---")
    print(f"Testing AR lags (p): 1-{max_p}, MA lags (q): 1-{max_q}")
    print("This may take several minutes...\n")

    best_aic = np.inf
    best_order = None
    results_table = []

    import warnings

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                print(f"  Testing ARMAX({p},{q})...", end='')

                # Suppress convergence warnings for cleaner output
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = sm.tsa.ARIMA(Y, exog=exog_vars, order=(p, 0, q))
                    fitted = model.fit()

                    aic = fitted.aic
                    results_table.append({'p': p, 'q': q, 'AIC': aic})

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)
                        print(f" AIC={aic:.2f} *** NEW BEST ***")
                    else:
                        print(f" AIC={aic:.2f}")

            except Exception as e:
                print(f" FAILED (convergence error)")
                # Skip models that fail to converge
                results_table.append({'p': p, 'q': q, 'AIC': np.nan})
                continue

    print(f"\n{'='*70}")
    print(f"OPTIMAL MODEL SELECTED: ARMAX{best_order} with AIC = {best_aic:.2f}")
    print(f"{'='*70}")

    # Display top 5 models
    results_df = pd.DataFrame(results_table).dropna()
    results_df = results_df.sort_values('AIC').head(10)

    print("\nTop 10 Models by AIC:")
    print(f"{'Rank':<6} {'Model':<15} {'AIC':<15}")
    print("-" * 40)
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        model_name = f"ARMAX({int(row['p'])},{int(row['q'])})"
        print(f"{idx:<6} {model_name:<15} {row['AIC']:<15.2f}")

    return best_order


def perform_multivariate_analysis(df, zone, use_log_transform=False, use_deseasonalized=False,
                                 run_ljungbox=False, run_hetero_tests=False, run_stationarity=False,
                                 optimize_armax_lags=False, run_tvp_wind_kalman=False,
                                 run_rolling_window=False, rolling_window_years=3,
                                 rolling_step_years=1, rolling_min_obs=24*180,
                                 run_quantile_regression=False):
    """
    Runs OLS and ARMAX-GARCHX with full control variables and optional diagnostic tests.

    Parameters:
    - use_log_transform: Use logged variables
    - use_deseasonalized: Use deseasonalized (logged) variables
    """
    print(f"\n--- RUNNING MULTIVARIATE ANALYSIS ({zone}) ---")

    # Determine dependent variable (Y) and exogenous variables based on flags
    if use_log_transform and use_deseasonalized:
        print("Using: Logged and Deseasonalized variables (Standard approach)")

        # Dependent variable: Price_Log_Deseasonalized
        Y = df['Price_Log_Deseasonalized']

        # Exogenous variables: deseasonalized logged versions (except Wind and Net_Exchange)
        exog_vars = [
            'Wind_Forecast_Log',  # NOT deseasonalized (Fredriksson doesn't deseasonalize wind)
            'Hydro_Reserves_Log_Deseasonalized',
            'Net_Exchange',  # NOT logged or deseasonalized
            'Consumption_Log_Deseasonalized'
        ]

        # Add commodity controls if available (deseasonalized versions)
        if 'Oil_Price_Log_Deseasonalized' in df.columns:
            exog_vars.append('Oil_Price_Log_Deseasonalized')
        if 'Gas_Price_Log_Deseasonalized' in df.columns:
            exog_vars.append('Gas_Price_Log_Deseasonalized')

    elif use_log_transform:
        print("Using: Logged variables only (no deseasonalization)")

        # Dependent variable: Price_Log
        Y = df['Price_Log']

        # Exogenous variables: logged versions
        exog_vars = [
            'Wind_Forecast_Log',
            'Hydro_Reserves_Log',
            'Net_Exchange',  # NOT logged
            'Consumption_Log'
        ]

        # Add commodity controls if available (logged versions)
        if 'Oil_Price_Log' in df.columns:
            exog_vars.append('Oil_Price_Log')
        if 'Gas_Price_Log' in df.columns:
            exog_vars.append('Gas_Price_Log')

    else:
        print("Using: Raw variables (no log transformation or deseasonalization)")

        # Dependent variable: Price
        Y = df['Price']

        # Exogenous variables: raw versions
        exog_vars = ['Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']

        # Add commodity controls if available (raw versions)
        if 'Oil_Price' in df.columns:
            exog_vars.append('Oil_Price')
        if 'Gas_Price' in df.columns:
            exog_vars.append('Gas_Price')

    print(f"Dependent variable: {Y.name}")
    print(f"Exogenous variables: {exog_vars}")

    # TVP Kalman Filter mode: run time-varying parameter analysis and return early
    if run_tvp_wind_kalman:
        run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, use_log_transform, plots_dir="plots")
        return None, None  # Early return, skip OLS/ARMAX

    # Rolling-window mode: run rolling window analysis and return early
    if run_rolling_window:
        run_rolling_window_analysis(df, zone, Y, exog_vars, use_log_transform,
                                    window_years=rolling_window_years,
                                    step_years=rolling_step_years,
                                    min_obs=rolling_min_obs,
                                    plots_dir="plots",
                                    results_dir="results")
        return None, None  # Early return, skip OLS/ARMAX

    # Quantile regression mode: run quantile regression analysis and return early
    if run_quantile_regression:
        run_quantile_regression_analysis(df, zone, use_log_transform,
                                         plots_dir="plots",
                                         results_dir="results")
        return None, None  # Early return, skip OLS/ARMAX

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

    # Determine optimal lags if enabled, otherwise use default (3,3)
    if optimize_armax_lags:
        optimal_order = select_armax_lags_aic(Y, df[exog_vars], max_p=10, max_q=10)
        armax_order = (optimal_order[0], 0, optimal_order[1])
    else:
        armax_order = (3, 0, 3)
        print(f"Using default ARMAX{armax_order} specification (set OPTIMIZE_ARMAX_LAGS=True for AIC-based selection)")

    # Mean Equation (Price Level)
    armax_res = sm.tsa.ARIMA(Y, exog=df[exog_vars], order=armax_order).fit()
    print(f"\nMEAN EQUATION (Price Level) - ARMAX{armax_order}:")
    print(armax_res.summary())

    # Optional: Diagnostic tests on ARMAX residuals
    if run_ljungbox:
        print("\n" + "="*70)
        print("DIAGNOSTIC TESTS ON ARMAX RESIDUALS")
        print("="*70)
        run_ljungbox_test(armax_res.resid, lags=[5, 10, 15, 20])

    if run_hetero_tests:
        run_heteroskedasticity_tests(armax_res.resid, nlags=10)

    # GARCHX component can be added here if ARCH-effects are confirmed
    return ols_model, armax_res


# --- 5. VISUALIZATION FUNCTIONS ---

def plot_time_series(df, zone, plots_dir='plots'):
    """Create time series plots for all variables."""

    print("\n--- Creating Time Series Plots ---")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Raw Time Series Data - {zone} (2021-2024)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh']

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
    filepath = os.path.join(plots_dir, f'raw_time_series_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_distributions(df, zone, plots_dir='plots'):
    """Create distribution plots (histograms + KDE) for all variables."""

    print("\n--- Creating Distribution Plots ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distribution of Raw Variables - {zone} (with outlier detection)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh']

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

        # Mark 6x std threshold (Fredriksson outlier definition)
        upper_6std = mean_val + 6 * std_val
        lower_3_7std = mean_val - 3.7 * std_val
        ax.axvline(upper_6std, color='orange', linestyle=':', linewidth=2, label=f'+6*std: {upper_6std:.2f}')
        ax.axvline(lower_3_7std, color='orange', linestyle=':', linewidth=2, label=f'-3.7*std: {lower_3_7std:.2f}')

        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{unit}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Count outliers using Fredriksson definition
        outliers_upper = (data > upper_6std).sum()
        outliers_lower = (data < lower_3_7std).sum()
        total_outliers = outliers_upper + outliers_lower

        # Add outlier count
        outlier_text = f'Outliers (Fredriksson):\nUpper (>6*std): {outliers_upper}\nLower (<-3.7*std): {outliers_lower}\nTotal: {total_outliers}'
        ax.text(0.98, 0.98, outlier_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Hide the last subplot if not used
    axes[-1].axis('off')

    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'raw_distributions_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close()


def plot_boxplots(df, zone, plots_dir='plots'):
    """Create box plots to visualize outliers."""

    print("\n--- Creating Box Plots ---")

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle(f'Box Plots for Outlier Detection - {zone} (Raw Data)', fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh']

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
    filepath = os.path.join(plots_dir, f'raw_boxplots_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close()


def detect_outliers_fredriksson(df, zone, plots_dir='plots'):
    """
    Detect outliers using Fredriksson (2016) methodology.

    Outlier definition:
    - Exceeds 6x standard deviation above the mean, OR
    - Lower than 3.7x standard deviation below the mean
    """

    print("\n" + "="*80)
    print(f"OUTLIER DETECTION - FREDRIKSSON (2016) METHODOLOGY - {zone}")
    print("="*80)
    print("Definition: Outliers exceed 6*std above mean OR fall below -3.7*std below mean\n")

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']

    outlier_summary = []

    for var in variables:
        data = df[var].dropna()
        mean_val = data.mean()
        std_val = data.std()

        # Fredriksson thresholds
        upper_threshold = mean_val + 6 * std_val
        lower_threshold = mean_val - 3.7 * std_val

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
        print(f"  Upper threshold (+6*std): {upper_threshold:.2f}")
        print(f"  Lower threshold (-3.7*std): {lower_threshold:.2f}")
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
            'N_Outliers_Upper': n_outliers_upper,
            'N_Outliers_Lower': n_outliers_lower,
            'N_Outliers_Total': n_outliers_total,
            'Pct_Outliers': pct_outliers
        })

    return pd.DataFrame(outlier_summary)


def plot_outliers_timeline(df, zone, plots_dir='plots'):
    """Plot time series highlighting detected outliers."""

    print("\n--- Creating Outlier Timeline Visualization ---")

    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle(f'Time Series with Detected Outliers - {zone} (Fredriksson Methodology)',
                 fontsize=16, fontweight='bold')

    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    units = ['EUR/MWh', 'MWh', 'MWh', 'MWh', 'MWh']

    for i, (var, color, unit) in enumerate(zip(variables, colors, units)):
        ax = axes[i]
        data = df[var]

        # Calculate thresholds
        mean_val = data.mean()
        std_val = data.std()
        upper_threshold = mean_val + 6 * std_val
        lower_threshold = mean_val - 3.7 * std_val

        # Identify outliers
        outliers = (data > upper_threshold) | (data < lower_threshold)

        # Plot normal data
        ax.plot(df.index, data, color=color, alpha=0.5, linewidth=0.5, label='Normal data')

        # Highlight outliers
        ax.scatter(df.index[outliers], data[outliers], color='red', s=20,
                  alpha=0.8, label=f'Outliers (n={outliers.sum()})', zorder=5)

        # Add threshold lines
        ax.axhline(upper_threshold, color='orange', linestyle='--',
                  linewidth=1.5, label=f'+6*std: {upper_threshold:.2f}')
        ax.axhline(lower_threshold, color='orange', linestyle='--',
                  linewidth=1.5, label=f'-3.7*std: {lower_threshold:.2f}')
        ax.axhline(mean_val, color='green', linestyle='-',
                  linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.2f}')

        ax.set_title(f'{var}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{unit}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'raw_outliers_timeline_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close()


def plot_scatter_matrix(df, zone, plots_dir='plots'):
    """Create scatter plot matrix to see relationships between variables."""

    print("\n--- Creating Scatter Plot Matrix ---")

    # Select variables for scatter matrix
    variables = ['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']

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

    plt.suptitle(f'Scatter Plot Matrix - Relationships Between Variables - {zone}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save to plots directory
    filepath = os.path.join(plots_dir, f'raw_scatter_matrix_{zone}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close()


def run_visualizations(data, zone, plots_dir='plots'):
    """Run all visualization functions."""

    print("\n" + "="*80)
    print("RUNNING RAW DATA VISUALIZATIONS")
    print("="*80)

    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"\nCreated directory: {plots_dir}/")

    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 8)

    # Basic statistics
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS (RAW DATA)")
    print("="*80)
    print(data[['Price', 'Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']].describe())

    # Detect outliers using Fredriksson methodology
    outlier_summary = detect_outliers_fredriksson(data, zone, plots_dir)

    # Save outlier summary to CSV in plots directory
    csv_path = os.path.join(plots_dir, f'outlier_summary_fredriksson_{zone}.csv')
    outlier_summary.to_csv(csv_path, index=False)
    print(f"\n  Saved outlier summary to: {csv_path}")

    # Create visualizations
    print("\n--- Generating Visualizations ---")
    plot_time_series(data, zone, plots_dir)
    plot_distributions(data, zone, plots_dir)
    plot_boxplots(data, zone, plots_dir)
    plot_outliers_timeline(data, zone, plots_dir)
    plot_scatter_matrix(data, zone, plots_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {plots_dir}/")
    print("\nGenerated files:")
    print(f"  1. raw_time_series_{zone}.png - Time series plots for all variables")
    print(f"  2. raw_distributions_{zone}.png - Distribution histograms with outlier thresholds")
    print(f"  3. raw_boxplots_{zone}.png - Box plots for outlier detection")
    print(f"  4. raw_outliers_timeline_{zone}.png - Time series with outliers highlighted")
    print(f"  5. raw_scatter_matrix_{zone}.png - Relationships between variables")
    print(f"  6. outlier_summary_fredriksson_{zone}.csv - Outlier detection statistics")


# --- 6. EXECUTION BLOCK ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ACTIVE_ZONE = 'SE1'

    # --- VISUALIZATION TOGGLE ---
    # Toggle for data visualization and outlier detection
    # When True: generates comprehensive visualizations of raw data and outlier detection
    # When False: skips visualization and proceeds directly to regression analysis
    RUN_VISUALIZATIONS = False

    # --- TRANSFORMATION TOGGLES (STANDARD APPROACH) ---
    # Toggle for logarithmic transformation (Fredriksson 2016 methodology)
    # When True: applies log() to Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price
    # Net_Exchange is NOT logged (can contain negative values)
    # STANDARD APPROACH: Log transformation is applied FIRST, before deseasonalization
    USE_LOG_TRANSFORM = True

    # Toggle for deseasonalization (Fredriksson 2016 methodology)
    # When True: deseasonalizes the LOGGED variables using dummy variable regression
    # STANDARD APPROACH: Deseasonalization is applied to LOGGED series (after log transformation)
    USE_DESEASONALIZED = True

    # --- OUTLIER HANDLING TOGGLES ---
    # Toggle for outlier replacement
    # When True: replaces outliers using selected method
    # When False: keeps outliers in the data (no replacement)
    HANDLE_OUTLIERS = True

    # Outlier handling method selection
    # 'fredriksson': Fredriksson (2016) methodology
    #   - Threshold: +6σ / -3.7σ (asymmetric)
    #   - Replacement: Mean of 24h and 48h before/after outlier
    # 'gianfreda': Gianfreda (2010) / Mugele et al. (2005) methodology
    #   - Threshold: ±3σ (symmetric)
    #   - Replacement: Capped at ±3σ for respective weekday
    OUTLIER_METHOD = 'gianfreda'  # Options: 'fredriksson' or 'gianfreda'

    # METHODOLOGICAL NOTE:
    # Fredriksson (2016) applies outlier filter TWICE:
    #   1st: On original price series (found 31 outliers)
    #   2nd: On deseasonalized price series (found 42 outliers)
    #
    # OUR APPROACH: Apply outlier filter ONCE, on logged-deseasonalized series
    # Rationale:
    #   - Seasonal patterns mask true outliers (e.g., high winter prices vs low summer)
    #   - Log transformation stabilizes variance
    #   - Deseasonalized mean ≈ 0 makes threshold more meaningful
    #   - Cleaner single-pass approach with stronger statistical justification
    #   - Fredriksson provides no theoretical justification for double application
    #
    # TODO: Future sensitivity analysis could compare single vs. double application

    # Toggle for linear interpolation of missing values
    # When True: fills missing values by linear interpolation between surrounding values
    # When False: drops all rows with missing values (original behavior)
    USE_LINEAR_INTERPOLATION = True

    # --- COMMODITY PRICE LAGGING ---
    # Commodity prices (oil & gas) are ALWAYS lagged by 24 hours (hardcoded in pipeline)
    # Rationale: Day-ahead electricity market uses commodity prices from bidding time (D-1)
    # This aligns with standard literature (Weron, Huisman, etc.)
    LAG_COMMODITY_HOURS = 24  # Applied automatically in lag_commodity_prices()

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
    # When False: Uses default ARMAX(3,3) specification
    # WARNING: This can take several minutes to run (tests 100 model combinations)
    OPTIMIZE_ARMAX_LAGS = False

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
    ROLLING_MIN_OBS = 24 * 180        # Minimum observations per window (6 months hourly data)

    # --- QUANTILE REGRESSION TOGGLE ---
    # When True: estimates wind coefficient across quantiles of price distribution (skips OLS/ARMAX)
    # When False: runs standard analysis
    RUN_QUANTILE_REGRESSION = False

    # Updated paths matching your local project directory
    # Master data files are stored in 'master data files/' folder
    PATHS = {
        'price': 'master data files/Spot_Prices.xlsx',
        'wind': 'master data files/Master_Wind_Forecast_Merged_2021_2024.xlsx',
        'hydro': 'master data files/Master_Hydro_Reservoir.xlsx',
        'exch': 'master data files/Master_Exchange_Merged_2021_2024.xlsx',
        'cons': 'master data files/Master_Consumption_2021_2024.xlsx',
        'commodities': 'master data files/Master_Commodities.xlsx'
    }

    try:
        # Load and clean full dataset
        data = load_all_thesis_data(PATHS, zone_price=ACTIVE_ZONE, use_interpolation=USE_LINEAR_INTERPOLATION)
        print(f"Merge successful. Total hourly observations: {len(data)}")

        # --- STEP 0: LAG COMMODITY PRICES (day-ahead market alignment) ---
        # Apply 24-hour lag to oil and gas prices to reflect information set at bidding time
        # Standard practice in electricity price modeling: commodity prices at time t should be
        # from t-24 (when day-ahead auction occurred), not from time t (delivery time)
        data = lag_commodity_prices(data, lag_hours=LAG_COMMODITY_HOURS)

        # --- STEP 1: VISUALIZATION OF RAW DATA (if enabled) ---
        if RUN_VISUALIZATIONS:
            run_visualizations(data, ACTIVE_ZONE)

        # --- STEP 2: LOG TRANSFORMATION (if enabled) ---
        # STANDARD APPROACH: Apply log transformation FIRST, before deseasonalization
        # This is the standard econometric approach for handling multiplicative seasonality
        # Note: Commodity prices are already lagged at this point
        if USE_LOG_TRANSFORM:
            data = apply_log_transform(data)

        # --- STEP 3: DESEASONALIZATION (if enabled) ---
        # STANDARD APPROACH: Deseasonalize the LOGGED variables (after log transformation)
        # Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL deseasonalization)
        # Hydro, Oil, Gas: Year + Month ONLY (PARTIAL - no intraday patterns)
        if USE_DESEASONALIZED:
            data = deseasonalize_logged_variables(data)

        # --- STEP 4: OUTLIER HANDLING (if enabled) ---
        # Apply outlier handling to logged-deseasonalized series (if both transformations enabled)
        # This provides the most meaningful outlier detection:
        #   - Log transformation stabilizes variance
        #   - Deseasonalization removes seasonal patterns
        #   - Threshold more meaningful on transformed data
        if HANDLE_OUTLIERS:
            if OUTLIER_METHOD == 'fredriksson':
                data, outlier_stats = handle_outliers_fredriksson(data,
                                                                  use_log_transform=USE_LOG_TRANSFORM,
                                                                  use_deseasonalized=USE_DESEASONALIZED)
            elif OUTLIER_METHOD == 'gianfreda':
                data, outlier_stats = handle_outliers_gianfreda(data,
                                                                use_log_transform=USE_LOG_TRANSFORM,
                                                                use_deseasonalized=USE_DESEASONALIZED)
            else:
                raise ValueError(f"Unknown outlier method: {OUTLIER_METHOD}. Choose 'fredriksson' or 'gianfreda'.")

        # --- STEP 5: REGRESSION ANALYSIS ---
        # Run regression models with optional diagnostic tests
        # Commodity prices used in regression are lagged by 24h (from Step 0)
        perform_multivariate_analysis(data, ACTIVE_ZONE,
                                      use_log_transform=USE_LOG_TRANSFORM,
                                      use_deseasonalized=USE_DESEASONALIZED,
                                      run_ljungbox=RUN_LJUNGBOX_TEST,
                                      run_hetero_tests=RUN_HETEROSKEDASTICITY_TESTS,
                                      run_stationarity=RUN_STATIONARITY_TESTS,
                                      optimize_armax_lags=OPTIMIZE_ARMAX_LAGS,
                                      run_tvp_wind_kalman=RUN_TVP_WIND_KALMAN,
                                      run_rolling_window=RUN_ROLLING_WINDOW,
                                      rolling_window_years=ROLLING_WINDOW_YEARS,
                                      rolling_step_years=ROLLING_STEP_YEARS,
                                      rolling_min_obs=ROLLING_MIN_OBS,
                                      run_quantile_regression=RUN_QUANTILE_REGRESSION)

    except Exception as e:
        print(f"Critical error during execution: {e}")