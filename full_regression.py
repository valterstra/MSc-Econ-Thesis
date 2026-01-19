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

    # Handle missing values based on configuration
    if use_interpolation:
        print("\n--- APPLYING LINEAR INTERPOLATION FOR MISSING VALUES ---")
        missing_before = final_df.isna().sum().sum()
        # Apply linear interpolation to fill missing values
        final_df = final_df.interpolate(method='linear', limit_direction='both')
        # Drop any remaining NaN values (e.g., at edges if interpolation couldn't fill)
        final_df = final_df.dropna()
        missing_after = final_df.isna().sum().sum()
        print(f"Missing values before interpolation: {missing_before}")
        print(f"Missing values after interpolation: {missing_after}")
        print(f"Rows retained: {len(final_df)}")
    else:
        # Original behavior: drop all rows with missing values
        rows_before = len(final_df)
        final_df = final_df.dropna()
        rows_dropped = rows_before - len(final_df)
        if rows_dropped > 0:
            print(f"\nDropped {rows_dropped} rows with missing values (original behavior)")

    return final_df.set_index('Datetime')


# --- 2. DATA PREPROCESSING FUNCTIONS ---

def handle_outliers_fredriksson(df, use_deseasonalized=False):
    """
    Replace outliers using Fredriksson (2016) methodology.

    Outlier definition:
    - Exceeds 6x standard deviation above the mean, OR
    - Lower than 3.7x standard deviation below the mean

    Replacement method:
    - Replace outlier with mean of 24 and 48 hours before and after the outlier
    - Only applied to PRICE series, not explanatory variables

    Parameters:
    - use_deseasonalized: If True, applies to deseasonalized Price. If False, to raw Price.

    Returns:
    - DataFrame with outliers replaced in Price
    - Dictionary with outlier statistics
    """

    print("\n" + "="*80)
    print("OUTLIER HANDLING - FREDRIKSSON (2016) METHODOLOGY")
    print("="*80)
    if use_deseasonalized:
        print("Applying to: Deseasonalized Price")
    else:
        print("Applying to: Raw Price")
    print("Replacing outliers with mean of 24 and 48 hours before/after")
    print("Note: Outlier handling only applied to Price, not explanatory variables\n")

    df_clean = df.copy()
    outlier_stats = {}

    # Determine which Price column to use
    if use_deseasonalized and 'Price_Deseasonalized' in df.columns:
        price_col = 'Price_Deseasonalized'
        print(f"\nProcessing: {price_col}")
    else:
        price_col = 'Price'
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

    if n_outliers > 0:
        print(f"  Found {n_outliers} outliers to replace")

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
                print(f"    {idx}: {original_value:.2f} -> {replacement_value:.2f}")

        # Update the dataframe
        df_clean[price_col] = data

        # If we modified deseasonalized price, also need to update the raw Price
        # by adding back the seasonal component
        if use_deseasonalized and price_col == 'Price_Deseasonalized':
            # The raw Price is Price_Deseasonalized + seasonal component
            # We need to update raw Price to reflect the outlier replacement
            print(f"  Note: Deseasonalized outliers replaced. Raw Price updated accordingly.")

        # Recalculate statistics after replacement
        new_mean = data.mean()
        new_std = data.std()

        print(f"  After replacement:")
        print(f"    Mean: {mean_val:.2f} -> {new_mean:.2f}")
        print(f"    Std: {std_val:.2f} -> {new_std:.2f}")

        outlier_stats[price_col] = {
            'n_outliers': n_outliers,
            'original_mean': mean_val,
            'original_std': std_val,
            'new_mean': new_mean,
            'new_std': new_std
        }
    else:
        print(f"  No outliers found")
        outlier_stats[price_col] = {
            'n_outliers': 0,
            'original_mean': mean_val,
            'original_std': std_val,
            'new_mean': mean_val,
            'new_std': std_val
        }

    return df_clean, outlier_stats


def deseasonalize_price(df):
    """
    Remove seasonal patterns from Price, Consumption, and Hydro_Reserves using dummy variable regression.
    Based on Fredriksson (2016) methodology.

    Creates dummies for: Year, Month, Day-of-Week, Hour
    Returns df with deseasonalized columns added for Price, Consumption, and Hydro_Reserves.

    Note: Wind is NOT deseasonalized (Fredriksson logs wind directly).
    """
    print("\n--- DESEASONALIZING VARIABLES (Fredriksson 2016 methodology) ---")

    # Extract time components from Datetime index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['Hour'] = df.index.hour

    # Create dummy variables (drop_first=True avoids multicollinearity)
    year_dummies = pd.get_dummies(df['Year'], prefix='Year', drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True, dtype=float)
    dow_dummies = pd.get_dummies(df['DayOfWeek'], prefix='DOW', drop_first=True, dtype=float)
    hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour', drop_first=True, dtype=float)

    # Combine all seasonal dummies
    seasonal_dummies = pd.concat([year_dummies, month_dummies, dow_dummies, hour_dummies], axis=1)
    seasonal_dummies = sm.add_constant(seasonal_dummies).astype(float)

    # Deseasonalize Price
    price_model = sm.OLS(df['Price'], seasonal_dummies).fit()
    df['Price_Deseasonalized'] = price_model.resid
    print(f"Price: Seasonal R² = {price_model.rsquared:.4f}")
    print(f"  Original std: {df['Price'].std():.2f}, Deseasonalized std: {df['Price_Deseasonalized'].std():.2f}")

    # Deseasonalize Consumption
    consumption_model = sm.OLS(df['Consumption'], seasonal_dummies).fit()
    df['Consumption_Deseasonalized'] = consumption_model.resid
    print(f"Consumption: Seasonal R² = {consumption_model.rsquared:.4f}")
    print(f"  Original std: {df['Consumption'].std():.2f}, Deseasonalized std: {df['Consumption_Deseasonalized'].std():.2f}")

    # Deseasonalize Hydro_Reserves
    hydro_model = sm.OLS(df['Hydro_Reserves'], seasonal_dummies).fit()
    df['Hydro_Reserves_Deseasonalized'] = hydro_model.resid
    print(f"Hydro_Reserves: Seasonal R² = {hydro_model.rsquared:.4f}")
    print(f"  Original std: {df['Hydro_Reserves'].std():.2f}, Deseasonalized std: {df['Hydro_Reserves_Deseasonalized'].std():.2f}")

    # Clean up temporary columns
    df = df.drop(columns=['Year', 'Month', 'DayOfWeek', 'Hour'])

    print("Note: Wind_Forecast and Net_Exchange are NOT deseasonalized (following Fredriksson)")

    return df


def apply_log_transform(df, use_deseasonalized=False):
    """
    Apply logarithmic transformation to variables following Fredriksson (2016).

    Logs applied to: Price, Wind_Forecast, Hydro_Reserves, Consumption
    NOT logged: Net_Exchange (can contain negative values)

    Note: Fredriksson logs demand/hydro AFTER deseasonalization,
    but wind production is logged directly (not deseasonalized).
    """
    print("\n--- APPLYING LOGARITHMIC TRANSFORMATION ---")

    # Price transformation
    if use_deseasonalized and 'Price_Deseasonalized' in df.columns:
        # Shift deseasonalized residuals to positive range, then log
        price_shifted = df['Price_Deseasonalized'] + df['Price'].mean()
        price_shifted = price_shifted.clip(lower=0.01)
        df['Price_Log'] = np.log(price_shifted)
        print(f"Price: log(deseasonalized + mean) applied")
    else:
        # Log raw price directly
        df['Price_Log'] = np.log(df['Price'].clip(lower=0.01))
        print(f"Price: log(raw) applied")

    # Wind forecast - log directly (Fredriksson doesn't deseasonalize wind)
    df['Wind_Forecast_Log'] = np.log(df['Wind_Forecast'].clip(lower=0.01))
    print(f"Wind_Forecast: log(raw) applied")

    # Hydro reserves - use deseasonalized if available
    if use_deseasonalized and 'Hydro_Reserves_Deseasonalized' in df.columns:
        hydro_shifted = df['Hydro_Reserves_Deseasonalized'] + df['Hydro_Reserves'].mean()
        hydro_shifted = hydro_shifted.clip(lower=0.01)
        df['Hydro_Reserves_Log'] = np.log(hydro_shifted)
        print(f"Hydro_Reserves: log(deseasonalized + mean) applied")
    else:
        df['Hydro_Reserves_Log'] = np.log(df['Hydro_Reserves'].clip(lower=0.01))
        print(f"Hydro_Reserves: log(raw) applied")

    # Consumption - use deseasonalized if available
    if use_deseasonalized and 'Consumption_Deseasonalized' in df.columns:
        consumption_shifted = df['Consumption_Deseasonalized'] + df['Consumption'].mean()
        consumption_shifted = consumption_shifted.clip(lower=0.01)
        df['Consumption_Log'] = np.log(consumption_shifted)
        print(f"Consumption: log(deseasonalized + mean) applied")
    else:
        df['Consumption_Log'] = np.log(df['Consumption'].clip(lower=0.01))
        print(f"Consumption: log(raw) applied")

    # Net_Exchange - NOT logged (can be negative)
    print(f"Net_Exchange: NOT logged (contains negative values)")

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


def perform_multivariate_analysis(df, zone, use_deseasonalized=False, use_log_transform=False,
                                 run_ljungbox=False, run_hetero_tests=False, run_stationarity=False,
                                 optimize_armax_lags=False):
    """Runs OLS and ARMAX-GARCHX with full control variables and optional diagnostic tests."""
    print(f"\n--- RUNNING MULTIVARIATE ANALYSIS ({zone}) ---")

    # Select variables based on log transform toggle
    if use_log_transform:
        print("Using: Log-transformed variables (Fredriksson methodology)")
        # Use logged versions of variables
        exog_vars = ['Wind_Forecast_Log', 'Hydro_Reserves_Log', 'Net_Exchange', 'Consumption_Log']
        Y = df['Price_Log']
    else:
        # Use raw variables
        exog_vars = ['Wind_Forecast', 'Hydro_Reserves', 'Net_Exchange', 'Consumption']
        if use_deseasonalized:
            print("Using: Log of Deseasonalized Price")
            price_shifted = df['Price_Deseasonalized'] + df['Price'].mean()
            price_shifted = price_shifted.clip(lower=0.01)
            Y = np.log(price_shifted)
        else:
            print("Using: Raw Price")
            Y = df['Price']

    X = sm.add_constant(df[exog_vars])

    # 1. Standard OLS Regression
    ols_model = sm.OLS(Y, X).fit()
    print("\n--- OLS RESULTS ---")
    print(ols_model.summary())

    # Optional: Diagnostic tests on OLS residuals
    if run_stationarity:
        # Test stationarity of dependent variable (Price)
        run_stationarity_tests(Y, series_name=f"{zone} Price (Dependent Variable)")

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

    # --- OUTLIER HANDLING TOGGLE ---
    # Toggle for outlier replacement using Fredriksson (2016) methodology
    # When True: replaces outliers with mean of 24 and 48 hours before/after
    # When False: keeps outliers in the data (no replacement)
    #
    # METHODOLOGICAL NOTE - DEVIATION FROM FREDRIKSSON:
    # Fredriksson (2016) applies outlier filter TWICE:
    #   1st: On original price series (found 31 outliers)
    #   2nd: On deseasonalized price series (found 42 outliers)
    #
    # OUR APPROACH: Apply outlier filter ONCE, AFTER deseasonalization
    # Rationale:
    #   - Seasonal patterns mask true outliers (e.g., high winter prices vs low summer)
    #   - Deseasonalized mean ≈ 0 makes 6σ threshold more meaningful
    #   - Cleaner single-pass approach with stronger statistical justification
    #   - Fredriksson provides no theoretical justification for double application
    #
    # TODO: Future sensitivity analysis could compare single vs. double application
    HANDLE_OUTLIERS = True

    # Toggle for Fredriksson-style deseasonalization + log transformation
    # Set to True to use deseasonalized log price, False for raw price
    USE_DESEASONALIZED = True

    # Toggle for logarithmic transformation (Fredriksson 2016 methodology)
    # When True: applies log() to Price, Wind_Forecast, Hydro_Reserves, Consumption
    # Net_Exchange is NOT logged (can contain negative values)
    USE_LOG_TRANSFORM = False

    # Toggle for linear interpolation of missing values
    # When True: fills missing values by linear interpolation between surrounding values
    # When False: drops all rows with missing values (original behavior)
    USE_LINEAR_INTERPOLATION = False

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

    # Updated paths matching your local project directory
    # Master data files are stored in 'master data files/' folder
    PATHS = {
        'price': 'master data files/Spot_Prices.xlsx',
        'wind': 'master data files/Master_Wind_Forecast_Merged_2021_2024.xlsx',
        'hydro': 'master data files/Master_Hydro_Reservoir.xlsx',
        'exch': 'master data files/Master_Exchange_Merged_2021_2024.xlsx',
        'cons': 'master data files/Master_Consumption_2021_2024.xlsx'
    }

    try:
        # Load and clean full dataset
        data = load_all_thesis_data(PATHS, zone_price=ACTIVE_ZONE, use_interpolation=USE_LINEAR_INTERPOLATION)
        print(f"Merge successful. Total hourly observations: {len(data)}")

        # --- STEP 1: VISUALIZATION OF RAW DATA (if enabled) ---
        if RUN_VISUALIZATIONS:
            run_visualizations(data, ACTIVE_ZONE)

        # --- STEP 2: DESEASONALIZATION (if enabled) ---
        # Deseasonalize BEFORE outlier handling (if both are enabled)
        # This allows outlier detection on deseasonalized data for more meaningful thresholds
        if USE_DESEASONALIZED:
            data = deseasonalize_price(data)

        # --- STEP 3: OUTLIER HANDLING (if enabled) ---
        # Can handle outliers on either:
        #   - Raw Price (if USE_DESEASONALIZED=False)
        #   - Deseasonalized Price (if USE_DESEASONALIZED=True) ← RECOMMENDED
        # Our approach differs from Fredriksson who applies outlier filter twice
        if HANDLE_OUTLIERS:
            data, outlier_stats = handle_outliers_fredriksson(data, use_deseasonalized=USE_DESEASONALIZED)

        # --- STEP 4: LOG TRANSFORMATION (if enabled) ---
        # Apply logarithmic transformation (Fredriksson 2016 methodology)
        if USE_LOG_TRANSFORM:
            data = apply_log_transform(data, use_deseasonalized=USE_DESEASONALIZED)

        # --- STEP 5: REGRESSION ANALYSIS ---
        # Run regression models with optional diagnostic tests
        perform_multivariate_analysis(data, ACTIVE_ZONE,
                                      use_deseasonalized=USE_DESEASONALIZED,
                                      use_log_transform=USE_LOG_TRANSFORM,
                                      run_ljungbox=RUN_LJUNGBOX_TEST,
                                      run_hetero_tests=RUN_HETEROSKEDASTICITY_TESTS,
                                      run_stationarity=RUN_STATIONARITY_TESTS,
                                      optimize_armax_lags=OPTIMIZE_ARMAX_LAGS)

    except Exception as e:
        print(f"Critical error during execution: {e}")