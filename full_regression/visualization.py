"""
################################################################################
#  [Module 09/10]  visualization.py  –  Plotting Functions and Data Export
#
#  Contains (in order):
#    --- Zone Comparison ---
#    1. plot_zone_comparisons()     : overlays SE1-SE4 price/wind/volatility comparisons
#
#    --- Time Series & Distributions ---
#    2. plot_time_series()          : raw/logged/deseasonalized variable time plots
#    3. plot_distributions()        : histogram + KDE of variable distributions
#    4. plot_boxplots()             : seasonal boxplots (hourly, daily, monthly)
#
#    --- Outlier Visualization ---
#    5. detect_outliers()           : identify outlier indices and statistics
#    6. plot_outliers_timeline()    : timeline plot of detected outliers
#
#    --- Scatter & Correlation ---
#    7. plot_scatter_matrix()       : pairwise scatter plots of all variables
#
#    --- Master Orchestrator ---
#    8. run_visualizations()        : generate all plots for a zone at given stage
#
#    --- Data Export ---
#    9. export_data_for_R()         : export processed data to CSV for R strucchange
#
#  Dependencies: none (self-contained)
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats

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


