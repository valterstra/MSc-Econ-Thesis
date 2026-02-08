"""
SE4-Germany Congestion Analysis Script

This script analyzes congestion frequency between SE4 (Sweden) and Germany
during the period when price data is available (July 2019 - December 2025).

Purpose: Determine if assuming "always congested" for the pre-2019 period is reasonable
for modeling purposes in the regional_data_combiner.py script.
"""

import pandas as pd
import os
import numpy as np

# Configuration
CONGESTION_EPSILON_EUR = 0.01  # Same threshold as regional_data_combiner.py

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
spot_price_file = os.path.join(script_dir, 'master data files', '2015-2025', 'Spot_Prices_2015_2025.xlsx')
output_dir = os.path.join(script_dir, 'results')


def load_and_filter_data():
    """
    Load spot price data and filter to rows where both SE4 and GER prices are available.

    Returns:
        pd.DataFrame: Filtered DataFrame with Timestamp, SE4_Price, GER_Price
    """
    print("Loading spot price data...")
    df = pd.read_excel(spot_price_file)

    # Filter to rows where both SE4 and GER prices are not null
    df_filtered = df[df['SE4_Price (EUR)'].notna() & df['GER_Price (EUR)'].notna()].copy()

    # Select and rename relevant columns
    df_filtered = df_filtered[['Timestamp', 'SE4_Price (EUR)', 'GER_Price (EUR)']].copy()
    df_filtered.columns = ['Timestamp', 'SE4_Price_EUR', 'GER_Price_EUR']

    print(f"Loaded {len(df_filtered):,} hours with both SE4 and GER price data")
    print(f"Period: {df_filtered['Timestamp'].min()} to {df_filtered['Timestamp'].max()}")

    return df_filtered


def calculate_congestion_metrics(df, epsilon=CONGESTION_EPSILON_EUR):
    """
    Calculate congestion metrics based on price differences.

    Args:
        df (pd.DataFrame): DataFrame with SE4 and GER prices
        epsilon (float): Congestion threshold in EUR

    Returns:
        pd.DataFrame: Enhanced DataFrame with congestion metrics
    """
    print(f"\nCalculating congestion metrics (epsilon = {epsilon} EUR)...")

    # Calculate price differences
    df['Price_Diff'] = df['SE4_Price_EUR'] - df['GER_Price_EUR']
    df['Abs_Price_Diff'] = df['Price_Diff'].abs()

    # Calculate congestion indicators
    df['Congestion_Dummy'] = (df['Abs_Price_Diff'] > epsilon).astype(int)
    df['SE4_Higher'] = (df['Price_Diff'] > epsilon).astype(int)  # Export congestion
    df['GER_Higher'] = (df['Price_Diff'] < -epsilon).astype(int)  # Import congestion

    # Add temporal columns for analysis
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    return df


def calculate_overall_statistics(df):
    """
    Calculate overall congestion statistics.

    Args:
        df (pd.DataFrame): DataFrame with congestion metrics

    Returns:
        dict: Dictionary of overall statistics
    """
    total_hours = len(df)
    congested_hours = df['Congestion_Dummy'].sum()
    se4_higher_hours = df['SE4_Higher'].sum()
    ger_higher_hours = df['GER_Higher'].sum()

    stats = {
        'total_hours': total_hours,
        'congested_hours': congested_hours,
        'congested_pct': (congested_hours / total_hours * 100) if total_hours > 0 else 0,
        'se4_higher_hours': se4_higher_hours,
        'se4_higher_pct': (se4_higher_hours / total_hours * 100) if total_hours > 0 else 0,
        'ger_higher_hours': ger_higher_hours,
        'ger_higher_pct': (ger_higher_hours / total_hours * 100) if total_hours > 0 else 0,
        'no_congestion_hours': total_hours - congested_hours,
        'no_congestion_pct': ((total_hours - congested_hours) / total_hours * 100) if total_hours > 0 else 0,
        'mean_price_diff': df['Price_Diff'].mean(),
        'median_price_diff': df['Price_Diff'].median(),
        'std_price_diff': df['Price_Diff'].std(),
        'min_price_diff': df['Price_Diff'].min(),
        'max_price_diff': df['Price_Diff'].max(),
        'mean_abs_price_diff': df['Abs_Price_Diff'].mean(),
        'start_date': df['Timestamp'].min(),
        'end_date': df['Timestamp'].max()
    }

    return stats


def calculate_yearly_statistics(df):
    """
    Calculate yearly congestion statistics.

    Args:
        df (pd.DataFrame): DataFrame with congestion metrics

    Returns:
        pd.DataFrame: Yearly statistics
    """
    yearly_stats = df.groupby('Year').agg({
        'Timestamp': 'count',
        'Congestion_Dummy': 'sum',
        'SE4_Higher': 'sum',
        'GER_Higher': 'sum',
        'Price_Diff': ['mean', 'median']
    }).reset_index()

    # Flatten column names
    yearly_stats.columns = ['Year', 'Total_Hours', 'Congested_Hours',
                            'SE4_Higher_Hours', 'GER_Higher_Hours',
                            'Mean_Price_Diff', 'Median_Price_Diff']

    # Calculate percentages
    yearly_stats['Congested_Pct'] = (yearly_stats['Congested_Hours'] / yearly_stats['Total_Hours'] * 100)
    yearly_stats['SE4_Higher_Pct'] = (yearly_stats['SE4_Higher_Hours'] / yearly_stats['Total_Hours'] * 100)
    yearly_stats['GER_Higher_Pct'] = (yearly_stats['GER_Higher_Hours'] / yearly_stats['Total_Hours'] * 100)

    return yearly_stats


def print_summary_report(overall_stats, yearly_stats):
    """
    Print formatted summary report to console.

    Args:
        overall_stats (dict): Overall statistics
        yearly_stats (pd.DataFrame): Yearly statistics
    """
    print("\n" + "="*80)
    print("SE4-GERMANY CONGESTION ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n{'DATA PERIOD':}")
    print(f"  Start Date: {overall_stats['start_date']}")
    print(f"  End Date:   {overall_stats['end_date']}")
    print(f"  Total Hours: {overall_stats['total_hours']:,}")

    print(f"\n{'OVERALL CONGESTION STATISTICS':}")
    print(f"  Congested Hours:     {overall_stats['congested_hours']:>10,} ({overall_stats['congested_pct']:>5.2f}%)")
    print(f"  No Congestion Hours: {overall_stats['no_congestion_hours']:>10,} ({overall_stats['no_congestion_pct']:>5.2f}%)")

    print(f"\n{'DIRECTIONAL ANALYSIS':}")
    print(f"  SE4 > GER (Export Congestion): {overall_stats['se4_higher_hours']:>10,} ({overall_stats['se4_higher_pct']:>5.2f}%)")
    print(f"  GER > SE4 (Import Congestion): {overall_stats['ger_higher_hours']:>10,} ({overall_stats['ger_higher_pct']:>5.2f}%)")

    print(f"\n{'PRICE SPREAD STATISTICS (SE4 - GER, EUR):':}")
    print(f"  Mean:     {overall_stats['mean_price_diff']:>8.2f}")
    print(f"  Median:   {overall_stats['median_price_diff']:>8.2f}")
    print(f"  Std Dev:  {overall_stats['std_price_diff']:>8.2f}")
    print(f"  Min:      {overall_stats['min_price_diff']:>8.2f}")
    print(f"  Max:      {overall_stats['max_price_diff']:>8.2f}")
    print(f"  Mean Abs: {overall_stats['mean_abs_price_diff']:>8.2f}")

    print(f"\n{'YEARLY BREAKDOWN':}")
    print("-" * 80)
    print(f"{'Year':<6} {'Hours':<8} {'Congested':<12} {'SE4>GER':<12} {'GER>SE4':<12} {'Mean Diff':>10}")
    print("-" * 80)

    for _, row in yearly_stats.iterrows():
        year_note = "*" if row['Year'] == 2019 else " "
        print(f"{int(row['Year']):<6}{year_note} {int(row['Total_Hours']):<8,} "
              f"{int(row['Congested_Hours']):>5,} ({row['Congested_Pct']:>5.2f}%) "
              f"{int(row['SE4_Higher_Hours']):>5,} ({row['SE4_Higher_Pct']:>5.2f}%) "
              f"{int(row['GER_Higher_Hours']):>5,} ({row['GER_Higher_Pct']:>5.2f}%) "
              f"{row['Mean_Price_Diff']:>10.2f}")

    print("-" * 80)
    print("* 2019 is partial year (July-December only)")

    print(f"\n{'KEY INSIGHTS':}")

    # Compare with reference congestion rates
    print(f"  - SE4-GER congestion rate: {overall_stats['congested_pct']:.1f}%")
    print(f"  - Reference rates: SE4-DK2 ~45%, SE4-SE3 ~38%")

    # Analyze directionality
    if overall_stats['ger_higher_pct'] > overall_stats['se4_higher_pct']:
        ratio = overall_stats['ger_higher_pct'] / max(overall_stats['se4_higher_pct'], 0.01)
        print(f"  - Heavily biased toward GER > SE4 (import congestion): {ratio:.1f}x more frequent")

    # Analyze temporal stability
    cv = yearly_stats['Congested_Pct'].std() / yearly_stats['Congested_Pct'].mean() * 100
    if cv < 10:
        print(f"  - Congestion rate is highly consistent across years (CV: {cv:.1f}%)")
    elif cv < 20:
        print(f"  - Congestion rate shows moderate variation across years (CV: {cv:.1f}%)")
    else:
        print(f"  - Congestion rate varies significantly across years (CV: {cv:.1f}%)")

    print(f"\n{'RECOMMENDATIONS FOR PRE-2019 MODELING':}")

    if overall_stats['congested_pct'] > 80:
        print("  - High congestion rate (>80%) supports 'always congested' assumption")
        print("  - Consider using congestion dummy = 1 for all pre-2019 hours")
    elif overall_stats['congested_pct'] > 60:
        print("  - Moderate-high congestion rate (60-80%) suggests frequent but not constant congestion")
        print("  - Options: (1) Use observed rate as probability, or (2) Create separate dummy for missing period")
    else:
        print("  - Lower congestion rate (<60%) suggests 'always congested' may overstate")
        print("  - Recommend creating separate dummy variable for data availability period")

    print("\n" + "="*80)


def save_results(df, yearly_stats, output_dir):
    """
    Save analysis results to CSV files.

    Args:
        df (pd.DataFrame): Detailed hourly data
        yearly_stats (pd.DataFrame): Yearly statistics
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save hourly data
    hourly_output_file = os.path.join(output_dir, 'se4_germany_congestion_hourly.csv')
    df.to_csv(hourly_output_file, index=False)
    print(f"\nSaved detailed hourly data to: {hourly_output_file}")

    # Save yearly summary
    yearly_output_file = os.path.join(output_dir, 'se4_germany_congestion_yearly_summary.csv')
    yearly_stats.to_csv(yearly_output_file, index=False)
    print(f"Saved yearly summary to: {yearly_output_file}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("SE4-GERMANY CONGESTION ANALYSIS")
    print("="*80)

    # Load data
    df = load_and_filter_data()

    # Calculate metrics
    df = calculate_congestion_metrics(df)

    # Calculate statistics
    overall_stats = calculate_overall_statistics(df)
    yearly_stats = calculate_yearly_statistics(df)

    # Output results
    print_summary_report(overall_stats, yearly_stats)
    save_results(df, yearly_stats, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
