import pandas as pd
import os
import pytz

# --- 1. CONFIGURATION ---
# Change this to 'SE1', 'SE2', 'SE3', or 'SE4' to switch regions
target_region = 'SE4'

# Extract region number (e.g., '1' from 'SE1')
region_number = target_region[-1]

# File paths for input data
spot_price_file = 'master data files/2015-2025/Spot_Prices_2015_2025.xlsx'
wind_forecast_file = f'Verified_S{region_number}_Wind_Forecast_2015_2025.xlsx'
exchange_file = f'Verified_{target_region}_Exchange_2015_2025.xlsx'
consumption_file = 'Master_Consumption_2015_2025.xlsx'

# Output file
output_file = f'master data files/2015-2025/Combined_{target_region}_Data_2015_2025.xlsx'

print(f"=" * 60)
print(f"  COMBINING DATA FOR {target_region}")
print(f"=" * 60)

# --- 2. CREATE GROUND TRUTH (2015-2025) ---
print("\nCreating ground truth template...")
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2015-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h', tz=tz)

ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
# Sequential ID for duplicate hours in October (DST transitions)
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()
print(f"Ground truth created: {len(ground_truth_df)} hours expected")

# --- 3. LOAD ALL DATA FILES ---
print("\nLoading data files...")

# Load Spot Prices (2015-2025)
print(f"  Loading spot prices...")
spot_df = pd.read_excel(spot_price_file)
spot_df['Timestamp'] = pd.to_datetime(spot_df['Timestamp'])
# Extract only the relevant region's price column
price_column = f'{target_region}_Price (EUR)'
spot_df = spot_df[['Timestamp', price_column]].copy()
spot_df.rename(columns={price_column: 'Spot_Price'}, inplace=True)
# Add Occurrence for DST handling
spot_df['Occurrence'] = spot_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(spot_df)} rows")

# Load Wind Forecast (2015-2025)
print(f"  Loading wind forecast for S{region_number}...")
wind_df = pd.read_excel(wind_forecast_file)
wind_df['Timestamp'] = pd.to_datetime(wind_df['Timestamp'])
wind_column = f'S{region_number}_Wind'
wind_df = wind_df[['Timestamp', wind_column]].copy()
wind_df.rename(columns={wind_column: 'Wind_Forecast'}, inplace=True)
# Add Occurrence for DST handling
wind_df['Occurrence'] = wind_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(wind_df)} rows")

# Load Exchange Forecast (2015-2025)
print(f"  Loading exchange forecast for {target_region}...")
exchange_df = pd.read_excel(exchange_file)
exchange_df['Timestamp'] = pd.to_datetime(exchange_df['Timestamp'])
exchange_column = f'{target_region}_Total_Net_Exchange'
exchange_df = exchange_df[['Timestamp', exchange_column]].copy()
exchange_df.rename(columns={exchange_column: 'Net_Exchange'}, inplace=True)
# Add Occurrence for DST handling
exchange_df['Occurrence'] = exchange_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(exchange_df)} rows")

# Load Consumption Forecast (2015-2025)
print(f"  Loading consumption forecast for {target_region}...")
consumption_df = pd.read_excel(consumption_file)
consumption_df['Timestamp'] = pd.to_datetime(consumption_df['Timestamp'])
consumption_df = consumption_df[['Timestamp', target_region]].copy()
consumption_df.rename(columns={target_region: 'Consumption_Forecast'}, inplace=True)
# Add Occurrence for DST handling
consumption_df['Occurrence'] = consumption_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(consumption_df)} rows")

# --- 4. MERGE ALL DATA INTO GROUND TRUTH ---
print("\nMerging datasets into ground truth template...")

# Start with ground truth as base
combined_df = ground_truth_df.copy()

# Merge each dataset and track missing values
datasets = [
    ('Spot_Price', spot_df, 'Spot_Price'),
    ('Wind_Forecast', wind_df, 'Wind_Forecast'),
    ('Net_Exchange', exchange_df, 'Net_Exchange'),
    ('Consumption_Forecast', consumption_df, 'Consumption_Forecast')
]

missing_report = {}

for name, df, col in datasets:
    # Merge with indicator to track matches
    combined_df = pd.merge(
        combined_df,
        df[['Timestamp', 'Occurrence', col]],
        on=['Timestamp', 'Occurrence'],
        how='left',
        indicator=f'_merge_{name}'
    )

    # Track missing values (both missing timestamps AND NaN values)
    # Missing timestamp: merge indicator shows 'left_only'
    # NaN value: timestamp exists but value is NaN
    missing_mask = (combined_df[f'_merge_{name}'] == 'left_only') | (combined_df[col].isna())
    missing_count = missing_mask.sum()
    missing_report[name] = {
        'count': missing_count,
        'timestamps': combined_df.loc[missing_mask, 'Timestamp'].tolist()
    }

    print(f"  Merged {name}: {len(df)} rows provided, {missing_count} hours missing/NaN")

    # Drop the merge indicator
    combined_df = combined_df.drop(columns=[f'_merge_{name}'])

# --- 5. VALIDATION REPORT ---
print("\n" + "=" * 60)
print("  VALIDATION REPORT")
print("=" * 60)
print(f"\nTarget Count (Ground Truth): {len(ground_truth_df)} hours")
print(f"Actual Count in Output:      {len(combined_df)} hours")
print(f"Date Range: {combined_df['Timestamp'].min()} to {combined_df['Timestamp'].max()}")

print("\n" + "-" * 60)
print("Missing Values Per Variable:")
print("-" * 60)

for name, info in missing_report.items():
    missing_count = info['count']
    pct = (missing_count / len(ground_truth_df)) * 100
    print(f"\n{name}:")
    print(f"  Missing: {missing_count} hours ({pct:.2f}%)")

    if missing_count > 0 and missing_count <= 50:
        # Show all missing timestamps if <= 50
        print(f"  Missing hours:")
        for ts in info['timestamps']:
            print(f"    {ts}")
    elif missing_count > 50:
        # Show first 10 and last 10 if more than 50
        print(f"  First 10 missing hours:")
        for ts in info['timestamps'][:10]:
            print(f"    {ts}")
        print(f"  ...")
        print(f"  Last 10 missing hours:")
        for ts in info['timestamps'][-10:]:
            print(f"    {ts}")

# --- 6. REORDER COLUMNS ---
# Put columns in logical order: Timestamp, Spot Price, Wind, Exchange, Consumption
column_order = ['Timestamp', 'Spot_Price', 'Wind_Forecast', 'Net_Exchange', 'Consumption_Forecast']
combined_df = combined_df[column_order]

# --- 7. SAVE OUTPUT ---
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("\n" + "=" * 60)
print(f"Saving combined data to:")
print(f"   {output_file}")
combined_df.to_excel(output_file, index=False)

print(f"\nSUCCESS! Combined {target_region} data saved.")
print("=" * 60)
