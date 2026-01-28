import pandas as pd
import os
import pytz

# --- 1. CONFIGURATION ---
# Change this to 'SE1', 'SE2', 'SE3', or 'SE4' to switch regions
target_region = 'SE1'

# Extract region number (e.g., '1' from 'SE1')
region_number = target_region[-1]

# Congestion threshold (Sandberg uses 1 öre; prices are in EUR/MWh)
CONGESTION_EPSILON_EUR = 0.01

# Define trading partners for each region (based on direct transmission connections)
TRADING_PARTNERS = {
    'SE1': ['FI', 'NO4', 'SE2'],
    'SE2': ['NO3', 'NO4', 'SE1', 'SE3'],
    'SE3': ['DK1', 'FI', 'NO1', 'SE2', 'SE4'],
    'SE4': []   # To be defined later
}

# Get script directory for absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')

# File paths for input data (absolute paths based on script location)
spot_price_file = os.path.join(project_root, 'master data files', '2015-2025', 'Spot_Prices_2015_2025.xlsx')
wind_forecast_file = os.path.join(project_root, f'Verified_S{region_number}_Wind_Forecast_2015_2025.xlsx')
exchange_file = os.path.join(project_root, f'Verified_{target_region}_Exchange_2015_2025.xlsx')
consumption_file = os.path.join(project_root, 'Master_Consumption_2015_2025.xlsx')
hydro_file = os.path.join(project_root, 'master data files', 'Master_Hydro_Reservoir.xlsx')

# Output file
output_file = os.path.join(project_root, 'master data files', '2015-2025', f'Combined_{target_region}_Data_2015_2025.xlsx')

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

# --- 3. LOAD ALL DATA FILES & CREATE DUMMIES ---
print("\nLoading data files...")

# Load Spot Prices (2015-2025) - keep all price columns for bottleneck calculation
print(f"  Loading spot prices...")
spot_df_full = pd.read_excel(spot_price_file)
spot_df_full['Timestamp'] = pd.to_datetime(spot_df_full['Timestamp'])

# Identify all price columns
price_columns = [col for col in spot_df_full.columns if col.endswith('_Price (EUR)')]

# Keep Timestamp and all price columns
spot_df = spot_df_full[['Timestamp'] + price_columns].copy()

# Create Spot_Price column for target region
target_price_col = f'{target_region}_Price (EUR)'
if target_price_col not in spot_df.columns:
    raise ValueError(f"Price column {target_price_col} not found in spot price data!")
spot_df['Spot_Price'] = spot_df[target_price_col]

# Add Occurrence for DST handling
spot_df['Occurrence'] = spot_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(spot_df)} rows with {len(price_columns)} price areas")

# --- 3b. CREATE BOTTLENECK DUMMIES ---
print(f"\n  Creating {target_region} bottleneck dummies (EPSILON = {CONGESTION_EPSILON_EUR} EUR/MWh)...")

# Get trading partners for the target region
trading_partners = TRADING_PARTNERS.get(target_region, [])

if not trading_partners:
    print(f"    WARNING: No trading partners defined for {target_region}, skipping bottleneck dummies...")
    bneck_cols = []
else:
    bneck_cols = []

    for other_area in trading_partners:
        # Construct the price column name
        other_col = f'{other_area}_Price (EUR)'

        # Skip if the price column doesn't exist in the data
        if other_col not in price_columns:
            print(f"    WARNING: {other_col} not found in spot price data, skipping...")
            continue

        # Create dummy name
        dummy_name = f'BNECK_{target_region}_{other_area}'

        # Calculate price difference
        price_diff = (spot_df['Spot_Price'] - spot_df[other_col]).abs()

        # Create dummy: 1 if |diff| > EPSILON, 0 otherwise
        # Preserve NaN if either price is NaN
        dummy = (price_diff > CONGESTION_EPSILON_EUR).astype(float)
        dummy = dummy.where(spot_df['Spot_Price'].notna() & spot_df[other_col].notna(), pd.NA)

        spot_df[dummy_name] = dummy
        bneck_cols.append(dummy_name)

    print(f"    Created {len(bneck_cols)} bottleneck dummies for {target_region}'s trading partners:")
    for col in bneck_cols:
        print(f"      {col}")

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

# Load Hydro Reserves (2015-2025)
print(f"  Loading hydro reserves for {target_region}...")
hydro_df = pd.read_excel(hydro_file)
hydro_df['Timestamp'] = pd.to_datetime(hydro_df['Timestamp'])
hydro_column = f'{target_region}_Hydro_Reserves'
hydro_df = hydro_df[['Timestamp', hydro_column]].copy()
hydro_df.rename(columns={hydro_column: 'Hydro_Reserves'}, inplace=True)
# Add Occurrence for DST handling
hydro_df['Occurrence'] = hydro_df.groupby('Timestamp').cumcount()
print(f"    Loaded {len(hydro_df)} rows")

# --- 4. MERGE ALL DATA INTO GROUND TRUTH ---
print("\nMerging datasets into ground truth template...")

# Start with ground truth as base
combined_df = ground_truth_df.copy()

# Merge each dataset and track missing values
# Build datasets list: start with Spot_Price, then bottleneck dummies, then other variables
datasets = [
    ('Spot_Price', spot_df, 'Spot_Price')
]

# Add bottleneck dummy datasets
for dummy_col in bneck_cols:
    datasets.append((dummy_col, spot_df, dummy_col))

# Add remaining variables
datasets.extend([
    ('Wind_Forecast', wind_df, 'Wind_Forecast'),
    ('Net_Exchange', exchange_df, 'Net_Exchange'),
    ('Consumption_Forecast', consumption_df, 'Consumption_Forecast'),
    ('Hydro_Reserves', hydro_df, 'Hydro_Reserves')
])

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
# Put columns in logical order: Timestamp, Spot_Price, bottleneck dummies, then other variables
column_order = ['Timestamp', 'Spot_Price'] + bneck_cols + ['Wind_Forecast', 'Net_Exchange', 'Consumption_Forecast', 'Hydro_Reserves']
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
