import pandas as pd
import os
import pytz

# --- 1. Configuration ---
folder_path = 'data/spot_price'
years_to_include = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
output_filename = 'master data files/2015-2025/Spot_Prices_2015_2025.xlsx'

# --- 2. Create the Timezone-Aware Ground Truth ---
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2015-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h', tz=tz)

# Create Ground Truth and add a 'Sequence' to distinguish duplicate hours in October
ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()

# --- 3. Load and Process User Files ---
all_extracted_data = []
files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx') and any(y in f for y in years_to_include)])

for file in files:
    path = os.path.join(folder_path, file)
    df = pd.read_excel(path, decimal=',', header=[0, 1])  #

    # Clean headers and timestamps
    df.columns = [f"{c[0]}_{c[1]}".strip('_') if "Unnamed" not in str(c[1]) else c[0] for c in df.columns]  #
    df['Timestamp'] = pd.to_datetime(df['Delivery Start (CET)'], dayfirst=True)  #

    # Identify price columns
    price_cols = [col for col in df.columns if 'Price' in col]  #

    # Check if data is sub-hourly (e.g., 15-minute intervals)
    df['Timestamp_Hour'] = df['Timestamp'].dt.floor('h')
    rows_per_hour = df.groupby('Timestamp_Hour').size()

    # Detect if consistently sub-hourly (more than 10% of hours have multiple readings)
    hours_with_multiple = (rows_per_hour > 1).sum()
    total_hours = len(rows_per_hour)
    is_subhourly = (hours_with_multiple / total_hours) > 0.1

    # Special handling for 2025 Oct-Dec quarterly data
    has_2025_q4_data = (df['Timestamp'] >= '2025-10-01').any()

    if is_subhourly:
        # Aggregate sub-hourly data to hourly by averaging price columns
        print(f"  {file}: Detected sub-hourly data, aggregating to hourly averages...")

        if has_2025_q4_data:
            # For 2025 Oct-Dec: aggregate quarterly readings to hourly
            # Special handling for DST duplicate on Oct 26, 2025 (8 readings for 02:00)
            dst_hour = pd.Timestamp('2025-10-26 02:00:00')
            dst_mask = (df['Timestamp_Hour'] == dst_hour)

            # For DST duplicate hour: first 4 readings → Occurrence 0, next 4 → Occurrence 1
            # For other hours: all get Occurrence 0
            df['Occurrence'] = 0
            if dst_mask.any():
                df.loc[dst_mask, 'Occurrence'] = df[dst_mask].groupby('Timestamp_Hour').cumcount() // 4

            # Aggregate by both Timestamp_Hour and Occurrence
            hourly_df = df.groupby(['Timestamp_Hour', 'Occurrence'])[price_cols].mean().reset_index()
            hourly_df.rename(columns={'Timestamp_Hour': 'Timestamp'}, inplace=True)

            all_extracted_data.append(hourly_df[['Timestamp', 'Occurrence'] + price_cols])
        else:
            # Normal sub-hourly handling with DST duplicate preservation
            readings_per_occurrence = rows_per_hour.mode()[0]  # Most common count (e.g., 4 for 15-min)
            df['Occurrence'] = df.groupby('Timestamp_Hour').cumcount() // readings_per_occurrence

            # Aggregate within each occurrence separately (preserves DST duplicates)
            hourly_df = df.groupby(['Timestamp_Hour', 'Occurrence'])[price_cols].mean().reset_index()
            hourly_df.rename(columns={'Timestamp_Hour': 'Timestamp'}, inplace=True)

            all_extracted_data.append(hourly_df[['Timestamp', 'Occurrence'] + price_cols])
    else:
        # Hourly data - just assign occurrence for DST duplicates
        df['Occurrence'] = df.groupby('Timestamp').cumcount()
        all_extracted_data.append(df[['Timestamp', 'Occurrence'] + price_cols])

merged_user_data = pd.concat(all_extracted_data, ignore_index=True)

# Filter to only include data from January 1, 2015 onwards
merged_user_data = merged_user_data[merged_user_data['Timestamp'] >= '2015-01-01']

# --- 4. Precise Merge ---
# Merging on both Timestamp AND Occurrence prevents the 8-row "shadow" expansion
final_verified_df = pd.merge(
    ground_truth_df,
    merged_user_data,
    on=['Timestamp', 'Occurrence'],
    how='outer',
    indicator=True
)

# --- 5. Corrected Integrity Report ---
gaps = final_verified_df[final_verified_df['_merge'] == 'left_only']
extras = final_verified_df[final_verified_df['_merge'] == 'right_only']

print("\n" + "=" * 40)
print("      DATA INTEGRITY REPORT")
print("=" * 40)
print(f"Expected Rows: {len(ground_truth_df)}")
print(f"Final Row Count: {len(final_verified_df)}")

if len(final_verified_df) == len(ground_truth_df) and gaps.empty and extras.empty:
    print(f"PERFECT MATCH: {len(ground_truth_df)} rows aligned exactly.")
else:
    if not gaps.empty:
        print(f"\nGAPS: {len(gaps)} hours missing from files.")
        if len(gaps) <= 20:
            print("Missing hours:")
            for idx, row in gaps.iterrows():
                print(f"  {row['Timestamp']}")
    if not extras.empty:
        print(f"\nEXTRAS: {len(extras)} unexpected rows found in files.")
        if len(extras) <= 20:
            print("Extra hours:")
            for idx, row in extras.iterrows():
                print(f"  {row['Timestamp']}")

# Save and Clean
final_verified_df.drop(columns=['Occurrence', '_merge']).to_excel(output_filename, index=False)