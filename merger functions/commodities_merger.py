import pandas as pd
import os
import re
import pytz

# Configuration
folder_path = 'data/commodities'
years_to_include = ['2015', '2016', '2017', '2018', '2019', '2020', '2021',
                    '2022', '2023', '2024', '2025', '2026']
output_filename = 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx'

print("=" * 60)
print("   LIGHT CRUDE OIL COMMODITIES MERGER")
print("=" * 60)
print(f"Source folder: {folder_path}")
print(f"Output file: {output_filename}")
print(f"Years to include: {', '.join(years_to_include)}")
print()

# Step 1: Create Ground Truth Skeleton
print("Creating ground truth skeleton...")
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2015-01-01 00:00:00', end='2025-12-31 23:00:00',
                         freq='h', tz=tz)
ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()

print(f"Ground truth created: {len(ground_truth_df)} hourly rows")
print(f"Date range: {ground_truth_df['Timestamp'].min()} to {ground_truth_df['Timestamp'].max()}")
print()

# Step 2: Discover and Filter CSV Files
print("Discovering CSV files...")
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"Total CSV files found: {len(all_files)}")

# Filter files by year
pattern = re.compile(r'LIGHT\.CMD-USD_Hour_(\d{4})-\d{2}-\d{2}_to_(\d{4})-\d{2}-\d{2}_UTC\.csv')
filtered_files = []

for file in all_files:
    match = pattern.match(file)
    if match:
        start_year, end_year = match.groups()
        if start_year in years_to_include or end_year in years_to_include:
            filtered_files.append(file)

filtered_files.sort()
print(f"Files matching year filter: {len(filtered_files)}")
print()

# Step 3: Process CSV Files
print("Processing CSV files...")
all_data = []

for i, filename in enumerate(filtered_files, 1):
    filepath = os.path.join(folder_path, filename)

    try:
        # Read CSV file
        df = pd.read_csv(filepath, decimal='.')

        # Parse UTC timestamp and convert to Europe/Stockholm timezone
        df['Timestamp'] = pd.to_datetime(df['UTC'], format='%d.%m.%Y %H:%M:%S.000 UTC', utc=True)
        df['Timestamp'] = df['Timestamp'].dt.tz_convert('Europe/Stockholm')
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

        # Add Occurrence column for DST handling
        df['Occurrence'] = df.groupby('Timestamp').cumcount()

        # Select relevant columns
        df = df[['Timestamp', 'Occurrence', 'Open', 'High', 'Low', 'Close', 'Volume']]

        all_data.append(df)

        if i % 20 == 0:
            print(f"  Processed {i}/{len(filtered_files)} files...")

    except Exception as e:
        print(f"  ⚠️ Error processing {filename}: {e}")

print(f"Successfully processed {len(all_data)} files")
print()

# Step 4: Concatenate All Data
print("Concatenating data from all files...")
merged_data = pd.concat(all_data, ignore_index=True)
print(f"Total rows before merge: {len(merged_data)}")

# Remove duplicates (files may have overlapping dates)
merged_data = merged_data.drop_duplicates(subset=['Timestamp', 'Occurrence'], keep='first')
print(f"Total rows after removing duplicates: {len(merged_data)}")
print()

# Step 5: Merge with Ground Truth
print("Merging with ground truth skeleton...")
final_df = pd.merge(
    ground_truth_df,
    merged_data,
    on=['Timestamp', 'Occurrence'],
    how='left',
    indicator=True
)

print(f"Final dataset rows: {len(final_df)}")
print()

# Step 6: Data Integrity Reporting
gaps = final_df[final_df['_merge'] == 'left_only']

print("=" * 60)
print("   DATA INTEGRITY REPORT")
print("=" * 60)
print(f"Target Rows (Ground Truth): {len(ground_truth_df)}")
print(f"Actual Rows in Output: {len(final_df)}")
print(f"Data Rows Matched: {len(final_df[final_df['_merge'] == 'both'])}")
print()

if not gaps.empty:
    print(f"WARNING: {len(gaps)} hours are missing data.")
    print()

    # Show sample of missing timestamps
    if len(gaps) <= 50:
        print("Missing timestamps:")
        print(gaps[['Timestamp']].to_string(index=False))
    else:
        print(f"First 20 missing timestamps:")
        print(gaps[['Timestamp']].head(20).to_string(index=False))
        print(f"...")
        print(f"Last 20 missing timestamps:")
        print(gaps[['Timestamp']].tail(20).to_string(index=False))
else:
    print("SUCCESS: Perfect match with ground truth - no missing data!")

print()
print("=" * 60)
print()

# Step 7: Export to Excel
print("Exporting to Excel...")
final_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
output_df = final_df.drop(columns=['Occurrence', '_merge'])[final_cols]

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

output_df.to_excel(output_filename, index=False)

print(f"SUCCESS: File saved to: {output_filename}")
print()

# Final verification
print("=" * 60)
print("   FINAL VERIFICATION")
print("=" * 60)
print(f"Output shape: {output_df.shape}")
print(f"Columns: {output_df.columns.tolist()}")
print(f"Date range: {output_df['Timestamp'].min()} to {output_df['Timestamp'].max()}")
print()
print("Missing values per column:")
print(output_df.isnull().sum())
print()
print("=" * 60)
print("   MERGER COMPLETE")
print("=" * 60)
