import pandas as pd
import os
import re
import pytz

# --- 1. Configuration ---
folder_path = 'data/production_forecast_s4'
years_to_include = ['2021', '2022', '2023', '2024']
output_filename = 'Verified_S4_Wind_Forecast_2021_2024.xlsx'

# --- 2. Create the Ground Truth (35,064 rows) ---
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2021-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h', tz=tz)

ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
# Sequential ID for duplicate hours in October
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()

# --- 3. Process S1 Files ---
all_days_data = []

# Filter files for 2021-2024 and remove duplicates like (1).xlsx
raw_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
unique_map = {}
for f in raw_files:
    match = re.search(r'(\d{4})-\d{2}-\d{2}', f)
    if match and match.group(1) in years_to_include:
        date_key = match.group(0)
        if date_key not in unique_map or '(' not in f:
            unique_map[date_key] = f

files = sorted(unique_map.values())
print(f"Processing {len(files)} files for S1...")

for file in files:
    path = os.path.join(folder_path, file)
    try:
        # Check Row 4 for "Wind Onshore" in Col B or C
        header = pd.read_excel(path, header=None, skiprows=3, nrows=1, usecols=[1, 2])
        labels = [str(header.iloc[0, 0]), str(header.iloc[0, 1])]

        target_col = None
        if "Wind Onshore" in labels[0]:
            target_col = 1
        elif "Wind Onshore" in labels[1]:
            target_col = 2

        if target_col is None:
            # If column is missing, we skip data extraction but row stays empty in merge
            continue

        # Read hourly data (Rows 6-30)
        df_raw = pd.read_excel(path, header=None, skiprows=5, nrows=30, usecols=[0, target_col], decimal=',')
        # Filter for rows with time periods like "00:00 - 01:00"
        df = df_raw[df_raw[0].astype(str).str.contains(' - ')].copy()
        df.columns = ['Period', 'S1_Wind']

        # Construct Timestamp
        date_str = re.search(r'\d{4}-\d{2}-\d{2}', file).group(0)
        df['Timestamp'] = pd.to_datetime(date_str + ' ' + df['Period'].str[:5])

        # Add Occurrence to handle October duplicates
        df['Occurrence'] = df.groupby('Timestamp').cumcount()

        all_days_data.append(df[['Timestamp', 'Occurrence', 'S1_Wind']])

    except Exception as e:
        print(f"Error reading {file}: {e}")

# --- 4. Merge and Audit ---
merged_s1 = pd.concat(all_days_data, ignore_index=True)

# Force S1 data into the 35,064-row Ground Truth
final_df = pd.merge(
    ground_truth_df,
    merged_s1,
    on=['Timestamp', 'Occurrence'],
    how='left',
    indicator=True
)

# --- 5. Reporting ---
gaps = final_df[final_df['_merge'] == 'left_only']
print("\n" + "=" * 40)
print(f"Target Count: {len(ground_truth_df)}")
print(f"Actual Count: {len(final_df)}")

if not gaps.empty:
    print(f"⚠️ MISSING DATA: {len(gaps)} hours are empty (missing files or columns).")
else:
    print("✅ PERFECT ALIGNMENT: All 35,064 hours are present in the structure.")

# Save final result
final_df.drop(columns=['Occurrence', '_merge']).to_excel(output_filename, index=False)