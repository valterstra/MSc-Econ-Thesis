import pandas as pd
import os
import re
import pytz

# --- 1. Configuration ---
folder_path = 'data/consumption_forecast'
years_to_include = ['2021', '2022', '2023', '2024']
output_filename = 'Master_Consumption_2021_2024.xlsx'

# Comprehensive list of all TSOs and Bidding Zones found in images
target_regions = [
    '50Hz', 'AMP', 'TBW', 'TTG',
    'DK1', 'DK2', 'FI', 'NO',
    'NO1', 'NO2', 'NO3', 'NO4', 'NO5',
    'SE1', 'SE2', 'SE3', 'SE4', 'DE'
]

# --- 2. Create the Ground Truth (35,064 rows) ---
# Handles CET/CEST transitions (23h/25h) for a perfect 4-year baseline
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2021-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h', tz=tz)

ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()

# --- 3. Process Consumption Files ---
all_days_data = []

if not os.path.exists(folder_path):
    print(f"❌ Error: Folder '{folder_path}' not found.")
else:
    # Filter files and exclude duplicates/intermediate saves
    raw_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    unique_map = {}
    for f in raw_files:
        match = re.search(r'(\d{4})-\d{2}-\d{2}', f)
        if match and match.group(1) in years_to_include:
            date_key = match.group(0)
            if date_key not in unique_map or '(' not in f:
                unique_map[date_key] = f

    files = sorted(unique_map.values())
    print(f"--- Extracting {len(target_regions)} zones from {len(files)} files ---")

    for file in files:
        path = os.path.join(folder_path, file)
        try:
            # Step A: Find the header row (it varies between Row 3, 4, and 5)
            temp_df = pd.read_excel(path, header=None, nrows=10)
            header_row_idx = None
            for idx, row in temp_df.iterrows():
                if any('(MWh)' in str(val) for val in row):
                    header_row_idx = idx
                    break

            if header_row_idx is None:
                continue

            # Step B: Map column labels to indices
            header_list = [str(val).strip() for val in temp_df.iloc[header_row_idx]]
            col_map = {'Timestamp': 0}
            for region in target_regions:
                for i, label in enumerate(header_list):
                    if label.startswith(region) and '(MWh)' in label:
                        col_map[region] = i
                        break

            # Step C: Read data and filter for hourly rows
            df_raw = pd.read_excel(path, header=None, skiprows=header_row_idx + 1, nrows=30,
                                   usecols=list(col_map.values()), decimal=',')

            inv_map = {v: k for k, v in col_map.items()}
            df_raw.columns = [inv_map[i] for i in range(len(df_raw.columns))]
            df = df_raw[df_raw['Timestamp'].astype(str).str.contains(' - ')].copy()

            # Step D: Standardize Timestamp and handle October duplicates
            date_str = re.search(r'\d{4}-\d{2}-\d{2}', file).group(0)
            df['Timestamp'] = pd.to_datetime(date_str + ' ' + df['Timestamp'].str[:5])
            df['Occurrence'] = df.groupby('Timestamp').cumcount()

            all_days_data.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")

# --- 4. Final Alignment & Reporting ---
if all_days_data:
    merged_data = pd.concat(all_days_data, ignore_index=True)

    # Perform the outer join to the 35,064-row skeleton
    final_df = pd.merge(ground_truth_df, merged_data, on=['Timestamp', 'Occurrence'],
                        how='left', indicator=True)

    # --- REPORTING BLOCK ---
    gaps = final_df[final_df['_merge'] == 'left_only']

    print("\n" + "=" * 40)
    print("      CONSUMPTION INTEGRITY REPORT")
    print("=" * 40)
    print(f"Target Rows (Ground Truth): {len(ground_truth_df)}")
    print(f"Actual Rows in Output:    {len(final_df)}")

    if not gaps.empty:
        print(f"⚠️ Alert: {len(gaps)} hours are missing data.")
        print("First few gaps:")
        print(gaps['Timestamp'].head(5))
    else:
        print("✅ Perfect match with ground truth.")

    # Ensure all target columns exist (filled with NA if missing)
    for col in target_regions:
        if col not in final_df.columns:
            final_df[col] = pd.NA

    # --- THE FIX ---
    # 1. Drop the helper columns from final_df first
    final_df = final_df.drop(columns=['Occurrence', '_merge'])

    # 2. Select and order your desired columns
    final_cols = ['Timestamp'] + target_regions

    # 3. Save the resulting DataFrame
    final_df[final_cols].to_excel(output_filename, index=False)

    print(f"\n✨ Final file saved: {output_filename}")