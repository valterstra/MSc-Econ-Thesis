import pandas as pd
import os
import re
import pytz

# --- 1. EASY SWITCH CONFIGURATION ---
# Change this to 'SE1', 'SE2', 'SE3', or 'SE4' to switch regions
target_region = 'SE4'

# Extracts the number from the region name (e.g., '1' from 'SE1')
region_number = target_region[-1]

# UPDATED: Matches your folder naming convention (e.g., data/exchange_forecast_s1)
folder_path = f'data/exchange_forecast_s{region_number}'
output_filename = f'Verified_{target_region}_Exchange_2015_2025.xlsx'
column_name = f'{target_region}_Total_Net_Exchange'

years_to_include = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
target_column_label = "Total Net Exchange"

# --- 2. Create the Ground Truth ---
# Ensuring daylight savings (23h/25h) are correctly handled
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2015-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h', tz=tz)

ground_truth_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})
# Sequential ID for duplicate hours in October
ground_truth_df['Occurrence'] = ground_truth_df.groupby('Timestamp').cumcount()

# --- 3. Process Regional Files ---
all_days_data = []

if not os.path.exists(folder_path):
    print(f"❌ Error: Folder '{folder_path}' not found.")
else:
    # Filter files for 2015-2025 and remove duplicates
    raw_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    unique_map = {}
    for f in raw_files:
        match = re.search(r'(\d{4})-\d{2}-\d{2}', f)
        if match and match.group(1) in years_to_include:
            date_key = match.group(0)
            if date_key not in unique_map or '(' not in f:
                unique_map[date_key] = f

    files = sorted(unique_map.values())
    print(f"--- Processing {target_region} from {folder_path} ---")
    print(f"Processing {len(files)} files for {target_region}...")

    # --- Check for Missing Days ---
    expected_dates = pd.date_range(start='2015-01-01', end='2025-12-31', freq='D')
    expected_date_strs = set(expected_dates.strftime('%Y-%m-%d'))

    actual_date_strs = set()
    for file in files:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', file)
        if match:
            actual_date_strs.add(match.group(0))

    missing_dates = sorted(expected_date_strs - actual_date_strs)
    if missing_dates:
        print(f"\n⚠️ MISSING FILES: {len(missing_dates)} days have no file in the folder:")
        for date in missing_dates:
            print(f"  {date}")
        print()
    else:
        print("✅ All expected day files are present in the folder.\n")

    for file in files:
        path = os.path.join(folder_path, file)
        try:
            # Search Row 5 (index 4) for the target column label
            header_row = pd.read_excel(path, header=None, skiprows=4, nrows=1)

            target_col_index = None
            for i, cell_value in enumerate(header_row.iloc[0]):
                if target_column_label in str(cell_value):
                    target_col_index = i
                    break

            if target_col_index is None:
                continue

            # Read hourly data starting at Row 6
            df_raw = pd.read_excel(path, header=None, skiprows=5, nrows=30,
                                   usecols=[0, target_col_index], decimal=',')

            # Filter for valid rows
            df = df_raw[df_raw[0].astype(str).str.contains(' - ')].copy()
            df.columns = ['Period', column_name]

            # Construct Timestamp
            date_str = re.search(r'\d{4}-\d{2}-\d{2}', file).group(0)
            df['Timestamp'] = pd.to_datetime(date_str + ' ' + df['Period'].str[:5])

            # Handle October duplicate hours
            df['Occurrence'] = df.groupby('Timestamp').cumcount()

            all_days_data.append(df[['Timestamp', 'Occurrence', column_name]])

        except Exception as e:
            print(f"Error reading {file}: {e}")

# --- 4. Merge into Ground Truth ---
if all_days_data:
    merged_data = pd.concat(all_days_data, ignore_index=True)
    final_df = pd.merge(ground_truth_df, merged_data, on=['Timestamp', 'Occurrence'],
                         how='left', indicator=True)

    # Reporting
    gaps = final_df[final_df['_merge'] == 'left_only']
    print("\n" + "=" * 40)
    print(f"Target Count: {len(ground_truth_df)}")
    print(f"Actual Count: {len(final_df)}")

    if not gaps.empty:
        print(f"⚠️ MISSING DATA: {len(gaps)} hours are empty (missing files or columns).")
        print("\nMissing hours:")
        print(gaps[['Timestamp']].to_string(index=False))
    else:
        print(f"✅ PERFECT ALIGNMENT: All {len(ground_truth_df)} hours are present in the structure.")

    # Clean up and export
    final_df.drop(columns=['Occurrence', '_merge']).to_excel(output_filename, index=False)
    print(f"✨ Verified file saved: {output_filename}")
else:
    print("No data found to merge.")