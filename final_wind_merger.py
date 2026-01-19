import pandas as pd
import os

# --- 1. Configuration ---
verified_files = [
    'Verified_S1_Wind_Forecast_2021_2024.xlsx',
    'Verified_S2_Wind_Forecast_2021_2024.xlsx',
    'Verified_S3_Wind_Forecast_2021_2024.xlsx',
    'Verified_S4_Wind_Forecast_2021_2024.xlsx'
]
output_filename = 'Master_Wind_Forecast_Merged_2021_2024.xlsx'

column_list = []

print("--- Starting Column-Based Master Merge ---")

# --- 2. Load and Prepare Columns ---
for i, file in enumerate(verified_files):
    if not os.path.exists(file):
        print(f"⚠️ Warning: {file} not found. Skipping.")
        continue

    print(f"Loading: {file}")
    df = pd.read_excel(file)

    if i == 0:
        # For S1, we keep everything (Day, Month, Year, Start Time, and S1_Wind)
        column_list.append(df)
    else:
        # For S2, S3, and S4, we only take the VERY LAST column (the wind data)
        # This prevents 'Day', 'Month', etc., from duplicating and causing MergeErrors
        wind_col_only = df.iloc[:, [-1]]
        column_list.append(wind_col_only)

# --- 3. Concatenate Side-by-Side ---
if column_list:
    # axis=1 tells pandas to stack these horizontally
    final_output = pd.concat(column_list, axis=1)

    # --- 4. Final Row Check ---
    print("\n" + "=" * 40)
    row_count = len(final_output)
    print(f"Final Row Count: {row_count}")

    if row_count == 35064:
        print("✅ Success: Master file matches the 35,064 hourly baseline.")
    else:
        print(f"⚠️ Row mismatch: Found {row_count} rows. Check for duplicates in source files.")

    # 5. Export
    final_output.to_excel(output_filename, index=False)
    print(f"✨ Master file saved as: {output_filename}")
else:
    print("❌ Error: No data was loaded.")