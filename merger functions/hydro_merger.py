import pandas as pd
import os
import pytz

# --- 1. Configuration ---
# Get the directory where this script is located, then construct paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, '..', 'data', 'hydro_reservoir_reserves')
output_path = os.path.join(script_dir, '..', 'master data files', 'Master_Hydro_Reservoir.xlsx')

# We include 2014 to get the data for the first days of 2015 and 2026 for the last days of 2025
years_to_load = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
areas = ['SE1', 'SE2', 'SE3', 'SE4']

# --- 2. Create the Hourly Ground Truth (2015-2025) ---
# 11 years: 96,432 hours (includes 3 leap years: 2016, 2020, 2024)
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2015-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h', tz=tz)
master_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})

print("--- Starting Hydro Weekly-to-Hourly Extrapolation with 2014 and 2026 Padding ---")

for area in areas:
    area_weekly_data = []

    for year in years_to_load:
        # Load reservoir files including 2014 and 2026
        file_name = f"reservoir_{area}_{year}.xlsx"
        path = os.path.join(folder_path, file_name)

        if not os.path.exists(path):
            print(f"WARNING - Missing: {file_name}")
            continue

        # Read weekly capacity data starting from row 4
        df = pd.read_excel(path, skiprows=3)

        # 'Week start' (Col B) and 'Current capacity (GWh)' (Col E)
        df = df[['Week start', 'Current capacity (GWh)']].copy()
        df['Timestamp'] = pd.to_datetime(df['Week start'])

        area_weekly_data.append(df[['Timestamp', 'Current capacity (GWh)']])

    if area_weekly_data:
        # Combine all years including 2014 and 2026
        combined_area_weekly = pd.concat(area_weekly_data).sort_values('Timestamp')

        # Merge into the target 2015-2025 skeleton
        # We perform an 'outer' merge initially so 2014 and 2026 dates are visible for filling
        temp_merged = pd.merge(master_df, combined_area_weekly, on='Timestamp', how='outer').sort_values('Timestamp')

        # EXTRAPOLATION: Forward fill handles start of 2015, backward fill handles end of 2025
        temp_merged[f'{area}_Hydro_Reserves'] = temp_merged['Current capacity (GWh)'].ffill().bfill()

        # Trim back to the target 2015-2025 window
        trimmed_data = temp_merged[temp_merged['Timestamp'].isin(master_df['Timestamp'])]

        # Add the finalized area column to master_df
        master_df[f'{area}_Hydro_Reserves'] = trimmed_data[f'{area}_Hydro_Reserves'].values
        print(f"OK - Extrapolated {area} (Gaps at start and end of period filled).")

# --- 3. Final Export ---
if len(master_df) == 96432:
    master_df.to_excel(output_path, index=False)
    print(f"\nSUCCESS! Hydro Master saved with {len(master_df)} rows.")
    print(f"Saved to: {output_path}")
    print(f"Data spans 2015-2025 (11 years of hourly hydro reservoir data).")
else:
    print(f"WARNING - Row count mismatch: Expected 96432, got {len(master_df)}.")