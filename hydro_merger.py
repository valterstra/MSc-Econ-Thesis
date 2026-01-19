import pandas as pd
import os
import pytz

# --- 1. Configuration ---
folder_path = 'data/hydro_reservoir_reserves'
# We include 2020 to get the data for the first three days of 2021
years_to_load = [2020, 2021, 2022, 2023, 2024]
output_filename = 'Master_Hydro_Reservoir.xlsx'
areas = ['SE1', 'SE2', 'SE3', 'SE4']

# --- 2. Create the Hourly Ground Truth (2021-2024 only) ---
# This remains our target window for 35,064 rows
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2021-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h', tz=tz)
master_df = pd.DataFrame({'Timestamp': gt_range.tz_localize(None)})

print("--- Starting Hydro Weekly-to-Hourly Extrapolation with 2020 Padding ---")

for area in areas:
    area_weekly_data = []

    for year in years_to_load:
        # Load reservoir files including 2020
        file_name = f"reservoir_{area}_{year}.xlsx"
        path = os.path.join(folder_path, file_name)

        if not os.path.exists(path):
            print(f"⚠️ Missing: {file_name}")
            continue

        # Read weekly capacity data starting from row 4
        df = pd.read_excel(path, skiprows=3)

        # 'Week start' (Col B) and 'Current capacity (GWh)' (Col E)
        df = df[['Week start', 'Current capacity (GWh)']].copy()
        df['Timestamp'] = pd.to_datetime(df['Week start'])

        area_weekly_data.append(df[['Timestamp', 'Current capacity (GWh)']])

    if area_weekly_data:
        # Combine all years including 2020
        combined_area_weekly = pd.concat(area_weekly_data).sort_values('Timestamp')

        # Merge into the target 2021-2024 skeleton
        # We perform an 'outer' merge initially so 2020 dates are visible to 'ffill'
        temp_merged = pd.merge(master_df, combined_area_weekly, on='Timestamp', how='outer').sort_values('Timestamp')

        # EXTRAPOLATION: Forward fill handles the gap between 2020-12-28 and 2021-01-04
        temp_merged[f'{area}_Hydro_Reserves'] = temp_merged['Current capacity (GWh)'].ffill()

        # Trim back to the target 2021-2024 window
        trimmed_data = temp_merged[temp_merged['Timestamp'].isin(master_df['Timestamp'])]

        # Add the finalized area column to master_df
        master_df[f'{area}_Hydro_Reserves'] = trimmed_data[f'{area}_Hydro_Reserves'].values
        print(f"✅ Extrapolated {area} (Gap at start of 2021 is now filled).")

# --- 3. Final Export ---
if len(master_df) == 35064:
    master_df.to_excel(output_filename, index=False)
    print(f"\n✨ Success! Hydro Master saved with {len(master_df)} rows.")
    print(f"The first rows of 2021 now contain capacity values from the end of 2020.")
else:
    print(f"⚠️ Row count mismatch: {len(master_df)}.")