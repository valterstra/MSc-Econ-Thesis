import pandas as pd
import os
import pytz

# --- 1. Configuration ---
folder_path = 'data/spot_price'
years_to_include = ['2021', '2022', '2023', '2024']
output_filename = 'Spot_Prices.xlsx'

# --- 2. Create the Timezone-Aware Ground Truth ---
tz = pytz.timezone('Europe/Stockholm')
gt_range = pd.date_range(start='2021-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h', tz=tz)

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

    # Create the same 'Occurrence' count to match the Ground Truth exactly
    df['Occurrence'] = df.groupby('Timestamp').cumcount()

    price_cols = [col for col in df.columns if 'Price' in col]  #
    all_extracted_data.append(df[['Timestamp', 'Occurrence'] + price_cols])

merged_user_data = pd.concat(all_extracted_data, ignore_index=True)

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
    print("✅ PERFECT MATCH: 35,064 rows aligned exactly.")
else:
    if not gaps.empty:
        print(f"⚠️ GAPS ({len(gaps)}): Hours missing from files.")
    if not extras.empty:
        print(f"⚠️ EXTRAS ({len(extras)}): Unexpected rows found in files.")

# Save and Clean
final_verified_df.drop(columns=['Occurrence', '_merge']).to_excel(output_filename, index=False)