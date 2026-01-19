import pandas as pd
import os

# --- 1. Configuration ---
# List of the verified exchange files you have generated
verified_files = [
    'Verified_SE1_Exchange_2021_2024.xlsx',
    'Verified_SE2_Exchange_2021_2024.xlsx',
    'Verified_SE3_Exchange_2021_2024.xlsx',
    'Verified_SE4_Exchange_2021_2024.xlsx'
]
output_filename = 'Master_Exchange_Merged_2021_2024.xlsx'

column_list = []

print("--- Starting Column-Based Master Exchange Merge ---")

# --- 2. Load and Prepare Columns ---
for i, file in enumerate(verified_files):
    if not os.path.exists(file):
        print(f"⚠️ Warning: {file} not found. Skipping.")
        continue

    print(f"Loading: {file}")
    df = pd.read_excel(file)

    if i == 0:
        # Keep taxonomy + SE1 data
        column_list.append(df)
    else:
        # For SE2-SE4, take only the last column (the exchange data)
        # This prevents taxonomy duplication
        exchange_col_only = df.iloc[:, [-1]]
        column_list.append(exchange_col_only)

# --- 3. Concatenate Horizontally ---
if column_list:
    # axis=1 stacks columns side-by-side
    final_output = pd.concat(column_list, axis=1)

    # --- 4. Quality Check ---
    print("\n" + "=" * 40)
    row_count = len(final_output)
    print(f"Final Row Count: {row_count}")

    if row_count == 35064:
        print("✅ Success: Master Exchange aligns with the 35,064 baseline.")
    else:
        print(f"⚠️ Row mismatch: Found {row_count} rows.")

    # 5. Save
    final_output.to_excel(output_filename, index=False)
    print(f"✨ Master Exchange file saved: {output_filename}")
else:
    print("❌ Error: No files were merged.")