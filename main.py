import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


# ============================================
# 1. DATA LOADING FUNCTIONS
# ============================================

def load_wind_data(file_path):
    """Loads wind data: Headers on row 11 (skip 10). Filters for 10:00 AM local."""
    # Using skip 10 as per your wind_test.csv structure
    df = pd.read_csv(file_path, sep=';', skiprows=10)

    # Combine Date and UTC Time
    df['Datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'])

    # Identify Wind column (Vindhasti)
    wind_col = df.columns[4]
    df.rename(columns={wind_col: 'Wind_10AM'}, inplace=True)
    df['Wind_10AM'] = pd.to_numeric(df['Wind_10AM'], errors='coerce')

    # Per report: Use wind at 10:00 AM local (approx. 09:00 UTC)
    df_10am = df[df['Datetime'].dt.hour == 9].copy()
    df_10am['Date'] = df_10am['Datetime'].dt.normalize()

    return df_10am[['Date', 'Wind_10AM']]


def load_temp_data(file_path):
    """Loads temp data: Headers on row 10 (skip 9). Calculates daily MAX temp."""
    # Using skip 9 as requested for temp_test.csv
    df = pd.read_csv(file_path, sep=';', skiprows=9)

    df['Date'] = pd.to_datetime(df['Datum']).dt.normalize()

    # Identify Temp column (Lufttempe)
    temp_col = df.columns[2]
    df.rename(columns={temp_col: 'Temp_Max'}, inplace=True)
    df['Temp_Max'] = pd.to_numeric(df['Temp_Max'], errors='coerce')

    # Per report: Use the daily highest reported temperature
    return df.groupby('Date')['Temp_Max'].max().reset_index()


def load_price_data(file_path):
    """Loads Nord Pool prices for SE3 and SE2 (for bottleneck proxy)."""
    # Price file headers usually on row 5
    df = pd.read_excel(file_path, skiprows=4)

    # Map SE3 and SE2 for the bottleneck calculation
    df.rename(columns={
        df.columns[0]: 'Date',
        'SE2 (EUR)': 'Price_SE2',
        'SE3 (EUR)': 'Price_SE3'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df[['Date', 'Price_SE2', 'Price_SE3']].dropna()


# ============================================
# 2. MAIN EXECUTION
# ============================================

def main():
    # File Paths using your provided filenames
    BASE_DIR = 'data'
    WIND_FILE = os.path.join(BASE_DIR, 'wind_test.csv')
    TEMP_FILE = os.path.join(BASE_DIR, 'temp_test.csv')
    PRICE_FILE = os.path.join(BASE_DIR, 'prices_test.xlsx')

    try:
        # 1. Load data
        wind_df = load_wind_data(WIND_FILE)
        temp_df = load_temp_data(TEMP_FILE)
        price_df = load_price_data(PRICE_FILE)

        # 2. Merge all features
        df = pd.merge(wind_df, temp_df, on='Date')
        df = pd.merge(df, price_df, on='Date')
        df.set_index('Date', inplace=True)

        # 3. Create RM1 Model Variables

        # Weekend Dummy: 1 for Saturday/Sunday
        df['Weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Nuclear Outage Dummy: Oskarshamn 3 major repair (Mar 29 - Oct 31, 2024)
        df['Nuclear_Outage'] = ((df.index >= '2024-03-29') & (df.index <= '2024-10-31')).astype(int)

        # Bottleneck Proxy: 1 if Price SE3 and SE2 differ by > 0.1 EUR (approx 1 öre)
        df['Bottleneck'] = (np.abs(df['Price_SE3'] - df['Price_SE2']) > 0.1).astype(int)

        # 4. Fit RM1 OLS Regression
        Y = df['Price_SE3']
        X = df[['Wind_10AM', 'Temp_Max', 'Weekend', 'Nuclear_Outage', 'Bottleneck']]
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        # 5. Output Results
        print("=" * 60)
        print("RM1 MODEL RESULTS: SE3 (STOCKHOLM) FULL YEAR 2024")
        print("=" * 60)
        print(model.summary())

    except Exception as e:
        print(f"Error executing regression: {e}")


if __name__ == "__main__":
    main()