import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt


# --- 1. DATA LOADING FUNCTIONS ---
# These follow your established file-handling logic

def load_price_data(file_path, area_zone='SE4'):
    price_df = pd.read_excel(file_path, header=[0, 1])
    time_col, price_col = price_df.columns[0], (area_zone, 'Price (EUR)')
    return pd.DataFrame({
        'Datetime': pd.to_datetime(price_df[time_col], dayfirst=True, errors='coerce'),
        'Price': pd.to_numeric(price_df[price_col], errors='coerce')
    }).dropna()


def load_production_data(file_path):
    df = pd.read_excel(file_path, header=0)
    df = df[df['Delivery period (CET)'].str.contains(':', na=False)].copy()
    df['StartHour'] = df['Delivery period (CET)'].str.split(' ').str[0]
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['StartHour'], errors='coerce')
    return df[['Datetime', 'Wind Onshore (MWh)']].rename(columns={'Wind Onshore (MWh)': 'Wind_Forecast'}).dropna()


def load_consumption_data(file_path, area_zone='SE4'):
    df = pd.read_excel(file_path, header=0)
    df = df[df['Delivery period (CET)'].str.contains(':', na=False)].copy()
    df['StartHour'] = df['Delivery period (CET)'].str.split(' ').str[0]
    date_clean = df['Date'].astype(str).str.split('T').str[0]
    df['Datetime'] = pd.to_datetime(date_clean + ' ' + df['StartHour'], errors='coerce')
    return df[['Datetime', f"{area_zone} (MWh)"]].rename(columns={f"{area_zone} (MWh)": 'Consumption'}).dropna()


def load_exchange_data(file_path):
    df = pd.read_excel(file_path, header=0)
    df = df[df['Delivery period (CET)'].str.contains(':', na=False)].copy()
    df['StartHour'] = df['Delivery period (CET)'].str.split(' ').str[0]
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['StartHour'], errors='coerce')
    target_col = 'SE4 Total Net Exchange (MWh)'
    return df[['Datetime', target_col]].rename(columns={target_col: 'Net_Exchange'}).dropna()


# --- 2. MODELING FUNCTIONS ---

def perform_standard_ols(df, zone):
    """Calculates standard Linear Regression (OLS)."""
    print(f"\n--- RUNNING STANDARD OLS REGRESSION ({zone}) ---")
    Y = df['Price']
    X = sm.add_constant(df[['Wind_Forecast', 'Consumption', 'Net_Exchange']])
    model = sm.OLS(Y, X).fit()
    print(model.summary())
    return model


def perform_thesis_model(df, zone):
    """Calculates ARMAX(3,3)-GARCHX(1,1) as per the SSE study."""
    print(f"\n--- RUNNING ARMAX-GARCHX FRAMEWORK ({zone}) ---")
    Y = df['Price']
    X_exog = df[['Wind_Forecast', 'Consumption', 'Net_Exchange']]

    # 1. Price Level (ARMAX)
    armax_res = sm.tsa.ARIMA(Y, exog=X_exog, order=(3, 0, 3)).fit()
    print("\nMEAN EQUATION (Price Level) RESULTS:")
    print(armax_res.summary())

    # 2. Volatility (GARCHX)
    garch_mod = arch_model(Y, x=X_exog, mean='ARX', lags=3, vol='Garch', p=1, q=1)
    garch_res = garch_mod.fit(disp='off')
    print("\nVARIANCE EQUATION (Volatility) RESULTS:")
    print(garch_res.summary())

    return armax_res, garch_res


# --- 3. EXECUTION BLOCK ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # SET THIS TO 'OLS' or 'ARMAX-GARCHX'
    MODEL_TO_RUN = 'ARMAX-GARCHX'
    TARGET_ZONE = 'SE4'

    PATHS = {
        'price': os.path.join('data', '2024_prices.xlsx'),
        'prod': os.path.join('data', 'merged_production_2024_s4.xlsx'),
        'cons': os.path.join('data', 'merged_consumption_2024.xlsx'),
        'exch': os.path.join('data', 'merged_exchange_2024_s4.xlsx')
    }

    try:
        # Data loading
        prices = load_price_data(PATHS['price'], TARGET_ZONE)
        production = load_production_data(PATHS['prod'])
        consumption = load_consumption_data(PATHS['cons'], TARGET_ZONE)
        exchange = load_exchange_data(PATHS['exch'])

        # Merge
        merged = pd.merge(prices, production, on='Datetime')
        merged = pd.merge(merged, consumption, on='Datetime')
        merged = pd.merge(merged, exchange, on='Datetime').dropna()
        merged = merged.set_index('Datetime')

        # Model Switching
        if MODEL_TO_RUN == 'OLS':
            perform_standard_ols(merged, TARGET_ZONE)
        elif MODEL_TO_RUN == 'ARMAX-GARCHX':
            perform_thesis_model(merged, TARGET_ZONE)
        else:
            print("Invalid MODEL_TO_RUN setting.")

    except Exception as e:
        print(f"Error encountered: {e}")