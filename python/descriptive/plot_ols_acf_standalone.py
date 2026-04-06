"""
Standalone: OLS preprocessing → AR[1,23,24,25] residual ACF (500 lags)
SE4, 2024-01-01 to 2025-12-31
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings('ignore')

DATA_DIR   = 'master data files/2015-2025'
OUTPUT_DIR = 'MSTL pipeline results'
ZONE       = 'SE4'
START      = '2024-01-01'
END        = '2025-12-31'
AR_LAGS    = [1]
N_LAGS     = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
path = os.path.join(DATA_DIR, f'Combined_{ZONE}_Data_2015_2025.xlsx')
df   = pd.read_excel(path, usecols=['Timestamp', 'Spot_Price'])
df.index = pd.to_datetime(df['Timestamp'])
df = df.drop(columns='Timestamp')
df = df.apply(pd.to_numeric, errors='coerce')
df.index.name = 'Datetime'
df = df[df.index >= pd.to_datetime(START)]
df = df[df.index <= pd.to_datetime(END)]
if df.index.duplicated().any():
    df = df.groupby(df.index).mean()
df = df.asfreq('h').ffill()

price = df['Spot_Price'].copy()

# ── 2. OLS preprocessing ──────────────────────────────────────────────────────
# Shift so min >= 0.01, then log
shift = max(0.0, 0.01 - price.min())
price_shifted = price + shift
price_log     = np.log(price_shifted)
price_log.name = 'Price_Log'

# Deseasonalize: hour, DOW, month, year dummies (full OLS)
idx = price_log.index
dummies = pd.DataFrame({
    'hour':  idx.hour,
    'dow':   idx.dayofweek,
    'month': idx.month,
    'year':  idx.year,
}, index=idx)

dummy_df = pd.concat([
    pd.get_dummies(dummies['hour'],  prefix='h',  drop_first=True).astype(float),
    pd.get_dummies(dummies['dow'],   prefix='d',  drop_first=True).astype(float),
    pd.get_dummies(dummies['month'], prefix='m',  drop_first=True).astype(float),
    pd.get_dummies(dummies['year'],  prefix='y',  drop_first=True).astype(float),
], axis=1)
X = sm.add_constant(dummy_df)

ols_res    = sm.OLS(price_log.astype(float), X.astype(float)).fit()
price_ds   = price_log - ols_res.fittedvalues
price_ds.name = 'Price_OLS_DS'
price_ds   = price_ds.dropna()

# ── 3. AR[1,23,24,25] ────────────────────────────────────────────────────────
ar_model  = AutoReg(price_ds, lags=AR_LAGS, trend='n')
ar_result = ar_model.fit(cov_type='HC0')

residuals = ar_result.resid.dropna()

# R² (manual)
ss_res = (residuals ** 2).sum()
ss_tot = ((price_ds - price_ds.mean()) ** 2).sum()
r2     = 1 - ss_res / ss_tot
print(f"R² = {r2:.3f}   N residuals = {len(residuals):,}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
plot_acf(residuals, lags=N_LAGS, ax=ax)
ax.set_title(f'OLS preprocessing → AR[1] residual ACF  (R²={r2:.3f})')
fig.tight_layout()

fname = f'acf_ols_ar1only_{ZONE}_{START.replace("-","")[:6]}_{END.replace("-","")[:6]}_200lags.png'
fpath = os.path.join(OUTPUT_DIR, fname)
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fpath}")
