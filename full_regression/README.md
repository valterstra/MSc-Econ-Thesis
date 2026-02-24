# full_regression — Modular Electricity Price Analysis

Modular refactoring of `full_regression.py` (6 112 lines → 10 focused files).
The **original `full_regression.py` is left completely untouched**.

---

## How to Run

```bash
# From the MSc-Econ-Thesis directory:
python -m full_regression.main
```

All configuration lives in `main.py` — that is the only file you need to edit.

---

## File Overview

> **Note:** Python module names cannot start with a digit, so files keep their plain names.
> Each file is tagged `[Module XX/10]` in its header docstring for quick orientation.

| # | File | Lines | Contents |
|---|------|-------|----------|
| 01 | [`config.py`](config.py) | ~35 | `TRADING_PARTNERS`, `ARMAX_*` global defaults |
| 02 | [`data_loading.py`](data_loading.py) | ~230 | `load_data()` |
| 03 | [`preprocessing.py`](preprocessing.py) | ~680 | `handle_negative_prices`, both outlier methods, `apply_log_transform`, `deseasonalize_logged_variables`, `preprocess_data_for_regression` |
| 04 | [`diagnostics.py`](diagnostics.py) | ~170 | `run_ljungbox_test`, `run_heteroskedasticity_tests`, `run_stationarity_tests` |
| 05 | [`utils.py`](utils.py) | ~230 | `run_tvp_wind_kalman_analysis`, `get_regression_variable_names` |
| 06 | [`regression_models.py`](regression_models.py) | ~820 | All ARMAX helpers, `_fit_armax_with_controls/fallback`, ARMAX lag selection, `fit_garchx_model` |
| 07 | [`regression_analysis.py`](regression_analysis.py) | ~390 | `perform_multivariate_analysis` (main OLS → ARMAX → GARCH coordinator) |
| 08 | [`structural_analysis.py`](structural_analysis.py) | ~2500 | `run_rolling_window_analysis`, structural break utilities, level & trend breaks, quantile regression |
| 09 | [`visualization.py`](visualization.py) | ~730 | All plotting functions + `export_data_for_R` |
| 10 | [`main.py`](main.py) | ~480 | Configuration block + execution pipeline |

---

## Module Details

### 01 — [`config.py`](config.py) — Global Constants
```
TRADING_PARTNERS       dict  congestion-dummy partners per zone
ARMAX_ALLOW_NONCONVERGED  bool  reject non-converged fits by default
ARMAX_MAXITER          int   optimiser iteration limit
ARMAX_SOLVER           str   'statespace'
ARMAX_USE_WARM_START   bool  initialise from OLS for faster convergence
ARMAX_ENABLE_FALLBACK_ORDERS  bool  try simpler orders on failure
ARMAX_FALLBACK_ORDERS  list  fallback ladder: [(1,0,1),(2,0,2),(3,0,3)]
ARMAX_BASELINE_SPEC    dict  default order/lags for baseline run
```
All values can be **overridden** in the `if __name__ == '__main__'` block of `main.py`.

---

### 02 — [`data_loading.py`](data_loading.py) — Data Loading
#### `load_data(paths, target_region, zone_hydro, use_interpolation, start_date, end_date, lag_commodity_hours=24)`
Loads combined regional Excel file, merges hydro reserves, Light Crude Oil (hourly), and TTF Gas (daily).
**Automatically lags Oil_Price and Gas_Price by `lag_commodity_hours` (default 24 h)** to align with day-ahead market bidding.

---

### 03 — [`preprocessing.py`](preprocessing.py) — Data Transformation Pipeline
Pipeline order:

1. **`handle_negative_prices(df, method)`**
   `method='clip'` → replace values < 0.01 with 0.01
   `method='shift'` → shift entire series so minimum = 0.01

2. **`handle_outliers_fredriksson(df, apply_to_raw)`**
   Fredriksson (2016): asymmetric thresholds `+6σ / −3.7σ`.
   Replacement: mean of ±24 h and ±48 h surrounding values.

3. **`handle_outliers_gianfreda(df, apply_to_raw)`**
   Gianfreda (2010): symmetric `±3σ` per weekday.
   Replacement: caps at the weekday-specific `±3σ` boundary.

4. **`apply_log_transform(df)`**
   Applies `log()` to: Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price.
   Net_Exchange is **not** logged (can be negative).

5. **`deseasonalize_logged_variables(df)`**
   Dummy-variable OLS regression on logged series.
   Full (Year+Month+DOW+Hour+Holiday) → Price, Consumption.
   Partial (Year+Month only) → Hydro, Oil, Gas.

6. **`preprocess_data_for_regression(df_raw, ...)`**
   Orchestrates steps 1–5 in one call. Used by rolling window analysis for window-local preprocessing.

---

### 04 — [`diagnostics.py`](diagnostics.py) — Statistical Tests
| Function | Test | H₀ |
|----------|------|----|
| `run_ljungbox_test(residuals)` | Ljung-Box | No autocorrelation |
| `run_heteroskedasticity_tests(residuals)` | Engle ARCH + LB on squared residuals | No ARCH effects |
| `run_stationarity_tests(series)` | ADF + DF-GLS | Unit root (non-stationary) |

---

### 05 — [`utils.py`](utils.py) — Utilities
**`run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars)`**
State-space Kalman filter for time-varying wind coefficient.
Uses Frisch-Waugh-Lovell partialling and a random-walk state equation.
⚠ Current hourly implementation shows excessive noise; see docstring for improvement options.

**`get_regression_variable_names(df, target_region)`**
Returns `(y_name, exog_vars)` including zone-specific bottleneck dummies.

---

### 06 — [`regression_models.py`](regression_models.py) — ARMAX & GARCH
**ARMAX internals (in call order):**
```
_attach_inferred_frequency()         attach freq to DatetimeIndex (avoid statsmodels warnings)
_validate_armax_baseline_spec()      validate spec dict
_prepare_baseline_armax_design()     build design matrix with optional sparse AR lags
_build_warm_start_params()           initialise from OLS for faster convergence
_fit_armax_with_controls()           core ARMAX fit; returns ok/model/diagnostics
_fit_armax_with_fallback()           try primary order, then fallback ladder
_diagnose_nonconvergence_simple()    classify non-convergence cause
```

**ARMAX grid search:**
```
_build_armax_search_grid()           build (p,q) combinations
_evaluate_armax_candidate()          fit + Ljung-Box for one (p,q)
_select_best_armax_candidate()       rank by AIC/BIC with eligibility filters
_save_armax_search_reports()         write CSV/Excel results
select_armax_lags_aic()              strict grid search
select_armax_lags_aic_checkpointed() same with resume support
```

**GARCH:**
```
fit_garchx_model(armax_residuals, df)   GARCH(1,1)-X; variance eq. uses Wind_Forecast_Log
```

---

### 07 — [`regression_analysis.py`](regression_analysis.py) — Main Orchestrator
**`perform_multivariate_analysis(df, zone, ...)`**
Coordinates the full analysis pipeline:
1. OLS with HAC standard errors
2. Optional stationarity / Ljung-Box / ARCH diagnostics
3. ARMAX (baseline spec or grid-search)
4. GARCH-X if ARCH effects detected
5. Early-return modes: TVP Kalman, rolling window, quantile regression, structural breaks

---

### 08 — [`structural_analysis.py`](structural_analysis.py) — Time-Varying & Break Analysis
**Rolling window:**
```
run_rolling_window_analysis(df, zone, ...)
  Overlapping OLS windows; window-local preprocessing supported.
  Saves coefficient plot and CSV.
```

**Break analysis utilities:**
```
_get_break_model_tag / _get_break_model_label   filesystem tag + plot label
_extract_armax_wind_coef()                       robust coefficient extraction
_estimate_rolling_wind_coefficients()            rolling OLS or ARMAX(3,0,3) betas
_run_dynamic_break_lr_tests()                    LR Chow tests at candidate dates
```

**Level breaks (Bai-Perron):**
```
run_structural_break_analysis(df, zone, Y, exog_vars, ...)
  Detects step changes in coefficient using ruptures (binary segmentation / PELT).
  Chow tests at detected breaks and known event dates.
```

**Trend breaks:**
```
run_trend_break_analysis_legacy(...)    segmented regression + BIC + sequential F-tests
run_trend_break_analysis_bp_supf(...)   Bai-Perron sequential supF tests + bootstrap inference
run_trend_break_analysis(...)           wrapper: dispatches to 'legacy' or 'bp_supf'
```

**Quantile regression:**
```
run_quantile_regression_analysis(df, zone, ...)
  Wind coefficient at quantiles 0.1, 0.25, 0.5, 0.75, 0.9.
```

---

### 09 — [`visualization.py`](visualization.py) — Plotting & Export
| Function | Output |
|----------|--------|
| `plot_zone_comparisons(...)` | SE1–SE4 overlay: price, log price, volatility, wind share |
| `plot_time_series(df, zone, stage)` | variable time series at 'raw'/'logged'/'deseasonalized' |
| `plot_distributions(df, zone, ...)` | histogram + KDE |
| `plot_boxplots(df, zone, ...)` | seasonal boxplots (hourly, daily, monthly) |
| `detect_outliers(df, zone, ...)` | identify outlier indices and statistics |
| `plot_outliers_timeline(df, zone, ...)` | timeline of detected outliers |
| `plot_scatter_matrix(df, zone, ...)` | pairwise scatter plots |
| `run_visualizations(data, zone, ...)` | master call: generates all plots for one stage |
| `export_data_for_R(data, zone, ...)` | CSV export for R `strucchange` package |

---

### 10 — [`main.py`](main.py) — Configuration & Entry Point
The **only file you need to edit**.
Key toggles (all in the `if __name__ == '__main__'` block):

```python
ACTIVE_ZONE                 = 'SE1'          # 'SE1'/'SE2'/'SE3'/'SE4'
NEGATIVE_PRICE_HANDLING     = 'shift'        # 'clip' or 'shift'
OUTLIER_METHOD              = 'fredriksson'  # 'fredriksson' or 'gianfreda'
HANDLE_OUTLIERS_BEFORE_LOG  = False          # True = before log, False = after deseasonalization
LAG_COMMODITY_HOURS         = 24             # day-ahead alignment
USE_LINEAR_INTERPOLATION    = True           # fill NaN by interpolation

RUN_LJUNGBOX_TEST           = True
RUN_HETEROSKEDASTICITY_TESTS= True
RUN_STATIONARITY_TESTS      = True

OPTIMIZE_ARMAX_LAGS         = False          # AIC/BIC grid search
ARMAX_BASELINE_SPEC         = {...}          # order, extra_ar_lags, label

RUN_TVP_WIND_KALMAN         = False
RUN_ROLLING_WINDOW          = False
RUN_QUANTILE_REGRESSION     = False
RUN_STRUCTURAL_BREAK        = False
STRUCTURAL_BREAK_TYPE       = 'trend'        # 'level' or 'trend'
TREND_BREAK_TEST_METHOD     = 'bp_supf'      # 'legacy' or 'bp_supf'

EXPORT_DATA_FOR_R           = False
RUN_VISUALIZATIONS          = False
RUN_ZONE_COMPARISONS        = False

PATHS = {
    'combined'   : 'master data files/2015-2025/Combined_SE1_Data_2015_2025.xlsx',
    'hydro'      : 'master data files/Master_Hydro_Reservoir.xlsx',
    'crude_oil'  : 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
    'commodities': 'master data files/Master_Commodities.xlsx'
}
```

---

## Module Dependency Graph

```
10  main.py
     ├── 02  data_loading        ← 01 config
     ├── 03  preprocessing       (self-contained)
     ├── 04  diagnostics         (self-contained)
     ├── 05  utils               ← 01 config
     ├── 06  regression_models   ← 01 config, 04 diagnostics
     ├── 07  regression_analysis ← 01 config, 04 diagnostics, 05 utils,
     │                              06 regression_models, 08 structural_analysis
     ├── 08  structural_analysis ← 01 config, 03 preprocessing, 05 utils,
     │                              06 regression_models
     └── 09  visualization       (self-contained)
```

---

## Preservation Check

All **47 functions** from the original `full_regression.py` are present.
No code was added or removed — only reorganized into logical modules.
Original file (`../full_regression.py`) is **untouched**.
