# Preprocessing Pipeline — Variable Reference

This document is the authoritative reference for how every variable is transformed
before it enters the regression. It mirrors the implementation in `preprocessing.py`
and is called by the Python export scripts via `preprocess_data_for_regression()`. 

---

## 1. Data Sources

| Variable | Source file | Frequency | Unit |
|----------|-------------|-----------|------|
| Price | `master data files/2015-2025/Combined_{ZONE}_Data_2015_2025.xlsx` | Hourly | EUR/MWh |
| Wind_Forecast | Same combined file | Hourly | MW |
| Net_Exchange | Same combined file | Hourly | MWh (positive = export) |
| Consumption | Same combined file | Hourly | MW |
| BNECK_{ZONE}_{partner} | Same combined file | Hourly | Binary 0/1 |
| Hydro_Reserves | `master data files/Master_Hydro_Reservoir.xlsx` | Weekly → merged hourly | GWh |
| Oil_Price | `master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx` (Close column) | Hourly | USD/barrel |
| Gas_Price | `master data files/Master_Commodities.xlsx` TTF column (skiprows=5, Bloomberg format) | Daily → broadcast to each hour | EUR/MWh |

Bottleneck trading partners by zone:

| Zone | Partners |
|------|----------|
| SE1 | FI, NO4, SE2 |
| SE2 | NO3, NO4, SE1, SE3 |
| SE3 | DK1, FI, NO1, SE2, SE4 |
| SE4 | (none defined) |

---

## 2. Pipeline — Ordered Stages

### Stage 0 — Load and Merge (`load_data`)

1. Read combined file; map to standard column names.
2. Merge Hydro_Reserves on exact hourly Datetime (left join).
3. Merge Oil_Price on exact hourly Datetime (left join).
4. Merge Gas_Price on **date only** — each hour within a day receives the same daily value.
5. **Lag Oil_Price and Gas_Price by 24 hours** (`shift(24)`). Rationale: day-ahead electricity
   prices are set the day before delivery; commodity prices known at decision time are those
   of the prior day.
6. Apply date filter (`start_date`, `end_date`) if specified.
7. Drop rows with any NaN (default). Interpolation available but not used in the thesis.
8. Set Datetime as index; infer hourly frequency.

After Stage 0, the dataframe contains:
`Price, Wind_Forecast, Net_Exchange, Consumption, Hydro_Reserves, Oil_Price, Gas_Price, BNECK_*`

---

### Stage 1 — Negative Price Handling (`handle_negative_prices`)

- **Checks** (report only): Price, Wind_Forecast, Hydro_Reserves, Consumption, Oil_Price, Gas_Price.
  Net_Exchange is not checked — negative values are expected and meaningful.
- **Action on Price**: if `min(Price) < 0.01`, shift the **entire Price series** upward by
  `0.01 − min(Price)`. Every observation is shifted by the same constant, preserving all
  relative differences. No observations are dropped.
- No other variable is modified in this stage.

---

### Stage 2 — Log Transformation (`apply_log_transform`)

Applied immediately after Stage 1, before deseasonalization (log-then-deseasonalize is the
standard approach because seasonal variation in electricity prices is multiplicative).

| Input column | Clipping before log | Output column |
|---|---|---|
| Price | None (negatives already handled) | `Price_Log` |
| Wind_Forecast | `clip(lower=0.01)` | `Wind_Forecast_Log` |
| Hydro_Reserves | `clip(lower=0.01)` | `Hydro_Reserves_Log` |
| Consumption | `clip(lower=0.01)` | `Consumption_Log` |
| Oil_Price | `clip(lower=0.01)` | `Oil_Price_Log` |
| Gas_Price | `clip(lower=0.01)` | `Gas_Price_Log` |
| Net_Exchange | **Not logged** — contains negative values | `Net_Exchange` (unchanged) |
| BNECK_* | **Not logged** — binary dummies | `BNECK_*` (unchanged) |

---

### Stage 3 — Deseasonalization (`deseasonalize_logged_variables`)

Each logged series is regressed on seasonal dummy variables using OLS. The **residual** from
that OLS is the deseasonalized series used in the main regression.

Two dummy sets are used:

- **FULL**: Year + Month + Day-of-Week + Hour + Swedish holiday indicator
- **PARTIAL**: Year + Month only

All dummy sets use `drop_first=True` (to avoid perfect multicollinearity) plus a constant term.

| Input column | Dummy set | Rationale | Output column |
|---|---|---|---|
| `Price_Log` | FULL | Electricity prices have strong intraday and day-of-week patterns | `Price_Log_Deseasonalized` |
| `Consumption_Log` | FULL | Consumption mirrors the same intraday/weekly cycles | `Consumption_Log_Deseasonalized` |
| `Hydro_Reserves_Log` | PARTIAL | Hydro is a weekly stock variable; no meaningful intraday cycle | `Hydro_Reserves_Log_Deseasonalized` |
| `Oil_Price_Log` | PARTIAL | Global commodity; no intraday variation, only seasonal trends | `Oil_Price_Log_Deseasonalized` |
| `Gas_Price_Log` | PARTIAL | Same as oil | `Gas_Price_Log_Deseasonalized` |
| `Wind_Forecast_Log` | **None** | Following Fredriksson (2016): wind is not deseasonalized | `Wind_Forecast_Log` (unchanged) |
| `Net_Exchange` | **None** | Following Fredriksson (2016): net exchange is not deseasonalized | `Net_Exchange` (unchanged) |
| `BNECK_*` | **None** | Binary dummies; deseasonalization not meaningful | `BNECK_*` (unchanged) |

Holiday indicator uses the `holidays` Python package with `holidays.Sweden()`.

---

### Stage 4 — Outlier Handling (`handle_outliers_gianfreda`)

Based on Gianfreda (2010) / Mugele et al. (2005).

- Applied **only to `Price_Log_Deseasonalized`**. No other variable is modified.
- For each day-of-week (Monday–Sunday) separately:
  - Compute mean and standard deviation of `Price_Log_Deseasonalized` for that weekday.
  - Any observation outside `[mean − 3σ, mean + 3σ]` is **capped** (Winsorized) to the threshold.
  - Replacement value = the threshold itself (not the weekday mean).
- Weekday-specific thresholds are used because price volatility differs systematically between
  weekdays and weekends.
- Outliers are replaced, not dropped; the total number of observations does not change.

---

## 3. Final Variables Entering the Regression

After all four stages, `get_regression_variable_names()` selects the following columns
in this fixed order (controls can be individually excluded via the `controls` dict):

| Role | Column name | Transformation summary |
|------|-------------|------------------------|
| **Dependent variable (Y)** | `Price_Log_Deseasonalized` | log → FULL deseasonalize → weekday ±3σ cap |
| Exogenous | `Wind_Forecast_Log` | log only (not deseasonalized) |
| Exogenous | `Hydro_Reserves_Log_Deseasonalized` | log → PARTIAL deseasonalize |
| Exogenous | `Net_Exchange` | raw (not logged, not deseasonalized) |
| Exogenous | `Consumption_Log_Deseasonalized` | log → FULL deseasonalize |
| Exogenous | `Oil_Price_Log_Deseasonalized` | 24h lag → log → PARTIAL deseasonalize |
| Exogenous | `Gas_Price_Log_Deseasonalized` | 24h lag → log → PARTIAL deseasonalize |
| Exogenous | `BNECK_{ZONE}_{partner}` × N | raw binary (one per trading partner) |

---

## 4. Column Name Reference (Python → Stata)

When data is exported to CSV for Stata via `arima_from_python.do`, columns are renamed:

| Python column | Stata column |
|---|---|
| `Price_Log_Deseasonalized` | `price_log_ds` |
| `Wind_Forecast_Log` | `wind_log` |
| `Hydro_Reserves_Log_Deseasonalized` | `hydro_log_ds` |
| `Net_Exchange` | `net_exchange` |
| `Consumption_Log_Deseasonalized` | `consump_log_ds` |
| `Oil_Price_Log_Deseasonalized` | `oil_log_ds` |
| `Gas_Price_Log_Deseasonalized` | `gas_log_ds` |
| `BNECK_{ZONE}_{partner}` | `bneck_{zone}_{partner}` (lowercased automatically) |

---

## 5. Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Log before deseasonalize | Seasonal patterns in electricity prices are multiplicative; log linearizes them before the additive dummy regression |
| Wind not deseasonalized | Following Fredriksson (2016); wind enters as a contemporaneous forecast, its seasonal component is treated as part of the signal |
| Net exchange not logged or deseasonalized | Can be negative (net importer); log undefined; no deseasonalization applied per Fredriksson |
| Hydro/Oil/Gas use PARTIAL dummies | These are slower-moving variables with no meaningful intraday or day-of-week cycle; Year + Month captures the relevant seasonal trend |
| Price/Consumption use FULL dummies | Strong and well-documented intraday and day-of-week seasonality in both series |
| Outlier cap on price only | Explanatory variables are not Winsorized; only the dependent variable receives outlier treatment per Gianfreda methodology |
| Weekday-specific outlier thresholds | Price variance differs between weekdays and weekends; a single global threshold would over-cap weekday peaks or under-cap weekend extremes |
| Commodity prices lagged 24h | Day-ahead market clears at noon D-1; the relevant commodity price signal is the prior day's closing price |
| Price shift (not drop) for negatives | Preserves all observations and relative differences; dropping negative-price hours would bias the sample away from stress events |
