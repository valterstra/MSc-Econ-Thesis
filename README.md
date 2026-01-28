# Electricity Price Analysis - SE1 (Stockholm)

## Master's Thesis Project: Merit Order Effect Analysis

This repository contains the analysis pipeline for studying electricity spot prices in the Swedish SE1 bidding zone (Stockholm), with a focus on outlier detection, deseasonalization, and multivariate regression analysis following Fredriksson (2016) methodology.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Pipeline](#data-pipeline)
3. [Execution Order and Critical Dependencies](#execution-order-and-critical-dependencies)
4. [Configuration Toggles](#configuration-toggles)
5. [Methodological Decisions](#methodological-decisions)
6. [Step-by-Step Explanation](#step-by-step-explanation)
7. [Toggle Interactions](#toggle-interactions)
8. [Recommended Configurations](#recommended-configurations)
9. [References](#references)

---

## Overview

This project analyzes hourly electricity spot prices and their relationship with:
- **Wind forecasts** (MW)
- **Hydro reservoir levels** (MWh)
- **Net exchange flows** (MWh)
- **Consumption** (MWh)

**Time Period**: 2021-2024 (hourly data, ~35,000 observations)

**Key Features**:
- Outlier detection and handling (Fredriksson 2016 methodology)
- Seasonal decomposition using dummy variable regression
- Log transformations for linearization
- Stationarity testing (ADF, DF-GLS)
- ARMAX-GARCHX modeling

---

## Data Pipeline

The analysis follows a **strict execution order** where each step depends on the previous ones:

```
┌─────────────────────────────────────┐
│  1. LOAD DATA                       │
│     - Merge all master files        │
│     - Create hourly time series     │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. HANDLE MISSING VALUES           │
│     - Drop rows (default)           │
│     - OR Linear interpolation       │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  3. VISUALIZE RAW DATA (Optional)   │
│     - Time series plots             │
│     - Distributions with outliers   │
│     - Box plots                     │
│     - Outlier detection (visual)    │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  4. DESEASONALIZE (Optional)        │
│     - Price, Consumption, Hydro     │
│     - Using Year/Month/DOW/Hour     │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  5. HANDLE OUTLIERS (Optional)      │
│     - 6σ / 3.7σ thresholds          │
│     - Replace with mean of ±24,48h  │
│     - ONLY applied to Price         │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  6. LOG TRANSFORM (Optional)        │
│     - log(Price), log(Wind)         │
│     - log(Hydro), log(Consumption)  │
│     - Net_Exchange NOT logged       │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  7. STATIONARITY TESTS (Optional)   │
│     - ADF, DF-GLS tests             │
│     - On final processed data       │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  8. REGRESSION ANALYSIS             │
│     - OLS baseline                  │
│     - ARMAX(p,q) with AIC selection │
│     - Diagnostic tests              │
└─────────────────────────────────────┘
```

---

## Execution Order and Critical Dependencies

### Why Order Matters

Each step **transforms the data**, affecting all downstream operations. The order is **not arbitrary** but based on statistical principles:

| Step | Must Come Before | Reason |
|------|------------------|--------|
| **Missing Values** | Everything | Can't analyze data with gaps |
| **Visualization** | Outlier Handling | Need to see raw outliers before removal |
| **Deseasonalize** | Outlier Handling | Seasonal patterns mask true outliers |
| **Outlier Handling** | Log Transform | Outliers distort logarithms |
| **Log Transform** | Stationarity Tests | Test the data that enters regression |
| **Stationarity Tests** | Regression | Validate modeling assumptions |

### Critical Dependency Chain

```
Missing Values → Deseasonalization → Outliers → Log → Tests → Regression
     ↓                 ↓                 ↓        ↓      ↓         ↓
  Required         Changes mean      Changes    Non-  Final    Uses
  for all          & variance        extreme   linear  data   processed
  analysis                           values    transform      data
```

---

## Configuration Toggles

All analysis options are controlled via boolean toggles in `full_regression.py` (lines ~960-1020):

### Data Processing Toggles

| Toggle | Default | Description |
|--------|---------|-------------|
| `RUN_VISUALIZATIONS` | `True` | Generate plots of raw data with outlier detection |
| `USE_LINEAR_INTERPOLATION` | `False` | Fill missing values (True) or drop rows (False) |
| `USE_DESEASONALIZED` | `False` | Remove seasonal patterns before analysis |
| `HANDLE_OUTLIERS` | `False` | Replace Price outliers using Fredriksson method |
| `USE_LOG_TRANSFORM` | `False` | Apply logarithmic transformation to variables |

### Diagnostic Test Toggles

| Toggle | Default | Description |
|--------|---------|-------------|
| `RUN_LJUNGBOX_TEST` | `False` | Test for autocorrelation in residuals |
| `RUN_HETEROSKEDASTICITY_TESTS` | `False` | ARCH effects and heteroskedasticity tests |
| `RUN_STATIONARITY_TESTS` | `False` | ADF and DF-GLS unit root tests |
| `OPTIMIZE_ARMAX_LAGS` | `False` | AIC-based lag selection (slow, tests 100 models) |

---

## Methodological Decisions

### 1. Missing Values: Drop vs. Interpolate

**Current Default**: Drop rows with missing values

**Rationale**:
- **Safer approach**: Doesn't create artificial data
- **Conservative**: Only 120 rows dropped (~0.3% of data)
- **Preserves integrity**: Real observations only

**When to use interpolation** (`USE_LINEAR_INTERPOLATION = True`):
- If missing values are random and sparse
- If data gaps are short (1-2 hours)
- ⚠️ **Caution**: Interpolated values can be flagged as outliers if the gap is large

---

### 2. Outlier Handling: Once vs. Twice

**Current Implementation**: Apply once, **AFTER** deseasonalization

**Deviation from Fredriksson (2016)**:
- **Fredriksson's approach**: Apply outlier filter TWICE
  1. First on raw price series (found 31 outliers)
  2. Then on deseasonalized series (found 42 outliers)

- **Our approach**: Apply ONCE, after deseasonalization

**Theoretical Justification**:

1. **Seasonal patterns mask true outliers**
   ```
   Example: 150 EUR/MWh at 6pm in January
   - Raw data: Might be flagged as outlier (mean=41.50)
   - After deseasonalization: Might be normal for winter peak
   ```

2. **Deseasonalized mean ≈ 0 is more meaningful**
   - Raw data: mean=41.50, std=48.51 (mixes summer/winter)
   - Deseasonalized: mean≈0, std=residual variation only
   - 6σ threshold on deseasonalized data captures truly unusual deviations

3. **Cleaner single-pass approach**
   - No risk of removing observations twice
   - One clear outlier detection step
   - Easier to document and reproduce

4. **Fredriksson provides no justification**
   - The paper simply states the double application
   - No theoretical or empirical reasoning given
   - Our approach is arguably more defensible

**Note**: This is documented as a TODO for future sensitivity analysis (compare results with both approaches).

---

### 3. Outlier Detection Thresholds

Following Fredriksson (2016):
- **Upper threshold**: mean + 6σ
- **Lower threshold**: mean - 3.7σ

**Why asymmetric?**
- Electricity prices have **right-skewed distribution** (extreme positive spikes)
- More lenient lower threshold (3.7σ vs 6σ) accounts for this skewness
- Negative prices are rare but possible (excess renewable generation)

**Replacement method**:
- Replace outlier with **mean of surrounding observations**:
  - 24 hours before
  - 48 hours before
  - 24 hours after
  - 48 hours after
- Uses available values if at data boundaries

---

### 4. What Gets Deseasonalized?

**Variables deseasonalized**:
- Price (EUR/MWh)
- Consumption (MWh)
- Hydro Reserves (MWh)

**Variables NOT deseasonalized**:
- Wind Forecast (no clear seasonal pattern in forecasts)
- Net Exchange (policy-driven, not seasonally structured)

**Method**: OLS regression with dummy variables
- Year dummies (2021, 2022, 2023, 2024)
- Month dummies (Jan-Dec)
- Day-of-week dummies (Mon-Sun)
- Hour dummies (0-23)

Residuals from this regression = deseasonalized series

---

### 5. Log Transformation

**Variables logged** (when `USE_LOG_TRANSFORM = True`):
- Price
- Wind Forecast
- Hydro Reserves
- Consumption

**Variable NOT logged**:
- Net Exchange (can be negative)

**Why log?**
- **Linearizes relationships**: Wind → Price effect may be log-linear
- **Stabilizes variance**: Reduces heteroskedasticity
- **Interpretability**: Coefficients = elasticities
- **Fredriksson (2016)**: Uses logged variables in final model

**Important**: Logs applied AFTER outlier handling to avoid distorting transformations

---

## Step-by-Step Explanation

### Step 1: Load Data

**Function**: `load_data()`

**Inputs**:
- `Combined_SE1_Data_2015_2025.xlsx` (contains Price, Wind, Exchange, Consumption)
- `Master_Hydro_Reservoir.xlsx` (SE1 hydro reserves)
- `Master_Commodities.xlsx` (Brent Oil, TTF Gas prices)

**Process**:
1. Load combined SE1 data (2015-2025 range)
2. Merge hydro reserves on Timestamp
3. Merge daily commodity prices on Date
4. Filter to 2021-2024 (matches hydro/commodity availability)
5. Handle missing values (interpolate or drop)
6. Return DataFrame indexed by Datetime

**Output**: DataFrame with hourly observations for all variables

---

### Step 2: Handle Missing Values

**Default**: Drop rows with any missing values

**Alternative**: Linear interpolation (`USE_LINEAR_INTERPOLATION = True`)

**Impact**:
- ~120 rows dropped (0.3% of data)
- If interpolating: Creates synthetic data points

**Why this comes first**:
- Statistical methods require complete observations
- Can't compute means, variances with gaps
- Visualization requires continuous time series

---

### Step 3: Visualize Raw Data

**Generated when** `RUN_VISUALIZATIONS = True`

**Outputs** (saved to `plots/` directory):
1. `raw_time_series_SE1.png`: Time series for all variables
2. `raw_distributions_SE1.png`: Histograms with outlier thresholds
3. `raw_boxplots_SE1.png`: Box plots with IQR outliers
4. `raw_outliers_timeline_SE1.png`: Time series with outliers highlighted
5. `raw_scatter_matrix_SE1.png`: Pairwise relationships
6. `outlier_summary_fredriksson_SE1.csv`: Outlier statistics

**Why visualize BEFORE cleaning**:
- See the data as it is (warts and all)
- Understand extent and distribution of outliers
- Validate that outlier detection thresholds make sense
- Document what was removed

**Outlier Detection** (visual only, not replacement):
- Marks observations exceeding 6σ / 3.7σ thresholds
- Shows 135 Price outliers (0.39% of data)
- Other variables: 0 outliers detected

---

### Step 4: Deseasonalize

**Activated when** `USE_DESEASONALIZED = True`

**Function**: `deseasonalize_price()`

**Method**: Dummy variable regression
```
Variable = β₀ + Σ(Year dummies) + Σ(Month dummies) +
           Σ(DOW dummies) + Σ(Hour dummies) + ε

Deseasonalized = ε (residuals)
```

**Variables processed**:
- Price → Price_Deseasonalized
- Consumption → Consumption_Deseasonalized
- Hydro_Reserves → Hydro_Reserves_Deseasonalized

**Output statistics**:
- Reports R² for seasonal model (how much variance explained by seasonality)
- Shows original vs. deseasonalized standard deviation

**Why this comes BEFORE outlier handling**:
- Raw mean (41.50) mixes high winter & low summer prices
- Deseasonalized mean ≈ 0 represents "typical price for that hour/day/month"
- Outliers are deviations from seasonal expectation, not absolute levels

---

### Step 5: Handle Outliers

**Activated when** `HANDLE_OUTLIERS = True`

**Function**: `handle_outliers_fredriksson()`

**Applies to**: Price only (or Price_Deseasonalized if step 4 was enabled)

**Detection**:
- Upper threshold: mean + 6σ
- Lower threshold: mean - 3.7σ

**Replacement**:
For each outlier at time t:
```
replacement = mean([t-24h, t-48h, t+24h, t+48h])
```

**Example** (SE1 data):
- Detects 135 outliers in raw Price (0.39% of data)
- Range: 333.69 to 590.00 EUR/MWh
- Mean replacement value typically 30-50 EUR/MWh

**Why this comes AFTER deseasonalization**:
- More meaningful thresholds (deviations from seasonal norm)
- Avoids removing "normal" seasonal extremes
- Focuses on truly anomalous observations

**Output**: Prints each outlier with original → replacement value

---

### Step 6: Log Transform

**Activated when** `USE_LOG_TRANSFORM = True`

**Function**: `apply_log_transform()`

**Transformations**:
```python
Price_Log = log(Price)  # or log(Price_Deseasonalized + mean) if deseasonalized
Wind_Forecast_Log = log(Wind_Forecast)
Hydro_Reserves_Log = log(Hydro_Reserves)  # or deseasonalized version
Consumption_Log = log(Consumption)  # or deseasonalized version
# Net_Exchange NOT logged (can be negative)
```

**Why AFTER outlier handling**:
- Outliers distort logarithms severely
- log(590) = 6.38 vs log(50) = 3.91 (extreme difference)
- Clean data first, then transform

**Why log at all**:
- Merit order effect is often log-linear (elasticity interpretation)
- Stabilizes variance across price levels
- Common in energy economics literature

---

### Step 7: Stationarity Tests

**Activated when** `RUN_STATIONARITY_TESTS = True`

**Function**: `run_stationarity_tests()`

**Tests**:
1. **Augmented Dickey-Fuller (ADF)**: Tests for unit root
2. **Dickey-Fuller GLS (DF-GLS)**: More powerful version of ADF

**Null hypothesis**: Series has a unit root (non-stationary)
**Rejection**: p < 0.05 → stationary

**Why this comes LAST** (before regression):
- Tests the actual data entering the regression model
- If you test raw data but regress on logged data, tests are meaningless
- Stationarity is a requirement for valid OLS inference

---

### Step 8: Regression Analysis

**Function**: `perform_multivariate_analysis()`

**Models**:
1. **OLS Baseline**: Simple linear regression
2. **ARMAX(3,3)**: Autoregressive moving average with exogenous variables
   - Can optimize lags if `OPTIMIZE_ARMAX_LAGS = True`

**Diagnostic Tests**:
- **Ljung-Box**: Autocorrelation in residuals
- **ARCH/Heteroskedasticity**: Time-varying variance
- **Residual Analysis**: Normality, patterns

**Output**: Full regression tables with coefficients, standard errors, diagnostics

---

## Toggle Interactions

### What Happens with Different Combinations?

| Configuration | Data State at Regression | Use Case |
|---------------|-------------------------|----------|
| **All FALSE** | Raw prices, original scale | Baseline comparison |
| `VISUALIZE` only | Same as above, with plots | Exploratory analysis |
| `DESEASONALIZED` only | Seasonal patterns removed | Focus on short-term variation |
| `HANDLE_OUTLIERS` only | Outliers replaced (raw mean) | Quick outlier removal |
| `LOG_TRANSFORM` only | Logged raw data (outliers still there!) | ⚠️ **Not recommended** |
| `DESEASONALIZED` + `HANDLE_OUTLIERS` | Outliers replaced in deseasonalized data | **Recommended preprocessing** |
| `DESEASONALIZED` + `HANDLE_OUTLIERS` + `LOG` | Fully processed, ready for regression | **"Golden Path"** |

### Examples with Actual Values

**Scenario: 590 EUR/MWh price spike on 2023-08-15 at 18:00**

| Configuration | How it's treated | Value in regression |
|---------------|------------------|---------------------|
| No processing | Kept as is | 590 EUR/MWh |
| Outlier only (raw) | Flagged (>332.53), replaced | ~45 EUR/MWh (mean of ±24,48h) |
| Deseasonalize only | Not flagged | ~150 (deseasonalized residual) |
| Deseasonalize + Outlier | Flagged in deseasonalized space | ~0 (deseasonalized mean) |
| + Log transform | Logged after replacement | log(45) ≈ 3.8 |

---

## Recommended Configurations

### Configuration 1: **Exploratory Analysis**
```python
RUN_VISUALIZATIONS = True
USE_DESEASONALIZED = False
HANDLE_OUTLIERS = False
USE_LOG_TRANSFORM = False
USE_LINEAR_INTERPOLATION = False
```
**Purpose**: See the raw data, understand patterns, identify outliers

---

### Configuration 2: **Basic Regression**
```python
RUN_VISUALIZATIONS = False
USE_DESEASONALIZED = True
HANDLE_OUTLIERS = True
USE_LOG_TRANSFORM = False
USE_LINEAR_INTERPOLATION = False
```
**Purpose**: Clean deseasonalized regression on original scale

---

### Configuration 3: **Full Fredriksson Replication** (Golden Path)
```python
RUN_VISUALIZATIONS = True           # Document raw data
USE_DESEASONALIZED = True           # Remove seasonality
HANDLE_OUTLIERS = True              # Clean anomalies
USE_LOG_TRANSFORM = True            # Log-linear specification
USE_LINEAR_INTERPOLATION = False    # Conservative missing data handling
RUN_LJUNGBOX_TEST = True            # Check autocorrelation
RUN_STATIONARITY_TESTS = True       # Validate assumptions
```
**Purpose**: Complete preprocessing pipeline for thesis analysis

---

### Configuration 4: **Sensitivity Analysis**
```python
# Run multiple times with different settings:
HANDLE_OUTLIERS = True/False        # Compare with/without outlier handling
USE_LINEAR_INTERPOLATION = True/False  # Test interpolation impact
OPTIMIZE_ARMAX_LAGS = True          # Find optimal lag structure
```
**Purpose**: Test robustness of results to preprocessing choices

---

## Files and Directory Structure

```
EconMScThesis/
│
├── full_regression.py              # Main analysis script
├── README.md                        # This file
│
├── master data files/               # Merged master data (not in git)
│   ├── Spot_Prices.xlsx
│   ├── Master_Wind_Forecast_Merged_2021_2024.xlsx
│   ├── Master_Hydro_Reservoir.xlsx
│   ├── Master_Exchange_Merged_2021_2024.xlsx
│   └── Master_Consumption_2021_2024.xlsx
│
├── merger functions/                # Data merger scripts
│   ├── consumption_forecast_merger.py
│   ├── exchange_merger.py
│   ├── final_exchange_merger.py
│   ├── final_wind_merger.py
│   ├── hydro_merger.py
│   ├── price_merger.py
│   └── production_forecast_merger.py
│
├── data/                            # Raw data files (not in git)
│   ├── spot_price/
│   ├── hydro_reservoir_reserves/
│   └── ...
│
├── plots/                           # Generated visualizations
│   ├── raw_time_series_SE1.png
│   ├── raw_distributions_SE1.png
│   ├── raw_boxplots_SE1.png
│   ├── raw_outliers_timeline_SE1.png
│   ├── raw_scatter_matrix_SE1.png
│   └── outlier_summary_fredriksson_SE1.csv
│
├── Sources/                         # Reference papers and documentation
│   └── 2016_Fredriksson_SSE_Thesis_3214.pdf
│
└── Verified files (intermediate data, not in git):
    ├── Verified_Complete_Consumption_2021_2024.xlsx
    ├── Verified_S1_Wind_Forecast_2021_2024.xlsx
    └── ...
```

---

## Running the Analysis

### Basic Usage

```bash
# Activate virtual environment
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Run full analysis
python full_regression.py
```

### Modify Configuration

Edit `full_regression.py` lines 960-1020:
```python
# Change toggles as needed
RUN_VISUALIZATIONS = True
USE_DESEASONALIZED = True
HANDLE_OUTLIERS = True
# ... etc
```

### View Results

- **Plots**: Check `plots/` directory
- **Console output**: Regression tables, diagnostics, outlier reports
- **CSV**: `plots/outlier_summary_fredriksson_SE1.csv`

---

## Key Results (SE1, 2021-2024)

### Data Summary
- **Total observations**: 35,064 hours
- **Missing values**: 120 rows dropped (0.34%)
- **Time span**: 2021-01-01 to 2024-12-31

### Outlier Detection (Raw Data)
- **Price**: 135 outliers (0.39%)
  - Range: 333.69 - 590.00 EUR/MWh
  - All above upper threshold (no negative outliers)
- **Other variables**: 0 outliers

### Descriptive Statistics (Raw Data)
| Variable | Mean | Std Dev | Min | Max | Unit |
|----------|------|---------|-----|-----|------|
| Price | 41.50 | 48.51 | -60.04 | 590.00 | EUR/MWh |
| Wind Forecast | 671.71 | 575.52 | 3.00 | 2832.00 | MWh |
| Hydro Reserves | 8531.92 | 3102.25 | 1381.00 | 13759.00 | MWh |
| Net Exchange | -1736.68 | 833.93 | -4481.60 | 776.40 | MWh |
| Consumption | 1244.67 | 199.54 | 717.00 | 1798.00 | MWh |

---

## Methodological Notes for Thesis

### Transparency and Replicability

1. **Document all choices**: Every toggle, every threshold, every decision
2. **Report sensitivity**: Show results with/without outlier handling
3. **Justify deviations**: Explain why we differ from Fredriksson
4. **Provide code**: Make analysis fully reproducible

### What to Report in Thesis

**Data section**:
- Number of observations, time period, missing values
- Outlier detection results (how many, what threshold)
- Descriptive statistics before/after preprocessing

**Methodology section**:
- Explain preprocessing pipeline (with flowchart from this README)
- Justify order of operations (reference this document)
- Note deviation from Fredriksson (once vs. twice outlier handling)
- Provide theoretical reasoning for our approach

**Results section**:
- Main results with recommended configuration
- Robustness checks with alternative configurations
- Sensitivity to outlier handling (compare with/without)

**Appendix**:
- Full code listing
- Additional diagnostic plots
- Alternative specifications

---

## Future Work / TODO

1. **Sensitivity Analysis**:
   - [ ] Compare single vs. double outlier application
   - [ ] Test impact of interpolation vs. dropping missing values
   - [ ] Vary outlier thresholds (5σ, 7σ, etc.)

2. **Methodological Extensions**:
   - [ ] Implement GARCHX for volatility modeling
   - [ ] Test alternative deseasonalization methods (STL, X-13)
   - [ ] Rolling window analysis (time-varying coefficients)

3. **Additional Diagnostics**:
   - [ ] Structural break tests
   - [ ] Cross-validation for lag selection
   - [ ] Forecast evaluation metrics

4. **Visualization Enhancements**:
   - [ ] Interactive plots (Plotly)
   - [ ] Before/after outlier handling comparison plots
   - [ ] Seasonal decomposition plots

---

## References

**Primary Reference**:
- Fredriksson, F. (2016). *The Merit Order Effect of Wind Power on the Swedish Electricity Market*. Stockholm School of Economics Master's Thesis.

**Methodology**:
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.

**Energy Economics**:
- Woo, C. K., et al. (2011). The impact of wind generation on the electricity spot-market price level and variance. *Energy Policy*, 39(7), 3939-3944.
- Ketterer, J. C. (2014). The impact of wind power generation on the electricity price in Germany. *Energy Economics*, 44, 270-280.

---

## Contact

**Author**: [Your Name]
**Institution**: [Your University]
**Email**: [Your Email]
**Date**: January 2026

---

## License

This code is provided for academic research purposes. Please cite appropriately if used in publications.

---

**Last Updated**: 2026-01-19
