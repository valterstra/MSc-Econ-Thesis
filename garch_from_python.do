clear all
set more off
set linesize 120

* ============================================================================
* garch_from_python.do
*
* Reads the CSV files exported by full_regression.py and estimates
* ARMAX(p,d,q)-GARCH(1,1) with Bollerslev-Wooldridge robust (QML) SEs.
*
* Mean eq.:  price_ds = c + β·X_t + AR(1..p) + MA(1..q) + ε_t
* Var. eq.:  h_t = ω + α·ε²_{t-1} + δ·h_{t-1}
*
* Usage:
*   1. Run full_regression.py  →  CSV files written to output/
*   2. Set DATA_FILE below to the exported armax_input filename
*   3. Run this do-file in Stata
*
* The do-file auto-detects all controls from the CSV columns.
* ARMA order is read from the companion metadata file.
* Only change DATA_FILE when you export a new window or zone.
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "output/armax_input_SE4_2024-01-01_2025-12-31.csv"
* -----------------------------------------------------------------------------


* ============================================================================
* 1. Read ARIMA order from companion metadata file
* ============================================================================
local meta_file = subinstr("$DATA_FILE", "armax_input_", "armax_meta_", 1)
import delimited using "`meta_file'", clear varnames(1)
local arma_p = arma_p[1]
local arma_d = arma_d[1]
local arma_q = arma_q[1]


* ============================================================================
* 2. Load model data and set up time series
* ============================================================================
quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)   // hourly (1h = 3,600,000 ms)
    drop timestamp
}

* Dependent variable
local depvar price_ds

* Controls: every column except the time index and dependent variable
ds stata_clock `depvar', not
local controls `r(varlist)'

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"


* ============================================================================
* 3. Estimate ARMAX(p,d,q)-GARCH(1,1) with QML robust SEs
*
* ar_spec / ma_spec are built dynamically from the metadata so the do-file
* works for any ARMA order without editing.
* vce(robust) = Bollerslev-Wooldridge (1992) sandwich SEs (QML).
* difficult   = BFGS stepping; helps convergence near unit-root AR processes.
* ============================================================================
local ar_spec ""
local ma_spec ""
if `arma_p' > 0  local ar_spec "ar(1/`arma_p')"
if `arma_q' > 0  local ma_spec "ma(1/`arma_q')"

quietly arch `depvar' `controls',   ///
    `ar_spec' `ma_spec'             ///
    arch(1) garch(1)                ///
    distribution(gaussian)          ///
    vce(robust)                     ///
    difficult


* ============================================================================
* 4. Collect results
* ============================================================================
quietly {
    estat ic
    matrix _ic  = r(S)
    scalar _aic = _ic[1,5]
    scalar _bic = _ic[1,6]

    * --- Mean equation ---
    scalar _b_cons  = _b[`depvar':_cons]
    scalar _se_cons = _se[`depvar':_cons]
    scalar _p_cons  = 2*(1 - normal(abs(_b_cons / _se_cons)))

    foreach v of local controls {
        scalar _b_`v'  = _b[`depvar':`v']
        scalar _se_`v' = _se[`depvar':`v']
        scalar _p_`v'  = 2*(1 - normal(abs(_b_`v' / _se_`v')))
    }

    forval i = 1/`arma_p' {
        scalar _b_ar`i'  = _b[ARMA:L`i'.ar]
        scalar _se_ar`i' = _se[ARMA:L`i'.ar]
        scalar _p_ar`i'  = 2*(1 - normal(abs(_b_ar`i' / _se_ar`i')))
    }

    forval i = 1/`arma_q' {
        scalar _b_ma`i'  = _b[ARMA:L`i'.ma]
        scalar _se_ma`i' = _se[ARMA:L`i'.ma]
        scalar _p_ma`i'  = 2*(1 - normal(abs(_b_ma`i' / _se_ma`i')))
    }

    * --- Variance equation ---
    scalar _b_omega  = _b[ARCH:_cons]
    scalar _se_omega = _se[ARCH:_cons]
    scalar _p_omega  = 2*(1 - normal(abs(_b_omega / _se_omega)))

    scalar _b_arch1  = _b[ARCH:L1.arch]
    scalar _se_arch1 = _se[ARCH:L1.arch]
    scalar _p_arch1  = 2*(1 - normal(abs(_b_arch1 / _se_arch1)))

    scalar _b_garch1  = _b[ARCH:L1.garch]
    scalar _se_garch1 = _se[ARCH:L1.garch]
    scalar _p_garch1  = 2*(1 - normal(abs(_b_garch1 / _se_garch1)))

    scalar _persistence = _b_arch1 + _b_garch1

    * --- Diagnostics on standardized residuals ---
    predict double _resid, residuals
    predict double _ht,    variance
    gen double _std_res    = _resid / sqrt(_ht)
    gen double _std_res_sq = _std_res^2

    wntestq _std_res, lags(24)
    scalar _Q_std   = r(stat)
    scalar _p_Q_std = r(p)

    wntestq _std_res_sq, lags(24)
    scalar _Q_sq   = r(stat)
    scalar _p_Q_sq = r(p)
}


* ============================================================================
* 5. Significance stars
* ============================================================================
local st_cons  = cond(_p_cons  < 0.01, "***", cond(_p_cons  < 0.05, "**", cond(_p_cons  < 0.10, "*", "")))
local st_omega = cond(_p_omega < 0.01, "***", cond(_p_omega < 0.05, "**", cond(_p_omega < 0.10, "*", "")))
local st_arch1 = cond(_p_arch1 < 0.01, "***", cond(_p_arch1 < 0.05, "**", cond(_p_arch1 < 0.10, "*", "")))
local st_garch1= cond(_p_garch1< 0.01, "***", cond(_p_garch1< 0.05, "**", cond(_p_garch1< 0.10, "*", "")))

foreach v of local controls {
    local st_`v' = cond(_p_`v' < 0.01, "***", cond(_p_`v' < 0.05, "**", cond(_p_`v' < 0.10, "*", "")))
}

forval i = 1/`arma_p' {
    local st_ar`i' = cond(_p_ar`i' < 0.01, "***", cond(_p_ar`i' < 0.05, "**", cond(_p_ar`i' < 0.10, "*", "")))
}

forval i = 1/`arma_q' {
    local st_ma`i' = cond(_p_ma`i' < 0.01, "***", cond(_p_ma`i' < 0.05, "**", cond(_p_ma`i' < 0.10, "*", "")))
}


* ============================================================================
* 6. Print results table
* ============================================================================
display _n "{hline 55}"
display "  ARMA(`arma_p',`arma_q')-GARCH(1,1) — QML Robust SEs"
display "{hline 55}"
display "  " _col(28) "Coef."  _col(38) "Std.Err."  _col(51) "Sig."
display "{hline 55}"
display "  Mean equation"
display "{hline 55}"

* Constant
display "  Constant"  _col(28) %9.5f _b_cons  _col(38) %8.4f _se_cons  "  `st_cons'"

* Controls
foreach v of local controls {
    display "  `v'"  _col(28) %9.5f _b_`v'  _col(38) %8.4f _se_`v'  "  `st_`v''"
}

* AR lags
forval i = 1/`arma_p' {
    display "  ar.L`i'"  _col(28) %9.5f _b_ar`i'  _col(38) %8.4f _se_ar`i'  "  `st_ar`i''"
}

* MA lags
forval i = 1/`arma_q' {
    display "  ma.L`i'"  _col(28) %9.5f _b_ma`i'  _col(38) %8.4f _se_ma`i'  "  `st_ma`i''"
}

display "{hline 55}"
display "  Variance equation"
display "{hline 55}"
display "  omega (ω)"  _col(28) %9.5f _b_omega  _col(38) %8.4f _se_omega  "  `st_omega'"
display "  ARCH(1) (α)"  _col(28) %9.5f _b_arch1  _col(38) %8.4f _se_arch1  "  `st_arch1'"
display "  GARCH(1) (δ)"  _col(28) %9.5f _b_garch1 _col(38) %8.4f _se_garch1 "  `st_garch1'"
display "{hline 55}"

* Persistence warning
display "  Persistence (α+δ)"  _col(28) %9.5f _persistence
if _persistence >= 1 {
    display "  WARNING: persistence >= 1 — variance non-stationary"
}

display "{hline 55}"
display "  N"               _col(28) %12.0f e(N)
display "  Log-likelihood"  _col(28) %12.3f e(ll)
display "  AIC"             _col(28) %12.3f _aic
display "  BIC"             _col(28) %12.3f _bic
display "{hline 55}"
display "  LB-Q(24) std. resid.     " %8.2f _Q_std  "  p=" %6.4f _p_Q_std
display "  LB-Q(24) sq. std. resid. " %8.2f _Q_sq   "  p=" %6.4f _p_Q_sq
display "  (sq. resid. LB-Q = remaining ARCH; ideally p > 0.05)"
display "{hline 55}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "  vce(robust) = Bollerslev-Wooldridge QML SEs"
display "{hline 55}"
