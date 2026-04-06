clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* garch_from_python_joint_t.do
*
* Joint ARMAX(1,0,1)-GARCH-X(1,1) with Student-t errors.
* Identical to garch_from_python.do but replaces distribution(gaussian)
* with distribution(t), with degrees of freedom fixed at T_DF.
*
* Additionally computes a pre-GARCH Ljung-Box diagnostic on squared raw
* ARMAX residuals (from a preliminary arima run) so you can compare:
*   lbq_sq_pre  — LBQ(24) on ε²_t from ARMAX only (no GARCH)
*   lbq_sq      — LBQ(24) on (ε_t/√h_t)² after joint GARCH
* The arima pre-run is purely diagnostic; it does not affect the joint
* estimation that follows.
*
* Usage:
*   1. Run python/stata_inputs/full_period_export.py  →  CSV files written to stata_input/
*   2. Set DATA_FILE, HET_ALL_CONTROLS, T_DF below
*   3. Run this do-file in Stata
*
* Output:
*   output from stata/garch_results/2025_results/
*       garch_results_{stub}_joint_tdf{T_DF}.csv
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/armax_garch/input_SE1_2025-01-01_2025-12-31_log.csv"
* 0 = wind_log only in variance eq. (baseline)
* 1 = all exogenous controls in variance eq. (extended GARCH-X)
global HET_ALL_CONTROLS 1
* Degrees of freedom for Student-t errors.
* Set to a positive integer >= 3 to fix df (e.g. 5, 6, 8).
* Set to 0 to estimate df freely from the data.
global T_DF 5
* ARMA order
local arma_p = 1
local arma_d = 0
local arma_q = 1
* -----------------------------------------------------------------------------

set maxiter 100

* Build distribution spec
if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
    local df_label "_tdf$T_DF"
}
else {
    local dist_spec "distribution(t)"
    local df_label "_tdfest"
}


* ============================================================================
* 1. Load data and set up time series
* ============================================================================
quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
    drop timestamp
}

local depvar price_ds

ds stata_clock `depvar', not
local controls `r(varlist)'

* Build het() variable list
if $HET_ALL_CONTROLS {
    local het_controls `controls'
    local het_label "all controls"
    local het_suffix "_hetfull"
}
else {
    local het_controls wind_log
    local het_label "wind_log only"
    local het_suffix "_hetwind"
}

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "Variance het(): `het_label'"
display    "Distribution: `dist_spec'"


* ============================================================================
* 2. Pre-GARCH diagnostic — ARMAX only, LBQ on squared raw residuals
*
* This preliminary arima run is used solely to compute how much squared-
* residual autocorrelation exists before the GARCH variance model is applied.
* Comparing lbq_sq_pre with lbq_sq (post-GARCH) shows how much volatility
* clustering the GARCH component absorbs.
* ============================================================================
display _n "--- Pre-GARCH diagnostic (arima only) ---"

capture arima `depvar' `controls', arima(`arma_p',`arma_d',`arma_q') vce(robust) difficult
if _rc != 0 {
    display "  NOTE: pre-GARCH arima did not converge (rc=" _rc ") — pre-GARCH LBQ set to missing"
    scalar _Q_sq_pre   = .
    scalar _p_Q_sq_pre = .
}
else {
    quietly {
        predict double _resid_pre, residuals
        gen double _resid_pre_sq = _resid_pre^2
        wntestq _resid_pre_sq, lags(24)
        scalar _Q_sq_pre   = r(stat)
        scalar _p_Q_sq_pre = r(p)
        drop _resid_pre _resid_pre_sq
    }
    display "  LBQ(24) on squared raw ARMAX residuals: " ///
        %8.2f _Q_sq_pre "  p=" %6.4f _p_Q_sq_pre
}


* ============================================================================
* 3. Estimate ARMAX(p,d,q)-GARCH-X(1,1) — joint MLE, t-distributed errors
* ============================================================================
local ar_spec ""
local ma_spec ""
if `arma_p' > 0  local ar_spec "ar(1/`arma_p')"
if `arma_q' > 0  local ma_spec "ma(1/`arma_q')"

capture arch `depvar' `controls',   ///
    `ar_spec' `ma_spec'             ///
    arch(1) garch(1)                ///
    het(`het_controls')             ///
    `dist_spec'                     ///
    vce(robust)                     ///
    difficult                       ///
    nrtolerance(1e-3)

if _rc != 0 & _rc != 430 {
    display as error "arch failed with rc=" _rc " — aborting"
    exit _rc
}
if _rc == 430 {
    display as text "NOTE: convergence not achieved (r(430)) — " ///
        "estimates at last iteration saved; treat with caution"
}


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
    scalar _b_omega  = _b[HET:_cons]
    scalar _se_omega = _se[HET:_cons]
    scalar _p_omega  = 2*(1 - normal(abs(_b_omega / _se_omega)))

    scalar _b_arch1  = _b[ARCH:L1.arch]
    scalar _se_arch1 = _se[ARCH:L1.arch]
    scalar _p_arch1  = 2*(1 - normal(abs(_b_arch1 / _se_arch1)))

    scalar _b_garch1  = _b[ARCH:L1.garch]
    scalar _se_garch1 = _se[ARCH:L1.garch]
    scalar _p_garch1  = 2*(1 - normal(abs(_b_garch1 / _se_garch1)))

    scalar _persistence = _b_arch1 + _b_garch1

    foreach v of local het_controls {
        scalar _b_het_`v'  = _b[HET:`v']
        scalar _se_het_`v' = _se[HET:`v']
        scalar _p_het_`v'  = 2*(1 - normal(abs(_b_het_`v' / _se_het_`v')))
    }

    * --- Post-GARCH diagnostics on standardized residuals ---
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
local st_cons   = cond(_p_cons   < 0.01, "***", cond(_p_cons   < 0.05, "**", cond(_p_cons   < 0.10, "*", "")))
local st_omega  = cond(_p_omega  < 0.01, "***", cond(_p_omega  < 0.05, "**", cond(_p_omega  < 0.10, "*", "")))
local st_arch1  = cond(_p_arch1  < 0.01, "***", cond(_p_arch1  < 0.05, "**", cond(_p_arch1  < 0.10, "*", "")))
local st_garch1 = cond(_p_garch1 < 0.01, "***", cond(_p_garch1 < 0.05, "**", cond(_p_garch1 < 0.10, "*", "")))
foreach v of local het_controls {
    local st_het_`v' = cond(_p_het_`v' < 0.01, "***", cond(_p_het_`v' < 0.05, "**", cond(_p_het_`v' < 0.10, "*", "")))
}
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
display _n "{hline 60}"
display "  ARMA(`arma_p',`arma_q')-GARCH-X(1,1) — Joint t($T_DF) QML  [het: `het_label']"
display "{hline 60}"
display "  " _col(28) "Coef."  _col(38) "Std.Err."  _col(51) "Sig."
display "{hline 60}"
display "  Mean equation"
display "{hline 60}"

display "  Constant"  _col(28) %9.5f _b_cons  _col(38) %8.4f _se_cons  "  `st_cons'"
foreach v of local controls {
    display "  `v'"  _col(28) %9.5f _b_`v'  _col(38) %8.4f _se_`v'  "  `st_`v''"
}
forval i = 1/`arma_p' {
    display "  ar.L`i'"  _col(28) %9.5f _b_ar`i'  _col(38) %8.4f _se_ar`i'  "  `st_ar`i''"
}
forval i = 1/`arma_q' {
    display "  ma.L`i'"  _col(28) %9.5f _b_ma`i'  _col(38) %8.4f _se_ma`i'  "  `st_ma`i''"
}

display "{hline 60}"
display "  Variance equation"
display "{hline 60}"
display "  omega (ω)"     _col(28) %9.5f _b_omega   _col(38) %8.4f _se_omega   "  `st_omega'"
display "  ARCH(1) (α)"  _col(28) %9.5f _b_arch1   _col(38) %8.4f _se_arch1   "  `st_arch1'"
display "  GARCH(1) (δ)" _col(28) %9.5f _b_garch1  _col(38) %8.4f _se_garch1  "  `st_garch1'"
foreach v of local het_controls {
    display "  `v' (γ)"  _col(28) %9.5f _b_het_`v' _col(38) %8.4f _se_het_`v' "  `st_het_`v''"
}
display "{hline 60}"

display "  Persistence (α+δ)"  _col(28) %9.5f _persistence
if _persistence >= 1 {
    display "  WARNING: persistence >= 1 — variance non-stationary"
}

display "{hline 60}"
display "  N"               _col(28) %12.0f e(N)
display "  Log-likelihood"  _col(28) %12.3f e(ll)
display "  AIC"             _col(28) %12.3f _aic
display "  BIC"             _col(28) %12.3f _bic
display "{hline 60}"
display "  LBQ(24) sq. raw ARMAX resid.    " %8.2f _Q_sq_pre "  p=" %6.4f _p_Q_sq_pre ///
    "  (pre-GARCH)"
display "  LBQ(24) std. resid.             " %8.2f _Q_std    "  p=" %6.4f _p_Q_std
display "  LBQ(24) sq. std. resid.         " %8.2f _Q_sq     "  p=" %6.4f _p_Q_sq ///
    "  (post-GARCH)"
display "  (comparing pre vs post shows how much ARCH the GARCH absorbed)"
display "{hline 60}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "  vce(robust) = Bollerslev-Wooldridge QML SEs"
display "{hline 60}"


* ============================================================================
* 7. Save results to CSV
* ============================================================================
local stub = subinstr("$DATA_FILE", "stata_input/armax_garch/input_", "", 1)
local stub = subinstr("`stub'", ".csv", "", 1)
local stub = subinstr("`stub'", "_log", "", 1)
capture mkdir "output from stata/garch_results/2025_results"

tempname out_post
tempfile out_file
postfile `out_post' str16 type str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

* Fit statistics
post `out_post' ("fit") ("N")   (e(N))  (.) (.) ("")
post `out_post' ("fit") ("ll")  (e(ll)) (.) (.) ("")
post `out_post' ("fit") ("AIC") (_aic)  (.) (.) ("")
post `out_post' ("fit") ("BIC") (_bic)  (.) (.) ("")

* LBQ diagnostics
local _rst_pre = cond(_p_Q_sq_pre < 0.01, "***", cond(_p_Q_sq_pre < 0.05, "**", cond(_p_Q_sq_pre < 0.10, "*", "")))
local _rst_std = cond(_p_Q_std    < 0.01, "***", cond(_p_Q_std    < 0.05, "**", cond(_p_Q_std    < 0.10, "*", "")))
local _rst_sq  = cond(_p_Q_sq     < 0.01, "***", cond(_p_Q_sq     < 0.05, "**", cond(_p_Q_sq     < 0.10, "*", "")))
post `out_post' ("lbq_sq_pre") ("L24") (_Q_sq_pre) (.) (_p_Q_sq_pre) ("`_rst_pre'")
post `out_post' ("lbq_std")    ("L24") (_Q_std)    (.) (_p_Q_std)    ("`_rst_std'")
post `out_post' ("lbq_sq")     ("L24") (_Q_sq)     (.) (_p_Q_sq)     ("`_rst_sq'")

* Mean equation coefficients
post `out_post' ("coef") ("_cons") (_b_cons) (_se_cons) (_p_cons) ("`st_cons'")
foreach v of local controls {
    post `out_post' ("coef") ("`v'") (_b_`v') (_se_`v') (_p_`v') ("`st_`v''")
}
forval i = 1/`arma_p' {
    post `out_post' ("coef") ("ar_L`i'") (_b_ar`i') (_se_ar`i') (_p_ar`i') ("`st_ar`i''")
}
forval i = 1/`arma_q' {
    post `out_post' ("coef") ("ma_L`i'") (_b_ma`i') (_se_ma`i') (_p_ma`i') ("`st_ma`i''")
}

* Variance equation coefficients
post `out_post' ("var_coef") ("omega")       (_b_omega)     (_se_omega)    (_p_omega)    ("`st_omega'")
post `out_post' ("var_coef") ("arch1")       (_b_arch1)     (_se_arch1)    (_p_arch1)    ("`st_arch1'")
post `out_post' ("var_coef") ("garch1")      (_b_garch1)    (_se_garch1)   (_p_garch1)   ("`st_garch1'")
foreach v of local het_controls {
    post `out_post' ("var_coef") ("het_`v'") (_b_het_`v') (_se_het_`v') (_p_het_`v') ("`st_het_`v''")
}
post `out_post' ("var_coef") ("persistence") (_persistence) (.) (.) ("")

postclose `out_post'

drop _std_res _std_res_sq _resid _ht

quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/garch_results/2025_results/garch_results_`stub'_joint`df_label'.csv", replace
}
display _n "Results saved to: output from stata/garch_results/2025_results/garch_results_`stub'_joint`df_label'.csv"
