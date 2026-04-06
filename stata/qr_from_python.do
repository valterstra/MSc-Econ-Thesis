clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* qr_from_python.do
*
* Reads the CSV exported by python/stata_inputs/full_period_export.py and estimates a
* Quantile Autoregression with exogenous controls (QAR-X) at five
* conditional quantiles: τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}.
*
* Specification (Koenker & Xiao 2006, JASA):
*
*   Q_τ(price_ds_t | Ω_{t-1}, X_t) = α_0(τ)
*                                    + Σ_j ρ_j(τ)·price_ds_{t-j}
*                                    + Σ_k β_k(τ)·x_kt
*
* This is the conditional quantile of the deseasonalised log price given
* its own lagged values and contemporaneous fundamentals.  The coefficient
* β_wind(τ) is the merit-order effect at quantile τ: how much a 1-unit
* increase in log wind generation shifts the τ-th conditional quantile of
* price.  If the merit-order effect is larger at high quantiles, β_wind(τ)
* will be more negative as τ → 1.
*
* AR lag order:
*   p* = arma_p         (from companion metadata, matches GARCH specification)
*   p* = arma_p + 1     if arma_q > 0  (extra AR lag approximates the MA
*                        component, which is not identifiable in linear QR)
*
* Standard errors:
*   vce(robust) — Huber-White sandwich, robust to heteroskedasticity.
*   After including p* AR lags the conditional-mean residuals should be
*   approximately white noise, making vce(robust) reasonable.
*   For stronger guarantees (block bootstrap): replace qreg ... vce(robust)
*   with  bsqreg ... reps(400)  and set an appropriate block length.
*
* Diagnostics:
*   Ljung-Box Q(24) on quantile residuals at each τ to check for remaining
*   serial correlation in the conditional quantile.
*
* Output (long format — one row per quantile × parameter):
*   Columns:  quantile, type, variable, value, se, pval, stars
*   type:     "coef"  — regression coefficients (controls + AR lags + const)
*             "fit"   — N and Koenker-Machado pseudo-R²
*             "lbq"   — Ljung-Box Q(24) on quantile residuals
*   Saved to: output from stata/qr_results/qr_results_<stub>.csv
*
* Usage:
*   1. Run python/stata_inputs/full_period_export.py → armax_input_*.csv + armax_meta_*.csv in
*      stata_input/armax/
*   2. Set DATA_FILE below to the desired armax_input CSV
*   3. Run this do-file in Stata
*   Only DATA_FILE needs to change between zones / windows.
* ============================================================================


* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/armax_garch/input_SE4_2025-01-01_2025-12-31_log.csv"
* Quantiles to estimate
local quantile_list "0.10 0.25 0.50 0.75 0.90 0.95 0.99"
* -----------------------------------------------------------------------------


* ============================================================================
* 1. ARMA order (hardcoded to match garch_from_python.do)
* ============================================================================
local arma_p = 1
local arma_d = 0
local arma_q = 1


* ============================================================================
* 2. Load data and set up time series
* ============================================================================
quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)   // hourly (1h = 3,600,000 ms)
    drop timestamp
}

local depvar price_ds

* Controls: every column except time index and dependent variable
ds stata_clock `depvar', not
local controls `r(varlist)'

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "ARMA order from metadata: p=`arma_p'  d=`arma_d'  q=`arma_q'"


* ============================================================================
* 3. Create lagged dependent variable as explicit AR regressors (QAR)
*
* Linear quantile regression cannot iterate MA terms; we instead include
* enough AR lags to whiten the conditional-mean dynamics.  When the
* companion GARCH used an MA component (arma_q > 0), we add one extra AR
* lag as a finite-order approximation.
* ============================================================================
local total_ar_lags = `arma_p'
if `arma_q' > 0  local total_ar_lags = `total_ar_lags' + 1

forval i = 1/`total_ar_lags' {
    quietly gen double L`i'_price_ds = L`i'.`depvar'
}

* Drop initial rows where any lag is missing
quietly drop if missing(L`total_ar_lags'_price_ds)

local ar_lags ""
forval i = 1/`total_ar_lags' {
    local ar_lags "`ar_lags' L`i'_price_ds"
}

display    "AR lags in QR: `ar_lags'  (total=`total_ar_lags')"
display    "Quantiles: `quantile_list'" _n


* ============================================================================
* 4. Open postfile for long-format results
* ============================================================================
capture mkdir "output from stata/qr_results"

* Build output stub from DATA_FILE path
local stub = subinstr("$DATA_FILE", "stata_input/armax_garch/input_", "", 1)
local stub = subinstr("`stub'", ".csv", "", 1)

tempname out_post
tempfile  out_file

postfile `out_post' ///
    double quantile  ///
    str16  type      ///
    str32  variable  ///
    double value     ///
    double se        ///
    double pval      ///
    str4   stars     ///
    using `out_file', replace


* ============================================================================
* 5. Main quantile loop
* ============================================================================
display "{hline 72}"
display "  QAR-X  —  conditional quantiles of `depvar'"
display "  AR lags: `ar_lags'"
display "  Controls: `controls'"
display "{hline 72}"

foreach tau of local quantile_list {

    * -----------------------------------------------------------------------
    * 5a. Estimate at this quantile
    * -----------------------------------------------------------------------
    quietly qreg `depvar' `controls' `ar_lags', quantile(`tau') vce(robust)

    local N_obs     = e(N)
    * Koenker-Machado pseudo-R²: 1 - weighted deviance / null deviance
    local pseudo_r2 = 1 - e(sum_rdev) / e(sum_adev)

    * -----------------------------------------------------------------------
    * 5b. Collect coefficients (stored as scalars for safe display in loops)
    * -----------------------------------------------------------------------
    * Constant
    scalar _b_cons  = _b[_cons]
    scalar _se_cons = _se[_cons]
    scalar _p_cons  = 2*(1 - normal(abs(scalar(_b_cons) / scalar(_se_cons))))

    * Exogenous controls
    foreach v of local controls {
        scalar _b_`v'  = _b[`v']
        scalar _se_`v' = _se[`v']
        scalar _p_`v'  = 2*(1 - normal(abs(scalar(_b_`v') / scalar(_se_`v'))))
    }

    * AR lags
    forval i = 1/`total_ar_lags' {
        scalar _b_ar`i'  = _b[L`i'_price_ds]
        scalar _se_ar`i' = _se[L`i'_price_ds]
        scalar _p_ar`i'  = 2*(1 - normal(abs(scalar(_b_ar`i') / scalar(_se_ar`i'))))
    }

    * -----------------------------------------------------------------------
    * 5c. Significance stars
    * -----------------------------------------------------------------------
    local st_cons = cond(scalar(_p_cons) < 0.01, "***", ///
                    cond(scalar(_p_cons) < 0.05, "**",  ///
                    cond(scalar(_p_cons) < 0.10, "*", "")))

    foreach v of local controls {
        local st_`v' = cond(scalar(_p_`v') < 0.01, "***", ///
                       cond(scalar(_p_`v') < 0.05, "**",  ///
                       cond(scalar(_p_`v') < 0.10, "*", "")))
    }

    forval i = 1/`total_ar_lags' {
        local st_ar`i' = cond(scalar(_p_ar`i') < 0.01, "***", ///
                         cond(scalar(_p_ar`i') < 0.05, "**",  ///
                         cond(scalar(_p_ar`i') < 0.10, "*", "")))
    }

    * -----------------------------------------------------------------------
    * 5d. Ljung-Box Q(24) on quantile residuals
    *     Tests for remaining serial correlation in the conditional quantile.
    *     Ideally p > 0.05; failure indicates AR lag order is insufficient.
    * -----------------------------------------------------------------------
    quietly {
        predict double _qr_resid, residuals
        wntestq _qr_resid, lags(24)
        scalar _Q_lbq = r(stat)
        scalar _p_lbq = r(p)
        drop _qr_resid
    }
    local st_lbq = cond(scalar(_p_lbq) < 0.01, "***", ///
                   cond(scalar(_p_lbq) < 0.05, "**",  ///
                   cond(scalar(_p_lbq) < 0.10, "*", "")))

    * -----------------------------------------------------------------------
    * 5e. Post results to file
    * -----------------------------------------------------------------------

    * Fit statistics
    post `out_post' (`tau') ("fit") ("N")         (`N_obs')     (.) (.)              ("")
    post `out_post' (`tau') ("fit") ("pseudo_r2")  (`pseudo_r2') (.) (.)              ("")

    * LBQ diagnostic
    post `out_post' (`tau') ("lbq") ("L24") (scalar(_Q_lbq)) (.) (scalar(_p_lbq)) ("`st_lbq'")

    * Constant
    post `out_post' (`tau') ("coef") ("_cons") ///
        (scalar(_b_cons)) (scalar(_se_cons)) (scalar(_p_cons)) ("`st_cons'")

    * Exogenous controls
    foreach v of local controls {
        post `out_post' (`tau') ("coef") ("`v'") ///
            (scalar(_b_`v')) (scalar(_se_`v')) (scalar(_p_`v')) ("`st_`v''")
    }

    * AR lags
    forval i = 1/`total_ar_lags' {
        post `out_post' (`tau') ("coef") ("ar_L`i'") ///
            (scalar(_b_ar`i')) (scalar(_se_ar`i')) (scalar(_p_ar`i')) ("`st_ar`i''")
    }

    * -----------------------------------------------------------------------
    * 5f. Print coefficient table for this quantile
    * -----------------------------------------------------------------------
    display _n "  τ = " %4.2f `tau' ///
        "   N=" %7.0f `N_obs' ///
        "   pseudo-R² = " %6.4f `pseudo_r2'
    display "  " _col(5) "Variable" _col(26) "Coef." _col(38) "Std.Err." _col(50) "p-val" _col(58) "Sig."
    display "  {hline 62}"

    display "  _cons" ///
        _col(26) %9.5f scalar(_b_cons) ///
        _col(38) %8.4f scalar(_se_cons) ///
        _col(50) %6.4f scalar(_p_cons) ///
        _col(58) "`st_cons'"

    foreach v of local controls {
        display "  `v'" ///
            _col(26) %9.5f scalar(_b_`v') ///
            _col(38) %8.4f scalar(_se_`v') ///
            _col(50) %6.4f scalar(_p_`v') ///
            _col(58) "`st_`v''"
    }

    forval i = 1/`total_ar_lags' {
        display "  ar_L`i'" ///
            _col(26) %9.5f scalar(_b_ar`i') ///
            _col(38) %8.4f scalar(_se_ar`i') ///
            _col(50) %6.4f scalar(_p_ar`i') ///
            _col(58) "`st_ar`i''"
    }

    display "  LB-Q(24) on resid.: stat=" %7.2f scalar(_Q_lbq) ///
        "   p=" %6.4f scalar(_p_lbq) "  `st_lbq'"
}


* ============================================================================
* 6. Save results
* ============================================================================
postclose `out_post'

quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/qr_results/qr_results_`stub'.csv", replace
}

display _n "{hline 72}"
display "Results saved to: output from stata/qr_results/qr_results_`stub'.csv"
display "{hline 72}"
display "*** p<0.01  ** p<0.05  * p<0.10"
display "SEs: vce(robust) — Huber-White heteroskedasticity-consistent"
display "For block-bootstrap SEs (stronger serial-correlation robustness):"
display "  Replace qreg ... vce(robust) with bsqreg ... reps(400)"
display "{hline 72}"
