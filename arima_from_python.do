clear all
set more off
set linesize 120

* ============================================================================
* arima_from_python.do
*
* Reads the CSV files exported by full_regression.py and estimates ARIMA
* with QML robust standard errors.
*
* Equivalent to Python:
*   sm.tsa.ARIMA(y, exog=X, order=(p,d,q)).fit(cov_type='robust')
*
* Usage:
*   1. Run full_regression.py  →  CSV files written to output/
*   2. Set DATA_FILE below to the exported armax_input filename
*   3. Run this do-file in Stata
*
* The do-file auto-detects all controls from the CSV columns.
* Only change DATA_FILE when you export a new window or zone.
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "output/armax_input_SE1_2024-01-01_2025-12-31.csv"
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
* 3. Estimate ARIMA with QML robust SEs
* ============================================================================
quietly arima `depvar' `controls', arima(`arma_p',`arma_d',`arma_q') vce(robust)


* ============================================================================
* 4. Collect results
* ============================================================================
quietly {
    estat ic
    matrix _ic  = r(S)
    scalar _aic = _ic[1,5]
    scalar _bic = _ic[1,6]

    * Constant
    scalar _b_cons  = _b[`depvar':_cons]
    scalar _se_cons = _se[`depvar':_cons]
    scalar _p_cons  = 2*(1 - normal(abs(_b_cons / _se_cons)))

    * Exogenous controls
    foreach v of local controls {
        scalar _b_`v'  = _b[`depvar':`v']
        scalar _se_`v' = _se[`depvar':`v']
        scalar _p_`v'  = 2*(1 - normal(abs(_b_`v' / _se_`v')))
    }

    * AR terms
    forval i = 1/`arma_p' {
        scalar _b_ar`i'  = _b[ARMA:L`i'.ar]
        scalar _se_ar`i' = _se[ARMA:L`i'.ar]
        scalar _p_ar`i'  = 2*(1 - normal(abs(_b_ar`i' / _se_ar`i')))
    }

    * MA terms
    forval i = 1/`arma_q' {
        scalar _b_ma`i'  = _b[ARMA:L`i'.ma]
        scalar _se_ma`i' = _se[ARMA:L`i'.ma]
        scalar _p_ma`i'  = 2*(1 - normal(abs(_b_ma`i' / _se_ma`i')))
    }

    scalar _b_sigma = _b[/sigma]
}


* ============================================================================
* 5. Significance stars
* ============================================================================
local st_cons = cond(_p_cons < 0.01, "***", cond(_p_cons < 0.05, "**", cond(_p_cons < 0.10, "*", "")))

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
display "  ARIMA(`arma_p',`arma_d',`arma_q') — QML Robust SEs"
display "{hline 55}"
display "  " _col(28) "Coef."  _col(38) "Std.Err."  _col(51) "Sig."
display "{hline 55}"

* Constant
display "  Constant"     _col(28) %9.5f _b_cons  _col(38) %8.4f _se_cons  "  `st_cons'"

* Controls
foreach v of local controls {
    display "  `v'"         _col(28) %9.5f _b_`v'   _col(38) %8.4f _se_`v'   "  `st_`v''"
}

* AR lags
forval i = 1/`arma_p' {
    display "  ar.L`i'"    _col(28) %9.5f _b_ar`i'  _col(38) %8.4f _se_ar`i' "  `st_ar`i''"
}

* MA lags
forval i = 1/`arma_q' {
    display "  ma.L`i'"    _col(28) %9.5f _b_ma`i'  _col(38) %8.4f _se_ma`i' "  `st_ma`i''"
}

display "{hline 55}"
display "  sigma"          _col(28) %9.5f _b_sigma
display "{hline 55}"
display "  N"              _col(28) %12.0f e(N)
display "  Log-likelihood" _col(28) %12.3f e(ll)
display "  AIC"            _col(28) %12.3f _aic
display "  BIC"            _col(28) %12.3f _bic
display "{hline 55}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "  vce(robust) = QML sandwich SEs"
display "{hline 55}"


* ============================================================================
* 7. Ljung-Box diagnostics on residuals
* ============================================================================
local lb_lags "1 2 3 4 5 10 15 20"

quietly predict double _resid, residuals

display _n "{hline 45}"
display "  Ljung-Box test on residuals"
display "{hline 45}"
display "  " _col(8) "Lag" _col(18) "Statistic" _col(32) "p-value"
display "{hline 45}"

foreach lag of local lb_lags {
    quietly wntestq _resid, lags(`lag')
    local _rs = r(stat)
    local _rp = r(p)
    local rstar = cond(`_rp' < 0.05, "*", " ")
    display "  " _col(8) %3.0f `lag' ///
                 _col(18) %9.2f `_rs' ///
                 _col(32) %6.4f `_rp' "`rstar'"
}
display "{hline 45}"
display "  p>0.05 = no remaining autocorrelation"
display "{hline 45}"

drop _resid


* ============================================================================
* 8. Save results to CSV
* ============================================================================
local stub = subinstr("$DATA_FILE", "output/armax_input_", "", 1)
local stub = subinstr("`stub'", ".csv", "", 1)
capture mkdir "output/arima_results"

tempname out_post
tempfile out_file
postfile `out_post' str16 type str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

* Fit statistics
post `out_post' ("fit") ("N")   (e(N))   (.) (.) ("")
post `out_post' ("fit") ("ll")  (e(ll))  (.) (.) ("")
post `out_post' ("fit") ("AIC") (_aic)   (.) (.) ("")
post `out_post' ("fit") ("BIC") (_bic)   (.) (.) ("")

* LB diagnostics
quietly predict double _resid2, residuals
foreach lag of local lb_lags {
    quietly wntestq _resid2, lags(`lag')
    local _rs = r(stat)
    local _rp = r(p)
    local _rst = cond(`_rp' < 0.01, "***", cond(`_rp' < 0.05, "**", cond(`_rp' < 0.10, "*", "")))
    post `out_post' ("lbq_res") ("L`lag'") (`_rs') (.) (`_rp') ("`_rst'")
}
drop _resid2

* Coefficients
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

postclose `out_post'

quietly {
    use `out_file', clear
    export delimited using "output/arima_results/arima_results_`stub'.csv", replace
}
display _n "Results saved to: output/arima_results/arima_results_`stub'.csv"
