clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* garch_from_python_joint_t_northern_spillover.do
*
* Joint ARMAX-GARCH-X with Student-t errors for SE3 / SE4, adding:
*   north_wind_log = ln(wind_SE1 + wind_SE2)
*
* Purpose:
*   test whether northern wind surplus affects southern prices beyond what
*   local wind already captures.
*
* Mean equation adds north_wind_log as an extra regressor.
* By default it is excluded from het(); flip HET_INCLUDE_NORTH to include it.
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/northern_spillover/northern_spillover_SE4_2023-01-01_2023-12-31_log.csv"
global HET_ALL_CONTROLS 1
global HET_INCLUDE_NORTH 0
global T_DF 5
local arma_p = 3
local arma_d = 0
local arma_q = 1
* -----------------------------------------------------------------------------

set maxiter 100

if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
    local df_label "_tdf$T_DF"
    local dist_display "Student-t (df=$T_DF, fixed)"
}
else {
    local dist_spec "distribution(t)"
    local df_label "_tdfest"
    local dist_display "Student-t (df estimated)"
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
local north_only "north_wind_log"
local het_base : list controls - north_only

if $HET_ALL_CONTROLS {
    local het_controls `het_base'
    local het_label "all controls (excl. north_wind_log)"
}
else {
    local het_controls wind_log
    local het_label "wind_log only"
}

if $HET_INCLUDE_NORTH {
    local het_controls `het_controls' north_wind_log
    local het_label "`het_label' + north_wind_log"
}

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "Variance het(): `het_label'"
display    "Distribution: `dist_display'"


* ============================================================================
* 2. Estimate model
* ============================================================================
capture arch `depvar' `controls',   ///
    ar(1/`arma_p') ma(1/`arma_q')   ///
    arch(1) garch(1)                ///
    het(`het_controls')             ///
    `dist_spec'                     ///
    vce(robust)                     ///
    difficult                       ///
    nrtolerance(1e-3)

if _rc != 0 & _rc != 430 {
    display as error "arch failed with rc=" _rc " - aborting"
    exit _rc
}
if _rc == 430 {
    display as text "NOTE: convergence not achieved (r(430)); last iteration saved"
}


* ============================================================================
* 3. Collect results
* ============================================================================
quietly {
    estat ic
    matrix _ic  = r(S)
    scalar _aic = _ic[1,5]
    scalar _bic = _ic[1,6]

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
* 4. Save results to CSV
* ============================================================================
local stub = subinstr("$DATA_FILE", "stata_input/northern_spillover/northern_spillover_", "", 1)
local stub = subinstr("`stub'", ".csv", "", 1)
local out_dir "output from stata/garch_results/northern_spillover"
capture mkdir "output from stata/garch_results"
capture mkdir "`out_dir'"

tempname out_post
tempfile out_file
postfile `out_post' str16 type str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

local st_cons   = cond(_p_cons   < 0.01, "***", cond(_p_cons   < 0.05, "**", cond(_p_cons   < 0.10, "*", "")))
local st_omega  = cond(_p_omega  < 0.01, "***", cond(_p_omega  < 0.05, "**", cond(_p_omega  < 0.10, "*", "")))
local st_arch1  = cond(_p_arch1  < 0.01, "***", cond(_p_arch1  < 0.05, "**", cond(_p_arch1  < 0.10, "*", "")))
local st_garch1 = cond(_p_garch1 < 0.01, "***", cond(_p_garch1 < 0.05, "**", cond(_p_garch1 < 0.10, "*", "")))

post `out_post' ("fit") ("N")           (e(N))       (.)           (.)           ("")
post `out_post' ("fit") ("ll")          (e(ll))      (.)           (.)           ("")
post `out_post' ("fit") ("AIC")         (_aic)       (.)           (.)           ("")
post `out_post' ("fit") ("BIC")         (_bic)       (.)           (.)           ("")
post `out_post' ("fit") ("Q_std")       (_Q_std)     (.)           (_p_Q_std)    ("")
post `out_post' ("fit") ("Q_sq")        (_Q_sq)      (.)           (_p_Q_sq)     ("")

post `out_post' ("coef") ("_cons")      (_b_cons)    (_se_cons)    (_p_cons)     ("`st_cons'")
foreach v of local controls {
    local st_`v' = cond(_p_`v' < 0.01, "***", cond(_p_`v' < 0.05, "**", cond(_p_`v' < 0.10, "*", "")))
    post `out_post' ("coef") ("`v'")    (_b_`v')     (_se_`v')     (_p_`v')      ("`st_`v''")
}
forval i = 1/`arma_p' {
    local st_ar`i' = cond(_p_ar`i' < 0.01, "***", cond(_p_ar`i' < 0.05, "**", cond(_p_ar`i' < 0.10, "*", "")))
    post `out_post' ("coef") ("ar_L`i'") (_b_ar`i')  (_se_ar`i')   (_p_ar`i')    ("`st_ar`i''")
}
forval i = 1/`arma_q' {
    local st_ma`i' = cond(_p_ma`i' < 0.01, "***", cond(_p_ma`i' < 0.05, "**", cond(_p_ma`i' < 0.10, "*", "")))
    post `out_post' ("coef") ("ma_L`i'") (_b_ma`i')  (_se_ma`i')   (_p_ma`i')    ("`st_ma`i''")
}

post `out_post' ("var_coef") ("omega")     (_b_omega)   (_se_omega)   (_p_omega)   ("`st_omega'")
post `out_post' ("var_coef") ("arch1")     (_b_arch1)   (_se_arch1)   (_p_arch1)   ("`st_arch1'")
post `out_post' ("var_coef") ("garch1")    (_b_garch1)  (_se_garch1)  (_p_garch1)  ("`st_garch1'")
foreach v of local het_controls {
    local st_het_`v' = cond(_p_het_`v' < 0.01, "***", cond(_p_het_`v' < 0.05, "**", cond(_p_het_`v' < 0.10, "*", "")))
    post `out_post' ("var_coef") ("het_`v'") (_b_het_`v') (_se_het_`v') (_p_het_`v') ("`st_het_`v''")
}
post `out_post' ("var_coef") ("persistence") (_persistence) (.) (.) ("")

postclose `out_post'

drop _std_res _std_res_sq _resid _ht

quietly {
    use `out_file', clear
    export delimited using "`out_dir'/garch_results_`stub'_joint`df_label'_northspill.csv", replace
}
display _n "Results saved to: `out_dir'/garch_results_`stub'_joint`df_label'_northspill.csv"
