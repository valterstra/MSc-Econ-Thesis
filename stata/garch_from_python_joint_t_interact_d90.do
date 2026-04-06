clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* garch_from_python_joint_t_interact_d90.do
*
* Joint ARMAX(1,0,1)-GARCH-X(1,1) with state-dependent wind elasticity.
* Supports Gaussian or Student-t errors via the GAUSSIAN flag.
*
* Extends garch_from_python_joint_t.do by adding:
*   dt       = 1 if price_ds > 90th percentile of price_ds (spike regime)
*   dt_wind  = dt * wind_log  (interaction term)
*
* Mean equation:
*   price_ds = a + b1*wind_log + b2*dt + b3*dt_wind + [other controls] + ARMA
*   b1           = wind elasticity in normal regime
*   b1 + b3      = wind elasticity in spike regime
*   b2           = level shift in spike regime
*
* Note: dt and dt_wind are NOT included in the variance equation (het()).
*
* Usage:
*   1. Run python/stata_inputs/full_period_export.py  ->  CSV files written to stata_input/
*   2. Set DATA_FILE, HET_ALL_CONTROLS, T_DF below
*   3. Run this do-file in Stata
*
* Output:
*   output from stata/garch_results/2025_results/
*       garch_results_{stub}_joint_gaussian_interact_d90.csv   (GAUSSIAN=1)
*       garch_results_{stub}_joint_tdf{T_DF}_interact_d90.csv  (GAUSSIAN=0)
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/armax_garch/input_SE1_2025-01-01_2025-12-31_log.csv"
* 0 = wind_log only in variance eq. (baseline)
* 1 = all exogenous controls in variance eq. (extended GARCH-X)
global HET_ALL_CONTROLS 1
* Distribution:
*   GAUSSIAN = 1  ->  Gaussian (normal) errors  [output suffix: _gaussian]
*   GAUSSIAN = 0  ->  Student-t errors, df set by T_DF below
global GAUSSIAN 0
* Degrees of freedom for Student-t errors (only used when GAUSSIAN=0).
* Set to a positive integer >= 3 to fix df. Set to 0 to estimate freely.
global T_DF 5
* Include interaction terms (dt, dt_wind) in the mean equation?
*   MEAN_INTERACT = 1  ->  dt and dt_wind enter mean equation  (default)
*   MEAN_INTERACT = 0  ->  no interactions; clean baseline  [no _interact_d90 suffix]
global MEAN_INTERACT 1
* Include standalone dt level-shift in mean equation?
*   INCLUDE_DT_LEVEL = 1  ->  both dt and dt_wind in mean (our spec, default)
*   INCLUDE_DT_LEVEL = 0  ->  only dt_wind in mean, no level shift (supervisor's spec)
*                            [output suffix: _nolevel]
*   Note: only relevant when MEAN_INTERACT = 1
global INCLUDE_DT_LEVEL 1
* Include dt_wind interaction term in the variance equation (het()) as well?
*   HET_INTERACT = 0  ->  dt_wind in mean equation only (baseline)
*   HET_INTERACT = 1  ->  dt_wind also enters het()  [output suffix: _hetinteract]
*   Note: HET_INTERACT is ignored when MEAN_INTERACT = 0
global HET_INTERACT 1
* Include bottom-10% low-price regime interaction?
*   INCLUDE_LOW_REGIME = 0  ->  spike regime only (baseline)
*   INCLUDE_LOW_REGIME = 1  ->  adds dl (price_ds < 10th pct) and dl_wind = dl*wind_log
*                              to mean equation alongside the spike terms
*                              [output suffix: _dualregime]
*   Note: only active when MEAN_INTERACT = 1
global INCLUDE_LOW_REGIME 0
* ARMA order
local arma_p = 1
local arma_d = 0
local arma_q = 1
* Spike threshold percentile
local spike_pct = 90
* -----------------------------------------------------------------------------

set maxiter 100

* Build distribution spec
if $GAUSSIAN {
    local dist_spec ""
    local df_label "_gaussian"
    local dist_display "Gaussian (normal)"
}
else if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
    local df_label "_tdf$T_DF"
    local dist_display "Student-t (df=$T_DF, fixed)"
}
else {
    local dist_spec "distribution(t)"
    local df_label "_tdfest"
    local dist_display "Student-t (df estimated)"
}


* Build interact suffix for output filename
if $MEAN_INTERACT & $INCLUDE_LOW_REGIME {
    local interact_suffix "_interact_d90_dualregime"
}
else if $MEAN_INTERACT {
    local interact_suffix "_interact_d90"
}
else {
    local interact_suffix ""
}

* Build level suffix (supervisor's spec omits dt level shift)
if $MEAN_INTERACT & !$INCLUDE_DT_LEVEL {
    local level_suffix "_nolevel"
}
else {
    local level_suffix ""
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


* ============================================================================
* 2. Generate spike dummy and interaction term (only when MEAN_INTERACT=1)
*    dt      = 1 if price_ds is above the spike_pct-th percentile
*    dt_wind = dt * wind_log
*
*    Both are created BEFORE the ds command so they are auto-detected as
*    controls and enter the mean equation automatically.
* ============================================================================
if $MEAN_INTERACT {
    quietly {
        summarize `depvar', detail
        scalar _p_threshold    = r(p`spike_pct')
        scalar _p_low_threshold = r(p10)
        gen byte   dt      = (`depvar' > _p_threshold)   if !missing(`depvar')
        gen double dt_wind = dt * wind_log
    }
    quietly count if dt == 1
    display _n "Spike threshold (`spike_pct'th pct of `depvar'): " %8.4f _p_threshold
    display    "Spike obs (dt=1): " r(N) " of " _N " (" %5.1f 100*r(N)/_N "%)"

    if $INCLUDE_LOW_REGIME {
        quietly {
            gen byte   dl      = (`depvar' < _p_low_threshold) if !missing(`depvar')
            gen double dl_wind = dl * wind_log
        }
        quietly count if dl == 1
        display    "Low threshold (10th pct of `depvar'):          " %8.4f _p_low_threshold
        display    "Low obs (dl=1):   " r(N) " of " _N " (" %5.1f 100*r(N)/_N "%)"
    }
}
else {
    display _n "MEAN_INTERACT=0: no spike dummy generated — clean baseline spec"
}


* ============================================================================
* 3. Build control lists
* ============================================================================
ds stata_clock `depvar', not
local controls `r(varlist)'

* het() controls: all controls EXCEPT the regime interaction terms
local spike_vars "dt dt_wind dl dl_wind"
local het_base : list controls - spike_vars

if $HET_ALL_CONTROLS {
    local het_controls `het_base'
    local het_label "all controls (excl. spike terms)"
    local het_suffix "_hetfull"
}
else {
    local het_controls wind_log
    local het_label "wind_log only"
    local het_suffix "_hetwind"
}

* Supervisor's spec: remove dt level-shift from mean equation controls
if $MEAN_INTERACT & !$INCLUDE_DT_LEVEL {
    local dt_only "dt"
    local controls : list controls - dt_only
}

* Optionally add regime interaction terms to variance equation (only possible when MEAN_INTERACT=1)
if $MEAN_INTERACT & $HET_INTERACT {
    local het_controls `het_controls' dt dt_wind
    if $INCLUDE_LOW_REGIME {
        local het_controls `het_controls' dl dl_wind
        local het_label "`het_label' + dt, dt_wind, dl, dl_wind"
    }
    else {
        local het_label "`het_label' + dt and dt_wind interaction"
    }
    local het_interact_suffix "_hetinteract"
}
else {
    local het_interact_suffix ""
}

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "Variance het(): `het_label'"
display    "Distribution: `dist_display'"


* ============================================================================
* 4. Estimate ARMAX(p,d,q)-GARCH-X(1,1) — joint MLE
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
* 5. Collect results
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

    * --- Implied elasticities (only when interaction terms present) ---
    if $MEAN_INTERACT {
        scalar _b_wind_normal = _b_wind_log
        scalar _b_wind_spike  = _b_wind_log + _b_dt_wind
        if $INCLUDE_LOW_REGIME {
            scalar _b_wind_low = _b_wind_log + _b_dl_wind
        }
    }
}


* ============================================================================
* 6. Significance stars
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
* 7. Print results table
* ============================================================================
display _n "{hline 70}"
display "  ARMA(`arma_p',`arma_q')-GARCH-X(1,1) — `dist_display'  [het: `het_label']"
if $MEAN_INTERACT {
    display "  State-dependent wind elasticity  (spike: `depvar' > `spike_pct'th pct)"
}
else {
    display "  No interaction terms (clean baseline)"
}
display "{hline 70}"
display "  " _col(28) "Coef."  _col(38) "Std.Err."  _col(51) "Sig."
display "{hline 70}"
display "  Mean equation"
display "{hline 70}"

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

if $MEAN_INTERACT {
    display "{hline 70}"
    display "  Implied wind elasticities"
    display "  Normal regime  (b_wind_log):            " %9.5f _b_wind_normal
    display "  Spike  regime  (b_wind_log + b_dt_wind):" %9.5f _b_wind_spike
    if $INCLUDE_LOW_REGIME {
        display "  Low    regime  (b_wind_log + b_dl_wind):" %9.5f _b_wind_low
    }
}
display "  Variance equation"
display "{hline 70}"
display "  omega (w)"     _col(28) %9.5f _b_omega   _col(38) %8.4f _se_omega   "  `st_omega'"
display "  ARCH(1) (a)"  _col(28) %9.5f _b_arch1   _col(38) %8.4f _se_arch1   "  `st_arch1'"
display "  GARCH(1) (d)" _col(28) %9.5f _b_garch1  _col(38) %8.4f _se_garch1  "  `st_garch1'"
foreach v of local het_controls {
    display "  `v' (g)"  _col(28) %9.5f _b_het_`v' _col(38) %8.4f _se_het_`v' "  `st_het_`v''"
}
display "{hline 70}"
display "  Persistence (a+d)"  _col(28) %9.5f _persistence
if _persistence >= 1 {
    display "  WARNING: persistence >= 1 — variance non-stationary"
}
display "{hline 70}"
display "  N"               _col(28) %12.0f e(N)
display "  Log-likelihood"  _col(28) %12.3f e(ll)
display "  AIC"             _col(28) %12.3f _aic
display "  BIC"             _col(28) %12.3f _bic
display "{hline 70}"
display "  LBQ(24) std. resid.             " %8.2f _Q_std    "  p=" %6.4f _p_Q_std
display "  LBQ(24) sq. std. resid.         " %8.2f _Q_sq     "  p=" %6.4f _p_Q_sq
display "{hline 70}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "  vce(robust) = Bollerslev-Wooldridge QML SEs"
display "{hline 70}"


* ============================================================================
* 8. Save results to CSV
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
local _rst_std = cond(_p_Q_std < 0.01, "***", cond(_p_Q_std < 0.05, "**", cond(_p_Q_std < 0.10, "*", "")))
local _rst_sq  = cond(_p_Q_sq  < 0.01, "***", cond(_p_Q_sq  < 0.05, "**", cond(_p_Q_sq  < 0.10, "*", "")))
post `out_post' ("lbq_std") ("L24") (_Q_std) (.) (_p_Q_std) ("`_rst_std'")
post `out_post' ("lbq_sq")  ("L24") (_Q_sq)  (.) (_p_Q_sq)  ("`_rst_sq'")

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

* Implied elasticities (only saved when interaction terms present)
if $MEAN_INTERACT {
    post `out_post' ("elasticity") ("wind_normal") (_b_wind_normal) (.) (.) ("")
    post `out_post' ("elasticity") ("wind_spike")  (_b_wind_spike)  (.) (.) ("")
    if $INCLUDE_LOW_REGIME {
        post `out_post' ("elasticity") ("wind_low") (_b_wind_low) (.) (.) ("")
    }
}

postclose `out_post'

drop _std_res _std_res_sq _resid _ht
capture drop dt dt_wind dl dl_wind

quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/garch_results/2025_results/garch_results_`stub'_joint`df_label'`interact_suffix'`level_suffix'`het_interact_suffix'.csv", replace
}
display _n "Results saved to: output from stata/garch_results/2025_results/garch_results_`stub'_joint`df_label'`interact_suffix'`level_suffix'`het_interact_suffix'.csv"
