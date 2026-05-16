clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\valte\Documents\MSc-Econ-Thesis"

* --- Configuration -----------------------------------------------------------
* Set the zone to estimate. Change to SE1 / SE2 / SE3 / SE4 as needed.
global ACTIVE_ZONE "SE4"
* Rolling-window year range.
global START_YEAR 2015
global END_YEAR 2025
* 0 = wind_log only in variance eq. (baseline)
* 1 = all exogenous controls in variance eq. (extended GARCH-X)
global HET_ALL_CONTROLS 1
* Pipeline suffix in the CSV filenames exported by rolling_window_export.py.
global PIPELINE "log_floor001"
* Degrees of freedom for Student-t errors.
* Set to a positive integer >= 3 to fix df (e.g. 5, 6, 8).
* Set to 0 to estimate df freely from the data.
global T_DF 5
* -----------------------------------------------------------------------------


* ============================================================================
* rolling_window_garch_joint_t.do
*
* Joint ARMAX-GARCH-X rolling window estimator with Student-t errors.
* Analogous to rolling_window_garch.do but replaces distribution(gaussian)
* with distribution(t), with degrees of freedom fixed at T_DF.
*
* Unlike the two-step version (rolling_window_garch_2step.do), this estimates
* the mean and variance equations simultaneously via full MLE. The t-distribution
* is applied to the joint likelihood, so it affects both equations.
*
* Motivation: joint Gaussian GARCH inflates alpha in fat-tailed windows
* (e.g. SE1 2018, 2020, 2021, 2025), causing persistence >= 1. Imposing
* t(df=5) correctly models fat-tailed electricity price residuals and resolves
* non-stationarity while retaining the efficiency of joint estimation.
*
* Output format: identical to rolling_window_garch.do —
*   long CSV, one row per (zone, year, parameter).
*   Columns: zone, start_year, end_year, type, variable, value, se, pval,
*            stars, converged
*   type values: "fit", "lbq_std", "lbq_sq", "coef", "var_coef"
*   converged: 1 = clean, 2 = r(430) non-convergence, 0 = hard failure
*
* Output:
*   output from stata/rolling_window_results/joint_t/
*       rolling_garch_joint_t_{ZONE}_1yr_tdf{T_DF}.csv            when PIPELINE = log
*       rolling_garch_joint_t_{ZONE}_1yr_{PIPELINE}_tdf{T_DF}.csv otherwise
*
* Usage:
*   1. Run rolling_window_export.py  →  CSVs in stata_input/rolling_windows/
*   2. Set ACTIVE_ZONE / START_YEAR / END_YEAR / HET_ALL_CONTROLS / PIPELINE / T_DF above
*   3. Run this do-file in Stata
* ============================================================================

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

capture mkdir "output from stata/rolling_window_results"
capture mkdir "output from stata/rolling_window_results/joint_t"

local pipeline_suffix ""
if "$PIPELINE" != "log" {
    local pipeline_suffix "_$PIPELINE"
}

* Open long-format postfile
tempname out_post
tempfile  out_file

postfile `out_post'             ///
    str4  zone                  ///
    int   start_year            ///
    int   end_year              ///
    str16 type                  ///
    str32 variable              ///
    double value                ///
    double se                   ///
    double pval                 ///
    str4  stars                 ///
    int   converged             ///
    using `out_file', replace


* ============================================================================
* Main loop: one zone × 11 windows (1-year, 1-year step, 2015–2025)
* ============================================================================
display _n "Zone: $ACTIVE_ZONE  |  Windows: $START_YEAR to $END_YEAR" _n
display    "Joint estimation: arch with `dist_spec'" _n

foreach zone in $ACTIVE_ZONE {
    forval start_year = $START_YEAR/$END_YEAR {
        local end_year = `start_year'

        local start = "`start_year'-01-01"
        local end   = "`end_year'-12-31"
        local csv   = "stata_input/rolling_windows/armax_input_`zone'_`start'_`end'_${PIPELINE}.csv"

        display _n "Processing `zone' `start_year'–`end_year' ..."

        * Skip if file missing
        capture confirm file "`csv'"
        if _rc != 0 {
            display "  SKIPPED: file not found (`csv')"
            continue
        }

        * Load data
        quietly {
            import delimited using "`csv'", clear varnames(1) case(lower)
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
        }
        else {
            local het_controls wind_log
            local het_label "wind_log only"
        }

        display "  Controls: `controls'"
        display "  Variance het(): `het_label'"


        * ====================================================================
        * Estimate ARMAX(1,0,1)-GARCH-X(1,1) — joint MLE, t-distributed errors
        * ====================================================================
        capture arch `depvar' `controls',   ///
            ar(1/3) ma(1)                   ///
            arch(1) garch(1)                ///
            het(`het_controls')             ///
            `dist_spec'                     ///
            vce(robust)                     ///
            difficult                       ///
            nrtolerance(1e-3)

        local _converged = 1
        if _rc != 0 & _rc != 430 {
            display "  FAILED (rc=" _rc ")"
            post `out_post' ("`zone'") (`start_year') (`end_year') ///
                ("fit") ("N") (.) (.) (.) ("") (0)
            continue
        }
        if _rc == 430 {
            display "  NOTE: non-convergence (r(430)) — estimates at last iteration saved"
            local _converged = 2
        }


        * ====================================================================
        * Collect results
        * ====================================================================
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

            scalar _b_ar1  = _b[ARMA:L1.ar]
            scalar _se_ar1 = _se[ARMA:L1.ar]
            scalar _p_ar1  = 2*(1 - normal(abs(_b_ar1 / _se_ar1)))

            scalar _b_ar2  = _b[ARMA:L2.ar]
            scalar _se_ar2 = _se[ARMA:L2.ar]
            scalar _p_ar2  = 2*(1 - normal(abs(_b_ar2 / _se_ar2)))

            scalar _b_ar3  = _b[ARMA:L3.ar]
            scalar _se_ar3 = _se[ARMA:L3.ar]
            scalar _p_ar3  = 2*(1 - normal(abs(_b_ar3 / _se_ar3)))

            scalar _b_ma1  = _b[ARMA:L1.ma]
            scalar _se_ma1 = _se[ARMA:L1.ma]
            scalar _p_ma1  = 2*(1 - normal(abs(_b_ma1 / _se_ma1)))

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

            * --- LB diagnostics on standardized residuals ---
            predict double _resid, residuals
            predict double _ht,    variance
            gen double _std_res    = _resid / sqrt(_ht)
            gen double _std_res_sq = _std_res^2

            wntestq _std_res,    lags(24)
            scalar _Q_std   = r(stat)
            scalar _p_Q_std = r(p)

            wntestq _std_res_sq, lags(24)
            scalar _Q_sq   = r(stat)
            scalar _p_Q_sq = r(p)

            drop _resid _ht _std_res _std_res_sq
        }


        * ====================================================================
        * Significance stars
        * ====================================================================
        local st_cons   = cond(_p_cons   < 0.01, "***", cond(_p_cons   < 0.05, "**", cond(_p_cons   < 0.10, "*", "")))
        local st_ar1    = cond(_p_ar1    < 0.01, "***", cond(_p_ar1    < 0.05, "**", cond(_p_ar1    < 0.10, "*", "")))
        local st_ar2    = cond(_p_ar2    < 0.01, "***", cond(_p_ar2    < 0.05, "**", cond(_p_ar2    < 0.10, "*", "")))
        local st_ar3    = cond(_p_ar3    < 0.01, "***", cond(_p_ar3    < 0.05, "**", cond(_p_ar3    < 0.10, "*", "")))
        local st_ma1    = cond(_p_ma1    < 0.01, "***", cond(_p_ma1    < 0.05, "**", cond(_p_ma1    < 0.10, "*", "")))
        local st_omega  = cond(_p_omega  < 0.01, "***", cond(_p_omega  < 0.05, "**", cond(_p_omega  < 0.10, "*", "")))
        local st_arch1  = cond(_p_arch1  < 0.01, "***", cond(_p_arch1  < 0.05, "**", cond(_p_arch1  < 0.10, "*", "")))
        local st_garch1 = cond(_p_garch1 < 0.01, "***", cond(_p_garch1 < 0.05, "**", cond(_p_garch1 < 0.10, "*", "")))

        foreach v of local controls {
            local st_`v' = cond(_p_`v' < 0.01, "***", cond(_p_`v' < 0.05, "**", cond(_p_`v' < 0.10, "*", "")))
        }
        foreach v of local het_controls {
            local st_het_`v' = cond(_p_het_`v' < 0.01, "***", cond(_p_het_`v' < 0.05, "**", cond(_p_het_`v' < 0.10, "*", "")))
        }


        * ====================================================================
        * Post results (long format)
        * ====================================================================

        * Fit statistics
        post `out_post' ("`zone'") (`start_year') (`end_year') ("fit") ("N")   (e(N))  (.) (.) ("") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("fit") ("ll")  (e(ll)) (.) (.) ("") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("fit") ("AIC") (_aic)  (.) (.) ("") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("fit") ("BIC") (_bic)  (.) (.) ("") (`_converged')

        * LB diagnostics
        local _rst_std = cond(_p_Q_std < 0.01, "***", cond(_p_Q_std < 0.05, "**", cond(_p_Q_std < 0.10, "*", "")))
        local _rst_sq  = cond(_p_Q_sq  < 0.01, "***", cond(_p_Q_sq  < 0.05, "**", cond(_p_Q_sq  < 0.10, "*",  "")))
        post `out_post' ("`zone'") (`start_year') (`end_year') ("lbq_std") ("L24") (_Q_std) (.) (_p_Q_std) ("`_rst_std'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("lbq_sq")  ("L24") (_Q_sq)  (.) (_p_Q_sq)  ("`_rst_sq'")  (`_converged')

        * Mean equation coefficients
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("_cons") (_b_cons) (_se_cons) (_p_cons) ("`st_cons'") (`_converged')
        foreach v of local controls {
            post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("`v'") (_b_`v') (_se_`v') (_p_`v') ("`st_`v''") (`_converged')
        }
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L1") (_b_ar1) (_se_ar1) (_p_ar1) ("`st_ar1'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L2") (_b_ar2) (_se_ar2) (_p_ar2) ("`st_ar2'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L3") (_b_ar3) (_se_ar3) (_p_ar3) ("`st_ar3'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ma_L1") (_b_ma1) (_se_ma1) (_p_ma1) ("`st_ma1'") (`_converged')

        * Variance equation coefficients
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("omega")       (_b_omega)    (_se_omega)    (_p_omega)    ("`st_omega'")   (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("arch1")       (_b_arch1)    (_se_arch1)    (_p_arch1)    ("`st_arch1'")   (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("garch1")      (_b_garch1)   (_se_garch1)   (_p_garch1)   ("`st_garch1'")  (`_converged')
        foreach v of local het_controls {
            post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("het_`v'") (_b_het_`v') (_se_het_`v') (_p_het_`v') ("`st_het_`v''") (`_converged')
        }
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("persistence") (_persistence) (.) (.) ("") (`_converged')

        display "  N=" e(N) "  AIC=" %9.2f _aic ///
            "  γ_wind=" %7.4f _b_het_wind_log ///
            "  persist=" %6.4f _persistence
    }
}


* ============================================================================
* Save results
* ============================================================================
postclose `out_post'

quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/rolling_window_results/joint_t/rolling_garch_joint_t_${ACTIVE_ZONE}_1yr`pipeline_suffix'`df_label'.csv", replace
}

display _n "Results saved to: output from stata/rolling_window_results/joint_t/rolling_garch_joint_t_${ACTIVE_ZONE}_1yr`pipeline_suffix'`df_label'.csv"
