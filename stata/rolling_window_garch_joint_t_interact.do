clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* --- Configuration -----------------------------------------------------------
* Set the zone to estimate. Change to SE1 / SE2 / SE3 / SE4 as needed.
global ACTIVE_ZONE "SE4"
* Degrees of freedom for Student-t errors.
* Set to a positive integer >= 3 to fix df (e.g. 5, 6, 8).
* Set to 0 to estimate df freely from the data.
global T_DF 5
* Tail percentile used for both spike and low regimes.
* Set to 10 for top/bottom 10% or 20 for top/bottom 20%.
global EXTREME_TAIL_PCT 30
* -----------------------------------------------------------------------------


* ============================================================================
* rolling_window_garch_joint_t_interact.do
*
* Joint ARMAX(3,0,1)-GARCH-X(1,1) rolling window estimator with Student-t
* errors and state-dependent wind elasticity (spike regime interaction).
*
* Extends rolling_window_garch_joint_t.do by adding, within each window:
*   dt      = 1 if price_ds > the top tail cutoff of price_ds (spike regime)
*   dt_wind = dt * wind_log  (interaction term)
*
* Mean equation (our spec — includes dt level shift):
*   price_ds = a + b1*wind_log + b2*dt + b3*dt_wind + [other controls] + ARMA(3,1)
*   b1           = wind elasticity in normal regime
*   b1 + b3      = wind elasticity in spike regime
*   b2           = level shift in spike regime
*
* Variance equation:
*   het() = all exogenous controls incl. dt, dt_wind, dl, dl_wind
*
* Spike threshold is re-estimated per window, so it captures
* "relatively high prices in that year" rather than a fixed absolute level.
*
* Output format: identical to rolling_window_garch_joint_t.do plus two
*   additional "elasticity" rows per window:
*   Columns: zone, start_year, end_year, type, variable, value, se, pval,
*            stars, converged
*   type values: "fit", "lbq_std", "lbq_sq", "coef", "var_coef", "elasticity"
*   converged: 1 = clean, 2 = r(430) non-convergence, 0 = hard failure
*
* Output:
*   output from stata/rolling_window_results/joint_t_interact/tail_{PCT}pct/
*       rolling_garch_joint_t_interact_{ZONE}_1yr_tdf{T_DF}.csv
*
* Usage:
*   1. Run rolling_window_export.py  ->  CSVs in stata_input/rolling_windows/
*   2. Set ACTIVE_ZONE / T_DF / EXTREME_TAIL_PCT above
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

local tail_pct = $EXTREME_TAIL_PCT
local spike_pct = 100 - `tail_pct'
local low_pct   = `tail_pct'

local output_root "output from stata/rolling_window_results"
local output_dir  "`output_root'/joint_t_interact/tail_`tail_pct'pct"

capture mkdir "`output_root'"
capture mkdir "`output_root'/joint_t_interact"
capture mkdir "`output_dir'"

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
* Main loop: one zone x 11 windows (1-year, 1-year step, 2015-2025)
* ============================================================================
display _n "Zone: $ACTIVE_ZONE  |  Windows: 2015 ... 2025  (11 windows)" _n
display    "Joint estimation: arch with `dist_spec'" _n
display    "State-dependent wind elasticity (spike/low tails: `tail_pct'% per window)" _n

foreach zone in $ACTIVE_ZONE {
    forval start_year = 2015/2025 {
        local end_year = `start_year'

        local start = "`start_year'-01-01"
        local end   = "`end_year'-12-31"
        local csv   = "stata_input/rolling_windows/armax_input_`zone'_`start'_`end'_log.csv"

        display _n "Processing `zone' `start_year'-`end_year' ..."

        * Skip if file missing
        capture confirm file "`csv'"
        if _rc != 0 {
            display "  SKIPPED: file not found (`csv')"
            continue
        }

        * --------------------------------------------------------------------
        * Load data
        * --------------------------------------------------------------------
        quietly {
            import delimited using "`csv'", clear varnames(1) case(lower)
            gen double stata_clock = clock(timestamp, "YMDhms")
            format stata_clock %tc
            duplicates drop stata_clock, force
            tsset stata_clock, delta(3600000)
            drop timestamp
        }

        local depvar price_ds

        * --------------------------------------------------------------------
        * Generate spike dummy and interaction term (window-specific threshold)
        * dt      = 1 if price_ds above the top tail cutoff of this window
        * dt_wind = dt * wind_log
        * --------------------------------------------------------------------
        quietly {
            centile `depvar', centile(`low_pct' `spike_pct')
            scalar _p_low_threshold = r(c_1)
            scalar _p_threshold     = r(c_2)
            gen byte   dt      = (`depvar' > _p_threshold)     if !missing(`depvar')
            gen double dt_wind = dt * wind_log
            gen byte   dl      = (`depvar' < _p_low_threshold) if !missing(`depvar')
            gen double dl_wind = dl * wind_log
        }
        quietly count if dt == 1
        display "  Spike threshold (top `tail_pct'%):   " %8.4f _p_threshold ///
            "  Spike obs: " r(N) " (" %4.1f 100*r(N)/_N "%)"
        quietly count if dl == 1
        display "  Low threshold (bottom `tail_pct'%): " %8.4f _p_low_threshold ///
            "  Low obs:   " r(N) " (" %4.1f 100*r(N)/_N "%)"

        * --------------------------------------------------------------------
        * Build control lists
        * dt, dt_wind, dl, dl_wind are now in the dataset, so ds picks them up
        * --------------------------------------------------------------------
        ds stata_clock `depvar', not
        local controls `r(varlist)'

        * Variance equation uses all controls including dt, dt_wind, dl, dl_wind (symmetric with mean)
        local het_controls `controls'
        local het_label "all controls incl. dt, dt_wind, dl, dl_wind"

        display "  Controls: `controls'"
        display "  Variance het(): `het_label'"


        * --------------------------------------------------------------------
        * Estimate ARMAX(1,0,1)-GARCH-X(1,1) — joint MLE, t-distributed errors
        * --------------------------------------------------------------------
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
            drop dt dt_wind dl dl_wind
            post `out_post' ("`zone'") (`start_year') (`end_year') ///
                ("fit") ("N") (.) (.) (.) ("") (0)
            continue
        }
        if _rc == 430 {
            display "  NOTE: non-convergence (r(430)) — estimates at last iteration saved"
            local _converged = 2
        }


        * --------------------------------------------------------------------
        * Collect results
        * --------------------------------------------------------------------
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

            * --- Implied elasticities (point estimates) ---
            scalar _b_wind_normal = _b_wind_log
            scalar _b_wind_spike  = _b_wind_log + _b_dt_wind
            scalar _b_wind_low    = _b_wind_log + _b_dl_wind

            * --- SE and p-values for implied elasticities via lincom ---
            * wind_normal: SE taken directly from wind_log
            scalar _se_wind_normal = _se_wind_log
            scalar _p_wind_normal  = _p_wind_log
            * wind_spike = wind_log + dt_wind: exact SE using full VCE
            quietly lincom [`depvar']wind_log + [`depvar']dt_wind
            scalar _se_wind_spike = r(se)
            scalar _p_wind_spike  = r(p)
            * wind_low = wind_log + dl_wind: exact SE using full VCE
            quietly lincom [`depvar']wind_log + [`depvar']dl_wind
            scalar _se_wind_low   = r(se)
            scalar _p_wind_low    = r(p)

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

            * --- Implied variance elasticities via lincom ---
            * het_wind_normal: taken directly from het_wind_log
            scalar _b_het_wind_normal  = _b_het_wind_log
            scalar _se_het_wind_normal = _se_het_wind_log
            scalar _p_het_wind_normal  = _p_het_wind_log
            * het_wind_spike = het_wind_log + het_dt_wind
            quietly lincom [HET]wind_log + [HET]dt_wind
            scalar _b_het_wind_spike  = r(estimate)
            scalar _se_het_wind_spike = r(se)
            scalar _p_het_wind_spike  = r(p)
            * het_wind_low = het_wind_log + het_dl_wind
            quietly lincom [HET]wind_log + [HET]dl_wind
            scalar _b_het_wind_low  = r(estimate)
            scalar _se_het_wind_low = r(se)
            scalar _p_het_wind_low  = r(p)

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

            drop _resid _ht _std_res _std_res_sq dt dt_wind dl dl_wind
        }


        * --------------------------------------------------------------------
        * Significance stars
        * --------------------------------------------------------------------
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
        local st_wind_normal = cond(_p_wind_normal < 0.01, "***", cond(_p_wind_normal < 0.05, "**", cond(_p_wind_normal < 0.10, "*", "")))
        local st_wind_spike  = cond(_p_wind_spike  < 0.01, "***", cond(_p_wind_spike  < 0.05, "**", cond(_p_wind_spike  < 0.10, "*", "")))
        local st_wind_low    = cond(_p_wind_low    < 0.01, "***", cond(_p_wind_low    < 0.05, "**", cond(_p_wind_low    < 0.10, "*", "")))
        local st_het_wind_normal = cond(_p_het_wind_normal < 0.01, "***", cond(_p_het_wind_normal < 0.05, "**", cond(_p_het_wind_normal < 0.10, "*", "")))
        local st_het_wind_spike  = cond(_p_het_wind_spike  < 0.01, "***", cond(_p_het_wind_spike  < 0.05, "**", cond(_p_het_wind_spike  < 0.10, "*", "")))
        local st_het_wind_low    = cond(_p_het_wind_low    < 0.01, "***", cond(_p_het_wind_low    < 0.05, "**", cond(_p_het_wind_low    < 0.10, "*", "")))


        * --------------------------------------------------------------------
        * Post results (long format)
        * --------------------------------------------------------------------

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

        * Mean equation coefficients (dt and dt_wind included via controls loop)
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("_cons") (_b_cons) (_se_cons) (_p_cons) ("`st_cons'") (`_converged')
        foreach v of local controls {
            post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("`v'") (_b_`v') (_se_`v') (_p_`v') ("`st_`v''") (`_converged')
        }
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L1") (_b_ar1) (_se_ar1) (_p_ar1) ("`st_ar1'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L2") (_b_ar2) (_se_ar2) (_p_ar2) ("`st_ar2'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ar_L3") (_b_ar3) (_se_ar3) (_p_ar3) ("`st_ar3'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("coef") ("ma_L1") (_b_ma1) (_se_ma1) (_p_ma1) ("`st_ma1'") (`_converged')

        * Implied wind elasticities (with lincom SEs and p-values)
        post `out_post' ("`zone'") (`start_year') (`end_year') ("elasticity") ("wind_normal") (_b_wind_normal) (_se_wind_normal) (_p_wind_normal) ("`st_wind_normal'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("elasticity") ("wind_spike")  (_b_wind_spike)  (_se_wind_spike)  (_p_wind_spike)  ("`st_wind_spike'")  (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("elasticity") ("wind_low")    (_b_wind_low)    (_se_wind_low)    (_p_wind_low)    ("`st_wind_low'")    (`_converged')

        * Variance equation coefficients
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("omega")       (_b_omega)    (_se_omega)    (_p_omega)    ("`st_omega'")   (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("arch1")       (_b_arch1)    (_se_arch1)    (_p_arch1)    ("`st_arch1'")   (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("garch1")      (_b_garch1)   (_se_garch1)   (_p_garch1)   ("`st_garch1'")  (`_converged')
        foreach v of local het_controls {
            post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("het_`v'") (_b_het_`v') (_se_het_`v') (_p_het_`v') ("`st_het_`v''") (`_converged')
        }
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_coef") ("persistence") (_persistence) (.) (.) ("") (`_converged')

        * Implied variance elasticities (with lincom SEs and p-values)
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_elasticity") ("wind_normal") (_b_het_wind_normal) (_se_het_wind_normal) (_p_het_wind_normal) ("`st_het_wind_normal'") (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_elasticity") ("wind_spike")  (_b_het_wind_spike)  (_se_het_wind_spike)  (_p_het_wind_spike)  ("`st_het_wind_spike'")  (`_converged')
        post `out_post' ("`zone'") (`start_year') (`end_year') ("var_elasticity") ("wind_low")    (_b_het_wind_low)    (_se_het_wind_low)    (_p_het_wind_low)    ("`st_het_wind_low'")    (`_converged')

        display "  N=" e(N) "  AIC=" %9.2f _aic ///
            "  wind_normal=" %7.4f _b_wind_normal ///
            "  wind_spike=" %7.4f _b_wind_spike ///
            "  wind_low=" %7.4f _b_wind_low ///
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
        "`output_dir'/rolling_garch_joint_t_interact_${ACTIVE_ZONE}_1yr`df_label'.csv", replace
}

display _n "Results saved to: `output_dir'/rolling_garch_joint_t_interact_${ACTIVE_ZONE}_1yr`df_label'.csv"
