clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* -----------------------------------------------------------------------------
* rolling_window_qarx.do
*
* Rolling-window QARX estimator using the same per-window CSV inputs as the
* rolling joint t-GARCH workflow, but with a deliberately simpler dynamic
* quantile specification:
*
*   Q_tau(price_ds_t | Omega_{t-1}, X_t)
*       = a(tau) + sum_j phi_j(tau) * price_ds_{t-j} + beta(tau)' * X_t
*
* Design choice:
*   - Same rolling windows and exogenous controls as the baseline workflow
*   - AR lags included explicitly as regressors
*   - No MA term and no GARCH variance equation
*
* This is intended as a distributional complement to the baseline rolling
* joint t-GARCH results, not as a full quantile analogue of the baseline
* ARMAX-GARCH-X model.
*
* Output format: one long CSV row per (zone, year, quantile, parameter)
*   Columns: zone, start_year, end_year, quantile, type, variable,
*            value, se, pval, stars, converged
*
*   type values:
*     "fit"  - N, pseudo_r2
*     "lbq"  - Ljung-Box Q(24) on quantile residuals
*     "coef" - coefficients on controls, AR lags, and constant
*
* Output:
*   output from stata/rolling_window_results/qarx/{quantile_tag}_ar{AR_LAGS}/
*       rolling_qarx_{ZONE}_1yr_{PIPELINE}.csv
*
* Usage:
*   1. Run rolling_window_export.py -> CSVs in stata_input/rolling_windows/
*   2. Set ACTIVE_ZONE / PIPELINE / AR_LAGS / QUANTILES below
*   3. Run this do-file in Stata
* -----------------------------------------------------------------------------

* --- Configuration -----------------------------------------------------------
* One or more zones, e.g. "SE4" or "SE1 SE2 SE3 SE4"
global ACTIVE_ZONE "SE1"
* File suffix used by rolling_window_export.py
global PIPELINE "log"
* Number of explicit AR lags in the conditional quantile equation
global AR_LAGS 3
* Quantiles to estimate
global QUANTILES "0.10 0.50 0.90"
* Wind regressor name in the rolling-window CSVs
global WIND_VAR "wind_log"
* -----------------------------------------------------------------------------

local quantiles $QUANTILES
local wind_var "$WIND_VAR"
local quantile_tag ""
foreach tau of local quantiles {
    local tau_tag : subinstr local tau "." "", all
    if "`quantile_tag'" == "" local quantile_tag "q`tau_tag'"
    else local quantile_tag "`quantile_tag'_q`tau_tag'"
}
local zone_tag "$ACTIVE_ZONE"
local zone_tag : subinstr local zone_tag " " "_", all

local output_root "output from stata/rolling_window_results"
local output_dir  "`output_root'/qarx/`quantile_tag'_ar$AR_LAGS"

capture mkdir "`output_root'"
capture mkdir "`output_root'/qarx"
capture mkdir "`output_dir'"

* Open long-format postfile
tempname out_post
tempfile  out_file

postfile `out_post'           ///
    str4   zone               ///
    int    start_year         ///
    int    end_year           ///
    double quantile           ///
    str16  type               ///
    str32  variable           ///
    double value              ///
    double se                 ///
    double pval               ///
    str4   stars              ///
    int    converged          ///
    using `out_file', replace


* ============================================================================
* Main loop: one zone x 11 windows (1-year, 1-year step, 2015-2025)
* ============================================================================
display _n "Zones: $ACTIVE_ZONE  |  Windows: 2015 ... 2025  |  AR lags: $AR_LAGS"
display    "Quantiles: $QUANTILES" _n

foreach zone in $ACTIVE_ZONE {
    forval start_year = 2015/2025 {
        local end_year = `start_year'

        local start = "`start_year'-01-01"
        local end   = "`end_year'-12-31"
        local csv   = "stata_input/rolling_windows/armax_input_`zone'_`start'_`end'_${PIPELINE}.csv"

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
        capture confirm variable `wind_var'
        if _rc != 0 {
            display as error "Configured wind variable not found: `wind_var'"
            exit 111
        }
        ds stata_clock `depvar', not
        local controls `r(varlist)'

        * Explicit AR terms replace the ARMA structure in the baseline model.
        local ar_lags ""
        forval i = 1/$AR_LAGS {
            quietly gen double L`i'_price_ds = L`i'.`depvar'
            local ar_lags "`ar_lags' L`i'_price_ds"
        }
        quietly drop if missing(L${AR_LAGS}_price_ds)

        display "  Controls: `controls'"
        display "  AR lags:  `ar_lags'"
        display "  N (after lag drop): " _N


        * --------------------------------------------------------------------
        * Quantile loop
        * --------------------------------------------------------------------
        foreach tau of local quantiles {
            capture qreg `depvar' `controls' `ar_lags', quantile(`tau') vce(robust)

            local _converged = 1
            if _rc != 0 {
                display "  FAILED at tau=" %4.2f `tau' " (rc=" _rc ")"
                post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                    ("fit") ("N") (.) (.) (.) ("") (0)
                continue
            }

            local N_obs = e(N)

            capture scalar _pseudo_r2 = e(r2_p)
            if _rc != 0 {
                scalar _pseudo_r2 = .
            }

            * ----------------------------------------------------------------
            * Collect coefficients
            * ----------------------------------------------------------------
            scalar _b_cons  = _b[_cons]
            scalar _se_cons = _se[_cons]
            scalar _p_cons  = 2*(1 - normal(abs(_b_cons / _se_cons)))
            local st_cons = cond(_p_cons < 0.01, "***", ///
                            cond(_p_cons < 0.05, "**",  ///
                            cond(_p_cons < 0.10, "*", "")))

            foreach v of local controls {
                scalar _b_`v'  = _b[`v']
                scalar _se_`v' = _se[`v']
                scalar _p_`v'  = 2*(1 - normal(abs(_b_`v' / _se_`v')))
                local st_`v' = cond(_p_`v' < 0.01, "***", ///
                               cond(_p_`v' < 0.05, "**",  ///
                               cond(_p_`v' < 0.10, "*", "")))
            }

            forval i = 1/$AR_LAGS {
                scalar _b_ar`i'  = _b[L`i'_price_ds]
                scalar _se_ar`i' = _se[L`i'_price_ds]
                scalar _p_ar`i'  = 2*(1 - normal(abs(_b_ar`i' / _se_ar`i')))
                local st_ar`i' = cond(_p_ar`i' < 0.01, "***", ///
                                 cond(_p_ar`i' < 0.05, "**",  ///
                                 cond(_p_ar`i' < 0.10, "*", "")))
            }

            * ----------------------------------------------------------------
            * Ljung-Box on quantile residuals
            * ----------------------------------------------------------------
            quietly {
                predict double _qresid, residuals
                wntestq _qresid, lags(24)
                scalar _Q_lbq = r(stat)
                scalar _p_lbq = r(p)
                drop _qresid
            }
            local st_lbq = cond(_p_lbq < 0.01, "***", ///
                           cond(_p_lbq < 0.05, "**",  ///
                           cond(_p_lbq < 0.10, "*", "")))

            * ----------------------------------------------------------------
            * Post results
            * ----------------------------------------------------------------
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("fit") ("N") (`N_obs') (.) (.) ("") (`_converged')
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("fit") ("pseudo_r2") (_pseudo_r2) (.) (.) ("") (`_converged')
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("lbq") ("L24") (_Q_lbq) (.) (_p_lbq) ("`st_lbq'") (`_converged')

            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("coef") ("_cons") (_b_cons) (_se_cons) (_p_cons) ("`st_cons'") (`_converged')

            foreach v of local controls {
                post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                    ("coef") ("`v'") (_b_`v') (_se_`v') (_p_`v') ("`st_`v''") (`_converged')
            }

            forval i = 1/$AR_LAGS {
                post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                    ("coef") ("ar_L`i'") (_b_ar`i') (_se_ar`i') (_p_ar`i') ("`st_ar`i''") (`_converged')
            }

            display "  tau=" %4.2f `tau' ///
                "  `wind_var'=" %9.5f _b[`wind_var'] ///
                "  ar_L1=" %7.4f _b_ar1 ///
                "  LBQ-p=" %6.4f _p_lbq
        }
    }
}


* ============================================================================
* Save results
* ============================================================================
postclose `out_post'

quietly {
    use `out_file', clear
    export delimited using ///
        "`output_dir'/rolling_qarx_`zone_tag'_1yr_${PIPELINE}.csv", replace
}

display _n "Results saved to: `output_dir'/rolling_qarx_`zone_tag'_1yr_${PIPELINE}.csv"
