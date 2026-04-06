clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* rolling_window_qr.do
*
* Reads per-window CSVs exported by rolling_window_export.py and estimates
* QAR-X (Quantile Autoregression with Exogenous Variables) at five
* conditional quantiles for each zone × year window (2015–2025,
* non-overlapping 1-year windows).
*
* Specification (Koenker & Xiao 2006, JASA):
*   Q_τ(price_ds_t | Ω_{t-1}, X_t) = α_0(τ)
*                                    + Σ_j ρ_j(τ)·price_ds_{t-j}
*                                    + Σ_k β_k(τ)·x_kt
*
*   τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
*
*   The coefficient β_wind(τ) is the merit-order effect at quantile τ:
*   how much a 1-unit increase in log wind generation shifts the τ-th
*   conditional quantile of the deseasonalised log price, given recent
*   price history and all other fundamentals.
*
* AR lag order (AR_LAGS global):
*   Quantile regression cannot estimate MA parameters iteratively.
*   Default AR_LAGS = 2, matching the ARMA(1,1) companion GARCH spec:
*     - 1 lag from the AR(1) component
*     - 1 extra lag to approximate the dropped MA(1) term
*   Increase AR_LAGS if Ljung-Box diagnostics show residual autocorrelation.
*
* Standard errors: vce(robust) — Huber-White sandwich, robust to
*   heteroskedasticity. Serial correlation in the conditional mean is
*   mitigated by the included AR lags.
*
* Output format: long CSV — one row per (zone, year, quantile, parameter).
*   Columns: zone, start_year, end_year, quantile, type, variable,
*            value, se, pval, stars
*   type values:
*     "fit"  — N (observations after lag drop)
*     "lbq"  — Ljung-Box Q(24) on quantile residuals (serial corr. check)
*     "coef" — regression coefficients (controls + AR lags + constant)
*
* Output:
*   output from stata/rolling_window_results/qr/
*       rolling_qr_{ZONE}_1yr_{PIPELINE}.csv
*
* Usage:
*   1. Run rolling_window_export.py → CSVs in stata_input/rolling_windows/
*   2. Set ACTIVE_ZONE / PIPELINE / AR_LAGS below
*   3. Run this do-file in Stata
* ============================================================================


* --- Configuration -----------------------------------------------------------
* Zone to estimate. Change to SE1 / SE2 / SE3 / SE4 as needed.
global ACTIVE_ZONE "SE4"
* Pipeline suffix in the CSV filenames (must match rolling_window_export.py).
* "log" matches LOG_PRICE=True, USE_BILATERAL_EXCHANGE=False, no net exchange.
global PIPELINE "log"
* Number of AR lags to include as explicit regressors.
* Default 2 = AR(1) from ARMA spec + 1 extra to approximate MA(1).
global AR_LAGS 2
* -----------------------------------------------------------------------------


capture mkdir "output from stata/rolling_window_results"
capture mkdir "output from stata/rolling_window_results/qr"

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
    using `out_file', replace


* ============================================================================
* Main loop: zone × 11 windows (1-year, 1-year step, 2015–2025)
* ============================================================================
display _n "Zone: $ACTIVE_ZONE  |  Windows: 2015–2025  |  AR lags: $AR_LAGS" _n

foreach zone in $ACTIVE_ZONE {
    forval start_year = 2015/2025 {
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

        * Create AR lags as explicit regressors (QAR approach).
        * These replace the ARMA AR/MA structure from the GARCH spec.
        forval i = 1/$AR_LAGS {
            quietly gen double L`i'_price_ds = L`i'.`depvar'
        }
        quietly drop if missing(L${AR_LAGS}_price_ds)

        local ar_lags ""
        forval i = 1/$AR_LAGS {
            local ar_lags "`ar_lags' L`i'_price_ds"
        }

        display "  Controls: `controls'"
        display "  AR lags:  `ar_lags'"
        display "  N (after lag drop): " _N


        * ====================================================================
        * Quantile loop: estimate QR at each τ, collect results, post
        * ====================================================================
        foreach tau in 0.10 0.25 0.50 0.75 0.90 0.95 0.99 {

            * -----------------------------------------------------------------
            * Estimate
            * -----------------------------------------------------------------
            quietly qreg `depvar' `controls' `ar_lags', ///
                quantile(`tau') vce(robust)

            local N_obs = e(N)

            * -----------------------------------------------------------------
            * Collect coefficients
            * -----------------------------------------------------------------

            * Constant
            scalar _b_cons  = _b[_cons]
            scalar _se_cons = _se[_cons]
            scalar _p_cons  = 2*(1 - normal(abs(scalar(_b_cons) / scalar(_se_cons))))
            local st_cons = cond(scalar(_p_cons) < 0.01, "***", ///
                            cond(scalar(_p_cons) < 0.05, "**",  ///
                            cond(scalar(_p_cons) < 0.10, "*", "")))

            * Exogenous controls
            foreach v of local controls {
                scalar _b_`v'  = _b[`v']
                scalar _se_`v' = _se[`v']
                scalar _p_`v'  = 2*(1 - normal(abs(scalar(_b_`v') / scalar(_se_`v'))))
                local st_`v' = cond(scalar(_p_`v') < 0.01, "***", ///
                               cond(scalar(_p_`v') < 0.05, "**",  ///
                               cond(scalar(_p_`v') < 0.10, "*", "")))
            }

            * AR lags
            forval i = 1/$AR_LAGS {
                scalar _b_ar`i'  = _b[L`i'_price_ds]
                scalar _se_ar`i' = _se[L`i'_price_ds]
                scalar _p_ar`i'  = 2*(1 - normal(abs(scalar(_b_ar`i') / scalar(_se_ar`i'))))
                local st_ar`i' = cond(scalar(_p_ar`i') < 0.01, "***", ///
                                 cond(scalar(_p_ar`i') < 0.05, "**",  ///
                                 cond(scalar(_p_ar`i') < 0.10, "*", "")))
            }

            * -----------------------------------------------------------------
            * Ljung-Box Q(24) on quantile residuals
            * Tests for remaining serial correlation in the conditional quantile.
            * Failure (p < 0.05) suggests AR_LAGS may need to be increased.
            * -----------------------------------------------------------------
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

            * -----------------------------------------------------------------
            * Post results
            * -----------------------------------------------------------------

            * Fit: observation count
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("fit") ("N") (`N_obs') (.) (.) ("")

            * LBQ diagnostic
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("lbq") ("L24") ///
                (scalar(_Q_lbq)) (.) (scalar(_p_lbq)) ("`st_lbq'")

            * Constant
            post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                ("coef") ("_cons") ///
                (scalar(_b_cons)) (scalar(_se_cons)) (scalar(_p_cons)) ("`st_cons'")

            * Exogenous controls
            foreach v of local controls {
                post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                    ("coef") ("`v'") ///
                    (scalar(_b_`v')) (scalar(_se_`v')) (scalar(_p_`v')) ("`st_`v''")
            }

            * AR lags
            forval i = 1/$AR_LAGS {
                post `out_post' ("`zone'") (`start_year') (`end_year') (`tau') ///
                    ("coef") ("ar_L`i'") ///
                    (scalar(_b_ar`i')) (scalar(_se_ar`i')) (scalar(_p_ar`i')) ("`st_ar`i''")
            }

            * Summary line: wind coefficient at this quantile
            display "  τ=" %4.2f `tau' ///
                "  wind=" %9.5f scalar(_b_wind_log) " (`st_wind_log')" ///
                "  ar_L1=" %7.4f scalar(_b_ar1) ///
                "  LBQ-p=" %6.4f scalar(_p_lbq)

        }
        * end quantile loop

    }
    * end year loop
}
* end zone loop


* ============================================================================
* Save results
* ============================================================================
postclose `out_post'

quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/rolling_window_results/qr/rolling_qr_${ACTIVE_ZONE}_1yr_${PIPELINE}.csv", ///
        replace
}

display _n "Results saved to: output from stata/rolling_window_results/qr/rolling_qr_${ACTIVE_ZONE}_1yr_${PIPELINE}.csv"
display "*** p<0.01  ** p<0.05  * p<0.10"
display "SEs: vce(robust) — Huber-White heteroskedasticity-consistent"
