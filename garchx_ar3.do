clear all
set more off
set linesize 120

* ============================================================================
* ARMAX(3)-GARCH(1,1)-X  —  SE4, 2015-2016
*
* Mean eq.:  price_log_ds = c + β·wind_t + [AR(1,2,3)] + ε_t
* Var. eq.:  σ²_t = exp(λ₀ + λ_w·wind_t) + α·ε²_{t-1} + δ·σ²_{t-1}
*
* NOTE: Stata's het(varlist) implements Harvey (1976) multiplicative
* heteroskedasticity. This differs from the additive GARCH-X in the Python
* files; λ_w and γ are not directly comparable in magnitude.
* ============================================================================

* ============================================================================
* 1–2. Import and time-series setup (suppressed)
* ============================================================================
quietly {
    import delimited "output/arx_input_SE4_2018-01-01_2019-12-31.csv", ///
        varnames(1) case(lower) clear
    gen double stata_clock = clock(datetime, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
}

* ============================================================================
* 3. Estimation
* ============================================================================
quietly arch price_log_ds wind_forecast_log_ds,  ///
    ar(1 2 3)                                    ///
    arch(1) garch(1)                             ///
    het(wind_forecast_log_ds)                    ///
    distribution(gaussian)                       ///
    difficult

* ============================================================================
* 4–5. Collect all results quietly, then print one clean table
* ============================================================================
quietly {

    * AIC
    estat ic
    matrix _ic = r(S)
    scalar _aic = _ic[1,5]

    * Coefficients and standard errors
    scalar b_cons_m    = _b[price_log_ds:_cons]
    scalar se_cons_m   = _se[price_log_ds:_cons]
    scalar p_cons_m    = 2*(1-normal(abs(b_cons_m / se_cons_m)))

    scalar b_wind_m    = _b[price_log_ds:wind_forecast_log_ds]
    scalar se_wind_m   = _se[price_log_ds:wind_forecast_log_ds]
    scalar p_wind_m    = 2*(1-normal(abs(b_wind_m / se_wind_m)))

    scalar b_ar1       = _b[ARMA:L1.ar]
    scalar se_ar1      = _se[ARMA:L1.ar]
    scalar p_ar1       = 2*(1-normal(abs(b_ar1 / se_ar1)))

    scalar b_ar2       = _b[ARMA:L2.ar]
    scalar se_ar2      = _se[ARMA:L2.ar]
    scalar p_ar2       = 2*(1-normal(abs(b_ar2 / se_ar2)))

    scalar b_ar3       = _b[ARMA:L3.ar]
    scalar se_ar3      = _se[ARMA:L3.ar]
    scalar p_ar3       = 2*(1-normal(abs(b_ar3 / se_ar3)))

    scalar b_het_cons  = _b[HET:_cons]
    scalar se_het_cons = _se[HET:_cons]
    scalar p_het_cons  = 2*(1-normal(abs(b_het_cons / se_het_cons)))

    scalar b_het_wind  = _b[HET:wind_forecast_log_ds]
    scalar se_het_wind = _se[HET:wind_forecast_log_ds]
    scalar p_het_wind  = 2*(1-normal(abs(b_het_wind / se_het_wind)))

    scalar b_arch1     = _b[ARCH:L1.arch]
    scalar se_arch1    = _se[ARCH:L1.arch]
    scalar p_arch1     = 2*(1-normal(abs(b_arch1 / se_arch1)))

    scalar b_garch1    = _b[ARCH:L1.garch]
    scalar se_garch1   = _se[ARCH:L1.garch]
    scalar p_garch1    = 2*(1-normal(abs(b_garch1 / se_garch1)))

    * Diagnostics
    predict double resid, residuals
    predict double ht,    variance
    gen double std_res    = resid / sqrt(ht)
    gen double std_res_sq = std_res^2

    wntestq std_res, lags(24)
    scalar Q_std   = r(stat)
    scalar p_Q_std = r(p)

    wntestq std_res_sq, lags(24)
    scalar Q_sq   = r(stat)
    scalar p_Q_sq = r(p)
}

* --- Significance stars ---
local st_cons_m   = cond(p_cons_m   < 0.01, "***", cond(p_cons_m   < 0.05, "**", cond(p_cons_m   < 0.10, "*", "")))
local st_wind_m   = cond(p_wind_m   < 0.01, "***", cond(p_wind_m   < 0.05, "**", cond(p_wind_m   < 0.10, "*", "")))
local st_ar1      = cond(p_ar1      < 0.01, "***", cond(p_ar1      < 0.05, "**", cond(p_ar1      < 0.10, "*", "")))
local st_ar2      = cond(p_ar2      < 0.01, "***", cond(p_ar2      < 0.05, "**", cond(p_ar2      < 0.10, "*", "")))
local st_ar3      = cond(p_ar3      < 0.01, "***", cond(p_ar3      < 0.05, "**", cond(p_ar3      < 0.10, "*", "")))
local st_het_cons = cond(p_het_cons < 0.01, "***", cond(p_het_cons < 0.05, "**", cond(p_het_cons < 0.10, "*", "")))
local st_het_wind = cond(p_het_wind < 0.01, "***", cond(p_het_wind < 0.05, "**", cond(p_het_wind < 0.10, "*", "")))
local st_arch1    = cond(p_arch1    < 0.01, "***", cond(p_arch1    < 0.05, "**", cond(p_arch1    < 0.10, "*", "")))
local st_garch1   = cond(p_garch1   < 0.01, "***", cond(p_garch1   < 0.05, "**", cond(p_garch1   < 0.10, "*", "")))

* --- Print table ---
display _n "{hline 45}"
display "  ARMAX(3)-GARCH(1,1)-X   SE4, 2015-2016"
display "{hline 45}"
display "                              (A)"
display "{hline 45}"
display "  Mean equation"
display "{hline 45}"
display "  Constant        " %9.5f b_cons_m    "`st_cons_m'"
display "                  (" %6.4f se_cons_m  ")"
display "  Log wind        " %9.5f b_wind_m    "`st_wind_m'"
display "                  (" %6.4f se_wind_m  ")"
display "  a1              " %9.5f b_ar1        "`st_ar1'"
display "                  (" %6.4f se_ar1      ")"
display "  a2              " %9.5f b_ar2        "`st_ar2'"
display "                  (" %6.4f se_ar2      ")"
display "  a3              " %9.5f b_ar3        "`st_ar3'"
display "                  (" %6.4f se_ar3      ")"
display "{hline 45}"
display "  Variance equation"
display "{hline 45}"
display "  Constant        " %9.5f b_het_cons  "`st_het_cons'"
display "                  (" %6.4f se_het_cons ")"
display "  Log wind        " %9.5f b_het_wind  "`st_het_wind'"
display "                  (" %6.4f se_het_wind ")"
display "  ARCH(1)         " %9.5f b_arch1      "`st_arch1'"
display "                  (" %6.4f se_arch1    ")"
display "  GARCH(1)        " %9.5f b_garch1     "`st_garch1'"
display "                  (" %6.4f se_garch1   ")"
display "{hline 45}"
display "  N               " %12.0f e(N)
display "  Log likelihood  " %12.3f e(ll)
display "  AIC             " %12.3f _aic
display "  Persistence     " %9.5f b_arch1 + b_garch1
display "{hline 45}"
display "  LB-Q(24) std. resid.    " %8.2f Q_std  "  p=" %6.4f p_Q_std
display "  LB-Q(24) sq. std. resid." %8.2f Q_sq   "  p=" %6.4f p_Q_sq
display "{hline 45}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "{hline 45}"

* ============================================================================
* 6. Export to Excel
* ============================================================================
capture mkdir results

putexcel set "results/garchx_ar3_SE4_results.xlsx", replace sheet("ARMAX3-GARCH-X")

* --- Headers ---
putexcel A1 = "Parameter"
putexcel B1 = "Coefficient"
putexcel C1 = "Std. Error"
putexcel D1 = "p-value"
putexcel E1 = "Sig."

* --- Mean equation ---
putexcel A2 = "Mean equation"
putexcel A3 = "Constant"
putexcel B3 = (b_cons_m)
putexcel C3 = (se_cons_m)
putexcel D3 = (p_cons_m)
putexcel E3 = "`st_cons_m'"

putexcel A4 = "Log wind"
putexcel B4 = (b_wind_m)
putexcel C4 = (se_wind_m)
putexcel D4 = (p_wind_m)
putexcel E4 = "`st_wind_m'"

putexcel A5 = "a1 (AR(1))"
putexcel B5 = (b_ar1)
putexcel C5 = (se_ar1)
putexcel D5 = (p_ar1)
putexcel E5 = "`st_ar1'"

putexcel A6 = "a2 (AR(2))"
putexcel B6 = (b_ar2)
putexcel C6 = (se_ar2)
putexcel D6 = (p_ar2)
putexcel E6 = "`st_ar2'"

putexcel A7 = "a3 (AR(3))"
putexcel B7 = (b_ar3)
putexcel C7 = (se_ar3)
putexcel D7 = (p_ar3)
putexcel E7 = "`st_ar3'"

* --- Variance equation ---
putexcel A8 = "Variance equation"
putexcel A9 = "Constant"
putexcel B9 = (b_het_cons)
putexcel C9 = (se_het_cons)
putexcel D9 = (p_het_cons)
putexcel E9 = "`st_het_cons'"

putexcel A10 = "Log wind"
putexcel B10 = (b_het_wind)
putexcel C10 = (se_het_wind)
putexcel D10 = (p_het_wind)
putexcel E10 = "`st_het_wind'"

putexcel A11 = "ARCH(1)"
putexcel B11 = (b_arch1)
putexcel C11 = (se_arch1)
putexcel D11 = (p_arch1)
putexcel E11 = "`st_arch1'"

putexcel A12 = "GARCH(1)"
putexcel B12 = (b_garch1)
putexcel C12 = (se_garch1)
putexcel D12 = (p_garch1)
putexcel E12 = "`st_garch1'"

* --- Fit statistics ---
putexcel A13 = "N"
putexcel B13 = (e(N))

putexcel A14 = "Log likelihood"
putexcel B14 = (e(ll))

putexcel A15 = "AIC"
putexcel B15 = (_aic)

putexcel A16 = "Persistence (a+d)"
putexcel B16 = (b_arch1 + b_garch1)

* --- Diagnostics ---
putexcel A17 = "LB-Q(24) std. resid."
putexcel B17 = (Q_std)
putexcel C17 = "p ="
putexcel D17 = (p_Q_std)

putexcel A18 = "LB-Q(24) sq. std. resid."
putexcel B18 = (Q_sq)
putexcel C18 = "p ="
putexcel D18 = (p_Q_sq)

putexcel A19 = "Note: *** p<0.01  ** p<0.05  * p<0.10"

display _n "Results exported to results/garchx_ar3_SE4_results.xlsx"
