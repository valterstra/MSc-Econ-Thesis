clear all
set more off
set linesize 120

* ============================================================================
* ARMAX(1,2,3,23,24,25,47,48,49)-GARCH(1,1)-X  —  SE4, 2018-2019
*
* Mean eq.:  price_log_ds = c + β·wind_t + [AR(1,2,3,23,24,25,47,48,49)] + ε_t
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
    ar(1 2 3 23 24 25 47 48 49)                  ///
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

    scalar b_ar23      = _b[ARMA:L23.ar]
    scalar se_ar23     = _se[ARMA:L23.ar]
    scalar p_ar23      = 2*(1-normal(abs(b_ar23 / se_ar23)))

    scalar b_ar24      = _b[ARMA:L24.ar]
    scalar se_ar24     = _se[ARMA:L24.ar]
    scalar p_ar24      = 2*(1-normal(abs(b_ar24 / se_ar24)))

    scalar b_ar25      = _b[ARMA:L25.ar]
    scalar se_ar25     = _se[ARMA:L25.ar]
    scalar p_ar25      = 2*(1-normal(abs(b_ar25 / se_ar25)))

    scalar b_ar47      = _b[ARMA:L47.ar]
    scalar se_ar47     = _se[ARMA:L47.ar]
    scalar p_ar47      = 2*(1-normal(abs(b_ar47 / se_ar47)))

    scalar b_ar48      = _b[ARMA:L48.ar]
    scalar se_ar48     = _se[ARMA:L48.ar]
    scalar p_ar48      = 2*(1-normal(abs(b_ar48 / se_ar48)))

    scalar b_ar49      = _b[ARMA:L49.ar]
    scalar se_ar49     = _se[ARMA:L49.ar]
    scalar p_ar49      = 2*(1-normal(abs(b_ar49 / se_ar49)))

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
local st_ar23     = cond(p_ar23     < 0.01, "***", cond(p_ar23     < 0.05, "**", cond(p_ar23     < 0.10, "*", "")))
local st_ar24     = cond(p_ar24     < 0.01, "***", cond(p_ar24     < 0.05, "**", cond(p_ar24     < 0.10, "*", "")))
local st_ar25     = cond(p_ar25     < 0.01, "***", cond(p_ar25     < 0.05, "**", cond(p_ar25     < 0.10, "*", "")))
local st_ar47     = cond(p_ar47     < 0.01, "***", cond(p_ar47     < 0.05, "**", cond(p_ar47     < 0.10, "*", "")))
local st_ar48     = cond(p_ar48     < 0.01, "***", cond(p_ar48     < 0.05, "**", cond(p_ar48     < 0.10, "*", "")))
local st_ar49     = cond(p_ar49     < 0.01, "***", cond(p_ar49     < 0.05, "**", cond(p_ar49     < 0.10, "*", "")))
local st_het_cons = cond(p_het_cons < 0.01, "***", cond(p_het_cons < 0.05, "**", cond(p_het_cons < 0.10, "*", "")))
local st_het_wind = cond(p_het_wind < 0.01, "***", cond(p_het_wind < 0.05, "**", cond(p_het_wind < 0.10, "*", "")))
local st_arch1    = cond(p_arch1    < 0.01, "***", cond(p_arch1    < 0.05, "**", cond(p_arch1    < 0.10, "*", "")))
local st_garch1   = cond(p_garch1   < 0.01, "***", cond(p_garch1   < 0.05, "**", cond(p_garch1   < 0.10, "*", "")))

* --- Print table ---
display _n "{hline 50}"
display "  ARMAX(1,2,3,23,24,25,47,48,49)-GARCH(1,1)-X"
display "{hline 50}"
display "                              (A)"
display "{hline 50}"
display "  Mean equation"
display "{hline 50}"
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
display "  a23             " %9.5f b_ar23       "`st_ar23'"
display "                  (" %6.4f se_ar23     ")"
display "  a24             " %9.5f b_ar24       "`st_ar24'"
display "                  (" %6.4f se_ar24     ")"
display "  a25             " %9.5f b_ar25       "`st_ar25'"
display "                  (" %6.4f se_ar25     ")"
display "  a47             " %9.5f b_ar47       "`st_ar47'"
display "                  (" %6.4f se_ar47     ")"
display "  a48             " %9.5f b_ar48       "`st_ar48'"
display "                  (" %6.4f se_ar48     ")"
display "  a49             " %9.5f b_ar49       "`st_ar49'"
display "                  (" %6.4f se_ar49     ")"
display "{hline 50}"
display "  Variance equation"
display "{hline 50}"
display "  Constant        " %9.5f b_het_cons  "`st_het_cons'"
display "                  (" %6.4f se_het_cons ")"
display "  Log wind        " %9.5f b_het_wind  "`st_het_wind'"
display "                  (" %6.4f se_het_wind ")"
display "  ARCH(1)         " %9.5f b_arch1      "`st_arch1'"
display "                  (" %6.4f se_arch1    ")"
display "  GARCH(1)        " %9.5f b_garch1     "`st_garch1'"
display "                  (" %6.4f se_garch1   ")"
display "{hline 50}"
display "  N               " %12.0f e(N)
display "  Log likelihood  " %12.3f e(ll)
display "  AIC             " %12.3f _aic
display "  Persistence     " %9.5f b_arch1 + b_garch1
display "{hline 50}"
display "  LB-Q(24) std. resid.    " %8.2f Q_std  "  p=" %6.4f p_Q_std
display "  LB-Q(24) sq. std. resid." %8.2f Q_sq   "  p=" %6.4f p_Q_sq
display "{hline 50}"
display "  *** p<0.01  ** p<0.05  * p<0.10"
display "{hline 50}"

* ============================================================================
* 6. Export to Excel
* ============================================================================
capture mkdir results

putexcel set "results/garchx_ar3_seasonal2_SE4_results.xlsx", replace sheet("ARMAX-seasonal2-GARCH-X")

* --- Headers ---
putexcel A1 = "Parameter"
putexcel B1 = "Coefficient"
putexcel C1 = "Std. Error"
putexcel D1 = "p-value"
putexcel E1 = "Sig."

* --- Mean equation ---
putexcel A2  = "Mean equation"
putexcel A3  = "Constant"
putexcel B3  = (b_cons_m)
putexcel C3  = (se_cons_m)
putexcel D3  = (p_cons_m)
putexcel E3  = "`st_cons_m'"

putexcel A4  = "Log wind"
putexcel B4  = (b_wind_m)
putexcel C4  = (se_wind_m)
putexcel D4  = (p_wind_m)
putexcel E4  = "`st_wind_m'"

putexcel A5  = "a1 (AR(1))"
putexcel B5  = (b_ar1)
putexcel C5  = (se_ar1)
putexcel D5  = (p_ar1)
putexcel E5  = "`st_ar1'"

putexcel A6  = "a2 (AR(2))"
putexcel B6  = (b_ar2)
putexcel C6  = (se_ar2)
putexcel D6  = (p_ar2)
putexcel E6  = "`st_ar2'"

putexcel A7  = "a3 (AR(3))"
putexcel B7  = (b_ar3)
putexcel C7  = (se_ar3)
putexcel D7  = (p_ar3)
putexcel E7  = "`st_ar3'"

putexcel A8  = "a23 (AR(23))"
putexcel B8  = (b_ar23)
putexcel C8  = (se_ar23)
putexcel D8  = (p_ar23)
putexcel E8  = "`st_ar23'"

putexcel A9  = "a24 (AR(24))"
putexcel B9  = (b_ar24)
putexcel C9  = (se_ar24)
putexcel D9  = (p_ar24)
putexcel E9  = "`st_ar24'"

putexcel A10 = "a25 (AR(25))"
putexcel B10 = (b_ar25)
putexcel C10 = (se_ar25)
putexcel D10 = (p_ar25)
putexcel E10 = "`st_ar25'"

putexcel A11 = "a47 (AR(47))"
putexcel B11 = (b_ar47)
putexcel C11 = (se_ar47)
putexcel D11 = (p_ar47)
putexcel E11 = "`st_ar47'"

putexcel A12 = "a48 (AR(48))"
putexcel B12 = (b_ar48)
putexcel C12 = (se_ar48)
putexcel D12 = (p_ar48)
putexcel E12 = "`st_ar48'"

putexcel A13 = "a49 (AR(49))"
putexcel B13 = (b_ar49)
putexcel C13 = (se_ar49)
putexcel D13 = (p_ar49)
putexcel E13 = "`st_ar49'"

* --- Variance equation ---
putexcel A14 = "Variance equation"
putexcel A15 = "Constant"
putexcel B15 = (b_het_cons)
putexcel C15 = (se_het_cons)
putexcel D15 = (p_het_cons)
putexcel E15 = "`st_het_cons'"

putexcel A16 = "Log wind"
putexcel B16 = (b_het_wind)
putexcel C16 = (se_het_wind)
putexcel D16 = (p_het_wind)
putexcel E16 = "`st_het_wind'"

putexcel A17 = "ARCH(1)"
putexcel B17 = (b_arch1)
putexcel C17 = (se_arch1)
putexcel D17 = (p_arch1)
putexcel E17 = "`st_arch1'"

putexcel A18 = "GARCH(1)"
putexcel B18 = (b_garch1)
putexcel C18 = (se_garch1)
putexcel D18 = (p_garch1)
putexcel E18 = "`st_garch1'"

* --- Fit statistics ---
putexcel A19 = "N"
putexcel B19 = (e(N))

putexcel A20 = "Log likelihood"
putexcel B20 = (e(ll))

putexcel A21 = "AIC"
putexcel B21 = (_aic)

putexcel A22 = "Persistence (a+d)"
putexcel B22 = (b_arch1 + b_garch1)

* --- Diagnostics ---
putexcel A23 = "LB-Q(24) std. resid."
putexcel B23 = (Q_std)
putexcel C23 = "p ="
putexcel D23 = (p_Q_std)

putexcel A24 = "LB-Q(24) sq. std. resid."
putexcel B24 = (Q_sq)
putexcel C24 = "p ="
putexcel D24 = (p_Q_sq)

putexcel A25 = "Note: *** p<0.01  ** p<0.05  * p<0.10"

display _n "Results exported to results/garchx_ar3_seasonal2_SE4_results.xlsx"
