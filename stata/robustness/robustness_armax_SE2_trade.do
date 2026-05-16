clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

/*============================================================================
  robustness_armax_SE2_trade.do

  SE2-specific robustness check: isolates the contribution of each
  individual exchange partner to the wind price effect.

  Background: in the main robustness check, removing all SE2 trade flows
  simultaneously caused the wind coefficient to roughly halve (-0.170 → -0.089).
  This script identifies WHICH partner drives that attenuation by dropping
  one at a time from the full model.

  Model: plain ARMAX(1,0,1) — no GARCH — Huber-White robust SEs.
  ARMA order fixed at (1,1).

  Specs (one partner removed at a time — all other controls unchanged):
    1  Full model          (NO3 + NO4 + SE1 + SE3)
    2  Drop netexch_no3    (NO4 + SE1 + SE3)
    3  Drop netexch_no4    (NO3 + SE1 + SE3)
    4  Drop netexch_se1    (NO3 + NO4 + SE3)
    5  Drop netexch_se3    (NO3 + NO4 + SE1)

  Output  →  output from stata/robustness/armax/robustness_armax_SE2_trade_2025.csv
============================================================================*/

global DATA_FILE "stata_input/armax/armax_input_SE2_2025-01-01_2025-12-31_log.csv"

capture mkdir "output from stata/robustness"
capture mkdir "output from stata/robustness/armax"

/*============================================================================
  1.  Load data and set up time series
============================================================================*/
quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
    drop timestamp
}

local depvar "price_ds"

ds stata_clock `depvar', not
local all_controls `r(varlist)'

display _n "Data loaded  :  " _N " obs"
display    "All controls :  `all_controls'"

/*============================================================================
  2.  Define specifications (one partner dropped at a time from full)
============================================================================*/
* Standard controls — stay in every spec
local std_c "wind_log hydro_log_ds consump_log_ds oil_log gas_log"

* Spec control lists
local c1 "`std_c' netexch_no3 netexch_no4 netexch_se1 netexch_se3"   // full
local c2 "`std_c' netexch_no4 netexch_se1 netexch_se3"               // drop NO3
local c3 "`std_c' netexch_no3 netexch_se1 netexch_se3"               // drop NO4
local c4 "`std_c' netexch_no3 netexch_no4 netexch_se3"               // drop SE1
local c5 "`std_c' netexch_no3 netexch_no4 netexch_se1"               // drop SE3

local n_specs = 5

local sid1 "full"
local sid2 "no_no3"
local sid3 "no_no4"
local sid4 "no_se1"
local sid5 "no_se3"

local slabel1 "Full model (all partners)"
local slabel2 "Drop NO3"
local slabel3 "Drop NO4"
local slabel4 "Drop SE1"
local slabel5 "Drop SE3"

/*============================================================================
  3.  Postfile for results
============================================================================*/
local outfile "output from stata/robustness/armax/robustness_armax_SE2_trade_2025.csv"

tempfile tmp
postfile rob_trade           ///
    double spec_num          ///
    str20  spec_id           ///
    str40  spec_label        ///
    str32  variable          ///
    double coef se pval      ///
    str5   stars             ///
    double obs ll aic bic    ///
    using `tmp', replace

/*============================================================================
  4.  Estimate each specification
============================================================================*/
forval s = 1/`n_specs' {

    local curr_c `c`s''

    display _n "── Spec `s' / `n_specs': `slabel`s'' " _dup(40) "─"
    display    "   Controls: `curr_c'"

    quietly arima `depvar' `curr_c', arima(1,0,1) vce(robust)

    quietly {
        estat ic
        matrix mat_ic  = r(S)
        scalar sc_aic  = mat_ic[1,5]
        scalar sc_bic  = mat_ic[1,6]
        scalar sc_ll   = e(ll)
        scalar sc_nobs = e(N)
    }

    * ── Constant ────────────────────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[`depvar':_cons]
        scalar sc_se   = _se[`depvar':_cons]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("_cons") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── Exogenous regressors ────────────────────────────────────────────
    foreach v of local curr_c {
        quietly {
            scalar sc_coef = _b[`depvar':`v']
            scalar sc_se   = _se[`depvar':`v']
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("`v'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    * ── AR(1) ───────────────────────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[ARMA:L1.ar]
        scalar sc_se   = _se[ARMA:L1.ar]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("ar_L1") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── MA(1) ───────────────────────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[ARMA:L1.ma]
        scalar sc_se   = _se[ARMA:L1.ma]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("ma_L1") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── Per-spec summary ────────────────────────────────────────────────
    display "   wind_log : b=" %8.5f _b[`depvar':wind_log] ///
            "  SE=" %7.5f _se[`depvar':wind_log]
    display "   AIC=" %10.2f sc_aic "  BIC=" %10.2f sc_bic "  N=" %6.0f sc_nobs
}

postclose rob_trade

/*============================================================================
  5.  Export CSV
============================================================================*/
use `tmp', clear
export delimited using "`outfile'", replace

display _n "Saved → `outfile'  (" _N " rows)"

/*============================================================================
  6.  Summary table
============================================================================*/
display _n "{hline 68}"
display    "  SE2 Trade Partner Analysis  —  ARMAX(1,0,1)  [no GARCH]"
display    "{hline 68}"
display    "  Spec  Label" _col(34) "b(wind_log)" _col(48) "SE" _col(58) "Stars"
display    "{hline 68}"

forval s = 1/`n_specs' {
    quietly {
        use `tmp', clear
        keep if spec_num == `s' & variable == "wind_log"
    }
    display "  `s'     `slabel`s''" ///
        _col(34) %8.5f coef[1] _col(48) %7.5f se[1] _col(58) stars[1]
}

display "{hline 68}"
display "vce(robust) = Huber-White sandwich SEs throughout"
display "{hline 68}"
