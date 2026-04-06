clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

/*============================================================================
  robustness_garch_SE2_trade.do

  SE2-specific robustness check: isolates the contribution of each
  individual exchange partner to the wind price effect.

  Background: in the main robustness check, removing all SE2 trade flows
  simultaneously caused the wind coefficient to roughly halve (-0.149 → -0.067).
  This script identifies WHICH partner drives that attenuation by dropping
  one at a time from the full model.

  Model: ARMAX(1,1)-GARCH-X(1,1), ARMA fixed at (1,1), QML robust SEs.
  Var. eq.: h_t = [ω + α·ε²_{t-1} + δ·h_{t-1}] · exp(Σ γ_k · x_kt)

  Specs (one partner removed at a time — all other controls unchanged):
    1  Full model          (NO3 + NO4 + SE1 + SE3)
    2  Drop netexch_no3    (NO4 + SE1 + SE3)
    3  Drop netexch_no4    (NO3 + SE1 + SE3)
    4  Drop netexch_se1    (NO3 + NO4 + SE3)
    5  Drop netexch_se3    (NO3 + NO4 + SE1)

  Output  →  output from stata/robustness/garch/robustness_garch_SE2_trade_2025.csv
============================================================================*/

global DATA_FILE "stata_input/armax/armax_input_SE2_2025-01-01_2025-12-31_log.csv"

capture mkdir "output from stata/robustness"
capture mkdir "output from stata/robustness/garch"

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

* All four exchange partners
local all_netexch "netexch_no3 netexch_no4 netexch_se1 netexch_se3"

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
local outfile "output from stata/robustness/garch/robustness_garch_SE2_trade_2025.csv"

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
set maxiter 100

forval s = 1/`n_specs' {

    local curr_c `c`s''

    display _n "── Spec `s' / `n_specs': `slabel`s'' " _dup(40) "─"
    display    "   Controls: `curr_c'"

    capture arch `depvar' `curr_c',  ///
        ar(1) ma(1)                  ///
        arch(1) garch(1)             ///
        het(`curr_c')                ///
        distribution(gaussian)       ///
        vce(robust)                  ///
        difficult                    ///
        nrtolerance(1e-3)

    if _rc != 0 & _rc != 430 {
        display as error "Spec `s': arch failed with rc=" _rc " — skipping"
        continue
    }
    if _rc == 430 {
        display as text "Spec `s': NOTE convergence not achieved (r(430)) — estimates at last iteration"
    }

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

    * ── Exogenous regressors (mean equation) ────────────────────────────
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

    * ── Variance equation: omega (HET:_cons), alpha, delta ──────────────
    quietly {
        scalar sc_coef = _b[HET:_cons]
        scalar sc_se   = _se[HET:_cons]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("omega") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    quietly {
        scalar sc_coef = _b[ARCH:L1.arch]
        scalar sc_se   = _se[ARCH:L1.arch]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("alpha") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    quietly {
        scalar sc_coef = _b[ARCH:L1.garch]
        scalar sc_se   = _se[ARCH:L1.garch]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("delta") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── GARCH-X: het() coefficients for all controls in this spec ───────
    foreach v of local curr_c {
        quietly {
            scalar sc_coef = _b[HET:`v']
            scalar sc_se   = _se[HET:`v']
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_trade (`s') ("`sid`s''") ("`slabel`s''") ("het_`v'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    * ── Per-spec summary ────────────────────────────────────────────────
    display "   wind_log (mean) : b=" %8.5f _b[`depvar':wind_log] ///
            "  SE=" %7.5f _se[`depvar':wind_log]
    display "   wind_log (var)  : γ=" %8.5f _b[HET:wind_log] ///
            "  SE=" %7.5f _se[HET:wind_log]
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
display _n "{hline 76}"
display    "  SE2 Trade Partner Analysis  —  ARMAX(1,1)-GARCH-X(1,1)"
display    "{hline 76}"
display    "  Spec  Label" _col(34) "Mean b(wind)" _col(48) "SE" _col(56) "Var γ(wind)" _col(70) "SE"
display    "{hline 76}"

forval s = 1/`n_specs' {
    quietly {
        use `tmp', clear
        keep if spec_num == `s' & (variable == "wind_log" | variable == "het_wind_log")
    }
    quietly {
        scalar sc_mean_b  = coef[1]
        scalar sc_mean_se = se[1]
        scalar sc_var_b   = coef[2]
        scalar sc_var_se  = se[2]
    }
    display "  `s'     `slabel`s''" ///
        _col(34) %8.5f sc_mean_b _col(48) %7.5f sc_mean_se ///
        _col(56) %8.5f sc_var_b  _col(70) %7.5f sc_var_se
}

display "{hline 76}"
display "vce(robust) = Bollerslev-Wooldridge QML SEs throughout"
display "{hline 76}"
