clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

/*============================================================================
  robustness_garch.do

  Robustness check: ARMAX(1,1)-GARCH-X(1,1) with progressive control removal.
  ARMA order is fixed at (1,1) across all specifications.

  Model:
    Mean eq. : price_ds = c + β·X_t + AR(1) + MA(1) + ε_t
    Var. eq. : h_t = [ω + α·ε²_{t-1} + δ·h_{t-1}] · exp(Σ γ_k · x_kt)
    (multiplicative GARCH-X via het(); same controls in both equations)

  Specs (accumulated removal — wind_log is never removed):
    1  Full model
    2  Drop gas_log
    3  Drop gas_log + oil_log
    4  Drop gas_log + oil_log + all netexch_* cols
    5  Spec 4  +  drop consump_log_ds
    6  Spec 5  +  drop hydro_log_ds   [wind_log + AR(1) + MA(1) only]

  Output  →  output from stata/robustness/garch/robustness_garch_<ZONE>_<PERIOD>.csv

  To switch zone / window: change the three globals below only.
============================================================================*/

* ── Configuration ──────────────────────────────────────────────────────────
global DATA_FILE "stata_input/armax/armax_input_SE1_2025-01-01_2025-12-31_log.csv"
global ZONE      "SE1"
global PERIOD    "2025"
* ───────────────────────────────────────────────────────────────────────────

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

local netexch_cols ""
foreach v of local all_controls {
    if substr("`v'", 1, 8) == "netexch_" {
        local netexch_cols `netexch_cols' `v'
    }
}

display _n "Data loaded  :  " _N " obs"
display    "All controls :  `all_controls'"
display    "NetExch cols :  " cond("`netexch_cols'"=="", "(none)", "`netexch_cols'")

/*============================================================================
  2.  Build control lists per specification (accumulated removal)
============================================================================*/
local rem_gas     "gas_log"
local rem_oil     "oil_log"
local rem_netexch "`netexch_cols'"
local rem_consump "consump_log_ds"
local rem_hydro   "hydro_log_ds"

local c1 `all_controls'
local c2 : list c1 - rem_gas
local c3 : list c2 - rem_oil
local c4 : list c3 - rem_netexch
local c5 : list c4 - rem_consump
local c6 : list c5 - rem_hydro

local n_specs = 6

local sid1 "full"
local sid2 "no_gas"
local sid3 "no_energy"
local sid4 "no_trade"
local sid5 "no_demand"
local sid6 "wind_arma"

local slabel1 "Full model"
local slabel2 "Drop gas"
local slabel3 "Drop energy prices"
local slabel4 "Drop trade flows"
local slabel5 "Drop consumption"
local slabel6 "Wind + ARMA(1,1) only"

/*============================================================================
  3.  Postfile for results
============================================================================*/
local outfile "output from stata/robustness/garch/robustness_garch_${ZONE}_${PERIOD}.csv"

tempfile tmp
postfile rob_garch           ///
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
    display    "   Controls (mean + variance): `curr_c'"

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

    * ── Constant (mean equation) ────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[`depvar':_cons]
        scalar sc_se   = _se[`depvar':_cons]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("_cons") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── Exogenous regressors (mean equation) ────────────────────────────
    foreach v of local curr_c {
        quietly {
            scalar sc_coef = _b[`depvar':`v']
            scalar sc_se   = _se[`depvar':`v']
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("`v'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    * ── AR(1) ───────────────────────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[ARMA:L1.ar]
        scalar sc_se   = _se[ARMA:L1.ar]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("ar_L1") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── MA(1) ───────────────────────────────────────────────────────────
    quietly {
        scalar sc_coef = _b[ARMA:L1.ma]
        scalar sc_se   = _se[ARMA:L1.ma]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("ma_L1") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── Variance equation: omega (HET:_cons), alpha, delta ──────────────
    * Note: with het(), omega lives at HET:_cons, not ARCH:_cons
    quietly {
        scalar sc_coef = _b[HET:_cons]
        scalar sc_se   = _se[HET:_cons]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("omega") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    quietly {
        scalar sc_coef = _b[ARCH:L1.arch]
        scalar sc_se   = _se[ARCH:L1.arch]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("alpha") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    quietly {
        scalar sc_coef = _b[ARCH:L1.garch]
        scalar sc_se   = _se[ARCH:L1.garch]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("delta") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    * ── GARCH-X: het() coefficients (variance equation, one per control) ─
    foreach v of local curr_c {
        quietly {
            scalar sc_coef = _b[HET:`v']
            scalar sc_se   = _se[HET:`v']
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_garch (`s') ("`sid`s''") ("`slabel`s''") ("het_`v'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    * ── Per-spec summary ────────────────────────────────────────────────
    display "   wind_log (mean) : b=" %8.5f _b[`depvar':wind_log] ///
            "  SE=" %7.5f _se[`depvar':wind_log]
    display "   wind_log (var)  : γ=" %8.5f _b[HET:wind_log] ///
            "  SE=" %7.5f _se[HET:wind_log]
    display "   AIC=" %10.2f sc_aic "  BIC=" %10.2f sc_bic "  N=" %6.0f sc_nobs
}

postclose rob_garch

/*============================================================================
  5.  Export CSV
============================================================================*/
use `tmp', clear
export delimited using "`outfile'", replace

display _n "Saved → `outfile'  (" _N " rows)"

/*============================================================================
  6.  Wind-coefficient stability summary table
============================================================================*/
display _n "{hline 72}"
display    "  Wind coefficient stability  —  ARMAX(1,1)-GARCH-X(1,1)"
display    "  Zone: ${ZONE}   Period: ${PERIOD}"
display    "{hline 72}"
display    "  #   Specification" _col(32) "Mean eq." _col(44) "SE" _col(52) "Var eq.(γ)" _col(64) "SE"
display    "{hline 72}"

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
    display "  `s'   `slabel`s''" ///
        _col(32) %8.5f sc_mean_b _col(44) %7.5f sc_mean_se ///
        _col(52) %8.5f sc_var_b  _col(64) %7.5f sc_var_se
}

display "{hline 72}"
display "vce(robust) = Bollerslev-Wooldridge QML SEs throughout"
display "{hline 72}"
