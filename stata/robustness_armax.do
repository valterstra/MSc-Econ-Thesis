clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

/*============================================================================
  robustness_armax.do

  Student-t ARMAX(3,0,1) robustness check with progressive control removal.
  No GARCH/X variance dynamics are estimated here. The mean equation matches
  the rolling-window joint specification: AR(3) + MA(1).

  Specs (accumulated removal - wind_log is never removed):
    1  Full model
    2  Drop gas_log
    3  Drop gas_log + oil_log
    4  Drop gas_log + oil_log + all netexch_* cols
    5  Spec 4 + drop consump_log_ds
    6  Spec 5 + drop hydro_log_ds   [wind_log + AR(3) + MA(1) only]

  Output -> output from stata/robustness/armax/robustness_armax_<ZONE>_<PERIOD>.csv
============================================================================*/

* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/armax_garch/input_SE4_2025-01-01_2025-12-31_log.csv"
global ZONE      "SE4"
global PERIOD    "2025"
global T_DF      5

local arma_p = 3
local arma_d = 0
local arma_q = 1

if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
}
else {
    local dist_spec "distribution(t)"
}

capture mkdir "output from stata/robustness"
capture mkdir "output from stata/robustness/armax"

/*============================================================================
  1. Load data and set up time series
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
  2. Build control lists per specification (accumulated removal)
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
local slabel6 "Wind + ARMA(3,1) only"

/*============================================================================
  3. Postfile for results
============================================================================*/
local outfile "output from stata/robustness/armax/robustness_armax_${ZONE}_${PERIOD}.csv"

tempfile tmp
postfile rob_armax           ///
    double spec_num          ///
    str20  spec_id           ///
    str40  spec_label        ///
    str32  variable          ///
    double coef se pval      ///
    str5   stars             ///
    double obs ll aic bic    ///
    using `tmp', replace

/*============================================================================
  4. Estimate each specification
============================================================================*/
forval s = 1/`n_specs' {
    local curr_c `c`s''

    display _n "Spec `s' / `n_specs': `slabel`s''"
    display    "   Controls: `curr_c'"

    quietly arch `depvar' `curr_c', ///
        ar(1/`arma_p') ma(1/`arma_q') ///
        `dist_spec' ///
        vce(robust) ///
        difficult ///
        nrtolerance(1e-3)

    if _rc != 0 & _rc != 430 {
        display as error "Spec `s': arch failed with rc=" _rc " - skipping"
        continue
    }
    if _rc == 430 {
        display as text "Spec `s': NOTE convergence not achieved (r(430)) - estimates at last iteration"
    }

    quietly {
        estat ic
        matrix mat_ic   = r(S)
        scalar sc_aic   = mat_ic[1,5]
        scalar sc_bic   = mat_ic[1,6]
        scalar sc_ll    = e(ll)
        scalar sc_nobs  = e(N)
    }

    quietly {
        scalar sc_coef = _b[`depvar':_cons]
        scalar sc_se   = _se[`depvar':_cons]
        scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
        local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
    }
    post rob_armax (`s') ("`sid`s''") ("`slabel`s''") ("_cons") ///
        (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

    foreach v of local curr_c {
        quietly {
            scalar sc_coef = _b[`depvar':`v']
            scalar sc_se   = _se[`depvar':`v']
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_armax (`s') ("`sid`s''") ("`slabel`s''") ("`v'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    forval i = 1/`arma_p' {
        quietly {
            scalar sc_coef = _b[ARMA:L`i'.ar]
            scalar sc_se   = _se[ARMA:L`i'.ar]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_armax (`s') ("`sid`s''") ("`slabel`s''") ("ar_L`i'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    forval i = 1/`arma_q' {
        quietly {
            scalar sc_coef = _b[ARMA:L`i'.ma]
            scalar sc_se   = _se[ARMA:L`i'.ma]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post rob_armax (`s') ("`sid`s''") ("`slabel`s''") ("ma_L`i'") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
    }

    display "   wind_log : b=" %8.5f _b[`depvar':wind_log] ///
            "  SE=" %7.5f _se[`depvar':wind_log]
    display "   AIC=" %10.2f sc_aic "  BIC=" %10.2f sc_bic "  N=" %6.0f sc_nobs
}

postclose rob_armax

/*============================================================================
  5. Export CSV
============================================================================*/
use `tmp', clear
export delimited using "`outfile'", replace

display _n "Saved -> `outfile'  (" _N " rows)"

/*============================================================================
  6. Wind-coefficient stability summary table
============================================================================*/
display _n "{hline 68}"
display    "  Wind coefficient stability - Student-t ARMAX(3,0,1) [no GARCH/X variance]"
display    "  Zone: ${ZONE}   Period: ${PERIOD}"
display    "{hline 68}"
display    "  #   Specification" _col(36) "b(wind_log)" _col(50) "SE" _col(60) "Stars"
display    "{hline 68}"

forval s = 1/`n_specs' {
    quietly {
        use `tmp', clear
        keep if variable == "wind_log" & spec_num == `s'
    }
    display "  `s'   `slabel`s''" _col(36) %9.5f coef[1] _col(50) %7.5f se[1] _col(60) stars[1]
}

display "{hline 68}"
display "vce(robust) = Huber-White sandwich SEs throughout"
display "{hline 68}"
