clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

/*============================================================================
  robustness_garch_SE2_trade.do

  Combined SE1 + SE2 trade-partner robustness check for the updated joint
  specification.

  For each zone, the script estimates Student-t ARMAX(3,0,1)-GARCH-X(1,1)
  models and drops one exchange partner at a time from the full zone-specific
  trade set.

  SE1 partners:
    netexch_fi, netexch_no4, netexch_se2

  SE2 partners:
    netexch_no3, netexch_no4, netexch_se1, netexch_se3

  Output -> output from stata/robustness/garch/robustness_garch_SE1_SE2_trade_2025.csv
============================================================================*/

global T_DF 5

local arma_p = 3
local arma_q = 1

if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
}
else {
    local dist_spec "distribution(t)"
}

capture mkdir "output from stata/robustness"
capture mkdir "output from stata/robustness/garch"

tempfile tmp
tempname out_post

postfile `out_post' ///
    str4   zone        ///
    double spec_num    ///
    str20  spec_id     ///
    str40  spec_label  ///
    str32  variable    ///
    double coef se pval ///
    str5   stars       ///
    double obs ll aic bic ///
    using `tmp', replace

/*============================================================================
  Main loop: SE1 first, then SE2
============================================================================*/
foreach zone in SE1 SE2 {

    if "`zone'" == "SE1" {
        local data_file "stata_input/armax_garch/input_SE1_2025-01-01_2025-12-31_log.csv"
        local std_c "wind_log hydro_log_ds consump_log_ds oil_log gas_log"
        local partner_vars "netexch_fi netexch_no4 netexch_se2"
        local zone_title "SE1"
    }
    else if "`zone'" == "SE2" {
        local data_file "stata_input/armax_garch/input_SE2_2025-01-01_2025-12-31_log.csv"
        local std_c "wind_log hydro_log_ds consump_log_ds oil_log gas_log"
        local partner_vars "netexch_no3 netexch_no4 netexch_se1 netexch_se3"
        local zone_title "SE2"
    }

    local full_c "`std_c' `partner_vars'"
    local n_partners : word count `partner_vars'
    local n_specs = `n_partners' + 1

    display _n "Zone: `zone_title'"
    display    "Controls: `full_c'"

    quietly {
        import delimited using "`data_file'", clear varnames(1) case(lower)
        gen double stata_clock = clock(timestamp, "YMDhms")
        format stata_clock %tc
        duplicates drop stata_clock, force
        tsset stata_clock, delta(3600000)
        drop timestamp
    }

    local depvar "price_ds"

    ds stata_clock `depvar', not
    local all_controls `r(varlist)'

    local spec_num = 1
    local curr_c "`full_c'"
    local curr_id "full"
    local curr_label "Full model"

    forval s = 1/`n_specs' {

        if `s' > 1 {
            local drop_var : word `=`s'-1' of `partner_vars'
            local curr_c : list full_c - drop_var
            local short = subinstr("`drop_var'", "netexch_", "", 1)
            local curr_id "no_`short'"
            local curr_label "Drop `drop_var'"
        }

        display _n "Spec `s' / `n_specs': `curr_label'"
        display    "   Controls: `curr_c'"

        capture arch `depvar' `curr_c', ///
            ar(1/`arma_p') ma(1/`arma_q') ///
            arch(1) garch(1) ///
            `dist_spec' ///
            het(`curr_c') ///
            vce(robust) ///
            difficult ///
            nrtolerance(1e-3)

        if _rc != 0 & _rc != 430 {
            display as error "Spec `s' in `zone_title' failed with rc=" _rc " - skipping"
            continue
        }
        if _rc == 430 {
            display as text "Spec `s' in `zone_title': NOTE convergence not achieved (r(430)) - estimates at last iteration"
        }

        quietly {
            estat ic
            matrix mat_ic  = r(S)
            scalar sc_aic  = mat_ic[1,5]
            scalar sc_bic  = mat_ic[1,6]
            scalar sc_ll   = e(ll)
            scalar sc_nobs = e(N)
        }

        quietly {
            scalar sc_coef = _b[`depvar':_cons]
            scalar sc_se   = _se[`depvar':_cons]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("_cons") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

        foreach v of local curr_c {
            quietly {
                scalar sc_coef = _b[`depvar':`v']
                scalar sc_se   = _se[`depvar':`v']
                scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
                local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
            }
            post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("`v'") ///
                (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
        }

        forval i = 1/`arma_p' {
            quietly {
                scalar sc_coef = _b[ARMA:L`i'.ar]
                scalar sc_se   = _se[ARMA:L`i'.ar]
                scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
                local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
            }
            post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("ar_L`i'") ///
                (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
        }

        forval i = 1/`arma_q' {
            quietly {
                scalar sc_coef = _b[ARMA:L`i'.ma]
                scalar sc_se   = _se[ARMA:L`i'.ma]
                scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
                local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
            }
            post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("ma_L`i'") ///
                (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
        }

        quietly {
            scalar sc_coef = _b[HET:_cons]
            scalar sc_se   = _se[HET:_cons]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("omega") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

        quietly {
            scalar sc_coef = _b[ARCH:L1.arch]
            scalar sc_se   = _se[ARCH:L1.arch]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("alpha") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

        quietly {
            scalar sc_coef = _b[ARCH:L1.garch]
            scalar sc_se   = _se[ARCH:L1.garch]
            scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
            local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
        }
        post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("delta") ///
            (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)

        foreach v of local curr_c {
            quietly {
                scalar sc_coef = _b[HET:`v']
                scalar sc_se   = _se[HET:`v']
                scalar sc_pval = 2*(1 - normal(abs(sc_coef / sc_se)))
                local  st      = cond(sc_pval<0.01,"***",cond(sc_pval<0.05,"**",cond(sc_pval<0.10,"*","")))
            }
            post `out_post' ("`zone_title'") (`s') ("`curr_id'") ("`curr_label'") ("het_`v'") ///
                (sc_coef) (sc_se) (sc_pval) ("`st'") (sc_nobs) (sc_ll) (sc_aic) (sc_bic)
        }

        display "   wind_log (mean) : b=" %8.5f _b[`depvar':wind_log] ///
                "  SE=" %7.5f _se[`depvar':wind_log]
        display "   wind_log (var)  : gamma=" %8.5f _b[HET:wind_log] ///
                "  SE=" %7.5f _se[HET:wind_log]
        display "   AIC=" %10.2f sc_aic "  BIC=" %10.2f sc_bic "  N=" %6.0f sc_nobs
    }
}

postclose `out_post'

/*============================================================================
  Export CSV
============================================================================*/
use `tmp', clear
local outfile "output from stata/robustness/garch/robustness_garch_SE1_SE2_trade_2025.csv"
export delimited using "`outfile'", replace

display _n "Saved -> `outfile'  (" _N " rows)"
