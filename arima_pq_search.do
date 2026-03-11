clear all
set more off
set linesize 120

* ============================================================================
* arima_pq_search.do
*
* Grid search over ARMA(p,q) orders, p in {0,...,3}, q in {0,...,3}.
* d is fixed from the companion metadata file.
*
* Each converged model is estimated ONCE. Results are stored with
* estimates store and a single postfile collects everything in one pass.
*
* Output: one long-format CSV with rows of three types:
*   type="fit"     : variable in (N, ll, AIC, BIC)   value=stat, se=., pval=.
*   type="lbq_res" : variable=L{lag}                  value=stat, se=., pval=p
*   type="lbq_sq"  : variable=L{lag}                  value=stat, se=., pval=p
*   type="coef"    : variable=regressor name           value=coef, se=se, pval=p
*
* Tables printed to console:
*   Table 1 — Fit statistics
*   Table 2 — LB-Q statistics  on residuals
*   Table 3 — LB-Q p-values    on residuals
*   Table 4 — LB-Q statistics  on squared residuals
*   Table 5 — LB-Q p-values    on squared residuals
*   Table 6 — Coefficients per spec (via estimates restore, no re-estimation)
*
* Usage:
*   1. Run full_regression.py  →  CSV files written to output/
*   2. Set DATA_FILE below
*   3. Run this do-file in Stata
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "output/armax_input_SE2_2024-01-01_2025-12-31.csv"
local lb_lags "1 2 3 4 5 10 15 20"
local n_lags  8
* -----------------------------------------------------------------------------


* ============================================================================
* 1. Read d from companion metadata file
* ============================================================================
local meta_file = subinstr("$DATA_FILE", "armax_input_", "armax_meta_", 1)
import delimited using "`meta_file'", clear varnames(1)
local arma_d = arma_d[1]


* ============================================================================
* 2. Load data and set up time series
* ============================================================================
quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
    drop timestamp
}

local depvar price_ds
ds stata_clock `depvar', not
local controls `r(varlist)'

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "d fixed at: `arma_d'"
display    "Grid:     p in {0,...,3}  x  q in {0,...,3}" _n


* ============================================================================
* 3. Grid search — estimate once, store everything
*
* Matrix layout (for console tables):
*   1-2   : p, q
*   3-6   : N, ll, aic, bic
*   7-14  : LB stats   on residuals     at each lag
*   15-22 : LB p-values on residuals    at each lag
*   23-30 : LB stats   on sq residuals  at each lag
*   31-38 : LB p-values on sq residuals at each lag
*   39    : converged (1/0)
* ============================================================================
local n_specs = 16
matrix RES = J(`n_specs', 39, .)

* Open single combined postfile
local stub = subinstr("$DATA_FILE", "output/armax_input_", "", 1)
local stub = subinstr("`stub'", ".csv", "", 1)
capture mkdir "output/pq_search_results"

tempname out_post
tempfile out_file
postfile `out_post' p q str16 type str32 variable ///
    double (value se pval) using `out_file', replace

local row = 0
forval p = 0/3 {
    forval q = 0/3 {
        local row = `row' + 1
        matrix RES[`row', 1] = `p'
        matrix RES[`row', 2] = `q'

        * --- Estimate ---
        capture quietly arima `depvar' `controls', ///
            arima(`p', `arma_d', `q')             ///
            vce(robust)

        if _rc != 0 {
            display "  ARMA(`p',`q'): FAILED (rc=" _rc ")"
            matrix RES[`row', 39] = 0
            continue
        }

        matrix RES[`row', 39] = 1

        * Store estimates for Table 6 (no re-estimation needed later)
        estimates store arma_`p'_`q'

        * --- Fit statistics ---
        quietly estat ic
        matrix _ic = r(S)
        local _N   = e(N)
        local _ll  = e(ll)
        local _aic = _ic[1,5]
        local _bic = _ic[1,6]

        matrix RES[`row', 3] = `_N'
        matrix RES[`row', 4] = `_ll'
        matrix RES[`row', 5] = `_aic'
        matrix RES[`row', 6] = `_bic'

        post `out_post' (`p') (`q') ("fit") ("N")   (`_N')   (.) (.)
        post `out_post' (`p') (`q') ("fit") ("ll")  (`_ll')  (.) (.)
        post `out_post' (`p') (`q') ("fit") ("AIC") (`_aic') (.) (.)
        post `out_post' (`p') (`q') ("fit") ("BIC") (`_bic') (.) (.)

        * --- LB tests ---
        quietly predict double _resid_pq, residuals
        quietly gen double _resid_pq_sq = _resid_pq^2

        local col_rs = 7
        local col_rp = 15
        local col_ss = 23
        local col_sp = 31

        foreach lag of local lb_lags {
            quietly wntestq _resid_pq, lags(`lag')
            local _stat = r(stat)
            local _pval = r(p)
            matrix RES[`row', `col_rs'] = `_stat'
            matrix RES[`row', `col_rp'] = `_pval'
            post `out_post' (`p') (`q') ("lbq_res") ("L`lag'") (`_stat') (.) (`_pval')
            local col_rs = `col_rs' + 1
            local col_rp = `col_rp' + 1

            quietly wntestq _resid_pq_sq, lags(`lag')
            local _stat = r(stat)
            local _pval = r(p)
            matrix RES[`row', `col_ss'] = `_stat'
            matrix RES[`row', `col_sp'] = `_pval'
            post `out_post' (`p') (`q') ("lbq_sq") ("L`lag'") (`_stat') (.) (`_pval')
            local col_ss = `col_ss' + 1
            local col_sp = `col_sp' + 1
        }

        drop _resid_pq _resid_pq_sq

        * --- Coefficients ---
        local _bc  = _b[`depvar':_cons]
        local _sec = _se[`depvar':_cons]
        local _pc  = 2*(1 - normal(abs(`_bc' / `_sec')))
        post `out_post' (`p') (`q') ("coef") ("_cons") (`_bc') (`_sec') (`_pc')

        foreach v of local controls {
            local _bv  = _b[`depvar':`v']
            local _sev = _se[`depvar':`v']
            local _pv  = 2*(1 - normal(abs(`_bv' / `_sev')))
            post `out_post' (`p') (`q') ("coef") ("`v'") (`_bv') (`_sev') (`_pv')
        }

        forval i = 1/`p' {
            local _bar  = _b[ARMA:L`i'.ar]
            local _sear = _se[ARMA:L`i'.ar]
            local _par  = 2*(1 - normal(abs(`_bar' / `_sear')))
            post `out_post' (`p') (`q') ("coef") ("ar_L`i'") (`_bar') (`_sear') (`_par')
        }

        forval i = 1/`q' {
            local _bma  = _b[ARMA:L`i'.ma]
            local _sema = _se[ARMA:L`i'.ma]
            local _pma  = 2*(1 - normal(abs(`_bma' / `_sema')))
            post `out_post' (`p') (`q') ("coef") ("ma_L`i'") (`_bma') (`_sema') (`_pma')
        }

        display "  ARMA(`p',`q'): done  " ///
            "AIC=" %9.2f `_aic' "  BIC=" %9.2f `_bic'
    }
}

postclose `out_post'


* ============================================================================
* Helper: lag header for LB tables
* ============================================================================
local lag_header "  p  q"
foreach lag of local lb_lags {
    local lag_header "`lag_header'      L`lag'"
}


* ============================================================================
* 4. Table 1 — Fit statistics
* ============================================================================
display _n "{hline 60}"
display "  TABLE 1: Fit statistics  (d=`arma_d')"
display "{hline 60}"
display "  p  q" _col(9) "N" _col(18) "Log-lik" _col(31) "AIC" _col(44) "BIC"
display "{hline 60}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]  ///
        _col(9)  %7.0f  RES[`r',3]                        ///
        _col(18) %11.3f RES[`r',4]                        ///
        _col(31) %11.3f RES[`r',5]                        ///
        _col(44) %11.3f RES[`r',6]
}
display "{hline 60}"


* ============================================================================
* 5. Table 2 — LB-Q statistics on residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 2: LB-Q statistics on residuals"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 7/14 {
        display _continue "  " %7.2f RES[`r',`c']
    }
    display ""
}
display "{hline 100}"


* ============================================================================
* 6. Table 3 — LB-Q p-values on residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 3: LB-Q p-values on residuals  (* = p<0.05)"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 15/22 {
        local pval = RES[`r', `c']
        local star = cond(`pval' < 0.05, "*", " ")
        display _continue "  " %6.4f `pval' "`star'"
    }
    display ""
}
display "{hline 100}"


* ============================================================================
* 7. Table 4 — LB-Q statistics on squared residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 4: LB-Q statistics on squared residuals"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 23/30 {
        display _continue "  " %7.2f RES[`r',`c']
    }
    display ""
}
display "{hline 100}"


* ============================================================================
* 8. Table 5 — LB-Q p-values on squared residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 5: LB-Q p-values on squared residuals  (* = p<0.05 → ARCH)"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 31/38 {
        local pval = RES[`r', `c']
        local star = cond(`pval' < 0.05, "*", " ")
        display _continue "  " %6.4f `pval' "`star'"
    }
    display ""
}
display "{hline 100}"
display "  LB-Q on residuals:         p>0.05 = no remaining autocorrelation"
display "  LB-Q on squared residuals: p<0.05 = ARCH effects present"
display "{hline 100}"


* ============================================================================
* 9. Table 6 — Coefficients via estimates restore (no re-estimation)
* ============================================================================
local row = 0
forval p = 0/3 {
    forval q = 0/3 {
        local row = `row' + 1
        if RES[`row', 39] == 0 continue

        quietly estimates restore arma_`p'_`q'

        display _n "{hline 58}"
        display "  ARMA(`p',`arma_d',`q') — Coefficients (QML robust SEs)"
        display "{hline 58}"
        display "  " _col(30) "Coef." _col(40) "Std.Err." _col(52) "Sig."
        display "{hline 58}"

        local _bc  = _b[`depvar':_cons]
        local _sec = _se[`depvar':_cons]
        local _pc  = 2*(1 - normal(abs(`_bc' / `_sec')))
        local st = cond(`_pc'<0.01,"***",cond(`_pc'<0.05,"**",cond(`_pc'<0.10,"*","")))
        display "  Constant" _col(30) %9.5f `_bc' _col(40) %8.4f `_sec' "  `st'"

        foreach v of local controls {
            local _bv  = _b[`depvar':`v']
            local _sev = _se[`depvar':`v']
            local _pv  = 2*(1 - normal(abs(`_bv' / `_sev')))
            local st = cond(`_pv'<0.01,"***",cond(`_pv'<0.05,"**",cond(`_pv'<0.10,"*","")))
            display "  `v'" _col(30) %9.5f `_bv' _col(40) %8.4f `_sev' "  `st'"
        }

        forval i = 1/`p' {
            local _bar  = _b[ARMA:L`i'.ar]
            local _sear = _se[ARMA:L`i'.ar]
            local _par  = 2*(1 - normal(abs(`_bar' / `_sear')))
            local st = cond(`_par'<0.01,"***",cond(`_par'<0.05,"**",cond(`_par'<0.10,"*","")))
            display "  ar.L`i'" _col(30) %9.5f `_bar' _col(40) %8.4f `_sear' "  `st'"
        }

        forval i = 1/`q' {
            local _bma  = _b[ARMA:L`i'.ma]
            local _sema = _se[ARMA:L`i'.ma]
            local _pma  = 2*(1 - normal(abs(`_bma' / `_sema')))
            local st = cond(`_pma'<0.01,"***",cond(`_pma'<0.05,"**",cond(`_pma'<0.10,"*","")))
            display "  ma.L`i'" _col(30) %9.5f `_bma' _col(40) %8.4f `_sema' "  `st'"
        }

        display "{hline 58}"
        display "  *** p<0.01  ** p<0.05  * p<0.10"
        display "{hline 58}"
    }
}


* ============================================================================
* 10. Save single combined CSV
* ============================================================================
quietly {
    use `out_file', clear
    export delimited using ///
        "output/pq_search_results/arima_results_`stub'.csv", replace
}
display _n "Results saved to: output/pq_search_results/arima_results_`stub'.csv"
