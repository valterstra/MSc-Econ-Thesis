clear all
set more off
set linesize 120

* Set working directory to project root
cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* garch_pq_search.do
*
* Grid search over ARMA(p,q) orders, p in {0,...,5}, q in {0,...,5},
* each estimated as ARMAX(p,q)-GARCH-X(1,1) with het(wind_log).
* d is fixed from the companion metadata file.
*
* Each converged model is estimated ONCE. Results are stored with
* estimates store and a single postfile collects everything in one pass.
*
* Output: one long-format CSV with rows of four types:
*   type="fit"      : variable in (N, ll, AIC, BIC)
*   type="lbq_std"  : LB-Q on standardized residuals     at each lag
*   type="lbq_sq"   : LB-Q on squared standardized resid at each lag
*   type="coef"     : mean equation coefficients
*   type="var_coef" : variance equation coefficients
*
* Tables printed to console:
*   Table 1 — Fit statistics
*   Table 2 — LB-Q statistics  on std. residuals
*   Table 3 — LB-Q p-values    on std. residuals
*   Table 4 — LB-Q statistics  on squared std. residuals
*   Table 5 — LB-Q p-values    on squared std. residuals
*   Table 6 — Coefficients per spec (via estimates restore, no re-estimation)
*
* Usage:
*   1. Run python/stata_inputs/full_period_export.py  →  CSV files written to output/
*   2. Set DATA_FILE below
*   3. Run this do-file in Stata
* ============================================================================

* --- Configuration -----------------------------------------------------------
global DATA_FILE "stata_input/armax_garch/input_SE4_2025-01-01_2025-12-31_log.csv"
* 0 = wind_log only in variance eq. (baseline)
* 1 = all exogenous controls in variance eq. (extended GARCH-X)
global HET_ALL_CONTROLS 1
local lb_lags "1 2 3 4 5 10 15 20"
local n_lags  8
* -----------------------------------------------------------------------------


* ============================================================================
* 1. Read d from companion metadata file
* ============================================================================
local meta_file = subinstr("$DATA_FILE", "input_", "meta_", 1)
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

* Build het() variable list
if $HET_ALL_CONTROLS {
    local het_controls `controls'
    local het_label "all controls"
    local het_suffix "_hetfull"
}
else {
    local het_controls wind_log
    local het_label "wind_log only"
    local het_suffix "_hetwind"
}

display _n "Sample:   " %tc stata_clock[1] " to " %tc stata_clock[_N]
display    "Controls: `controls'"
display    "Variance het(): `het_label'"
display    "d fixed at: `arma_d'"
display    "Grid:     p in {0,...,5}  x  q in {0,...,5}" _n


* ============================================================================
* 3. Grid search — estimate once, store everything
*
* Matrix layout (for console tables):
*   1-2   : p, q
*   3-6   : N, ll, aic, bic
*   7-14  : LB stats   on std. residuals     at each lag
*   15-22 : LB p-values on std. residuals    at each lag
*   23-30 : LB stats   on sq std. residuals  at each lag
*   31-38 : LB p-values on sq std. residuals at each lag
*   39    : converged (1/0)
* ============================================================================
local n_specs = 36
matrix RES = J(`n_specs', 39, .)

* Open single combined postfile
local stub = subinstr("$DATA_FILE", "stata_input/armax_garch/input_", "", 1)
local zone = substr("`stub'", 1, strpos("`stub'", "_") - 1)
capture mkdir "output from stata/pq_results_tdf5"

tempname out_post
tempfile out_file
postfile `out_post' p q str16 type str32 variable ///
    double (value se pval) using `out_file', replace

set maxiter 100

local row = 0
forval p = 0/5 {
    forval q = 0/5 {
        local row = `row' + 1
        matrix RES[`row', 1] = `p'
        matrix RES[`row', 2] = `q'

        * --- Build ar/ma spec (arch requires explicit options, not arima()) ---
        local ar_spec ""
        local ma_spec ""
        if `p' > 0  local ar_spec "ar(1/`p')"
        if `q' > 0  local ma_spec "ma(1/`q')"

        * --- Estimate ---
        capture quietly arch `depvar' `controls',   ///
            `ar_spec' `ma_spec'                     ///
            arch(1) garch(1)                        ///
            het(`het_controls')                     ///
            distribution(t 5)                       ///
            vce(robust)                             ///
            difficult                               ///
            nrtolerance(1e-3)

        if _rc != 0 & _rc != 430 {
            display "  ARMA(`p',`q'): FAILED (rc=" _rc ")"
            matrix RES[`row', 39] = 0
            continue
        }

        * 1 = clean convergence; 2 = r(430) non-convergence, estimates at last iter
        if _rc == 430 {
            display "  ARMA(`p',`q'): non-convergence (r(430)) — estimates at last iteration saved"
            matrix RES[`row', 39] = 2
        }
        else {
            matrix RES[`row', 39] = 1
        }

        * Store estimates for Table 6 (no re-estimation needed later)
        estimates store garch_`p'_`q'

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

        * --- LB tests on standardized residuals ---
        quietly predict double _resid_pq,  residuals
        quietly predict double _ht_pq,     variance
        quietly gen double _std_res_pq    = _resid_pq / sqrt(_ht_pq)
        quietly gen double _std_res_pq_sq = _std_res_pq^2

        local col_rs = 7
        local col_rp = 15
        local col_ss = 23
        local col_sp = 31

        foreach lag of local lb_lags {
            quietly wntestq _std_res_pq, lags(`lag')
            local _stat = r(stat)
            local _pval = r(p)
            matrix RES[`row', `col_rs'] = `_stat'
            matrix RES[`row', `col_rp'] = `_pval'
            post `out_post' (`p') (`q') ("lbq_std") ("L`lag'") (`_stat') (.) (`_pval')
            local col_rs = `col_rs' + 1
            local col_rp = `col_rp' + 1

            quietly wntestq _std_res_pq_sq, lags(`lag')
            local _stat = r(stat)
            local _pval = r(p)
            matrix RES[`row', `col_ss'] = `_stat'
            matrix RES[`row', `col_sp'] = `_pval'
            post `out_post' (`p') (`q') ("lbq_sq") ("L`lag'") (`_stat') (.) (`_pval')
            local col_ss = `col_ss' + 1
            local col_sp = `col_sp' + 1
        }

        drop _resid_pq _ht_pq _std_res_pq _std_res_pq_sq

        * --- Mean equation coefficients ---
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

        * --- Variance equation coefficients ---
        local _bomega  = _b[HET:_cons]
        local _seomega = _se[HET:_cons]
        local _pomega  = 2*(1 - normal(abs(`_bomega' / `_seomega')))
        post `out_post' (`p') (`q') ("var_coef") ("omega") (`_bomega') (`_seomega') (`_pomega')

        local _ba1  = _b[ARCH:L1.arch]
        local _sea1 = _se[ARCH:L1.arch]
        local _pa1  = 2*(1 - normal(abs(`_ba1' / `_sea1')))
        post `out_post' (`p') (`q') ("var_coef") ("arch1") (`_ba1') (`_sea1') (`_pa1')

        local _bg1  = _b[ARCH:L1.garch]
        local _seg1 = _se[ARCH:L1.garch]
        local _pg1  = 2*(1 - normal(abs(`_bg1' / `_seg1')))
        post `out_post' (`p') (`q') ("var_coef") ("garch1") (`_bg1') (`_seg1') (`_pg1')

        foreach v of local het_controls {
            local _bhw  = _b[HET:`v']
            local _sehw = _se[HET:`v']
            local _phw  = 2*(1 - normal(abs(`_bhw' / `_sehw')))
            post `out_post' (`p') (`q') ("var_coef") ("het_`v'") (`_bhw') (`_sehw') (`_phw')
        }

        local _persist = `_ba1' + `_bg1'
        post `out_post' (`p') (`q') ("var_coef") ("persistence") (`_persist') (.) (.)

        display "  ARMA(`p',`q')-GARCH-X: done  " ///
            "AIC=" %9.2f `_aic' "  BIC=" %9.2f `_bic' ///
            "  persistence=" %6.4f `_persist'
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
display "  TABLE 1: Fit statistics  (GARCH-X, d=`arma_d')"
display "{hline 60}"
display "  p  q" _col(9) "N" _col(18) "Log-lik" _col(31) "AIC" _col(44) "BIC"
display "{hline 60}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    local _nc_flag = cond(RES[`r', 39] == 2, " (nc)", "")
    display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]  ///
        _col(9)  %7.0f  RES[`r',3]                        ///
        _col(18) %11.3f RES[`r',4]                        ///
        _col(31) %11.3f RES[`r',5]                        ///
        _col(44) %11.3f RES[`r',6] "`_nc_flag'"
}
display "{hline 60}"


* ============================================================================
* 5. Table 2 — LB-Q statistics on std. residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 2: LB-Q statistics on standardized residuals"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    local _nc_flag = cond(RES[`r', 39] == 2, " (nc)", "")
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 7/14 {
        display _continue "  " %7.2f RES[`r',`c']
    }
    display "`_nc_flag'"
}
display "{hline 100}"


* ============================================================================
* 6. Table 3 — LB-Q p-values on std. residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 3: LB-Q p-values on standardized residuals  (* = p<0.05)"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    local _nc_flag = cond(RES[`r', 39] == 2, " (nc)", "")
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 15/22 {
        local pval = RES[`r', `c']
        local star = cond(`pval' < 0.05, "*", " ")
        display _continue "  " %6.4f `pval' "`star'"
    }
    display "`_nc_flag'"
}
display "{hline 100}"


* ============================================================================
* 7. Table 4 — LB-Q statistics on squared std. residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 4: LB-Q statistics on squared standardized residuals"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    local _nc_flag = cond(RES[`r', 39] == 2, " (nc)", "")
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 23/30 {
        display _continue "  " %7.2f RES[`r',`c']
    }
    display "`_nc_flag'"
}
display "{hline 100}"


* ============================================================================
* 8. Table 5 — LB-Q p-values on squared std. residuals
* ============================================================================
display _n "{hline 100}"
display "  TABLE 5: LB-Q p-values on sq. std. residuals  (* = p<0.05 → remaining ARCH)"
display "{hline 100}"
display "`lag_header'"
display "{hline 100}"
forval r = 1/`n_specs' {
    if RES[`r', 39] == 0 {
        display "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2] "  FAILED"
        continue
    }
    local _nc_flag = cond(RES[`r', 39] == 2, " (nc)", "")
    display _continue "  " %1.0f RES[`r',1] "  " %1.0f RES[`r',2]
    forval c = 31/38 {
        local pval = RES[`r', `c']
        local star = cond(`pval' < 0.05, "*", " ")
        display _continue "  " %6.4f `pval' "`star'"
    }
    display "`_nc_flag'"
}
display "{hline 100}"
display "  LB-Q on std. residuals:         p>0.05 = no remaining autocorrelation"
display "  LB-Q on sq. std. residuals:     p<0.05 = remaining ARCH effects"
display "{hline 100}"


* ============================================================================
* 9. Table 6 — Coefficients via estimates restore (no re-estimation)
* ============================================================================
local row = 0
forval p = 0/5 {
    forval q = 0/5 {
        local row = `row' + 1
        if RES[`row', 39] == 0 continue

        quietly estimates restore garch_`p'_`q'

        local _nc_hdr = cond(RES[`row', 39] == 2, " [non-convergence]", "")
        display _n "{hline 60}"
        display "  ARMA(`p',`arma_d',`q')-GARCH-X(1,1) — Coefficients (QML robust SEs)`_nc_hdr'"
        display "{hline 60}"
        display "  " _col(30) "Coef." _col(40) "Std.Err." _col(52) "Sig."
        display "{hline 60}"
        display "  Mean equation"
        display "{hline 60}"

        * Constant
        local _bc  = _b[`depvar':_cons]
        local _sec = _se[`depvar':_cons]
        local _pc  = 2*(1 - normal(abs(`_bc' / `_sec')))
        local st = cond(`_pc'<0.01,"***",cond(`_pc'<0.05,"**",cond(`_pc'<0.10,"*","")))
        display "  Constant" _col(30) %9.5f `_bc' _col(40) %8.4f `_sec' "  `st'"

        * Controls
        foreach v of local controls {
            local _bv  = _b[`depvar':`v']
            local _sev = _se[`depvar':`v']
            local _pv  = 2*(1 - normal(abs(`_bv' / `_sev')))
            local st = cond(`_pv'<0.01,"***",cond(`_pv'<0.05,"**",cond(`_pv'<0.10,"*","")))
            display "  `v'" _col(30) %9.5f `_bv' _col(40) %8.4f `_sev' "  `st'"
        }

        * AR lags
        forval i = 1/`p' {
            local _bar  = _b[ARMA:L`i'.ar]
            local _sear = _se[ARMA:L`i'.ar]
            local _par  = 2*(1 - normal(abs(`_bar' / `_sear')))
            local st = cond(`_par'<0.01,"***",cond(`_par'<0.05,"**",cond(`_par'<0.10,"*","")))
            display "  ar.L`i'" _col(30) %9.5f `_bar' _col(40) %8.4f `_sear' "  `st'"
        }

        * MA lags
        forval i = 1/`q' {
            local _bma  = _b[ARMA:L`i'.ma]
            local _sema = _se[ARMA:L`i'.ma]
            local _pma  = 2*(1 - normal(abs(`_bma' / `_sema')))
            local st = cond(`_pma'<0.01,"***",cond(`_pma'<0.05,"**",cond(`_pma'<0.10,"*","")))
            display "  ma.L`i'" _col(30) %9.5f `_bma' _col(40) %8.4f `_sema' "  `st'"
        }

        display "{hline 60}"
        display "  Variance equation"
        display "{hline 60}"

        * Omega
        local _bomega  = _b[HET:_cons]
        local _seomega = _se[HET:_cons]
        local _pomega  = 2*(1 - normal(abs(`_bomega' / `_seomega')))
        local st = cond(`_pomega'<0.01,"***",cond(`_pomega'<0.05,"**",cond(`_pomega'<0.10,"*","")))
        display "  omega (ω)" _col(30) %9.5f `_bomega' _col(40) %8.4f `_seomega' "  `st'"

        * ARCH
        local _ba1  = _b[ARCH:L1.arch]
        local _sea1 = _se[ARCH:L1.arch]
        local _pa1  = 2*(1 - normal(abs(`_ba1' / `_sea1')))
        local st = cond(`_pa1'<0.01,"***",cond(`_pa1'<0.05,"**",cond(`_pa1'<0.10,"*","")))
        display "  ARCH(1) (α)" _col(30) %9.5f `_ba1' _col(40) %8.4f `_sea1' "  `st'"

        * GARCH
        local _bg1  = _b[ARCH:L1.garch]
        local _seg1 = _se[ARCH:L1.garch]
        local _pg1  = 2*(1 - normal(abs(`_bg1' / `_seg1')))
        local st = cond(`_pg1'<0.01,"***",cond(`_pg1'<0.05,"**",cond(`_pg1'<0.10,"*","")))
        display "  GARCH(1) (δ)" _col(30) %9.5f `_bg1' _col(40) %8.4f `_seg1' "  `st'"

        * het controls in variance
        foreach v of local het_controls {
            local _bhw  = _b[HET:`v']
            local _sehw = _se[HET:`v']
            local _phw  = 2*(1 - normal(abs(`_bhw' / `_sehw')))
            local st = cond(`_phw'<0.01,"***",cond(`_phw'<0.05,"**",cond(`_phw'<0.10,"*","")))
            display "  `v' (γ)" _col(30) %9.5f `_bhw' _col(40) %8.4f `_sehw' "  `st'"
        }

        * Persistence
        local _persist = `_ba1' + `_bg1'
        display "  Persistence (α+δ)" _col(30) %9.5f `_persist'
        if `_persist' >= 1 {
            display "  WARNING: persistence >= 1"
        }

        display "{hline 60}"
        display "  *** p<0.01  ** p<0.05  * p<0.10"
        display "{hline 60}"
    }
}


* ============================================================================
* 10. Save single combined CSV
* ============================================================================
quietly {
    use `out_file', clear
    export delimited using ///
        "output from stata/pq_results_tdf5/garch_results_`zone'_tdf5.csv", replace
}
display _n "Results saved to: output from stata/pq_results_tdf5/garch_results_`zone'_tdf5.csv"
