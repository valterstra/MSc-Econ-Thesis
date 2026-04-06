clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

global DATA_FILE "stata_input/local_congestion/local_congestion_SE2_2022-01-01_2022-12-31_log.csv"
global HET_ALL_CONTROLS 1
global HET_INCLUDE_CONGESTION 0
global T_DF 5
local arma_p = 3
local arma_d = 0
local arma_q = 1

if $T_DF > 0 {
    local dist_spec "distribution(t $T_DF)"
    local df_label "_tdf$T_DF"
}
else {
    local dist_spec "distribution(t)"
    local df_label "_tdfest"
}

quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
    drop timestamp
}

local depvar price_ds
local cong_terms "d_congested wind_cong"

ds stata_clock `depvar', not
local controls `r(varlist)'
local het_base : list controls - cong_terms

if $HET_ALL_CONTROLS {
    local het_controls `het_base'
}
else {
    local het_controls wind_log
}

if $HET_INCLUDE_CONGESTION {
    local het_controls `het_controls' d_congested wind_cong
}

capture arch `depvar' `controls',   ///
    ar(1/`arma_p') ma(1/`arma_q')   ///
    arch(1) garch(1)                ///
    het(`het_controls')             ///
    `dist_spec'                     ///
    vce(robust)                     ///
    difficult                       ///
    nrtolerance(1e-3)

if _rc != 0 & _rc != 430 {
    display as error "arch failed with rc=" _rc
    exit _rc
}

quietly {
    estat ic
    matrix _ic  = r(S)
    scalar _aic = _ic[1,5]
    scalar _bic = _ic[1,6]

    scalar _b_cons  = _b[`depvar':_cons]
    scalar _se_cons = _se[`depvar':_cons]
    scalar _p_cons  = 2*(1 - normal(abs(_b_cons / _se_cons)))

    foreach v of local controls {
        scalar _b_`v'  = _b[`depvar':`v']
        scalar _se_`v' = _se[`depvar':`v']
        scalar _p_`v'  = 2*(1 - normal(abs(_b_`v' / _se_`v')))
    }
}

local out_dir "output from stata/garch_results/local_congestion"
capture mkdir "output from stata/garch_results"
capture mkdir "`out_dir'"

tempname out_post
tempfile out_file
postfile `out_post' str16 type str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

local st_cons = cond(_p_cons < 0.01, "***", cond(_p_cons < 0.05, "**", cond(_p_cons < 0.10, "*", "")))
post `out_post' ("fit") ("AIC") (_aic) (.) (.) ("")
post `out_post' ("fit") ("BIC") (_bic) (.) (.) ("")
post `out_post' ("coef") ("_cons") (_b_cons) (_se_cons) (_p_cons) ("`st_cons'")

foreach v of local controls {
    local st_`v' = cond(_p_`v' < 0.01, "***", cond(_p_`v' < 0.05, "**", cond(_p_`v' < 0.10, "*", "")))
    post `out_post' ("coef") ("`v'") (_b_`v') (_se_`v') (_p_`v') ("`st_`v''")
}

postclose `out_post'

quietly {
    use `out_file', clear
    local stub = subinstr("$DATA_FILE", "stata_input/local_congestion/local_congestion_", "", 1)
    local stub = subinstr("`stub'", ".csv", "", 1)
    export delimited using "`out_dir'/garch_results_`stub'_joint`df_label'_localcong.csv", replace
}
