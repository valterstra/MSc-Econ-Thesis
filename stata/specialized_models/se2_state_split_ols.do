clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

global DATA_FILE "stata_input/state_split/state_split_SE2_2023-01-01_2023-12-31_thr5p0_log.csv"

quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    drop if missing(high_sep)
    gen double stata_clock = clock(timestamp, "YMDhms")
    format stata_clock %tc
    duplicates drop stata_clock, force
    tsset stata_clock, delta(3600000)
    drop timestamp
}

local depvar price_ds
local controls "wind_log consump_log_ds"

capture mkdir "output from stata"
capture mkdir "output from stata/ols_results"
capture mkdir "output from stata/ols_results/state_split"

tempname out_post
tempfile out_file
postfile `out_post' str12 sample str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

foreach sample in lowsep highsep {
    preserve
    if "`sample'" == "lowsep" {
        keep if high_sep == 0
    }
    else {
        keep if high_sep == 1
    }

    quietly regress `depvar' `controls', vce(robust)

    foreach v in _cons wind_log consump_log_ds {
        capture scalar _b_tmp = _b[`v']
        if _rc == 0 {
            scalar _se_tmp = _se[`v']
            scalar _p_tmp  = 2*ttail(e(df_r), abs(_b_tmp/_se_tmp))
            local st = cond(_p_tmp < 0.01, "***", cond(_p_tmp < 0.05, "**", cond(_p_tmp < 0.10, "*", "")))
            post `out_post' ("`sample'") ("`v'") (_b_tmp) (_se_tmp) (_p_tmp) ("`st'")
        }
    }

    post `out_post' ("`sample'") ("N") (e(N)) (.) (.) ("")
    post `out_post' ("`sample'") ("R2") (e(r2)) (.) (.) ("")
    restore
}

postclose `out_post'

quietly {
    use `out_file', clear
    local stub = subinstr("$DATA_FILE", "stata_input/state_split/state_split_", "", 1)
    local stub = subinstr("`stub'", ".csv", "", 1)
    export delimited using ///
        "output from stata/ols_results/state_split/ols_results_`stub'_statesplit.csv", replace
}
