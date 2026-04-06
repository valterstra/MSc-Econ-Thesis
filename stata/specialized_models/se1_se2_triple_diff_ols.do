clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

global DATA_FILE "stata_input/triple_diff/triple_diff_SE1_SE2_2024-01-01_2025-12-31_log.csv"

quietly {
    import delimited using "$DATA_FILE", clear varnames(1) case(lower)
    egen long time_id = group(timestamp occurrence)
}

local depvar price_ds
local controls "is_se2 post_fbmc se2_post wind_log wind_se2 wind_post_fbmc wind_se2_post_fbmc consump_log_ds"

quietly regress `depvar' `controls', vce(cluster time_id)

tempname out_post
tempfile out_file
postfile `out_post' str32 variable double (value se pval) str4 stars ///
    using `out_file', replace

foreach v in _cons is_se2 post_fbmc se2_post wind_log wind_se2 wind_post_fbmc wind_se2_post_fbmc consump_log_ds {
    scalar _b_tmp = _b[`v']
    scalar _se_tmp = _se[`v']
    scalar _p_tmp = 2*ttail(e(df_r), abs(_b_tmp/_se_tmp))
    local st = cond(_p_tmp < 0.01, "***", cond(_p_tmp < 0.05, "**", cond(_p_tmp < 0.10, "*", "")))
    post `out_post' ("`v'") (_b_tmp) (_se_tmp) (_p_tmp) ("`st'")
}

post `out_post' ("N") (e(N)) (.) (.) ("")
post `out_post' ("R2") (e(r2)) (.) (.) ("")
postclose `out_post'

local out_dir "output from stata/ols_results/triple_diff"
capture mkdir "output from stata"
capture mkdir "output from stata/ols_results"
capture mkdir "`out_dir'"

quietly {
    use `out_file', clear
    local stub = subinstr("$DATA_FILE", "stata_input/triple_diff/triple_diff_", "", 1)
    local stub = subinstr("`stub'", ".csv", "", 1)
    export delimited using ///
        "`out_dir'/ols_results_`stub'_triplediff.csv", replace
}
