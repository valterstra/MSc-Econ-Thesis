clear all
set more off
set linesize 120

cd "C:\Users\ValterAdmin\Documents\VS code projects\EconMScThesis"

* ============================================================================
* spillover_collinearity_checks.do
*
* Diagnostics for northern spillover specifications in SE3 / SE4.
*
* Checks:
* 1. Pairwise correlation between local and northern wind
* 2. VIF from OLS approximation of the mean equation
* 3. Nested OLS specifications
* 4. Residualized northern wind test
* ============================================================================

global CHECK_YEAR "2025"

foreach zone in SE2 SE3 SE4 {

    local data_file "stata_input/northern_spillover/northern_spillover_`zone'_${CHECK_YEAR}-01-01_${CHECK_YEAR}-12-31_log.csv"

    display _n "{hline 78}"
    display "Zone: `zone'"
    display "File: `data_file'"
    display "{hline 78}"

    quietly import delimited using "`data_file'", clear varnames(1) case(lower)

    * 1. Pairwise correlation
    display _n "1. Pairwise Correlation"
    pwcorr wind_log north_wind_log, sig

    * 2. VIF on OLS approximation
    display _n "2. OLS Approximation + VIF"
    ds timestamp price_ds wind_log north_wind_log, not
    local other_controls `r(varlist)'
    regress price_ds wind_log north_wind_log `other_controls'
    vif

    * 3. Nested specifications
    display _n "3. Nested OLS Specifications"
    display "Local wind only"
    regress price_ds wind_log `other_controls'

    display _n "Northern wind only"
    regress price_ds north_wind_log `other_controls'

    display _n "Both local and northern wind"
    regress price_ds wind_log north_wind_log `other_controls'

    * 4. Residualized northern wind
    display _n "4. Residualized Northern Wind"
    regress north_wind_log wind_log
    predict double north_resid, resid
    regress price_ds wind_log north_resid `other_controls'
    drop north_resid
}
