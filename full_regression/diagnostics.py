"""
################################################################################
#  [Module 04/10]  diagnostics.py  –  Statistical Diagnostic Tests
#
#  Contains:
#    1. run_ljungbox_test()              : autocorrelation test on residuals
#    2. run_heteroskedasticity_tests()   : Engle ARCH + Ljung-Box on squared residuals
#    3. run_stationarity_tests()         : ADF and DF-GLS unit root tests
#
#  All functions follow Fredriksson (2016) methodology.
#
#  Dependencies: none (self-contained)
################################################################################
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS


# --- 3. DIAGNOSTIC TEST FUNCTIONS ---

def run_ljungbox_test(residuals, lags=[5, 10, 15, 20], return_results=False, print_output=True):
    """
    Ljung-Box test for autocorrelation in residuals.

    Tests the null hypothesis that residuals are independently distributed (no autocorrelation).
    Low p-values (< 0.05) indicate significant autocorrelation.

    Following Fredriksson (2016), tests at multiple lag lengths.

    Parameters:
    - residuals: Residual series to test
    - lags: List of lag values to test
    - return_results: If True, return DataFrame with results
    - print_output: If True, print formatted table (default behavior)

    Returns:
    - If return_results=True: DataFrame with columns [lag, test_stat, p_value, reject_h0]
    - If return_results=False: None (backward compatible)
    """
    if print_output:
        print("\n--- LJUNG-BOX TEST FOR AUTOCORRELATION ---")
        print("H0: Residuals are independently distributed (no autocorrelation)")
        print("Reject H0 if p-value < 0.05\n")

    # Run test at multiple lags
    lb_results = acorr_ljungbox(residuals, lags=lags, return_df=True)

    if print_output:
        print(f"{'Lag':<10} {'Test Statistic':<20} {'P-value':<15} {'Result'}")
        print("-" * 60)

    results_data = []
    for lag in lags:
        if lag in lb_results.index:
            stat = lb_results.loc[lag, 'lb_stat']
            pval = lb_results.loc[lag, 'lb_pvalue']
            reject_h0 = pval < 0.05

            if print_output:
                result = "REJECT H0 (autocorr present)" if reject_h0 else "Fail to reject H0"
                print(f"{lag:<10} {stat:<20.4f} {pval:<15.4f} {result}")

            results_data.append({
                'lag': lag,
                'test_stat': stat,
                'p_value': pval,
                'reject_h0': reject_h0
            })

    if return_results:
        return pd.DataFrame(results_data)
    return None


def run_heteroskedasticity_tests(residuals, nlags=10):
    """
    Tests for heteroskedasticity and ARCH effects.

    1. Engle's ARCH test (Lagrange Multiplier test)
    2. Ljung-Box Q test on squared residuals

    Following Fredriksson (2016) Table 2.
    """
    print("\n--- HETEROSKEDASTICITY AND ARCH EFFECTS TESTS ---")

    # 1. Engle's ARCH Test (Lagrange Multiplier)
    print("\n1. ENGLE'S ARCH TEST (Lagrange Multiplier)")
    print("   H0: No ARCH effects (homoskedastic residuals)")
    print("   Reject H0 if p-value < 0.05\n")

    try:
        # ARCH test with specified lags
        lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=nlags)

        print(f"   LM Statistic: {lm_stat:.4f}")
        print(f"   LM P-value:   {lm_pval:.4f}")
        print(f"   F-Statistic:  {f_stat:.4f}")
        print(f"   F P-value:    {f_pval:.4f}")

        if lm_pval < 0.05:
            print(f"   Result: REJECT H0 - ARCH effects detected (use GARCH model)")
        else:
            print(f"   Result: Fail to reject H0 - No significant ARCH effects")

    except Exception as e:
        print(f"   Error running ARCH test: {e}")

    # 2. Ljung-Box Q test on squared residuals
    print("\n2. LJUNG-BOX Q TEST ON SQUARED RESIDUALS")
    print("   H0: No autocorrelation in squared residuals")
    print("   Reject H0 if p-value < 0.05\n")

    try:
        squared_resid = residuals ** 2
        lb_squared = acorr_ljungbox(squared_resid, lags=[5, 10, 15, 20], return_df=True)

        print(f"   {'Lag':<10} {'Q-Statistic':<20} {'P-value':<15} {'Result'}")
        print("   " + "-" * 60)

        for lag in [5, 10, 15, 20]:
            if lag in lb_squared.index:
                stat = lb_squared.loc[lag, 'lb_stat']
                pval = lb_squared.loc[lag, 'lb_pvalue']
                result = "REJECT H0 (heteroskedasticity)" if pval < 0.05 else "Fail to reject H0"
                print(f"   {lag:<10} {stat:<20.4f} {pval:<15.4f} {result}")

    except Exception as e:
        print(f"   Error running Ljung-Box on squared residuals: {e}")


def run_stationarity_tests(series, series_name="Series"):
    """
    Stationarity tests: Augmented Dickey-Fuller (ADF) and Dickey-Fuller GLS (DF-GLS).

    H0: Series has a unit root (non-stationary)
    Reject H0 if p-value < 0.05 (series is stationary)

    Following Fredriksson (2016) Table G in appendix.
    """
    print(f"\n--- STATIONARITY TESTS: {series_name} ---")
    print("H0: Series has a unit root (non-stationary)")
    print("Reject H0 if p-value < 0.05 (series is stationary)\n")

    # 1. Augmented Dickey-Fuller (ADF) Test
    print("1. AUGMENTED DICKEY-FULLER (ADF) TEST")
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_stat, adf_pval = adf_result[0], adf_result[1]
        adf_lags = adf_result[2]

        print(f"   ADF Statistic: {adf_stat:.4f}")
        print(f"   P-value:       {adf_pval:.4f}")
        print(f"   Lags used:     {adf_lags}")
        print(f"   Critical values: 1%={adf_result[4]['1%']:.3f}, 5%={adf_result[4]['5%']:.3f}, 10%={adf_result[4]['10%']:.3f}")

        if adf_pval < 0.05:
            print(f"   Result: REJECT H0 - Series is STATIONARY")
        else:
            print(f"   Result: Fail to reject H0 - Series is NON-STATIONARY")

    except Exception as e:
        print(f"   Error running ADF test: {e}")

    # 2. Dickey-Fuller GLS (DF-GLS) Test
    print("\n2. DICKEY-FULLER GLS (DF-GLS) TEST")
    try:
        dfgls = DFGLS(series.dropna())
        dfgls_stat = dfgls.stat
        dfgls_pval = dfgls.pvalue

        print(f"   DF-GLS Statistic: {dfgls_stat:.4f}")
        print(f"   P-value:          {dfgls_pval:.4f}")
        print(f"   Critical values:  1%={dfgls.critical_values['1%']:.3f}, 5%={dfgls.critical_values['5%']:.3f}, 10%={dfgls.critical_values['10%']:.3f}")

        if dfgls_pval < 0.05:
            print(f"   Result: REJECT H0 - Series is STATIONARY")
        else:
            print(f"   Result: Fail to reject H0 - Series is NON-STATIONARY")

    except Exception as e:
        print(f"   Error running DF-GLS test: {e}")
