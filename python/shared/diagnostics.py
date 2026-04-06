import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS

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


def run_collinearity_diagnostics(df, exog_vars, zone='SE1', results_dir='results',
                                 vif_severe_threshold=10.0, top_corr_pairs=10):
    """
    Diagnose multicollinearity for exogenous variables used in ARMAX.
    Saves VIF and top absolute-correlation pairs to CSV.
    """
    print("\n--- COLLINEARITY DIAGNOSTICS (EXOGENOUS VARIABLES) ---")

    X = df[exog_vars].copy()
    rows_before = len(X)
    X = X.dropna()
    rows_after = len(X)
    if rows_after == 0:
        print("No observations available after dropping NaNs. Skipping collinearity diagnostics.")
        return None
    if rows_after < rows_before:
        print(f"Dropped {rows_before - rows_after} rows with missing exogenous values for diagnostics.")

    X = X.astype(float)

    # Condition number on exogenous design with constant.
    try:
        cond_no = float(np.linalg.cond(sm.add_constant(X).values))
    except Exception:
        cond_no = np.nan

    vif_rows = []
    X_values = X.values
    for i, var in enumerate(exog_vars):
        try:
            vif_val = float(variance_inflation_factor(X_values, i))
        except Exception:
            vif_val = np.nan
        vif_rows.append({'variable': var, 'vif': vif_val, 'severe_vif_gt_10': bool(vif_val > vif_severe_threshold) if pd.notna(vif_val) else False})
    vif_df = pd.DataFrame(vif_rows).sort_values('vif', ascending=False, na_position='last')

    corr_abs = X.corr().abs()
    corr_pairs = []
    cols = list(corr_abs.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append({
                'var1': cols[i],
                'var2': cols[j],
                'abs_corr': float(corr_abs.iloc[i, j])
            })
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('abs_corr', ascending=False).head(int(top_corr_pairs))

    severe_count = int(vif_df['severe_vif_gt_10'].sum())
    print(f"Observations used: {rows_after:,}")
    print(f"Condition number (X with constant): {cond_no:.2e}" if pd.notna(cond_no) else "Condition number: NaN")
    print(f"Variables with VIF > {vif_severe_threshold:.0f}: {severe_count}/{len(vif_df)}")

    print("\nTop VIF values:")
    for _, row in vif_df.head(min(10, len(vif_df))).iterrows():
        vif_txt = f"{row['vif']:.2f}" if pd.notna(row['vif']) else "NaN"
        flag = "  [SEVERE]" if row['severe_vif_gt_10'] else ""
        print(f"  {row['variable']:<35} VIF={vif_txt}{flag}")

    print(f"\nTop {min(int(top_corr_pairs), len(corr_pairs_df))} absolute correlation pairs:")
    for _, row in corr_pairs_df.iterrows():
        print(f"  {row['var1']} vs {row['var2']}: |corr|={row['abs_corr']:.3f}")

    os.makedirs(results_dir, exist_ok=True)
    vif_path = os.path.join(results_dir, f'collinearity_vif_{zone}.csv')
    corr_path = os.path.join(results_dir, f'collinearity_corr_pairs_{zone}.csv')
    vif_df.to_csv(vif_path, index=False)
    corr_pairs_df.to_csv(corr_path, index=False)
    print(f"\nSaved VIF diagnostics: {vif_path}")
    print(f"Saved correlation diagnostics: {corr_path}")

    return {
        'condition_number': cond_no,
        'severe_vif_count': severe_count,
        'vif_path': vif_path,
        'corr_path': corr_path
    }

