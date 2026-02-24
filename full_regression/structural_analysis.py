"""
################################################################################
#  [Module 08/10]  structural_analysis.py  –  Rolling Windows, Structural Breaks & Quantile Regression
#
#  Contains (in order):
#    --- Rolling Window Coefficient Estimation ---
#    1.  run_rolling_window_analysis()           : OLS on overlapping windows
#                                                  (supports window-local preprocessing)
#
#    --- Helper Utilities (used by break analysis) ---
#    2.  _get_break_model_tag()                  : filesystem-safe output tag
#    3.  _get_break_model_label()                : human-readable label for plots
#    4.  _extract_armax_wind_coef()              : robust extraction of wind coef from ARMAX
#
#    --- Rolling Coefficient Estimation (for Break Analysis) ---
#    5.  _estimate_rolling_wind_coefficients()   : estimate rolling OLS or ARMAX coefficients
#    6.  _run_dynamic_break_lr_tests()           : LR Chow tests at candidate break dates
#
#    --- Level Breaks (Bai-Perron) ---
#    7.  run_structural_break_analysis()         : step changes in coefficient using ruptures
#
#    --- Trend Breaks ---
#    8.  run_trend_break_analysis_legacy()       : segmented linear regression + BIC
#    9.  run_trend_break_analysis_bp_supf()      : Bai-Perron sequential supF tests
#    10. run_trend_break_analysis()              : wrapper that dispatches to 'legacy' or 'bp_supf'
#
#    --- Quantile Regression ---
#    11. run_quantile_regression_analysis()      : wind coefficient across price quantiles
#
#  Dependencies: config, preprocessing, utils, regression_models
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy import stats

from .preprocessing import preprocess_data_for_regression
from .utils import get_regression_variable_names
from .regression_models import _fit_armax_with_fallback
from .config import (
    ARMAX_ALLOW_NONCONVERGED, ARMAX_MAXITER, ARMAX_SOLVER,
    ARMAX_USE_WARM_START, ARMAX_ENABLE_FALLBACK_ORDERS, ARMAX_FALLBACK_ORDERS
)

def run_rolling_window_analysis(df, zone, Y=None, exog_vars=None,
                                window_years=3, step_years=1, min_obs=24*180,
                                plots_dir="plots", results_dir="results",
                                use_window_local_preprocessing=False,
                                target_region='SE1',
                                negative_price_handling='clip',
                                outlier_method='fredriksson',
                                handle_outliers_before_log=False):
    """
    Estimate wind coefficient using overlapping rolling windows with OLS.

    use_window_local_preprocessing=False:
      expects preprocessed df + Y/exog_vars (legacy behavior).
    use_window_local_preprocessing=True:
      applies full preprocessing independently inside each window.
    """
    from dateutil.relativedelta import relativedelta

    print("\n" + "="*80)
    print("ROLLING-WINDOW WIND COEFFICIENT ESTIMATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Window size: {window_years} years")
    step_months = int(round(step_years * 12)) if step_years < 1 else None
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")

    if use_window_local_preprocessing:
        y_name, exog_names = get_regression_variable_names(df, target_region=target_region)
        tmp = df.sort_index().copy()
        print("  Preprocessing mode: window-local")
        print(f"  Raw data range: {tmp.index.min()} to {tmp.index.max()}")
        print(f"  Total raw observations: {len(tmp):,}")
    else:
        if Y is None or exog_vars is None:
            raise ValueError("Legacy rolling mode requires Y and exog_vars.")
        y_name = Y.name
        exog_names = exog_vars
        cols_needed = [y_name] + exog_names
        tmp = df[cols_needed].dropna().copy()
        tmp = tmp.sort_index()
        print("  Preprocessing mode: full-sample (legacy)")
        print(f"  Data range: {tmp.index.min()} to {tmp.index.max()}")
        print(f"  Total observations after cleaning: {len(tmp):,}")

    wind_col = [col for col in exog_names if 'Wind' in col and 'Forecast' in col][0]
    control_cols = [col for col in exog_names if col != wind_col]
    print(f"\nTarget variable: {wind_col}")
    print(f"Control variables: {control_cols}")

    results = []
    start_date = tmp.index.min()
    end_of_data = tmp.index.max()

    total_windows = 0
    temp_start = start_date
    while temp_start <= end_of_data:
        total_windows += 1
        if step_months:
            temp_start = temp_start + relativedelta(months=step_months)
        else:
            temp_start = temp_start + relativedelta(years=int(step_years))

    print(f"\n--- Estimating Rolling Windows ---")
    print(f"Total candidate windows: {total_windows}\n")

    processed_count = 0
    valid_count = 0
    current_start = start_date

    while current_start <= end_of_data:
        processed_count += 1
        window_end = current_start + relativedelta(years=window_years)
        raw_window = tmp[(tmp.index >= current_start) & (tmp.index < window_end)]

        if len(raw_window) < min_obs:
            if step_months:
                current_start = current_start + relativedelta(months=step_months)
            else:
                current_start = current_start + relativedelta(years=int(step_years))
            continue

        if use_window_local_preprocessing:
            try:
                window_processed = preprocess_data_for_regression(
                    raw_window,
                    negative_price_handling=negative_price_handling,
                    outlier_method=outlier_method,
                    handle_outliers_before_log=handle_outliers_before_log,
                    suppress_output=True,
                    save_temp_plots=False
                )
            except Exception as e:
                print(f"[{processed_count}/{total_windows}] Window {current_start.date()} skipped (preprocessing failed: {e})")
                if step_months:
                    current_start = current_start + relativedelta(months=step_months)
                else:
                    current_start = current_start + relativedelta(years=int(step_years))
                continue

            cols_needed = [y_name] + exog_names
            window_data = window_processed[cols_needed].dropna().copy()
            window_data = window_data.sort_index()
        else:
            window_data = raw_window

        if len(window_data) >= min_obs:
            valid_count += 1
            actual_start_year = window_data.index.min().year
            actual_end_year = window_data.index.max().year
            print(f"[{processed_count}/{total_windows}] Window {actual_start_year} to {actual_end_year}... ", end="")

            X = sm.add_constant(window_data[exog_names])
            y = window_data[y_name]
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})

            window_midpoint = current_start + relativedelta(months=window_years * 6)

            results.append({
                'window_start': current_start,
                'window_end': window_data.index.max(),
                'window_midpoint': window_midpoint,
                'beta_wind': model.params[wind_col],
                'se_wind': model.bse[wind_col],
                't_stat': model.tvalues[wind_col],
                'pvalue': model.pvalues[wind_col],
                'n_obs': len(window_data),
                'r_squared': model.rsquared
            })

            print(f"beta_wind={model.params[wind_col]:.4f}, p={model.pvalues[wind_col]:.4f}")

        if step_months:
            current_start = current_start + relativedelta(months=step_months)
        else:
            current_start = current_start + relativedelta(years=int(step_years))

    if not results:
        print("\nWARNING: No valid windows found. Check data range and window parameters.")
        return

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("ROLLING-WINDOW SUMMARY STATISTICS")
    print("="*80)
    print(f"\nNumber of windows analyzed: {len(results_df)}")
    print(f"Valid windows / candidates: {valid_count}/{total_windows}")
    print(f"\nWind coefficient (beta):")
    print(f"  Mean:   {results_df['beta_wind'].mean():.6f}")
    print(f"  Std:    {results_df['beta_wind'].std():.6f}")
    print(f"  Min:    {results_df['beta_wind'].min():.6f}")
    print(f"  Max:    {results_df['beta_wind'].max():.6f}")

    sig_count = (results_df['pvalue'] < 0.05).sum()
    print(f"\nSignificance at 5% level: {sig_count}/{len(results_df)} windows "
          f"({100*sig_count/len(results_df):.1f}%)")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'rolling_wind_coef_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    midpoints = pd.to_datetime(results_df['window_midpoint'])
    beta_values = results_df['beta_wind'].values
    se_values = results_df['se_wind'].values

    upper_95 = beta_values + 1.96 * se_values
    lower_95 = beta_values - 1.96 * se_values

    ax.plot(midpoints, beta_values, color='blue', linewidth=2, marker='o',
            markersize=6, label=r'$\beta_{wind}$ coefficient')
    ax.fill_between(midpoints, lower_95, upper_95, color='blue', alpha=0.2,
                    label='95% CI')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
               label='Zero')

    mean_beta = results_df['beta_wind'].mean()
    ax.axhline(y=mean_beta, color='red', linestyle=':', linewidth=1.5,
               label=f'Mean = {mean_beta:.4f}')

    ax.set_title(f'Rolling-Window Wind Coefficient - {zone}\n'
                 f'({window_years}-year windows, {step_years}-year steps, Newey-West SE)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Window Midpoint', fontsize=12)
    ax.set_ylabel(r'$\beta_{wind}$ (Wind Coefficient)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = (f'Windows: {len(results_df)}\n'
                  f'Mean: {mean_beta:.4f}\n'
                  f'Std: {results_df["beta_wind"].std():.4f}\n'
                  f'Sig (p<0.05): {sig_count}/{len(results_df)}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    zone_plots_dir = os.path.join(plots_dir, zone)
    os.makedirs(zone_plots_dir, exist_ok=True)
    plot_path = os.path.join(zone_plots_dir, f'rolling_wind_coef_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("ROLLING-WINDOW ANALYSIS COMPLETE")
    print("="*80)

def _get_break_model_tag(estimation_model, dynamic_armax_order=(3, 0, 3)):
    """Return filesystem-safe tag used in break-analysis outputs."""
    if estimation_model == 'ols':
        return 'ols'
    if estimation_model == 'dynamic_armax':
        p, d, q = dynamic_armax_order
        return f'dynamic_armax_{p}{d}{q}'
    raise ValueError(
        f"Unknown structural-break estimation model: '{estimation_model}'. "
        "Allowed values: 'ols', 'dynamic_armax'."
    )


def _get_break_model_label(estimation_model, dynamic_armax_order=(3, 0, 3)):
    """Return human-readable label for plot titles and summaries."""
    if estimation_model == 'ols':
        return 'OLS'
    if estimation_model == 'dynamic_armax':
        return f'Dynamic ARMAX{dynamic_armax_order}'
    raise ValueError(
        f"Unknown structural-break estimation model: '{estimation_model}'. "
        "Allowed values: 'ols', 'dynamic_armax'."
    )


def _extract_armax_wind_coef(armax_model, wind_col):
    """Robustly extract wind coefficient and standard error from ARIMAResults."""
    param_names = list(armax_model.param_names)
    params = np.asarray(armax_model.params)
    bse = np.asarray(armax_model.bse)
    pvalues = np.asarray(armax_model.pvalues)

    if wind_col in param_names:
        idx = param_names.index(wind_col)
        return params[idx], bse[idx], pvalues[idx]

    normalized = [name.replace('x', '').replace('.', '').strip().lower() for name in param_names]
    target_norm = wind_col.lower()
    for idx, name_norm in enumerate(normalized):
        if target_norm in name_norm:
            return params[idx], bse[idx], pvalues[idx]

    raise ValueError(
        f"Could not locate wind coefficient '{wind_col}' in ARMAX parameter names: {param_names}"
    )



def _estimate_rolling_wind_coefficients(tmp, Y_name, exog_vars, wind_col,
                                        window_years=1, step_years=1/12,
                                        min_obs=24*365 - 24*30,
                                        estimation_model='ols',
                                        dynamic_armax_order=(3, 0, 3)):
    """
    Estimate rolling-window wind coefficients with either OLS or dynamic ARMAX.
    Returns: rolling_df, total_candidate_windows, failure_stats.
    """
    from dateutil.relativedelta import relativedelta

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    rolling_results = []
    failure_stats = {
        'failed_windows_total': 0,
        'failed_windows_exception': 0,
        'failed_windows_nonconvergence': 0,
        'used_nonconverged_windows': 0
    }

    # First pass: build candidate windows so we can report progress/ETA accurately.
    candidate_windows = []
    start_date = tmp.index.min()
    end_of_data = tmp.index.max()
    current_start = start_date

    while current_start <= end_of_data:
        window_end = current_start + relativedelta(years=window_years)
        window_data = tmp[(tmp.index >= current_start) & (tmp.index < window_end)]

        if len(window_data) >= min_obs:
            midpoint = current_start + relativedelta(months=window_years * 6)
            candidate_windows.append((current_start, window_end, midpoint, window_data))

        if step_months:
            current_start = current_start + relativedelta(months=step_months)
        else:
            current_start = current_start + relativedelta(years=int(step_years))

    total_candidate_windows = len(candidate_windows)
    batch_start_time = time.perf_counter()

    # Second pass: estimate each candidate and print progress after each attempt.
    for i, (current_start, window_end, midpoint, window_data) in enumerate(candidate_windows, start=1):
        status_label = "ok"
        try:
            if estimation_model == 'ols':
                X = sm.add_constant(window_data[exog_vars])
                y = window_data[Y_name]
                model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
                rolling_results.append({
                    'midpoint': midpoint,
                    'beta_wind': model.params[wind_col],
                    'se_wind': model.bse[wind_col],
                    'pvalue': model.pvalues[wind_col],
                    'r_squared': model.rsquared,
                    'n_obs_window': len(window_data),
                    'model_used': 'ols',
                    'fit_status': 'ok'
                })
            elif estimation_model == 'dynamic_armax':
                y = window_data[Y_name]
                X_exog = window_data[exog_vars]
                fit_result = _fit_armax_with_fallback(
                    y=y,
                    X_exog=X_exog,
                    primary_order=dynamic_armax_order,
                    context_label=f"rolling window {current_start} -> {window_end}",
                    allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                    maxiter=ARMAX_MAXITER,
                    solver=ARMAX_SOLVER,
                    use_warm_start=ARMAX_USE_WARM_START,
                    enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                    fallback_orders=ARMAX_FALLBACK_ORDERS
                )
                if not fit_result['ok']:
                    failure_stats['failed_windows_total'] += 1
                    if fit_result['fail_reason'] == 'non_converged':
                        failure_stats['failed_windows_nonconvergence'] += 1
                        status_label = "failed_nonconverged"
                    else:
                        failure_stats['failed_windows_exception'] += 1
                        status_label = "failed_exception"
                else:
                    armax_model = fit_result['model']
                    beta_wind, se_wind, pvalue = _extract_armax_wind_coef(armax_model, wind_col)
                    fit_status = 'ok'
                    if not fit_result.get('converged', True):
                        failure_stats['used_nonconverged_windows'] += 1
                        fit_status = 'nonconverged_used'
                        status_label = "used_nonconverged"
                    diag = fit_result.get('diagnostics', {})
                    status_label = (
                        f"{status_label}:order={fit_result.get('selected_order', dynamic_armax_order)}"
                        f",iter={diag.get('iterations')},warn={diag.get('warnflag')}"
                    )
                    rolling_results.append({
                        'midpoint': midpoint,
                        'beta_wind': beta_wind,
                        'se_wind': se_wind,
                        'pvalue': pvalue,
                        'r_squared': np.nan,
                        'n_obs_window': len(window_data),
                        'model_used': 'dynamic_armax',
                        'fit_status': fit_status
                    })
            else:
                raise ValueError(
                    f"Unknown structural-break estimation model: '{estimation_model}'. "
                    "Allowed values: 'ols', 'dynamic_armax'."
                )
        except Exception:
            failure_stats['failed_windows_total'] += 1
            failure_stats['failed_windows_exception'] += 1
            status_label = "failed_exception"

        elapsed = time.perf_counter() - batch_start_time
        avg_per_window = elapsed / i
        remaining = total_candidate_windows - i
        eta_sec = avg_per_window * remaining
        print(f"[Window {i}/{total_candidate_windows}] status={status_label} "
              f"remaining={remaining} eta={eta_sec/60:.1f}m")

    rolling_df = pd.DataFrame(rolling_results)
    return rolling_df, total_candidate_windows, failure_stats


def _run_dynamic_break_lr_tests(tmp, Y_name, exog_vars, wind_col, test_dates, dynamic_armax_order=(3, 0, 3)):
    """
    Run ARMAX likelihood-ratio split tests at break dates.
    H0: one ARMAX model for full sample. H1: separate ARMAX models pre/post break.
    """
    results = []
    test_stats = {
        'total_test_dates': len(test_dates),
        'skipped_insufficient_obs': 0,
        'skipped_fit_fail_total': 0,
        'skipped_fit_fail_nonconvergence': 0,
        'skipped_fit_fail_exception': 0,
        'used_nonconverged_test_dates': 0
    }

    for test_info in test_dates:
        break_date = test_info['date']
        source = test_info['source']
        print(f"\nTesting break at {break_date.strftime('%Y-%m-%d')} ({source}) [ARMAX LR]:")

        pre_break = tmp[tmp.index < break_date]
        post_break = tmp[tmp.index >= break_date]

        if len(pre_break) < 100 or len(post_break) < 100:
            print(f"  Skipped: Insufficient observations (pre={len(pre_break)}, post={len(post_break)})")
            test_stats['skipped_insufficient_obs'] += 1
            continue

        try:
            fit_full = _fit_armax_with_fallback(
                y=tmp[Y_name],
                X_exog=tmp[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR full sample ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )
            fit_pre = _fit_armax_with_fallback(
                y=pre_break[Y_name],
                X_exog=pre_break[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR pre-break ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )
            fit_post = _fit_armax_with_fallback(
                y=post_break[Y_name],
                X_exog=post_break[exog_vars],
                primary_order=dynamic_armax_order,
                context_label=f"LR post-break ({break_date})",
                allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
                fallback_orders=ARMAX_FALLBACK_ORDERS
            )

            fit_bundle = [fit_full, fit_pre, fit_post]
            if not all(fr['ok'] for fr in fit_bundle):
                test_stats['skipped_fit_fail_total'] += 1
                if any(fr['fail_reason'] == 'non_converged' for fr in fit_bundle):
                    test_stats['skipped_fit_fail_nonconvergence'] += 1
                    print("  Skipped: Dynamic model fit non-converged")
                else:
                    test_stats['skipped_fit_fail_exception'] += 1
                    first_err = next((fr['error'] for fr in fit_bundle if not fr['ok']), "Unknown dynamic fit error")
                    print(f"  Skipped: Dynamic model fit failed ({first_err})")
                continue

            if any(not fr.get('converged', True) for fr in fit_bundle):
                test_stats['used_nonconverged_test_dates'] += 1

            model_full = fit_full['model']
            model_pre = fit_pre['model']
            model_post = fit_post['model']

            ll_full = float(model_full.llf)
            ll_pre = float(model_pre.llf)
            ll_post = float(model_post.llf)
            lr_df = int(max(len(model_full.params), 1))
            lr_stat = 2.0 * ((ll_pre + ll_post) - ll_full)
            p_value = 1.0 - stats.chi2.cdf(lr_stat, lr_df)

            beta_pre, se_pre, _ = _extract_armax_wind_coef(model_pre, wind_col)
            beta_post, se_post, _ = _extract_armax_wind_coef(model_post, wind_col)
            beta_change = beta_post - beta_pre
            beta_change_pct = (beta_change / abs(beta_pre) * 100.0) if beta_pre != 0 else np.nan

            print(f"  Pre-break:  n={len(pre_break):,}, beta_wind={beta_pre:.4f} (SE={se_pre:.4f})")
            print(f"  Post-break: n={len(post_break):,}, beta_wind={beta_post:.4f} (SE={se_post:.4f})")
            print(f"  Change in beta_wind: {beta_change:.4f} ({beta_change_pct:.1f}%)")
            print(f"  LR statistic: {lr_stat:.2f} (df={lr_df})")
            print(f"  p-value: {p_value:.4e}")
            print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
            print(f"  Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")

            results.append({
                'break_date': break_date,
                'source': source,
                'n_pre': len(pre_break),
                'n_post': len(post_break),
                'beta_wind_pre': beta_pre,
                'beta_wind_post': beta_post,
                'se_pre': se_pre,
                'se_post': se_post,
                'beta_change': beta_change,
                'beta_change_pct': beta_change_pct,
                'lr_statistic': lr_stat,
                'lr_df': lr_df,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'test_type': 'ARMAX_LR'
            })
        except Exception as e:
            print(f"  Skipped: Dynamic model fit failed ({e})")
            test_stats['skipped_fit_fail_total'] += 1
            test_stats['skipped_fit_fail_exception'] += 1
            continue

    return results, test_stats


def run_structural_break_analysis(df, zone, Y, exog_vars,
                                  max_breaks=5, min_segment_length=None,
                                  known_break_dates=None, trimming=0.15,
                                  window_years=1, step_years=1/12,
                                  min_obs=24*365 - 24*30,
                                  config_label=None,
                                  plots_dir="plots", results_dir="results",
                                  estimation_model='ols',
                                  dynamic_armax_order=(3, 0, 3)):
    """
    Detect structural breaks in the wind coefficient using Bai-Perron methodology.

    This function:
    1. Estimates rolling window coefficients to visualize coefficient evolution
    2. Applies Bai-Perron change point detection to identify structural breaks
    3. Runs Chow tests at detected break points and known event dates
    4. Generates CUSUM plots for parameter stability diagnostics
    5. Outputs comprehensive results and visualizations

    Note: Always uses logged variables (Wind_Forecast_Log).

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - Y: Dependent variable (Series)
    - exog_vars: List of exogenous variable column names
    - max_breaks: Maximum number of breaks to detect (default 5)
    - min_segment_length: Minimum observations between breaks (default: 10% of data)
    - known_break_dates: List of dates to test with Chow test (e.g., ['2022-02-24'] for Ukraine invasion).
                          Set to None or [] to only use Bai-Perron detected breaks.
    - trimming: Fraction of data to trim from endpoints for break detection (default 0.15)
    - window_years: Rolling window size in years (default 1)
    - step_years: Step size between windows in years (default 1/12 = 1 month)
    - min_obs: Minimum observations required per window (default 24*365 - 24*30)
    - config_label: Custom label for this configuration (e.g., '1y_window_1m_step').
                    If None, auto-generated from window_years and step_years.
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns:
    - Dictionary with break detection results
    """
    from dateutil.relativedelta import relativedelta

    # Generate config label for file naming if not provided
    if config_label is None:
        step_months = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months:
            step_label = f"{step_months}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"

        config_label = f"{window_label}_window_{step_label}_step"

    print("\n" + "="*80)
    print("STRUCTURAL BREAK ANALYSIS - BAI-PERRON METHODOLOGY")
    print("="*80)

    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Identify wind column from exog_vars
    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    print(f"\nConfiguration:")
    print(f"  Zone: {zone}")
    print(f"  Config label: {config_label}")
    print(f"  Target coefficient: {wind_col}")
    print(f"  Rolling window: {window_years} year(s)")
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")
    print(f"  Maximum breaks to detect: {max_breaks}")
    print(f"  Trimming (endpoints): {trimming*100:.0f}%")
    print(f"  Estimation model: {model_label}")
    if known_break_dates:
        print(f"  Known event dates to test: {known_break_dates}")
    else:
        print(f"  Known event dates: None (using only Bai-Perron detection)")

    # Prepare clean data
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy()
    tmp = tmp.sort_index()

    n_obs = len(tmp)
    print(f"\nData range: {tmp.index.min()} to {tmp.index.max()}")
    print(f"Total observations: {n_obs:,}")

    # Set minimum segment length (default: 10% of data or ~1 year of hourly data)
    if min_segment_length is None:
        min_segment_length = max(int(n_obs * 0.10), 24 * 365)  # At least 1 year
    print(f"Minimum segment length: {min_segment_length:,} observations (~{min_segment_length/(24*365):.1f} years)")

    # Create output directories with structural break subdirectories
    zone_plots_dir = os.path.join(plots_dir, zone, "structural_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)

    zone_results_dir = os.path.join(results_dir, "structural_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    results = {
        'zone': zone,
        'config_label': config_label,
        'window_years': window_years,
        'step_years': step_years,
        'min_obs': min_obs,
        'n_obs': n_obs,
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'detected_breaks': [],
        'chow_tests': [],
        'bic_scores': {}
    }

    # =========================================================================
    # STEP 1: ROLLING WINDOW ESTIMATION (for coefficient time series)
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)
    print(f"Window settings: {window_years} year(s) window, "
          f"{f'{step_months} month(s)' if step_months else f'{step_years} year(s)'} step")
    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )

    if rolling_df.empty:
        print("No valid rolling windows available for break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'estimation_model': estimation_model,
            'model_tag': model_tag,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {len(rolling_df)} rolling window coefficients")
    print(f"Successful windows: {len(rolling_df)}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Coefficient range: [{rolling_df['beta_wind'].min():.4f}, {rolling_df['beta_wind'].max():.4f}]")
    results['rolling_fit_stats'] = {
        'total_candidate_windows': int(total_candidate_windows),
        'successful_windows': int(len(rolling_df)),
        **failure_stats
    }

    # =========================================================================
    # STEP 2: BAI-PERRON CHANGE POINT DETECTION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: BAI-PERRON CHANGE POINT DETECTION")
    print("-"*80)

    # Prepare signal for change point detection
    beta_signal = rolling_df['beta_wind'].values.reshape(-1, 1)

    # Method 1: PELT (Pruned Exact Linear Time) - optimal for multiple breaks
    print("\nMethod 1: PELT algorithm (optimal partitioning)")
    algo_pelt = rpt.Pelt(model="rbf", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    # Use BIC-like penalty: pen = log(n) * dim * sigma^2
    # Adjust penalty to control number of breaks
    sigma = np.std(beta_signal)
    pen = np.log(len(beta_signal)) * sigma**2 * 2  # Moderate penalty

    try:
        breaks_pelt = algo_pelt.predict(pen=pen)
        # Remove the last element (which is always n)
        breaks_pelt = [b for b in breaks_pelt if b < len(beta_signal)]
        print(f"  PELT detected {len(breaks_pelt)} break(s) at indices: {breaks_pelt}")
    except Exception as e:
        print(f"  PELT failed: {e}")
        breaks_pelt = []

    # Method 2: Binary Segmentation (faster, good approximation)
    print("\nMethod 2: Binary Segmentation algorithm")
    algo_binseg = rpt.Binseg(model="l2", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    try:
        breaks_binseg = algo_binseg.predict(n_bkps=max_breaks)
        breaks_binseg = [b for b in breaks_binseg if b < len(beta_signal)]
        print(f"  BinSeg detected {len(breaks_binseg)} break(s) at indices: {breaks_binseg}")
    except Exception as e:
        print(f"  BinSeg failed: {e}")
        breaks_binseg = []

    # Method 3: Dynamic Programming (exact solution)
    print("\nMethod 3: Dynamic Programming (exact, slower)")
    algo_dynp = rpt.Dynp(model="l2", min_size=max(3, len(beta_signal)//20)).fit(beta_signal)

    # Test different numbers of breaks and compute BIC
    print("\n  Testing different numbers of breaks (BIC selection):")
    bic_results = []

    for n_breaks in range(0, max_breaks + 1):
        try:
            if n_breaks == 0:
                # No breaks: cost is total variance
                cost = np.sum((beta_signal - np.mean(beta_signal))**2)
                n_params = 1
            else:
                breaks = algo_dynp.predict(n_bkps=n_breaks)
                breaks = [0] + [b for b in breaks if b < len(beta_signal)] + [len(beta_signal)]

                # Calculate cost (sum of squared residuals within segments)
                cost = 0
                for i in range(len(breaks) - 1):
                    segment = beta_signal[breaks[i]:breaks[i+1]]
                    if len(segment) > 0:
                        cost += np.sum((segment - np.mean(segment))**2)
                n_params = n_breaks + 1  # n_breaks + 1 segment means

            # BIC = n*log(RSS/n) + k*log(n)
            n = len(beta_signal)
            bic = n * np.log(cost / n + 1e-10) + n_params * np.log(n)
            bic_results.append({'n_breaks': n_breaks, 'bic': bic, 'cost': cost})
            print(f"    {n_breaks} breaks: BIC = {bic:.2f}")

        except Exception as e:
            print(f"    {n_breaks} breaks: Failed ({e})")

    # Select optimal number of breaks by BIC
    if bic_results:
        bic_df = pd.DataFrame(bic_results)
        optimal_n_breaks = bic_df.loc[bic_df['bic'].idxmin(), 'n_breaks']
        print(f"\n  Optimal number of breaks (BIC): {int(optimal_n_breaks)}")
        results['bic_scores'] = bic_results

        # Get break points for optimal model
        if optimal_n_breaks > 0:
            optimal_breaks = algo_dynp.predict(n_bkps=int(optimal_n_breaks))
            optimal_breaks = [b for b in optimal_breaks if b < len(beta_signal)]
        else:
            optimal_breaks = []
    else:
        optimal_n_breaks = 0
        optimal_breaks = []

    # Convert break indices to dates
    detected_break_dates = []
    for brk_idx in optimal_breaks:
        if brk_idx < len(rolling_df):
            break_date = rolling_df.iloc[brk_idx]['midpoint']
            detected_break_dates.append(break_date)
            print(f"\n  Break detected at: {break_date.strftime('%Y-%m-%d')}")

    results['detected_breaks'] = detected_break_dates
    results['optimal_n_breaks'] = int(optimal_n_breaks)

    # =========================================================================
    # STEP 3: BREAK TESTS AT DETECTED AND KNOWN BREAK DATES
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: BREAK TESTS FOR STRUCTURAL BREAKS")
    print("-"*80)

    # Combine detected breaks with known event dates
    test_dates = []

    # Add detected break dates
    for bd in detected_break_dates:
        test_dates.append({'date': bd, 'source': 'Bai-Perron detected'})

    # Add known event dates
    if known_break_dates:
        for date_str in known_break_dates:
            test_dates.append({'date': pd.to_datetime(date_str), 'source': 'Known event'})

    # Run break tests
    chow_results = []
    dynamic_lr_test_stats = None
    if estimation_model == 'dynamic_armax':
        chow_results, dynamic_lr_test_stats = _run_dynamic_break_lr_tests(
            tmp=tmp,
            Y_name=Y.name,
            exog_vars=exog_vars,
            wind_col=wind_col,
            test_dates=test_dates,
            dynamic_armax_order=dynamic_armax_order
        )
        results['dynamic_lr_test_stats'] = dynamic_lr_test_stats
        print(f"Dynamic LR test summary: total={dynamic_lr_test_stats['total_test_dates']}, "
              f"skipped_insufficient={dynamic_lr_test_stats['skipped_insufficient_obs']}, "
              f"skipped_fit_fail={dynamic_lr_test_stats['skipped_fit_fail_total']}, "
              f"used_nonconverged={dynamic_lr_test_stats['used_nonconverged_test_dates']}")
    else:
        for test_info in test_dates:
            break_date = test_info['date']
            source = test_info['source']

            print(f"\nTesting break at {break_date.strftime('%Y-%m-%d')} ({source}):")

            pre_break = tmp[tmp.index < break_date]
            post_break = tmp[tmp.index >= break_date]

            if len(pre_break) < 100 or len(post_break) < 100:
                print(f"  Skipped: Insufficient observations (pre={len(pre_break)}, post={len(post_break)})")
                continue

            X_full = sm.add_constant(tmp[exog_vars])
            y_full = tmp[Y.name]
            model_full = sm.OLS(y_full, X_full).fit()
            rss_full = model_full.ssr
            k = len(model_full.params)

            X_pre = sm.add_constant(pre_break[exog_vars])
            y_pre = pre_break[Y.name]
            model_pre = sm.OLS(y_pre, X_pre).fit()
            rss_pre = model_pre.ssr

            X_post = sm.add_constant(post_break[exog_vars])
            y_post = post_break[Y.name]
            model_post = sm.OLS(y_post, X_post).fit()
            rss_post = model_post.ssr

            rss_unrestricted = rss_pre + rss_post
            n = len(tmp)
            f_stat = ((rss_full - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

            beta_pre = model_pre.params[wind_col]
            beta_post = model_post.params[wind_col]
            se_pre = model_pre.bse[wind_col]
            se_post = model_post.bse[wind_col]

            print(f"  Pre-break:  n={len(pre_break):,}, beta_wind={beta_pre:.4f} (SE={se_pre:.4f})")
            print(f"  Post-break: n={len(post_break):,}, beta_wind={beta_post:.4f} (SE={se_post:.4f})")
            print(f"  Change in beta_wind: {beta_post - beta_pre:.4f} ({((beta_post - beta_pre)/abs(beta_pre))*100:.1f}%)")
            print(f"  Chow F-statistic: {f_stat:.2f}")
            print(f"  p-value: {p_value:.4e}")
            print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
            print(f"  Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")

            chow_results.append({
                'break_date': break_date,
                'source': source,
                'n_pre': len(pre_break),
                'n_post': len(post_break),
                'beta_wind_pre': beta_pre,
                'beta_wind_post': beta_post,
                'se_pre': se_pre,
                'se_post': se_post,
                'beta_change': beta_post - beta_pre,
                'beta_change_pct': ((beta_post - beta_pre)/abs(beta_pre))*100,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01,
                'test_type': 'CHOW_OLS'
            })

    results['chow_tests'] = chow_results

    # =========================================================================
    # STEP 4: CUSUM TEST FOR PARAMETER STABILITY
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: CUSUM TEST FOR PARAMETER STABILITY")
    print("-"*80)

    # Run recursive OLS and compute CUSUM
    X_full = sm.add_constant(tmp[exog_vars])
    y_full = tmp[Y.name].values

    # For CUSUM, we need recursive residuals
    # Simplified approach: compute rolling prediction errors
    print("\nComputing CUSUM statistics...")

    # Start recursive estimation after initial window
    init_window = max(len(exog_vars) * 10, 24 * 30)  # At least 1 month
    recursive_residuals = []
    recursive_dates = []

    for t in range(init_window, n_obs, 24 * 7):  # Weekly steps for speed
        # Estimate on data up to t
        X_t = sm.add_constant(tmp[exog_vars].iloc[:t])
        y_t = tmp[Y.name].iloc[:t]
        model_t = sm.OLS(y_t, X_t).fit()

        # One-step-ahead prediction error
        if t < n_obs:
            # Get the next observation's exog values and manually add constant
            # (sm.add_constant behaves inconsistently with single-row DataFrames)
            X_next_raw = tmp[exog_vars].iloc[t].values
            X_next = np.concatenate([[1.0], X_next_raw])  # Prepend constant
            y_next = tmp[Y.name].iloc[t]
            pred = model_t.predict(X_next.reshape(1, -1))[0]
            resid = y_next - pred
            recursive_residuals.append(resid)
            recursive_dates.append(tmp.index[t])

    recursive_residuals = np.array(recursive_residuals)
    sigma_resid = np.std(recursive_residuals)

    # Standardized cumulative sum
    cusum = np.cumsum(recursive_residuals) / (sigma_resid * np.sqrt(len(recursive_residuals)))

    # Critical values (5% significance): ±0.948 * sqrt(n) at endpoints
    # Linear boundaries that start at 0 and reach ±0.948*sqrt(n)
    n_cusum = len(cusum)
    t_values = np.arange(1, n_cusum + 1)
    upper_bound = 0.948 * np.sqrt(n_cusum) * (t_values / n_cusum)
    lower_bound = -upper_bound

    # Check for boundary violations
    violations = (cusum > upper_bound) | (cusum < lower_bound)
    n_violations = np.sum(violations)

    print(f"CUSUM observations: {n_cusum}")
    print(f"Boundary violations: {n_violations} ({100*n_violations/n_cusum:.1f}%)")
    if n_violations > 0:
        first_violation_idx = np.where(violations)[0][0]
        first_violation_date = recursive_dates[first_violation_idx]
        print(f"First violation at: {first_violation_date}")
        results['cusum_first_violation'] = first_violation_date
    else:
        print("No boundary violations detected (parameters appear stable)")
        results['cusum_first_violation'] = None

    results['cusum_violations'] = n_violations

    # =========================================================================
    # STEP 5: GENERATE PLOTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: GENERATING DIAGNOSTIC PLOTS")
    print("-"*80)

    # Plot 1: Coefficient evolution with detected breaks
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel A: Rolling coefficient with breaks
    ax1 = axes[0]
    midpoints = pd.to_datetime(rolling_df['midpoint'])
    beta_values = rolling_df['beta_wind'].values
    se_values = rolling_df['se_wind'].values

    ax1.plot(midpoints, beta_values, color='blue', linewidth=2, label=r'$\beta_{wind}$')
    ax1.fill_between(midpoints, beta_values - 1.96*se_values, beta_values + 1.96*se_values,
                     color='blue', alpha=0.2, label='95% CI')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Mark detected breaks
    for i, break_date in enumerate(detected_break_dates):
        ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                    label='Detected break' if i == 0 else None)

    # Mark known event dates
    if known_break_dates:
        for i, date_str in enumerate(known_break_dates):
            ax1.axvline(x=pd.to_datetime(date_str), color='orange', linestyle=':',
                        linewidth=2, label='Known event' if i == 0 else None)

    ax1.set_title(
        f'Wind Coefficient Evolution with Structural Breaks - {zone}\n(Model: {model_label})',
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Date')
    ax1.set_ylabel(r'$\beta_{wind}$')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel B: BIC by number of breaks
    ax2 = axes[1]
    if results['bic_scores']:
        bic_df = pd.DataFrame(results['bic_scores'])
        ax2.bar(bic_df['n_breaks'], bic_df['bic'], color='steelblue', edgecolor='black')
        ax2.axvline(x=results['optimal_n_breaks'], color='red', linestyle='--',
                    linewidth=2, label=f'Optimal: {results["optimal_n_breaks"]} breaks')
        ax2.set_xlabel('Number of Breaks')
        ax2.set_ylabel('BIC')
        ax2.set_title('Model Selection: BIC by Number of Breaks', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'BIC analysis not available\n(ruptures package not installed)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Model Selection: BIC by Number of Breaks', fontsize=12, fontweight='bold')

    # Panel C: CUSUM plot
    ax3 = axes[2]
    ax3.plot(recursive_dates, cusum, color='blue', linewidth=1.5, label='CUSUM')
    ax3.plot(recursive_dates, upper_bound, color='red', linestyle='--', linewidth=1.5, label='5% bounds')
    ax3.plot(recursive_dates, lower_bound, color='red', linestyle='--', linewidth=1.5)
    ax3.fill_between(recursive_dates, lower_bound, upper_bound, color='red', alpha=0.1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('CUSUM')
    ax3.set_title('CUSUM Test for Parameter Stability', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(zone_plots_dir, f'sb_{config_label}_{model_tag}_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6: SAVING RESULTS")
    print("-"*80)

    # Save rolling coefficients
    rolling_csv = os.path.join(zone_results_dir, f'sb_rolling_coef_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    # Save break-test results
    if chow_results:
        chow_df = pd.DataFrame(chow_results)
        if estimation_model == 'dynamic_armax':
            chow_csv = os.path.join(zone_results_dir, f'sb_dynamic_lr_tests_{config_label}_{model_tag}_{zone}.csv')
        else:
            chow_csv = os.path.join(zone_results_dir, f'sb_chow_tests_{config_label}_{model_tag}_{zone}.csv')
        chow_df.to_csv(chow_csv, index=False)
        print(f"Saved break-test results: {chow_csv}")

    # Save summary
    summary_path = os.path.join(zone_results_dir, f'sb_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"STRUCTURAL BREAK ANALYSIS SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Config label: {config_label}\n")
        f.write(f"  Estimation model: {model_label}\n")
        f.write(f"  Rolling window: {window_years} year(s)\n")
        if step_months:
            f.write(f"  Step size: {step_months} month(s)\n")
        else:
            f.write(f"  Step size: {step_years} year(s)\n")
        f.write(f"  Minimum observations per window: {min_obs:,}\n\n")
        if 'rolling_fit_stats' in results:
            rfs = results['rolling_fit_stats']
            f.write("  Rolling fit diagnostics:\n")
            f.write(f"    Candidate windows: {rfs['total_candidate_windows']}\n")
            f.write(f"    Successful windows: {rfs['successful_windows']}\n")
            f.write(f"    Failed windows: {rfs['failed_windows_total']}\n")
            f.write(f"      Non-converged: {rfs['failed_windows_nonconvergence']}\n")
            f.write(f"      Exceptions: {rfs['failed_windows_exception']}\n")
            f.write(f"    Used non-converged fits: {rfs.get('used_nonconverged_windows', 0)}\n\n")

        f.write(f"Data range: {tmp.index.min()} to {tmp.index.max()}\n")
        f.write(f"Total observations: {n_obs:,}\n\n")

        f.write("-"*80 + "\n")
        f.write("BAI-PERRON BREAK DETECTION\n")
        f.write("-"*80 + "\n")
        if results['optimal_n_breaks'] is not None:
            f.write(f"Optimal number of breaks (BIC): {results['optimal_n_breaks']}\n")
            if detected_break_dates:
                f.write("Detected break dates:\n")
                for bd in detected_break_dates:
                    f.write(f"  - {bd.strftime('%Y-%m-%d')}\n")
            else:
                f.write("No breaks detected.\n")
        else:
            f.write("Bai-Perron analysis not available (ruptures not installed)\n")

        f.write("\n" + "-"*80 + "\n")
        if estimation_model == 'dynamic_armax':
            f.write("ARMAX LR TEST RESULTS\n")
            if dynamic_lr_test_stats is not None:
                f.write(f"  Test dates evaluated: {dynamic_lr_test_stats['total_test_dates']}\n")
                f.write(f"  Skipped (insufficient obs): {dynamic_lr_test_stats['skipped_insufficient_obs']}\n")
                f.write(f"  Skipped (fit failures): {dynamic_lr_test_stats['skipped_fit_fail_total']}\n")
                f.write(f"    Non-converged: {dynamic_lr_test_stats['skipped_fit_fail_nonconvergence']}\n")
                f.write(f"    Exceptions: {dynamic_lr_test_stats['skipped_fit_fail_exception']}\n")
                f.write(f"  Used non-converged test dates: {dynamic_lr_test_stats['used_nonconverged_test_dates']}\n\n")
        else:
            f.write("CHOW TEST RESULTS\n")
        f.write("-"*80 + "\n")
        for cr in chow_results:
            f.write(f"\nBreak date: {cr['break_date'].strftime('%Y-%m-%d')} ({cr['source']})\n")
            f.write(f"  Pre-break beta_wind:  {cr['beta_wind_pre']:.4f} (SE={cr['se_pre']:.4f})\n")
            f.write(f"  Post-break beta_wind: {cr['beta_wind_post']:.4f} (SE={cr['se_post']:.4f})\n")
            f.write(f"  Change: {cr['beta_change']:.4f} ({cr['beta_change_pct']:.1f}%)\n")
            if 'lr_statistic' in cr:
                f.write(f"  LR-statistic: {cr['lr_statistic']:.2f} (df={cr['lr_df']}), p-value: {cr['p_value']:.4e}\n")
            else:
                f.write(f"  F-statistic: {cr['f_statistic']:.2f}, p-value: {cr['p_value']:.4e}\n")
            f.write(f"  Significant at 5%: {'YES' if cr['significant_5pct'] else 'NO'}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("CUSUM TEST\n")
        f.write("-"*80 + "\n")
        f.write(f"Boundary violations: {results['cusum_violations']}\n")
        if results['cusum_first_violation']:
            f.write(f"First violation: {results['cusum_first_violation']}\n")
        else:
            f.write("No violations (parameters appear stable)\n")

    print(f"Saved summary: {summary_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("STRUCTURAL BREAK ANALYSIS COMPLETE")
    print("="*80)

    print(f"\nKey findings for {zone}:")
    if results['optimal_n_breaks'] is not None:
        print(f"  - Detected {results['optimal_n_breaks']} structural break(s) via Bai-Perron")
    if chow_results:
        sig_chow = sum(1 for cr in chow_results if cr['significant_5pct'])
        if estimation_model == 'dynamic_armax':
            print(f"  - {sig_chow}/{len(chow_results)} break dates significant at 5% (ARMAX LR)")
        else:
            print(f"  - {sig_chow}/{len(chow_results)} break dates significant at 5% (Chow test)")
    print(f"  - CUSUM violations: {results['cusum_violations']}")

    return results


def run_trend_break_analysis_legacy(df, zone, Y, exog_vars,
                             max_breaks=5, min_segment_pct=0.10,
                             trimming=0.15,
                             window_years=1, step_years=1/12,
                             min_obs=24*365 - 24*30,
                             config_label=None,
                             plots_dir="plots", results_dir="results", show_progress=True,
                             estimation_model='ols',
                             dynamic_armax_order=(3, 0, 3)):
    """
    Detect structural breaks in the TREND of wind coefficient using SEQUENTIAL TESTING.

    This function tests for changes in the SLOPE of coefficient evolution over time,
    using sequential hypothesis testing (0 vs 1 break, 1 vs 2 breaks, etc.).

    Methodology:
    1. Estimate rolling window coefficients
    2. For m = 0, 1, 2, ..., max_breaks:
       - Find optimal break locations using dynamic programming
       - Calculate BIC and F-statistic for m vs m-1 breaks
    3. Select optimal number of breaks via BIC
    4. Perform sequential F-tests for significance

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - Y: Dependent variable (Series)
    - exog_vars: List of exogenous variable column names
    - max_breaks: Maximum number of trend breaks to test (default: 5)
    - min_segment_pct: Minimum segment length as fraction of total windows (default: 0.10)
    - trimming: Fraction of data to trim from endpoints (default 0.15)
    - window_years: Rolling window size in years (default 1)
    - step_years: Step size between windows in years (default 1/12 = 1 month)
    - min_obs: Minimum observations required per window (default 24*365 - 24*30)
    - config_label: Custom label for this configuration (auto-generated if None)
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving results
    - show_progress: Whether to show progress indicators

    Returns:
    - Dictionary with trend break detection results
    """
    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Generate config label if not provided
    if config_label is None:
        step_months_label = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months_label:
            step_label = f"{step_months_label}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"

        config_label = f"{window_label}_window_{step_label}_step_trend"

    step_months = int(round(step_years * 12)) if step_years < 1 else None

    print("\n" + "="*80)
    print("SEQUENTIAL TREND BREAK ANALYSIS")
    print("="*80)

    # Identify wind column
    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]

    print(f"\nConfiguration:")
    print(f"  Zone: {zone}")
    print(f"  Config label: {config_label}")
    print(f"  Target coefficient: {wind_col}")
    print(f"  Rolling window: {window_years} year(s)")
    if step_months:
        print(f"  Step size: {step_months} month(s)")
    else:
        print(f"  Step size: {step_years} year(s)")
    print(f"  Minimum observations per window: {min_obs:,}")
    print(f"  Maximum breaks to test: {max_breaks}")
    print(f"  Minimum segment: {min_segment_pct*100:.0f}% of windows")
    print(f"  Trimming (endpoints): {trimming*100:.0f}%")
    print(f"  Estimation model: {model_label}")

    # Prepare clean data
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy()
    tmp = tmp.sort_index()

    n_obs = len(tmp)
    print(f"\nData range: {tmp.index.min()} to {tmp.index.max()}")
    print(f"Total observations: {n_obs:,}")

    # Create output directories
    zone_plots_dir = os.path.join(plots_dir, zone, "trend_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)

    zone_results_dir = os.path.join(results_dir, "trend_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: ROLLING WINDOW ESTIMATION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)

    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )
    if rolling_df.empty:
        print("No valid rolling windows available for trend break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'method': 'legacy',
            'model_tag': model_tag,
            'estimation_model': estimation_model,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }
    rolling_df['time_idx'] = np.arange(len(rolling_df))
    n_windows = len(rolling_df)

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {n_windows} rolling window coefficients")
    print(f"Successful windows: {n_windows}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Coefficient range: [{rolling_df['beta_wind'].min():.4f}, {rolling_df['beta_wind'].max():.4f}]")

    # Minimum segment length
    min_segment = max(int(n_windows * min_segment_pct), 5)
    print(f"Minimum segment length: {min_segment} windows")

    # =========================================================================
    # STEP 2: HELPER FUNCTIONS FOR SEGMENTED REGRESSION
    # =========================================================================

    def fit_segment(data):
        """Fit linear trend to a segment, return RSS and model"""
        if len(data) < 3:
            return np.inf, None
        X = sm.add_constant(data['time_idx'])
        y = data['beta_wind']
        model = sm.OLS(y, X).fit()
        return model.ssr, model

    def find_optimal_breaks_dp(n_breaks):
        """
        Find optimal break locations for exactly n_breaks using dynamic programming.
        Returns: (break_indices, total_rss, segment_models)
        """
        if n_breaks == 0:
            rss, model = fit_segment(rolling_df)
            return [], rss, [model]

        # Trim indices
        trim_n = int(n_windows * trimming)

        # Dynamic programming approach
        # Cost[i][k] = minimum RSS for first i observations with k breaks
        # We need to find k break points that partition [0, n) into k+1 segments

        # For simplicity with small max_breaks, use recursive search with memoization
        best_breaks = None
        best_rss = np.inf
        best_models = None

        def get_segment_rss(start, end):
            """Get RSS for segment [start, end)"""
            if end - start < 3:
                return np.inf, None
            segment_data = rolling_df.iloc[start:end]
            return fit_segment(segment_data)

        def search_breaks(remaining_breaks, start_idx, current_breaks):
            """Recursive search for optimal break locations"""
            nonlocal best_breaks, best_rss, best_models

            if remaining_breaks == 0:
                # No more breaks to place, fit final segment
                rss_final, model_final = get_segment_rss(start_idx, n_windows)
                if model_final is None:
                    return

                # Calculate total RSS
                total_rss = 0
                models = []
                prev_idx = 0
                for brk in current_breaks:
                    rss_seg, model_seg = get_segment_rss(prev_idx, brk)
                    if model_seg is None:
                        return
                    total_rss += rss_seg
                    models.append(model_seg)
                    prev_idx = brk
                total_rss += rss_final
                models.append(model_final)

                if total_rss < best_rss:
                    best_rss = total_rss
                    best_breaks = current_breaks.copy()
                    best_models = models
                return

            # Try placing next break
            # Break can be placed from (start_idx + min_segment) to (n_windows - remaining_breaks*min_segment - min_segment)
            earliest = max(start_idx + min_segment, trim_n)
            latest = n_windows - remaining_breaks * min_segment - min_segment
            latest = min(latest, n_windows - trim_n)

            for brk in range(earliest, latest + 1):
                current_breaks.append(brk)
                search_breaks(remaining_breaks - 1, brk, current_breaks)
                current_breaks.pop()

        search_breaks(n_breaks, 0, [])

        return best_breaks if best_breaks else [], best_rss, best_models

    # =========================================================================
    # STEP 3: SEQUENTIAL TESTING
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: SEQUENTIAL TREND BREAK TESTING")
    print("-"*80)

    model_results = []

    for m in range(0, max_breaks + 1):
        print(f"\n--- Testing {m} break(s) ---")

        breaks, rss, models = find_optimal_breaks_dp(m)

        if models is None or rss == np.inf:
            print(f"  Could not fit model with {m} breaks (insufficient segment length)")
            break

        # Calculate BIC
        k = 2 * (m + 1)  # Each segment has intercept + slope
        bic = n_windows * np.log(rss / n_windows) + k * np.log(n_windows)

        # Get break dates
        break_dates = [rolling_df.iloc[b]['midpoint'] for b in breaks] if breaks else []

        # Extract slopes for each segment
        slopes = []
        for i, model in enumerate(models):
            slope = model.params['time_idx']
            se = model.bse['time_idx']
            slopes.append({'slope': slope, 'se': se})

        print(f"  Break locations: {breaks if breaks else 'None'}")
        if break_dates:
            print(f"  Break dates: {[d.strftime('%Y-%m-%d') for d in break_dates]}")
        print(f"  RSS: {rss:.4f}")
        print(f"  BIC: {bic:.2f}")
        slopes_str = [f"{s['slope']:.6f}" for s in slopes]
        print(f"  Segment slopes: {slopes_str}")

        model_results.append({
            'n_breaks': m,
            'breaks': breaks,
            'break_dates': break_dates,
            'rss': rss,
            'bic': bic,
            'models': models,
            'slopes': slopes
        })

    # =========================================================================
    # STEP 4: SEQUENTIAL F-TESTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: SEQUENTIAL F-TESTS (m vs m+1 breaks)")
    print("-"*80)

    f_test_results = []

    for i in range(len(model_results) - 1):
        m0 = model_results[i]
        m1 = model_results[i + 1]

        rss0 = m0['rss']
        rss1 = m1['rss']
        k0 = 2 * (m0['n_breaks'] + 1)
        k1 = 2 * (m1['n_breaks'] + 1)

        # F-statistic: (RSS0 - RSS1) / (k1 - k0) / (RSS1 / (n - k1))
        if rss1 > 0 and (k1 - k0) > 0:
            f_stat = ((rss0 - rss1) / (k1 - k0)) / (rss1 / (n_windows - k1))
            p_value = 1 - stats.f.cdf(f_stat, k1 - k0, n_windows - k1)
        else:
            f_stat = np.nan
            p_value = np.nan

        result = {
            'test': f"{m0['n_breaks']} vs {m1['n_breaks']} breaks",
            'f_stat': f_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05 if not np.isnan(p_value) else False,
            'significant_1pct': p_value < 0.01 if not np.isnan(p_value) else False
        }
        f_test_results.append(result)

        sig_5 = "YES" if result['significant_5pct'] else "NO"
        sig_1 = "YES" if result['significant_1pct'] else "NO"
        print(f"\n  {m0['n_breaks']} vs {m1['n_breaks']} breaks:")
        print(f"    F-statistic: {f_stat:.2f}")
        print(f"    p-value: {p_value:.4e}")
        print(f"    Significant at 5%: {sig_5}")
        print(f"    Significant at 1%: {sig_1}")

    # =========================================================================
    # STEP 5: MODEL SELECTION
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: MODEL SELECTION SUMMARY")
    print("-"*80)

    # BIC-based selection
    bic_values = [m['bic'] for m in model_results]
    optimal_by_bic = np.argmin(bic_values)
    optimal_model = model_results[optimal_by_bic]

    print(f"\n  BIC comparison:")
    for m in model_results:
        marker = " <-- OPTIMAL" if m['n_breaks'] == optimal_model['n_breaks'] else ""
        print(f"    {m['n_breaks']} breaks: BIC = {m['bic']:.2f}{marker}")

    # Sequential testing selection (stop when F-test is not significant)
    optimal_by_seq = 0
    for i, f_result in enumerate(f_test_results):
        if f_result['significant_5pct']:
            optimal_by_seq = i + 1
        else:
            break

    print(f"\n  Optimal by BIC: {optimal_model['n_breaks']} break(s)")
    print(f"  Optimal by sequential F-test (5%): {optimal_by_seq} break(s)")

    # Use BIC-selected model
    selected_model = optimal_model

    print(f"\n  SELECTED MODEL: {selected_model['n_breaks']} break(s)")
    if selected_model['break_dates']:
        print(f"  Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}")

    # Print segment details
    print(f"\n  Segment details:")
    segment_starts = [0] + selected_model['breaks']
    segment_ends = selected_model['breaks'] + [n_windows]

    for i, (start, end, slope_info) in enumerate(zip(segment_starts, segment_ends, selected_model['slopes'])):
        start_date = rolling_df.iloc[start]['midpoint']
        end_date = rolling_df.iloc[min(end-1, n_windows-1)]['midpoint']
        print(f"    Segment {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"      Slope: {slope_info['slope']:.6f} (SE: {slope_info['se']:.6f})")

    # =========================================================================
    # STEP 6: GENERATE PLOTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 5: GENERATING DIAGNOSTIC PLOTS")
    print("-"*80)

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    # Coefficient evolution with segmented trend lines
    midpoints = pd.to_datetime(rolling_df['midpoint'])
    beta_values = rolling_df['beta_wind'].values

    ax1.plot(midpoints, beta_values, 'o', color='steelblue', alpha=0.5, markersize=4, label='Rolling coefficients')

    # Plot fitted trend lines for each segment
    colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
    segment_starts = [0] + selected_model['breaks']
    segment_ends = selected_model['breaks'] + [n_windows]

    for i, (start, end, model) in enumerate(zip(segment_starts, segment_ends, selected_model['models'])):
        segment_data = rolling_df.iloc[start:end]
        X_plot = sm.add_constant(segment_data['time_idx'])
        y_pred = model.predict(X_plot)
        segment_midpoints = pd.to_datetime(segment_data['midpoint'])
        color = colors[i % len(colors)]
        slope = selected_model['slopes'][i]['slope']
        ax1.plot(segment_midpoints, y_pred, '-', color=color, linewidth=2.5,
                 label=f'Segment {i+1} (slope={slope:.4f})')

    # Mark break points
    for i, break_date in enumerate(selected_model['break_dates']):
        ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                    label='Break' if i == 0 else None)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_title(f'Wind Coefficient Evolution with {selected_model["n_breaks"]} Trend Break(s) - {zone}\n'
                  f'(Model: {model_label})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(r'$\beta_{wind}$')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(zone_plots_dir, f'tb_{config_label}_{model_tag}_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 6: SAVING RESULTS")
    print("-"*80)

    # Save rolling coefficients
    rolling_csv = os.path.join(zone_results_dir, f'tb_rolling_coef_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    # Save model comparison
    model_comparison = []
    for m in model_results:
        row = {
            'n_breaks': m['n_breaks'],
            'bic': m['bic'],
            'rss': m['rss'],
            'break_dates': ';'.join([d.strftime('%Y-%m-%d') for d in m['break_dates']]) if m['break_dates'] else '',
            'slopes': ';'.join([f"{s['slope']:.6f}" for s in m['slopes']])
        }
        model_comparison.append(row)

    comparison_df = pd.DataFrame(model_comparison)
    comparison_csv = os.path.join(zone_results_dir, f'tb_model_comparison_{config_label}_{model_tag}_{zone}.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved model comparison: {comparison_csv}")

    # Save F-test results
    ftest_df = pd.DataFrame(f_test_results)
    ftest_csv = os.path.join(zone_results_dir, f'tb_ftest_results_{config_label}_{model_tag}_{zone}.csv')
    ftest_df.to_csv(ftest_csv, index=False)
    print(f"Saved F-test results: {ftest_csv}")

    # Save summary
    summary_path = os.path.join(zone_results_dir, f'tb_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SEQUENTIAL TREND BREAK ANALYSIS SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Config label: {config_label}\n")
        f.write(f"  Estimation model: {model_label}\n")
        f.write(f"  Rolling window: {window_years} year(s)\n")
        if step_months:
            f.write(f"  Step size: {step_months} month(s)\n")
        else:
            f.write(f"  Step size: {step_years} year(s)\n")
        f.write(f"  Minimum observations per window: {min_obs:,}\n")
        f.write(f"  Maximum breaks tested: {max_breaks}\n\n")
        f.write("  Rolling fit diagnostics:\n")
        f.write(f"    Candidate windows: {total_candidate_windows}\n")
        f.write(f"    Successful windows: {n_windows}\n")
        f.write(f"    Failed windows: {failure_stats['failed_windows_total']}\n")
        f.write(f"      Non-converged: {failure_stats['failed_windows_nonconvergence']}\n")
        f.write(f"      Exceptions: {failure_stats['failed_windows_exception']}\n")
        f.write(f"    Used non-converged fits: {failure_stats.get('used_nonconverged_windows', 0)}\n\n")

        f.write(f"Data: {n_windows} rolling windows\n\n")

        f.write("-"*80 + "\n")
        f.write("MODEL COMPARISON (BIC)\n")
        f.write("-"*80 + "\n")
        for m in model_results:
            marker = " <-- SELECTED" if m['n_breaks'] == selected_model['n_breaks'] else ""
            f.write(f"  {m['n_breaks']} breaks: BIC = {m['bic']:.2f}{marker}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("SEQUENTIAL F-TESTS\n")
        f.write("-"*80 + "\n")
        for r in f_test_results:
            f.write(f"  {r['test']}: F = {r['f_stat']:.2f}, p = {r['p_value']:.4e}\n")
            f.write(f"    Significant at 5%: {'YES' if r['significant_5pct'] else 'NO'}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("SELECTED MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of breaks: {selected_model['n_breaks']}\n")
        if selected_model['break_dates']:
            f.write(f"Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}\n")

        f.write("\nSegment details:\n")
        for i, slope_info in enumerate(selected_model['slopes']):
            f.write(f"  Segment {i+1}: slope = {slope_info['slope']:.6f} (SE: {slope_info['se']:.6f})\n")

    print(f"Saved summary: {summary_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SEQUENTIAL TREND BREAK ANALYSIS COMPLETE")
    print("="*80)

    print(f"\nKey findings for {zone}:")
    print(f"  - Optimal breaks (BIC): {selected_model['n_breaks']}")
    if selected_model['break_dates']:
        print(f"  - Break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}")
    slopes_final = [f"{s['slope']:.6f}" for s in selected_model['slopes']]
    print(f"  - Segment slopes: {slopes_final}")

    # Return results
    results = {
        'zone': zone,
        'config_label': config_label,
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'rolling_fit_stats': {
            'total_candidate_windows': int(total_candidate_windows),
            'successful_windows': int(n_windows),
            **failure_stats
        },
        'n_windows': n_windows,
        'model_results': model_results,
        'f_test_results': f_test_results,
        'selected_model': selected_model,
        'optimal_by_bic': optimal_by_bic,
        'optimal_by_seq': optimal_by_seq,
        'rolling_df': rolling_df
    }

    return results


def run_trend_break_analysis_bp_supf(df, zone, Y, exog_vars,
                                     max_breaks=5, min_segment_pct=0.10,
                                     trimming=0.15,
                                     window_years=1, step_years=1/12,
                                     min_obs=24*365 - 24*30,
                                     config_label=None,
                                     plots_dir="plots", results_dir="results",
                                     bp_inference_mode='both',
                                     bp_significance_level=0.05,
                                     bp_bootstrap_reps=999,
                                     bp_bootstrap_block_length=8,
                                     bp_random_seed=42,
                                     estimation_model='ols',
                                     dynamic_armax_order=(3, 0, 3)):
    """
    Bai-Perron style sequential supF trend-break analysis.

    This applies supF(l+1|l) testing to the rolling wind-coefficient trend series,
    with optional tabulated critical values and bootstrap inference.
    """
    model_tag = _get_break_model_tag(estimation_model, dynamic_armax_order)
    model_label = _get_break_model_label(estimation_model, dynamic_armax_order)

    # Bai-Perron (2003) Table 2b: Asymptotic critical values for sequential test
    # F_T(l+1|l), epsilon = 0.10. Index: q -> alpha -> [ell=0..9].
    BP_TABLE2B_EPS_010 = {
        2: {
            0.10: [10.37, 13.20, 13.79, 14.37, 14.66, 15.07, 15.42, 15.81, 16.09, 16.09],
            0.05: [12.25, 13.83, 14.73, 15.46, 16.13, 16.55, 16.82, 17.07, 17.34, 17.58],
            0.025: [13.86, 15.51, 16.55, 17.07, 17.58, 17.98, 18.19, 18.55, 18.92, 19.02],
            0.01: [16.19, 17.88, 18.31, 18.96, 19.63, 20.09, 20.30, 20.87, 20.97, 21.13]
        }
    }

    if config_label is None:
        step_months_label = int(round(step_years * 12)) if step_years < 1 else None
        if window_years == int(window_years):
            window_label = f"{int(window_years)}y"
        else:
            window_label = f"{window_years:.1f}y"
        if step_months_label:
            step_label = f"{step_months_label}m"
        else:
            step_label = f"{int(step_years)}y" if step_years == int(step_years) else f"{step_years:.1f}y"
        config_label = f"{window_label}_window_{step_label}_step_trend_bp"

    step_months = int(round(step_years * 12)) if step_years < 1 else None
    rng = np.random.default_rng(bp_random_seed)

    print("\n" + "="*80)
    print("BAI-PERRON STYLE SEQUENTIAL SUPF TREND BREAK ANALYSIS")
    print("="*80)
    print(f"Inference mode: {bp_inference_mode}")
    print(f"Significance level: {bp_significance_level}")
    print(f"Bootstrap reps: {bp_bootstrap_reps}")
    print(f"Bootstrap block length: {bp_bootstrap_block_length}")
    print(f"Estimation model: {model_label}")

    wind_col = [col for col in exog_vars if 'Wind' in col and 'Forecast' in col][0]
    cols_needed = [Y.name] + exog_vars
    tmp = df[cols_needed].dropna().copy().sort_index()
    n_obs = len(tmp)

    zone_plots_dir = os.path.join(plots_dir, zone, "trend_break_analysis", model_tag)
    os.makedirs(zone_plots_dir, exist_ok=True)
    zone_results_dir = os.path.join(results_dir, "trend_break_analysis", model_tag)
    os.makedirs(zone_results_dir, exist_ok=True)

    print("\n" + "-"*80)
    print("STEP 1: ESTIMATING ROLLING WINDOW COEFFICIENTS")
    print("-"*80)
    rolling_df, total_candidate_windows, failure_stats = _estimate_rolling_wind_coefficients(
        tmp=tmp,
        Y_name=Y.name,
        exog_vars=exog_vars,
        wind_col=wind_col,
        window_years=window_years,
        step_years=step_years,
        min_obs=min_obs,
        estimation_model=estimation_model,
        dynamic_armax_order=dynamic_armax_order
    )
    if rolling_df.empty:
        print("No rolling windows available for trend break analysis.")
        return {
            'zone': zone,
            'config_label': config_label,
            'method': 'bp_supf',
            'model_tag': model_tag,
            'estimation_model': estimation_model,
            'failure_stats': failure_stats,
            'error': 'no_windows'
        }

    rolling_df['time_idx'] = np.arange(len(rolling_df))
    y_arr = rolling_df['beta_wind'].to_numpy(dtype=float)
    time_idx = rolling_df['time_idx'].to_numpy(dtype=float)
    n = len(y_arr)
    min_segment = max(int(n * min_segment_pct), 5)
    trim_n = int(n * trimming)
    q = 2  # Intercept + slope restrictions for an added break
    max_l_table = 9  # Table 2b provides ell columns 0..9

    if bp_inference_mode in ('tables', 'both') and abs(float(trimming) - 0.10) > 1e-9:
        raise ValueError(
            f"BP table mode requires trimming=0.10 (epsilon=0.10), got {trimming}. "
            "Set STRUCTURAL_BREAK_TRIMMING=0.10 or use bp_inference_mode='bootstrap'."
        )

    if bp_significance_level not in (0.10, 0.05, 0.025, 0.01):
        raise ValueError(
            f"Unsupported bp_significance_level={bp_significance_level}. "
            "Choose one of: 0.10, 0.05, 0.025, 0.01"
        )

    failed_windows = int(failure_stats['failed_windows_total'])
    print(f"Estimated {n} rolling window coefficients")
    print(f"Successful windows: {n}/{total_candidate_windows}")
    print(f"Failed window fits: {failed_windows}")
    if estimation_model == 'dynamic_armax':
        print(f"Used non-converged window fits: {failure_stats['used_nonconverged_windows']}")
    if failed_windows > 0:
        print(f"  - Non-converged: {failure_stats['failed_windows_nonconvergence']}")
        print(f"  - Exceptions:    {failure_stats['failed_windows_exception']}")
    print(f"Minimum segment length: {min_segment}")

    def fit_segment_range(y_values, start, end):
        if end - start < 3:
            return np.inf, None, None, None
        x_seg = time_idx[start:end]
        X = np.column_stack([np.ones(end - start), x_seg])
        y_seg = y_values[start:end]
        beta, _, _, _ = np.linalg.lstsq(X, y_seg, rcond=None)
        fitted = X @ beta
        resid = y_seg - fitted
        rss = float(np.sum(resid ** 2))
        return rss, beta, fitted, resid

    def evaluate_break_list(y_values, breaks):
        starts = [0] + list(breaks)
        ends = list(breaks) + [n]
        total_rss = 0.0
        models = []
        for s, e in zip(starts, ends):
            rss, beta, fitted, resid = fit_segment_range(y_values, s, e)
            if beta is None:
                return np.inf, None
            total_rss += rss
            models.append({'start': s, 'end': e, 'beta': beta, 'rss': rss, 'fitted': fitted, 'resid': resid})
        return total_rss, models

    def find_optimal_breaks_dp(y_values, n_breaks):
        if n_breaks == 0:
            rss, models = evaluate_break_list(y_values, [])
            return [], rss, models

        best_breaks = None
        best_rss = np.inf
        best_models = None

        def search(remaining_breaks, start_idx, current_breaks):
            nonlocal best_breaks, best_rss, best_models
            if remaining_breaks == 0:
                rss, models = evaluate_break_list(y_values, current_breaks)
                if models is not None and rss < best_rss:
                    best_rss = rss
                    best_breaks = current_breaks.copy()
                    best_models = models
                return

            earliest = max(start_idx + min_segment, trim_n)
            latest = n - remaining_breaks * min_segment - min_segment
            latest = min(latest, n - trim_n)
            if earliest > latest:
                return

            for brk in range(earliest, latest + 1):
                current_breaks.append(brk)
                search(remaining_breaks - 1, brk, current_breaks)
                current_breaks.pop()

        search(n_breaks, 0, [])
        return best_breaks if best_breaks else [], best_rss, best_models

    def supf_lplus1_given_l(y_values, breaks_l, rss_l):
        starts = [0] + list(breaks_l)
        ends = list(breaks_l) + [n]
        k_u = 2 * (len(breaks_l) + 2)
        best_f = -np.inf
        best_break = None
        best_rss_u = None

        for s, e in zip(starts, ends):
            c_start = s + min_segment
            c_end = e - min_segment
            if c_start > c_end:
                continue
            for c in range(c_start, c_end + 1):
                candidate_breaks = sorted(list(breaks_l) + [c])
                rss_u, _ = evaluate_break_list(y_values, candidate_breaks)
                if np.isinf(rss_u) or (n - k_u) <= 0 or rss_u <= 0:
                    continue
                f_stat = ((rss_l - rss_u) / q) / (rss_u / (n - k_u))
                if f_stat > best_f:
                    best_f = f_stat
                    best_break = c
                    best_rss_u = rss_u

        if best_break is None:
            return np.nan, None, np.nan
        return float(best_f), int(best_break), float(best_rss_u)

    def moving_block_sample(resid_values, block_len):
        T = len(resid_values)
        if T <= 1:
            return resid_values.copy()
        block_len = max(2, min(block_len, T))
        starts = np.arange(0, T - block_len + 1)
        draws = []
        while len(draws) < T:
            st = rng.choice(starts)
            draws.extend(resid_values[st:st + block_len].tolist())
        return np.array(draws[:T], dtype=float)

    print("\n" + "-"*80)
    print("STEP 2: MODEL GRID (0..M breaks)")
    print("-"*80)
    model_results = []
    for m in range(0, max_breaks + 1):
        brks, rss, models = find_optimal_breaks_dp(y_arr, m)
        if models is None or np.isinf(rss):
            break
        k = 2 * (m + 1)
        bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)
        break_dates = [rolling_df.iloc[b]['midpoint'] for b in brks] if brks else []
        slopes = [{'slope': mdl['beta'][1], 'se': np.nan} for mdl in models]
        model_results.append({
            'n_breaks': m,
            'breaks': brks,
            'break_dates': break_dates,
            'rss': rss,
            'bic': bic,
            'models': models,
            'slopes': slopes
        })
        print(f"{m} breaks: RSS={rss:.4f}, BIC={bic:.2f}")

    print("\n" + "-"*80)
    print("STEP 3: SEQUENTIAL SUPF(l+1|l)")
    print("-"*80)
    supf_results = []
    optimal_by_supf = 0

    for l in range(0, min(max_breaks, len(model_results) - 1)):
        if bp_inference_mode in ('tables', 'both') and l > max_l_table:
            print(f"Stopping sequential test at l={l-1}: Table 2b supports ell up to {max_l_table}.")
            break

        model_l = model_results[l]
        rss_l = model_l['rss']
        breaks_l = model_l['breaks']
        supf_obs, best_split, rss_u = supf_lplus1_given_l(y_arr, breaks_l, rss_l)

        cv_table = np.nan
        p_boot = np.nan
        cv_boot = np.nan

        # Table-based critical value (Bai-Perron Table 2b, epsilon=0.10)
        if q in BP_TABLE2B_EPS_010 and bp_significance_level in BP_TABLE2B_EPS_010[q]:
            cv_table = BP_TABLE2B_EPS_010[q][bp_significance_level][l]

        if bp_inference_mode in ('bootstrap', 'both'):
            boot_stats = []
            fitted_l = np.empty(n)
            resid_l = np.empty(n)
            for mdl in model_l['models']:
                s, e = mdl['start'], mdl['end']
                fitted_l[s:e] = mdl['fitted']
                resid_l[s:e] = mdl['resid']
            resid_l = resid_l - np.mean(resid_l)

            for _ in range(bp_bootstrap_reps):
                e_star = moving_block_sample(resid_l, bp_bootstrap_block_length)
                y_star = fitted_l + e_star
                rss_l_star, _ = evaluate_break_list(y_star, breaks_l)
                supf_star, _, _ = supf_lplus1_given_l(y_star, breaks_l, rss_l_star)
                if not np.isnan(supf_star):
                    boot_stats.append(supf_star)

            if boot_stats:
                boot_stats = np.array(boot_stats, dtype=float)
                cv_boot = float(np.quantile(boot_stats, 1 - bp_significance_level))
                p_boot = float(np.mean(boot_stats >= supf_obs))

        # Decision rule
        reject_5 = False
        decision_source = 'none'
        if bp_inference_mode == 'tables' and not np.isnan(cv_table):
            reject_5 = supf_obs > cv_table
            decision_source = 'table'
        elif bp_inference_mode == 'tables' and np.isnan(cv_table):
            raise ValueError(
                f"No BP Table 2b critical value for q={q}, alpha={bp_significance_level}, l={l}."
            )
        elif bp_inference_mode == 'bootstrap':
            reject_5 = (p_boot < bp_significance_level) if not np.isnan(p_boot) else False
            decision_source = 'bootstrap'
        else:  # both
            # In 'both', table is primary for decision; bootstrap is reported as robustness.
            if not np.isnan(cv_table):
                reject_5 = supf_obs > cv_table
                decision_source = 'table_primary'
            elif not np.isnan(p_boot):
                reject_5 = p_boot < bp_significance_level
                decision_source = 'bootstrap_fallback'

        supf_results.append({
            'l': l,
            'test': f"supF({l+1}|{l})",
            'supf_stat': supf_obs,
            'best_split_idx': best_split,
            'best_split_date': rolling_df.iloc[best_split]['midpoint'] if best_split is not None else pd.NaT,
            'rss_l': rss_l,
            'rss_l_plus_1_best': rss_u,
            'cv_table': cv_table,
            'cv_boot': cv_boot,
            'p_boot': p_boot,
            'alpha': bp_significance_level,
            'reject_alpha': bool(reject_5),
            'reject_5pct': bool(reject_5),
            'decision_source': decision_source
        })

        print(f"supF({l+1}|{l})={supf_obs:.3f}, reject@{int((1-bp_significance_level)*100)}%={reject_5} ({decision_source})")
        if reject_5:
            optimal_by_supf = l + 1
        else:
            break

    optimal_by_bic = int(np.argmin([m['bic'] for m in model_results])) if model_results else 0
    selected_model = model_results[optimal_by_supf] if model_results else None

    # Plot selected model
    if selected_model is not None:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        midpoints = pd.to_datetime(rolling_df['midpoint'])
        beta_values = rolling_df['beta_wind'].values
        ax1.plot(midpoints, beta_values, 'o', color='steelblue', alpha=0.5, markersize=4, label='Rolling coefficients')

        colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
        segment_starts = [0] + selected_model['breaks']
        segment_ends = selected_model['breaks'] + [n]
        for i, (s, e, mdl) in enumerate(zip(segment_starts, segment_ends, selected_model['models'])):
            x_seg = time_idx[s:e]
            y_pred = mdl['beta'][0] + mdl['beta'][1] * x_seg
            ax1.plot(midpoints[s:e], y_pred, '-', color=colors[i % len(colors)], linewidth=2.5,
                     label=f"Segment {i+1} (slope={mdl['beta'][1]:.4f})")
        for i, break_date in enumerate(selected_model['break_dates']):
            ax1.axvline(x=break_date, color='red', linestyle='--', linewidth=2,
                        label='Break' if i == 0 else None)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_title(
            f'BP-supF Trend Breaks ({selected_model["n_breaks"]}) - {zone}\n(Model: {model_label})',
            fontsize=12, fontweight='bold'
        )
        ax1.set_xlabel('Date')
        ax1.set_ylabel(r'$\beta_{wind}$')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(zone_plots_dir, f'tb_bp_{config_label}_{model_tag}_{zone}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    # Save outputs
    rolling_csv = os.path.join(zone_results_dir, f'tb_rolling_coef_bp_{config_label}_{model_tag}_{zone}.csv')
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling coefficients: {rolling_csv}")

    model_comp_rows = []
    for m in model_results:
        model_comp_rows.append({
            'n_breaks': m['n_breaks'],
            'bic': m['bic'],
            'rss': m['rss'],
            'break_dates': ';'.join([d.strftime('%Y-%m-%d') for d in m['break_dates']]) if m['break_dates'] else ''
        })
    model_comp_df = pd.DataFrame(model_comp_rows)
    model_comp_csv = os.path.join(zone_results_dir, f'tb_bp_model_comparison_{config_label}_{model_tag}_{zone}.csv')
    model_comp_df.to_csv(model_comp_csv, index=False)
    print(f"Saved model comparison: {model_comp_csv}")

    supf_df = pd.DataFrame(supf_results)
    supf_csv = os.path.join(zone_results_dir, f'tb_bp_supf_seq_{config_label}_{model_tag}_{zone}.csv')
    supf_df.to_csv(supf_csv, index=False)
    print(f"Saved supF results: {supf_csv}")

    summary_path = os.path.join(zone_results_dir, f'tb_bp_summary_{config_label}_{model_tag}_{zone}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BP STYLE SEQUENTIAL SUPF TREND BREAK SUMMARY - {zone}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Config label: {config_label}\n")
        f.write(f"Estimation model: {model_label}\n")
        f.write(f"Inference mode: {bp_inference_mode}\n")
        f.write(f"Epsilon (trimming): {trimming}\n")
        f.write(f"q restrictions per added break: {q}\n")
        f.write(f"Significance level: {bp_significance_level}\n")
        f.write(f"Bootstrap reps: {bp_bootstrap_reps}\n")
        f.write(f"Bootstrap block length: {bp_bootstrap_block_length}\n")
        f.write(f"Random seed: {bp_random_seed}\n\n")
        f.write(f"Data windows: {n}\n")
        f.write("Rolling fit diagnostics:\n")
        f.write(f"  Candidate windows: {total_candidate_windows}\n")
        f.write(f"  Successful windows: {n}\n")
        f.write(f"  Failed windows: {failure_stats['failed_windows_total']}\n")
        f.write(f"    Non-converged: {failure_stats['failed_windows_nonconvergence']}\n")
        f.write(f"    Exceptions: {failure_stats['failed_windows_exception']}\n")
        f.write(f"  Used non-converged fits: {failure_stats.get('used_nonconverged_windows', 0)}\n")
        f.write(f"Optimal breaks by supF: {optimal_by_supf}\n")
        f.write(f"Optimal breaks by BIC: {optimal_by_bic}\n")
        if selected_model and selected_model['break_dates']:
            f.write(f"Selected break dates: {[d.strftime('%Y-%m-%d') for d in selected_model['break_dates']]}\n")
    print(f"Saved summary: {summary_path}")

    return {
        'zone': zone,
        'config_label': config_label,
        'method': 'bp_supf',
        'estimation_model': estimation_model,
        'model_tag': model_tag,
        'rolling_fit_stats': {
            'total_candidate_windows': int(total_candidate_windows),
            'successful_windows': int(n),
            **failure_stats
        },
        'rolling_df': rolling_df,
        'model_results': model_results,
        'supf_results': supf_results,
        'selected_model': selected_model,
        'optimal_by_supf': optimal_by_supf,
        'optimal_by_bic': optimal_by_bic
    }


def run_trend_break_analysis(df, zone, Y, exog_vars,
                             max_breaks=5, min_segment_pct=0.10,
                             trimming=0.15,
                             window_years=1, step_years=1/12,
                             min_obs=24*365 - 24*30,
                             config_label=None,
                             plots_dir="plots", results_dir="results", show_progress=True,
                             trend_break_test_method='legacy',
                             bp_inference_mode='both',
                             bp_significance_level=0.05,
                             bp_bootstrap_reps=999,
                             bp_bootstrap_block_length=8,
                             bp_random_seed=42,
                             bp_use_hac_se=True,
                             estimation_model='ols',
                             dynamic_armax_order=(3, 0, 3)):
    """
    Dispatcher for trend-break analysis.
    - legacy: existing BIC + sequential F implementation
    - bp_supf: Bai-Perron style sequential supF implementation
    """
    if trend_break_test_method == 'legacy':
        return run_trend_break_analysis_legacy(
            df, zone, Y, exog_vars,
            max_breaks=max_breaks,
            min_segment_pct=min_segment_pct,
            trimming=trimming,
            window_years=window_years,
            step_years=step_years,
            min_obs=min_obs,
            config_label=config_label,
            plots_dir=plots_dir,
            results_dir=results_dir,
            show_progress=show_progress,
            estimation_model=estimation_model,
            dynamic_armax_order=dynamic_armax_order
        )

    if trend_break_test_method == 'bp_supf':
        return run_trend_break_analysis_bp_supf(
            df, zone, Y, exog_vars,
            max_breaks=max_breaks,
            min_segment_pct=min_segment_pct,
            trimming=trimming,
            window_years=window_years,
            step_years=step_years,
            min_obs=min_obs,
            config_label=config_label,
            plots_dir=plots_dir,
            results_dir=results_dir,
            bp_inference_mode=bp_inference_mode,
            bp_significance_level=bp_significance_level,
            bp_bootstrap_reps=bp_bootstrap_reps,
            bp_bootstrap_block_length=bp_bootstrap_block_length,
            bp_random_seed=bp_random_seed,
            estimation_model=estimation_model,
            dynamic_armax_order=dynamic_armax_order
        )

    raise ValueError(
        f"Unknown trend_break_test_method='{trend_break_test_method}'. "
        "Choose 'legacy' or 'bp_supf'."
    )


def run_quantile_regression_analysis(df, zone,
                                     plots_dir="plots", results_dir="results"):
    """
    Estimate wind coefficient across quantiles of the price distribution.

    Uses logged variables (NOT deseasonalized) with calendar dummies included directly
    in the regression to control for seasonality (FULL basis: Year+Month+DOW+Hour+Holiday).

    Note: Always uses logged variables.

    Parameters:
    - df: DataFrame with all variables
    - zone: Price zone identifier
    - plots_dir: Directory for saving plots
    - results_dir: Directory for saving CSV results

    Returns: None (saves outputs to files)
    """
    # Hardcoded quantiles and seasonality settings (not user-configurable)
    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    print("\n" + "="*80)
    print("QUANTILE REGRESSION ANALYSIS")
    print("="*80)

    # --- Step 1: Determine dependent variable (logged, NOT deseasonalized) ---
    y_col = 'Price_Log'

    print(f"\nDependent variable: {y_col}")
    print("  (Using logged price, NOT deseasonalized - seasonality handled via dummies)")

    # --- Step 2: Determine economic regressors (logged, NOT deseasonalized) ---
    econ_vars = [
        'Wind_Forecast_Log',
        'Consumption_Log',
        'Hydro_Reserves_Log',
        'Net_Exchange',  # NOT logged
        'Oil_Price_Log',
        'Gas_Price_Log'
    ]

    print(f"\nEconomic regressors: {econ_vars}")

    # --- Step 3: Build calendar/seasonal dummies (FULL basis) ---
    print("\nBuilding seasonality controls (FULL basis: Year+Month+DOW+Hour+Holiday)...")

    # Create a working copy to avoid modifying original
    tmp = df.copy()

    # Extract time components from datetime index
    tmp['Year'] = tmp.index.year
    tmp['Month'] = tmp.index.month
    tmp['DayOfWeek'] = tmp.index.dayofweek  # 0=Monday, 6=Sunday
    tmp['Hour'] = tmp.index.hour

    # Create holiday indicator for Swedish holidays
    try:
        import holidays
        swedish_holidays = holidays.Sweden(years=range(tmp.index.year.min(), tmp.index.year.max() + 1))
        tmp['Holiday'] = tmp.index.to_series().apply(lambda x: 1 if x.date() in swedish_holidays else 0).values
        print("  Holiday dummies created using Swedish holiday calendar")
    except ImportError:
        tmp['Holiday'] = 0
        print("  WARNING: 'holidays' package not installed; Holiday set to 0 (no crash)")

    # Create dummy variables with drop_first=True to avoid multicollinearity
    year_dummies = pd.get_dummies(tmp['Year'], prefix='Year', drop_first=True)
    month_dummies = pd.get_dummies(tmp['Month'], prefix='Month', drop_first=True)
    dow_dummies = pd.get_dummies(tmp['DayOfWeek'], prefix='DOW', drop_first=True)
    hour_dummies = pd.get_dummies(tmp['Hour'], prefix='Hour', drop_first=True)

    print(f"  Year dummies: {len(year_dummies.columns)} columns")
    print(f"  Month dummies: {len(month_dummies.columns)} columns")
    print(f"  DOW dummies: {len(dow_dummies.columns)} columns")
    print(f"  Hour dummies: {len(hour_dummies.columns)} columns")
    print(f"  Holiday: 1 column (binary indicator)")

    # --- Step 4: Assemble data matrix ---
    # Combine all regressors
    cols_needed = [y_col] + econ_vars
    data_subset = tmp[cols_needed].copy()

    # Add seasonal dummies
    data_subset = pd.concat([data_subset, year_dummies, month_dummies, dow_dummies, hour_dummies], axis=1)
    data_subset['Holiday'] = tmp['Holiday'].values

    # Drop rows with NA and sort by index
    data_subset = data_subset.dropna()
    data_subset = data_subset.sort_index()

    print(f"\nData range: {data_subset.index.min()} to {data_subset.index.max()}")
    print(f"Observations after cleaning: {len(data_subset):,}")

    # Build y and X
    y = data_subset[y_col].astype(float)

    # X includes: constant + economic vars + seasonal dummies + holiday
    seasonal_cols = list(year_dummies.columns) + list(month_dummies.columns) + \
                    list(dow_dummies.columns) + list(hour_dummies.columns) + ['Holiday']
    X_cols = econ_vars + seasonal_cols

    # Ensure all columns are numeric (convert to float64)
    X_data = data_subset[X_cols].astype(float)
    X = sm.add_constant(X_data)

    print(f"\nTotal regressors (incl. const): {X.shape[1]}")
    print(f"  Economic controls: {len(econ_vars)}")
    print(f"  Seasonal controls: {len(seasonal_cols)}")

    # --- Step 5: Run quantile regressions ---
    print(f"\n--- Estimating Quantile Regressions ---")
    print(f"Quantiles: {QUANTILES}")
    # TODO: Block bootstrap can be added later for time-series-robust inference

    results = []
    print(f"\nEstimating quantile regressions for {len(QUANTILES)} quantiles...\n")

    for idx, q in enumerate(QUANTILES, 1):
        print(f"[{idx}/{len(QUANTILES)}] Quantile q={q:.2f}... ", end="")

        model = sm.QuantReg(y, X)
        res = model.fit(q=q)

        # Extract coefficients for key variables
        result_row = {
            'quantile': q,
            'beta_wind': res.params[wind_col],
            'se_wind': res.bse[wind_col] if wind_col in res.bse.index else np.nan,
            'p_wind': res.pvalues[wind_col] if wind_col in res.pvalues.index else np.nan,
            'beta_demand': res.params[demand_col],
            'se_demand': res.bse[demand_col] if demand_col in res.bse.index else np.nan,
            'p_demand': res.pvalues[demand_col] if demand_col in res.pvalues.index else np.nan,
            'beta_hydro': res.params[hydro_col],
            'se_hydro': res.bse[hydro_col] if hydro_col in res.bse.index else np.nan,
            'p_hydro': res.pvalues[hydro_col] if hydro_col in res.pvalues.index else np.nan,
            'n_obs': int(res.nobs)
        }

        # Add oil/gas if available
        if oil_col and oil_col in res.params.index:
            result_row['beta_oil'] = res.params[oil_col]
            result_row['se_oil'] = res.bse[oil_col] if oil_col in res.bse.index else np.nan
            result_row['p_oil'] = res.pvalues[oil_col] if oil_col in res.pvalues.index else np.nan

        if gas_col and gas_col in res.params.index:
            result_row['beta_gas'] = res.params[gas_col]
            result_row['se_gas'] = res.bse[gas_col] if gas_col in res.bse.index else np.nan
            result_row['p_gas'] = res.pvalues[gas_col] if gas_col in res.pvalues.index else np.nan

        results.append(result_row)
        print(f"β_wind={res.params[wind_col]:.4f}, p={res.pvalues[wind_col] if wind_col in res.pvalues.index else np.nan:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # --- Step 6: Print summary ---
    print("\n" + "="*80)
    print("QUANTILE REGRESSION RESULTS SUMMARY")
    print("="*80)
    print(f"\nDependent variable: {y_col}")
    print(f"Seasonality basis: FULL (Year+Month+DOW+Hour+Holiday)")
    print(f"Observations: {results_df['n_obs'].iloc[0]:,}")

    print(f"\nWind coefficient by quantile:")
    print(f"{'Quantile':<10} {'Beta':<12} {'SE':<12} {'p-value':<10}")
    print("-" * 44)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_wind'] < 0.01 else "**" if row['p_wind'] < 0.05 else "*" if row['p_wind'] < 0.1 else ""
        print(f"{row['quantile']:<10.2f} {row['beta_wind']:<12.6f} {row['se_wind']:<12.6f} {row['p_wind']:<10.4f} {sig}")

    print(f"\nDemand coefficient by quantile:")
    print(f"{'Quantile':<10} {'Beta':<12} {'SE':<12} {'p-value':<10}")
    print("-" * 44)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_demand'] < 0.01 else "**" if row['p_demand'] < 0.05 else "*" if row['p_demand'] < 0.1 else ""
        print(f"{row['quantile']:<10.2f} {row['beta_demand']:<12.6f} {row['se_demand']:<12.6f} {row['p_demand']:<10.4f} {sig}")

    # --- Step 7: Save CSV output ---
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'quantreg_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    # --- Step 8: Create plots ---
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Wind coefficient across quantiles
    fig, ax = plt.subplots(figsize=(10, 6))

    quantiles = results_df['quantile'].values
    beta_wind = results_df['beta_wind'].values
    se_wind = results_df['se_wind'].values

    # 95% CI
    upper_95 = beta_wind + 1.96 * se_wind
    lower_95 = beta_wind - 1.96 * se_wind

    ax.plot(quantiles, beta_wind, 'o-', linewidth=2, markersize=8, label=r'$\beta_{wind}$')
    ax.fill_between(quantiles, lower_95, upper_95, alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel(r'$\beta_{wind}$ (Wind Coefficient)', fontsize=12)
    ax.set_title(f'Quantile Regression: Wind Coefficient - {zone}\n(FULL seasonality: Year+Month+DOW+Hour+Holiday)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(quantiles)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'quantreg_beta_wind_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: Demand coefficient across quantiles
    fig, ax = plt.subplots(figsize=(10, 6))

    beta_demand = results_df['beta_demand'].values
    se_demand = results_df['se_demand'].values

    upper_95 = beta_demand + 1.96 * se_demand
    lower_95 = beta_demand - 1.96 * se_demand

    ax.plot(quantiles, beta_demand, 'o-', linewidth=2, markersize=8, label=r'$\beta_{demand}$')
    ax.fill_between(quantiles, lower_95, upper_95, alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel(r'$\beta_{demand}$ (Demand Coefficient)', fontsize=12)
    ax.set_title(f'Quantile Regression: Demand Coefficient - {zone}\n(FULL seasonality: Year+Month+DOW+Hour+Holiday)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(quantiles)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'quantreg_beta_demand_{zone}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("QUANTILE REGRESSION ANALYSIS COMPLETE")
    print("="*80)


