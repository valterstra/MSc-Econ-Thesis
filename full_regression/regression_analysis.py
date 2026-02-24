"""
################################################################################
#  [Module 07/10]  regression_analysis.py  –  Main Regression Orchestrator
#
#  Contains:
#    1. perform_multivariate_analysis()  : runs OLS → optional ARMAX → optional GARCH-X
#       Parameters control which analyses run:
#         run_stationarity / run_ljungbox / run_hetero_tests  : diagnostic toggles
#         optimize_armax_lags / use_checkpointed_lag_selection : ARMAX lag selection
#         run_tvp_wind_kalman                                  : early return to TVP Kalman
#         run_rolling_window                                   : early return to rolling window
#         run_quantile_regression                              : early return to quantile reg
#         run_structural_break / structural_break_type         : early return to break analysis
#
#  All functions are imported from other modules — this file is the coordinator.
#
#  Dependencies: config, diagnostics, utils, regression_models, structural_analysis
################################################################################
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

from .config import (
    ARMAX_ALLOW_NONCONVERGED, ARMAX_MAXITER, ARMAX_SOLVER,
    ARMAX_USE_WARM_START, ARMAX_ENABLE_FALLBACK_ORDERS, ARMAX_FALLBACK_ORDERS
)
from .diagnostics import run_ljungbox_test, run_heteroskedasticity_tests, run_stationarity_tests
from .utils import get_regression_variable_names, run_tvp_wind_kalman_analysis
from .regression_models import (
    _validate_armax_baseline_spec, _prepare_baseline_armax_design,
    _fit_armax_with_fallback, _diagnose_nonconvergence_simple,
    select_armax_lags_aic, select_armax_lags_aic_checkpointed,
    fit_garchx_model
)
from .structural_analysis import (
    run_rolling_window_analysis, run_structural_break_analysis,
    run_trend_break_analysis, run_quantile_regression_analysis
)

def perform_multivariate_analysis(df, zone, target_region='SE1',
                                 run_ljungbox=False, run_hetero_tests=False, run_stationarity=False,
                                 optimize_armax_lags=False, use_checkpointed_lag_selection=True,
                                 armax_search_p_min=0, armax_search_p_max=10,
                                 armax_search_q_min=0, armax_search_q_max=10,
                                 armax_search_exclude_00=True,
                                 armax_search_require_convergence=True,
                                 armax_search_require_ljungbox_pass=True,
                                 armax_search_selection_criterion='bic',
                                 armax_search_save_top_n=20,
                                 run_tvp_wind_kalman=False,
                                 run_rolling_window=False, rolling_window_years=3,
                                 rolling_step_years=1, rolling_min_obs=24*180,
                                 run_quantile_regression=False,
                                 run_structural_break=False, structural_break_type='level',
                                 structural_break_max_breaks=5,
                                 structural_break_trimming=0.15, structural_break_known_dates=None,
                                 structural_break_window_years=1, structural_break_step_years=1/12,
                                 structural_break_min_obs=24*365 - 24*30,
                                 trend_break_test_method='legacy',
                                 bp_inference_mode='both',
                                 bp_significance_level=0.05,
                                 bp_bootstrap_reps=999,
                                 bp_bootstrap_block_length=8,
                                 bp_random_seed=42,
                                 bp_use_hac_se=True,
                                 structural_break_estimation_model='ols',
                                 armax_baseline_spec=None):
    """
    Runs OLS, ARMAX, and conditionally GARCH-X with full control variables.

    GARCH-X is fitted only if ARCH effects detected in ARMAX residuals (p < 0.05).

    Note: Always uses logged and deseasonalized variables (standard approach).

    Parameters:
    - df: DataFrame with all variables
    - zone: Zone identifier for display purposes
    - target_region: Target region for bottleneck dummies (default 'SE1')

    Returns:
    - ols_model: OLS regression results
    - armax_res: ARMAX model results
    - garch_res: GARCH-X results (None if not fitted)
    """
    print(f"\n--- RUNNING MULTIVARIATE ANALYSIS ({zone}) ---")
    print("Using: Logged and Deseasonalized variables (Standard approach)")

    y_name, exog_vars = get_regression_variable_names(df, target_region=target_region)
    Y = df[y_name]

    print(f"Dependent variable: {Y.name}")
    print(f"Exogenous variables: {exog_vars}")

    # TVP Kalman Filter mode: run time-varying parameter analysis and return early
    if run_tvp_wind_kalman:
        run_tvp_wind_kalman_analysis(df, zone, Y, exog_vars, plots_dir="plots")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Rolling-window mode: run rolling window analysis and return early
    if run_rolling_window:
        run_rolling_window_analysis(df, zone, Y, exog_vars,
                                    window_years=rolling_window_years,
                                    step_years=rolling_step_years,
                                    min_obs=rolling_min_obs,
                                    plots_dir="plots",
                                    results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Quantile regression mode: run quantile regression analysis and return early
    if run_quantile_regression:
        run_quantile_regression_analysis(df, zone,
                                         plots_dir="plots",
                                         results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    # Structural break mode: run analysis and return early
    if run_structural_break:
        if structural_break_type == 'trend':
            # Trend break analysis: detects changes in coefficient slope (sequential testing)
            run_trend_break_analysis(df, zone, Y, exog_vars,
                                    max_breaks=structural_break_max_breaks,
                                    trimming=structural_break_trimming,
                                    window_years=structural_break_window_years,
                                    step_years=structural_break_step_years,
                                    min_obs=structural_break_min_obs,
                                    trend_break_test_method=trend_break_test_method,
                                    bp_inference_mode=bp_inference_mode,
                                    bp_significance_level=bp_significance_level,
                                    bp_bootstrap_reps=bp_bootstrap_reps,
                                    bp_bootstrap_block_length=bp_bootstrap_block_length,
                                    bp_random_seed=bp_random_seed,
                                    bp_use_hac_se=bp_use_hac_se,
                                    estimation_model=structural_break_estimation_model,
                                    dynamic_armax_order=(3, 0, 3),
                                    plots_dir="plots",
                                    results_dir="results")
        else:  # 'level' or default
            # Level break analysis: detects step changes in coefficient mean (Bai-Perron)
            run_structural_break_analysis(df, zone, Y, exog_vars,
                                          max_breaks=structural_break_max_breaks,
                                          trimming=structural_break_trimming,
                                          known_break_dates=structural_break_known_dates,
                                          window_years=structural_break_window_years,
                                          step_years=structural_break_step_years,
                                          min_obs=structural_break_min_obs,
                                          estimation_model=structural_break_estimation_model,
                                          dynamic_armax_order=(3, 0, 3),
                                          plots_dir="plots",
                                          results_dir="results")
        return None, None, None  # Early return, skip OLS/ARMAX

    X = sm.add_constant(df[exog_vars])

    # 1. Standard OLS Regression
    ols_model = sm.OLS(Y, X).fit()
    print("\n--- OLS RESULTS ---")
    print(ols_model.summary())

    # Optional: Diagnostic tests on OLS residuals
    if run_stationarity:
        # Test stationarity of ALL variables used in the regression
        print("\n" + "="*80)
        print("STATIONARITY TESTS FOR ALL REGRESSION VARIABLES")
        print("="*80)

        # Test dependent variable (Price)
        run_stationarity_tests(Y, series_name=f"{zone} {Y.name} (Dependent Variable)")

        # Test all independent variables
        for var in exog_vars:
            run_stationarity_tests(df[var], series_name=f"{zone} {var} (Independent Variable)")

    if run_ljungbox:
        # Test for autocorrelation in OLS residuals
        run_ljungbox_test(ols_model.resid, lags=[5, 10, 15, 20])

    if run_hetero_tests:
        # Test for heteroskedasticity and ARCH effects in OLS residuals
        run_heteroskedasticity_tests(ols_model.resid, nlags=10)

    # 2. ARMAX(3,3)-GARCHX(1,1) Framework
    print(f"\n--- ARMAX-GARCHX RESULTS ---")
    baseline_spec = _validate_armax_baseline_spec(armax_baseline_spec)

    # Determine optimal lags if enabled, otherwise use default (3,3)
    if optimize_armax_lags:
        if len(baseline_spec.get('extra_ar_lags', [])) > 0:
            print(
                f"Note: baseline sparse AR lags {baseline_spec['extra_ar_lags']} are ignored when OPTIMIZE_ARMAX_LAGS=True "
                "(search uses contiguous ARMA(p,q) only)."
            )
        if use_checkpointed_lag_selection:
            # Use new checkpointed version with Ljung-Box diagnostics
            optimal_order, search_df = select_armax_lags_aic_checkpointed(
                Y, df[exog_vars],
                zone=zone,
                p_min=armax_search_p_min,
                p_max=armax_search_p_max,
                q_min=armax_search_q_min,
                q_max=armax_search_q_max,
                exclude_00=armax_search_exclude_00,
                require_convergence=armax_search_require_convergence,
                require_ljungbox_pass=armax_search_require_ljungbox_pass,
                selection_criterion=armax_search_selection_criterion,
                checkpoint_file=None,  # Auto-generate based on zone
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                save_top_n=armax_search_save_top_n
            )
        else:
            # Use strict version (no checkpointing)
            optimal_order, search_df = select_armax_lags_aic(
                Y, df[exog_vars],
                zone=zone,
                p_min=armax_search_p_min,
                p_max=armax_search_p_max,
                q_min=armax_search_q_min,
                q_max=armax_search_q_max,
                exclude_00=armax_search_exclude_00,
                require_convergence=armax_search_require_convergence,
                require_ljungbox_pass=armax_search_require_ljungbox_pass,
                selection_criterion=armax_search_selection_criterion,
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER,
                use_warm_start=ARMAX_USE_WARM_START,
                save_top_n=armax_search_save_top_n
            )
        if optimal_order is None:
            raise RuntimeError(
                "No admissible ARMAX order found in lag search. "
                "Check results/armax_search_all_<ZONE>.csv and relax search constraints or widen p/q range."
            )
        armax_order = (optimal_order[0], 0, optimal_order[1])
        Y_armax = Y
        X_armax = df[exog_vars]
        added_lag_cols = []
        dropped_rows = 0
    else:
        if baseline_spec.get('enabled', True):
            armax_order = baseline_spec['order']
            extra_lags = baseline_spec.get('extra_ar_lags', [])
            spec_label = baseline_spec.get('label', f'ARMAX{armax_order}')
        else:
            armax_order = (1, 0, 1)
            extra_lags = []
            spec_label = "ARMAX baseline disabled -> fallback ARMAX(1,0,1)"

        Y_armax, X_armax, added_lag_cols = _prepare_baseline_armax_design(
            Y,
            df[exog_vars],
            extra_ar_lags=extra_lags,
            drop_initial_nan=baseline_spec.get('drop_initial_nan', True)
        )
        dropped_rows = int(len(Y) - len(Y_armax))
        print(
            f"Using baseline spec: {spec_label}, "
            f"order={armax_order}, extra_ar_lags={extra_lags}"
        )
        if len(added_lag_cols) > 0:
            print(f"Added lagged dependent regressors: {added_lag_cols}")
        print(f"ARMAX estimation sample: n={len(Y_armax)} (dropped {dropped_rows} rows due to lagging/alignment)")
        print(f"Set OPTIMIZE_ARMAX_LAGS=True for AIC/BIC-based contiguous lag search.")

    # Mean Equation (Price Level)
    print(f"\n--- Fitting ARMAX{armax_order} model ---")

    armax_fit = _fit_armax_with_fallback(
        y=Y_armax,
        X_exog=X_armax,
        primary_order=armax_order,
        context_label=f"main ARMAX ({zone})",
        allow_nonconverged=ARMAX_ALLOW_NONCONVERGED,
        maxiter=ARMAX_MAXITER,
        solver=ARMAX_SOLVER,
        use_warm_start=ARMAX_USE_WARM_START,
        enable_fallback_orders=ARMAX_ENABLE_FALLBACK_ORDERS,
        fallback_orders=ARMAX_FALLBACK_ORDERS
    )
    if not armax_fit['ok']:
        attempts = armax_fit.get('attempts', [])
        print("ARMAX fitting failed. Attempt diagnostics:")
        for a in attempts:
            diag = a.get('diagnostics', {})
            print(f"  order={a.get('order')} ok={a.get('ok')} converged={a.get('converged')} "
                  f"iter={diag.get('iterations')} warnflag={diag.get('warnflag')} err={a.get('error')}")
        raise RuntimeError(
            "No converged ARMAX model found. "
            "You can relax ARMAX_ALLOW_NONCONVERGED=True to use non-converged estimates."
        )
    armax_res = armax_fit['model']
    selected_order = armax_fit.get('selected_order', armax_order)
    diag = armax_fit.get('diagnostics', {})
    print(f"ARMAX fit status: converged={armax_fit.get('converged')} "
          f"order_used={selected_order} iterations={diag.get('iterations')} "
          f"warnflag={diag.get('warnflag')} fopt={diag.get('fopt')}")
    attempts = armax_fit.get('attempts', [])
    primary_attempt = None
    for a in attempts:
        if tuple(a.get('order', ())) == tuple(armax_order):
            primary_attempt = a
            break

    if primary_attempt is not None and not primary_attempt.get('converged', False):
        pdiag = primary_attempt.get('diagnostics', {}) or {}
        label, explanation = _diagnose_nonconvergence_simple(
            primary_attempt,
            configured_maxiter=ARMAX_MAXITER
        )
        print("\n" + "="*80)
        print("ARMAX NON-CONVERGENCE DIAGNOSIS (PRIMARY ORDER)")
        print("="*80)
        print(f"Attempted order: ARMAX{armax_order}")
        print(f"Converged:       {primary_attempt.get('converged')}")
        print(f"Iterations:      {pdiag.get('iterations')}")
        print(f"Warnflag:        {pdiag.get('warnflag')}")
        print(f"Gradient max abs:{pdiag.get('gradient_max_abs')}")
        print(f"Diagnosis:       {label}")
        print(f"Why:             {explanation}")
        if tuple(selected_order) != tuple(armax_order):
            selected_attempt = next(
                (a for a in attempts if tuple(a.get('order', ())) == tuple(selected_order)),
                None
            )
            sdiag = (selected_attempt or {}).get('diagnostics', {}) or {}
            print(f"Fallback succeeded with ARMAX{selected_order} "
                  f"(iterations={sdiag.get('iterations')}, warnflag={sdiag.get('warnflag')}).")
        print("="*80)
    if armax_fit.get('used_nonconverged', False):
        print("WARNING: Using non-converged ARMAX fit due to ARMAX_ALLOW_NONCONVERGED=True.")

    print(f"\nMEAN EQUATION (Price Level) - ARMAX{selected_order}:")
    print(armax_res.summary())

    # Optional: Diagnostic tests on ARMAX residuals
    arch_detected = False
    if run_ljungbox:
        print("\n" + "="*70)
        print("DIAGNOSTIC TESTS ON ARMAX RESIDUALS")
        print("="*70)
        run_ljungbox_test(armax_res.resid, lags=[5, 10, 15, 20])

    if run_hetero_tests:
        # Run tests and check if ARCH effects detected
        run_heteroskedasticity_tests(armax_res.resid, nlags=10)

        # Check ARCH test result
        lm_stat, lm_pval, f_stat, f_pval = het_arch(armax_res.resid, nlags=10)
        if lm_pval < 0.05:
            arch_detected = True
            print(f"\n{'='*70}")
            print(f"ARCH EFFECTS DETECTED (p={lm_pval:.4f} < 0.05)")
            print(f"Proceeding with GARCH-X modeling...")
            print(f"{'='*70}")

    # GARCH-X component: Fit if ARCH effects confirmed
    garch_res = None
    if arch_detected and FIT_GARCH_IF_ARCH:
        # Always use logged wind variable
        wind_var = 'Wind_Forecast_Log'

        garch_res, garch_diagnostics = fit_garchx_model(
            armax_res.resid,
            df,
            wind_var=wind_var,
            p=GARCH_ORDER[0],
            q=GARCH_ORDER[1],
            show_diagnostics=True
        )

        # Compare AIC/BIC
        if garch_res is not None:
            print(f"\n{'='*70}")
            print("MODEL COMPARISON")
            print(f"{'='*70}")
            print(f"ARMAX({selected_order[0]},{selected_order[2]}) AIC: {armax_res.aic:.2f}")
            print(f"ARMAX-GARCH({GARCH_ORDER[0]},{GARCH_ORDER[1]})-X AIC: {garch_res.aic:.2f}")
            improvement = armax_res.aic - garch_res.aic
            print(f"AIC Improvement: {improvement:.2f} {'(better)' if improvement > 0 else '(worse)'}")
            print(f"{'='*70}")
    elif not arch_detected:
        print(f"\n{'='*70}")
        print("NO ARCH EFFECTS DETECTED - GARCH modeling not necessary")
        print("ARMAX model is sufficient (constant variance assumption holds)")
        print(f"{'='*70}")

    return ols_model, armax_res, garch_res


