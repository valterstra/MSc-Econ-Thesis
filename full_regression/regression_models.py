"""
################################################################################
#  [Module 06/10]  regression_models.py  –  ARMAX, GARCH, and Lag Selection
#
#  Contains:
#    --- ARMAX Internals ---
#    1.  _attach_inferred_frequency()        : attach freq metadata to avoid statsmodels warnings
#    2.  _validate_armax_baseline_spec()     : validate baseline ARMAX spec dict
#    3.  _prepare_baseline_armax_design()    : build design matrix with optional extra AR lags
#    4.  _build_warm_start_params()          : warm-start from OLS for faster ARMAX convergence
#    5.  _fit_armax_with_controls()          : core ARMAX fit with convergence handling
#    6.  _fit_armax_with_fallback()          : ARMAX with fallback order ladder
#    7.  _diagnose_nonconvergence_simple()   : classify non-convergence reason
#
#    --- ARMAX Grid Search ---
#    8.  _evaluate_armax_candidate()         : evaluate single ARMAX(p,q) candidate
#    9.  _select_best_armax_candidate()      : rank candidates by AIC/BIC + eligibility
#    10. _build_armax_search_grid()          : build (p,q) grid
#    11. _save_armax_search_reports()        : save top-N search results to CSV/Excel
#    12. select_armax_lags_aic()             : strict grid search without checkpointing
#    13. select_armax_lags_aic_checkpointed(): checkpointed grid search with resume support
#
#    --- GARCH ---
#    14. fit_garchx_model()                  : GARCH(1,1)-X on ARMAX residuals
#                                             (variance eq. uses Wind_Forecast_Log)
#
#  Dependencies: config, diagnostics
################################################################################
"""

import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from .config import (
    ARMAX_ALLOW_NONCONVERGED, ARMAX_MAXITER, ARMAX_SOLVER,
    ARMAX_USE_WARM_START, ARMAX_ENABLE_FALLBACK_ORDERS, ARMAX_FALLBACK_ORDERS
)
from .diagnostics import run_ljungbox_test, run_heteroskedasticity_tests

def _attach_inferred_frequency(y, X_exog):
    """
    Attach inferred frequency metadata to datetime index when possible.
    This avoids noisy statsmodels date-index warnings without changing values.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        return y, X_exog, None

    inferred = y.index.freqstr or pd.infer_freq(y.index)
    if not inferred:
        return y, X_exog, None

    try:
        new_idx = pd.DatetimeIndex(y.index.values, freq=inferred)
        y_adj = y.copy()
        x_adj = X_exog.copy()
        y_adj.index = new_idx
        x_adj.index = new_idx
        return y_adj, x_adj, inferred
    except Exception:
        return y, X_exog, None


def _validate_armax_baseline_spec(spec):
    """
    Validate and normalize baseline ARMAX specification.
    Returns a normalized spec dict.
    """
    default_spec = {
        'enabled': True,
        'order': (1, 0, 1),
        'extra_ar_lags': [],
        'drop_initial_nan': True,
        'label': 'ARMAX(1,0,1)'
    }
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("ARMAX baseline spec must be a dictionary.")

    normalized = default_spec.copy()
    normalized.update(spec)

    order = normalized.get('order')
    if (not isinstance(order, (tuple, list))) or len(order) != 3:
        raise ValueError("ARMAX baseline spec 'order' must be a tuple/list of length 3, e.g. (1, 0, 1).")
    try:
        order = (int(order[0]), int(order[1]), int(order[2]))
    except Exception:
        raise ValueError("ARMAX baseline spec 'order' values must be integers.")
    if order[0] < 0 or order[2] < 0:
        raise ValueError("ARMAX baseline spec requires non-negative AR/MA orders.")
    normalized['order'] = order

    raw_lags = normalized.get('extra_ar_lags', [])
    if raw_lags is None:
        raw_lags = []
    if not isinstance(raw_lags, (list, tuple)):
        raise ValueError("ARMAX baseline spec 'extra_ar_lags' must be a list/tuple of positive integers.")
    lag_list = []
    for lag in raw_lags:
        if isinstance(lag, bool):
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' cannot contain boolean values.")
        try:
            lag_i = int(lag)
        except Exception:
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' must contain integers.")
        if lag_i <= 0:
            raise ValueError("ARMAX baseline spec 'extra_ar_lags' must contain positive integers.")
        lag_list.append(lag_i)
    if len(set(lag_list)) != len(lag_list):
        raise ValueError("ARMAX baseline spec 'extra_ar_lags' contains duplicates.")
    normalized['extra_ar_lags'] = sorted(lag_list)
    normalized['drop_initial_nan'] = bool(normalized.get('drop_initial_nan', True))

    label = normalized.get('label')
    normalized['label'] = str(label) if label is not None else f"ARMAX{normalized['order']}"
    normalized['enabled'] = bool(normalized.get('enabled', True))

    return normalized


def _prepare_baseline_armax_design(y, X_exog, extra_ar_lags=None, drop_initial_nan=True):
    """
    Build ARMAX design matrix for baseline run, optionally augmenting exogenous set
    with sparse lagged dependent terms (e.g., y_lag_23, y_lag_24, y_lag_25).
    """
    y_aligned = y.copy()
    x_aug = X_exog.copy()
    added_cols = []

    lag_list = [] if extra_ar_lags is None else list(extra_ar_lags)
    for lag in lag_list:
        col = f"y_lag_{int(lag)}"
        x_aug[col] = y_aligned.shift(int(lag))
        added_cols.append(col)

    if drop_initial_nan and len(added_cols) > 0:
        joined = pd.concat([y_aligned.rename('_y'), x_aug], axis=1).dropna()
        y_aligned = joined['_y']
        x_aug = joined.drop(columns=['_y'])

    return y_aligned, x_aug, added_cols


def _build_warm_start_params(y_fit, x_fit, order):
    """
    Build start params by mapping ARMAX(0,0,0)-X estimates into target order params.
    Falls back to statsmodels defaults if mapping fails.
    """
    model = sm.tsa.ARIMA(y_fit, exog=x_fit, order=order)
    start_params = np.asarray(model.start_params, dtype=float).copy()
    param_names = list(model.param_names)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=ValueWarning)
            base_fit = sm.tsa.ARIMA(y_fit, exog=x_fit, order=(0, 0, 0)).fit(
                method='statespace',
                method_kwargs={'maxiter': 100}
            )
        base_map = dict(zip(base_fit.param_names, np.asarray(base_fit.params)))
        for i, name in enumerate(param_names):
            if name in base_map:
                start_params[i] = base_map[name]
    except Exception:
        # Keep default start params if warm-start fitting fails.
        pass

    return start_params


def _fit_armax_with_controls(y, X_exog, order=(3, 0, 3), context_label="",
                             accept_nonconverged=True, maxiter=300, solver='statespace',
                             use_warm_start=True):
    """
    Fit ARMAX quietly and enforce convergence acceptance.
    Returns dict with keys: ok, model, fail_reason, error, freq, converged, diagnostics.
    """
    y_fit, x_fit, freq = _attach_inferred_frequency(y, X_exog)

    diagnostics = {
        'order': order,
        'converged': False,
        'iterations': None,
        'warnflag': None,
        'fopt': None,
        'gradient_max_abs': None,
        'context': context_label
    }

    try:
        fit_kwargs = {
            'method': solver,
            'method_kwargs': {'maxiter': maxiter}
        }
        if use_warm_start:
            fit_kwargs['start_params'] = _build_warm_start_params(y_fit, x_fit, order)

        with warnings.catch_warnings():
            # Keep console noise minimal while still handling failures explicitly.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=ValueWarning)
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"statsmodels\.tsa\.statespace\.sarimax"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Non-stationary starting autoregressive parameters found.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Non-invertible starting MA parameters found.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="A date index has been provided, but it has no associated frequency information.*"
            )
            fitted = sm.tsa.ARIMA(y_fit, exog=x_fit, order=order).fit(**fit_kwargs)

        mle_retvals = getattr(fitted, 'mle_retvals', {})
        converged = bool(mle_retvals.get('converged', False))
        gopt = mle_retvals.get('gopt', None)
        diagnostics.update({
            'converged': converged,
            'iterations': mle_retvals.get('iterations', None),
            'warnflag': mle_retvals.get('warnflag', None),
            'fopt': mle_retvals.get('fopt', None),
            'gradient_max_abs': float(np.max(np.abs(gopt))) if gopt is not None else None
        })

        if not converged and not accept_nonconverged:
            return {
                'ok': False,
                'model': None,
                'fail_reason': 'non_converged',
                'error': f"Non-converged MLE fit ({context_label})",
                'freq': freq,
                'converged': False,
                'diagnostics': diagnostics
            }

        return {
            'ok': True,
            'model': fitted,
            'fail_reason': None,
            'error': None,
            'freq': freq,
            'converged': converged,
            'diagnostics': diagnostics
        }
    except Exception as e:
        return {
            'ok': False,
            'model': None,
            'fail_reason': 'exception',
            'error': f"{type(e).__name__}: {e}",
            'freq': freq,
            'converged': False,
            'diagnostics': diagnostics
        }


def _fit_armax_with_fallback(y, X_exog, primary_order=(3, 0, 3), context_label="",
                             allow_nonconverged=False, maxiter=300, solver='statespace',
                             use_warm_start=True, enable_fallback_orders=True,
                             fallback_orders=None):
    """
    Fit ARMAX using a fallback ladder. Prefers converged models.
    """
    orders = [primary_order]
    if enable_fallback_orders:
        if fallback_orders is None:
            fallback_orders = []
        for order in fallback_orders:
            if order not in orders:
                orders.append(order)

    attempts = []
    first_nonconverged_ok = None

    for order in orders:
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog,
            order=order,
            context_label=context_label,
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start
        )
        attempts.append({
            'order': order,
            'ok': fit['ok'],
            'converged': fit.get('converged', False),
            'fail_reason': fit.get('fail_reason'),
            'error': fit.get('error'),
            'diagnostics': fit.get('diagnostics', {})
        })

        if fit['ok'] and fit.get('converged', False):
            fit['selected_order'] = order
            fit['attempts'] = attempts
            fit['used_nonconverged'] = False
            return fit

        if fit['ok'] and first_nonconverged_ok is None:
            first_nonconverged_ok = fit
            first_nonconverged_ok['selected_order'] = order

    if first_nonconverged_ok is not None and allow_nonconverged:
        first_nonconverged_ok['attempts'] = attempts
        first_nonconverged_ok['used_nonconverged'] = True
        return first_nonconverged_ok

    return {
        'ok': False,
        'model': None,
        'fail_reason': 'non_converged',
        'error': f"No converged ARMAX fit found for {context_label}",
        'freq': None,
        'converged': False,
        'diagnostics': None,
        'selected_order': None,
        'attempts': attempts,
        'used_nonconverged': False
    }


def _diagnose_nonconvergence_simple(attempt, configured_maxiter=None):
    """
    Classify non-convergence reason using lightweight diagnostics from one attempt.
    """
    if attempt is None:
        return 'unknown', "No diagnostics available for this failed fit."

    if attempt.get('ok') is False and attempt.get('fail_reason') == 'exception':
        err = attempt.get('error', 'Unknown exception')
        return 'unknown', f"Fit failed with exception before optimizer convergence checks: {err}"

    diag = attempt.get('diagnostics', {}) or {}
    iterations = diag.get('iterations', None)
    warnflag = diag.get('warnflag', None)
    grad = diag.get('gradient_max_abs', None)

    if iterations == 0:
        return (
            'immediate_stop',
            "Optimizer stopped at iteration 0: it could not take a valid first improvement step "
            "(often due to hard AR/MA constraints or poor local geometry)."
        )

    if configured_maxiter is not None and iterations is not None:
        if iterations >= int(round(0.95 * configured_maxiter)):
            return (
                'hit_iteration_limit',
                "Optimizer was still working but reached the iteration limit before satisfying convergence tolerance."
            )

    if warnflag not in (0, None):
        return (
            'unstable_or_numerical',
            f"Optimizer ended with warnflag={warnflag}; convergence likely blocked by numerical instability "
            "or an ill-conditioned likelihood surface."
        )

    if grad is not None and grad > 1e-3:
        return (
            'unstable_or_numerical',
            f"Gradient remained materially non-zero ({grad:.3e}), suggesting it did not settle at an optimum."
        )

    return (
        'unknown',
        "Non-convergence occurred, but diagnostics are inconclusive from this run."
    )




def _evaluate_armax_candidate(Y, exog_vars, p, q,
                            ljungbox_lags=(5, 10, 15, 20),
                            maxiter=300,
                            solver='statespace',
                            use_warm_start=True):
    """Evaluate a single ARMAX(p,0,q) candidate without fallback."""
    order = (p, 0, q)
    timestamp = pd.Timestamp.now().isoformat()

    result = {
        'p': p,
        'q': q,
        'status': 'exception',
        'converged': False,
        'iterations': np.nan,
        'warnflag': np.nan,
        'fopt': np.nan,
        'gradient_max_abs': np.nan,
        'aic': np.nan,
        'bic': np.nan,
        'hqic': np.nan,
        'passes_ljungbox': False,
        'diagnosis_label': 'unknown',
        'diagnosis_why': '',
        'error_message': '',
        'timestamp': timestamp
    }

    for lag in ljungbox_lags:
        result[f'ljungbox_lag_{lag}_stat'] = np.nan
        result[f'ljungbox_lag_{lag}_pval'] = np.nan

    fit = _fit_armax_with_controls(
        y=Y,
        X_exog=exog_vars,
        order=order,
        context_label=f"lag_search ARMAX({p},{q})",
        accept_nonconverged=True,
        maxiter=maxiter,
        solver=solver,
        use_warm_start=use_warm_start
    )

    if not fit['ok']:
        result['status'] = 'exception'
        result['error_message'] = fit.get('error', 'Unknown fit failure')
        label, why = _diagnose_nonconvergence_simple(
            {'ok': False, 'fail_reason': 'exception', 'error': result['error_message'], 'diagnostics': {}},
            configured_maxiter=maxiter
        )
        result['diagnosis_label'] = label
        result['diagnosis_why'] = why
        return result

    model = fit['model']
    diag = fit.get('diagnostics', {}) or {}
    converged = bool(fit.get('converged', False))

    result['converged'] = converged
    result['status'] = 'converged' if converged else 'nonconverged'
    result['iterations'] = diag.get('iterations', np.nan)
    result['warnflag'] = diag.get('warnflag', np.nan)
    result['fopt'] = diag.get('fopt', np.nan)
    result['gradient_max_abs'] = diag.get('gradient_max_abs', np.nan)
    result['aic'] = float(getattr(model, 'aic', np.nan))
    result['bic'] = float(getattr(model, 'bic', np.nan))
    result['hqic'] = float(getattr(model, 'hqic', np.nan))
    if converged:
        result['diagnosis_label'] = 'converged'
        result['diagnosis_why'] = 'Converged successfully.'
    else:
        label, why = _diagnose_nonconvergence_simple(
            {'ok': True, 'converged': False, 'diagnostics': diag, 'fail_reason': None, 'error': None},
            configured_maxiter=maxiter
        )
        result['diagnosis_label'] = label
        result['diagnosis_why'] = why

    try:
        lb_results = run_ljungbox_test(
            model.resid,
            lags=list(ljungbox_lags),
            return_results=True,
            print_output=False
        )
        for _, row in lb_results.iterrows():
            lag = int(row['lag'])
            result[f'ljungbox_lag_{lag}_stat'] = row['test_stat']
            result[f'ljungbox_lag_{lag}_pval'] = row['p_value']
        result['passes_ljungbox'] = bool((lb_results['p_value'] > 0.05).all())
    except Exception as e:
        result['passes_ljungbox'] = False
        if not result['error_message']:
            result['error_message'] = f"Ljung-Box failed: {e}"

    return result


def _select_best_armax_candidate(results_df,
                                require_convergence=True,
                                require_ljungbox_pass=True,
                                criterion='bic'):
    """Select best candidate from evaluated grid based on eligibility and criterion."""
    if results_df.empty:
        return None, pd.DataFrame(), {
            'tested': 0,
            'converged': 0,
            'eligible': 0,
            'exceptions': 0,
            'nonconverged': 0
        }

    criterion_col = criterion.lower()
    if criterion_col not in {'aic', 'bic'}:
        raise ValueError(f"Invalid criterion '{criterion}'. Use 'aic' or 'bic'.")

    df = results_df.copy()
    mask = pd.Series(True, index=df.index)
    if require_convergence:
        mask &= (df['converged'] == True)
    if require_ljungbox_pass:
        mask &= (df['passes_ljungbox'] == True)
    mask &= df[criterion_col].notna()

    df['eligible_for_selection'] = mask
    eligible = df[df['eligible_for_selection']].copy()

    if eligible.empty:
        best_order = None
    else:
        best_idx = eligible[criterion_col].idxmin()
        best_row = eligible.loc[best_idx]
        best_order = (int(best_row['p']), int(best_row['q']))

    stats = {
        'tested': int(len(df)),
        'converged': int((df['converged'] == True).sum()),
        'eligible': int(len(eligible)),
        'exceptions': int((df['status'] == 'exception').sum()),
        'nonconverged': int((df['status'] == 'nonconverged').sum())
    }

    return best_order, df, stats


def _build_armax_search_grid(p_min, p_max, q_min, q_max, exclude_00=True):
    """Build grid list [(p,q), ...] for ARMAX search."""
    grid = []
    for p in range(int(p_min), int(p_max) + 1):
        for q in range(int(q_min), int(q_max) + 1):
            if exclude_00 and p == 0 and q == 0:
                continue
            grid.append((p, q))
    return grid


def _save_armax_search_reports(results_df, zone='SE1', top_n=20,
                              criterion='bic', results_dir='results'):
    """Persist full and top-N search reports to CSV and Excel."""
    os.makedirs(results_dir, exist_ok=True)

    all_path = os.path.join(results_dir, f'armax_search_all_{zone}.csv')
    results_df.to_csv(all_path, index=False)
    all_xlsx_path = os.path.join(results_dir, f'armax_search_all_{zone}.xlsx')
    try:
        results_df.to_excel(all_xlsx_path, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write Excel full report: {e}")

    criterion_col = criterion.lower()
    eligible = results_df[results_df.get('eligible_for_selection', False) == True].copy()
    if not eligible.empty and criterion_col in eligible.columns:
        top_df = eligible.sort_values(criterion_col).head(int(top_n))
    else:
        top_df = pd.DataFrame(columns=results_df.columns)

    top_path = os.path.join(results_dir, f'armax_search_top_{zone}.csv')
    top_df.to_csv(top_path, index=False)
    top_xlsx_path = os.path.join(results_dir, f'armax_search_top_{zone}.xlsx')
    try:
        top_df.to_excel(top_xlsx_path, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write Excel top report: {e}")

    return all_path, top_path, all_xlsx_path, top_xlsx_path


def select_armax_lags_aic(Y, exog_vars, zone='SE1',
                          p_min=0, p_max=10, q_min=0, q_max=10,
                          exclude_00=True,
                          require_convergence=True,
                          require_ljungbox_pass=True,
                          selection_criterion='bic',
                          ljungbox_lags=(5, 10, 15, 20),
                          maxiter=300,
                          solver='statespace',
                          use_warm_start=True,
                          save_top_n=20,
                          results_dir='results'):
    """Strict ARMAX lag search without fallback; returns best order and full results."""
    grid = _build_armax_search_grid(p_min, p_max, q_min, q_max, exclude_00=exclude_00)

    print("\n--- ARMAX LAG SELECTION (STRICT GRID SEARCH) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min}..{p_max}, q={q_min}..{q_max}, exclude_00={exclude_00}")
    print(f"Eligibility: require_convergence={require_convergence}, require_ljungbox_pass={require_ljungbox_pass}")
    print(f"Selection criterion: {selection_criterion.upper()}")
    print(f"Models to test: {len(grid)}")

    rows = []
    total = len(grid)
    for i, (p, q) in enumerate(grid, start=1):
        print(f"[{i}/{total}] Testing ARMAX({p},{q})... ", end="")
        row = _evaluate_armax_candidate(
            Y=Y,
            exog_vars=exog_vars,
            p=p,
            q=q,
            ljungbox_lags=ljungbox_lags,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start
        )
        rows.append(row)
        diag_short = row.get('diagnosis_label', '')
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
            f"diag={diag_short}"
        )

    results_df = pd.DataFrame(rows)
    best_order, results_df, stats = _select_best_armax_candidate(
        results_df,
        require_convergence=require_convergence,
        require_ljungbox_pass=require_ljungbox_pass,
        criterion=selection_criterion
    )
    all_path, top_path, all_xlsx_path, top_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        top_n=save_top_n,
        criterion=selection_criterion,
        results_dir=results_dir
    )

    print("\n" + "="*80)
    print("ARMAX SEARCH SUMMARY")
    print("="*80)
    print(f"Tested: {stats['tested']} | Converged: {stats['converged']} | Eligible: {stats['eligible']} | "
          f"Nonconverged: {stats['nonconverged']} | Exceptions: {stats['exceptions']}")
    print(f"Saved full report: {all_path}")
    print(f"Saved full Excel:  {all_xlsx_path}")
    print(f"Saved top report:  {top_path}")
    print(f"Saved top Excel:   {top_xlsx_path}")

    if best_order is None:
        print("No admissible ARMAX order found under current eligibility constraints.")
    else:
        print(f"Selected order: ARMAX{best_order} by {selection_criterion.upper()}")

    return best_order, results_df


def select_armax_lags_aic_checkpointed(Y, exog_vars, zone='SE1',
                                       p_min=0, p_max=10, q_min=0, q_max=10,
                                       exclude_00=True,
                                       require_convergence=True,
                                       require_ljungbox_pass=True,
                                       selection_criterion='bic',
                                       checkpoint_file=None,
                                       ljungbox_lags=(5, 10, 15, 20),
                                       save_interval=1,
                                       maxiter=300,
                                       solver='statespace',
                                       use_warm_start=True,
                                       save_top_n=20,
                                       results_dir='results'):
    """Checkpointed strict ARMAX lag search (no fallback)."""
    os.makedirs(results_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(results_dir, f'armax_lag_selection_checkpoint_{zone}.csv')

    grid = _build_armax_search_grid(p_min, p_max, q_min, q_max, exclude_00=exclude_00)
    print("\n--- ARMAX LAG SELECTION (CHECKPOINTED, STRICT) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min}..{p_max}, q={q_min}..{q_max}, exclude_00={exclude_00}")
    print(f"Selection criterion: {selection_criterion.upper()}")
    print(f"Checkpoint file: {checkpoint_file}")

    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        completed_specs = set()
        if {'p', 'q', 'status'}.issubset(checkpoint_df.columns):
            for _, row in checkpoint_df.iterrows():
                status = str(row.get('status', ''))
                if status in {'converged', 'nonconverged', 'exception', 'completed', 'failed'}:
                    completed_specs.add((int(row['p']), int(row['q'])))
        print(f"Resuming: found {len(completed_specs)} tested specifications")
    else:
        checkpoint_df = pd.DataFrame()
        completed_specs = set()
        print("Starting fresh (no existing checkpoint)")

    total = len(grid)
    tested_before = len(completed_specs)
    print(f"Models to test: {total - tested_before} (Total: {total}, Already tested: {tested_before})")

    tested_counter = tested_before
    for p, q in grid:
        if (p, q) in completed_specs:
            continue

        tested_counter += 1
        print(f"[{tested_counter}/{total}] Testing ARMAX({p},{q})... ", end="")
        row = _evaluate_armax_candidate(
            Y=Y,
            exog_vars=exog_vars,
            p=p,
            q=q,
            ljungbox_lags=ljungbox_lags,
            maxiter=maxiter,
            solver=solver,
            use_warm_start=use_warm_start
        )
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
            f"diag={row.get('diagnosis_label', '')}"
        )

        row_df = pd.DataFrame([row])
        if checkpoint_df.empty:
            checkpoint_df = row_df
        else:
            checkpoint_df = pd.concat([checkpoint_df, row_df], ignore_index=True)

        if len(checkpoint_df) % max(int(save_interval), 1) == 0:
            checkpoint_df.to_csv(checkpoint_file, index=False)
            checkpoint_xlsx = os.path.splitext(checkpoint_file)[0] + '.xlsx'
            try:
                checkpoint_df.to_excel(checkpoint_xlsx, index=False)
            except Exception as e:
                print(f"WARNING: Failed to write checkpoint Excel report: {e}")

    checkpoint_df.to_csv(checkpoint_file, index=False)
    checkpoint_xlsx = os.path.splitext(checkpoint_file)[0] + '.xlsx'
    try:
        checkpoint_df.to_excel(checkpoint_xlsx, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write checkpoint Excel report: {e}")

    best_order, results_df, stats = _select_best_armax_candidate(
        checkpoint_df.copy(),
        require_convergence=require_convergence,
        require_ljungbox_pass=require_ljungbox_pass,
        criterion=selection_criterion
    )
    all_path, top_path, all_xlsx_path, top_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        top_n=save_top_n,
        criterion=selection_criterion,
        results_dir=results_dir
    )

    print("\n" + "="*80)
    print("ARMAX CHECKPOINT SEARCH SUMMARY")
    print("="*80)
    print(f"Tested: {stats['tested']} | Converged: {stats['converged']} | Eligible: {stats['eligible']} | "
          f"Nonconverged: {stats['nonconverged']} | Exceptions: {stats['exceptions']}")
    print(f"Checkpoint:       {checkpoint_file}")
    print(f"Checkpoint Excel: {checkpoint_xlsx}")
    print(f"Saved full report: {all_path}")
    print(f"Saved full Excel:  {all_xlsx_path}")
    print(f"Saved top report:  {top_path}")
    print(f"Saved top Excel:   {top_xlsx_path}")

    if best_order is None:
        print("No admissible ARMAX order found under current eligibility constraints.")
    else:
        print(f"Selected order: ARMAX{best_order} by {selection_criterion.upper()}")

    return best_order, results_df

def fit_garchx_model(armax_residuals, df, wind_var='Wind_Forecast_Log',
                     p=1, q=1, show_diagnostics=True):
    """
    Fits GARCH(p,q)-X model on ARMAX residuals with Wind as exogenous variable.

    Parameters:
    - armax_residuals: Residuals from ARMAX mean equation (pd.Series)
    - df: DataFrame with all variables (must include wind_var)
    - wind_var: Name of wind variable for variance equation
    - p: GARCH lag order (default 1)
    - q: ARCH lag order (default 1)
    - show_diagnostics: Run tests on standardized residuals

    Returns:
    - garch_res: Fitted ARCH model object (or None if fitting fails)
    - diagnostics: Dict with test results (or None)
    """
    print(f"\n--- Fitting GARCH({p},{q})-X model with {wind_var} in variance equation ---")

    try:
        # Align wind variable with residuals index
        wind_series = df.loc[armax_residuals.index, wind_var]

        # Check for NaNs
        if wind_series.isna().any():
            print(f"WARNING: {wind_var} contains NaN values. Dropping NaNs...")
            valid_idx = ~(armax_residuals.isna() | wind_series.isna())
            armax_residuals = armax_residuals[valid_idx]
            wind_series = wind_series[valid_idx]

        # Specify GARCH(p,q)-X model
        garch_spec = arch_model(
            armax_residuals,
            vol='GARCH',
            p=p, q=q,
            x=wind_series.values.reshape(-1, 1),  # Must be 2D array
            rescale=False
        )

        # Fit with MLE
        garch_res = garch_spec.fit(disp='off', show_warning=False)

        # Display results
        print(f"\nVARIANCE EQUATION - GARCH({p},{q})-X:")
        print(garch_res.summary())

        # Extract standardized residuals
        std_resid = garch_res.std_resid

        # Run diagnostics if requested
        diagnostics = None
        if show_diagnostics:
            print("\n" + "="*80)
            print("DIAGNOSTIC TESTS ON GARCH STANDARDIZED RESIDUALS")
            print("="*80)
            print("(These should show NO autocorrelation and NO ARCH effects)")

            # Ljung-Box test
            run_ljungbox_test(std_resid, lags=[5, 10, 15, 20])

            # Heteroskedasticity tests
            run_heteroskedasticity_tests(std_resid, nlags=10)

            diagnostics = {
                'std_resid_mean': std_resid.mean(),
                'std_resid_std': std_resid.std(),
                'aic': garch_res.aic,
                'bic': garch_res.bic
            }

        return garch_res, diagnostics

    except Exception as e:
        print(f"ERROR: GARCH fitting failed - {str(e)}")
        print("Continuing with ARMAX-only results...")
        return None, None


