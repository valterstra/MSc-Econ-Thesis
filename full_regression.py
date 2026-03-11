import pandas as pd
import numpy as np
import os
import warnings
import time
import io
import contextlib
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from arch.unitroot import DFGLS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (save plots only, no display)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import holidays
import ruptures as rpt
from scipy import stats
# --- Price transformation mode ---
# True  → log(price) then deseasonalize  (Method A, standard log-linear)
# False → deseasonalize price in levels  (semi-log: price not logged)
LOG_PRICE = True

if LOG_PRICE:
    from preprocessing import (
        TRADING_PARTNERS,
        load_data, handle_negative_prices, apply_log_transform,
        deseasonalize_logged_variables, handle_outliers_gianfreda,
        preprocess_data_for_regression
    )
else:
    from preprocessing2 import (
        TRADING_PARTNERS,
        load_data, handle_negative_prices, apply_log_transform,
        deseasonalize_logged_variables, handle_outliers_gianfreda,
        preprocess_data_for_regression
    )
from visualizations import (
    plot_zone_comparisons, plot_time_series, plot_distributions,
    plot_boxplots, detect_outliers, plot_outliers_timeline,
    plot_scatter_matrix, run_visualizations
)
from diagnostics import (
    run_ljungbox_test, run_heteroskedasticity_tests,
    run_stationarity_tests, run_collinearity_diagnostics
)

# ARMAX fitting defaults
ARMAX_ALLOW_NONCONVERGED = False
ARMAX_MAXITER = 300
ARMAX_SOLVER = 'statespace'
ARMAX_ENABLE_FALLBACK_ORDERS = False
ARMAX_FALLBACK_ORDERS = []
ARMAX_BASELINE_SPEC = {
    'order': (1, 0, 1),
    'extra_ar_lags': []
}

# --- 4. MODELING FUNCTIONS ---

# Ordered mapping from CONTROLS key → column name (wind first, descending importance)
_CONTROLS_COL_MAP = [
    ('Wind',        'Wind_Forecast_Log'),
    ('Hydro',       'Hydro_Reserves_Log_Deseasonalized'),
    ('NetExchange', 'Net_Exchange'),
    ('Consumption', 'Consumption_Log_Deseasonalized'),
    ('Oil',         'Oil_Price_Log_Deseasonalized'),
    ('Gas',         'Gas_Price_Log_Deseasonalized'),
]

def get_regression_variable_names(df, target_region='SE1', controls=None):
    """Return dependent variable name and exogenous variable names used in regression.

    controls: dict with keys matching _CONTROLS_COL_MAP plus 'Bottlenecks'.
              Missing keys default to True (include). Pass an empty dict to
              include everything; pass None for the same effect.
    """
    if controls is None:
        controls = {}
    y_name = 'Price_DS'
    exog_vars = [col for key, col in _CONTROLS_COL_MAP if controls.get(key, True)]

    if controls.get('Bottlenecks', True):
        for partner in TRADING_PARTNERS.get(target_region, []):
            bneck_col = f'BNECK_{target_region}_{partner}'
            if bneck_col in df.columns:
                exog_vars.append(bneck_col)

    return y_name, exog_vars


def run_rolling_window_analysis(df, zone, Y=None, exog_vars=None,
                                window_years=3, step_years=1, min_obs=24*180,
                                plots_dir="plots", results_dir="results",
                                use_window_local_preprocessing=False,
                                controls=None,
                                target_region='SE1'):
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
        y_name, exog_names = get_regression_variable_names(
            df,
            target_region=target_region,
            controls=controls
        )
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
                    suppress_output=True
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

    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)
    csv_path = os.path.join(zone_results_dir, f'rolling_wind_coef_{zone}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    excel_path = os.path.join(zone_results_dir, f'rolling_wind_coef_{zone}.xlsx')
    results_df.to_excel(excel_path, index=False)
    print(f"Saved results to: {excel_path}")

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
    for x, y in zip(midpoints, beta_values):
        ax.annotate(
            f'{y:.3f}',
            xy=(x, y),
            xytext=(0, 14),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

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
        'order': (1, 0, 1),
        'extra_ar_lags': []
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
    normalized['label'] = f"ARMAX{normalized['order']}"
    normalized['drop_initial_nan'] = True

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


def _build_manual_armax_start_params(y_fit, x_fit, order,
                                     ar_overrides=None, ma_overrides=None,
                                     sigma2_override=None, exog_source='ols'):
    """
    Build a constrained start vector using interpretable coefficient guesses.

    exog_source='ols':
      set constant / exogenous coefficients from OLS(y ~ const + X).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ValueWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="A date index has been provided, but it has no associated frequency information.*"
        )
        model = sm.tsa.ARIMA(y_fit, exog=x_fit, order=order)
    start_params = np.asarray(model.start_params, dtype=float).copy()
    param_names = list(model.param_names)

    if exog_source == 'ols':
        try:
            ols_fit = sm.OLS(y_fit, sm.add_constant(x_fit)).fit()
            ols_map = {'const': float(ols_fit.params['const'])}
            for col in x_fit.columns:
                if col in ols_fit.params.index:
                    ols_map[col] = float(ols_fit.params[col])
            for i, name in enumerate(param_names):
                if name in ols_map:
                    start_params[i] = ols_map[name]
        except Exception:
            pass

    ar_overrides = {} if ar_overrides is None else dict(ar_overrides)
    ma_overrides = {} if ma_overrides is None else dict(ma_overrides)
    for i, name in enumerate(param_names):
        if name.startswith('ar.L'):
            lag = int(name.split('ar.L')[-1])
            if lag in ar_overrides:
                start_params[i] = float(ar_overrides[lag])
        elif name.startswith('ma.L'):
            lag = int(name.split('ma.L')[-1])
            if lag in ma_overrides:
                start_params[i] = float(ma_overrides[lag])
        elif name == 'sigma2' and sigma2_override is not None:
            start_params[i] = float(sigma2_override)

    return start_params


def _fit_armax(y, X_exog, order, maxiter=None, manual_start_params=None):
    """Fit ARMAX with QML robust standard errors. Raises on failure."""
    if maxiter is None:
        maxiter = ARMAX_MAXITER
    fit_kwargs = {
        'method': ARMAX_SOLVER,
        'method_kwargs': {'maxiter': maxiter},
        'cov_type': 'robust'
    }
    if manual_start_params is not None:
        fit_kwargs['start_params'] = np.asarray(manual_start_params, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=ValueWarning)
        warnings.filterwarnings("ignore", category=UserWarning,
                                module=r"statsmodels\.tsa\.statespace\.sarimax")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Non-stationary starting autoregressive parameters found.*")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Non-invertible starting MA parameters found.*")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="A date index has been provided, but it has no associated frequency information.*")
        return sm.tsa.ARIMA(y, exog=X_exog, order=order).fit(**fit_kwargs)


def _print_armax_results(result):
    """Print ARMAX coefficient table with p-values."""
    SEP = "-" * 52
    print(f"\nARMAX{result.model.order}")
    print(SEP)
    print(f"  {'Parameter':<28} {'Coef':>10}   {'p-value':>8}  Sig.")
    print(SEP)
    for name, coef, pval in zip(result.param_names, result.params, result.pvalues):
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {name:<28} {coef:>10.5f}   {pval:>8.4f}  {stars}")
    print(SEP)
    print(f"  {'N':<28} {int(result.nobs):>10}")
    labels = getattr(result.model.data, 'row_labels', None)
    if labels is not None and len(labels) > 0:
        print(f"  {'Sample':<28} {str(labels[0])[:10]} to {str(labels[-1])[:10]}")
    print(f"  {'Log-likelihood':<28} {result.llf:>10.3f}")
    print(f"  {'AIC':<28} {result.aic:>10.3f}")
    print(f"  {'BIC':<28} {result.bic:>10.3f}")
    print(SEP)
    print("  *** p<0.01  ** p<0.05  * p<0.10")


def run_armax_same_order_diagnostic(y, X_exog, order, zone='SE1',
                                    solver='statespace', base_maxiter=300,
                                    results_dir='results'):
    """
    Re-fit the same ARMAX order under a small set of optimizer variants to
    diagnose whether non-convergence looks like an iteration-limit issue,
    a start-value issue, or a broader numerical instability.
    """
    p, d, q = order
    print("\n" + "="*80)
    print("ARMAX SAME-ORDER CONVERGENCE DIAGNOSTIC")
    print("="*80)
    print(f"Zone: {zone}")
    print(f"Order under diagnosis: ARMAX{order}")

    high_maxiter = max(int(base_maxiter), 1000)
    variants = [
        {
            'variant': 'current_settings',
            'solver': solver,
            'maxiter': int(base_maxiter),
        }
    ]
    if high_maxiter > int(base_maxiter):
        variants.append({
            'variant': 'higher_maxiter',
            'solver': solver,
            'maxiter': high_maxiter,
        })

    rows = []
    for spec in variants:
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog,
            order=order,
            context_label=f"same_order_diag {spec['variant']} ({zone})",
            accept_nonconverged=True,
            maxiter=spec['maxiter'],
            solver=spec['solver'],
        )
        diag = fit.get('diagnostics', {}) or {}
        model = fit.get('model')
        aic = float(getattr(model, 'aic', np.nan)) if model is not None else np.nan
        bic = float(getattr(model, 'bic', np.nan)) if model is not None else np.nan
        ar_root_min_abs = np.nan
        ma_root_min_abs = np.nan
        try:
            arroots = getattr(model, 'arroots', None)
            if arroots is not None and len(arroots) > 0:
                ar_root_min_abs = float(np.min(np.abs(arroots)))
        except Exception:
            pass
        try:
            maroots = getattr(model, 'maroots', None)
            if maroots is not None and len(maroots) > 0:
                ma_root_min_abs = float(np.min(np.abs(maroots)))
        except Exception:
            pass

        row = {
            'variant': spec['variant'],
            'solver': spec['solver'],
            'maxiter_requested': spec['maxiter'],
            'ok': fit.get('ok', False),
            'converged': fit.get('converged', False),
            'iterations': diag.get('iterations'),
            'warnflag': diag.get('warnflag'),
            'gradient_max_abs': diag.get('gradient_max_abs'),
            'optimizer_task': diag.get('optimizer_task'),
            'start_params_source': diag.get('start_params_source'),
            'start_params_max_abs': diag.get('start_params_max_abs'),
            'aic': aic,
            'bic': bic,
            'ar_root_min_abs': ar_root_min_abs,
            'ma_root_min_abs': ma_root_min_abs,
            'error': fit.get('error'),
            'diagnosis_label': '',
            'diagnosis_why': ''
        }
        label, why = _diagnose_nonconvergence_simple(
            {
                'ok': fit.get('ok', False),
                'converged': fit.get('converged', False),
                'diagnostics': diag,
                'fail_reason': fit.get('fail_reason'),
                'error': fit.get('error')
            },
            configured_maxiter=spec['maxiter']
        )
        row['diagnosis_label'] = label
        row['diagnosis_why'] = why
        rows.append(row)

    diag_df = pd.DataFrame(rows)

    print("\nVariant results:")
    for _, row in diag_df.iterrows():
        print(
            f"  {row['variant']}: conv={row['converged']} iter={row['iterations']} "
            f"warn={row['warnflag']} grad={row['gradient_max_abs'] if pd.notna(row['gradient_max_abs']) else np.nan:.3e} "
            f"maxiter={row['maxiter_requested']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"diag={row['diagnosis_label']}"
        )
        if pd.notna(row['ar_root_min_abs']):
            print(f"    AR root min abs: {row['ar_root_min_abs']:.6f}")
        if pd.notna(row['ma_root_min_abs']):
            print(f"    MA root min abs: {row['ma_root_min_abs']:.6f}")
        if row['optimizer_task']:
            print(f"    Optimizer task: {row['optimizer_task']}")

    converged_mask = diag_df['converged'] == True
    if converged_mask.any():
        best_conv = diag_df.loc[converged_mask].sort_values('bic').iloc[0]
        print("\nDiagnostic read:")
        print(
            f"  A converged fit exists for the same order under variant '{best_conv['variant']}'. "
            f"This points to an optimizer setup / starting value issue rather than the order itself."
        )
    else:
        print("\nDiagnostic read:")
        if (diag_df['iterations'].fillna(0) < diag_df['maxiter_requested']).all():
            print("  All variants stopped well before the iteration cap. This does not look like a maxiter problem.")
        if diag_df['gradient_max_abs'].dropna().gt(1e-3).any():
            print("  At least one variant ended with a materially non-zero gradient, consistent with numerical instability.")
        if diag_df['ar_root_min_abs'].dropna().between(1.0, 1.05, inclusive='both').any():
            print("  The AR root is very close to the unit-circle boundary in at least one variant, which can make MLE unstable.")

    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)
    csv_path = os.path.join(zone_results_dir, f'armax_same_order_diagnostic_{zone}_{p}_{d}_{q}.csv')
    diag_df.to_csv(csv_path, index=False)
    print(f"Saved same-order diagnostic table: {csv_path}")

    return diag_df


def run_armax_fixed_order_exog_sweep(y, X_exog, order, zone='SE1',
                                     solver='statespace', maxiter=300,
                                     results_dir='results'):
    """
    Diagnose same-order convergence by incrementally adding exogenous variables.
    This helps separate AR/MA-order instability from instability introduced by
    specific controls.
    """
    print("\n" + "="*80)
    print("ARMAX FIXED-ORDER EXOG SWEEP")
    print("="*80)
    print(f"Zone: {zone}")
    print(f"Fixed order: ARMAX{order}")

    stages = _build_staged_armax_exog_sequence(list(X_exog.columns))
    if not stages:
        print("No exogenous variables available for sweep.")
        return pd.DataFrame()

    rows = []
    total = len(stages)
    for idx, cols in enumerate(stages, start=1):
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog[cols],
            order=order,
            context_label=f"fixed_order_exog_sweep stage={idx} ({zone})",
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
        )
        diag = fit.get('diagnostics', {}) or {}
        model = fit.get('model')
        aic = float(getattr(model, 'aic', np.nan)) if model is not None else np.nan
        bic = float(getattr(model, 'bic', np.nan)) if model is not None else np.nan
        ar_root_min_abs = np.nan
        ma_root_min_abs = np.nan
        try:
            arroots = getattr(model, 'arroots', None)
            if arroots is not None and len(arroots) > 0:
                ar_root_min_abs = float(np.min(np.abs(arroots)))
        except Exception:
            pass
        try:
            maroots = getattr(model, 'maroots', None)
            if maroots is not None and len(maroots) > 0:
                ma_root_min_abs = float(np.min(np.abs(maroots)))
        except Exception:
            pass

        label, why = _diagnose_nonconvergence_simple(
            {
                'ok': fit.get('ok', False),
                'converged': fit.get('converged', False),
                'diagnostics': diag,
                'fail_reason': fit.get('fail_reason'),
                'error': fit.get('error')
            },
            configured_maxiter=maxiter
        )

        row = {
            'stage': idx,
            'n_exog': len(cols),
            'added_var': cols[-1],
            'exog_vars': " | ".join(cols),
            'ok': fit.get('ok', False),
            'converged': fit.get('converged', False),
            'iterations': diag.get('iterations'),
            'warnflag': diag.get('warnflag'),
            'gradient_max_abs': diag.get('gradient_max_abs'),
            'optimizer_task': diag.get('optimizer_task'),
            'aic': aic,
            'bic': bic,
            'ar_root_min_abs': ar_root_min_abs,
            'ma_root_min_abs': ma_root_min_abs,
            'diagnosis_label': label,
            'diagnosis_why': why,
            'error': fit.get('error')
        }
        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    print("\nStage results:")
    for _, row in sweep_df.iterrows():
        print(
            f"  Stage {int(row['stage'])}/{total}: added={row['added_var']} "
            f"conv={row['converged']} iter={row['iterations']} warn={row['warnflag']} "
            f"grad={row['gradient_max_abs'] if pd.notna(row['gradient_max_abs']) else np.nan:.3e} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"diag={row['diagnosis_label']}"
        )
        if pd.notna(row['ar_root_min_abs']):
            print(f"    AR root min abs: {row['ar_root_min_abs']:.6f}")
        if row['optimizer_task']:
            print(f"    Optimizer task: {row['optimizer_task']}")

    print("\nSweep read:")
    if (sweep_df['converged'] == True).all():
        print("  All staged exogenous sets converged. The baseline failure is not reproduced in this sweep.")
    else:
        first_fail = sweep_df[sweep_df['converged'] != True].iloc[0]
        print(
            f"  First non-converged stage: {int(first_fail['stage'])} after adding {first_fail['added_var']} "
            f"(n_exog={int(first_fail['n_exog'])})."
        )
        earlier = sweep_df[sweep_df['stage'] < first_fail['stage']]
        if not earlier.empty and (earlier['converged'] == True).all():
            print("  Earlier smaller exogenous sets converged, so the added controls likely worsen conditioning.")
        elif first_fail['stage'] == 1:
            print("  The order is already unstable even with the smallest exogenous set, so the core AR structure is implicated.")
        if sweep_df['ar_root_min_abs'].dropna().between(1.0, 1.05, inclusive='both').any():
            print("  AR roots remain close to the unit-circle boundary in at least one stage, consistent with persistence-driven instability.")

    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)
    p, d, q = order
    csv_path = os.path.join(zone_results_dir, f'armax_fixed_order_exog_sweep_{zone}_{p}_{d}_{q}.csv')
    sweep_df.to_csv(csv_path, index=False)
    print(f"Saved fixed-order exog sweep: {csv_path}")

    return sweep_df


def run_armax_manual_start_optimizer_diagnostic(y, X_exog, order, zone='SE1',
                                                solver='statespace', maxiter=300,
                                                results_dir='results'):
    """
    Test whether the same ARMAX order can be rescued by keeping all controls but
    changing the constrained AR/sigma start values and the inner optimizer.
    """
    print("\n" + "="*80)
    print("ARMAX MANUAL-START / OPTIMIZER DIAGNOSTIC")
    print("="*80)
    print(f"Zone: {zone}")
    print(f"Order: ARMAX{order}")

    sigma2_base = max(float(y.var()), 1e-6)
    trial_specs = [
        {'variant': 'bfgs_ar_0.50_sigma_var', 'optimizer_method': 'bfgs', 'ar1': 0.50, 'sigma2': sigma2_base},
        {'variant': 'bfgs_ar_0.80_sigma_var', 'optimizer_method': 'bfgs', 'ar1': 0.80, 'sigma2': sigma2_base},
        {'variant': 'bfgs_ar_0.95_sigma_var', 'optimizer_method': 'bfgs', 'ar1': 0.95, 'sigma2': sigma2_base},
        {'variant': 'powell_ar_0.50_sigma_var', 'optimizer_method': 'powell', 'ar1': 0.50, 'sigma2': sigma2_base},
        {'variant': 'powell_ar_0.80_sigma_var', 'optimizer_method': 'powell', 'ar1': 0.80, 'sigma2': sigma2_base},
        {'variant': 'powell_ar_0.95_sigma_var', 'optimizer_method': 'powell', 'ar1': 0.95, 'sigma2': sigma2_base},
    ]

    rows = []
    best_converged_fit = None
    best_converged_bic = None
    for spec in trial_specs:
        start_params = _build_manual_armax_start_params(
            y_fit=y,
            x_fit=X_exog,
            order=order,
            ar_overrides={1: spec['ar1']},
            sigma2_override=spec['sigma2'],
            exog_source='ols'
        )
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog,
            order=order,
            context_label=f"manual_start_diag {spec['variant']} ({zone})",
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
            trace_optimizer=False,
            optimizer_method=spec['optimizer_method'],
            manual_start_params=start_params
        )
        diag = fit.get('diagnostics', {}) or {}
        model = fit.get('model')
        aic = float(getattr(model, 'aic', np.nan)) if model is not None else np.nan
        bic = float(getattr(model, 'bic', np.nan)) if model is not None else np.nan
        est_ar1 = np.nan
        est_sigma2 = np.nan
        if model is not None:
            try:
                params_map = dict(zip(model.param_names, np.asarray(model.params)))
                est_ar1 = float(params_map.get('ar.L1', np.nan))
                est_sigma2 = float(params_map.get('sigma2', np.nan))
            except Exception:
                pass
        label, why = _diagnose_nonconvergence_simple(
            {
                'ok': fit.get('ok', False),
                'converged': fit.get('converged', False),
                'diagnostics': diag,
                'fail_reason': fit.get('fail_reason'),
                'error': fit.get('error')
            },
            configured_maxiter=maxiter
        )
        row = {
            'variant': spec['variant'],
            'optimizer_method': spec['optimizer_method'],
            'start_ar1': spec['ar1'],
            'start_sigma2': spec['sigma2'],
            'converged': fit.get('converged', False),
            'iterations': diag.get('iterations'),
            'warnflag': diag.get('warnflag'),
            'gradient_max_abs': diag.get('gradient_max_abs'),
            'aic': aic,
            'bic': bic,
            'est_ar1': est_ar1,
            'est_sigma2': est_sigma2,
            'diagnosis_label': label,
            'diagnosis_why': why,
            'error': fit.get('error')
        }
        rows.append(row)
        if fit.get('converged', False) and pd.notna(bic):
            if best_converged_bic is None or bic < best_converged_bic:
                best_converged_bic = bic
                best_converged_fit = {
                    'variant': spec['variant'],
                    'optimizer_method': spec['optimizer_method'],
                    'start_params': start_params.tolist(),
                    'fit': fit
                }

    diag_df = pd.DataFrame(rows)
    print("\nManual-start results:")
    for _, row in diag_df.iterrows():
        print(
            f"  {row['variant']}: conv={row['converged']} iter={row['iterations']} "
            f"warn={row['warnflag']} grad={row['gradient_max_abs'] if pd.notna(row['gradient_max_abs']) else np.nan:.3e} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"est_ar1={row['est_ar1'] if pd.notna(row['est_ar1']) else np.nan:.6f}"
        )

    if best_converged_fit is not None:
        print("\nDiagnostic read:")
        print(
            f"  A converged same-order fit was found using optimizer={best_converged_fit['optimizer_method']} "
            f"under variant '{best_converged_fit['variant']}'."
        )
        print("  This means the order with all controls is not intrinsically impossible; the default optimizer path is the problem.")
    else:
        print("\nDiagnostic read:")
        print("  No converged same-order fit was found in the manual-start / optimizer sweep.")

    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)
    p, d, q = order
    csv_path = os.path.join(zone_results_dir, f'armax_manual_start_optimizer_diag_{zone}_{p}_{d}_{q}.csv')
    diag_df.to_csv(csv_path, index=False)
    print(f"Saved manual-start diagnostic: {csv_path}")

    return diag_df, best_converged_fit


def _fit_same_order_bfgs_rescue(y, X_exog, order, zone='SE1',
                                solver='statespace', maxiter=300):
    """
    Retry the same ARMAX order with a more reliable optimizer setup:
    BFGS plus OLS-based exogenous starts and a moderate AR(1) start.
    """
    sigma2_base = max(float(y.var()), 1e-6)
    start_params = _build_manual_armax_start_params(
        y_fit=y,
        x_fit=X_exog,
        order=order,
        ar_overrides={1: 0.80} if int(order[0]) >= 1 else None,
        sigma2_override=sigma2_base,
        exog_source='ols'
    )
    fit = _fit_armax_with_controls(
        y=y,
        X_exog=X_exog,
        order=order,
        context_label=f"same_order_bfgs_rescue ({zone})",
        accept_nonconverged=True,
        maxiter=maxiter,
        solver=solver,
        trace_optimizer=True,
        optimizer_method='bfgs',
        manual_start_params=start_params
    )
    fit['selected_order'] = order
    fit['used_nonconverged'] = False
    fit['rescue_strategy'] = 'bfgs_manual_ols_start'
    fit['rescue_start_params'] = start_params.tolist()
    return fit


def _print_armax_fit_report(fit_result, order, label, trace_snapshots=(1, 5)):
    """Print a standardized ARMAX fit report for one optimizer path."""
    diag = fit_result.get('diagnostics', {}) or {}
    model = fit_result.get('model')

    print("\n" + "="*80)
    print(label)
    print("="*80)
    if not fit_result.get('converged', False):
        print("Status:           did not converge")
    else:
        print("Status:           converged")
    print("="*80)

    if model is not None:
        if not fit_result.get('converged', False):
            print("NOTE: Estimates below come from a non-converged fit and are shown for diagnosis only.")
        print()
        print(model.summary())


def _fit_armax_with_fallback(y, X_exog, primary_order=(3, 0, 3), context_label="",
                             allow_nonconverged=False, maxiter=300, solver='statespace',
                             enable_fallback_orders=True,
                             fallback_orders=None, trace_primary_optimizer=False):
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

    for idx, order in enumerate(orders):
        fit = _fit_armax_with_controls(
            y=y,
            X_exog=X_exog,
            order=order,
            context_label=context_label,
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
            trace_optimizer=bool(trace_primary_optimizer and idx == 0)
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
    optimizer_task = diag.get('optimizer_task', None)

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
            f"Optimizer ended with warnflag={warnflag}; task={optimizer_task!r}. "
            "Convergence likely blocked by numerical instability or an ill-conditioned likelihood surface."
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


def _print_optimizer_trace_snapshots(diag, snapshot_iters=(1, 5)):
    """Print selected optimizer parameter vectors from a traced fit."""
    param_names = diag.get('param_names') or []
    start_vec = diag.get('start_params_vector') or []
    trace = diag.get('iteration_trace') or []

    if not param_names or not start_vec:
        print("Optimizer trace: start vector unavailable.")
        return

    def _print_vector(label, values):
        print(f"{label}:")
        for name, value in zip(param_names, values):
            print(f"  {name:<35} {value: .6f}")

    print("\nOptimizer parameter trace:")
    _print_vector("  Start vector", start_vec)

    if not trace:
        print("  No per-iteration trace captured.")
        return

    trace_len = len(trace)
    for snap in snapshot_iters:
        if 1 <= int(snap) <= trace_len:
            _print_vector(f"  Iteration {int(snap)}", trace[int(snap) - 1])

    if trace_len > 0:
        _print_vector(f"  Last recorded iterate (iteration {trace_len})", trace[-1])


def _build_staged_armax_exog_sequence(exog_vars):
    """
    Build staged exogenous-variable sets for convergence diagnostics.
    Sequence: Wind -> Hydro -> Net Exchange -> Consumption -> Oil -> Gas -> BNECK dummies.
    """
    if not exog_vars:
        return []

    preferred_order = [
        'Wind_Forecast_Log',
        'Hydro_Reserves_Log_Deseasonalized',
        'Net_Exchange',
        'Consumption_Log_Deseasonalized',
        'Oil_Price_Log_Deseasonalized',
        'Gas_Price_Log_Deseasonalized'
    ]

    selected = []
    seen = set()
    for name in preferred_order:
        if name in exog_vars and name not in seen:
            selected.append(name)
            seen.add(name)

    for name in exog_vars:
        if name.startswith('BNECK_') and name not in seen:
            selected.append(name)
            seen.add(name)

    for name in exog_vars:
        if name not in seen:
            selected.append(name)
            seen.add(name)

    stages = []
    running = []
    for name in selected:
        running.append(name)
        stages.append(running.copy())

    return stages


def run_armax_staged_convergence_diagnostic(Y, df, exog_vars, zone='SE1',
                                            order=(1, 0, 1), maxiter=300,
                                            solver='statespace'):
    """
    Diagnose ARMAX convergence by incrementally adding exogenous variables.
    Prints stage-by-stage diagnostics to terminal.
    """
    print("\n" + "="*80)
    print("ARMAX STAGED CONVERGENCE DIAGNOSTIC")
    print("="*80)
    print(f"Zone: {zone}")
    print(f"Fixed order: ARMAX{order}")
    print(f"Solver/maxiter: {solver}/{maxiter}")

    stages = _build_staged_armax_exog_sequence(exog_vars)
    if not stages:
        print("No exogenous variables available for staged diagnostic.")
        return pd.DataFrame()

    rows = []
    total = len(stages)
    for i, stage_vars in enumerate(stages, start=1):
        stage_label = f"Stage {i}/{total}"
        X_stage = df[stage_vars]
        fit = _fit_armax_with_controls(
            y=Y,
            X_exog=X_stage,
            order=order,
            context_label=f"staged_{zone}_{i}",
            accept_nonconverged=True,
            maxiter=maxiter,
            solver=solver,
        )

        diag = fit.get('diagnostics', {}) or {}
        converged = bool(fit.get('converged', False))
        status = 'exception'
        if fit.get('ok', False):
            status = 'converged' if converged else 'nonconverged'

        aic = np.nan
        bic = np.nan
        beta_wind = np.nan
        se_wind = np.nan
        if fit.get('ok', False) and fit.get('model') is not None:
            model = fit['model']
            aic = getattr(model, 'aic', np.nan)
            bic = getattr(model, 'bic', np.nan)
            try:
                beta_wind, se_wind, _ = _extract_armax_wind_coef(model, 'Wind_Forecast_Log')
            except Exception:
                beta_wind, se_wind = np.nan, np.nan

        if status == 'nonconverged':
            label, why = _diagnose_nonconvergence_simple(
                {'ok': True, 'converged': False, 'diagnostics': diag, 'fail_reason': None, 'error': None},
                configured_maxiter=maxiter
            )
        elif status == 'exception':
            label, why = _diagnose_nonconvergence_simple(
                {'ok': False, 'fail_reason': 'exception', 'error': fit.get('error'), 'diagnostics': diag},
                configured_maxiter=maxiter
            )
        else:
            label, why = 'converged', 'Converged successfully.'

        iterations = diag.get('iterations', None)
        warnflag = diag.get('warnflag', None)
        grad = diag.get('gradient_max_abs', None)

        print(
            f"{stage_label} | k={len(stage_vars):2d} | status={status:12s} "
            f"| conv={converged} | iter={iterations} | warn={warnflag} | "
            f"grad={grad if grad is not None else np.nan:.4f} | "
            f"AIC={aic if pd.notna(aic) else np.nan:.2f} | "
            f"beta_wind={beta_wind if pd.notna(beta_wind) else np.nan:.4f}"
        )
        print(f"  vars={stage_vars}")
        if status != 'converged':
            print(f"  diagnosis={label} | {why}")

        rows.append({
            'stage': i,
            'n_exog': len(stage_vars),
            'vars': ', '.join(stage_vars),
            'status': status,
            'converged': converged,
            'iterations': iterations,
            'warnflag': warnflag,
            'gradient_max_abs': grad,
            'aic': aic,
            'bic': bic,
            'beta_wind': beta_wind,
            'se_wind': se_wind,
            'diagnosis_label': label,
            'diagnosis_why': why
        })

    out_df = pd.DataFrame(rows)
    first_fail = out_df[out_df['status'] != 'converged']
    print("\n--- STAGED DIAGNOSTIC SUMMARY ---")
    print(f"Total stages: {len(out_df)}")
    print(f"Converged stages: {(out_df['status'] == 'converged').sum()}")
    if first_fail.empty:
        print("First failing stage: none (all stages converged)")
    else:
        ff = first_fail.iloc[0]
        print(
            f"First failing stage: {int(ff['stage'])} "
            f"(k={int(ff['n_exog'])}, status={ff['status']}, diagnosis={ff['diagnosis_label']})"
        )
    print("="*80)

    return out_df


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



def _evaluate_armax_candidate(Y, exog_vars, p, q,
                            ljungbox_lags=(5, 10, 15, 20),
                            maxiter=300,
                            solver='statespace',
                            max_p_for_schema=10,
                            max_q_for_schema=10):
    """Evaluate a single ARMAX(p,0,q) candidate without fallback."""
    order = (p, 0, q)

    def _sig_stars(p_value):
        if p_value is None or pd.isna(p_value):
            return ""
        if p_value < 0.01:
            return "***"
        if p_value < 0.05:
            return "**"
        if p_value < 0.10:
            return "*"
        return ""

    result = {
        'p': p,
        'q': q,
        'status': 'exception',
        'converged': False,
        'aic': np.nan,
        'bic': np.nan,
        'beta_wind': np.nan,
        'wind_se': np.nan,
        'wind_stars': '',
        'passes_ljungbox': False,
    }

    for lag in ljungbox_lags:
        result[f'ljungbox_lag_{lag}_stat'] = np.nan
        result[f'ljungbox_lag_{lag}_pval'] = np.nan

    max_p_for_schema = int(max(0, max_p_for_schema))
    max_q_for_schema = int(max(0, max_q_for_schema))
    for i in range(1, max_p_for_schema + 1):
        result[f'ar{i}_coef'] = np.nan
        result[f'ar{i}_se'] = np.nan
        result[f'ar{i}_stars'] = ''
    for i in range(1, max_q_for_schema + 1):
        result[f'ma{i}_coef'] = np.nan
        result[f'ma{i}_se'] = np.nan
        result[f'ma{i}_stars'] = ''

    fit = _fit_armax_with_controls(
        y=Y,
        X_exog=exog_vars,
        order=order,
        context_label=f"lag_search ARMAX({p},{q})",
        accept_nonconverged=True,
        maxiter=maxiter,
        solver=solver,
    )

    if not fit['ok']:
        result['status'] = 'exception'
        return result

    model = fit['model']
    diag = fit.get('diagnostics', {}) or {}
    converged = bool(fit.get('converged', False))

    result['converged'] = converged
    result['status'] = 'converged' if converged else 'nonconverged'
    result['aic'] = float(getattr(model, 'aic', np.nan))
    result['bic'] = float(getattr(model, 'bic', np.nan))

    try:
        beta_wind, se_wind, p_wind = _extract_armax_wind_coef(model, 'Wind_Forecast_Log')
        result['beta_wind'] = float(beta_wind)
        result['wind_se'] = float(se_wind)
        result['wind_stars'] = _sig_stars(p_wind)
    except Exception:
        pass

    try:
        param_names = list(model.param_names)
        params = np.asarray(model.params)
        bse = np.asarray(model.bse)
        pvals = np.asarray(model.pvalues)
        for idx, name in enumerate(param_names):
            if name.startswith('ar.L'):
                lag_str = name.split('ar.L')[-1]
                lag = int(lag_str)
                if 1 <= lag <= max_p_for_schema:
                    pval = float(pvals[idx])
                    result[f'ar{lag}_coef'] = float(params[idx])
                    result[f'ar{lag}_se'] = float(bse[idx])
                    result[f'ar{lag}_stars'] = _sig_stars(pval)
            elif name.startswith('ma.L'):
                lag_str = name.split('ma.L')[-1]
                lag = int(lag_str)
                if 1 <= lag <= max_q_for_schema:
                    pval = float(pvals[idx])
                    result[f'ma{lag}_coef'] = float(params[idx])
                    result[f'ma{lag}_se'] = float(bse[idx])
                    result[f'ma{lag}_stars'] = _sig_stars(pval)
    except Exception:
        pass

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
        result['passes_ljungbox'] = bool((lb_results['p_value'] >= 0.05).all())
    except Exception as e:
        result['passes_ljungbox'] = False
        pass

    return result


def _select_best_armax_candidate(results_df):
    """Select best candidate from evaluated grid without adding extra report columns."""
    if results_df.empty:
        return None, pd.DataFrame(), {
            'tested': 0,
            'converged': 0,
            'eligible': 0,
            'exceptions': 0,
            'nonconverged': 0
        }

    df = results_df.copy()
    mask = df['bic'].notna()
    mask &= (df['passes_ljungbox'] == True)
    eligible = df[mask].copy()

    if eligible.empty:
        best_order = None
    else:
        best_idx = eligible['bic'].idxmin()
        best_row = eligible.loc[best_idx]
        best_order = (int(best_row['p']), int(best_row['q']))

    top_pass_lb = (
        df[df['passes_ljungbox'] == True]
        .copy()
        .sort_values(by='bic', ascending=True)
        .head(3)
    )

    stats = {
        'tested': int(len(df)),
        'converged': int((df['converged'] == True).sum()),
        'eligible': int(len(eligible)),
        'exceptions': int((df['status'] == 'exception').sum()),
        'nonconverged': int((df['status'] == 'nonconverged').sum()),
        'passes_ljungbox': int((df['passes_ljungbox'] == True).sum()),
        'top_pass_ljungbox': top_pass_lb.to_dict('records')
    }

    return best_order, df, stats


def _build_armax_search_grid(p_min, p_max, q_min, q_max):
    """Build grid list [(p,q), ...] for ARMAX search, excluding (0,0)."""
    grid = []
    for p in range(int(p_min), int(p_max) + 1):
        for q in range(int(q_min), int(q_max) + 1):
            if p == 0 and q == 0:
                continue
            grid.append((p, q))
    return grid


def _save_armax_search_reports(results_df, zone='SE1', results_dir='results'):
    """Persist the full ARMAX lag-search report to CSV and Excel."""
    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)

    all_path = os.path.join(zone_results_dir, f'armax_search_all_{zone}.csv')
    results_df.to_csv(all_path, index=False)
    all_xlsx_path = os.path.join(zone_results_dir, f'armax_search_all_{zone}.xlsx')
    try:
        results_df.to_excel(all_xlsx_path, index=False)
    except Exception as e:
        print(f"WARNING: Failed to write Excel full report: {e}")

    return all_path, all_xlsx_path


def select_armax_lags_aic(Y, exog_vars, zone='SE1',
                          p_min=0, p_max=10, q_min=0, q_max=10,
                          ljungbox_lags=(5, 10, 15, 20),
                          maxiter=300,
                          solver='statespace',
                          results_dir='results'):
    """Strict ARMAX lag search without fallback; returns best order and full results."""
    p_min_eff = int(p_min)
    q_min_eff = int(q_min)
    p_max_eff = int(p_max)
    q_max_eff = int(q_max)
    if p_min_eff < 0 or q_min_eff < 0 or p_max_eff < 0 or q_max_eff < 0:
        raise ValueError(
            f"Invalid ARMAX grid: lag bounds must be non-negative, got p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )
    if p_min_eff > p_max_eff or q_min_eff > q_max_eff:
        raise ValueError(
            f"Invalid ARMAX grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )

    grid = _build_armax_search_grid(p_min_eff, p_max_eff, q_min_eff, q_max_eff)
    max_p_schema = int(max(0, p_max_eff))
    max_q_schema = int(max(0, q_max_eff))

    print("\n--- ARMAX LAG SELECTION (STRICT GRID SEARCH) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}, excluding (0,0)")
    print(f"Ljung-Box pass rule: all configured Ljung-Box p-values >= 0.05 at lags {list(ljungbox_lags)}")
    print("Selection criterion: BIC")
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
            max_p_for_schema=max_p_schema,
            max_q_for_schema=max_q_schema
        )
        rows.append(row)
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
        )

    results_df = pd.DataFrame(rows)
    best_order, results_df, stats = _select_best_armax_candidate(results_df)
    all_path, all_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        results_dir=results_dir
    )

    if stats['top_pass_ljungbox']:
        print("\nTop 3 models passing Ljung-Box, ranked by BIC:")
        for rank, row in enumerate(stats['top_pass_ljungbox'], start=1):
            print(
                f"  {rank}. ARMAX({int(row['p'])},{int(row['q'])}) | "
                f"status={row['status']} | conv={row['converged']} | "
                f"BIC={row['bic']:.2f} | AIC={row['aic']:.2f}"
            )
    else:
        print("\nNo tested models passed the Ljung-Box screen.")

    if best_order is None:
        print("No admissible ARMAX order found.")
    else:
        print(f"Selected order: ARMAX{best_order} by BIC")

    return best_order, results_df


def select_armax_lags_aic_checkpointed(Y, exog_vars, zone='SE1',
                                       p_min=0, p_max=10, q_min=0, q_max=10,
                                       checkpoint_file=None,
                                       ljungbox_lags=(5, 10, 15, 20),
                                       save_interval=1,
                                       maxiter=300,
                                       solver='statespace',
                                       results_dir='results'):
    """Checkpointed strict ARMAX lag search (no fallback)."""
    zone_results_dir = os.path.join(results_dir, zone)
    os.makedirs(zone_results_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(zone_results_dir, f'armax_lag_selection_checkpoint_{zone}.csv')

    p_min_eff = int(p_min)
    q_min_eff = int(q_min)
    p_max_eff = int(p_max)
    q_max_eff = int(q_max)
    if p_min_eff < 0 or q_min_eff < 0 or p_max_eff < 0 or q_max_eff < 0:
        raise ValueError(
            f"Invalid ARMAX grid: lag bounds must be non-negative, got p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )
    if p_min_eff > p_max_eff or q_min_eff > q_max_eff:
        raise ValueError(
            f"Invalid ARMAX grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}."
        )

    grid = _build_armax_search_grid(p_min_eff, p_max_eff, q_min_eff, q_max_eff)
    max_p_schema = int(max(0, p_max_eff))
    max_q_schema = int(max(0, q_max_eff))
    print("\n--- ARMAX LAG SELECTION (CHECKPOINTED, STRICT) ---")
    print(f"Zone: {zone}")
    print(f"Grid: p={p_min_eff}..{p_max_eff}, q={q_min_eff}..{q_max_eff}, excluding (0,0)")
    print(f"Ljung-Box pass rule: all configured Ljung-Box p-values >= 0.05 at lags {list(ljungbox_lags)}")
    print("Selection criterion: BIC")

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
            max_p_for_schema=max_p_schema,
            max_q_for_schema=max_q_schema
        )
        print(
            f"status={row['status']} conv={row['converged']} "
            f"AIC={row['aic'] if pd.notna(row['aic']) else np.nan:.2f} "
            f"BIC={row['bic'] if pd.notna(row['bic']) else np.nan:.2f} "
            f"LB={'PASS' if row['passes_ljungbox'] else 'FAIL'} "
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

    best_order, results_df, stats = _select_best_armax_candidate(checkpoint_df.copy())
    all_path, all_xlsx_path = _save_armax_search_reports(
        results_df,
        zone=zone,
        results_dir=results_dir
    )

    if stats['top_pass_ljungbox']:
        print("\nTop 3 models passing Ljung-Box, ranked by BIC:")
        for rank, row in enumerate(stats['top_pass_ljungbox'], start=1):
            print(
                f"  {rank}. ARMAX({int(row['p'])},{int(row['q'])}) | "
                f"status={row['status']} | conv={row['converged']} | "
                f"BIC={row['bic']:.2f} | AIC={row['aic']:.2f}"
            )
    else:
        print("\nNo tested models passed the Ljung-Box screen.")

    if best_order is None:
        print("No admissible ARMAX order found.")
    else:
        print(f"Selected order: ARMAX{best_order} by BIC")

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


def run_ols_variable_inclusion_analysis(df, zone, target_region='SE1', results_dir='results'):
    """
    Run a nested OLS variable-inclusion diagnostic using a fixed theory-driven block order.

    The diagnostic always tests the full candidate ladder on a common complete-case sample
    and prints a console summary before returning structured results.
    """
    del results_dir  # Reserved for interface consistency; console output only in this version.

    y_name = 'Price_DS'
    block_specs = [
        ('wind', ['Wind_Forecast_Log']),
        ('hydro', ['Hydro_Reserves_Log_Deseasonalized']),
        ('exchange', ['Net_Exchange']),
        ('demand', ['Consumption_Log_Deseasonalized']),
        ('oil', ['Oil_Price_Log_Deseasonalized']),
        ('gas', ['Gas_Price_Log_Deseasonalized']),
    ]

    bottleneck_cols = []
    for partner in TRADING_PARTNERS.get(target_region, []):
        col = f'BNECK_{target_region}_{partner}'
        if col in df.columns:
            bottleneck_cols.append(col)

    block_specs.append(('bottlenecks', bottleneck_cols))

    if y_name not in df.columns:
        raise ValueError(
            f"Required dependent variable '{y_name}' not found for OLS inclusion diagnostic."
        )
    if 'Wind_Forecast_Log' not in df.columns:
        raise ValueError(
            "Required core regressor 'Wind_Forecast_Log' not found for OLS inclusion diagnostic."
        )

    missing_columns = {}
    available_blocks = []
    union_regressors = []

    for block_name, expected_cols in block_specs:
        available_cols = [col for col in expected_cols if col in df.columns]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            missing_columns[block_name] = missing_cols
        if available_cols:
            available_blocks.append({'name': block_name, 'columns': available_cols})
            union_regressors.extend(available_cols)

    sample_cols = [y_name] + union_regressors
    sample_df = df[sample_cols].dropna().copy()

    def _fmt_num(value, fmt):
        return format(value, fmt) if pd.notna(value) else "-"

    def _fmt_pvalue(value):
        return f"{value:.3g}" if pd.notna(value) else "-"

    def _wrap_cols(columns, indent="    ", width=88):
        if not columns:
            return [f"{indent}-"]
        lines = []
        current = indent
        for col in columns:
            token = col if current == indent else f", {col}"
            if len(current) + len(token) > width and current != indent:
                lines.append(current)
                current = indent + col
            else:
                current += token
        lines.append(current)
        return lines

    print("\n" + "=" * 80)
    print(f"OLS VARIABLE INCLUSION DIAGNOSTIC ({zone})")
    print("=" * 80)
    print(f"Dependent variable: {y_name}")
    print("Purpose: evaluate incremental explanatory value of theory-driven control blocks")
    print("Decision guide:")
    print("  - Lower partial F-test p-value: newly added block contributes jointly")
    print("  - Higher adjusted R2 is better")
    print("  - Lower AIC/BIC is better")
    print("  - Large wind-coefficient changes indicate substantive sensitivity")
    print("All models use the same complete-case sample.")
    print("Gas and bottleneck blocks are tested regardless of INCLUDE_GAS_PRICE / INCLUDE_BOTTLENECKS.")

    if missing_columns:
        print("\nUnavailable candidate columns detected:")
        for block_name, cols in missing_columns.items():
            print(f"  {block_name}: missing {cols}")

    if not bottleneck_cols:
        print(f"\nNo bottleneck columns available for zone {target_region}; bottleneck block will be skipped.")

    print(f"\nCommon sample size: {len(sample_df):,}")
    print(f"Block order: {[block['name'] for block in available_blocks]}")

    result = {
        'zone': zone,
        'dependent_variable': y_name,
        'sample_n': int(len(sample_df)),
        'available_blocks': [block['name'] for block in available_blocks],
        'models': []
    }

    if sample_df.empty:
        print("No observations remain after dropping missing values on the full candidate set.")
        print("=" * 80)
        return result

    Y = sample_df[y_name]
    included_blocks = []
    regressors = []
    previous_model = None
    previous_wind_coef = None

    print("\n--- NESTED MODEL RESULTS ---")

    for model_index, block in enumerate(available_blocks, start=1):
        included_blocks.append(block['name'])
        regressors.extend(block['columns'])
        model_id = f"M{model_index}"
        model_label = f"M{model_index}: {' + '.join(included_blocks)}"

        try:
            X = sm.add_constant(sample_df[regressors].astype(float))
            model = sm.OLS(Y, X).fit()
        except Exception as e:
            print(f"\n{model_label}")
            print(f"  ERROR: failed to fit model after adding block '{block['name']}': {type(e).__name__}: {e}")
            print("=" * 80)
            return result

        wind_coef = model.params.get('Wind_Forecast_Log', np.nan)
        wind_se = model.bse.get('Wind_Forecast_Log', np.nan)
        wind_p = model.pvalues.get('Wind_Forecast_Log', np.nan)

        delta_wind = np.nan
        delta_wind_pct = np.nan
        partial_f_stat = np.nan
        partial_f_pvalue = np.nan
        partial_f_df_num = np.nan
        partial_f_df_den = np.nan

        if previous_model is not None:
            try:
                partial_f_stat, partial_f_pvalue, partial_f_df_den = model.compare_f_test(previous_model)
                partial_f_df_num = float(model.df_model - previous_model.df_model)
            except Exception:
                partial_f_stat = np.nan
                partial_f_pvalue = np.nan
                partial_f_df_num = np.nan
                partial_f_df_den = np.nan

            if pd.notna(previous_wind_coef) and pd.notna(wind_coef):
                delta_wind = float(wind_coef - previous_wind_coef)
                if abs(previous_wind_coef) > 1e-12:
                    delta_wind_pct = float((delta_wind / abs(previous_wind_coef)) * 100.0)

        model_info = {
            'model_id': model_id,
            'model_label': model_label,
            'added_block': block['name'],
            'included_blocks': included_blocks.copy(),
            'regressors': regressors.copy(),
            'nobs': int(model.nobs),
            'k_exog': int(len(regressors)),
            'adj_r2': float(model.rsquared_adj),
            'aic': float(model.aic),
            'bic': float(model.bic),
            'wind_coef': float(wind_coef) if pd.notna(wind_coef) else np.nan,
            'wind_se': float(wind_se) if pd.notna(wind_se) else np.nan,
            'wind_pvalue': float(wind_p) if pd.notna(wind_p) else np.nan,
            'delta_wind_coef': delta_wind,
            'delta_wind_pct': delta_wind_pct,
            'partial_f_stat': float(partial_f_stat) if pd.notna(partial_f_stat) else np.nan,
            'partial_f_pvalue': float(partial_f_pvalue) if pd.notna(partial_f_pvalue) else np.nan,
            'partial_f_df_num': float(partial_f_df_num) if pd.notna(partial_f_df_num) else np.nan,
            'partial_f_df_den': float(partial_f_df_den) if pd.notna(partial_f_df_den) else np.nan,
        }
        result['models'].append(model_info)

        print("\n" + "-" * 80)
        print(f"{model_id} | added block: {block['name']}")
        print(f"Blocks included: {' -> '.join(included_blocks)}")
        print("Regressors:")
        for line in _wrap_cols(regressors):
            print(line)
        print(
            f"Sample n={int(model.nobs):,} | k={len(regressors)} | "
            f"Adj R2={model.rsquared_adj:.6f} | AIC={model.aic:.3f} | BIC={model.bic:.3f}"
        )
        print(
            f"Wind: coef={_fmt_num(wind_coef, '.6f')} | "
            f"SE={_fmt_num(wind_se, '.6f')} | p={_fmt_pvalue(wind_p)}"
        )

        if previous_model is not None:
            delta_adj_r2 = model.rsquared_adj - previous_model.rsquared_adj
            delta_aic = model.aic - previous_model.aic
            delta_bic = model.bic - previous_model.bic
            print(
                f"Delta vs previous: dAdjR2={delta_adj_r2:+.6f} | "
                f"dAIC={delta_aic:+.3f} | dBIC={delta_bic:+.3f}"
            )
            if pd.notna(partial_f_stat):
                print(
                    f"Partial F-test: F={partial_f_stat:.4f} | p={_fmt_pvalue(partial_f_pvalue)} | "
                    f"df_num={partial_f_df_num:.0f} | df_den={partial_f_df_den:.0f}"
                )
            else:
                print("Partial F-test: unavailable")
            if pd.notna(delta_wind):
                if pd.notna(delta_wind_pct):
                    print(f"Wind change vs previous: {delta_wind:+.6f} ({delta_wind_pct:+.2f}%)")
                else:
                    print(f"Wind change vs previous: {delta_wind:+.6f}")

        previous_model = model
        previous_wind_coef = wind_coef

    print("\n--- COMPARISON SUMMARY ---")
    header = (
        f"{'Model':<5} {'Block':<13} {'AdjR2':>9} {'dAdjR2':>10} "
        f"{'AIC':>11} {'dAIC':>10} {'Wind':>11} {'Wind p':>10} {'F p':>10}"
    )
    print(header)
    print("-" * len(header))
    previous_row = None
    for row in result['models']:
        delta_adj_r2 = row['adj_r2'] - previous_row['adj_r2'] if previous_row is not None else np.nan
        delta_aic = row['aic'] - previous_row['aic'] if previous_row is not None else np.nan
        print(
            f"{row['model_id']:<5} {row['added_block']:<13} "
            f"{row['adj_r2']:>9.6f} {_fmt_num(delta_adj_r2, '+.6f'):>10} "
            f"{row['aic']:>11.3f} {_fmt_num(delta_aic, '+.3f'):>10} "
            f"{row['wind_coef']:>11.6f} {_fmt_pvalue(row['wind_pvalue']):>10} "
            f"{_fmt_pvalue(row['partial_f_pvalue']):>10}"
        )
        previous_row = row

    print("=" * 80)
    return result


def perform_multivariate_analysis(df, zone, target_region='SE1',
                                  run_ols=True, run_armax=True,
                                  run_ljungbox=False, run_hetero_tests=False, run_stationarity=False,
                                  run_collinearity_checks=False,
                                  controls=None,
                                 run_armax_staged_convergence=False,
                                 run_ols_variable_inclusion_diagnostic=False,
                                 armax_staged_order=(1, 0, 1),
                                 armax_staged_maxiter=300,
                                 armax_staged_solver='statespace',
                                  optimize_armax_lags=False, use_checkpointed_lag_selection=True,
                                  armax_search_p_min=0, armax_search_p_max=10,
                                  armax_search_q_min=0, armax_search_q_max=10,
                                 run_rolling_window=False, rolling_window_years=3,
                                 rolling_step_years=1, rolling_min_obs=24*180,
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

    y_name, exog_vars = get_regression_variable_names(
        df,
        target_region=target_region,
        controls=controls
    )
    Y = df[y_name]

    print(f"Dependent variable: {Y.name}")
    print(f"Controls: { {k: v for k, v in (controls or {}).items()} }")
    print(f"Exogenous variables: {exog_vars}")

    # Rolling-window mode: run rolling window analysis and return early
    if run_rolling_window:
        run_rolling_window_analysis(df, zone, Y, exog_vars,
                                    window_years=rolling_window_years,
                                    step_years=rolling_step_years,
                                    min_obs=rolling_min_obs,
                                    plots_dir="plots",
                                    results_dir="results",
                                    controls=controls)
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

    # OLS variable inclusion mode: run nested model comparisons and return early
    if run_ols_variable_inclusion_diagnostic:
        run_ols_variable_inclusion_analysis(
            df=df,
            zone=zone,
            target_region=target_region,
            results_dir='results'
        )
        return None, None, None  # Early return, skip standard OLS/ARMAX pipeline

    # Staged ARMAX convergence diagnostic mode: incrementally add exogenous controls
    if run_armax_staged_convergence:
        run_armax_staged_convergence_diagnostic(
            Y=Y,
            df=df,
            exog_vars=exog_vars,
            zone=zone,
            order=armax_staged_order,
            maxiter=armax_staged_maxiter,
            solver=armax_staged_solver,
        )
        return None, None, None  # Early return, skip standard OLS/ARMAX pipeline

    X = sm.add_constant(df[exog_vars])

    # 1. Standard OLS Regression
    ols_model = None
    if run_ols:
        ols_model = sm.OLS(Y, X).fit()
        print("\n--- OLS RESULTS ---")
        print(ols_model.summary())

    # Optional: Diagnostic tests on OLS residuals
    if run_ols and run_stationarity:
        # Test stationarity of ALL variables used in the regression
        print("\n" + "="*80)
        print("STATIONARITY TESTS FOR ALL REGRESSION VARIABLES")
        print("="*80)

        # Test dependent variable (Price)
        run_stationarity_tests(Y, series_name=f"{zone} {Y.name} (Dependent Variable)")

        # Test all independent variables
        for var in exog_vars:
            run_stationarity_tests(df[var], series_name=f"{zone} {var} (Independent Variable)")

    if run_ols and run_ljungbox:
        # Test for autocorrelation in OLS residuals
        run_ljungbox_test(ols_model.resid, lags=[5, 10, 15, 20])

    if run_ols and run_hetero_tests:
        # Test for heteroskedasticity and ARCH effects in OLS residuals
        run_heteroskedasticity_tests(ols_model.resid, nlags=10)

    if run_ols and run_collinearity_checks:
        run_collinearity_diagnostics(
            df=df,
            exog_vars=exog_vars,
            zone=zone,
            results_dir='results',
            vif_severe_threshold=10.0,
            top_corr_pairs=10
        )

    # 2. ARMAX(3,3)-GARCHX(1,1) Framework
    if not run_armax:
        return ols_model, None, None

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
                checkpoint_file=None,  # Auto-generate based on zone
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER
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
                maxiter=ARMAX_MAXITER,
                solver=ARMAX_SOLVER
            )
        if optimal_order is None:
            raise RuntimeError(
                "No admissible ARMAX order found in lag search. "
                "Check results/<ZONE>/armax_search_all_<ZONE>.csv and adjust the search range if needed."
            )
        armax_order = (optimal_order[0], 0, optimal_order[1])
        Y_armax = Y
        X_armax = df[exog_vars]
        added_lag_cols = []
        dropped_rows = 0
    else:
        armax_order = baseline_spec['order']
        extra_lags = baseline_spec.get('extra_ar_lags', [])
        spec_label = baseline_spec.get('label', f'ARMAX{armax_order}')

        Y_armax, X_armax, added_lag_cols = _prepare_baseline_armax_design(
            Y,
            df[exog_vars],
            extra_ar_lags=extra_lags,
            drop_initial_nan=True
        )
        dropped_rows = int(len(Y) - len(Y_armax))
        if len(added_lag_cols) > 0:
            print(f"Added lagged dependent regressors: {added_lag_cols}")

    # Export ARIMA input data to CSV for external use (e.g. Stata)
    _out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(_out_dir, exist_ok=True)
    _armax_input = pd.concat([Y_armax.rename(y_name), X_armax], axis=1)
    _armax_input.index.name = 'timestamp'
    # Rename to short Stata-friendly column names (max 32 chars, no ambiguity)
    _stata_rename = {
        'Price_DS':                        'price_ds',
        'Wind_Forecast_Log':               'wind_log',
        'Hydro_Reserves_Log_Deseasonalized': 'hydro_log_ds',
        'Net_Exchange':                    'net_exchange',
        'Consumption_Log_Deseasonalized':  'consump_log_ds',
        'Oil_Price_Log_Deseasonalized':    'oil_log_ds',
        'Gas_Price_Log_Deseasonalized':    'gas_log_ds',
    }
    _armax_input = _armax_input.rename(columns=_stata_rename)
    _start = str(df.index[0])[:10]
    _end = str(df.index[-1])[:10]
    _csv_path = os.path.join(_out_dir, f'armax_input_{zone}_{_start}_{_end}.csv')
    _armax_input.to_csv(_csv_path)
    print(f"Exported ARIMA input ({len(_armax_input)} obs) to {_csv_path}")
    # Export ARIMA order as companion metadata file (read by arima_from_python.do)
    _p, _d, _q = armax_order
    _extra_lags_nums = [int(c.replace('y_lag_', '')) for c in added_lag_cols]
    _extra_lags_str = ','.join(str(l) for l in _extra_lags_nums)
    pd.DataFrame({'arma_p': [_p], 'arma_d': [_d], 'arma_q': [_q],
                  'extra_ar_lags': [_extra_lags_str]}).to_csv(
        os.path.join(_out_dir, f'armax_meta_{zone}_{_start}_{_end}.csv'), index=False
    )

    # Mean Equation (Price Level)
    print(f"\n--- Fitting ARMAX{armax_order} model ---")

    armax_res = sm.tsa.ARIMA(Y_armax, exog=X_armax, order=armax_order).fit(cov_type='robust')
    _print_armax_results(armax_res)
    selected_order = armax_order
    armax_model_for_diagnostics = armax_res

    # Optional: Diagnostic tests on ARMAX residuals
    arch_detected = False
    should_run_arch_screen = bool(run_hetero_tests or FIT_GARCH_IF_ARCH)
    if armax_model_for_diagnostics is not None and run_ljungbox:
        print("\n" + "="*70)
        print("DIAGNOSTIC TESTS ON ARMAX RESIDUALS")
        print("="*70)
        run_ljungbox_test(armax_model_for_diagnostics.resid, lags=[5, 10, 15, 20])

    if armax_model_for_diagnostics is not None and should_run_arch_screen:
        # Run tests and check if ARCH effects detected
        run_heteroskedasticity_tests(armax_model_for_diagnostics.resid, nlags=10)

        # Check ARCH test result
        lm_stat, lm_pval, f_stat, f_pval = het_arch(armax_model_for_diagnostics.resid, nlags=10)
        if lm_pval < 0.05:
            arch_detected = True
            print(f"\n{'='*70}")
            print(f"ARCH EFFECTS DETECTED (p={lm_pval:.4f} < 0.05)")
            print("Proceeding with GARCH-X modeling...")
            print(f"{'='*70}")

    # GARCH-X component: Fit if ARCH effects confirmed
    garch_res = None
    if armax_model_for_diagnostics is not None and arch_detected and FIT_GARCH_IF_ARCH:
        # Always use logged wind variable
        wind_var = 'Wind_Forecast_Log'

        garch_res, garch_diagnostics = fit_garchx_model(
            armax_model_for_diagnostics.resid,
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
            print(f"ARMAX({selected_order[0]},{selected_order[2]}) AIC: {armax_model_for_diagnostics.aic:.2f}")
            print(f"ARMAX-GARCH({GARCH_ORDER[0]},{GARCH_ORDER[1]})-X AIC: {garch_res.aic:.2f}")
            improvement = armax_model_for_diagnostics.aic - garch_res.aic
            print(f"AIC Improvement: {improvement:.2f} {'(better)' if improvement > 0 else '(worse)'}")
            print(f"{'='*70}")
    elif should_run_arch_screen and not arch_detected:
        print(f"\n{'='*70}")
        print("NO ARCH EFFECTS DETECTED - GARCH modeling not necessary")
        print("ARMAX model is sufficient (constant variance assumption holds)")
        print(f"{'='*70}")

    return ols_model, armax_res, garch_res


# --- 6. EXECUTION BLOCK ---

if __name__ == "__main__":
    # --- REGION & DATE RANGE ---
    ACTIVE_ZONE = 'SE1'       # Options: 'SE1', 'SE2', 'SE3', 'SE4'
    START_DATE  = '2024-01-01'
    END_DATE    = '2025-12-31'
    PLOT_OUTLIERS = False  # Show outlier bounds on Price_DS before capping

    # --- DIAGNOSTICS ---
    SHOW_PREPROCESSING_STATISTICS = False  # Print detailed preprocessing output
    RUN_LJUNGBOX_TEST = False               # Ljung-Box autocorrelation test on residuals
    RUN_HETEROSKEDASTICITY_TESTS = False    # Engle ARCH + Ljung-Box on squared residuals
    RUN_STATIONARITY_TESTS = False         # ADF and DF-GLS unit root tests
    RUN_COLLINEARITY_DIAGNOSTICS = False   # VIF, correlations, condition number

    # --- CONTROLS (ordered by importance; set False to exclude from regression) ---
    CONTROLS = {
        'Wind':        True,   # Wind_Forecast_Log
        'Hydro':       True,   # Hydro_Reserves_Log_Deseasonalized
        'NetExchange': True,   # Net_Exchange
        'Consumption': True,   # Consumption_Log_Deseasonalized
        'Oil':         True,   # Oil_Price_Log_Deseasonalized
        'Gas':         True,  # Gas_Price_Log_Deseasonalized
        'Bottlenecks': True,  # BNECK_* congestion dummies
    }

    # --- MODEL RUNS ---
    RUN_OLS = False    # Run standard OLS
    RUN_ARMAX = True   # Run ARMAX (and conditionally GARCH)

    # --- ARMAX SPECIFICATION ---
    OPTIMIZE_ARMAX_LAGS = False          # Grid-search optimal p,q via BIC (slow)
    USE_CHECKPOINTED_LAG_SELECTION = False  # Save/resume lag search progress; only if OPTIMIZE_ARMAX_LAGS=True
    ARMAX_SEARCH_P_MIN = 0               # AR search range (inclusive); only if OPTIMIZE_ARMAX_LAGS=True
    ARMAX_SEARCH_P_MAX = 10
    ARMAX_SEARCH_Q_MIN = 1               # MA search range (inclusive); only if OPTIMIZE_ARMAX_LAGS=True
    ARMAX_SEARCH_Q_MAX = 1
    ARMAX_BASELINE_SPEC = {              # Fixed spec used when OPTIMIZE_ARMAX_LAGS=False
        'order': (3, 0, 2),             # (p, d, q)
        'extra_ar_lags': []             # Sparse additional AR lags, e.g. [23, 24, 25]
    }

    # --- GARCH ---
    FIT_GARCH_IF_ARCH = False   # Fit GARCH-X if ARCH effects detected in ARMAX residuals
    GARCH_ORDER = (1, 1)        # (p, q) order

    # --- DIAGNOSTICS: STAGED ---
    RUN_ARMAX_STAGED_CONVERGENCE = False       # Incremental exog controls convergence check; exits early
    ARMAX_STAGED_ORDER = (1, 0, 1)
    RUN_OLS_VARIABLE_INCLUSION_DIAGNOSTIC = False  # Nested OLS model comparisons; exits early

    # --- ROLLING WINDOW ---
    RUN_ROLLING_WINDOW = False      # Rolling wind coefficient estimation; skips OLS/ARMAX
    ROLLING_WINDOW_YEARS = 1        # Window size in years
    ROLLING_STEP_YEARS = 1          # Step size in years
    ROLLING_MIN_OBS = 24 * 365 - 24 * 30  # Minimum observations per window

    # --- STRUCTURAL BREAK ---
    RUN_STRUCTURAL_BREAK = False                    # Detect structural breaks in wind coefficient
    STRUCTURAL_BREAK_TYPE = 'trend'                 # 'level' (Bai-Perron step) or 'trend' (slope change)
    STRUCTURAL_BREAK_ESTIMATION_MODEL = 'ols'       # 'ols' or 'dynamic_armax'
    STRUCTURAL_BREAK_MAX_BREAKS = 3                 # Max breaks to test
    STRUCTURAL_BREAK_TRIMMING = 0.1                 # Endpoint trim fraction
    STRUCTURAL_BREAK_KNOWN_DATES = None             # Chow test dates, e.g. ['2022-02-24']
    STRUCTURAL_BREAK_WINDOW_YEARS = 1               # Rolling window size for coefficient estimation
    STRUCTURAL_BREAK_STEP_YEARS = 3/12              # Step size in years
    STRUCTURAL_BREAK_MIN_OBS = 24 * 365 - 24 * 30  # Minimum observations per window
    TREND_BREAK_TEST_METHOD = 'bp_supf'             # 'legacy' or 'bp_supf' (Bai-Perron supF)
    BP_INFERENCE_MODE = 'tables'                    # 'both', 'tables', or 'bootstrap'
    BP_SIGNIFICANCE_LEVEL = 0.05                    # One of: 0.10, 0.05, 0.025, 0.01
    BP_BOOTSTRAP_REPS = 999
    BP_BOOTSTRAP_BLOCK_LENGTH = 8
    BP_RANDOM_SEED = 42
    BP_USE_HAC_SE = True

    # --- RARELY USED ---
    RUN_ZONE_COMPARISONS = False   # Overlay SE1-SE4 comparison plots
    RUN_VISUALIZATIONS = False     # Raw and logged data visualizations

    # Data file paths - dynamically set based on ACTIVE_ZONE
    PATHS = {
        'combined': f'master data files/2015-2025/Combined_{ACTIVE_ZONE}_Data_2015_2025.xlsx',
        'hydro': 'master data files/Master_Hydro_Reservoir.xlsx',
        'crude_oil': 'master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx',
        'commodities': 'master data files/Master_Commodities.xlsx'  # Still used for TTF Gas
    }

    # --- ZONE COMPARISON PLOTS (runs before zone-specific pipeline) ---
    if RUN_ZONE_COMPARISONS:
        plot_zone_comparisons(start_date='2015-01-01', end_date='2025-12-31', plots_dir='plots')

    try:
        def _preprocess_output_context():
            if SHOW_PREPROCESSING_STATISTICS:
                return contextlib.nullcontext()
            return contextlib.redirect_stdout(io.StringIO())

        # Load and clean full dataset (filtered to 2021-2024 for hydro/commodity availability)
        # Note: Commodity prices are automatically lagged within load_data()
        with _preprocess_output_context():
            data = load_data(
                PATHS,
                target_region=ACTIVE_ZONE,
                zone_hydro=ACTIVE_ZONE,
                use_interpolation=True,
                start_date=START_DATE,
                end_date=END_DATE,
                lag_commodity_hours=24
            )
            print(f"Merge successful. Total hourly observations: {len(data)}")
        if not SHOW_PREPROCESSING_STATISTICS:
            print("Preprocessing loaded. Set SHOW_PREPROCESSING_STATISTICS=True to view detailed preprocessing output.")

        _price_mode = "log(Price) → deseasonalize (Method A)" if LOG_PRICE else "deseasonalize Price in levels (semi-log)"
        print(f"Price transformation mode: {_price_mode}")

        # --- ROLLING WINDOW MODE (WINDOW-LOCAL PREPROCESSING) ---
        # Run rolling-window analysis directly on raw data so each window is transformed locally.
        # This avoids full-sample transformation leakage into local coefficient estimates.
        if RUN_ROLLING_WINDOW:
            run_rolling_window_analysis(
                data,
                ACTIVE_ZONE,
                window_years=ROLLING_WINDOW_YEARS,
                step_years=ROLLING_STEP_YEARS,
                min_obs=ROLLING_MIN_OBS,
                plots_dir="plots",
                results_dir="results",
                use_window_local_preprocessing=True,
                controls=CONTROLS,
                target_region=ACTIVE_ZONE,
            )
            raise SystemExit(0)

        # --- STEP 1: VISUALIZATION OF RAW DATA (if enabled) ---
        # Visualize pure, untouched raw data BEFORE any transformations
        # This shows the true state of the data including any negative values or quality issues
        if RUN_VISUALIZATIONS:
            run_visualizations(data, ACTIVE_ZONE, stage='raw')

        # --- STEP 2: CHECK NEGATIVE VALUES & HANDLE PRICE ---
        # Check all variables for negative values and handle Price if needed
        # Net_Exchange is NOT checked or modified (expected to have negative values)
        # NOTE: This happens AFTER raw visualization so we can see the original data quality issues
        with _preprocess_output_context():
            data = handle_negative_prices(data)

        # --- STEP 3: LOG TRANSFORMATION (always applied) ---
        # STANDARD APPROACH: Apply log transformation FIRST, before deseasonalization
        # This is the standard econometric approach for handling multiplicative seasonality
        # Note: Commodity prices are already lagged at this point
        # Note: Price negative values handled in STEP 2 (before log transformation)
        with _preprocess_output_context():
            data = apply_log_transform(data)

        # Visualize logged data (if enabled)
        # This shows data AFTER negative handling and log transformation
        if RUN_VISUALIZATIONS:
            # Create temporary dataframe with logged variables mapped to base names for visualization
            data_logged_viz = data.copy()
            data_logged_viz['Price'] = data_logged_viz['Price_Log']
            data_logged_viz['Wind_Forecast'] = data_logged_viz['Wind_Forecast_Log']
            data_logged_viz['Hydro_Reserves'] = data_logged_viz['Hydro_Reserves_Log']
            data_logged_viz['Consumption'] = data_logged_viz['Consumption_Log']
            data_logged_viz['Oil_Price'] = data_logged_viz['Oil_Price_Log']
            # Net_Exchange stays the same (not logged)
            run_visualizations(data_logged_viz, ACTIVE_ZONE, stage='logged')

        # --- STEP 4: DESEASONALIZATION (always applied) ---
        # STANDARD APPROACH: Deseasonalize the LOGGED variables (after log transformation)
        # Price & Consumption: Year + Month + DOW + Hour + Holiday (FULL deseasonalization)
        # Hydro, Oil, Gas: Year + Month ONLY (PARTIAL - no intraday patterns)
        with _preprocess_output_context():
            data = deseasonalize_logged_variables(data)

        # --- OUTLIER VISUALISATION (optional) ---
        if PLOT_OUTLIERS:
            import matplotlib.pyplot as plt
            import os
            _ds = data['Price_DS'].copy()
            _dow = _ds.index.dayofweek
            _fig, _ax = plt.subplots(figsize=(16, 5))
            _ax.plot(_ds.index, _ds.values, color='steelblue', linewidth=0.6, label='Price_DS')
            _colors = plt.cm.tab10.colors
            for _day in range(7):
                _mask = _dow == _day
                _day_data = _ds[_mask]
                _mu, _sigma = _day_data.mean(), _day_data.std()
                _upper, _lower = _mu + 3 * _sigma, _mu - 3 * _sigma
                _ax.axhline(_upper, color=_colors[_day], linewidth=0.5, linestyle='--', alpha=0.6)
                _ax.axhline(_lower, color=_colors[_day], linewidth=0.5, linestyle='--', alpha=0.6)
                _out_mask = _mask & ((_ds > _upper) | (_ds < _lower))
                _ax.scatter(_ds.index[_out_mask], _ds.values[_out_mask],
                            color=_colors[_day], s=20, zorder=5)
            _ax.set_title(f'Price_DS with weekday ±3σ outlier bounds — {ACTIVE_ZONE} {START_DATE}:{END_DATE}')
            _ax.set_ylabel('Price_DS')
            _ax.legend(['Price_DS'] + [f'day {d} bounds/outliers' for d in range(7)],
                       fontsize=7, ncol=4, loc='upper right')
            plt.tight_layout()
            _fig.savefig('_outlier_preview.png', dpi=150, bbox_inches='tight')
            plt.close(_fig)
            os.startfile(os.path.abspath('_outlier_preview.png'))

        # --- STEP 5: OUTLIER HANDLING (after log + deseasonalization) ---
        with _preprocess_output_context():
            print("\n" + "="*80)
            print("STEP 5: OUTLIER HANDLING (AFTER TRANSFORMATIONS)")
            print("="*80)
            print("Applying outlier detection and replacement to transformed Price series\n")

            data, outlier_stats = handle_outliers_gianfreda(data)

        # --- STEP 6: REGRESSION ANALYSIS ---
        # Run regression models with optional diagnostic tests
        # Commodity prices used in regression are lagged by 24h (from load_data)
        ols_model, armax_res, garch_res = perform_multivariate_analysis(data, ACTIVE_ZONE,
                                      target_region=ACTIVE_ZONE,
                                      run_ols=RUN_OLS,
                                      run_armax=RUN_ARMAX,
                                      run_ljungbox=RUN_LJUNGBOX_TEST,
                                      run_hetero_tests=RUN_HETEROSKEDASTICITY_TESTS,
                                      run_stationarity=RUN_STATIONARITY_TESTS,
                                      run_collinearity_checks=RUN_COLLINEARITY_DIAGNOSTICS,
                                      controls=CONTROLS,
                                      run_armax_staged_convergence=RUN_ARMAX_STAGED_CONVERGENCE,
                                      run_ols_variable_inclusion_diagnostic=RUN_OLS_VARIABLE_INCLUSION_DIAGNOSTIC,
                                      armax_staged_order=ARMAX_STAGED_ORDER,
                                      armax_staged_maxiter=300,
                                      armax_staged_solver='statespace',
                                       optimize_armax_lags=OPTIMIZE_ARMAX_LAGS,
                                       use_checkpointed_lag_selection=USE_CHECKPOINTED_LAG_SELECTION,
                                        armax_search_p_min=ARMAX_SEARCH_P_MIN,
                                        armax_search_p_max=ARMAX_SEARCH_P_MAX,
                                        armax_search_q_min=ARMAX_SEARCH_Q_MIN,
                                        armax_search_q_max=ARMAX_SEARCH_Q_MAX,
                                      run_rolling_window=RUN_ROLLING_WINDOW,
                                      rolling_window_years=ROLLING_WINDOW_YEARS,
                                      rolling_step_years=ROLLING_STEP_YEARS,
                                      rolling_min_obs=ROLLING_MIN_OBS,
                                      run_structural_break=RUN_STRUCTURAL_BREAK,
                                      structural_break_type=STRUCTURAL_BREAK_TYPE,
                                      structural_break_max_breaks=STRUCTURAL_BREAK_MAX_BREAKS,
                                      structural_break_trimming=STRUCTURAL_BREAK_TRIMMING,
                                      structural_break_known_dates=STRUCTURAL_BREAK_KNOWN_DATES,
                                      structural_break_window_years=STRUCTURAL_BREAK_WINDOW_YEARS,
                                      structural_break_step_years=STRUCTURAL_BREAK_STEP_YEARS,
                                      structural_break_min_obs=STRUCTURAL_BREAK_MIN_OBS,
                                      trend_break_test_method=TREND_BREAK_TEST_METHOD,
                                      bp_inference_mode=BP_INFERENCE_MODE,
                                      bp_significance_level=BP_SIGNIFICANCE_LEVEL,
                                      bp_bootstrap_reps=BP_BOOTSTRAP_REPS,
                                      bp_bootstrap_block_length=BP_BOOTSTRAP_BLOCK_LENGTH,
                                      bp_random_seed=BP_RANDOM_SEED,
                                      bp_use_hac_se=BP_USE_HAC_SE,
                                      structural_break_estimation_model=STRUCTURAL_BREAK_ESTIMATION_MODEL,
                                      armax_baseline_spec=ARMAX_BASELINE_SPEC)

    except Exception as e:
        print(f"Critical error during execution: {e}")
