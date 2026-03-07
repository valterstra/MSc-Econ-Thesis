from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class JointGarchXResult:
    success: bool
    message: str
    optimizer: str
    n_obs: int
    n_ar_lags: int
    mu: float
    ar_coefficients: np.ndarray
    beta_x: float
    omega: float
    alpha1: float
    beta1: float
    gamma_x: float
    loglik: float
    aic: float
    bic: float
    persistence: float
    residuals: np.ndarray
    conditional_variance: np.ndarray
    standard_errors: dict[str, float]
    pvalues: dict[str, float]
    x_center_mean: float


def _as_1d_array(values, name):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _compute_aic_bic(loglik, n_obs, n_params):
    aic = 2 * n_params - 2 * loglik
    bic = np.log(n_obs) * n_params - 2 * loglik
    return float(aic), float(bic)


def _build_joint_series(y, x, ar_lags):
    y = _as_1d_array(y, "y")
    x = _as_1d_array(x, "x")
    if y.size != x.size:
        raise ValueError("y and x must have the same length.")
    if ar_lags < 1:
        raise ValueError("ar_lags must be at least 1.")
    if y.size <= ar_lags:
        raise ValueError("y must have more observations than ar_lags.")

    x_centered = x - x.mean()
    y_current = y[ar_lags:]
    x_current = x[ar_lags:]
    x_centered_current = x_centered[ar_lags:]
    lag_columns = [y[ar_lags - lag : y.size - lag] for lag in range(1, ar_lags + 1)]
    y_lagged_matrix = np.column_stack(lag_columns)

    return y, x, y_current, y_lagged_matrix, x_current, x_centered_current


def _compute_joint_residuals(params, y, x, ar_lags):
    phi = params[1 : 1 + ar_lags]
    beta_x = params[1 + ar_lags]
    _, _, y_current, y_lagged_matrix, x_current, _ = _build_joint_series(y, x, ar_lags)
    mu = params[0]
    residuals = y_current - mu - y_lagged_matrix @ phi - beta_x * x_current
    return residuals


def _compute_joint_variance(params, residuals, x_centered_current, initial_variance, ar_lags):
    omega = params[2 + ar_lags]
    alpha1 = params[3 + ar_lags]
    beta1 = params[4 + ar_lags]
    gamma_x = params[5 + ar_lags]
    residuals = np.asarray(residuals, dtype=float)
    x_centered_current = np.asarray(x_centered_current, dtype=float)
    h = np.empty_like(residuals)
    h[0] = max(float(initial_variance), 1e-6)

    for t in range(1, residuals.size):
        h[t] = omega + alpha1 * residuals[t - 1] ** 2 + beta1 * h[t - 1] + gamma_x * x_centered_current[t]
        if h[t] <= 1e-10 or not np.isfinite(h[t]):
            raise ValueError("Non-positive conditional variance encountered.")

    return h


def _negative_loglik_joint(params, y, x, ar_lags):
    try:
        y_full, _, _, _, _, x_centered_current = _build_joint_series(y, x, ar_lags)
        residuals = _compute_joint_residuals(params, y, x, ar_lags)
        initial_variance = max(np.var(y_full, ddof=1), 1e-6)
        conditional_variance = _compute_joint_variance(
            params,
            residuals,
            x_centered_current,
            initial_variance,
            ar_lags,
        )
    except ValueError:
        return 1e12

    ll = -0.5 * (
        np.log(2 * np.pi)
        + np.log(conditional_variance)
        + residuals**2 / conditional_variance
    )
    return float(-np.sum(ll))


def _approx_hessian(func, x0, epsilon=1e-5):
    x0 = np.asarray(x0, dtype=float)
    n_params = x0.size
    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        for j in range(n_params):
            step_i = np.zeros(n_params, dtype=float)
            step_j = np.zeros(n_params, dtype=float)
            step_i[i] = epsilon
            step_j[j] = epsilon

            f_pp = func(x0 + step_i + step_j)
            f_pm = func(x0 + step_i - step_j)
            f_mp = func(x0 - step_i + step_j)
            f_mm = func(x0 - step_i - step_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)

    return hessian


def _build_inference(params, objective, param_names):
    standard_errors = {name: float("nan") for name in param_names}
    pvalues = {name: float("nan") for name in param_names}

    try:
        hessian = _approx_hessian(objective, params)
        covariance = np.linalg.inv(hessian)
        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0) or np.any(~np.isfinite(diagonal)):
            return standard_errors, pvalues

        se = np.sqrt(diagonal)
        z_stats = params / se
        tail_probs = 2 * (1 - norm.cdf(np.abs(z_stats)))

        standard_errors = {
            name: float(value) for name, value in zip(param_names, se, strict=True)
        }
        pvalues = {
            name: float(value) for name, value in zip(param_names, tail_probs, strict=True)
        }
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        pass

    return standard_errors, pvalues


def _minimize_with_fallback(objective, initial_params, bounds, constraints):
    attempts = [
        ("SLSQP", dict(method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})),
        ("L-BFGS-B", dict(method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})),
    ]

    best_result = None
    best_name = None

    for optimizer_name, kwargs in attempts:
        result = minimize(objective, x0=initial_params, **kwargs)
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_name = optimizer_name
        if result.success:
            return result, optimizer_name

    return best_result, best_name


def _safe_joint_outputs(params, y, x, ar_lags):
    try:
        y_full, x_full, _, _, _, x_centered_current = _build_joint_series(y, x, ar_lags)
        residuals = _compute_joint_residuals(params, y_full, x_full, ar_lags)
        conditional_variance = _compute_joint_variance(
            params,
            residuals,
            x_centered_current,
            initial_variance=max(np.var(y_full, ddof=1), 1e-6),
            ar_lags=ar_lags,
        )
        return residuals, conditional_variance
    except ValueError:
        y_array = _as_1d_array(y, "y")
        n_obs = y_array.size - ar_lags
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)


def fit_joint_arx_garchx(y, x, ar_lags=1):
    y, x, y_current, y_lagged_matrix, x_current, _ = _build_joint_series(y, x, ar_lags)

    ols_design = np.column_stack([np.ones_like(y_current), y_lagged_matrix, x_current])
    ols_beta, _, _, _ = np.linalg.lstsq(ols_design, y_current, rcond=None)
    provisional_residuals = y_current - ols_design @ ols_beta
    provisional_variance = max(np.var(provisional_residuals, ddof=1), 1e-6)
    gamma_bound = max(10 * np.var(y, ddof=1) / max(np.std(x - x.mean(), ddof=1), 1e-6), 1e-3)

    initial_params = np.concatenate(
        [
            [ols_beta[0]],
            ols_beta[1 : 1 + ar_lags],
            [ols_beta[-1], 0.05 * provisional_variance, 0.10, 0.80, 0.0],
        ]
    ).astype(float)
    bounds = [(None, None)] + [(-0.999, 0.999)] * ar_lags + [
        (None, None),
        (1e-8, None),
        (0.0, 0.999),
        (0.0, 0.999),
        (-gamma_bound, gamma_bound),
    ]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[3 + ar_lags] - p[4 + ar_lags]}]

    objective = lambda p: _negative_loglik_joint(p, y, x, ar_lags)
    result, optimizer_name = _minimize_with_fallback(objective, initial_params, bounds, constraints)
    residuals, conditional_variance = _safe_joint_outputs(result.x, y, x, ar_lags)
    loglik = -objective(result.x)
    aic, bic = _compute_aic_bic(loglik, residuals.size, len(result.x))
    param_names = ["mu"] + [f"phi_lag_{lag}" for lag in range(1, ar_lags + 1)] + [
        "beta_x",
        "omega",
        "alpha1",
        "beta1",
        "gamma_x",
    ]
    standard_errors, pvalues = _build_inference(result.x, objective, param_names)

    success = bool(result.success and np.all(np.isfinite(conditional_variance)))
    message = str(result.message)
    if not np.all(np.isfinite(conditional_variance)):
        message = f"{message} | final variance recursion invalid"

    return JointGarchXResult(
        success=success,
        message=message,
        optimizer=optimizer_name,
        n_obs=int(residuals.size),
        n_ar_lags=int(ar_lags),
        mu=float(result.x[0]),
        ar_coefficients=np.asarray(result.x[1 : 1 + ar_lags], dtype=float),
        beta_x=float(result.x[1 + ar_lags]),
        omega=float(result.x[2 + ar_lags]),
        alpha1=float(result.x[3 + ar_lags]),
        beta1=float(result.x[4 + ar_lags]),
        gamma_x=float(result.x[5 + ar_lags]),
        loglik=float(loglik),
        aic=aic,
        bic=bic,
        persistence=float(result.x[3 + ar_lags] + result.x[4 + ar_lags]),
        residuals=residuals,
        conditional_variance=conditional_variance,
        standard_errors=standard_errors,
        pvalues=pvalues,
        x_center_mean=float(x.mean()),
    )
