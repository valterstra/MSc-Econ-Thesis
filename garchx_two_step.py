from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GarchResult:
    success: bool
    message: str
    n_obs: int
    omega: float
    alpha1: float
    beta1: float
    loglik: float
    aic: float
    bic: float
    persistence: float
    conditional_variance: np.ndarray
    standard_errors: dict[str, float]
    pvalues: dict[str, float]
    optimizer: str


@dataclass
class GarchXResult(GarchResult):
    gamma_x: float
    x_center_mean: float


def _as_1d_array(values, name):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _compute_aic_bic(loglik, n_obs, n_params):
    aic = 2 * n_params - 2 * loglik
    bic = np.log(n_obs) * n_params - 2 * loglik
    return float(aic), float(bic)


def compute_conditional_variance_garch(params, residuals):
    omega, alpha1, beta1 = params
    residuals = _as_1d_array(residuals, "residuals")
    h = np.empty_like(residuals)
    h[0] = max(np.var(residuals, ddof=1), 1e-6)

    for t in range(1, residuals.size):
        h[t] = omega + alpha1 * residuals[t - 1] ** 2 + beta1 * h[t - 1]
        if h[t] <= 1e-10 or not np.isfinite(h[t]):
            raise ValueError("Non-positive conditional variance encountered.")

    return h


def compute_conditional_variance_garchx(params, residuals, x):
    omega, alpha1, beta1, gamma_x = params
    residuals = _as_1d_array(residuals, "residuals")
    x = _as_1d_array(x, "x")
    if residuals.size != x.size:
        raise ValueError("residuals and x must have the same length.")

    centered_x = x - x.mean()
    h = np.empty_like(residuals)
    h[0] = max(np.var(residuals, ddof=1), 1e-6)

    for t in range(1, residuals.size):
        h[t] = omega + alpha1 * residuals[t - 1] ** 2 + beta1 * h[t - 1] + gamma_x * centered_x[t]
        if h[t] <= 1e-10 or not np.isfinite(h[t]):
            raise ValueError("Non-positive conditional variance encountered.")

    return h


def negative_loglik_garch(params, residuals):
    try:
        h = compute_conditional_variance_garch(params, residuals)
    except ValueError:
        return 1e12

    residuals = _as_1d_array(residuals, "residuals")
    ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + residuals**2 / h)
    return float(-np.sum(ll))


def negative_loglik_garchx(params, residuals, x):
    try:
        h = compute_conditional_variance_garchx(params, residuals, x)
    except ValueError:
        return 1e12

    residuals = _as_1d_array(residuals, "residuals")
    ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + residuals**2 / h)
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


def _safe_compute_variance_garch(params, residuals):
    try:
        return compute_conditional_variance_garch(params, residuals)
    except ValueError:
        return np.full_like(_as_1d_array(residuals, "residuals"), np.nan)


def _safe_compute_variance_garchx(params, residuals, x):
    try:
        return compute_conditional_variance_garchx(params, residuals, x)
    except ValueError:
        return np.full_like(_as_1d_array(residuals, "residuals"), np.nan)


def fit_garch(residuals):
    residuals = _as_1d_array(residuals, "residuals")
    variance = max(np.var(residuals, ddof=1), 1e-6)
    initial_params = np.array([0.05 * variance, 0.10, 0.80], dtype=float)
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]

    objective = lambda p: negative_loglik_garch(p, residuals)
    result, optimizer_name = _minimize_with_fallback(objective, initial_params, bounds, constraints)
    omega, alpha1, beta1 = result.x
    conditional_variance = _safe_compute_variance_garch(result.x, residuals)
    loglik = -objective(result.x)
    aic, bic = _compute_aic_bic(loglik, residuals.size, len(result.x))
    standard_errors, pvalues = _build_inference(result.x, objective, ["omega", "alpha1", "beta1"])
    success = bool(result.success and np.all(np.isfinite(conditional_variance)))
    message = str(result.message)
    if not np.all(np.isfinite(conditional_variance)):
        message = f"{message} | final variance recursion invalid"

    return GarchResult(
        success=success,
        message=message,
        n_obs=int(residuals.size),
        omega=float(omega),
        alpha1=float(alpha1),
        beta1=float(beta1),
        loglik=float(loglik),
        aic=aic,
        bic=bic,
        persistence=float(alpha1 + beta1),
        conditional_variance=conditional_variance,
        standard_errors=standard_errors,
        pvalues=pvalues,
        optimizer=optimizer_name,
    )


def fit_garchx(residuals, x):
    residuals = _as_1d_array(residuals, "residuals")
    x = _as_1d_array(x, "x")
    if residuals.size != x.size:
        raise ValueError("residuals and x must have the same length.")

    variance = max(np.var(residuals, ddof=1), 1e-6)
    centered_x = x - x.mean()
    gamma_bound = max(10 * variance / max(centered_x.std(ddof=1), 1e-6), 1e-3)
    initial_params = np.array([0.05 * variance, 0.10, 0.80, 0.0], dtype=float)
    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999), (-gamma_bound, gamma_bound)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]

    objective = lambda p: negative_loglik_garchx(p, residuals, x)
    result, optimizer_name = _minimize_with_fallback(objective, initial_params, bounds, constraints)
    omega, alpha1, beta1, gamma_x = result.x
    conditional_variance = _safe_compute_variance_garchx(result.x, residuals, x)
    loglik = -objective(result.x)
    aic, bic = _compute_aic_bic(loglik, residuals.size, len(result.x))
    standard_errors, pvalues = _build_inference(
        result.x,
        objective,
        ["omega", "alpha1", "beta1", "gamma_x"],
    )

    success = bool(result.success and np.all(np.isfinite(conditional_variance)))
    message = str(result.message)
    if not np.all(np.isfinite(conditional_variance)):
        message = f"{message} | final variance recursion invalid"

    return GarchXResult(
        success=success,
        message=message,
        n_obs=int(residuals.size),
        omega=float(omega),
        alpha1=float(alpha1),
        beta1=float(beta1),
        gamma_x=float(gamma_x),
        x_center_mean=float(x.mean()),
        loglik=float(loglik),
        aic=aic,
        bic=bic,
        persistence=float(alpha1 + beta1),
        conditional_variance=conditional_variance,
        standard_errors=standard_errors,
        pvalues=pvalues,
        optimizer=optimizer_name,
    )
