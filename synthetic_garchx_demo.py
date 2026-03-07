from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_synthetic_garchx_data(
    n_obs=10_000,
    phi=0.90,
    beta_x=-0.10,
    x_phi=0.85,
    x_scale=0.30,
    omega=0.15,
    alpha=0.10,
    beta_h=0.82,
    gamma_x=0.08,
    burn_in=1_000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    total_obs = n_obs + burn_in

    x_latent = np.zeros(total_obs)
    x = np.zeros(total_obs)
    h = np.zeros(total_obs)
    eps = np.zeros(total_obs)
    y = np.zeros(total_obs)

    x_shocks = rng.normal(0.0, 1.0, total_obs)
    z = rng.normal(0.0, 1.0, total_obs)

    h[0] = omega / max(1.0 - alpha - beta_h, 0.05)
    x[0] = 1.0

    for t in range(1, total_obs):
        x_latent[t] = x_phi * x_latent[t - 1] + x_shocks[t]
        x[t] = np.exp(x_scale * x_latent[t])
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta_h * h[t - 1] + gamma_x * x[t]
        eps[t] = np.sqrt(max(h[t], 1e-8)) * z[t]
        y[t] = phi * y[t - 1] + beta_x * x[t] + eps[t]

    dt_index = pd.date_range("2020-01-01 00:00:00", periods=n_obs, freq="h")
    data = pd.DataFrame(
        {
            "Datetime": dt_index,
            "Y": y[burn_in:],
            "X": x[burn_in:],
            "True_H": h[burn_in:],
            "True_Eps": eps[burn_in:],
        }
    ).set_index("Datetime").asfreq("h")

    return data


def fit_ar1(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return SARIMAX(
            endog=data["Y"],
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)


def fit_arx1(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return SARIMAX(
            endog=data["Y"],
            exog=data[["X"]],
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)


def fit_aux_variance_regression(arx_model, data):
    aux = pd.DataFrame(
        {
            "eps2": np.square(np.asarray(arx_model.resid)),
            "X": data["X"].to_numpy(),
        },
        index=data.index,
    )
    aux["eps2_lag1"] = aux["eps2"].shift(1)
    aux = aux.dropna()

    x = sm.add_constant(aux[["eps2_lag1", "X"]])
    model = sm.OLS(aux["eps2"], x).fit()
    return model


def meets_requirements(ar1_model, arx1_model, aux_var_model):
    ar1_phi = float(ar1_model.params["ar.L1"])
    arx1_phi = float(arx1_model.params["ar.L1"])
    x_beta = float(arx1_model.params["X"])
    x_pvalue = float(arx1_model.pvalues["X"])
    variance_x = float(aux_var_model.params["X"])
    variance_x_p = float(aux_var_model.pvalues["X"])

    return (
        0.85 <= ar1_phi <= 0.95
        and 0.85 <= arx1_phi <= 0.95
        and -0.12 <= x_beta <= -0.03
        and x_pvalue < 0.01
        and variance_x > 0
        and variance_x_p < 0.01
    )


def search_for_dataset(n_obs=10_000, max_tries=300):
    base_configs = [
        {
            "phi": 0.90,
            "beta_x": -0.08,
            "x_phi": 0.85,
            "x_scale": 0.30,
            "omega": 0.15,
            "alpha": 0.10,
            "beta_h": 0.82,
            "gamma_x": 0.08,
        },
        {
            "phi": 0.90,
            "beta_x": -0.10,
            "x_phi": 0.80,
            "x_scale": 0.28,
            "omega": 0.14,
            "alpha": 0.10,
            "beta_h": 0.84,
            "gamma_x": 0.10,
        },
        {
            "phi": 0.88,
            "beta_x": -0.08,
            "x_phi": 0.82,
            "x_scale": 0.32,
            "omega": 0.16,
            "alpha": 0.12,
            "beta_h": 0.80,
            "gamma_x": 0.09,
        },
    ]

    for seed in range(1, max_tries + 1):
        config = base_configs[(seed - 1) % len(base_configs)]
        data = generate_synthetic_garchx_data(n_obs=n_obs, seed=seed, **config)
        ar1_model = fit_ar1(data)
        arx1_model = fit_arx1(data)
        aux_var_model = fit_aux_variance_regression(arx1_model, data)

        if meets_requirements(ar1_model, arx1_model, aux_var_model):
            return data, ar1_model, arx1_model, aux_var_model, seed, config

    raise RuntimeError("No synthetic GARCH-X dataset met the target requirements within max_tries.")


def summarize_results(data, ar1_model, arx1_model, aux_var_model, seed, config):
    print("Synthetic GARCH-X data search succeeded")
    print(f"Observations: {len(data)}")
    print(f"Seed used: {seed}")
    print(f"Generation parameters: {config}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print("")
    print("AR(1) fit on Y")
    print(f"Intercept: {ar1_model.params['intercept']:.4f}")
    print(f"AR(1) coefficient: {ar1_model.params['ar.L1']:.6f}")
    print(f"AR(1) p-value: {ar1_model.pvalues['ar.L1']:.4g}")
    print("")
    print("ARX(1) fit on Y with X")
    print(f"Intercept: {arx1_model.params['intercept']:.4f}")
    print(f"AR(1) coefficient: {arx1_model.params['ar.L1']:.6f}")
    print(f"X coefficient: {arx1_model.params['X']:.6f}")
    print(f"AR(1) p-value: {arx1_model.pvalues['ar.L1']:.4g}")
    print(f"X p-value: {arx1_model.pvalues['X']:.4g}")
    print("")
    print("Auxiliary variance check on ARX residuals")
    print(f"lagged eps^2 coefficient: {aux_var_model.params['eps2_lag1']:.6f}")
    print(f"X coefficient in variance proxy regression: {aux_var_model.params['X']:.6f}")
    print(f"X p-value in variance proxy regression: {aux_var_model.pvalues['X']:.4g}")


def export_dataset(data, seed, config):
    output_path = Path(__file__).resolve().parent / "output" / "synthetic_garchx_demo_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = data.reset_index().copy()
    export_df["seed"] = seed
    export_df["phi_true"] = config["phi"]
    export_df["beta_x_true"] = config["beta_x"]
    export_df["x_phi_true"] = config["x_phi"]
    export_df["omega_true"] = config["omega"]
    export_df["alpha_true"] = config["alpha"]
    export_df["beta_h_true"] = config["beta_h"]
    export_df["gamma_x_true"] = config["gamma_x"]
    export_df.to_csv(output_path, index=False)

    print("")
    print(f"Exported synthetic dataset to: {output_path}")


if __name__ == "__main__":
    data, ar1_model, arx1_model, aux_var_model, seed, config = search_for_dataset()
    summarize_results(data, ar1_model, arx1_model, aux_var_model, seed, config)
    export_dataset(data, seed, config)
