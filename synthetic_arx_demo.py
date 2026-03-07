from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_synthetic_data(
    n_obs=10_000,
    phi=0.90,
    beta=-0.08,
    x_phi=0.75,
    sigma_x=1.0,
    sigma_y=1.0,
    burn_in=500,
    seed=42,
):
    rng = np.random.default_rng(seed)
    total_obs = n_obs + burn_in

    x = np.zeros(total_obs)
    y = np.zeros(total_obs)

    x_shocks = rng.normal(0.0, sigma_x, total_obs)
    y_shocks = rng.normal(0.0, sigma_y, total_obs)

    for t in range(1, total_obs):
        x[t] = x_phi * x[t - 1] + x_shocks[t]
        y[t] = phi * y[t - 1] + beta * x[t] + y_shocks[t]

    x = x[burn_in:]
    y = y[burn_in:]
    dt_index = pd.date_range("2020-01-01 00:00:00", periods=n_obs, freq="h")

    return pd.DataFrame(
        {
            "Datetime": dt_index,
            "Y": y,
            "X": x,
        }
    ).set_index("Datetime").asfreq("h")


def fit_ar1(data):
    model = SARIMAX(
        endog=data["Y"],
        order=(1, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model


def fit_arx1(data):
    model = SARIMAX(
        endog=data["Y"],
        exog=data[["X"]],
        order=(1, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model


def meets_requirements(ar1_model, arx1_model):
    ar1_phi = float(ar1_model.params["ar.L1"])
    arx1_phi = float(arx1_model.params["ar.L1"])
    x_beta = float(arx1_model.params["X"])
    x_pvalue = float(arx1_model.pvalues["X"])

    return (
        0.85 <= ar1_phi <= 0.95
        and 0.85 <= arx1_phi <= 0.95
        and -0.12 <= x_beta <= -0.03
        and 0.001 <= x_pvalue < 0.01
    )


def search_for_dataset(n_obs=10_000, max_tries=200):
    base_configs = [
        {"phi": 0.90, "beta": -0.06, "x_phi": 0.70, "sigma_x": 1.0, "sigma_y": 1.4},
        {"phi": 0.90, "beta": -0.08, "x_phi": 0.75, "sigma_x": 1.0, "sigma_y": 1.5},
        {"phi": 0.90, "beta": -0.10, "x_phi": 0.80, "sigma_x": 0.9, "sigma_y": 1.5},
        {"phi": 0.88, "beta": -0.08, "x_phi": 0.75, "sigma_x": 1.0, "sigma_y": 1.4},
    ]

    for seed in range(1, max_tries + 1):
        config = base_configs[(seed - 1) % len(base_configs)]
        data = generate_synthetic_data(n_obs=n_obs, seed=seed, **config)
        ar1_model = fit_ar1(data)
        arx1_model = fit_arx1(data)

        if meets_requirements(ar1_model, arx1_model):
            return data, ar1_model, arx1_model, seed, config

    raise RuntimeError("No synthetic dataset met the target requirements within max_tries.")


def summarize_results(data, ar1_model, arx1_model, seed, config):
    print("Synthetic data search succeeded")
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


def export_dataset(data, seed, config):
    output_path = Path(__file__).resolve().parent / "output" / "synthetic_arx_demo_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = data.reset_index().copy()
    export_df["seed"] = seed
    export_df["phi_true"] = config["phi"]
    export_df["beta_true"] = config["beta"]
    export_df["x_phi_true"] = config["x_phi"]
    export_df["sigma_x_true"] = config["sigma_x"]
    export_df["sigma_y_true"] = config["sigma_y"]
    export_df.to_csv(output_path, index=False)

    print("")
    print(f"Exported synthetic dataset to: {output_path}")


if __name__ == "__main__":
    data, ar1_model, arx1_model, seed, config = search_for_dataset()
    summarize_results(data, ar1_model, arx1_model, seed, config)
    export_dataset(data, seed, config)
