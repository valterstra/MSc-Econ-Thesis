from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import holidays as holidays_lib
except ImportError as exc:
    raise ImportError(
        "The 'holidays' package is required. Install with: pip install holidays"
    ) from exc


def build_paths(active_zone="SE1"):
    base_dir = Path(__file__).resolve().parent
    return {
        "combined": base_dir / "master data files" / "2015-2025" / f"Combined_{active_zone}_Data_2015_2025.xlsx",
    }


def load_price_and_wind_data(paths, start_date=None, end_date=None):
    df_combined = pd.read_excel(
        paths["combined"],
        usecols=["Timestamp", "Spot_Price", "Wind_Forecast", "Consumption_Forecast"],
    )

    data = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(df_combined["Timestamp"]),
            "Price": pd.to_numeric(df_combined["Spot_Price"], errors="coerce"),
            "Wind_Forecast": pd.to_numeric(df_combined["Wind_Forecast"], errors="coerce"),
            "Consumption_Forecast": pd.to_numeric(df_combined["Consumption_Forecast"], errors="coerce"),
        }
    )

    if start_date is not None:
        start_ts = pd.to_datetime(start_date).normalize()
        data = data[data["Datetime"] >= start_ts]
    if end_date is not None:
        end_ts = (
            pd.to_datetime(end_date).normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(hours=1)
        )
        data = data[data["Datetime"] <= end_ts]

    data = data.sort_values("Datetime").set_index("Datetime")
    return data


# ---------------------------------------------------------------------------
# Step 0 — Fill isolated missing price observations
# ---------------------------------------------------------------------------

def fill_missing_prices(price):
    """
    For each NaN at t, replace with (price[t-1] + price[t+1]) / 2.
    Always uses the original series for neighbour lookup (no chaining).
    If only one immediate neighbour exists (sample edge), uses that value alone.
    """
    original = price.copy()
    filled = price.copy()
    n_filled = 0

    def _get_scalar(s, t):
        """Return first non-NaN value at t, handling duplicate DST timestamps."""
        if t not in s.index:
            return np.nan
        val = s[t]
        if isinstance(val, pd.Series):
            val = val.dropna()
            return float(val.iloc[0]) if not val.empty else np.nan
        return np.nan if pd.isna(val) else float(val)

    for t in original.index[original.isna()]:
        prev_t = t - pd.Timedelta(hours=1)
        next_t = t + pd.Timedelta(hours=1)
        neighbours = []
        v_prev = _get_scalar(original, prev_t)
        v_next = _get_scalar(original, next_t)
        if not np.isnan(v_prev):
            neighbours.append(v_prev)
        if not np.isnan(v_next):
            neighbours.append(v_next)
        if neighbours:
            filled[t] = np.mean(neighbours)
            n_filled += 1

    return filled, n_filled


# ---------------------------------------------------------------------------
# Steps 1 & 4 — Fredriksson / Li (2015) outlier replacement
# ---------------------------------------------------------------------------

def apply_outlier_filter(series):
    """
    Flag: value > mean + 6*sd  or  value < mean - 3.7*sd
    Replace each flagged observation with the mean of available values at
    t-24h, t-48h, t+24h, t+48h (always from the original, pre-replacement series).
    Returns: filtered series, n_outliers, mean, sd (all computed on original).
    """
    original = series.copy()
    s = series.copy()
    mean_val = float(original.mean())
    sd_val = float(original.std())

    upper = mean_val + 6.0 * sd_val
    lower = mean_val - 3.7 * sd_val

    outlier_mask = (original > upper) | (original < lower)
    n_outliers = int(outlier_mask.sum())

    for t in original.index[outlier_mask]:
        neighbours = []
        for delta_h in [24, -24, 48, -48]:
            nt = t + pd.Timedelta(hours=delta_h)
            if nt in original.index and not pd.isna(original[nt]):
                neighbours.append(float(original[nt]))
        if neighbours:
            s[t] = np.mean(neighbours)

    return s, n_outliers, mean_val, sd_val


# ---------------------------------------------------------------------------
# Step 2 — Seasonal OLS and Step 3 — Mean alignment
# ---------------------------------------------------------------------------

def build_holiday_dummy(index):
    """
    Binary dummy: 1 for any hour that falls on a public holiday in Sweden
    or Denmark AND is a weekday (Mon–Fri).
    """
    years = list(index.year.unique())
    se_hols = set(holidays_lib.Sweden(years=years).keys())
    dk_hols = set(holidays_lib.Denmark(years=years).keys())
    all_hols = se_hols | dk_hols

    flag = pd.Series(
        [
            1.0 if (ts.date() in all_hols and ts.dayofweek < 5) else 0.0
            for ts in index
        ],
        index=index,
        name="holiday",
    )
    return flag


def deseasonalize_price(price_filtered_1):
    """
    Step 2: OLS with constant, year dummies, month dummies, weekday dummies,
            hour-of-day dummies, and a weekday-holiday dummy.
    Step 3: Mean-align the residuals so that mean(u_aligned) == mean(price_filtered_1).

    Returns s_hat, u_raw, u_aligned, seasonal_model.
    """
    idx = price_filtered_1.index

    cal = pd.DataFrame(
        {
            "year": pd.Categorical(idx.year),
            "month": pd.Categorical(idx.month),
            "weekday": pd.Categorical(idx.dayofweek),
            "hour": pd.Categorical(idx.hour),
        },
        index=idx,
    )
    dummies = pd.get_dummies(
        cal,
        columns=["year", "month", "weekday", "hour"],
        drop_first=True,
        dtype=float,
    )
    holiday = build_holiday_dummy(idx)
    design = sm.add_constant(pd.concat([dummies, holiday], axis=1))

    # missing='drop' so NaN rows in endog are excluded from the fit but predictions
    # are still computed for the full design matrix (design is always complete).
    seasonal_model = sm.OLS(price_filtered_1, design, missing='drop').fit()
    s_hat = pd.Series(
        seasonal_model.predict(design.values), index=idx, name="seasonal_component"
    )
    u_raw = (price_filtered_1 - s_hat).rename("u_raw")  # NaN where endog was NaN

    # Step 3 — mean alignment (nanmean so NaN rows don't propagate)
    m1 = float(np.nanmean(price_filtered_1.values))
    m2 = float(np.nanmean(u_raw.values))
    u_aligned = (u_raw + (m1 - m2)).rename("deseasonalised_price_pre_outlier2")

    return s_hat, u_raw, u_aligned, seasonal_model


# ---------------------------------------------------------------------------
# Master price preprocessing function (Steps 0–5)
# ---------------------------------------------------------------------------

def preprocess_price(price_raw):
    """
    Full Fredriksson / Li (2015) price preprocessing pipeline.

    Parameters
    ----------
    price_raw : pd.Series  (DatetimeIndex, hourly)

    Returns
    -------
    result : pd.DataFrame
        price_raw | price_filtered_1 | seasonal_component
        | deseasonalised_price_pre_outlier2 | deseasonalised_price
        | log_deseasonalised_price
    summary : dict
        n_missing_filled_step0, n_outliers_replaced_step1,
        n_outliers_replaced_step4, plus mean & sd of each stage.
    """
    # Step 0
    price_gap_filled, n_filled = fill_missing_prices(price_raw)

    # Step 1
    price_filtered_1, n_out1, mean_raw, sd_raw = apply_outlier_filter(price_gap_filled)

    # Steps 2–3
    s_hat, u_raw, u_aligned, _ = deseasonalize_price(price_filtered_1)

    # Step 4
    deseasonalised_price, n_out2, mean_ds_pre, sd_ds_pre = apply_outlier_filter(u_aligned)

    # Step 5 — non-positive check before logging
    nonpositive_mask = deseasonalised_price <= 0
    n_nonpos = int(nonpositive_mask.sum())
    if n_nonpos > 0:
        print(f"\nOption A: excluding {n_nonpos} non-positive deseasonalised price observation(s).")
        deseasonalised_price = deseasonalised_price.where(~nonpositive_mask, other=np.nan)

    log_ds_price = np.log(deseasonalised_price).rename("log_deseasonalised_price")

    result = pd.DataFrame(
        {
            "price_raw": price_raw,
            "price_filtered_1": price_filtered_1,
            "seasonal_component": s_hat,
            "deseasonalised_price_pre_outlier2": u_aligned,
            "deseasonalised_price": deseasonalised_price,
            "log_deseasonalised_price": log_ds_price,
        }
    )

    summary = {
        "n_missing_filled_step0": n_filled,
        "n_outliers_replaced_step1": n_out1,
        "n_outliers_replaced_step4": n_out2,
        "n_nonpositive_dropped_step5": n_nonpos,
        "mean_price_raw": mean_raw,
        "sd_price_raw": sd_raw,
        "mean_price_filtered_1": float(price_filtered_1.mean()),
        "sd_price_filtered_1": float(price_filtered_1.std()),
        "mean_deseasonalised_pre_outlier2": float(u_aligned.mean()),
        "sd_deseasonalised_pre_outlier2": float(u_aligned.std()),
        "mean_deseasonalised": float(deseasonalised_price.mean()),
        "sd_deseasonalised": float(deseasonalised_price.std()),
        "mean_log_deseasonalised": float(log_ds_price.mean()),
        "sd_log_deseasonalised": float(log_ds_price.std()),
    }

    return result, summary


# ---------------------------------------------------------------------------
# Wind preprocessing (Steps 3–5)
# ---------------------------------------------------------------------------

def preprocess_wind(wind_raw):
    """
    Fredriksson wind preprocessing.

    Step 3 — Missing values: same neighbour-average rule as for price.
      For each missing t: wind[t] = mean of available values at t-1h and t+1h.
      Uses the original series for lookups (no chaining).

    Step 4 — No outlier filtering applied to wind.

    Step 5 — Log transform with non-positive guard.
      ln(wind_zone_mwh) applied directly, no floor or offset.
      If any value <= 0, execution stops and details are reported.

    Returns
    -------
    result : pd.DataFrame
        wind_forecast_zone_mwh  (raw, before any filling)
        wind_zone_mwh           (after missing-value filling)
        log_wind_zone           (natural log of wind_zone_mwh)
    summary : dict
    """
    # Step 3 — fill missing observations (same neighbour logic as price)
    wind_filled, n_filled = fill_missing_prices(wind_raw)

    # Step 5 — non-positive guard before logging
    nonpositive_mask = wind_filled <= 0
    n_nonpos = int(nonpositive_mask.sum())
    if n_nonpos > 0:
        bad = wind_filled[nonpositive_mask]
        print(f"\n{'=' * 60}")
        print(f"STOP: {n_nonpos} non-positive value(s) in wind_zone_mwh after gap filling.")
        print("Cannot take log. Fredriksson does not describe handling for these.")
        print("\nAffected timestamps and values:")
        print(bad.to_string())
        print(f"\nOption A (closest to thesis): drop the {n_nonpos} affected timestamp(s)")
        print("  from all series so the ARX sample is consistent.")
        print("Option B (pragmatic, not in thesis): replace non-positive values with")
        print("  the smallest strictly positive wind value in the sample, then log.")
        print(f"{'=' * 60}\n")
        raise ValueError(
            f"{n_nonpos} non-positive wind value(s) before log step. "
            "See output above. Provide explicit instruction to proceed."
        )

    log_wind = np.log(wind_filled).rename("log_wind_zone")

    result = pd.DataFrame(
        {
            "wind_forecast_zone_mwh": wind_raw,
            "wind_zone_mwh": wind_filled,
            "log_wind_zone": log_wind,
        }
    )

    summary = {
        "n_missing_filled": n_filled,
        "min_wind_zone_mwh": float(wind_filled.min()),
        "mean_wind_zone_mwh": float(wind_filled.mean()),
        "max_wind_zone_mwh": float(wind_filled.max()),
        "all_strictly_positive": bool((wind_filled.dropna() > 0).all()),
        "min_log_wind_zone": float(log_wind.min()),
        "mean_log_wind_zone": float(log_wind.mean()),
        "max_log_wind_zone": float(log_wind.max()),
    }

    return result, summary


# ---------------------------------------------------------------------------
# Consumption preprocessing (fill missing → deseasonalise → log)
# ---------------------------------------------------------------------------

def preprocess_consumption(consumption_raw):
    """
    Preprocessing for forecasted consumption.

    Step 1 — Fill isolated missing values: same neighbour-average rule as price/wind.
    Step 2 — Deseasonalise: same OLS methodology as price (calendar dummies + mean align).
              No outlier filtering before or after.
    Step 3 — Log transform. Non-positive values after mean alignment are reported and dropped.

    Returns
    -------
    result : pd.DataFrame
        consumption_forecast_raw  (original series)
        consumption_forecast      (after gap filling)
        consumption_log_ds        (log of mean-aligned deseasonalised series)
    summary : dict
    """
    # Fill missing
    consumption_filled, n_filled = fill_missing_prices(consumption_raw)

    # Deseasonalise (reuse price OLS machinery).
    _, _, u_aligned, _ = deseasonalize_price(consumption_filled)

    # Non-positive guard before logging
    nonpositive_mask = u_aligned <= 0
    n_nonpos = int(nonpositive_mask.sum())
    if n_nonpos > 0:
        bad = u_aligned[nonpositive_mask]
        print(f"\n{n_nonpos} non-positive deseasonalised consumption value(s) — dropping before log.")
        print(bad.to_string())
        u_aligned = u_aligned.where(~nonpositive_mask, other=np.nan)

    log_consumption = np.log(u_aligned).rename("consumption_log_ds")

    result = pd.DataFrame(
        {
            "consumption_forecast_raw": consumption_raw,
            "consumption_forecast": consumption_filled,
            "consumption_log_ds": log_consumption,
        }
    )

    summary = {
        "n_missing_filled": n_filled,
        "n_nonpositive_dropped": n_nonpos,
        "mean_consumption_raw": float(consumption_filled.mean()),
        "mean_consumption_log_ds": float(log_consumption.mean()),
        "sd_consumption_log_ds": float(log_consumption.std()),
    }

    return result, summary


# ---------------------------------------------------------------------------
# ARX mean model
# ---------------------------------------------------------------------------

def run_arx_model(data, ar_lags=1):
    """
    Estimate ARMAX with QML robust SEs (equivalent to Stata: arima ..., vce(robust)).
    ar_lags may be an int (contiguous lags 1..p) or a list of specific lags.
    """
    endog = pd.Series(data["Price_Log_DS"].to_numpy(), name="Price_Log_DS")
    exog = data[["Wind_Forecast_Log_DS"]].reset_index(drop=True)
    model = _ARIMA(
        endog=endog,
        exog=exog,
        order=(ar_lags, 0, 0),
        trend="c",
    ).fit(cov_type="robust")
    return model


def print_mean_model_results(result):
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ACTIVE_ZONE = "SE1"
    START_DATE = "2023-01-01"
    END_DATE = "2025-12-31"
    AR_LAGS = [1]
    PLOT_ONLY = True

    paths = build_paths(ACTIVE_ZONE)
    raw_data = load_price_and_wind_data(paths, start_date=START_DATE, end_date=END_DATE)

    if PLOT_ONLY:
        negative = raw_data["Price"][raw_data["Price"] < 0]
        pct_negative = 100 * len(negative) / len(raw_data["Price"].dropna())
        if len(negative) > 0:
            print(f"First negative price: {negative.index[0]}  ({negative.iloc[0]:.2f} EUR/MWh)")
        else:
            print("No negative prices in this period.")
        print(f"Negative prices: {len(negative)} obs  ({pct_negative:.2f}% of sample)")

        price = raw_data["Price"].dropna()
        mean_p = float(price.mean())
        sd_p = float(price.std())
        lower_bound = mean_p - 3 * sd_p
        upper_bound = mean_p + 3 * sd_p
        outliers = price[(price < lower_bound) | (price > upper_bound)]
        included = price[(price >= lower_bound) & (price <= upper_bound)]

        print(f"\n--- Before outlier removal ---")
        print(f"  Largest value:  {price.max():.2f} EUR/MWh  ({price.idxmax()})")
        print(f"  Smallest value: {price.min():.2f} EUR/MWh  ({price.idxmin()})")
        print(f"  Mean:           {mean_p:.2f} EUR/MWh")

        print(f"\n--- After ±3σ outlier removal ({len(outliers)} obs removed) ---")
        print(f"  Upper bound (mean + 3σ): {upper_bound:.2f} EUR/MWh")
        print(f"  Lower bound (mean − 3σ): {lower_bound:.2f} EUR/MWh")
        print(f"  Largest value kept:      {included.max():.2f} EUR/MWh")
        print(f"  Smallest value kept:     {included.min():.2f} EUR/MWh")

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(raw_data.index, raw_data["Price"], linewidth=0.6, color="steelblue", zorder=1)
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="Zero")
        ax.axhline(lower_bound, color="orange", linewidth=1.0, linestyle="--",
                   label=f"mean − 3σ  ({lower_bound:.1f})")
        ax.axhline(upper_bound, color="orange", linewidth=1.0, linestyle="--",
                   label=f"mean + 3σ  ({upper_bound:.1f})")
        if len(outliers) > 0:
            ax.scatter(outliers.index, outliers.values, color="red", s=12, zorder=2,
                       label=f"Outliers ({len(outliers)})")
        ax.set_title(f"Spot price — {ACTIVE_ZONE}  {START_DATE} to {END_DATE}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        # Price: full Fredriksson pipeline (Steps 0–5)
        price_result, price_summary = preprocess_price(raw_data["Price"])

        print(f"=== Price preprocessing summary: {ACTIVE_ZONE}  {START_DATE} to {END_DATE} ===")
        for k, v in price_summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Wind: Fredriksson pipeline (Steps 3–5, no deseasonalisation, no outlier filter)
        wind_result, wind_summary = preprocess_wind(raw_data["Wind_Forecast"])

        print(f"\n=== Wind preprocessing summary ===")
        for k, v in wind_summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Consumption: fill missing → deseasonalise (same OLS as price) → log
        consumption_result, consumption_summary = preprocess_consumption(raw_data["Consumption_Forecast"])

        print(f"\n=== Consumption preprocessing summary ===")
        for k, v in consumption_summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Align to timestamps where all series are non-missing
        model_data = pd.concat(
            [
                price_result["log_deseasonalised_price"].rename("Price_Log_DS"),
                wind_result["log_wind_zone"].rename("Wind_Forecast_Log_DS"),
                consumption_result["consumption_log_ds"].rename("Consumption_Log_DS"),
            ],
            axis=1,
        ).dropna()

        # Save model input columns
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"arx_input_{ACTIVE_ZONE}_{START_DATE}_{END_DATE}.csv"
        model_data.to_csv(out_path)
        print(f"\nSaved ARX input ({len(model_data)} obs) to {out_path}")

        # Run ARX model
        model = run_arx_model(model_data, ar_lags=AR_LAGS)
        print(f"Date range: {model_data.index.min()} to {model_data.index.max()}")
        print_mean_model_results(model)
