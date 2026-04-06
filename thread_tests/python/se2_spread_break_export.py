"""
se2_spread_break_export.py

Builds a Stata-ready SE2 spread-break export over one common sample.

Dependent variables:
- spread_se2_se1 = raw SpotPrice_SE2 - SpotPrice_SE1
- spread_se2_se3 = raw SpotPrice_SE2 - SpotPrice_SE3

Key regressors:
- wind_log (processed SE2 local wind)
- post_fbmc
- wind_post_fbmc = wind_log * post_fbmc
- consump_log_ds
"""

import contextlib
import io
import os
from datetime import date

import pandas as pd

from preprocessing import load_data, preprocess_data_for_regression


START_DATE = date.fromisoformat(os.getenv("START_DATE", "2024-01-01"))
END_DATE = date.fromisoformat(os.getenv("END_DATE", "2025-12-31"))
FBMC_DATE = pd.Timestamp(os.getenv("FBMC_DATE", "2024-10-30 00:00:00"))
SEASONAL_INTERACTIONS = "hour_dow_month"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "stata_input", "spread_break")
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    "hydro": os.path.join(BASE_DIR, "master data files", "Master_Hydro_Reservoir.xlsx"),
    "crude_oil": os.path.join(
        BASE_DIR, "master data files", "2015-2025", "Light_Crude_Oil_2015_2025.xlsx"
    ),
    "commodities": os.path.join(BASE_DIR, "master data files", "Master_Commodities.xlsx"),
}


def _window_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE) + pd.Timedelta(hours=23)
    return start_ts, end_ts


def _load_processed_se2() -> pd.DataFrame:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    paths = {
        **SHARED_PATHS,
        "combined": os.path.join(
            BASE_DIR, "master data files", "2015-2025", "Combined_SE2_Data_2015_2025.xlsx"
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths,
            target_region="SE2",
            zone_hydro="SE2",
            use_interpolation=True,
            start_date=start_str,
            end_date=end_str,
            lag_commodity_hours=24,
            use_bilateral_exchange=True,
        )
        return preprocess_data_for_regression(
            df_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )


def _read_spot_price(zone: str) -> pd.DataFrame:
    start_ts, end_ts = _window_bounds()
    path = os.path.join(
        BASE_DIR, "master data files", "2015-2025", f"Combined_{zone}_Data_2015_2025.xlsx"
    )
    df = pd.read_excel(path, usecols=["Timestamp", "Spot_Price"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].copy()
    df["occurrence"] = df.groupby("Timestamp").cumcount()
    return df.rename(columns={"Spot_Price": f"price_{zone.lower()}_raw"})


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    df_proc = _load_processed_se2()
    df_proc = df_proc.reset_index().rename(columns={"Datetime": "Timestamp"})
    df_proc["occurrence"] = df_proc.groupby("Timestamp").cumcount()

    se1 = _read_spot_price("SE1")
    se2 = _read_spot_price("SE2")
    se3 = _read_spot_price("SE3")

    prices = se2.merge(se1, on=["Timestamp", "occurrence"], how="inner")
    prices = prices.merge(se3, on=["Timestamp", "occurrence"], how="inner")
    prices["spread_se2_se1"] = prices["price_se2_raw"] - prices["price_se1_raw"]
    prices["spread_se2_se3"] = prices["price_se2_raw"] - prices["price_se3_raw"]

    out = df_proc.merge(
        prices[
            [
                "Timestamp",
                "occurrence",
                "spread_se2_se1",
                "spread_se2_se3",
                "price_se1_raw",
                "price_se2_raw",
                "price_se3_raw",
            ]
        ],
        on=["Timestamp", "occurrence"],
        how="inner",
    )

    out = out[
        [
            "Timestamp",
            "spread_se2_se1",
            "spread_se2_se3",
            "price_se1_raw",
            "price_se2_raw",
            "price_se3_raw",
            "Wind_Forecast_Log",
            "Consumption_Log_Deseasonalized",
        ]
    ].copy()
    out["post_fbmc"] = (out["Timestamp"] >= FBMC_DATE).astype(int)
    out["wind_post_fbmc"] = out["Wind_Forecast_Log"] * out["post_fbmc"]

    out = out.rename(
        columns={
            "Timestamp": "timestamp",
            "Wind_Forecast_Log": "wind_log",
            "Consumption_Log_Deseasonalized": "consump_log_ds",
        }
    )

    out_path = os.path.join(
        OUT_DIR,
        f"spread_break_SE2_{start_str}_{end_str}_log.csv",
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
