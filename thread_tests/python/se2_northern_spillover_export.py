"""
se2_northern_spillover_export.py

Builds an SE2 regression export for testing whether SE1 wind affects SE2 prices
beyond SE2's own local wind.

Adds:
- north_wind_log = ln(wind_SE1)
"""

import io
import os
import contextlib
from datetime import date

import numpy as np
import pandas as pd

from preprocessing import load_data, preprocess_data_for_regression


START_DATE = date.fromisoformat(os.getenv("START_DATE", "2025-01-01"))
END_DATE = date.fromisoformat(os.getenv("END_DATE", "2025-12-31"))
SEASONAL_INTERACTIONS = "hour_dow_month"
USE_BILATERAL_EXCHANGE = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "stata_input", "northern_spillover")
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    "hydro": os.path.join(BASE_DIR, "master data files", "Master_Hydro_Reservoir.xlsx"),
    "crude_oil": os.path.join(
        BASE_DIR, "master data files", "2015-2025", "Light_Crude_Oil_2015_2025.xlsx"
    ),
    "commodities": os.path.join(BASE_DIR, "master data files", "Master_Commodities.xlsx"),
}


def _load_zone_raw(zone: str) -> pd.DataFrame:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    paths = {
        **SHARED_PATHS,
        "combined": os.path.join(
            BASE_DIR, "master data files", "2015-2025", f"Combined_{zone}_Data_2015_2025.xlsx"
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        return load_data(
            paths,
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=start_str,
            end_date=end_str,
            lag_commodity_hours=24,
            use_bilateral_exchange=USE_BILATERAL_EXCHANGE,
        )


def _load_se1_wind_raw() -> pd.DataFrame:
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE) + pd.Timedelta(hours=23)
    path = os.path.join(
        BASE_DIR, "master data files", "2015-2025", "Combined_SE1_Data_2015_2025.xlsx"
    )
    df = pd.read_excel(path, usecols=["Timestamp", "Wind_Forecast"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].copy()
    df["occurrence"] = df.groupby("Timestamp").cumcount()
    df = df.rename(columns={"Wind_Forecast": "north_wind_raw"})
    return df


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    df_raw = _load_zone_raw("SE2")
    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    base = df_raw.reset_index().rename(columns={"Datetime": "Timestamp"})
    base["occurrence"] = base.groupby("Timestamp").cumcount()
    base = base[["Timestamp", "occurrence"]]

    se1_wind = _load_se1_wind_raw()
    merged = base.merge(se1_wind, on=["Timestamp", "occurrence"], how="left")
    merged["north_wind_raw"] = merged["north_wind_raw"].interpolate(
        method="linear", limit_direction="both"
    )
    merged["north_wind_log"] = np.log(merged["north_wind_raw"].clip(lower=0.01))

    north_log = merged[["Timestamp", "occurrence", "north_wind_log"]]
    df_proc_reset = df_proc.reset_index().rename(columns={"Datetime": "Timestamp"})
    df_proc_reset["occurrence"] = df_proc_reset.groupby("Timestamp").cumcount()
    df_proc_reset = df_proc_reset.merge(north_log, on=["Timestamp", "occurrence"], how="left")

    netexch_cols = sorted(c for c in df_proc_reset.columns if c.startswith("NetExch_"))
    out = df_proc_reset[
        ["Timestamp", "Price_DS", "Wind_Forecast_Log", "Consumption_Log_Deseasonalized", "north_wind_log"]
        + netexch_cols
    ].copy()
    out = out.rename(
        columns={
            "Timestamp": "timestamp",
            "Price_DS": "price_ds",
            "Wind_Forecast_Log": "wind_log",
            "Consumption_Log_Deseasonalized": "consump_log_ds",
            **{col: col.lower() for col in netexch_cols},
        }
    )

    out_path = os.path.join(
        OUT_DIR, f"northern_spillover_SE2_{start_str}_{end_str}_log.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
