"""
se2_congestion_spillover_export.py

SE2 congestion-spillover export:
- north_wind_log = ln(wind_SE1)
- d_congested    = 1 if |P_SE1 - P_SE2| > threshold
- north_wind_cong = north_wind_log * d_congested
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
CONGESTION_EPSILON_EUR = float(os.getenv("CONGESTION_EPSILON_EUR", "5.0"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "stata_input", "congestion_spillover")
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


def _load_se2_raw() -> pd.DataFrame:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    paths = {
        **SHARED_PATHS,
        "combined": os.path.join(
            BASE_DIR, "master data files", "2015-2025", "Combined_SE2_Data_2015_2025.xlsx"
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        return load_data(
            paths,
            target_region="SE2",
            zone_hydro="SE2",
            use_interpolation=True,
            start_date=start_str,
            end_date=end_str,
            lag_commodity_hours=24,
            use_bilateral_exchange=USE_BILATERAL_EXCHANGE,
        )


def _read_combined(zone: str, cols: list[str]) -> pd.DataFrame:
    start_ts, end_ts = _window_bounds()
    path = os.path.join(
        BASE_DIR, "master data files", "2015-2025", f"Combined_{zone}_Data_2015_2025.xlsx"
    )
    df = pd.read_excel(path, usecols=cols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].copy()
    df["occurrence"] = df.groupby("Timestamp").cumcount()
    return df


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    df_raw = _load_se2_raw()
    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    se1 = _read_combined("SE1", ["Timestamp", "Wind_Forecast", "Spot_Price"])
    se2 = _read_combined("SE2", ["Timestamp", "Spot_Price"])

    aux = se1.rename(columns={"Wind_Forecast": "north_wind_raw", "Spot_Price": "price_se1_raw"})
    aux = aux.merge(
        se2.rename(columns={"Spot_Price": "price_se2_raw"}),
        on=["Timestamp", "occurrence"],
        how="inner",
    )
    aux["north_wind_log"] = np.log(aux["north_wind_raw"].clip(lower=0.01))
    aux["d_congested"] = (
        (aux["price_se1_raw"] - aux["price_se2_raw"]).abs() > CONGESTION_EPSILON_EUR
    ).astype(int)
    aux["north_wind_cong"] = aux["north_wind_log"] * aux["d_congested"]

    df_proc_reset = df_proc.reset_index().rename(columns={"Datetime": "Timestamp"})
    df_proc_reset["occurrence"] = df_proc_reset.groupby("Timestamp").cumcount()
    df_proc_reset = df_proc_reset.merge(
        aux[["Timestamp", "occurrence", "north_wind_log", "d_congested", "north_wind_cong"]],
        on=["Timestamp", "occurrence"],
        how="left",
    )

    netexch_cols = sorted(c for c in df_proc_reset.columns if c.startswith("NetExch_"))
    out = df_proc_reset[
        ["Timestamp", "Price_DS", "Wind_Forecast_Log", "Consumption_Log_Deseasonalized",
         "north_wind_log", "d_congested", "north_wind_cong"] + netexch_cols
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
        OUT_DIR, f"congestion_spillover_SE2_{start_str}_{end_str}_log.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
