"""
congestion_spillover_export.py

Builds SE3 / SE4 regression exports for testing whether the northern spillover
effect weakens when the relevant corridor is congested.

Variables added:
- north_wind_log   = ln(wind_SE1 + wind_SE2)
- d_congested      = 1 if the relevant raw price corridor is binding
- north_wind_cong  = north_wind_log * d_congested

Corridors:
- SE3: SE2-SE3
- SE4: SE3-SE4
"""

import io
import os
import contextlib
from datetime import date

import numpy as np
import pandas as pd

from preprocessing import load_data, preprocess_data_for_regression


ZONES = ["SE3", "SE4"]
START_DATE = date.fromisoformat(os.getenv("START_DATE", "2025-01-01"))
END_DATE = date.fromisoformat(os.getenv("END_DATE", "2025-12-31"))
SEASONAL_INTERACTIONS = "hour_dow_month"
USE_BILATERAL_EXCHANGE = True
CONGESTION_EPSILON_EUR = 5.0

CONTROLS = {
    "Wind": True,
    "Hydro": False,
    "Consumption": True,
    "Oil": False,
    "Gas": False,
}

STATA_RENAME = {
    "Price_DS": "price_ds",
    "Wind_Forecast_Log": "wind_log",
    "Consumption_Log_Deseasonalized": "consump_log_ds",
}

_CONTROL_COLS = {
    "Wind": "Wind_Forecast_Log",
    "Hydro": "Hydro_Reserves_Log_Deseasonalized",
    "Consumption": "Consumption_Log_Deseasonalized",
    "Oil": "Oil_Price_Log",
    "Gas": "Gas_Price_Log",
}
EXOG_VARS = [_CONTROL_COLS[k] for k, v in CONTROLS.items() if v]

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


def _build_auxiliary_frame() -> pd.DataFrame:
    se1 = _read_combined("SE1", ["Timestamp", "Wind_Forecast"])
    se2 = _read_combined("SE2", ["Timestamp", "Wind_Forecast", "Spot_Price"])
    se3 = _read_combined("SE3", ["Timestamp", "Spot_Price"])
    se4 = _read_combined("SE4", ["Timestamp", "Spot_Price"])

    aux = se1.rename(columns={"Wind_Forecast": "wind_se1_raw"})
    aux = aux.merge(
        se2.rename(columns={"Wind_Forecast": "wind_se2_raw", "Spot_Price": "price_se2_raw"}),
        on=["Timestamp", "occurrence"],
        how="inner",
    )
    aux = aux.merge(
        se3.rename(columns={"Spot_Price": "price_se3_raw"}),
        on=["Timestamp", "occurrence"],
        how="inner",
    )
    aux = aux.merge(
        se4.rename(columns={"Spot_Price": "price_se4_raw"}),
        on=["Timestamp", "occurrence"],
        how="inner",
    )

    aux["north_wind_raw"] = aux["wind_se1_raw"] + aux["wind_se2_raw"]
    aux["north_wind_log"] = np.log(aux["north_wind_raw"].clip(lower=0.01))
    aux["d_congested_se2se3"] = (
        (aux["price_se2_raw"] - aux["price_se3_raw"]).abs() > CONGESTION_EPSILON_EUR
    ).astype(int)
    aux["d_congested_se3se4"] = (
        (aux["price_se3_raw"] - aux["price_se4_raw"]).abs() > CONGESTION_EPSILON_EUR
    ).astype(int)

    return aux[
        [
            "Timestamp",
            "occurrence",
            "north_wind_log",
            "d_congested_se2se3",
            "d_congested_se3se4",
        ]
    ]


def build_zone_export(zone: str, aux: pd.DataFrame) -> pd.DataFrame:
    df_raw = _load_zone_raw(zone)
    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    base = df_raw.reset_index().rename(columns={"Datetime": "Timestamp"})
    base["occurrence"] = base.groupby("Timestamp").cumcount()
    df_proc_reset = df_proc.reset_index().rename(columns={"Datetime": "Timestamp"})
    df_proc_reset["occurrence"] = df_proc_reset.groupby("Timestamp").cumcount()
    df_proc_reset = df_proc_reset.merge(
        base[["Timestamp", "occurrence"]],
        on=["Timestamp", "occurrence"],
        how="left",
    )
    df_proc_reset = df_proc_reset.merge(aux, on=["Timestamp", "occurrence"], how="left")

    if zone == "SE3":
        df_proc_reset["d_congested"] = df_proc_reset["d_congested_se2se3"]
    else:
        df_proc_reset["d_congested"] = df_proc_reset["d_congested_se3se4"]

    df_proc_reset["north_wind_cong"] = (
        df_proc_reset["north_wind_log"] * df_proc_reset["d_congested"]
    )

    standard_exog = [v for v in EXOG_VARS if v in df_proc_reset.columns]
    netexch_cols = sorted(c for c in df_proc_reset.columns if c.startswith("NetExch_"))
    export_cols = (
        ["Price_DS"]
        + standard_exog
        + ["north_wind_log", "d_congested", "north_wind_cong"]
        + netexch_cols
    )

    out = df_proc_reset[["Timestamp"] + export_cols].copy()
    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in out.columns}
    rename_dict.update({col: col.lower() for col in netexch_cols})
    out = out.rename(columns=rename_dict)
    out = out.rename(columns={"Timestamp": "timestamp"})
    return out


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    aux = _build_auxiliary_frame()

    print(f"Congestion spillover export: {start_str} -> {end_str}")
    print(f"Congestion epsilon: {CONGESTION_EPSILON_EUR} EUR/MWh")
    print(f"Output dir: {OUT_DIR}")

    for zone in ZONES:
        zone_df = build_zone_export(zone, aux)
        out_path = os.path.join(
            OUT_DIR, f"congestion_spillover_{zone}_{start_str}_{end_str}_log.csv"
        )
        zone_df.to_csv(out_path, index=False)
        print(f"  [{zone}] wrote {len(zone_df)} rows -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
