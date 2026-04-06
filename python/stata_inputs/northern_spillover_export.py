"""
northern_spillover_export.py

Builds SE3/SE4 regression exports for testing whether northern wind surplus
affects southern prices beyond local wind. Uses the same preprocessing pipeline
as the existing full-period / ACT exports, then adds:

- north_wind_log = ln(wind_SE1 + wind_SE2)

Exports:
- stata_input/northern_spillover/northern_spillover_SE3_2024-01-01_2025-12-31_log.csv
- stata_input/northern_spillover/northern_spillover_SE4_2024-01-01_2025-12-31_log.csv
"""

import io
import os
import sys
import contextlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from preprocessing import load_data, preprocess_data_for_regression


ZONES = ["SE3", "SE4"]
START_DATE = date.fromisoformat(os.getenv("START_DATE", "2025-01-01"))
END_DATE = date.fromisoformat(os.getenv("END_DATE", "2025-12-31"))
SEASONAL_INTERACTIONS = "hour_dow_month"

USE_BILATERAL_EXCHANGE = True
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

BASE_DIR = str(PROJECT_ROOT)
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


def _load_north_wind_raw() -> pd.DataFrame:
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE) + pd.Timedelta(hours=23)

    def read_zone(zone: str, label: str) -> pd.DataFrame:
        path = os.path.join(
            BASE_DIR, "master data files", "2015-2025", f"Combined_{zone}_Data_2015_2025.xlsx"
        )
        df = pd.read_excel(path, usecols=["Timestamp", "Wind_Forecast"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].copy()
        df["occurrence"] = df.groupby("Timestamp").cumcount()
        df = df.rename(columns={"Wind_Forecast": label})
        return df

    se1 = read_zone("SE1", "wind_se1_raw")
    se2 = read_zone("SE2", "wind_se2_raw")
    north = se1.merge(se2, on=["Timestamp", "occurrence"], how="inner")
    north["north_wind_raw"] = north["wind_se1_raw"] + north["wind_se2_raw"]
    return north[["Timestamp", "occurrence", "north_wind_raw"]]


def build_zone_export(zone: str, north_wind_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = _load_zone_raw(zone)
    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    # Align northern wind to the same timestamp/duplicate-hour structure used by load_data.
    base = df_raw.reset_index().rename(columns={"Datetime": "Timestamp"})
    base["occurrence"] = base.groupby("Timestamp").cumcount()
    base = base[["Timestamp", "occurrence"]]

    north = base.merge(north_wind_raw, on=["Timestamp", "occurrence"], how="left")
    north["north_wind_raw"] = north["north_wind_raw"].interpolate(
        method="linear", limit_direction="both"
    )
    north["north_wind_log"] = np.log(north["north_wind_raw"].clip(lower=0.01))

    north_log = north[["Timestamp", "occurrence", "north_wind_log"]].copy()
    df_proc_reset = df_proc.reset_index().rename(columns={"Datetime": "Timestamp"})
    df_proc_reset["occurrence"] = df_proc_reset.groupby("Timestamp").cumcount()
    df_proc_reset = df_proc_reset.merge(north_log, on=["Timestamp", "occurrence"], how="left")

    standard_exog = [v for v in EXOG_VARS if v in df_proc_reset.columns]
    netexch_cols = sorted(c for c in df_proc_reset.columns if c.startswith("NetExch_"))
    export_cols = ["Price_DS"] + standard_exog + ["north_wind_log"] + netexch_cols

    out = df_proc_reset[["Timestamp"] + export_cols].copy()
    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in out.columns}
    rename_dict.update({col: col.lower() for col in netexch_cols})
    out = out.rename(columns=rename_dict)
    out = out.rename(columns={"Timestamp": "timestamp"})
    return out


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    print(f"Northern spillover export: {start_str} -> {end_str}")
    print(f"Zones: {', '.join(ZONES)}")
    print(f"Output dir: {OUT_DIR}")

    north_wind_raw = _load_north_wind_raw()
    for zone in ZONES:
        zone_df = build_zone_export(zone, north_wind_raw)
        zone_path = os.path.join(
            OUT_DIR,
            f"northern_spillover_{zone}_{start_str}_{end_str}_log.csv",
        )
        zone_df.to_csv(zone_path, index=False)
        print(f"  [{zone}] wrote {len(zone_df)} rows -> {os.path.basename(zone_path)}")


if __name__ == "__main__":
    main()
