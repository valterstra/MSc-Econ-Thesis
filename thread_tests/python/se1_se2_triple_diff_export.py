"""
se1_se2_triple_diff_export.py

Builds a stacked SE1/SE2 panel for a post-FBMC triple-difference test.

Main coefficient of interest in Stata:
- wind_se2_post_fbmc = wind_log * is_se2 * post_fbmc
"""

import contextlib
import io
import os
from datetime import date

import pandas as pd

from preprocessing import load_data, preprocess_data_for_regression


ZONES = ["SE1", "SE2"]
START_DATE = date.fromisoformat(os.getenv("START_DATE", "2024-01-01"))
END_DATE = date.fromisoformat(os.getenv("END_DATE", "2025-12-31"))
FBMC_DATE = pd.Timestamp(os.getenv("FBMC_DATE", "2024-10-30 00:00:00"))
SEASONAL_INTERACTIONS = "hour_dow_month"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "stata_input", "triple_diff")
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    "hydro": os.path.join(BASE_DIR, "master data files", "Master_Hydro_Reservoir.xlsx"),
    "crude_oil": os.path.join(
        BASE_DIR, "master data files", "2015-2025", "Light_Crude_Oil_2015_2025.xlsx"
    ),
    "commodities": os.path.join(BASE_DIR, "master data files", "Master_Commodities.xlsx"),
}


def _load_zone(zone: str) -> pd.DataFrame:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    paths = {
        **SHARED_PATHS,
        "combined": os.path.join(
            BASE_DIR, "master data files", "2015-2025", f"Combined_{zone}_Data_2015_2025.xlsx"
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths,
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=start_str,
            end_date=end_str,
            lag_commodity_hours=24,
            use_bilateral_exchange=True,
        )
        df_proc = preprocess_data_for_regression(
            df_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )

    out = df_proc[["Price_DS", "Wind_Forecast_Log", "Consumption_Log_Deseasonalized"]].copy()
    out = out.reset_index().rename(
        columns={
            "Datetime": "timestamp",
            "Price_DS": "price_ds",
            "Wind_Forecast_Log": "wind_log",
            "Consumption_Log_Deseasonalized": "consump_log_ds",
        }
    )
    out["occurrence"] = out.groupby("timestamp").cumcount()
    out["zone"] = zone
    out["is_se2"] = int(zone == "SE2")
    out["post_fbmc"] = (out["timestamp"] >= FBMC_DATE).astype(int)
    out["se2_post"] = out["is_se2"] * out["post_fbmc"]
    out["wind_se2"] = out["wind_log"] * out["is_se2"]
    out["wind_post_fbmc"] = out["wind_log"] * out["post_fbmc"]
    out["wind_se2_post_fbmc"] = out["wind_log"] * out["is_se2"] * out["post_fbmc"]
    return out


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    frames = [_load_zone(zone) for zone in ZONES]
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "occurrence", "zone"]
    ).reset_index(drop=True)

    out_path = os.path.join(
        OUT_DIR,
        f"triple_diff_SE1_SE2_{start_str}_{end_str}_log.csv",
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
