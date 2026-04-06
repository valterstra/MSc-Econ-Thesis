"""
act_interaction_export.py

Builds ACT-break regression exports for all four price zones using the same
preprocessing pipeline and control specification as rolling_window_export.py,
but on one common 2024-01-01 to 2025-12-31 sample, then adds:

- D_post_ACT
- wind_post_act = wind_log * D_post_ACT

Outputs:
- Per-zone Stata-ready CSVs for separate equations
- One stacked all-zones CSV with common columns only
"""

import io
import os
import contextlib
from datetime import date

import pandas as pd

from preprocessing import load_data, preprocess_data_for_regression


ZONES = ["SE1", "SE2", "SE3", "SE4"]
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
ACT_DATE = pd.Timestamp("2024-10-30 00:00:00")

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
    "Hydro_Reserves_Log_Deseasonalized": "hydro_log_ds",
    "Consumption_Log_Deseasonalized": "consump_log_ds",
    "Oil_Price_Log": "oil_log",
    "Gas_Price_Log": "gas_log",
}

_CONTROL_COLS = {
    "Wind": "Wind_Forecast_Log",
    "Hydro": "Hydro_Reserves_Log_Deseasonalized",
    "Consumption": "Consumption_Log_Deseasonalized",
    "Oil": "Oil_Price_Log",
    "Gas": "Gas_Price_Log",
}
EXOG_VARS = [_CONTROL_COLS[k] for k, v in CONTROLS.items() if v]

COMMON_STACKED_COLS = [
    "timestamp",
    "zone",
    "price_ds",
    "wind_log",
    "consump_log_ds",
    "d_post_act",
    "wind_post_act",
]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "stata_input", "act_interaction")
os.makedirs(OUT_DIR, exist_ok=True)

SHARED_PATHS = {
    "hydro": os.path.join(BASE_DIR, "master data files", "Master_Hydro_Reservoir.xlsx"),
    "crude_oil": os.path.join(
        BASE_DIR, "master data files", "2015-2025", "Light_Crude_Oil_2015_2025.xlsx"
    ),
    "commodities": os.path.join(BASE_DIR, "master data files", "Master_Commodities.xlsx"),
}


def build_zone_export(zone: str) -> pd.DataFrame:
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
            use_bilateral_exchange=USE_BILATERAL_EXCHANGE,
        )
        df_proc = preprocess_data_for_regression(
            df_raw,
            suppress_output=True,
            seasonal_interactions=SEASONAL_INTERACTIONS,
        )

    standard_exog = [v for v in EXOG_VARS if v in df_proc.columns]
    if USE_BILATERAL_EXCHANGE:
        netexch_cols = sorted(c for c in df_proc.columns if c.startswith("NetExch_"))
    else:
        netexch_cols = ["Net_Exchange"] if "Net_Exchange" in df_proc.columns else []
    export_cols = ["Price_DS"] + standard_exog + netexch_cols

    out = df_proc[export_cols].copy()
    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in out.columns}
    if USE_BILATERAL_EXCHANGE:
        rename_dict.update({col: col.lower() for col in netexch_cols})
    elif "Net_Exchange" in out.columns:
        rename_dict["Net_Exchange"] = "net_exchange"
    out = out.rename(columns=rename_dict)

    out.index.name = "timestamp"
    out["zone"] = zone
    out["d_post_act"] = (out.index >= ACT_DATE).astype(int)
    out["wind_post_act"] = out["wind_log"] * out["d_post_act"]

    ordered = ["zone", "d_post_act", "wind_post_act", "price_ds", "wind_log"]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]
    return out


def main() -> None:
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    file_suffix = "log" if USE_BILATERAL_EXCHANGE else "log_netexch"
    zone_frames = []

    print(f"ACT interaction export: {start_str} -> {end_str}")
    print(f"ACT break date: {ACT_DATE}")
    print(f"Output dir: {OUT_DIR}")

    for zone in ZONES:
        print(f"\n{zone}")
        zone_df = build_zone_export(zone)
        zone_frames.append(zone_df.reset_index())

        zone_path = os.path.join(
            OUT_DIR,
            f"act_interaction_{zone}_{start_str}_{end_str}_{file_suffix}.csv",
        )
        zone_df.to_csv(zone_path)
        print(f"  wrote {len(zone_df)} rows -> {os.path.basename(zone_path)}")

    stacked = pd.concat(zone_frames, ignore_index=True, sort=False)
    stacked = stacked.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    stacked_common = stacked.reindex(columns=COMMON_STACKED_COLS)
    stacked_path = os.path.join(
        OUT_DIR,
        f"act_interaction_allzones_{start_str}_{end_str}_{file_suffix}.csv",
    )
    stacked_common.to_csv(stacked_path, index=False)

    meta = pd.DataFrame(
        {
            "start_date": [start_str],
            "end_date": [end_str],
            "act_break_date": [str(ACT_DATE)],
            "seasonal_interactions": [SEASONAL_INTERACTIONS],
            "use_interpolation": [1],
            "lag_commodity_hours": [24],
            "use_bilateral_exchange": [int(USE_BILATERAL_EXCHANGE)],
            "controls": [",".join([k for k, v in CONTROLS.items() if v])],
            "per_zone_files": [1],
            "stacked_common_file": [1],
        }
    )
    meta_path = os.path.join(
        OUT_DIR,
        f"act_interaction_meta_{start_str}_{end_str}_{file_suffix}.csv",
    )
    meta.to_csv(meta_path, index=False)

    print(f"\nStacked file -> {os.path.basename(stacked_path)}")
    print(f"Metadata     -> {os.path.basename(meta_path)}")


if __name__ == "__main__":
    main()
