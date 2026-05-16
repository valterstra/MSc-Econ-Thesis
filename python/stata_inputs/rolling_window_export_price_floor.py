"""
rolling_window_export_price_floor.py

Exports annual rolling-window Stata inputs for the capped-price log robustness
check. The baseline shifted-log exporter is untouched.

Price transformation:
    Price_Log = log(max(Price, 0.01))

Output:
    stata_input/rolling_windows/
        armax_input_{ZONE}_{YEAR}-01-01_{YEAR}-12-31_log_floor001.csv
        armax_meta_{ZONE}_{YEAR}-01-01_{YEAR}-12-31_log_floor001.csv

In Stata, use:
    global PIPELINE "log_floor001"

By default this exporter runs SE1 only. To run several zones later:
    $env:ZONES = "SE1 SE2 SE3 SE4"; python python/stata_inputs/rolling_window_export_price_floor.py
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from preprocessing.preprocessing_floor import load_data, preprocess_data_for_regression


ZONES = os.getenv("ZONES", "SE1").split()
WINDOW_YEARS = 1
STEP_YEARS = 1
START_YEAR = 2015
END_YEAR = 2025
WINDOWS = [
    (y, y + WINDOW_YEARS - 1)
    for y in range(START_YEAR, END_YEAR - WINDOW_YEARS + 2, STEP_YEARS)
]

ARMA_ORDER = (1, 0, 1)
SEASONAL_INTERACTIONS = "hour_dow_month"
PIPELINE = "log_floor001"
OVERWRITE = True

USE_BILATERAL_EXCHANGE = False
CONTROLS = {
    "Wind": True,
    "Hydro": True,
    "Consumption": True,
    "Oil": True,
    "Gas": True,
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

OUT_DIR = PROJECT_ROOT / "stata_input" / "rolling_windows"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_PATHS = {
    "hydro": "master data files/Master_Hydro_Reservoir.xlsx",
    "crude_oil": "master data files/2015-2025/Light_Crude_Oil_2015_2025.xlsx",
    "commodities": "master data files/Master_Commodities.xlsx",
}


def load_bilateral_exchange(zone: str, start_str: str, end_str: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "master data files" / "verified exchange" / f"Verified_{zone}_Exchange_2015_2025.xlsx"
    df = pd.read_excel(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    prefix = f"{zone}_NetExch_"
    bilateral_cols = [c for c in df.columns if c.startswith(prefix)]
    if not bilateral_cols:
        raise ValueError(f"No verified bilateral exchange columns found in {path}")

    rename_map = {col: col.replace(f"{zone}_", "", 1) for col in bilateral_cols}
    out = df[["Timestamp", *bilateral_cols]].rename(columns=rename_map).copy()
    out = out[
        (out["Timestamp"] >= pd.to_datetime(start_str))
        & (out["Timestamp"] <= pd.to_datetime(end_str))
    ]
    out["Occurrence"] = out.groupby("Timestamp").cumcount()
    return out


def attach_bilateral_exchange(output_df: pd.DataFrame, zone: str, start_str: str, end_str: str) -> pd.DataFrame:
    bilateral = load_bilateral_exchange(zone, start_str, end_str)

    merged = output_df.reset_index().rename(columns={"index": "Timestamp", "timestamp": "Timestamp"})
    merged["Timestamp"] = pd.to_datetime(merged["Timestamp"])
    merged["Occurrence"] = merged.groupby("Timestamp").cumcount()
    merged["TargetCount"] = merged.groupby("Timestamp")["Occurrence"].transform("count")

    bilateral_counts = bilateral.groupby("Timestamp")["Occurrence"].count()
    merged["BilateralCount"] = merged["Timestamp"].map(bilateral_counts)
    if merged["BilateralCount"].isna().any():
        missing_ts = merged.loc[merged["BilateralCount"].isna(), "Timestamp"].drop_duplicates().head(5)
        raise ValueError(
            f"Missing verified bilateral exchange timestamps for {zone} {start_str}-{end_str}: "
            f"{missing_ts.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()}"
        )

    # DST fall-back hours can appear more often in price data than in exchange
    # data. Match the existing rolling inputs by repeating each verified
    # exchange observation across the corresponding block of duplicate price rows.
    merged["BilateralOccurrence"] = (
        merged["Occurrence"] * merged["BilateralCount"].astype(int) // merged["TargetCount"]
    ).astype(int)

    bilateral = bilateral.rename(columns={"Occurrence": "BilateralOccurrence"})
    merged = merged.merge(
        bilateral,
        on=["Timestamp", "BilateralOccurrence"],
        how="left",
        validate="many_to_one",
    )
    if merged.filter(regex=r"^NetExch_").isna().any().any():
        raise ValueError(f"Missing verified bilateral exchange values after merge for {zone} {start_str}-{end_str}")

    merged = merged.drop(
        columns=["Occurrence", "TargetCount", "BilateralCount", "BilateralOccurrence"]
    ).rename(columns={"Timestamp": "timestamp"})
    return merged.set_index("timestamp")


def export_window(zone: str, start_year: int, end_year: int) -> None:
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    baseline_path = OUT_DIR / f"armax_input_{zone}_{start}_{end}_log.csv"
    csv_path = OUT_DIR / f"armax_input_{zone}_{start}_{end}_{PIPELINE}.csv"
    meta_path = OUT_DIR / f"armax_meta_{zone}_{start}_{end}_{PIPELINE}.csv"

    if csv_path.exists() and meta_path.exists() and not OVERWRITE:
        print(f"  [{zone} {start_year}-{end_year}] already exists - skipping")
        return

    paths = {
        **SHARED_PATHS,
        "combined": f"master data files/2015-2025/Combined_{zone}_Data_2015_2025.xlsx",
    }

    with contextlib.redirect_stdout(io.StringIO()):
        df_raw = load_data(
            paths,
            target_region=zone,
            zone_hydro=zone,
            use_interpolation=True,
            start_date=start,
            end_date=end,
            lag_commodity_hours=24,
            use_bilateral_exchange=USE_BILATERAL_EXCHANGE,
        )

    df_proc = preprocess_data_for_regression(
        df_raw,
        suppress_output=True,
        seasonal_interactions=SEASONAL_INTERACTIONS,
    )

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path, dtype=str)
        price_ds = df_proc[["Price_DS"]].reset_index().rename(
            columns={"Datetime": "timestamp", "Price_DS": "price_ds_floor"}
        )
        price_ds["timestamp"] = pd.to_datetime(price_ds["timestamp"])
        price_ds["Occurrence"] = price_ds.groupby("timestamp").cumcount()

        armax_input = baseline.copy()
        armax_input["_timestamp_key"] = pd.to_datetime(armax_input["timestamp"])
        armax_input["Occurrence"] = armax_input.groupby("_timestamp_key").cumcount()
        armax_input = armax_input.merge(
            price_ds,
            left_on=["_timestamp_key", "Occurrence"],
            right_on=["timestamp", "Occurrence"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_floor_key"),
        )
        if armax_input["price_ds_floor"].isna().any():
            raise ValueError(f"Missing capped price_ds values when matching {baseline_path.name}")

        armax_input["price_ds"] = armax_input["price_ds_floor"].map(lambda x: f"{x:.17g}")
        armax_input = armax_input.drop(
            columns=["_timestamp_key", "Occurrence", "timestamp_floor_key", "price_ds_floor"]
        )
    else:
        armax_input = build_full_input_from_processed(df_proc, zone, start, end)

    armax_input.to_csv(csv_path, index=False)

    p, d, q = ARMA_ORDER
    pd.DataFrame(
        {"arma_p": [p], "arma_d": [d], "arma_q": [q], "extra_ar_lags": [""]}
    ).to_csv(meta_path, index=False)

    print(f"  [{zone} {start_year}-{end_year}] {len(armax_input)} obs -> {csv_path.name}")


def build_full_input_from_processed(
    df_proc: pd.DataFrame,
    zone: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fallback path for zones/windows without an existing baseline rolling CSV."""
    standard_exog = [v for v in EXOG_VARS if v in df_proc.columns]

    armax_input = pd.concat([df_proc["Price_DS"], df_proc[standard_exog]], axis=1)
    armax_input.index.name = "timestamp"

    rename_dict = {k: v for k, v in STATA_RENAME.items() if k in armax_input.columns}
    armax_input = armax_input.rename(columns=rename_dict)
    armax_input = attach_bilateral_exchange(armax_input, zone, start, end)
    armax_input = armax_input.rename(
        columns={col: col.lower() for col in armax_input.columns if col.startswith("NetExch_")}
    )
    return armax_input.reset_index()


if __name__ == "__main__":
    total = len(ZONES) * len(WINDOWS)
    done = 0
    errors = []

    print(f"Capped-price rolling export: {len(ZONES)} zones x {len(WINDOWS)} windows = {total} exports")
    print(f"Window size: {WINDOW_YEARS} year  |  Step: {STEP_YEARS} year")
    print(f"Windows: {WINDOWS[0][0]}-{WINDOWS[0][1]} ... {WINDOWS[-1][0]}-{WINDOWS[-1][1]}")
    print(f"Seasonal interactions: {SEASONAL_INTERACTIONS}")
    print(f"Fixed ARMA order: {ARMA_ORDER}")
    print(f"Controls: {[k for k, v in CONTROLS.items() if v]}  |  Exchange: verified bilateral")
    print(f"File suffix: _{PIPELINE}")
    print(f"Overwrite: {OVERWRITE}")
    print(f"Output directory: {OUT_DIR}\n")

    for zone in ZONES:
        print(f"\n{'=' * 60}")
        print(f"  Zone: {zone}")
        print(f"{'=' * 60}")
        for start_year, end_year in WINDOWS:
            try:
                export_window(zone, start_year, end_year)
                done += 1
            except Exception as exc:
                msg = f"[{zone} {start_year}-{end_year}] ERROR: {exc}"
                print(f"  {msg}")
                errors.append(msg)

    print(f"\n{'=' * 60}")
    print(f"Done: {done}/{total} windows exported successfully")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  {err}")
    print(f"{'=' * 60}")
