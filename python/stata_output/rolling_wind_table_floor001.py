"""
rolling_wind_table_floor001.py

Builds one wide table for the capped-price rolling joint t(5) results.
Rows are years (2015-2025). For each zone, the table includes the wind_log
coefficient from the mean equation and the het_wind_log coefficient from the
variance equation, plus SE, p-value, stars, and convergence code.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "output from stata" / "rolling_window_results" / "joint_t"
OUT_PATH = RESULTS_DIR / "rolling_wind_table_joint_t5_floor001.csv"
ZONES = ["SE1", "SE2", "SE3", "SE4"]
YEARS = list(range(2015, 2026))


def load_zone(zone: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"rolling_garch_joint_t_{zone}_1yr_log_floor001_tdf5.csv"
    df = pd.read_csv(path)
    df["stars"] = df["stars"].fillna("")
    return df


def extract_series(df: pd.DataFrame, type_val: str, variable: str, prefix: str) -> pd.DataFrame:
    sub = df[(df["type"] == type_val) & (df["variable"] == variable)].copy()
    sub = sub[["start_year", "value", "se", "pval", "stars", "converged"]]
    return sub.rename(
        columns={
            "start_year": "year",
            "value": f"{prefix}_coef",
            "se": f"{prefix}_se",
            "pval": f"{prefix}_pval",
            "stars": f"{prefix}_stars",
            "converged": f"{prefix}_converged",
        }
    )


def build_table() -> pd.DataFrame:
    table = pd.DataFrame({"year": YEARS})

    for zone in ZONES:
        df = load_zone(zone)
        mean_part = extract_series(df, "coef", "wind_log", f"{zone.lower()}_mean")
        var_part = extract_series(df, "var_coef", "het_wind_log", f"{zone.lower()}_var")
        zone_part = mean_part.merge(var_part, on="year", how="outer", validate="one_to_one")
        table = table.merge(zone_part, on="year", how="left", validate="one_to_one")

    return table.sort_values("year").reset_index(drop=True)


def main() -> None:
    table = build_table()
    table.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(table)} rows -> {OUT_PATH}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
