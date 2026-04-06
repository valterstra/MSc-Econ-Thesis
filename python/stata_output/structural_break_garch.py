"""
structural_break_garch.py

Structural break analysis on GARCH rolling-window estimates from Stata.

Reads per-zone long-format CSVs produced by stata/rolling_window_garch.do and
tests for structural breaks in key GARCH parameters (γ_wind, β_wind,
persistence) over the annual rolling windows 2015–2025.

Steps
-----
1. Load and reshape the long-format Stata output into a wide annual panel.
2. Bai-Perron style change-point detection (via `ruptures`) on the γ_wind
   series (het_wind_log) — the primary coefficient of interest.
3. Wald z-test for a level shift at known event years (meta-regression with
   regime dummy, WLS weighted by 1/se²).
4. CUSUM of the standardised annual coefficient estimates.
5. Multi-panel plots: coefficient evolution + CUSUM + persistence.
6. Save results to output from stata/structural_break/.

Usage
-----
  python structural_break_garch.py
  python structural_break_garch.py --zones SE1 SE2 SE3 SE4
  python structural_break_garch.py --zone SE4 --pipeline log_netexch
         --het hetwind --breaks 2020 2022

Output
------
  output from stata/structural_break/
      sb_coef_evolution_{zone}_{tag}.png
      sb_cusum_{zone}_{tag}.png
      sb_results_{zone}_{tag}.csv
      sb_summary_{zone}_{tag}.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Optional – Bai-Perron style change-point detection
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    warnings.warn(
        "ruptures not installed — change-point detection disabled. "
        "Install with: pip install ruptures",
        ImportWarning,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "output from stata" / "rolling_window_results"
OUTPUT_DIR = PROJECT_ROOT / "output from stata" / "structural_break"

# Known economic event years to test for regime shifts
DEFAULT_BREAK_YEARS = [2020, 2022]   # COVID onset, Ukraine war

# Parameters we track over time
MEAN_PARAMS   = ["wind_log", "consump_log_ds", "net_exchange", "ar_L1", "ma_L1", "_cons"]
VAR_PARAMS    = ["het_wind_log", "arch1", "garch1", "omega", "persistence"]
FOCUS_PARAMS  = ["het_wind_log", "wind_log", "persistence"]   # main focus for break tests


# ===========================================================================
# Data loading and reshaping
# ===========================================================================

def load_garch_csv(zone: str, pipeline: str, het: str) -> pd.DataFrame:
    """Load the Stata rolling-window GARCH CSV for one zone/pipeline/het."""
    if pipeline:
        fname = f"rolling_garch_{zone}_1yr_{pipeline}_{het}.csv"
    else:
        fname = f"rolling_garch_{zone}_1yr_{het}.csv"
    fpath = INPUT_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Expected file not found: {fpath}")
    df = pd.read_csv(fpath)
    return df


def reshape_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the long-format Stata output to a wide annual panel.

    Returns a DataFrame indexed by `year` with columns:
        {variable}_value, {variable}_se, {variable}_pval, {variable}_converged
    for all coef/var_coef/fit rows.
    """
    rows = []
    for year, grp in df.groupby("start_year"):
        row = {"year": int(year)}
        for _, r in grp.iterrows():
            var = str(r["variable"]).strip()
            row[f"{var}_value"] = r["value"] if pd.notna(r["value"]) else np.nan
            if pd.notna(r.get("se")):
                row[f"{var}_se"] = r["se"]
            if pd.notna(r.get("pval")):
                row[f"{var}_pval"] = r["pval"]
            row[f"{var}_converged"] = int(r["converged"]) if pd.notna(r.get("converged")) else np.nan
        rows.append(row)
    wide = pd.DataFrame(rows).set_index("year").sort_index()
    return wide


# ===========================================================================
# Change-point detection
# ===========================================================================

def detect_changepoints(series: np.ndarray, years: list[int],
                        max_breaks: int = 4) -> dict:
    """
    Apply Bai-Perron style change-point detection to a 1-D signal.

    Uses PELT (optimal) and BinSeg + BIC model selection from `ruptures`.
    Returns a dict with keys:
        optimal_n_breaks, break_years, bic_scores, method
    """
    result = {
        "optimal_n_breaks": 0,
        "break_years": [],
        "bic_scores": [],
        "method": "none",
    }

    n = len(series)
    if n < 4:
        return result

    signal = series.reshape(-1, 1)

    if not HAS_RUPTURES:
        return result

    # BIC selection: test 0..max_breaks breakpoints
    bic_rows = []
    algo_dynp = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(signal)
    for k in range(0, min(max_breaks, n // 2) + 1):
        try:
            if k == 0:
                cost = np.sum((series - series.mean()) ** 2)
            else:
                bkps = algo_dynp.predict(n_bkps=k)
                bkps_full = [0] + [b for b in bkps if b < n] + [n]
                cost = 0.0
                for i in range(len(bkps_full) - 1):
                    seg = series[bkps_full[i]:bkps_full[i + 1]]
                    cost += float(np.sum((seg - seg.mean()) ** 2))
            n_params = k + 1
            bic = n * np.log(cost / n + 1e-12) + n_params * np.log(n)
            bic_rows.append({"n_breaks": k, "bic": bic, "cost": cost})
        except Exception:
            pass

    if not bic_rows:
        return result

    bic_df = pd.DataFrame(bic_rows)
    optimal_k = int(bic_df.loc[bic_df["bic"].idxmin(), "n_breaks"])
    result["bic_scores"] = bic_rows
    result["optimal_n_breaks"] = optimal_k

    if optimal_k > 0:
        try:
            bkps = algo_dynp.predict(n_bkps=optimal_k)
            break_indices = [b for b in bkps if b < n]
            break_years = [years[i] for i in break_indices]
            result["break_years"] = break_years
            result["method"] = "DynP+BIC"
        except Exception:
            pass

    return result


# ===========================================================================
# Wald test (meta-regression regime shift)
# ===========================================================================

def wald_regime_test(values: np.ndarray, ses: np.ndarray, years: np.ndarray,
                     break_year: int) -> dict:
    """
    Test for a level shift in a coefficient series at `break_year`.

    Fits a WLS meta-regression:
        β̂_t = α + δ·D_t + ε_t,   w_t = 1/se²_t
    where D_t = 1 if year >= break_year.

    Returns dict with keys: break_year, n_pre, n_post, mean_pre, mean_post,
        delta, se_delta, z_stat, p_value, significant_5pct, significant_1pct
    """
    mask = np.isfinite(values) & np.isfinite(ses) & (ses > 0)
    y = values[mask]
    s = ses[mask]
    yr = years[mask]

    D = (yr >= break_year).astype(float)
    pre_mask = yr < break_year
    post_mask = yr >= break_year
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())

    if n_pre < 1 or n_post < 1:
        return {"break_year": break_year, "n_pre": n_pre, "n_post": n_post,
                "skipped": True}

    w = 1.0 / s**2
    X = np.column_stack([np.ones(len(y)), D])
    W = np.diag(w)

    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_hat = np.linalg.solve(XtWX, XtWy)
        cov = np.linalg.inv(XtWX)
        delta = beta_hat[1]
        se_delta = np.sqrt(cov[1, 1])
        z_stat = delta / se_delta if se_delta > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan

        mean_pre = float(np.average(y[pre_mask[mask]], weights=w[pre_mask[mask]]))
        mean_post = float(np.average(y[post_mask[mask]], weights=w[post_mask[mask]]))

        return {
            "break_year": break_year,
            "n_pre": n_pre,
            "n_post": n_post,
            "mean_pre": mean_pre,
            "mean_post": mean_post,
            "delta": delta,
            "se_delta": se_delta,
            "z_stat": z_stat,
            "p_value": p_value,
            "significant_5pct": bool(p_value < 0.05) if np.isfinite(p_value) else False,
            "significant_1pct": bool(p_value < 0.01) if np.isfinite(p_value) else False,
            "skipped": False,
        }
    except np.linalg.LinAlgError:
        return {"break_year": break_year, "n_pre": n_pre, "n_post": n_post,
                "skipped": True, "error": "LinAlgError"}


# ===========================================================================
# CUSUM on coefficient series
# ===========================================================================

def cusum_of_coefs(values: np.ndarray, ses: np.ndarray) -> dict:
    """
    CUSUM of standardised annual coefficient deviations.

    Standardises each estimate by its SE, then computes cumulative sum.
    Critical bounds at 5% level: ±0.948*sqrt(n)*(t/n).

    Returns dict with keys: cusum, upper, lower, violations, first_violation_idx
    """
    mask = np.isfinite(values) & np.isfinite(ses) & (ses > 0)
    z = np.where(mask, (values - np.nanmean(values)) / ses, np.nan)
    z_clean = z[mask]
    n = len(z_clean)
    if n == 0:
        return {"cusum": np.array([]), "upper": np.array([]),
                "lower": np.array([]), "violations": 0,
                "first_violation_idx": None}

    # Normalised CUSUM
    sigma = np.std(z_clean, ddof=1) if np.std(z_clean, ddof=1) > 0 else 1.0
    cusum = np.cumsum(z_clean) / (sigma * np.sqrt(n))

    t_vals = np.arange(1, n + 1)
    upper = 0.948 * np.sqrt(n) * t_vals / n
    lower = -upper

    violations = (cusum > upper) | (cusum < lower)
    first_viol = int(np.where(violations)[0][0]) if violations.any() else None

    return {
        "cusum": cusum,
        "upper": upper,
        "lower": lower,
        "violations": int(violations.sum()),
        "first_violation_idx": first_viol,
        "mask": mask,
    }


# ===========================================================================
# Plotting
# ===========================================================================

PARAM_LABELS = {
    "het_wind_log": r"$\gamma_{\mathrm{wind}}$ (variance eq.)",
    "wind_log":     r"$\beta_{\mathrm{wind}}$ (mean eq.)",
    "persistence":  r"$\alpha + \beta$ (GARCH persistence)",
    "arch1":        r"$\alpha_1$ (ARCH)",
    "garch1":       r"$\beta_1$ (GARCH)",
    "omega":        r"$\omega$ (variance intercept)",
    "ar_L1":        r"AR(1)",
    "ma_L1":        r"MA(1)",
    "consump_log_ds": "Consumption (log DS)",
    "net_exchange": "Net Exchange",
    "_cons":        "Constant",
}

EVENT_COLORS = {
    2020: ("#e07b00", "COVID-19 onset"),
    2022: ("#c0392b", "Ukraine war"),
    2021: ("#8e44ad", "Energy crisis"),
}


def _add_event_lines(ax, break_years, detected_years=None):
    added = set()
    for yr in break_years:
        color, label = EVENT_COLORS.get(yr, ("#555555", f"{yr}"))
        kw = dict(color=color, linestyle="--", linewidth=1.4, alpha=0.85)
        if label not in added:
            ax.axvline(x=yr, label=label, **kw)
            added.add(label)
        else:
            ax.axvline(x=yr, **kw)
    if detected_years:
        for i, yr in enumerate(detected_years):
            kw = dict(color="#1a7abf", linestyle=":", linewidth=1.8, alpha=0.9)
            if i == 0:
                ax.axvline(x=yr, label="Detected break", **kw)
            else:
                ax.axvline(x=yr, **kw)


def plot_coef_evolution(wide: pd.DataFrame, zone: str, tag: str,
                        break_years: list[int], detected_breaks: dict,
                        out_dir: Path) -> Path:
    """
    Three-panel figure:
      Top:    γ_wind (variance eq.) with CI and break markers
      Middle: β_wind (mean eq.) with CI
      Bottom: GARCH persistence (arch1 + garch1) — no SE for persistence
    """
    years = wide.index.values.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"GARCH-X Parameter Evolution — {zone}  ({tag})",
        fontsize=13, fontweight="bold", y=0.995,
    )

    params_panels = [
        ("het_wind_log", "coef"),
        ("wind_log",     "coef"),
        ("persistence",  "fit"),   # no se column
    ]

    for ax, (param, _) in zip(axes, params_panels):
        val_col = f"{param}_value"
        se_col  = f"{param}_se"

        if val_col not in wide.columns:
            ax.text(0.5, 0.5, f"{param} not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        vals = wide[val_col].values.astype(float)
        has_se = se_col in wide.columns
        ses = wide[se_col].values.astype(float) if has_se else None

        ax.plot(years, vals, "o-", color="#2c5f8a", linewidth=2,
                markersize=5, zorder=3)
        ax.axhline(y=0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)

        if has_se and ses is not None:
            ci_lo = vals - 1.96 * ses
            ci_hi = vals + 1.96 * ses
            ax.fill_between(years, ci_lo, ci_hi, color="#2c5f8a", alpha=0.18,
                            label="95% CI")

        # Mark non-converged windows
        conv_col = f"{param}_converged"
        if conv_col in wide.columns:
            nc_mask = wide[conv_col].values == 2
            if nc_mask.any():
                ax.scatter(years[nc_mask], vals[nc_mask], marker="x",
                           color="darkorange", s=80, zorder=5,
                           label="Non-converged")

        _add_event_lines(ax, break_years,
                         detected_years=detected_breaks.get(param, []))

        ylabel = PARAM_LABELS.get(param, param)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Year", fontsize=10)
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    fpath = out_dir / f"sb_coef_evolution_{zone}_{tag}.png"
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


def plot_cusum(wide: pd.DataFrame, zone: str, tag: str,
               break_years: list[int], out_dir: Path) -> Path:
    """
    Two-panel CUSUM figure for het_wind_log and wind_log.
    """
    years = wide.index.values.astype(int)
    params = [("het_wind_log", "variance eq."), ("wind_log", "mean eq.")]

    fig, axes = plt.subplots(len(params), 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"CUSUM of Standardised Wind Coefficient Estimates — {zone}  ({tag})",
        fontsize=12, fontweight="bold",
    )

    for ax, (param, desc) in zip(axes, params):
        val_col = f"{param}_value"
        se_col  = f"{param}_se"
        if val_col not in wide.columns or se_col not in wide.columns:
            ax.text(0.5, 0.5, f"{param} not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        vals = wide[val_col].values.astype(float)
        ses  = wide[se_col].values.astype(float)
        res  = cusum_of_coefs(vals, ses)

        if len(res["cusum"]) == 0:
            continue

        cusum_years = years[res["mask"]]
        ax.plot(cusum_years, res["cusum"], "o-", color="#1a7abf", linewidth=2,
                markersize=5, label="CUSUM", zorder=3)
        ax.plot(cusum_years, res["upper"], "--", color="#c0392b", linewidth=1.4,
                label="5% bounds")
        ax.plot(cusum_years, res["lower"], "--", color="#c0392b", linewidth=1.4)
        ax.fill_between(cusum_years, res["lower"], res["upper"],
                        color="#c0392b", alpha=0.08)
        ax.axhline(y=0, color="black", linewidth=0.7, alpha=0.4)

        _add_event_lines(ax, break_years)

        n_viol = res["violations"]
        viol_str = f"  ({n_viol} violation{'s' if n_viol != 1 else ''})"
        ax.set_ylabel(f"CUSUM — {PARAM_LABELS.get(param, param)}\n{viol_str}",
                      fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Year", fontsize=10)
    plt.tight_layout()

    fpath = out_dir / f"sb_cusum_{zone}_{tag}.png"
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


def plot_full_panel(wide: pd.DataFrame, zone: str, tag: str,
                    break_years: list[int], detected_breaks: dict,
                    out_dir: Path) -> Path:
    """
    Extended 6-panel figure covering all key GARCH-X parameters.
    """
    years = wide.index.values.astype(int)
    params = [
        "het_wind_log", "wind_log", "arch1", "garch1", "persistence", "omega"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()
    fig.suptitle(
        f"GARCH-X Parameter Evolution (all) — {zone}  ({tag})",
        fontsize=13, fontweight="bold",
    )

    for ax, param in zip(axes, params):
        val_col = f"{param}_value"
        se_col  = f"{param}_se"

        if val_col not in wide.columns:
            ax.text(0.5, 0.5, f"{param} not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(PARAM_LABELS.get(param, param), fontsize=10)
            continue

        vals = wide[val_col].values.astype(float)
        has_se = se_col in wide.columns
        ses = wide[se_col].values.astype(float) if has_se else None

        ax.plot(years, vals, "o-", color="#2c5f8a", linewidth=2,
                markersize=5, zorder=3)
        ax.axhline(y=0, color="black", linewidth=0.7, alpha=0.4)

        if has_se and ses is not None:
            ax.fill_between(years, vals - 1.96 * ses, vals + 1.96 * ses,
                            color="#2c5f8a", alpha=0.18)

        _add_event_lines(ax, break_years,
                         detected_years=detected_breaks.get(param, []))

        ax.set_title(PARAM_LABELS.get(param, param), fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fpath = out_dir / f"sb_full_panel_{zone}_{tag}.png"
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


# ===========================================================================
# Core analysis per zone
# ===========================================================================

def analyse_zone(zone: str, pipeline: str, het: str,
                 break_years: list[int], max_breaks: int,
                 out_dir: Path) -> None:
    """Run the full structural break pipeline for one zone."""
    print("\n" + "=" * 72)
    tag = f"{pipeline}_{het}" if pipeline else het
    print(f"ZONE: {zone}   tag: {tag}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load & reshape
    # ------------------------------------------------------------------
    try:
        raw = load_garch_csv(zone, pipeline, het)
    except FileNotFoundError as e:
        print(f"  SKIPPED — {e}")
        return

    wide = reshape_to_wide(raw)
    years = wide.index.values.astype(int)
    print(f"  Loaded {len(raw)} rows -> {len(wide)} annual windows  "
          f"({years[0]}-{years[-1]})")

    converged_col = "N_converged"
    if "N_converged" in wide.columns:
        n_nc = (wide["N_converged"] == 2).sum()
        if n_nc:
            print(f"  Warning: {n_nc} window(s) flagged as non-converged (converged=2)")

    # ------------------------------------------------------------------
    # 2. Change-point detection (Bai-Perron via ruptures)
    # ------------------------------------------------------------------
    print("\n--- Change-point detection ---")
    detected_breaks: dict[str, list[int]] = {}
    cp_results: list[dict] = []

    for param in FOCUS_PARAMS:
        val_col = f"{param}_value"
        if val_col not in wide.columns:
            continue
        vals = wide[val_col].values.astype(float)
        ok = np.isfinite(vals)
        if ok.sum() < 4:
            print(f"  {param}: too few valid observations — skipped")
            continue

        cp = detect_changepoints(vals[ok], list(years[ok]), max_breaks=max_breaks)
        detected_breaks[param] = cp["break_years"]

        print(f"  {param}: optimal_breaks={cp['optimal_n_breaks']}, "
              f"break_years={cp['break_years']}, "
              f"method={cp['method']}")
        for row in cp.get("bic_scores", []):
            cp_results.append({
                "zone": zone, "param": param,
                "n_breaks": row["n_breaks"], "bic": row["bic"],
            })

    if not HAS_RUPTURES:
        print("  (ruptures not available — change-point detection skipped)")

    # ------------------------------------------------------------------
    # 3. Wald tests at known event years
    # ------------------------------------------------------------------
    print("\n--- Wald tests at known event years ---")
    wald_rows = []
    for param in FOCUS_PARAMS:
        val_col = f"{param}_value"
        se_col  = f"{param}_se"
        if val_col not in wide.columns:
            continue
        if se_col not in wide.columns:
            print(f"  {param}: no SE column — Wald test skipped")
            continue
        vals = wide[val_col].values.astype(float)
        ses  = wide[se_col].values.astype(float)

        for by in break_years:
            if by not in range(years[0] + 1, years[-1] + 1):
                print(f"  {param} @ {by}: outside window range — skipped")
                continue
            res = wald_regime_test(vals, ses, years.astype(float), by)
            res["zone"]  = zone
            res["param"] = param
            wald_rows.append(res)

            if res.get("skipped"):
                print(f"  {param} @ {by}: skipped — "
                      f"n_pre={res['n_pre']}, n_post={res['n_post']}")
            else:
                sig = ("***" if res["significant_1pct"] else
                       "**"  if res["significant_5pct"] else "")
                print(f"  {param} @ {by}:  delta={res['delta']:+.4f}  "
                      f"z={res['z_stat']:+.2f}  p={res['p_value']:.4f}  {sig}")

    # ------------------------------------------------------------------
    # 4. CUSUM statistics (console only — plot below)
    # ------------------------------------------------------------------
    print("\n--- CUSUM summary ---")
    cusum_rows = []
    for param in FOCUS_PARAMS:
        val_col = f"{param}_value"
        se_col  = f"{param}_se"
        if val_col not in wide.columns or se_col not in wide.columns:
            continue
        vals = wide[val_col].values.astype(float)
        ses  = wide[se_col].values.astype(float)
        res  = cusum_of_coefs(vals, ses)
        first_yr = (years[res["mask"]][res["first_violation_idx"]]
                    if res["first_violation_idx"] is not None else None)
        print(f"  {param}: violations={res['violations']}, "
              f"first_at_year={first_yr}")
        cusum_rows.append({
            "zone": zone, "param": param,
            "n_violations": res["violations"],
            "first_violation_year": first_yr,
        })

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_coef_evolution(wide, zone, tag, break_years, detected_breaks, out_dir)
    plot_cusum(wide, zone, tag, break_years, out_dir)
    plot_full_panel(wide, zone, tag, break_years, detected_breaks, out_dir)

    # ------------------------------------------------------------------
    # 6. Save CSV results
    # ------------------------------------------------------------------
    print("\n--- Saving results ---")

    if wald_rows:
        wald_df = pd.DataFrame(wald_rows)
        wald_path = out_dir / f"sb_wald_tests_{zone}_{tag}.csv"
        wald_df.to_csv(wald_path, index=False)
        print(f"  Saved: {wald_path}")

    if cp_results:
        cp_df = pd.DataFrame(cp_results)
        cp_path = out_dir / f"sb_changepoint_bic_{zone}_{tag}.csv"
        cp_df.to_csv(cp_path, index=False)
        print(f"  Saved: {cp_path}")

    # Wide panel (all parameters over years)
    wide_path = out_dir / f"sb_wide_panel_{zone}_{tag}.csv"
    wide.to_csv(wide_path)
    print(f"  Saved: {wide_path}")

    # ------------------------------------------------------------------
    # 7. Text summary
    # ------------------------------------------------------------------
    sum_path = out_dir / f"sb_summary_{zone}_{tag}.txt"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"STRUCTURAL BREAK ANALYSIS — {zone}  ({tag})\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Annual windows: {years[0]}–{years[-1]}  (N={len(wide)})\n\n")

        f.write("-" * 72 + "\n")
        f.write("CHANGE-POINT DETECTION (Bai-Perron / DynP+BIC)\n")
        f.write("-" * 72 + "\n")
        if HAS_RUPTURES:
            for param in FOCUS_PARAMS:
                by = detected_breaks.get(param, [])
                f.write(f"  {param}: {len(by)} break(s) -> {by}\n")
        else:
            f.write("  ruptures not installed - skipped\n")

        f.write("\n" + "-" * 72 + "\n")
        f.write("WALD TESTS (WLS meta-regression with regime dummy)\n")
        f.write("-" * 72 + "\n")
        for r in wald_rows:
            if r.get("skipped"):
                f.write(f"  {r['param']} @ {r['break_year']}: SKIPPED\n")
                continue
            sig = ("***" if r["significant_1pct"] else
                   "**"  if r["significant_5pct"] else "")
            f.write(
                f"  {r['param']} @ {r['break_year']}:  "
                f"pre={r['mean_pre']:+.4f}  post={r['mean_post']:+.4f}  "
                f"Δ={r['delta']:+.4f}  z={r['z_stat']:+.2f}  "
                f"p={r['p_value']:.4f}  {sig}\n"
            )

        f.write("\n" + "-" * 72 + "\n")
        f.write("CUSUM SUMMARY\n")
        f.write("-" * 72 + "\n")
        for r in cusum_rows:
            f.write(f"  {r['param']}: {r['n_violations']} violation(s)  "
                    f"first={r['first_violation_year']}\n")

    print(f"  Saved: {sum_path}")
    print(f"\nZone {zone} done.")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Structural break analysis on Stata GARCH rolling results."
    )
    p.add_argument("--zones", nargs="+", default=["SE1", "SE2", "SE3", "SE4"],
                   help="Price zones to analyse (default: all four)")
    p.add_argument("--zone", help="Single zone (overrides --zones)")
    p.add_argument("--pipeline", default="log_netexch",
                   help="Pipeline suffix in CSV filenames (default: log_netexch; "
                        "use '' for no pipeline)")
    p.add_argument("--het", default="hetwind",
                   choices=["hetwind", "hetfull"],
                   help="Het specification (default: hetwind)")
    p.add_argument("--breaks", nargs="*", type=int,
                   default=DEFAULT_BREAK_YEARS,
                   help="Known event years to test (default: 2020 2022)")
    p.add_argument("--max-breaks", type=int, default=3,
                   help="Max breaks for BIC selection (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    zones = [args.zone] if args.zone else args.zones

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = args.pipeline.strip()
    het      = args.het
    break_years = sorted(set(args.breaks)) if args.breaks else []

    print("Structural Break Analysis — GARCH Rolling Windows")
    print(f"  Zones:      {zones}")
    print(f"  Pipeline:   {pipeline or '(none)'}")
    print(f"  Het spec:   {het}")
    print(f"  Break years tested: {break_years}")
    print(f"  Max breaks (BIC):   {args.max_breaks}")
    print(f"  Output dir: {OUTPUT_DIR}")

    for zone in zones:
        analyse_zone(
            zone=zone,
            pipeline=pipeline,
            het=het,
            break_years=break_years,
            max_breaks=args.max_breaks,
            out_dir=OUTPUT_DIR,
        )

    print("\n" + "=" * 72)
    print("All done.")


if __name__ == "__main__":
    main()
