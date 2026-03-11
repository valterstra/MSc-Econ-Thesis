"""
Market size comparison: Day-Ahead vs Intraday auctions, Sweden 2025
Buy and Sell sides reported separately (apples-to-apples).
"""

import pandas as pd
import os

DATA_DIR = (
    r"C:\Users\patri\OneDrive - Handelshögskolan i Stockholm"
    r"\Master Thesis Economics 2025_2026 - General"
    r"\07_Code and Data\market_size"
)

FILES = {
    "Day-Ahead":  "AuctionVolume_2025_DayAhead_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_1": "AuctionVolume_2025_SIDC_IntradayAuction1_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_2": "AuctionVolume_2025_SIDC_IntradayAuction2_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_3": "AuctionVolume_2025_SIDC_IntradayAuction3_SE4,SE2,SE1,SE3_Yearly.csv",
}

ZONES = ["SE1", "SE2", "SE3", "SE4"]

# ── Load data ────────────────────────────────────────────────────────────────
def load_sides(filename):
    """Return {zone: {'buy': TWh, 'sell': TWh}} for one market file."""
    df = pd.read_csv(os.path.join(DATA_DIR, filename), sep=";")
    row = df.iloc[0]
    result = {}
    for z in ZONES:
        result[z] = {
            "buy":  row[f"{z} Total Volume Buy (MWh)"]  / 1e6,
            "sell": row[f"{z} Total Volume Sell (MWh)"] / 1e6,
        }
    return result

data = {name: load_sides(fname) for name, fname in FILES.items()}

# ── Aggregate to Sweden totals ───────────────────────────────────────────────
def sweden(mkt_data):
    return {
        "buy":  sum(mkt_data[z]["buy"]  for z in ZONES),
        "sell": sum(mkt_data[z]["sell"] for z in ZONES),
    }

# Add Sweden column to each market
for mkt in data:
    data[mkt]["Sweden"] = sweden(data[mkt])

COLS = ZONES + ["Sweden"]

# Build intraday total and grand total
def add(a, b):
    return {c: {"buy": a[c]["buy"] + b[c]["buy"], "sell": a[c]["sell"] + b[c]["sell"]} for c in COLS}

data["Intraday_Total"] = add(add(data["Intraday_1"], data["Intraday_2"]), data["Intraday_3"])
data["Total"]          = add(data["Day-Ahead"], data["Intraday_Total"])

# ── Formatting helpers ───────────────────────────────────────────────────────
W = 13

def fmt(v):    return f"{v:>{W}.2f}"
def fmtp(v):   return f"{v:>{W}.2f}"

LABELS = {
    "Day-Ahead":     "Day-Ahead",
    "Intraday_1":    "Intraday Auction 1",
    "Intraday_2":    "Intraday Auction 2",
    "Intraday_3":    "Intraday Auction 3",
    "Intraday_Total":"Intraday Total",
    "Total":         "TOTAL",
}

col_hdr  = f"{'Market':<22}{'Side':>6}" + "".join(f"{c:>{W}}" for c in COLS)
sep      = "-" * len(col_hdr)
eq       = "=" * len(col_hdr)

PRINT_ORDER = ["Day-Ahead", "Intraday_1", "Intraday_2", "Intraday_3", "Intraday_Total", "Total"]

def print_table(title, value_fn, keys=PRINT_ORDER):
    print(f"\n{eq}")
    print(f"  {title}")
    print(eq)
    print(col_hdr)
    print(sep)
    for key in keys:
        prefix = "> " if key in ("Intraday_Total", "Total") else "  "
        label  = LABELS[key]
        for side in ("buy", "sell"):
            vals = "".join(fmt(value_fn(key, c, side)) for c in COLS)
            print(f"{prefix}{label:<20}{side:>6}{vals}")
        if key != keys[-1]:
            print(sep)

# ── Volume table [TWh] ───────────────────────────────────────────────────────
print_table(
    "ENERGY TRADED IN SWEDEN 2025  [TWh]",
    lambda mkt, col, side: data[mkt][col][side],
)

# ── Share of total [%] ───────────────────────────────────────────────────────
def pct(mkt, col, side):
    return data[mkt][col][side] / data["Total"][col][side] * 100

print_table(
    "SHARE OF TOTAL ENERGY TRADED  [%]",
    pct,
    keys=["Day-Ahead", "Intraday_1", "Intraday_2", "Intraday_3", "Intraday_Total"],
)

print(sep)
# Sanity check row
check_row = "".join(fmtp(pct("Day-Ahead", c, "buy") + pct("Intraday_Total", c, "buy")) for c in COLS)
print(f"  {'DA + ID (check)':<20}{'buy':>6}{check_row}")
check_row = "".join(fmtp(pct("Day-Ahead", c, "sell") + pct("Intraday_Total", c, "sell")) for c in COLS)
print(f"  {'':20}{'sell':>6}{check_row}")
print()
