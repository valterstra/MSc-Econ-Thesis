"""
Market size comparison: Day-Ahead vs Intraday auctions, Sweden 2025
Energy traded = (Total Volume Buy + Total Volume Sell) / 2 per zone
(Buy and Sell sides differ because submitted bids exceed cleared volume;
averaging gives the best proxy for matched/cleared volume.)
"""

import pandas as pd
import os

DATA_DIR = (
    r"C:\Users\patri\OneDrive - Handelshögskolan i Stockholm"
    r"\Master Thesis Economics 2025_2026 - General"
    r"\07_Code and Data\market_size"
)

FILES = {
    "Day-Ahead":   "AuctionVolume_2025_DayAhead_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_1":  "AuctionVolume_2025_SIDC_IntradayAuction1_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_2":  "AuctionVolume_2025_SIDC_IntradayAuction2_SE4,SE2,SE1,SE3_Yearly.csv",
    "Intraday_3":  "AuctionVolume_2025_SIDC_IntradayAuction3_SE4,SE2,SE1,SE3_Yearly.csv",
}

ZONES = ["SE1", "SE2", "SE3", "SE4"]

# ── Load data ────────────────────────────────────────────────────────────────
def load_traded(filename):
    """Return dict {zone: traded_TWh} for one market file."""
    df = pd.read_csv(os.path.join(DATA_DIR, filename), sep=";")
    row = df.iloc[0]
    result = {}
    for z in ZONES:
        buy  = row[f"{z} Total Volume Buy (MWh)"]
        sell = row[f"{z} Total Volume Sell (MWh)"]
        result[z] = (buy + sell) / 2 / 1e6   # MWh -> TWh
    return result

data = {name: load_traded(fname) for name, fname in FILES.items()}

# ── Build summary table ──────────────────────────────────────────────────────
markets = list(data.keys())
cols = ZONES + ["Sweden"]

rows = {}
for mkt, zone_vals in data.items():
    row = {z: zone_vals[z] for z in ZONES}
    row["Sweden"] = sum(zone_vals.values())
    rows[mkt] = row

rows["Intraday_Total"] = {
    c: rows["Intraday_1"][c] + rows["Intraday_2"][c] + rows["Intraday_3"][c]
    for c in cols
}
rows["Total"] = {
    c: rows["Day-Ahead"][c] + rows["Intraday_Total"][c]
    for c in cols
}

# ── Print results ────────────────────────────────────────────────────────────
W = 14

def fmt(v): return f"{v:>{W}.2f}"

header = f"{'Market':<22}" + "".join(f"{c:>{W}}" for c in cols)
sep    = "-" * len(header)

print("\n" + "=" * len(header))
print("  ENERGY TRADED IN SWEDEN 2025  [TWh]")
print("=" * len(header))
print(header)
print(sep)

labels = {
    "Day-Ahead":     "Day-Ahead",
    "Intraday_1":    "Intraday Auction 1",
    "Intraday_2":    "Intraday Auction 2",
    "Intraday_3":    "Intraday Auction 3",
    "Intraday_Total":"Intraday Total",
    "Total":         "TOTAL",
}

for key in ["Day-Ahead", "Intraday_1", "Intraday_2", "Intraday_3", "Intraday_Total", "Total"]:
    prefix = "  "
    if key in ("Intraday_Total", "Total"):
        prefix = "> "
    print(f"{prefix}{labels[key]:<20}" + "".join(fmt(rows[key][c]) for c in cols))

# ── Percentages ──────────────────────────────────────────────────────────────
print("\n" + "=" * len(header))
print("  SHARE OF TOTAL ENERGY TRADED  [%]")
print("=" * len(header))
print(header)
print(sep)

for key in ["Day-Ahead", "Intraday_1", "Intraday_2", "Intraday_3", "Intraday_Total"]:
    prefix = "  "
    if key == "Intraday_Total":
        prefix = "> "
    pct_row = {c: rows[key][c] / rows["Total"][c] * 100 for c in cols}
    print(f"{prefix}{labels[key]:<20}" + "".join(f"{pct_row[c]:>{W}.2f}" for c in cols))

print(sep)
# Sanity check: DA + ID_Total = 100%
check = {c: (rows["Day-Ahead"][c] + rows["Intraday_Total"][c]) / rows["Total"][c] * 100 for c in cols}
print(f"  {'DA + ID (check)':<20}" + "".join(f"{check[c]:>{W}.2f}" for c in cols))

print()
print("Note: Traded volume = (Total Volume Buy + Total Volume Sell) / 2 per zone.")
print("      Buy != Sell reflects submitted-bid asymmetry; the average approximates")
print("      the matched/cleared volume on each auction.")
