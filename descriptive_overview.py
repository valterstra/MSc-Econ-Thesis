"""
descriptive_overview.py  –  Descriptive Overview: Swedish Electricity Market
═══════════════════════════════════════════════════════════════════════════════

Produces:
  1.  price_series_raw.png         –  SE1–SE4 daily avg spot prices (EUR/MWh), 2015–2025
  2.  price_series_log.png         –  SE1–SE4 ln(price + shift) daily avg, 2015–2025
                                      shift = |global_min_daily| + 1 (ensures all args > 0)
  3.  volatility_raw.png           –  SE1–SE4 daily price volatility (intra-day std), 2015–2025
  4.  volatility_log.png           –  SE1–SE4 daily volatility of ln(price + shift), 2015–2025
                                      shift = |global_min_hourly| + 1
  5.  monthly_price_raw.png        –  SE1–SE4 monthly average of daily VWAP (EUR/MWh)
  6.  monthly_price_log.png        –  SE1–SE4 monthly average of ln(VWAP + shift)
  7.  monthly_volatility_raw.png   –  SE1–SE4 monthly average of daily intra-day std
  8.  monthly_volatility_log.png   –  SE1–SE4 monthly average of daily std of ln(price + shift)
  9.  cross_border_flows_2025.tex  –  LaTeX table: net exchange by zone, 2025
  10. production_mix_2025.tex      –  LaTeX table: wind / other / consumption, 2025

Data limitations:
  · Only aggregate Net_Exchange (MW, hourly) is available per zone. Bilateral
    flows between specific trading pairs are not in the dataset.
  · Sign convention for Net_Exchange: positive = net import; negative = net export
    (consistent with the energy balance: Net_Import = Consumption − Production).
  · Total_Forecast covers all generation types combined; only Wind_Forecast is
    separately available. Nuclear / hydro / solar / thermal cannot be individually
    decomposed from this dataset.

Usage:
    python descriptive_overview.py   (from MSc-Econ-Thesis root)
"""

import gc
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  edit here only
# ══════════════════════════════════════════════════════════════════════════════
ZONES      = ['SE1', 'SE2', 'SE3', 'SE4']
YEAR_TABLE = 2025
OUTPUT_DIR = 'results/descriptive_overview'

ZONE_COLORS = {
    'SE1': '#4477AA',   # blue
    'SE2': '#228833',   # green
    'SE3': '#CCBB44',   # gold
    'SE4': '#EE6677',   # red
}

# Trading partners derived from bottleneck (BNECK) column names in source data
TRADING_PARTNERS = {
    'SE1': ['FI', 'NO4', 'SE2'],
    'SE2': ['NO3', 'NO4', 'SE1', 'SE3'],
    'SE3': ['DK1', 'FI', 'NO1', 'SE2', 'SE4'],
    'SE4': ['DK2', 'SE3'],
}

PATHS = {
    'prices': 'master data files/2015-2025/Spot_Prices_2015_2025.xlsx',
    'SE1':    'master data files/2015-2025/Combined_SE1_Data_2015_2025.xlsx',
    'SE2':    'master data files/2015-2025/Combined_SE2_Data_2015_2025.xlsx',
    'SE3':    'master data files/2015-2025/Combined_SE3_Data_2015_2025.xlsx',
    'SE4':    'master data files/2015-2025/Combined_SE4_Data_2015_2025.xlsx',
}
# ══════════════════════════════════════════════════════════════════════════════


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_fig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f"    Saved -> {path}")


def _save_tex(content, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f"    Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  1.  PRICE SERIES PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _load_hourly_prices():
    """Return hourly price DataFrame indexed by Timestamp, columns = ZONES."""
    df = pd.read_excel(PATHS['prices'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    df = df.rename(columns={f'SE{i}_Price (EUR)': f'SE{i}' for i in range(1, 5)})
    return df[ZONES]


def _load_hourly_consumption():
    """Return hourly Consumption_Forecast (MW) per zone, indexed by Timestamp."""
    frames = {}
    for zone in ZONES:
        z = pd.read_excel(PATHS[zone])
        z['Timestamp'] = pd.to_datetime(z['Timestamp'])
        z = z.set_index('Timestamp')
        frames[zone] = z['Consumption_Forecast']
    return pd.DataFrame(frames)


def _daily_vwap(price_df, cons_df):
    """
    Volume-weighted average price per day per zone.

    VWAP_d = Σ_h (price_h × consumption_h) / Σ_h consumption_h

    Consumption_Forecast (MW) serves as the hourly trading volume proxy
    (MW × 1 h = MWh of energy cleared at the spot price in that hour).
    """
    daily = pd.DataFrame(index=pd.date_range(
        price_df.index.min().date(), price_df.index.max().date(), freq='D'
    ))
    for zone in ZONES:
        p = price_df[zone]
        w = cons_df[zone].clip(lower=0)          # weight must be non-negative
        num = (p * w).resample('D').sum()
        den = w.resample('D').sum().replace(0, np.nan)
        daily[zone] = num / den
    return daily


def plot_price_series():
    print("\n[1/3]  Price series plots …")

    price_df = _load_hourly_prices()
    cons_df  = _load_hourly_consumption()

    # ── Volume-weighted daily average prices (VWAP) ───────────────────────────
    daily = _daily_vwap(price_df, cons_df)
    print(f"    VWAP computed: {daily.shape[0]} days, "
          f"global min = {daily.min().min():.2f} EUR/MWh")

    # ── Figure A: raw VWAP ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(daily.index, daily[zone],
                color=ZONE_COLORS[zone], lw=0.9, alpha=0.85, label=zone)
    ax.axhline(0, color='#555555', lw=0.7, ls='--', alpha=0.45,
               label='Price = 0')
    ax.set_title(
        'Swedish Electricity Spot Prices – SE1 to SE4\n'
        'Volume-Weighted Daily Average (Consumption-Weighted VWAP), 2015–2025',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price (EUR/MWh)', fontsize=10)
    ax.legend(fontsize=10, ncol=5, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'price_series_raw.png')

    # ── Log transformation: shift all VWAP values so minimum → 1 ────────────
    # shift = −min(VWAP_global) + 1  ⟹  ln(VWAP + shift) ≥ ln(1) = 0 always
    global_min_daily = daily[ZONES].min().min()
    shift_d = -global_min_daily + 1
    log_daily = np.log(daily[ZONES] + shift_d)
    print(f"    Daily VWAP shift = {shift_d:.4f}  "
          f"(global_min = {global_min_daily:.4f})")

    # ── Figure B: ln(VWAP + shift) ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(log_daily.index, log_daily[zone],
                color=ZONE_COLORS[zone], lw=0.9, alpha=0.85, label=zone)
    ax.set_title(
        f'Swedish Electricity Spot Prices – ln(VWAP + {shift_d:.2f}) (Daily), 2015–2025\n'
        f'Shift = −min(VWAP) + 1 = {shift_d:.4f} EUR/MWh  |  '
        f'min(VWAP) = {global_min_daily:.4f} EUR/MWh',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(f'ln(VWAP + {shift_d:.2f})  [EUR/MWh]', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'price_series_log.png')

    # ── Daily volatility: intra-day std of hourly prices ─────────────────────
    daily_std = price_df[ZONES].resample('D').std()

    # ── Figure C: raw daily volatility ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(daily_std.index, daily_std[zone],
                color=ZONE_COLORS[zone], lw=0.9, alpha=0.85, label=zone)
    ax.set_title(
        'Swedish Electricity Price Volatility – SE1 to SE4\n'
        'Daily Intra-Day Standard Deviation of Hourly Spot Prices, 2015–2025',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price Std Dev (EUR/MWh)', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'volatility_raw.png')

    # ── Log-transform hourly prices, then compute daily std ──────────────────
    # Same shift principle applied to raw hourly prices for consistency
    global_min_hourly = price_df[ZONES].min().min()
    shift_h = -global_min_hourly + 1
    log_price_df = np.log(price_df[ZONES] + shift_h)
    daily_log_std = log_price_df.resample('D').std()
    print(f"    Hourly price shift = {shift_h:.4f}  "
          f"(global_min = {global_min_hourly:.4f})")

    # ── Figure D: log daily volatility ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(daily_log_std.index, daily_log_std[zone],
                color=ZONE_COLORS[zone], lw=0.9, alpha=0.85, label=zone)
    ax.set_title(
        f'Swedish Electricity Price Volatility – SE1 to SE4\n'
        f'Daily Std Dev of ln(Price + {shift_h:.2f}), 2015–2025  |  '
        f'Shift = {shift_h:.4f} EUR/MWh',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(f'Std Dev of ln(Price + {shift_h:.2f})', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'volatility_log.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  Monthly averages  –  Figures E–H  (all zones per graph)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Figure E: monthly average raw VWAP (price level) ─────────────────────
    monthly_raw = daily[ZONES].resample('ME').mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(monthly_raw.index, monthly_raw[zone],
                color=ZONE_COLORS[zone], lw=1.5, marker='o', ms=3.5,
                alpha=0.9, label=zone)
    ax.axhline(0, color='#555555', lw=0.7, ls='--', alpha=0.45,
               label='Price = 0')
    ax.set_title(
        'Swedish Electricity Spot Prices – SE1 to SE4\n'
        'Monthly Average of Daily VWAP (EUR/MWh), 2015–2025',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price (EUR/MWh)', fontsize=10)
    ax.legend(fontsize=10, ncol=5, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'monthly_price_raw.png')

    # ── Figure F: monthly average log VWAP ───────────────────────────────────
    monthly_log = log_daily[ZONES].resample('ME').mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(monthly_log.index, monthly_log[zone],
                color=ZONE_COLORS[zone], lw=1.5, marker='o', ms=3.5,
                alpha=0.9, label=zone)
    ax.set_title(
        f'Swedish Electricity Spot Prices – ln(VWAP + {shift_d:.2f}) – SE1 to SE4\n'
        f'Monthly Average, 2015–2025  |  Shift = {shift_d:.4f} EUR/MWh',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(f'ln(VWAP + {shift_d:.2f})  [EUR/MWh]', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'monthly_price_log.png')

    # ── Figure G: monthly average raw volatility ──────────────────────────────
    monthly_vol_raw = daily_std[ZONES].resample('ME').mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(monthly_vol_raw.index, monthly_vol_raw[zone],
                color=ZONE_COLORS[zone], lw=1.5, marker='o', ms=3.5,
                alpha=0.9, label=zone)
    ax.set_title(
        'Swedish Electricity Price Volatility – SE1 to SE4\n'
        'Monthly Average of Daily Intra-Day Std Dev (EUR/MWh), 2015–2025',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Mean Daily Std Dev (EUR/MWh)', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'monthly_volatility_raw.png')

    # ── Figure H: monthly average log volatility ──────────────────────────────
    monthly_vol_log = daily_log_std[ZONES].resample('ME').mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in ZONES:
        ax.plot(monthly_vol_log.index, monthly_vol_log[zone],
                color=ZONE_COLORS[zone], lw=1.5, marker='o', ms=3.5,
                alpha=0.9, label=zone)
    ax.set_title(
        f'Swedish Electricity Price Volatility – SE1 to SE4\n'
        f'Monthly Average of Daily Std Dev of ln(Price + {shift_h:.2f}), 2015–2025',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(f'Mean Std Dev of ln(Price + {shift_h:.2f})', fontsize=10)
    ax.legend(fontsize=10, ncol=4, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    _save_fig(fig, 'monthly_volatility_log.png')


# ══════════════════════════════════════════════════════════════════════════════
#  2.  CROSS-BORDER FLOWS TABLE  (LaTeX)
# ══════════════════════════════════════════════════════════════════════════════

def build_flows_table():
    print("\n[2/3]  Cross-border flows table (2025) …")

    rows = []
    for zone in ZONES:
        df = pd.read_excel(PATHS[zone])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        ne = df.loc[df['Timestamp'].dt.year == YEAR_TABLE, 'Net_Exchange'].fillna(0)

        # Sign convention: positive = net import, negative = net export
        gross_import_gwh = ne[ne > 0].sum() / 1_000   # MWh → GWh
        gross_export_gwh = ne[ne < 0].abs().sum() / 1_000
        net_gwh          = ne.sum() / 1_000            # positive = net importer

        rows.append(dict(
            zone      = zone,
            partners  = ', '.join(TRADING_PARTNERS[zone]),
            exp_gwh   = gross_export_gwh,
            imp_gwh   = gross_import_gwh,
            net_gwh   = net_gwh,
        ))
        print(f"    {zone}: gross export = {gross_export_gwh:,.0f} GWh | "
              f"gross import = {gross_import_gwh:,.0f} GWh | net = {net_gwh:+,.0f} GWh")

    # Sweden aggregate — internal SE*↔SE* flows cancel in the net sum,
    # but appear in gross import and gross export totals for individual zones.
    tot_exp = sum(r['exp_gwh'] for r in rows)
    tot_imp = sum(r['imp_gwh'] for r in rows)
    tot_net = sum(r['net_gwh'] for r in rows)
    rows.append(dict(
        zone     = 'Sweden (all zones)',
        partners = 'see above',
        exp_gwh  = tot_exp,
        imp_gwh  = tot_imp,
        net_gwh  = tot_net,
    ))

    tex = _flows_latex(rows)
    _save_tex(tex, 'cross_border_flows_2025.tex')
    print("\n-- LaTeX output --------------------------------------------------------------")
    print(tex)


def _flows_latex(rows):
    lines = []
    lines.append(r'% Generated by descriptive_overview.py')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Cross-Border Electricity Flows by Swedish Bidding Zone, 2025}'
    )
    lines.append(r'\label{tab:cross_border_flows_2025}')
    lines.append(r'\begin{tabular}{llrrr}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Zone} & \textbf{Connected Partners} '
        r'& \textbf{Gross Export} & \textbf{Gross Import} & \textbf{Net Balance} \\'
    )
    lines.append(
        r'& & \textbf{(GWh)} & \textbf{(GWh)} & \textbf{(GWh)} \\'
    )
    lines.append(r'\midrule')

    for r in rows:
        if r['zone'] == 'Sweden (all zones)':
            lines.append(r'\midrule')
            zone_cell = r'\textbf{Sweden (all zones)}'
        else:
            zone_cell = r['zone']

        net_fmt = f"{r['net_gwh']:+,.0f}"
        lines.append(
            f"{zone_cell} & {r['partners']} "
            f"& {r['exp_gwh']:,.0f} "
            f"& {r['imp_gwh']:,.0f} "
            f"& {net_fmt} \\\\"
        )

    lines.append(r'\bottomrule')
    lines.append(
        r'\multicolumn{5}{p{0.97\linewidth}}{\footnotesize'
    )
    lines.append(
        r'  \textit{Note:} Net Balance = Gross Import $-$ Gross Export; positive'
        r' values denote net-importing zones. Gross import and export figures are'
        r' derived from the aggregate hourly \texttt{Net\_Exchange} variable (MW),'
        r' where positive values represent net imports and negative values net exports.'
        r' Bilateral flows per individual trading partner are not available in the'
        r' source dataset; connected partners are inferred from interconnection'
        r' bottleneck variables. The Sweden aggregate row sums across all four bidding'
        r' zones: net exchange cancels for intra-Swedish flows, but the gross'
        r' import/export totals include those internal flows.}'
    )
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  PRODUCTION MIX TABLE  (LaTeX)
# ══════════════════════════════════════════════════════════════════════════════

def build_production_table():
    print("\n[3/3]  Production mix table (2025) …")

    rows = []
    agg_wind = agg_total = agg_cons = 0.0

    for zone in ZONES:
        df = pd.read_excel(PATHS[zone])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df2025 = df[df['Timestamp'].dt.year == YEAR_TABLE]

        wind_gwh  = df2025['Wind_Forecast'].fillna(0).sum() / 1_000
        total_gwh = df2025['Total_Forecast'].fillna(0).sum() / 1_000
        other_gwh = max(total_gwh - wind_gwh, 0.0)
        cons_gwh  = df2025['Consumption_Forecast'].fillna(0).sum() / 1_000
        bal_gwh   = total_gwh - cons_gwh

        wind_pct  = 100.0 * wind_gwh  / total_gwh if total_gwh else 0.0
        other_pct = 100.0 - wind_pct

        rows.append(dict(
            zone      = zone,
            wind_gwh  = wind_gwh,
            wind_pct  = wind_pct,
            other_gwh = other_gwh,
            other_pct = other_pct,
            total_gwh = total_gwh,
            cons_gwh  = cons_gwh,
            bal_gwh   = bal_gwh,
        ))
        agg_wind  += wind_gwh
        agg_total += total_gwh
        agg_cons  += cons_gwh

        print(f"    {zone}: wind = {wind_gwh:,.0f} GWh ({wind_pct:.1f}%) | "
              f"other = {other_gwh:,.0f} GWh ({other_pct:.1f}%) | "
              f"total = {total_gwh:,.0f} GWh | cons = {cons_gwh:,.0f} GWh | "
              f"balance = {bal_gwh:+,.0f} GWh")

    # Sweden totals
    sw_wind_pct  = 100.0 * agg_wind / agg_total if agg_total else 0.0
    sw_other_gwh = agg_total - agg_wind
    sw_bal        = agg_total - agg_cons
    rows.append(dict(
        zone      = 'Sweden (total)',
        wind_gwh  = agg_wind,
        wind_pct  = sw_wind_pct,
        other_gwh = sw_other_gwh,
        other_pct = 100.0 - sw_wind_pct,
        total_gwh = agg_total,
        cons_gwh  = agg_cons,
        bal_gwh   = sw_bal,
    ))

    tex = _production_latex(rows)
    _save_tex(tex, 'production_mix_2025.tex')
    print("\n-- LaTeX output --------------------------------------------------------------")
    print(tex)


def _production_latex(rows):
    lines = []
    lines.append(r'% Generated by descriptive_overview.py')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Electricity Generation Mix and Consumption by Swedish Bidding Zone, 2025}'
    )
    lines.append(r'\label{tab:production_mix_2025}')
    lines.append(r'\begin{tabular}{lrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(
        r' & \multicolumn{2}{c}{\textbf{Wind}} '
        r'& \multicolumn{2}{c}{\textbf{Other$^\dagger$}} '
        r'& \textbf{Total Prod.} & \textbf{Consumption} & \textbf{Balance} \\'
    )
    lines.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    lines.append(
        r'\textbf{Zone} '
        r'& \textbf{GWh} & \textbf{\%} '
        r'& \textbf{GWh} & \textbf{\%} '
        r'& \textbf{(GWh)} & \textbf{(GWh)} & \textbf{(GWh)} \\'
    )
    lines.append(r'\midrule')

    for r in rows:
        if r['zone'] == 'Sweden (total)':
            lines.append(r'\midrule')
            zone_cell = r'\textbf{Sweden (total)}'
        else:
            zone_cell = r['zone']

        lines.append(
            f"{zone_cell} "
            f"& {r['wind_gwh']:,.0f} & {r['wind_pct']:.1f} "
            f"& {r['other_gwh']:,.0f} & {r['other_pct']:.1f} "
            f"& {r['total_gwh']:,.0f} "
            f"& {r['cons_gwh']:,.0f} "
            f"& {r['bal_gwh']:+,.0f} \\\\"
        )

    lines.append(r'\bottomrule')
    lines.append(r'\multicolumn{8}{p{0.99\linewidth}}{\footnotesize')
    lines.append(
        r'  $^\dagger$\,\textit{Other} denotes total generation forecast minus wind'
        r' forecast and encompasses nuclear, hydro, solar, and thermal generation.'
        r' A disaggregated breakdown by source is not available in the underlying'
        r' dataset. All values are annual sums of hourly day-ahead forecasts (MW\,h);'
        r' divide by 1{,}000 to convert to TWh.'
        r' \textit{Balance} = Total Generation $-$ Consumption; positive values'
        r' indicate a net production surplus (potential net exporter).'
        r' Figures may deviate from official statistics (Eurostat, ENTSO-E) because'
        r' they are based on day-ahead forecasts rather than realised generation and'
        r' consumption.}'
    )
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  DESCRIPTIVE OVERVIEW  -  Swedish Electricity Market")
    print(f"  Zones: {', '.join(ZONES)}  |  Table year: {YEAR_TABLE}")
    print("=" * 72)

    plot_price_series()
    build_flows_table()
    build_production_table()

    print("\n" + "=" * 72)
    print(f"  Done.  Outputs saved to: {OUTPUT_DIR}/")
    print("    price_series_raw.png          (VWAP daily avg, raw)")
    print("    price_series_log.png          (VWAP daily avg, ln-transformed)")
    print("    volatility_raw.png            (daily intra-day std, raw)")
    print("    volatility_log.png            (daily std of ln-prices)")
    print("    monthly_price_raw.png         (monthly avg of daily VWAP)")
    print("    monthly_price_log.png         (monthly avg of ln(VWAP + shift))")
    print("    monthly_volatility_raw.png    (monthly avg of daily std, raw)")
    print("    monthly_volatility_log.png    (monthly avg of daily std of ln-prices)")
    print("    cross_border_flows_2025.tex")
    print("    production_mix_2025.tex")
    print("=" * 72)
