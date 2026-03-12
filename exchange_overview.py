"""
exchange_overview.py  –  Cross-Border Exchange and Negative Prices
═══════════════════════════════════════════════════════════════════

Produces:
    results/descriptive_overview/negative_prices_by_zone.png
        – Annual count of hours with negative spot price per zone, 2015–2025.

    results/descriptive_overview/net_exports_by_zone.png
        – Annual net exports (TWh) per zone, 2015–2025.

    results/descriptive_overview/cross_border_matrix_2025.tex
        – LaTeX matrix table: net exchange (GWh) per zone × trading partner, 2025.

Data sources:
    master data files/2015-2025/Spot_Prices_2015_2025.xlsx  (hourly, EUR/MWh)
    export_total/Exchange_YYYY_SE1,SE2,SE3,SE4_Monthly.csv  (monthly MWh)

Usage:
    python exchange_overview.py   (from MSc-Econ-Thesis root)
"""

import gc
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EXCHANGE_FOLDER = (
    r'C:\Users\patri\OneDrive - Handelshögskolan i Stockholm'
    r'\Master Thesis Economics 2025_2026 - General'
    r'\07_Code and Data\export_total'
)
SPOT_PRICE_FILE = 'master data files/2015-2025/Spot_Prices_2015_2025.xlsx'
OUTPUT_DIR     = 'results/descriptive_overview'
YEARS          = list(range(2015, 2026))
ZONES          = ['SE1', 'SE2', 'SE3', 'SE4']

ZONE_COLORS = {
    'SE1': '#08306b',
    'SE2': '#2171b5',
    'SE3': '#6baed6',
    'SE4': '#bdd7e7',
}

# Raw partner label  ->  aggregated country name shown in table
COUNTRY_MAP = {
    'FI':    'Finland',
    'NO1':   'Norway',
    'NO2':   'Norway',
    'NO3':   'Norway',
    'NO4':   'Norway',
    'NO5':   'Norway',
    'DK1':   'Denmark',
    'DK2':   'Denmark',
    'DE-LU': 'Germany',
    'LT':    'Lithuania',
    'PL':    'Poland',
    'SE1':   'SE1',
    'SE2':   'SE2',
    'SE3':   'SE3',
    'SE4':   'SE4',
}

# Desired column order in the matrix table
PARTNER_ORDER = ['SE1', 'SE2', 'SE3', 'SE4',
                 'Finland', 'Norway', 'Denmark',
                 'Germany', 'Lithuania', 'Poland']


# ══════════════════════════════════════════════════════════════════════════════
#  1. NEGATIVE PRICES
# ══════════════════════════════════════════════════════════════════════════════

def load_spot_prices():
    """Load hourly spot price data for all zones; return DataFrame with Year column."""
    print('  Loading spot price data …')
    price_cols = ['Timestamp'] + [f'{z}_Price (EUR)' for z in ZONES]
    df = pd.read_excel(SPOT_PRICE_FILE, usecols=price_cols)
    df['Year'] = pd.to_datetime(df['Timestamp']).dt.year
    return df[df['Year'].between(2015, 2025)]


def count_negative_prices(price_df=None):
    """Return DataFrame(Zone, Year, NegativeHours) from hourly spot price file."""
    df = price_df if price_df is not None else load_spot_prices()
    records = []
    for zone in ZONES:
        col = f'{zone}_Price (EUR)'
        for year, grp in df.groupby('Year'):
            n_neg = int((grp[col] < 0).sum())
            records.append({'Zone': zone, 'Year': year, 'NegativeHours': n_neg})
    return pd.DataFrame(records)


def plot_negative_prices(neg_df):
    fig, ax = plt.subplots(figsize=(13, 6))

    x      = np.arange(len(YEARS))
    width  = 0.18
    offsets = np.linspace(-(len(ZONES) - 1) / 2,
                           (len(ZONES) - 1) / 2, len(ZONES)) * width

    for i, zone in enumerate(ZONES):
        zdf    = neg_df[neg_df['Zone'] == zone].set_index('Year')
        values = [int(zdf.loc[y, 'NegativeHours']) if y in zdf.index else 0
                  for y in YEARS]
        ax.bar(x + offsets[i], values, width=width,
               color=ZONE_COLORS[zone], label=zone,
               edgecolor='white', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Hours with Negative Price', fontsize=10)
    ax.set_xlabel('Year', fontsize=10)
    ax.legend(title='Bidding Zone', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'negative_prices_by_zone.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
#  2. NET EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

def load_exchange_data():
    """Load all yearly exchange CSVs; return combined DataFrame with 'Year' column."""
    files = sorted([
        f for f in os.listdir(EXCHANGE_FOLDER)
        if f.endswith('.csv') and any(str(y) in f for y in YEARS)
    ])
    frames = []
    for fname in files:
        path = os.path.join(EXCHANGE_FOLDER, fname)
        df   = pd.read_csv(path, sep=';')
        m    = re.search(r'(\d{4})', fname)
        df['Year'] = int(m.group(1)) if m else None
        frames.append(df)
        print(f'  Loaded: {fname}  ({len(df)} rows)')
    return pd.concat(frames, ignore_index=True)


def compute_net_exports(exdf):
    """
    Return DataFrame indexed by Year, columns = ZONES, values = net exports (TWh).
    Net Export = Export − Import = −(Net Position), where Net Position = Import − Export.
    """
    records = []
    for year, grp in exdf.groupby('Year'):
        row = {'Year': year}
        for zone in ZONES:
            col = f'{zone} Total Net Position (MWh)'
            row[zone] = (-grp[col].sum() / 1e6) if col in grp.columns else 0.0
        records.append(row)
    return pd.DataFrame(records).set_index('Year')


def plot_net_exports(net_df):
    fig, ax = plt.subplots(figsize=(13, 6))

    years   = [y for y in YEARS if y in net_df.index]
    x       = np.arange(len(years))
    width   = 0.18
    offsets = np.linspace(-(len(ZONES) - 1) / 2,
                           (len(ZONES) - 1) / 2, len(ZONES)) * width

    for i, zone in enumerate(ZONES):
        values = [net_df.loc[y, zone] for y in years]
        ax.bar(x + offsets[i], values, width=width,
               color=ZONE_COLORS[zone], label=zone,
               edgecolor='white', linewidth=0.4)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Net Exports (TWh)', fontsize=10)
    ax.set_xlabel('Year', fontsize=10)
    ax.legend(title='Bidding Zone', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'net_exports_by_zone.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
#  3. TRADING PARTNER MATRIX (2025)
# ══════════════════════════════════════════════════════════════════════════════

def compute_partner_matrix(exdf):
    """
    Return dict: zone -> {country: net_export_MWh} for 2025.
    Net export = Export(zone->partner) − Import(partner->zone).
    Trading partners are aggregated by country (NO1/NO3/NO4 -> Norway, etc.).
    """
    df25 = exdf[exdf['Year'] == 2025]

    matrix = {}
    for zone in ZONES:
        exp_pat = re.compile(rf'^{zone} {zone}->(.+) Export \(MWh\)$')
        imp_pat = re.compile(rf'^{zone} (.+)->{zone} Import \(MWh\)$')

        exp_cols = {}
        imp_cols = {}
        for col in df25.columns:
            m = exp_pat.match(col)
            if m:
                exp_cols[m.group(1)] = col
            m2 = imp_pat.match(col)
            if m2:
                imp_cols[m2.group(1)] = col

        country_net = {}
        for partner in set(list(exp_cols) + list(imp_cols)):
            exp = df25[exp_cols[partner]].sum() if partner in exp_cols else 0.0
            imp = df25[imp_cols[partner]].sum() if partner in imp_cols else 0.0
            net = exp - imp  # positive = net exporter to this partner
            country = COUNTRY_MAP.get(partner, partner)
            country_net[country] = country_net.get(country, 0.0) + net

        matrix[zone] = country_net

    return matrix


def _fmt(mwh):
    """Format MWh value as signed GWh with thousand separator; '---' if near zero."""
    gwh = mwh / 1_000
    if abs(gwh) < 1.0:
        return '---'
    sign = '+' if gwh >= 0 else ''
    return f'${sign}{int(round(gwh)):,}$'


def write_matrix_tex(matrix):
    """Write the cross-border net exchange matrix table to LaTeX."""
    # Determine which partners actually appear in the data
    all_partners = set()
    for zone_data in matrix.values():
        all_partners.update(zone_data.keys())

    cols = [c for c in PARTNER_ORDER if c in all_partners]
    n_total = len(cols) + 1                     # +1 for Zone column
    col_spec = 'l' + 'r' * len(cols)

    def _hdr(s):
        return f'\\textbf{{{s}}}'

    lines = [
        '% Generated by exchange_overview.py',
        '% Source: ENTSO-E export_total monthly MWh files',
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Cross-Border Net Exchange by Swedish Bidding Zone and Trading Partner, 2025 (GWh)}',
        r'\label{tab:cross_border_matrix_2025}',
        r'\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
    ]

    # Header
    header_cells = [_hdr(c) for c in cols]
    lines.append(r'\textbf{Zone} & ' + ' & '.join(header_cells) + r' \\')
    lines.append(r'\midrule')

    for zone in ZONES:
        cells = []
        for col in cols:
            if col == zone:
                cells.append(r'\multicolumn{1}{c}{---}')
            else:
                val = matrix[zone].get(col, 0.0)
                cells.append(_fmt(val))
        lines.append(f'\\textbf{{{zone}}} & ' + ' & '.join(cells) + r' \\')

    lines += [
        r'\midrule',
    ]

    # Sweden total row (sum over zones, excluding intra-Swedish flows)
    external_partners = [c for c in cols if c not in ZONES]
    total_cells = []
    for col in cols:
        if col in ZONES:
            total_cells.append(r'\multicolumn{1}{c}{---}')
        else:
            total = sum(matrix[z].get(col, 0.0) for z in ZONES)
            total_cells.append(_fmt(total))

    lines.append(
        r'\textbf{Sweden (total)} & '
        + ' & '.join(total_cells) + r' \\'
    )

    lines += [
        r'\bottomrule',
        f'\\multicolumn{{{n_total}}}{{p{{0.99\\linewidth}}}}{{\\footnotesize',
        r'  \textit{Note:} Values are annual net exchanges in GWh (positive = net export',
        r'  from the row zone to the column partner; negative = net import).',
        r'  Norway aggregates NO1, NO3, NO4; Denmark aggregates DK1, DK2.',
        r'  SE-zone columns show intra-Swedish bilateral net flows; the Sweden total row',
        r'  excludes intra-Swedish flows and sums only external partner columns.',
        r'  Source: ENTSO-E realised monthly cross-border exchange data.}',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]

    out = os.path.join(OUTPUT_DIR, 'cross_border_matrix_2025.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  EXCHANGE OVERVIEW  –  Swedish Electricity Market (SE1–SE4)')
    print('  Period: 2015–2025  |  Sources: ENTSO-E Exchange + Spot Price data')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\n[1/4]  Counting hours with negative spot prices …')
    neg_df = count_negative_prices()
    for zone in ZONES:
        zdf = neg_df[neg_df['Zone'] == zone].set_index('Year')
        totals = zdf['NegativeHours'].sum()
        print(f'  {zone}: {totals} total negative-price hours across 2015–2025')
    plot_negative_prices(neg_df)

    print('\n[2/4]  Loading exchange CSV files …')
    exdf = load_exchange_data()
    print(f'  Combined: {len(exdf)} monthly observations.')

    print('\n[3/4]  Plotting net exports …')
    net_df = compute_net_exports(exdf)
    for zone in ZONES:
        print(f'  {zone}: {net_df[zone].min():.1f}–{net_df[zone].max():.1f} TWh/year  '
              f'(range 2015–2025)')
    plot_net_exports(net_df)

    print('\n[4/4]  Computing cross-border matrix (2025) …')
    matrix = compute_partner_matrix(exdf)
    for zone in ZONES:
        top = sorted(matrix[zone].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_str = ', '.join(f'{k}: {v/1e6:.1f} TWh' for k, v in top)
        print(f'  {zone} top partners: {top_str}')
    write_matrix_tex(matrix)

    print('\n' + '=' * 70)
    print('  Done.')
    print('=' * 70)
