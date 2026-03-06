"""
production_overview.py  –  Production Mix by Zone: SE1–SE4, 2015–2025
═══════════════════════════════════════════════════════════════════════

Produces:
    production_mix_by_zone.png  –  2×2 stacked bar charts (one per zone)
                                   Annual totals (TWh), 2015–2025,
                                   broken down by generation type.

Data source:
    production_total/Production_YYYY_SE1,SE2,SE3,SE4_Monthly.csv
    (semicolon-delimited, monthly MWh totals per generation type)

Usage:
    python production_overview.py   (from MSc-Econ-Thesis root)
"""

import gc
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FOLDER_PATH = (
    r'C:\Users\patri\OneDrive - Handelshögskolan i Stockholm'
    r'\Master Thesis Economics 2025_2026 - General'
    r'\07_Code and Data\production_total'
)
YEARS = [str(y) for y in range(2015, 2026)]
OUTPUT_DIR = 'results/production_overview'
OUTPUT_FILE = 'production_mix_by_zone.png'

ZONES = ['SE1', 'SE2', 'SE3', 'SE4']

# All possible generation types (superset across all files and zones)
PROD_TYPES = [
    'Nuclear',
    'Hydro Water Reservoir',
    'Wind Onshore',
    'Wind Offshore',
    'Solar',
    'Fossil Gas',
    'Marine',
    'Other',
]

# Consistent colors per generation type
TYPE_COLORS = {
    'Nuclear':              '#E64B35',   # red
    'Hydro Water Reservoir':'#4DBBD5',   # teal/blue
    'Wind Onshore':         '#3C5488',   # dark blue
    'Wind Offshore':        '#7E9CCF',   # light blue
    'Solar':                '#F39B7F',   # orange
    'Fossil Gas':           '#8D8D8D',   # grey
    'Marine':               '#00A087',   # green
    'Other':                '#B09C85',   # brown
}

ZONE_TITLES = {
    'SE1': 'SE1 – Northern Sweden',
    'SE2': 'SE2 – North-Central Sweden',
    'SE3': 'SE3 – South-Central Sweden',
    'SE4': 'SE4 – Southern Sweden',
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (mirrors price_merger.py folder-scan approach)
# ══════════════════════════════════════════════════════════════════════════════

def load_production_data():
    """Load and concatenate all yearly CSVs, return a unified DataFrame."""
    files = sorted([
        f for f in os.listdir(FOLDER_PATH)
        if f.endswith('.csv') and any(y in f for y in YEARS)
    ])

    all_frames = []
    for file in files:
        path = os.path.join(FOLDER_PATH, file)
        df = pd.read_csv(path, sep=';')
        df['Date'] = pd.to_datetime(df['Delivery Date Start CET'])
        df['Year'] = df['Date'].dt.year
        all_frames.append(df)
        print(f'  Loaded: {file}  ({len(df)} rows)')

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined[combined['Year'].between(2015, 2025)]
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_by_zone_year(df):
    """
    Returns a dict: zone -> DataFrame(index=Year, columns=PROD_TYPES) in TWh.
    Missing generation types are filled with 0.
    """
    zone_data = {}
    for zone in ZONES:
        rows = []
        for year, grp in df.groupby('Year'):
            row = {'Year': year}
            for ptype in PROD_TYPES:
                col = f'{zone} {ptype} (MWh)'
                if col in grp.columns:
                    row[ptype] = grp[col].fillna(0).sum() / 1e6   # MWh → TWh
                else:
                    row[ptype] = 0.0
            rows.append(row)
        zone_df = pd.DataFrame(rows).set_index('Year')
        zone_data[zone] = zone_df
    return zone_data


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_production_mix(zone_data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    years = list(range(2015, 2026))
    x = range(len(years))

    # Collect legend handles (only types that have any data)
    legend_handles = []
    legend_labels = []

    for ax, zone in zip(axes_flat, ZONES):
        zdf = zone_data[zone]

        # Only plot types that have at least some non-zero data in this zone
        active_types = [t for t in PROD_TYPES if zdf[t].sum() > 0]

        bottoms = [0.0] * len(years)
        for ptype in active_types:
            values = [zdf.loc[y, ptype] if y in zdf.index else 0.0 for y in years]
            bar = ax.bar(
                x,
                values,
                bottom=bottoms,
                color=TYPE_COLORS[ptype],
                label=ptype,
                width=0.72,
                edgecolor='white',
                linewidth=0.4,
            )
            bottoms = [b + v for b, v in zip(bottoms, values)]

            # Collect legend entries (avoid duplicates)
            if ptype not in legend_labels:
                legend_handles.append(bar)
                legend_labels.append(ptype)

        ax.set_title(ZONE_TITLES[zone], fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Annual Production (TWh)', fontsize=10)
        ax.set_xlabel('Year', fontsize=10)
        ax.grid(axis='y', alpha=0.25, linewidth=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Swedish Electricity Production by Generation Type and Bidding Zone\n'
        'Annual Totals, 2015–2025 (TWh)',
        fontsize=14, fontweight='bold', y=1.01,
    )

    # Shared legend below all subplots
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        ncol=len(legend_labels),
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
        title='Generation Type',
        title_fontsize=10,
    )

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f'\n  Saved -> {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  PRODUCTION MIX OVERVIEW  –  Swedish Electricity Market (SE1–SE4)')
    print('  Period: 2015–2025  |  Granularity: Annual (aggregated from monthly)')
    print('=' * 70)

    print('\n[1/3]  Loading CSV files …')
    df = load_production_data()
    print(f'  Combined: {len(df)} monthly observations across all zones and years.')

    print('\n[2/3]  Aggregating to annual totals per zone …')
    zone_data = aggregate_by_zone_year(df)
    for zone in ZONES:
        totals = zone_data[zone].sum(axis=1)
        print(f'  {zone}: {totals.min():.1f}–{totals.max():.1f} TWh/year')

    print('\n[3/3]  Plotting stacked bar charts …')
    plot_production_mix(zone_data)

    print('\n' + '=' * 70)
    print(f'  Done.  Output saved to: {OUTPUT_DIR}/{OUTPUT_FILE}')
    print('=' * 70)
