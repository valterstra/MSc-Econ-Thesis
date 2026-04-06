"""
baseline_table.py

Reads ARMA(1,1)-GARCH(1,1)-X baseline results for 2024-01-01 to 2025-12-31
from output from stata/garch_results/ and produces:

  1. Terminal table  — coefficient (stars) / (SE) format, all controls
  2. LaTeX table     — saved to output from stata/baseline_results/baseline_garch_table.tex

Zones as columns (SE1-SE4), all mean + variance equation coefficients as rows.
Missing zones (file not yet generated) are shown as blank columns.

Usage:
    python baseline_table.py
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

ZONES       = ['SE1', 'SE2', 'SE3', 'SE4']
RESULTS_DIR = 'output from stata/garch_results'
STUB        = '2025-01-01_2025-12-31_log_hetfull'
OUT_DIR     = 'output from stata/baseline_results'

# Human-readable labels for known variables
VAR_LABELS = {
    # Mean equation
    '_cons':            'Constant',
    'wind_log':         'Wind (log)',
    'hydro_log_ds':     'Hydro (log, ds.)',
    'consump_log_ds':   'Consumption (log, ds.)',
    'oil_log':          'Oil price (log)',
    'gas_log':          'Gas price (log)',
    'ar_L1':            'AR(1)',
    'ma_L1':            'MA(1)',
    # Variance equation
    'omega':                    r'$\omega$ (intercept)',
    'arch1':                    r'$\alpha$ ARCH(1)',
    'garch1':                   r'$\delta$ GARCH(1)',
    'het_wind_log':             r'$\gamma$ Wind (log)',
    'het_hydro_log_ds':         r'$\gamma$ Hydro (log, ds.)',
    'het_consump_log_ds':       r'$\gamma$ Consumption (log, ds.)',
    'het_oil_log':              r'$\gamma$ Oil price (log)',
    'het_gas_log':              r'$\gamma$ Gas price (log)',
    'persistence':              r'Persistence ($\alpha+\delta$)',
}

# Fixed row order (net exchange rows inserted dynamically between gas and AR)
MEAN_ROWS_FIXED = [
    ('coef', '_cons'),
    ('coef', 'wind_log'),
    ('coef', 'hydro_log_ds'),
    ('coef', 'consump_log_ds'),
    ('coef', 'oil_log'),
    ('coef', 'gas_log'),
    # netexch rows inserted here
    ('coef', 'ar_L1'),
    ('coef', 'ma_L1'),
]

VAR_ROWS_FIXED = [
    ('var_coef', 'omega'),
    ('var_coef', 'arch1'),
    ('var_coef', 'garch1'),
    ('var_coef', 'het_wind_log'),
    ('var_coef', 'het_hydro_log_ds'),
    ('var_coef', 'het_consump_log_ds'),
    ('var_coef', 'het_oil_log'),
    ('var_coef', 'het_gas_log'),
    # het_netexch rows inserted here
    ('var_coef', 'persistence'),
]


# ── Load data ──────────────────────────────────────────────────────────────────

def load_zone(zone: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f'garch_results_{zone}_{STUB}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['stars'] = df['stars'].fillna('')
    return df


def fmt(val: float) -> str:
    """Format a coefficient: 4 decimal places normally, scientific notation if very small."""
    if val is None:
        return ''
    if abs(val) < 0.0001 and val != 0:
        return f'{val:.2e}'
    return f'{val:.4f}'


def get_cell(df: pd.DataFrame, type_val: str, var_val: str) -> tuple:
    """Returns (value, se, stars) or (None, None, '') if not found."""
    if df is None:
        return None, None, ''
    row = df[(df['type'] == type_val) & (df['variable'] == var_val)]
    if row.empty:
        return None, None, ''
    r = row.iloc[0]
    return r['value'], r.get('se', np.nan), r['stars']


def get_netexch_vars(data: dict) -> list:
    """Returns sorted list of all netexch variable names found across all zones."""
    netexch = set()
    for df in data.values():
        if df is None:
            continue
        for _, row in df[df['type'] == 'coef'].iterrows():
            if row['variable'].startswith('netexch_'):
                netexch.add(row['variable'])
    return sorted(netexch)


def get_het_netexch_vars(data: dict) -> list:
    het = set()
    for df in data.values():
        if df is None:
            continue
        for _, row in df[df['type'] == 'var_coef'].iterrows():
            if row['variable'].startswith('het_netexch_'):
                het.add(row['variable'])
    return sorted(het)


def format_partner(varname: str) -> str:
    """'netexch_fi' -> 'Net export: FI',  'het_netexch_fi' -> 'Net export: FI (var.)'"""
    if varname.startswith('het_netexch_'):
        partner = varname.replace('het_netexch_', '').upper()
        return rf'$\gamma$ Net export: {partner}'
    else:
        partner = varname.replace('netexch_', '').upper()
        return f'Net export: {partner}'


def build_row_list(data: dict) -> list:
    """Returns ordered list of (type, variable) tuples for all rows."""
    netexch     = get_netexch_vars(data)
    het_netexch = get_het_netexch_vars(data)

    rows = []
    # Mean equation (insert netexch after gas_log, before ar_L1)
    for t, v in MEAN_ROWS_FIXED:
        if v == 'ar_L1':
            for n in netexch:
                rows.append(('coef', n))
        rows.append((t, v))

    # Variance equation (insert het_netexch after het_gas_log, before persistence)
    for t, v in VAR_ROWS_FIXED:
        if v == 'persistence':
            for n in het_netexch:
                rows.append(('var_coef', n))
        rows.append((t, v))

    return rows


# ── Terminal table ─────────────────────────────────────────────────────────────

def print_baseline_table(data: dict) -> None:
    rows = build_row_list(data)
    col_w = 18
    total_w = 28 + col_w * len(ZONES)

    present = [z for z in ZONES if data[z] is not None]
    missing = [z for z in ZONES if data[z] is None]
    if missing:
        print(f'  Note: missing zones (not yet estimated): {", ".join(missing)}')

    print(f'\n{"="*total_w}')
    print(f'  ARMA(1,1)-GARCH(1,1)-X  Baseline 2025  [all controls in variance eq.]')
    print(f'  *** p<0.01   ** p<0.05   * p<0.10   SEs in parentheses')
    print(f'{"="*total_w}')
    print(f'{"Variable":<28}' + ''.join(f'{z:>{col_w}}' for z in ZONES))
    print(f'{"-"*total_w}')

    prev_type = None
    for type_val, var_val in rows:
        # Section header
        if type_val != prev_type:
            section = 'Mean equation' if type_val == 'coef' else 'Variance equation'
            if prev_type is not None:
                print(f'{"-"*total_w}')
            print(f'  {section}')
            prev_type = type_val

        # Label
        label = VAR_LABELS.get(var_val)
        if label is None:
            if var_val.startswith('netexch_') or var_val.startswith('het_netexch_'):
                label = format_partner(var_val)
            else:
                label = var_val

        # Coefficient row
        coef_row = f'  {label:<26}'
        se_row   = f'  {"":<26}'
        for zone in ZONES:
            val, se, stars = get_cell(data[zone], type_val, var_val)
            if val is None:
                coef_row += f'{"":>{col_w}}'
                se_row   += f'{"":>{col_w}}'
            elif var_val == 'persistence':
                coef_row += f'{fmt(val)}{stars}'.rjust(col_w)
                se_row   += f'{"":>{col_w}}'
            else:
                coef_row += f'{fmt(val)}{stars}'.rjust(col_w)
                se_row   += f'({fmt(se)})'.rjust(col_w)

        print(coef_row)
        if not all(var_val == 'persistence' for _, v in [(type_val, var_val)]):
            # Print SE row (skip for persistence which has no SE)
            if var_val != 'persistence':
                print(se_row)

    # Fit statistics
    print(f'{"-"*total_w}')
    print(f'  Fit statistics')
    print(f'{"-"*total_w}')
    for fit_var, label in [('N', 'N'), ('ll', 'Log-likelihood'), ('AIC', 'AIC'), ('BIC', 'BIC')]:
        row_str = f'  {label:<26}'
        for zone in ZONES:
            val, _, _ = get_cell(data[zone], 'fit', fit_var)
            if val is None:
                row_str += f'{"":>{col_w}}'
            elif fit_var == 'N':
                row_str += f'{int(val):,}'.rjust(col_w)
            else:
                row_str += f'{val:,.1f}'.rjust(col_w)
        print(row_str)

    for lbq_type, label in [('lbq_std', 'LBQ(24) std. resid.'), ('lbq_sq', 'LBQ(24) sq. resid.')]:
        row_str = f'  {label:<26}'
        for zone in ZONES:
            val, _, stars = get_cell(data[zone], lbq_type, 'L24')
            prow = df_zone = data[zone]
            if prow is not None:
                sub = prow[(prow['type'] == lbq_type) & (prow['variable'] == 'L24')]
                pval = sub['pval'].values[0] if not sub.empty else None
            else:
                pval = None
            if val is None:
                row_str += f'{"":>{col_w}}'
            else:
                row_str += f'{val:.1f}{stars}'.rjust(col_w)
        print(row_str)

    print(f'{"="*total_w}')


# ── LaTeX table ────────────────────────────────────────────────────────────────

def write_latex_table(data: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'baseline_garch_table.tex')

    rows = build_row_list(data)
    n_zones = len(ZONES)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{ARMA(1,1)--GARCH(1,1)-X Baseline Estimates, 2025}')
    lines.append(r'\label{tab:baseline_garch}')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{l' + 'c' * n_zones + '}')
    lines.append(r'\toprule')
    lines.append(' & ' + ' & '.join(ZONES) + r' \\')
    lines.append(r'\midrule')

    prev_type = None
    for type_val, var_val in rows:
        if type_val != prev_type:
            if prev_type is not None:
                lines.append(r'\addlinespace[4pt]')
                lines.append(r'\midrule')
            section = r'\textit{Mean equation}' if type_val == 'coef' else r'\textit{Variance equation}'
            lines.append(r'\multicolumn{' + str(n_zones + 1) + r'}{l}{' + section + r'} \\')
            lines.append(r'\addlinespace[2pt]')
            prev_type = type_val

        label = VAR_LABELS.get(var_val)
        if label is None:
            if var_val.startswith('netexch_') or var_val.startswith('het_netexch_'):
                partner = var_val.replace('netexch_', '').replace('het_', '').replace('netexch_', '').upper()
                label = f'Net export: {partner}'
            else:
                label = var_val

        # Coefficient row
        cells = []
        se_cells = []
        for zone in ZONES:
            val, se, stars = get_cell(data[zone], type_val, var_val)
            if val is None:
                cells.append('')
                se_cells.append('')
            elif var_val == 'persistence':
                cells.append(f'${fmt(val)}$')
                se_cells.append('')
            else:
                cells.append(f'${fmt(val)}${stars}')
                se_cells.append(f'$({fmt(se)})$')

        lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r' \\')
        if var_val != 'persistence':
            lines.append(' & ' + ' & '.join(se_cells) + r' \\[3pt]')

    # Fit statistics
    lines.append(r'\addlinespace[4pt]')
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{' + str(n_zones + 1) + r'}{l}{\textit{Fit statistics}} \\')
    lines.append(r'\addlinespace[2pt]')

    for fit_var, label in [('N', '$N$'), ('ll', 'Log-likelihood'), ('AIC', 'AIC'), ('BIC', 'BIC')]:
        cells = []
        for zone in ZONES:
            val, _, _ = get_cell(data[zone], 'fit', fit_var)
            if val is None:
                cells.append('')
            elif fit_var == 'N':
                cells.append(f'${int(val):,}$'.replace(',', '{,}'))
            else:
                cells.append(f'${val:,.1f}$'.replace(',', '{,}'))
        lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r' \\')

    for lbq_type, label in [('lbq_std', r'LBQ(24) std.\ resid.'), ('lbq_sq', r'LBQ(24) sq.\ resid.')]:
        cells = []
        for zone in ZONES:
            df = data[zone]
            if df is None:
                cells.append('')
                continue
            sub = df[(df['type'] == lbq_type) & (df['variable'] == 'L24')]
            if sub.empty:
                cells.append('')
            else:
                val   = sub['value'].values[0]
                stars = sub['stars'].values[0] if sub['stars'].values[0] else ''
                cells.append(f'${val:.1f}${stars}')
        lines.append(f'\\quad {label} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')  # resizebox
    lines.append(r'\begin{tablenotes}')
    lines.append(r'\footnotesize')
    lines.append(r'\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. '
                 r'Standard errors (in parentheses) are Bollerslev--Wooldridge QML robust. '
                 r'The variance equation includes all exogenous controls multiplicatively via \texttt{het()}. '
                 r'Persistence $= \hat{\alpha} + \hat{\delta}$. '
                 r'LBQ(24) is the Ljung--Box $Q$-statistic at 24 lags on standardised residuals.')
    lines.append(r'\end{tablenotes}')
    lines.append(r'\end{table}')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {out_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    data = {zone: load_zone(zone) for zone in ZONES}

    present = [z for z in ZONES if data[z] is not None]
    print(f'Zones loaded: {present}')

    print_baseline_table(data)
    write_latex_table(data)
