"""merit_order_graph.py  –  Merit Order Curve: Swedish Electricity Market
═══════════════════════════════════════════════════════════════════════════════

Produces:
  merit_order_sweden.png  –  Illustrative merit order supply stack for Sweden
                             showing the Merit Order Effect of Wind & Solar

Note:
  Values are directionally correct based on publicly available information
  about Sweden's generation mix. They are NOT derived from the primary dataset.

  Sources:
    · Swedish Energy Agency (Energimyndigheten) – installed capacity 2023/24
    · Nord Pool / ENTSO-E – variable cost estimates by technology
    · IEA World Energy Outlook – technology cost benchmarks

  Approximate marginal costs (variable O&M + fuel, EUR/MWh):
    Solar / Wind  :  ~0–2   (zero fuel, minimal variable O&M)
    Hydro         :  ~2–8   (water value / reservoir opportunity cost)
    Nuclear       :  ~8–15  (uranium fuel + variable O&M)
    Biomass / CHP :  ~35–55 (biomass fuel, district-heat credit)
    Gas / Oil     :  ~70–100 (gas price + EU ETS carbon cost)

  Approximate installed capacity (GW, 2023/24):
    Solar:  ~3.5   Wind: ~14.0   Hydro: ~16.5
    Nuclear: ~6.9  Biomass/CHP: ~4.0  Gas/Oil: ~2.0

Usage:
    python merit_order_graph.py   (from MSc-Econ-Thesis root)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTPUT_DIR = 'results/descriptive_overview'

# ══════════════════════════════════════════════════════════════════════════════
#  MERIT ORDER DATA
# ══════════════════════════════════════════════════════════════════════════════

# (label, capacity_GW, marginal_cost_EUR/MWh, bar_facecolor)
SEGMENTS = [
    ('Solar',          3.5,   1,  '#F9E547'),
    ('Wind',          14.0,   3,  '#7DCEA0'),
    ('Hydro',         16.5,   6,  '#5DADE2'),
    ('Nuclear',        6.9,  13,  '#BB8FCE'),
    ('Biomass /\nCHP', 4.0,  45,  '#F0A058'),
    ('Gas / Oil',      2.0,  88,  '#EC7063'),
]

# High-demand scenario for Sweden (GW)
# ~24 GW is a realistic high-demand winter hour; places "old supply" demand
# intersection in the Biomass/CHP zone → clearly visible merit order effect
DEMAND_GW = 24.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cumulative_widths(segments):
    """Return list of (x_start, x_end) for each segment."""
    xs = []
    cum = 0.0
    for _, cap, _, _ in segments:
        xs.append((cum, cum + cap))
        cum += cap
    return xs, cum          # (bounds list, total_capacity)


def _price_at_demand(segments, demand_gw):
    """Return the marginal cost of the segment that clears the demand."""
    cum = 0.0
    for _, cap, mc, _ in segments:
        cum += cap
        if demand_gw <= cum:
            return mc
    return segments[-1][2]  # demand exceeds total supply


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PLOT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bounds, total_cap = _cumulative_widths(SEGMENTS)

    new_price = _price_at_demand(SEGMENTS, DEMAND_GW)

    # "Old" supply = same stack minus Solar and Wind (zero-MC renewables)
    old_segments = [s for s in SEGMENTS if s[0] not in ('Solar', 'Wind')]
    old_price    = _price_at_demand(old_segments, DEMAND_GW)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6.5))

    # ── 1. Gray background bars (full height to marginal cost) ────────────────
    for (x0, x1), (name, cap, mc, color) in zip(bounds, SEGMENTS):
        rect = Rectangle(
            (x0, 0), cap, mc,
            facecolor=color, edgecolor='white',
            linewidth=1.2, alpha=0.72, zorder=2,
        )
        ax.add_patch(rect)

    # ── 2. Green supply (step) curve ──────────────────────────────────────────
    sx, sy = [], []
    for (x0, x1), (_, _, mc, _) in zip(bounds, SEGMENTS):
        sx += [x0, x1]
        sy += [mc, mc]

    ax.plot(sx, sy, color='#27AE60', lw=2.8, zorder=5, solid_capstyle='round')

    # Vertical drops between steps
    for i in range(len(SEGMENTS) - 1):
        x_join = bounds[i][1]
        mc_left  = SEGMENTS[i][2]
        mc_right = SEGMENTS[i + 1][2]
        ax.plot([x_join, x_join], [mc_left, mc_right],
                color='#27AE60', lw=2.8, zorder=5)

    # ── 3. Segment labels ─────────────────────────────────────────────────────
    for (x0, x1), (name, cap, mc, _) in zip(bounds, SEGMENTS):
        cx = (x0 + x1) / 2
        if mc >= 40 and cap >= 3:       # tall bars wide enough → white centred text
            ax.text(cx, mc * 0.46, name,
                    ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white',
                    zorder=6, linespacing=1.35)
        elif mc >= 40 and cap < 3:      # tall but narrow (Gas/Oil) → vertical white text
            ax.text(cx, mc * 0.46, name,
                    ha='center', va='center', rotation=90,
                    fontsize=9, fontweight='bold', color='white',
                    zorder=6)
        elif mc >= 12:                  # medium height → white centred text
            ax.text(cx, mc * 0.50, name,
                    ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white',
                    zorder=6, linespacing=1.35)
        else:                           # short bars (Solar, Wind, Hydro) → label above bar
            ax.text(cx, mc + 2.0, name,
                    ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color='#1A3A5C',
                    zorder=6, linespacing=1.35)

    # ── 4. Demand vertical line ───────────────────────────────────────────────
    ax.axvline(DEMAND_GW, color='#555555', lw=1.6, ls='--', alpha=0.70, zorder=4)
    ax.text(
        DEMAND_GW + 0.3, 99,
        'High-Demand\nHour',
        color='#444444', fontsize=9, va='top', linespacing=1.4,
    )

    # ── 5. New price marker (demand meets supply WITH wind+solar) ─────────────
    ax.hlines(new_price, 0, DEMAND_GW,
              colors='#27AE60', lw=1.4, ls='--', alpha=0.75, zorder=4)
    ax.plot(DEMAND_GW, new_price, 'o',
            color='#27AE60', ms=11, zorder=7, markeredgecolor='white', markeredgewidth=1.8)
    ax.text(
        DEMAND_GW - 0.5, new_price + 1.5,
        'New Electricity\nPrice',
        ha='right', va='bottom', color='#27AE60',
        fontsize=10, fontweight='bold', linespacing=1.3,
    )

    # ── 6. Old price marker (demand meets supply WITHOUT wind+solar) ──────────
    ax.hlines(old_price, 0, DEMAND_GW,
              colors='#7F8C8D', lw=1.4, ls='--', alpha=0.75, zorder=4)
    ax.plot(DEMAND_GW, old_price, 'o',
            color='#7F8C8D', ms=11, zorder=7, markeredgecolor='white', markeredgewidth=1.8)
    ax.text(
        DEMAND_GW - 0.5, old_price + 1.5,
        'Electricity Price\n(without Wind & Solar, based on Biomass/CHP)',
        ha='right', va='bottom', color='#555555',
        fontsize=10, fontweight='bold', linespacing=1.3,
    )

    # ── 7. Merit Order Effect downward arrow + label ──────────────────────────
    # Placed in the Wind zone (x≈10), left of demand line — mirrors reference image
    arrow_x = 10.0
    ax.annotate(
        '',
        xy=(arrow_x, new_price + 1),
        xytext=(arrow_x, old_price - 1),
        arrowprops=dict(
            arrowstyle='->', color='#27AE60', lw=2.4,
            mutation_scale=18,
        ),
        zorder=6,
    )
    ax.text(
        arrow_x + 0.5, (new_price + old_price) / 2 + 4,
        'Merit Order\nEffect',
        ha='left', va='center',
        color='#1A7D3E', fontsize=10.5, fontweight='bold', linespacing=1.35,
    )

    # ── 8. "Zero marginal cost" bracket for Solar + Wind region ───────────────
    solar_wind_end = bounds[1][1]          # end of Wind segment
    ax.annotate(
        '',
        xy=(0.2, -1),
        xytext=(solar_wind_end - 0.2, -1),
        xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle='<->', color='#1A7D3E', lw=1.3),
        annotation_clip=False,
        zorder=6,
    )
    ax.text(
        solar_wind_end / 2, -4,
        'Marginal Cost ≈ 0\n(Wind & Solar)',
        ha='center', va='top',
        color='#1A7D3E', fontsize=8.5, style='italic', linespacing=1.3,
    )

    # ── 9. Axes, grid, and labels ─────────────────────────────────────────────
    ax.set_xlim(0, total_cap + 0.8)
    ax.set_ylim(-14, 105)
    ax.set_xlabel('Installed Capacity (GW)', fontsize=11)
    ax.set_ylabel('Marginal Cost (EUR/MWh)', fontsize=11)
    ax.set_title(
        'Merit Order Curve – Swedish Electricity Market\n'
        'Illustrative Supply Stack Ordered by Marginal Cost',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, alpha=0.22, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    # x-axis ticks at segment boundaries + demand line
    tick_positions = sorted(
        set([b[0] for b in bounds] + [bounds[-1][1]] + [DEMAND_GW])
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{t:.1f}' for t in tick_positions], fontsize=8.5, rotation=35)

    # ── 10. Source note ───────────────────────────────────────────────────────
    fig.text(
        0.12, 0.01,
        'Note: Illustrative figure. Capacity and marginal cost values are directionally correct '
        'approximations based on Swedish Energy Agency data and Nord Pool/ENTSO-E estimates. '
        'Demand set to 24 GW (~high-demand winter hour). Not derived from the primary dataset.',
        fontsize=7.5, color='#666666', style='italic',
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(OUTPUT_DIR, 'merit_order_sweden.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
