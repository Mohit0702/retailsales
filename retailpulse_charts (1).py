"""
RetailPulse — Sales Insights Analysis
Author : Mohit Mahajan | github.com/Mohit0702
Dataset: retailpulse_sales.csv (1,000 transactions, 2023)

What this script does:
  - Cleans and analyses multi-region retail sales data
  - Generates 6 chart images used by the dashboard
  - All heavy lifting is here — HTML just displays the outputs

Run:
    pip install pandas numpy matplotlib seaborn
    python retailpulse_charts.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE  = "retailpulse_sales.csv"
OUTPUT_DIR = "charts"          # all PNGs go here

PINE  = "#142420"
CARD  = "#1e332c"
CREAM = "#f3ecde"
GOLD  = "#e8a33d"
SAGE  = "#84a696"
SLATE = "#3a4f47"
MUTED = "#a9b3ab"

CAT_COLORS = {"Beauty": GOLD, "Clothing": SAGE, "Electronics": "#c9c3b4"}
MONTHS     = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

plt.rcParams.update({
    "figure.facecolor" : PINE,
    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : SLATE,
    "axes.labelcolor"  : MUTED,
    "text.color"       : CREAM,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "grid.color"       : SLATE,
    "grid.linewidth"   : 0.6,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────
def load(path):
    print(f"[+] Loading {path} ...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[!] {path} not found. Place it in the same folder.")

    # 2024 has only 2 rows (partial) — drop so graphs are clean
    df = df[df["year"] == 2023].copy()
    df["month_name"] = df["month"].apply(lambda m: MONTHS[m - 1])
    df["revenue"]    = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    print(f"    {len(df):,} transactions loaded after cleaning")
    return df


# ── KPIs ──────────────────────────────────────────────────────────────────────
def compute_kpis(df):
    return {
        "total_revenue" : df["revenue"].sum(),
        "total_orders"  : len(df),
        "total_units"   : df["quantity"].sum(),
        "avg_order"     : df["revenue"].mean(),
        "top_category"  : df.groupby("category")["revenue"].sum().idxmax(),
        "best_month"    : MONTHS[df.groupby("month")["revenue"].sum().idxmax() - 1],
    }


# ── CHART 1 — monthly trend ───────────────────────────────────────────────────
def chart_monthly_trend(df):
    monthly = (df.groupby("month")["revenue"]
                 .sum()
                 .reindex(range(1, 13), fill_value=0))

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=PINE)
    ax.fill_between(range(12), monthly.values, alpha=0.18, color=GOLD)
    ax.plot(range(12), monthly.values, color=GOLD, lw=2.4,
            marker="o", ms=5.5, zorder=3)

    # annotate peak and trough
    peak_i  = monthly.values.argmax()
    trough_i = monthly.values.argmin()
    ax.annotate(f"Peak\n₹{monthly.values[peak_i]:,.0f}",
                xy=(peak_i, monthly.values[peak_i]),
                xytext=(peak_i, monthly.values[peak_i] + 3000),
                ha="center", fontsize=8, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.9))
    ax.annotate(f"Low\n₹{monthly.values[trough_i]:,.0f}",
                xy=(trough_i, monthly.values[trough_i]),
                xytext=(trough_i, monthly.values[trough_i] - 4200),
                ha="center", fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_ylabel("Revenue (₹)", fontsize=9)
    ax.set_title("Monthly Revenue Trend — 2023", color=CREAM, fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.35)
    ax.set_facecolor(CARD)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_monthly_trend.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_monthly_trend.png saved")


# ── CHART 2 — category revenue (horizontal bar) ───────────────────────────────
def chart_category_revenue(df):
    cat = (df.groupby("category")["revenue"]
             .sum()
             .sort_values(ascending=True))   # ascending so biggest is on top

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PINE)
    colors = [CAT_COLORS[c] for c in cat.index]
    bars = ax.barh(cat.index, cat.values, color=colors,
                   edgecolor="none", height=0.5)

    total = cat.sum()
    for bar, val in zip(bars, cat.values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"₹{val:,.0f}  ({val/total*100:.1f}%)",
                va="center", fontsize=8, color=MUTED)

    ax.set_title("Revenue by Category", color=CREAM, fontsize=12, pad=10)
    ax.set_xlabel("Revenue (₹)", fontsize=9)
    ax.grid(axis="x", alpha=0.35)
    ax.set_facecolor(CARD)
    ax.set_xlim(0, cat.max() * 1.35)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_category_revenue.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_category_revenue.png saved")


# ── CHART 3 — price tier revenue (bar) ───────────────────────────────────────
def chart_price_tiers(df):
    tiers = (df.groupby("unit_price")["revenue"]
               .sum()
               .sort_index())

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PINE)
    bar_colors = [GOLD if v == tiers.max() else SAGE for v in tiers.values]
    bars = ax.bar([f"₹{p}" for p in tiers.index], tiers.values,
                  color=bar_colors, edgecolor="none", width=0.55)

    for bar, val in zip(bars, tiers.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                f"₹{val:,.0f}", ha="center", fontsize=8, color=MUTED)

    ax.set_title("Revenue by Price Tier", color=CREAM, fontsize=12, pad=10)
    ax.set_ylabel("Revenue (₹)", fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    ax.set_facecolor(CARD)

    # note: gold bar = best performing tier
    gold_patch = mpatches.Patch(color=GOLD, label="Best performing tier")
    ax.legend(handles=[gold_patch], fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_price_tiers.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_price_tiers.png saved")


# ── CHART 4 — heatmap category × price ───────────────────────────────────────
def chart_heatmap(df):
    pivot = (df.groupby(["category", "unit_price"])["revenue"]
               .sum()
               .unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=PINE)
    sns.heatmap(pivot, ax=ax, cmap="YlOrBr", linewidths=0.5,
                linecolor=PINE, annot=True, fmt=",",
                annot_kws={"size": 8, "color": PINE},
                cbar_kws={"shrink": 0.8, "label": "Revenue (₹)"})

    ax.set_title("Revenue Heatmap: Category × Price Tier",
                 color=CREAM, fontsize=12, pad=10)
    ax.set_xlabel("Unit Price (₹)", fontsize=9)
    ax.set_ylabel("", fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_heatmap.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_heatmap.png saved")


# ── CHART 5 — stacked monthly by category ─────────────────────────────────────
def chart_stacked_monthly(df):
    stacked = (df.groupby(["month", "category"])["revenue"]
                 .sum()
                 .unstack(fill_value=0)
                 .reindex(range(1, 13), fill_value=0))

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=PINE)
    bottom = np.zeros(12)
    for cat_name in ["Beauty", "Clothing", "Electronics"]:
        if cat_name in stacked.columns:
            vals = stacked[cat_name].values
            ax.bar(range(12), vals, bottom=bottom,
                   color=CAT_COLORS[cat_name], edgecolor="none",
                   label=cat_name, width=0.65)
            bottom += vals

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_ylabel("Revenue (₹)", fontsize=9)
    ax.set_title("Monthly Revenue Split by Category — 2023",
                 color=CREAM, fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper left",
              facecolor=CARD, edgecolor=SLATE, labelcolor=CREAM)
    ax.grid(axis="y", alpha=0.35)
    ax.set_facecolor(CARD)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_stacked_monthly.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_stacked_monthly.png saved")


# ── CHART 6 — units sold by category (donut) ──────────────────────────────────
def chart_units_donut(df):
    units = df.groupby("category")["quantity"].sum()

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=PINE)
    wedges, texts, autotexts = ax.pie(
        units.values,
        labels=units.index,
        autopct="%1.1f%%",
        colors=[CAT_COLORS[c] for c in units.index],
        startangle=140,
        wedgeprops=dict(width=0.55, edgecolor=PINE, linewidth=2),
        pctdistance=0.78,
    )
    for t in texts:
        t.set_color(CREAM); t.set_fontsize(9)
    for at in autotexts:
        at.set_color(PINE); at.set_fontsize(8); at.set_fontweight("bold")

    ax.set_title("Units Sold by Category", color=CREAM, fontsize=12, pad=10)
    ax.set_facecolor(PINE)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/chart_units_donut.png",
                dpi=150, bbox_inches="tight", facecolor=PINE)
    plt.close()
    print("    chart_units_donut.png saved")


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
def print_summary(kpis):
    print("\n" + "="*50)
    print("  RETAILPULSE — KEY FINDINGS (2023)")
    print("="*50)
    print(f"  Total Revenue  : ₹{kpis['total_revenue']:,.0f}")
    print(f"  Total Orders   : {kpis['total_orders']:,}")
    print(f"  Units Sold     : {kpis['total_units']:,}")
    print(f"  Avg Order Value: ₹{kpis['avg_order']:.0f}")
    print(f"  Top Category   : {kpis['top_category']}")
    print(f"  Best Month     : {kpis['best_month']}")
    print("="*50)
    print(f"\n  All charts saved to ./{OUTPUT_DIR}/")
    print("  Open retailpulse_dashboard.html to view the full dashboard.\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df   = load(DATA_FILE)
    kpis = compute_kpis(df)

    print("[+] Generating charts ...")
    chart_monthly_trend(df)
    chart_category_revenue(df)
    chart_price_tiers(df)
    chart_heatmap(df)
    chart_stacked_monthly(df)
    chart_units_donut(df)

    print_summary(kpis)
