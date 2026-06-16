"""
RetailPulse — Sales Insights Analysis
Author : Mohit Mahajan | github.com/Mohit0702
Dataset: Multi-region retail sales data (2023–2024)
         1,000 transactions · 3 categories · 5 price tiers

Run:
    pip install pandas numpy matplotlib seaborn
    python retailpulse_analysis.py

Output: retailpulse_analysis.png
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE = "retailpulse_sales.csv"
OUT_FILE  = "retailpulse_analysis.png"

# same palette as the dashboard so the portfolio looks cohesive
PINE   = "#142420"
CARD   = "#1e332c"
CREAM  = "#f3ecde"
GOLD   = "#e8a33d"
SAGE   = "#84a696"
SLATE  = "#3a4f47"
MUTED  = "#a9b3ab"
WHITE  = "#f3ecde"

CAT_COLORS = {
    "Beauty":      GOLD,
    "Clothing":    SAGE,
    "Electronics": "#c9c3b4",
}
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
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
    "grid.linewidth"   : 0.5,
    "font.family"      : "DejaVu Sans",
})


# ── LOAD & VALIDATE ───────────────────────────────────────────────────────────
def load(path):
    print(f"[+] Loading {path} ...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[!] File not found: {path}\n    Make sure retailpulse_sales.csv is in the same folder.")

    required = {"year","month","category","quantity","unit_price","revenue"}
    if not required.issubset(df.columns):
        sys.exit(f"[!] Missing columns: {required - set(df.columns)}")

    df["month_name"] = df["month"].apply(lambda m: MONTHS[m-1])
    print(f"    {len(df):,} transactions | {df['year'].nunique()} year(s) | "
          f"{df['category'].nunique()} categories")
    return df


# ── ANALYSIS ──────────────────────────────────────────────────────────────────
def analyse(df):
    print("[+] Running analysis ...")

    # overall KPIs
    kpi = {
        "total_revenue" : df["revenue"].sum(),
        "total_units"   : df["quantity"].sum(),
        "total_orders"  : len(df),
        "avg_order"     : df["revenue"].mean(),
    }

    # category breakdown
    cat = (df.groupby("category")
             .agg(revenue=("revenue","sum"), units=("quantity","sum"), orders=("revenue","count"))
             .reset_index()
             .sort_values("revenue", ascending=False))
    cat["revenue_share"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(1)

    # monthly trend (2023 only — 2024 has just 1 day of data)
    monthly = (df[df["year"]==2023]
               .groupby("month")["revenue"]
               .sum()
               .reindex(range(1,13), fill_value=0)
               .reset_index())
    monthly.columns = ["month", "revenue"]
    monthly["month_name"] = monthly["month"].apply(lambda m: MONTHS[m-1])

    # price tier analysis
    tiers = (df.groupby("unit_price")
               .agg(revenue=("revenue","sum"), units=("quantity","sum"), orders=("revenue","count"))
               .reset_index()
               .sort_values("revenue", ascending=False))

    # category × price heatmap
    heatmap = (df.groupby(["category","unit_price"])["revenue"]
                 .sum()
                 .unstack(fill_value=0))

    # month × category stacked (2023)
    stacked = (df[df["year"]==2023]
               .groupby(["month","category"])["revenue"]
               .sum()
               .unstack(fill_value=0)
               .reindex(range(1,13), fill_value=0))

    # top finding: which month had the best sales?
    best_month = monthly.loc[monthly["revenue"].idxmax()]
    worst_month = monthly.loc[monthly["revenue"].idxmin()]
    print(f"    Best month  : {best_month['month_name']} — ₹{best_month['revenue']:,}")
    print(f"    Worst month : {worst_month['month_name']} — ₹{worst_month['revenue']:,}")
    print(f"    Top category: {cat.iloc[0]['category']} — ₹{cat.iloc[0]['revenue']:,} "
          f"({cat.iloc[0]['revenue_share']}% share)")

    return kpi, cat, monthly, tiers, heatmap, stacked


# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot(df, kpi, cat, monthly, tiers, heatmap, stacked):
    print("[+] Building charts ...")

    fig = plt.figure(figsize=(20, 16), facecolor=PINE)
    fig.suptitle("RetailPulse — Sales Insights Dashboard  ·  2023",
                 fontsize=17, fontweight="bold", color=GOLD, y=0.97)
    fig.text(0.5, 0.955,
             "Mohit Mahajan  ·  github.com/Mohit0702  ·  Python · Pandas · Matplotlib · Seaborn",
             ha="center", fontsize=9, color=MUTED, style="italic")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── KPI row (text boxes) ──────────────────────────────────────────────
    kpi_labels = [
        ("Total Revenue",  f"₹{kpi['total_revenue']:,.0f}"),
        ("Total Orders",   f"{kpi['total_orders']:,}"),
        ("Units Sold",     f"{kpi['total_units']:,}"),
        ("Avg Order Value",f"₹{kpi['avg_order']:,.0f}"),
    ]
    for i, (label, val) in enumerate(kpi_labels):
        ax = fig.add_subplot(gs[0, i] if i < 3 else None)
        if i == 3:
            break
        ax.set_facecolor(CARD)
        ax.text(0.5, 0.62, val, ha="center", va="center",
                fontsize=22, fontweight="bold", color=GOLD, transform=ax.transAxes)
        ax.text(0.5, 0.28, label, ha="center", va="center",
                fontsize=9, color=MUTED, transform=ax.transAxes, letter_spacing=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SLATE)

    # 4th KPI in the same row
    ax_kpi4 = fig.add_axes([0.68, 0.71, 0.28, 0.17])   # manual placement
    ax_kpi4.set_facecolor(CARD)
    ax_kpi4.text(0.5, 0.62, f"₹{kpi['avg_order']:,.0f}",
                 ha="center", va="center", fontsize=22, fontweight="bold",
                 color=GOLD, transform=ax_kpi4.transAxes)
    ax_kpi4.text(0.5, 0.28, "Avg Order Value",
                 ha="center", va="center", fontsize=9, color=MUTED,
                 transform=ax_kpi4.transAxes)
    ax_kpi4.set_xticks([]); ax_kpi4.set_yticks([])
    for spine in ax_kpi4.spines.values():
        spine.set_edgecolor(SLATE)

    # ── Monthly revenue trend ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.fill_between(monthly["month_name"], monthly["revenue"],
                     alpha=0.15, color=GOLD)
    ax1.plot(monthly["month_name"], monthly["revenue"],
             color=GOLD, lw=2.2, marker="o", ms=5)

    # annotate peak & trough — took me a while to get the offset right
    peak = monthly.loc[monthly["revenue"].idxmax()]
    trough = monthly.loc[monthly["revenue"].idxmin()]
    ax1.annotate(f"Peak\n₹{peak['revenue']:,}",
                 xy=(peak["month_name"], peak["revenue"]),
                 xytext=(peak["month_name"], peak["revenue"]+2500),
                 ha="center", fontsize=7.5, color=GOLD,
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8))
    ax1.annotate(f"Low\n₹{trough['revenue']:,}",
                 xy=(trough["month_name"], trough["revenue"]),
                 xytext=(trough["month_name"], trough["revenue"]-3500),
                 ha="center", fontsize=7.5, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

    ax1.set_title("Monthly Revenue Trend — 2023", color=CREAM, fontsize=11, pad=8)
    ax1.set_ylabel("Revenue (₹)"); ax1.grid(axis="y", alpha=0.4)
    ax1.tick_params(axis="x", rotation=30)

    # ── Category revenue bars ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    bar_colors = [CAT_COLORS[c] for c in cat["category"]]
    bars = ax2.barh(cat["category"], cat["revenue"],
                    color=bar_colors, edgecolor="none", height=0.5)
    ax2.set_title("Revenue by Category", color=CREAM, fontsize=11, pad=8)
    ax2.grid(axis="x", alpha=0.4)
    for bar, share in zip(bars, cat["revenue_share"]):
        ax2.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
                 f"{share}%", va="center", fontsize=8, color=MUTED)

    # ── Price tier revenue ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    tier_cols = [GOLD if i==0 else SAGE for i in range(len(tiers))]
    ax3.bar([f"₹{p}" for p in tiers["unit_price"]], tiers["revenue"],
            color=tier_cols, edgecolor="none")
    ax3.set_title("Revenue by Price Tier", color=CREAM, fontsize=11, pad=8)
    ax3.set_ylabel("Revenue (₹)"); ax3.grid(axis="y", alpha=0.4)

    # ── Heatmap: category × price ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    sns.heatmap(heatmap, ax=ax4, cmap="YlOrBr", linewidths=0.4,
                linecolor=PINE, annot=True, fmt=",", annot_kws={"size":8},
                cbar_kws={"shrink":0.7})
    ax4.set_title("Revenue Heatmap: Category × Price", color=CREAM, fontsize=11, pad=8)
    ax4.set_xlabel("Unit Price (₹)"); ax4.set_ylabel("")
    ax4.tick_params(colors=MUTED)

    # ── Stacked bar: monthly category split ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    bottom = np.zeros(12)
    for cat_name in ["Beauty", "Clothing", "Electronics"]:
        if cat_name in stacked.columns:
            vals = stacked[cat_name].values
            ax5.bar(range(12), vals, bottom=bottom,
                    color=CAT_COLORS[cat_name], edgecolor="none", label=cat_name)
            bottom += vals
    ax5.set_xticks(range(12))
    ax5.set_xticklabels([m[:1] for m in MONTHS], fontsize=8)
    ax5.set_title("Monthly Sales by Category", color=CREAM, fontsize=11, pad=8)
    ax5.legend(fontsize=7, loc="upper left")
    ax5.grid(axis="y", alpha=0.4)

    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=PINE, edgecolor="none")
    print(f"    Saved → {OUT_FILE}")
    plt.show()


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_summary(kpi, cat, monthly, tiers):
    best = monthly.loc[monthly["revenue"].idxmax()]
    worst = monthly.loc[monthly["revenue"].idxmin()]
    print("\n" + "="*55)
    print("  RETAILPULSE — KEY FINDINGS")
    print("="*55)
    print(f"  Total Revenue  : ₹{kpi['total_revenue']:,}")
    print(f"  Total Orders   : {kpi['total_orders']:,}")
    print(f"  Units Sold     : {kpi['total_units']:,}")
    print(f"  Avg Order Value: ₹{kpi['avg_order']:.0f}")
    print(f"  Top Category   : {cat.iloc[0]['category']} "
          f"({cat.iloc[0]['revenue_share']}% revenue share)")
    print(f"  Best Month     : {best['month_name']} — ₹{best['revenue']:,}")
    print(f"  Worst Month    : {worst['month_name']} — ₹{worst['revenue']:,}")
    print(f"  Best Price Tier: ₹{tiers.iloc[0]['unit_price']} "
          f"— ₹{tiers.iloc[0]['revenue']:,} revenue")
    print("="*55)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load(DATA_FILE)
    kpi, cat, monthly, tiers, heatmap, stacked = analyse(df)
    plot(df, kpi, cat, monthly, tiers, heatmap, stacked)
    print_summary(kpi, cat, monthly, tiers)
