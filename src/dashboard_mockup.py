"""
dashboard_mockup.py
-------------------
Render a Power-BI-style dashboard mockup (PNG) from the scored data in
data/powerbi_churn_ready.csv. This mirrors the layout described in
powerbi/POWERBI_GUIDE.md and acts as:

  * a design reference for building the real Power BI report, and
  * a placeholder for images/powerbi_dashboard_mockup.png in the README until
    you drop in your own Power BI Desktop screenshot.

    python src/dashboard_mockup.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = PROJECT_ROOT / "data" / "powerbi_churn_ready.csv"
OUT = PROJECT_ROOT / "images" / "powerbi_dashboard_mockup.png"

BG = "#F4F6F9"
CARD = "#FFFFFF"
ACCENT = "#3D5A80"
CHURN = "#E4572E"
KEEP = "#4C9F70"


def kpi(ax, title, value):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.22, title, ha="center", va="center",
            fontsize=10, color="#555", transform=ax.transAxes)


def main():
    df = pd.read_csv(CSV)
    total = len(df)
    churn_rate = df["Actual_Churn"].mean()
    high_risk = (df["Risk_Band"] == "High").sum()
    avg_prob = df["Churn_Probability"].mean()
    avg_tenure = df["Tenure"].mean()

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.5, 1, 1],
                  hspace=0.45, wspace=0.25,
                  left=0.05, right=0.96, top=0.90, bottom=0.07)

    fig.suptitle("Customer Churn Dashboard", x=0.05, ha="left",
                 fontsize=20, fontweight="bold", color="#222")
    fig.text(0.05, 0.925, "Scored on unseen customers  •  model: Gradient Boosting",
             ha="left", fontsize=10, color="#777")

    # KPI row
    kpi(fig.add_subplot(gs[0, 0]), "Total Customers", f"{total:,}")
    kpi(fig.add_subplot(gs[0, 1]), "Churn Rate", f"{churn_rate:.1%}")
    kpi(fig.add_subplot(gs[0, 2]), "High-Risk Customers", f"{high_risk:,}")
    kpi(fig.add_subplot(gs[0, 3]), "Avg Churn Probability", f"{avg_prob:.1%}")

    # Churn rate by contract length
    ax1 = fig.add_subplot(gs[1, 0:2]); ax1.set_facecolor(CARD)
    rate = df.groupby("Contract Length")["Actual_Churn"].mean().sort_values(ascending=False)
    ax1.bar(rate.index, rate.values, color=ACCENT)
    for i, v in enumerate(rate.values):
        ax1.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Churn Rate by Contract Length", fontsize=11, loc="left")
    ax1.set_ylim(0, max(rate.values) * 1.25)

    # Risk band distribution
    ax2 = fig.add_subplot(gs[1, 2]); ax2.set_facecolor(CARD)
    band = df["Risk_Band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    ax2.pie(band.values, labels=band.index, autopct="%1.0f%%",
            colors=[KEEP, "#E6B800", CHURN], startangle=90,
            wedgeprops={"width": 0.45})
    ax2.set_title("Customers by Risk Band", fontsize=11)

    # Churn by support calls
    ax3 = fig.add_subplot(gs[1, 3]); ax3.set_facecolor(CARD)
    sc = df.groupby("Support Calls")["Actual_Churn"].mean()
    ax3.plot(sc.index, sc.values, marker="o", color=CHURN)
    ax3.set_title("Churn by Support Calls", fontsize=11)
    ax3.set_ylim(0, 1)

    # Churn probability distribution
    ax4 = fig.add_subplot(gs[2, 0:2]); ax4.set_facecolor(CARD)
    ax4.hist(df["Churn_Probability"], bins=25, color=ACCENT, alpha=0.85)
    ax4.set_title("Distribution of Predicted Churn Probability", fontsize=11, loc="left")
    ax4.set_xlabel("Churn Probability")

    # Churn rate by subscription type
    ax5 = fig.add_subplot(gs[2, 2:4]); ax5.set_facecolor(CARD)
    sub = df.groupby("Subscription Type")["Actual_Churn"].mean().sort_values(ascending=False)
    ax5.barh(sub.index, sub.values, color=ACCENT)
    for i, v in enumerate(sub.values):
        ax5.text(v, i, f" {v:.0%}", va="center", fontsize=9)
    ax5.set_title("Churn Rate by Subscription Type", fontsize=11, loc="left")
    ax5.set_xlim(0, max(sub.values) * 1.25)

    fig.savefig(OUT, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
