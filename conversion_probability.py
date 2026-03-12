"""
Rugby conversion probability vs. lateral distance from goal posts.

Data
----
goal_kicking_data.csv — 24,682 goal-kick events from WhartonSABI/rugby-ep
(sourced from Quarrie & Hopkins data, covering international / top-tier matches
2000-2012).  Encoding: Type=2 → conversion attempt; Quality=1 → made,
Quality=2 → missed.  X1 Metres is the lateral position of the kick (0–70 m
across the pitch width, posts at X=35 m).

Each record reflects the kicker's own choice of how far back to stand, so the
empirical success rate already embeds the optimal-distance strategy.

Aggregation: 2.5 m bins on lateral offset |X1 - 35|, weighted by bin count.
A GAM-style smoothing spline is fit through the bin midpoints.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
df   = pd.read_csv("goal_kicking_data.csv")
conv = df[df["Type"] == 2].copy()
conv["lateral_m"] = (conv["X1 Metres"] - 35).abs()

bins = np.arange(0, 37.5, 2.5)
conv["bin"] = pd.cut(conv["lateral_m"], bins=bins, include_lowest=True)

agg = (
    conv.groupby("bin", observed=True)
        .agg(attempts=("Quality", "count"), made=("Quality", lambda x: (x == 1).sum()))
)
agg["prob"]    = agg["made"] / agg["attempts"]
agg["bin_mid"] = [i.mid for i in agg.index]

bin_mid = agg["bin_mid"].values
bin_p   = agg["prob"].values
n       = agg["attempts"].values        # use attempt counts as weights

# ---------------------------------------------------------------------------
# Fit a weighted smoothing spline (s controls smoothness)
# ---------------------------------------------------------------------------
spl    = UnivariateSpline(bin_mid, bin_p, w=np.sqrt(n), k=3, s=len(bin_mid))
x_fine = np.linspace(0, 35, 500)
y_fit  = np.clip(spl(x_fine), 0, 1)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Raw empirical bars (with 95% Wilson CI)
z = 1.96
for xm, p, ni in zip(bin_mid, bin_p, n):
    lo = (p + z**2/(2*ni) - z * np.sqrt(p*(1-p)/ni + z**2/(4*ni**2))) / (1 + z**2/ni)
    hi = (p + z**2/(2*ni) + z * np.sqrt(p*(1-p)/ni + z**2/(4*ni**2))) / (1 + z**2/ni)
    ax.plot([xm, xm], [lo*100, hi*100], color="#1a6faf", lw=1.2, alpha=0.5)

ax.scatter(bin_mid, bin_p * 100, s=n / 8, color="#1a6faf", zorder=5,
           label="Empirical (2.5 m bins, size ∝ attempts)", alpha=0.85)
ax.plot(x_fine, y_fit * 100, color="#e05a00", lw=2.5, zorder=6,
        label="Smoothing spline fit")

# Annotate each bin
for xm, p, ni in zip(bin_mid, bin_p, n):
    ax.annotate(f"{p*100:.0f}%\n(n={ni})",
                xy=(xm, p*100), xytext=(0, 9), textcoords="offset points",
                fontsize=6.5, ha="center", color="#333333")

ax.set_xlabel("Lateral distance from goal posts  (m)\n0 = directly under posts, 35 = touchline",
              fontsize=12)
ax.set_ylabel("Conversion success probability  (%)", fontsize=12)
ax.set_title(
    "Rugby Union — Empirical Conversion Probability by Try Location\n"
    "13,338 conversion attempts · international & top-tier matches · 2000–2012"
    "\n(WhartonSABI / Quarrie & Hopkins dataset)",
    fontsize=12, pad=10
)

ax.set_xlim(-1, 37)
ax.set_ylim(30, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.grid(axis="both", linestyle="--", alpha=0.35)
ax.legend(fontsize=10)

for xv, label in [(0, "Under posts"), (35, "Touchline")]:
    ax.axvline(xv, color="gray", lw=0.8, ls=":")
    ax.text(xv + (0.5 if xv == 0 else -0.5), 103, label,
            ha="left" if xv == 0 else "right", fontsize=8, color="gray")

fig.tight_layout()
out = "conversion_probability.png"
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
