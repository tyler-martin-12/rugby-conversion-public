"""
Expected points (try + conversion) vs. lateral try location —
visualised as a filled hill curve rising above the try line on an
abstract top-down rugby pitch.

Coordinate mapping (y-axis):
  try line  (y = 0)  ↔  5.0 pts  (try alone)
  top       (y = 30) ↔  7.0 pts  (try + certain conversion)

The curve peaks upward at centre and dips toward the wings.
Data: 13,338 conversion attempts, goal_kicking_data.csv
(WhartonSABI / Quarrie & Hopkins, 2000–2012).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------------
# Load & fit
# ---------------------------------------------------------------------------
df   = pd.read_csv("goal_kicking_data.csv")
conv = df[df["Type"] == 2].copy()
conv["lateral_m"] = (conv["X1 Metres"] - 35).abs()

bins = np.arange(0, 37.5, 2.5)
conv["bin"] = pd.cut(conv["lateral_m"], bins=bins, include_lowest=True)
agg = (
    conv.groupby("bin", observed=True)
        .agg(attempts=("Quality", "count"),
             made=("Quality", lambda x: (x == 1).sum()))
)
agg["prob"]    = agg["made"] / agg["attempts"]
agg["bin_mid"] = [i.mid for i in agg.index]

bin_mid = agg["bin_mid"].values
bin_p   = agg["prob"].values
n       = agg["attempts"].values

spl = UnivariateSpline(bin_mid, bin_p, w=np.sqrt(n), k=3, s=len(bin_mid))

HALF_WIDTH = 35.0
lat    = np.linspace(-HALF_WIDTH, HALF_WIDTH, 700)
p_conv = np.clip(spl(np.abs(lat)), 0, 1)
ep     = 5 + 2 * p_conv

# ---------------------------------------------------------------------------
# Map EP → y  (5 pts = 0, 7 pts = Y_TOP)
# ---------------------------------------------------------------------------
Y_TOP   = 30.0          # full chart height above try line
Y_BOT   = -4.0          # small in-goal strip below

def ep_to_y(e):
    return (e - 5.0) * (Y_TOP / 2.0)   # 2 pt range → Y_TOP metres

curve_y = ep_to_y(ep)

# EP tick positions
ep_ticks = [5.0, 5.5, 6.0, 6.5, 7.0]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
POST_SEP = 5.6 / 2

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#2d6a2d")
ax.set_facecolor("#2d6a2d")

# ── Grass stripes (vertical bands across the chart height) ───────────────────
stripe_colors = ["#2d6a2d", "#316e31"]
for i, y_start in enumerate(np.arange(Y_BOT, Y_TOP + 5, 5)):
    ax.add_patch(mpatches.Rectangle(
        (-HALF_WIDTH, y_start), 2 * HALF_WIDTH, 5,
        color=stripe_colors[int(i) % 2], zorder=0
    ))

# In-goal darker tint below try line
ax.add_patch(mpatches.Rectangle(
    (-HALF_WIDTH, Y_BOT), 2 * HALF_WIDTH, abs(Y_BOT),
    color="#1a451a", zorder=1, alpha=0.7
))

# ── EP fill: try line up to curve ─────────────────────────────────────────────
ax.fill_between(lat, 0, curve_y, color="#f5c518", alpha=0.55, zorder=2, linewidth=0)

# ── EP curve ──────────────────────────────────────────────────────────────────
ax.plot(lat, curve_y, color="white", lw=2.5, zorder=5)

# ── Pitch lines ───────────────────────────────────────────────────────────────
ax.axvline(-HALF_WIDTH, color="white", lw=2,   zorder=3)   # left touchline
ax.axvline( HALF_WIDTH, color="white", lw=2,   zorder=3)   # right touchline
ax.axhline(0,           color="white", lw=3,   zorder=4)   # try line
ax.axhline(Y_BOT,       color="white", lw=1.2, zorder=3, alpha=0.6)   # dead-ball

# ── Goal posts ────────────────────────────────────────────────────────────────
post_kw = dict(color="#FFD700", lw=3.5, zorder=6)
ax.plot([-POST_SEP, -POST_SEP], [Y_BOT, 0], **post_kw)
ax.plot([ POST_SEP,  POST_SEP], [Y_BOT, 0], **post_kw)
ax.plot([-POST_SEP,  POST_SEP], [0,     0], **post_kw)

# ── Dashed EP gridlines ───────────────────────────────────────────────────────
for pts in ep_ticks:
    y = ep_to_y(pts)
    if pts > 5.0:   # try line already drawn solid
        ax.axhline(y, color="white", lw=0.8, ls="--", alpha=0.35, zorder=2)

# ── Y-axis EP scale (right edge) ─────────────────────────────────────────────
scale_x = HALF_WIDTH + 1.5
for pts in ep_ticks:
    y = ep_to_y(pts)
    ax.plot([scale_x, scale_x + 0.8], [y, y], color="white", lw=1.2, zorder=7)
    ax.text(scale_x + 1.2, y, f"{pts:.1f}",
            color="white", fontsize=9, va="center", zorder=7)

ax.text(scale_x + 2.8, ep_to_y(6.0), "Expected\npoints",
        color="white", fontsize=9, ha="center", va="center",
        rotation=90, zorder=7)

# ── Labels ────────────────────────────────────────────────────────────────────
ax.text(-HALF_WIDTH + 1, 0.6,   "TRY LINE", color="white", fontsize=9,  zorder=8)
ax.text(-HALF_WIDTH + 1, Y_BOT + 0.4, "IN-GOAL", color="white", fontsize=8,
        alpha=0.65, zorder=8)

# Key value annotations
ep_centre = float(5 + 2 * np.clip(spl(0), 0, 1))
ep_tl     = float(ep[0])
pct_centre = int(round(np.clip(spl(0), 0, 1) * 100))
pct_tl     = int(round(p_conv[0] * 100))
ax.annotate(f"{ep_centre:.2f} pts  ({pct_centre}%)",
            xy=(0, ep_to_y(ep_centre)),
            xytext=(9, ep_to_y(ep_centre) + 1.5),
            color="white", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
ax.annotate(f"{ep_tl:.2f} pts  ({pct_tl}%)",
            xy=(-HALF_WIDTH, ep_to_y(ep_tl)),
            xytext=(-HALF_WIDTH + 5, ep_to_y(ep_tl) + 2),
            color="white", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

# ── Title & axes ──────────────────────────────────────────────────────────────
ax.set_title(
    "Expected Points by Try Location",
    fontsize=13, color="white", pad=10
)

# Source in bottom-right corner
ax.text(HALF_WIDTH + 7.8, Y_BOT - 0.3,
        "WhartonSABI / Quarrie & Hopkins · 13,338 conversion attempts · 2000–2012",
        color="white", fontsize=6.5, alpha=0.55, ha="right", va="top", zorder=9)

ax.set_xlim(-HALF_WIDTH - 2, HALF_WIDTH + 8)
ax.set_ylim(Y_BOT - 0.5, Y_TOP + 1.5)
ax.set_aspect("equal")

ax.set_xticks([-35, -25, -15, -5, 0, 5, 15, 25, 35])
ax.set_xticklabels(
    ["TL\n−35", "−25", "−15", "−5", "0\n(posts)", "5", "15", "25", "TL\n35"],
    color="white", fontsize=8.5
)
ax.tick_params(axis="x", colors="white")
ax.yaxis.set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlabel("Lateral position along try line  (m from goal posts)", fontsize=11,
              color="white", labelpad=8)

fig.tight_layout()
out = "expected_points_pitch.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
