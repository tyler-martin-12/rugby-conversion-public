"""
Naive mental model: conversion probability vs. lateral try location.
Model: >90% (flat) in the middle third of the field, then linearly
decreasing to ~30% at the touchlines.

Plotted in the same pitch style as expected_points_pitch.py.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Naive model
# ---------------------------------------------------------------------------
HALF_WIDTH   = 35.0
MIDDLE_THIRD = HALF_WIDTH / 3          # ≈ 11.67 m — edge of middle third
P_CENTRE     = 0.92                    # flat rate inside middle third
P_EDGE       = 0.30                    # rate at touchline

lat    = np.linspace(-HALF_WIDTH, HALF_WIDTH, 700)
abs_l  = np.abs(lat)

# Piecewise: flat in middle third, then linear ramp out to touchline
p_naive = np.where(
    abs_l <= MIDDLE_THIRD,
    P_CENTRE,
    P_CENTRE + (abs_l - MIDDLE_THIRD) / (HALF_WIDTH - MIDDLE_THIRD) * (P_EDGE - P_CENTRE)
)
ep = 5 + 2 * p_naive

# ---------------------------------------------------------------------------
# Map EP → y  (5 pts = 0, 7 pts = Y_TOP)  — same scale as existing plot
# ---------------------------------------------------------------------------
Y_TOP = 30.0
Y_BOT = -4.0

def ep_to_y(e):
    return (e - 5.0) * (Y_TOP / 2.0)

curve_y  = ep_to_y(ep)
ep_ticks = [5.0, 5.5, 6.0, 6.5, 7.0]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
POST_SEP = 5.6 / 2

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#2d6a2d")
ax.set_facecolor("#2d6a2d")

# Grass stripes
stripe_colors = ["#2d6a2d", "#316e31"]
for i, y_start in enumerate(np.arange(Y_BOT, Y_TOP + 5, 5)):
    ax.add_patch(mpatches.Rectangle(
        (-HALF_WIDTH, y_start), 2 * HALF_WIDTH, 5,
        color=stripe_colors[int(i) % 2], zorder=0
    ))

# In-goal darker tint
ax.add_patch(mpatches.Rectangle(
    (-HALF_WIDTH, Y_BOT), 2 * HALF_WIDTH, abs(Y_BOT),
    color="#1a451a", zorder=1, alpha=0.7
))

# Middle-third shading to make the flat zone visible
ax.add_patch(mpatches.Rectangle(
    (-MIDDLE_THIRD, 0), 2 * MIDDLE_THIRD, ep_to_y(5 + 2 * P_CENTRE),
    color="#ffffff", alpha=0.06, zorder=2
))
ax.axvline(-MIDDLE_THIRD, color="white", lw=1, ls=":", alpha=0.45, zorder=4)
ax.axvline( MIDDLE_THIRD, color="white", lw=1, ls=":", alpha=0.45, zorder=4)
ax.text(0, ep_to_y(5 + 2 * P_CENTRE) + 0.7, "middle third",
        color="white", fontsize=7.5, ha="center", alpha=0.55, zorder=8)

# EP fill and curve
ax.fill_between(lat, 0, curve_y, color="#f5c518", alpha=0.55, zorder=2, linewidth=0)
ax.plot(lat, curve_y, color="white", lw=2.5, zorder=5)

# Pitch lines
ax.axvline(-HALF_WIDTH, color="white", lw=2,   zorder=3)
ax.axvline( HALF_WIDTH, color="white", lw=2,   zorder=3)
ax.axhline(0,           color="white", lw=3,   zorder=4)
ax.axhline(Y_BOT,       color="white", lw=1.2, zorder=3, alpha=0.6)

# Goal posts
post_kw = dict(color="#FFD700", lw=3.5, zorder=6)
ax.plot([-POST_SEP, -POST_SEP], [Y_BOT, 0], **post_kw)
ax.plot([ POST_SEP,  POST_SEP], [Y_BOT, 0], **post_kw)
ax.plot([-POST_SEP,  POST_SEP], [0,     0], **post_kw)

# Dashed EP gridlines
for pts in ep_ticks:
    y = ep_to_y(pts)
    if pts > 5.0:
        ax.axhline(y, color="white", lw=0.8, ls="--", alpha=0.35, zorder=2)

# Y-axis EP scale (right edge)
scale_x = HALF_WIDTH + 1.5
for pts in ep_ticks:
    y = ep_to_y(pts)
    ax.plot([scale_x, scale_x + 0.8], [y, y], color="white", lw=1.2, zorder=7)
    ax.text(scale_x + 1.2, y, f"{pts:.1f}",
            color="white", fontsize=9, va="center", zorder=7)
ax.text(scale_x + 2.8, ep_to_y(6.0), "Expected\npoints",
        color="white", fontsize=9, ha="center", va="center",
        rotation=90, zorder=7)

# Labels
ax.text(-HALF_WIDTH + 1, 0.6,   "TRY LINE", color="white", fontsize=9,  zorder=8)
ax.text(-HALF_WIDTH + 1, Y_BOT + 0.4, "IN-GOAL", color="white", fontsize=8,
        alpha=0.65, zorder=8)

# Key value annotations
ep_centre = float(5 + 2 * P_CENTRE)
ep_edge   = float(5 + 2 * P_EDGE)
ax.annotate(f"{ep_centre:.2f} pts  ({int(P_CENTRE*100)}%)",
            xy=(0, ep_to_y(ep_centre)),
            xytext=(9, ep_to_y(ep_centre) + 1.5),
            color="white", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
ax.annotate(f"{ep_edge:.2f} pts  ({int(P_EDGE*100)}%)",
            xy=(-HALF_WIDTH, ep_to_y(ep_edge)),
            xytext=(-HALF_WIDTH + 5, ep_to_y(ep_edge) + 2),
            color="white", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

# Title & subtitle
ax.set_title(
    "Expected Points by Try Location  —  Naive Mental Model",
    fontsize=13, color="white", pad=10
)
ax.text(0, Y_TOP + 0.3,
        f"Assumed: {int(P_CENTRE*100)}% in middle third  →  linear drop to {int(P_EDGE*100)}% at touchline",
        color="white", fontsize=8.5, ha="center", alpha=0.75, zorder=9)

# Source note
ax.text(HALF_WIDTH + 7.8, Y_BOT - 0.3,
        "Illustrative — not from data",
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
out = "naive_model_pitch.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
