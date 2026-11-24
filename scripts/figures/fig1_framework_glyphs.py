"""Small vector glyphs for the Figure 1 framework (millimetre coordinates, 1 unit = 1 mm).

Complements drawn_icons.py with the pictograms that the framework figure needs and the
overview figure did not: a time-course icon, a human cohort icon, and the four finding glyphs.
Each glyph is centred on (x, y) and returns nothing; text is added by the caller so that the
layout check sees every label.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

import drawn_icons as di
from drawn_icons import BLUE, GREEN, INK, ORANGE, PURPLE, RED, tint

BROWN = "#937860"          # tolerance-associated (as in Figure 5)
GREY_BAR = "#C9CDD3"


def time_course(ax, x, y, z=3):
    """Mouse beside a time axis with sampling points; about 25 x 9 mm."""
    di.mouse(ax, x - 5.6, y + 0.9, s=0.62, z=z)
    x0, x1, yb = x + 2.6, x + 12.2, y - 1.6
    di.line(ax, [(x0, yb), (x1 - 0.8, yb)], c=INK, lw=0.5, z=z)
    di.poly(ax, [(x1, yb), (x1 - 1.1, yb + 0.6), (x1 - 1.1, yb - 0.6)], INK, ec="none", z=z)
    ts = np.linspace(x0 + 0.6, x1 - 2.0, 4)
    for k, t in enumerate(ts):
        di.line(ax, [(t, yb - 0.5), (t, yb + 0.5)], c=INK, lw=0.4, z=z)
        h = 1.2 + 2.6 * np.sin(np.pi * (k + 0.6) / 4.6)
        di.dot(ax, t, yb + 1.2 + h, 0.55, tint(RED, 0.15 + 0.15 * k), ec=INK, lw=0.25, z=z + 0.1)
    return x - 12.4, x1


def human(ax, x, y, z=3):
    """Person silhouette, blood tube and a monocyte; about 22 x 9 mm."""
    px = x - 6.0
    body = di.blob_xy(px, y - 3.0, 3.4, sx=1.15, sy=0.85, seed=4)
    body = body[body[:, 1] >= y - 4.2]
    di.poly(ax, np.vstack([body, [(px + 3.6, y - 4.2), (px - 3.6, y - 4.2)]]), "#DADDE3", z=z)
    di.dot(ax, px, y + 2.0, 2.0, "#DADDE3", ec=INK, z=z + 0.1)
    tx = x + 2.4
    di.add(ax, FancyBboxPatch((tx - 1.0, y - 4.2), 2.0, 7.6, boxstyle="round,pad=0,rounding_size=0.9",
                              fc="#F4F6F9", ec=INK, lw=di.LW, zorder=z))
    di.add(ax, FancyBboxPatch((tx - 0.8, y - 4.0), 1.6, 4.6, boxstyle="round,pad=0,rounding_size=0.7",
                              fc=tint(RED, 0.1), ec="none", zorder=z + 0.1))
    di.add(ax, Rectangle((tx - 1.25, y + 3.4), 2.5, 1.0, fc=tint(PURPLE, 0.3), ec=INK, lw=di.LW,
                         zorder=z + 0.2))
    mx = x + 8.0
    di.poly(ax, di.blob_xy(mx, y + 0.2, 2.3, wobble=0.06, ripple=(17, 0.05), seed=2),
            tint(BROWN, 0.55), z=z)
    di.nucleus(ax, mx + 0.2, y + 0.1, 1.05, tint(BROWN, 0.05), ang=200, z=z + 0.2)
    return px - 4.0, mx + 2.5


def mini_bars(ax, x, yb, h_sham, h_clp, colour, z=3, w=1.9, gap=0.6):
    """Two bars (sham grey, CLP coloured) on a short baseline."""
    di.add(ax, Rectangle((x - gap / 2 - w, yb), w, h_sham, fc=GREY_BAR, ec=INK, lw=0.3, zorder=z))
    di.add(ax, Rectangle((x + gap / 2, yb), w, h_clp, fc=colour, ec=INK, lw=0.3, zorder=z))
    di.line(ax, [(x - gap / 2 - w - 1.0, yb), (x + gap / 2 + w + 1.0, yb)], c=INK, lw=0.5, z=z + 0.1)


def tolerance_curve(ax, x0, x1, yb, h, x8, z=3):
    """Brown sigmoid rising before x8 and staying high, with a dotted baseline."""
    xs = np.linspace(x0, x1, 200)
    ys = yb + h / (1 + np.exp(-(xs - (x8 - 3.0)) / 1.1))
    di.poly(ax, np.vstack([np.column_stack([xs, ys]), [(x1, yb), (x0, yb)]]), tint(BROWN, 0.75),
            ec="none", z=z)
    di.line(ax, np.column_stack([xs, ys]), c=BROWN, lw=1.0, z=z + 0.1)
    di.line(ax, [(x0, yb), (x1 + 0.6, yb)], c=INK, lw=0.5, z=z + 0.2)
    return xs, ys


def overlap_circles(ax, x, y, r=4.6, d=2.6, z=3):
    """Disulfide HMGB1 (orange) and LPS (red) responses as two nearly coincident discs."""
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x - d, y), r, fc=ORANGE, ec="none", alpha=0.55, zorder=z))
    ax.add_patch(Circle((x + d, y), r, fc=RED, ec="none", alpha=0.45, zorder=z + 0.1))
    ax.add_patch(Circle((x - d, y), r, fc="none", ec=tint(ORANGE, -0.3), lw=0.5, zorder=z + 0.2))
    ax.add_patch(Circle((x + d, y), r, fc="none", ec=tint(RED, -0.3), lw=0.5, zorder=z + 0.2))


def up_arrow(ax, x, y0, y1, colour, z=3):
    ax.add_patch(FancyArrow(x, y0, 0, y1 - y0, width=0.9, head_width=2.2, head_length=1.6,
                            length_includes_head=True, fc=colour, ec="none", zorder=z))


__all__ = ["BROWN", "BLUE", "GREEN", "time_course", "human", "mini_bars", "tolerance_curve",
           "overlap_circles", "up_arrow"]
