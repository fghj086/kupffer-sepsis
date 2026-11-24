#!/usr/bin/env python
"""Figure 1 - study overview, drawn entirely in code (matplotlib patches and paths).

No downloaded or raster images are used. Layout, left to right: stimuli (PAMP, DAMP, cytokines)
-> zoom into a septic liver sinusoid (resident Kupffer cell, recruited macrophage) -> five data
layers -> source attribution. The build fails if any label touches drawn content or another label.
Outputs: figures/Fig1_drawn.pdf, figures/Fig1_drawn.svg, figures/Fig1_drawn.png (300 dpi)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon  # noqa: E402
from scipy.spatial import Voronoi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drawn_icons as di  # noqa: E402
from drawn_icons import BLUE, GREEN, INK, LILAC, ORANGE, PEACH, PURPLE, RED, tint  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
W, H = 180.0, 94.5                        # mm
Y0 = 3.0                                  # lower y limit (mm)
FS = 7                                    # pt
TEXT_GREY = "#333333"
C = np.array([72.0, 44.0])                # sinusoid zoom centre
R = 31.0                                  # zoom radius
BAND = 7.6                                # half width of the sinusoid lumen
DATA_X, DATA_Y = 127.0, (88.0, 70.0, 52.0, 34.0, 16.0)
RING = np.array([165.0, 52.0])
LABELS: list = []                         # (text artist) filled by label()


def label(ax, x, y, s, color=TEXT_GREY, ha="center", va="top"):
    t = ax.text(x, y, s, fontsize=FS, color=color, ha=ha, va=va, zorder=20)
    LABELS.append(t)
    return t


# ---------------------------------------------------------------- sinusoid zoom
def edge(x, side):
    """Upper (side=+1) / lower (side=-1) endothelial boundary of the lumen."""
    u = (x - C[0]) / R
    return C[1] + side * (BAND + 0.7 * np.sin(2.3 * u + (0.6 if side > 0 else 2.1))) + 0.9 * np.sin(1.4 * u)


def hepatocyte_plates(ax, clip):
    rng = np.random.default_rng(21)
    s = 8.2
    pts = []
    for i, yy in enumerate(np.arange(C[1] - R - 2 * s, C[1] + R + 2 * s, s * 0.866)):
        for xx in np.arange(C[0] - R - 2 * s, C[0] + R + 2 * s, s):
            pts.append((xx + (s / 2 if i % 2 else 0) + rng.normal(0, 0.9), yy + rng.normal(0, 0.9)))
    vor = Voronoi(np.array(pts))
    for p_idx, reg_idx in enumerate(vor.point_region):
        reg = vor.regions[reg_idx]
        if -1 in reg or not reg:
            continue
        poly = vor.vertices[reg]
        cx, cy = poly.mean(axis=0)
        if np.hypot(cx - C[0], cy - C[1]) > R + 6:
            continue
        di.poly(ax, poly, PEACH, ec=tint(PEACH, -0.42), lw=0.4, z=1, clip=clip)
        if abs(cy - C[1]) > BAND - 1:
            di.dot(ax, cx, cy, 1.55, tint(PEACH, -0.14), ec=tint(PEACH, -0.42), lw=0.3, z=1.2, clip=clip)
            di.granules(ax, cx, cy, 3.0, 4, tint(PEACH, -0.3), seed=p_idx, clip=clip, z=1.1,
                        size=0.17, avoid=(cx, cy, 2.0))


def endothelium(ax, clip):
    xs_all = np.linspace(C[0] - R - 2, C[0] + R + 2, 400)
    lumen = np.vstack([np.column_stack([xs_all, edge(xs_all, 1)]),
                       np.column_stack([xs_all[::-1], edge(xs_all[::-1], -1)])])
    di.poly(ax, lumen, tint(RED, 0.72), ec="none", z=2, clip=clip)
    t = 1.55
    for side, off in ((1, 0.0), (-1, 3.8)):
        x0 = C[0] - R - 6 + off
        k = 0
        while x0 < C[0] + R + 2:
            L = 8.6 + (k % 3) * 0.7
            xs = np.linspace(x0, x0 + L, 60)
            u = (xs - x0) / L
            yb = edge(xs, side)
            outer = yb + side * (0.25 + t * np.clip(np.sin(np.pi * u), 0, 1) ** 0.7)
            inner = yb - side * 0.35 * np.sin(np.pi * u)
            cell = np.vstack([np.column_stack([xs, outer]), np.column_stack([xs[::-1], inner[::-1]])])
            di.poly(ax, cell, LILAC, lw=0.35, z=2.5, clip=clip)
            xm = x0 + L * 0.5
            ym = edge(xm, side) + side * 0.65
            slope = np.degrees(np.arctan((edge(xm + 0.5, side) - edge(xm - 0.5, side))))
            di.add(ax, Ellipse((xm, ym), 3.0, 0.85, angle=slope, fc=tint(PURPLE, 0.35), ec=INK, lw=0.3,
                               zorder=2.6), clip)
            x0 += L - 0.6
            k += 1


def sinusoid(ax):
    clip = Circle(tuple(C), R, fc="white", ec="none", zorder=0.9)
    ax.add_patch(clip)
    hepatocyte_plates(ax, clip)
    endothelium(ax, clip)
    for x, y, a in ((45.5, 47.0, 15), (49.5, 39.0, -20), (75.0, 49.2, 5), (81.0, 38.2, 10),
                    (95.5, 47.0, 30), (99.5, 39.8, 0), (88.5, 51.0, -15), (51.5, 47.6, -10)):
        di.rbc(ax, x, y, 1.55, 0.9, a, clip=clip, z=3)
    di.kupffer_cell(ax, C[0] - 10.0, C[1] + 0.2, 3.9, clip=clip, z=6, ang=5)
    di.recruited_macrophage(ax, C[0] + 13.5, C[1] + 0.4, 4.6, clip=clip, z=6)
    ax.add_patch(Circle(tuple(C), R, fc="none", ec=INK, lw=0.9, zorder=8))


def zoom_cone(ax, p):
    d = np.linalg.norm(C - p)
    base = np.arctan2(p[1] - C[1], p[0] - C[0])
    half = np.arccos(R / d)
    tangents = [C + R * np.array([np.cos(base + s * half), np.sin(base + s * half)]) for s in (1, -1)]
    arc_t = np.linspace(base + half, base - half, 60)
    arc = np.column_stack([C[0] + R * np.cos(arc_t), C[1] + R * np.sin(arc_t)])
    ax.add_patch(Polygon(np.vstack([p, arc]), closed=True, fc="#ECEDEF", ec="none", zorder=0.3))
    for tp in tangents:
        di.line(ax, [p, tp], c=di.CONNECT, lw=0.5, z=0.4)
    di.dot(ax, p[0], p[1], 1.3, "none", ec="white", lw=0.8, z=4)


# ---------------------------------------------------------------- columns
def stimuli(ax):
    for x, y, a, s in ((8.5, 86.0, 18, 0), (16.5, 88.5, -12, 1), (11.5, 79.5, 28, 2)):
        di.bacterium(ax, x, y, 7.2, 2.5, a, seed=s, z=4)
    for x, y, a in ((23.0, 86.5, 30), (26.5, 82.0, -40), (22.0, 79.5, 160), (27.5, 88.5, 80)):
        di.lps_fragment(ax, x, y, 0.75, a)
    label(ax, 17, 74.5, "PAMP (LPS)", RED)

    di.hepatocyte(ax, 12.5, 56.0, 6.6, ruptured=True, seed=6, z=3)
    rng = np.random.default_rng(9)
    for k in range(9):
        x = 19.5 + k * 1.25 + rng.uniform(-0.3, 0.3)
        y = 56.0 + rng.normal(0, 1.2 + k * 0.28)
        di.dot(ax, x, y, 0.55 - k * 0.02, ORANGE, ec=INK, lw=0.25, z=4)
    label(ax, 17, 45.5, "DAMP (HMGB1)", ORANGE)

    for x, y, r in ((10.0, 27.5, 1.25), (15.5, 30.0, 1.0), (21.5, 28.0, 1.35), (13.0, 22.5, 1.4),
                    (19.5, 22.5, 1.05), (25.0, 24.0, 0.85), (8.5, 21.5, 0.8)):
        di.cytokine(ax, x, y, r)
    label(ax, 17, 17.5, "Cytokines", PURPLE)
    for y in (83.5, 56.0, 25.5):
        di.connector(ax, (31.0, y), (C[0] - R + 1.0, C[1]))


def data_layers(ax):
    drawers = (di.plate, di.tubes_sorter, di.single_cell, di.spatial, di.proteome)
    names = ("Stimulus references", "Sorted cells", "Single cell", "Spatial", "Proteome")
    for y, f, n in zip(DATA_Y, drawers, names):
        x0, x1 = f(ax, DATA_X, y)
        label(ax, DATA_X, y - 6.0, n)
        di.connector(ax, (C[0] + R - 1.0, C[1]), (x0 - 1.0, y))
        di.connector(ax, (x1 + 1.0, y), (RING[0] - 13.0, RING[1]), arrow=(y == DATA_Y[2]))


def attribution(ax):
    di.split_ring(ax, RING[0], RING[1], 10.5, 1.9)
    di.kupffer_cell(ax, RING[0], RING[1], 2.6, z=4, seed=9, phagosome=False, ang=20)
    label(ax, RING[0], RING[1] - 13.0, "Source attribution")


def header(ax):
    di.mouse(ax, 46.0, 89.0, s=1.05)
    di.liver(ax, 63.5, 83.5, 24.0, 11.0)
    zoom_cone(ax, np.array([71.5, 88.0]))
    label(ax, 91.0, 90.0, "Sepsis (CLP, LPS)", ha="left", va="center")


def cell_labels(ax):
    label(ax, C[0] - 12.0, C[1] - R - 3.0, "Resident Kupffer cell", BLUE)
    label(ax, C[0] + 15.5, C[1] - R - 3.0, "Recruited macrophage", GREEN)


def build():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    LABELS.clear()
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, Y0 + H)
    ax.set_aspect("equal")
    ax.axis("off")
    sinusoid(ax)
    header(ax)
    stimuli(ax)
    data_layers(ax)
    attribution(ax)
    cell_labels(ax)
    return fig, ax


# ---------------------------------------------------------------- layout check
def layout_report(fig, dpi=600, pad_px=9) -> list[str]:
    """Rasterize drawings without text; every label box (padded) must contain only white pixels."""
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    for t in LABELS:
        t.set_visible(False)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].astype(int)
    for t in LABELS:
        t.set_visible(True)
    canvas.draw()
    rend = canvas.get_renderer()
    hpx, wpx = img.shape[:2]
    boxes, bad = [], []
    for t in LABELS:
        bb = t.get_window_extent(rend)
        x0, x1 = int(bb.x0) - pad_px, int(np.ceil(bb.x1)) + pad_px
        y0, y1 = hpx - int(np.ceil(bb.y1)) - pad_px, hpx - int(bb.y0) + pad_px
        boxes.append((t.get_text(), x0, y0, x1, y1))
        if x0 < 0 or y0 < 0 or x1 > wpx or y1 > hpx:
            bad.append(f"OUTSIDE {t.get_text()}")
            continue
        n = int((img[y0:y1, x0:x1].min(axis=2) < 250).sum())
        if n:
            bad.append(f"LABEL ON DRAWING {t.get_text()}: {n} px")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            p, q = boxes[i], boxes[j]
            if min(p[3], q[3]) > max(p[1], q[1]) and min(p[4], q[4]) > max(p[2], q[2]):
                bad.append(f"LABEL OVERLAP {p[0]} / {q[0]}")
    return bad


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig, _ = build()
    bad = layout_report(fig)
    for b in bad:
        print(b)
    if bad:
        sys.exit(1)
    fig.set_dpi(300)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"Fig1_drawn.{ext}", facecolor="white", metadata={"Creator": None} if ext == "svg" else None)
    fig.savefig(OUT / "Fig1_drawn.png", dpi=300, facecolor="white")
    print("layout check passed:", len(LABELS), "labels")


if __name__ == "__main__":
    main()
