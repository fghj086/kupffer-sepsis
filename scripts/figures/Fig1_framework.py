#!/usr/bin/env python
"""Figure 1 - study framework: the question, the data and design, and the findings.

Drawn entirely in code (matplotlib patches, paths and live text; no raster or generated
images). Three bands: (a) the question, a liver sinusoid receiving bacterial products,
danger signals and cytokines; (b) six cards with the data sets, designs and sample sizes;
(c) four finding cards with a schematic glyph, a headline and the supporting number.
The build fails if any label touches drawn content, another label or leaves its card.
Outputs: figures/Fig1_framework.{pdf,svg,png}, figures/source_data/Fig1_framework.tsv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drawn_icons as di  # noqa: E402
import fig1_framework_glyphs as gl  # noqa: E402
import Fig1_overview_drawn as ov  # noqa: E402
import figstyle  # noqa: E402
from drawn_icons import BLUE, GREEN, INK, ORANGE, PURPLE, RED, tint  # noqa: E402
from fig1_framework_glyphs import BROWN  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
W, H = 180.0, 158.0                     # mm
FS, FS_T = 6.5, 7.0                     # body and title sizes (pt)
TEXT = "#333333"
SLATE = "#55606E"
ARROW = "#A9AEB5"
LABELS: list = []                       # (text, bounds or None)
SOURCE: list = []                       # rows of the source-data table

FONT = ["Arial", "Liberation Sans"]

# ---------------------------------------------------------------- text helpers
_R = None


def _width_mm(s: str, fs: float, bold: bool) -> float:
    prop = FontProperties(family=FONT, size=fs, weight="bold" if bold else "normal")
    w, _, _ = _R.get_text_width_height_descent(s, prop, ismath=False)
    return w / _R.dpi * 25.4


def wrap(s: str, width: float, fs: float = FS, bold: bool = False) -> str:
    """Fewest lines that fit `width`, with the longest line as short as possible."""
    from itertools import combinations
    words = s.split(" ")
    n, cur = 1, ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and _width_mm(trial, fs, bold) > width:
            n, cur = n + 1, word
        else:
            cur = trial
    if n == 1:
        return s
    best, best_w = None, None
    for cuts in combinations(range(1, len(words)), n - 1):
        b = (0,) + cuts + (len(words),)
        lines = [" ".join(words[b[i]:b[i + 1]]) for i in range(n)]
        mw = max(_width_mm(ln, fs, bold) for ln in lines)
        if mw <= width and (best_w is None or mw < best_w):
            best, best_w = lines, mw
    return "\n".join(best)


def label(ax, x, y, s, fs=FS, color=TEXT, bold=False, ha="center", va="top", bounds=None,
          panel="", role="", z=20):
    t = ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, linespacing=1.18,
                fontweight="bold" if bold else "normal")
    LABELS.append((t, bounds))
    SOURCE.append({"panel": panel, "element": role, "text": s.replace("\n", " ")})
    return t


def lines_height(s: str, fs: float) -> float:
    return (s.count("\n") + 1) * fs * 1.18 * 25.4 / 72


def card(ax, x0, y0, w, h, fc, ec, lw=0.6, r=1.8, z=1):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, zorder=z)
    p._bg = True
    ax.add_patch(p)
    return (x0, y0, x0 + w, y0 + h)


def arrow(ax, a, b, rad=0.0, lw=0.7, color=ARROW, z=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=2.4,head_width=1.3",
                                 connectionstyle=f"arc3,rad={rad}", lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


# ---------------------------------------------------------------- panel a: the question
C_MM, R_MM = np.array([58.0, 128.5]), 19.5


def stimuli(ax):
    for x, y, a, s in ((7.5, 146.5, 18, 0), (14.5, 148.0, -14, 1), (10.0, 142.5, 24, 2)):
        di.bacterium(ax, x, y, 5.6, 1.9, a, seed=s, z=4)
    for x, y, a in ((20.5, 147.5, 30), (23.5, 144.0, -40), (19.5, 142.0, 160), (24.5, 148.5, 80)):
        di.lps_fragment(ax, x, y, 0.6, a)
    label(ax, 16, 139.0, "Bacterial products\n(LPS)", color=RED, panel="a", role="stimulus")

    di.hepatocyte(ax, 10.0, 126.5, 4.4, ruptured=True, seed=6, z=3)
    rng = np.random.default_rng(9)
    for k in range(8):
        di.dot(ax, 15.6 + k * 1.05 + rng.uniform(-0.2, 0.2), 126.5 + rng.normal(0, 0.8 + k * 0.22),
               0.45, ORANGE, ec=INK, lw=0.25, z=4)
    label(ax, 16, 120.0, "Danger signals\n(HMGB1)", color=ORANGE, panel="a", role="stimulus")

    for x, y, r in ((9.0, 109.5, 0.95), (13.5, 111.5, 0.8), (18.5, 110.0, 1.05), (11.5, 106.5, 1.05),
                    (16.5, 106.3, 0.85), (22.0, 107.8, 0.7), (6.5, 106.0, 0.65)):
        di.cytokine(ax, x, y, r)
    label(ax, 16, 103.8, "Cytokines", color=PURPLE, panel="a", role="stimulus")
    for y in (145.0, 126.5, 108.5):
        di.connector(ax, (27.5, y), (C_MM[0] - R_MM - 0.6, C_MM[1]), lw=0.7)


def sinusoid(fig, ax):
    """The sinusoid of the overview figure, scaled into an inset of radius R_MM."""
    m = ov.R + 0.6
    s = R_MM / ov.R
    half = m * s
    ins = fig.add_axes([(C_MM[0] - half) / W, (C_MM[1] - half) / H, 2 * half / W, 2 * half / H])
    ins.set_xlim(ov.C[0] - m, ov.C[0] + m)
    ins.set_ylim(ov.C[1] - m, ov.C[1] + m)
    ins.set_aspect("equal")
    ins.axis("off")
    ov.sinusoid(ins)
    # resident and recruited cells sit at C - 10 and C + 13.5 in the overview frame
    label(ax, C_MM[0] - 9.5 * s - 3.0, C_MM[1] - R_MM - 1.6, "Kupffer cell", color=BLUE,
          panel="a", role="cell")
    label(ax, C_MM[0] + 13.5 * s + 4.0, C_MM[1] - R_MM - 1.6, "Recruited\nmacrophage", color=GREEN,
          panel="a", role="cell")
    return ins


QUESTIONS = (
    ("Q1", "Does septic inflammation erode the Kupffer cell scavenger program?", BLUE),
    ("Q2", "Which stimuli does the macrophage response resemble?", SLATE),
)


def questions(ax):
    x0, x1 = 90.0, 178.5
    cy = C_MM[1]
    for (tag, q, c), (yb, yt) in zip(QUESTIONS, ((cy + 3.0, cy + 17.0), (cy - 17.0, cy - 3.0))):
        b = card(ax, x0, yb, x1 - x0, yt - yb, tint(c, 0.92), tint(c, 0.45))
        bx, by = x0 + 7.0, (yb + yt) / 2
        ax.add_patch(Circle((bx, by), 4.0, fc=c, ec="none", zorder=3))
        t = ax.text(bx, by, tag, fontsize=7.5, color="white", fontweight="bold", ha="center",
                    va="center", zorder=21)
        t._overlap_exempt = True      # deliberately sits on its badge
        SOURCE.append({"panel": "a", "element": "question tag", "text": tag})
        s = wrap(q, x1 - x0 - 17.0, fs=7.5, bold=False)
        label(ax, x0 + 13.5, by, s, fs=7.5, color=TEXT, ha="left", va="center", bounds=b,
              panel="a", role="question")
        arrow(ax, (C_MM[0] + R_MM + 1.0, C_MM[1] + (3.0 if tag == "Q1" else -3.0)),
              (x0 - 0.8, by), rad=-0.12 if tag == "Q1" else 0.12)


# ---------------------------------------------------------------- panel b: data and design
DATA = (
    ("Stimulus references", ("41 responses, 11 studies",
                             "resolution tested across 4 laboratories at in vivo noise"), "plate"),
    ("Sorted liver cells", ("Kupffer cells 2 h; 5 cell types 8 h", "sham vs CLP, 4 + 4 mice"), "sort"),
    ("Single cells", ("12 individual septic livers", "pooled sham vs CLP"), "sc"),
    ("Time courses", ("LPS 0–5 d; graded CLP (45 mice)", "cytokine blockade"), "time"),
    ("Spatial + proteome", ("zone-wise LPS response", "liver proteome 24 h"), "spat"),
    ("Human", ("tolerant monocytes, septic patients", "5 blood cohorts (897)"), "human"),
)
B_TOP, B_BOT, B_ICON = 96.0, 64.0, 89.0


def spatial_proteome_inset(fig, cx, cy):
    """Spatial section above, proteome trace below, scaled into one inset."""
    s = 0.56
    w, h = 23.0 * s, 21.0 * s
    ins = fig.add_axes([(cx - w / 2 + 0.0) / W, (cy - h / 2) / H, w / W, h / H])
    ins.set_xlim(-11.5, 11.5)
    ins.set_ylim(-10.5, 10.5)
    ins.set_aspect("equal")
    ins.axis("off")
    di.spatial(ins, 0, 5.0)
    di.proteome(ins, 0, -5.4)
    return ins


def data_cards(fig, ax):
    n, gap, x_start = len(DATA), 1.8, 1.5
    w = (W - 2 * x_start - (n - 1) * gap) / n
    tops = []
    for k, (title, body, icon) in enumerate(DATA):
        x0 = x_start + k * (w + gap)
        cx = x0 + w / 2
        b = card(ax, x0, B_BOT, w, B_TOP - B_BOT, "#F5F6F8", "#C9CED6")
        if icon == "plate":
            di.plate(ax, cx, B_ICON)
        elif icon == "sort":
            di.tubes_sorter(ax, cx, B_ICON)
        elif icon == "sc":
            di.single_cell(ax, cx, B_ICON)
        elif icon == "time":
            gl.time_course(ax, cx, B_ICON)
        elif icon == "spat":
            spatial_proteome_inset(fig, cx, B_ICON - 0.6)
        else:
            gl.human(ax, cx, B_ICON)
        ttl = wrap(title, w - 1.8, FS_T, bold=True)
        y = 81.6
        label(ax, cx, y, ttl, fs=FS_T, bold=True, color=SLATE, bounds=b, panel="b", role="card title")
        y -= lines_height(ttl, FS_T) + 0.9
        for line in body:
            s = wrap(line, w - 1.8)
            label(ax, cx, y, s, bounds=b, panel="b", role="card detail")
            y -= lines_height(s, FS) + 0.7
        tops.append(y)
    return w


BRIDGE = "Decomposition into reference responses + within-liver and sham-controlled comparisons"


def bridge(ax):
    y = 55.6
    tw = _width_mm(BRIDGE, FS, False)
    b = card(ax, W / 2 - tw / 2 - 4.0, y - 2.9, tw + 8.0, 5.8, "#EEF0F3", "none", r=2.9)
    label(ax, W / 2, y, BRIDGE, color=SLATE, va="center", bounds=b, panel="b-c", role="approach")
    arrow(ax, (W / 2, B_BOT - 0.4), (W / 2, y + 3.0), lw=0.8)
    arrow(ax, (W / 2, y - 3.0), (W / 2, C_TOP + 1.6), lw=0.8)


# ---------------------------------------------------------------- panel c: findings
FINDINGS = (
    ("Inflammation uncoupled from scavenger program",
     "scavenger lead over other leukocytes: sham 3.62 vs CLP 3.56 log2", BLUE),
    ("Kupffer cells > recruited macrophages",
     "scavenger and identity genes higher in 12/12 livers", GREEN),
    ("Tolerance-type program within hours",
     "from 8 h in every septic mouse; resident and recruited", BROWN),
    ("HMGB1 and LPS: one program",
     "r = 0.945; microbial-ligand responses transfer across laboratories", ORANGE),
)
C_TOP, C_BOT, G_Y = 47.0, 10.6, 39.8


def glyph_uncoupled(ax, x0):
    di.kupffer_cell(ax, x0 + 6.5, G_Y + 0.8, 1.75, z=4, seed=9, phagosome=False, ang=20)
    yb = G_Y - 2.6
    gl.mini_bars(ax, x0 + 19.0, yb, 1.6, 6.8, RED)
    gl.mini_bars(ax, x0 + 33.0, yb, 5.4, 5.3, BLUE)
    label(ax, x0 + 19.0, yb - 0.8, "Inflammation", color=tint(RED, -0.15), panel="c1", role="glyph")
    label(ax, x0 + 33.0, yb - 0.8, "Scavenger", color=BLUE, panel="c1", role="glyph")
    ly = yb - 5.2                                     # legend: grey sham, filled CLP
    from matplotlib.patches import Rectangle
    for lx, txt, fills in ((x0 + 14.0, "Sham", (gl.GREY_BAR,)), (x0 + 26.0, "CLP", (RED, BLUE))):
        for i, fc in enumerate(fills):
            ax.add_patch(Rectangle((lx + i * 0.8 * (2 / len(fills)), ly - 0.8), 1.6 / len(fills), 1.6,
                                   fc=fc, ec="none", zorder=4))
        ax.add_patch(Rectangle((lx, ly - 0.8), 1.6, 1.6, fc="none", ec=INK, lw=0.3, zorder=4.1))
        label(ax, lx + 2.4, ly, txt, ha="left", va="center", panel="c1", role="glyph legend")


def glyph_kc_vs_rec(ax, x0):
    di.kupffer_cell(ax, x0 + 10.0, G_Y + 1.0, 1.75, z=4, seed=3, phagosome=False, ang=5)
    di.recruited_macrophage(ax, x0 + 31.0, G_Y + 1.0, 2.9, z=4)
    t = label(ax, x0 + 20.5, G_Y + 1.0, ">", fs=12, color=SLATE, bold=True, va="center",
              panel="c2", role="glyph")
    t.set_va("center_baseline")
    label(ax, x0 + 10.0, G_Y - 4.3, "Kupffer", color=BLUE, panel="c2", role="glyph")
    label(ax, x0 + 31.0, G_Y - 4.3, "Recruited", color=GREEN, panel="c2", role="glyph")


def glyph_tolerance(ax, x0):
    xa, xb, yb = x0 + 6.0, x0 + 36.0, G_Y - 3.6
    x8 = x0 + 18.0
    xs, ys = gl.tolerance_curve(ax, xa, xb, yb, 7.6, x8)
    for xt, tk in ((xa, "0 h"), (x8, "8 h"), (x0 + 32.0, "24 h")):
        di.line(ax, [(xt, yb), (xt, yb - 0.7)], c=INK, lw=0.45, z=4)
        label(ax, xt, yb - 1.1, tk, panel="c3", role="glyph")
    for xt, c in ((x8 - 1.1, BLUE), (x8 + 1.1, GREEN)):     # resident and recruited, both by 8 h
        yt = np.interp(xt, xs, ys)
        di.dot(ax, xt, yt, 0.85, c, ec="white", lw=0.5, z=6)


def glyph_one_program(ax, x0):
    cx = x0 + 21.0
    gl.overlap_circles(ax, cx, G_Y + 0.8, r=4.4, d=2.3)
    label(ax, cx - 7.6, G_Y + 0.8, "HMGB1", color=tint(ORANGE, -0.25), ha="right", va="center",
          panel="c4", role="glyph")
    label(ax, cx + 7.6, G_Y + 0.8, "LPS", color=tint(RED, -0.15), ha="left", va="center",
          panel="c4", role="glyph")


NOTE = "Severe CLP deepens Kupffer identity loss"
Y_HEAD = 27.8


def finding_cards(ax):
    n, gap, x_start = len(FINDINGS), 3.0, 1.5
    w = (W - 2 * x_start - (n - 1) * gap) / n
    glyphs = (glyph_uncoupled, glyph_kc_vs_rec, glyph_tolerance, glyph_one_program)
    for k, ((head, ev, c), g) in enumerate(zip(FINDINGS, glyphs)):
        x0 = x_start + k * (w + gap)
        b = card(ax, x0, C_BOT, w, C_TOP - C_BOT, tint(c, 0.93), tint(c, 0.5))
        g(ax, x0)
        hd = wrap(head, w - 3.0, FS_T, bold=True)
        y = Y_HEAD
        label(ax, x0 + w / 2, y, hd, fs=FS_T, bold=True, color=tint(c, -0.35), bounds=b,
              panel=f"c{k + 1}", role="finding")
        y -= lines_height(hd, FS_T) + 1.0
        evs = wrap(ev, w - 3.0)
        label(ax, x0 + w / 2, y, evs, bounds=b, panel=f"c{k + 1}", role="evidence")
        if k == 0:                   # additional finding, as a slim note under card 1
            tw = _width_mm(NOTE, FS, False)
            nb = card(ax, x0, 2.4, tw + 7.0, 6.0, "white", tint(c, 0.5), r=3.0)
            ax.add_patch(Circle((x0 + 3.0, 5.4), 0.9, fc=c, ec="none", zorder=3))
            label(ax, x0 + 4.8, 5.4, NOTE, color=tint(c, -0.35), ha="left", va="center", bounds=nb,
                  panel="c1", role="additional finding")
    return w


# ---------------------------------------------------------------- assembly
BANDS = (("a", 101.0, 152.0, "The question"), ("b", B_BOT, B_TOP, "What we examined"),
         ("c", C_BOT, C_TOP, "What we found"))


def build():
    global _R
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    LABELS.clear()
    SOURCE.clear()
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)
    _R = fig.canvas.get_renderer()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    stimuli(ax)
    sinusoid(fig, ax)
    questions(ax)
    data_cards(fig, ax)
    bridge(ax)
    finding_cards(ax)
    # panel letters via the house aligner: one invisible frame per band
    for letter, y0, y1, _ in BANDS:
        fr = fig.add_axes([1.5 / W, y0 / H, (W - 3.0) / W, (y1 - y0) / H])
        fr.set_axis_off()
        fr._frame = True
        figstyle.panel_label(fr, letter)
    for a_ in fig.axes:              # no axes draws ticks; drop their hidden labels too
        a_.set_xticks([])
        a_.set_yticks([])
    figstyle.align_panel_labels(fig, pad_pt=2.0)
    r = fig.canvas.get_renderer()
    for t in [t for t in fig.texts if getattr(t, "_panel_letter", False)]:
        bb = t.get_window_extent(r)
        x = fig.transFigure.inverted().transform((bb.x1, bb.y0))[0]
        head = dict((b[0], b[3]) for b in BANDS)[t.get_text().lower()]
        h = fig.text(x + 1.4 / W, t.get_position()[1], head, fontsize=FS_T, fontweight="bold",
                     color=SLATE, ha="left", va="bottom")
        LABELS.append((h, None))
        SOURCE.append({"panel": t.get_text(), "element": "band heading", "text": head})
        LABELS.append((t, None))
    return fig, ax


# ---------------------------------------------------------------- layout check
def _render(fig, dpi):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].astype(int), canvas.get_renderer()


def layout_report(fig, ax, dpi=600, pad_px=5, margin_mm=0.8) -> list[str]:
    """Every label must sit on empty card or page, inside its card, apart from other labels.

    The figure is rasterised twice without text: once complete, once with only the card
    backgrounds. Inside each (padded) label box the two images must be identical, so a label
    may lie on a card fill but never on an icon, arrow, glyph or card edge.
    """
    fig.set_dpi(dpi)
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    for t in texts:
        t.set_visible(False)
    full, _ = _render(fig, dpi)
    hidden = []
    for a in ax.get_children():
        if getattr(a, "_bg", False) or a is ax.patch or not a.get_visible():
            continue
        if isinstance(a, (matplotlib.patches.Patch, matplotlib.lines.Line2D,
                          matplotlib.collections.Collection)):
            a.set_visible(False)
            hidden.append(a)
    for other in fig.axes:
        if other is not ax and not getattr(other, "_frame", False) and other.get_visible():
            other.set_visible(False)
            hidden.append(other)
    bg, _ = _render(fig, dpi)
    for a in hidden:
        a.set_visible(True)
    for t in texts:
        t.set_visible(True)
    _, rend = _render(fig, dpi)
    hpx, wpx = full.shape[:2]
    px_per_mm = dpi / 25.4
    bad, boxes = [], []
    for t, bounds in LABELS:
        bb = t.get_window_extent(rend)
        x0, x1 = int(bb.x0) - pad_px, int(np.ceil(bb.x1)) + pad_px
        y0, y1 = hpx - int(np.ceil(bb.y1)) - pad_px, hpx - int(bb.y0) + pad_px
        name = t.get_text().replace("\n", " ")[:40]
        boxes.append((name, x0, y0, x1, y1))
        if x0 < 0 or y0 < 0 or x1 > wpx or y1 > hpx:
            bad.append(f"OUTSIDE FIGURE {name}")
            continue
        n = int((np.abs(full[y0:y1, x0:x1] - bg[y0:y1, x0:x1]).max(axis=2) > 12).sum())
        if n:
            bad.append(f"LABEL ON DRAWING {name}: {n} px")
        if bounds is not None:
            m = margin_mm * px_per_mm
            bx0, by0, bx1, by1 = (bounds[0] * px_per_mm, bounds[1] * px_per_mm,
                                  bounds[2] * px_per_mm, bounds[3] * px_per_mm)
            if bb.x0 < bx0 + m or bb.x1 > bx1 - m or bb.y0 < by0 + m or bb.y1 > by1 - m:
                bad.append(f"OUTSIDE CARD {name}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            p, q = boxes[i], boxes[j]
            if min(p[3], q[3]) > max(p[1], q[1]) and min(p[4], q[4]) > max(p[2], q[2]):
                bad.append(f"LABEL OVERLAP {p[0]} / {q[0]}")
    bad += [f"figstyle: {m}" for m in figstyle.overlap_report(fig)]
    return bad


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig, ax = build()
    bad = layout_report(fig, ax)
    for b in bad:
        print(b)
    if bad and "--draft" not in sys.argv:
        sys.exit(1)
    fig.set_dpi(300)
    fig.savefig(OUT / "Fig1_framework.pdf", facecolor="white")
    fig.savefig(OUT / "Fig1_framework.svg", facecolor="white", metadata={"Creator": None})
    fig.savefig(OUT / "Fig1_framework.png", dpi=300, facecolor="white")
    src = OUT / "source_data"
    src.mkdir(exist_ok=True)
    pd.DataFrame(SOURCE).to_csv(src / "Fig1_framework.tsv", sep="\t", index=False)
    print(f"layout check {'passed' if not bad else 'FAILED (draft written)'}:",
          len(LABELS), "labels;", f"{W:.0f} x {H:.0f} mm")


if __name__ == "__main__":
    main()
