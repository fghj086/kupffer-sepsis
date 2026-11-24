"""Shared figure style (300 dpi, PDF + PNG).

Every figure script imports this, writes its panel inputs to
figures/source_data/<figure>_<panel>.tsv, and saves PDF + PNG. Retouching a panel
must never require rerunning an analysis stage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

GROUP_COLORS = {
    "control": "#8C8C8C", "healthy": "#8C8C8C",
    "survivor": "#4C72B0", "nonsurvivor": "#C44E52",
    "septic_shock": "#DD8452",
    "sham": "#4C72B0", "CLP": "#C44E52", "CLPDCA": "#55A868",
    "septic": "#C44E52", "recD1": "#DD8452", "recD2": "#CCB974", "recD3": "#55A868",
}

MODULE_COLORS = {"M1": "#4C72B0", "M2": "#DD8452", "M3": "#55A868",
                 "M4": "#C44E52", "M5": "#8172B3"}


LETTER_SIZE = 8.0

# stimulus classes and cell populations, matched to Figure 1
CLASS_COLORS = {"PAMP": "#C44E52", "DAMP": "#DD8452", "cytokine": "#8172B3",
                "resolution": "#937860", "mixed": "#B0B0B0"}
CLASS_LABELS = {"PAMP": "PAMP", "DAMP": "DAMP", "cytokine": "Cytokine",
                "resolution": "Tolerance-associated", "mixed": "LPS/HMGB1 shared"}
KC_BLUE, REC_GREEN, GREY = "#4C72B0", "#55A868", "#8C8C8C"
COND_COLORS = {"PBS": "#8C8C8C", "frHMGB1": "#F2C49B", "dsHMGB1": "#DD8452",
               "LPS": "#C44E52", "LPS_IFNg": "#7A2427"}
COND_LABELS = {"PBS": "PBS", "frHMGB1": "Reduced HMGB1", "dsHMGB1": "Disulfide HMGB1",
               "LPS": "LPS", "LPS_IFNg": "LPS + IFN-γ"}
REF_LABELS = {
    "LPS_24h": "LPS", "dsHMGB1": "Disulfide HMGB1", "LPS_24h+dsHMGB1": "LPS/HMGB1 shared",
    "KLA_6h": "Kdo2-lipid A", "KLA_6h+LPS_24h+dsHMGB1": "LPS/HMGB1/KLA shared",
    "Pam3CSK4": "Pam3CSK4", "PolyIC": "Poly(I:C)", "CpG": "CpG",
    "TissueHom_6h": "Tissue homogenate", "Hemin": "Hemin", "ApoptoticHepCM": "Apoptotic hepatocytes",
    "TNF": "TNF", "IL1b": "IL-1β", "IFNg": "IFN-γ", "IFNb": "IFN-β", "IL6": "IL-6", "IL10": "IL-10",
    "KC_TolLPS_6h": "LPS-tolerant KC", "Tolerance_shift": "LPS tolerance", "Dex": "Dexamethasone",
    "Dex_on_LPS": "Dexamethasone on LPS", "liv_LPS": "LPS (liver)", "liv_TNF": "TNF (liver)",
    "liv_IL1b": "IL-1β (liver)", "liv_IFNg": "IFN-γ (liver)", "liv_IL6": "IL-6 (liver)",
    "liv_IL10": "IL-10 (liver)", "liv_IL18": "IL-18 (liver)", "liv_IL1b+liv_TNF": "IL-1β/TNF (liver)",
    "LPS_IFNg_24h": "LPS + IFN-γ", "KLA_IFNg_24h": "KLA + IFN-γ", "LPS_RAW_14h": "LPS, RAW264.7",
    "KC_LPS_6h": "LPS, cultured KC", "LPS_24h_M1": "LPS, other BMDM",
    "liv_TNFIFNg": "TNF + IFN-γ (liver)", "liv_TNFIL6": "TNF + IL-6 (liver)",
    "liv_IFNgIL1b": "IFN-γ + IL-1β (liver)", "liv_IL1bIL6": "IL-1β + IL-6 (liver)",
    "liv_IFNgIL6": "IFN-γ + IL-6 (liver)", "liv_TNFIL1b": "TNF + IL-1β (liver)",
    "IFNg_24h": "IFN-γ 24 h", "PolyIC_RAW": "Poly(I:C), RAW264.7", "frHMGB1": "Reduced HMGB1",
    "TissueHom_24h": "Tissue homogenate 24 h",
}
MM = 1 / 25.4


def apply_style(base_fontsize: float = 7.0) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Register a panel letter; save_fig places all of a figure's letters together.

    Letters are positioned from the drawn extent of each panel rather than from
    offsets inside it, so that letters on one row share a baseline and letters in
    one column share a left edge, however large each panel is. `dx` and `dy` are
    accepted for compatibility and no longer used.
    """
    import os
    # PANEL_CASE=upper prints panel letters in capitals; the default is lower case
    if os.environ.get("PANEL_CASE", "").lower() == "upper":
        letter = letter.upper()
    fig = ax.figure
    if not hasattr(fig, "_panel_labels"):
        fig._panel_labels = []
    fig._panel_labels.append((ax, letter))


def _spans_overlap(a0, a1, b0, b1) -> bool:
    inter = min(a1, b1) - max(a0, b0)
    return inter > 0.5 * min(a1 - a0, b1 - b0)


def align_panel_labels(fig, pad_pt: float = 4.0, col_tol: float = 0.07,
                       max_shift: float = 0.12) -> None:
    """Place registered panel letters on shared rows and columns.

    Rows: panels whose axes overlap vertically by more than half of the shorter
    one form a row, and every letter of the row sits just above the highest drawn
    top in it (titles and labels included). Columns: letters whose panels start
    within `col_tol` of the figure width share the leftmost of their left edges.
    """
    items = getattr(fig, "_panel_labels", [])
    if not items:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rec = []
    for ax, letter in items:
        tb = inv.transform_bbox(ax.get_tightbbox(r))
        pos = ax.get_position()
        rec.append({"ax": ax, "letter": letter, "x0": tb.x0, "top": tb.y1,
                    "y0": pos.y0, "y1": pos.y1, "px0": pos.x0, "px1": pos.x1})
    # rows
    rows = []
    for it in sorted(rec, key=lambda d: -d["y1"]):
        for row in rows:
            if any(_spans_overlap(it["y0"], it["y1"], o["y0"], o["y1"]) for o in row):
                row.append(it); break
        else:
            rows.append([it])
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows:
        y = max(o["top"] for o in row) + pad
        for o in row:
            o["y"] = y
    # columns: panels that overlap horizontally, or whose drawn left edges lie
    # within `col_tol` of each other, share one left edge (the leftmost)
    parent = list(range(len(rec)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i_ in range(len(rec)):
        for j_ in range(i_ + 1, len(rec)):
            a_, b_ = rec[i_], rec[j_]
            # overlap is measured against the wider panel, so a full-width panel
            # does not chain every column of the row above into one
            inter = min(a_["px1"], b_["px1"]) - max(a_["px0"], b_["px0"])
            wide = max(a_["px1"] - a_["px0"], b_["px1"] - b_["px0"])
            if inter > 0.5 * wide or abs(a_["x0"] - b_["x0"]) < col_tol:
                parent[find(i_)] = find(j_)
    groups = {}
    for k, o in enumerate(rec):
        groups.setdefault(find(k), []).append(o)
    for col in groups.values():
        x = min(o["x0"] for o in col)
        for o in col:
            # a narrow panel under a wide one keeps its own edge rather than
            # borrowing one far to its left
            o["x"] = x if o["x0"] - x < max_shift else o["x0"]
    for o in rec:
        t = fig.text(o["x"], o["y"], o["letter"], fontsize=LETTER_SIZE, fontweight="bold",
                     ha="left", va="bottom")
        t._panel_letter = True
    fig._panel_labels = []


def save_fig(fig, root: Path, name: str) -> list[Path]:
    import os
    align_panel_labels(fig)
    if os.environ.get("FIG_OVERLAP_REPORT"):
        for msg in overlap_report(fig):
            print(f"OVERLAP {name}: {msg}")
    # FIG_OUT_DIR renders a separate copy (e.g. capital panel letters) without
    # overwriting figures/
    outdir = Path(os.environ["FIG_OUT_DIR"]) if os.environ.get("FIG_OUT_DIR") else root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def save_source(df: pd.DataFrame, root: Path, figure: str, panel: str) -> Path:
    import os
    d = (Path(os.environ["FIG_OUT_DIR"]) / "source_data" if os.environ.get("FIG_OUT_DIR")
         else root / "figures" / "source_data")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{figure}_{panel}.tsv"
    df.to_csv(p, sep="\t", index=False, float_format="%.6g")
    return p


ACRONYMS = {
    "Nadpplus": "NADP+", "Nadph": "NADPH", "Nadp": "NADP", "Nad": "NAD",
    "Perk": "PERK", "Atp": "ATP", "Dna": "DNA", "Rna": "RNA", "Mrna": "mRNA",
    "Tca": "TCA", "Ros": "ROS", "Tnf": "TNF", "Nfkb": "NF-κB", "Il": "IL",
    "Coa": "CoA", "Acyl": "acyl", "Eif2ak4": "EIF2AK4", "Gcn2": "GCN2",
    "Ucp1": "UCP1", "Iv": "IV", "Iii": "III", "Ii": "II", "I": "I",
    "Mhc": "MHC", "Hla": "HLA", "Nk": "NK", "Net": "NET", "Ifn": "IFN",
    "Gtpase": "GTPase", "Rho": "Rho", "Icosanoid": "eicosanoid",
}


PHRASES = {
    "unfolded protein response": "UPR",
    "reactive oxygen species": "ROS",
    "oxidative phosphorylation": "OXPHOS",
    "electron transport chain": "electron transport chain",
}

_ACR_LC = {k.lower(): v for k, v in ACRONYMS.items()}


def tidy_label(text: str, max_len: int | None = None) -> str:
    """Restore acronym casing in MSigDB-derived labels and optionally truncate."""
    out = str(text)
    for a, b in PHRASES.items():
        out = out.replace(a, b)
    words = [_ACR_LC.get(w.lower(), w) for w in out.split(" ")]
    out = " ".join(words)
    if out:
        out = out[0].upper() + out[1:]
    if max_len and len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def overlap_report(fig, point_pad: float = 1.0) -> list[str]:
    """Text that collides with other text or sits on plotted points.

    Every visible text artist (labels, annotations, legends, tick labels) is
    measured after drawing. Two texts whose boxes intersect, or a text box that
    covers a scatter point, are reported. Heatmap cell values are exempt because
    they are meant to sit on their cells.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.legend import Legend
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels exist for ticks outside the view as well; only the drawn ones count
    all_ticks, drawn = set(), set()
    for ax_ in fig.axes:
        for axis in (ax_.xaxis, ax_.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {id(tk.label1), id(tk.label2)}
            for tk in axis._update_ticks():
                drawn |= {id(tk.label1), id(tk.label2)}
    texts = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_ticks and id(t) not in drawn:
            continue
        if getattr(t, "_overlap_exempt", False):
            continue
        texts.append(t)
    # Text.get_window_extent on an Annotation measures the text alone; the
    # annotation's own method also spans its leader line
    boxes = [(t, mpl.text.Text.get_window_extent(t, r)) for t in texts]
    issues = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            if ti.get_rotation() or tj.get_rotation():
                continue        # axis-aligned boxes of rotated labels overlap falsely
            if bi.overlaps(bj) and bi.width > 0 and bj.width > 0:
                issues.append(f"text '{ti.get_text()[:30]}' overlaps '{tj.get_text()[:30]}'")
    legend_boxes = [lg.get_window_extent(r) for lg in fig.findobj(Legend)]
    for col in fig.findobj(PathCollection):
        if not col.get_visible() or col.axes is None or col.get_offsets().size == 0:
            continue
        pts = col.get_offset_transform().transform(np.asarray(col.get_offsets()))
        for t, b in boxes:
            if t.axes is not col.axes and t.figure is fig and t.axes is not None:
                continue
            inside = ((pts[:, 0] > b.x0 + point_pad) & (pts[:, 0] < b.x1 - point_pad) &
                      (pts[:, 1] > b.y0 + point_pad) & (pts[:, 1] < b.y1 - point_pad))
            if inside.any():
                issues.append(f"text '{t.get_text()[:30]}' covers {int(inside.sum())} point(s)")
        for b in legend_boxes:
            inside = ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) &
                      (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1))
            if inside.any() and col.get_label().startswith("_"):
                issues.append(f"legend covers {int(inside.sum())} point(s)")
    return sorted(set(issues))


def place_labels(ax, xs, ys, labels, fontsize: float = 5.4, italic: bool = False,
                 color: str = "#222222", min_r: float = 9, max_r: float = 70,
                 avoid_axes=None) -> list:
    """Annotate points with labels that land in empty space.

    For each label, candidate positions on rings of growing radius around its
    point are scored against every plotted point in the axes, every label already
    placed, the legend and the axes frame; the first position touching nothing is
    taken, otherwise the one covering the fewest points. A thin leader line joins
    label and point. Axes limits are frozen first, so later autoscaling cannot
    move the points out from under their labels.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Bbox
    fig = ax.figure
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = []
    for col in ax.collections:
        if isinstance(col, PathCollection) and col.get_offsets().size:
            pts.append(col.get_offset_transform().transform(np.asarray(col.get_offsets())))
    for ln in ax.lines:
        xy = ax.transData.transform(np.asarray(ln.get_xydata(), float))
        if len(xy) > 1:
            # sample along each segment so a label cannot sit across a line
            seg = [np.linspace(xy[i], xy[i + 1], 12) for i in range(len(xy) - 1)]
            xy = np.vstack(seg)
        if len(xy):
            pts.append(xy)
    from matplotlib.collections import LineCollection
    for lc in ax.collections:
        if isinstance(lc, LineCollection):
            for seg in lc.get_segments():
                if len(seg) > 1:
                    sxy = ax.transData.transform(np.asarray(seg, float))
                    pts.append(np.vstack([np.linspace(sxy[i], sxy[i + 1], 12)
                                          for i in range(len(sxy) - 1)]))
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    frame = ax.get_window_extent(r)
    blocked = [lg.get_window_extent(r) for lg in ([ax.get_legend()] if ax.get_legend() else [])]
    blocked += [mpl.text.Text.get_window_extent(t, r) for t in ax.texts if t.get_text().strip()]
    for other in (avoid_axes or []):
        blocked.append(other.get_window_extent(r))
    px = fig.dpi / 72.0
    placed, out = [], []
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    for x, y, lab in zip(xs, ys, labels):
        t = ax.text(0, 0, lab, fontsize=fontsize, fontstyle="italic" if italic else "normal")
        w, h = t.get_window_extent(r).width, t.get_window_extent(r).height
        t.remove()
        p0 = ax.transData.transform((x, y))
        best, best_cost = None, None
        for rad in np.arange(min_r, max_r + 1, 6):
            for a in angles:
                cx, cy = p0[0] + rad * px * np.cos(a), p0[1] + rad * px * np.sin(a)
                b = Bbox.from_bounds(cx - w / 2 - 1.5, cy - h / 2 - 1.5, w + 3, h + 3)
                if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
                    continue
                if any(b.overlaps(o) for o in placed + blocked):
                    continue
                cover = int(((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) &
                             (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)).sum())
                cost = cover * 1000 + rad
                if best_cost is None or cost < best_cost:
                    best, best_cost = (cx - p0[0], cy - p0[1], b), cost
            if best_cost is not None and best_cost < 1000:
                break
        if best is None:
            raise SystemExit(f"place_labels: no room for '{lab}'")
        dx, dy, b = best
        placed.append(b)
        out.append(ax.annotate(lab, (x, y), xytext=(dx / px, dy / px),
                               textcoords="offset points", ha="center", va="center",
                               fontsize=fontsize, color=color, zorder=7,
                               fontstyle="italic" if italic else "normal",
                               # an invisible plate around the text clips the leader
                               # line at its edge, so the line never runs through it
                               bbox=dict(boxstyle="square,pad=0.15", fc="none", ec="none"),
                               arrowprops=dict(arrowstyle="-", lw=0.35, color="#777777",
                                               shrinkA=0, shrinkB=1.5)))
    return out
