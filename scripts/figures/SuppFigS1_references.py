#!/usr/bin/env python
"""Supplementary Figure S1 - the stimulus reference panel: how similar the references are,
how well one stimulus agrees across studies, and leave-one-replicate-out recovery.

Reads results/06_attribution only.
Outputs: figures/SuppFigS1.pdf/.png, figures/source_data/SuppFigS1_<panel>.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R06 = ROOT / "results/06_attribution"
FIG = "SuppFigS1"
PANEL = "macrophage_resolution"


def read(p):
    return pd.read_csv(p, sep="\t")


def panel_corr(ax, cax):
    c = read(R06 / "reference_correlation.tsv")
    c = c[c.panel == PANEL]
    fs.save_source(c, ROOT, FIG, "a")
    refs = list(dict.fromkeys(list(c.reference_a) + list(c.reference_b)))
    M = pd.DataFrame(np.eye(len(refs)), index=refs, columns=refs)
    for _, r in c.iterrows():
        M.loc[r.reference_a, r.reference_b] = M.loc[r.reference_b, r.reference_a] = r.pearson_r
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(refs)):
        for j in range(len(refs)):
            if i != j and abs(M.values[i, j]) >= 0.9:
                t = ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=4,
                            color="white")
                t._overlap_exempt = True
    lab = [fs.REF_LABELS.get(r, r) for r in refs]
    ax.set_xticks(range(len(refs)), lab, rotation=60, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(refs)), lab, fontsize=5.5)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Pearson r", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
    ax.set_title("Reference similarity (r ≥ 0.90 printed)", loc="left")


def panel_cross(ax):
    x = read(R06 / "cross_study_similarity.tsv")
    x["kind"] = np.where(x.relation.str.startswith("same stimulus"), "Same stimulus,\nother study",
                         "Other stimulus,\nsame study")
    x["tissue"] = np.where(x.reference_a.str.startswith("liv_") | x.reference_b.str.startswith("liv_"),
                           "liver", "macrophage")
    fs.save_source(x, ROOT, FIG, "c")
    rng = np.random.default_rng(3)
    groups = [("Same stimulus,\nother study", "macrophage"), ("Same stimulus,\nother study", "liver"),
              ("Other stimulus,\nsame study", "macrophage"), ("Other stimulus,\nsame study", "liver")]
    for k, (kind, tis) in enumerate(groups):
        d = x[(x.kind == kind) & (x.tissue == tis)]
        col = fs.KC_BLUE if tis == "macrophage" else "#937860"
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(d)), d.pearson_r, s=8, color=col, lw=0,
                   alpha=0.85)
        ax.plot([k - 0.25, k + 0.25], [d.pearson_r.median()] * 2, color="#333333", lw=0.9)
    ax.set_xticks(range(4), ["Same,\nother\nstudy", "Same,\nvs\nliver",
                             "Other,\nsame\nstudy", "Other,\nliver\nstudy"], fontsize=5.5)
    ax.set_xlabel("Stimulus pair")
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.set_ylim(-0.45, 1.05)
    ax.set_ylabel("Pearson r")
    ax.text(3.45, -0.42, "blue: macrophage\nbrown: liver", ha="right", va="bottom", fontsize=5,
            color="#444444")
    ax.set_title("Study versus stimulus", loc="left")


def panel_loro(ax):
    lo = read(R06 / "controls_loro.tsv")
    lo = lo[lo.panel == PANEL]
    fs.save_source(lo, ROOT, FIG, "b")
    s = lo.groupby("reference").self_share.median().sort_values()
    y = np.arange(len(s))
    for yi, ref in zip(y, s.index):
        d = lo[lo.reference == ref]
        ax.scatter(d.self_share, [yi] * len(d), s=7, lw=0,
                   color=np.where(d.top_is_self, "#555555", fs.CLASS_COLORS["PAMP"]))
        ax.plot([s[ref]] * 2, [yi - 0.3, yi + 0.3], color="#333333", lw=0.8)
    ax.set_yticks(y, [fs.REF_LABELS.get(r, r) for r in s.index], fontsize=5.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share returned to the held-out reference")
    ax.text(0.02, len(s) - 0.6, "red: another reference ranked first", fontsize=5,
            color=fs.CLASS_COLORS["PAMP"], va="top")
    ax.set_title("Leave-one-replicate-out", loc="left")


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(177 * fs.MM, 150 * fs.MM))
    a = fig.add_axes([0.15, 0.27, 0.45, 0.66]); cax = fig.add_axes([0.615, 0.5, 0.01, 0.2])
    panel_corr(a, cax)
    c = fig.add_axes([0.80, 0.52, 0.185, 0.41]); panel_loro(c)
    b = fig.add_axes([0.73, 0.08, 0.255, 0.27]); panel_cross(b)
    for ax, l in zip([a, c, b], "abc"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
