#!/usr/bin/env python
"""Supplementary Figure S3 - robustness of the tolerance-associated component.

(a) Per-mouse gain in held-out-gene R² when the two tolerance references are added (left) and
    when they are added beyond the acute KC LPS reference (right): each CLP mouse vs the sham
    mean (filled) and each sham mouse vs the other shams (open; noise control), with the
    95th percentile of the gene-permutation null for that mouse (tick). Sorted KC 2 h
    (GSE328774), KC 8 h and CD45+ non-KC 8 h (GSE311736); gene universe as Figure 5d.
(b) Gain from adding each single real reference to the base panel, per target (gene universe
    covering every candidate reference), with the two tolerance references highlighted and
    dexamethasone, IL-10 and the acute KC LPS reference marked.

Reads results/10_sepsis_specificity only.
Outputs: figures/SuppFigS3.pdf, figures/SuppFigS3.png, figures/source_data/SuppFigS3_<panel>.tsv
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
R10 = ROOT / "results/10_sepsis_specificity"
FIG = "SuppFigS3"
TOL = fs.CLASS_COLORS["resolution"]

UNITS = {"KC_2h": "KC, 2 h", "KC_8h": "KC, 8 h", "CD45nonKC_8h": "CD45$^+$ non-KC, 8 h"}
CMP = {"tolerance step (P2 vs P1)": "Tolerance references added",
       "tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)": "Beyond acute KC LPS response"}
CONTRAST = {"CLP_vs_sham_mean": ("CLP mouse vs sham mean", 0.16, TOL),
            "sham_vs_other_sham (noise control)": ("Sham mouse vs other shams", -0.16, "white")}
TARGETS = {"KC_CLP_2h": "KC, 2 h", "KC_CLP_8h": "KC, 8 h",
           "CD45nonKC_CLP_8h": "CD45$^+$ non-KC, 8 h", "sc_all_Kupffer": "Kupffer cells (sc)",
           "sc_myeloid_Recruited Mo-macrophages": "Recruited macrophages (sc)"}
MARK = {  # reference -> label, marker, face, edge, size, row offset
    "KC_TolLPS_6h": ("LPS-tolerant KC", "o", TOL, TOL, 3.6, 0.36),
    "Tolerance_shift": ("LPS tolerance", "D", TOL, TOL, 3.0, 0.24),
    "KC_LPS_6h": ("LPS, cultured KC", "v", "#C44E52", "#C44E52", 3.6, -0.12),
    "Dex": ("Dexamethasone", "^", "#8172B3", "#8172B3", 3.6, -0.22),
    "Dex_on_LPS": ("Dexamethasone on LPS", "^", "white", "#8172B3", 3.6, -0.32),
    "IL10": ("IL-10", "s", "#55A868", "#55A868", 3.0, -0.42),
}


def read(p):
    return pd.read_csv(p, sep="\t")


# ---------------------------------------------------------------- a
def panel_a(axs):
    d = read(R10 / "tolerance_per_mouse.tsv")
    d = d[(d.universe == "stage09") & d.comparison.isin(CMP)]
    s = read(R10 / "tolerance_per_mouse_summary.tsv")
    s = s[(s.universe == "stage09") & s.comparison.isin(CMP)]
    fs.save_source(d[["unit", "library", "contrast", "comparison", "n_genes", "heldout_r2_P1",
                      "gain_mean", "gain_sd", "null_q95", "p_null_ge_observed",
                      "above_null_q95"]], ROOT, FIG, "a")
    fs.save_source(s[["unit", "contrast", "comparison", "n_mice", "n_gain_positive",
                      "n_above_null_q95", "gain_min", "gain_median", "gain_max"]],
                   ROOT, FIG, "a_summary")
    y0 = {u: len(UNITS) - 1 - i for i, u in enumerate(UNITS)}
    for k, (ax, cmp_) in enumerate(zip(axs, CMP)):
        for (u, con), g in d[d.comparison == cmp_].groupby(["unit", "contrast"]):
            lab, off, fc = CONTRAST[con]
            y = y0[u] + off + np.linspace(-0.07, 0.07, len(g))
            ax.errorbar(g.gain_mean, y, xerr=g.gain_sd, fmt="o", ms=3, mfc=fc, mec=TOL,
                        mew=0.6, color=TOL, lw=0.5, capsize=0, zorder=3)
            ax.scatter(g.null_q95, y, marker="|", s=14, color="#222222", lw=0.7, zorder=4)
            n = s[(s.unit == u) & (s.contrast == con) & (s.comparison == cmp_)].iloc[0]
            ax.text(1.02, y0[u] + off, f"{int(n.n_above_null_q95)}/{int(n.n_mice)}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.2)
        ax.axvline(0, color="#333333", lw=0.4)
        for yy in (0.5, 1.5):
            ax.axhline(yy, color="#E0E0E0", lw=0.5)
        ax.set_xlim(-0.004, 0.036); ax.set_xticks([0, 0.01, 0.02, 0.03])
        ax.set_ylim(-0.5, len(UNITS) - 0.5)
        ax.set_xlabel("Gain in held-out-gene R²")
        ax.set_title(CMP[cmp_], loc="left", fontsize=6)
        ax.text(1.02, len(UNITS) - 0.45, "Above\nnull", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=5, color="#555555", linespacing=1.05)
        if k == 0:
            ax.set_yticks(list(y0.values()), list(UNITS.values()))
        else:
            ax.set_yticks(list(y0.values()), [])
            ax.tick_params(left=False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=v[2], mec=TOL, mew=0.6)
         for v in CONTRAST.values()]
    h.append(plt.Line2D([], [], ls="", marker="|", ms=5, mec="#222222", mew=0.8))
    axs[0].legend(h, [v[0] for v in CONTRAST.values()] + ["Permutation null, 95th percentile"],
                  loc="lower left", bbox_to_anchor=(-0.02, 1.12), ncol=3, frameon=False,
                  fontsize=5.2, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)


# ---------------------------------------------------------------- b
def panel_b(ax):
    g = read(R10 / "tolerance_real_reference_gains.tsv")
    g = g[(g.universe == "union") & (g.n_added == 1) & g.target.isin(TARGETS)]
    r = read(R10 / "tolerance_real_reference_rank.tsv")
    r = r[(r.universe == "union") & r.target.isin(TARGETS)].set_index("target")
    fs.save_source(g[["target", "cell", "universe", "n_genes", "heldout_r2_P1", "added",
                      "is_tolerance", "reference_class", "gain_mean", "gain_sd",
                      "frac_repeats_gt0"]], ROOT, FIG, "b")
    fs.save_source(r.reset_index()[["target", "universe", "n_candidates", "best_nontolerance",
                                    "best_nontolerance_gain", "gain_KC_TolLPS_6h",
                                    "rank_KC_TolLPS_6h_among_singles", "gain_Tolerance_shift",
                                    "rank_Tolerance_shift_among_singles"]], ROOT, FIG, "b_rank")
    rng = np.random.default_rng(5)
    y0 = {t: len(TARGETS) - 1 - i for i, t in enumerate(TARGETS)}
    for t, d in g.groupby("target"):
        rest = d[~d.added.isin(MARK)]
        ax.scatter(rest.gain_mean, y0[t] + 0.05 + rng.uniform(-0.06, 0.06, len(rest)), s=5,
                   color="#C8C8C8", lw=0, zorder=2)
        for ref, (lab, mk, fc, ec, ms, off) in MARK.items():
            v = d[d.added == ref].gain_mean
            ax.plot(v, [y0[t] + off] * len(v), ls="", marker=mk, ms=ms, mfc=fc, mec=ec,
                    mew=0.6, zorder=4 if ref in ("KC_TolLPS_6h", "Tolerance_shift") else 3)
        rr = r.loc[t]
        ax.text(1.02, y0[t], f"{int(rr.rank_KC_TolLPS_6h_among_singles)}, "
                f"{int(rr.rank_Tolerance_shift_among_singles)}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.2)
    n = int(r.n_candidates.iloc[0]) + 1
    ax.text(1.02, len(TARGETS) - 0.45, f"Rank of the\ntolerance\nreferences\n(each of {n})",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5,
            color="#555555", linespacing=1.05)
    ax.axvline(0, color="#333333", lw=0.4)
    for yy in np.arange(0.5, len(TARGETS) - 1, 1.0):
        ax.axhline(yy, color="#E0E0E0", lw=0.5)
    ax.set_yticks(list(y0.values()), list(TARGETS.values()))
    ax.set_ylim(-0.62, len(TARGETS) - 0.5)
    ax.set_xlim(-0.006, 0.122); ax.set_xticks([0, 0.03, 0.06, 0.09, 0.12])
    ax.set_xlabel("Gain in held-out-gene R² from one added reference")
    h = [plt.Line2D([], [], ls="", marker=v[1], ms=v[4], mfc=v[2], mec=v[3], mew=0.6)
         for v in MARK.values()]
    h.append(plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="#C8C8C8", mec="#C8C8C8"))
    ax.legend(h, [v[0] for v in MARK.values()] + ["Other references"], loc="center left",
              bbox_to_anchor=(1.2, 0.5), frameon=False, fontsize=5.2, handletextpad=0.3,
              borderaxespad=0.1, labelspacing=0.6)


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(180 * fs.MM, 140 * fs.MM))
    a1 = fig.add_axes([0.16, 0.62, 0.3, 0.26])
    a2 = fig.add_axes([0.57, 0.62, 0.3, 0.26])
    panel_a([a1, a2])
    b = fig.add_axes([0.16, 0.08, 0.5, 0.38]); panel_b(b)
    for ax, l in zip([a1, b], "ab"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
