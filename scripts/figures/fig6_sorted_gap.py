"""Figure 6c - Kupffer cells vs CD45+ non-KC cells in sorted liver, sham and CLP 8 h (GSE311736).

Per library: scavenger-receptor and identity scores (gene sets free of annotation genes; mean
gene-centred log2 normalised counts, both sorted fractions on one scale). Group means are joined
from sham to CLP; the KC − CD45+ non-KC gap is printed for each condition, with the change in the
gap (CLP − sham, Welch 95% CI).

Reads results/10_sepsis_specificity only (sorted_library_scores.tsv,
sorted_kc_vs_cd45_interaction.tsv, sorted_within_fraction_change.tsv).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

R10 = Path(__file__).resolve().parents[2] / "results/10_sepsis_specificity"
SCORES = {"clearance_strict": "Scavenger receptors",
          "identity_not_annotation": "KC identity"}
FRACTIONS = {"KC": ("Kupffer cells", fs.KC_BLUE), "CD45_nonKC": ("CD45$^+$ non-KC", "#8C8C8C")}
CONDS = ["sham", "CLP"]


def _sgn(v):
    return f"{v:+.2f}".replace("-", "−")


def _num(v):
    return f"{v:.2f}".replace("-", "−")


def load(root: Path, fig: str):
    s = pd.read_csv(R10 / "sorted_library_scores.tsv", sep="\t")
    s = s[s.dataset == "GSE311736"]
    i = pd.read_csv(R10 / "sorted_kc_vs_cd45_interaction.tsv", sep="\t")
    i = i[i.score.isin(SCORES) & (i.analysis == "unpaired (libraries independent)")]
    w = pd.read_csv(R10 / "sorted_within_fraction_change.tsv", sep="\t")
    w = w[w.score.isin(SCORES) & w.unit.str.endswith("GSE311736")]
    fs.save_source(s[["sample", "cell_type", "condition", "replicate"] + list(SCORES)],
                   root, fig, "c_libraries")
    fs.save_source(i[["score", "analysis", "gap_sham", "gap_clp", "delta_disease",
                      "welch_ci_low", "welch_ci_high", "p_welch_contrast"]], root, fig, "c_gap")
    fs.save_source(w[["unit", "score", "n_case", "n_ref", "mean_case", "mean_ref", "diff",
                      "welch_ci_low", "welch_ci_high", "boot_ci_low", "boot_ci_high"]],
                   root, fig, "c_within")
    return s, i


def panel(axs, root: Path, fig: str):
    s, gi = load(root, fig)
    rng = np.random.default_rng(11)
    for k, (ax, sc) in enumerate(zip(axs, SCORES)):
        means = {}
        for ct, (lab, col) in FRACTIONS.items():
            m = []
            for x, cond in enumerate(CONDS):
                v = s[(s.cell_type == ct) & (s.condition == cond)][sc].values
                xx = x + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(xx, v, s=9, facecolor="white" if cond == "sham" else col,
                           edgecolor=col, lw=0.6, zorder=3)
                m.append(v.mean())
            ax.plot([0, 1], m, color=col, lw=1.0, zorder=2)
            for x in (0, 1):
                ax.plot([x - 0.17, x + 0.17], [m[x]] * 2, color=col, lw=1.2, zorder=4)
            means[ct] = m
        r = gi[gi.score == sc].iloc[0]
        # gap brackets outside the points: sham on the left, CLP on the right
        for x, gap, side in ((0, r.gap_sham, -1), (1, r.gap_clp, 1)):
            xb = x + side * 0.3
            lo, hi = means["CD45_nonKC"][x], means["KC"][x]
            ax.plot([xb, xb], [lo, hi], color="#333333", lw=0.6)
            for yy in (lo, hi):
                ax.plot([xb, xb - side * 0.06], [yy, yy], color="#333333", lw=0.6)
            ax.text(xb + side * 0.06, (lo + hi) / 2, _num(gap), rotation=90, ha="center",
                    va="center", fontsize=5.2)
        ax.text(0.5, 0.0, f"Change in gap\n{_sgn(r.delta_disease)} "
                f"({_num(r.welch_ci_low)} to {_num(r.welch_ci_high)})",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5, linespacing=1.1)
        ax.set_xticks([0, 1], ["Sham", "CLP"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-4.3, 3.4); ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.axhline(0, color=fs.GREY, lw=0.4, ls=":", zorder=0)
        ax.set_title(SCORES[sc], loc="left", fontsize=6)
        if k == 0:
            ax.set_ylabel("Score (relative log$_2$ expression)")
        else:
            ax.set_yticklabels([])
    h = [plt.Line2D([], [], color=col, lw=1.0, marker="o", ms=3, mfc=col, mec=col)
         for _, col in FRACTIONS.values()]
    axs[0].legend(h, [v[0] for v in FRACTIONS.values()], loc="lower left",
                  bbox_to_anchor=(-0.05, 1.08), ncol=2, frameon=False, fontsize=5.2,
                  handlelength=1.4, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
