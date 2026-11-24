#!/usr/bin/env python
"""Supplementary Figure S2 - attribution to each single reference (LPS and disulfide HMGB1 fitted
as separate references) for the targets of Figure 5 and the LPS liver time course.

Reads results/06_attribution only.
Outputs: figures/SuppFigS2.pdf/.png, figures/source_data/SuppFigS2_a.tsv
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
FIG = "SuppFigS2"
PANEL = "macrophage_resolution_unmerged"
TARGETS = {
    "KC_CLP_2h": "KC, 2 h", "KC_CLP_8h": "KC, 8 h", "CD45nonKC_CLP_8h": "CD45$^+$ non-KC, 8 h",
    "LSEC_CLP_8h": "LSEC, 8 h", "HSC_CLP_8h": "Stellate, 8 h", "HEP_CLP_8h": "Hepatocyte, 8 h",
    "HEP_FSP_GSE303231": "Hepatocyte, FSP", "sc_all_Kupffer": "Kupffer cells (sc)",
    "sc_myeloid_Recruited Mo-macrophages": "Recruited macrophages (sc)",
    "liver_proteome_CLP_24h": "Liver proteome, 24 h",
    "LPS_LPS_d0.25_LPS_timecourse": "LPS liver, 6 h", "LPS_LPS_d0.5_LPS_timecourse": "LPS liver, 12 h",
    "LPS_LPS_d1_LPS_timecourse": "LPS liver, 1 d", "LPS_LPS_d2_LPS_timecourse": "LPS liver, 2 d",
    "LPS_LPS_d3_LPS_timecourse": "LPS liver, 3 d",
}


def main():
    fs.apply_style()
    a = pd.read_csv(R06 / "attribution.tsv", sep="\t")
    a = a[(a.panel == PANEL) & a.target.isin(TARGETS)].copy()
    fs.save_source(a[["target", "panel", "reference", "stimulus_class", "share", "share_ci_low",
                      "share_ci_high", "boot_nonzero", "r2_total"]], ROOT, FIG, "a")
    cls_order = ["PAMP", "DAMP", "cytokine", "resolution"]
    refs = (a.drop_duplicates("reference").assign(o=lambda d: d.stimulus_class.map(
        {c: i for i, c in enumerate(cls_order)})).sort_values(["o"], kind="stable").reference.tolist())
    M = a.pivot_table(index="reference", columns="target", values="share").reindex(
        index=refs, columns=list(TARGETS))
    fig = plt.figure(figsize=(177 * fs.MM, 120 * fs.MM))
    ax = fig.add_axes([0.2, 0.25, 0.68, 0.68])
    im = ax.imshow(M.values, cmap="Greys", vmin=0, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if v >= 0.05:
                t = ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                            color="white" if v > 0.4 else "#222222")
                t._overlap_exempt = True
    ax.set_xticks(range(M.shape[1]), [TARGETS[t] for t in M.columns], rotation=45, ha="right")
    cls = a.drop_duplicates("reference").set_index("reference").stimulus_class
    ax.set_yticks(range(M.shape[0]), [fs.REF_LABELS.get(r, r) for r in M.index])
    for tl, r in zip(ax.get_yticklabels(), M.index):
        tl.set_color(fs.CLASS_COLORS[cls[r]])
    ax.tick_params(length=0)
    ax.axvline(9.5, color="#333333", lw=1)
    cax = fig.add_axes([0.9, 0.55, 0.012, 0.25])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Share (≥ 0.05 printed)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("Share on each reference", loc="left")
    # key to the stimulus class of each row label
    h = [plt.Rectangle((0, 0), 1, 1, color=fs.CLASS_COLORS[c]) for c in cls_order]
    fig.legend(h, [fs.CLASS_LABELS[c] for c in cls_order], title="Row label: stimulus class",
               title_fontsize=5.5, loc="upper left", bbox_to_anchor=(0.895, 0.5), frameon=False,
               fontsize=5.5, handlelength=0.9, handletextpad=0.3, alignment="left")
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
