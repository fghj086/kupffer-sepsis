#!/usr/bin/env python
"""Figure 5 - across liver cell types, the septic response resolves into PAMP, cytokine and
tolerance-associated components with no separable HMGB1-like share; cytokine loss-of-function
shifts NF-kB, KC identity and acute-phase scores; the tolerance references improve prediction of
held-out genes in macrophage targets.

Reads results/06_attribution, results/04_timecourse and results/09_identifiability only.
Outputs: figures/Fig5.pdf, figures/Fig5.png, figures/source_data/Fig5_<panel>.tsv
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
R04 = ROOT / "results/04_timecourse"
R06 = ROOT / "results/06_attribution"
R09 = ROOT / "results/09_identifiability"
FIG = "Fig5"
PANEL = "macrophage_resolution"   # the one panel fitted to every target
CLASSES = ["PAMP", "mixed", "DAMP", "cytokine", "resolution"]
TARGETS = {
    "KC_CLP_2h": ("KC, 2 h", "Sorted cells"),
    "KC_CLP_8h": ("KC, 8 h", "Sorted cells"),
    "CD45nonKC_CLP_8h": ("CD45$^+$ non-KC, 8 h", "Sorted cells"),
    "LSEC_CLP_8h": ("LSEC, 8 h", "Sorted cells"),
    "HSC_CLP_8h": ("Stellate, 8 h", "Sorted cells"),
    "HEP_CLP_8h": ("Hepatocyte, 8 h", "Sorted cells"),
    "HEP_FSP_GSE303231": ("Hepatocyte, FSP", "Sorted cells"),
    "sc_all_Kupffer": ("Kupffer cells", "Single cell"),
    "sc_myeloid_Recruited Mo-macrophages": ("Recruited macrophages", "Single cell"),
    "liver_proteome_CLP_24h": ("Liver proteome, 24 h", "Protein"),
}
BLOCK = {  # target perturbation -> label, challenge
    "LPSantiTNF": ("Anti-TNF", "LPS"), "LPSTNFKO": ("TNF KO", "LPS"),
    "LPSIRF1KO": ("IRF1 KO", "LPS"), "LPSSTAT1KO": ("STAT1 KO", "LPS"),
    "LPSIFNgKOantiTNF": ("IFN-γ KO + anti-TNF", "LPS"),
    "LPSIL1bKOantiTNF": ("IL-1β KO + anti-TNF", "LPS"),
    "LPSIL18KOantiTNF": ("IL-18 KO + anti-TNF", "LPS"),
    "antiTNF": ("Anti-TNF", "CLP"), "TNFKO": ("TNF KO", "CLP"),
    "IFNgKOantiTNF": ("IFN-γ KO + anti-TNF", "CLP"),
    "IL1bKOantiTNF": ("IL-1β KO + anti-TNF", "CLP"),
    "IL18KOantiTNF": ("IL-18 KO + anti-TNF", "CLP"),
}
SCORES = {"nfkb_inflammatory": "NF-κB inflammatory", "kc_identity": "KC identity",
          "hepatocyte_acute_phase": "Acute phase"}


def read(p):
    return pd.read_csv(p, sep="\t")


def panel_classes(ax):
    b = read(R06 / "attribution_by_class.tsv")
    b = b[(b.panel == PANEL) & b.target.isin(TARGETS)].copy()
    fs.save_source(b, ROOT, FIG, "a")
    W = b.pivot_table(index="target", columns="stimulus_class", values="share").reindex(
        index=list(TARGETS), columns=CLASSES).fillna(0)
    r2 = b.drop_duplicates("target").set_index("target").r2_total.reindex(W.index)
    y = np.arange(len(W))[::-1]
    left = np.zeros(len(W))
    for c in CLASSES:
        ax.barh(y, W[c].values, left=left, color=fs.CLASS_COLORS[c], height=0.72,
                label=fs.CLASS_LABELS[c], edgecolor="white", lw=0.3)
        left += W[c].values
    for yi, v in zip(y, r2.values):
        ax.text(1.03, yi, f"{v:.2f}", va="center", fontsize=5.5, color="#444444")
    ax.text(1.03, len(W) - 0.35, "R²", va="bottom", fontsize=5.5, color="#444444")
    ax.set_yticks(y, [TARGETS[t][0] for t in W.index])
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.6, len(W) - 0.4)
    ax.set_xlabel("Share of fitted response")
    for g_end in (2.5, 0.5):   # separators between data types
        ax.axhline(g_end, color="#333333", lw=0.4, xmin=-0.55, clip_on=False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.01), ncol=3, frameon=False, fontsize=5.5,
              handlelength=1.0, columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
    ax.set_title("Stimulus classes by target", loc="left", pad=26)


def panel_terms(ax):
    a = read(R06 / "attribution.tsv")
    a = a[(a.panel == f"{PANEL}_unmerged") & a.target.isin(TARGETS)
          & a.reference.isin(["LPS_24h", "dsHMGB1", "TissueHom_6h"])].copy()
    fs.save_source(a[["target", "panel", "reference", "share", "share_ci_low", "share_ci_high",
                      "boot_nonzero", "r2_total"]], ROOT, FIG, "b")
    refs = {"LPS_24h": (fs.CLASS_COLORS["PAMP"], "o", "LPS"),
            "dsHMGB1": (fs.CLASS_COLORS["DAMP"], "D", "Disulfide HMGB1"),
            "TissueHom_6h": ("#8C5A2B", "s", "Tissue homogenate")}
    y0 = np.arange(len(TARGETS))[::-1]
    for k, (ref, (col, mk, lab)) in enumerate(refs.items()):
        d = a[a.reference == ref].set_index("target").reindex(list(TARGETS))
        y = y0 + (1 - k) * 0.22
        ax.errorbar(d.share, y, xerr=[d.share - d.share_ci_low, d.share_ci_high - d.share],
                    fmt=mk, ms=2.8, color=col, mfc=col if ref != "dsHMGB1" else "white",
                    lw=0.6, capsize=0, label=lab)
    ax.set_yticks(y0, [""] * len(TARGETS))
    ax.set_ylim(-0.6, len(TARGETS) - 0.4)
    ax.set_xlim(-0.02, 0.55)
    ax.axvline(0, color="#333333", lw=0.4)
    for g_end in (2.5, 0.5):
        ax.axhline(g_end, color="#333333", lw=0.4)
    ax.set_xlabel("Share, single reference (95% CI)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), frameon=False, fontsize=5.5,
              handletextpad=0.2, borderaxespad=0, ncol=1, labelspacing=0.25)
    ax.set_title("LPS and HMGB1 fitted apart", loc="left", pad=36)


def panel_block(axes):
    p = read(R04 / "perturbation_effects.tsv")
    p = p[(p.comparison == "blockade_vs_stimulus") & p.perturbation.isin(BLOCK)
          & p.score.isin(SCORES)].copy()
    fs.save_source(p[["file_key", "perturbation", "description", "reference", "score", "n_test",
                      "n_ref", "diff_log2", "diff_ci_low", "diff_ci_high", "hedges_g", "p_welch",
                      "fdr"]], ROOT, FIG, "c")
    order = list(BLOCK)
    y = np.arange(len(order))[::-1].astype(float)
    y[[BLOCK[o][1] == "CLP" for o in order]] -= 0.6   # gap between challenge models
    for ax, (sc, lab) in zip(axes, SCORES.items()):
        d = p[p.score == sc].set_index("perturbation").loc[order]
        cols = [fs.CLASS_COLORS["PAMP"] if BLOCK[o][1] == "LPS" else "#333333" for o in order]
        for yi, (o, row), col in zip(y, d.iterrows(), cols):
            filled = row.fdr < 0.05
            ax.errorbar(row.diff_log2, yi, xerr=[[row.diff_log2 - row.diff_ci_low],
                                                 [row.diff_ci_high - row.diff_log2]],
                        fmt="o", ms=3, color=col, mfc=col if filled else "white", lw=0.6,
                        capsize=0)
        ax.axvline(0, color="#333333", lw=0.4)
        ax.set_xlabel(f"{lab}\n(log$_2$ change)")
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_xlim(-2.6, 2.6); ax.set_xticks([-2, 0, 2])
    axes[0].set_yticks(y, [BLOCK[o][0] for o in order])
    for ax in axes[1:]:
        ax.set_yticks(y, [""] * len(order))
    n_lps = sum(BLOCK[o][1] == "LPS" for o in order)
    tr = axes[-1].get_yaxis_transform()
    axes[-1].text(1.04, y[0] + 0.3, "LPS", ha="left", va="top", fontsize=5.5,
                  color=fs.CLASS_COLORS["PAMP"], transform=tr)
    axes[-1].text(1.04, y[n_lps] + 0.3, "CLP,\nsevere", ha="left", va="top", fontsize=5.5,
                  color="#333333", transform=tr, linespacing=1.0)
    for t, o in zip(axes[0].get_yticklabels(), order):
        t.set_color(fs.CLASS_COLORS["PAMP"] if BLOCK[o][1] == "LPS" else "#333333")
    axes[0].set_title("Cytokine loss of function vs challenge alone (filled: FDR < 0.05)",
                      loc="left")


TOL_TARGETS = {"KC_CLP_2h": "KC, 2 h", "KC_CLP_8h": "KC, 8 h",
               "CD45nonKC_CLP_8h": "CD45$^+$ non-KC, 8 h", "sc_all_Kupffer": "Kupffer cells (sc)",
               "sc_myeloid_Recruited Mo-macrophages": "Recruited\nmacrophages (sc)"}
TOL_CMP = {"P2": "Tolerance references added",
           "tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)": "Beyond acute KC LPS response"}


def panel_tolerance(ax):
    """Held-out-gene R2 gain from adding the two tolerance references."""
    r = read(R09 / "panel_sensitivity_heldout_r2.tsv")
    r = r[r.target.isin(TOL_TARGETS) & r.comparison.isin(TOL_CMP)].copy()
    fs.save_source(r[["target", "cell", "time_h", "n_genes", "comparison", "heldout_r2",
                      "delta_heldout_r2", "delta_sd", "frac_repeats_delta_gt0", "null_delta_mean",
                      "null_delta_q95", "p_null_ge_observed"]], ROOT, FIG, "d")
    order = list(TOL_TARGETS)
    y0 = np.arange(len(order))[::-1].astype(float)
    sty = {"P2": ("o", fs.CLASS_COLORS["resolution"], fs.CLASS_COLORS["resolution"], 0.14),
           "tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)":
               ("D", fs.CLASS_COLORS["resolution"], "white", -0.14)}
    for cmp_, (mk, col, mfc, off) in sty.items():
        d = r[r.comparison == cmp_].set_index("target").reindex(order)
        ax.errorbar(d.delta_heldout_r2, y0 + off, xerr=d.delta_sd, fmt=mk, ms=3, color=col,
                    mfc=mfc, mec=col, mew=0.6, lw=0.6, capsize=0, zorder=3)
    nul = r[r.comparison == "P2"].set_index("target").reindex(order)
    for yi, q in zip(y0, nul.null_delta_q95):
        ax.plot([q, q], [yi + 0.02, yi + 0.3], color="#222222", lw=0.9, zorder=4)
    ax.axvline(0, color="#333333", lw=0.4)
    ax.set_yticks(y0, [TOL_TARGETS[t] for t in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.005, 0.1); ax.set_xticks([0, 0.05, 0.1])
    ax.set_xlabel("Gain in held-out-gene R²")
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=fs.CLASS_COLORS["resolution"],
                    mec=fs.CLASS_COLORS["resolution"]),
         plt.Line2D([], [], ls="", marker="D", ms=3, mfc="white",
                    mec=fs.CLASS_COLORS["resolution"], mew=0.6),
         plt.Line2D([], [], ls="", marker="|", ms=5, mec="#222222", mew=0.9)]
    ax.legend(h, list(TOL_CMP.values()) + ["Permutation null, 95th percentile"],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=5,
              handletextpad=0.3, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("Tolerance references", loc="left", pad=30)


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(177 * fs.MM, 140 * fs.MM))
    a = fig.add_axes([0.2, 0.56, 0.36, 0.33]); panel_classes(a)
    b = fig.add_axes([0.66, 0.56, 0.31, 0.33]); panel_terms(b)
    c1 = fig.add_axes([0.2, 0.08, 0.135, 0.33])
    c2 = fig.add_axes([0.355, 0.08, 0.135, 0.33])
    c3 = fig.add_axes([0.51, 0.08, 0.135, 0.33])
    panel_block([c1, c2, c3])
    d = fig.add_axes([0.82, 0.08, 0.165, 0.26]); panel_tolerance(d)
    for ax, l in zip([a, b, c1, d], "abcd"):
        fs.panel_label(ax, l)
    # panel a's separators reach left of its tick labels; c shares that column
    fs.align_panel_labels(fig, col_tol=0.1)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
