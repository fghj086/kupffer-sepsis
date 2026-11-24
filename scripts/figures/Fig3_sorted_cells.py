#!/usr/bin/env python
"""Figure 3 - the early Kupffer-cell response is a change of state, not of identity.

Reads results/02_sorted_bulk only.
Outputs: figures/Fig3.pdf, figures/Fig3.png, figures/source_data/Fig3_<panel>.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R02 = ROOT / "results/02_sorted_bulk"
FIG = "Fig3"

PROGRAMS = {"inflammation_hallmark_tnfa_nfkb": "TNF/\nNF-κB",
            "clearance_curated": "Clear-\nance",
            "kc_identity": "KC\nidentity",
            "monocyte_recruited": "Recruited\nmarkers"}
PROG_COL = {"inflammation_hallmark_tnfa_nfkb": "#333333", "clearance_curated": "#8172B3",
            "kc_identity": fs.KC_BLUE, "monocyte_recruited": fs.REC_GREEN}
UNITS = {"GSE328774_KC": "KC 2 h", "GSE311736_KC": "KC 8 h",
         "GSE311736_CD45_nonKC": "CD45$^+$ non-KC 8 h", "GSE311736_LSEC": "LSEC 8 h",
         "GSE311736_HSC": "Stellate 8 h", "GSE311736_HEP": "Hepatocyte 8 h",
         "GSE303231_HEP": "Hepatocyte, FSP"}
HEAT = {"inflammation_hallmark_tnfa_nfkb": "TNF/NF-κB",
        "inflammation_hallmark_inflammatory": "Inflammatory response",
        "chemokines_cytokines": "Chemokines/cytokines",
        "acute_phase": "Acute phase", "hepatocyte_stress": "Cell stress",
        "clearance_curated": "Clearance", "go_phagocytosis": "Phagocytosis",
        "reactome_scavenger": "Scavenger receptors", "kc_identity": "KC identity",
        "monocyte_recruited": "Recruited markers"}
NORM = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4)
CMAP = plt.cm.RdBu_r
GENES = ["Hmgb1", "Ager", "Tlr4", "Ly96", "Cd14", "Myd88", "Casp4", "Gsdmd"]


def read(p):
    return pd.read_csv(p, sep="\t")


def panel_per_mouse(ax, unit, title):
    p = read(R02 / "program_scores.tsv")
    p = p[(p.unit == unit) & p.program.isin(PROGRAMS)].copy()
    fs.save_source(p[["unit", "sample", "condition", "program", "score_log2"]], ROOT, FIG,
                   f"a_{UNITS[unit].split()[1]}h")
    rng = np.random.default_rng(1)
    for i, prog in enumerate(PROGRAMS):
        for j, cond in enumerate(["sham", "CLP"]):
            v = p[(p.program == prog) & (p.condition == cond)].score_log2.values
            x = i + (j - 0.5) * 0.36 + rng.uniform(-0.06, 0.06, len(v))
            fc = "white" if cond == "sham" else PROG_COL[prog]
            ax.scatter(x, v, s=11, facecolor=fc, edgecolor=PROG_COL[prog], lw=0.6, zorder=3)
            ax.plot([i + (j - 0.5) * 0.36 - 0.12, i + (j - 0.5) * 0.36 + 0.12], [v.mean()] * 2,
                    color="#333333", lw=0.8, zorder=4)
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.set_xticks(range(len(PROGRAMS)), list(PROGRAMS.values()), fontsize=5.5)
    ax.set_xlim(-0.6, len(PROGRAMS) - 0.4)
    ax.set_ylim(-1.2, 2.8)
    ax.set_title(title, loc="left")


def panel_coupling(ax):
    c = read(R02 / "coupling_stats.tsv")
    c = c[c.analysis == "inflammation_minus_program_per_mouse"]
    progs = {"clearance_curated": "Clearance", "go_phagocytosis": "Phagocytosis",
             "kc_identity": "KC identity", "monocyte_recruited": "Recruited markers"}
    units = {"GSE328774_KC": ("KC 2 h", fs.KC_BLUE, "o"), "GSE311736_KC": ("KC 8 h", fs.KC_BLUE, "s"),
             "GSE311736_CD45_nonKC": ("CD45$^+$ non-KC 8 h", fs.REC_GREEN, "D")}
    c = c[c.program_b.isin(progs) & c.unit.isin(units)].copy()
    fs.save_source(c[["unit", "program_a", "program_b", "n_case", "n_ref", "diff", "diff_ci_low",
                      "diff_ci_high", "hedges_g", "p_welch", "p_mannwhitney"]], ROOT, FIG, "b")
    for k, (u, (lab, col, mk)) in enumerate(units.items()):
        d = c[c.unit == u].set_index("program_b").loc[list(progs)]
        y = np.arange(len(progs))[::-1] + (1 - k) * 0.22
        ax.errorbar(d["diff"], y, xerr=[d["diff"] - d.diff_ci_low, d.diff_ci_high - d["diff"]],
                    fmt=mk, ms=3.2, color=col, mfc=col if k != 1 else "white", lw=0.7, capsize=0,
                    label=lab)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(np.arange(len(progs))[::-1], list(progs.values()))
    ax.set_xlabel("Inflammation gain minus\nprogram gain (log$_2$, CLP − sham)")
    ax.set_xlim(-1.8, 2.9)
    ax.set_ylim(-0.6, len(progs) - 0.1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=5.5, ncol=3,
              handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    ax.set_title("Inflammation outpaces other programs", loc="left", pad=14)


def panel_heat(ax):
    r = read(R02 / "celltype_response.tsv")
    r = r[(r.level == "program") & (r.stratum == "all") & r.feature.isin(HEAT)]
    fs.save_source(r[["unit", "feature", "n_genes", "n_case", "n_ref", "diff", "diff_ci_low",
                      "diff_ci_high", "hedges_g", "p_welch"]], ROOT, FIG, "c")
    M = r.pivot_table(index="feature", columns="unit", values="diff").reindex(
        index=list(HEAT), columns=list(UNITS))
    im = ax.imshow(M.values, cmap=CMAP, norm=NORM, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#EEEEEE", lw=0))
                t = ax.text(j, i, "–", ha="center", va="center", fontsize=5, color=fs.GREY)
            else:
                t = ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 2.4 else "#222222")
            t._overlap_exempt = True
    ax.set_xticks(range(M.shape[1]), [UNITS[u] for u in M.columns], rotation=40, ha="right")
    ax.set_yticks(range(M.shape[0]), [HEAT[f] for f in M.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Which sorted cell responds", loc="left")
    return im


def panel_genes(ax):
    r = read(R02 / "celltype_response.tsv")
    r = r[(r.level == "gene") & (r.stratum == "all") & r.feature.isin(GENES)].copy()
    fs.save_source(r[["unit", "feature", "diff", "lfcSE", "padj", "cpm_ref_mean", "cpm_case_mean",
                      "tested"]], ROOT, FIG, "d")
    units = list(UNITS)
    for j, u in enumerate(units):
        for i, g in enumerate(GENES):
            d = r[(r.unit == u) & (r.feature == g)]
            if d.empty or pd.isna(d["diff"].iloc[0]):
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=fs.GREY)._overlap_exempt = True
                continue
            v, q = d["diff"].iloc[0], d.padj.iloc[0]
            col = CMAP(NORM(float(np.clip(v, -2, 4))))
            size = 8 + 32 * min(-np.log10(max(q, 1e-10)) / 10, 1) if pd.notna(q) else 8
            ax.scatter(j, i, s=size, color=col, edgecolor="#333333" if (pd.notna(q) and q < 0.05)
                       else "none", lw=0.4, label="_cell")
    ax.set_xticks(range(len(units)), [UNITS[u] for u in units], rotation=40, ha="right")
    ax.set_yticks(range(len(GENES)), GENES, fontstyle="italic")
    ax.set_xlim(-0.6, len(units) - 0.4); ax.set_ylim(len(GENES) - 0.5, -0.5)
    ax.tick_params(length=0)
    ax.set_title("HMGB1 and sensing genes", loc="left")


def panel_projection(ax_s, ax_a):
    s = read(R02 / "signature_projection.tsv")
    s = s[(s.scale == "z") & s.signature.isin(["shared_inflammatory", "HMGB1_vs_LPS_axis"])]
    fs.save_source(s, ROOT, FIG, "e")
    units = list(UNITS)
    x = np.arange(len(units))
    for ax, sig, col, lab in [(ax_s, "shared_inflammatory", "#555555", "Shared (z)"),
                              (ax_a, "HMGB1_vs_LPS_axis", fs.CLASS_COLORS["PAMP"],
                               "HMGB1 vs LPS (z)")]:
        d = s[s.signature == sig].set_index("unit").loc[units]
        ax.errorbar(x, d["diff"], yerr=[d["diff"] - d.diff_ci_low, d.diff_ci_high - d["diff"]],
                    fmt="o", ms=3, color=col, lw=0.7, capsize=0)
        ax.axhline(0, color="#333333", lw=0.5)
        ax.set_ylabel(lab)
        ax.set_xlim(-0.6, len(units) - 0.4)
    ax_s.set_xticks(x, [""] * len(units))
    ax_a.set_xticks(x, [UNITS[u] for u in units], rotation=40, ha="right")
    ax_s.set_ylim(-0.5, 6.2)
    ax_a.set_ylim(-3.2, 1.4)
    ax_a.text(len(units) - 0.45, -3.1, "LPS-like", fontsize=5.5, ha="right",
              color=fs.CLASS_COLORS["PAMP"], va="bottom")
    ax_a.text(len(units) - 0.45, 1.3, "HMGB1-like", fontsize=5.5, ha="right",
              color=fs.CLASS_COLORS["DAMP"], va="top")
    ax_s.set_title("Stimulus signatures", loc="left")


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(177 * fs.MM, 150 * fs.MM))
    top = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.93, bottom=0.7, wspace=0.42,
                           width_ratios=[1, 1, 1.05])
    a1 = fig.add_subplot(top[0, 0]); panel_per_mouse(a1, "GSE328774_KC", "KC, 2 h after CLP")
    a1.set_ylabel("Program score (log$_2$)")
    a2 = fig.add_subplot(top[0, 1], sharey=a1); panel_per_mouse(a2, "GSE311736_KC", "KC, 8 h after CLP")
    a2.tick_params(labelleft=False)
    b = fig.add_subplot(top[0, 2]); panel_coupling(b)
    a1.scatter([], [], s=11, facecolor="white", edgecolor="#333333", lw=0.6, label="Sham")
    a1.scatter([], [], s=11, facecolor="#333333", edgecolor="#333333", lw=0.6, label="CLP")
    a1.legend(loc="upper right", frameon=False, fontsize=5.5, handletextpad=0.1, borderaxespad=0.1)
    c = fig.add_axes([0.135, 0.2, 0.25, 0.34]); im = panel_heat(c)
    d = fig.add_axes([0.47, 0.2, 0.19, 0.34]); panel_genes(d)
    cax = fig.add_axes([0.675, 0.38, 0.01, 0.14])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log$_2$ change,\nCLP − sham", fontsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
    h = [plt.scatter([], [], s=8 + 32 * k, color="#BBBBBB", edgecolor=ec, lw=0.4)
         for k, ec in [(0.2, "none"), (1.0, "none"), (1.0, "#333333")]]
    fig.legend(h, ["FDR 0.01", "FDR ≤ 10$^{-10}$", "FDR < 0.05"], loc="upper left",
               bbox_to_anchor=(0.662, 0.35), frameon=False, fontsize=5, handletextpad=0.1,
               labelspacing=0.6, borderaxespad=0)
    es = fig.add_axes([0.835, 0.385, 0.155, 0.155]); ea = fig.add_axes([0.835, 0.2, 0.155, 0.155])
    panel_projection(es, ea)
    for ax, l in zip([a1, b, c, d, es], "abcde"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
