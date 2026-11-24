#!/usr/bin/env python
"""Figure 6 - single-cell, spatial and protein layers.

(a) GSE279167 pooled libraries: macrophage composition, sham vs CLP (effect sizes only).
(b) GSE290684 (12 livers, CLP 24 h): within-liver KC-marker-positive minus recruited difference,
    beyond the shared inflammatory score, for gene sets free of annotation genes.
(c) GSE311736 sorted cells (4 sham + 4 CLP 8 h): Kupffer cells vs CD45+ non-KC cells in
    scavenger-receptor and identity scores; the gap is present in sham and unchanged after CLP
    (module fig6_sorted_gap.py).
(d) GSE248904 spatial (2 control + 2 LPS): hepatocyte stress by zone; myeloid neighbourhood vs
    hepatocyte stress, adjusted for zonation.
(e) PXD062961 CLP liver proteome vs GSE224127 CLP liver transcripts.

Reads results/03_single_cell, results/04_spatial, results/05_proteomics and
results/10_sepsis_specificity only.
Outputs: figures/Fig6.pdf, figures/Fig6.png, figures/source_data/Fig6_<panel>.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import fig6_sorted_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R03 = ROOT / "results/03_single_cell"
R04 = ROOT / "results/04_spatial"
R05 = ROOT / "results/05_proteomics"
FIG = "Fig6"

POPS = {"Resident KC": ("Resident KC (marker-high)", fs.KC_BLUE),
        "KC-like (KC-marker-intermediate)": ("KC-marker-intermediate", "#9DB7DA"),
        "Recruited Mo-macrophages": ("Recruited macrophages", fs.REC_GREEN),
        "Monocytes": ("Monocytes", "#B5D8AE")}
IND = {"identity_not_annotation": "KC identity\n(no annotation genes)",
       "clearance_strict": "Scavenger receptors\n(no annotation genes)",
       "inflammatory": "Inflammatory",
       "sig_HMGB1_vs_LPS_axis": "HMGB1-vs-LPS axis"}
ARMS = {"vehicle": ("Vehicle livers (n = 6)", "o", "#333333", 0.22),
        "TPPU": ("TPPU livers (n = 6)", "o", "white", 0.0),
        "all_12": ("All 12 livers", "D", "#C44E52", -0.22)}
PREDICTORS = {"kc_identity": ("KC identity", fs.KC_BLUE),
              "recruited_mono_mac": ("Recruited", fs.REC_GREEN),
              "nfkb_inflammatory": ("NF-κB", "#333333")}
NEIGH = {"own_unit": "own tile", "adjacent": "adjacent", "ring": "ring"}


def read(p, **kw):
    return pd.read_csv(p, sep="\t", **kw)


def fmt_p(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"P = {m}×10$^{{{int(e)}}}$"
    return f"P = {p:.2f}"


# ---------------------------------------------------------------- a: pooled composition
def panel_umap(axs):
    u = read(R03 / "umap_GSE279167_myeloid.tsv.gz", compression="gzip",
             usecols=["cell", "condition", "population", "umap1", "umap2"])
    fs.save_source(u, ROOT, FIG, "a_umap")
    xl = (u.umap1.min() - 0.5, u.umap1.max() + 0.5)
    yl = (u.umap2.min() - 0.5, u.umap2.max() + 0.5)
    rng = np.random.default_rng(6)
    for ax, cond, lab in zip(axs, ["sham", "CLP"], ["Sham", "CLP"]):
        bg = u[u.condition != cond]
        ax.scatter(bg.umap1, bg.umap2, s=0.4, color="#E6E6E6", lw=0, rasterized=True)
        d = u[u.condition == cond].iloc[rng.permutation((u.condition == cond).sum())]
        ax.scatter(d.umap1, d.umap2, s=0.5, lw=0, rasterized=True,
                   c=[POPS[p][1] for p in d.population])
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{lab} pool ({len(d):,} cells)", loc="left", fontsize=6)
    axs[0].set_ylabel("UMAP 2", fontsize=5.5)
    axs[0].set_xlabel("UMAP 1", fontsize=5.5); axs[1].set_xlabel("UMAP 1", fontsize=5.5)


def panel_composition(ax):
    c = read(R03 / "myeloid_composition_contrast.tsv")
    keep = ["Resident KC", "KC-like (KC-marker-intermediate)", "Recruited Mo-macrophages"]
    c = c[(c.dataset == "GSE279167") & (c.scope == "macrophages") & c.population.isin(keep)]
    fs.save_source(c[["dataset", "scope", "population", "reference", "comparison",
                      "ref_mean_fraction", "alt_mean_fraction", "delta_fraction", "n_ref",
                      "n_alt"]], ROOT, FIG, "a_composition")
    c = c.set_index("population").loc[keep]
    for x, col in enumerate(["ref_mean_fraction", "alt_mean_fraction"]):
        bottom = 0.0
        for pop in keep:
            v = c.loc[pop, col]
            ax.bar(x, v, bottom=bottom, width=0.62, color=POPS[pop][1], edgecolor="white", lw=0.4)
            if v >= 0.08:
                t = ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if pop != keep[1] else "#222222")
                t._overlap_exempt = True
            bottom += v
    ax.set_xticks([0, 1], ["Sham", "CLP"])
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Fraction of macrophages")
    ax.set_title("One pool per condition", loc="left", fontsize=6)


# ---------------------------------------------------------------- b: KC minus recruited
def panel_beyond(ax):
    """KC-marker-positive minus recruited within each liver, beyond the shared score, for gene
    sets that share no gene with the cell annotation (summaries per treatment arm)."""
    k = read(R03 / "kc_vs_recruited_independent.tsv")
    k = k[k.score.isin(IND) & (k.adjustment == "adjusted_for_shared")].copy()
    fs.save_source(k, ROOT, FIG, "b")
    order = list(IND)
    y = np.arange(len(order))[::-1].astype(float)
    y[:3] += 0.5                                       # gap between programs and the axis
    for yi, sc in zip(y, order):
        for cs, (lab, mk, fc, off) in ARMS.items():
            r = k[(k.score == sc) & (k.condition_set == cs)].iloc[0]
            ax.plot([r.ci_low, r.ci_high], [yi + off] * 2, color="#333333" if cs != "all_12"
                    else "#C44E52", lw=0.6, zorder=2)
            ax.plot(r.mean_kc_minus_recruited, yi + off, ls="", marker=mk, ms=3.4 if mk != "D" else 3,
                    mfc=fc, mec="#333333" if cs != "all_12" else "#C44E52", mew=0.6, zorder=3)
        for j, cs in enumerate(["vehicle", "all_12"]):
            r = k[(k.score == sc) & (k.condition_set == cs)].iloc[0]
            ax.text(1.04 + 0.2 * j, yi, f"{int(r.n_livers_kc_higher)}/{int(r.n_livers)}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.3)
    tr = ax.get_yaxis_transform()
    ax.text(1.04, y[0] + 0.45, "Vehicle", transform=tr, ha="left", va="bottom", fontsize=5,
            color="#555555")
    ax.text(1.24, y[0] + 0.45, "All", transform=tr, ha="left", va="bottom", fontsize=5,
            color="#555555")
    ax.text(1.04, y[0] + 0.85, "KC higher", transform=tr, ha="left", va="bottom", fontsize=5,
            color="#555555")
    ax.text(3.35, (y[2] + y[3]) / 2, "programs: log CPM\naxis: mean t", ha="right",
            va="center", fontsize=4.8, color="#777777", linespacing=1.1)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(y, [IND[s] for s in order])
    ax.set_ylim(-0.6, y[0] + 1.3)
    ax.set_xlim(-0.8, 3.4); ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("KC − recruited, within liver,\nbeyond shared score (95% CI)")
    h = [plt.Line2D([], [], ls="", marker=mk, ms=3.4 if mk != "D" else 3, mfc=fc,
                    mec="#333333" if cs != "all_12" else "#C44E52", mew=0.6)
         for cs, (lab, mk, fc, off) in ARMS.items()]
    ax.legend(h, [v[0] for v in ARMS.values()], loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              ncol=1, frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.25)
    ax.set_title("CLP 24 h, GSE290684", loc="left", pad=30)


# ---------------------------------------------------------------- d: spatial
def panel_spatial_stress(ax):
    e = read(R04 / "spatial_effects.tsv")
    e = e[(e.score == "hepatocyte_stress") & e.subregion.isin(["Pericentral", "Periportal"])]
    fs.save_source(e, ROOT, FIG, "d_stress")
    for i, zone in enumerate(["Pericentral", "Periportal"]):
        r = e[e.subregion == zone].iloc[0]
        for j, (col, lab) in enumerate([("control_animals", "Control"), ("lps_animals", "LPS")]):
            v = np.array([float(x) for x in str(r[col]).split(";")])
            x = i + (j - 0.5) * 0.4 + np.array([-0.05, 0.05])
            fc = "white" if j == 0 else fs.CLASS_COLORS["PAMP"]
            ax.scatter(x, v, s=14, facecolor=fc, edgecolor=fs.CLASS_COLORS["PAMP"], lw=0.6,
                       zorder=3)
        ax.text(i, 1.5, f"+{r.lps_minus_control:.2f}", ha="center", va="bottom", fontsize=5.3)
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.set_xticks([0, 1], ["Peri-\ncentral", "Peri-\nportal"])
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-1.5, 2.0)
    ax.set_ylabel("Hepatocyte stress score")
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc=fc, mec=fs.CLASS_COLORS["PAMP"],
                    mew=0.6) for fc in ("white", fs.CLASS_COLORS["PAMP"])]
    ax.legend(h, ["Control", "LPS 24 h"], loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("Per animal", loc="left", fontsize=6)


def panel_spatial_assoc(ax):
    c = read(R04 / "colocalization.tsv")
    c = c[(c.resolution == "tile_0.2") & (c.treatment != "contrast")]
    fs.save_source(c, ROOT, FIG, "d_association")
    samples = {"CTRL_1": ("Control 1", "white", "o"), "CTRL_2": ("Control 2", "white", "s"),
               "LPS_1": ("LPS 1", fs.CLASS_COLORS["PAMP"], "o"),
               "LPS_2": ("LPS 2", fs.CLASS_COLORS["PAMP"], "s")}
    rows = [(p, n) for p in PREDICTORS for n in NEIGH]
    y = np.arange(len(rows))[::-1].astype(float)
    y -= np.repeat([0, 0.5, 1.0], 3)
    for yi, (p, n) in zip(y, rows):
        for k, (smp, (lab, fc, mk)) in enumerate(samples.items()):
            d = c[(c.predictor == p) & (c.neighbourhood == n) & (c["sample"] == smp)]
            if d.empty:
                continue
            d = d.iloc[0]
            yy = yi + (k - 1.5) * 0.17
            ax.plot([d.beta_ci_low, d.beta_ci_high], [yy, yy], color=PREDICTORS[p][1], lw=0.5)
            ax.scatter(d.beta_adj, yy, s=8, marker=mk, facecolor=fc,
                       edgecolor=PREDICTORS[p][1] if fc == "white" else fc, lw=0.5, zorder=3)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(y, [f"{PREDICTORS[p][0]}, {NEIGH[n]}" for p, n in rows], fontsize=5.2)
    for t, (p, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(PREDICTORS[p][1])
    ax.set_xlim(-0.25, 0.25); ax.set_xticks([-0.2, 0, 0.2])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("β, hepatocyte stress per SD\n(zonation-adjusted, 95% CI)")
    h = [plt.Line2D([], [], ls="", marker=mk, ms=3, mfc=fc, mec="#555555" if fc == "white" else fc,
                    mew=0.5) for _, fc, mk in samples.values()]
    ax.legend(h, [v[0] for v in samples.values()], loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, frameon=False, fontsize=5, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.1)
    ax.set_title("Myeloid surroundings", loc="left", fontsize=6, pad=16)


# ---------------------------------------------------------------- e: proteome
def panel_proteome(ax):
    p = read(R05 / "protein_de_PXD062961.tsv")
    t = read(R05 / "transcript_de_GSE224127_CLP_d1_liver.tsv")
    m = p[p.tested == True].merge(t, on="gene", suffixes=("_protein", "_transcript"))  # noqa: E712
    m = m.dropna(subset=["log2FC", "lfc"])
    m = m.rename(columns={"log2FC": "protein_log2FC", "lfc": "transcript_log2FC"})
    m["protein_fdr_sig"] = m.padj_protein < 0.05
    fs.save_source(m[["gene", "protein_log2FC", "padj_protein", "transcript_log2FC",
                      "padj_transcript", "protein_fdr_sig"]], ROOT, FIG, "e")
    conc = read(R05 / "transcript_protein_concordance.tsv")
    conc = conc[(conc.protein_dataset == "PXD062961") &
                (conc.transcript_dataset == "GSE224127_CLP_d1_liver")].iloc[0]
    fs.save_source(conc.to_frame().T, ROOT, FIG, "e_concordance")
    ns, sg = m[~m.protein_fdr_sig], m[m.protein_fdr_sig]
    agree = np.sign(sg.protein_log2FC) == np.sign(sg.transcript_log2FC)
    ax.scatter(ns.transcript_log2FC, ns.protein_log2FC, s=1.5, color="#DDDDDD", lw=0,
               rasterized=True, label="_ns")
    ax.scatter(sg.transcript_log2FC[agree], sg.protein_log2FC[agree], s=5, color="#4C72B0",
               lw=0, alpha=0.85, label=f"Same direction ({int(agree.sum())})")
    ax.scatter(sg.transcript_log2FC[~agree], sg.protein_log2FC[~agree], s=5, color="#DD8452",
               lw=0, alpha=0.9, label=f"Opposite ({int((~agree).sum())})")
    hl = m[m.gene.isin(["Clec4f", "Hmgb1"])]
    ax.scatter(hl.transcript_log2FC, hl.protein_log2FC, s=16, facecolor="none",
               edgecolor="#C44E52", lw=0.8, zorder=5, label="_hl")
    ax.axhline(0, color="#333333", lw=0.4); ax.axvline(0, color="#333333", lw=0.4)
    ax.set_xlim(-6.5, 9.5); ax.set_ylim(-3.2, 6.0)
    ax.set_xlabel("Transcript log$_2$FC, CLP 1 d (GSE224127)")
    ax.set_ylabel("Protein log$_2$FC, CLP 24 h (PXD062961)")
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handletextpad=0.1, borderaxespad=0.1,
              title=f"Protein FDR < 0.05 (n = {int(conc.n_protein_fdr_sig)})", title_fontsize=5.2,
              alignment="left")
    ax.text(9.3, -3.0, f"{conc.frac_sign_agree_sig * 100:.0f}% same direction\n"
            f"ρ = {conc.spearman_rho:.2f} (all {int(conc.n_genes_shared):,} genes)",
            ha="right", va="bottom", fontsize=5.3, linespacing=1.15)
    labs = [f"{g.upper()}" for g in hl.gene]
    fs.place_labels(ax, hl.transcript_log2FC.values, hl.protein_log2FC.values, labs,
                    fontsize=5.6, italic=False, color="#C44E52", min_r=10, max_r=60)
    ax.set_title("Liver proteome vs transcriptome", loc="left")


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(180 * fs.MM, 200 * fs.MM))
    # row 1: a (embedding + composition)
    a1 = fig.add_axes([0.035, 0.725, 0.2, 0.19])
    a2 = fig.add_axes([0.25, 0.725, 0.2, 0.19])
    panel_umap([a1, a2])
    a3 = fig.add_axes([0.53, 0.725, 0.075, 0.19]); panel_composition(a3)
    h = [plt.Rectangle((0, 0), 1, 1, color=POPS[p][1]) for p in POPS]
    fig.legend(h, [POPS[p][0] for p in POPS], loc="center left", bbox_to_anchor=(0.66, 0.82),
               ncol=1, frameon=False, fontsize=5.5, handlelength=0.9, handletextpad=0.3,
               borderaxespad=0, labelspacing=0.5)
    # row 2: b (within-liver KC - recruited) and c (sorted sham vs CLP)
    b = fig.add_axes([0.19, 0.42, 0.2, 0.19]); panel_beyond(b)
    c1 = fig.add_axes([0.66, 0.42, 0.13, 0.19])
    c2 = fig.add_axes([0.83, 0.42, 0.13, 0.19])
    fig6_sorted_gap.panel([c1, c2], ROOT, FIG)
    # row 3: d (spatial) and e (proteome)
    d1 = fig.add_axes([0.075, 0.05, 0.11, 0.22]); panel_spatial_stress(d1)
    d2 = fig.add_axes([0.32, 0.05, 0.15, 0.22]); panel_spatial_assoc(d2)
    e = fig.add_axes([0.62, 0.05, 0.355, 0.24]); panel_proteome(e)
    for ax, l in zip([a1, b, c1, d1, e], "abcde"):
        fs.panel_label(ax, l)
    # dataset headers sit on the baseline of their panel letter, just right of it
    fs.align_panel_labels(fig)
    letters = {t.get_text(): t for t in fig.texts if getattr(t, "_panel_letter", False)}
    heads = {"a": "Liver myeloid cells, GSE279167 (pooled libraries; effect sizes only)",
             "c": "Sorted liver cells, CLP 8 h, GSE311736 (4 + 4 mice)",
             "d": "Spatial liver, GSE248904 (2 control + 2 LPS; exploratory)"}
    for k, txt in heads.items():
        x, y = letters[k].get_position()
        fig.text(x + 0.022, y, txt, fontsize=6.5, ha="left", va="bottom")
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
