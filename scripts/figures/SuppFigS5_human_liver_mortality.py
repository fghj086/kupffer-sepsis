#!/usr/bin/env python
"""Supplementary Figure S5 - septic human liver and blood mortality.

(a) GSE141864 septic liver (4 septic shock vs 1 control organ; descriptive).
(b) Blood cohorts, 28-day mortality: per-cohort and pooled odds ratios, unadjusted, adjusted
    for blood-composition markers, and adjusted for deconvolved neutrophil/monocyte proportions
    (three cohorts).

Reads results/07_human only.
Outputs: figures/SuppFigS5.pdf, figures/SuppFigS5.png, figures/source_data/SuppFigS5_<panel>.tsv
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
R07 = ROOT / "results/07_human"
FIG = "SuppFigS5"

COHORTS = {"MARS": "MARS", "GAINS": "GAinS (E-MTAB-5273)", "GAINS4421": "GAinS (E-MTAB-4421)",
           "GSE95233": "GSE95233", "GSE222393": "GSE222393", "pooled (REML)": "Pooled"}
MODELS = {"unadjusted": ("Unadjusted", "#333333", "white", "o"),
          "composition markers": ("Composition markers", "#C44E52", "#C44E52", "o"),
          "deconvolved proportions": ("Deconvolved proportions", "#4C72B0", "#4C72B0", "s")}


def read(p):
    return pd.read_csv(p, sep="\t")


def contrasts():
    return read(R07 / "contrasts.tsv")


def scores():
    return read(R07 / "scores_by_sample.tsv")


# ---------------------------------------------------------------- a
def panel_a(ax):
    scs = {"shared_inflammatory": "Shared", "HMGB1_vs_LPS_axis": "HMGB1-vs-LPS axis",
           "pena_endotoxin_tolerance": "Endotoxin tolerance", "tolerance_KC": "Tolerant KC",
           "tolerance_BMDM": "Tolerant BMDM", "kc_identity": "KC identity",
           "clearance": "Clearance"}
    s = scores()
    s = s[(s.dataset == "GSE141864_liver") & s.score.isin(scs)]
    c = contrasts()
    c = c[(c.dataset == "GSE141864_liver") & c.score.isin(scs)]
    fs.save_source(s[["dataset", "sample", "subject", "group", "score", "value"]], ROOT, FIG, "a")
    fs.save_source(c[["dataset", "contrast", "score", "n1", "n0", "estimate", "ci_low", "ci_high",
                      "effect_only", "note"]], ROOT, FIG, "a_contrasts")
    rng = np.random.default_rng(3)
    for i, sc in enumerate(scs):
        y = len(scs) - 1 - i
        d = s[(s.score == sc) & (s.group == "septic_shock")]
        ax.scatter(d.value, y + rng.uniform(-0.12, 0.12, len(d)), s=8, color="#C44E52", lw=0,
                   zorder=3)
        ax.scatter(s[(s.score == sc) & (s.group == "control")].value, [y], s=14, marker="|",
                   color="#333333", lw=0.8, zorder=2)
        r = c[c.score == sc].iloc[0]
        ax.plot([r.estimate] * 2, [y - 0.28, y + 0.28], color="#C44E52", lw=0.9, zorder=4)
    ax.axvline(0, color="#BBBBBB", lw=0.4, ls=":")
    ax.set_yticks(range(len(scs))[::-1], list(scs.values()))
    ax.set_xlim(-1.6, 1.8); ax.set_ylim(-0.6, len(scs) - 0.4)
    ax.set_xlabel("Score vs control liver")
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, color="#C44E52"),
         plt.Line2D([], [], color="#C44E52", lw=0.9),
         plt.Line2D([], [], ls="", marker="|", ms=4, color="#333333", mew=0.8)]
    ax.legend(h, ["Septic shock (n = 4)", "Mean", "Control (n = 1)"], loc="lower left",
              bbox_to_anchor=(0, 1.0), frameon=False, fontsize=5, ncol=1, handletextpad=0.3,
              borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("Post-mortem liver", loc="left", pad=24, fontsize=6)


# ---------------------------------------------------------------- b
def panel_b(axs):
    m = read(R07 / "mortality_models.tsv")
    m = m[(m.analysis == "28-day mortality") &
          m.score.isin(["pena_endotoxin_tolerance", "HMGB1_vs_LPS_axis"])].copy()
    # one row per cohort, then the five-cohort pool and the three-cohort pool (the cohorts
    # with deconvolved proportions), each with its unadjusted estimate in the same cohorts
    rows = []
    cohorts = [c for c in COHORTS if not c.startswith("pooled")]
    for ds in cohorts:
        for mod in MODELS:
            rows.append((ds, ds, mod, mod))
    rows += [("pooled5", "pooled (REML)", "unadjusted", "unadjusted"),
             ("pooled5", "pooled (REML)", "composition markers", "composition markers"),
             ("pooled3", "pooled (REML)", "unadjusted (cohorts of 'deconvolved proportions')",
              "unadjusted"),
             ("pooled3", "pooled (REML)", "deconvolved proportions", "deconvolved proportions")]
    sel = []
    for row, ds, mod, sty in rows:
        r = m[(m.dataset == ds) & (m.model == mod)].copy()
        sel.append(r.assign(row=row, style=sty))
    sel = pd.concat(sel)
    fs.save_source(sel[["row", "dataset", "score", "model", "covariates", "n_cohorts",
                        "n_patients", "n_deaths", "OR_per_SD", "OR_ci_low", "OR_ci_high",
                        "OR_hk_ci_low", "OR_hk_ci_high", "OR_mhk_ci_low", "OR_mhk_ci_high", "p",
                        "hk_p", "mhk_p", "I2", "tau2"]], ROOT, FIG, "b")
    titles = {"pena_endotoxin_tolerance": "Endotoxin-tolerance score",
              "HMGB1_vs_LPS_axis": "HMGB1-vs-LPS axis"}
    order = cohorts + ["pooled5", "pooled3"]
    ypos = {r: len(order) - 1 - i for i, r in enumerate(order)}
    off = {"unadjusted": 0.27, "composition markers": 0.0, "deconvolved proportions": -0.27}
    for k, (ax, sc) in enumerate(zip(axs, titles)):
        d = sel[sel.score == sc]
        for _, r in d.iterrows():
            lab, col, mfc, mk = MODELS[r["style"]]
            pooled = r.row.startswith("pooled")
            # pooled rows carry two estimates each, spread wider so their labels fit
            y = ypos[r.row] + (off[r["style"]] if not pooled else
                               (0.22 if r["style"] == "unadjusted" else -0.22))
            if pooled:
                ax.plot([r.OR_mhk_ci_low, r.OR_mhk_ci_high], [y, y], color=col, lw=2.2,
                        alpha=0.25, solid_capstyle="butt")
            ax.plot([r.OR_ci_low, r.OR_ci_high], [y, y], color=col, lw=0.7)
            ax.plot(r.OR_per_SD, y, ls="", marker="D" if pooled else mk,
                    ms=3.4 if pooled else 2.6, mfc=mfc, mec=col, mew=0.6, zorder=3)
            if pooled:
                txt = (f"{r.OR_per_SD:.2f} ({r.OR_mhk_ci_low:.2f}–{r.OR_mhk_ci_high:.2f})")
                ax.text(1.04, y, txt, color=col, fontsize=4.9, transform=ax.get_yaxis_transform(),
                        ha="left", va="center")
        ax.axvline(1, color="#333333", lw=0.5)
        ax.axhline(1.5, color="#DDDDDD", lw=0.5)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlim(0.2, 14)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("Odds ratio for 28-day death per SD")
        ax.set_title(titles[sc], loc="left", fontsize=6)
        ax.text(1.04, 1.56, "Pooled OR (modified\nHartung–Knapp interval)", color="#555555",
                transform=ax.get_yaxis_transform(), fontsize=4.9, ha="left", va="bottom",
                linespacing=1.05)
        if k == 0:
            n = d.drop_duplicates("row").set_index("row")
            labs = [f"{COHORTS[q]} ({int(n.loc[q, 'n_deaths'])}/{int(n.loc[q, 'n_patients'])})"
                    for q in cohorts]
            p5 = d[(d.row == "pooled5")].iloc[0]; p3 = d[(d.row == "pooled3")].iloc[0]
            labs += [f"Pooled, {int(p5.n_cohorts)} cohorts ({int(p5.n_deaths)}/{int(p5.n_patients)})",
                     f"Pooled, {int(p3.n_cohorts)} cohorts ({int(p3.n_deaths)}/{int(p3.n_patients)})"]
            ax.set_yticks([ypos[q] for q in order], labs)
        else:
            ax.set_yticks([ypos[q] for q in order], [])
            ax.tick_params(left=False)
    h = [plt.Line2D([], [], ls="", marker=v[3], ms=2.8, mfc=v[2], mec=v[1], mew=0.6)
         for v in MODELS.values()]
    h.append(plt.Line2D([], [], color="#888888", lw=2.2, alpha=0.35))
    axs[0].legend(h, [v[0] for v in MODELS.values()] + ["Pooled: modified Hartung–Knapp interval"],
                  loc="lower left", bbox_to_anchor=(-0.9, 1.1), ncol=4, frameon=False, fontsize=5,
                  handletextpad=0.5, columnspacing=1.4, borderaxespad=0.1)


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(180 * fs.MM, 135 * fs.MM))
    a = fig.add_axes([0.15, 0.62, 0.3, 0.27]); panel_a(a)
    b1 = fig.add_axes([0.2, 0.07, 0.2, 0.32])
    b2 = fig.add_axes([0.62, 0.07, 0.2, 0.32])
    panel_b([b1, b2])
    for ax, l in zip([a, b1], "ab"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
