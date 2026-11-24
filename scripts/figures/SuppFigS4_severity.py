#!/usr/bin/env python
"""Supplementary Figure S4 - macrophage programs scale with CLP severity and track hepatocyte injury.

(a) KC identity by CLP grade (45 mice, three time points), centred within time point.
(b) Tolerance reference responses by CLP grade.
(c) Association of macrophage programs with hepatocyte injury beyond NF-κB inflammation,
    CLP and LPS mice (standardised beta, 95% CI).
(d) Severity slopes of KC identity and the tolerance scores, unadjusted and adjusted for
    cell-composition scores (recruited, neutrophil, leukocyte, pan-macrophage) and NF-κB.

Reads results/08_injury_link only.
Outputs: figures/SuppFigS4.pdf, figures/SuppFigS4.png, figures/source_data/SuppFigS4_<panel>.tsv
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
R08 = ROOT / "results/08_injury_link"
FIG = "SuppFigS4"

GRADES = {"mild": 1, "moderate": 2, "severe": 3}
STRATA = {"CLP_d0.25_d1_d0.25": ("6 h", "#C9A227", "^"), "CLP_d0.5_d0.5": ("12 h", "#DD8452", "s"),
          "CLP_d0.25_d1_d1.0": ("1 d", "#7A2427", "o")}
SCORES = {"kc_identity": "KC identity",
          "res_KC_TolLPS_6h": "Tolerant KC + LPS\n(vs control)",
          "res_KC_tol_vs_LPS": "Tolerant − naive KC,\nLPS-restimulated",
          "res_Tolerance_shift": "Tolerant − naive BMDM,\nLPS-restimulated"}
PROGS = {"kc_identity": ("KC identity", fs.KC_BLUE), "clearance_phagocytosis": ("Clearance", "#8172B3"),
         "recruited_mono_mac": ("Recruited", fs.REC_GREEN)}
OUTCOMES = {"hepatocyte_acute_phase": "acute phase", "hepatocyte_function_loss": "function loss"}
COVS = {"unadjusted": ("Unadjusted", "white", "o"),
        "inflammation_only": ("+ NF-κB score", "fill", "o"),
        "inflammation_composition": ("+ NF-κB + macrophage content", "fill", "s")}


def read(p):
    return pd.read_csv(p, sep="\t")


def fmt(v):
    return f"{v:.2f}".replace("-", "−")


def grade_panel(ax, score, mice, dose, ylabel=False, legend=False):
    rng = np.random.default_rng(11)
    for st, (lab, col, mk) in STRATA.items():
        d = mice[mice.stratum == st]
        x = d.grade + (list(STRATA).index(st) - 1) * 0.16 + rng.uniform(-0.04, 0.04, len(d))
        ax.scatter(x, d[f"{score}_centred"], s=9, marker=mk, facecolor="none", edgecolor=col,
                   lw=0.6, zorder=3, label=lab)
    r = dose[(dose.score == score)].iloc[0]
    xx = np.array([0.7, 3.3])
    ax.plot(xx, r.slope_per_grade_log2 * (xx - 2), color="#333333", lw=0.8, zorder=2)
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.set_xticks([1, 2, 3], ["Mild", "Moder-\nate", "Severe"])
    ax.set_xlim(0.55, 3.45)
    lines = [f"ρ = {fmt(r.spearman_rho)}",
             f"slope {fmt(r.slope_per_grade_log2)} ({fmt(r.slope_ci_low)} to {fmt(r.slope_ci_high)})"]
    if pd.notna(r.slope_adj_inflammation_log2):
        lines.append(f"NF-κB-adjusted {fmt(r.slope_adj_inflammation_log2)}")
    ax.text(0.0, 1.03, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=4.9, linespacing=1.15, color="#444444")
    ax.text(0.0, 1.03 + 0.073 * (len(lines) + 0.3), SCORES[score], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6, linespacing=1.1)
    if ylabel:
        ax.set_ylabel("Score minus time-point mean\n(log$_2$)")
    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=5, handletextpad=0.1,
                  borderaxespad=0.1, labelspacing=0.2)


def forest(ax, m, cohort, strata, title, labels=True):
    d = m[(m.cohort == cohort) & (m.adjustment_strata == strata) & m.program.isin(PROGS) &
          m.covariates.isin(COVS)]
    rows = [(o, p) for o in OUTCOMES for p in PROGS]
    y = np.arange(len(rows))[::-1].astype(float)
    y[3:] -= 0.5
    for yi, (o, p) in zip(y, rows):
        col = PROGS[p][1]
        for k, (cv, (_, fill, mk)) in enumerate(COVS.items()):
            r = d[(d.outcome == o) & (d.program == p) & (d.covariates == cv)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = yi + (1 - k) * 0.24
            ax.plot([r.beta_ci_low, r.beta_ci_high], [yy, yy], color=col, lw=0.6)
            ax.plot(r.std_beta, yy, ls="", marker=mk, ms=3, mec=col, mew=0.6,
                    mfc="white" if fill == "white" else col)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(y, [f"{PROGS[p][0]} → {OUTCOMES[o]}" for o, p in rows] if labels else [])
    if labels:
        for t, (_, p) in zip(ax.get_yticklabels(), rows):
            t.set_color(PROGS[p][1])
    else:
        ax.tick_params(left=False)
    ax.set_xlim(-1.5, 1.0); ax.set_xticks([-1, -0.5, 0, 0.5])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("β per SD (95% CI)")
    n = int(d.n_mice.iloc[0])
    ax.set_title(f"{title} (n = {n})", loc="left", fontsize=6)
    return d


COMP_SCORES = {"kc_identity": ("KC identity", fs.KC_BLUE),
               "kc_identity_per_macrophage": ("KC identity per macrophage", "#9DB7DA"),
               "res_KC_tol_vs_LPS": ("Tolerant − naive KC", fs.CLASS_COLORS["resolution"]),
               "res_Tolerance_shift": ("Tolerant − naive BMDM", fs.CLASS_COLORS["resolution"])}
COMP_MODELS = {"unadjusted": ("Unadjusted", "white", "o"),
               "all_composition": ("+ composition", "fill", "o"),
               "all_composition_nfkb": ("+ composition + NF-κB", "fill", "s")}


def composition_forest(ax):
    """Severity slopes before and after adjustment for cell-composition scores."""
    c = read(R08 / "dose_response_composition.tsv")
    c = c[(c.grade_range == "CLP_grades_only") & c.score.isin(COMP_SCORES)
          & c.model.isin(COMP_MODELS)]
    # per-macrophage identity is shown unadjusted only: adjusting it for the pan-macrophage
    # score returns the KC identity model above it
    c = c[(c.score != "kc_identity_per_macrophage") | (c.model == "unadjusted")]
    fs.save_source(c, ROOT, FIG, "d")
    order = list(COMP_SCORES)
    y0 = np.arange(len(order))[::-1].astype(float)
    tr = ax.get_yaxis_transform()
    for yi, sc in zip(y0, order):
        col = COMP_SCORES[sc][1]
        for k, (mod, (lab, fill, mk)) in enumerate(COMP_MODELS.items()):
            r = c[(c.score == sc) & (c.model == mod)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = yi + (1 - k) * 0.24
            ax.plot([r.slope_ci_low, r.slope_ci_high], [y, y], color=col, lw=0.6)
            ax.plot(r.slope_per_grade_log2, y, ls="", marker=mk, ms=3, mec=col, mew=0.6,
                    mfc="white" if fill == "white" else col)
            ax.text(1.05 + 0.28 * k, yi, fmt(r.slope_per_grade_log2), transform=tr, ha="left",
                    va="center", fontsize=5.2)
    for k, (lab, _, _) in enumerate(COMP_MODELS.values()):
        ax.text(1.05 + 0.28 * k, y0[0] + 0.55, lab.replace(" + NF", "\n+ NF"), transform=tr,
                ha="left", va="bottom", fontsize=5, color="#555555", linespacing=1.0)
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(y0, [COMP_SCORES[s][0] for s in order])
    for t, s in zip(ax.get_yticklabels(), order):
        t.set_color(COMP_SCORES[s][1] if s != "kc_identity_per_macrophage" else fs.KC_BLUE)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.65, 0.15)
    ax.set_xlabel("Slope per CLP grade, log$_2$ (95% CI)")
    h = [plt.Line2D([], [], ls="", marker=mk, ms=3, mec="#555555", mew=0.6,
                    mfc="white" if fill == "white" else "#555555")
         for _, fill, mk in COMP_MODELS.values()]
    ax.legend(h, [v[0] for v in COMP_MODELS.values()], loc="center left",
              bbox_to_anchor=(1.95, 0.5), frameon=False, fontsize=5.2, handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.4)
    ax.set_title("Severity slope with composition adjustment (n = 45)", loc="left", fontsize=6,
                 pad=16)


def main():
    fs.apply_style()
    mice = read(R08 / "scores_by_mouse.tsv")
    mice = mice[(mice.model == "CLP") & mice.group.isin(GRADES) & mice.stratum.isin(STRATA)].copy()
    mice["grade"] = mice.group.map(GRADES)
    for s in SCORES:
        mice[f"{s}_centred"] = mice[s] - mice.groupby("stratum")[s].transform("mean")
    dose = read(R08 / "dose_response.tsv")
    dose = dose[(dose.grade_range == "CLP_grades_only") & (dose.stratum == "pooled_within_stratum")
                & dose.score.isin(SCORES)]
    fs.save_source(mice[["column", "file_key", "group", "grade", "time_d", "mouse", "stratum"] +
                        [c for s in SCORES for c in (s, f"{s}_centred")]], ROOT, FIG, "ab_mice")
    fs.save_source(dose, ROOT, FIG, "ab_trend")
    assert len(mice) == 45, len(mice)

    fig = plt.figure(figsize=(180 * fs.MM, 178 * fs.MM))
    a = fig.add_axes([0.08, 0.665, 0.17, 0.205])
    grade_panel(a, "kc_identity", mice, dose, ylabel=True, legend=True)
    bx = [fig.add_axes([0.375 + 0.225 * k, 0.665, 0.16, 0.205]) for k in range(3)]
    for ax, s in zip(bx, list(SCORES)[1:]):
        grade_panel(ax, s, mice, dose)

    m = read(R08 / "injury_models.tsv")
    c1 = fig.add_axes([0.2, 0.3, 0.2, 0.25])
    c2 = fig.add_axes([0.46, 0.3, 0.2, 0.25])
    c3 = fig.add_axes([0.72, 0.3, 0.2, 0.25])
    sel = pd.concat([forest(c1, m, "CLP", "batch_time", "CLP, within time point"),
                     forest(c2, m, "LPS", "batch", "LPS, within study", labels=False),
                     forest(c3, m, "LPS", "batch_time", "LPS, within time point", labels=False)])
    fs.save_source(sel[["cohort", "adjustment_strata", "primary", "covariates", "outcome", "program",
                        "n_mice", "std_beta", "beta_ci_low", "beta_ci_high", "p_beta", "fdr",
                        "partial_r", "lobo_n_same_sign", "lobo_n_fits"]], ROOT, FIG, "c")
    h = [plt.Line2D([], [], ls="", marker=mk, ms=3, mec="#555555", mew=0.6,
                    mfc="white" if fill == "white" else "#555555") for _, fill, mk in COVS.values()]
    c1.legend(h, [v[0] for v in COVS.values()], loc="lower left", bbox_to_anchor=(0, 1.08),
              ncol=3, frameon=False, fontsize=5.2, handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.1)
    d = fig.add_axes([0.2, 0.045, 0.3, 0.13]); composition_forest(d)
    for ax, l in zip([a, bx[0], c1, d], "abcd"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
