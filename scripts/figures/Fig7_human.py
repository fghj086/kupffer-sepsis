#!/usr/bin/env python
"""Figure 7 - human cells and septic patients.

(a) Endotoxin-tolerised human PBMC (GSE22248, per donor) and monocytes (GSE15219): tolerance scores.
(b) HMGB1 in human monocytes/macrophages: GSE254271, GSE162834; GSE78162 qPCR panel.
(c) GSE46955 blood monocytes: tolerance-like state in sepsis, reversed with recovery.
Post-mortem liver and blood mortality panels are in SuppFigS5_human_liver_mortality.py.

Reads results/07_human only.
Outputs: figures/Fig7.pdf, figures/Fig7.png, figures/source_data/Fig7_<panel>.tsv
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
FIG = "Fig7"

SC = {"tolerance_KC": ("Tolerant KC (mouse)", fs.KC_BLUE, "o"),
      "tolerance_BMDM": ("Tolerant BMDM (mouse)", "#8172B3", "s"),
      "pena_endotoxin_tolerance": ("Endotoxin tolerance (human)", "#937860", "D")}


def sgn(v, plus=True):
    """Signed number with a typographic minus."""
    t = f"{v:+.2f}" if plus else f"{v:.2f}"
    return t.replace("-", "−")


def read(p):
    return pd.read_csv(p, sep="\t")


def contrasts():
    return read(R07 / "contrasts.tsv")


def scores():
    return read(R07 / "scores_by_sample.tsv")


# ---------------------------------------------------------------- a
def panel_a_donor(ax):
    s = scores()
    s = s[(s.dataset == "GSE22248") & (s.score == "tolerance_KC")]
    fs.save_source(s[["dataset", "sample", "subject", "group", "score", "value"]], ROOT, FIG, "a_donor")
    order = ["untreated", "LPS", "LPS_LPS"]
    w = s.pivot_table(index="subject", columns="group", values="value")[order]
    for _, r in w.iterrows():
        ax.plot(range(3), r.values, "-o", color=fs.KC_BLUE, ms=2.8, lw=0.6, mfc="white", mew=0.6)
    c = contrasts()
    c = c[(c.dataset == "GSE22248") & (c.score == "tolerance_KC") & (c.contrast == "LPS_LPS vs LPS")].iloc[0]
    ax.text(1.5, 1.55, f"{sgn(c.estimate)}\n({c.ci_low:.2f}–{c.ci_high:.2f})", ha="center",
            va="bottom", fontsize=5.2, linespacing=1.1)
    ax.set_xticks(range(3), ["None", "LPS", "LPS→\nLPS"])
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(-3.2, 2.8)
    ax.set_ylabel("Tolerant-KC score")
    ax.set_title("PBMC, 4 donors", loc="left", fontsize=6)


def panel_a_forest(ax):
    c = contrasts()
    rows = [("GSE22248", "LPS_LPS vs LPS", "PBMC, 2nd LPS"),
            ("GSE15219", "T_L3h vs L3h", "Monocytes, LPS 3 h"),
            ("GSE15219", "T_L6h vs L6h", "Monocytes, LPS 6 h"),
            ("GSE15219", "T_L24h vs L24h", "Monocytes, LPS 24 h")]
    sel = pd.concat([c[(c.dataset == d) & (c.contrast == k) & c.score.isin(SC)] for d, k, _ in rows])
    fs.save_source(sel[["dataset", "contrast", "score", "design", "n1", "n0", "estimate", "ci_low",
                        "ci_high", "hedges_g", "p", "note"]], ROOT, FIG, "a_contrasts")
    for i, (d, k, lab) in enumerate(rows):
        for j, (sc, (sl, col, mk)) in enumerate(SC.items()):
            r = sel[(sel.dataset == d) & (sel.contrast == k) & (sel.score == sc)].iloc[0]
            y = len(rows) - 1 - i + (1 - j) * 0.22
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.6)
            ax.plot(r.estimate, y, marker=mk, ms=3, color=col, ls="")
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(range(len(rows))[::-1], [r[2] for r in rows])
    ax.set_xlim(-6, 13); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Tolerant − naive (score units, 95% CI)")
    h = [plt.Line2D([], [], color=v[1], marker=v[2], ms=3, lw=0.6) for v in SC.values()]
    ax.legend(h, [v[0] for v in SC.values()], loc="lower left", bbox_to_anchor=(0, 1.0),
              frameon=False, fontsize=5, handlelength=1.4, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.2)
    ax.set_title("Tolerant vs naive restimulation", loc="left", pad=28, fontsize=6)


# ---------------------------------------------------------------- b
def panel_b(ax):
    c = contrasts()
    groups = [("GSE254271", "HMGB1 vs GMCSF", "HMGB1, 72 h (4 donors)", "#DD8452", "o"),
              ("GSE254271", "S100A9 vs GMCSF", "S100A9, same donors", "#8C8C8C", "s"),
              ("GSE162834", "HMGB1 vs control", "HMGB1 (1 donor, 3 cultures)", "#7A2427", "D")]
    scs = {"shared_inflammatory": "Shared", "HMGB1_vs_LPS_axis": "HMGB1-vs-LPS axis",
           "LPS_biased": "LPS-biased", "HMGB1_biased": "HMGB1-biased",
           "pena_endotoxin_tolerance": "Endotoxin tolerance"}
    sel = pd.concat([c[(c.dataset == d) & (c.contrast == k) & c.score.isin(scs)] for d, k, *_ in groups])
    g78 = c[c.dataset == "GSE78162"]
    fs.save_source(pd.concat([sel, g78])[["dataset", "contrast", "score", "design", "n1", "n0",
                                          "estimate", "ci_low", "ci_high", "hedges_g", "p"]],
                   ROOT, FIG, "b")
    for i, sc in enumerate(scs):
        for j, (d, k, lab, col, mk) in enumerate(groups):
            r = sel[(sel.dataset == d) & (sel.contrast == k) & (sel.score == sc)].iloc[0]
            y = len(scs) - 1 - i + (1 - j) * 0.24
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.6)
            ax.plot(r.estimate, y, marker=mk, ms=3, color=col, ls="", label=lab if i == 0 else "_")
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(range(len(scs))[::-1], list(scs.values()))
    ax.set_xlim(-1.4, 4.0); ax.set_ylim(-0.6, len(scs) - 0.4)
    ax.set_xlabel("Stimulus − control (95% CI)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=5,
              handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    v = g78.set_index("score").estimate
    ax.text(1.0, 1.02, "GSE78162, 135-gene panel:\nHMGB1 = "
            f"{v['fitted scale of LPS for HMGB1']:.2f} × LPS response\n"
            f"r = {v['gene-level Pearson r']:.2f}, R² = {v['R2 of HMGB1 by scaled LPS']:.2f}",
            ha="right", va="bottom", fontsize=5, linespacing=1.15, color="#444444",
            transform=ax.transAxes)
    ax.set_title("HMGB1 in human myeloid cells", loc="left", pad=28, fontsize=6)


# ---------------------------------------------------------------- c
GRP = {"healthy": ("Healthy", "#8C8C8C"), "sepsis": ("Sepsis", "#C44E52"),
       "recovery": ("Recovery", "#55A868")}


def panel_c_pena(ax):
    s = scores()
    s = s[(s.dataset == "GSE46955") & (s.score == "pena_endotoxin_tolerance") & (s.stim == "basal")]
    fs.save_source(s[["dataset", "sample", "subject", "group", "state", "stim", "score", "value"]],
                   ROOT, FIG, "c_tolerance")
    rng = np.random.default_rng(7)
    xs = {}
    for i, st in enumerate(GRP):
        d = s[s.state == st]
        x = i + rng.uniform(-0.1, 0.1, len(d))
        xs.update({(st, sub): xx for sub, xx in zip(d.subject, x)})
        ax.scatter(x, d.value, s=10, color=GRP[st][1], lw=0, zorder=3)
    w = s[s.state.isin(["sepsis", "recovery"])].pivot_table(index="subject", columns="state",
                                                            values="value").dropna()
    for sub, r in w.iterrows():
        ax.plot([xs[("sepsis", sub)], xs[("recovery", sub)]], [r.sepsis, r.recovery],
                color="#BBBBBB", lw=0.4, zorder=1)
    c = contrasts()
    c = c[(c.dataset == "GSE46955") & (c.score == "pena_endotoxin_tolerance")].set_index("contrast")
    a, b = c.loc["sepsis_basal vs healthy_basal"], c.loc["recovery_basal vs sepsis_basal"]
    ax.text(0.5, 4.3, sgn(a.estimate), ha="center", fontsize=5.2)
    ax.text(1.5, 4.3, sgn(b.estimate), ha="center", fontsize=5.2)
    ax.set_xticks(range(3), [v[0] for v in GRP.values()], rotation=30, ha="right")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-3.2, 5.0)
    ax.set_ylabel("Endotoxin-tolerance score")
    ax.set_title("Unstimulated", loc="left", fontsize=6)


def panel_c_lps(ax):
    s = scores()
    s = s[(s.dataset == "GSE46955") & (s.score == "shared_inflammatory")]
    fs.save_source(s[["dataset", "sample", "subject", "group", "state", "stim", "score", "value"]],
                   ROOT, FIG, "c_lps_response")
    for i, st in enumerate(GRP):
        w = s[s.state == st].pivot_table(index="subject", columns="stim", values="value").dropna()
        for _, r in w.iterrows():
            ax.plot([i - 0.17, i + 0.17], [r.basal, r.LPS], color=GRP[st][1], lw=0.5, alpha=0.8)
        ax.scatter(np.full(len(w), i - 0.17), w.basal, s=7, facecolor="white",
                   edgecolor=GRP[st][1], lw=0.5, zorder=3)
        ax.scatter(np.full(len(w), i + 0.17), w.LPS, s=7, color=GRP[st][1], lw=0, zorder=3)
    c = contrasts()
    c = c[(c.dataset == "GSE46955") & (c.score == "shared_inflammatory")].set_index("contrast")
    a, b = c.loc["LPS response: sepsis vs healthy"], c.loc["LPS response: recovery vs sepsis"]
    ax.text(0.5, 3.25, sgn(a.estimate), ha="center", fontsize=5.2)
    ax.text(1.5, 3.25, sgn(b.estimate), ha="center", fontsize=5.2)
    ax.set_xticks(range(3), [v[0] for v in GRP.values()], rotation=30, ha="right")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-1.6, 3.7)
    ax.set_ylabel("Shared inflammatory score")
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc=fc, mec="#555555", mew=0.5)
         for fc in ("white", "#555555")]
    ax.legend(h, ["Basal", "LPS ex vivo"], loc="lower right", frameon=False, fontsize=5,
              handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("Ex vivo LPS response", loc="left", fontsize=6)


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(180 * fs.MM, 125 * fs.MM))
    a1 = fig.add_axes([0.07, 0.62, 0.1, 0.26]); panel_a_donor(a1)
    a2 = fig.add_axes([0.32, 0.62, 0.17, 0.26]); panel_a_forest(a2)
    b = fig.add_axes([0.665, 0.62, 0.32, 0.26]); panel_b(b)
    c1 = fig.add_axes([0.07, 0.1, 0.25, 0.3]); panel_c_pena(c1)
    c2 = fig.add_axes([0.45, 0.1, 0.25, 0.3]); panel_c_lps(c2)
    for ax, l in zip([a1, b, c1], "abc"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
