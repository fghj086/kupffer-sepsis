#!/usr/bin/env python
"""Figure 4 - liver time courses after LPS and CLP: inflammation peaks early, KC identity falls,
and the late drift of the HMGB1-vs-LPS axis is not attributed to an HMGB1-like input.

Reads results/04_timecourse and results/06_attribution only.
Outputs: figures/Fig4.pdf, figures/Fig4.png, figures/source_data/Fig4_<panel>.tsv
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
R04 = ROOT / "results/04_timecourse"
R06 = ROOT / "results/06_attribution"
FIG = "Fig4"

SCORES = {"nfkb_inflammatory": ("NF-κB inflammatory", "#222222"),
          "hepatocyte_acute_phase": ("Acute phase", "#DA8BC3"),
          "kc_identity": ("KC identity", fs.KC_BLUE),
          "recruited_mono_mac": ("Recruited markers", fs.REC_GREEN),
          "clearance_phagocytosis": ("Clearance", "#8172B3")}
TIMES = {0.25: "6 h", 0.5: "12 h", 1: "1 d", 2: "2 d", 3: "3 d", 5: "5 d"}
GRADES = {"mild": ("Mild", "#C9C9C9", "^"), "moderate": ("Moderate", "#8C8C8C", "s"),
          "severe": ("Severe", "#333333", "o")}
CLASSES = ["PAMP", "mixed", "DAMP", "cytokine", "resolution"]


def read(p):
    return pd.read_csv(p, sep="\t")


def tx(t):
    return np.log2(np.asarray(t, float))


def time_axis(ax, times):
    ax.set_xticks(tx(times), [TIMES[t] for t in times])
    ax.set_xlim(tx(min(times)) - 0.4, tx(max(times)) + 0.4)


def panel_lps(ax):
    t = read(R04 / "timecourse_effects.tsv")
    t = t[(t.model == "LPS") & t.score.isin(SCORES)].copy()
    fs.save_source(t[["model", "time_d", "score", "n_test", "n_ref", "diff_log2", "diff_ci_low",
                      "diff_ci_high", "hedges_g", "fdr"]], ROOT, FIG, "a")
    for k, (sc, (lab, col)) in enumerate(SCORES.items()):
        d = t[t.score == sc].sort_values("time_d")
        x = tx(d.time_d) + (k - 2) * 0.05
        ax.errorbar(x, d.diff_log2, yerr=[d.diff_log2 - d.diff_ci_low, d.diff_ci_high - d.diff_log2],
                    fmt="-o", ms=2.6, lw=0.9, elinewidth=0.6, color=col, capsize=0, label=lab)
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    time_axis(ax, sorted(TIMES))
    ax.set_xlabel("Time after LPS")
    ax.set_ylabel("Liver score, log$_2$ vs day 0")
    ax.set_ylim(-2.2, 9.2)
    ax.legend(loc="upper right", frameon=False, fontsize=5.2, handlelength=1.2, borderaxespad=0.1,
              labelspacing=0.25, handletextpad=0.3)
    ax.set_title("LPS, whole liver", loc="left")


def panel_clp(ax):
    t = read(R04 / "timecourse_effects.tsv")
    t = t[(t.model == "CLP") & t.score.isin(SCORES)].copy()
    fs.save_source(t[["model", "arm", "time_d", "score", "n_test", "n_ref", "diff_log2",
                      "diff_ci_low", "diff_ci_high", "hedges_g", "fdr"]], ROOT, FIG, "b")
    cols = [(g, tt) for g in GRADES for tt in (0.25, 0.5, 1)]
    M = np.full((len(SCORES), len(cols)), np.nan)
    Q = np.full_like(M, np.nan)
    for i, sc in enumerate(SCORES):
        for j, (g, tt) in enumerate(cols):
            d = t[(t.score == sc) & (t.arm == g) & (t.time_d == tt)]
            if len(d):
                M[i, j], Q[i, j] = d.diff_log2.iloc[0], d.fdr.iloc[0]
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            s = f"{M[i, j]:.1f}" + ("*" if Q[i, j] < 0.05 else "")
            tt = ax.text(j, i, s, ha="center", va="center", fontsize=5,
                         color="white" if abs(M[i, j]) > 2.4 else "#222222")
            tt._overlap_exempt = True
    ax.set_xticks(range(len(cols)), [TIMES[tt] for _, tt in cols], fontsize=5.5)
    ax.set_yticks(range(len(SCORES)), [v[0] for v in SCORES.values()])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j in (2.5, 5.5):
        ax.axvline(j, color="white", lw=1.5)
    for k, g in enumerate(GRADES):
        ax.text(1 + 3 * k, -0.75, GRADES[g][0], ha="center", va="bottom", fontsize=6)
    ax.set_title("CLP grade, whole liver", loc="left", pad=14)
    return im


def panel_axis(ax):
    r = read(R04 / "signature_relative.tsv")
    r = r[r.model.isin(["LPS", "CLP"])].copy()
    fs.save_source(r[["model", "arm", "time_d", "contrast_diff", "contrast_ci_low",
                      "contrast_ci_high", "shared_diff", "shared_ci_low", "shared_ci_high",
                      "contrast_per_shared"]], ROOT, FIG, "c")
    d = r[r.model == "LPS"].sort_values("time_d")
    ax.errorbar(d.shared_diff, d.contrast_diff,
                xerr=[d.shared_diff - d.shared_ci_low, d.shared_ci_high - d.shared_diff],
                yerr=[d.contrast_diff - d.contrast_ci_low, d.contrast_ci_high - d.contrast_diff],
                fmt="-o", ms=3.4, lw=0.9, elinewidth=0.5, color=fs.CLASS_COLORS["PAMP"],
                capsize=0, label="LPS", zorder=3)
    for g, (lab, col, mk) in GRADES.items():
        c = r[(r.model == "CLP") & (r.arm == g)].sort_values("time_d")
        ax.plot(c.shared_diff, c.contrast_diff, "-", marker=mk, ms=3, lw=0.7, color=col,
                label=f"CLP, {lab.lower()}", zorder=2)
    ax.axhline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.axvline(0, color=fs.GREY, lw=0.4, ls=":")
    ax.set_xlim(-0.4, 3.1); ax.set_ylim(-1.1, 1.9)
    ax.set_xlabel("Shared score (z vs control)")
    ax.set_ylabel("HMGB1-vs-LPS axis (z vs control)")
    ax.text(3.05, 1.85, "HMGB1-like", ha="right", va="top", fontsize=5.5,
            color=fs.CLASS_COLORS["DAMP"])
    ax.text(3.05, -1.05, "LPS-like", ha="right", va="bottom", fontsize=5.5,
            color=fs.CLASS_COLORS["PAMP"])
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handlelength=1.2,
              borderaxespad=0.1, labelspacing=0.25, handletextpad=0.3)
    fs.place_labels(ax, d.shared_diff.values, d.contrast_diff.values,
                    [TIMES[t] for t in d.time_d], fontsize=5.2, color=fs.CLASS_COLORS["PAMP"],
                    min_r=9, max_r=45)
    ax.set_title("Axis drifts as shared response wanes", loc="left")


def panel_attr(ax, model, group, title, legend=False):
    o = read(R06 / "attribution_over_time.tsv")
    o = o[(o.model == model) & (o.group == group) & (o.panel == "macrophage_resolution")]
    fs.save_source(o[["model", "group", "time_d", "panel", "stimulus_class", "share", "delta_r2",
                      "r2_total"]], ROOT, FIG, f"d_{model}_{group}")
    W = o.pivot_table(index="time_d", columns="stimulus_class", values="share").reindex(
        columns=CLASSES).fillna(0)
    r2 = o.drop_duplicates("time_d").set_index("time_d").r2_total.reindex(W.index)
    x = np.arange(len(W))
    bottom = np.zeros(len(W))
    for c in CLASSES:
        alpha = np.where(r2.values < 0.05, 0.35, 1.0)
        for xi in range(len(W)):
            ax.bar(x[xi], W[c].values[xi], bottom=bottom[xi], color=fs.CLASS_COLORS[c],
                   width=0.72, alpha=alpha[xi], edgecolor="white", lw=0.3,
                   label=fs.CLASS_LABELS[c] if xi == 0 else "_")
        bottom += W[c].values
    for xi, v in zip(x, r2.values):
        ax.text(xi, 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5, color="#444444")
    ax.set_xticks(x, [TIMES[t] for t in W.index])
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.6, len(W) - 0.4)
    ax.set_title(title, loc="left", pad=8)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=5.5,
                  handlelength=1.0, borderaxespad=0)
    return o


def panel_hmgb1(ax):
    o = read(R06 / "attribution_over_time.tsv")
    o = o[o.panel.isin(["macrophage_resolution_unmerged", "liver_plus_unmerged"])]
    o = o.drop_duplicates(["target", "panel"])
    sel = o[((o.model == "LPS") | ((o.model == "CLP") & (o.group == "severe")))].copy()
    fs.save_source(sel[["model", "group", "time_d", "panel", "hmgb1_reference", "hmgb1_share",
                        "hmgb1_share_ci_low", "hmgb1_share_ci_high", "r2_total"]], ROOT, FIG, "e")
    sty = {("LPS", "macrophage_resolution_unmerged"): ("LPS, macrophage panel", fs.CLASS_COLORS["PAMP"], "o", "white"),
           ("LPS", "liver_plus_unmerged"): ("LPS, liver panel", fs.CLASS_COLORS["PAMP"], "o", None),
           ("CLP", "macrophage_resolution_unmerged"): ("CLP severe, macrophage panel", "#333333", "s", "white"),
           ("CLP", "liver_plus_unmerged"): ("CLP severe, liver panel", "#333333", "s", None)}
    for k, ((m, p), (lab, col, mk, mfc)) in enumerate(sty.items()):
        d = sel[(sel.model == m) & (sel.panel == p)].sort_values("time_d")
        x = tx(d.time_d) + (k - 1.5) * 0.07
        ax.errorbar(x, d.hmgb1_share, yerr=[d.hmgb1_share - d.hmgb1_share_ci_low,
                                            d.hmgb1_share_ci_high - d.hmgb1_share],
                    fmt=mk, ms=3, color=col, mfc=mfc or col, lw=0.6, capsize=0, label=lab)
    time_axis(ax, sorted(TIMES))
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel("Time after challenge")
    ax.set_ylabel("Share on disulfide HMGB1\nreference (95% CI)")
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.3)
    ax.set_title("HMGB1-like share stays near zero", loc="left")


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(177 * fs.MM, 135 * fs.MM))
    a = fig.add_axes([0.075, 0.6, 0.22, 0.33]); panel_lps(a)
    b = fig.add_axes([0.42, 0.6, 0.21, 0.30]); im = panel_clp(b)
    cax = fig.add_axes([0.64, 0.66, 0.009, 0.18])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("log$_2$ vs sham (* FDR < 0.05)", fontsize=5.5)
    c = fig.add_axes([0.765, 0.6, 0.225, 0.33]); panel_axis(c)
    d1 = fig.add_axes([0.075, 0.1, 0.3, 0.33]); panel_attr(d1, "LPS", "LPS", "LPS, whole liver (R² above bars)")
    d1.set_ylabel("Share of fitted response")
    d2 = fig.add_axes([0.395, 0.1, 0.15, 0.33]); panel_attr(d2, "CLP", "severe", "CLP, severe",
                                                             legend=True)
    d2.tick_params(labelleft=False)
    e = fig.add_axes([0.765, 0.1, 0.225, 0.33]); panel_hmgb1(e)
    for ax, l in zip([a, b, c, d1, e], "abcde"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
