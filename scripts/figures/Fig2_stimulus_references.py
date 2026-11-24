#!/usr/bin/env python
"""Figure 2 - disulfide HMGB1 and LPS evoke one macrophage program at different magnitudes,
and a magnitude-adjusted axis still tells them apart; attribution recovers known inputs.

Reads results/01_perturbation, results/06_attribution and results/09_identifiability only.
Outputs: figures/Fig2.pdf, figures/Fig2.png, figures/source_data/Fig2_<panel>.tsv
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
R01 = ROOT / "results/01_perturbation"
R06 = ROOT / "results/06_attribution"
R09 = ROOT / "results/09_identifiability"
FIG = "Fig2"
PADJ, LFC = 0.05, 1.0          # responsive-gene rule of stage 01 (config/01_perturbation.yaml)
ORDER = ["PBS", "frHMGB1", "dsHMGB1", "LPS", "LPS_IFNg"]
NOISE = {"mouse_KC_2h": ("Mouse sampling, KC 2 h", "#333333", "-"),
         "mouse_KC_8h": ("Mouse sampling, KC 8 h", "#333333", "--"),
         "residmatched_KC_2h": ("Fit residual, KC 2 h", "#C44E52", "-"),
         "residmatched_KC_8h": ("Fit residual, KC 8 h", "#C44E52", "--")}


def read(p):
    return pd.read_csv(p, sep="\t")


def panel_scatter(ax):
    ds, lp = read(R01 / "de_dsHMGB1.tsv"), read(R01 / "de_LPS.tsv")
    m = ds.merge(lp, on="gene", suffixes=("_ds", "_lps"))
    resp = lambda d, s: (d[f"padj_{s}"] < PADJ) & (d[f"log2FC_{s}"].abs() >= LFC)  # noqa: E731
    m = m[resp(m, "ds") | resp(m, "lps")].copy()
    ov = read(R01 / "overlap_stats.tsv").set_index("metric")["value"]
    n_union = int(read(R01 / "overlap_stats.tsv").set_index("metric").loc[
        "pearson_log2FC_union_responsive", "n_genes"])
    assert len(m) == n_union, (len(m), n_union)
    fs.save_source(m[["gene", "log2FC_lps", "log2FC_ds", "padj_lps", "padj_ds"]], ROOT, FIG, "a")
    ax.scatter(m.log2FC_lps, m.log2FC_ds, s=1.2, lw=0, color="#555555", alpha=0.35,
               rasterized=True, label="_genes")
    lim = [-11, 17]
    ax.plot(lim, lim, ls=":", lw=0.6, color=fs.GREY)
    slope = ov["slope_dsHMGB1_on_LPS_log2FC"]
    xx = np.array(lim)
    ax.plot(xx, slope * xx, lw=0.9, color=fs.CLASS_COLORS["DAMP"])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("LPS vs PBS, log$_2$FC")
    ax.set_ylabel("Disulfide HMGB1 vs PBS, log$_2$FC")
    ax.text(0.04, 0.96, f"r = {ov['pearson_log2FC_union_responsive']:.3f}\n"
            f"slope = {slope:.2f}\n{n_union:,} genes", transform=ax.transAxes,
            va="top", ha="left", fontsize=6)
    ax.text(0.97, 0.06, "identity", transform=ax.transAxes, ha="right", fontsize=5.5,
            color=fs.GREY)
    ax.set_title("Shared response, smaller in HMGB1", loc="left")


def panel_sizes(ax):
    rs = read(R01 / "response_sizes.tsv")
    rs = rs[rs.contrast.isin(["frHMGB1", "dsHMGB1", "LPS", "LPS_IFNg"])].set_index("contrast").loc[
        ["frHMGB1", "dsHMGB1", "LPS", "LPS_IFNg"]].reset_index()
    fs.save_source(rs, ROOT, FIG, "b")
    y = np.arange(len(rs))[::-1]
    cols = [fs.COND_COLORS[c] for c in rs.contrast]
    ax.barh(y, rs.n_up, color=cols, height=0.65, edgecolor="none")
    ax.barh(y, -rs.n_down, color=cols, height=0.65, alpha=0.5, edgecolor="none")
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(y, [fs.COND_LABELS[c] for c in rs.contrast])
    ax.set_xlim(-6300, 5600)
    ax.set_xticks([-3000, 0, 3000], ["3,000", "0", "3,000"])
    ax.set_xlabel("Genes down | up vs PBS")
    for yi, (u, d) in zip(y, zip(rs.n_up, rs.n_down)):
        ax.text(u + 80, yi, f"{u:,}", va="center", fontsize=5.5)
        ax.text(-d - 80, yi, f"{d:,}", va="center", ha="right", fontsize=5.5)
    ax.set_title("Response size", loc="left")


def panel_heldout(ax):
    """Held-out technical replicates on the shared score and the adjusted axis, with the
    dsHMGB1-vs-LPS separation (Hedges' g) printed on the panel."""
    h = read(R01 / "heldout_scores.tsv")
    h = h[h.set == "held_out"].copy()
    fs.save_source(h[["sample", "condition", "fold", "shared_inflammatory", "axis_adj_shared"]],
                   ROOT, FIG, "c")
    s = read(R01 / "separation_stats.tsv")
    sep = s[(s.analysis == "pairwise_held_out") & (s.group1 == "dsHMGB1") & (s.group2 == "LPS")]
    keep = ["shared_inflammatory", "HMGB1_vs_LPS_axis", "axis_adj_shared"]
    sep = sep[sep.score.isin(keep)].set_index("score").loc[keep].reset_index()
    acc = s[s.analysis == "nearest_centroid_accuracy"][["score", "diff"]].rename(
        columns={"diff": "accuracy"})
    out = sep[["score", "n1", "n2", "diff", "hedges_g", "welch_p", "exact_perm_p"]].merge(
        acc, on="score", how="left")
    fs.save_source(out, ROOT, FIG, "c_separation")
    for c in ORDER:
        d = h[h.condition == c]
        ax.scatter(d.shared_inflammatory, d.axis_adj_shared, s=11, color=fs.COND_COLORS[c],
                   edgecolor="white", lw=0.3, zorder=3, label="_" + c)
    ax.axhline(0, color=fs.GREY, lw=0.5, ls=":")
    ax.set_xlabel("Shared inflammatory score")
    ax.set_ylabel("HMGB1-vs-LPS axis,\nadjusted for shared score")
    ax.set_xlim(-1.6, 1.4); ax.set_ylim(-2.3, 3.2)
    lab = dict(fontsize=5, va="center")
    ax.text(0.5, 2.2, "Disulfide\nHMGB1", ha="right", color=fs.COND_COLORS["dsHMGB1"],
            linespacing=1.0, **lab)
    ax.text(0.55, -0.9, "LPS", ha="right", color=fs.COND_COLORS["LPS"], **lab)
    ax.text(0.55, -1.55, "LPS + IFN-γ", ha="right", color=fs.COND_COLORS["LPS_IFNg"], **lab)
    ax.text(-0.85, 0.55, "PBS, reduced\nHMGB1", ha="left", color="#777777", linespacing=1.0,
            **lab)
    g = out.set_index("score")
    txt = (f"Disulfide HMGB1 vs LPS, g:\naxis {g.loc['axis_adj_shared', 'hedges_g']:.1f}; "
           f"shared {g.loc['shared_inflammatory', 'hedges_g']:.1f}")
    ax.text(-1.55, -2.2, txt.replace("-1", "−1"), ha="left", va="bottom", fontsize=4.8,
            color="#444444", linespacing=1.1)
    ax.set_title("Held-out technical replicates", loc="left")


def panel_power(ax):
    """Power to detect an HMGB1-like component mixed into a cross-laboratory LPS-like response."""
    d = read(R09 / "hmgb1_detection.tsv")
    d = d[(d.panel == "macrophage_resolution_unmerged") & (d.statistic == "hmgb1_share")
          & (d.source != "within_GSE200210") & d.noise.isin(NOISE)]
    # the four cross-laboratory sources carry equal draws, so the mean of their powers is the
    # power of the pooled draws
    p = (d.groupby(["noise", "f"], as_index=False)
         .agg(power_pooled=("power_pooled", "mean"), n_sources=("source", "nunique"),
              n_draws_per_source=("n_draws", "first"), thr_pooled=("thr_pooled", "first")))
    fs.save_source(p, ROOT, FIG, "d")
    for nz, (lab, col, ls) in NOISE.items():
        q = p[p.noise == nz].sort_values("f")
        ax.plot(q.f, q.power_pooled, color=col, ls=ls, lw=0.9, marker="o", ms=2, label=lab)
    ax.axhline(0.8, color=fs.GREY, lw=0.5, ls=":")
    ax.set_xlim(-0.02, 0.72); ax.set_ylim(0, 1.3)
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_yticks([0, 0.5, 0.8, 1], ["0", "0.5", "0.8", "1"])
    ax.set_xlabel("True HMGB1-like fraction")
    ax.set_ylabel("Detection power")
    h = [plt.Line2D([], [], color="#333333", lw=0.9), plt.Line2D([], [], color="#C44E52", lw=0.9),
         plt.Line2D([], [], color="#777777", lw=0.9), plt.Line2D([], [], color="#777777", lw=0.9,
                                                                 ls="--")]
    ax.legend(h, ["Mouse sampling", "Fit residual", "KC 2 h", "KC 8 h"], loc="upper left",
              ncol=2, frameon=False, fontsize=4.8, handlelength=1.6, handletextpad=0.3,
              borderaxespad=0.1, labelspacing=0.25, columnspacing=0.6)
    ax.set_title("HMGB1-like part detected", loc="left")


def panel_transfer(ax):
    """Leave-one-study-out class attribution of held-out stimuli."""
    t = read(R09 / "transfer_summary.tsv")
    t = t[(t.panel == "paper") & (t.method == "nnls") & (t.contrast == "stimulus")]
    order = ["PAMP", "DAMP", "cytokine", "resolution"]
    allr = t[t.expected_class == "all"].iloc[0]
    t = t.set_index("expected_class").loc[order].reset_index()
    fs.save_source(pd.concat([t, allr.to_frame().T], ignore_index=True), ROOT, FIG, "e")
    x = np.arange(len(t))
    ax.bar(x, t.top_class_accuracy, width=0.62, color=[fs.CLASS_COLORS[c] for c in order],
           edgecolor="none")
    for xi, ch in zip(x, t.mean_chance_share):
        ax.plot([xi - 0.38, xi + 0.38], [ch, ch], color="#222222", lw=0.9, zorder=3)
    for xi, v, ch in zip(x, t.top_class_accuracy, t.mean_chance_share):
        ax.text(xi, max(v, ch) + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=5)
    labs = {"PAMP": "PAMP", "DAMP": "DAMP", "cytokine": "Cyto-\nkine",
            "resolution": "Toler-\nance"}
    ax.set_xticks(x, [f"{labs[c]}\n(n = {int(n)})" for c, n in zip(order, t.n_targets)],
                  fontsize=5)
    ax.set_xlim(-0.6, len(t) - 0.4); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Correct top class")
    h = [plt.Line2D([], [], color="#222222", lw=0.9)]
    ax.legend(h, ["Chance"], loc="upper right", frameon=False, fontsize=5, handlelength=1.2,
              handletextpad=0.3, borderaxespad=0.1)
    ax.text(0.98, 0.82, f"All {int(allr.n_targets)}: {allr.top_class_accuracy:.2f}\n"
            f"(chance {allr.mean_chance_share:.2f})", transform=ax.transAxes, ha="right",
            va="top", fontsize=4.8, color="#444444", linespacing=1.1)
    ax.set_title("Leave-one-study-out", loc="left")


def panel_mixture(ax):
    m = read(R06 / "controls_mixture.tsv")
    m = m[m.panel == "macrophage_resolution"].copy()
    m["pair"] = m.reference_a + " + " + m.reference_b
    fs.save_source(m, ROOT, FIG, "f")
    hm = (m.reference_a == "LPS_24h") & (m.reference_b == "dsHMGB1")
    jit = (m.groupby(["true_weight_a"]).cumcount() % 7 - 3) * 0.008
    ax.scatter(m.true_weight_a[~hm] + jit[~hm], m.recovered_weight_a[~hm], s=5, lw=0,
               color=fs.GREY, alpha=0.6, label="Other reference pairs")
    ax.scatter(m.true_weight_a[hm] + jit[hm], m.recovered_weight_a[hm], s=9, lw=0.3,
               edgecolor="white", color=fs.CLASS_COLORS["DAMP"], label="LPS + disulfide HMGB1")
    ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="#333333")
    c = read(R06 / "controls.tsv")
    err = c[(c.control == "synthetic mixture") & (c.item == "LPS_24h + dsHMGB1")]["value"].iloc[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("True weight of first reference")
    ax.set_ylabel("Recovered weight")
    ax.text(0.97, 0.04, f"LPS + HMGB1\nmedian |error| = {err:.3f}", transform=ax.transAxes,
            ha="right", fontsize=5.5)
    ax.legend(loc="upper left", frameon=False, fontsize=5.5, handletextpad=0.1, borderaxespad=0.2)
    ax.set_title("Synthetic mixtures", loc="left")


def panel_heldout_stimuli(ax):
    h = read(R06 / "controls_heldout.tsv")
    h = h.sort_values(["panel", "share_on_expected"], ascending=[False, True]).reset_index(drop=True)
    fs.save_source(h, ROOT, FIG, "g")
    y = np.arange(len(h))
    ok = h.top_in_expected & (h.share_on_expected >= 0.5)
    cols = np.where(ok, "#555555", "#D9D9D9")
    ax.barh(y, h.share_on_expected, color=cols, height=0.65, edgecolor="#555555", lw=0.4)
    ax.axvline(0.5, color=fs.GREY, lw=0.5, ls=":")
    ax.set_yticks(y, [fs.REF_LABELS.get(t, t) for t in h.target])
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Share on expected references")
    for yi, (v, top) in enumerate(zip(h.share_on_expected, h.top_reference)):
        if v < 0.5:
            ax.text(v + 0.03, yi, f"top: {fs.REF_LABELS.get(top, top)}", va="center", fontsize=5,
                    color="#555555", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_title("Held-out stimuli", loc="left")


def main():
    fs.apply_style()
    fig = plt.figure(figsize=(177 * fs.MM, 128 * fs.MM))
    top, bot, h = 0.58, 0.1, 0.33
    axa = fig.add_axes([0.075, top, 0.17, h]); panel_scatter(axa)
    axb = fig.add_axes([0.375, top, 0.13, h]); panel_sizes(axb)
    axc = fig.add_axes([0.62, top, 0.135, h]); panel_heldout(axc)
    axd = fig.add_axes([0.83, top, 0.155, h]); panel_power(axd)
    axe = fig.add_axes([0.075, bot, 0.17, h]); panel_transfer(axe)
    axf = fig.add_axes([0.375, bot, 0.2, h]); panel_mixture(axf)
    axg = fig.add_axes([0.765, bot, 0.22, h]); panel_heldout_stimuli(axg)
    for ax, l in zip([axa, axb, axc, axd, axe, axf, axg], "abcdefg"):
        fs.panel_label(ax, l)
    fs.save_fig(fig, ROOT, FIG)


if __name__ == "__main__":
    main()
