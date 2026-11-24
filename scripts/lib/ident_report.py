"""Stage 09: plain-English SUMMARY.md assembled from the stage 09 tables (numbers are read back
from the written TSVs, never typed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CROSS = ["KLA_6h", "KC_LPS_6h", "LPS_RAW_14h", "LPS_24h_M1"]


def _r(out, name):
    return pd.read_csv(out / name, sep="\t")


def _f(x, nd=2):
    return "not reached (> 0.7)" if pd.isna(x) else f"{x:.{nd}g}"


def section_a(out) -> list[str]:
    lim = _r(out, "hmgb1_detection_limits.tsv")
    det = _r(out, "hmgb1_detection.tsv")
    L = ["## A. Detection limit of an HMGB1-like component", "",
         "Minimum HMGB1-like fraction f detected with >= 80% power (threshold = 95th percentile at "
         "f = 0 pooled over the four cross-laboratory LPS-like responses; 'within' = optimistic "
         "GSE200210 LPS replicate). Columns: noise setting.", ""]
    for stat, panel in [("hmgb1_share", "macrophage_unmerged"),
                        ("hmgb1_share", "macrophage_resolution_unmerged"),
                        ("hmgb1_share", "hmgb1_vs_lps"), ("axis", "none")]:
        d = lim[(lim.statistic == stat) & (lim.panel == panel) & (lim.threshold == "pooled")]
        piv = d.pivot(index="source", columns="noise", values="min_f_power80")
        L += [f"**{stat} / {panel}**", "", "| LPS-like source | " + " | ".join(piv.columns) + " |",
              "|---" * (len(piv.columns) + 1) + "|"]
        for src, r in piv.iterrows():
            L.append(f"| {src} | " + " | ".join(_f(v) for v in r.values) + " |")
        L.append("")
    fp = det[(det.statistic == "hmgb1_share") & (det.f == 0) & det.source.isin(CROSS)]
    fp = fp.groupby(["panel", "noise"]).agg(mean_share_f0=("mean", "mean"),
                                            p_gt_005_f0=("p_share_gt_fixed", "mean")).reset_index()
    L += ["False dsHMGB1 weight at f = 0 (pure cross-laboratory LPS-like response), mean over the "
          "four sources:", "", "| panel | noise | mean share | P(share > 0.05) |", "|---|---|---|---|"]
    L += [f"| {r.panel} | {r.noise} | {r.mean_share_f0:.3f} | {r.p_gt_005_f0:.2f} |" for r in fp.itertuples()]
    ub = _r(out, "hmgb1_upper_bound.tsv")
    u = ub[ub.source == "cross_lab_pooled"].drop_duplicates(["target", "noise", "panel", "statistic"])
    L += ["", "Upper bound on the HMGB1-like fraction in observed responses (smallest f at which the "
          "observed statistic would be seen in < 5% of simulations; cross-laboratory sources pooled):", "",
          "| target | noise | panel | statistic | observed | smallest f excluded |", "|---|---|---|---|---|---|"]
    L += [f"| {r.target} | {r.noise} | {r.panel} | {r.statistic} | {r.observed:.3f} | "
          + (f"{_f(r.smallest_f_excluded)} |" if r.f0_compatible else
             "observed outside the pure LPS-like range (f = 0 rejected) |") for r in u.itertuples()]
    return L + [""]


def section_b(out) -> list[str]:
    h = _r(out, "methods_heldout_summary.tsv")
    mm = _r(out, "methods_mixture.tsv")
    L = ["## B. Methods compared", "", "Held-out stimuli (11 controls of stage 06):", "",
         "| method | top reference in expected set | mean share on expected | pass (top in expected and share >= 0.5) |",
         "|---|---|---|---|"]
    L += [f"| {r.method} | {r.top_in_expected:.2f} | {r.mean_share_on_expected:.2f} | {r.pass_rate:.2f} |"
          for r in h.itertuples()]
    x = mm[mm.source.isin(CROSS)].groupby(["method", "f"]).power_pooled.mean().unstack()
    L += ["", "Mixture detection power (cross-laboratory sources and both mouse-noise settings "
          "averaged; pooled 5% threshold); columns are f:", "",
          "| method | " + " | ".join(f"{c:g}" for c in x.columns) + " |", "|---" * (len(x.columns) + 1) + "|"]
    L += [f"| {m} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |" for m, r in x.iterrows()]
    return L + [""]


def section_c(out) -> list[str]:
    s = _r(out, "transfer_summary.tsv")
    s = s[s.contrast == "stimulus"]
    L = ["## C. Leave-one-series-out transfer of stimulus classes", "",
         "| panel | method | expected class | n | top-class accuracy | mean expected-class share | chance share |",
         "|---|---|---|---|---|---|---|"]
    L += [f"| {r.panel} | {r.method} | {r.expected_class} | {r.n_targets} | {r.top_class_accuracy:.2f} | "
          f"{r.mean_expected_class_share:.2f} | {r.mean_chance_share:.2f} |" for r in s.itertuples()]
    return L + [""]


def section_d(out) -> list[str]:
    p = _r(out, "panel_sensitivity_heldout_r2.tsv")
    sh = _r(out, "panel_sensitivity_shares.tsv")
    L = ["## D. Tolerance-associated component across nested panels", "",
         "Held-out-gene R2 gain; P2 = adding the two tolerance references to P1; unique = P3 vs "
         "P1+glucocorticoid/IL-10; null q95 = 95th percentile of the gain with permuted tolerance "
         "references. Tolerance share in P2 and P3 from the full-gene fits.", "",
         "| target | R2 P1 | gain P2 | null q95 | unique gain | null q95 | gain KC_LPS alone | tol. gain beyond KC_LPS | tol. share P2 | tol. share P3 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t, d in p.groupby("target", sort=False):
        g = d.set_index("comparison")
        s = sh[sh.target == t].set_index("panel")
        u = g.loc["tolerance_unique (P3 vs P1+GC)"]
        L.append(f"| {t} | {g.loc['P1', 'heldout_r2']:.3f} | {g.loc['P2', 'delta_heldout_r2']:.3f} | "
                 f"{g.loc['P2', 'null_delta_q95']:.3f} | {u.delta_heldout_r2:.3f} | {u.null_delta_q95:.3f} | "
                 f"{g.loc['KC_LPS (P1+KC_LPS vs P1)', 'delta_heldout_r2']:.3f} | "
                 f"{g.loc['tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)', 'delta_heldout_r2']:.3f} | "
                 f"{s.loc['P2', 'tolerance_share']:.2f} | {s.loc['P3', 'tolerance_share']:.2f} |")
    return L + [""]


def write_summary(out: Path, cfg, settings) -> Path:
    ns = _r(out, "noise_settings.tsv")
    nm = _r(out, "noise_model.tsv")
    L = ["# Stage 09 identifiability: key numbers", "",
         "Generated by scripts/stages/09_identifiability.py from the tables in this folder.", "",
         "Noise settings (root-mean-square of the noise relative to a unit-RMS signal): "
         + "; ".join(f"mouse_{r.study} = {r.noise_to_signal_rms:.2f} (sampling error of the observed "
                     f"4-vs-4 log2FC)" for r in nm.itertuples()) + "; "
         + "; ".join(f"{k} = " + ", ".join(f"{r.source}:{r.level_rms:.2f}" for r in d.itertuples())
                     for k, d in ns[ns.kind == "resid"].groupby("setting")), ""]
    L += section_a(out) + section_b(out) + section_c(out) + section_d(out)
    p = out / "SUMMARY.md"
    p.write_text("\n".join(L) + "\n")
    return p
