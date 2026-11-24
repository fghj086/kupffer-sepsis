"""Stage 06 positive controls, run before any septic-liver target is attributed.

1. Held-out stimuli (never in a panel): LPS+IFN-g BMDM (two studies), LPS in RAW cells, cultured
   Kupffer cells and a second BMDM study, and in vivo cytokine pairs; each must load mainly on
   its known components.
2. Leave-one-replicate-out: replicate k of a reference (against the control mean without
   control k) is attributed with that reference rebuilt from the remaining replicates.
3. Synthetic mixtures of two held-out replicate vectors with known weights.
Responsive-gene sets come from the full data of each study (the held-out replicate contributes
to gene selection, not to the reference values); this is stated with the results.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lib.attr_model import attribute, fit_nnls, unit_rms
from lib.attr_panel import informative


def _swap(rs, panel, ref, rebuilt, genes):
    M = rs.matrix(panel, genes).copy()
    M[ref] = rebuilt.reindex(genes)
    return M


def heldout_controls(rs, cfg, rng) -> pd.DataFrame:
    rows = []
    for tgt, spec in cfg["controls"]["heldout"].items():
        panel = cfg["panels"][spec["panel"]]
        y = rs.vec[tgt]
        g = informative(rs, panel, y)
        res, s = attribute(y.reindex(g), rs.matrix(panel, g), 200, rng)
        exp = set(spec["expect"])
        top = res.sort_values("share", ascending=False).reference.iloc[0]
        rows.append({"control": "held-out stimulus", "target": tgt, "panel": spec["panel"],
                     "expected": ";".join(spec["expect"]), "top_reference": top,
                     "top_in_expected": top in exp,
                     "share_on_expected": float(res[res.reference.isin(exp)].share.sum()),
                     "r2_total": s["r2_total"], "n_genes": s["n_genes"],
                     "loadings": ";".join(f"{r.reference}={r.share:.2f}" for r in
                                          res.sort_values("share", ascending=False).itertuples()
                                          if r.share > 0.02)})
    return pd.DataFrame(rows)


def loro_controls(rs, cfg, panel_name) -> pd.DataFrame:
    panel = cfg["panels"][panel_name]
    g = informative(rs, panel)
    rows = []
    for ref in panel:
        for col, v, rebuilt, drop in rs.held[ref]:
            M = _swap(rs, panel, ref, rebuilt, g)
            ok = M.notna().all(1) & v.reindex(g).notna()
            Y = unit_rms(v.reindex(g)[ok].values)
            R = unit_rms(M[ok].values)
            b, r2 = fit_nnls(Y, R)
            sh = b / b.sum() if b.sum() > 0 else b
            j = panel.index(ref)
            rows.append({"control": "leave-one-replicate-out", "panel": panel_name,
                         "reference": ref, "heldout_column": col,
                         "dropped_columns": ";".join(drop), "self_share": float(sh[j]),
                         "top_reference": panel[int(np.argmax(b))],
                         "top_is_self": int(np.argmax(b)) == j, "r2_total": r2,
                         "n_genes": int(ok.sum())})
    return pd.DataFrame(rows)


def mixture_controls(rs, cfg, panel_name, rng) -> pd.DataFrame:
    panel = cfg["panels"][panel_name]
    g = informative(rs, panel)
    rows = []
    for a, b in cfg["controls"]["mixture_pairs"]:
        if a not in panel or b not in panel:
            continue
        ha, hb = rs.held[a], rs.held[b]
        for (ka, (ca, va, ra, _)), (kb, (cb, vb, rb, _)) in itertools.product(
                enumerate(ha), enumerate(hb)):
            if ca == cb:
                continue
            M = _swap(rs, panel, a, ra, g)
            M[b] = rb.reindex(g)
            ok = M.notna().all(1) & va.reindex(g).notna() & vb.reindex(g).notna()
            R = unit_rms(M[ok].values)
            A, B = unit_rms(va.reindex(g)[ok].values), unit_rms(vb.reindex(g)[ok].values)
            for w in cfg["controls"]["mixture_weights"]:
                coef, r2 = fit_nnls(unit_rms(w * A + (1 - w) * B), R)
                ia, ib = panel.index(a), panel.index(b)
                ab = coef[ia] + coef[ib]
                rows.append({"control": "synthetic mixture", "panel": panel_name,
                             "reference_a": a, "reference_b": b, "column_a": ca, "column_b": cb,
                             "true_weight_a": w,
                             "recovered_weight_a": coef[ia] / ab if ab > 0 else np.nan,
                             "share_on_pair": ab / coef.sum() if coef.sum() > 0 else np.nan,
                             "r2_total": r2})
    return pd.DataFrame(rows)


def summarise_controls(held, loro, mix) -> pd.DataFrame:
    rows = []
    for r in held.itertuples():
        rows.append({"control": r.control, "panel": r.panel, "item": r.target,
                     "n": 1, "metric": "share on expected references", "value": r.share_on_expected,
                     "pass": bool(r.top_in_expected and r.share_on_expected >= 0.5),
                     "detail": r.loadings})
    for (p, ref), d in loro.groupby(["panel", "reference"], sort=False):
        rows.append({"control": "leave-one-replicate-out", "panel": p, "item": ref, "n": len(d),
                     "metric": "fraction of held-out replicates with self as top reference",
                     "value": float(d.top_is_self.mean()),
                     "pass": bool(d.top_is_self.mean() >= 0.5),
                     "detail": f"median self share {d.self_share.median():.2f}; "
                               f"top otherwise: {';'.join(sorted(set(d.top_reference) - {ref}))}"})
    if len(mix):
        for (p, a, b), d in mix.groupby(["panel", "reference_a", "reference_b"], sort=False):
            err = (d.recovered_weight_a - d.true_weight_a).abs()
            rows.append({"control": "synthetic mixture", "panel": p, "item": f"{a} + {b}",
                         "n": len(d), "metric": "median |recovered - true weight|",
                         "value": float(err.median()),
                         "pass": bool(err.median() <= 0.15 and d.share_on_pair.median() >= 0.6),
                         "detail": f"median share on the pair {d.share_on_pair.median():.2f}; "
                                   f"90th pct error {err.quantile(0.9):.2f}"})
    return pd.DataFrame(rows)
