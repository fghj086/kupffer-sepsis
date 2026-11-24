"""Stage 09 C: does stimulus attribution transfer between laboratories at the class level?

Every single-stimulus reference is attributed with a panel from which all references of its own
GEO series are removed (the three GSE224127 count files count as one series). The expected class
is the reference's own class (PAMP / DAMP / cytokine / resolution). Two panels:
  paper     macrophage references -> macrophage_resolution, liver references -> liver_plus
  extended  every single-stimulus reference of the other series (macrophage and liver)
Combined stimuli and the inactive fully reduced HMGB1 arm are not attributed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.attr_model import fit_nnls, unit_rms
from lib.attr_panel import informative
from lib.ident_methods import correlations


def series(study: str) -> str:
    return study.split("_")[0]


def _fit(method, y, R):
    if method == "nnls":
        b, r2 = fit_nnls(y, R)
        return b, r2
    return np.clip(correlations(y, R, method), 0, None), np.nan


def leave_one_study_out(rs, cfg06, cfg) -> pd.DataFrame:
    refs = cfg06["references"]
    c = cfg["transfer"]
    single = [r for r, v in refs.items() if v["cls"] in c["classes"] and r not in c["exclude"]]
    tissue = {r: "liver" if refs[r]["study"].startswith("GSE224127") else "macrophage" for r in single}
    rows = []
    for tgt in single:
        s = series(refs[tgt]["study"])
        panels = {
            "paper": [r for r in cfg06["panels"]["liver_plus" if tissue[tgt] == "liver"
                                                else "macrophage_resolution"]
                      if series(refs[r]["study"]) != s],
            "extended": [r for r in single if series(refs[r]["study"]) != s],
        }
        for pname, panel in panels.items():
            g = informative(rs, panel, rs.vec[tgt])
            y = unit_rms(rs.vec[tgt].reindex(g).values.astype(float))
            R = unit_rms(rs.matrix(panel, g).values.astype(float))
            cls = np.array([refs[r]["cls"] for r in panel])
            classes = sorted(set(cls))
            for method in c["methods"]:
                b, r2 = _fit(method, y, R)
                w = b / b.sum() if b.sum() > 0 else b
                cs = {k: float(w[cls == k].sum()) for k in classes}
                top_cls = max(cs, key=cs.get) if b.sum() > 0 else "none"
                top_ref = panel[int(np.argmax(w))] if b.sum() > 0 else "none"
                exp = refs[tgt]["cls"]
                rows.append({
                    "target": tgt, "series_left_out": s, "tissue": tissue[tgt],
                    "contrast": "difference" if tgt in c["difference_contrasts"] else "stimulus",
                    "expected_class": exp, "panel": pname, "method": method,
                    "n_references": len(panel), "n_genes": len(g),
                    "expected_class_in_panel": exp in classes,
                    "top_class": top_cls, "top_class_correct": top_cls == exp,
                    "expected_class_share": cs.get(exp, 0.0),
                    "chance_share": float(np.mean(cls == exp)),
                    "top_reference": top_ref, "r2_total": r2,
                    "class_shares": ";".join(f"{k}={v:.2f}" for k, v in cs.items())})
    return pd.DataFrame(rows)


def transfer_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = t[t.expected_class_in_panel]
    for keys, d in t.groupby(["panel", "method", "contrast"]):
        for cl, dd in [("all", d)] + list(d.groupby("expected_class")):
            rows.append(dict(zip(["panel", "method", "contrast"], keys)) | {
                "expected_class": cl, "n_targets": len(dd),
                "top_class_accuracy": float(dd.top_class_correct.mean()),
                "mean_expected_class_share": float(dd.expected_class_share.mean()),
                "mean_chance_share": float(dd.chance_share.mean())})
    return pd.DataFrame(rows)
