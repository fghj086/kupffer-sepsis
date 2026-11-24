"""Stage 09 D: is a tolerance-associated component needed, whatever its share?

Nested panels (no merging of collinear references):
  P1 microbial / damage / cytokine
  P2 P1 + endotoxin-tolerance references
  P3 P2 + glucocorticoid and IL-10 references
  P1+GC  P1 + glucocorticoid and IL-10 (so that P3 - P1+GC is the unique gain of tolerance)
Each target is fitted on one fixed gene set (informative genes of P3 measured in the target), so
panels are compared on the same genes. Fit quality is the uncentred R2 on held-out genes: genes are
split into k folds, NNLS is fitted on k-1 folds and predicts the left-out fold; the split is
repeated and paired across panels. Null: the tolerance references with their gene labels
permuted, which gives the held-out gain expected from adding two arbitrary vectors with the same
value distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.attr_model import fit_nnls, unit_rms
from lib.attr_panel import informative


def nested_panels(c) -> dict:
    b, t, g, k = c["base"], c["tolerance"], c["glucocorticoid_il10"], c["cell_matched_lps"]
    return {"P1": b, "P2": b + t, "P3": b + t + g, "P1+GC": b + g,
            "P1+KC_LPS": b + k, "P2+KC_LPS": b + t + k}


def heldout_r2(y, R, splits) -> float:
    sse = sst = 0.0
    for tr, te in splits:
        b, _ = fit_nnls(y[tr], R[tr])
        sse += float(np.sum((y[te] - R[te] @ b) ** 2))
        sst += float(np.sum(y[te] ** 2))
    return 1 - sse / sst


def make_splits(n, k, rng):
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def panel_sensitivity(targets, rs, cfg, cls, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["panel_sensitivity"]
    P = nested_panels(c)
    tol, gc = c["tolerance"], c["glucocorticoid_il10"]
    rows, shares = [], []
    for t in targets:
        if t["kind"] != "stimulus":
            continue
        allr = P["P3"] + c["cell_matched_lps"]
        g = informative(rs, P["P3"], t["y"])
        g = g[rs.matrix(allr, g).notna().all(axis=1).values]
        y = unit_rms(t["y"].reindex(g).values.astype(float))
        Rall = pd.DataFrame(unit_rms(rs.matrix(allr, g).values.astype(float)),
                            index=g, columns=allr)
        reps = [make_splits(len(g), c["n_folds"], rng) for _ in range(c["n_repeats"])]
        cv = {p: np.array([heldout_r2(y, Rall[names].values, s) for s in reps]) for p, names in P.items()}
        for ref in tol:
            names = [r for r in P["P3"] if r != ref]
            cv[f"P3-{ref}"] = np.array([heldout_r2(y, Rall[names].values, s) for s in reps])
        # permutation null for the tolerance step
        null_step, null_unique = [], []
        for _ in range(c["n_perm"]):
            Rp = Rall.copy()
            for ref in tol:
                Rp[ref] = rng.permutation(Rp[ref].values)
            s = reps[rng.integers(len(reps))]
            r1 = heldout_r2(y, Rall[P["P1"]].values, s)
            r2 = heldout_r2(y, Rp[P["P2"]].values, s)
            r1g = heldout_r2(y, Rall[P["P1+GC"]].values, s)
            r3 = heldout_r2(y, Rp[P["P3"]].values, s)
            null_step.append(r2 - r1)
            null_unique.append(r3 - r1g)
        meta = {k: t.get(k) for k in ("target", "cell", "time_h", "family")} | {"n_genes": len(g)}
        comps = {"P1": ("P1", None), "P2": ("P2", "P1"), "P3": ("P3", "P2"),
                 "P1+GC": ("P1+GC", "P1"), "tolerance_unique (P3 vs P1+GC)": ("P3", "P1+GC")}
        comps |= {f"drop {r} from P3": (f"P3-{r}", "P3") for r in tol}
        # control: is the gain specific to tolerance, or to a Kupffer-cell LPS response from the
        # same study (cultured KC, LPS 6 h without pre-exposure)?
        comps |= {"KC_LPS (P1+KC_LPS vs P1)": ("P1+KC_LPS", "P1"),
                  "tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)": ("P2+KC_LPS", "P1+KC_LPS")}
        for label, (a, b) in comps.items():
            d = cv[a] - cv[b] if b else None
            row = meta | {"comparison": label, "heldout_r2": float(cv[a].mean()),
                          "heldout_r2_sd": float(cv[a].std(ddof=1))}
            if d is not None:
                row |= {"delta_heldout_r2": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                        "frac_repeats_delta_gt0": float(np.mean(d > 0))}
            if label == "P2":
                row |= _null(null_step, d)
            if label.startswith("tolerance_unique"):
                row |= _null(null_unique, d)
            rows.append(row)
        # full-data shares per panel
        for p, names in P.items():
            b, r2 = fit_nnls(y, Rall[names].values)
            w = b / b.sum() if b.sum() > 0 else b
            sh = dict(zip(names, w))
            shares.append(meta | {"panel": p, "r2_full_genes": r2,
                                  "tolerance_share": sum(sh.get(r, 0) for r in tol),
                                  "glucocorticoid_il10_share": sum(sh.get(r, 0) for r in gc),
                                  "PAMP_share": sum(v for r, v in sh.items() if cls[r] == "PAMP"),
                                  "DAMP_share": sum(v for r, v in sh.items() if cls[r] == "DAMP"),
                                  "cytokine_share": sum(v for r, v in sh.items() if cls[r] == "cytokine"),
                                  "dsHMGB1_share": sh.get("dsHMGB1", 0.0),
                                  **{f"share_{r}": sh.get(r, np.nan)
                                     for r in tol + gc + c["cell_matched_lps"]}})
    return pd.DataFrame(rows), pd.DataFrame(shares)


def _null(null, d):
    null = np.asarray(null)
    return {"null_delta_mean": float(null.mean()), "null_delta_q95": float(np.quantile(null, 0.95)),
            "p_null_ge_observed": float((1 + np.sum(null >= d.mean())) / (1 + len(null)))}


def reference_correlations(rs, refs) -> pd.DataFrame:
    g = informative(rs, refs)
    M = rs.matrix(refs, g).dropna()
    C = np.corrcoef(M.values.T)
    return pd.DataFrame([{"reference_a": a, "reference_b": b, "pearson_r": C[i, j], "n_genes": len(M)}
                         for i, a in enumerate(refs) for j, b in enumerate(refs) if j > i])
