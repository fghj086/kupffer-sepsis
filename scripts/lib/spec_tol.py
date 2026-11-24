"""Stage 10, part 4: how specific and how robust is the tolerance-associated component?

4a  Real-reference control. For each target, the held-out-gene R2 gain from adding one real
    reference to panel P1 (stage 09 panel_sensitivity.base), for every reference of the stage 06
    RefSet that is neither in P1 nor a tolerance reference, next to the gain from each tolerance
    reference and from the tolerance pair; also every pair of non-tolerance references, so that
    the two-reference tolerance step is ranked among two-reference steps.
    Gene universes: "union" = responsive genes of P1, tolerance and every candidate reference
    (no reference is favoured by the gene choice); "stage09" = responsive genes of P3 (the stage
    09 choice, which contains the tolerance-responsive genes but not those of most candidates).
4b  Per mouse. Each CLP mouse vs the sham mean of its study and sorted fraction (sorted KC 2 h and
    8 h, CD45+ non-KC 8 h): tolerance step P2 vs P1 and tolerance beyond the cultured-KC LPS
    response (P2+KC_LPS vs P1+KC_LPS), with a null from permuting the gene labels of the
    tolerance references (stage 09). Sham mice against the mean of the other sham mice give the
    gain expected from mouse-to-mouse noise alone.
Held-out R2 and splits reuse ident_panels (stage 09); NNLS from attr_model.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lib.attr_model import unit_rms
from lib.attr_panel import informative
from lib.attr_ref import median_ratio_sf
from lib.ident_panels import heldout_r2, make_splits


def _design(rs, refs, y, universe_refs):
    g = informative(rs, universe_refs, y)
    g = g[rs.matrix(refs, g).notna().all(axis=1).values]
    yy = unit_rms(y.reindex(g).values.astype(float))
    R = pd.DataFrame(unit_rms(rs.matrix(refs, g).values.astype(float)), index=g, columns=refs)
    return yy, R


def _cv(y, R, names, reps):
    return np.array([heldout_r2(y, R[list(names)].values, s) for s in reps])


def real_reference_gains(targets, rs, ps_cfg, tcfg, cls, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    P1, tol = list(ps_cfg["base"]), list(ps_cfg["tolerance"])
    P3 = P1 + tol + list(ps_cfg["glucocorticoid_il10"])
    cand = [r for r in rs.vec if r not in P1 + tol and r not in tcfg["exclude_candidates"]]
    allrefs = P1 + tol + cand
    universes = {"union": allrefs, "stage09": P3}
    by = {t["target"]: t for t in targets}
    rows, summ = [], []
    for name in tcfg["targets"]:
        if name not in by:
            continue
        t = by[name]
        for uname, uref in universes.items():
            y, R = _design(rs, allrefs, t["y"], uref)
            reps = [make_splits(len(y), tcfg["n_folds"], rng) for _ in range(tcfg["n_repeats"])]
            base = _cv(y, R, P1, reps)
            meta = {"target": name, "cell": t.get("cell"), "universe": uname, "n_genes": len(y),
                    "heldout_r2_P1": base.mean()}
            single = {}
            for r in tol + cand:
                d = _cv(y, R, P1 + [r], reps) - base
                single[r] = d.mean()
                rows.append(meta | {"added": r, "n_added": 1, "is_tolerance": r in tol,
                                    "reference_class": cls.get(r), "gain_mean": d.mean(),
                                    "gain_sd": d.std(ddof=1), "frac_repeats_gt0": float(np.mean(d > 0))})
            d = _cv(y, R, P1 + tol, reps) - base
            pair_tol = d.mean()
            rows.append(meta | {"added": "+".join(tol), "n_added": 2, "is_tolerance": True,
                                "reference_class": "resolution", "gain_mean": pair_tol,
                                "gain_sd": d.std(ddof=1), "frac_repeats_gt0": float(np.mean(d > 0))})
            pair_gains = {}
            if uname == tcfg["pairs_universe"]:
                for a, b in itertools.combinations(cand, 2):
                    pair_gains[f"{a}+{b}"] = (_cv(y, R, P1 + [a, b], reps) - base).mean()
            nontol = pd.Series({k: v for k, v in single.items() if k not in tol}).sort_values(ascending=False)
            s = meta | {"n_candidates": len(nontol), "best_nontolerance": nontol.index[0],
                        "best_nontolerance_gain": nontol.iloc[0],
                        "median_nontolerance_gain": float(nontol.median())}
            for r in tol:
                s[f"gain_{r}"] = single[r]
                s[f"rank_{r}_among_singles"] = int(1 + np.sum(nontol.values > single[r]))
                s[f"frac_nontolerance_ge_{r}"] = float(np.mean(nontol.values >= single[r]))
            s["gain_tolerance_pair"] = pair_tol
            if pair_gains:
                pg = pd.Series(pair_gains).sort_values(ascending=False)
                s |= {"n_nontolerance_pairs": len(pg), "best_nontolerance_pair": pg.index[0],
                      "best_nontolerance_pair_gain": pg.iloc[0],
                      "rank_tolerance_pair_among_pairs": int(1 + np.sum(pg.values > pair_tol)),
                      "frac_nontolerance_pairs_ge_tolerance_pair": float(np.mean(pg.values >= pair_tol))}
            summ.append(s)
    return pd.DataFrame(rows), pd.DataFrame(summ)


# ------------------------------------------------------------------ 4b per mouse
def per_mouse_vectors(counts: pd.DataFrame, meta: pd.DataFrame, min_mean: float) -> dict:
    """{label: (kind, log2 vector)}: CLP mouse i minus sham mean; sham mouse j minus the mean of
    the other sham mice (noise-only contrast)."""
    C = counts[meta.index]
    C = C[C.mean(axis=1) >= min_mean]
    L = np.log2(C / median_ratio_sf(C) + 1.0)
    clp = list(meta.index[meta.condition == "CLP"])
    sham = list(meta.index[meta.condition == "sham"])
    out = {c: ("CLP_vs_sham_mean", L[c] - L[sham].mean(axis=1)) for c in clp}
    for s in sham:
        rest = [x for x in sham if x != s]
        out[s] = ("sham_vs_other_sham (noise control)", L[s] - L[rest].mean(axis=1))
    return out


def per_mouse_gains(vectors: dict, unit: str, rs, ps_cfg, tcfg, rng) -> pd.DataFrame:
    P1, tol, kcl = list(ps_cfg["base"]), list(ps_cfg["tolerance"]), list(ps_cfg["cell_matched_lps"])
    P3 = P1 + tol + list(ps_cfg["glucocorticoid_il10"])
    cand = [r for r in rs.vec if r not in P1 + tol and r not in tcfg["exclude_candidates"]]
    panels = {"P1": P1, "P2": P1 + tol, "P1+KC_LPS": P1 + kcl, "P2+KC_LPS": P1 + tol + kcl}
    comps = {"tolerance step (P2 vs P1)": ("P2", "P1"),
             "tolerance beyond KC_LPS (P2+KC_LPS vs P1+KC_LPS)": ("P2+KC_LPS", "P1+KC_LPS")}
    pm = tcfg["per_mouse"]
    rows = []
    for lib, (kind, v) in vectors.items():
        for uname, uref in {"stage09": P3 + kcl, "union": P1 + tol + cand}.items():
            refs = P1 + tol + (cand if uname == "union" else kcl)
            y, R = _design(rs, refs, v, uref)
            reps = [make_splits(len(y), tcfg["n_folds"], rng) for _ in range(tcfg["n_repeats"])]
            cv = {p: _cv(y, R, n, reps) for p, n in panels.items()}
            null = {c: [] for c in comps}
            for _ in range(pm["n_perm"]):
                Rp = R.copy()
                for r in tol:
                    Rp[r] = rng.permutation(Rp[r].values)
                s = reps[rng.integers(len(reps))]
                r_ = {p: heldout_r2(y, (Rp if p.startswith("P2") else R)[n].values, s)
                      for p, n in panels.items()}
                for c, (a, b) in comps.items():
                    null[c].append(r_[a] - r_[b])
            base = {"unit": unit, "library": lib, "contrast": kind, "universe": uname, "n_genes": len(y),
                    "heldout_r2_P1": cv["P1"].mean()}
            for c, (a, b) in comps.items():
                d = cv[a] - cv[b]
                nl = np.asarray(null[c])
                rows.append(base | {"comparison": c, "gain_mean": d.mean(), "gain_sd": d.std(ddof=1),
                                    "null_mean": nl.mean(), "null_q95": np.quantile(nl, 0.95),
                                    "p_null_ge_observed": (1 + np.sum(nl >= d.mean())) / (1 + len(nl)),
                                    "above_null_q95": bool(d.mean() > np.quantile(nl, 0.95))})
            if uname == "union":
                g = {r: (_cv(y, R, P1 + [r], reps) - cv["P1"]).mean() for r in tol + cand}
                nontol = np.array([g[r] for r in cand])
                for r in tol:
                    rows.append(base | {"comparison": f"single {r} vs real non-tolerance singles",
                                        "gain_mean": g[r], "best_nontolerance_gain": nontol.max(),
                                        "best_nontolerance": cand[int(np.argmax(nontol))],
                                        "rank_among_singles": int(1 + np.sum(nontol > g[r])),
                                        "n_candidates": len(cand)})
    return pd.DataFrame(rows)


def per_mouse_summary(pm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = pm[pm.comparison.str.startswith("tolerance")]
    for (unit, contrast, uni, comp), g in d.groupby(["unit", "contrast", "universe", "comparison"]):
        rows.append({"unit": unit, "contrast": contrast, "universe": uni, "comparison": comp,
                     "n_mice": len(g), "n_gain_positive": int((g.gain_mean > 0).sum()),
                     "n_above_null_q95": int(g.above_null_q95.sum()),
                     "gain_min": g.gain_mean.min(), "gain_median": g.gain_mean.median(),
                     "gain_max": g.gain_mean.max()})
    s = pm[pm.comparison.str.startswith("single")]
    for (unit, contrast, comp), g in s.groupby(["unit", "contrast", "comparison"]):
        rows.append({"unit": unit, "contrast": contrast, "universe": "union", "comparison": comp,
                     "n_mice": len(g), "n_rank1": int((g.rank_among_singles == 1).sum()),
                     "rank_median": g.rank_among_singles.median(),
                     "gain_median": g.gain_mean.median()})
    return pd.DataFrame(rows)
