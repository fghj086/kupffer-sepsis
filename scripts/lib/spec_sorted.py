"""Stage 10, part 1: KC vs CD45+ non-KC in sorted liver cells (GSE311736, CLP 8 h vs sham) and
KC alone at 2 h (GSE328774).

Scores per library: mean gene-centred log2(median-of-ratios normalised count + 1) over a gene
set (stage 02 normalisation, computed jointly over the KC and CD45+ non-KC libraries so that the
two sorted fractions are on one scale); signed axis = mean of the HMGB1-like half minus mean of
the LPS-like half (tc_scores.score_sets). Replicate pairing (same mouse behind KC replicate k and
CD45+ non-KC replicate k) is not recorded in GEO; it is examined from shared per-replicate
residual expression across all five sorted fractions.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from scripts.lib.sorted_stats import filter_genes, hedges_g, log_norm
from scripts.lib.spec_stats import boot_groups, did_unpaired, exact_perm_p, two_group
from scripts.lib.tc_scores import score_sets


def library_scores(counts, meta, sets, weights, scfg) -> tuple[pd.DataFrame, dict]:
    C = filter_genes(counts[meta.index], scfg["min_count"], scfg["min_samples"])
    L = log_norm(C, scfg["pseudocount"])
    S, used = score_sets(L, list(L.columns), sets, weights)
    S = S.join(meta[["dataset", "cell_type", "condition", "library"]])
    S["replicate"] = S.library.str.extract(r"-(\d+)$")[0].astype(int)
    return S, {k: len(v) for k, v in used.items()}


def _vals(S, ct, cond, score):
    d = S[(S.cell_type == ct) & (S.condition == cond)].sort_values("replicate")
    return d[score].values, d.replicate.values


def interaction(S, scores, n_boot, rng) -> pd.DataFrame:
    rows = []
    for sc in scores:
        kc_c, r1 = _vals(S, "KC", "CLP", sc)
        cd_c, r2 = _vals(S, "CD45_nonKC", "CLP", sc)
        kc_s, r3 = _vals(S, "KC", "sham", sc)
        cd_s, r4 = _vals(S, "CD45_nonKC", "sham", sc)
        u = did_unpaired(kc_c, cd_c, kc_s, cd_s, n_boot, rng)
        rows.append({"score": sc, "analysis": "unpaired (libraries independent)"} | u)
        assert (r1 == r2).all() and (r3 == r4).all()
        dc, ds = kc_c - cd_c, kc_s - cd_s
        bt = boot_groups([dc, ds], lambda a, b: a.mean() - b.mean(), n_boot, rng)
        g, glo, ghi = hedges_g(dc, ds)
        rows.append({"score": sc, "analysis": "paired by replicate number (KC_k - CD45_k per mouse)",
                     "gap_sham": ds.mean(), "gap_clp": dc.mean(), "delta_disease": dc.mean() - ds.mean(),
                     "boot_ci_low": np.quantile(bt, 0.025), "boot_ci_high": np.quantile(bt, 0.975),
                     "hedges_g": g, "g_ci_low": glo, "g_ci_high": ghi,
                     "p_exact_permutation": exact_perm_p(dc, ds),
                     "per_mouse_gap_clp": ",".join(f"{x:.3f}" for x in dc),
                     "per_mouse_gap_sham": ",".join(f"{x:.3f}" for x in ds)})
    out = pd.DataFrame(rows)
    out["direction"] = np.where(np.sign(out.gap_clp) != np.sign(out.gap_sham), "gap reverses sign",
                                np.where(out.gap_clp.abs() > out.gap_sham.abs(), "gap widens", "gap narrows"))
    return out


def adjusted_interaction(S, scores, shared, n_boot, rng) -> pd.DataFrame:
    """Scores residualised on the shared inflammatory score (one fit over the 16 libraries with a
    cell-type intercept), then the unpaired difference-in-differences: KC vs CD45+ non-KC change
    at equal shared inflammatory score."""
    rows = []
    kc = (S.cell_type == "KC").astype(float).values
    for sc in [s for s in scores if s != shared]:
        X = np.column_stack([np.ones(len(S)), kc, S[shared].values])
        b = np.linalg.lstsq(X, S[sc].values, rcond=None)[0]
        R = S.assign(resid=S[sc].values - X @ b)
        g = [_vals(R, ct, c, "resid")[0] for ct, c in
             [("KC", "CLP"), ("CD45_nonKC", "CLP"), ("KC", "sham"), ("CD45_nonKC", "sham")]]
        rows.append({"score": sc, "analysis": "unpaired, adjusted for shared inflammatory score",
                     "slope_on_shared": b[2]} | did_unpaired(*g, n_boot, rng))
    return pd.DataFrame(rows)


def within_changes(S_by_unit: dict, scores, n_boot, rng) -> pd.DataFrame:
    rows = []
    for unit, S in S_by_unit.items():
        for sc in scores:
            a = S[S.condition == "CLP"][sc].values
            b = S[S.condition == "sham"][sc].values
            rows.append({"unit": unit, "score": sc} | two_group(a, b, n_boot, rng))
    return pd.DataFrame(rows)


def composition(counts, meta, markers: dict, scfg) -> pd.DataFrame:
    """Marker levels in CD45+ non-KC (and KC for reference), sham vs CLP, log2 norm and CPM."""
    C = filter_genes(counts[meta.index], 1, 1)
    L = log_norm(C, scfg["pseudocount"])
    cpm = C / C.sum() * 1e6
    rows = []
    for grp, genes in markers.items():
        for gname in genes:
            if gname not in L.index:
                continue
            for ct in ("CD45_nonKC", "KC"):
                r = {"marker_group": grp, "gene": gname, "cell_type": ct}
                for cond in ("sham", "CLP"):
                    idx = meta.index[(meta.cell_type == ct) & (meta.condition == cond)]
                    r[f"log2norm_{cond}"] = float(L.loc[gname, idx].mean())
                    r[f"cpm_{cond}"] = float(cpm.loc[gname, idx].mean())
                r["log2fc_clp_vs_sham"] = r["log2norm_CLP"] - r["log2norm_sham"]
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ replicate pairing
def _resid(counts, meta, ct, cond, min_mean, pseudocount):
    idx = meta.index[(meta.cell_type == ct) & (meta.condition == cond)]
    C = counts[idx]
    C = C[C.mean(axis=1) >= min_mean]
    L = log_norm(C, pseudocount)
    R = L.sub(L.mean(axis=1), axis=0)
    R.columns = meta.loc[idx, "library"].str.extract(r"-(\d+)$")[0].astype(int).values
    return R


def _corr_block(A, B):
    g = A.index.intersection(B.index)
    a = A.loc[g].values
    b = B.loc[g].values
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return pd.DataFrame(a.T @ b / len(g), index=A.columns, columns=B.columns), len(g)


def pairing_evidence(counts, meta, scfg, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per condition, residual (library minus arm mean) expression of replicate k in one sorted
    fraction correlated with replicate j in another. If the same mouse (or the same processing
    batch) is behind replicate k of every fraction, matching replicates correlate more.
    Exact P per fraction pair (all relabelings of one side); combined P over all fraction pairs by
    permuting replicate labels independently within each fraction (Monte-Carlo)."""
    pair_rows, comb_rows = [], []
    cts = sorted(meta.cell_type.unique())
    for cond in ("CLP", "sham"):
        R = {ct: _resid(counts, meta, ct, cond, scfg["pairing_min_mean_count"], scfg["pseudocount"])
             for ct in cts}
        blocks = {}
        for a, b in itertools.combinations(cts, 2):
            M, n = _corr_block(R[a], R[b])
            reps = sorted(set(M.index) & set(M.columns))
            M = M.loc[reps, reps]
            blocks[(a, b)] = M
            diag = float(np.mean(np.diag(M.values)))
            perm = [np.mean([M.values[i, p[i]] for i in range(len(reps))])
                    for p in itertools.permutations(range(len(reps)))]
            pair_rows.append({"condition": cond, "fraction_a": a, "fraction_b": b, "n_genes": n,
                              "n_replicates": len(reps), "mean_r_matching": diag,
                              "mean_r_nonmatching": float(M.values[~np.eye(len(reps), dtype=bool)].mean()),
                              "p_exact_label_permutation": float(np.mean(np.array(perm) >= diag - 1e-12)),
                              "matrix": ";".join(",".join(f"{x:.3f}" for x in r) for r in M.values)})
        # combined statistic over all fraction pairs
        reps_of = {ct: list(R[ct].columns) for ct in cts}

        def total(lab):
            s, k = 0.0, 0
            for (a, b), M in blocks.items():
                for r in M.index:
                    ra, rb = lab[a][r], lab[b][r]
                    if ra in M.index and rb in M.columns:
                        s += M.loc[ra, rb]
                        k += 1
            return s / k
        ident = {ct: {r: r for r in reps_of[ct]} for ct in cts}
        obs = total(ident)
        null = []
        for _ in range(scfg["n_perm_pairing"]):
            lab = {ct: dict(zip(reps_of[ct], rng.permutation(reps_of[ct]))) for ct in cts}
            null.append(total(lab))
        null = np.array(null)
        comb_rows.append({"condition": cond, "fractions": ",".join(cts), "mean_r_matching_all_pairs": obs,
                          "null_mean": null.mean(), "null_q95": np.quantile(null, 0.95),
                          "p_monte_carlo": (1 + np.sum(null >= obs)) / (1 + len(null)),
                          "n_perm": len(null)})
    return pd.DataFrame(pair_rows), pd.DataFrame(comb_rows)


def kc_cd45_pairing_row(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs[(pairs.fraction_a == "CD45_nonKC") & (pairs.fraction_b == "KC")]
