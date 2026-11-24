"""Stage 01 helpers: leave-one-replicate-out validation of the stimulus signatures.

For replicate k, every arm loses replicate k; DESeq2, the scaling slope and all signatures
are rebuilt from the 10 remaining libraries (2 per arm); the 5 held-out libraries are scored
with genes z-scored against the training libraries only. Nothing estimated from a held-out
library enters its own score.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .perturb_de import all_contrasts, log_cpm
from .perturb_sig import AXIS, build_signatures, residualise, score, zscore_ref

log = logging.getLogger("01_perturbation")


def loro_scores(counts, meta, cfg):
    de, sg, sc = cfg["de"], cfg["signatures"], cfg["scoring"]
    lc = log_cpm(counts, sc["pseudocount"])
    held, folds, fsigs = [], [], []
    for k in sorted(meta.replicate.unique()):
        tr = meta.index[meta.replicate != k]
        te = meta.index[meta.replicate == k]
        res, slope = all_contrasts(counts[tr], meta, cfg["reference"], cfg["stimuli"],
                                   de["alpha"], de["min_abs_lfc"], de["n_cpus"])
        sigs, ncand = build_signatures(res, de, sg)
        fsigs.append(sigs.assign(fold=k))
        z = zscore_ref(lc, tr)
        s_tr, s_te = score(z[tr], sigs), score(z[te], sigs)
        r_tr, r_te, b, a = residualise(s_tr, s_te, AXIS, "shared_inflammatory")
        s_te = s_te.assign(axis_adj_shared=r_te, fold=k, set="held_out")
        s_tr = s_tr.assign(axis_adj_shared=r_tr, fold=k, set="training")
        held.append(pd.concat([s_te, s_tr]))
        folds.append({"fold": k, "slope_ds_on_lps": slope, "axis_on_shared_slope": b,
                      "axis_on_shared_intercept": a,
                      **{f"n_candidates_{n}": v for n, v in ncand.items()},
                      **{f"n_genes_{n}": int((sigs.signature == n).sum())
                         for n in sigs.signature.unique()}})
        log.info("fold %s: slope %.3f, candidates %s", k, slope, ncand)
    out = pd.concat(held)
    out.index.name = "sample"
    out = out.join(meta[["condition", "replicate"]])
    return out.reset_index(), pd.DataFrame(folds), pd.concat(fsigs, ignore_index=True)


def _exact_perm_p(x, y) -> float:
    """Two-sided exact label-permutation P for a difference in means (all splits)."""
    pool = np.concatenate([x, y])
    obs = abs(x.mean() - y.mean())
    n, cnt, tot = len(x), 0, 0
    for idx in itertools.combinations(range(len(pool)), n):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        cnt += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12
        tot += 1
    return cnt / tot


def _hedges_g(x, y) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
    return float(d * (1 - 3 / (4 * (nx + ny) - 9)))


def separation_stats(scores: pd.DataFrame, conds) -> pd.DataFrame:
    """Pairwise separation of held-out libraries on each score, plus a linear model of the
    axis on condition with the shared score (and held-out replicate) as covariates."""
    h = scores[scores.set == "held_out"]
    rows = []
    metrics = ["shared_inflammatory", "HMGB1_biased", "LPS_biased", AXIS, "axis_adj_shared"]
    for met in metrics:
        for c1, c2 in itertools.combinations(conds, 2):
            x = h.loc[h.condition == c1, met].values
            y = h.loc[h.condition == c2, met].values
            fold_order = sum(
                h[(h.condition == c1) & (h.fold == k)][met].values[0]
                > h[(h.condition == c2) & (h.fold == k)][met].values[0]
                for k in sorted(h.fold.unique()))
            rows.append({"analysis": "pairwise_held_out", "score": met, "group1": c1,
                         "group2": c2, "n1": len(x), "n2": len(y),
                         "mean1": x.mean(), "mean2": y.mean(), "diff": x.mean() - y.mean(),
                         "hedges_g": _hedges_g(x, y),
                         "welch_p": stats.ttest_ind(x, y, equal_var=False).pvalue,
                         "exact_perm_p": _exact_perm_p(x, y),
                         "folds_group1_higher": int(fold_order),
                         "n_folds": int(h.fold.nunique())})
    # axis ~ condition + shared (+ replicate), dsHMGB1 as reference level
    d = h.copy()
    d["condition"] = pd.Categorical(d["condition"], categories=["dsHMGB1"] +
                                    [c for c in conds if c != "dsHMGB1"])
    d = d.rename(columns={AXIS: "axis"})
    for form, label in [("axis ~ C(condition) + shared_inflammatory", "lm_condition_plus_shared"),
                        ("axis ~ C(condition) + shared_inflammatory + C(fold)",
                         "lm_condition_plus_shared_plus_replicate")]:
        fit = smf.ols(form, data=d).fit()
        for term in fit.params.index:
            rows.append({"analysis": label, "score": AXIS, "group1": term,
                         "group2": "dsHMGB1 (reference)" if "condition" in term else "",
                         "n1": int(fit.nobs), "diff": fit.params[term],
                         "se": fit.bse[term], "welch_p": fit.pvalues[term],
                         "r2": fit.rsquared})
    return pd.DataFrame(rows)


def independence_stats(scores: pd.DataFrame, full_scores: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the axis with the shared score: full-data scores (all 15 and the
    12 stimulated libraries) and held-out scores (stimulated libraries)."""
    rows = []
    h = scores[scores.set == "held_out"]
    for label, d in [("full_data_all", full_scores),
                     ("full_data_stimulated", full_scores[full_scores.condition != "PBS"]),
                     ("full_data_ds_lps_lpsifng", full_scores[full_scores.condition.isin(
                         ["dsHMGB1", "LPS", "LPS_IFNg"])]),
                     ("held_out_stimulated", h[h.condition != "PBS"]),
                     ("held_out_ds_lps_lpsifng", h[h.condition.isin(
                         ["dsHMGB1", "LPS", "LPS_IFNg"])])]:
        r, p = stats.pearsonr(d["shared_inflammatory"], d[AXIS])
        rho, ps = stats.spearmanr(d["shared_inflammatory"], d[AXIS])
        rows.append({"analysis": "axis_vs_shared_correlation", "score": AXIS,
                     "group1": label, "n1": len(d), "pearson_r": r, "pearson_p": p,
                     "spearman_rho": rho, "spearman_p": ps})
    return pd.DataFrame(rows)


def nearest_centroid(scores: pd.DataFrame, feats) -> pd.DataFrame:
    """Classify each held-out library to the nearest training-condition centroid in the
    given score space (scores standardised with training SDs within the fold)."""
    out = []
    for k, f in scores.groupby("fold"):
        tr, te = f[f.set == "training"], f[f.set == "held_out"]
        sd = tr[feats].std(ddof=1)
        cen = tr.groupby("condition")[feats].mean()
        for s, r in te.iterrows():
            d = (((cen - r[feats]) / sd) ** 2).sum(axis=1) ** 0.5
            out.append({"fold": k, "sample": r["sample"], "condition": r["condition"],
                        "features": "+".join(feats), "predicted": d.idxmin(),
                        "correct": d.idxmin() == r["condition"]})
    return pd.DataFrame(out)
