"""Stage 09 B: attribution methods compared on the same held-out stimuli and mixtures.

nnls          non-negative least squares (stage 06 method)
nn_ridge      non-negative ridge: NNLS on [R; sqrt(lambda n) I], lambda chosen by 5-fold
              cross-validation over genes
nn_enet       non-negative elastic net (sklearn ElasticNetCV, positive, no intercept; alpha and
              l1_ratio chosen by 5-fold cross-validation over genes)
pearson / spearman   reference with the highest correlation; "share" is the positive part of the
              correlations normalised to sum to one (a ranking, not a mixture estimate)
All inputs are unit root-mean-square vectors on the informative genes, as in stage 06.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from lib.attr_model import fit_nnls, unit_rms
from lib.attr_panel import informative

METHODS = ["nnls", "nn_ridge", "nn_enet", "pearson", "spearman"]


def _ridge_nnls(y, R, lam):
    k = R.shape[1]
    A = np.vstack([R, np.sqrt(lam * len(y)) * np.eye(k)])
    b, _ = nnls(A, np.r_[y, np.zeros(k)], maxiter=50 * k)
    return b


def nn_ridge(y, R, lams, seed, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    err = np.zeros(len(lams))
    for tr, te in kf.split(y):
        for i, lam in enumerate(lams):
            b = _ridge_nnls(y[tr], R[tr], lam)
            err[i] += np.sum((y[te] - R[te] @ b) ** 2)
    return _ridge_nnls(y, R, lams[int(np.argmin(err))])


def nn_enet(y, R, l1_ratios, seed, n_folds=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = ElasticNetCV(l1_ratio=l1_ratios, positive=True, fit_intercept=False, alphas=30,
                         cv=KFold(n_folds, shuffle=True, random_state=seed), max_iter=5000)
        m.fit(R, y)
    return m.coef_


def correlations(y, R, kind):
    f = (lambda a, b: stats.pearsonr(a, b)[0]) if kind == "pearson" else \
        (lambda a, b: stats.spearmanr(a, b)[0])
    return np.array([f(y, R[:, j]) for j in range(R.shape[1])])


def weights(method, y, R, mcfg, seed) -> np.ndarray:
    """Non-negative weights normalised to shares (all zero -> zeros)."""
    if method == "nnls":
        b, _ = fit_nnls(y, R)
    elif method == "nn_ridge":
        b = nn_ridge(y, R, mcfg["ridge_lambdas"], seed)
    elif method == "nn_enet":
        b = nn_enet(y, R, mcfg["enet_l1_ratios"], seed)
    else:
        b = np.clip(correlations(y, R, method), 0, None)
    s = b.sum()
    return b / s if s > 0 else b


def heldout_methods(rs, cfg06, mcfg, seed) -> pd.DataFrame:
    rows = []
    for tgt, spec in cfg06["controls"]["heldout"].items():
        panel = cfg06["panels"][spec["panel"]]
        g = informative(rs, panel, rs.vec[tgt])
        y = unit_rms(rs.vec[tgt].reindex(g).values.astype(float))
        R = unit_rms(rs.matrix(panel, g).values.astype(float))
        exp = set(spec["expect"])
        for m in METHODS:
            w = weights(m, y, R, mcfg, seed)
            order = np.argsort(-w)
            top = panel[order[0]] if w.sum() > 0 else "none"
            rows.append({"target": tgt, "panel": spec["panel"], "method": m,
                         "expected": ";".join(spec["expect"]), "top_reference": top,
                         "top_in_expected": top in exp,
                         "share_on_expected": float(sum(w[panel.index(e)] for e in exp if e in panel)),
                         "n_genes": len(g),
                         "loadings": ";".join(f"{panel[j]}={w[j]:.2f}" for j in order if w[j] > 0.02)})
    return pd.DataFrame(rows)


def heldout_summary(h: pd.DataFrame) -> pd.DataFrame:
    return h.groupby("method", as_index=False).agg(
        n_controls=("target", "size"), top_in_expected=("top_in_expected", "mean"),
        mean_share_on_expected=("share_on_expected", "mean"),
        pass_rate=("share_on_expected", lambda s: float(np.mean(
            (s >= 0.5) & h.loc[s.index, "top_in_expected"]))))


def mixture_methods(sim, settings, f_grid, n_draws, panel, mcfg, rng, seed) -> pd.DataFrame:
    """Per draw: dsHMGB1 share for each method; for correlation methods the statistic is
    r(dsHMGB1) - r(LPS_24h) as well, because a ranking has no mixture weight."""
    rows = []
    for src in sim.sources:
        for sname, spec in settings.items():
            for f, ks, Y in sim.simulate(src, spec, f_grid, n_draws, rng):
                for i, (k, yv) in enumerate(zip(ks, Y)):
                    pos, names, R = sim.panel_refs(panel, src, k)
                    y = unit_rms(yv[pos])
                    jd, jl = names.index("dsHMGB1"), names.index("LPS_24h")
                    for m in METHODS:
                        if m in ("pearson", "spearman"):
                            r = correlations(y, R, m)
                            stat, top = r[jd] - r[jl], names[int(np.argmax(r))]
                        else:
                            w = weights(m, y, R, mcfg, seed + i)
                            stat, top = w[jd], names[int(np.argmax(w))] if w.sum() > 0 else "none"
                        rows.append({"source": src, "noise": sname, "f": f, "draw": i, "method": m,
                                     "value": stat, "top_reference": top})
    return pd.DataFrame(rows)


def mixture_method_summary(d: pd.DataFrame, q=0.95) -> pd.DataFrame:
    null = d[(d.f == 0) & (d.source != "within_GSE200210")]
    thr = null.groupby(["noise", "method"]).value.quantile(q).rename("thr_pooled").reset_index()
    x = d.merge(thr, on=["noise", "method"])
    x["det"] = x.value > x.thr_pooled
    x["top_lps_like"] = x.top_reference.isin(["LPS_24h", "KLA_6h"])
    x["top_hmgb1"] = x.top_reference.eq("dsHMGB1")
    return x.groupby(["method", "noise", "source", "f"], as_index=False).agg(
        n_draws=("value", "size"), mean_statistic=("value", "mean"),
        thr_pooled=("thr_pooled", "first"), power_pooled=("det", "mean"),
        top_is_lps_like=("top_lps_like", "mean"), top_is_dsHMGB1=("top_hmgb1", "mean"))
