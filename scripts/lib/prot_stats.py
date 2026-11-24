"""Statistics for stage 05 (protein layer).

Moderated two-group t statistics follow limma's empirical Bayes variance shrinkage
(Smyth 2004, Stat Appl Genet Mol Biol 3:3): residual variances are shrunk towards a common
prior whose scale and degrees of freedom are estimated from all proteins.
Proteins may have different residual degrees of freedom because missing values are not imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats


# ------------------------------------------------------------------ helpers
def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    q = p[ok]
    o = np.argsort(q)
    r = q[o] * len(q) / np.arange(1, len(q) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    res = np.empty_like(q)
    res[o] = np.minimum(r, 1.0)
    out[ok] = res
    return out


def _trigamma_inverse(x: float) -> float:
    """Newton iteration used by limma::trigammaInverse."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_f_dist(s2: np.ndarray, df: np.ndarray) -> tuple[float, float]:
    """Estimate prior (s0^2, d0) of residual variances (limma::fitFDist, no covariate)."""
    ok = np.isfinite(s2) & (s2 > 0) & (df > 0)
    s2, df = s2[ok], df[ok]
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = np.sum((e - emean) ** 2) / (len(e) - 1) - np.mean(special.polygamma(1, df / 2))
    if evar > 0:
        d0 = 2 * _trigamma_inverse(evar)
        s02 = float(np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2)))
    else:
        d0 = np.inf
        s02 = float(np.exp(emean))
    return s02, float(d0)


def moderated_t(x: pd.DataFrame, groups: pd.Series, test: str, ref: str,
                min_valid: int) -> pd.DataFrame:
    """Two-group moderated t on log2 abundance (rows = proteins, columns = samples)."""
    a = x.loc[:, groups.index[groups == test]].to_numpy(float)
    b = x.loc[:, groups.index[groups == ref]].to_numpy(float)
    na, nb = (~np.isnan(a)).sum(1), (~np.isnan(b)).sum(1)
    keep = (na >= min_valid) & (nb >= min_valid)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        ma, mb = np.nanmean(a, 1), np.nanmean(b, 1)
        ssa = np.nansum((a - ma[:, None]) ** 2, 1)
        ssb = np.nansum((b - mb[:, None]) ** 2, 1)
    df = (na + nb - 2).astype(float)
    with np.errstate(all="ignore"):
        s2 = (ssa + ssb) / df
    s02, d0 = fit_f_dist(s2[keep], df[keep])
    np.seterr(all="ignore")
    if np.isinf(d0):
        s2post = np.full_like(s2, s02)
        dft = np.full_like(df, 1e6)
    else:
        s2post = (d0 * s02 + df * s2) / (d0 + df)
        dft = df + d0
    lfc = ma - mb
    se = np.sqrt(s2post * (1 / na + 1 / nb))
    t = lfc / se
    p = 2 * stats.t.sf(np.abs(t), dft)
    tcrit = stats.t.ppf(0.975, dft)
    out = pd.DataFrame({
        "n_test": na, "n_ref": nb, "mean_test": ma, "mean_ref": mb, "log2FC": lfc,
        "ci_low": lfc - tcrit * se, "ci_high": lfc + tcrit * se,
        "s2": s2, "s2_post": s2post, "t_mod": t, "df_total": dft, "pvalue": p,
    }, index=x.index)
    out.loc[~keep, ["log2FC", "ci_low", "ci_high", "t_mod", "pvalue", "s2_post"]] = np.nan
    np.seterr(all="warn")
    out["tested"] = keep
    out["padj"] = bh(out["pvalue"].to_numpy())
    out.attrs["prior_s02"], out.attrs["prior_d0"] = s02, d0
    return out


def on_off(x: pd.DataFrame, groups: pd.Series, test: str, ref: str, frac: float) -> pd.Series:
    """'on_in_test' / 'on_in_ref' when one group is detected in >= frac and the other never."""
    a = x.loc[:, groups.index[groups == test]].notna().mean(1)
    b = x.loc[:, groups.index[groups == ref]].notna().mean(1)
    lab = pd.Series("", index=x.index)
    lab[(a >= frac) & (b == 0)] = "on_in_" + test
    lab[(b >= frac) & (a == 0)] = "on_in_" + ref
    return lab


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * j)


# ------------------------------------------------------------------ set scores
def sample_set_score(x: pd.DataFrame, members: list[str]) -> tuple[pd.Series, list[str]]:
    """Mean of per-protein z scores (across the samples of one study) over detected members.

    A member counts only if it is quantified in every sample, so all samples are scored
    from the same proteins.
    """
    present = [g for g in members if g in x.index and x.loc[g].notna().all()]
    if not present:
        return pd.Series(np.nan, index=x.columns), []
    z = x.loc[present]
    z = z.sub(z.mean(1), axis=0).div(z.std(1, ddof=1).replace(0, np.nan), axis=0)
    return z.mean(0), present


def rank_set_test(t: pd.Series, members: list[str], weights: pd.Series | None = None,
                  n_perm: int = 10000, rng: np.random.Generator | None = None) -> dict:
    """Rank-based enrichment of a protein set within one moderated-t ranking.

    Returns the AUC of members versus all other tested proteins (Mann-Whitney), the mean
    (signed) moderated t of members, and a two-sided gene-label permutation P for that mean.
    """
    t = t.dropna()
    idx = [g for g in members if g in t.index]
    res = {"n_members": len(members), "n_detected": len(idx)}
    if len(idx) < 3:
        return {**res, "auc": np.nan, "mwu_p": np.nan, "mean_t": np.nan, "perm_p": np.nan}
    w = np.ones(len(idx)) if weights is None else np.sign(weights.reindex(idx).to_numpy(float))
    signed = t.loc[idx].to_numpy() * w
    others = t.drop(idx).to_numpy()
    if weights is None:
        u, p = stats.mannwhitneyu(t.loc[idx], others, alternative="two-sided")
        auc = u / (len(idx) * len(others))
    else:
        auc, p = np.nan, np.nan
    obs = signed.mean()
    rng = rng or np.random.default_rng(0)
    allt = t.to_numpy()
    perm = np.empty(n_perm)
    for i in range(n_perm):
        pick = rng.choice(len(allt), len(idx), replace=False)
        perm[i] = (allt[pick] * w).mean()
    pp = (np.sum(np.abs(perm - perm.mean()) >= abs(obs - perm.mean())) + 1) / (n_perm + 1)
    return {**res, "auc": auc, "mwu_p": p, "mean_t": obs, "perm_p": pp}


def difference_test(t: pd.Series, set_a: list[str], set_b: list[str], n_perm: int,
                    rng: np.random.Generator) -> dict:
    """Mean t of set A minus mean t of set B; P from permuting labels within A ∪ B."""
    t = t.dropna()
    a = [g for g in set_a if g in t.index and g not in set_b]
    b = [g for g in set_b if g in t.index and g not in set_a]
    if len(a) < 3 or len(b) < 3:
        return {"n_a": len(a), "n_b": len(b), "diff_mean_t": np.nan, "perm_p": np.nan}
    va, vb = t.loc[a].to_numpy(), t.loc[b].to_numpy()
    obs = va.mean() - vb.mean()
    pool = np.concatenate([va, vb])
    perm = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pool)
        perm[i] = pool[:len(a)].mean() - pool[len(a):].mean()
    pp = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return {"n_a": len(a), "n_b": len(b), "diff_mean_t": obs, "perm_p": pp}
