"""Stage 07 statistics: subject-level effect sizes with intervals, exact small-sample tests,
Firth-penalised logistic regression and REML random-effects pooling."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats


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
    res[o] = np.minimum(r, 1)
    out[ok] = res
    return out


def _exact_perm_p(x1, x0, max_enum=200000, n_rand=100000, seed=0):
    """Two-sided permutation p for the difference in means (exact when enumerable)."""
    x = np.concatenate([x1, x0])
    n1 = len(x1)
    obs = abs(x1.mean() - x0.mean())
    ncomb = math.comb(len(x), n1)
    if ncomb <= max_enum:
        ge, tot = 0, 0
        s = x.sum()
        for idx in itertools.combinations(range(len(x)), n1):
            a = x[list(idx)].sum()
            d = abs(a / n1 - (s - a) / (len(x) - n1))
            ge += d >= obs - 1e-12
            tot += 1
        return ge / tot, ncomb
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_rand):
        p = rng.permutation(x)
        ge += abs(p[:n1].mean() - p[n1:].mean()) >= obs - 1e-12
    return (ge + 1) / (n_rand + 1), ncomb


def hedges_g(x1, x0):
    n1, n0 = len(x1), len(x0)
    sp = math.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n0 - 1) * np.var(x0, ddof=1)) / (n1 + n0 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (x1.mean() - x0.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n0) - 9)
    g = j * d
    se = math.sqrt((n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0)))
    return g, g - 1.96 * se, g + 1.96 * se


def two_group(x1, x0, level=0.95) -> dict:
    x1 = np.asarray(pd.Series(x1).dropna(), float)
    x0 = np.asarray(pd.Series(x0).dropna(), float)
    out = {"n1": len(x1), "n0": len(x0), "mean1": np.nan, "mean0": np.nan, "estimate": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "hedges_g": np.nan, "g_ci_low": np.nan,
           "g_ci_high": np.nan, "p": np.nan, "test": ""}
    if len(x1) < 2 or len(x0) < 2:
        return out
    d = x1.mean() - x0.mean()
    v1, v0 = np.var(x1, ddof=1) / len(x1), np.var(x0, ddof=1) / len(x0)
    se = math.sqrt(v1 + v0)
    df = (v1 + v0) ** 2 / (v1 ** 2 / (len(x1) - 1) + v0 ** 2 / (len(x0) - 1)) if se > 0 else np.nan
    t = stats.t.ppf(0.5 + level / 2, df) if se > 0 else np.nan
    g, gl, gh = hedges_g(x1, x0)
    small = math.comb(len(x1) + len(x0), len(x1)) <= 200000
    if small:
        p, _ = _exact_perm_p(x1, x0)
        test = "exact permutation (difference in means)"
    else:
        p = stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue
        test = "Mann-Whitney"
    out.update({"mean1": x1.mean(), "mean0": x0.mean(), "estimate": d,
                "ci_low": d - t * se, "ci_high": d + t * se, "hedges_g": g,
                "g_ci_low": gl, "g_ci_high": gh, "p": p, "test": test})
    return out


def paired(d, level=0.95) -> dict:
    """Within-subject differences: mean, t interval, exact sign-flip p (n <= 20) or Wilcoxon."""
    d = np.asarray(pd.Series(d).dropna(), float)
    n = len(d)
    out = {"n1": n, "n0": n, "mean1": np.nan, "mean0": np.nan, "estimate": np.nan, "ci_low": np.nan,
           "ci_high": np.nan, "hedges_g": np.nan, "g_ci_low": np.nan, "g_ci_high": np.nan,
           "p": np.nan, "test": ""}
    if n < 2:
        return out
    m, s = d.mean(), d.std(ddof=1)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    dz = m / s if s > 0 else np.nan
    j = 1 - 3 / (4 * (n - 1) - 1) if n > 2 else 1.0
    gz = j * dz
    se_g = math.sqrt(1 / n + gz ** 2 / (2 * n)) if s > 0 else np.nan
    if n <= 20:
        obs = abs(m)
        ge = 0
        for signs in itertools.product([1, -1], repeat=n):
            ge += abs((d * np.array(signs)).mean()) >= obs - 1e-12
        p, test = ge / 2 ** n, "exact sign-flip (paired)"
    else:
        p, test = stats.wilcoxon(d).pvalue, "Wilcoxon signed-rank"
    out.update({"estimate": m, "ci_low": m - t * s / math.sqrt(n), "ci_high": m + t * s / math.sqrt(n),
                "hedges_g": gz, "g_ci_low": gz - 1.96 * se_g, "g_ci_high": gz + 1.96 * se_g,
                "p": p, "test": test})
    return out


# ------------------------------------------------------------------ Firth logistic regression
def _firth_fit(X, y, fixed=None, max_iter=200, tol=1e-9):
    """Penalised ML (Jeffreys prior). fixed: {column index: value} held constant."""
    n, k = X.shape
    b = np.zeros(k)
    if fixed:
        for j, v in fixed.items():
            b[j] = v
    free = [j for j in range(k) if not fixed or j not in fixed]

    def pll(beta):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        return np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet

    for _ in range(max_iter):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        u = X.T @ (y - p + h * (0.5 - p))
        Iff = info[np.ix_(free, free)]
        step = np.linalg.solve(Iff, u[free])
        mx = np.max(np.abs(step))
        if mx > 5:
            step *= 5 / mx
        old = pll(b)
        nb = b.copy()
        nb[free] += step
        k_half = 0
        while pll(nb) < old - 1e-12 and k_half < 30:
            step /= 2
            nb = b.copy()
            nb[free] += step
            k_half += 1
        b = nb
        if np.max(np.abs(step)) < tol:
            break
    return b, pll(b)


def firth_logistic(x, y, level=0.95, covars=None) -> dict:
    """Firth logistic regression of y on one standardised predictor (plus optional covariate
    columns); penalised likelihood-ratio test and profile penalised-likelihood interval for the
    predictor's coefficient (as in logistf)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cols = [np.ones_like(x), x]
    if covars is not None:
        cv = np.asarray(covars, float)
        cols += [cv] if cv.ndim == 1 else [cv[:, j] for j in range(cv.shape[1])]
    X = np.column_stack(cols)
    b, ll = _firth_fit(X, y)
    _, ll0 = _firth_fit(X, y, fixed={1: 0.0})
    lr = 2 * (ll - ll0)
    p = stats.chi2.sf(max(lr, 0), 1)
    crit = stats.chi2.ppf(level, 1) / 2

    def f(v):
        return ll - _firth_fit(X, y, fixed={1: v})[1] - crit
    lo = hi = np.nan
    try:
        lo = optimize.brentq(f, b[1] - 20, b[1])
    except ValueError:
        pass
    try:
        hi = optimize.brentq(f, b[1], b[1] + 20)
    except ValueError:
        pass
    # standard error implied by the profile interval (for pooling)
    se = (hi - lo) / (2 * stats.norm.ppf(0.5 + level / 2)) if np.isfinite(lo) and np.isfinite(hi) else np.nan
    return {"log_or": b[1], "ci_low": lo, "ci_high": hi, "se": se, "p": p,
            "n": int(len(y)), "events": int(y.sum())}


def reml_pool(est, se, level=0.95) -> dict:
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, v = est[ok], se[ok] ** 2
    k = len(est)
    if k == 0:
        return {"k": 0}
    tau2 = 0.0
    for _ in range(500):
        w = 1 / (v + tau2)
        mu = np.sum(w * est) / np.sum(w)
        new = np.sum(w ** 2 * ((est - mu) ** 2 - v)) / np.sum(w ** 2) + 1 / np.sum(w)
        new = max(new, 0.0)
        if abs(new - tau2) < 1e-10:
            tau2 = new
            break
        tau2 = new
    w = 1 / (v + tau2)
    mu = np.sum(w * est) / np.sum(w)
    se_mu = math.sqrt(1 / np.sum(w))
    z = stats.norm.ppf(0.5 + level / 2)
    q = np.sum((est - np.sum(est / v) / np.sum(1 / v)) ** 2 / v)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    out = {"k": k, "estimate": mu, "ci_low": mu - z * se_mu, "ci_high": mu + z * se_mu,
           "p": 2 * stats.norm.sf(abs(mu / se_mu)), "tau2": tau2, "I2": i2,
           "n_same_sign": int(max((est > 0).sum(), (est < 0).sum())),
           "sign_of_pooled_matches": int(((est > 0) == (mu > 0)).sum())}
    if k > 1:   # Hartung-Knapp: weighted residual variance, t with k - 1 df
        qhk = np.sum(w * (est - mu) ** 2) / (k - 1)
        se_hk = math.sqrt(qhk / np.sum(w))
        t = stats.t.ppf(0.5 + level / 2, k - 1)
        se_mhk = math.sqrt(max(qhk, 1.0) / np.sum(w))   # modified HK: never narrower than REML
        out.update({"hk_ci_low": mu - t * se_hk, "hk_ci_high": mu + t * se_hk,
                    "hk_p": 2 * stats.t.sf(abs(mu / se_hk), k - 1) if se_hk > 0 else np.nan,
                    "mhk_ci_low": mu - t * se_mhk, "mhk_ci_high": mu + t * se_mhk,
                    "mhk_p": 2 * stats.t.sf(abs(mu / se_mhk), k - 1)})
    return out
