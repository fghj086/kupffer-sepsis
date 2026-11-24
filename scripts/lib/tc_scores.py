"""Gene-set scoring and mouse-level contrasts for stage 04 (GSE224127 liver).

Each count file is one sequencing batch and is analysed on its own. Within a file:
log2 normalised CPM (median-of-ratios size factors); a gene-set score for a mouse is the mean, over set genes expressed in that
file, of the gene's log2 expression minus its mean in the file's reference group. A score
is therefore an average log2 fold change versus the reference and is comparable in units
between files (no counts are merged across files).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def size_factors(X: pd.DataFrame) -> pd.Series:
    """Median-of-ratios size factors (DESeq2 style) over genes detected in every library.
    Unlike total-count CPM, this is not shifted when a few very abundant acute-phase
    transcripts take a larger share of the library after infection."""
    pos = X[(X > 0).all(axis=1)]
    lg = np.log(pos)
    sf = np.exp(lg.sub(lg.mean(axis=1), axis=0).median(axis=0))
    return sf / np.exp(np.log(sf).mean())


def log_cpm(X: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """log2(normalised counts per million + pseudocount); normalisation by median-of-ratios
    size factors, scaled to the file's mean library size."""
    norm = X.div(size_factors(X), axis=1)
    return np.log2(norm / X.sum(axis=0).mean() * 1e6 + pseudocount)


def expressed_genes(X: pd.DataFrame, genes, min_mean_cpm: float) -> list[str]:
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    present = [g for g in genes if g in cpm.index]
    return [g for g in present if cpm.loc[g].mean() >= min_mean_cpm]


def score_sets(L: pd.DataFrame, ref_cols, sets: dict[str, list[str]],
               weights: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-sample scores (columns = sets). Returns scores and genes used per set."""
    centred = L.sub(L[ref_cols].mean(axis=1), axis=0)
    out, used = {}, {}
    for name, genes in sets.items():
        g = [x for x in genes if x in centred.index]
        used[name] = g
        if not g:
            out[name] = pd.Series(np.nan, index=L.columns)
            continue
        if weights and name in weights:
            w = weights[name].reindex(g).astype(float)
            pos, neg = w[w > 0], w[w < 0]
            if len(pos) and len(neg):
                # signed axis: the + and - halves are averaged separately, as in stage 01
                out[name] = (centred.loc[pos.index].mul(pos, axis=0).sum(0) / pos.sum()
                             - centred.loc[neg.index].mul(-neg, axis=0).sum(0) / (-neg.sum()))
            else:
                out[name] = (centred.loc[g].mul(w, axis=0).sum(axis=0) / w.abs().sum())
        else:
            out[name] = centred.loc[g].mean(axis=0)
    return pd.DataFrame(out), used


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Hedges' g (a minus b) with an approximate 95% CI (normal approximation)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (na + nb) - 9)
    g = d * j
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g, g - 1.96 * se, g + 1.96 * se


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean difference (a - b) with Welch 95% CI and two-sided P; mice are the unit."""
    na, nb = len(a), len(b)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if na < 2 or nb < 2 or se == 0:
        return {"diff": diff, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    t = diff / se
    q = stats.t.ppf(0.975, df)
    return {"diff": diff, "ci_low": diff - q * se, "ci_high": diff + q * se,
            "p": 2 * stats.t.sf(abs(t), df)}


def contrast_table(S: pd.DataFrame, design: pd.DataFrame, comparisons) -> pd.DataFrame:
    """comparisons: iterable of dicts with keys label fields + 'test_cols', 'ref_cols'."""
    rows = []
    for comp in comparisons:
        meta = {k: v for k, v in comp.items() if k not in ("test_cols", "ref_cols")}
        for score in S.columns:
            a = S.loc[comp["test_cols"], score].dropna().to_numpy()
            b = S.loc[comp["ref_cols"], score].dropna().to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue
            w = welch(a, b)
            g, gl, gh = hedges_g(a, b)
            rows.append({**meta, "score": score, "n_test": len(a), "n_ref": len(b),
                         "mean_test": a.mean(), "mean_ref": b.mean(),
                         "diff_log2": w["diff"], "diff_ci_low": w["ci_low"],
                         "diff_ci_high": w["ci_high"], "hedges_g": g,
                         "g_ci_low": gl, "g_ci_high": gh, "p_welch": w["p"]})
    return pd.DataFrame(rows)


def add_fdr(df: pd.DataFrame, by=None, col: str = "p_welch") -> pd.DataFrame:
    df = df.copy()
    df["fdr"] = np.nan
    groups = [df.index] if by is None else [ix for _, ix in df.groupby(by).groups.items()]
    for ix in groups:
        sub = df.loc[ix, col]
        ok = sub.notna()
        if ok.any():
            df.loc[sub[ok].index, "fdr"] = multipletests(sub[ok], method="fdr_bh")[1]
    return df


def bootstrap_peak(S: pd.DataFrame, score: str, comps: list[dict], sign: int,
                   rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Fraction of mouse-level bootstrap resamples in which each comparison (time point)
    shows the largest signed difference from its own reference group."""
    draws = []
    for c in comps:
        a = S.loc[c["test_cols"], score].to_numpy()
        b = S.loc[c["ref_cols"], score].to_numpy()
        ia = rng.integers(0, len(a), (n_boot, len(a)))
        ib = rng.integers(0, len(b), (n_boot, len(b)))
        draws.append(sign * (a[ia].mean(1) - b[ib].mean(1)))
    win = np.argmax(np.vstack(draws), axis=0)
    return np.bincount(win, minlength=len(comps)) / n_boot
