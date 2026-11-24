"""Stage 06: within-study reference vectors, responsive genes and replicate-level vectors.

A reference vector is the difference of arm means of log2(normalised level + prior) between a
stimulus arm and the study's own unstimulated arm. Responsive genes are called within the same
study: pydeseq2 (Wald test, replicate as unit) for raw counts, a moderated t on log2 values for
deposited normalised tables.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("06_attribution")


def median_ratio_sf(C: pd.DataFrame) -> pd.Series:
    C = C.astype(float)
    pos = (C > 0).all(axis=1)
    lg = np.log(C[pos])
    ref = lg.mean(axis=1)
    sf = np.exp((lg.sub(ref, axis=0)).median(axis=0))
    return sf / np.exp(np.log(sf).mean())


def to_level(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Linear expression level on a CPM-like scale (FPKM kept as deposited)."""
    if kind == "log2":
        return (2.0 ** X - 1.0).clip(lower=0)
    if kind == "counts":
        N = X / median_ratio_sf(X)
        return N / (N.sum(axis=0).mean() / 1e6)
    # normalised linear values: TMM/DESeq2-normalised counts are rescaled to CPM,
    # FPKM-like tables (column sums far below 1e6) are kept
    tot = X.sum(axis=0).mean()
    return X / (tot / 1e6) if tot > 2e6 else X.astype(float)


def log_level(level: pd.DataFrame, prior: float) -> pd.DataFrame:
    return np.log2(level + prior)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    o = np.argsort(q)
    r = q[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.minimum(r, 1)
    out[ok] = res
    return out


def moderated_t(L: pd.DataFrame, a: list, b: list) -> pd.DataFrame:
    """Two-group t on log2 values with a simple empirical-Bayes variance prior (limma-like)."""
    xa, xb = L[a].values, L[b].values
    na, nb = xa.shape[1], xb.shape[1]
    d = na + nb - 2
    s2 = (((xa - xa.mean(1, keepdims=True)) ** 2).sum(1) +
          ((xb - xb.mean(1, keepdims=True)) ** 2).sum(1)) / max(d, 1)
    s2 = np.maximum(s2, 1e-8)
    d0, s0 = 4.0, np.median(s2)
    s2p = (d0 * s0 + d * s2) / (d0 + d)
    diff = xa.mean(1) - xb.mean(1)
    t = diff / np.sqrt(s2p * (1 / na + 1 / nb))
    p = 2 * stats.t.sf(np.abs(t), d + d0)
    return pd.DataFrame({"lfc": diff, "stat": t, "pvalue": p, "padj": _bh(p)}, index=L.index)


def deseq_test(C: pd.DataFrame, arms: pd.Series, test: str, ctrl: str, n_cpus: int) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    cols = list(arms[arms.isin([test, ctrl])].index)
    meta = pd.DataFrame({"arm": arms[cols].map({test: "B_test", ctrl: "A_ctrl"})}, index=cols)
    counts = C[cols].T.round().astype(int)
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    inf = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~arm", inference=inf, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["arm", "B_test", "A_ctrl"], inference=inf, quiet=True)
        st.summary()
    r = st.results_df
    return pd.DataFrame({"lfc": r["log2FoldChange"], "stat": r["stat"], "pvalue": r["pvalue"],
                         "padj": r["padj"]}, index=r.index)


class Study:
    """Expression of one study with arm labels; builds references within the study."""

    def __init__(self, name, X, arms, kind, cfg):
        self.name, self.kind, self.arms = name, kind, arms
        self.raw = X
        self.level = to_level(X, kind)
        self.L = log_level(self.level, cfg["expression"]["prior_cpm"])
        self.cfg = cfg

    def cols(self, arm):
        return list(self.arms[self.arms == arm].index)

    def expressed(self, test, ctrl):
        lv = self.level
        m = np.maximum(lv[self.cols(test)].mean(1), lv[self.cols(ctrl)].mean(1))
        return m >= self.cfg["expression"]["min_level"]

    def reference(self, test, ctrl, drop=()):
        t = [c for c in self.cols(test) if c not in drop]
        c = [x for x in self.cols(ctrl) if x not in drop]
        return self.L[t].mean(1) - self.L[c].mean(1)

    def de(self, test, ctrl):
        if self.kind == "counts":
            r = deseq_test(self.raw, self.arms, test, ctrl, self.cfg["responsive"]["n_cpus"])
        else:
            r = moderated_t(self.L, self.cols(test), self.cols(ctrl))
        return r.reindex(self.L.index)

    def heldout(self, test, ctrl):
        """Replicate-level vectors: test replicate k against the control mean without control k,
        and the reference rebuilt from the remaining replicates (both arms)."""
        tc, cc = self.cols(test), self.cols(ctrl)
        out = []
        for k, t in enumerate(tc):
            ck = cc[k % len(cc)] if len(cc) > 1 else None
            c_rest = [c for c in cc if c != ck] if ck else cc
            v = self.L[t] - self.L[c_rest].mean(1)
            drop = (t,) + ((ck,) if ck else ())
            out.append((t, v, self.reference(test, ctrl, drop=drop), drop))
        return out


def responsive_genes(de: pd.DataFrame, expressed: pd.Series, cfg) -> pd.Index:
    r = cfg["responsive"]
    d = de[expressed.reindex(de.index).fillna(False).values]
    hit = d[(d.padj < r["padj"]) & (d.lfc.abs() >= r["min_abs_lfc"])]
    hit = hit.reindex(hit.stat.abs().sort_values(ascending=False).index)
    return hit.index[: r["top_n_per_reference"]]
