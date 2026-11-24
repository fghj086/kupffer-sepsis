"""Program scores, pseudobulk, shift-share decomposition, ligand-receptor products and
stimulus-signature projection for one annotated scRNA-seq dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def pseudobulk(ad, keys: list[str], layer: str = "counts", min_cells: int = 20):
    """Sum raw counts per group; returns (logCPM genes x groups, group metadata)."""
    X = ad.layers[layer]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    g = ad.obs[keys].astype(str).agg("|".join, axis=1)
    codes, uniq = pd.factorize(g)
    ind = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                        shape=(len(uniq), len(codes)))
    S = np.asarray((ind @ X).todense())
    n = np.bincount(codes)
    meta = pd.DataFrame([u.split("|") for u in uniq], columns=keys)
    meta["n_cells"] = n
    meta["pb_id"] = uniq
    keep = n >= min_cells
    S, meta = S[keep], meta[keep].reset_index(drop=True)
    lib = S.sum(1, keepdims=True)
    cpm = np.log2(S / lib * 1e6 + 1)
    return pd.DataFrame(cpm.T, index=ad.var_names, columns=meta["pb_id"]), meta


def pb_program_scores(logcpm: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """Mean of gene-centred logCPM per set (genes centred across all pseudobulks given),
    so scores are comparable across populations and conditions within one dataset."""
    expressed = logcpm.index[(logcpm > 1).mean(1) >= 0.25]
    c = logcpm.loc[expressed].sub(logcpm.loc[expressed].mean(1), axis=0)
    out = {}
    for k, genes in sets.items():
        g = [x for x in genes if x in c.index]
        out[k] = c.loc[g].mean(0) if len(g) >= 2 else pd.Series(np.nan, index=c.columns)
        out[f"{k}__n_genes"] = pd.Series(len(g), index=c.columns)
    return pd.DataFrame(out)


def shift_share(w0: pd.Series, w1: pd.Series, s0: pd.Series, s1: pd.Series) -> dict:
    """Exact two-factor decomposition of the change in a weighted mean,
    total = sum(w1*s1) - sum(w0*s0) = composition + within, with midpoint weights:
    composition = sum((w1-w0)*(s0+s1)/2), within = sum((w0+w1)/2*(s1-s0))."""
    idx = w0.index.union(w1.index)
    w0, w1 = w0.reindex(idx, fill_value=0), w1.reindex(idx, fill_value=0)
    s0 = s0.reindex(idx)
    s1 = s1.reindex(idx)
    s0, s1 = s0.fillna(s1), s1.fillna(s0)  # population absent in one condition
    comp = ((w1 - w0) * (s0 + s1) / 2)
    within = ((w0 + w1) / 2 * (s1 - s0))
    total = (w1 * s1).sum() - (w0 * s0).sum()
    return {"total": total, "composition": comp.sum(), "within": within.sum(),
            "per_pop_composition": comp, "per_pop_within": within}


def ulm_scores(ad, net: pd.DataFrame, layer=None) -> pd.DataFrame:
    """Univariate linear model per cell (as decoupler 'ulm'): regress the cell's
    log-normalised expression over all genes on the signature weight vector
    (0 for genes outside the signature); the score is the slope t-value."""
    X = ad.X if layer is None else ad.layers[layer]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    G = X.shape[1]
    sq = np.asarray(X.multiply(X).sum(1)).ravel()
    sm = np.asarray(X.sum(1)).ravel()
    syy = sq - sm ** 2 / G
    res, ngenes = {}, {}
    for sig, d in net.groupby("signature"):
        w = d.groupby("gene")["weight"].sum()
        w = w[w.index.isin(ad.var_names)]
        ngenes[sig] = len(w)
        x = pd.Series(0.0, index=ad.var_names)
        x.loc[w.index] = w.values
        xc = x.values - x.values.mean()
        sxx = (xc ** 2).sum()
        sxy = X @ xc
        slope = sxy / sxx
        rss = np.clip(syy - slope ** 2 * sxx, 1e-12, None)
        se = np.sqrt(rss / (G - 2) / sxx)
        res[sig] = slope / se
    df = pd.DataFrame(res, index=ad.obs_names)
    df.attrs["n_genes"] = ngenes
    return df


def mean_expr(ad, genes: list[str], groupby: list[str]) -> pd.DataFrame:
    """Mean log-normalised expression and detection fraction per group for each gene."""
    g = [x for x in genes if x in ad.var_names]
    X = ad[:, g].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, columns=g, index=ad.obs_names)
    keys = ad.obs[groupby].astype(str)
    m = df.groupby([keys[k] for k in groupby]).mean()
    f = (df > 0).groupby([keys[k] for k in groupby]).mean()
    n = df.groupby([keys[k] for k in groupby]).size().rename("n_cells")
    out = m.stack().rename("mean_lognorm").to_frame().join(f.stack().rename("frac_detected"))
    out.index.names = groupby + ["gene"]
    return out.reset_index().merge(n.reset_index(), on=groupby)


def lr_products(ad, ligand: str, receptors: list[str], cond_key: str, src_key: str,
                tgt_key: str, unit_key: str | None = None) -> pd.DataFrame:
    """Expression-product score: mean ligand in source x mean receptor in target,
    per condition (and per sample when unit_key is given)."""
    by = [cond_key] + ([unit_key] if unit_key else [])
    L = mean_expr(ad, [ligand], by + [src_key])
    R = mean_expr(ad, receptors, by + [tgt_key])
    L = L.rename(columns={src_key: "source", "mean_lognorm": "ligand_mean",
                          "frac_detected": "ligand_frac", "n_cells": "n_source"}).drop(columns="gene")
    R = R.rename(columns={tgt_key: "target", "gene": "receptor", "mean_lognorm": "receptor_mean",
                          "frac_detected": "receptor_frac", "n_cells": "n_target"})
    out = L.merge(R, on=by)
    out["ligand"] = ligand
    out["score"] = out["ligand_mean"] * out["receptor_mean"]
    return out


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp_ = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp_ if sp_ > 0 else np.nan
