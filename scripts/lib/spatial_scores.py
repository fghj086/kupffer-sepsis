"""Stage 04 spatial helpers (GSE248904, whole-mouse Array-seq, LPS 24 h vs control, 2+2).

Exploratory: animals are the unit and only effect sizes are reported.
Spot scores are means of per-gene z-scores of log-normalised expression, with z computed
over the liver spots of all four sections together, so that LPS and control share one scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def log_norm(X, lib: np.ndarray, target: float = 1e4):
    """log1p(count / library size * target); library size = all-gene UMI total per spot."""
    X = sp.csr_matrix(X, dtype=np.float32)
    lib = np.where(np.asarray(lib, float) > 0, lib, 1.0)
    Xn = sp.diags((target / lib).astype(np.float32)) @ X
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def ring_neighbours(coords: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    """Neighbours ranked k_from..k_to by distance: a non-adjacent 'ring' used as a
    reference for how much of an association is local rather than regional."""
    nn = NearestNeighbors(n_neighbors=k_to + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, k_from:k_to + 1]


def gene_matrix(Xn, var_names, genes) -> pd.DataFrame:
    idx = {g: i for i, g in enumerate(var_names)}
    present = [g for g in genes if g in idx]
    if not present:
        return pd.DataFrame(index=range(Xn.shape[0]))
    M = Xn[:, [idx[g] for g in present]]
    return pd.DataFrame(M.toarray(), columns=present)


def zscore_cols(M: pd.DataFrame) -> pd.DataFrame:
    sd = M.std(axis=0, ddof=0).replace(0, np.nan)
    return (M - M.mean(axis=0)) / sd


def set_score(Xn, var_names, genes) -> tuple[np.ndarray, list[str]]:
    M = gene_matrix(Xn, var_names, genes)
    if M.shape[1] == 0:
        return np.full(Xn.shape[0], np.nan), []
    Z = zscore_cols(M).dropna(axis=1, how="all")
    return Z.mean(axis=1).to_numpy(), list(Z.columns)


def neighbours(coords: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]


def spatial_lag(x: np.ndarray, nb: np.ndarray) -> np.ndarray:
    return np.nanmean(x[nb], axis=1)


def morans_i(x: np.ndarray, nb: np.ndarray) -> float:
    """Moran's I with row-standardised k-nearest-neighbour weights."""
    z = x - np.nanmean(x)
    lag = z[nb].mean(axis=1)
    return float(np.nansum(z * lag) / np.nansum(z * z))


def std_ols(y: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Standardised OLS coefficients (all variables z-scored) with 95% CI. Spots are not
    independent, so the CI is descriptive only; inference is made across animals."""
    import statsmodels.api as sm
    df = pd.concat([pd.Series(y, name="y"), X.reset_index(drop=True)], axis=1).dropna()
    df = (df - df.mean()) / df.std(ddof=0)
    fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit()
    ci = fit.conf_int()
    return pd.DataFrame({"beta": fit.params, "ci_low": ci[0], "ci_high": ci[1]}).drop("const")


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp_ = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp_ * (1 - 3 / (4 * (na + nb) - 9)) if sp_ > 0 else np.nan


def bin_counts(X, coords: np.ndarray, lib: np.ndarray, size: float, min_spots: int):
    """Sum counts of spots in square tiles of side `size` (coordinate units). Returns the
    tile count matrix, tile library sizes, tile centres, spots per tile and spot->tile map."""
    ij = np.floor(coords / size).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    uniq, inv, n = np.unique(key, return_inverse=True, return_counts=True)
    A = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))),
                      shape=(len(uniq), len(inv)))
    Xb = (A @ sp.csr_matrix(X, dtype=np.float32)).tocsr()
    libb = A @ np.asarray(lib, np.float64)
    cen = (A @ coords) / n[:, None]
    ok = n >= min_spots
    return Xb[ok], libb[ok], cen[ok], n[ok], np.where(ok[inv], inv, -1), ok
