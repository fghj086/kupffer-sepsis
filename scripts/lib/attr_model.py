"""Stage 06: non-negative attribution of a log2FC vector onto stimulus reference vectors.

All vectors are scaled to unit root-mean-square over the informative genes, so a coefficient
expresses the contribution of a reference pattern, not of its absolute response size. Fit quality
is the uncentred R^2 (log2FC = 0 is the meaningful origin):
    R^2 = 1 - ||y - R b||^2 / ||y||^2.
The unique contribution of reference j is the drop in R^2 when j is removed and the rest refitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def unit_rms(v: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.mean(np.square(v), axis=0))
    return v / np.where(s > 0, s, 1.0)


def r2(y, R, b) -> float:
    ss = float(np.sum(y ** 2))
    return 1.0 - float(np.sum((y - R @ b) ** 2)) / ss if ss > 0 else np.nan


def fit_nnls(y: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    b, _ = nnls(R, y, maxiter=50 * R.shape[1])
    return b, r2(y, R, b)


def fit_ridge(y, R, alpha):
    """Signed ridge fit, used for differences between two perturbed arms (blockade contrasts)."""
    k = R.shape[1]
    b = np.linalg.solve(R.T @ R + alpha * len(y) * np.eye(k), R.T @ y)
    return b, r2(y, R, b)


def drop_one(y, R, b_full, r2_full, signed=False, alpha=0.1):
    out = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        keep = [i for i in range(R.shape[1]) if i != j]
        if b_full[j] == 0 and not signed:
            out[j] = 0.0
            continue
        _, r = (fit_ridge(y, R[:, keep], alpha) if signed else fit_nnls(y, R[:, keep]))
        out[j] = r2_full - r
    return out


def shared_axis(R: np.ndarray) -> np.ndarray:
    """First uncentred singular vector of the reference matrix (the response shared by all
    references), oriented so that most references load positively."""
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    a = u[:, 0] * s[0]
    if np.sum(vt[0]) < 0:
        a = -a
    return unit_rms(a)


def attribute(y: pd.Series, refs: pd.DataFrame, n_boot: int, rng: np.random.Generator,
              signed=False, alpha=0.1) -> tuple[pd.DataFrame, dict]:
    """y and refs share the gene index (informative genes, no missing values)."""
    Y = unit_rms(y.values.astype(float))
    R = unit_rms(refs.values.astype(float))
    fit = (lambda yy, RR: fit_ridge(yy, RR, alpha)) if signed else fit_nnls
    b, r2f = fit(Y, R)
    d = drop_one(Y, R, b, r2f, signed, alpha)
    a = shared_axis(R)
    bs, r2s = fit_nnls(Y, a[:, None])
    n = len(Y)
    boots = np.zeros((n_boot, R.shape[1]))
    r2b = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bb, rr = fit(Y[idx], R[idx])
        boots[i], r2b[i] = bb, rr
    tot = np.abs(b).sum()
    share_b = np.abs(boots) / np.maximum(np.abs(boots).sum(1, keepdims=True), 1e-12)
    res = pd.DataFrame({
        "reference": refs.columns,
        "coefficient": b,
        "ci_low": np.quantile(boots, 0.025, axis=0),
        "ci_high": np.quantile(boots, 0.975, axis=0),
        "share": np.abs(b) / tot if tot > 0 else np.nan,
        "share_ci_low": np.quantile(share_b, 0.025, axis=0),
        "share_ci_high": np.quantile(share_b, 0.975, axis=0),
        "boot_nonzero": (boots != 0).mean(0),
        "delta_r2": d,
    })
    summ = {"r2_total": r2f, "r2_ci_low": float(np.quantile(r2b, 0.025)),
            "r2_ci_high": float(np.quantile(r2b, 0.975)), "r2_shared_axis": r2s,
            "n_genes": n, "fit": "signed_ridge" if signed else "nnls"}
    return res, summ


def class_shares(res: pd.DataFrame, cls: dict) -> pd.DataFrame:
    r = res.assign(cls=res.reference.map(cls))
    return r.groupby("cls", as_index=False)[["share", "delta_r2"]].sum()


def merge_collinear(refs: pd.DataFrame, r_max: float) -> tuple[pd.DataFrame, list]:
    """Average (unit-RMS) reference pairs whose Pearson r over the informative genes exceeds
    r_max; repeat until no pair exceeds it. Returns the merged panel and the merge log."""
    R = pd.DataFrame(unit_rms(refs.values.astype(float)), index=refs.index, columns=refs.columns)
    merged = []
    while R.shape[1] > 1:
        C = np.corrcoef(R.values.T)
        np.fill_diagonal(C, -1)
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if C[i, j] <= r_max:
            break
        a, b = sorted([R.columns[i], R.columns[j]])
        new = "+".join(sorted(f"{a}+{b}".split("+")))
        merged.append((a, b, float(C[i, j]), new))
        v = unit_rms(((R[a] + R[b]) / 2).values)
        R = R.drop(columns=[a, b])
        R[new] = v
    return R, merged
