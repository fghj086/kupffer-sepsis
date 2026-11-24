"""Whole-blood deconvolution with a sorted-cell reference and per-cohort platform calibration.

Reference: GSE60424 RNA-seq of neutrophils, monocytes, B cells, CD4 and CD8 T cells and NK cells
sorted from the whole blood of healthy donors and patients with sepsis (mean CPM per cell type).
Platform calibration: expression units differ gene by gene between platforms, so each gene's
reference profile is multiplied by a factor that makes the reference mixture at a typical adult
leukocyte differential equal the mean of the cohort's healthy controls. The calibrated reference is
then fitted to every sample by non-negative least squares; proportions are normalized to sum 1.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

TYPES = ["Neutrophils", "Monocytes", "B-cells", "CD4", "CD8", "NK"]
LABEL = {"Neutrophils": "neutrophil", "Monocytes": "monocyte", "B-cells": "B cell", "CD4": "CD4 T cell",
         "CD8": "CD8 T cell", "NK": "NK cell"}


def _matrix_field(path: Path, key: str) -> list[str]:
    with gzip.open(path, "rt") as f:
        for line in f:
            if key in line:
                return [x.strip('"').split(": ", 1)[-1] for x in line.rstrip("\n").split("\t")[1:]]
    raise KeyError(key)


def reference(raw: Path, sym: pd.Series, donors: tuple[str, ...]) -> pd.DataFrame:
    """Mean CPM per sorted cell type over donors with the given disease status."""
    d = pd.read_csv(raw / "GSE60424_GEOSubmit_FC1to11_normalized_counts.txt.gz", sep=r"\s+", index_col=0)
    sm = raw / "GSE60424_series_matrix.txt.gz"
    title = _matrix_field(sm, "!Sample_title")
    meta = pd.DataFrame({"ct": _matrix_field(sm, "celltype:"), "ds": _matrix_field(sm, "diseasestatus:")},
                        index=title).loc[d.columns]
    d = d[d.index.isin(sym.index)]
    d.index = sym[d.index].values
    d = d.groupby(level=0).sum()
    cpm = d.div(d.sum(), axis=1) * 1e6
    keep = meta.ct.isin(TYPES) & meta.ds.isin(donors)
    return cpm.loc[:, keep].T.groupby(meta.ct[keep]).mean().T[TYPES]


def signature(ref: pd.DataFrame, n: int, margin: float, min_log2: float = 5.0) -> list[str]:
    """Up to n genes per type whose log2 CPM exceeds every other type by at least `margin`."""
    lg = np.log2(ref + 1)
    out = []
    for t in TYPES:
        m = lg[t] - lg.drop(columns=t).max(axis=1)
        ok = (m >= margin) & (lg[t] >= min_log2) & ~lg.index.str.startswith(("MT-", "RPL", "RPS"))
        out += list(m[ok].sort_values(ascending=False).index[:n])
    return list(dict.fromkeys(out))


def calibrate(ref: pd.DataFrame, lin_controls: pd.DataFrame, prior: pd.Series, genes: list[str]) -> pd.DataFrame:
    """Reference in the cohort's units: per-gene factor = control mean / reference mixture at the prior."""
    g = [x for x in genes if x in lin_controls.index and x in ref.index]
    mix = ref.loc[g, TYPES] @ prior[TYPES]
    f = lin_controls.loc[g].mean(axis=1) / mix.replace(0, np.nan)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return ref.loc[f.index, TYPES].mul(f, axis=0)


def deconvolve(lin: pd.DataFrame, cref: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """NNLS on the signature genes, each gene scaled by its mean over the cohort's samples."""
    g = [x for x in genes if x in lin.index and x in cref.index]
    scale = lin.loc[g].mean(axis=1).replace(0, np.nan)
    B = lin.loc[g].div(scale, axis=0).fillna(0)
    R = cref.loc[g].div(scale, axis=0).fillna(0)
    P = np.array([nnls(R.values, B[c].values)[0] for c in B.columns])
    P = P / P.sum(axis=1, keepdims=True)
    return pd.DataFrame(P, index=lin.columns, columns=TYPES)


def gse185263_healthy(raw_root: Path, sym: pd.Series) -> pd.DataFrame:
    """CPM of the healthy donors sequenced with the GSE222393 patients' admission samples."""
    raw = raw_root / "GSE222393"
    title = _matrix_field(raw / "GSE185263_series_matrix.txt.gz", "!Sample_title")
    state = _matrix_field(raw / "GSE185263_series_matrix.txt.gz", "disease state:")
    healthy = [t for t, s in zip(title, state) if s == "healthy"]
    C = pd.read_csv(raw / "GSE185263_raw_counts.csv.gz", index_col=0)[healthy]
    C.index = C.index.str.split(".").str[0]
    gs = C.index.map(sym)
    C = C[gs.notna()].groupby(gs[gs.notna()]).sum()
    return C / C.sum(axis=0) * 1e6
