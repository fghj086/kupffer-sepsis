"""Loading of the two liver scRNA-seq datasets into AnnData objects with raw counts."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """Sample-level metadata rows (!Sample_*) of a GEO series matrix."""
    rows = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0][8:]
            vals = [v.strip('"') for v in parts[1:]]
            i = 1
            k = key
            while k in rows:
                i += 1
                k = f"{key}_{i}"
            rows[k] = vals
    return pd.DataFrame(rows)


def _counts_matrix(ad: sc.AnnData):
    """Integer counts and gene names. GSE290684 files store scaled HVGs in X and
    log1p(count / n_count * 1e4) for all genes in .raw; counts are recovered exactly as
    round(expm1(x) * n_count / 1e4) and checked against the stored per-cell n_count."""
    if ad.raw is not None and "n_count" in ad.obs:
        R = ad.raw.X.tocsr().astype(np.float64)
        R.data = np.expm1(R.data)
        R = sp.diags(ad.obs["n_count"].values / 1e4) @ R
        dev = float(np.abs(R.data - np.round(R.data)).max()) if R.nnz else 0.0
        R.data = np.round(R.data)
        R.eliminate_zeros()
        tot_err = float(np.abs(np.asarray(R.sum(1)).ravel() - ad.obs["n_count"].values).max())
        if dev > 0.05 or tot_err > 0.5:
            raise ValueError(f"count recovery failed (max dev {dev}, total err {tot_err})")
        return R.tocsr().astype(np.float32), ad.raw.var_names, f"raw_log1p_inverted(maxdev={dev:.4f},total_err={tot_err:.1f})"
    M = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
    d = M.data[: min(M.nnz, 200000)]
    if d.size and np.allclose(d, np.round(d)) and d.min() >= 0:
        return M.astype(np.float32), ad.var_names, "X"
    raise ValueError("no integer count matrix found")


def load_gse279167(cfg: dict, root: Path) -> sc.AnnData:
    d = root / cfg["raw_dir"]
    ads = []
    for gsm, info in cfg["files"].items():
        a = sc.read_10x_h5(d / info["file"])
        a.var_names_make_unique()
        a.obs_names = [f"{gsm}_{b}" for b in a.obs_names]
        a.obs["library"] = gsm
        a.obs["sample"] = info["sample"]
        a.obs["condition"] = info["condition"]
        ads.append(a)
    ad = sc.concat(ads, join="inner", merge="same")
    ad.X = ad.X.tocsr().astype(np.float32)
    return ad


def gse290684_design(root: Path, raw_dir: str) -> pd.DataFrame:
    sm = parse_series_matrix(root / raw_dir / "GSE290684_series_matrix.txt.gz")
    ch = [c for c in sm.columns if c.startswith("characteristics_ch1")]
    des = pd.DataFrame({"library": sm["geo_accession"], "title": sm["title"],
                        "library_name": sm["description"]})
    for c in ch:
        k = sm[c].str.split(": ", n=1).str[0].iloc[0]
        des[k.replace(" ", "_")] = sm[c].str.split(": ", n=1).str[1]
    des["file"] = sm["supplementary_file_1"].str.split("/").str[-1]
    # All 12 libraries are CLP 24 h (series overall design); "control" = saline gavage.
    des["condition"] = np.where(des["treatment"].str.upper() == "TPPU", "CLP_TPPU", "CLP_vehicle")
    des["sample"] = des["title"]
    return des


def load_gse290684(cfg: dict, root: Path) -> tuple[sc.AnnData, pd.DataFrame]:
    des = gse290684_design(root, cfg["raw_dir"])
    d = root / cfg["raw_dir"]
    ads, src = [], []
    for _, r in des.iterrows():
        f = d / r["file"]
        if not f.exists():
            gz = Path(str(f) + ".gz")
            f = gz if gz.exists() else f
        a = sc.read_h5ad(f)
        M, genes, where = _counts_matrix(a)
        b = sc.AnnData(M, obs=pd.DataFrame(index=[f"{r['library']}_{x}" for x in a.obs_names]),
                       var=pd.DataFrame(index=pd.Index(genes).astype(str)))
        b.var_names_make_unique()
        b.obs["library"] = r["library"]
        b.obs["sample"] = r["sample"]
        b.obs["condition"] = r["condition"]
        ads.append(b)
        src.append({"library": r["library"], "counts_from": where, "cells_in_file": a.n_obs,
                    "genes_in_file": len(genes),
                    "obs_columns_in_file": ";".join(map(str, a.obs.columns))})
    ad = sc.concat(ads, join="inner", merge="same")
    ad.X = ad.X.tocsr().astype(np.float32)
    des = des.merge(pd.DataFrame(src), on="library")
    return ad, des


def species_check(var_names) -> dict:
    v = pd.Index(var_names)
    mouse_like = v.str.match(r"^[A-Z][a-z0-9]+").mean()
    human_like = v.str.match(r"^[A-Z0-9]+$").mean()
    return {"frac_mouse_case": float(mouse_like), "frac_upper_case": float(human_like),
            "has_mt-": bool(v.str.startswith("mt-").any()), "has_MT-": bool(v.str.startswith("MT-").any())}
