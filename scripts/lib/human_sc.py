"""Stage 07: blood monocytes in GSE252331 single-cell libraries, summed per donor.

Cells pass QC (genes >= min_genes, mitochondrial fraction < max_pct_mt), are log-normalised and
clustered (Leiden on PCA of highly variable genes). Clusters whose mean classical- or non-classical-monocyte
marker score is highest among canonical blood cell-type marker sets are labelled monocytes (an
annotation; the two subsets are pooled). Raw counts of those cells are summed per donor and converted to log2 CPM; the donor
is the unit of every comparison.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io

TYPES = {"CD14 monocyte": ["CD14", "LYZ", "VCAN", "S100A9", "FCN1"],
         "CD16 monocyte": ["FCGR3A", "MS4A7", "LST1", "CDKN1C"],
         "T cell": ["CD3E", "CD3D", "IL7R", "CD8A"],
         "NK cell": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
         "B cell": ["MS4A1", "CD79A", "CD79B"],
         "Plasmablast": ["JCHAIN", "MZB1"],
         "DC": ["FCER1A", "CLEC10A", "CD1C", "LILRA4"],
         "Platelet": ["PPBP", "PF4"],
         "Erythroid": ["HBB", "HBA1"]}


def _read(raw: Path, prefix: str, ens2sym: pd.Series) -> ad.AnnData:
    X = scipy.io.mmread(gzip.open(raw / f"{prefix}_quants_mat.mtx.gz")).tocsr().astype(np.float32)
    genes = pd.read_csv(raw / f"{prefix}_quants_mat_cols.txt.gz", header=None)[0].astype(str).str.split(".").str[0]
    cells = pd.read_csv(raw / f"{prefix}_quants_mat_rows.txt.gz", header=None)[0].astype(str)
    if X.shape == (len(genes), len(cells)):
        X = X.T.tocsr()
    a = ad.AnnData(X, obs=pd.DataFrame(index=prefix + ":" + cells.values), var=pd.DataFrame(index=genes.values))
    s = a.var_names.map(ens2sym)
    a = a[:, s.notna()].copy()
    a.var_names = s[s.notna()].astype(str)
    a.var_names_make_unique()
    return a


def donor_monocytes(raw: Path, ens2sym: pd.Series, cfg: dict, seed: int):
    brief = (raw / "gsm_brief.txt").read_text().replace("\r", "")
    meta = []
    for b in brief.split("^SAMPLE = ")[1:]:
        title = re.search(r"!Sample_title = (.*)", b).group(1)
        if not title.endswith("GEX"):
            continue
        f = re.search(r"suppl/(GSM\d+_[^/\s]+?)_quants_mat\.mtx\.gz", b).group(1)
        grp = ("healthy" if title.startswith("Healthy") else "sepsis_day4" if "Day 4" in title
               else "CCI_day14" if "CCI" in title else "RAP_day14")
        meta.append({"prefix": f, "subject": title.replace(", GEX", "").split(",")[0], "group": grp})
    meta = pd.DataFrame(meta)
    parts = []
    for _, m in meta.iterrows():
        a = _read(raw, m.prefix, ens2sym)
        sc.pp.filter_cells(a, min_genes=cfg["min_genes"])
        a.obs["subject"], a.obs["group"] = m.subject, m.group
        parts.append(a)
    a = ad.concat(parts, join="inner")
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    a = a[a.obs.pct_counts_mt < cfg["max_pct_mt"]].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="subject")
    sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.leiden(a, resolution=cfg["leiden_resolution"], random_state=seed, flavor="igraph",
                 n_iterations=2, directed=False)
    for t, g in TYPES.items():
        sc.tl.score_genes(a, [x for x in g if x in a.var_names], score_name=f"s_{t}", random_state=seed)
    cl = a.obs.groupby("leiden", observed=True)[[f"s_{t}" for t in TYPES]].mean()
    lab = cl.idxmax(axis=1).str[2:]
    # classical and non-classical monocyte clusters are pooled: without cross-donor integration some
    # clusters are donor-specific, and a cluster-level CD14 / FCGR3A split assigned whole donors
    lab = lab.replace({"CD14 monocyte": "monocyte", "CD16 monocyte": "monocyte"})
    a.obs["cell_type"] = a.obs.leiden.map(lab).astype(str)
    mono = a[a.obs.cell_type == "monocyte"]
    counts = {}
    ncell = mono.obs.groupby("subject").size()
    for s in ncell.index:
        X = mono[mono.obs.subject == s].layers["counts"]
        counts[s] = np.asarray(X.sum(axis=0)).ravel()
    C = pd.DataFrame(counts, index=mono.var_names)
    comp = pd.crosstab(a.obs.subject, a.obs.cell_type)
    return C, ncell, meta, comp
