"""Stage 06 loaders: one function per reference study.

Every loader returns (X, arms) where X is genes x samples on the deposited scale
(raw counts, normalised linear values, or log2 values), indexed by mouse gene symbol and
restricted to protein-coding genes (Ensembl 102 map), and arms is a Series column -> arm label.
Columns that match no arm pattern are dropped (other genotypes, other treatments).
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def gene_map(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _collapse_symbols(X: pd.DataFrame) -> pd.DataFrame:
    """Duplicate symbols: keep the row with the largest total signal."""
    X = X.loc[X.index.notna() & (X.index != "")]
    tot = X.sum(axis=1)
    X = X.loc[tot.sort_values(ascending=False).index]
    return X[~X.index.duplicated(keep="first")]


def ensembl_to_symbol(X: pd.DataFrame, gm: pd.DataFrame, coding_only=True) -> pd.DataFrame:
    X = X.copy()
    X.index = X.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    m = gm.set_index("ensembl_gene_id").reindex(X.index)
    keep = m["gene_name"].notna()
    if coding_only:
        keep &= m["gene_biotype"].eq("protein_coding")
    X = X[keep.values]
    X.index = m.loc[keep.values, "gene_name"].values
    return _collapse_symbols(X)


def restrict_symbols(X: pd.DataFrame, gm: pd.DataFrame, coding_only=True) -> pd.DataFrame:
    ok = gm if not coding_only else gm[gm.gene_biotype.eq("protein_coding")]
    X = X[X.index.isin(set(ok.gene_name))]
    return _collapse_symbols(X)


def assign_arms(cols, patterns: dict) -> pd.Series:
    out = {}
    for c in cols:
        for pat, arm in patterns.items():
            if re.search(pat, str(c)):
                out[c] = arm
                break
    return pd.Series(out, dtype=object)


def _select(X, patterns):
    arms = assign_arms(X.columns, patterns)
    return X[arms.index], arms


# ---------------------------------------------------------------------------------- loaders
def load_gse200210(root: Path, spec: dict, gm: pd.DataFrame, cfg01: dict):
    from lib.perturb_de import sample_table
    c = cfg01["data"]
    cnt = pd.read_csv(root / c["counts"], sep="\t", index_col=0)
    st = sample_table(root / c["series_matrix"], cnt.columns, cfg01["conditions"])
    X = ensembl_to_symbol(cnt, gm)
    return X[st.index], st["condition"].astype(object)


def load_gse232827(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], index_col=0)
    X = X.set_index("X")
    X = ensembl_to_symbol(X, gm)
    return _select(X, spec["arms"])


def load_gse315641(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], sep="\t", index_col=0)
    return _select(restrict_symbols(X, gm), spec["arms"])


def load_gse72964(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], sep="\t", index_col=0)
    sym = X["Annotation/Divergence"].astype(str).str.split("|").str[0]
    fp = X[[c for c in X.columns if c.endswith("FPKM")]].copy()
    fp.index = sym.values
    fp = fp.groupby(level=0).sum()          # RefSeq isoforms of one gene -> gene
    return _select(restrict_symbols(fp, gm), spec["arms"])


def load_gse221734(root, spec, gm, _):
    cols = {}
    for f in sorted((root / spec["path"]).glob("GSM*.gene.fpkm.txt.gz")):
        d = pd.read_csv(f, sep="\t", index_col=0)
        name = f.name.split("_", 1)[1].replace(".gene.fpkm.txt.gz", "")
        cols[name] = d["expected_count"]
    X = pd.DataFrame(cols).fillna(0).round().astype(np.int64)
    return _select(restrict_symbols(X, gm), spec["arms"])


def load_gse193577(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], sep="\t", index_col=0)
    return _select(ensembl_to_symbol(X, gm), spec["arms"])


def load_gse240633(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], sep="\t", index_col=0)
    X = X[[c for c in X.columns if c.endswith("_count")]].fillna(0).round().astype(np.int64)
    return _select(ensembl_to_symbol(X, gm), spec["arms"])


def load_gse300953(root, spec, gm, _):
    cols = {}
    for f in sorted((root / spec["path"]).glob("GSM*_processed.txt.gz")):
        d = pd.read_csv(f, sep="\t", index_col=0)
        cols[d.columns[0]] = d.iloc[:, 0]
    X = pd.DataFrame(cols)
    # values are log2 TMM-CPM; collapse duplicates on the linear scale
    lin = restrict_symbols(2.0 ** X - 1.0, gm).clip(lower=0)
    return _select(np.log2(lin + 1.0), spec["arms"])


def load_gse157518(root, spec, gm, _):
    X = pd.read_csv(root / spec["path"], sep="\t")   # header is one field short: row ids
    X = X.set_index("EnsemblID").select_dtypes("number")
    return _select(ensembl_to_symbol(X, gm), spec["arms"])


def load_gse306502(root, spec, gm, _):
    cols = {}
    for f in sorted((root / spec["path"]).glob("GSM*_featurecounts_fragments.txt.gz")):
        d = pd.read_csv(f, sep="\t", comment="#", index_col=0)
        name = f.name.split("_")[1]
        cols[name] = d.iloc[:, -1]
    X = pd.DataFrame(cols)
    return _select(ensembl_to_symbol(X, gm), spec["arms"])


def load_gse224127(root, spec, gm, cfg):
    """Liver columns of one GSE224127 count file; arms from the stage 04 design table."""
    from lib.tc_io import load_liver_counts
    gc = cfg["gse224127"]
    des = pd.read_csv(root / gc["design"], sep="\t")
    des = des[(des.file_key == spec["file_key"]) & (des.author_qc_flag == 0)]
    fname = des["count_file"].iloc[0]
    p = root / gc["raw_dir"] / fname
    if not p.exists():
        p = root / gc["extra_dir"] / fname
    X = load_liver_counts(p)
    X = restrict_symbols(X, gm)
    des = des.set_index("column")
    cols = [c for c in X.columns if c in des.index]
    return X[cols], des.loc[cols, "group"].astype(object)


def load_study(name: str, spec: dict, root: Path, gm: pd.DataFrame, cfg: dict, cfg01: dict):
    fn = globals()[f"load_{spec['loader']}"]
    extra = cfg01 if spec["loader"] == "gse200210" else cfg
    X, arms = fn(root, spec, gm, extra)
    X = X.loc[:, arms.index]
    if X.shape[1] == 0:
        raise SystemExit(f"{name}: no columns matched the arm patterns")
    return X, arms


def study_inputs(cfg: dict, root: Path) -> list[Path]:
    out = []
    for spec in cfg["studies"].values():
        if "path" in spec:
            p = root / spec["path"]
            out += sorted(p.glob("*")) if p.is_dir() else [p]
    return out
