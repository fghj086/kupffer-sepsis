"""Loaders for the processed protein tables used in stage 05.

Each loader returns (abundance, design, annotation):
  abundance  DataFrame, rows = gene symbol (one row per gene), columns = sample id, raw scale
  design     DataFrame with columns sample, group, source_column
  annotation DataFrame indexed by gene symbol: protein_group, genes_all, n_genes_in_group
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _collapse_to_gene(mat: pd.DataFrame, genes: pd.Series, groups: pd.Series):
    """One row per gene: the first gene of each protein group names it; when several protein
    groups share a gene, the group quantified in most samples (then highest median) is kept."""
    first = genes.fillna("").astype(str).str.split(";").str[0].str.strip()
    ok = first != ""
    mat, first, genes, groups = mat[ok], first[ok], genes[ok], groups[ok]
    nvalid = mat.notna().sum(1)
    med = mat.median(1)
    order = pd.DataFrame({"gene": first, "nvalid": nvalid, "med": med}).sort_values(
        ["gene", "nvalid", "med"], ascending=[True, False, False])
    keep = order.drop_duplicates("gene").index
    out = mat.loc[keep]
    out.index = first.loc[keep].to_numpy()
    ann = pd.DataFrame({"protein_group": groups.loc[keep].to_numpy(),
                        "genes_all": genes.loc[keep].to_numpy()}, index=out.index)
    ann["n_genes_in_group"] = ann["genes_all"].str.count(";") + 1
    ann["n_groups_sharing_gene"] = first.value_counts().reindex(out.index).to_numpy()
    return out.sort_index(), ann.sort_index()


def load_diann_pg(zip_path: Path, design_xlsx: Path):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith("report.pg_matrix.tsv")
                and "first-pass" not in n][0]
        pg = pd.read_csv(io.BytesIO(z.read(name)), sep="\t")
    des = pd.read_excel(design_xlsx)
    des.columns = [c.strip() for c in des.columns]
    raw_cols = [c for c in pg.columns if c.lower().endswith(".raw")]
    base = {re.split(r"[\\/]", c)[-1]: c for c in raw_cols}
    rows = []
    for _, r in des.iterrows():
        fn = re.split(r"[\\/]", str(r["Rawfile"]))[-1]
        rows.append({"sample": str(r["Experiment"]), "group": str(r[next(c for c in des.columns if "ondition" in c)]),
                     "replicate": str(r["Replicate"]), "source_column": base[fn]})
    design = pd.DataFrame(rows)
    mat = pg[design["source_column"]].copy()
    mat.columns = design["sample"].to_list()
    mat = mat.replace(0, np.nan)
    ab, ann = _collapse_to_gene(mat, pg["Genes"], pg["Protein.Group"])
    return ab, design, ann


def load_copy_number_xlsx(xlsx: Path, sheet: str, groups: dict):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    sym = df["Symbol"].astype(str).str.strip()
    rows = []
    for col in df.columns[1:]:
        for prefix, lab in groups.items():
            if str(col).startswith(prefix):
                rep = str(col)[len(prefix):]
                rows.append({"sample": f"{lab}_{rep}", "group": lab, "replicate": rep,
                             "source_column": col})
    design = pd.DataFrame(rows)
    mat = df[design["source_column"]].apply(pd.to_numeric, errors="coerce")
    mat.columns = design["sample"].to_list()
    mat = mat.where(mat > 0)
    ab, ann = _collapse_to_gene(mat, sym, sym)
    ann["protein_group"] = "symbol_only"
    return ab, design, ann


def log_normalise(ab: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    x = np.log2(ab)
    if method == "median":
        x = x - x.median(0) + x.median(0).median()
    return x


def withdrawn_symbol_map(mgi_report: Path) -> dict:
    """Withdrawn MGI symbol -> current symbol (only one-to-one replacements)."""
    m = pd.read_csv(mgi_report, sep="\t", usecols=["Marker Symbol", "Status",
                                                   "Current Marker Symbol (if withdrawn)"],
                    dtype=str)
    m = m[(m["Status"] == "W") & m["Current Marker Symbol (if withdrawn)"].notna()]
    m = m[~m["Current Marker Symbol (if withdrawn)"].str.contains(r"[,|]")]
    m = m.drop_duplicates("Marker Symbol", keep=False)
    return dict(zip(m["Marker Symbol"], m["Current Marker Symbol (if withdrawn)"]))


def harmonise_symbols(ab: pd.DataFrame, ann: pd.DataFrame, mapping: dict):
    """Rename withdrawn symbols to current ones unless the current symbol is already a row."""
    present = set(ab.index)
    new = [mapping[g] if g in mapping and mapping[g] not in present else g for g in ab.index]
    renamed = pd.Series(new, index=ab.index)
    ann = ann.copy()
    ann["symbol_in_source"] = ab.index
    ab = ab.copy()
    ab.index, ann.index = renamed.to_numpy(), renamed.to_numpy()
    keep = ~ab.index.duplicated()
    return ab[keep], ann[keep], int((renamed != renamed.index).sum())
