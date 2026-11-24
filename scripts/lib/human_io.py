"""Stage 07 loaders: each returns (log2 expression, genes x samples, symbols) and a per-sample
design table with columns sample, subject, group (+ dataset-specific columns).
Every dataset is read and normalised on its own."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ GEO helpers
def series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta: dict[str, list] = {}
    rows, cols, inside = [], None, False
    with gzip.open(path, "rt", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("!series_matrix_table_begin"):
                inside = True
                continue
            if ln.startswith("!series_matrix_table_end"):
                break
            if inside:
                parts = [x.strip().strip('"') for x in ln.rstrip("\n").split("\t")]
                if cols is None:
                    cols = parts[1:]
                else:
                    rows.append(parts)
            elif ln.startswith("!Sample_"):
                k = ln.split("\t")[0][1:]
                v = [x.strip().strip('"') for x in ln.rstrip("\n").split("\t")[1:]]
                meta.setdefault(k, []).append(v)
    flat = {}
    for k, vals in meta.items():
        for i, v in enumerate(vals):
            flat[k if len(vals) == 1 else f"{k}_{i}"] = v
    m = pd.DataFrame(flat)
    m.index = m["Sample_geo_accession"]
    if rows:
        x = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=cols)
        x = x.apply(pd.to_numeric, errors="coerce")
    else:
        x = pd.DataFrame()
    return x, m


def characteristics(m: pd.DataFrame) -> pd.DataFrame:
    """'key: value' characteristic columns -> one column per key."""
    out = pd.DataFrame(index=m.index)
    for c in [c for c in m.columns if c.startswith("Sample_characteristics_ch1")]:
        for idx, v in m[c].items():
            if ":" in v:
                k, val = v.split(":", 1)
                out.loc[idx, k.strip()] = val.strip()
    return out


def gpl_table(path: Path, id_col: str, sym_col: str) -> pd.Series:
    lines = []
    with gzip.open(path, "rt", errors="replace") as fh:
        inside = False
        for ln in fh:
            if ln.startswith("!platform_table_begin"):
                inside = True
                continue
            if ln.startswith("!platform_table_end"):
                break
            if inside:
                lines.append(ln.rstrip("\n").split("\t"))
    t = pd.DataFrame(lines[1:], columns=lines[0])
    s = t.set_index(id_col)[sym_col].astype(str)
    if sym_col == "gene_assignment":     # Affymetrix: 'NM_x // SYMBOL // ...'
        s = s.str.split(" // ").str[1]
    s = s.replace({"": np.nan, "---": np.nan, "nan": np.nan}).dropna()
    return s[~s.str.contains(" ")]


def collapse(x: pd.DataFrame, id2sym: pd.Series) -> pd.DataFrame:
    """Probe -> gene by the probe with the highest mean expression."""
    sym = id2sym.reindex(x.index)
    x = x[sym.notna().values]
    sym = sym.dropna()
    mean = x.mean(axis=1)
    keep = mean.groupby(sym.values).idxmax()
    g = x.loc[keep.values]
    g.index = keep.index
    return g


def log2_if_linear(x: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    return np.log2(x.clip(lower=0) + prior) if np.nanmax(x.values) > 100 else x


def log2cpm(counts: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    lib = counts.sum(axis=0)
    return np.log2(counts / lib * 1e6 + prior)


# ------------------------------------------------------------------ datasets
def gse141864(root: Path, spec: dict, upd: dict):
    x, m = series_matrix(root / spec["matrix"])
    ch = characteristics(m)
    sym = gpl_table(root / spec["platform"], "ID", "gene_assignment")
    g = collapse(x, sym)
    d = pd.DataFrame({"sample": m.index,
                      "subject": ch["patients id nr"].str.replace("Nr ", "patient_", regex=False).values,
                      "group": np.where(ch["diagnosis"].str.startswith("Meningococcal"), "septic_shock", "control"),
                      "tissue": ch["tissue"].str.strip().str.lower().values,
                      "storage": ch["type of storage tissue"].values,
                      "title": m["Sample_title"].values})
    return g, d


def gse22248(root: Path, spec: dict, gpl: Path):
    x, m = series_matrix(root / spec["matrix"])
    ch = characteristics(m)
    g = collapse(x, gpl_table(gpl, "ID", "Symbol"))
    tr = ch["treatment"].str.lower()
    grp = np.select([tr.str.contains("twice"), tr.str.contains("lps")], ["LPS_LPS", "LPS"], "untreated")
    d = pd.DataFrame({"sample": m.index, "subject": ch["individual"].values, "group": grp})
    return g, d


def gse15219(root: Path, spec: dict, gpl: Path):
    x, m = series_matrix(root / spec["matrix"])
    g = log2_if_linear(collapse(x, gpl_table(gpl, "ID", "SYMBOL")), prior=16.0)
    t = m["Sample_title"]
    grp = t.str.replace(r"\s*rep\d", "", regex=True).str.replace("+", "_", regex=False)
    d = pd.DataFrame({"sample": m.index, "subject": "rep" + t.str.extract(r"rep(\d)")[0].values,
                      "group": grp.values})
    return g, d


def gse254271(root: Path, spec: dict, entrez2sym: pd.Series):
    x, m = series_matrix(root / spec["matrix"])
    ent = pd.Series(x.index.str.replace("_at", "", regex=False), index=x.index)
    sym = ent.map(entrez2sym).dropna()
    g = collapse(x, sym)
    ch = characteristics(m)
    grp = ch["treatment"].str.replace("GM-CSF", "", regex=False).str.replace(r"[+\s]+", "_", regex=True) \
        .str.strip("_").replace("", "GMCSF")
    d = pd.DataFrame({"sample": m.index, "group": grp.values,
                      "subject": "donor" + m["Sample_title"].str.extract(r"replicate (\d)")[0].values})
    return g, d


def gse162834(root: Path, spec: dict):
    c = pd.read_csv(root / spec["file"], sep="\t", index_col=0)
    meta = pd.read_csv(root / spec["file"].replace("rawCounts", "sample_metadata"), sep="\t")
    c = c.groupby(level=0).sum()
    c = c[(c >= 10).sum(axis=1) >= 3]
    g = log2cpm(c)
    d = meta.rename(columns={"Sample": "sample", "Group": "group"})
    d["group"] = d.group.str.replace("D3_", "", regex=False)
    d["subject"] = "culture_" + d["sample"]
    return g, d


def gse78162(root: Path, spec: dict):
    x, m = series_matrix(root / spec["matrix"])
    g = np.log2(x.clip(lower=1e-4))          # 2^-dCt x 100 relative to ACTB
    g = g.drop(index=[i for i in ["ACTB"] if i in g.index])
    desc = m["Sample_description_1"]
    d = pd.DataFrame({"sample": m.index, "group": desc.str.split("-").str[0].values,
                      "subject": "culture_" + desc.values})
    return g, d


def gse46955(root: Path, spec: dict, gpl: Path):
    x, m = series_matrix(root / spec["matrix"])
    g = collapse(x, gpl_table(gpl, "ID", "Symbol"))
    desc = m["Sample_description"]
    subj = desc.str.extract(r"^([A-Za-z]+\d+)")[0]
    state = np.select([desc.str.contains("hMo"), desc.str.contains("sepsis"), desc.str.contains("basal")],
                      ["healthy", "sepsis", "recovery"], "unknown")
    lps = desc.str.contains("LPS").map({True: "LPS", False: "basal"})
    d = pd.DataFrame({"sample": m.index, "subject": subj.values, "state": state, "stim": lps.values,
                      "description": desc.values})
    # healthy donors and patients share 'D' prefixes: patient IDs are made unique by state class
    d["subject"] = np.where(d.state == "healthy", "healthy_" + d.subject, "patient_" + d.subject)
    d["group"] = d.state + "_" + d.stim
    return g, d


def gse233483(root: Path, spec: dict):
    t = pd.read_csv(root / spec["file"], sep="\t")
    cols = [c for c in t.columns if c.startswith("KuLO_")]
    x = t.groupby("Gene")[cols].sum()
    x = x[(x >= 1).sum(axis=1) >= 2]
    g = np.log2(x + 1)                      # reads per million (AmpliSeq)
    grp = pd.Series(cols).str.replace(r"KuLO_|_\d$", "", regex=True).map(
        {"D14": "control", "sep": "LPS_IFNg", "ReD1": "recovery_d1", "ReD2": "recovery_d2", "ReD3": "recovery_d3"})
    d = pd.DataFrame({"sample": cols, "group": grp.values, "subject": ["culture_" + c for c in cols]})
    return g, d


# ------------------------------------------------------------------ blood cohorts (prepared matrices)
def blood_cohort(name: str, processed: Path, meta_dir: Path):
    x = pd.read_csv(processed / f"{name}_expr.tsv.gz", sep="\t", index_col=0)
    x = x[~x.index.isna()]
    x.index = x.index.astype(str)
    x = x[~x.index.str.match(r"^\d+-(Mar|Sep|Dec)$|^---$")]
    s = pd.read_csv(meta_dir / f"{name}_samples.tsv", sep="\t")
    s = s[s["sample"].isin(x.columns)].copy()
    if "patient" not in s:
        s["patient"] = s["sample"]
    s["subject"] = s["patient"].astype(str)
    if "group" not in s:
        s["group"] = np.where(s["died"].notna(), "sepsis", "other")
    s["group"] = s["group"].replace({"survivor": "sepsis", "nonsurvivor": "sepsis", "septic_shock": "sepsis"})
    if "day" not in s:
        s["day"] = np.where(s.group == "sepsis", 1, np.nan)
    return x[s["sample"]], s.reset_index(drop=True)
