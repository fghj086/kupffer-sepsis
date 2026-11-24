"""Loading and sample design for stage 02 (sorted liver cell bulk RNA-seq)."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lib.repro import project_root

ROOT = project_root()


def _p(rel: str) -> Path:
    return ROOT / rel


def read_series_matrix(rel: str) -> pd.DataFrame:
    """Sample-level fields of a GEO series matrix (one row per GSM)."""
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(_p(rel), "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
            rows.setdefault(parts[0][8:], []).append(parts[1:])
    out = {}
    for key, vals in rows.items():
        for i, v in enumerate(vals):
            if key == "characteristics_ch1" and v and ":" in v[0]:
                out["ch_" + v[0].split(":", 1)[0].strip().replace(" ", "_")] = \
                    [x.split(":", 1)[1].strip() if ":" in x else x for x in v]
            else:
                out[key if i == 0 else f"{key}_{i + 1}"] = v
    return pd.DataFrame(out)


def gene_annotation(gtf_rel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gene table (id, symbol, type) and transcript -> gene table from a GENCODE GTF."""
    genes, tx = [], []
    pat = re.compile(r'(\w+) "([^"]*)"')
    with gzip.open(_p(gtf_rel), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] not in ("gene", "transcript"):
                continue
            a = dict(pat.findall(f[8]))
            gid = a["gene_id"].split(".")[0]
            if f[2] == "gene":
                genes.append((gid, a["gene_name"], a["gene_type"], f[0]))
            else:
                tx.append((a["transcript_id"].split(".")[0], gid))
    g = pd.DataFrame(genes, columns=["gene_id", "symbol", "gene_type", "chrom"])
    g = g[~g.gene_id.str.endswith("_PAR_Y")].drop_duplicates("gene_id").set_index("gene_id")
    t = pd.DataFrame(tx, columns=["transcript_id", "gene_id"]).drop_duplicates("transcript_id")
    return g, t.set_index("transcript_id")


def to_symbols(counts: pd.DataFrame, genes: pd.DataFrame, protein_coding_only: bool) -> pd.DataFrame:
    """Ensembl gene counts -> symbol counts (duplicated symbols summed)."""
    ids = counts.index.str.split(".").str[0]
    counts = counts.groupby(ids).sum()
    ann = genes.reindex(counts.index)
    keep = ann.symbol.notna()
    if protein_coding_only:
        keep &= ann.gene_type.eq("protein_coding")
    counts = counts[keep.values]
    return counts.groupby(ann.symbol[keep].values).sum()


def load_gse328774(cfg: dict, genes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]["GSE328774"]
    sm = read_series_matrix(d["series_matrix"])
    raw = pd.read_csv(_p(d["counts"]), sep="\t", index_col=0)
    lib = sm["description"].str.replace("Library name: ", "", regex=False)
    meta = pd.DataFrame({
        "dataset": "GSE328774", "gsm": sm["geo_accession"].values, "library": lib.values,
        "cell_type": "KC", "condition": np.where(sm["ch_treatment"].str.upper() == "CLP", "CLP", "sham"),
        "time_h": 2, "sex": "not_stated", "title": sm["title"].values,
    })
    meta["sample"] = "GSE328774_" + meta.library
    meta = meta.set_index("sample")
    counts = raw[meta.library.values]
    counts.columns = meta.index
    return to_symbols(counts, genes, cfg["filter"]["protein_coding_only"]), meta


def load_gse311736(cfg: dict, genes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]["GSE311736"]
    sm = read_series_matrix(d["series_matrix"])
    raw = pd.read_csv(_p(d["counts"]), sep="\t", index_col=0)
    lib = sm["description_2"]
    ct = lib.str.split("_").str[0].replace({"CD45": "CD45_nonKC"})
    meta = pd.DataFrame({
        "dataset": "GSE311736", "gsm": sm["geo_accession"].values, "library": lib.values,
        "cell_type": ct.values,
        "condition": np.where(sm["ch_treatment"].str.upper() == "CLP", "CLP", "sham"),
        "time_h": 8, "sex": "male", "title": sm["title"].values,
    })
    meta["sample"] = "GSE311736_" + meta.library
    meta = meta.set_index("sample")
    counts = raw[meta.library.values]
    counts.columns = meta.index
    return to_symbols(counts, genes, cfg["filter"]["protein_coding_only"]), meta


def load_gse303231(cfg: dict, genes: pd.DataFrame, tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """kallisto transcript estimates: lanes summed per library, transcripts summed per gene, rounded."""
    d = cfg["data"]["GSE303231"]
    sm = read_series_matrix(d["series_matrix"])
    raw = pd.read_csv(_p(d["counts"]), sep="\t", index_col=0)
    design = cfg["gse303231_samples"]
    sm["code"] = sm["title"].str.split(" - ").str[0].str.strip()
    sm = sm[sm.code.isin(design)].copy()
    lanes = {c: [x for x in raw.columns if x.rsplit("_", 1)[0] == c] for c in design}
    lib = pd.DataFrame({c: raw[cols].sum(axis=1) for c, cols in lanes.items()})
    tid = lib.index.str.split(".").str[0]
    gid = tx.gene_id.reindex(tid)
    mapped = gid.notna().values
    gcounts = lib[mapped].groupby(gid[mapped].values).sum().round().astype(int)
    meta = pd.DataFrame({
        "dataset": "GSE303231", "gsm": sm["geo_accession"].values, "library": sm.code.values,
        "cell_type": "HEP",
        "condition": [("sepsis" if design[c]["condition"] == "sepsis" else "control") for c in sm.code],
        "time_h": np.nan, "sex": [design[c]["sex"] for c in sm.code], "title": sm["title"].values,
    })
    # check the configured sex/condition against the GEO title of every library
    for _, r in meta.iterrows():
        t = r.title.lower()
        assert r.sex in t and (("septic" in t) == (r.condition == "sepsis")), r.title
    meta["sample"] = "GSE303231_" + meta.library
    meta = meta.set_index("sample")
    gcounts = gcounts[meta.library.values]
    gcounts.columns = meta.index
    info = {"transcripts": int(len(lib)), "transcripts_mapped": int(mapped.sum()),
            "reads_mapped_fraction": float(lib[mapped].values.sum() / lib.values.sum()),
            "lanes_per_library": {c: len(v) for c, v in lanes.items()}}
    return to_symbols(gcounts, genes, cfg["filter"]["protein_coding_only"]), meta, info


def read_gmt(path_rel: str) -> dict[str, list[str]]:
    out = {}
    with open(_p(path_rel)) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[f[0]] = [g for g in f[2:] if g]
    return out


def build_programs(cfg: dict) -> dict[str, list[str]]:
    sets = {}
    for lib in cfg["data"]["gmt"].values():
        sets.update(read_gmt(lib))
    gl = dict(cfg["genes"])
    excl = set(gl["kc_identity"]) | set(gl["monocyte_recruited"])
    gl["clearance_minus_identity"] = [g for g in gl["clearance"] if g not in excl]
    progs = {}
    for name, src in cfg["programs"].items():
        progs[name] = list(gl[src]) if src in gl else list(sets[src])
    return progs
