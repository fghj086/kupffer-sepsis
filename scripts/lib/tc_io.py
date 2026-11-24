"""GSE224127 liver loading for stage 04.

Count-matrix columns are <tissue>_<time label>_<mouse>. Treatment is a property of the
mouse, so annotation is joined on the mouse number. Two annotation sources exist:
the authors' GSE224127_metadata.txt (mice 1-294, with a QC flag) and the GEO series
matrices (all deposited samples, with genotype). The author file is preferred where it
exists; GEO fills the later experiments (mice >= 295 and the mild/moderate CLP arms).
Both labels are kept so any disagreement is visible in design.tsv.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_geo_samples(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        d: dict[str, list[str]] = {}
        n_desc = 0
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("!series_matrix_table_begin"):
                    break
                v = [x.strip('"') for x in line.rstrip("\n").split("\t")]
                key, vals = v[0], v[1:]
                if key == "!Sample_title":
                    d["title"] = vals
                elif key == "!Sample_geo_accession":
                    d["gsm"] = vals
                elif key == "!Sample_characteristics_ch1":
                    tag = vals[0].split(": ")[0]
                    d[tag] = [x.split(": ", 1)[-1] for x in vals]
                elif key == "!Sample_description":
                    d[f"description{n_desc}"] = vals
                    n_desc += 1
        frames.append(pd.DataFrame(d))
    m = pd.concat(frames, ignore_index=True).fillna("")
    m["tissue_code"] = m["title"].str.split("_").str[0].str.upper()
    m["mouse"] = m["title"].str.extract(r"(\d+)$")[0].astype(int)
    exp = m.filter(like="description").apply(
        lambda r: next((x for x in r if re.fullmatch(r"experiment\d+", x)), ""), axis=1)
    m["geo_experiment"] = exp
    return m


def read_author_metadata(path) -> pd.DataFrame:
    a = pd.read_csv(path, sep="\t")
    a["tissue_code"] = a["id"].str.split("_").str[0].str.upper()
    a["mouse"] = a["id"].str.extract(r"_(\d+)$")[0].astype(int)
    return a


def load_liver_counts(path, tissue: str = "LI") -> pd.DataFrame:
    """Genes x liver columns (integer counts). Handles CR, LF and CRLF line endings."""
    X = pd.read_csv(path, sep="\t", index_col=0, engine="c")
    X = X.loc[X.index.notna()]
    X.index = X.index.astype(str)
    X = X.groupby(level=0).sum()  # duplicated symbols, if any
    cols = [c for c in X.columns if c.split("_")[0].upper() == tissue]
    return X[cols].astype(np.int64)


def _strip_time(cond: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)(?:_d([0-9.]+))?$", cond)
    return m.group(1), (m.group(2) or "")


def annotate_columns(cols, file_key: str, geo: pd.DataFrame, author: pd.DataFrame,
                     tissue: str = "LI") -> pd.DataFrame:
    g = geo[geo.tissue_code == tissue].set_index("mouse")
    gany = geo.drop_duplicates("mouse").set_index("mouse")  # mouse-level fallback
    a = author[author.tissue_code == tissue].set_index("mouse")
    rows = []
    for c in cols:
        mouse = int(c.split("_")[-1])
        col_time = c.split("_")[1].lstrip("d")
        geo_tr = g.loc[mouse, "treatment"] if mouse in g.index else (
            gany.loc[mouse, "treatment"] if mouse in gany.index else "")
        geno = g.loc[mouse, "genotype"] if mouse in g.index else (
            gany.loc[mouse, "genotype"] if mouse in gany.index else "")
        gsm = g.loc[mouse, "gsm"] if mouse in g.index else ""
        auth = a.loc[mouse, "condition"] if mouse in a.index else ""
        qc = int(a.loc[mouse, "qc"]) if mouse in a.index else 0
        rows.append({"column": c, "file_key": file_key, "mouse": mouse, "gsm": gsm,
                     "geo_treatment": geo_tr, "genotype": geno or "Wild type",
                     "author_condition": auth, "author_qc_flag": qc,
                     "column_time_d": col_time})
    return pd.DataFrame(rows)


def harmonise(ann: pd.DataFrame) -> pd.DataFrame:
    """Add model, group and time_d columns used by every analysis."""
    out = []
    for _, r in ann.iterrows():
        src = r.author_condition or r.geo_treatment
        base, t_auth = _strip_time(src)
        _, t_geo = _strip_time(r.geo_treatment) if r.geo_treatment else ("", "")
        b = base
        if b.startswith("CLP_"):
            b = b[4:]
        if b == "Control" or b == "Control_d0":
            group = "control"
        elif b == "sham":
            group = "sham"
        else:
            group = b
        geno = r.genotype
        if geno != "Wild type" and geno.replace(" KO", "") not in group:
            group = f"{group}_{geno.replace(' ', '')}"
        model = ("CLP" if ("CLP" in r.file_key) else
                 "LPS" if r.file_key.startswith("LPS") else "cytokine")
        time_d = r.column_time_d
        d = r.to_dict()
        d.update({"condition_source": "author_metadata" if r.author_condition else "GEO",
                  "group": group, "model": model, "time_d": float(time_d),
                  "time_label_author": t_auth if r.author_condition else "",
                  "time_label_geo": t_geo})
        out.append(d)
    return pd.DataFrame(out)


def build_design(cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    gc = cfg["gse224127"]
    raw, extra = root / gc["raw_dir"], root / gc["extra_dir"]
    geo = read_geo_samples([raw / f for f in gc["series_matrices"]])
    author = read_author_metadata(extra / gc["author_metadata"])
    designs, counts = [], {}
    for key, spec in gc["files"].items():
        p = (raw if spec["path"] == "raw" else extra) / spec["file"]
        X = load_liver_counts(p, gc["tissue"])
        ann = harmonise(annotate_columns(X.columns, key, geo, author, gc["tissue"]))
        ann["count_file"] = spec["file"]
        designs.append(ann)
        counts[key] = X
    design = pd.concat(designs, ignore_index=True)
    return design, counts


def input_paths(cfg: dict, root: Path) -> list[Path]:
    gc = cfg["gse224127"]
    raw, extra = root / gc["raw_dir"], root / gc["extra_dir"]
    ps = [raw / f for f in gc["series_matrices"]] + [extra / gc["author_metadata"]]
    for spec in gc["files"].values():
        ps.append((raw if spec["path"] == "raw" else extra) / spec["file"])
    return ps
