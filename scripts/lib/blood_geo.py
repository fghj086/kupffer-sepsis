"""GEO series-matrix parsing and probe-to-gene collapsing for the human blood cohorts."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _unquote(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith('"') and tok.endswith('"'):
        tok = tok[1:-1]
    return tok.replace('""', '"').strip()


def parse_series_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expression, sample_metadata).

    Expression is probes x samples (float, NaN preserved); sample metadata is
    samples x annotation fields. Repeated !Sample_characteristics_ch1 lines are
    split on the first ': ' into named columns; unnamed ones get char_<i>.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    sample_lines: list[tuple[str, list[str]]] = []
    table_rows: list[list[str]] = []
    in_table = False
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                if line:
                    table_rows.append(line.split("\t"))
                continue
            if line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0][len("!Sample_"):]
                sample_lines.append((key, [_unquote(x) for x in parts[1:]]))

    header = [_unquote(x) for x in table_rows[0]]
    expr = pd.DataFrame(
        [r[1:] for r in table_rows[1:]],
        index=[_unquote(r[0]) for r in table_rows[1:]],
        columns=header[1:],
    ).apply(pd.to_numeric, errors="coerce")
    expr.index.name = "probe"

    gsm = None
    for key, vals in sample_lines:
        if key == "geo_accession":
            gsm = vals
            break
    if gsm is None:
        raise ValueError(f"no !Sample_geo_accession in {path}")

    meta: dict[str, list[str]] = {}
    char_i = 0
    for key, vals in sample_lines:
        if key == "characteristics_ch1":
            names = [v.split(": ", 1)[0] if ": " in v else "" for v in vals]
            named = [n for n in names if n]
            field = (max(set(named), key=named.count) if named
                     else f"char_{char_i}")
            field = re.sub(r"[^0-9a-zA-Z]+", "_", field).strip("_").lower()
            char_i += 1
            body = [v.split(": ", 1)[1] if ": " in v else v for v in vals]
            while field in meta:
                field = f"{field}_2"
            meta[field] = body
        elif key in ("title", "source_name_ch1", "description",
                     "platform_id", "geo_accession"):
            col = key if key not in meta else f"{key}_2"
            meta[col] = vals

    n = len(gsm)
    meta = {k: v for k, v in meta.items() if len(v) == n}
    md = pd.DataFrame(meta, index=gsm)
    md.index.name = "sample_id"
    expr = expr.loc[:, [c for c in md.index if c in expr.columns]]
    return expr, md.loc[expr.columns]


def load_platform_annotation(path: str | Path) -> pd.Series:
    """probe -> single gene symbol; multi-mapping (///) probes are dropped."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows, header = [], None
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            if header is None:
                if line.startswith("ID\t"):
                    header = line.rstrip("\n").split("\t")
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(parts[:len(header)])
    ann = pd.DataFrame(rows, columns=header)
    sym = ann.set_index("ID")["Gene symbol"].astype(str).str.strip()
    sym = sym[(sym != "") & (sym != "nan")]
    sym = sym[~sym.str.contains("///", regex=False)]
    return sym


def collapse_probes(expr: pd.DataFrame, probe2gene: pd.Series,
                    how: str = "max_mean") -> pd.DataFrame:
    """Collapse probes to gene symbols. 'max_mean' keeps the highest-mean probe."""
    common = expr.index.intersection(probe2gene.index)
    e = expr.loc[common]
    g = probe2gene.loc[common]
    if how == "max_mean":
        order = e.mean(axis=1).sort_values(ascending=False).index
        e = e.loc[order]
        g = g.loc[order]
        keep = ~g.duplicated(keep="first")
        out = e.loc[keep.values]
        out.index = g.loc[keep.values].values
    elif how == "median":
        out = e.groupby(g.values).median()
    else:
        raise ValueError(how)
    out.index.name = "gene"
    return out.sort_index()


