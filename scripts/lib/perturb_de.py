"""Stage 01 helpers: GSE200210 loading, sample mapping and DESeq2 contrasts (pydeseq2)."""
from __future__ import annotations

import gzip
import logging
import re
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger("01_perturbation")


def read_series_samples(path) -> pd.DataFrame:
    """Sample table (GSM, title, description, treatment) from a GEO series matrix."""
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, *vals = line.rstrip("\n").split("\t")
            vals = [v.strip('"') for v in vals]
            if key == "!Sample_characteristics_ch1":
                tag = vals[0].split(":")[0].strip()
                if tag == "treatment":
                    rows["treatment"] = [v.split(":", 1)[1].strip() for v in vals]
                continue
            if key in ("!Sample_geo_accession", "!Sample_title", "!Sample_description"):
                rows[key.replace("!Sample_", "")] = vals
    return pd.DataFrame(rows)


def sample_table(series_path, counts_cols, cond_map: dict) -> pd.DataFrame:
    """Map count-matrix columns to conditions via the series-matrix description field.

    The count file headers (e.g. 'dsHMGB1-2', 'M1-1', 'PBS3') equal !Sample_description.
    The condition is cross-checked against the free-text treatment field.
    """
    sm = read_series_samples(series_path)
    if set(sm["description"]) != set(counts_cols):
        raise SystemExit(f"count columns {sorted(counts_cols)} != series descriptions "
                         f"{sorted(sm['description'])}")
    out = []
    for _, r in sm.iterrows():
        m = re.match(r"^(PBS|frHMGB1|dsHMGB1|LPS|M1)-?(\d)$", r["description"])
        if not m:
            raise SystemExit(f"unparsed sample description {r['description']}")
        prefix, rep = m.group(1), int(m.group(2))
        cond = cond_map[prefix]
        trt = r["treatment"]
        expect = {"PBS": "PBS", "frHMGB1": "frHMGB1", "dsHMGB1": "dsHMGB1",
                  "LPS": "LPS-EK", "LPS_IFNg": "IFN-gamma"}[cond]
        if expect not in trt or (cond == "LPS" and "IFN" in trt):
            raise SystemExit(f"treatment text '{trt}' inconsistent with {cond}")
        if f"replicate {rep}" not in r["title"]:
            raise SystemExit(f"replicate mismatch for {r['description']}")
        out.append({"sample": r["description"], "gsm": r["geo_accession"],
                    "condition": cond, "replicate": rep, "treatment": trt})
    return pd.DataFrame(out).set_index("sample")


def load_counts(counts_path, gene_map_path, flt: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw counts (genes x samples) with symbols; Ensembl IDs collapsed to unique symbols.

    Duplicate symbols keep the Ensembl gene with the highest total count.
    """
    cnt = pd.read_csv(counts_path, sep="\t", index_col=0)
    cnt.index = cnt.index.str.replace(r"\.\d+$", "", regex=True)
    gm = pd.read_csv(gene_map_path, sep="\t").set_index("ensembl_gene_id")
    ann = gm.reindex(cnt.index)
    keep = ann["gene_name"].notna()
    if flt.get("protein_coding_only", True):
        keep &= ann["gene_biotype"].eq("protein_coding")
    cnt, ann = cnt[keep], ann[keep]
    expressed = (cnt >= flt["min_count"]).sum(axis=1) >= flt["min_samples"]
    cnt, ann = cnt[expressed], ann[expressed]
    tot = cnt.sum(axis=1)
    order = tot.sort_values(ascending=False).index
    ann_o = ann.loc[order]
    first = ~ann_o["gene_name"].duplicated()
    ids = ann_o.index[first]
    out = cnt.loc[ids].copy()
    sym = ann.loc[ids, "gene_name"]
    gene_tab = pd.DataFrame({"ensembl_gene_id": ids, "gene": sym.values}).set_index("gene")
    out.index = sym.values
    out.index.name = "gene"
    return out.astype(int), gene_tab


def log_cpm(counts: pd.DataFrame, pseudo: float = 1.0) -> pd.DataFrame:
    """Per-sample log2(CPM + pseudo); sample-independent, so held-out libraries are not
    normalised with training libraries."""
    lib = counts.sum(axis=0)
    return np.log2(counts.div(lib, axis=1) * 1e6 + pseudo)


def fit_deseq(counts: pd.DataFrame, meta: pd.DataFrame, ref: str, n_cpus: int = 8):
    """Fit ~condition on the given libraries (genes x samples input)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference

    md = meta.loc[counts.columns, ["condition"]].copy()
    levels = [ref] + sorted(c for c in md["condition"].unique() if c != ref)
    md["condition"] = pd.Categorical(md["condition"], categories=levels)
    inf = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts.T, metadata=md, design="~condition",
                           refit_cooks=True,
                           inference=inf, quiet=True)
        dds.deseq2()
    return dds


def contrast(dds, num: str, den: str, alpha: float, n_cpus: int = 8) -> pd.DataFrame:
    """Wald test num vs den; returns baseMean, log2FC, lfcSE, stat, pvalue, padj."""
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = DeseqStats(dds, contrast=["condition", num, den], alpha=alpha,
                        inference=DefaultInference(n_cpus=n_cpus), quiet=True)
        st.summary()
    res = st.results_df.rename(columns={"log2FoldChange": "log2FC"}).copy()
    res.index.name = "gene"
    # genes removed by independent filtering keep padj = NaN -> treat as not significant
    return res[["baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj"]]


def scaling_slope(ds: pd.DataFrame, lps: pd.DataFrame, alpha: float, min_lfc: float) -> float:
    """Least-squares slope (through the origin) of dsHMGB1 log2FC on LPS log2FC over genes
    responsive to either stimulus: how much of the LPS response dsHMGB1 reproduces when the
    two are treated as one programme of different magnitude."""
    u = responsive(ds, alpha, min_lfc).union(responsive(lps, alpha, min_lfc))
    x, y = lps.loc[u, "log2FC"].values, ds.loc[u, "log2FC"].values
    return float((x * y).sum() / (x * x).sum())


def scaled_contrast(dds, slope: float, alpha: float, n_cpus: int = 8) -> pd.DataFrame:
    """Wald test of (dsHMGB1 - PBS) - slope * (LPS - PBS): the part of the dsHMGB1 response
    that a magnitude-scaled LPS response does not explain."""
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    cols = list(dds.obsm["design_matrix"].columns)
    c = np.zeros(len(cols))
    c[cols.index("condition[T.dsHMGB1]")] = 1.0
    c[cols.index("condition[T.LPS]")] = -slope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = DeseqStats(dds, contrast=c, alpha=alpha,
                        inference=DefaultInference(n_cpus=n_cpus), quiet=True)
        st.summary()
    res = st.results_df.rename(columns={"log2FoldChange": "log2FC"}).copy()
    res.index.name = "gene"
    return res[["baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj"]]


def all_contrasts(counts, meta, ref, stimuli, alpha, min_lfc, n_cpus=8):
    """Every stimulus present vs reference, the direct dsHMGB1 vs LPS contrast and the
    magnitude-adjusted dsHMGB1 vs scaled-LPS contrast. Returns (results dict, slope)."""
    dds = fit_deseq(counts, meta, ref, n_cpus)
    present = set(meta.loc[counts.columns, "condition"])
    res = {s: contrast(dds, s, ref, alpha, n_cpus) for s in stimuli if s in present}
    res["dsHMGB1_vs_LPS"] = contrast(dds, "dsHMGB1", "LPS", alpha, n_cpus)
    slope = scaling_slope(res["dsHMGB1"], res["LPS"], alpha, min_lfc)
    res["dsHMGB1_vs_scaledLPS"] = scaled_contrast(dds, slope, alpha, n_cpus)
    return res, slope


def responsive(df: pd.DataFrame, alpha: float, min_lfc: float, direction: str = "any"):
    sig = df["padj"].fillna(1.0) < alpha
    if direction == "up":
        return df.index[sig & (df["log2FC"] >= min_lfc)]
    if direction == "down":
        return df.index[sig & (df["log2FC"] <= -min_lfc)]
    return df.index[sig & (df["log2FC"].abs() >= min_lfc)]
