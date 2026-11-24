"""Stage 08: per-mouse scores for GSE224127 liver (macrophage programs, inflammation, injury).

Scores follow stage 04 exactly (file-by-file log2 normalised CPM, centred on the file's reference
group, set genes restricted to those expressed in all three time-course files). Added here:
hepatocyte function loss (Cyp/negative acute-phase genes) and resolution-like programs scored
with signed weights from the stage 06 macrophage references.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib.tc_design import REFERENCE
from lib.tc_io import build_design
from lib.tc_scores import expressed_genes, log_cpm, score_sets

TIMECOURSE_FILES = ("LPS_timecourse", "CLP_d0.5", "CLP_d0.25_d1")


def core_genes(counts: dict, genes, min_cpm: float) -> list[str]:
    keep = set(genes)
    for fk in TIMECOURSE_FILES:
        keep &= set(expressed_genes(counts[fk], genes, min_cpm))
    present = set.intersection(*[set(X.index) for X in counts.values()])
    return [g for g in genes if g in keep and g in present]


def load_refset(cache_dir: Path):
    ps = sorted(cache_dir.glob("refset_*.pkl"))
    if len(ps) != 1:
        raise SystemExit(f"expected one stage 06 reference cache in {cache_dir}, found {len(ps)}")
    with open(ps[0], "rb") as fh:
        return pickle.load(fh), ps[0]


def resolution_weights(rs, cfg: dict, exclude: set[str]) -> tuple[dict, list[dict]]:
    """Signed weight vectors for the resolution-like programs; genes in `exclude` dropped."""
    out, rows = {}, []
    for name, spec in cfg["resolution_references"].items():
        a = spec["ref"]
        if "minus" in spec:
            b = spec["minus"]
            genes = pd.Index(sorted(set(rs.genes[a]) | set(rs.genes[b])))
            w = (rs.vec[a].reindex(genes) - rs.vec[b].reindex(genes)).dropna()
            w = w[w.abs() >= cfg["min_abs_weight"]]
        else:
            w = rs.vec[a].reindex(rs.genes[a]).dropna()
        n0 = len(w)
        w = w[~w.index.isin(exclude)]
        out[name] = w
        rows.append({"program": name, "reference": a, "minus": spec.get("minus", ""),
                     "n_responsive": n0, "n_after_exclusion": len(w),
                     "n_up": int((w > 0).sum()), "n_down": int((w < 0).sum())})
    return out, rows


def compute_scores(cfg: dict, cfg04: dict, root: Path, rs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide per-mouse scores (index = column name) + design + genes used."""
    design, counts = build_design(cfg04, root)
    design["reference_group"] = design.file_key.map(REFERENCE)
    design = design[design.author_qc_flag == 0]
    ex = cfg04["expression"]
    gs = cfg04["gene_sets"]
    sets = {k: core_genes(counts, v, ex["min_mean_cpm"]) for k, v in gs.items()}
    sets["hepatocyte_function"] = core_genes(counts, cfg["hepatocyte_function_genes"], ex["min_mean_cpm"])
    exclude = set().union(*[set(v) for v in gs.values()]) | set(cfg["hepatocyte_function_genes"])
    W, wrows = resolution_weights(rs, cfg, exclude)
    wsets = {}
    for k, w in W.items():
        g = core_genes(counts, list(w.index), ex["min_mean_cpm"])
        if len(g) < cfg["min_genes"]:
            raise SystemExit(f"{k}: only {len(g)} expressed genes")
        wsets[k] = w.reindex(g)
    frames, used = [], []
    for fk, X in counts.items():
        d = design[design.file_key == fk]
        L = log_cpm(X, ex["pseudocount"])[d.column.tolist()]
        ref = d.loc[d.group == REFERENCE[fk], "column"].tolist()
        S, _ = score_sets(L, ref, sets)
        S2, _ = score_sets(L, ref, {k: list(w.index) for k, w in wsets.items()}, wsets)
        S = pd.concat([S, S2], axis=1)
        S["hepatocyte_function_loss"] = -S.pop("hepatocyte_function")
        frames.append(S)
    for k, g in {**sets, **{k: list(w.index) for k, w in wsets.items()}}.items():
        used.append({"set": k, "n_genes": len(g), "genes": ",".join(g)})
    for r in wrows:
        r["n_expressed_used"] = len(wsets[r["program"]])
    gtab = pd.DataFrame(used).merge(pd.DataFrame(wrows).rename(columns={"program": "set"}),
                                    on="set", how="left")
    return pd.concat(frames), design.set_index("column"), gtab


def check_against_stage04(S: pd.DataFrame, scores04: pd.DataFrame, names) -> pd.DataFrame:
    w = scores04.pivot_table(index="column", columns="score", values="value")
    rows = []
    for n in names:
        a, b = S[n].align(w[n], join="inner")
        rows.append({"score": n, "n": len(a), "max_abs_diff": float((a - b).abs().max())})
    return pd.DataFrame(rows)


def mouse_table(S: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    keep = ["file_key", "model", "group", "genotype", "time_d", "mouse"]
    return design[keep].join(S, how="inner").reset_index().rename(columns={"index": "column"})
