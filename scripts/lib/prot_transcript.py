"""Transcript-level comparators for stage 05 and protein-transcript concordance."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.lib.prot_stats import moderated_t


def clp_liver_transcripts(spec: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole-liver CLP vs sham log2 CPM moderated t from one GSE224127 count file.

    Uses the sample annotation written by stage 04 (results/04_timecourse/design.tsv).
    Returns (de_table, design_used).
    """
    des = pd.read_csv(root / spec["design"], sep="\t")
    des = des[(des["file_key"] == spec["file_key"]) & (des["time_d"] == spec["time_d"])
              & des["column"].str.startswith(spec["tissue_prefix"])]
    des = des[des["group"].isin(spec["test_groups"] + [spec["reference_group"]])].copy()
    des["arm"] = np.where(des["group"] == spec["reference_group"], "reference", "test")
    counts = pd.read_csv(root / spec["counts"], sep="\t", index_col=0)
    counts = counts.loc[:, des["column"]]
    counts = counts.groupby(level=0).sum()
    cpm = counts / counts.sum(0) * 1e6
    cpm = cpm[cpm.mean(1) >= spec["min_mean_cpm"]]
    x = np.log2(cpm + 1.0)
    g = pd.Series(des["arm"].to_numpy(), index=des["column"].to_numpy())
    de = moderated_t(x, g, "test", "reference", min_valid=2)
    de = de.rename(columns={"log2FC": "lfc", "t_mod": "stat"})
    return de[["lfc", "stat", "pvalue", "padj"]], des[["column", "group", "time_d", "arm"]]


def load_transcript_table(spec: dict, root: Path) -> tuple[pd.DataFrame | None, str]:
    """Return (table indexed by gene with columns lfc, pvalue, stat) and a status string."""
    if "glob" in spec:
        hits = sorted(glob.glob(str(root / spec["glob"])))
        if not hits:
            return None, "pending: no table matching " + spec["glob"]
        df = pd.read_csv(hits[0], sep="\t")
        gcol = next(c for c in ("gene", "Gene", "symbol", "names") if c in df.columns)
        lcol = next(c for c in ("log2FoldChange", "log2FC", "lfc", "logfoldchanges",
                                "effect") if c in df.columns)
        pcol = next((c for c in ("pvalue", "pvals", "p") if c in df.columns), None)
        out = pd.DataFrame({"lfc": df[lcol].to_numpy(),
                            "pvalue": df[pcol].to_numpy() if pcol else np.nan},
                           index=df[gcol].astype(str).to_numpy())
        return out[~out.index.duplicated()], "ok: " + Path(hits[0]).name
    p = root / spec["table"]
    if not p.exists():
        return None, "pending: " + spec["table"] + " not present"
    df = pd.read_csv(p, sep="\t")
    out = pd.DataFrame({"lfc": df[spec["lfc"]].to_numpy(), "pvalue": df[spec["p"]].to_numpy()},
                       index=df[spec["gene"]].astype(str).to_numpy())
    out = out.dropna(subset=["lfc"])
    return out[~out.index.duplicated()], "ok"


def concordance(prot: pd.DataFrame, tx: pd.DataFrame, fdr: float) -> dict:
    """Direction and size agreement between protein log2FC and transcript log2FC."""
    common = prot.index[prot["tested"]].intersection(tx.index)
    pr, tr = prot.loc[common, "log2FC"], tx.loc[common, "lfc"]
    ok = pr.notna() & tr.notna()
    pr, tr = pr[ok], tr[ok]
    res = {"n_genes_shared": int(len(pr))}
    if len(pr) < 10:
        return res
    rho, p = stats.spearmanr(pr, tr)
    res.update({"spearman_rho": rho, "spearman_p": p})
    sig = prot.loc[pr.index, "padj"] < fdr
    n_sig = int(sig.sum())
    res["n_protein_fdr_sig"] = n_sig
    if n_sig >= 5:
        agree = int((np.sign(pr[sig]) == np.sign(tr[sig])).sum())
        res.update({"sign_agree_sig": agree, "frac_sign_agree_sig": agree / n_sig,
                    "binom_p_sig": stats.binomtest(agree, n_sig, 0.5).pvalue})
        rs, ps = stats.spearmanr(pr[sig], tr[sig])
        res.update({"spearman_rho_sig": rs, "spearman_p_sig": ps})
    return res
