"""Stage 01 helpers: stimulus-response signatures and sample scoring.

Signatures (all up-regulated programmes, mouse symbols):
  shared_inflammatory  up vs PBS in both dsHMGB1 and LPS, |dsHMGB1 - LPS| < threshold
  HMGB1_biased         up vs PBS in dsHMGB1, higher than LPS (direct contrast) and
                       higher than a magnitude-scaled LPS response
  LPS_biased           up vs PBS in LPS, higher than dsHMGB1 (direct contrast) and
                       higher than the scaled expectation (ds - slope*LPS < 0)
  HMGB1_vs_LPS_axis    HMGB1_biased (+1) and LPS_biased (-1): a log-ratio score
                       (mean z of HMGB1-biased genes minus mean z of LPS-biased genes)
Requiring the magnitude-adjusted contrast keeps genes that are merely more induced by the
stronger stimulus out of the biased sets, so the axis is not a proxy for response size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .perturb_de import responsive

SIG_NAMES = ["shared_inflammatory", "HMGB1_biased", "LPS_biased"]
AXIS = "HMGB1_vs_LPS_axis"


def _sig(df, alpha):
    return df["padj"].fillna(1.0) < alpha


def build_signatures(res: dict, de_cfg: dict, sig_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return a long table (signature, gene, weight, rank, rank_stat) and candidate counts."""
    a, m = de_cfg["alpha"], de_cfg["min_abs_lfc"]
    ds, lp = res["dsHMGB1"], res["LPS"]
    dvl, adj = res["dsHMGB1_vs_LPS"], res["dsHMGB1_vs_scaledLPS"]
    up_ds, up_lp = responsive(ds, a, m, "up"), responsive(lp, a, m, "up")
    ca, cl = sig_cfg["biased_padj_contrast"], sig_cfg["biased_min_lfc_contrast"]

    shared = up_ds.intersection(up_lp)
    shared = shared[dvl.loc[shared, "log2FC"].abs() < sig_cfg["shared_max_abs_lfc_ds_vs_lps"]]
    shared_rank = np.minimum(ds.loc[shared, "log2FC"], lp.loc[shared, "log2FC"])

    h = up_ds[_sig(dvl.loc[up_ds], ca) & (dvl.loc[up_ds, "log2FC"] >= cl)]
    h = h[_sig(adj.loc[h], ca) & (adj.loc[h, "log2FC"] >= cl)]
    h_rank = dvl.loc[h, "stat"]

    l_ = up_lp[_sig(dvl.loc[up_lp], ca) & (dvl.loc[up_lp, "log2FC"] <= -cl)]
    l_ = l_[_sig(adj.loc[l_], ca) & (adj.loc[l_, "log2FC"] <= -cl)]
    l_rank = -adj.loc[l_, "stat"]

    n = sig_cfg["max_size"]
    rows, counts = [], {}
    for name, rank in [("shared_inflammatory", shared_rank), ("HMGB1_biased", h_rank),
                       ("LPS_biased", l_rank)]:
        counts[name] = int(len(rank))
        top = rank.sort_values(ascending=False).iloc[:n]
        for i, (g, v) in enumerate(top.items(), 1):
            rows.append({"signature": name, "gene": g, "weight": 1.0, "rank": i,
                         "rank_stat": float(v)})
    tab = pd.DataFrame(rows)
    axis = tab[tab.signature.isin(["HMGB1_biased", "LPS_biased"])].copy()
    axis["weight"] = np.where(axis.signature == "HMGB1_biased", 1.0, -1.0)
    axis["signature"] = AXIS
    return pd.concat([tab, axis], ignore_index=True), counts


def zscore_ref(logexpr: pd.DataFrame, ref_cols) -> pd.DataFrame:
    """z-score each gene using mean and SD of the reference libraries only."""
    ref = logexpr[list(ref_cols)]
    mu, sd = ref.mean(axis=1), ref.std(axis=1, ddof=1)
    sd = sd.where(sd > 1e-8, np.nan)
    return logexpr.sub(mu, axis=0).div(sd, axis=0)


def score(z: pd.DataFrame, sigs: pd.DataFrame) -> pd.DataFrame:
    """Signed weighted mean z per signature. For the axis the + and - halves are averaged
    separately, so it equals mean z(HMGB1-biased) - mean z(LPS-biased) even when the
    two halves differ in size."""
    out = {}
    for name, g in sigs.groupby("signature", sort=False):
        g = g[g.gene.isin(z.index)]
        if name == AXIS:
            pos, neg = g[g.weight > 0].gene, g[g.weight < 0].gene
            out[name] = z.loc[pos].mean(axis=0) - z.loc[neg].mean(axis=0)
        else:
            w = g.set_index("gene")["weight"]
            out[name] = z.loc[w.index].mul(w, axis=0).sum(axis=0) / w.abs().sum()
    return pd.DataFrame(out)


def residualise(train: pd.DataFrame, test: pd.DataFrame, y: str, x: str):
    """OLS y ~ x fitted on training scores, applied to both sets. Returns (train_res,
    test_res, slope, intercept)."""
    b, a = np.polyfit(train[x].values, train[y].values, 1)
    return (train[y] - (a + b * train[x]), test[y] - (a + b * test[x]), float(b), float(a))
