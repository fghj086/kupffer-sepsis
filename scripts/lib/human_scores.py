"""Stage 07 sample scores.

Per gene, log2 expression is centred on the mean of the dataset's reference samples and scaled by
the pooled within-group standard deviation (the stage 02 convention), so that a reference group of
one or two samples still gives a usable scale. A signature score is the mean z of its genes; for a
signed set (genes with weight +1 and -1) it is mean z(+1 genes) - mean z(-1 genes), which for the
HMGB1-vs-LPS axis is mean z(HMGB1-biased) - mean z(LPS-biased).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardise(x: pd.DataFrame, groups: pd.Series, ref_mask: pd.Series) -> pd.DataFrame:
    groups = groups.reindex(x.columns)
    ref_mask = ref_mask.reindex(x.columns).fillna(False).astype(bool)
    if ref_mask.sum() == 0:
        ref_mask[:] = True
    centre = x.loc[:, ref_mask.values].mean(axis=1)
    ss = pd.Series(0.0, index=x.index)
    df = 0
    for _, cols in groups.groupby(groups).groups.items():
        sub = x[list(cols)]
        if sub.shape[1] > 1:
            ss += ((sub.sub(sub.mean(axis=1), axis=0)) ** 2).sum(axis=1)
            df += sub.shape[1] - 1
    if df == 0:
        sd = x.std(axis=1, ddof=1)
    else:
        sd = np.sqrt(ss / df)
    sd = sd.where(sd > 1e-6)
    z = x.sub(centre, axis=0).div(sd, axis=0)
    return z.dropna(how="all")


def score(z: pd.DataFrame, sets: pd.DataFrame, names: list, min_genes: int,
          marker_sets: tuple = (), min_markers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, ng = {}, []
    present = set(z.index)
    for name in names:
        s = sets[(sets.signature == name) & sets.gene.notna()]
        s = s[s.gene.isin(present)]
        pos = s[s.weight > 0].gene.tolist()
        neg = s[s.weight < 0].gene.tolist()
        n_tot = int(sets[(sets.signature == name)].shape[0])
        ng.append({"score": name, "n_genes_used": len(pos) + len(neg), "n_up": len(pos),
                   "n_down": len(neg), "n_genes_in_set": n_tot})
        if len(pos) + len(neg) < (min_markers if name in marker_sets else min_genes):
            out[name] = pd.Series(np.nan, index=z.columns)
            continue
        if pos and neg:
            out[name] = z.loc[pos].mean(axis=0) - z.loc[neg].mean(axis=0)
        elif pos:
            out[name] = z.loc[pos].mean(axis=0)
        else:
            out[name] = -z.loc[neg].mean(axis=0)
    return pd.DataFrame(out), pd.DataFrame(ng)


def tidy(scores: pd.DataFrame, design: pd.DataFrame, dataset: str, ngenes: pd.DataFrame) -> pd.DataFrame:
    long = scores.rename_axis("sample").reset_index().melt(id_vars="sample", var_name="score",
                                                          value_name="value")
    keep = [c for c in ["sample", "subject", "group", "day", "tissue", "storage", "state", "stim",
                        "died"] if c in design.columns]
    long = long.merge(design[keep], on="sample", how="left")
    long = long.merge(ngenes[["score", "n_genes_used"]], on="score", how="left")
    long.insert(0, "dataset", dataset)
    return long
