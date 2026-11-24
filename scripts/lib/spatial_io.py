"""GSE248904 liver-spot extraction for stage 04 spatial.

The deposited object (2.3 M spots x 32 k genes, 13.8 GB) carries the authors' organ
annotation ('Organ' == 'LI') and liver zonation subregions ('Pericentral', 'Periportal').
Liver spots and the genes needed here are cached once to data/processed/ so that the
analysis can be rerun without opening the full object.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

OBS_KEEP = ["Sample", "Treatment", "batch", "Organ", "Subregion", "x_plotting", "y_plotting",
            "total_counts", "n_genes_by_counts", "pct_counts_mt"]


def extract_liver(h5ad: Path, out: Path, genes: list[str], organ: str = "LI",
                  chunk: int = 20000) -> ad.AnnData:
    if out.exists():
        return ad.read_h5ad(out)
    a = ad.read_h5ad(h5ad, backed="r")
    obs = a.obs
    ix = np.where(obs["Organ"].astype(str).values == organ)[0]
    vn = pd.Index(a.var_names)
    keep = [g for g in dict.fromkeys(genes) if g in vn]
    gi = vn.get_indexer(keep)
    blocks = []
    for s in range(0, len(ix), chunk):
        blk = a.X[ix[s:s + chunk]]
        blocks.append(sp.csr_matrix(blk)[:, gi])
    X = sp.vstack(blocks).tocsr()
    o = obs.iloc[ix][OBS_KEEP].copy()
    for c in ("Sample", "Treatment", "batch", "Organ", "Subregion"):
        o[c] = o[c].astype(str)
    sub = ad.AnnData(X=X, obs=o, var=pd.DataFrame(index=keep))
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out, compression="gzip")
    a.file.close()
    return sub
