"""Stage 09: mouse-to-mouse variation of sorted Kupffer cells, used as realistic noise.

For each sorted-KC study (GSE328774 CLP 2 h, GSE311736 CLP 8 h) the per-mouse expression is
log2(median-of-ratios normalised count + 1). Within each arm (CLP, sham) every mouse is expressed
as its deviation from the arm mean, rescaled by sqrt(n / (n - 1)) so that each deviation vector
has the per-gene variance of one mouse. A noise draw is a Gaussian combination of these real
deviation vectors, which keeps their gene-gene covariance (co-regulated modules, admixture,
dissociation responses) instead of assuming independent genes.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib.attr_ref import median_ratio_sf

log = logging.getLogger("09_identifiability")


def _kc_counts(cfg02: dict, cache: Path) -> dict:
    """{study: (counts, meta)} for the sorted-KC libraries; cached because the GTF parse is slow."""
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    from scripts.lib.sorted_io import gene_annotation, load_gse311736, load_gse328774
    genes, _ = gene_annotation(cfg02["data"]["gtf"])
    out = {}
    for name, loader in (("GSE328774", load_gse328774), ("GSE311736", load_gse311736)):
        c, m = loader(cfg02, genes)
        m = m[m.cell_type == "KC"]
        out[name] = (c[m.index], m)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


class MouseNoise:
    """Residual deviation vectors of individual mice and the observed CLP effect of one study."""

    def __init__(self, name, counts: pd.DataFrame, meta: pd.DataFrame, min_mean: float):
        keep = counts.mean(axis=1) >= min_mean
        C = counts[keep]
        L = np.log2(C / median_ratio_sf(C) + 1.0)
        self.name = name
        res, self.n_arm = [], {}
        for arm in ("CLP", "sham"):
            cols = list(meta.index[meta.condition == arm])
            n = len(cols)
            self.n_arm[arm] = n
            dev = L[cols].sub(L[cols].mean(axis=1), axis=0) * np.sqrt(n / (n - 1))
            res.append(dev)
        self.E = pd.concat(res, axis=1)                      # genes x mice
        clp, sham = meta.index[meta.condition == "CLP"], meta.index[meta.condition == "sham"]
        self.effect = L[clp].mean(axis=1) - L[sham].mean(axis=1)   # observed CLP log2FC
        # sampling error of a group-mean difference relative to one mouse
        self.mean_factor = float(np.sqrt(1 / self.n_arm["CLP"] + 1 / self.n_arm["sham"]))

    @property
    def genes(self) -> pd.Index:
        return self.E.index

    def draw(self, genes: pd.Index, rng: np.random.Generator, n: int) -> np.ndarray:
        """n draws (n x genes) of one-mouse noise with the empirical covariance."""
        E = self.E.reindex(genes).values
        z = rng.standard_normal((n, E.shape[1]))
        return z @ E.T / np.sqrt(E.shape[1])

    def group_mean_scale(self, scale_genes: pd.Index) -> float:
        """Multiplier that turns a one-mouse draw into the error of the observed 4-vs-4 log2FC,
        expressed relative to the root-mean-square of the observed CLP effect on scale_genes
        (simulated signals have unit root-mean-square on the same genes)."""
        rms_eff = float(np.sqrt(np.mean(self.effect.reindex(scale_genes).values ** 2)))
        return self.mean_factor / rms_eff

    def summary(self, scale_genes: pd.Index) -> dict:
        sd = self.E.reindex(scale_genes).std(axis=1, ddof=0)
        rms_eff = float(np.sqrt(np.mean(self.effect.reindex(scale_genes).values ** 2)))
        return {"study": self.name, "n_clp": self.n_arm["CLP"], "n_sham": self.n_arm["sham"],
                "n_genes_measured": len(self.genes), "n_scale_genes": len(scale_genes),
                "rms_clp_log2fc": rms_eff, "median_mouse_sd": float(sd.median()),
                "rms_group_mean_error": float(np.sqrt(np.mean(sd ** 2))) * self.mean_factor,
                "noise_to_signal_rms": float(np.sqrt(np.mean(sd ** 2))) * self.mean_factor / rms_eff}


def load_noise(cfg: dict, cfg02: dict, root: Path) -> dict[str, MouseNoise]:
    data = _kc_counts(cfg02, root / cfg["cache_dir"] / "kc_counts.pkl")
    out = {}
    for key, spec in cfg["noise"]["studies"].items():
        c, m = data[spec["accession"]]
        out[key] = MouseNoise(key, c, m, cfg["noise"]["min_mean_count"])
        log.info("noise %s: %d genes, %s", key, len(out[key].genes), out[key].n_arm)
    return out
