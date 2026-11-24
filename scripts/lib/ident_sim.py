"""Stage 09 A: detection limit of an HMGB1-like component in an LPS-like response.

Simulated response  y = (1 - f) * LPS_true + f * dsHMGB1_k + noise,  with both components scaled
to unit root-mean-square on the scale genes (informative genes of the macrophage_resolution panel),
which is the scaling the stage 06 fits use.

* LPS_true comes from another laboratory (KDO2-lipid A BMDM, LPS-stimulated cultured Kupffer
  cells, LPS in RAW264.7, LPS 24 h BMDM of a second study) or, as the optimistic case, from a
  GSE200210 LPS replicate.
* dsHMGB1_k is GSE200210 disulfide-HMGB1 replicate k against PBS replicate k. The panel's
  GSE200210 references (LPS_24h, dsHMGB1) are rebuilt from the other two replicates of every arm,
  so the simulated HMGB1 component shares no library with the panel.
* noise: real mouse-to-mouse deviations of sorted Kupffer cells (lib.ident_noise), either at the
  sampling error of the observed 4-vs-4 log2FC ("mouse") or scaled so the noise equals the residual
  left by the panel in the real Kupffer-cell fit ("fit residual").
The source reference itself is removed from the panel when it is a panel member (KDO2-lipid A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.attr_model import fit_nnls, unit_rms
from lib.attr_panel import informative


def gse200210_folds(rs) -> list[dict]:
    st = rs.studies["GSE200210"]
    L, cols = st.L, st.cols
    pbs, ds, lps = cols("PBS"), cols("dsHMGB1"), cols("LPS")
    out = []
    for k in range(min(len(pbs), len(ds), len(lps))):
        rest = lambda c: [x for i, x in enumerate(c) if i != k]
        out.append({"k": k,
                    "ds_comp": L[ds[k]] - L[pbs[k]],
                    "lps_comp": L[lps[k]] - L[pbs[k]],
                    "dsHMGB1": L[rest(ds)].mean(1) - L[rest(pbs)].mean(1),
                    "LPS_24h": L[rest(lps)].mean(1) - L[rest(pbs)].mean(1)})
    return out


def axis_scores(Y: np.ndarray, iH, iL, iS) -> np.ndarray:
    """HMGB1-vs-LPS axis of stage 01 on log2FC vectors: mean over HMGB1-biased genes minus mean
    over LPS-biased genes, divided by the root-mean-square over the scale genes (so the score
    does not follow response size). Y: draws x genes."""
    Y = np.atleast_2d(Y)
    rms = np.sqrt(np.mean(Y[:, iS] ** 2, axis=1))
    return (Y[:, iH].mean(1) - Y[:, iL].mean(1)) / np.where(rms > 0, rms, 1.0)


class MixtureSim:
    def __init__(self, rs, cfg06, cfg, noise, signatures: pd.DataFrame):
        self.rs, self.cfg = rs, cfg
        self.folds = gse200210_folds(rs)
        a = cfg["mixture"]
        self.sources = a["lps_sources"]
        H = set(signatures.gene[signatures.signature == "HMGB1_biased"])
        Lb = set(signatures.gene[signatures.signature == "LPS_biased"])
        self.panels = {p: list(cfg06["panels"][p.replace("_unmerged", "")]) for p in a["panels"]}
        pan_genes = set().union(*[set(informative(rs, v)) for v in self.panels.values()])
        noise_genes = set.intersection(*[set(n.genes) for n in noise.values()])
        vecs = [rs.vec[s] for s in self.sources if s != "within_GSE200210"]
        vecs += [f[x] for f in self.folds for x in ("ds_comp", "lps_comp", "dsHMGB1", "LPS_24h")]
        ok = pd.concat(vecs, axis=1).notna().all(axis=1)
        avail = set(ok.index[ok.values]) & noise_genes
        self.U = pd.Index(sorted((pan_genes | H | Lb) & avail))
        scale = set(informative(rs, cfg06["panels"]["macrophage_resolution"])) & set(self.U)
        self.S0 = pd.Index(sorted(scale))
        pos = {g: i for i, g in enumerate(self.U)}
        self.iH = np.array([pos[g] for g in sorted(H & set(self.U))])
        self.iL = np.array([pos[g] for g in sorted(Lb & set(self.U))])
        self.iS = np.array([pos[g] for g in self.S0])
        self.noise = noise
        self._R = {}

    # -- components ------------------------------------------------------------------------
    def _unit(self, v: pd.Series) -> np.ndarray:
        x = v.reindex(self.U).values.astype(float)
        return x / np.sqrt(np.mean(x[self.iS] ** 2))

    def lps_true(self, source: str, k: int) -> np.ndarray:
        if source == "within_GSE200210":
            return self._unit(self.folds[k]["lps_comp"])
        return self._unit(self.rs.vec[source])

    def panel_refs(self, panel: str, source: str, k: int):
        """Unit-RMS panel matrix on the panel's informative genes (within U), fold-k rebuilt
        GSE200210 references, source reference removed. Returns (gene positions, names, R)."""
        key = (panel, source, k)
        if key not in self._R:
            names = [r for r in self.panels[panel] if r != source]
            g = informative(self.rs, names).intersection(self.U)
            M = self.rs.matrix(names, g).copy()
            for ref in ("LPS_24h", "dsHMGB1"):
                if ref in names:
                    M[ref] = self.folds[k][ref].reindex(g)
            pos = self.U.get_indexer(g)
            self._R[key] = (pos, names, unit_rms(M.values.astype(float)))
        return self._R[key]

    # -- noise -----------------------------------------------------------------------------
    def draw_noise(self, spec: dict, n: int, rng, source=None) -> np.ndarray:
        """kind none | mouse | resid; for resid, level is the noise root-mean-square relative to
        the unit signal, either one number or a dict per LPS source (residual-matched)."""
        if spec["kind"] == "none":
            return np.zeros((n, len(self.U)))
        nm = self.noise[spec["pattern"]]
        e = nm.draw(self.U, rng, n)
        if spec["kind"] == "mouse":
            return e * nm.group_mean_scale(self.S0)
        rms = np.sqrt(np.mean(e[:, self.iS] ** 2, axis=1, keepdims=True))
        lv = spec["level"][source] if isinstance(spec["level"], dict) else spec["level"]
        return e / rms * lv

    def simulate(self, source: str, spec: dict, f_grid, n: int, rng):
        """Yields (f, ks, Y) with Y draws x U; the same noise draws are reused across f so that
        power curves are paired."""
        ks = np.arange(n) % len(self.folds)
        E = self.draw_noise(spec, n, rng, source)
        L = np.stack([self.lps_true(source, k) for k in ks])
        D = np.stack([self._unit(self.folds[k]["ds_comp"]) for k in ks])
        for f in f_grid:
            yield f, ks, (1 - f) * L + f * D + E

    def fit_draws(self, panel, source, ks, Y) -> tuple[np.ndarray, np.ndarray]:
        """dsHMGB1 share and R2 for every draw."""
        share, r2s = np.full(len(Y), np.nan), np.full(len(Y), np.nan)
        for i, (k, y) in enumerate(zip(ks, Y)):
            pos, names, R = self.panel_refs(panel, source, k)
            b, r2 = fit_nnls(unit_rms(y[pos]), R)
            j = names.index("dsHMGB1")
            share[i] = b[j] / b.sum() if b.sum() > 0 else 0.0
            r2s[i] = r2
        return share, r2s

    def misfit(self, source: str, panel: str) -> float:
        """Residual fraction (1 - R2) of a pure, noise-free LPS-like source fitted by the panel,
        averaged over the GSE200210 folds: the cross-laboratory mismatch alone."""
        ks = np.arange(len(self.folds))
        L = np.stack([self.lps_true(source, k) for k in ks])
        _, r2 = self.fit_draws(panel, source, ks, L)
        return float(1 - r2.mean())

    def matched_level(self, r2_obs: float, panel: str) -> dict:
        """Noise level per source so that mismatch + noise leaves the observed residual fraction:
        r = (m + s^2) / (1 + s^2)  ->  s^2 = (r - m) / (1 - r), floored at 0."""
        r = 1 - r2_obs
        return {src: float(np.sqrt(max(r - self.misfit(src, panel), 0.0) / (1 - r)))
                for src in self.sources}

    # -- observed targets ------------------------------------------------------------------
    def observed(self, y: pd.Series, panel: str) -> dict:
        """Fit a real target with the full-data panel on the same genes as the simulation."""
        names = self.panels[panel]
        g = informative(self.rs, names, y).intersection(self.U)
        R = unit_rms(self.rs.matrix(names, g).values.astype(float))
        b, r2 = fit_nnls(unit_rms(y.reindex(g).values.astype(float)), R)
        yy = y.reindex(self.U).values.astype(float)
        ok = ~np.isnan(yy)
        iH, iL, iS = (np.intersect1d(ix, np.where(ok)[0]) for ix in (self.iH, self.iL, self.iS))
        ax = float(axis_scores(np.nan_to_num(yy)[None, :], iH, iL, iS)[0])
        return {"hmgb1_share": float(b[names.index("dsHMGB1")] / b.sum()) if b.sum() > 0 else 0.0,
                "r2": r2, "axis": ax, "n_genes_fit": len(g), "n_axis_hmgb1_genes": len(iH),
                "n_axis_lps_genes": len(iL)}
