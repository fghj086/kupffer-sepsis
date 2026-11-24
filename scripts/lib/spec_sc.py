"""Stage 10, part 2: KC vs recruited myeloid gap in the pooled GSE279167 libraries (sham pool vs
CLP pool), and its change with sepsis.

Scores per condition x population: pseudobulk (summed raw counts) log2 CPM, genes centred across
the six condition x population pseudobulks, mean over the set (as stage 03 pb_program_scores,
with the expressed-gene filter fixed on the full data so bootstrap replicates use the same genes);
stimulus signatures as the mean per-cell ULM t (as stage 03). The cell bootstrap resamples cells
within each condition x population; it describes cell sampling only. Each condition is one pooled
library (sham 3 mice, CLP 4 mice), so there is no biological replication and no P value.
Cell-level adjustment for the shared inflammatory score uses sc_beyond.cell_did_beyond_shared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scripts.lib import sc_beyond as B

CONDS = ("sham", "CLP")


def assign_groups(obs: pd.DataFrame, kc_pops, comparators: dict) -> pd.Series:
    pop = obs["population"].astype(str)
    g = pd.Series(np.nan, index=obs.index, dtype=object)
    g[pop.isin(kc_pops)] = "KC"
    for name, pops in comparators.items():
        g[pop.isin(pops)] = name
    return g


class PseudobulkScorer:
    """Fixed groups (condition x population), fixed expressed genes, resamplable cell weights."""

    def __init__(self, ad, groups: pd.Series, sets: dict, sig: pd.DataFrame):
        keep = groups.notna().values
        self.X = ad.layers["counts"][keep]
        self.X = self.X.tocsr() if sp.issparse(self.X) else sp.csr_matrix(self.X)
        self.sig = sig.loc[keep].values
        self.sig_names = list(sig.columns)
        o = ad.obs.loc[keep]
        self.keys = list(zip(o["condition"].astype(str), groups[keep]))
        self.labels = sorted(set(self.keys))
        self.cells = {k: np.where([x == k for x in self.keys])[0] for k in self.labels}
        self.n_cells = {k: len(v) for k, v in self.cells.items()}
        self.var = pd.Index(ad.var_names)
        lc = self._logcpm(self._weights(None))
        expressed = (lc > 1).mean(1) >= 0.25
        self.sets = {k: np.where(self.var.isin(v) & expressed)[0] for k, v in sets.items()}
        self.n_genes = {k: len(v) for k, v in self.sets.items()}

    def _weights(self, rng):
        rows, cols, vals = [], [], []
        for i, k in enumerate(self.labels):
            idx = self.cells[k]
            if rng is None:
                w = np.ones(len(idx))
            else:
                w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
            rows += [i] * len(idx)
            cols += list(idx)
            vals += list(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(self.labels), self.X.shape[0]))

    def _logcpm(self, W):
        S = np.asarray((W @ self.X).todense())
        return np.log2(S / S.sum(1, keepdims=True) * 1e6 + 1).T  # genes x groups

    def scores(self, rng=None) -> pd.DataFrame:
        W = self._weights(rng)
        lc = self._logcpm(W)
        out = {}
        for k, gi in self.sets.items():
            c = lc[gi] - lc[gi].mean(1, keepdims=True)
            out[k] = c.mean(0)
        wsum = np.asarray(W.sum(1)).ravel()
        Sg = (W @ self.sig) / wsum[:, None]
        for j, n in enumerate(self.sig_names):
            out[n] = Sg[:, j]
        return pd.DataFrame(out, index=pd.MultiIndex.from_tuples(self.labels, names=["condition", "group"]))


def gaps(S: pd.DataFrame, comparators) -> pd.DataFrame:
    rows = []
    for comp in comparators:
        for sc in S.columns:
            g = {c: S.loc[(c, "KC"), sc] - S.loc[(c, comp), sc] for c in CONDS}
            rows.append({"comparator": comp, "score": sc, "gap_sham": g["sham"], "gap_clp": g["CLP"],
                         "change_clp_minus_sham": g["CLP"] - g["sham"],
                         "kc_sham": S.loc[("sham", "KC"), sc], "kc_clp": S.loc[("CLP", "KC"), sc],
                         "comp_sham": S.loc[("sham", comp), sc], "comp_clp": S.loc[("CLP", comp), sc]})
    return pd.DataFrame(rows)


def gap_table(scorer: PseudobulkScorer, comparators, n_boot, rng) -> pd.DataFrame:
    est = gaps(scorer.scores(), comparators)
    boots = [gaps(scorer.scores(rng), comparators) for _ in range(n_boot)]
    cols = ["gap_sham", "gap_clp", "change_clp_minus_sham"]
    arr = np.stack([b[cols].values for b in boots])
    for j, c in enumerate(cols):
        est[f"{c}_cellboot_ci_low"] = np.quantile(arr[:, :, j], 0.025, axis=0)
        est[f"{c}_cellboot_ci_high"] = np.quantile(arr[:, :, j], 0.975, axis=0)
    est["direction"] = np.where(np.sign(est.gap_clp) != np.sign(est.gap_sham), "gap reverses sign",
                                np.where(est.gap_clp.abs() > est.gap_sham.abs(), "gap widens", "gap narrows"))
    est["n_genes"] = est.score.map(scorer.n_genes)
    for c in CONDS:
        est[f"n_cells_kc_{c}"] = scorer.n_cells.get((c, "KC"))
        est[f"n_cells_comp_{c}"] = [scorer.n_cells.get((c, k)) for k in est.comparator]
    est["ci_meaning"] = "cell bootstrap: cell sampling only, not biological replication"
    return est


def adjusted_did(obs: pd.DataFrame, groups: pd.Series, comparators, programs, shared,
                 n_boot, rng) -> pd.DataFrame:
    """Per cell: z(program) ~ KC * CLP + z(shared) (stage 03 cell_did_beyond_shared), for each
    comparator; interaction = change of the KC-minus-comparator gap with sepsis, unadjusted and at
    equal shared inflammatory score (SD units of the program across the cells used)."""
    rows = []
    for comp in comparators:
        m = groups.isin(["KC", comp]).values
        o = obs.loc[m].copy()
        o["population"] = np.where(groups[m] == "KC", B.KC, B.REC)
        est = B.cell_did_beyond_shared(o, programs, shared, "sham", "CLP")
        strata = [np.where(((o.condition == c) & (o.population == p)).values)[0]
                  for c in CONDS for p in (B.KC, B.REC)]
        bt = []
        for _ in range(n_boot):
            idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
            ob = o.iloc[idx].reset_index(drop=True)
            bt.append(B.cell_did_beyond_shared(ob, programs, shared, "sham", "CLP")
                      [["beta_kc_unadjusted", "beta_kc_adjusted"]].values)
        bt = np.stack(bt)
        for j, c in enumerate(["beta_kc_unadjusted", "beta_kc_adjusted"]):
            est[f"{c}_cellboot_ci_low"] = np.quantile(bt[:, :, j], 0.025, axis=0)
            est[f"{c}_cellboot_ci_high"] = np.quantile(bt[:, :, j], 0.975, axis=0)
        est["comparator"] = comp
        rows.append(est)
    out = pd.concat(rows, ignore_index=True)
    out["ci_meaning"] = "cell bootstrap: cell sampling only, not biological replication"
    return out
