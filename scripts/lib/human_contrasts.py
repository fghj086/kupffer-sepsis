"""Stage 07 contrasts on sample scores (subject / donor as unit).

Each contrast row: dataset, family, contrast, score, design, n1, n0, estimate (difference in score,
z units), ci_low, ci_high, hedges_g (+ interval), p, test. Contrasts whose design cannot support a
test (one control, n = 2 per arm, culture replicates of one donor) carry effect sizes only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib import human_stats as hs


def _wide(sc: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return sc[sc.dataset == dataset]


def _row(dataset, family, contrast, score, design, res, effect_only=False, note=""):
    r = {"dataset": dataset, "family": family, "contrast": contrast, "score": score, "design": design,
         **res, "effect_only": effect_only, "note": note}
    if effect_only:
        r["p"] = np.nan
        r["test"] = "effect size only"
    return r


def unpaired(sc, dataset, family, g1, g0, scores, effect_only=False, label=None, note="", sel=None):
    d = _wide(sc, dataset) if sel is None else sel
    rows = []
    for s in scores:
        x = d[d.score == s]
        x1 = x[x.group == g1].groupby("subject").value.mean()
        x0 = x[x.group == g0].groupby("subject").value.mean()
        rows.append(_row(dataset, family, label or f"{g1} vs {g0}", s, "unpaired, subject means",
                         hs.two_group(x1, x0), effect_only, note))
    return rows


def paired(sc, dataset, family, g1, g0, scores, effect_only=False, label=None, note="", sel=None):
    d = _wide(sc, dataset) if sel is None else sel
    rows = []
    for s in scores:
        x = d[d.score == s].pivot_table(index="subject", columns="group", values="value", aggfunc="mean")
        if g1 not in x or g0 not in x:
            continue
        diff = (x[g1] - x[g0]).dropna()
        res = hs.paired(diff)
        res["mean1"], res["mean0"] = x.loc[diff.index, g1].mean(), x.loc[diff.index, g0].mean()
        rows.append(_row(dataset, family, label or f"{g1} vs {g0}", s, "paired within subject",
                         res, effect_only, note))
    return rows


def vs_single_control(sc, dataset, family, g1, g0, scores, label, note, sel):
    """Patients against the only available control sample: the control's variance cannot be
    estimated, so the interval reflects between-patient spread only and no P value is given."""
    rows = []
    for s in scores:
        x = sel[sel.score == s]
        x1 = x[x.group == g1].groupby("subject").value.mean()
        x0 = x[x.group == g0].value
        if len(x0) != 1 or len(x1) < 2:
            continue
        res = hs.paired(x1 - float(x0.iloc[0]))
        res["mean1"], res["mean0"], res["n0"] = x1.mean(), float(x0.iloc[0]), 1
        res["hedges_g"] = res["g_ci_low"] = res["g_ci_high"] = np.nan
        rows.append(_row(dataset, family, label, s, "patients vs single control", res, True, note))
    return rows


def response_shift(sc, dataset, family, scores, stim, base, s1, s0, paired_states: bool, label, note=""):
    """Ex vivo response (stim - basal within subject) compared between clinical states."""
    d = _wide(sc, dataset)
    rows = []
    for s in scores:
        x = d[d.score == s].pivot_table(index=["subject", "state"], columns="stim", values="value")
        if stim not in x or base not in x:     # score not computable in this dataset
            continue
        resp = (x[stim] - x[base]).dropna().reset_index()
        r1 = resp[resp.state == s1].set_index("subject")[0]
        r0 = resp[resp.state == s0].set_index("subject")[0]
        if paired_states:
            common = r1.index.intersection(r0.index)
            res = hs.paired(r1[common] - r0[common])
            res["mean1"], res["mean0"] = r1[common].mean(), r0[common].mean()
            design = "paired within patient (response = LPS - basal)"
        else:
            res = hs.two_group(r1, r0)
            design = "unpaired subjects (response = LPS - basal)"
        rows.append(_row(dataset, family, label, s, design, res, False, note))
    return rows


def gene_level_identifiability(x: pd.DataFrame, d: pd.DataFrame, a: str, b: str, ctrl: str,
                               dataset: str, family: str, genes=None) -> list:
    """Within one study: correlation of two stimulus responses across genes and the fraction of
    the first response left after removing a fitted multiple of the second."""
    m = {g: x[d[d.group == g]["sample"]].mean(axis=1) for g in (a, b, ctrl)}
    ra, rb = (m[a] - m[ctrl]), (m[b] - m[ctrl])
    keep = (ra.abs() >= 1) | (rb.abs() >= 1)
    if genes is not None:
        keep &= ra.index.isin(genes)
    ra, rb = ra[keep], rb[keep]
    if len(ra) < 5:
        return []
    r = float(np.corrcoef(ra, rb)[0, 1])
    k = float((ra * rb).sum() / (rb * rb).sum())
    resid = ra - k * rb
    r2 = 1 - float((resid ** 2).sum() / ((ra - ra.mean()) ** 2).sum())
    base = {"dataset": dataset, "family": family, "design": "genes as units (descriptive)",
            "n1": int(keep.sum()), "n0": np.nan, "effect_only": True, "test": "effect size only"}
    return [{**base, "contrast": f"{a} response vs {b} response", "score": "gene-level Pearson r",
             "estimate": r},
            {**base, "contrast": f"{a} response vs {b} response", "score": f"fitted scale of {b} for {a}",
             "estimate": k},
            {**base, "contrast": f"{a} response vs {b} response", "score": f"R2 of {a} by scaled {b}",
             "estimate": r2}]


def add_fdr(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["fdr_bh_family"] = np.nan
    for fam, idx in c.groupby("family").groups.items():
        p = c.loc[idx, "p"].values
        c.loc[idx, "fdr_bh_family"] = hs.bh(p)
    return c
