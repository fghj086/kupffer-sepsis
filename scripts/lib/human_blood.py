"""Stage 07 secondary analysis: sepsis blood cohorts (patients as units).

* sepsis (first sample, day <= 1) vs healthy/non-infected controls, per cohort, pooled (REML)
* within-patient change from the first sample to the latest sample (day >= 3), per cohort
* 28-day mortality on the first sample: Firth logistic per cohort (score standardised within the
  cohort), log odds ratios pooled by REML random effects
Overlapping patients between the two GAinS deposits are kept only in the longitudinal deposit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib import human_stats as hs


def first_sample(s: pd.DataFrame, day_max: float) -> pd.DataFrame:
    p = s[s.group == "sepsis"].copy()
    if p["day"].notna().any():
        p = p[p["day"].isna() | (p["day"] <= day_max)]
        p = p.sort_values("day").drop_duplicates("subject", keep="first")
    else:
        p = p.drop_duplicates("subject")
    return p


def cohort_contrasts(name: str, sc: pd.DataFrame, design: pd.DataFrame, scores: list, day_max: float):
    rows = []
    d = sc[sc.dataset == name]
    first = first_sample(design, day_max)
    ctrl = design[design.group == "control"]
    for s in scores:
        x = d[d.score == s].set_index("sample").value
        if len(ctrl) >= 3:
            r = hs.two_group(x.reindex(first["sample"]).groupby(first.subject.values).mean(),
                             x.reindex(ctrl["sample"]).groupby(ctrl.subject.values).mean())
            rows.append({"dataset": name, "family": "blood: sepsis vs control", "contrast":
                         "sepsis first sample vs control", "score": s, "design": "unpaired subjects", **r})
        # longitudinal: first vs last sample on day >= 3 within patient
        if design["day"].notna().sum() > 0:
            sp = design[(design.group == "sepsis") & design["day"].notna()]
            late = sp[sp.day >= 3].sort_values("day").drop_duplicates("subject", keep="last")
            common = sorted(set(late.subject) & set(first.subject))
            if len(common) >= 5:
                a = x.reindex(late.set_index("subject").loc[common, "sample"]).values
                b = x.reindex(first.set_index("subject").loc[common, "sample"]).values
                r = hs.paired(pd.Series(a - b))
                r["mean1"], r["mean0"] = np.nanmean(a), np.nanmean(b)
                rows.append({"dataset": name, "family": "blood: change over time", "contrast":
                             "latest sample (day >= 3) vs first sample", "score": s,
                             "design": "paired within patient", **r})
    return rows


def mortality(per_cohort: dict, scores: list, min_deaths: int, level: float):
    rows, pooled = [], []
    for s in scores:
        est = []
        for name, (sc, first) in per_cohort.items():
            f = first.dropna(subset=["died"])
            x = sc[(sc.dataset == name) & (sc.score == s)].set_index("sample").value.reindex(f["sample"])
            ok = x.notna().values
            y = f["died"].astype(float).values[ok]
            xv = x.values[ok]
            if y.sum() < min_deaths or (len(y) - y.sum()) < min_deaths:
                continue
            xz = (xv - xv.mean()) / xv.std(ddof=1)
            r = hs.firth_logistic(xz, y, level)
            rows.append({"dataset": name, "family": "blood: 28-day mortality", "contrast":
                         "per SD of score, first sample", "score": s,
                         "design": "Firth logistic, profile penalised-likelihood interval",
                         "n1": int(y.sum()), "n0": int(len(y) - y.sum()),
                         "estimate": r["log_or"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "p": r["p"], "test": "penalised likelihood-ratio", "se": r["se"]})
            est.append((r["log_or"], r["se"]))
        if est:
            pr = hs.reml_pool([e[0] for e in est], [e[1] for e in est], level)
            pooled.append({"dataset": "pooled", "family": "blood: 28-day mortality", "contrast":
                           "per SD of score, first sample (REML random effects)", "score": s,
                           "design": f"{pr['k']} cohorts", "n1": np.nan, "n0": np.nan,
                           "estimate": pr["estimate"], "ci_low": pr["ci_low"], "ci_high": pr["ci_high"],
                           "p": pr["p"], "test": f"REML; tau2={pr['tau2']:.3g}; I2={pr['I2']:.2f}"})
    return rows + pooled


def pool_two_group(rows: list, family: str, contrast: str, level: float):
    df = pd.DataFrame([r for r in rows if r["family"] == family])
    out = []
    if df.empty:
        return out
    for s, g in df.groupby("score"):
        g = g.dropna(subset=["hedges_g"])
        if len(g) < 2:
            continue
        se = (g.g_ci_high - g.g_ci_low) / (2 * 1.96)
        pr = hs.reml_pool(g.hedges_g.values, se.values, level)
        out.append({"dataset": "pooled", "family": family, "contrast": contrast + " (REML, Hedges g)",
                    "score": s, "design": f"{pr['k']} cohorts", "hedges_g": pr["estimate"],
                    "g_ci_low": pr["ci_low"], "g_ci_high": pr["ci_high"], "p": pr["p"],
                    "test": f"REML; tau2={pr['tau2']:.3g}; I2={pr['I2']:.2f}"})
    return out
