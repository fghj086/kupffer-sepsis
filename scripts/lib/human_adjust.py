"""Stage 07: blood analyses adjusted for leukocyte composition, age and sex.

Composition covariates
  * marker scores, computed in every cohort from its own matrix: mean log2 expression of canonical
    neutrophil, monocyte and lymphocyte genes, standardised within the cohort;
  * deconvolved proportions (neutrophil, monocyte, lymphocyte = B + CD4 + CD8 + NK) of septic samples
    from the GSE60424-referenced deconvolution of stage 00b (septic samples of GAinS
    E-MTAB-5273, GSE95233, GSE57065 and GSE222393) — mortality sensitivity models.
Sepsis vs control: ordinary least squares of the subject-level score on sepsis status (+ covariates),
HC3 standard errors, coefficients pooled across cohorts by REML.
Mortality: one admission sample per patient, Firth logistic per cohort (score per SD), REML pooling
with Wald and Hartung-Knapp intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib import human_stats as hs

LYMPH = ["B cell", "CD4 T cell", "CD8 T cell", "NK cell"]


def marker_scores(x: pd.DataFrame, markers: dict) -> pd.DataFrame:
    out = {}
    for k, genes in markers.items():
        g = [m for m in genes if m in x.index]
        v = x.loc[g].mean(axis=0)
        out[f"mk_{k}"] = (v - v.mean()) / v.std(ddof=1)
        out[f"mk_{k}_n_genes"] = len(g)
    return pd.DataFrame(out)


def deconvolved(path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d = d.set_index("sample")
    lym = d[[f"prop_{t}" for t in LYMPH]].sum(axis=1)
    return pd.DataFrame({"dc_neutrophil": d["prop_neutrophil"], "dc_monocyte": d["prop_monocyte"],
                         "dc_lymphocyte": lym})


def _demo(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    if "age" in d and pd.to_numeric(d["age"], errors="coerce").notna().mean() > 0.8:
        out["age"] = pd.to_numeric(d["age"], errors="coerce")
    if "sex" in d and d["sex"].isin(["M", "F"]).mean() > 0.8:
        out["sex_male"] = (d["sex"] == "M").astype(float).where(d["sex"].isin(["M", "F"]))
    return out


def _std(df):
    return (df - df.mean()) / df.std(ddof=1)


def sepsis_vs_control(name, sc, design, mk, scores, day_max, first_fn):
    """Rows: unadjusted / composition / composition + age + sex, sepsis coefficient (z units)."""
    first = first_fn(design, day_max)
    ctrl = design[design.group == "control"]
    if len(ctrl) < 3:
        return []
    sub = pd.concat([first, ctrl]).drop_duplicates("sample").set_index("sample")
    sub["sepsis"] = (sub.group == "sepsis").astype(float)
    cov = mk.reindex(sub.index)[[c for c in mk.columns if not c.endswith("_n_genes")]]
    demo = _demo(sub)
    rows = []
    for s in scores:
        y = sc[(sc.dataset == name) & (sc.score == s)].set_index("sample").value.reindex(sub.index)
        for model, X in [("unadjusted", sub[["sepsis"]]),
                         ("composition markers", pd.concat([sub[["sepsis"]], cov], axis=1)),
                         ("composition markers + age + sex", pd.concat([sub[["sepsis"]], cov, demo], axis=1))]:
            if model.endswith("sex") and demo.shape[1] == 0:
                continue
            dfm = pd.concat([y.rename("y"), X], axis=1).dropna()
            if dfm.sepsis.nunique() < 2 or len(dfm) < X.shape[1] + 5:
                continue
            fit = sm.OLS(dfm.y, sm.add_constant(dfm.drop(columns="y"))).fit(cov_type="HC3")
            ci = fit.conf_int().loc["sepsis"]
            rows.append({"dataset": name, "analysis": "sepsis vs control", "score": s, "model": model,
                         "covariates": ",".join(c for c in X.columns if c != "sepsis") or "none",
                         "n_sepsis": int(dfm.sepsis.sum()), "n_control": int((1 - dfm.sepsis).sum()),
                         "estimate": fit.params["sepsis"], "se": fit.bse["sepsis"], "ci_low": ci[0],
                         "ci_high": ci[1], "p": fit.pvalues["sepsis"],
                         "estimate_type": "difference in score (z units), OLS with HC3 errors"})
    return rows


def pool(rows: list, keys: list, level: float, est="estimate", se="se") -> list:
    df = pd.DataFrame(rows)
    out = []
    for k, g in df.groupby(keys):
        g = g.dropna(subset=[est, se])
        if len(g) < 2:
            continue
        pr = hs.reml_pool(g[est].values, g[se].values, level)
        r = dict(zip(keys, k))
        r.update({"dataset": "pooled (REML)", "n_cohorts": pr["k"], "estimate": pr["estimate"],
                  "ci_low": pr["ci_low"], "ci_high": pr["ci_high"], "p": pr["p"], "tau2": pr["tau2"],
                  "I2": pr["I2"], "n_cohorts_same_sign_as_pooled": pr["sign_of_pooled_matches"],
                  "hk_ci_low": pr.get("hk_ci_low"), "hk_ci_high": pr.get("hk_ci_high"),
                  "hk_p": pr.get("hk_p"), "mhk_ci_low": pr.get("mhk_ci_low"),
                  "mhk_ci_high": pr.get("mhk_ci_high"), "mhk_p": pr.get("mhk_p"),
                  **{c: g[c].sum() for c in ("n_sepsis", "n_control", "n_patients", "n_deaths")
                     if c in g.columns}})
        out.append(r)
    return out


def mortality_models(name, sc, first, mk, dc, scores, tol_scores, min_deaths, level):
    f = first.dropna(subset=["died"]).drop_duplicates("subject").set_index("sample")
    y_all = f["died"].astype(float)
    cov = mk.reindex(f.index)[[c for c in mk.columns if not c.endswith("_n_genes")]]
    demo = _demo(f)
    dcov = dc.reindex(f.index) if dc is not None else None
    have_dc = dcov is not None and dcov.notna().all(axis=1).mean() > 0.9
    S = sc[sc.dataset == name].pivot_table(index="sample", columns="score", values="value").reindex(f.index)
    rows = []
    specs = [("unadjusted", None), ("composition markers", cov)]
    if demo.shape[1]:
        specs.append(("composition markers + age + sex", pd.concat([cov, demo], axis=1)))
    if have_dc:
        # the three proportions sum to 1, so with an intercept one must be left out (collinear
        # design otherwise); lymphocytes are the omitted remainder
        specs.append(("deconvolved proportions", dcov[["dc_neutrophil", "dc_monocyte"]]))
    for s in scores:
        model_list = list(specs)
        if s in tol_scores:
            sh = S[["shared_inflammatory"]]
            model_list += [("shared inflammatory score", sh),
                           ("shared inflammatory score + composition markers", pd.concat([sh, cov], axis=1))]
            if demo.shape[1]:
                model_list.append(("shared inflammatory score + composition markers + age + sex",
                                   pd.concat([sh, cov, demo], axis=1)))
        for model, C in model_list:
            dfm = pd.concat([S[s].rename("x"), y_all.rename("y")] + ([C] if C is not None else []), axis=1).dropna()
            ev = dfm.y.sum()
            if ev < min_deaths or len(dfm) - ev < min_deaths:
                continue
            xz = (dfm.x - dfm.x.mean()) / dfm.x.std(ddof=1)
            cv = _std(dfm.drop(columns=["x", "y"])).values if C is not None else None
            r = hs.firth_logistic(xz.values, dfm.y.values, level, covars=cv)
            rows.append({"dataset": name, "analysis": "28-day mortality", "score": s, "model": model,
                         "covariates": ",".join(C.columns) if C is not None else "none",
                         "n_patients": len(dfm), "n_deaths": int(ev), "estimate": r["log_or"],
                         "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"],
                         "estimate_type": "log odds ratio per SD, Firth logistic (profile interval)"})
    return rows


def finish(df: pd.DataFrame, log_or: bool) -> pd.DataFrame:
    df = df.copy()
    if log_or:
        for a, b in [("estimate", "OR_per_SD"), ("ci_low", "OR_ci_low"), ("ci_high", "OR_ci_high"),
                     ("hk_ci_low", "OR_hk_ci_low"), ("hk_ci_high", "OR_hk_ci_high"),
                     ("mhk_ci_low", "OR_mhk_ci_low"), ("mhk_ci_high", "OR_mhk_ci_high")]:
            if a in df:
                df[b] = np.exp(df[a])
    df["fdr_bh_model"] = np.nan
    for _, idx in df[df.dataset == "pooled (REML)"].groupby("model").groups.items():
        df.loc[idx, "fdr_bh_model"] = hs.bh(df.loc[idx, "p"].values)
    return df
