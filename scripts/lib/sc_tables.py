"""Tidy result tables for stage 03 (population scores, contrasts, decomposition, coupling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scripts.lib.sc_scores import cohen_d, pb_program_scores, pseudobulk, shift_share

KC_LIKE = "KC-like (KC-marker-intermediate)"
MACRO = ["Resident KC", KC_LIKE, "Recruited Mo-macrophages"]
MONO = ["Monocytes"]
SCOPES = {"macrophages": MACRO, "macrophages_plus_monocytes": MACRO + MONO}
# lineage-collapsed view: KC-marker-positive vs recruited (Mo-mac + monocytes)
COLLAPSE = {"Resident KC": "KC-marker-positive", KC_LIKE: "KC-marker-positive",
            "Recruited Mo-macrophages": "Recruited (Mo-mac + monocytes)",
            "Monocytes": "Recruited (Mo-mac + monocytes)"}
RECRUITED = {"Recruited Mo-macrophages", "Monocytes"}


def cell_population_means(obs: pd.DataFrame, score_cols, by=("condition", "sample")):
    g = obs.groupby(list(by) + ["population"], observed=True)
    m = g[score_cols].mean().add_suffix("__mean")
    s = g[score_cols].std().add_suffix("__sd")
    n = g.size().rename("n_cells")
    return pd.concat([m, s, n], axis=1).reset_index()


def pseudobulk_scores(my, sets, by=("condition", "sample")):
    lc, meta = pseudobulk(my, list(by) + ["population"], min_cells=20)
    sc_ = pb_program_scores(lc, sets)
    out = meta.merge(sc_.reset_index().rename(columns={"index": "pb_id"}), on="pb_id")
    return out.drop(columns="pb_id")


def contrasts(pb: pd.DataFrame, cell: pd.DataFrame, obs: pd.DataFrame, score_cols, ref: str,
              alt: str, sample_level: bool) -> pd.DataFrame:
    """Per population x score: difference alt - ref. With sample_level, pseudobulk scores of
    samples are the units (Hedges-type d, Mann-Whitney and Welch P). Without it (pooled
    libraries), only effect sizes: pseudobulk delta and cell-level Cohen's d."""
    rows = []
    for pop in sorted(obs["population"].unique()):
        for sc_ in score_cols:
            r = {"population": pop, "score": sc_, "reference": ref, "comparison": alt}
            a = pb[(pb.population == pop) & (pb.condition == ref)][sc_].dropna()
            b = pb[(pb.population == pop) & (pb.condition == alt)][sc_].dropna()
            r.update(pb_ref_mean=a.mean(), pb_alt_mean=b.mean(), pb_delta=b.mean() - a.mean(),
                     n_ref_units=len(a), n_alt_units=len(b))
            ca = obs[(obs.population == pop) & (obs.condition == ref)][sc_]
            cb = obs[(obs.population == pop) & (obs.condition == alt)][sc_]
            r.update(cell_delta=cb.mean() - ca.mean(), cell_cohen_d=cohen_d(ca, cb),
                     n_ref_cells=len(ca), n_alt_cells=len(cb))
            if sample_level and len(a) >= 3 and len(b) >= 3:
                r["sample_hedges_g"] = cohen_d(a, b) * (1 - 3 / (4 * (len(a) + len(b)) - 9))
                r["p_mannwhitney"] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                r["p_welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows.append(r)
    df = pd.DataFrame(rows)
    if "p_mannwhitney" in df:
        from statsmodels.stats.multitest import multipletests
        for col in ["p_mannwhitney", "p_welch"]:
            df[col.replace("p_", "fdr_")] = np.nan
            for s, d in df.groupby("score"):
                ok = d[col].notna()
                if ok.any():
                    df.loc[d.index[ok], col.replace("p_", "fdr_")] = multipletests(
                        d.loc[ok, col], method="fdr_bh")[1]
    return df


def composition(obs: pd.DataFrame, ref: str, alt: str, sample_level: bool) -> pd.DataFrame:
    rows = []
    for scope, pops in SCOPES.items():
        o = obs[obs.population.isin(pops)]
        tab = o.groupby(["condition", "sample", "population"], observed=True).size().unstack(fill_value=0)
        frac = tab.div(tab.sum(1), axis=0)
        frac["Recruited (Mo-mac + monocytes)"] = frac[[c for c in frac.columns if c in RECRUITED]].sum(1)
        frac = frac.stack().rename("fraction").reset_index()
        n = tab.stack().rename("n_cells").reset_index()
        frac = frac.merge(n, on=["condition", "sample", "population"], how="left")
        frac.insert(0, "scope", scope)
        rows.append(frac)
    df = pd.concat(rows)
    summ = []
    for (scope, pop), d in df.groupby(["scope", "population"]):
        a, b = d[d.condition == ref]["fraction"], d[d.condition == alt]["fraction"]
        r = {"scope": scope, "population": pop, "reference": ref, "comparison": alt,
             "ref_mean_fraction": a.mean(), "alt_mean_fraction": b.mean(),
             "delta_fraction": b.mean() - a.mean(), "n_ref": len(a), "n_alt": len(b)}
        if sample_level and len(a) >= 3 and len(b) >= 3:
            r["p_mannwhitney"] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        summ.append(r)
    return df, pd.DataFrame(summ)


def decomposition(obs: pd.DataFrame, score_cols, ref: str, alt: str, unit_level: bool):
    """Shift-share split of the change in the mean score of all cells in a scope.
    Pooled design: weights and population means taken from the pooled library.
    Sample design: per-sample weights and population means, averaged within condition."""
    rows, per_pop = [], []
    views = [(k, obs[obs.population.isin(v)]) for k, v in SCOPES.items()]
    col = obs[obs.population.isin(COLLAPSE)].copy()
    col["population"] = col["population"].astype(str).map(COLLAPSE)
    views.append(("kc_marker_vs_recruited", col))
    for scope, o in views:
        for sc_ in score_cols:
            def summ(cond):
                d = o[o.condition == cond]
                if not unit_level:
                    w = d.population.value_counts(normalize=True)
                    s = d.groupby("population", observed=True)[sc_].mean()
                    return w, s
                W = d.groupby("sample", observed=True).population.value_counts(normalize=True).unstack(fill_value=0)
                S = d.groupby(["sample", "population"], observed=True)[sc_].mean().unstack()
                return W.mean(0), S.mean(0)
            w0, s0 = summ(ref)
            w1, s1 = summ(alt)
            res = shift_share(w0, w1, s0, s1)
            rows.append({"scope": scope, "score": sc_, "reference": ref, "comparison": alt,
                         "mean_ref": (w0 * s0.reindex(w0.index)).sum(),
                         "mean_alt": (w1 * s1.reindex(w1.index)).sum(),
                         "total_change": res["total"], "composition_part": res["composition"],
                         "within_population_part": res["within"],
                         "composition_share": res["composition"] / res["total"] if res["total"] else np.nan})
            pp_ = pd.DataFrame({"composition": res["per_pop_composition"],
                                "within": res["per_pop_within"],
                                "weight_ref": w0.reindex(res["per_pop_within"].index),
                                "weight_alt": w1.reindex(res["per_pop_within"].index),
                                "score_ref": s0.reindex(res["per_pop_within"].index),
                                "score_alt": s1.reindex(res["per_pop_within"].index)})
            pp_.index.name = "population"
            pp_ = pp_.reset_index()
            pp_.insert(0, "score", sc_)
            pp_.insert(0, "scope", scope)
            per_pop.append(pp_)
    return pd.DataFrame(rows), pd.concat(per_pop)


def coupling(obs: pd.DataFrame, x: str, ys, by=("condition", "population")) -> pd.DataFrame:
    """Spearman rho between the inflammatory score and each clearance/identity score across
    cells within a population x condition (descriptive; cells are not replicates)."""
    rows = []
    for key, d in obs.groupby(list(by), observed=True):
        for y in ys:
            if len(d) < 30:
                continue
            rho = stats.spearmanr(d[x], d[y]).statistic
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                             x=x, y=y, spearman_rho=rho, n_cells=len(d)))
    return pd.DataFrame(rows)


def sample_coupling(pb: pd.DataFrame, x: str, ys) -> pd.DataFrame:
    """Across-sample coupling of pseudobulk scores within a population (samples as units)."""
    rows = []
    for pop, d in pb.groupby("population"):
        for y in ys:
            dd = d[[x, y]].dropna()
            if len(dd) < 6:
                continue
            r = stats.spearmanr(dd[x], dd[y])
            rows.append({"population": pop, "x": x, "y": y, "spearman_rho": r.statistic,
                         "p_spearman": r.pvalue, "n_samples": len(dd)})
    return pd.DataFrame(rows)
