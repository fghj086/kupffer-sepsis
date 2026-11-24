"""Stage 08 statistics: severity dose-response, injury models adjusted for inflammation and
macrophage abundance, and arm-level tracking across cytokine perturbations.

Mouse is the unit. Stratum fixed effects are implemented by centring every variable within
stratum; bootstrap resamples mice within stratum (and grade, for the dose-response).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def centre(df: pd.DataFrame, cols, strata) -> pd.DataFrame:
    return df[cols] - df.groupby(strata)[cols].transform("mean")


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _strat_boot_idx(groups: np.ndarray, rng) -> np.ndarray:
    idx = []
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        idx.append(rng.choice(m, len(m), replace=True))
    return np.concatenate(idx)


def _fdr(df, col, by):
    df["fdr"] = np.nan
    for _, g in df.groupby(by):
        ok = g[col].notna()
        if ok.any():
            df.loc[g.index[ok], "fdr"] = multipletests(g.loc[ok, col], method="fdr_bh")[1]
    return df


# ------------------------------------------------------------------ 1. dose-response
def dose_response(M: pd.DataFrame, scores, grade_map, rng, n_boot, inflammation) -> pd.DataFrame:
    rows = []
    M = M.assign(grade=M.group.map(grade_map), stratum=M.file_key + "_d" + M.time_d.astype(str))
    for label, sub in [("sham_to_severe", M), ("CLP_grades_only", M[M.grade > 0])]:
        sub = sub.reset_index(drop=True)
        for s in scores:
            for st, g in sub.groupby("stratum"):
                r, p = stats.spearmanr(g.grade, g[s])
                rows.append({"grade_range": label, "score": s, "stratum": st, "n_mice": len(g),
                             "spearman_rho": r, "p_spearman": p})
            cols = list(dict.fromkeys([s, "grade", inflammation]))
            C = centre(sub, cols, "stratum")
            r, p = stats.spearmanr(C.grade, C[s])
            x, y, z = C.grade.values, C[s].values, C[inflammation].values
            b = _ols(y, x[:, None])[0]
            resid = y - b * x
            dof = len(y) - 1 - sub.stratum.nunique()
            se = np.sqrt(resid @ resid / dof / (x @ x))
            b_adj = _ols(y, np.c_[x, z])[0] if s != inflammation else np.nan
            cells = (sub.stratum + "_" + sub.grade.astype(str)).values
            bt, bta = np.empty(n_boot), np.empty(n_boot)
            for i in range(n_boot):
                ix = _strat_boot_idx(cells, rng)
                Cb = centre(sub.iloc[ix].reset_index(drop=True), cols, "stratum")
                bt[i] = _ols(Cb[s].values, Cb.grade.values[:, None])[0]
                bta[i] = (_ols(Cb[s].values, np.c_[Cb.grade.values, Cb[inflammation].values])[0]
                          if s != inflammation else np.nan)
            sd = C[s].std(ddof=1)
            rows.append({"grade_range": label, "score": s, "stratum": "pooled_within_stratum",
                         "n_mice": len(sub), "spearman_rho": r, "p_spearman": p,
                         "slope_per_grade_log2": b, "slope_ci_low": np.quantile(bt, .025),
                         "slope_ci_high": np.quantile(bt, .975),
                         "p_trend": 2 * stats.t.sf(abs(b / se), dof),
                         "slope_per_grade_sd": b / sd,
                         "slope_adj_inflammation_log2": b_adj,
                         "slope_adj_ci_low": np.nanquantile(bta, .025) if s != inflammation else np.nan,
                         "slope_adj_ci_high": np.nanquantile(bta, .975) if s != inflammation else np.nan})
    out = pd.DataFrame(rows)
    pooled = out[out.stratum == "pooled_within_stratum"].copy()
    pooled = _fdr(pooled, "p_trend", ["grade_range"])
    return out.merge(pooled[["grade_range", "score", "stratum", "fdr"]],
                     on=["grade_range", "score", "stratum"], how="left").rename(columns={"fdr": "fdr_trend"})


# ------------------------------------------------------------------ 2. injury models
def _std_design(df, y, preds, strata):
    C = centre(df, [y] + preds, strata)
    C = C / C.std(ddof=1)
    return C[y].values, C[preds].values


def _partial(y, x, Z):
    if Z.shape[1]:
        ry = y - Z @ _ols(y, Z)
        rx = x - Z @ _ols(x, Z)
    else:
        ry, rx = y, x
    return ry, rx


def injury_model(df, outcome, program, covars, strata, rng, n_boot) -> dict:
    preds = [program] + covars
    df = df.dropna(subset=[outcome] + preds).reset_index(drop=True)
    y, X = _std_design(df, outcome, preds, strata)
    beta = _ols(y, X)
    n, k, ns = len(y), X.shape[1], df.groupby(strata).ngroups
    dof = n - k - ns
    res = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtXi) * (res @ res) / dof)
    r2_full = 1 - res @ res / (y @ y)
    Z = X[:, 1:]
    r2_cov = 1 - np.sum((y - Z @ _ols(y, Z)) ** 2) / (y @ y) if Z.shape[1] else 0.0
    ry, rx = _partial(y, X[:, 0], Z)
    pr = float(np.corrcoef(ry, rx)[0, 1])
    prs = float(stats.spearmanr(ry, rx)[0])
    p_pr = 2 * stats.t.sf(abs(pr) * np.sqrt(dof / max(1e-12, 1 - pr ** 2)), dof)
    groups = df.groupby(strata).ngroup().values
    bt = np.empty((n_boot, 2))
    for i in range(n_boot):
        ix = _strat_boot_idx(groups, rng)
        yb, Xb = _std_design(df.iloc[ix].reset_index(drop=True), outcome, preds, strata)
        bt[i, 0] = _ols(yb, Xb)[0]
        ryb, rxb = _partial(yb, Xb[:, 0], Xb[:, 1:])
        bt[i, 1] = np.corrcoef(ryb, rxb)[0, 1]
    vif = 1 / (1 - (1 - np.sum((X[:, 0] - Z @ _ols(X[:, 0], Z)) ** 2) / (X[:, 0] @ X[:, 0]))) if Z.shape[1] else 1.0
    return {"n_mice": n, "n_strata": ns, "std_beta": beta[0], "se": se[0],
            "p_beta": 2 * stats.t.sf(abs(beta[0] / se[0]), dof),
            "beta_ci_low": np.nanquantile(bt[:, 0], .025), "beta_ci_high": np.nanquantile(bt[:, 0], .975),
            "boot_frac_same_sign": float(np.mean(np.sign(bt[:, 0]) == np.sign(beta[0]))),
            "partial_r": pr, "partial_r_ci_low": np.nanquantile(bt[:, 1], .025),
            "partial_r_ci_high": np.nanquantile(bt[:, 1], .975), "partial_spearman": prs,
            "p_partial": p_pr, "r2_covariates": r2_cov, "delta_r2_program": r2_full - r2_cov,
            "vif_program": vif,
            **{f"std_beta_{c}": b for c, b in zip(covars, beta[1:])}}


def leave_one_out(df, outcome, program, covars, strata, unit) -> dict:
    preds = [program] + covars
    bs = []
    for u in df[unit].unique():
        sub = df[df[unit] != u].dropna(subset=[outcome] + preds).reset_index(drop=True)
        if len(sub) < len(preds) + 5:
            continue
        y, X = _std_design(sub, outcome, preds, strata)
        bs.append(_ols(y, X)[0])
    bs = np.array(bs)
    return {"lobo_unit": unit, "lobo_n_fits": len(bs), "lobo_beta_min": bs.min(),
            "lobo_beta_max": bs.max()}, bs


def injury_table(M, cohort, cfg, rng) -> pd.DataFrame:
    sc, adj = cfg["scores"], cfg["adjustment"]
    covsets = {"inflammation_composition": [sc["inflammation"], sc["composition"]],
               "inflammation_only": [sc["inflammation"]], "unadjusted": []}
    rows = []
    for adj_name, strata in adj.items():
        lobo_unit = "file_key" if (adj_name == "batch" or cohort == "LPS") else "stratum"
        for cov_name, covars in covsets.items():
            for out in sc["outcomes"]:
                for prog in sc["programs"]:
                    r = injury_model(M, out, prog, covars, strata, rng, cfg["stats"]["n_boot"])
                    lo, bs = leave_one_out(M, out, prog, covars, strata, lobo_unit)
                    lo["lobo_n_same_sign"] = int(np.sum(np.sign(bs) == np.sign(r["std_beta"])))
                    rows.append({"cohort": cohort, "adjustment_strata": adj_name,
                                 "primary": adj_name == cfg["primary_adjustment"][cohort],
                                 "covariates": cov_name, "outcome": out, "program": prog, **r, **lo})
    t = pd.DataFrame(rows)
    return _fdr(t, "p_beta", ["cohort", "adjustment_strata", "covariates", "outcome"])


def joint_table(M, cohort, cfg, rng) -> pd.DataFrame:
    """All programs entered together with inflammation and composition (primary strata only)."""
    sc = cfg["scores"]
    strata = cfg["adjustment"][cfg["primary_adjustment"][cohort]]
    preds = sc["programs"] + [sc["inflammation"], sc["composition"]]
    rows = []
    for out in sc["outcomes"]:
        df = M.dropna(subset=[out] + preds).reset_index(drop=True)
        y, X = _std_design(df, out, preds, strata)
        beta = _ols(y, X)
        groups = df.groupby(strata).ngroup().values
        bt = np.array([_ols(*_std_design(df.iloc[_strat_boot_idx(groups, rng)].reset_index(drop=True),
                                         out, preds, strata)) for _ in range(cfg["stats"]["n_boot"])])
        for j, p in enumerate(preds):
            rows.append({"cohort": cohort, "adjustment_strata": cfg["primary_adjustment"][cohort],
                         "primary": True, "covariates": "joint_all_programs", "outcome": out,
                         "program": p, "n_mice": len(y), "std_beta": beta[j],
                         "beta_ci_low": np.quantile(bt[:, j], .025),
                         "beta_ci_high": np.quantile(bt[:, j], .975),
                         "boot_frac_same_sign": float(np.mean(np.sign(bt[:, j]) == np.sign(beta[j])))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. perturbation tracking
def hedges(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    return j * (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def arm_shifts(M, pcfg, score_names) -> pd.DataFrame:
    rows = []
    for fam, spec in pcfg.items():
        if fam == "families":
            continue
        for fk in spec["files"]:
            d = M[M.file_key == fk]
            ref = d[d.group == spec["reference"]]
            for arm, g in d[d.group != spec["reference"]].groupby("group"):
                r = {"arm_family": fam, "file_key": fk, "arm": arm, "reference": spec["reference"],
                     "n_arm": len(g), "n_reference": len(ref)}
                for s in score_names:
                    r[f"delta_{s}"] = g[s].mean() - ref[s].mean()
                    r[f"g_{s}"] = hedges(g[s].values, ref[s].values)
                rows.append(r)
    return pd.DataFrame(rows)


def tracking(A: pd.DataFrame, cfg, rng) -> pd.DataFrame:
    sc = cfg["scores"]
    inf = f"delta_{sc['inflammation']}"
    rows = []
    for fam, members in cfg["perturbations"]["families"].items():
        sub = A[A.arm_family.isin(members)].reset_index(drop=True)
        sub = sub.assign(il6_arm=sub.arm.str.contains("IL6").astype(float))
        n = len(sub)
        for out in sc["outcomes"]:
            yo = f"delta_{out}"
            for prog in sc["programs"]:
                xp = f"delta_{prog}"
                for cov_name, covs in [("unadjusted", []), ("inflammation_shift", [inf]),
                                       ("inflammation_shift_plus_IL6_arm", [inf, "il6_arm"])]:
                    if cov_name.endswith("IL6_arm") and fam != "cytokine_injection":
                        continue
                    cols = [yo, xp] + covs

                    def fit(D):
                        Zs = (D[cols] - D[cols].mean()) / D[cols].std(ddof=1)
                        Zs = Zs.fillna(0.0)
                        X = Zs[[xp] + covs].values
                        b = _ols(Zs[yo].values, X)[0]
                        ry, rx = _partial(Zs[yo].values, X[:, 0], X[:, 1:])
                        return b, np.corrcoef(ry, rx)[0, 1], stats.spearmanr(ry, rx)[0]
                    b, pr, ps = fit(sub)
                    bt = np.array([fit(sub.iloc[rng.integers(0, n, n)].reset_index(drop=True))
                                   for _ in range(cfg["stats"]["n_boot"])])
                    rows.append({"arm_family": fam, "n_arms": n, "outcome_shift": out,
                                 "program_shift": prog, "covariates": cov_name,
                                 "std_beta": b, "beta_ci_low": np.nanquantile(bt[:, 0], .025),
                                 "beta_ci_high": np.nanquantile(bt[:, 0], .975),
                                 "partial_r": pr, "partial_r_ci_low": np.nanquantile(bt[:, 1], .025),
                                 "partial_r_ci_high": np.nanquantile(bt[:, 1], .975),
                                 "partial_spearman": ps,
                                 "boot_frac_same_sign": float(np.nanmean(np.sign(bt[:, 0]) == np.sign(b)))})
    return pd.DataFrame(rows)
