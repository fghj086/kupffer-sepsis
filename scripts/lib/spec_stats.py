"""Small-sample statistics for stage 10: exact two-sample permutation, group bootstrap,
Welch contrast of two independent differences, Hedges g (from sorted_stats)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from scripts.lib.sorted_stats import hedges_g


def exact_perm_p(a, b) -> float:
    """Two-sided exact permutation P for the difference in means (all relabelings)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = np.r_[a, b]
    n, na = len(x), len(a)
    obs = abs(a.mean() - b.mean())
    hits = tot = 0
    for idx in itertools.combinations(range(n), na):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        d = abs(x[m].mean() - x[~m].mean())
        hits += d >= obs - 1e-12
        tot += 1
    return hits / tot


def boot_groups(groups: list[np.ndarray], fn, n_boot: int, rng) -> np.ndarray:
    """Resample units with replacement within each group; fn(*resampled groups) -> statistic."""
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = fn(*[g[rng.integers(0, len(g), len(g))] for g in groups])
    return out


def two_group(case, ref, n_boot: int, rng) -> dict:
    """Case minus reference: difference, Welch CI, percentile bootstrap CI, Hedges g, exact P."""
    a, b = np.asarray(case, float), np.asarray(ref, float)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.nan
    tq = stats.t.ppf(0.975, dfw) if se > 0 else np.nan
    bt = boot_groups([a, b], lambda x, y: x.mean() - y.mean(), n_boot, rng)
    g, glo, ghi = hedges_g(a, b)
    return {"n_case": len(a), "n_ref": len(b), "mean_case": a.mean(), "mean_ref": b.mean(),
            "diff": diff, "welch_ci_low": diff - tq * se, "welch_ci_high": diff + tq * se,
            "boot_ci_low": np.quantile(bt, 0.025), "boot_ci_high": np.quantile(bt, 0.975),
            "hedges_g": g, "g_ci_low": glo, "g_ci_high": ghi, "p_exact_permutation": exact_perm_p(a, b)}


def did_unpaired(kc_clp, cd_clp, kc_sham, cd_sham, n_boot: int, rng) -> dict:
    """(KC - other)_CLP - (KC - other)_sham with independent libraries: Welch contrast
    (Satterthwaite df) and a bootstrap resampling libraries within each of the four groups."""
    g = [np.asarray(x, float) for x in (kc_clp, cd_clp, kc_sham, cd_sham)]
    c = np.array([1, -1, -1, 1])
    est = float(sum(ci * x.mean() for ci, x in zip(c, g)))
    v = np.array([x.var(ddof=1) / len(x) for x in g])
    se = float(np.sqrt(v.sum()))
    df = v.sum() ** 2 / np.sum(v ** 2 / (np.array([len(x) for x in g]) - 1)) if se > 0 else np.nan
    tq = stats.t.ppf(0.975, df) if se > 0 else np.nan
    bt = boot_groups(g, lambda a, b, cc, d: (a.mean() - b.mean()) - (cc.mean() - d.mean()), n_boot, rng)
    return {"gap_sham": g[2].mean() - g[3].mean(), "gap_clp": g[0].mean() - g[1].mean(),
            "delta_disease": est, "welch_ci_low": est - tq * se, "welch_ci_high": est + tq * se,
            "welch_df": df, "p_welch_contrast": 2 * stats.t.sf(abs(est) / se, df) if se > 0 else np.nan,
            "boot_ci_low": np.quantile(bt, 0.025), "boot_ci_high": np.quantile(bt, 0.975)}
