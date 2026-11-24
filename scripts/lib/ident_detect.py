"""Stage 09 A: run the mixture simulation, summarise detection power and bound the HMGB1-like
fraction compatible with the observed Kupffer-cell and macrophage responses.

Two statistics: the dsHMGB1 share of the NNLS fit (per panel) and the stage 01 HMGB1-vs-LPS axis
(panel-free). Detection thresholds are the 95th percentile of the statistic at f = 0, either
for the same LPS source ("setting", knows the true LPS-like response) or pooled over the
cross-laboratory LPS sources ("pooled", the realistic case where the LPS-like response of the
target is not known in advance). The fixed threshold share > 0.05 is reported as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WITHIN = "within_GSE200210"


def run_draws(sim, settings: dict, f_grid, n_draws: int, rng, panels=None) -> pd.DataFrame:
    """settings: name -> noise spec. One row per draw x f x statistic (panel or axis)."""
    panels = panels or list(sim.panels)
    rows = []
    for src in sim.sources:
        for sname, spec in settings.items():
            n = len(sim.folds) if spec["kind"] == "none" else n_draws
            for f, ks, Y in sim.simulate(src, spec, f_grid, n, rng):
                ax = sim_axis(sim, Y)
                rows.append(pd.DataFrame({"source": src, "noise": sname, "f": f, "draw": np.arange(n),
                                          "statistic": "axis", "panel": "none", "value": ax,
                                          "r2": np.nan}))
                for p in panels:
                    sh, r2 = sim.fit_draws(p, src, ks, Y)
                    rows.append(pd.DataFrame({"source": src, "noise": sname, "f": f,
                                              "draw": np.arange(n), "statistic": "hmgb1_share",
                                              "panel": p, "value": sh, "r2": r2}))
    return pd.concat(rows, ignore_index=True)


def sim_axis(sim, Y):
    from lib.ident_sim import axis_scores
    return axis_scores(Y, sim.iH, sim.iL, sim.iS)


def _thresholds(d: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    null = d[d.f == 0]
    own = null.groupby(["source", "noise", "panel", "statistic"]).value.quantile(q).rename("thr_setting")
    pooled = null[null.source != WITHIN].groupby(["noise", "panel", "statistic"]).value.quantile(q)
    return own.reset_index(), pooled.rename("thr_pooled").reset_index()


def detection_table(d: pd.DataFrame, q=0.95, fixed=0.05) -> pd.DataFrame:
    own, pooled = _thresholds(d, q)
    x = d.merge(own, on=["source", "noise", "panel", "statistic"]).merge(
        pooled, on=["noise", "panel", "statistic"], how="left")
    x["det_setting"] = x.value > x.thr_setting
    x["det_pooled"] = x.value > x.thr_pooled
    x["det_fixed"] = (x.value > fixed) & (x.statistic == "hmgb1_share")
    g = x.groupby(["source", "noise", "panel", "statistic", "f"], as_index=False)
    out = g.agg(n_draws=("value", "size"), mean=("value", "mean"), median=("value", "median"),
                q05=("value", lambda v: v.quantile(0.05)), q95=("value", lambda v: v.quantile(0.95)),
                p_share_gt_fixed=("det_fixed", "mean"), thr_setting=("thr_setting", "first"),
                power_setting=("det_setting", "mean"), thr_pooled=("thr_pooled", "first"),
                power_pooled=("det_pooled", "mean"), mean_r2=("r2", "mean"))
    out.loc[out.statistic == "axis", "p_share_gt_fixed"] = np.nan
    return out


def limits_table(det: pd.DataFrame, power=0.8) -> pd.DataFrame:
    rows = []
    for (src, nz, p, stat), d in det.groupby(["source", "noise", "panel", "statistic"]):
        d = d.sort_values("f")
        cal = {"setting": "power_setting", "pooled": "power_pooled"}
        if stat == "hmgb1_share":
            cal["fixed_share_gt_0.05"] = "p_share_gt_fixed"
        for cname, col in cal.items():
            hit = d[(d.f > 0) & (d[col] >= power)]
            f0 = d[d.f == 0][col]
            rows.append({"source": src, "noise": nz, "panel": p, "statistic": stat,
                         "threshold": cname, "false_positive_rate_f0": float(f0.iloc[0]) if len(f0) else np.nan,
                         "min_f_power80": float(hit.f.min()) if len(hit) else np.nan,
                         "power_at_f0.1": _at(d, col, 0.1), "power_at_f0.2": _at(d, col, 0.2),
                         "power_at_f0.3": _at(d, col, 0.3), "power_at_f0.5": _at(d, col, 0.5)})
    return pd.DataFrame(rows)


def _at(d, col, f):
    m = d[np.isclose(d.f, f)][col]
    return float(m.iloc[0]) if len(m) else np.nan


def upper_bounds(d: pd.DataFrame, obs: dict, target: str, alpha=0.05) -> pd.DataFrame:
    """For each f: P(simulated statistic <= observed). f is excluded when that probability is
    below alpha (the observation would be unusually HMGB1-poor for that fraction). Rows are given
    per LPS source and for the cross-laboratory sources pooled."""
    rows = []
    d = d.copy()
    pooled = d[d.source != WITHIN].assign(source="cross_lab_pooled")
    for (src, nz, p, stat), g in pd.concat([d, pooled]).groupby(["source", "noise", "panel", "statistic"]):
        o = obs[p]["axis"] if stat == "axis" else obs[p]["hmgb1_share"]
        if o is None or np.isnan(o):
            continue
        prob = g.groupby("f").value.apply(lambda v: float(np.mean(v <= o + 1e-12)))
        excl = prob[prob < alpha]
        compat = prob[prob >= alpha]
        # observed value outside the range of pure LPS-like simulations (f = 0 itself rejected):
        # no fraction is compatible and no upper bound is reported
        f0_ok = bool(prob.loc[prob.index.min()] >= alpha)
        if not f0_ok:
            excl = excl.iloc[:0]
        for f, pr in prob.items():
            rows.append({"target": target, "source": src, "noise": nz, "panel": p,
                         "statistic": stat, "observed": o, "f": f, "p_sim_le_observed": pr,
                         "excluded": bool(pr < alpha), "f0_compatible": f0_ok,
                         "smallest_f_excluded": float(excl.index.min()) if len(excl) else np.nan,
                         "largest_f_compatible": float(compat.index.max()) if len(compat) else np.nan})
    return pd.DataFrame(rows)
