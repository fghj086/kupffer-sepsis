"""Timing summaries for the liver time courses (stage 04)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .tc_scores import bootstrap_peak


def timing_table(S: pd.DataFrame, effects: pd.DataFrame, comps: list[dict], fdr: float,
                 rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows = []
    for (model, arm), e_arm in effects.groupby(["model", "arm"]):
        arm_comps = sorted([c for c in comps if c["model"] == model and c["arm"] == arm],
                           key=lambda c: c["time_d"])
        times = [c["time_d"] for c in arm_comps]
        for score, e in e_arm.groupby("score"):
            e = e.set_index("time_d").loc[times]
            ipk = int(np.nanargmax(np.abs(e["diff_log2"].to_numpy())))
            peak = e.iloc[ipk]
            sign = int(np.sign(peak["diff_log2"])) or 1
            prob = bootstrap_peak(S, score, arm_comps, sign, rng, n_boot)
            sig = e[e["fdr"] < fdr]
            up = sig[sig["diff_log2"] > 0]
            dn = sig[sig["diff_log2"] < 0]
            half = e[(sign * e["diff_log2"]) >= 0.5 * abs(peak["diff_log2"])]
            rows.append({
                "model": model, "arm": arm, "score": score,
                "times_tested_d": ";".join(f"{t:g}" for t in times),
                "peak_time_d": times[ipk], "peak_direction": "up" if sign > 0 else "down",
                "peak_diff_log2": peak["diff_log2"], "peak_hedges_g": peak["hedges_g"],
                "peak_time_boot_prob": prob[ipk],
                "boot_prob_by_time": ";".join(f"{t:g}:{p:.3f}" for t, p in zip(times, prob)),
                "first_time_half_peak_d": half.index.min() if len(half) else np.nan,
                "earliest_sig_up_d": up.index.min() if len(up) else np.nan,
                "earliest_sig_down_d": dn.index.min() if len(dn) else np.nan,
                "n_times_sig": len(sig),
            })
    return pd.DataFrame(rows)


def kc_recruited_coupling(S: pd.DataFrame, design: pd.DataFrame, effects: pd.DataFrame,
                          fdr: float) -> pd.DataFrame:
    """Does KC identity fall while recruited-myeloid markers rise? Per model/arm/time:
    both effects side by side, plus the across-mouse Spearman correlation of the two
    scores within each file (mice as the unit)."""
    rows = []
    w = effects[effects.score.isin(["kc_identity", "recruited_mono_mac", "pan_macrophage",
                                    "kc_minus_panmac"])]
    wide = w.pivot_table(index=["model", "arm", "time_d", "file_key"], columns="score",
                         values=["diff_log2", "fdr"])
    for idx, r in wide.iterrows():
        d = dict(zip(["model", "arm", "time_d", "file_key"], idx))
        for s in ("kc_identity", "recruited_mono_mac", "pan_macrophage", "kc_minus_panmac"):
            d[f"{s}_diff"] = r.get(("diff_log2", s), np.nan)
            d[f"{s}_fdr"] = r.get(("fdr", s), np.nan)
        d["kc_down_and_recruited_up"] = bool(
            d["kc_identity_fdr"] < fdr and d["kc_identity_diff"] < 0
            and d["recruited_mono_mac_fdr"] < fdr and d["recruited_mono_mac_diff"] > 0)
        rows.append(d)
    out = pd.DataFrame(rows)
    cors = []
    for fk, d in design.groupby("file_key"):
        if fk not in ("LPS_timecourse", "CLP_d0.5", "CLP_d0.25_d1"):
            continue
        x = S.loc[d.column, "kc_identity"]
        y = S.loc[d.column, "recruited_mono_mac"]
        rho, p = stats.spearmanr(x, y)
        cors.append({"file_key": fk, "n_mice": len(d), "spearman_kc_vs_recruited": rho,
                     "p_spearman": p})
    return out.merge(pd.DataFrame(cors), on="file_key", how="left")
