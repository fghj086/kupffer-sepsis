"""Projection of BMDM stimulus signatures (stage 01) onto GSE224127 liver.

Scores are weighted means of per-gene log2 differences from each file's reference group,
so the "HMGB1-like" and "LPS-like" scores are transcriptional resemblance only.
CLP and LPS are compared through their effect sizes (each against its own in-batch
reference), never through a merged model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_signatures(path: Path) -> dict[str, pd.Series] | None:
    if not path.exists():
        return None
    s = pd.read_csv(path, sep="\t")
    need = {"signature", "gene", "weight"}
    if not need.issubset(s.columns):
        raise SystemExit(f"{path} lacks columns {need - set(s.columns)}")
    return {k: g.set_index("gene")["weight"].astype(float) for k, g in s.groupby("signature")}


def axis_names(sigs: dict) -> dict[str, str | None]:
    """Identify the shared, contrast (HMGB1-versus-LPS) and biased signatures by name."""
    low = {k.lower(): k for k in sigs}

    def pick(test):
        return next((v for k, v in low.items() if test(k)), None)
    return {"shared": pick(lambda k: "shared" in k),
            "contrast": pick(lambda k: "axis" in k or "contrast" in k or "_vs_" in k),
            "hmgb1": pick(lambda k: "hmgb1" in k and "biased" in k),
            "lps": pick(lambda k: "lps" in k and "biased" in k)}


def add_derived_contrast(S: pd.DataFrame, ax: dict) -> pd.DataFrame:
    """If no explicit contrast axis is supplied, define it as HMGB1-biased minus LPS-biased."""
    if ax["contrast"] is None and ax["hmgb1"] and ax["lps"]:
        S = S.copy()
        S["contrast_hmgb1_minus_lps(derived)"] = S[ax["hmgb1"]] - S[ax["lps"]]
        ax["contrast"] = "contrast_hmgb1_minus_lps(derived)"
    return S


def relative_contrast(S: pd.DataFrame, comps: list[dict], ax: dict, rng, n_boot: int):
    """Per comparison: contrast-axis effect, shared-axis effect, and their ratio, with
    mouse-level bootstrap CIs (same resamples for both axes)."""
    rows, boots = [], {}
    if ax["contrast"] is None or ax["shared"] is None:
        return pd.DataFrame(), boots
    for c in comps:
        a_c = S.loc[c["test_cols"], ax["contrast"]].to_numpy()
        b_c = S.loc[c["ref_cols"], ax["contrast"]].to_numpy()
        a_s = S.loc[c["test_cols"], ax["shared"]].to_numpy()
        b_s = S.loc[c["ref_cols"], ax["shared"]].to_numpy()
        ia = rng.integers(0, len(a_c), (n_boot, len(a_c)))
        ib = rng.integers(0, len(b_c), (n_boot, len(b_c)))
        dc = a_c[ia].mean(1) - b_c[ib].mean(1)
        ds = a_s[ia].mean(1) - b_s[ib].mean(1)
        key = (c.get("model", c.get("file_key")), c.get("arm", c.get("perturbation")),
               c.get("time_d", np.nan))
        boots[key] = (dc, ds)
        ratio = dc / np.where(np.abs(ds) < 1e-9, np.nan, ds)
        rows.append({**{k: v for k, v in c.items() if k not in ("test_cols", "ref_cols")},
                     "contrast_axis": ax["contrast"], "shared_axis": ax["shared"],
                     "contrast_diff": a_c.mean() - b_c.mean(),
                     "contrast_ci_low": np.nanpercentile(dc, 2.5),
                     "contrast_ci_high": np.nanpercentile(dc, 97.5),
                     "shared_diff": a_s.mean() - b_s.mean(),
                     "shared_ci_low": np.nanpercentile(ds, 2.5),
                     "shared_ci_high": np.nanpercentile(ds, 97.5),
                     "contrast_per_shared": (a_c.mean() - b_c.mean()) / (a_s.mean() - b_s.mean()),
                     "ratio_ci_low": np.nanpercentile(ratio, 2.5),
                     "ratio_ci_high": np.nanpercentile(ratio, 97.5)})
    return pd.DataFrame(rows), boots


def clp_minus_lps(boots: dict, pairs) -> pd.DataFrame:
    """CLP-severe versus LPS at matched days: difference in contrast-axis effect and in
    contrast-per-shared ratio (independent bootstrap distributions, mice resampled)."""
    rows = []
    for t_clp, t_lps in pairs:
        k1, k2 = ("CLP", "severe", t_clp), ("LPS", "LPS", t_lps)
        if k1 not in boots or k2 not in boots:
            continue
        (c1, s1), (c2, s2) = boots[k1], boots[k2]
        dd = c1 - c2
        r1, r2 = c1 / s1, c2 / s2
        dr = r1 - r2
        rows.append({"clp_time_d": t_clp, "lps_time_d": t_lps,
                     "contrast_diff_clp_minus_lps": np.nanmedian(dd),
                     "ci_low": np.nanpercentile(dd, 2.5), "ci_high": np.nanpercentile(dd, 97.5),
                     "boot_p_two_sided": 2 * min((dd > 0).mean(), (dd < 0).mean()),
                     "ratio_diff_clp_minus_lps": np.nanmedian(dr),
                     "ratio_ci_low": np.nanpercentile(dr, 2.5),
                     "ratio_ci_high": np.nanpercentile(dr, 97.5)})
    return pd.DataFrame(rows)
