"""Stage 06: identifiability of the HMGB1-like component and its behaviour over time.

magnitude_matched(): in GSE200210, how much of the disulfide-HMGB1 response is a scaled LPS
response (fitted scale and the stage 01 scale), and whether what remains is reproducible between
independent culture replicates (replicate i of each arm against PBS replicate i only, so that
residuals of different replicates share no samples).
per_mouse_timecourse(): attribution (no merging of collinear references) of every mouse in the GSE224127 time courses, summarised by
arm with a bootstrap over mice.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lib.attr_model import fit_nnls, unit_rms


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def magnitude_matched(rs, cfg, rng, n_boot=1000) -> pd.DataFrame:
    st = rs.studies["GSE200210"]
    g = pd.Index(sorted(set(rs.genes["dsHMGB1"]) | set(rs.genes["LPS_24h"])))
    ds, lps = rs.vec["dsHMGB1"].reindex(g).values, rs.vec["LPS_24h"].reindex(g).values
    beta = float(ds @ lps / (lps @ lps))
    ss = float(ds @ ds)
    rows = [{"quantity": "fitted LPS scale for dsHMGB1", "value": beta},
            {"quantity": "R2 of dsHMGB1 by fitted-scale LPS", "value": 1 - float(np.sum((ds - beta * lps) ** 2)) / ss}]
    s0 = cfg["controls"]["lps_scale"]
    rows.append({"quantity": f"R2 of dsHMGB1 by LPS x {s0}", "value": 1 - float(np.sum((ds - s0 * lps) ** 2)) / ss})
    rows.append({"quantity": "Pearson r dsHMGB1 vs LPS (responsive genes)", "value": _r(ds, lps)})
    # paired replicate vectors (replicate i vs PBS i)
    L = st.L.reindex(g)
    rep = lambda arm: [c for c in st.cols(arm)]
    pbs, dsc, lpc = rep("PBS"), rep("dsHMGB1"), rep("LPS")
    n = min(len(pbs), len(dsc), len(lpc))
    dsv = [L[dsc[i]] - L[pbs[i]] for i in range(n)]
    lpv = [L[lpc[i]] - L[pbs[i]] for i in range(n)]
    res = [dsv[i] - beta * lpv[i] for i in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    r_res = [_r(res[i], res[j]) for i, j in pairs]
    r_ds = [_r(dsv[i], dsv[j]) for i, j in pairs]
    # bootstrap over genes for the mean residual reproducibility
    boots = []
    m = len(g)
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        boots.append(np.mean([_r(res[i].values[idx], res[j].values[idx]) for i, j in pairs]))
    rows += [{"quantity": "replicate reproducibility of dsHMGB1 response (mean r, independent PBS)",
              "value": float(np.mean(r_ds))},
             {"quantity": "replicate reproducibility of dsHMGB1 minus fitted-scale LPS (mean r)",
              "value": float(np.mean(r_res)), "ci_low": float(np.quantile(boots, 0.025)),
              "ci_high": float(np.quantile(boots, 0.975))}]
    for (i, j), a, b in zip(pairs, r_res, r_ds):
        rows.append({"quantity": f"residual r, replicates {i + 1} vs {j + 1}", "value": a})
    # share of the reproducible dsHMGB1 signal: residual reproducibility relative to total
    rows.append({"quantity": "n responsive genes (dsHMGB1 or LPS)", "value": float(m)})
    return pd.DataFrame(rows)


def per_mouse_timecourse(targets, rs, cfg, panels, informative, rng, n_boot=1000) -> pd.DataFrame:
    rows = []
    for t in targets:
        if "per_mouse" not in t:
            continue
        for pname in panels:
            panel = cfg["panels"][pname]
            g = informative(rs, panel, t["y"])
            R = unit_rms(rs.matrix(panel, g).values)
            shares = []
            for col, v in t["per_mouse"].items():
                b, r2 = fit_nnls(unit_rms(v.reindex(g).values), R)
                shares.append(np.r_[b / b.sum() if b.sum() > 0 else b, r2])
            S = np.array(shares)
            k = len(S)
            bt = np.array([S[rng.integers(0, k, k)].mean(0) for _ in range(n_boot)])
            for j, ref in enumerate(panel + ["r2_total"]):
                rows.append({"target": t["target"], "model": t.get("model"), "group": t.get("group"),
                             "time_d": t.get("time_d"), "panel": pname, "reference": ref,
                             "n_mice": k, "mean": S[:, j].mean(),
                             "ci_low": np.quantile(bt[:, j], 0.025),
                             "ci_high": np.quantile(bt[:, j], 0.975)})
    return pd.DataFrame(rows)
