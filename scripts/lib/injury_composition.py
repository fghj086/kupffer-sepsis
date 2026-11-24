"""Stage 08 addendum: does the severity gradient in Kupffer-cell identity and tolerance-like
programs remain after accounting for leukocyte composition of the bulk liver?

Composition proxies (recruited monocyte/macrophage, neutrophil, total leukocyte, pan-macrophage)
are scored per mouse exactly like the stage 04 sets (file by file, centred on the file's
reference group, genes expressed in all three time-course files). The per-grade slope is then
re-estimated with each proxy, all proxies together, and all proxies plus the NF-kB score as
covariates, with stratum (file x time) fixed effects and the same bootstrap as the main
dose-response (mice resampled within stratum x grade).
A KC identity score relative to the pan-macrophage score (difference of the two log2 scores)
expresses identity per macrophage rather than macrophage abundance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lib.injury_scores import core_genes
from lib.injury_stats import _ols, _strat_boot_idx, centre
from lib.tc_design import REFERENCE
from lib.tc_io import build_design
from lib.tc_scores import log_cpm, score_sets


def proxy_scores(cfg: dict, cfg04: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-mouse composition-proxy scores (index = column name) and the genes used."""
    design, counts = build_design(cfg04, root)
    design = design[design.author_qc_flag == 0]
    ex = cfg04["expression"]
    sets = {k: core_genes(counts, v, ex["min_mean_cpm"])
            for k, v in cfg["composition_proxies"].items()}
    frames = []
    for fk, X in counts.items():
        d = design[design.file_key == fk]
        L = log_cpm(X, ex["pseudocount"])[d.column.tolist()]
        ref = d.loc[d.group == REFERENCE[fk], "column"].tolist()
        S, _ = score_sets(L, ref, sets)
        frames.append(S)
    genes = pd.DataFrame([{"set": k, "n_genes": len(g), "genes": ",".join(g),
                           "requested": ",".join(cfg["composition_proxies"][k])}
                          for k, g in sets.items()])
    return pd.concat(frames), genes


def _slope(sub: pd.DataFrame, y: str, covars: list[str]) -> tuple[float, float, int]:
    cols = list(dict.fromkeys([y, "grade"] + covars))
    C = centre(sub, cols, "stratum")
    X = np.c_[C.grade.values, C[covars].values] if covars else C.grade.values[:, None]
    yv = C[y].values
    b = _ols(yv, X)
    dof = len(yv) - X.shape[1] - sub.stratum.nunique()
    res = yv - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (res @ res) / dof)
    return float(b[0]), float(se[0]), dof


def adjusted_dose_response(M: pd.DataFrame, outcomes, models: dict, grade_map: dict,
                           rng, n_boot: int) -> pd.DataFrame:
    """Per-grade slope of each outcome under each covariate model (within stratum)."""
    M = M.assign(grade=M.group.map(grade_map))
    rows = []
    for label, sub in [("CLP_grades_only", M[M.grade > 0]), ("sham_to_severe", M)]:
        sub = sub.reset_index(drop=True)
        cells = (sub.stratum + "_" + sub.grade.astype(str)).values
        boots = [_strat_boot_idx(cells, rng) for _ in range(n_boot)]
        for y in outcomes:
            sd = centre(sub, [y], "stratum")[y].std(ddof=1)
            base = None
            for mname, covars in models.items():
                covars = [c for c in covars if c != y]
                b, se, dof = _slope(sub, y, covars)
                bt = np.array([_slope(sub.iloc[ix].reset_index(drop=True), y, covars)[0]
                               for ix in boots])
                if base is None:
                    base = b
                C = centre(sub, [y, "grade"], "stratum")
                r = stats.spearmanr(C.grade, C[y])
                rows.append({"grade_range": label, "score": y, "model": mname,
                             "covariates": "+".join(covars) if covars else "none",
                             "n_mice": len(sub), "n_strata": sub.stratum.nunique(),
                             "spearman_rho_unadjusted": r[0],
                             "slope_per_grade_log2": b,
                             "slope_ci_low": np.quantile(bt, .025),
                             "slope_ci_high": np.quantile(bt, .975),
                             "p_trend": 2 * stats.t.sf(abs(b / se), dof),
                             "slope_per_grade_sd": b / sd,
                             "fraction_of_unadjusted_slope": b / base if base else np.nan})
    return pd.DataFrame(rows)
