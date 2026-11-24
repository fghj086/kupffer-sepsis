"""KC-marker-positive vs recruited myeloid cells beyond the shared inflammatory response,
functional uncoupling index and pseudobulk log2 fold changes (stage 03)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scripts.lib.sc_scores import cohen_d, pseudobulk

KC, REC = "KC-marker-positive", "Recruited (Mo-mac + monocytes)"


def _ols(y, X):
    """OLS coefficients and classical SEs (X includes the intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = (res @ res / dof) * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def cell_beyond_shared(obs: pd.DataFrame, programs, shared: str, conds, sample_fe: bool):
    """Per cell, within the given conditions: z(program) ~ KC + z(shared) [+ sample dummies].
    beta_KC is the KC-minus-recruited difference in SD units of the program at equal shared
    inflammatory score. Cells are not replicates: effect sizes only."""
    o = obs[obs.condition.isin(conds) & obs.population.isin([KC, REC])]
    rows = []
    for p in programs:
        d = o[[p, shared, "population", "sample"]].dropna()
        y = (d[p] - d[p].mean()) / d[p].std()
        s = (d[shared] - d[shared].mean()) / d[shared].std()
        kc = (d.population == KC).astype(float).values
        cols = [np.ones(len(d)), kc, s.values]
        if sample_fe and d["sample"].nunique() > 1:
            dm = pd.get_dummies(d["sample"], drop_first=True, dtype=float).values
            cols += list(dm.T)
        X = np.column_stack(cols)
        b, se = _ols(y.values, X)
        b0, _ = _ols(y.values, np.column_stack([np.ones(len(d)), kc] + cols[3:]))
        rows.append({"program": p, "shared_score": shared, "conditions": "+".join(conds),
                     "level": "cells", "beta_kc_unadjusted": b0[1], "beta_kc_adjusted": b[1],
                     "beta_shared": b[2], "n_kc": int(kc.sum()), "n_recruited": int((1 - kc).sum()),
                     "sample_fixed_effects": sample_fe})
    return pd.DataFrame(rows)


def cell_did_beyond_shared(obs, programs, shared, ref, alt):
    """Pooled design (GSE279167): z(program) ~ KC * CLP + z(shared); the interaction is the
    difference between KC-marker-positive and recruited cells in their CLP response at equal
    shared inflammatory score (effect size only)."""
    o = obs[obs.population.isin([KC, REC]) & obs.condition.isin([ref, alt])]
    rows = []
    for p in programs:
        d = o[[p, shared, "population", "condition"]].dropna()
        y = ((d[p] - d[p].mean()) / d[p].std()).values
        s = ((d[shared] - d[shared].mean()) / d[shared].std()).values
        kc = (d.population == KC).astype(float).values
        cl = (d.condition == alt).astype(float).values
        b, _ = _ols(y, np.column_stack([np.ones(len(d)), kc, cl, kc * cl, s]))
        b0, _ = _ols(y, np.column_stack([np.ones(len(d)), kc, cl, kc * cl]))
        rows.append({"program": p, "shared_score": shared, "conditions": f"{alt}_vs_{ref}",
                     "level": "cells_interaction", "beta_kc_unadjusted": b0[3],
                     "beta_kc_adjusted": b[3], "beta_shared": b[4],
                     "n_kc": int(kc.sum()), "n_recruited": int((1 - kc).sum()),
                     "sample_fixed_effects": False})
    return pd.DataFrame(rows)


def sample_beyond_shared(samp: pd.DataFrame, programs, shared, conds):
    """Samples as units (GSE290684). Per-sample population means are residualised on the
    shared score (one linear fit over all sample x population units), then KC minus recruited
    is taken within each sample: mean paired difference, paired d, Wilcoxon signed-rank P."""
    s = samp[samp.condition.isin(conds) & samp.population.isin([KC, REC])]
    rows = []
    for p in programs:
        d = s[["sample", "population", p, shared]].dropna()
        X = np.column_stack([np.ones(len(d)), d[shared].values])
        b, _ = _ols(d[p].values, X)
        d = d.assign(resid=d[p].values - X @ b)
        w = d.pivot(index="sample", columns="population", values=["resid", p, shared]).dropna()
        for kind, col in [("adjusted", "resid"), ("unadjusted", p)]:
            diff = w[(col, KC)] - w[(col, REC)]
            r = {"program": p, "shared_score": shared, "conditions": "+".join(conds),
                 "level": f"samples_{kind}", "mean_kc_minus_recruited": diff.mean(),
                 "paired_d": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan,
                 "n_samples": len(diff), "n_samples_kc_higher": int((diff > 0).sum())}
            if len(diff) >= 5:
                r["p_wilcoxon_signed_rank"] = stats.wilcoxon(diff).pvalue
            rows.append(r)
        sd = w[(shared, KC)] - w[(shared, REC)]
        rows[-1]["shared_kc_minus_recruited"] = sd.mean()
        rows[-2]["shared_kc_minus_recruited"] = sd.mean()
    return pd.DataFrame(rows)


def uncoupling(samp: pd.DataFrame, infl: str, programs, ref: str, alt: str, sample_level: bool):
    """Per population: (inflammation - program) per unit, each score centred on the reference
    group mean, so the index is delta(inflammation) - delta(program) (as in the sorted-cell
    analysis). Samples are units only where sample_level is True."""
    rows = []
    for pop, d in samp.groupby("population"):
        for p in programs:
            dd = d[["condition", infl, p]].dropna()
            r0 = dd[dd.condition == ref]
            a = dd[dd.condition == alt]
            if r0.empty or a.empty:
                continue
            idx_ref = (r0[infl] - r0[infl].mean()) - (r0[p] - r0[p].mean())
            idx_alt = (a[infl] - r0[infl].mean()) - (a[p] - r0[p].mean())
            r = {"population": pop, "inflammation_score": infl, "program": p,
                 "reference": ref, "comparison": alt,
                 "delta_inflammation": a[infl].mean() - r0[infl].mean(),
                 "delta_program": a[p].mean() - r0[p].mean(),
                 "uncoupling_index": idx_alt.mean() - idx_ref.mean(),
                 "n_ref": len(r0), "n_alt": len(a)}
            if sample_level and len(r0) >= 3 and len(a) >= 3:
                r["hedges_g"] = cohen_d(idx_ref, idx_alt) * (1 - 3 / (4 * (len(r0) + len(a)) - 9))
                r["p_mannwhitney"] = stats.mannwhitneyu(idx_alt, idx_ref).pvalue
            r["pattern"] = ("opposite (inflammation up, program down)"
                            if r["delta_inflammation"] > 0 and r["delta_program"] < 0 else
                            "same direction" if np.sign(r["delta_inflammation"]) == np.sign(r["delta_program"])
                            else "opposite (inflammation down, program up)")
            rows.append(r)
    return pd.DataFrame(rows)


def pseudobulk_log2fc(my, ref: str, alt: str, label: str, prior: float = 1.0,
                      min_cpm: float = 5.0) -> pd.DataFrame:
    """Per population: log2((mean CPM alt + prior) / (mean CPM ref + prior)), CPM from
    per-sample pseudobulk sums (one pseudobulk per library). Genes kept if mean CPM >= min_cpm
    in either group."""
    lc, meta = pseudobulk(my, ["condition", "sample", "population"], min_cells=20)
    cpm = (2 ** lc - 1)
    out = []
    for pop, m in meta.groupby("population"):
        a = m[m.condition == alt]["pb_id"]
        r = m[m.condition == ref]["pb_id"]
        if a.empty or r.empty:
            continue
        ca, cr = cpm[a].mean(1), cpm[r].mean(1)
        keep = (ca >= min_cpm) | (cr >= min_cpm)
        lfc = np.log2((ca + prior) / (cr + prior))[keep]
        out.append(pd.DataFrame({"population": pop, "gene": lfc.index, "log2fc": lfc.values,
                                 "mean_cpm_ref": cr[keep].values, "mean_cpm_alt": ca[keep].values,
                                 "n_ref": len(r), "n_alt": len(a), "contrast": label}))
    return pd.concat(out) if out else pd.DataFrame()


# ---------------------------------------------------------------- orchestration
SHARED = "sig_shared_inflammatory"
BEYOND_PROGRAMS = ["sig_HMGB1_vs_LPS_axis", "sig_HMGB1_biased", "sig_LPS_biased",
                   "clearance_curated", "clearance_curated_no_annotation", "phagocytosis_GO",
                   "phagocytosis_GO_no_annotation", "kc_identity", "inflammatory",
                   "inflammation_hallmark_tnfa_nfkb"]
UNC_INFL = ["inflammation_hallmark_tnfa_nfkb", "inflammatory"]
UNC_PROGRAMS = ["clearance_curated", "clearance_curated_no_annotation", "phagocytosis_GO", "kc_identity"]


def run_dataset(gse, myc_obs, pb_c, pb_s, ref, alt, sample_level):
    """myc_obs: collapsed-view cell table with scores; pb_c / pb_s: per-sample population
    tables (pseudobulk programs + mean ULM signatures) for collapsed view and subsets."""
    progs = [p for p in BEYOND_PROGRAMS if p in myc_obs]
    beyond = []
    if SHARED in myc_obs:
        if not sample_level:
            beyond += [cell_beyond_shared(myc_obs, progs, SHARED, [alt], False),
                       cell_beyond_shared(myc_obs, progs, SHARED, [ref], False),
                       cell_did_beyond_shared(myc_obs, progs, SHARED, ref, alt)]
        else:
            beyond += [cell_beyond_shared(myc_obs, progs, SHARED, [ref], True),
                       cell_beyond_shared(myc_obs, progs, SHARED, [ref, alt], True),
                       sample_beyond_shared(pb_c, progs, SHARED, [ref]),
                       sample_beyond_shared(pb_c, progs, SHARED, [alt]),
                       sample_beyond_shared(pb_c, progs, SHARED, [ref, alt])]
    beyond = pd.concat(beyond).assign(dataset=gse) if beyond else pd.DataFrame()
    unc = []
    for view, pb in [("kc_marker_vs_recruited", pb_c), ("myeloid_subsets", pb_s)]:
        for infl in [i for i in UNC_INFL if i in pb]:
            u = uncoupling(pb, infl, [p for p in UNC_PROGRAMS if p in pb], ref, alt, sample_level)
            unc.append(u.assign(view=view))
    unc = pd.concat(unc).assign(dataset=gse, unit="library (sample)" if sample_level
                                else "pooled library (no replicates)")
    return beyond, unc


def replication(beyond: pd.DataFrame) -> pd.DataFrame:
    """KC-minus-recruited difference at equal shared score in septic liver, both datasets:
    GSE279167 CLP library (cells) vs GSE290684 CLP vehicle (cells with sample effects, and
    samples). Same sign = replicated direction."""
    if beyond.empty:
        return pd.DataFrame()
    a = beyond[(beyond.dataset == "GSE279167") & (beyond.level == "cells") & (beyond.conditions == "CLP")]
    b = beyond[(beyond.dataset == "GSE290684") & (beyond.level == "cells") & (beyond.conditions == "CLP_vehicle")]
    c = beyond[(beyond.dataset == "GSE290684") & (beyond.level == "samples_adjusted") & (beyond.conditions == "CLP_vehicle")]
    if a.empty or b.empty or c.empty:
        return pd.DataFrame({"status": ["needs both datasets"]})
    m = a.set_index("program")[["beta_kc_unadjusted", "beta_kc_adjusted"]].add_prefix("GSE279167_CLP_cells_")
    m = m.join(b.set_index("program")[["beta_kc_unadjusted", "beta_kc_adjusted"]].add_prefix("GSE290684_vehicle_cells_"))
    m = m.join(c.set_index("program")[["mean_kc_minus_recruited", "paired_d", "n_samples_kc_higher",
                                       "n_samples", "p_wilcoxon_signed_rank"]].add_prefix("GSE290684_vehicle_samples_adj_"))
    s1 = np.sign(m["GSE279167_CLP_cells_beta_kc_adjusted"])
    s2 = np.sign(m["GSE290684_vehicle_cells_beta_kc_adjusted"])
    s3 = np.sign(m["GSE290684_vehicle_samples_adj_mean_kc_minus_recruited"])
    m["direction_replicated"] = (s1 == s2) & (s2 == s3)
    m["shared_score"] = SHARED
    return m.reset_index()


def compare_sorted(unc: pd.DataFrame, sorted_path) -> pd.DataFrame:
    """Direction of the uncoupling index (Hallmark TNFA/NF-kB inflammation) in sorted bulk
    cells (stage 02, read only) next to the single-cell populations (CLP vs sham)."""
    from pathlib import Path
    rows = []
    p = Path(sorted_path)
    if p.exists():
        s = pd.read_csv(p, sep="\t")
        s = s[(s.analysis == "inflammation_minus_program_per_mouse")]
        prog_map = {"clearance_curated": "clearance_curated", "go_phagocytosis": "phagocytosis_GO",
                    "kc_identity": "kc_identity", "clearance_curated_no_identity": "clearance_curated_no_annotation"}
        for _, r in s[s.program_b.isin(prog_map)].iterrows():
            rows.append({"source": f"sorted bulk {r['unit']}", "program": prog_map[r["program_b"]],
                         "uncoupling_index": r["diff"], "hedges_g": r["hedges_g"],
                         "p": r.get("p_mannwhitney"), "inflammation_score": r["program_a"],
                         "contrast": "CLP vs sham"})
    u = unc[(unc.dataset == "GSE279167") & (unc.view == "kc_marker_vs_recruited")
            & (unc.inflammation_score == "inflammation_hallmark_tnfa_nfkb")]
    for _, r in u.iterrows():
        rows.append({"source": f"scRNA GSE279167 {r['population']}", "program": r["program"],
                     "uncoupling_index": r["uncoupling_index"], "hedges_g": np.nan, "p": np.nan,
                     "inflammation_score": r["inflammation_score"], "contrast": "CLP vs sham (pooled)"})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sign"] = np.sign(out["uncoupling_index"])
        out["agrees_with_sorted_KC_majority"] = np.nan
        for prog, d in out.groupby("program"):
            kc = d[d.source.str.contains("sorted bulk GSE3(?:28774|11736)_KC", regex=True)]
            if len(kc):
                maj = np.sign(kc["uncoupling_index"].sum())
                out.loc[d.index, "agrees_with_sorted_KC_majority"] = d["sign"] == maj
    return out
