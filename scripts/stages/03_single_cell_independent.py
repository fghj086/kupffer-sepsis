"""Stage 03 add-on: KC-marker-positive vs recruited myeloid cells within each GSE290684 liver,
with scores that share no gene with the annotation markers.

* identity_not_annotation: KC identity genes never used for any cell label
  (programs.kc_identity minus every cell_type_markers / myeloid_subsets gene);
* clearance_strict: curated clearance genes minus every annotation gene, every identity gene
  and C1qc (paralogue of the annotation markers C1qa/C1qb);
* the original scores (kc_identity, clearance_curated, clearance_curated_no_annotation,
  inflammatory) and the HMGB1-vs-LPS axis for reference.
Libraries (livers) are the units: per liver, KC minus recruited (pseudobulk centred logCPM;
mean per-cell ULM t for signatures), unadjusted and residualised on the shared inflammatory
score as in stage 03. Vehicle livers are the main analysis; TPPU and all 12 are reported
alongside. Also: fraction of macrophages labelled KC-marker-positive per liver.
Run after 03_single_cell_analyse.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib import repro  # noqa: E402
from scripts.lib import sc_tables as T  # noqa: E402
from scripts.lib.sc_beyond import KC, REC  # noqa: E402
from scripts.lib.sc_scores import ulm_scores  # noqa: E402


def build_sets(base: dict, extra_exclude) -> tuple[dict, pd.DataFrame]:
    ann = set()
    for v in base["cell_type_markers"].values():
        ann |= set(v)
    for v in base["myeloid_subsets"].values():
        if isinstance(v, list):
            ann |= set(v)
    prog = base["programs"]
    ident = set(prog["kc_identity"])
    sets = {k: list(prog[k]) for k in ["kc_identity", "clearance_curated", "inflammatory"]}
    old_ann = ident.copy()
    for v in base["myeloid_subsets"].values():
        if isinstance(v, list):
            old_ann |= set(v)
    sets["clearance_curated_no_annotation"] = [g for g in prog["clearance_curated"] if g not in old_ann]
    sets["identity_not_annotation"] = [g for g in prog["kc_identity"] if g not in ann]
    sets["clearance_strict"] = [g for g in prog["clearance_curated"]
                                if g not in ann | ident | set(extra_exclude)]
    rows = [{"set": k, "n_genes": len(v), "genes": ",".join(v),
             "shares_gene_with_annotation": bool(set(v) & ann)} for k, v in sets.items()]
    return sets, pd.DataFrame(rows)


def paired_rows(pb: pd.DataFrame, scores, shared: str, conds, label: str) -> list[dict]:
    s = pb[pb.condition.isin(conds) & pb.population.isin([KC, REC])]
    rows = []
    for p in scores:
        d = s[list(dict.fromkeys(["sample", "population", p, shared]))].dropna()
        if p != shared:
            X = np.column_stack([np.ones(len(d)), d[shared].values])
            b = np.linalg.lstsq(X, d[p].values, rcond=None)[0]
            d = d.assign(resid=d[p].values - X @ b)
            kinds = [("adjusted_for_shared", "resid"), ("unadjusted", p)]
        else:
            kinds = [("unadjusted", p)]
        for kind, col in kinds:
            w = d.pivot(index="sample", columns="population", values=col).dropna()
            diff = (w[KC] - w[REC]).values
            n = len(diff)
            m, sd = diff.mean(), diff.std(ddof=1)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            rows.append({"condition_set": label, "conditions": "+".join(conds), "score": p,
                         "adjustment": kind, "n_livers": n,
                         "mean_kc_minus_recruited": m, "ci_low": m - half, "ci_high": m + half,
                         "paired_d": m / sd if sd > 0 else np.nan,
                         "n_livers_kc_higher": int((diff > 0).sum()),
                         "p_wilcoxon_signed_rank": stats.wilcoxon(diff).pvalue if n >= 5 else np.nan})
    return rows


def kc_fraction(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = obs.assign(pop=obs.population.astype(str))
    rows = []
    for (cond, smp), g in o.groupby(["condition", "sample"], observed=True):
        mac = g[g["pop"].isin(T.MACRO)]
        allm = g[g["pop"].isin(T.MACRO + T.MONO)]
        kc = mac["pop"].isin(["Resident KC", T.KC_LIKE])
        rows.append({"condition": cond, "sample": smp, "n_macrophages": len(mac),
                     "n_macrophages_plus_monocytes": len(allm),
                     "frac_kc_marker_positive_of_macrophages": kc.mean(),
                     "frac_resident_kc_of_macrophages": (mac["pop"] == "Resident KC").mean(),
                     "frac_kc_marker_positive_of_macrophages_plus_monocytes": kc.sum() / len(allm)})
    per = pd.DataFrame(rows)
    summ = []
    for c in [c for c in per.columns if c.startswith("frac_")]:
        a = per.loc[per.condition == "CLP_vehicle", c]
        b = per.loc[per.condition == "CLP_TPPU", c]
        summ.append({"measure": c, "vehicle_mean": a.mean(), "vehicle_min": a.min(),
                     "vehicle_max": a.max(), "tppu_mean": b.mean(), "tppu_min": b.min(),
                     "tppu_max": b.max(), "n_vehicle": len(a), "n_tppu": len(b),
                     "p_mannwhitney": stats.mannwhitneyu(a, b).pvalue})
    return per, pd.DataFrame(summ)


def main():
    cfg = repro.load_config("config/03_single_cell_independent.yaml")
    base = repro.load_config(cfg["base_config"])
    repro.set_global_seed(cfg["seed"])
    root = repro.project_root()
    out = repro.results_dir(cfg["stage"])
    ic = cfg["independent"]
    h5 = root / base["processed_dir"] / f"03_{ic['dataset']}_myeloid.h5ad"
    my = sc.read_h5ad(h5)
    sets, gs = build_sets(base, ic["extra_exclude"])
    sig_path = root / base["signatures_path"]
    net = pd.read_csv(sig_path, sep="\t")[["signature", "gene", "weight"]].dropna()
    net = net[net.signature.isin([ic["shared_signature"], ic["axis_signature"]])]
    S = ulm_scores(my, net)
    S.columns = [f"sig_{c}" for c in S.columns]
    my.obs = my.obs.join(S)
    per, summ = kc_fraction(my.obs)

    myc = my[my.obs["population"].astype(str).map(T.COLLAPSE).notna()].copy()
    myc.obs["population"] = myc.obs["population"].astype(str).map(T.COLLAPSE)
    pb = T.pseudobulk_scores(myc, sets)
    sm = myc.obs.groupby(["condition", "sample", "population"], observed=True)[list(S.columns)].mean().reset_index()
    pb = pb.merge(sm, on=["condition", "sample", "population"], how="left")
    shared = f"sig_{ic['shared_signature']}"
    scores = list(sets) + [f"sig_{ic['axis_signature']}", shared]
    rows = []
    for label, conds in ic["condition_sets"].items():
        rows += paired_rows(pb, scores, shared, conds, label)
    res = pd.DataFrame(rows)
    res = res.merge(gs.rename(columns={"set": "score"})[["score", "n_genes", "genes"]], on="score", how="left")

    outputs = [repro.write_table(res, out / "kc_vs_recruited_independent.tsv"),
               repro.write_table(gs, out / "kc_vs_recruited_independent_gene_sets.tsv"),
               repro.write_table(per, out / "kc_fraction_per_liver.tsv"),
               repro.write_table(summ, out / "kc_fraction_vehicle_vs_tppu.tsv")]
    repro.append_provenance(cfg["stage"], "03_single_cell_independent", inputs=[h5, sig_path],
                            outputs=outputs, cfg=cfg,
                            notes="KC vs recruited per GSE290684 liver with annotation-independent "
                                  "identity and clearance scores; KC fraction per liver.")
    print(res.round(3).to_string())
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
