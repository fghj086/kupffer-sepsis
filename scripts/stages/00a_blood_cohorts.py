#!/usr/bin/env python
"""Stage 00a - gene matrices and harmonised sample tables of the human sepsis blood cohorts.

Each cohort is processed on its own (no cross-platform merging). Repeatedly sampled cohorts pass a
sample identity check: genetic sex (chromosome Y genes minus XIST) must agree across a person's
samples; samples that disagree are removed, and a sex annotation contradicted by all of a person's
samples is replaced by the genetic call.

Outputs
  data/human/processed/<cohort>_expr.tsv.gz        genes x samples, log2
  results/00a_blood_cohorts/<cohort>_samples.tsv
  results/00a_blood_cohorts/sample_identity.tsv
  results/00a_blood_cohorts/cohort_summary.tsv
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import blood_cohorts as co  # noqa: E402
from lib import blood_identity as sid  # noqa: E402
from lib import repro  # noqa: E402

ROOT = repro.project_root()
REPEATED = ["GAINS", "GSE95233", "GSE222393", "GSE57065"]


def write_matrix(x: pd.DataFrame, path: Path) -> None:
    text = x.to_csv(sep="\t", float_format="%.5f")
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))


def died_from_group(s: pd.DataFrame) -> pd.Series:
    return s["group"].map({"nonsurvivor": 1, "survivor": 0})


def main() -> None:
    cfg = repro.load_config("config/00a_blood_cohorts.yaml")
    repro.set_global_seed(cfg["seed"])
    raw = ROOT / cfg["raw"]
    plat = raw / "platforms/GPL10558_full.txt"
    proc = ROOT / cfg["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    out = repro.results_dir(cfg["stage"])

    data = {}
    data["GAINS"] = co.gains_5273(raw, plat)
    data["GAINS4421"] = co.gains_4421(raw, plat)
    data["MARS"] = co.mars(raw)
    g, s = co.series_cohort(raw, "GSE95233", "GPL570", co.harmonise_gse95233)
    data["GSE95233"] = (g, s.assign(died=died_from_group(s)))
    g, s = co.series_cohort(raw, "GSE57065", "GPL570", co.harmonise_gse57065)
    data["GSE57065"] = (g, s.assign(died=np.nan))
    data["GSE222393"] = co.gse222393(raw, cfg["preprocess"]["min_mean_cpm"])

    excl = cfg["sample_identity"]["exclude"] or {}
    ident_rows, summary, paths = [], [], []
    for name, (g, s) in data.items():
        s = s.loc[g.columns].copy()
        s.index.name = "sample"
        removed = 0
        if name in REPEATED:
            have_sex = s["sex"].notna()
            sx = sid.genetic_sex(g[s.index[have_sex]], cfg["sample_identity"]["sex_gap_min"], name)
            t = sid.identity_table(s[have_sex], sx)
            t["removed"] = np.where(t["conflict"], "genetic sex differs from the patient", "")
            for k, why in excl.items():
                if k in t.index:
                    t.loc[k, "removed"] = why
            ident_rows.append(t.reset_index().assign(cohort=name))
            drop = t.index[t["removed"] != ""]
            removed = len(drop)
            s.loc[t.index, "sex"] = t["sex"]
            s = s.drop(index=drop)
        g = g[s.index]
        write_matrix(g, proc / f"{name}_expr.tsv.gz")
        paths.append(proc / f"{name}_expr.tsv.gz")
        paths.append(repro.write_table(s.reset_index(), out / f"{name}_samples.tsv"))
        pat = s.loc[s["group"].isin(["survivor", "nonsurvivor"])] if "group" in s else s.iloc[0:0]
        summary.append({"cohort": name, "n_genes": g.shape[0], "n_samples": g.shape[1],
                        "n_patients_with_outcome": pat["patient"].nunique() if len(pat) else np.nan,
                        "n_deaths": pat.loc[pat["died"] == 1, "patient"].nunique() if len(pat) else np.nan,
                        "n_samples_removed_identity": removed})
        print(summary[-1])
    paths.append(repro.write_table(pd.concat(ident_rows, ignore_index=True), out / "sample_identity.tsv"))
    paths.append(repro.write_table(pd.DataFrame(summary), out / "cohort_summary.tsv"))
    repro.write_provenance(cfg["stage"], [ROOT / "config/00a_blood_cohorts.yaml"], paths, cfg,
                           notes="Cohorts processed independently; genetic-sex identity check on repeated cohorts.")


if __name__ == "__main__":
    main()
