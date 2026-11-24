#!/usr/bin/env python
"""Stage 00b - leukocyte proportions of septic whole-blood samples by reference deconvolution.

Reference: GSE60424 neutrophils, monocytes, B, CD4, CD8 and NK cells sorted from the whole blood of
healthy donors and patients with sepsis (lib/blood_deconv.py). Each cohort's platform is calibrated on
its healthy controls (GSE185263 healthy donors for GSE222393) with a typical adult leukocyte
differential, and every septic sample is fitted by non-negative least squares on the cell-type
signature genes; proportions sum to 1.

Outputs (results/00b_blood_deconvolution/)
  sample_values.tsv        sample, cohort, patient, day, prop_<cell type>
  signature_genes.tsv, signature_coverage.tsv, reference_profiles.tsv, PROVENANCE.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import blood_cohorts, blood_deconv as deconv, repro  # noqa: E402

ROOT = repro.project_root()
TYPES = [deconv.LABEL[t] for t in deconv.TYPES]


def linear(e: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Back to linear units: log2(x) for arrays, log2(CPM + 1) for the RNA-seq cohort."""
    return (2 ** e) - (1 if cohort == "GSE222393" else 0)


def controls(cohort: str, e: pd.DataFrame, samples: pd.DataFrame, healthy222: pd.DataFrame) -> pd.DataFrame:
    if cohort == "GSE222393":
        return healthy222
    return linear(e[samples.index[samples.group == "control"]], cohort)


def main() -> None:
    cfg = repro.load_config("config/00b_blood_deconvolution.yaml")
    repro.set_global_seed(cfg["seed"])
    out = repro.results_dir(cfg["stage"])
    dc = cfg["deconvolution"]
    raw = ROOT / cfg["raw"]
    proc = ROOT / cfg["processed"]
    tabs = ROOT / cfg["cohort_tables"]

    sym = blood_cohorts._ensembl_to_symbol(raw / "GSE279452/GSE279452_features.tsv.gz")
    ref = deconv.reference(raw / dc["reference"], sym, tuple(dc["donors"]))
    prior = pd.Series(dc["healthy_prior"])
    prior = prior / prior.sum()
    healthy222 = deconv.gse185263_healthy(raw, sym)
    sig = deconv.signature(ref, int(dc["genes_per_type"]), float(dc["min_margin"]))

    props, cov, inputs = [], [], []
    for c in cfg["cohorts"]:
        e = pd.read_csv(proc / f"{c}_expr.tsv.gz", sep="\t", index_col=0)
        s = pd.read_csv(tabs / f"{c}_samples.tsv", sep="\t", index_col=0)
        inputs += [proc / f"{c}_expr.tsv.gz", tabs / f"{c}_samples.tsv"]
        pts = s[s["group"] != "control"]              # septic samples
        ctrl = controls(c, e, s, healthy222)
        cref = deconv.calibrate(ref, ctrl, prior, sig)
        p = deconv.deconvolve(linear(e[pts.index], c), cref, sig).rename(columns=deconv.LABEL)
        props.append(p.add_prefix("prop_").assign(cohort=c, patient=pts["patient"], day=pts["day"]))
        cov.append({"cohort": c, "signature_genes_measured": sum(x in cref.index for x in sig),
                    "signature_genes": len(sig), "healthy_controls": ctrl.shape[1]})
        print(c, len(p), p.mean().round(3).to_dict())
    props = pd.concat(props)
    props = props[["cohort", "patient", "day", *[f"prop_{t}" for t in TYPES]]]

    paths = [repro.write_table(props.reset_index(names="sample"), out / "sample_values.tsv"),
             repro.write_table(pd.DataFrame(cov), out / "signature_coverage.tsv"),
             repro.write_table(pd.DataFrame({"gene": sig}), out / "signature_genes.tsv"),
             repro.write_table(ref.loc[sig].rename(columns=deconv.LABEL).reset_index(names="gene"),
                               out / "reference_profiles.tsv")]
    repro.write_provenance(cfg["stage"], inputs, paths, cfg,
                           notes="GSE60424-referenced NNLS deconvolution, per-platform calibration on healthy controls")


if __name__ == "__main__":
    main()
