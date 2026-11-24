"""Stage 04a — liver time courses, cytokine perturbations and stimulus-signature projection
(GSE224127, whole-liver bulk RNA-seq; the mouse is the unit of replication).

Outputs (results/04_timecourse/): design.tsv, gene_sets_used.tsv, scores_by_sample.tsv,
timecourse_effects.tsv, timing.tsv, kc_recruited_coupling.tsv, perturbation_effects.tsv,
signature_projection.tsv (+ signature_relative.tsv, signature_clp_vs_lps.tsv when the
stage-01 signatures exist), PROVENANCE.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import (ROOT, load_config, results_dir, set_global_seed,  # noqa: E402
                       write_provenance, write_table)
from lib.tc_design import (REFERENCE, perturbation_comparisons,  # noqa: E402
                           timecourse_comparisons)
from lib.tc_io import build_design, input_paths  # noqa: E402
from lib.tc_scores import (add_fdr, contrast_table, expressed_genes, log_cpm,  # noqa: E402
                           score_sets)
from lib.tc_signatures import (add_derived_contrast, axis_names, clp_minus_lps,  # noqa: E402
                               read_signatures, relative_contrast)
from lib.tc_timing import kc_recruited_coupling, timing_table  # noqa: E402


TIMECOURSE_FILES = ("LPS_timecourse", "CLP_d0.5", "CLP_d0.25_d1")


def core_genes(counts, genes, min_cpm):
    """Set genes expressed in all three time-course files: one gene list per score, used
    unchanged in every file so that scores mean the same thing across batches."""
    keep = set(genes)
    for fk in TIMECOURSE_FILES:
        keep &= set(expressed_genes(counts[fk], genes, min_cpm))
    present = set.intersection(*[set(X.index) for X in counts.values()])
    return [g for g in genes if g in keep and g in present]


def score_all(design, counts, cfg, weights=None):
    """Scores for every liver mouse, file by file. Returns wide scores + genes used."""
    ex = cfg["expression"]
    if weights is None:
        sets = {k: core_genes(counts, v, ex["min_mean_cpm"])
                for k, v in cfg["gene_sets"].items()}
        present = set.intersection(*[set(X.index) for X in counts.values()])
        for g in cfg["single_genes"]:
            if g in present:
                sets[f"gene_{g}"] = [g]
        requested = {**cfg["gene_sets"], **{f"gene_{g}": [g] for g in cfg["single_genes"]}}
    else:
        sets = {k: core_genes(counts, list(w.index), ex["min_mean_cpm"])
                for k, w in weights.items()}
        requested = {k: list(w.index) for k, w in weights.items()}
    frames, used_rows = [], []
    for fk, X in counts.items():
        d = design[design.file_key == fk]
        ref = d.loc[d.group == REFERENCE[fk], "column"].tolist()
        S, _ = score_sets(log_cpm(X, ex["pseudocount"]), ref, sets, weights)
        if weights is None:
            S["kc_minus_panmac"] = S["kc_identity"] - S["pan_macrophage"]
        cpm_mean = (X.div(X.sum(0), axis=1) * 1e6).mean(1)
        for k, g in sets.items():
            used_rows.append({"file_key": fk, "set": k, "n_requested": len(requested[k]),
                              "n_used": len(g), "genes_used": ",".join(g),
                              "mean_cpm_in_file": cpm_mean[g].mean() if g else np.nan})
        frames.append(S)
    return pd.concat(frames), pd.DataFrame(used_rows)


def long_scores(S, design):
    m = S.reset_index().rename(columns={"index": "column"}).melt(
        id_vars="column", var_name="score", value_name="value")
    keep = ["column", "file_key", "model", "group", "genotype", "time_d", "mouse"]
    return design[keep].merge(m, on="column")


def main():
    cfg = load_config("config/04_timecourse_spatial.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    st = cfg["stats"]
    out = results_dir(cfg["stage_timecourse"])

    design, counts = build_design(cfg, ROOT)
    design["reference_group"] = design.file_key.map(REFERENCE)
    S, used = score_all(design, counts, cfg)

    tc = timecourse_comparisons(design)
    eff = add_fdr(contrast_table(S, design, tc), by=["model"])
    timing = timing_table(S, eff, tc, st["fdr"], rng, st["n_boot"])
    coupling = kc_recruited_coupling(S, design, eff, st["fdr"])

    pc = perturbation_comparisons(design)
    pert = add_fdr(contrast_table(S, design, pc), by=["comparison"])

    outputs = {
        "design.tsv": design, "gene_sets_used.tsv": used,
        "scores_by_sample.tsv": long_scores(S, design),
        "timecourse_effects.tsv": eff, "timing.tsv": timing,
        "kc_recruited_coupling.tsv": coupling, "perturbation_effects.tsv": pert,
    }

    sig_path = ROOT / cfg["signatures"]["path"]
    sigs = read_signatures(sig_path)
    if sigs is None:
        outputs["signature_projection.tsv"] = pd.DataFrame(
            [{"status": "pending", "detail": f"{cfg['signatures']['path']} not found"}])
        note_sig = "signatures pending"
    else:
        Sg, used_sig = score_all(design, counts, cfg, weights=sigs)
        ax = axis_names(sigs)
        Sg = add_derived_contrast(Sg, ax)
        proj = pd.concat([
            contrast_table(Sg, design, tc).assign(layer="timecourse"),
            contrast_table(Sg, design, pc).assign(layer="perturbation")], ignore_index=True)
        proj = add_fdr(proj, by=["layer"])
        rel, boots = relative_contrast(Sg, tc + pc, ax, rng, st["n_boot"])
        pairs = [(0.25, 0.25), (0.5, 0.5), (1.0, 1.0)]
        outputs.update({
            "signature_projection.tsv": proj, "signature_relative.tsv": rel,
            "signature_clp_vs_lps.tsv": clp_minus_lps(boots, pairs),
            "signature_genes_used.tsv": used_sig,
            "signature_scores_by_sample.tsv": long_scores(Sg, design)})
        note_sig = f"signatures projected: {sorted(sigs)}; axes {ax}"

    paths = []
    for name, df in outputs.items():
        paths.append(write_table(df, out / name))
    ins = input_paths(cfg, ROOT) + ([sig_path] if sigs is not None else [])
    write_provenance(cfg["stage_timecourse"], inputs=ins, outputs=paths, cfg=cfg,
                     notes="GSE224127 liver; each count file analysed separately; "
                           "mouse is the unit; Welch tests with BH FDR; " + note_sig)
    print(f"[04_timecourse] wrote {len(paths)} tables to {out}")


if __name__ == "__main__":
    main()
