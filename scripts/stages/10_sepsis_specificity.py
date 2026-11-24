#!/usr/bin/env python
"""Stage 10 — are Kupffer-cell (KC) vs recruited-macrophage differences specific to sepsis, and
is the tolerance-associated component robust at the level of single animals?

1  Sorted cells (GSE311736, CLP 8 h vs sham, KC and CD45+ non-KC): change of the KC minus CD45+
   non-KC gap with sepsis, within-fraction changes (and KC at 2 h, GSE328774), composition
   markers of the CD45+ non-KC fraction, and whether replicate k of the fractions comes from one
   mouse.
2  Single cell (GSE279167, pooled sham vs pooled CLP): KC minus recruited gap per condition and its
   change; cell bootstrap (cell sampling only); adjustment for the shared inflammatory score.
3  Healthy-liver baseline: checked in data/raw only.
4  Tolerance: (a) gain of the tolerance references ranked among the gains of every other real
   reference; (b) per-mouse tolerance gain with a permutation null and sham noise controls.
Outputs in results/10_sepsis_specificity/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import (load_config, project_root, results_dir, set_global_seed,  # noqa: E402
                               write_provenance, write_table)
from scripts.lib.sorted_io import gene_annotation, load_gse311736, load_gse328774  # noqa: E402
from scripts.lib import spec_sorted as SS  # noqa: E402
from scripts.lib import spec_sc as SC  # noqa: E402
from lib import spec_tol as TL  # noqa: E402
from lib.attr_panel import load_or_build  # noqa: E402
from lib import attr_targets as tg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("10_sepsis_specificity")
ROOT = project_root()


def score_inputs(cfg):
    gs = pd.read_csv(ROOT / cfg["inputs"]["gene_sets"], sep="\t").set_index("set")
    sets = {k: gs.loc[k, "genes"].split(",") for k in cfg["scores"]["gene_sets"]}
    sig = pd.read_csv(ROOT / cfg["inputs"]["signatures"], sep="\t")
    weights = {}
    for s in (cfg["scores"]["shared_signature"], cfg["scores"]["axis_signature"]):
        d = sig[sig.signature == s].drop_duplicates("gene")
        sets[s] = list(d.gene)
        weights[s] = d.set_index("gene").weight.astype(float)
    return sets, weights, sig


def part1(cfg, cfg02, sets, weights, rng, save):
    genes, _ = gene_annotation(cfg02["data"]["gtf"])
    c8, m8 = load_gse311736(cfg02, genes)
    c2, m2 = load_gse328774(cfg02, genes)
    sc_ = cfg["sorted"]
    myl = m8[m8.cell_type.isin(["KC", "CD45_nonKC"])]
    S8, n8 = SS.library_scores(c8, myl, sets, weights, sc_)
    S2, n2 = SS.library_scores(c2, m2, sets, weights, sc_)
    scores = list(sets)
    S_all = pd.concat([S8.assign(normalisation="GSE311736 KC + CD45+ non-KC jointly"),
                       S2.assign(normalisation="GSE328774 KC")])
    save(S_all.reset_index(names="sample"), "sorted_library_scores.tsv")
    save(pd.DataFrame([{"dataset": d, "score": k, "n_genes_used": v} for d, n in
                       (("GSE311736", n8), ("GSE328774", n2)) for k, v in n.items()]),
         "sorted_genes_used.tsv")
    inter = SS.interaction(S8, scores, sc_["n_boot"], rng)
    adj = SS.adjusted_interaction(S8, scores, cfg["scores"]["shared_signature"], sc_["n_boot"], rng)
    save(pd.concat([inter, adj], ignore_index=True), "sorted_kc_vs_cd45_interaction.tsv")
    within = SS.within_changes({"KC_8h_GSE311736": S8[S8.cell_type == "KC"],
                                "CD45nonKC_8h_GSE311736": S8[S8.cell_type == "CD45_nonKC"],
                                "KC_2h_GSE328774": S2}, scores, sc_["n_boot"], rng)
    save(within, "sorted_within_fraction_change.tsv")
    save(SS.composition(c8, myl, sc_["composition_markers"], sc_), "sorted_cd45_composition_markers.tsv")
    pairs, comb = SS.pairing_evidence(c8, m8, sc_, rng)
    save(pairs, "sorted_replicate_pairing_by_fraction.tsv")
    save(comb, "sorted_replicate_pairing_combined.tsv")
    save(metadata_pairing(), "sorted_replicate_pairing_metadata.tsv")
    return {"GSE311736": (c8, m8), "GSE328774": (c2, m2)}


def metadata_pairing() -> pd.DataFrame:
    """What the GSE311736 series matrix states about animals (read from the deposited record)."""
    from scripts.lib.sorted_io import read_series_matrix
    cfg02 = load_config("config/02_sorted_bulk.yaml")
    sm = read_series_matrix(cfg02["data"]["GSE311736"]["series_matrix"])
    ch = [c for c in sm.columns if c.startswith("ch_")]
    animal_fields = [c for c in sm.columns if any(k in c.lower() for k in ("mouse", "animal", "individual", "subject", "donor"))]
    return pd.DataFrame([
        {"item": "characteristics fields", "value": ",".join(ch)},
        {"item": "fields naming an animal / individual", "value": ",".join(animal_fields) or "none"},
        {"item": "titles", "value": "purified <fraction>, <CLP|Sham>, replicate <1-4>"},
        {"item": "distinct BioSample per library",
         "value": str(sm.filter(like="relation").iloc[:, 0].nunique() == len(sm))},
        {"item": "treatment protocol", "value": sm["treatment_protocol_ch1"].iloc[0]},
        {"item": "conclusion", "value": "GEO does not record which mouse each library came from; "
                                        "replicate numbers are not documented as animal identifiers"}])


def part2(cfg, cfg03, rng, save):
    import scanpy as sc
    from scripts.lib.sc_preprocess import score_sets
    from scripts.lib.sc_scores import ulm_scores
    s = cfg["single_cell"]
    my = sc.read_h5ad(ROOT / cfg["inputs"]["sc_myeloid"])
    gs = pd.read_csv(ROOT / cfg["inputs"]["gene_sets"], sep="\t").set_index("set")
    sets = {k: gs.loc[k, "genes"].split(",") for k in cfg["scores"]["gene_sets"]}
    net = pd.read_csv(ROOT / cfg["inputs"]["signatures"], sep="\t")[["signature", "gene", "weight"]].dropna()
    sh, ax = cfg["scores"]["shared_signature"], cfg["scores"]["axis_signature"]
    U = ulm_scores(my, net[net.signature.isin([sh, ax])])
    U.columns = [f"sig_{c}" for c in U.columns]
    my.obs = my.obs.join(U)
    score_sets(my, sets, cfg["seed"])
    groups = SC.assign_groups(my.obs, s["kc_populations"], s["comparators"])
    counts = my.obs.assign(group=groups).groupby(["condition", "population", "group"], observed=True,
                                                 dropna=False).size().rename("n_cells").reset_index()
    save(counts, "sc_cell_counts.tsv")
    scorer = SC.PseudobulkScorer(my, groups, sets, U)
    gap = SC.gap_table(scorer, list(s["comparators"]), s["n_boot"], rng)
    save(gap, "sc_kc_vs_recruited_gap.tsv")
    progs = list(sets) + [f"sig_{ax}"]
    adj = SC.adjusted_did(my.obs, groups, list(s["comparators"]), progs, f"sig_{sh}",
                          s["n_boot_cells_ols"], rng)
    save(adj, "sc_kc_vs_recruited_interaction_cells.tsv")


def part3(cfg, save):
    raw = ROOT / "data/raw"
    present = sorted(p.name for p in raw.iterdir())
    rows = [{"accession": a, "present_in_data_raw": any(a in p for p in present)}
            for a in cfg["baseline"]["candidates"]]
    rows.append({"accession": "data/raw contents", "present_in_data_raw": ",".join(present)})
    rows.append({"accession": "conclusion", "present_in_data_raw":
                 "no healthy-liver dataset with KC and monocyte-derived macrophages is downloaded; "
                 "the sham arms of GSE311736 and GSE279167 are the within-study baselines; "
                 "GSE290684 is CLP only"})
    save(pd.DataFrame(rows), "baseline_reference.tsv")


def part4(cfg, rng, save, sorted_data):
    cfg06, cfg01, cfg09 = (load_config(cfg["inputs"][k]) for k in ("config06", "config01", "config09"))
    rs = load_or_build(cfg06, cfg01, ROOT, ROOT / "data/processed/06_attribution")
    cls = {k: v["cls"] for k, v in cfg06["references"].items()}
    ps, tc = cfg09["panel_sensitivity"], cfg["tolerance"]
    targets = tg.stage02_targets(cfg06, ROOT) + tg.stage03_targets(cfg06, ROOT)
    g, s = TL.real_reference_gains(targets, rs, ps, tc, cls, rng)
    save(g, "tolerance_real_reference_gains.tsv")
    save(s, "tolerance_real_reference_rank.tsv")
    log.info("4a done")
    pm = []
    for unit, u in tc["per_mouse"]["units"].items():
        c, m = sorted_data[u["dataset"]]
        m = m[m.cell_type == u["cell_type"]]
        vec = TL.per_mouse_vectors(c, m, tc["per_mouse"]["min_mean_count"])
        pm.append(TL.per_mouse_gains(vec, unit, rs, ps, tc, rng))
        log.info("4b %s done", unit)
    pm = pd.concat(pm, ignore_index=True)
    save(pm, "tolerance_per_mouse.tsv")
    save(TL.per_mouse_summary(pm), "tolerance_per_mouse_summary.tsv")
    return [ROOT / cfg06["targets"]["stage02"]["dir"] / v["file"]
            for v in cfg06["targets"]["stage02"]["files"].values()]


def main():
    cfg = load_config("config/10_sepsis_specificity.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = results_dir(cfg["stage"])
    outputs = []

    def save(df, name):
        outputs.append(write_table(df, out / name))

    cfg02 = load_config(cfg["inputs"]["config02"])
    cfg03 = load_config(cfg["inputs"]["config03"])
    sets, weights, _ = score_inputs(cfg)
    sorted_data = part1(cfg, cfg02, sets, weights, rng, save)
    log.info("part 1 done")
    part2(cfg, cfg03, rng, save)
    log.info("part 2 done")
    part3(cfg, save)
    extra_inputs = part4(cfg, rng, save, sorted_data)
    inputs = [ROOT / cfg["inputs"][k] for k in ("gene_sets", "signatures", "sc_myeloid")]
    inputs += [ROOT / cfg02["data"][d][f] for d in ("GSE311736", "GSE328774") for f in ("counts", "series_matrix")]
    inputs += extra_inputs + [ROOT / "data/processed/03_GSE279167.h5ad"]
    write_provenance(cfg["stage"], inputs=sorted(set(map(str, inputs))), outputs=outputs, cfg=cfg,
                     notes="KC vs CD45+ non-KC / recruited gap in sham and CLP (sorted GSE311736, "
                           "GSE328774; pooled GSE279167 with cell bootstrap); replicate pairing; "
                           "tolerance gain ranked among real references and per mouse.")
    log.info("done")


if __name__ == "__main__":
    main()
