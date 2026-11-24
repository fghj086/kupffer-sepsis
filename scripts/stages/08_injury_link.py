#!/usr/bin/env python
"""Stage 08 — do Kupffer-cell identity, clearance, recruited-macrophage and resolution-like
programs relate to hepatocyte injury in septic liver beyond the shared inflammatory magnitude
and macrophage abundance? (GSE224127 whole liver; mouse is the unit.)

1. severity dose-response across CLP grades (within file and time point)
2. hepatocyte acute-phase and function-loss scores regressed on each program, adjusted for the
   NF-kB/inflammatory score and pan-macrophage abundance (CLP and LPS mice separately)
3. arm-level tracking across cytokine injections and knockout/blockade arms (effect sizes only)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import (load_config, project_root, read_table, results_dir,  # noqa: E402
                       set_global_seed, write_provenance, write_table)
from lib import injury_scores as isc  # noqa: E402
from lib import injury_stats as ist  # noqa: E402
from lib import injury_composition as icomp  # noqa: E402
from lib.tc_io import input_paths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("08_injury_link")
ROOT = project_root()


def main():
    cfg = load_config("config/08_injury_link.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    cfg04 = load_config(cfg["inputs"]["config_04"])
    out = results_dir(cfg["stage"])
    outputs = []

    def save(df, name):
        outputs.append(write_table(df, out / name))

    rs, ref_cache = isc.load_refset(ROOT / cfg["inputs"]["refset_dir"])
    S, design, genes = isc.compute_scores(cfg, cfg04, ROOT, rs)
    chk = isc.check_against_stage04(S, read_table(ROOT / cfg["inputs"]["scores_04"]),
                                    list(cfg04["gene_sets"].keys()))
    if (chk.max_abs_diff > 1e-4).any():
        raise SystemExit(f"recomputed scores differ from stage 04:\n{chk}")
    M = isc.mouse_table(S, design)
    M["stratum"] = M.file_key + "_d" + M.time_d.astype(str)
    sc = cfg["scores"]
    score_names = sc["outcomes"] + sc["programs"] + [sc["inflammation"], sc["composition"]]
    save(genes, "gene_sets_used.tsv")
    save(chk, "check_vs_stage04.tsv")
    save(M[["column", "file_key", "model", "group", "genotype", "time_d", "mouse", "stratum"]
           + score_names], "scores_by_mouse.tsv")

    # 1. dose-response in CLP
    clp = M[M.file_key.isin(cfg["clp"]["files"])].reset_index(drop=True)
    dr = ist.dose_response(clp, score_names, cfg["clp"]["grade"], rng, cfg["stats"]["n_boot"],
                           sc["inflammation"])
    save(dr, "dose_response.tsv")
    log.info("dose-response done (%d CLP-file mice)", len(clp))

    # 2. injury link beyond inflammation and abundance
    lps = M[M.file_key.isin(cfg["lps"]["files"]) & M.group.isin(cfg["lps"]["groups"])
            & (M.genotype == "Wild type")].reset_index(drop=True)
    tabs = []
    for cohort, D in [("CLP", clp), ("LPS", lps)]:
        tabs.append(ist.injury_table(D, cohort, cfg, rng))
        tabs.append(ist.joint_table(D, cohort, cfg, rng))
        log.info("%s injury models done (%d mice)", cohort, len(D))
    save(pd.concat(tabs, ignore_index=True), "injury_models.tsv")

    # 3. perturbation arms as quasi-interventions
    A = ist.arm_shifts(M, {k: v for k, v in cfg["perturbations"].items()}, score_names)
    save(A, "perturbation_shifts.tsv")
    save(ist.tracking(A, cfg, rng), "perturbation_tracking.tsv")

    # 1b. severity slopes adjusted for leukocyte composition of the bulk liver
    P, pgenes = icomp.proxy_scores(cfg, cfg04, ROOT)
    Mc = M.drop(columns=[c for c in P.columns if c in M]).merge(
        P, left_on="column", right_index=True, how="left")
    Mc["kc_identity_per_macrophage"] = Mc.kc_identity - Mc.pan_macrophage
    save(pgenes, "composition_gene_sets.tsv")
    save(Mc[["column", "file_key", "group", "time_d", "mouse", "stratum"]
            + list(P.columns) + ["kc_identity_per_macrophage"]], "composition_scores_by_mouse.tsv")
    clp_c = Mc[Mc.file_key.isin(cfg["clp"]["files"])].reset_index(drop=True)
    rng_c = np.random.default_rng(cfg["seed"] + cfg["composition_seed_offset"])
    save(icomp.adjusted_dose_response(clp_c, cfg["composition_outcomes"], cfg["composition_models"],
                                      cfg["clp"]["grade"], rng_c, cfg["stats"]["n_boot"]),
         "dose_response_composition.tsv")
    log.info("composition-adjusted dose-response done")

    inputs = (input_paths(cfg04, ROOT) + [ROOT / cfg["inputs"]["design_04"],
              ROOT / cfg["inputs"]["scores_04"], ref_cache])
    write_provenance(cfg["stage"], inputs=inputs, outputs=outputs, cfg=cfg,
                     notes="GSE224127 liver, mouse-level scores; resolution-like programs scored "
                           "with stage 06 reference vectors (outcome, NF-kB and identity genes removed).",
                     extra={"n_mice_clp": len(clp), "n_mice_lps": len(lps), "n_arms": len(A)})


if __name__ == "__main__":
    main()
