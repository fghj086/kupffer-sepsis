#!/usr/bin/env python
"""Stage 06 — how much of each septic-liver response resembles a bacterial-PAMP, a DAMP
(HMGB1-like, tissue damage, heme, dying cells) or a secondary-cytokine stimulus, and how the mix
changes with cell type and time.

Order: build references -> positive controls -> identifiability of the HMGB1-like component ->
attribution of stage 02/03/04/05 contrasts. Outputs in results/06_attribution/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, project_root, results_dir, set_global_seed, write_provenance, write_table  # noqa: E402
from lib import attr_controls as ctl  # noqa: E402
from lib.attr_damp import magnitude_matched, per_mouse_timecourse  # noqa: E402
from lib.attr_io import study_inputs  # noqa: E402
from lib.attr_model import attribute, merge_collinear  # noqa: E402
from lib.attr_panel import informative, load_or_build  # noqa: E402
from lib import attr_targets as tg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("06_attribution")
ROOT = project_root()


def correlations(rs, cfg) -> pd.DataFrame:
    rows = []
    for pname, panel in cfg["panels"].items():
        g = informative(rs, panel)
        M = rs.matrix(panel, g)
        C = np.corrcoef(M.values.T)
        for i, a in enumerate(panel):
            for j, b in enumerate(panel):
                if j <= i:
                    continue
                rows.append({"panel": pname, "reference_a": a, "reference_b": b,
                             "pearson_r": C[i, j], "n_genes": len(g),
                             "collinear_flag": C[i, j] > cfg["model"]["collinear_r"]})
    return pd.DataFrame(rows)


def cross_study(rs, cfg) -> pd.DataFrame:
    """Same stimulus in different studies versus different stimuli within one study."""
    def r_of(a, b):
        g = pd.Index(sorted(set(rs.genes[a]) | set(rs.genes[b])))
        M = rs.matrix([a, b], g).dropna()
        return float(np.corrcoef(M.values.T)[0, 1]), len(M)
    rows = []
    for a, b in cfg["cross_study_pairs"]:
        r, n = r_of(a, b)
        rows.append({"reference_a": a, "reference_b": b, "relation": "same stimulus, different study",
                     "pearson_r": r, "n_genes": n})
    refs = cfg["references"]
    for a in refs:
        for b in refs:
            if a < b and refs[a]["study"] == refs[b]["study"] and "combo" not in (refs[a]["cls"], refs[b]["cls"]):
                r, n = r_of(a, b)
                rows.append({"reference_a": a, "reference_b": b,
                             "relation": f"different stimulus, same study ({refs[a]['study']})",
                             "pearson_r": r, "n_genes": n})
    return pd.DataFrame(rows)


def run_target(t, rs, cfg, panel_name, rng, cls, merge=True):
    panel = cfg["panels"][panel_name]
    g = informative(rs, panel, t["y"])
    r_max = cfg["model"]["collinear_r"] if merge else np.inf
    refs, merged = merge_collinear(rs.matrix(panel, g), r_max)
    signed = t["kind"] == "difference"
    res, s = attribute(t["y"].reindex(g), refs, cfg["model"]["n_boot"], rng, signed=signed,
                       alpha=cfg["model"]["ridge_alpha"])
    res["stimulus_class"] = [";".join(sorted({cls[x] for x in r.split("+")})) for r in res.reference]
    meta = {k: t.get(k) for k in ("target", "source", "cell", "time_h", "family", "kind")}
    label = panel_name if merge else f"{panel_name}_unmerged"
    for k, v in {**meta, "panel": label, **s,
                 "merged": ";".join(f"{a}+{b}(r={r:.2f})" for a, b, r, _ in merged)}.items():
        res[k] = v
    return res


def class_table(att: pd.DataFrame) -> pd.DataFrame:
    a = att[att.fit == "nnls"].copy()
    a["stimulus_class"] = a.stimulus_class.where(~a.stimulus_class.str.contains(";"), "mixed")
    out = a.groupby(["target", "panel", "stimulus_class"], as_index=False).agg(
        share=("share", "sum"), delta_r2=("delta_r2", "sum"))
    keep = a.drop_duplicates(["target", "panel"])[["target", "panel", "cell", "time_h",
                                                   "r2_total", "r2_shared_axis"]]
    return out.merge(keep, on=["target", "panel"])


def main():
    cfg = load_config("config/06_attribution.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    cfg01 = load_config("config/01_perturbation.yaml")
    out = results_dir(cfg["stage"])
    rs = load_or_build(cfg, cfg01, ROOT, ROOT / "data/processed/06_attribution")
    cls = {k: v["cls"] for k, v in cfg["references"].items()}
    outputs = []

    def save(df, name):
        outputs.append(write_table(df, out / name))

    # ---- references
    save(pd.DataFrame(cfg["candidates"], columns=["accession", "design", "decision", "reason"]),
         "reference_candidates.tsv")
    save(pd.DataFrame(rs.meta), "reference_panel.tsv")
    corr = correlations(rs, cfg)
    save(corr, "reference_correlation.tsv")
    save(cross_study(rs, cfg), "cross_study_similarity.tsv")

    # ---- positive controls (before any target)
    held = ctl.heldout_controls(rs, cfg, rng)
    loro = pd.concat([ctl.loro_controls(rs, cfg, p) for p in cfg["controls_panels"]["loro"]])
    mix = pd.concat([ctl.mixture_controls(rs, cfg, p, rng) for p in cfg["controls_panels"]["mixture"]])
    save(held, "controls_heldout.tsv")
    save(loro, "controls_loro.tsv")
    save(mix, "controls_mixture.tsv")
    save(ctl.summarise_controls(held, loro, mix), "controls.tsv")
    save(magnitude_matched(rs, cfg, rng), "hmgb1_identifiability.tsv")
    log.info("controls done")

    # ---- targets
    targets = (tg.stage02_targets(cfg, ROOT) + tg.stage05_targets(cfg, ROOT)
               + tg.stage04_targets(cfg, ROOT, rs) + tg.stage03_targets(cfg, ROOT))
    log.info("%d targets", len(targets))
    rows = []
    for t in targets:
        for pname in cfg["target_panels"][t["family"]]:
            rows.append(run_target(t, rs, cfg, pname, rng, cls))
            if pname in cfg["model"]["unmerged_sensitivity"] and rows[-1]["merged"].iloc[0]:
                rows.append(run_target(t, rs, cfg, pname, rng, cls, merge=False))
    att = pd.concat(rows, ignore_index=True)
    lead = ["target", "panel", "reference", "stimulus_class", "coefficient", "ci_low", "ci_high",
            "share", "share_ci_low", "share_ci_high", "boot_nonzero", "delta_r2", "r2_total",
            "r2_ci_low", "r2_ci_high", "r2_shared_axis", "n_genes", "fit", "merged"]
    att = att[lead + [c for c in att.columns if c not in lead]]
    save(att, "attribution.tsv")
    save(class_table(att), "attribution_by_class.tsv")
    tt = pd.DataFrame([{k: t.get(k) for k in ("target", "source", "family", "cell", "time_h",
                                              "kind", "n_test", "n_ref", "n_cells_sham",
                                              "n_cells_clp")} | {"n_genes_target": int(t["y"].notna().sum())}
                       for t in targets])
    save(tt, "targets.tsv")

    # ---- over time (GSE224127 time courses): group-level and per mouse
    tc = att[att.target.isin([t["target"] for t in targets if "per_mouse" in t])].copy()
    tmeta = pd.DataFrame([{"target": t["target"], "model": t["model"], "group": t["group"],
                           "time_d": t["time_d"]} for t in targets if "per_mouse" in t])
    ot = class_table(tc).merge(tmeta, on="target")
    hm = tc[tc.reference.str.contains("dsHMGB1")][["target", "panel", "reference", "share",
                                                   "share_ci_low", "share_ci_high", "delta_r2"]]
    hm = hm.rename(columns={"reference": "hmgb1_reference", "share": "hmgb1_share",
                            "share_ci_low": "hmgb1_share_ci_low",
                            "share_ci_high": "hmgb1_share_ci_high", "delta_r2": "hmgb1_delta_r2"})
    ot = ot.merge(hm, on=["target", "panel"], how="left").sort_values(
        ["model", "group", "panel", "time_d", "stimulus_class"])
    save(ot, "attribution_over_time.tsv")
    pm = per_mouse_timecourse(targets, rs, cfg, ["macrophage_resolution", "liver", "liver_plus"],
                              informative, rng)
    save(pm, "attribution_over_time_per_mouse.tsv")

    inputs = study_inputs(cfg, ROOT) + [p for t in targets for p in t.get("inputs", [])]
    inputs += [ROOT / cfg["gse224127"]["design"]]
    write_provenance(cfg["stage"], inputs=sorted(set(map(str, inputs))), outputs=outputs, cfg=cfg,
                     notes="Within-study reference vectors (mean log2 differences); NNLS on unit-RMS "
                           "vectors over responsive genes; uncentred R2; gene bootstrap CIs; "
                           "positive controls run before targets.")
    log.info("done")


if __name__ == "__main__":
    main()
