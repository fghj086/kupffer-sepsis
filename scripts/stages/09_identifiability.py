#!/usr/bin/env python
"""Stage 09 — how identifiable is the stage 06 stimulus attribution under realistic conditions?

A  detection limit of an HMGB1-like component: LPS-like responses from other laboratories,
   GSE200210 disulfide-HMGB1 replicates not used by the panel, real Kupffer-cell mouse-to-mouse
   noise; and the HMGB1-like fraction compatible with the observed macrophage responses.
B  NNLS against non-negative ridge, non-negative elastic net and correlation ranking.
C  leave-one-series-out attribution of every single-stimulus reference (class transfer).
D  nested panels and held-out-gene R2 for the tolerance-associated component.
Reuses the stage 06 reference cache and target builders. Outputs in results/09_identifiability/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.repro import load_config, project_root, results_dir, set_global_seed, write_provenance, write_table  # noqa: E402
from lib.attr_io import study_inputs  # noqa: E402
from lib.attr_panel import load_or_build  # noqa: E402
from lib import attr_targets as tg  # noqa: E402
from lib.ident_noise import load_noise  # noqa: E402
from lib.ident_sim import MixtureSim  # noqa: E402
from lib import ident_detect as det  # noqa: E402
from lib import ident_methods as mth  # noqa: E402
from lib.ident_transfer import leave_one_study_out, transfer_summary  # noqa: E402
from lib.ident_panels import panel_sensitivity, reference_correlations  # noqa: E402
from lib.ident_report import write_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("09_identifiability")
ROOT = project_root()


def observed_table(sim, targets, cfg) -> tuple[pd.DataFrame, dict]:
    by = {t["target"]: t for t in targets}
    rows, obs = [], {}
    for name in cfg["mixture"]["upper_bound_targets"]:
        if name not in by:
            log.warning("upper-bound target %s not built", name)
            continue
        obs[name] = {}
        for p in sim.panels:
            o = sim.observed(by[name]["y"], p)
            obs[name][p] = o
            rows.append({"target": name, "panel": p} | o)
        obs[name]["none"] = {"axis": obs[name][p]["axis"], "hmgb1_share": np.nan}
    return pd.DataFrame(rows), obs


def main():
    cfg = load_config("config/09_identifiability.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    cfg06, cfg01, cfg02 = (load_config(cfg["inputs"][k]) for k in ("config06", "config01", "config02"))
    out = results_dir(cfg["stage"])
    outputs = []

    def save(df, name):
        outputs.append(write_table(df, out / name))

    rs = load_or_build(cfg06, cfg01, ROOT, ROOT / "data/processed/06_attribution")
    cls = {k: v["cls"] for k, v in cfg06["references"].items()}
    noise = load_noise(cfg, cfg02, ROOT)
    sig = pd.read_csv(ROOT / cfg["inputs"]["signatures"], sep="\t")
    sim = MixtureSim(rs, cfg06, cfg, noise, sig)
    log.info("simulation universe %d genes, scale genes %d, axis genes %d/%d",
             len(sim.U), len(sim.S0), len(sim.iH), len(sim.iL))
    save(pd.DataFrame([n.summary(sim.S0) for n in noise.values()]), "noise_model.tsv")

    targets = (tg.stage02_targets(cfg06, ROOT) + tg.stage05_targets(cfg06, ROOT)
               + tg.stage04_targets(cfg06, ROOT, rs) + tg.stage03_targets(cfg06, ROOT))
    obs_tab, obs = observed_table(sim, targets, cfg)
    save(obs_tab, "observed_targets.tsv")

    # ---- A: detection limits
    a = cfg["mixture"]
    rp = a["resid_panel"]
    level = lambda name: float(np.sqrt((1 - obs[name][rp]["r2"]) / obs[name][rp]["r2"]))
    settings = {"none": {"kind": "none"}}
    for key in noise:
        settings[f"mouse_{key}"] = {"kind": "mouse", "pattern": key}
    # "fitresid": all of the observed residual treated as noise (conservative, double counts the
    # cross-laboratory mismatch); "residmatched": noise topped up so that mismatch + noise leaves the
    # observed residual for each LPS-like source
    matched = lambda name: sim.matched_level(obs[name][rp]["r2"], rp)
    for name, key in a["upper_bound_targets"].items():
        if name.startswith("KC_CLP") and name in obs:
            settings[f"fitresid_{key}"] = {"kind": "resid", "pattern": key, "level": level(name)}
            settings[f"residmatched_{key}"] = {"kind": "resid", "pattern": key, "level": matched(name)}
    draws = det.run_draws(sim, settings, a["f_grid"], a["n_draws"], rng)
    dt = det.detection_table(draws, a["null_quantile"], a["fixed_share_threshold"])
    save(dt, "hmgb1_detection.tsv")
    save(det.limits_table(dt, a["power"]), "hmgb1_detection_limits.tsv")
    save(pd.DataFrame([{"setting": k, "kind": v["kind"], "pattern": v.get("pattern"), "source": src,
                        "level_rms": lv} for k, v in settings.items()
                       for src, lv in (v["level"].items() if isinstance(v.get("level"), dict)
                                       else [("all", v.get("level"))])]), "noise_settings.tsv")
    log.info("A detection done (%d draws rows)", len(draws))

    ub = []
    for name, key in a["upper_bound_targets"].items():
        if name not in obs:
            continue
        own = {f"mouse_{key}": settings[f"mouse_{key}"],
               "residmatched_own": {"kind": "resid", "pattern": key, "level": matched(name)},
               "fitresid_own": {"kind": "resid", "pattern": key, "level": level(name)}}
        d = det.run_draws(sim, own, a["f_grid"], a["n_draws_upper_bound"], rng)
        t = det.upper_bounds(d, obs[name], name, a["alpha_upper_bound"])
        lv = own["residmatched_own"]["level"]
        t["noise_level_rms"] = [level(name) if nz == "fitresid_own" else
                                lv.get(src, np.nan) if nz == "residmatched_own" else np.nan
                                for nz, src in zip(t.noise, t.source)]
        ub.append(t)
    save(pd.concat(ub, ignore_index=True), "hmgb1_upper_bound.tsv")
    log.info("A upper bounds done")

    # ---- B: methods
    m = cfg["methods"]
    hm = mth.heldout_methods(rs, cfg06, m, cfg["seed"])
    save(hm, "methods_heldout.tsv")
    save(mth.heldout_summary(hm), "methods_heldout_summary.tsv")
    mm = mth.mixture_methods(sim, {k: settings[k] for k in m["mixture_noise"]}, m["mixture_f_grid"],
                             m["mixture_n_draws"], m["mixture_panel"], m, rng, cfg["seed"])
    save(mth.mixture_method_summary(mm, a["null_quantile"]), "methods_mixture.tsv")
    log.info("B methods done")

    # ---- C: leave one series out
    lo = leave_one_study_out(rs, cfg06, cfg)
    save(lo, "transfer_leave_one_series_out.tsv")
    save(transfer_summary(lo), "transfer_summary.tsv")
    log.info("C transfer done")

    # ---- D: panel sensitivity
    ps, sh = panel_sensitivity(targets, rs, cfg, cls, rng)
    save(ps, "panel_sensitivity_heldout_r2.tsv")
    save(sh, "panel_sensitivity_shares.tsv")
    save(reference_correlations(rs, cfg["panel_sensitivity"]["correlation_refs"]),
         "tolerance_reference_correlation.tsv")
    log.info("D panel sensitivity done")

    outputs.append(write_summary(out, cfg, settings))
    inputs = study_inputs(cfg06, ROOT) + [p for t in targets for p in t.get("inputs", [])]
    inputs += [ROOT / cfg["inputs"]["signatures"], ROOT / cfg02["data"]["GSE328774"]["counts"],
               ROOT / cfg02["data"]["GSE311736"]["counts"]]
    write_provenance(cfg["stage"], inputs=sorted(set(map(str, inputs))), outputs=outputs, cfg=cfg,
                     notes="Mixtures of cross-laboratory LPS-like references and held-out GSE200210 "
                           "disulfide-HMGB1 replicates with sorted-KC mouse-to-mouse noise; methods "
                           "comparison; leave-one-series-out; nested panels with held-out-gene R2.")
    log.info("done")


if __name__ == "__main__":
    main()
