"""Stage 01 — stimulus-response signatures from GSE200210 (mouse BMDM, 24 h).

1. DESeq2 (pydeseq2) per stimulus vs PBS, culture replicate as unit; dsHMGB1 vs LPS direct
   and magnitude-adjusted contrasts; response sizes and dsHMGB1/LPS overlap.
2. Signatures: shared inflammatory, HMGB1-biased, LPS-biased, HMGB1-vs-LPS axis.
   signatures.tsv is written immediately (downstream stages poll it).
3. Leave-one-replicate-out validation of the signatures on held-out libraries.
4. Mechanism-candidate and pathogen-clearance genes; clearance gene-set tests.
5. Over-representation and preranked enrichment (MSigDB mouse collections).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import (load_config, project_root, results_dir, set_global_seed,  # noqa: E402
                       write_provenance, write_table)
from lib.perturb_de import all_contrasts, load_counts, log_cpm, responsive, sample_table  # noqa: E402
from lib.perturb_sig import AXIS, build_signatures, score, zscore_ref  # noqa: E402
from lib.perturb_validate import (independence_stats, loro_scores, nearest_centroid,  # noqa: E402
                                  separation_stats)
from lib.perturb_annot import (candidate_table, clearance_set_tests, ora, prerank,  # noqa: E402
                               read_gmt)

STAGE = "01_perturbation"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(STAGE)


def response_summary(res, de, slope):
    a, m = de["alpha"], de["min_abs_lfc"]
    rows = []
    for k, d in res.items():
        rows.append({"contrast": k, "n_tested": int(d["padj"].notna().sum()),
                     "n_up": len(responsive(d, a, m, "up")),
                     "n_down": len(responsive(d, a, m, "down")),
                     "n_responsive": len(responsive(d, a, m))})
    ds, lp = res["dsHMGB1"], res["LPS"]
    out = []
    for direction in ["any", "up", "down"]:
        A, B = set(responsive(ds, a, m, direction)), set(responsive(lp, a, m, direction))
        out.append({"metric": f"jaccard_{direction}", "value": len(A & B) / len(A | B),
                    "n_dsHMGB1": len(A), "n_LPS": len(B), "n_both": len(A & B)})
    U = responsive(ds, a, m).union(responsive(lp, a, m))
    x, y = lp.loc[U, "log2FC"], ds.loc[U, "log2FC"]
    out += [{"metric": "pearson_log2FC_union_responsive", "value": stats.pearsonr(x, y)[0],
             "n_genes": len(U)},
            {"metric": "spearman_log2FC_union_responsive", "value": stats.spearmanr(x, y)[0],
             "n_genes": len(U)},
            {"metric": "slope_dsHMGB1_on_LPS_log2FC", "value": slope, "n_genes": len(U)},
            {"metric": "frac_union_same_direction", "value": float((np.sign(x) == np.sign(y)).mean()),
             "n_genes": len(U)}]
    fr = res["frHMGB1"]
    out.append({"metric": "frHMGB1_n_responsive", "value": len(responsive(fr, a, m))})
    return pd.DataFrame(rows), pd.DataFrame(out)


def main():
    cfg = load_config(f"config/{STAGE}.yaml")
    set_global_seed(cfg["seed"])
    R, out = project_root(), results_dir(STAGE)
    dcfg, de = cfg["data"], cfg["de"]
    inputs = [R / dcfg["counts"], R / dcfg["series_matrix"], R / dcfg["gene_map"],
              *[R / p for p in dcfg["gmt"].values()]]
    outputs = []

    counts, gene_tab = load_counts(R / dcfg["counts"], R / dcfg["gene_map"], cfg["filter"])
    meta = sample_table(R / dcfg["series_matrix"], counts.columns, cfg["conditions"])
    order = [s for c in [cfg["reference"], *cfg["stimuli"]] for s in meta.index[meta.condition == c]]
    counts, meta = counts[order], meta.loc[order]
    outputs.append(write_table(meta.reset_index(), out / "samples.tsv"))
    log.info("%d genes x %d libraries", *counts.shape)

    # 1. differential expression on all 15 libraries
    res, slope = all_contrasts(counts, meta, cfg["reference"], cfg["stimuli"],
                               de["alpha"], de["min_abs_lfc"], de["n_cpus"])
    for k, d in res.items():
        d = d.join(gene_tab).reset_index()
        outputs.append(write_table(d, out / f"de_{k}.tsv"))
    sizes, overlap = response_summary(res, de, slope)
    outputs += [write_table(sizes, out / "response_sizes.tsv"),
                write_table(overlap, out / "overlap_stats.tsv")]

    # 2. signatures from all libraries (written first: downstream stages poll this file)
    sigs, ncand = build_signatures(res, de, cfg["signatures"])
    outputs.append(write_table(sigs[["signature", "gene", "weight", "rank", "rank_stat"]],
                               out / "signatures.tsv"))
    log.info("signatures written: candidates %s", ncand)
    lc = log_cpm(counts, cfg["scoring"]["pseudocount"])
    full = score(zscore_ref(lc, counts.columns), sigs).join(meta[["condition", "replicate"]])
    outputs.append(write_table(full.reset_index().rename(columns={"index": "sample"}),
                               out / "full_data_scores.tsv"))

    # 3. leave-one-replicate-out validation
    conds = [cfg["reference"], *cfg["stimuli"]]
    loro, folds, fsigs = loro_scores(counts, meta, cfg)
    for k, f in fsigs.groupby("fold"):
        for name in ["shared_inflammatory", "HMGB1_biased", "LPS_biased"]:
            a = set(sigs.loc[sigs.signature == name, "gene"])
            b = set(f.loc[f.signature == name, "gene"])
            folds.loc[folds.fold == k, f"recovered_full_{name}"] = len(a & b) / len(a)
            folds.loc[folds.fold == k, f"jaccard_full_{name}"] = len(a & b) / len(a | b)
    sep = pd.concat([separation_stats(loro, conds), independence_stats(loro, full)],
                    ignore_index=True)
    nc = pd.concat([nearest_centroid(loro, f) for f in
                    (["shared_inflammatory"], [AXIS], ["axis_adj_shared"],
                     ["shared_inflammatory", AXIS])], ignore_index=True)
    acc = nc.groupby("features").agg(n=("correct", "size"), n_correct=("correct", "sum"))
    for feats, r in acc.iterrows():
        sep.loc[len(sep)] = {"analysis": "nearest_centroid_accuracy", "score": feats,
                             "n1": r["n"], "diff": r["n_correct"] / r["n"]}
    sig_cnt = sigs.groupby("signature").size()
    folds = folds.assign(**{f"n_genes_full_{k}": v for k, v in sig_cnt.items()},
                         **{f"n_candidates_full_{k}": v for k, v in ncand.items()},
                         slope_full=slope)
    outputs += [write_table(loro, out / "heldout_scores.tsv"),
                write_table(sep, out / "separation_stats.tsv"),
                write_table(nc, out / "heldout_classification.tsv"),
                write_table(folds, out / "loro_folds.tsv")]

    # 4. mechanism and clearance genes
    cg = cfg["candidate_genes"]
    gmts = {k: read_gmt(R / p) for k, p in dcfg["gmt"].items()}
    allsets = {**gmts["go_bp"], **gmts["reactome"]}
    clear_sets = {s: allsets[s] for s in cg["clearance_gene_sets"]}
    clear_sets["curated_clearance_receptors"] = cg["clearance"]
    cand = candidate_table({"mechanism": cg["mechanism"], "clearance": cg["clearance"]},
                           counts, meta, res, gene_tab, cfg["scoring"]["pseudocount"])
    outputs += [write_table(cand, out / "candidate_genes.tsv"),
                write_table(clearance_set_tests(clear_sets, res, counts.index),
                            out / "clearance_sets.tsv")]

    # 5. enrichment
    ec = cfg["enrichment"]
    libs = {k: gmts[k] for k in ec["libraries"]}
    enr = []
    for name in ["shared_inflammatory", "HMGB1_biased", "LPS_biased"]:
        e = ora(sigs.loc[sigs.signature == name, "gene"].tolist(), counts.index, libs,
                ec["min_set_size"], ec["max_set_size"])
        enr.append(e.assign(analysis="ORA", query=name))
    for k in ["dsHMGB1_vs_scaledLPS", "dsHMGB1_vs_LPS"]:
        for lib in ["hallmark"]:
            e = prerank(res[k]["stat"], gmts[lib], ec, cfg["seed"])
            enr.append(e.assign(analysis="prerank", query=k, library=lib))
        e = prerank(res[k]["stat"], clear_sets, {**ec, "min_set_size": 5}, cfg["seed"])
        enr.append(e.assign(analysis="prerank", query=k, library="clearance"))
    enr = pd.concat(enr, ignore_index=True)
    enr["source"] = f"MSigDB {dcfg['msigdb_version']} (mouse)"
    cols = ["analysis", "query", "library", "term", "overlap", "set_size", "query_size",
            "odds_ratio", "es", "nes", "pvalue", "fdr", "genes", "source"]
    outputs.append(write_table(enr[[c for c in cols if c in enr.columns]],
                               out / "enrichment.tsv"))

    write_provenance(STAGE, inputs, outputs, cfg,
                     notes="GSE200210 BMDM 24 h; DESeq2 ~condition; LORO by replicate index",
                     extra={"n_genes": int(counts.shape[0]), "scaling_slope": slope,
                            "signature_candidates": ncand})
    log.info("done")


if __name__ == "__main__":
    main()
