"""Stage 05 — protein layer for liver macrophage programs after sepsis or LPS.

Processed protein tables only (see config/05_proteomics.yaml and 00_download_05.sh).
For each study separately: log2, per-sample median centring, no imputation, moderated t
between the two groups with animals (or culture replicates) as units, BH FDR.
Then: program scores per sample, transcript-protein concordance, and rank-based
enrichment of the stimulus signatures from stage 01.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import (load_config, project_root, results_dir,  # noqa: E402
                               set_global_seed, write_provenance, write_table)
from scripts.lib.prot_io import (harmonise_symbols, load_copy_number_xlsx, load_diann_pg,  # noqa: E402
                                 log_normalise, withdrawn_symbol_map)
from scripts.lib.prot_stats import (difference_test, hedges_g, moderated_t, on_off,  # noqa: E402
                                    rank_set_test, sample_set_score)
from scripts.lib.prot_transcript import (clp_liver_transcripts, concordance,  # noqa: E402
                                         load_transcript_table)

ROOT = project_root()


def load_dataset(key: str, spec: dict, symmap: dict):
    if spec["kind"] == "diann_pg_matrix":
        ab, des, ann = load_diann_pg(ROOT / spec["zip"], ROOT / spec["design_xlsx"])
        inputs = [ROOT / spec["zip"], ROOT / spec["design_xlsx"]]
    else:
        ab, des, ann = load_copy_number_xlsx(ROOT / spec["xlsx"], spec["sheet"], spec["groups"])
        inputs = [ROOT / spec["xlsx"]]
    des.insert(0, "dataset", key)
    ab, ann, n_renamed = harmonise_symbols(ab, ann, symmap)
    print(f"[{key}] withdrawn symbols renamed to current MGI symbols: {n_renamed}")
    return ab, des, ann, inputs


def differential(key, spec, x, des, ann, cfg):
    c = spec["contrast"]
    g = pd.Series(des["group"].to_numpy(), index=des["sample"].to_numpy())
    n_min = int(min((g == c["test"]).sum(), (g == c["reference"]).sum()))
    min_valid = cfg["preprocessing"]["min_valid_per_group"].get(n_min, max(2, n_min - 1))
    de = moderated_t(x, g, c["test"], c["reference"], min_valid)
    de["on_off"] = on_off(x, g, c["test"], c["reference"],
                          cfg["preprocessing"]["on_off_min_present"])
    de = de.join(ann)
    de.insert(0, "contrast", f"{c['test']}_vs_{c['reference']}")
    de.insert(0, "dataset", key)
    de.index.name = "gene"
    return de, g, min_valid


def program_scores(key, spec, x, g, de, cfg):
    c = spec["contrast"]
    rows, per_sample = [], []
    sets = dict(cfg["protein_sets"])
    sets.update({f"single_{p}": [p] for p in cfg["single_proteins"]})
    t = de.loc[de["tested"], "t_mod"]
    rng = np.random.default_rng(cfg["seed"])
    for name, members in sets.items():
        score, present = sample_set_score(x, members)
        a = score[g.index[g == c["test"]]].to_numpy(float)
        b = score[g.index[g == c["reference"]]].to_numpy(float)
        tested = [m for m in members if m in t.index]
        r = {"dataset": key, "contrast": f"{c['test']}_vs_{c['reference']}", "program": name,
             "n_members": len(members), "n_scored_all_samples": len(present),
             "members_scored": ",".join(present), "n_tested": len(tested),
             "mean_log2FC_members": de.loc[tested, "log2FC"].mean() if tested else np.nan,
             "members_up_fdr": ",".join(m for m in tested if de.loc[m, "padj"] < cfg["stats"]["fdr"]
                                         and de.loc[m, "log2FC"] > 0),
             "members_down_fdr": ",".join(m for m in tested if de.loc[m, "padj"] < cfg["stats"]["fdr"]
                                           and de.loc[m, "log2FC"] < 0),
             "members_on_off": ",".join(f"{m}:{de.loc[m, 'on_off']}" for m in members
                                        if m in de.index and de.loc[m, "on_off"])}
        if present:
            r.update({"score_diff": np.nanmean(a) - np.nanmean(b), "hedges_g": hedges_g(a, b),
                      "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue
                      if len(present) and np.nanstd(np.r_[a, b]) > 0 else np.nan})
        if len(tested) >= 3 and not name.startswith("single_"):
            rt = rank_set_test(t, members, None, cfg["stats"]["n_perm"], rng)
            r.update({"auc_vs_background": rt["auc"], "mwu_p": rt["mwu_p"],
                      "mean_t_members": rt["mean_t"], "perm_p": rt["perm_p"]})
        if len(tested) == 1 and name.startswith("single_"):
            m = tested[0]
            r.update({"log2FC": de.loc[m, "log2FC"], "pvalue": de.loc[m, "pvalue"],
                      "padj": de.loc[m, "padj"]})
        rows.append(r)
        for s, v in score.items():
            per_sample.append({"dataset": key, "sample": s, "group": g[s], "program": name,
                               "score": v})
    return pd.DataFrame(rows), pd.DataFrame(per_sample)


def marker_rows(key, de, cfg, tx):
    """Every program member and single protein: protein effect next to paired transcript effects."""
    rows = []
    sets = dict(cfg["protein_sets"])
    sets["hmgb1_axis"] = cfg["single_proteins"]
    for name, members in sets.items():
        for m in members:
            r = {"dataset": key, "program": name, "gene": m, "detected": m in de.index}
            if m in de.index:
                d = de.loc[m]
                r.update({k: d[k] for k in ("n_test", "n_ref", "log2FC", "ci_low", "ci_high",
                                            "pvalue", "padj", "on_off")})
            for tk in cfg["concordance_pairs"].get(key, []):
                if tk in tx and m in tx[tk].index:
                    r[f"tx_lfc_{tk}"] = tx[tk].loc[m, "lfc"]
            rows.append(r)
    return pd.DataFrame(rows)


def signature_enrichment(key, de, sig, cfg, rng):
    t = de.loc[de["tested"], "t_mod"]
    rows = []
    for name, sub in sig.groupby("signature"):
        w = sub.set_index("gene")["weight"]
        signed = (w < 0).any()
        rt = rank_set_test(t, list(w.index), w if signed else None, cfg["stats"]["n_perm"], rng)
        det = [x for x in w.index if x in t.index]
        rows.append({"dataset": key, "signature": name, "statistic":
                     "mean signed t (positive = HMGB1 side)" if signed else "mean t",
                     **rt, "mean_log2FC_detected": de.loc[det, "log2FC"].mean() if det else np.nan,
                     "n_detected_up_fdr": int(((de.loc[det, "padj"] < cfg["stats"]["fdr"])
                                               & (de.loc[det, "log2FC"] > 0)).sum()),
                     "n_detected_down_fdr": int(((de.loc[det, "padj"] < cfg["stats"]["fdr"])
                                                 & (de.loc[det, "log2FC"] < 0)).sum()),
                     "detected_genes": ",".join(det)})
    sets = {n: list(s["gene"]) for n, s in sig.groupby("signature")}
    if {"HMGB1_biased", "LPS_biased"} <= sets.keys():
        d = difference_test(t, sets["HMGB1_biased"], sets["LPS_biased"], cfg["stats"]["n_perm"], rng)
        rows.append({"dataset": key, "signature": "HMGB1_biased_minus_LPS_biased",
                     "statistic": "difference in mean t (HMGB1_biased - LPS_biased)",
                     "n_members": np.nan, "n_detected": f"{d['n_a']}+{d['n_b']}",
                     "mean_t": d["diff_mean_t"], "perm_p": d["perm_p"]})
    return pd.DataFrame(rows)


def main():
    cfg = load_config("config/05_proteomics.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = results_dir(cfg["stage"])
    inputs, outputs = [], []
    designs, scores, sample_scores, conc, enr, markers = [], [], [], [], [], []
    des_by = {}

    symmap = withdrawn_symbol_map(ROOT / cfg["symbol_map"])
    inputs.append(ROOT / cfg["symbol_map"])
    sig_path = ROOT / cfg["signatures"]["path"]
    sig = pd.read_csv(sig_path, sep="\t") if sig_path.exists() else None
    if sig is not None:
        inputs.append(sig_path)

    # transcript comparators
    tx, tx_status = {}, {}
    for k, spec in cfg["transcripts"].items():
        if "counts" in spec:
            if (ROOT / spec["counts"]).exists() and (ROOT / spec["design"]).exists():
                de_tx, des_tx = clp_liver_transcripts(spec, ROOT)
                tx[k], tx_status[k] = de_tx, f"ok: computed here, n={len(des_tx)} livers"
                p = write_table(de_tx.rename_axis("gene").reset_index(), out / f"transcript_de_{k}.tsv")
                pd_ = write_table(des_tx, out / f"transcript_design_{k}.tsv")
                outputs += [p, pd_]
                inputs += [ROOT / spec["counts"], ROOT / spec["design"]]
            else:
                tx_status[k] = "pending: inputs not present"
        else:
            tab, st = load_transcript_table(spec, ROOT)
            tx_status[k] = st
            if tab is not None:
                tx[k] = tab
                if "table" in spec:
                    inputs.append(ROOT / spec["table"])

    for key, spec in cfg["datasets"].items():
        ab, des, ann, ins = load_dataset(key, spec, symmap)
        inputs += ins
        x = log_normalise(ab, cfg["preprocessing"]["normalise"])
        de, g, min_valid = differential(key, spec, x, des, ann, cfg)
        des["material"], des["model"], des["time_h"], des["unit"] = (
            spec["material"], spec["model"], spec["time_h"], spec["unit"])
        des["n_proteins_quantified"] = [int(ab[s].notna().sum()) for s in des["sample"]]
        des["min_valid_per_group"] = min_valid
        designs.append(des)
        des_by[key] = des
        p = write_table(de.reset_index(), out / f"protein_de_{key}.tsv")
        outputs.append(p)
        sc, ss = program_scores(key, spec, x, g, de, cfg)
        scores.append(sc)
        sample_scores.append(ss)
        for tk in cfg["concordance_pairs"].get(key, []):
            row = {"protein_dataset": key, "protein_contrast": de["contrast"].iloc[0],
                   "transcript_dataset": tk, "status": tx_status.get(tk, "not configured")}
            if tk in tx:
                row.update(concordance(de, tx[tk], cfg["stats"]["fdr"]))
            conc.append(row)
        markers.append(marker_rows(key, de, cfg, tx))
        if sig is not None:
            enr.append(signature_enrichment(key, de, sig, cfg, rng))
        print(f"[{key}] proteins={len(ab)} tested={int(de['tested'].sum())} "
              f"FDR<{cfg['stats']['fdr']}={int((de['padj'] < cfg['stats']['fdr']).sum())}")

    files = {
        "design.tsv": pd.concat(designs),
        "protein_scores.tsv": pd.concat(scores),
        "protein_scores_per_sample.tsv": pd.concat(sample_scores),
        "transcript_protein_concordance.tsv": pd.DataFrame(conc),
        "marker_proteins.tsv": pd.concat(markers),
    }
    if enr:
        files["signature_enrichment.tsv"] = pd.concat(enr)
    else:
        files["signature_enrichment.tsv"] = pd.DataFrame(
            [{"status": f"pending: {cfg['signatures']['path']} not present"}])
    for name, df in files.items():
        outputs.append(write_table(df, out / name))
    for extra in ("candidates.tsv", "raw_checksums.tsv"):
        if (out / extra).exists():
            outputs.append(out / extra)
    write_provenance(cfg["stage"], inputs=sorted(set(map(str, inputs))), outputs=outputs, cfg=cfg,
                     notes="Processed protein tables only; per-study analysis; no imputation; "
                           "limma-style moderated t; transcript comparators: " +
                           "; ".join(f"{k}={v}" for k, v in tx_status.items()))


if __name__ == "__main__":
    main()
