"""Stage 03b: myeloid composition vs within-population state, ligand-receptor products and
stimulus-signature projection in the two liver scRNA-seq datasets.

GSE279167: one pooled library per condition -> effect sizes only, no P values.
GSE290684: 12 libraries, all CLP 24 h (vehicle vs TPPU) -> libraries are the units.
Run after 03_single_cell_process.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib import repro  # noqa: E402
from scripts.lib import sc_beyond as B  # noqa: E402
from scripts.lib import sc_tables as T  # noqa: E402
from scripts.lib.sc_preprocess import score_sets  # noqa: E402
from scripts.lib.sc_scores import lr_products, ulm_scores  # noqa: E402

STAGE = "03_single_cell"
CONTRAST = {"GSE279167": ("sham", "CLP", False), "GSE290684": ("CLP_vehicle", "CLP_TPPU", True)}


def gene_sets(cfg, var_names) -> tuple[dict, pd.DataFrame]:
    sets = {k: v for k, v in cfg["programs"].items() if isinstance(v, list)}
    try:
        import gseapy as gp
        lib = gp.get_library(cfg["programs"]["go_library"], organism="Mouse")
        hs = lib[cfg["programs"]["go_term"]]
        lut = {g.upper(): g for g in var_names}
        sets["phagocytosis_GO"] = sorted({lut[h.upper()] for h in hs if h.upper() in lut})
        hm = gp.get_library(cfg["programs"]["hallmark_library"], organism="Mouse")
        sets["inflammation_hallmark_tnfa_nfkb"] = sorted(
            {lut[h.upper()] for h in hm[cfg["programs"]["hallmark_term"]] if h.upper() in lut})
    except Exception as e:  # offline: the GO / Hallmark sets are skipped and noted
        print("GO/Hallmark set unavailable:", e)
    # clearance sets without any gene used to annotate myeloid subsets or KC identity,
    # so KC-vs-recruited differences are not built in by the labelling
    ann = set(cfg["programs"]["kc_identity"])
    for k, v in cfg["myeloid_subsets"].items():
        if isinstance(v, list):
            ann |= set(v)
    for base in ["clearance_curated", "phagocytosis_GO"]:
        if base in sets:
            sets[f"{base}_no_annotation"] = [g for g in sets[base] if g not in ann]
    rows = [{"set": k, "gene": g, "present": g in var_names} for k, v in sets.items() for g in v]
    return sets, pd.DataFrame(rows)


def load_signatures(root, cfg):
    p = root / cfg["signatures_path"]
    if not p.exists():
        return None, p
    net = pd.read_csv(p, sep="\t")
    return net[["signature", "gene", "weight"]].dropna(), p


def population_tables(my, sets, scores, sig_cols, ref, alt, sample_level):
    """Per (sample, population) program scores (pseudobulk and mean cell score) and the
    alt-vs-ref contrast for each population of the given view."""
    obs = my.obs
    cell = T.cell_population_means(obs, scores)
    pb = T.pseudobulk_scores(my, sets)
    if sig_cols:  # signature scores at sample level: mean of per-cell ULM t
        sm = obs.groupby(["condition", "sample", "population"], observed=True)[sig_cols].mean().reset_index()
        pb = pb.merge(sm, on=["condition", "sample", "population"], how="left")
    ids = ["condition", "sample", "population", "n_cells"]
    pb_l = pb.melt(id_vars=ids, value_vars=[c for c in scores if c in pb], var_name="score",
                   value_name="value")
    pb_l["level"] = np.where(pb_l.score.str.startswith("sig_"), "mean_cell_ulm_t",
                             "pseudobulk_centred_logCPM")
    cl = cell.melt(id_vars=ids, value_vars=[f"{c}__mean" for c in scores], var_name="score",
                   value_name="value")
    cl["score"] = cl["score"].str.replace("__mean", "")
    cl = cl[~cl.score.str.startswith("sig_")]
    cl["level"] = "mean_cell_score_genes"
    ps = pd.concat([pb_l, cl])
    ps.insert(0, "dataset", "")
    con = T.contrasts(pb, cell, obs, [c for c in scores if c in pb], ref, alt, sample_level)
    con.insert(0, "dataset", "")
    return ps, con, pb


def umap_table(ad, cols, n, seed, gse, name, out):
    idx = np.arange(ad.n_obs)
    if ad.n_obs > n:
        idx = np.sort(np.random.default_rng(seed).choice(ad.n_obs, n, replace=False))
    o = ad.obs.iloc[idx]
    df = pd.DataFrame({"cell": o.index, "dataset": gse, "condition": o["condition"].values,
                       "sample": o["sample"].values, "population": o["population"].values,
                       "umap1": ad.obsm["X_umap"][idx, 0], "umap2": ad.obsm["X_umap"][idx, 1]})
    for c in cols:
        df[c] = o[c].values
    f = out / f"umap_{gse}_{name}.tsv.gz"
    df.to_csv(f, sep="\t", index=False, float_format="%.4g", compression="gzip")
    return f


def run(gse, cfg, root, out, net):
    seed = cfg["seed"]
    ref, alt, sample_level = CONTRAST[gse]
    pdir = root / cfg["processed_dir"]
    ad = sc.read_h5ad(pdir / f"03_{gse}.h5ad")
    my = sc.read_h5ad(pdir / f"03_{gse}_myeloid.h5ad")
    sets, gs = gene_sets(cfg, my.var_names)
    gs.insert(0, "dataset", gse)
    prog = list(sets)
    score_sets(my, sets, seed)
    score_sets(ad, {k: sets[k] for k in ["inflammatory", "kc_identity", "clearance_curated"]}, seed)
    tabs = {"gene_sets": gs}

    sig_cols = []
    if net is not None:
        S = ulm_scores(my, net)
        S.columns = [f"sig_{c}" for c in S.columns]
        my.obs = my.obs.join(S)
        Sa = ulm_scores(ad, net)
        Sa.columns = [f"sig_{c}" for c in Sa.columns]
        ad.obs = ad.obs.join(Sa)
        sig_cols = list(S.columns)
        by = ["condition", "population"] + (["sample"] if sample_level else [])
        proj = ad.obs.groupby(by, observed=True)[sig_cols].agg(["mean", "median"])
        proj.columns = [f"{a}__{b}" for a, b in proj.columns]
        proj = proj.join(ad.obs.groupby(by, observed=True).size().rename("n_cells")).reset_index()
        proj.insert(0, "dataset", gse)
        proj.insert(1, "level", "per_sample" if sample_level else "per_condition")
        tabs["signature_projection"] = proj
        ng = pd.Series(S.attrs.get("n_genes", {}) if hasattr(S, "attrs") else {})
        tabs["signature_genes_used"] = pd.DataFrame(
            {"dataset": gse, "signature": [c[4:] for c in sig_cols],
             "n_genes_in_data": [Sa.attrs["n_genes"][c[4:]] for c in sig_cols]})
        if sample_level:
            proj2 = ad.obs.groupby(["condition", "population"], observed=True)[sig_cols].mean()
            proj2 = proj2.add_suffix("__mean").reset_index()
            proj2.insert(0, "dataset", gse)
            proj2.insert(1, "level", "per_condition")
            tabs["signature_projection"] = pd.concat([proj, proj2])

    scores = prog + sig_cols
    obs = my.obs
    # second view: KC-marker-positive (resident + KC-like) vs recruited (Mo-mac + monocytes)
    myc = my.copy()
    myc.obs["population"] = myc.obs["population"].astype(str).map(T.COLLAPSE)
    myc = myc[myc.obs["population"].notna()].copy()
    ps_all, con_all, pbs = [], [], {}
    for view, m in [("myeloid_subsets", my), ("kc_marker_vs_recruited", myc)]:
        ps, con, pbv = population_tables(m, sets, scores, sig_cols, ref, alt, sample_level)
        pbs[view] = pbv
        tabs.setdefault("pseudobulk_log2fc", []).append(
            B.pseudobulk_log2fc(m, ref, alt, f"{alt}_vs_{ref}").assign(dataset=gse, view=view))
        ps.insert(1, "view", view)
        con.insert(1, "view", view)
        ps_all.append(ps)
        con_all.append(con)
    tabs["population_scores"] = pd.concat(ps_all).assign(dataset=gse)
    tabs["population_contrasts"] = pd.concat(con_all).assign(dataset=gse)
    pb = pbs["myeloid_subsets"]
    tabs["pseudobulk_log2fc"] = pd.concat(tabs["pseudobulk_log2fc"])
    beyond, unc = B.run_dataset(gse, myc.obs, pbs["kc_marker_vs_recruited"], pb, ref, alt,
                                sample_level)
    tabs["population_beyond_shared"] = beyond
    tabs["uncoupling"] = unc
    comp, comp_s = T.composition(obs, ref, alt, sample_level)
    comp.insert(0, "dataset", gse)
    comp_s.insert(0, "dataset", gse)
    tabs["myeloid_composition"] = comp
    tabs["myeloid_composition_contrast"] = comp_s
    dec, dec_pop = T.decomposition(obs, scores, ref, alt, sample_level)
    dec.insert(0, "dataset", gse)
    dec_pop.insert(0, "dataset", gse)
    tabs["decomposition"] = dec
    tabs["decomposition_by_population"] = dec_pop

    ys = [c for c in ["clearance_curated", "phagocytosis_GO", "kc_identity"] if c in obs]
    cp = T.coupling(obs, "inflammatory", ys)
    cp.insert(0, "dataset", gse)
    cp.insert(1, "level", "cells_within_population_condition")
    if sample_level:
        cs = T.sample_coupling(pb, "inflammatory", ys)
        cs.insert(0, "dataset", gse)
        cs.insert(1, "level", "samples_within_population")
        cp = pd.concat([cp, cs])
    tabs["coupling"] = cp

    lr_cfg = cfg["ligand_receptor"]
    keep = ad.obs["population"].value_counts()
    ok = keep.index[keep >= 30]
    sub = ad[ad.obs["population"].isin(ok) & ~ad.obs["population"].str.contains("excluded|Low quality")]
    lr = lr_products(sub, lr_cfg["ligand"], lr_cfg["receptors"], "condition", "population",
                     "population", "sample" if sample_level else None)
    lr.insert(0, "dataset", gse)
    tabs["lr_scores"] = lr

    n = cfg["processing"]["umap_max_cells"]
    f1 = umap_table(ad, ["inflammatory", "kc_identity", "clearance_curated"] + sig_cols, n, seed,
                    gse, "all", out)
    f2 = umap_table(my, prog + sig_cols, n, seed, gse, "myeloid", out)
    return tabs, [f1, f2]


def lr_contrast(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gse, (ref, alt, sl) in CONTRAST.items():
        d = lr[lr.dataset == gse]
        if d.empty:
            continue
        g = d.groupby(["condition", "source", "target", "receptor"])[["score", "ligand_mean", "receptor_mean"]].mean()
        a, b = g.xs(ref, level="condition"), g.xs(alt, level="condition")
        c = (b - a).add_prefix("delta_").join(a.add_prefix("ref_")).join(b.add_prefix("alt_")).reset_index()
        c.insert(0, "comparison", alt)
        c.insert(0, "reference", ref)
        c.insert(0, "dataset", gse)
        rows.append(c)
    return pd.concat(rows)


def main():
    cfg = repro.load_config("config/03_single_cell.yaml")
    repro.set_global_seed(cfg["seed"])
    root = repro.project_root()
    out = repro.results_dir(STAGE)
    net, sig_path = load_signatures(root, cfg)
    print("signatures:", "found" if net is not None else f"pending ({sig_path} missing)")
    allt, files = {}, []
    for gse in (sys.argv[1:] or list(cfg["datasets"])):
        tabs, fs = run(gse, cfg, root, out, net)
        files += fs
        for k, v in tabs.items():
            allt.setdefault(k, []).append(v)
    outputs = []
    for k, v in allt.items():
        outputs.append(repro.write_table(pd.concat(v), out / f"{k}.tsv"))
    if "lr_scores" in allt:
        outputs.append(repro.write_table(lr_contrast(pd.concat(allt["lr_scores"])), out / "lr_contrasts.tsv"))
    if "population_beyond_shared" in allt:
        bey = pd.concat(allt["population_beyond_shared"])
        outputs.append(repro.write_table(B.replication(bey), out / "beyond_shared_replication.tsv"))
    if "uncoupling" in allt:
        sp_ = root / "results/02_sorted_bulk/coupling_stats.tsv"
        outputs.append(repro.write_table(B.compare_sorted(pd.concat(allt["uncoupling"]), sp_),
                                         out / "uncoupling_vs_sorted.tsv"))
    if net is None:
        (out / "signature_projection.tsv").write_text(
            f"status\tpending\nreason\t{cfg['signatures_path']} not present at run time\n")
        outputs.append(out / "signature_projection.tsv")
    pre = [out / f for f in ["design.tsv", "qc_summary.tsv", "cluster_annotation.tsv",
                             "myeloid_subcluster_annotation.tsv", "markers.tsv", "cell_counts.tsv",
                             "raw_checksums.tsv", "processed_objects.tsv"]]
    pdir = root / cfg["processed_dir"]
    inputs = [pdir / f"03_{g}{s}.h5ad" for g in cfg["datasets"] for s in ("", "_myeloid")]
    if net is not None:
        inputs.append(sig_path)
    if (root / "results/02_sorted_bulk/coupling_stats.tsv").exists():
        inputs.append(root / "results/02_sorted_bulk/coupling_stats.tsv")
    repro.write_provenance(STAGE, inputs=inputs, outputs=pre + outputs + files, cfg=cfg,
                           notes="03_single_cell_process.py then 03_single_cell_analyse.py; "
                                 "raw inputs checksummed in raw_checksums.tsv",
                           extra={"signatures": "used" if net is not None else "pending"})
    print("done")


if __name__ == "__main__":
    main()
