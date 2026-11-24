"""Stage 02 — sorted liver cell bulk RNA-seq after polymicrobial sepsis.

Questions: (1) Kupffer-cell response at 2 h (GSE328774) versus 8 h (GSE311736), compared through
effect sizes on shared genes; (2) which sorted cell type carries the inflammatory response and
expresses the HMGB1-axis genes; (3) inflammation versus clearance and KC-identity programmes in
sorted KC and CD45+ non-KC cells; (4) hepatocyte acute-phase response (GSE311736, GSE303231 with
sex); (5) projection of the stage-01 stimulus signatures. The mouse (one library) is the unit;
each dataset and sorted cell type is modelled separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import (load_config, project_root, results_dir, set_global_seed,  # noqa: E402
                               write_provenance, write_table)
from scripts.lib.sorted_io import (build_programs, gene_annotation, load_gse303231,  # noqa: E402
                                   load_gse311736, load_gse328774)
from scripts.lib.sorted_stats import filter_genes, group_compare, lfc_concordance, run_deseq  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from scripts.lib.sorted_analysis import (coupling, gene_rows, kc_admixture, purity_markers,  # noqa: E402
                                         program_effects, project_signatures, unit_scores)

STAGE = "02_sorted_bulk"
ROOT = project_root()


def main() -> None:
    cfg = load_config("config/02_sorted_bulk.yaml")
    set_global_seed(cfg["seed"])
    out = results_dir(STAGE)
    f, de_cfg = cfg["filter"], cfg["de"]
    genes, tx = gene_annotation(cfg["data"]["gtf"])

    c1, m1 = load_gse328774(cfg, genes)
    c2, m2 = load_gse311736(cfg, genes)
    c3, m3, info3 = load_gse303231(cfg, genes, tx)
    meta = pd.concat([m1, m2, m3])
    meta["unit"] = meta.dataset + "_" + meta.cell_type
    meta["library_size"] = pd.concat([c1.sum(), c2.sum(), c3.sum()]).reindex(meta.index)
    design_out = meta.reset_index()
    design_out["biological_unit"] = "mouse"
    write_table(design_out, out / "design.tsv")

    # ---- per-unit DE ------------------------------------------------------------------
    counts_all = {"GSE328774": c1, "GSE311736": c2, "GSE303231": c3}
    case_ref = {}
    units = {}
    for unit, d in meta.groupby("unit"):
        case = "sepsis" if d.dataset.iloc[0] == "GSE303231" else "CLP"
        ref = "control" if case == "sepsis" else "sham"
        case_ref[unit] = (case, ref)
        units[unit] = filter_genes(counts_all[d.dataset.iloc[0]][d.index], f["min_count"], f["min_samples"])
    de = {}
    for unit, cnt in units.items():
        md = meta.loc[cnt.columns, ["condition", "sex"]].copy()
        case, ref = case_ref[unit]
        design = "~sex + condition" if md.sex.nunique() > 1 else "~condition"
        de[unit] = run_deseq(cnt, md, design, ["condition", case, ref], de_cfg["n_cpus"],
                             de_cfg["ci_level"], de_cfg["alpha"])
        de[unit].insert(0, "unit", unit)
        write_table(de[unit], out / f"de_{unit}.tsv")
        print(f"[de] {unit}: {cnt.shape[1]} libraries, {len(de[unit])} genes, "
              f"{int((de[unit].padj < de_cfg['alpha']).sum())} FDR<{de_cfg['alpha']}")
    # sex-stratified fits in GSE303231 and the sex difference in log2FC
    hep = meta[meta.unit == "GSE303231_HEP"]
    strat = {}
    for sx in ("female", "male"):
        idx = hep.index[hep.sex == sx]
        cnt = filter_genes(c3[idx], f["min_count"], f["min_samples"])
        strat[sx] = run_deseq(cnt, meta.loc[idx, ["condition"]], "~condition",
                              ["condition", "sepsis", "control"], de_cfg["n_cpus"], de_cfg["ci_level"])
        strat[sx].insert(0, "unit", f"GSE303231_HEP_{sx}")
        write_table(strat[sx], out / f"de_GSE303231_HEP_{sx}.tsv")
    sx = strat["male"].merge(strat["female"], on="gene", suffixes=("_male", "_female"))
    sx["lfc_male_minus_female"] = sx.log2FoldChange_male - sx.log2FoldChange_female
    sx["se_diff"] = np.sqrt(sx.lfcSE_male ** 2 + sx.lfcSE_female ** 2)
    sx["p_sex_difference"] = 2 * stats.norm.sf((sx.lfc_male_minus_female / sx.se_diff).abs())
    ok = sx.p_sex_difference.notna()
    sx.loc[ok, "padj_sex_difference"] = multipletests(sx.loc[ok, "p_sex_difference"], method="fdr_bh")[1]
    write_table(sx[["gene", "log2FoldChange_male", "lfcSE_male", "padj_male", "log2FoldChange_female",
                    "lfcSE_female", "padj_female", "lfc_male_minus_female", "se_diff",
                    "p_sex_difference", "padj_sex_difference"]], out / "de_GSE303231_HEP_sex_difference.tsv")

    # ---- Q1: KC 2 h versus 8 h ---------------------------------------------------------
    a, b = de["GSE328774_KC"], de["GSE311736_KC"]
    conc = pd.DataFrame(lfc_concordance(a, b, de_cfg["alpha"], de_cfg["min_abs_lfc"]))
    conc.insert(0, "comparison", "KC_2h(a)_vs_KC_8h(b)")
    # same comparison against the other 8 h cell types (does the 2 h KC response resemble 8 h KC most?)
    extra = []
    for u in [u for u in de if u.startswith("GSE311736_") and u != "GSE311736_KC"]:
        r = pd.DataFrame(lfc_concordance(a, de[u], de_cfg["alpha"], de_cfg["min_abs_lfc"]))
        r.insert(0, "comparison", f"KC_2h(a)_vs_{u.split('_', 1)[1]}_8h(b)")
        extra.append(r)
    write_table(pd.concat([conc] + extra), out / "de_kc_2h_vs_8h_concordance.tsv")
    m = a.merge(b, on="gene", suffixes=("_2h", "_8h"))
    al = de_cfg["alpha"]; ml = de_cfg["min_abs_lfc"]
    s2 = (m.padj_2h < al) & (m.log2FoldChange_2h.abs() >= ml)
    s8 = (m.padj_8h < al) & (m.log2FoldChange_8h.abs() >= ml)
    m["class"] = np.select([s2 & s8 & (np.sign(m.log2FoldChange_2h) == np.sign(m.log2FoldChange_8h)),
                            s2 & s8, s2 & ~s8, ~s2 & s8], ["shared_same_direction", "shared_opposite",
                           "2h_only", "8h_only"], "neither")
    write_table(m[["gene", "baseMean_2h", "log2FoldChange_2h", "lfcSE_2h", "padj_2h", "baseMean_8h",
                   "log2FoldChange_8h", "lfcSE_8h", "padj_8h", "class"]], out / "de_kc_2h_vs_8h_genes.tsv")

    # ---- programme scores and per-unit responses (Q2-Q4) ---------------------------------
    programs = build_programs(cfg)
    gl = cfg["genes"]
    gene_panel = list(dict.fromkeys(gl["chemokines_cytokines"] + gl["hmgb1_axis"] + gl["kc_identity"]
                                    + gl["monocyte_recruited"] + gl["clearance"] + gl["acute_phase"]
                                    + gl["hepatocyte_stress"]))
    all_scores, responses, z_by_unit, l2_by_unit = [], [], {}, {}
    for unit in units:
        idx = meta.index[meta.unit == unit]
        cnt = counts_all[meta.loc[idx[0], "dataset"]][idx]
        case, ref = case_ref[unit]
        sc, z, lx = unit_scores(cnt, meta.loc[idx], programs, cfg, case, ref, unit)
        z_by_unit[unit] = z
        gg = meta.condition.reindex(lx.columns)
        l2_by_unit[unit] = lx.loc[z.index].sub(lx.loc[z.index, gg == ref].mean(axis=1), axis=0)
        all_scores.append(sc)
        strata = meta.loc[idx, "sex"] if meta.loc[idx, "sex"].nunique() > 1 else None
        responses.append(program_effects(sc, meta, case, ref, de[unit], programs, al, strata))
        responses.append(gene_rows(de[unit], cnt, meta.loc[idx], gene_panel, case, ref, unit))
    for sxn, d in strat.items():
        idx = hep.index[hep.sex == sxn]
        g = gene_rows(d, c3[idx], meta.loc[idx], gene_panel, "sepsis", "control", "GSE303231_HEP")
        g["stratum"] = sxn
        responses.append(g)
    scores = pd.concat(all_scores).merge(meta[["dataset", "cell_type", "condition", "sex", "time_h"]],
                                         left_on="sample", right_index=True)
    write_table(scores, out / "program_scores.tsv")
    resp = pd.concat(responses, ignore_index=True)
    resp = resp.merge(meta.drop_duplicates("unit").set_index("unit")[["dataset", "cell_type", "time_h"]],
                      left_on="unit", right_index=True)
    # rank the cell types of GSE311736 by inflammatory programme effect
    pr = resp[(resp.level == "program") & (resp.stratum == "all")].copy()
    pr["rank_within_dataset"] = pr.groupby(["dataset", "feature"])["diff"].rank(ascending=False)  # log2-scale change
    resp.loc[pr.index, "rank_within_dataset"] = pr.rank_within_dataset
    write_table(resp, out / "celltype_response.tsv")

    unit_counts = {u: counts_all[meta.loc[meta.unit == u, "dataset"].iloc[0]][meta.index[meta.unit == u]]
                   for u in meta.unit.unique()}
    write_table(purity_markers(unit_counts, meta, cfg["purity_markers"]), out / "purity_markers.tsv")

    # ---- Q3: coupling and competing explanations ----------------------------------------
    infl = ["inflammation_hallmark_tnfa_nfkb", "inflammation_hallmark_inflammatory", "chemokines_cytokines"]
    clear = ["clearance_curated", "clearance_curated_no_identity", "go_phagocytosis",
             "reactome_scavenger", "kc_identity", "monocyte_recruited"]
    sc_all = pd.concat(all_scores)
    cp = [coupling(sc_all, meta, u, infl, clear)
          for u in ("GSE328774_KC", "GSE311736_KC", "GSE311736_CD45_nonKC")]
    # opposite-direction test within each mouse arm: per-mouse inflammation minus clearance score
    opp = []
    for u in ("GSE328774_KC", "GSE311736_KC", "GSE311736_CD45_nonKC"):
        w = sc_all[sc_all.unit == u].pivot(index="sample", columns="program", values="score_log2")
        case, ref = case_ref[u]
        for a_ in infl[:1]:
            for b_ in clear:
                r = {"unit": u, "analysis": "inflammation_minus_program_per_mouse",
                     "program_a": a_, "program_b": b_}
                r.update(group_compare(w[a_] - w[b_], meta.condition, case, ref))
                opp.append(r)
    adm, markers = kc_admixture({u: counts_all[meta.loc[meta.unit == u, "dataset"].iloc[0]][meta.index[meta.unit == u]]
                                 for u in ("GSE311736_KC", "GSE311736_CD45_nonKC")},
                                meta, de["GSE311736_CD45_nonKC"], cfg)
    ar = {"unit": "GSE311736_KC", "analysis": "nonKC_leukocyte_admixture_index",
          "program_a": "admixture_index", "program_b": f"{len(markers)} marker genes"}
    ar.update(group_compare(adm.set_index("sample").admixture_index, meta.condition, "CLP", "sham"))
    write_table(adm, out / "kc_admixture_per_sample.tsv")
    coup = pd.concat(cp + [pd.DataFrame(opp), pd.DataFrame([ar])], ignore_index=True)
    write_table(coup, out / "coupling_stats.tsv")

    # ---- Q5: stimulus-signature projection ----------------------------------------------
    sig_path = ROOT / cfg["data"]["signatures"]
    proj_note = "pending: results/01_perturbation/signatures.tsv not present"
    outputs_extra = []
    if sig_path.exists():
        sig = pd.read_csv(sig_path, sep="\t")
        if "weight" not in sig:
            sig["weight"] = 1.0
        ps, pc = [], []
        for scale, mats in (("z", z_by_unit), ("log2", l2_by_unit)):
            a_, b_ = project_signatures(sig, mats, meta, case_ref, cfg["scoring"]["min_genes"])
            a_.insert(1, "scale", scale); b_.insert(1, "scale", scale)
            ps.append(a_); pc.append(b_)
        ps, pc = pd.concat(ps), pd.concat(pc)
        ps = ps.merge(meta[["dataset", "cell_type", "condition", "sex"]], left_on="sample", right_index=True)
        write_table(ps, out / "signature_scores_per_sample.tsv")
        write_table(pc, out / "signature_projection.tsv")
        outputs_extra = [out / "signature_scores_per_sample.tsv", out / "signature_projection.tsv"]
        proj_note = f"projected {sig.signature.nunique()} signatures"
    print("[projection]", proj_note)

    outputs = sorted(p for p in out.glob("*.tsv")) + outputs_extra
    inputs = [ROOT / v for d in ("GSE328774", "GSE311736", "GSE303231") for v in cfg["data"][d].values()]
    inputs += [ROOT / cfg["data"]["gtf"]] + [ROOT / v for v in cfg["data"]["gmt"].values()]
    if sig_path.exists():
        inputs.append(sig_path)
    write_provenance(STAGE, inputs=inputs, outputs=list(dict.fromkeys(outputs)), cfg=cfg,
                     notes=proj_note, extra={"GSE303231_aggregation": info3,
                                             "admixture_markers": markers})


if __name__ == "__main__":
    main()
