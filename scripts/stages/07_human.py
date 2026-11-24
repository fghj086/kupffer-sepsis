#!/usr/bin/env python
"""Stage 07 - human relevance of the mouse findings, with patients and donors as units.

Questions
  1. Septic human liver (GSE141864, post-mortem, meningococcal septic shock): PAMP-like and
     tolerance-like, without an HMGB1-biased excess?  (one control liver: effect sizes only)
  2. Human stimulated myeloid cells: does the mouse HMGB1-vs-LPS axis place a human HMGB1 response
     on the HMGB1 side, and does the mouse tolerance programme mark human endotoxin tolerance?
  3. Sepsis monocytes and blood (secondary): PAMP-like / tolerance-like rather than HMGB1-biased,
     change over time, relation to 28-day mortality.
Every dataset is normalised and scored on its own; only effect directions and sizes are compared.

Outputs (results/07_human/): candidates.tsv, design.tsv, scores_by_sample.tsv, contrasts.tsv,
signatures_human.tsv, score_gene_counts.tsv, sc_monocytes_per_donor.tsv, ortholog_map_source.txt,
PROVENANCE.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import repro  # noqa: E402
from lib import human_adjust as ha  # noqa: E402
from lib import human_blood as hb  # noqa: E402
from lib import human_contrasts as hc  # noqa: E402
from lib import human_genesets as hg  # noqa: E402
from lib import human_io as hio  # noqa: E402
from lib import human_scores as hsc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("07_human")
ROOT = repro.project_root()
STAGE = "07_human"


def run_dataset(name, x, design, ref_groups, sets, cfg, group_col="group"):
    design = design.copy()
    design["sample"] = design["sample"].astype(str)
    x = x[design["sample"]]
    groups = pd.Series(design[group_col].values, index=design["sample"].values)
    ref = groups.isin(ref_groups)
    z = hsc.standardise(x, groups, ref)
    sc_, ng = hsc.score(z, sets, cfg["scores"], cfg["min_genes"], tuple(cfg["marker_sets"]),
                        cfg["min_genes_markers"])
    ng.insert(0, "dataset", name)
    log.info("%s: %d genes x %d samples; groups %s", name, x.shape[0], x.shape[1],
             design[group_col].value_counts().to_dict())
    return hsc.tidy(sc_, design, name, ng), ng


def design_rows(name, d, role, unit, note):
    g = d.groupby("group").agg(n_samples=("sample", "size"), n_subjects=("subject", "nunique")).reset_index()
    g.insert(0, "dataset", name)
    g["role"], g["unit"], g["note"] = role, unit, note
    return g


def main() -> None:
    cfg = repro.load_config("config/07_human.yaml")
    repro.set_global_seed(cfg["seed"])
    out = repro.results_dir(STAGE)
    raw = ROOT / "data/raw"
    S = cfg["scores"]
    lvl = cfg["ci_level"]

    sets, info = hg.build_sets(cfg, ROOT)
    repro.write_table(sets, out / "signatures_human.tsv")
    upd = hg.hgnc_updater(ROOT / cfg["hgnc"])
    ent = pd.read_csv(ROOT / cfg["hgnc"], sep="\t", dtype=str, usecols=["symbol", "entrez_id"]).dropna()
    entrez2sym = ent.drop_duplicates("entrez_id").set_index("entrez_id")["symbol"]
    ens2sym = hg.hgnc_ensembl(ROOT / cfg["hgnc"])

    scores, ngs, designs, C = [], [], [], []
    ds = cfg["datasets"]
    plat = lambda g: raw / "07_platforms" / f"{g}_data.txt.gz"  # noqa: E731

    # 1. septic liver (and spleen as a second macrophage-rich organ), FFPE only
    x, d = hio.gse141864(ROOT, ds["GSE141864"], upd)
    for tissue in ("liver", "spleen"):
        dd = d[(d.tissue == tissue) & (d.storage == "FFPE")]
        s, ng = run_dataset(f"GSE141864_{tissue}", x, dd, ["control"], sets, cfg)
        scores.append(s); ngs.append(ng)
        designs.append(design_rows(f"GSE141864_{tissue}", dd, "septic organ (post-mortem)", "patient",
                                   "FFPE only; fresh-frozen septic samples have no matched control"))
        C += hc.vs_single_control(s, f"GSE141864_{tissue}", f"GSE141864_{tissue}", "septic_shock", "control", S,
                                  "septic shock vs non-infectious death", "one control organ", s)
    ff = d[(d.tissue == "liver") & (d.storage == "FF")]
    designs.append(design_rows("GSE141864_liver_FF", ff, "septic liver, fresh frozen", "patient",
                               "not scored: no fresh-frozen control"))

    # 2. human KC-containing liver organoids (one iPSC line, n = 2)
    x, d = hio.gse233483(ROOT, ds["GSE233483"])
    s, ng = run_dataset("GSE233483", x, d, ["control"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE233483", d, "liver organoid with Kupffer-like cells", "culture", "one iPSC line"))
    for g1, g0 in [("LPS_IFNg", "control"), ("recovery_d1", "LPS_IFNg"), ("recovery_d3", "LPS_IFNg"),
                   ("recovery_d3", "control")]:
        C += hc.unpaired(s, "GSE233483", "GSE233483", g1, g0, S, effect_only=True)

    # 3. human endotoxin tolerance references
    x, d = hio.gse22248(ROOT, ds["GSE22248"], plat("GPL6883"))
    s, ng = run_dataset("GSE22248", x, d, ["untreated"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE22248", d, "PBMC endotoxin tolerance", "donor (paired)", ""))
    note = "Pena signatures were defined in this dataset (not an independent test)"
    for g1, g0 in [("LPS", "untreated"), ("LPS_LPS", "untreated"), ("LPS_LPS", "LPS")]:
        C += hc.paired(s, "GSE22248", "GSE22248", g1, g0, S, note=note)
    x, d = hio.gse15219(ROOT, ds["GSE15219"], plat("GPL6255"))
    s, ng = run_dataset("GSE15219", x, d, ["Control"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE15219", d, "monocyte endotoxin tolerance", "culture (2 per arm)", ""))
    for g1, g0 in [("T_L3h", "L3h"), ("T_L6h", "L6h"), ("T_L24h", "L24h"), ("Tol", "Control"),
                   ("L3h", "Control")]:
        C += hc.unpaired(s, "GSE15219", "GSE15219", g1, g0, S, effect_only=True)

    # 4. human HMGB1-stimulated myeloid cells
    x, d = hio.gse254271(ROOT, ds["GSE254271"], entrez2sym)
    s, ng = run_dataset("GSE254271", x, d, ["GMCSF"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE254271", d, "monocytes + HMGB1 (72 h, GM-CSF)", "donor (paired)", ""))
    for g1, g0 in [("HMGB1", "GMCSF"), ("HMGB1_TAK-242", "HMGB1"), ("HMGB1_FPS-ZM1", "HMGB1"),
                   ("S100A9", "GMCSF")]:
        C += hc.paired(s, "GSE254271", "GSE254271", g1, g0, S)
    x, d = hio.gse162834(ROOT, ds["GSE162834"])
    s, ng = run_dataset("GSE162834", x, d, ["control"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE162834", d, "monocytes + HMGB1", "culture (one donor)", ""))
    C += hc.unpaired(s, "GSE162834", "GSE162834", "HMGB1", "control", S, effect_only=True)
    x, d = hio.gse78162(ROOT, ds["GSE78162"])
    _, ng = run_dataset("GSE78162", x, d, ["Control"], sets, cfg)
    ngs.append(ng)                     # gene coverage only; no signature scores from this panel
    designs.append(design_rows("GSE78162", d, "CSF1 macrophages: LPS vs HMGB1 (qPCR panel)",
                               "culture (one donor)", "150-gene panel"))
    # 150-gene panel: signature scores rest on too few genes, so only gene-level comparisons
    C += hc.gene_level_identifiability(x, d, "HMGB1", "LPS", "Control", "GSE78162", "GSE78162")

    # 5. sepsis monocytes, basal and ex vivo LPS
    x, d = hio.gse46955(ROOT, ds["GSE46955"], plat("GPL6104"))
    s, ng = run_dataset("GSE46955", x, d, ["healthy_basal"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE46955", d, "blood monocytes, Gram-negative sepsis", "subject", ""))
    C += hc.unpaired(s, "GSE46955", "GSE46955", "sepsis_basal", "healthy_basal", S)
    C += hc.paired(s, "GSE46955", "GSE46955", "recovery_basal", "sepsis_basal", S)
    C += hc.response_shift(s, "GSE46955", "GSE46955", S, "LPS", "basal", "sepsis", "healthy", False,
                           "LPS response: sepsis vs healthy")
    C += hc.response_shift(s, "GSE46955", "GSE46955", S, "LPS", "basal", "recovery", "sepsis", True,
                           "LPS response: recovery vs sepsis")

    # 6. sepsis monocytes, single cell (GSE252331), summed per donor
    from lib import human_sc
    cnt, ncell, meta, comp = human_sc.donor_monocytes(ROOT / cfg["human_raw"] / "GSE252331", ens2sym,
                                                       cfg["single_cell"], cfg["seed"])
    keep = ncell[ncell >= cfg["single_cell"]["min_monocytes_per_donor"]].index
    cnt = cnt[keep]
    cnt = cnt[(cnt >= 10).sum(axis=1) >= 3]
    xs = hio.log2cpm(cnt)
    dsc = meta[meta.subject.isin(keep)].rename(columns={"prefix": "sample"})
    dsc["sample"] = dsc.subject
    s, ng = run_dataset("GSE252331_mono", xs, dsc, ["healthy"], sets, cfg)
    scores.append(s); ngs.append(ng)
    designs.append(design_rows("GSE252331_mono", dsc, "blood monocytes (single cell, summed per donor)",
                               "donor", "cell-type label is a marker annotation"))
    for g1, g0 in [("sepsis_day4", "healthy"), ("RAP_day14", "healthy"), ("CCI_day14", "healthy"),
                   ("CCI_day14", "RAP_day14")]:
        C += hc.unpaired(s, "GSE252331_mono", "GSE252331_mono", g1, g0, S)
    mono = pd.DataFrame({"subject": ncell.index, "n_monocytes": ncell.values})
    mono = mono.merge(meta[["subject", "group"]], on="subject").merge(
        comp.sum(axis=1).rename("n_cells").reset_index(), on="subject")
    repro.write_table(mono, out / "sc_monocytes_per_donor.tsv")

    # 7. sepsis blood cohorts (secondary)
    bl, per_cohort = [], {}
    cc = cfg["composition"]
    dc = ha.deconvolved(ROOT / cc["deconvolved"])
    adj_sc, adj_mort, mk_all = [], [], []
    for name in cfg["datasets"]["blood"]["cohorts"]:
        x, d = hio.blood_cohort(name, ROOT / cfg["human_processed"], ROOT / cfg["cohort_metadata"])
        if name == "GAINS4421":
            g5273 = pd.read_csv(ROOT / cfg["cohort_metadata"] / "GAINS_samples.tsv", sep="\t")
            d = d[~d.subject.isin(set(g5273.patient))]
        ref = ["control"] if (d.group == "control").sum() >= 3 else list(d.group.unique())
        s, ng = run_dataset(name, x, d, ref, sets, cfg)
        scores.append(s); ngs.append(ng)
        designs.append(design_rows(name, d, "sepsis blood", "patient",
                                   "GAinS E-MTAB-4421 patients also in E-MTAB-5273 removed" if name == "GAINS4421" else ""))
        dmax = cfg["mortality"]["admission_day_max"]
        bl += hb.cohort_contrasts(name, s, d, S, dmax)
        mk = ha.marker_scores(x, cc["markers"])
        mk_all.append(mk.assign(dataset=name).reset_index(names="sample"))
        adj_sc += ha.sepsis_vs_control(name, s, d, mk, cc["sepsis_vs_control_scores"], dmax, hb.first_sample)
        if name in cfg["mortality"]["cohorts"]:
            first = hb.first_sample(d, dmax)
            per_cohort[name] = (s, first)
            adj_mort += ha.mortality_models(name, s, first, mk, dc, cfg["mortality"]["scores"],
                                            cc["tolerance_scores"], cfg["mortality"]["min_deaths"], lvl)
    bl += hb.pool_two_group(bl, "blood: sepsis vs control", "sepsis first sample vs control", lvl)
    bl += hb.mortality(per_cohort, cfg["mortality"]["scores"], cfg["mortality"]["min_deaths"], lvl)

    comp_adj = pd.DataFrame(adj_sc)
    comp_adj = ha.finish(pd.concat([comp_adj, pd.DataFrame(ha.pool(adj_sc, ["analysis", "score", "model"], lvl))],
                                   ignore_index=True), log_or=False)
    repro.write_table(comp_adj, out / "composition_adjusted.tsv")
    # unadjusted fits restricted to the cohorts with deconvolved proportions / age and sex, so that
    # adjusted and unadjusted pooled estimates rest on the same patients
    sub = []
    for m_adj in ("deconvolved proportions", "composition markers + age + sex"):
        ds_m = {r["dataset"] for r in adj_mort if r["model"] == m_adj}
        sub += [{**r, "model": f"unadjusted (cohorts of '{m_adj}')"} for r in adj_mort
                if r["model"] == "unadjusted" and r["dataset"] in ds_m]
    mort = ha.finish(pd.concat([pd.DataFrame(adj_mort),
                                pd.DataFrame(ha.pool(adj_mort + sub, ["analysis", "score", "model"], lvl))],
                               ignore_index=True), log_or=True)
    repro.write_table(mort, out / "mortality_models.tsv")
    repro.write_table(pd.concat(mk_all, ignore_index=True), out / "blood_composition_markers.tsv")

    # ------------------------------------------------------------------ write
    sc_all = pd.concat(scores, ignore_index=True)
    repro.write_table(sc_all, out / "scores_by_sample.tsv")
    repro.write_table(pd.concat(ngs, ignore_index=True), out / "score_gene_counts.tsv")
    repro.write_table(pd.concat(designs, ignore_index=True), out / "design.tsv")
    con = pd.DataFrame(C + bl)
    con = hc.add_fdr(con)
    repro.write_table(con, out / "contrasts.tsv")
    cand = pd.DataFrame(cfg["candidates"], columns=["accession", "question", "design", "decision", "reason"])
    repro.write_table(cand, out / "candidates.tsv")
    (out / "ortholog_map_source.txt").write_text(
        "Mouse-to-human orthologues: MGI vertebrate homology report HOM_MouseHumanSequence.rpt\n"
        "(https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt), downloaded by\n"
        "scripts/stages/00_download_07.sh; md5 in results/07_human/raw_checksums.tsv.\n"
        "Homology classes (DB Class Key) with exactly one mouse (taxon 10090) and one human (taxon 9606)\n"
        "gene were used (one-to-one orthologues); mouse genes in one-to-many or many-to-many classes are\n"
        "left unmapped and counted in signatures_human.tsv (gene = empty).\n"
        f"Homology classes: {info['homology_classes']}; one-to-one: {info['one_to_one_classes']}; "
        f"mouse symbols mapped: {info['mouse_genes_mapped']}.\n"
        "Human symbols from published lists and older arrays were updated with HGNC previous/alias\n"
        "symbols (hgnc_complete_set.txt) when the old symbol mapped to a single approved symbol.\n"
        "Fraction of genes mapped per signature: "
        + "; ".join(f"{k}={v:.2f}" for k, v in info["mapped_fraction"].items()) + "\n")
    inputs = [ROOT / cfg["mouse_signatures"], Path(info["stage06_cache"]), ROOT / cfg["orthologs"]["file"],
              ROOT / cfg["hgnc"], ROOT / cfg["published"]["pena_pdf_text"],
              ROOT / cfg["composition"]["deconvolved"]]
    outputs = [out / f for f in ["candidates.tsv", "design.tsv", "scores_by_sample.tsv", "contrasts.tsv",
                                 "signatures_human.tsv", "score_gene_counts.tsv", "sc_monocytes_per_donor.tsv",
                                 "ortholog_map_source.txt", "raw_checksums.tsv",
                                 "composition_adjusted.tsv", "mortality_models.tsv", "blood_composition_markers.tsv"]]
    repro.write_provenance(STAGE, inputs, outputs, cfg,
                           notes="human relevance; raw file checksums in raw_checksums.tsv")
    log.info("done: %d contrast rows", len(con))


if __name__ == "__main__":
    main()
