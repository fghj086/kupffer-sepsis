"""Per-unit summaries for stage 02: programme comparisons, gene rows, coupling, admixture, projection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scripts.lib.sorted_stats import group_compare, log_cpm, log_norm, score, zscore_vs_reference


def unit_scores(counts, meta, programs, cfg, case, ref, unit):
    """Sample-level programme scores for one dataset x cell type."""
    f = cfg["filter"]
    keep = (counts >= f["min_count"]).sum(axis=1) >= f["min_samples"]
    lx = log_norm(counts[keep], cfg["scoring"]["pseudocount"])
    z = zscore_vs_reference(lx[keep], meta.condition, ref)
    g = meta.condition.reindex(lx.columns)
    centred = lx[keep].sub(lx[keep].loc[:, g == ref].mean(axis=1), axis=0)  # log2 scale, sham = 0
    rows = []
    for name, genes in programs.items():
        mg = cfg["scoring"]["min_genes"] if len(genes) >= 5 else 1
        s, n = score(z, genes, min_genes=mg)
        s2, _ = score(centred.loc[z.dropna().index], genes, min_genes=mg)
        if s is None:
            continue
        for smp, v in s.items():
            rows.append({"unit": unit, "sample": smp, "program": name, "score": v,
                         "score_log2": s2[smp], "n_genes": n, "n_genes_in_set": len(set(genes))})
    sc = pd.DataFrame(rows)
    return sc, z, lx


def program_effects(sc, meta, case, ref, de, programs, alpha, strata=None):
    """CLP minus sham per programme. Primary (diff, CIs, hedges_g, P): the log2-scale score, i.e. the
    mean change in log2 normalised expression over the set, comparable across cell types and in the
    same units as the gene rows. Secondary (z_*): the z-scale score (pooled within-group SD units),
    which up-weights low-variance genes. Plus gene-level direction counts from the DE table."""
    out = []
    for (unit, prog), d in sc.groupby(["unit", "program"]):
        s = d.set_index("sample").score
        subsets = [("all", s.index)]
        if strata is not None:
            for lv in strata.reindex(s.index).unique():
                subsets.append((lv, s.index[strata.reindex(s.index) == lv]))
        genes = set(programs[prog])
        dd = de[de.gene.isin(genes)]
        for lab, idx in subsets:
            row = {"unit": unit, "stratum": lab, "level": "program", "feature": prog,
                   "n_genes": int(d.n_genes.iloc[0])}
            # primary: log2 scale (same units as the gene-level log2FC rows)
            row.update(group_compare(d.set_index("sample").score_log2[idx], meta.condition, case, ref))
            zz = group_compare(s[idx], meta.condition, case, ref)
            row.update({"z_diff": zz["diff"], "z_diff_ci_low": zz["diff_ci_low"],
                        "z_diff_ci_high": zz["diff_ci_high"], "z_p_welch": zz["p_welch"]})
            if lab == "all":
                row.update({"n_up_fdr": int(((dd.padj < alpha) & (dd.log2FoldChange > 0)).sum()),
                            "n_down_fdr": int(((dd.padj < alpha) & (dd.log2FoldChange < 0)).sum()),
                            "median_gene_lfc": float(dd.log2FoldChange.median()) if len(dd) else np.nan,
                            "frac_genes_lfc_pos": float((dd.log2FoldChange > 0).mean()) if len(dd) else np.nan})
            out.append(row)
    return pd.DataFrame(out)


def gene_rows(de, counts, meta, genes, case, ref, unit):
    """Per-gene log2FC with Wald CI and group mean CPM (genes below the filter reported as such)."""
    out = []
    g = meta.condition.reindex(counts.columns)
    cpm = counts / counts.sum() * 1e6
    d = de.set_index("gene")
    for gene in genes:
        row = {"unit": unit, "stratum": "all", "level": "gene", "feature": gene}
        if gene in cpm.index:
            row["cpm_ref_mean"] = float(cpm.loc[gene, g == ref].mean())
            row["cpm_case_mean"] = float(cpm.loc[gene, g == case].mean())
        if gene in d.index and pd.notna(d.loc[gene, "log2FoldChange"]):
            r = d.loc[gene]
            row.update({"diff": r.log2FoldChange, "diff_ci_low": r.ci_low, "diff_ci_high": r.ci_high,
                        "lfcSE": r.lfcSE, "pvalue": r.pvalue, "padj": r.padj, "tested": True})
        else:
            row["tested"] = False
        out.append(row)
    return pd.DataFrame(out)


def coupling(sc, meta, unit, infl, clear):
    """Across mice: Spearman of log2-scale inflammation and clearance scores (all mice, and within each arm)."""
    w = sc[sc.unit == unit].pivot(index="sample", columns="program", values="score_log2")
    out = []
    for a in infl:
        for b in clear:
            if a not in w or b not in w:
                continue
            for lab, idx in [("all_mice", w.index)] + [
                    (lv, w.index[meta.condition.reindex(w.index) == lv])
                    for lv in meta.condition.reindex(w.index).unique()]:
                x, y = w.loc[idx, a], w.loc[idx, b]
                r, p = stats.spearmanr(x, y)
                out.append({"unit": unit, "analysis": "score_correlation", "program_a": a, "program_b": b,
                            "subset": lab, "n": len(idx), "spearman_rho": r, "p": p})
    return pd.DataFrame(out)


def kc_admixture(counts_by_unit, meta, de_cd45, cfg):
    """Fraction of CD45+ non-KC-like transcripts in each sorted-KC library (GSE311736).

    Marker genes: >= min_ratio-fold higher in sham CD45+ non-KC than in sham KC, CPM >= min_cpm in
    CD45+ non-KC, and |log2FC| <= max_abs_lfc in CD45+ non-KC after CLP. For each KC library the
    admixture index is the median over markers of CPM(KC library) / mean CPM(CD45+ non-KC, same arm).
    """
    a = cfg["admixture"]
    kc, cd = counts_by_unit["GSE311736_KC"], counts_by_unit["GSE311736_CD45_nonKC"]
    genes = kc.index.intersection(cd.index)
    cpm_kc = (kc.loc[genes] / kc.sum() * 1e6)
    cpm_cd = (cd.loc[genes] / cd.sum() * 1e6)
    cond = meta.condition
    sham_kc = cpm_kc.loc[:, cond.reindex(cpm_kc.columns) == "sham"].mean(axis=1)
    sham_cd = cpm_cd.loc[:, cond.reindex(cpm_cd.columns) == "sham"].mean(axis=1)
    stable = de_cd45.set_index("gene").log2FoldChange.abs().reindex(genes) <= a["max_abs_lfc_cd45"]
    sel = (sham_cd >= a["min_cpm_cd45"]) & (sham_cd >= a["min_ratio"] * (sham_kc + 0.1)) & stable.fillna(False)
    markers = list(genes[sel.values])
    rows = []
    for smp in cpm_kc.columns:
        arm = cond[smp]
        ref = cpm_cd.loc[markers, cond.reindex(cpm_cd.columns) == arm].mean(axis=1)
        rows.append({"sample": smp, "condition": arm,
                     "admixture_index": float(np.median(cpm_kc.loc[markers, smp] / ref))})
    return pd.DataFrame(rows), markers


def project_signatures(sig, z_by_unit, meta, case_ref, min_genes):
    """Weighted mean z of each stimulus signature per sample; CLP minus sham per unit."""
    scores, comps = [], []
    for unit, z in z_by_unit.items():
        case, ref = case_ref[unit]
        for name, d in sig.groupby("signature"):
            d = d.drop_duplicates("gene")
            s, n = score(z, d.gene.values, d.weight.values, min_genes=min_genes)
            if s is None:
                continue
            for smp, v in s.items():
                scores.append({"unit": unit, "sample": smp, "signature": name, "score": v,
                               "n_genes": n, "n_genes_in_signature": len(d)})
            r = {"unit": unit, "signature": name, "n_genes": n, "n_genes_in_signature": len(d)}
            r.update(group_compare(s, meta.condition, case, ref))
            comps.append(r)
    comps = pd.DataFrame(comps)
    if len(comps):
        comps["padj_welch_bh"] = multipletests(comps.p_welch, method="fdr_bh")[1]
    return pd.DataFrame(scores), comps


def purity_markers(counts, meta, markers):
    """Mean CPM of cell-type marker genes per unit and arm (sort purity / carry-over check)."""
    rows = []
    for (unit, cond), d in meta.groupby(["unit", "condition"]):
        c = counts[unit][d.index]
        cpm = c / c.sum() * 1e6
        for ct, genes in markers.items():
            for gname in genes:
                v = cpm.loc[gname] if gname in cpm.index else None
                rows.append({"unit": unit, "condition": cond, "marker_of": ct, "gene": gname,
                             "cpm_mean": float(v.mean()) if v is not None else np.nan,
                             "cpm_min": float(v.min()) if v is not None else np.nan,
                             "cpm_max": float(v.max()) if v is not None else np.nan})
    return pd.DataFrame(rows)
