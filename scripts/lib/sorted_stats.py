"""Differential expression, sample-level programme scores and effect sizes for stage 02."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def filter_genes(counts: pd.DataFrame, min_count: int, min_samples: int) -> pd.DataFrame:
    return counts[(counts >= min_count).sum(axis=1) >= min_samples]


def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame, design: str, contrast: list,
              n_cpus: int = 8, ci_level: float = 0.95, alpha: float = 0.05) -> pd.DataFrame:
    """pydeseq2 Wald test; returns MLE log2FC with Wald CI (mouse = one library = unit)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    inf = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(counts=counts.T.astype(int), metadata=meta, design=design,
                       refit_cooks=True, inference=inf, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, alpha=alpha, inference=inf, quiet=True)
    st.summary()
    res = st.results_df.copy()
    z = stats.norm.ppf(0.5 + ci_level / 2)
    res["ci_low"] = res.log2FoldChange - z * res.lfcSE
    res["ci_high"] = res.log2FoldChange + z * res.lfcSE
    res.index.name = "gene"
    return res.reset_index()


def log_cpm(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    lib = counts.sum(axis=0)
    return np.log2(counts / lib * 1e6 + pseudocount)


def median_ratio_size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios size factors (genes with a zero in any library are skipped)."""
    pos = counts[(counts > 0).all(axis=1)]
    logc = np.log(pos)
    return np.exp((logc.sub(logc.mean(axis=1), axis=0)).median(axis=0))


def log_norm(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """log2(median-of-ratios normalised count + pseudocount); robust to a few highly induced genes."""
    return np.log2(counts / median_ratio_size_factors(counts) + pseudocount)


def zscore_vs_reference(logx: pd.DataFrame, group: pd.Series, ref: str) -> pd.DataFrame:
    """Per gene: centre on the reference-group mean, scale by the pooled within-group SD."""
    g = group.reindex(logx.columns)
    mu_ref = logx.loc[:, g == ref].mean(axis=1)
    ss, df = 0.0, 0
    for lv in g.unique():
        x = logx.loc[:, g == lv]
        ss = ss + ((x.sub(x.mean(axis=1), axis=0)) ** 2).sum(axis=1)
        df += x.shape[1] - 1
    sd = np.sqrt(ss / max(df, 1))
    sd = sd.where(sd > 0)
    return logx.sub(mu_ref, axis=0).div(sd, axis=0)


def score(z: pd.DataFrame, genes, weights=None, min_genes: int = 5):
    """Weighted mean z over the genes present (weights normalised by sum |w|)."""
    w = pd.Series(1.0, index=list(genes)) if weights is None else pd.Series(weights, index=list(genes))
    w = w[~w.index.duplicated()]
    w = w[w.index.isin(z.dropna(how="any").index)]
    if len(w) < min_genes:
        return None, len(w)
    s = (z.loc[w.index].mul(w, axis=0)).sum(axis=0) / w.abs().sum()
    return s, len(w)


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Hedges g (a minus b) with an approximate 95% CI (Hedges & Olkin variance)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if not sp > 0:
        return np.nan, np.nan, np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    g = j * (a.mean() - b.mean()) / sp
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g, g - 1.96 * se, g + 1.96 * se


def group_compare(s: pd.Series, group: pd.Series, case: str, ref: str) -> dict:
    """Case minus reference: mean difference with Welch CI, Hedges g, Welch t and exact Mann-Whitney P."""
    a = s[group.reindex(s.index) == case].values
    b = s[group.reindex(s.index) == ref].values
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.nan
    tq = stats.t.ppf(0.975, dfw) if se > 0 else np.nan
    g, glo, ghi = hedges_g(a, b)
    return {
        "n_case": len(a), "n_ref": len(b), "mean_case": a.mean(), "mean_ref": b.mean(),
        "diff": diff, "diff_ci_low": diff - tq * se, "diff_ci_high": diff + tq * se,
        "hedges_g": g, "g_ci_low": glo, "g_ci_high": ghi,
        "p_welch": stats.ttest_ind(a, b, equal_var=False).pvalue,
        "p_mannwhitney": stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue,
    }


def lfc_concordance(a: pd.DataFrame, b: pd.DataFrame, alpha: float, min_lfc: float) -> list[dict]:
    """Compare two DE tables on their shared genes: correlation and sign concordance."""
    m = a.merge(b, on="gene", suffixes=("_a", "_b")).dropna(subset=["log2FoldChange_a", "log2FoldChange_b"])
    out = []
    sig_a = (m.padj_a < alpha) & (m.log2FoldChange_a.abs() >= min_lfc)
    sig_b = (m.padj_b < alpha) & (m.log2FoldChange_b.abs() >= min_lfc)
    for label, sel in [("all_shared", np.ones(len(m), bool)), ("responsive_in_either", sig_a | sig_b),
                       ("responsive_in_both", sig_a & sig_b), ("responsive_in_a", sig_a),
                       ("responsive_in_b", sig_b)]:
        x = m.loc[sel, "log2FoldChange_a"].values
        y = m.loc[sel, "log2FoldChange_b"].values
        row = {"gene_subset": label, "n_genes": int(sel.sum())}
        if sel.sum() >= 3:
            row.update({
                "pearson_r": stats.pearsonr(x, y)[0], "spearman_rho": stats.spearmanr(x, y)[0],
                "sign_concordance": float(np.mean(np.sign(x) == np.sign(y))),
                "p_sign_binomial": stats.binomtest(int(np.sum(np.sign(x) == np.sign(y))), len(x)).pvalue,
                "slope_b_on_a": float(np.polyfit(x, y, 1)[0]),
            })
        out.append(row)
    # overlap of responsive genes (hypergeometric, universe = shared tested genes)
    up_a, up_b = sig_a & (m.log2FoldChange_a > 0), sig_b & (m.log2FoldChange_b > 0)
    for lab, sa, sb in [("up", up_a, up_b), ("down", sig_a & ~up_a, sig_b & ~up_b)]:
        k, na, nb, n = int((sa & sb).sum()), int(sa.sum()), int(sb.sum()), len(m)
        out.append({"gene_subset": f"overlap_{lab}", "n_genes": k, "n_a": na, "n_b": nb, "universe": n,
                    "jaccard": k / max(na + nb - k, 1),
                    "p_hypergeom": stats.hypergeom.sf(k - 1, n, na, nb) if na and nb else np.nan})
    return out
