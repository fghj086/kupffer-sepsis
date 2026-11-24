"""Stage 01 helpers: candidate-gene tables, clearance gene-set tests and enrichment."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("01_perturbation")


def read_gmt(path) -> dict[str, list[str]]:
    out = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[f[0]] = [g for g in f[2:] if g]
    return out


def candidate_table(groups: dict[str, list[str]], counts, meta, res: dict,
                    gene_tab: pd.DataFrame, pseudo: float) -> pd.DataFrame:
    """Per gene: mean CPM per condition plus log2FC / padj for every contrast.
    Genes not passing the expression filter are listed with detected = False."""
    lib = counts.sum(axis=0)
    cpm = counts.div(lib, axis=1) * 1e6
    conds = list(dict.fromkeys(meta.loc[counts.columns, "condition"]))
    rows = []
    for grp, genes in groups.items():
        for g in genes:
            r = {"group": grp, "gene": g, "detected": g in counts.index,
                 "ensembl_gene_id": gene_tab["ensembl_gene_id"].get(g, "")}
            if r["detected"]:
                for c in conds:
                    smp = meta.index[meta.condition == c]
                    r[f"meanCPM_{c}"] = float(cpm.loc[g, smp].mean())
                for k, d in res.items():
                    r[f"log2FC_{k}"] = float(d.loc[g, "log2FC"])
                    r[f"padj_{k}"] = float(d.loc[g, "padj"]) if pd.notna(d.loc[g, "padj"]) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def clearance_set_tests(sets: dict[str, list[str]], res: dict, universe) -> pd.DataFrame:
    """Gene-set summaries for clearance programmes.

    mean_log2FC_<contrast>: average response of set genes.
    competitive test: Mann-Whitney of the Wald statistic of set genes vs all other tested
    genes (two-sided), for dsHMGB1 vs LPS and for the magnitude-adjusted contrast.
    """
    rows = []
    universe = pd.Index(universe)
    for name, genes in sets.items():
        g = universe.intersection(genes)
        r = {"gene_set": name, "n_genes_in_set": len(genes), "n_tested": len(g)}
        for k, d in res.items():
            r[f"mean_log2FC_{k}"] = float(d.loc[g, "log2FC"].mean())
        for k in ["dsHMGB1_vs_LPS", "dsHMGB1_vs_scaledLPS"]:
            st = res[k]["stat"].dropna()
            ins, out = st[st.index.isin(g)], st[~st.index.isin(g)]
            u = stats.mannwhitneyu(ins, out, alternative="two-sided")
            r[f"median_stat_{k}"] = float(ins.median())
            r[f"auc_{k}"] = float(u.statistic / (len(ins) * len(out)))
            r[f"mwu_p_{k}"] = float(u.pvalue)
        rows.append(r)
    df = pd.DataFrame(rows)
    for k in ["dsHMGB1_vs_LPS", "dsHMGB1_vs_scaledLPS"]:
        df[f"mwu_fdr_{k}"] = multipletests(df[f"mwu_p_{k}"], method="fdr_bh")[1]
    return df


def ora(query: list[str], universe, libs: dict[str, dict], min_size: int,
        max_size: int) -> pd.DataFrame:
    """Hypergeometric over-representation against the tested-gene universe; BH within
    each library."""
    U = set(universe)
    q = set(query) & U
    out = []
    for lib, gsets in libs.items():
        rows = []
        for term, genes in gsets.items():
            s = set(genes) & U
            if not (min_size <= len(s) <= max_size):
                continue
            k = len(q & s)
            if k == 0:
                continue
            p = stats.hypergeom.sf(k - 1, len(U), len(s), len(q))
            rows.append({"library": lib, "term": term, "overlap": k, "set_size": len(s),
                         "query_size": len(q), "odds_ratio": (k / len(q)) / (len(s) / len(U)),
                         "pvalue": p, "genes": ";".join(sorted(q & s))})
        d = pd.DataFrame(rows)
        if len(d):
            # BH over every eligible term in the library (terms with zero overlap have P=1)
            n_terms = sum(min_size <= len(set(g) & U) <= max_size for g in gsets.values())
            pv = np.concatenate([d.pvalue.values, np.ones(n_terms - len(d))])
            d["fdr"] = multipletests(pv, method="fdr_bh")[1][:len(d)]
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def prerank(stat: pd.Series, gsets: dict, cfg: dict, seed: int) -> pd.DataFrame:
    import gseapy as gp

    rnk = stat.dropna().sort_values(ascending=False)
    pre = gp.prerank(rnk=rnk, gene_sets=gsets, min_size=cfg["min_set_size"],
                     max_size=cfg["max_set_size"], permutation_num=cfg["prerank_permutations"],
                     seed=seed, threads=8, outdir=None, verbose=False, no_plot=True)
    r = pre.res2d.rename(columns={"Term": "term", "NES": "nes", "NOM p-val": "pvalue",
                                  "FDR q-val": "fdr", "Lead_genes": "genes", "ES": "es"})
    r["overlap"] = r["Tag %"].astype(str)
    return r[["term", "es", "nes", "pvalue", "fdr", "overlap", "genes"]]
