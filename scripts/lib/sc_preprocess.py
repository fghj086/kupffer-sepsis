"""QC, normalisation, clustering and marker-based annotation for one scRNA-seq dataset.

Every dataset is processed on its own. Harmony is applied only across the libraries of a
single dataset (never across studies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def qc_filter(ad: sc.AnnData, q: dict, seed: int) -> tuple[sc.AnnData, pd.DataFrame]:
    """Per-library QC; returns the filtered object and a per-library QC summary."""
    ad.var["mt"] = ad.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    ad.obs["doublet_score"] = np.nan
    ad.obs["predicted_doublet"] = False
    if q.get("scrublet", True):
        for lib in ad.obs["library"].unique():
            m = (ad.obs["library"] == lib).values
            sub = ad[m].copy()
            try:
                sc.pp.scrublet(sub, random_state=seed)
                ad.obs.loc[sub.obs_names, "doublet_score"] = sub.obs["doublet_score"].values
                ad.obs.loc[sub.obs_names, "predicted_doublet"] = sub.obs["predicted_doublet"].values
            except Exception as e:  # scrublet can fail on tiny libraries
                print(f"scrublet skipped for {lib}: {e}")
    before = ad.obs.groupby("library", observed=True).size().rename("cells_raw")
    keep = (
        (ad.obs["n_genes_by_counts"] >= q["min_genes"])
        & (ad.obs["n_genes_by_counts"] <= q["max_genes"])
        & (ad.obs["total_counts"] >= q["min_counts"])
        & (ad.obs["pct_counts_mt"] <= q["max_pct_mt"])
        & (~ad.obs["predicted_doublet"].astype(bool))
    )
    ad = ad[keep.values].copy()
    sc.pp.filter_genes(ad, min_cells=q["min_cells_per_gene"])
    after = ad.obs.groupby("library", observed=True).size().rename("cells_qc")
    summ = pd.concat([before, after], axis=1).reset_index()
    med = ad.obs.groupby("library", observed=True)[["total_counts", "n_genes_by_counts",
                                                    "pct_counts_mt"]].median()
    summ = summ.merge(med.add_prefix("median_").reset_index(), on="library", how="left")
    return ad, summ


def normalise_embed(ad: sc.AnnData, p: dict, seed: int, batch: str | None) -> sc.AnnData:
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=p["target_sum"])
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=p["n_hvg"], flavor="seurat",
                                batch_key=batch)
    embed(ad, p["n_pcs"], p["n_neighbors"], p["leiden_resolution"], seed, batch, key="leiden")
    return ad


def embed(ad, n_pcs, n_neighbors, res, seed, batch, key="leiden", hvg=True):
    sub = ad[:, ad.var["highly_variable"]].copy() if hvg else ad.copy()
    sc.pp.scale(sub, max_value=10)
    sc.pp.pca(sub, n_comps=n_pcs, random_state=seed)
    ad.obsm["X_pca"] = sub.obsm["X_pca"]
    rep = "X_pca"
    if batch is not None and ad.obs[batch].nunique() > 1:
        import harmonypy
        ho = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs, batch, random_state=seed,
                                   max_iter_harmony=20)
        z = np.asarray(ho.Z_corr)
        ad.obsm["X_pca_harmony"] = z.T if z.shape[0] != ad.n_obs else z
        rep = "X_pca_harmony"
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep=rep, random_state=seed)
    sc.tl.umap(ad, random_state=seed)
    sc.tl.leiden(ad, resolution=res, key_added=key, random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)


def score_sets(ad: sc.AnnData, sets: dict, seed: int, prefix: str = "") -> list[str]:
    """scanpy score_genes per set (control genes matched on expression bins); returns columns."""
    cols = []
    for name, genes in sets.items():
        g = [x for x in genes if x in ad.var_names]
        col = f"{prefix}{name}"
        if len(g) < 2:
            ad.obs[col] = np.nan
        else:
            sc.tl.score_genes(ad, g, score_name=col, random_state=seed, use_raw=False)
        cols.append(col)
    return cols


def annotate_major(ad: sc.AnnData, markers: dict, seed: int, key="leiden",
                   min_hb_frac: float = 0.05) -> pd.DataFrame:
    """Each cluster takes the cell type whose marker score has the highest cluster mean
    (scores z-scored across clusters first, so sets of different size are comparable)."""
    cols = score_sets(ad, markers, seed, prefix="mk_")
    means = ad.obs.groupby(key, observed=True)[cols].mean()
    z = (means - means.mean()) / means.std(ddof=0)
    z.columns = [c[3:] for c in z.columns]
    lab = z.idxmax(axis=1)
    # haemoglobin read fraction separates true erythroid cells from low-complexity droplets
    # dominated by ambient RNA, which otherwise score highest on the erythroid set
    C = ad.layers["counts"]
    hb = ad.var_names.str.startswith(("Hba-", "Hbb-"))
    frac = np.asarray(C[:, hb].sum(1)).ravel() / np.maximum(np.asarray(C.sum(1)).ravel(), 1)
    hb_med = pd.Series(frac, index=ad.obs_names).groupby(ad.obs[key], observed=True).median()
    lab = lab.where(~((lab == "Erythroid") & (hb_med.reindex(lab.index) < min_hb_frac)),
                    "Low quality (ambient-rich)")
    ad.obs["cell_type"] = ad.obs[key].map(lab).astype("category")
    out = z.copy()
    out["median_hb_read_fraction"] = hb_med.reindex(out.index).values
    out["median_n_genes"] = ad.obs.groupby(key, observed=True)["n_genes_by_counts"].median().reindex(out.index).values
    out["assigned"] = lab
    out["n_cells"] = ad.obs[key].value_counts().reindex(out.index).values
    return out.reset_index().rename(columns={key: "cluster"})


def annotate_myeloid(ad: sc.AnnData, cfg: dict, seed: int) -> tuple[sc.AnnData, pd.DataFrame]:
    """Recluster cells annotated Kupffer or MonoMac and label subclusters by rule."""
    p, ms = cfg["processing"], cfg["myeloid_subsets"]
    my = ad[ad.obs["cell_type"].isin(cfg["myeloid_parents"])].copy()
    batch = "library" if my.obs["library"].nunique() > 2 and cfg["_integrate"] else None
    sc.pp.highly_variable_genes(my, n_top_genes=2000, flavor="seurat",
                                batch_key=batch)
    embed(my, p["myeloid_n_pcs"], p["n_neighbors"], p["myeloid_resolution"], seed, batch,
          key="my_leiden")
    sets = {k: v for k, v in ms.items() if isinstance(v, list)}
    cols = score_sets(my, sets, seed, prefix="ms_")
    means = my.obs.groupby("my_leiden", observed=True)[cols].mean()
    z = (means - means.mean()) / means.std(ddof=0)
    z.columns = [c[3:] for c in z.columns]

    hi, lo = ms["kc_resident_min_z"], ms["kc_like_min_z"]

    def rule(r):
        if r["neutrophil_contam"] > 1 and r["kc_core"] < lo:
            return "drop"
        if r["kc_core"] >= hi:
            return "Resident KC"
        if r["kc_core"] >= lo:
            return "KC-like (KC-marker-intermediate)"
        return "Monocytes" if r["monocyte"] >= r["macrophage"] else "Recruited Mo-macrophages"

    lab = z.apply(rule, axis=1)
    my.obs["myeloid_subset"] = my.obs["my_leiden"].map(lab)
    tab = z.copy()
    tab["assigned"] = lab
    tab["n_cells"] = my.obs["my_leiden"].value_counts().reindex(tab.index).values
    my = my[my.obs["myeloid_subset"] != "drop"].copy()
    return my, tab.reset_index().rename(columns={"my_leiden": "subcluster"})


def top_markers(ad: sc.AnnData, key: str, n: int = 15) -> pd.DataFrame:
    """Wilcoxon one-vs-rest ranking used only to document labels (no inference)."""
    sc.tl.rank_genes_groups(ad, key, method="wilcoxon", use_raw=False, pts=True)
    df = sc.get.rank_genes_groups_df(ad, None)
    df = df.sort_values(["group", "scores"], ascending=[True, False]).groupby("group").head(n)
    return df[["group", "names", "logfoldchanges", "pct_nz_group", "pct_nz_reference"]].rename(
        columns={"names": "gene", "logfoldchanges": "log2fc_vs_rest"})


def as_csr(x):
    return x.tocsr() if sp.issparse(x) else sp.csr_matrix(x)
