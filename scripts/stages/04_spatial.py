"""Stage 04b — liver in whole-mouse spatial transcriptomics after LPS (GSE248904, Array-seq,
2 control + 2 LPS 24 h). Exploratory: the animal is the unit, effect sizes only.

Outputs (results/04_spatial/): liver_spots_summary.tsv (animal x subregion x score),
spatial_effects.tsv (LPS minus control per score), colocalization.tsv (per animal and
LPS-minus-control; spot and ~200-um tile resolution), morans_i.tsv, tile_scores.tsv,
spot_scores_subsample.tsv.gz, genes_used.tsv,
PROVENANCE.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import (ROOT, load_config, results_dir, set_global_seed,  # noqa: E402
                       write_provenance, write_table)
from lib.spatial_io import extract_liver  # noqa: E402
from lib.spatial_scores import (bin_counts, hedges_g, log_norm, morans_i,  # noqa: E402
                                neighbours, ring_neighbours, set_score, spatial_lag,
                                std_ols)


def score_table(Xn, var_names, cfg):
    sc = cfg["spatial"]
    sets = {k: cfg["gene_sets"][k] for k in ("kc_identity", "recruited_mono_mac",
                                               "nfkb_inflammatory", "clearance_phagocytosis")}
    sets.update({"hepatocyte_stress": sc["hepatocyte_stress"],
                 "hepatocyte_markers": sc["hepatocyte_markers"],
                 "pericentral": sc["zonation"]["pericentral"],
                 "periportal": sc["zonation"]["periportal"]})
    S, used = {}, []
    for k, genes in sets.items():
        S[k], g = set_score(Xn, list(var_names), genes)
        used.append({"score": k, "n_requested": len(genes), "n_used": len(g),
                     "genes_used": ",".join(g)})
    S = pd.DataFrame(S)
    S["zonation_index"] = S["pericentral"] - S["periportal"]  # + pericentral, - periportal
    return S, pd.DataFrame(used)


def tiles(a, cfg):
    """Tile-level table: counts summed over spots in each tile, scored on a shared scale."""
    sc = cfg["spatial"]
    Xs, libs, meta = [], [], []
    for smp in sorted(a.obs["Sample"].unique()):
        m = (a.obs["Sample"] == smp).to_numpy()
        o = a.obs.loc[m]
        coords = o[["x_plotting", "y_plotting"]].to_numpy()
        Xb, lb, cen, n, spot2tile, ok = bin_counts(a.X[m], coords, o["total_counts"].to_numpy(),
                                                   sc["bin_size"], sc["bin_min_spots"])
        pp = pd.Series((o["Subregion"] == "Periportal").to_numpy()).groupby(spot2tile).mean()
        meta.append(pd.DataFrame({"Sample": smp, "Treatment": o["Treatment"].iloc[0],
                                  "x_plotting": cen[:, 0], "y_plotting": cen[:, 1],
                                  "n_spots": n, "total_counts": lb,
                                  "periportal_fraction": pp.reindex(range(len(n))).to_numpy()}))
        Xs.append(Xb)
        libs.append(lb)
    import scipy.sparse as sp
    X = sp.vstack(Xs).tocsr()
    S, _ = score_table(log_norm(X, np.concatenate(libs)), a.var_names, cfg)
    return pd.concat([pd.concat(meta, ignore_index=True), S], axis=1)


def per_animal(df, scores):
    rows = []
    for (smp, trt), d in df.groupby(["Sample", "Treatment"], observed=True):
        for sub, dd in [("all_liver", d)] + [(s, x) for s, x in d.groupby("Subregion", observed=True)
                                             if s in ("Pericentral", "Periportal")]:
            for s in scores:
                rows.append({"sample": smp, "treatment": trt, "subregion": sub,
                             "score": s, "n_spots": len(dd), "mean": dd[s].mean(),
                             "median": dd[s].median()})
    return pd.DataFrame(rows)


def group_effects(summ, value="mean"):
    rows = []
    for (sub, s), d in summ.groupby(["subregion", "score"]):
        a = d.loc[d.treatment == "LPS", value].to_numpy()
        b = d.loc[d.treatment == "Control", value].to_numpy()
        rows.append({"subregion": sub, "score": s, "n_lps": len(a), "n_control": len(b),
                     "lps_animals": ";".join(f"{x:.3f}" for x in a),
                     "control_animals": ";".join(f"{x:.3f}" for x in b),
                     "lps_minus_control": a.mean() - b.mean(), "hedges_g": hedges_g(a, b),
                     "ranges_separate": bool(a.min() > b.max() or a.max() < b.min())})
    return pd.DataFrame(rows)


def colocalise(d, cfg, k, ring_ranks, pp):
    """Hepatocyte stress regressed on the neighbourhood mean (spatial lag) of each myeloid
    score, adjusted for the unit's own myeloid score, zonation index, periportal
    annotation, hepatocyte marker score and depth. The same model with a non-adjacent ring
    of neighbours shows how much of any association is local rather than regional."""
    sc = cfg["spatial"]
    coords = d[["x_plotting", "y_plotting"]].to_numpy()
    nb = neighbours(coords, k)
    ring = ring_neighbours(coords, *ring_ranks)
    nn_dist = np.median(np.linalg.norm(coords[nb[:, 0]] - coords, axis=1))
    covs = {"zonation_index": d["zonation_index"].to_numpy(), "periportal": pp,
            "hepatocyte_markers": d["hepatocyte_markers"].to_numpy(),
            "log_total_counts": np.log(d["total_counts"].to_numpy())}
    rows, mor = [], []
    y = d["hepatocyte_stress"].to_numpy()
    for m in sc["myeloid_sets"]:
        x = d[m].to_numpy()
        rows.append({"predictor": m, "neighbourhood": "own_unit",
                     "spearman_stress_vs_lag": pd.Series(y).corr(pd.Series(x), "spearman"),
                     **_beta(y, pd.DataFrame({"lag": x, **covs}))})
        for kind, nbr in (("adjacent", nb), ("ring", ring)):
            lag = spatial_lag(x, nbr)
            rho = pd.Series(y).corr(pd.Series(lag), method="spearman")
            rows.append({"predictor": m, "neighbourhood": kind,
                         "spearman_stress_vs_lag": rho,
                         **_beta(y, pd.DataFrame({"lag": lag, "own": x, **covs}))})
    for s in ["hepatocyte_stress", "zonation_index"] + sc["myeloid_sets"]:
        mor.append({"score": s, "morans_i": morans_i(d[s].to_numpy(), nb)})
    return pd.DataFrame(rows), pd.DataFrame(mor), nn_dist


def _beta(y, X):
    fit = std_ols(y, X)
    return {"beta_adj": fit.loc["lag", "beta"], "beta_ci_low": fit.loc["lag", "ci_low"],
            "beta_ci_high": fit.loc["lag", "ci_high"],
            "beta_zonation": fit.loc["zonation_index", "beta"]}


def run_coloc(df, cfg, resolution, k, ring_ranks, pp_col):
    col_rows, mor_rows = [], []
    for (smp, trt), d in df.groupby(["Sample", "Treatment"], observed=True):
        pp = (d["Subregion"] == "Periportal").astype(float).to_numpy() if pp_col is None \
            else d[pp_col].to_numpy()
        c, m, nn_dist = colocalise(d, cfg, k, ring_ranks, pp)
        col_rows.append(c.assign(sample=smp, treatment=trt, n_units=len(d),
                                 median_nn_distance=nn_dist, resolution=resolution))
        mor_rows.append(m.assign(sample=smp, treatment=trt, resolution=resolution))
    col = pd.concat(col_rows, ignore_index=True)
    g = []
    for (p, kind), d in col.groupby(["predictor", "neighbourhood"]):
        a_, b_ = d.loc[d.treatment == "LPS"], d.loc[d.treatment == "Control"]
        g.append({"sample": "LPS_minus_control", "treatment": "contrast", "predictor": p,
                  "neighbourhood": kind, "resolution": resolution,
                  "beta_adj": a_.beta_adj.mean() - b_.beta_adj.mean(),
                  "spearman_stress_vs_lag": (a_.spearman_stress_vs_lag.mean()
                                             - b_.spearman_stress_vs_lag.mean())})
    return pd.concat([col, pd.DataFrame(g)], ignore_index=True), pd.concat(mor_rows)


def main():
    cfg = load_config("config/04_timecourse_spatial.yaml")
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    sc = cfg["spatial"]
    out = results_dir(cfg["stage_spatial"])
    genes = (sum((cfg["gene_sets"][k] for k in cfg["gene_sets"]), [])
             + sc["hepatocyte_stress"] + sc["hepatocyte_markers"]
             + sc["zonation"]["pericentral"] + sc["zonation"]["periportal"])
    a = extract_liver(ROOT / sc["h5ad"], ROOT / sc["liver_cache"], genes)

    S, used = score_table(log_norm(a.X, a.obs["total_counts"].to_numpy()), a.var_names, cfg)
    S.index = a.obs_names
    df = pd.concat([a.obs, S], axis=1)
    scores = list(S.columns)
    summ = per_animal(df, scores)
    eff = group_effects(summ)

    col_s, mor_s = run_coloc(df, cfg, "spot", sc["n_neighbors"], sc["ring_neighbours"], None)
    T = tiles(a, cfg)
    col_t, mor_t = run_coloc(T, cfg, f"tile_{sc['bin_size']:g}", sc["bin_n_neighbors"],
                             sc["bin_ring_neighbours"], "periportal_fraction")
    col = pd.concat([col_s, col_t], ignore_index=True)
    mor = pd.concat([mor_s, mor_t], ignore_index=True)

    sub = pd.concat([d.sample(min(len(d), sc["subsample_per_sample"]),
                              random_state=int(rng.integers(1 << 31)))
                     for _, d in df.groupby("Sample", observed=True)])
    keep = ["Sample", "Treatment", "Subregion", "x_plotting", "y_plotting", "total_counts"]
    sub = sub[keep + scores].rename_axis("spot").reset_index()

    paths = [
        write_table(summ, out / "liver_spots_summary.tsv"),
        write_table(eff, out / "spatial_effects.tsv"),
        write_table(col, out / "colocalization.tsv"),
        write_table(mor, out / "morans_i.tsv"),
        write_table(used, out / "genes_used.tsv"),
        write_table(T, out / "tile_scores.tsv"),
    ]
    p = out / "spot_scores_subsample.tsv.gz"
    sub.to_csv(p, sep="\t", index=False, float_format="%.5g", compression="gzip")
    paths.append(p)
    write_provenance(cfg["stage_spatial"], inputs=[ROOT / (sc["h5ad"] + ".gz"), ROOT / sc["liver_cache"]],
                     outputs=paths, cfg=cfg,
                     notes="GSE248904 liver spots (authors' Organ == LI annotation); 2 control "
                           "+ 2 LPS animals; effect sizes only; spot-level CIs descriptive")
    print(f"[04_spatial] liver spots: {a.n_obs}; wrote {len(paths)} files to {out}")


if __name__ == "__main__":
    main()
