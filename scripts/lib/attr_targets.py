"""Stage 06 targets: log2FC vectors of septic-liver contrasts from stages 02-05.

Bulk DE tables (stage 02) are read as written, with genes of low mean count removed and the
per-gene log2FC shrunk towards 0 by its standard error (normal-normal shrinkage,
lfc * tau^2 / (tau^2 + se^2)) so that noisy low-count genes do not dominate the fit.
GSE224127 liver contrasts (stage 04 design) are computed here, each count file against its own
reference arm, both as group means and per mouse.
Pooled single-cell libraries (GSE279167) give one pseudobulk log2FC per population (effect size;
cells are not replicates).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib.attr_ref import Study

log = logging.getLogger("06_attribution")


def shrink(lfc: pd.Series, se: pd.Series) -> pd.Series:
    ok = lfc.notna() & se.notna()
    tau2 = max(float(np.var(lfc[ok]) - np.median(se[ok] ** 2)), 0.05)
    return lfc * tau2 / (tau2 + se ** 2)


def stage02_targets(cfg, root: Path) -> list[dict]:
    t = cfg["targets"]["stage02"]
    out = []
    for name, spec in t["files"].items():
        p = root / t["dir"] / spec["file"]
        if not p.exists():
            log.warning("missing %s", p)
            continue
        d = pd.read_csv(p, sep="\t")
        d = d[d.baseMean >= t["min_base_mean"]].dropna(subset=["log2FoldChange", "lfcSE"])
        d = d.drop_duplicates("gene").set_index("gene")
        y = shrink(d["log2FoldChange"], d["lfcSE"])
        out.append({"target": name, "source": f"stage02:{spec['file']}", "family": spec["family"],
                    "cell": spec["cell"], "time_h": spec["time_h"], "kind": "stimulus", "y": y,
                    "inputs": [p]})
    return out


def stage05_targets(cfg, root: Path) -> list[dict]:
    t = cfg["targets"]["stage05"]
    out = []
    for name, spec in t["files"].items():
        p = root / t["dir"] / spec["file"]
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t")
        d = d[d["tested"].astype(str).eq("True")].dropna(subset=["log2FC"])
        d = d.drop_duplicates("gene").set_index("gene")
        out.append({"target": name, "source": f"stage05:{spec['file']}", "family": spec["family"],
                    "cell": spec["cell"], "time_h": spec["time_h"], "kind": "stimulus",
                    "y": d["log2FC"], "inputs": [p]})
    return out


def _gse_study(rs, cfg, root, file_key):
    """Study object for one GSE224127 count file (reuses the reference cache when possible)."""
    for st in rs.studies.values():
        if getattr(st, "file_key", None) == file_key:
            return st
    from lib import attr_io
    gm = attr_io.gene_map(root / cfg["gene_map"])
    X, arms = attr_io.load_gse224127(root, {"file_key": file_key}, gm, cfg)
    st = Study(f"GSE224127_{file_key}", X, arms, "counts", cfg)
    st.file_key = file_key
    rs.studies[st.name] = st
    return st


def stage04_targets(cfg, root: Path, rs) -> list[dict]:
    des = pd.read_csv(root / cfg["gse224127"]["design"], sep="\t")
    des = des[des.author_qc_flag == 0]
    out = []
    for spec in cfg["targets"]["stage04"]["timecourse"]:
        st = _gse_study(rs, cfg, root, spec["file_key"])
        d = des[des.file_key == spec["file_key"]].set_index("column")
        d = d.loc[[c for c in st.L.columns if c in d.index]]
        ref_cols = list(d.index[d.group == spec["reference"]])
        for (grp, tday), g in d[d.group != spec["reference"]].groupby(["group", "time_d"]):
            # reference arm at the same time point when the file has one, else the file's reference
            same_t = [c for c in ref_cols if d.loc[c, "time_d"] == tday]
            rc = same_t if same_t else ref_cols
            cols = list(g.index)
            base = f"{spec['model']}_{grp}_d{tday:g}_{spec['file_key']}"
            y = st.L[cols].mean(1) - st.L[rc].mean(1)
            per_mouse = {c: st.L[c] - st.L[rc].mean(1) for c in cols}
            out.append({"target": base, "source": f"GSE224127:{spec['file_key']}",
                        "family": "tissue_like", "cell": "whole liver", "time_h": 24 * tday,
                        "model": spec["model"], "group": grp, "time_d": tday,
                        "kind": "stimulus", "y": y, "per_mouse": per_mouse,
                        "n_test": len(cols), "n_ref": len(rc), "inputs": []})
    for spec in cfg["targets"]["stage04"]["blockade"]:
        st = _gse_study(rs, cfg, root, spec["file_key"])
        d = des[des.file_key == spec["file_key"]].set_index("column")
        d = d.loc[[c for c in st.L.columns if c in d.index]]
        tc = list(d.index[d.group == spec["test"]])
        rc = list(d.index[d.group == spec["reference"]])
        if not tc or not rc:
            log.warning("blockade %s: arms missing", spec)
            continue
        y = st.L[tc].mean(1) - st.L[rc].mean(1)
        out.append({"target": f"block_{spec['test']}_vs_{spec['reference']}_{spec['file_key']}",
                    "source": f"GSE224127:{spec['file_key']}", "family": "tissue_like",
                    "cell": "whole liver", "time_h": 12, "kind": "difference", "y": y,
                    "n_test": len(tc), "n_ref": len(rc), "inputs": []})
    return out


def stage03_targets(cfg, root: Path) -> list[dict]:
    """Pseudobulk CLP-vs-sham log2FC per population from the stage 03 GSE279167 objects."""
    import anndata as ad
    out = []
    for p in [root / c for c in cfg["targets"]["stage03"]["candidates"]]:
        if p.exists():
            log.info("stage 03 DE table found: %s (not used: pseudobulk computed from counts)", p)
    for fname, col in [("data/processed/03_GSE279167.h5ad", "cell_type"),
                       ("data/processed/03_GSE279167_myeloid.h5ad", "population")]:
        p = root / fname
        if not p.exists():
            log.warning("stage 03 object missing: %s", p)
            continue
        a = ad.read_h5ad(p)
        C = a.layers["counts"]
        for pop in a.obs[col].astype(str).unique():
            cnt = {}
            for cond in ("sham", "CLP"):
                m = ((a.obs[col].astype(str) == pop) & (a.obs["condition"] == cond)).values
                cnt[cond] = (int(m.sum()), np.asarray(C[m].sum(axis=0)).ravel())
            n_min = min(cnt["sham"][0], cnt["CLP"][0])
            if n_min < cfg["targets"]["stage03"].get("min_cells", 30):
                continue
            cpm = {k: v[1] / v[1].sum() * 1e6 for k, v in cnt.items()}
            keep = np.maximum(cpm["sham"], cpm["CLP"]) >= 10
            lfc = np.log2(cpm["CLP"] + 1) - np.log2(cpm["sham"] + 1)
            y = pd.Series(lfc, index=a.var_names)[keep]
            fam = "macrophage_like" if col == "population" or pop in ("Kupffer", "MonoMac") \
                else "tissue_like"
            out.append({"target": f"sc_{'myeloid' if col == 'population' else 'all'}_{pop}",
                        "source": f"GSE279167 pseudobulk ({fname.split('/')[-1]})",
                        "family": fam, "cell": pop, "time_h": np.nan, "kind": "stimulus",
                        "y": y, "n_cells_sham": cnt["sham"][0], "n_cells_clp": cnt["CLP"][0],
                        "inputs": [p]})
    return out
