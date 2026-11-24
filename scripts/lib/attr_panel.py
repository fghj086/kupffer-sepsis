"""Stage 06: build every reference (vector, responsive genes, replicate-level vectors).

Outputs are cached under data/processed/06_attribution/ so that rerunning the attribution does
not repeat the per-study DESeq2 fits; the cache is keyed by a hash of the config sections that define the references.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib import attr_io
from lib.attr_ref import Study, responsive_genes

log = logging.getLogger("06_attribution")


class RefSet:
    def __init__(self):
        self.vec: dict[str, pd.Series] = {}        # full-data reference vectors (log2 scale)
        self.genes: dict[str, pd.Index] = {}       # responsive genes
        self.de: dict[str, pd.DataFrame] = {}
        self.held: dict[str, list] = {}            # (column, held-out vector, rebuilt ref, dropped)
        self.meta: list[dict] = []
        self.studies: dict[str, Study] = {}

    def matrix(self, names, genes=None) -> pd.DataFrame:
        M = pd.concat({n: self.vec[n] for n in names}, axis=1)
        return M if genes is None else M.reindex(genes)


def sanity_gse232827(st: Study):
    """Column prefixes are mapped to treatments by GEO order; confirm with canonical genes."""
    ref = lambda t: st.reference(t, "untreated")
    ifnb, ifng, tnf = ref("IFNb"), ref("IFNg"), ref("TNF")
    checks = {"Ifit1 up after IFN-b": ifnb.get("Ifit1", 0) > 2,
              "Ciita up after IFN-g more than IFN-b": ifng.get("Ciita", 0) > ifnb.get("Ciita", 0) + 1,
              "Tnfaip3 up after TNF": tnf.get("Tnfaip3", 0) > 1}
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"GSE232827 arm mapping failed checks: {bad}")
    return checks


def build_refset(cfg: dict, cfg01: dict, root: Path) -> RefSet:
    gm = attr_io.gene_map(root / cfg["gene_map"])
    rs = RefSet()
    for name, spec in cfg["studies"].items():
        X, arms = attr_io.load_study(name, spec, root, gm, cfg, cfg01)
        st = Study(name, X, arms, spec["kind"], cfg)
        rs.studies[name] = st
        log.info("%s: %d genes x %d samples %s", name, X.shape[0], X.shape[1],
                 arms.value_counts().to_dict())
        if name == "GSE232827":
            log.info("GSE232827 arm checks passed: %s", sanity_gse232827(st))
    for ref, r in cfg["references"].items():
        st = rs.studies[r["study"]]
        sp = cfg["studies"][r["study"]]
        rs.vec[ref] = st.reference(r["test"], r["control"])
        de = st.de(r["test"], r["control"])
        rs.de[ref] = de
        rs.genes[ref] = responsive_genes(de, st.expressed(r["test"], r["control"]), cfg)
        rs.held[ref] = st.heldout(r["test"], r["control"])
        n_up = int(((de.padj < cfg["responsive"]["padj"]) & (de.lfc >= cfg["responsive"]["min_abs_lfc"])).sum())
        n_dn = int(((de.padj < cfg["responsive"]["padj"]) & (de.lfc <= -cfg["responsive"]["min_abs_lfc"])).sum())
        rs.meta.append({
            "reference": ref, "study": r["study"], "cell": sp["cell"],
            "stimulus_arm": r["test"], "control_arm": r["control"],
            "n_stimulus": len(st.cols(r["test"])), "n_control": len(st.cols(r["control"])),
            "time_h": r["time_h"], "stimulus_class": r["cls"], "data_type": sp["kind"],
            "de_method": "pydeseq2 Wald" if sp["kind"] == "counts" else "moderated t (log2)",
            "n_genes_measured": int(rs.vec[ref].notna().sum()),
            "n_responsive_up": n_up, "n_responsive_down": n_dn,
            "n_genes_used_for_panel": len(rs.genes[ref]),
            "response_rms_log2fc": float(np.sqrt(np.mean(rs.vec[ref].reindex(rs.genes[ref]) ** 2)))
            if len(rs.genes[ref]) else np.nan,
        })
        log.info("reference %s: %d up / %d down", ref, n_up, n_dn)
    return rs


def load_or_build(cfg, cfg01, root: Path, cache_dir: Path) -> RefSet:
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib, json
    sub = {k: cfg[k] for k in ("studies", "references", "expression", "responsive", "gene_map",
                               "gse224127")}
    key = hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]
    p = cache_dir / f"refset_{key}.pkl"
    if p.exists():
        log.info("reference cache %s", p.name)
        with open(p, "rb") as fh:
            return pickle.load(fh)
    rs = build_refset(cfg, cfg01, root)
    for old in cache_dir.glob("refset_*.pkl"):
        old.unlink()
    with open(p, "wb") as fh:
        pickle.dump(rs, fh)
    return rs


def informative(rs: RefSet, panel: list, target: pd.Series | None = None) -> pd.Index:
    g = pd.Index(sorted(set().union(*[set(rs.genes[n]) for n in panel])))
    M = rs.matrix(panel, g)
    ok = M.notna().all(axis=1)
    if target is not None:
        ok &= target.reindex(g).notna()
    return g[ok.values]
