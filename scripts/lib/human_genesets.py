"""Stage 07: gene sets carried to human genes.

* Mouse -> human orthologues: MGI HOM_MouseHumanSequence.rpt homology classes (DB Class Key);
  only classes with exactly one mouse and one human gene are used (one-to-one orthologues).
* Human symbols in older lists and arrays are updated with HGNC previous / alias symbols.
* Published human lists (Pena et al. 2014, Supplementary Tables 1-2) are parsed from the
  supplement text, keeping the sign of the reported fold change.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ orthologues and symbols
def ortholog_map(path: Path) -> tuple[pd.Series, dict]:
    h = pd.read_csv(path, sep="\t", dtype=str)
    key = "DB Class Key"
    h = h[h["NCBI Taxon ID"].isin(["10090", "9606"])]
    cnt = h.groupby([key, "NCBI Taxon ID"]).size().unstack(fill_value=0)
    one = cnt[(cnt.get("10090", 0) == 1) & (cnt.get("9606", 0) == 1)].index
    sub = h[h[key].isin(one)]
    m = sub[sub["NCBI Taxon ID"] == "10090"].set_index(key)["Symbol"]
    hs = sub[sub["NCBI Taxon ID"] == "9606"].set_index(key)["Symbol"]
    mp = pd.Series(hs.reindex(m.index).values, index=m.values).dropna()
    stats = {"homology_classes": int(cnt.shape[0]), "one_to_one_classes": int(len(one)),
             "mouse_genes_mapped": int(mp.size)}
    return mp, stats


def hgnc_updater(path: Path) -> dict:
    """old/alias symbol -> approved symbol (only unambiguous old symbols)."""
    t = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "prev_symbol", "alias_symbol"])
    approved = set(t.symbol)
    out: dict[str, set] = {}
    for col in ("prev_symbol", "alias_symbol"):
        for sym, olds in zip(t.symbol, t[col].fillna("")):
            for o in olds.split("|"):
                o = o.strip()
                if o and o not in approved:
                    out.setdefault(o, set()).add(sym)
    return {k: next(iter(v)) for k, v in out.items() if len(v) == 1}


def hgnc_ensembl(path: Path) -> pd.Series:
    t = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"]).dropna()
    return t.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["symbol"]


def update_symbols(genes, upd: dict, universe=None) -> list:
    """Approved symbol for each gene; keeps a symbol already present in the data universe."""
    out = []
    for g in genes:
        if universe is not None and g in universe:
            out.append(g)
        else:
            out.append(upd.get(g, g))
    return out


# ------------------------------------------------------------------ published human lists
_ROW = re.compile(r"^\s*([A-Za-z0-9\-\.]+)\s+\S.*?\s(-?\d+\.\d)\s*$")


def parse_pena(txt: Path) -> pd.DataFrame:
    """Supplementary Tables 1 (endotoxin tolerance) and 2 (inflammatory) of Pena et al. 2014."""
    lines = Path(txt).read_text().splitlines()
    rows, cur = [], None
    for ln in lines:
        if ln.startswith("Supplementary Table 1"):
            cur = "pena_endotoxin_tolerance"
            continue
        if ln.startswith("Supplementary Table 2"):
            cur = "pena_inflammatory"
            continue
        if ln.lstrip().startswith("Supplementary Table 3"):
            break
        if cur is None:
            continue
        m = _ROW.match(ln)
        if m and m.group(1) not in ("Gene", "Symbol"):
            fc = float(m.group(2))
            rows.append({"signature": cur, "gene_published": m.group(1), "fold_change": fc,
                         "weight": 1.0 if fc > 0 else -1.0})
    df = pd.DataFrame(rows)
    n = df.groupby("signature").size().to_dict()
    if n.get("pena_endotoxin_tolerance") != 99:
        raise SystemExit(f"Pena 2014 Table 1 parse: expected 99 genes, got {n}")
    return df


# ------------------------------------------------------------------ mouse tolerance programmes
def mouse_tolerance(cfg: dict, root: Path) -> tuple[pd.DataFrame, str]:
    sys.path.insert(0, str(root / "scripts"))
    import pickle
    paths = sorted(glob.glob(str(root / cfg["mouse_tolerance"]["cache_glob"])))
    if len(paths) != 1:
        raise SystemExit(f"expected one stage 06 reference cache, found {paths}")
    with open(paths[0], "rb") as fh:
        rs = pickle.load(fh)
    mt = cfg["mouse_tolerance"]
    rows = []
    for name, c in mt["contrasts"].items():
        st = rs.studies[c["study"]]
        de = st.de(c["test"], c["control"])
        expressed = st.expressed(c["test"], c["control"])
        de = de[expressed.reindex(de.index).fillna(False).values].dropna(subset=["padj", "lfc"])
        ok = de[(de.padj < mt["padj"]) & (de.lfc.abs() >= mt["min_abs_lfc"])].copy()
        ok["rank_stat"] = ok.lfc.abs() * -np.log10(ok.padj.clip(lower=1e-300))
        for sign, part in ((1.0, ok[ok.lfc > 0]), (-1.0, ok[ok.lfc < 0])):
            part = part.sort_values("rank_stat", ascending=False).head(mt["max_genes_per_direction"])
            for g, r in part.iterrows():
                rows.append({"signature": name, "gene": g, "weight": sign, "lfc": r.lfc, "padj": r.padj})
    return pd.DataFrame(rows), paths[0]


# ------------------------------------------------------------------ assembly
def build_sets(cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    o2o, ostats = ortholog_map(root / cfg["orthologs"]["file"])
    upd = hgnc_updater(root / cfg["hgnc"])
    rows = []
    sig = pd.read_csv(root / cfg["mouse_signatures"], sep="\t")
    for _, r in sig.iterrows():
        rows.append({"signature": r.signature, "source": "mouse stage 01 (GSE200210)",
                     "gene_source": r.gene, "weight": float(r.weight)})
    tol, cache = mouse_tolerance(cfg, root)
    for _, r in tol.iterrows():
        rows.append({"signature": r.signature, "source": f"mouse stage 06 reference cache",
                     "gene_source": r.gene, "weight": r.weight})
    for name, genes in cfg["mouse_sets_to_map"].items():
        for g in genes:
            rows.append({"signature": name, "source": "mouse curated (stage 02)", "gene_source": g,
                         "weight": 1.0})
    mouse = pd.DataFrame(rows)
    mouse["gene"] = mouse.gene_source.map(o2o)
    pena = parse_pena(root / cfg["published"]["pena_pdf_text"])
    pena = pena.rename(columns={"gene_published": "gene_source"})
    pena["source"] = "Pena et al. 2014 Suppl. Tables 1-2"
    pena["gene"] = update_symbols(pena.gene_source, upd)
    human = []
    for name, genes in cfg["human_sets"].items():
        for g in genes:
            human.append({"signature": name, "source": "human markers (config)", "gene_source": g,
                          "gene": g, "weight": 1.0})
    sets = pd.concat([mouse, pena[["signature", "source", "gene_source", "gene", "weight"]],
                      pd.DataFrame(human)], ignore_index=True)
    # clearance without the identity markers used for the KC and recruited-cell annotations
    ident = set(cfg["human_sets"]["kc_identity"]) | set(cfg["human_sets"]["monocyte_recruited"]) | {"CLEC4F"}
    cl = sets[(sets.signature == "clearance") & sets.gene.notna() & ~sets.gene.isin(ident)].copy()
    cl["signature"] = "clearance_no_identity"
    sets = pd.concat([sets, cl], ignore_index=True)
    # a gene in both halves of a signed set (possible after mapping) is dropped from that set
    dup = sets.dropna(subset=["gene"]).groupby(["signature", "gene"]).weight.nunique()
    bad = dup[dup > 1].index
    sets = sets[~sets.set_index(["signature", "gene"]).index.isin(bad)]
    sets = pd.concat([sets[sets.gene.notna()].drop_duplicates(["signature", "gene"], keep="first"),
                      sets[sets.gene.isna()]], ignore_index=True)
    info = {**ostats, "stage06_cache": cache,
            "mapped_fraction": sets.groupby("signature").gene.apply(lambda s: float(s.notna().mean())).to_dict()}
    return sets.reset_index(drop=True), info
