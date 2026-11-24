"""Stage 03a: QC, clustering and annotation of the two liver scRNA-seq datasets.

Each dataset is processed separately. GSE279167 has one pooled library per condition
(sham: 3 mice; CLP: 4 mice), so it is not integrated (library = condition). GSE290684
libraries are integrated with Harmony within that dataset only.
Outputs: data/processed/03_<GSE>.h5ad (+ _myeloid.h5ad) and annotation tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib import repro  # noqa: E402
from scripts.lib import sc_io, sc_preprocess as pp  # noqa: E402

STAGE = "03_single_cell"


def process(gse: str, cfg: dict, root: Path, out: Path):
    dcfg = cfg["datasets"][gse]
    seed = cfg["seed"]
    if gse == "GSE279167":
        ad = sc_io.load_gse279167(dcfg, root)
        des = (ad.obs[["library", "sample", "condition"]].drop_duplicates().reset_index(drop=True))
        des["mice_pooled"] = des["condition"].map({"sham": 3, "CLP": 4})
        des["timepoint"] = "not stated in GEO record"
        des["replicate_unit"] = "none (one pooled library per condition)"
    else:
        ad, des = sc_io.load_gse290684(dcfg, root)
        des["timepoint"] = "24 h after CLP"
        des["replicate_unit"] = "library"
    des.insert(0, "dataset", gse)
    sp_chk = sc_io.species_check(ad.var_names)
    print(gse, ad.shape, sp_chk)

    ad, qc = pp.qc_filter(ad, cfg["qc"], seed)
    qc.insert(0, "dataset", gse)
    batch = "library" if dcfg["integrate"] else None
    ad = pp.normalise_embed(ad, cfg["processing"], seed, batch)

    ann = pp.annotate_major(ad, cfg["cell_type_markers"], seed,
                           min_hb_frac=cfg["qc"]["erythroid_min_hb_fraction"])
    ann.insert(0, "dataset", gse)
    cfg["_integrate"] = dcfg["integrate"]
    my, myann = pp.annotate_myeloid(ad, cfg, seed)
    myann.insert(0, "dataset", gse)

    ad.obs["population"] = ad.obs["cell_type"].astype(str)
    is_my = ad.obs["cell_type"].isin(cfg["myeloid_parents"])
    ad.obs.loc[is_my, "population"] = "Myeloid (neutrophil-like, excluded)"
    ad.obs.loc[my.obs_names, "population"] = my.obs["myeloid_subset"].astype(str)
    ad.obs["population"] = ad.obs["population"].astype("category")
    my.obs["population"] = my.obs["myeloid_subset"].astype("category")

    mk = pp.top_markers(ad, "population", n=15)
    mk.insert(0, "level", "population")
    mk2 = pp.top_markers(my, "population", n=25)
    mk2.insert(0, "level", "myeloid_within")
    markers = pd.concat([mk, mk2])
    markers.insert(0, "dataset", gse)

    counts = (ad.obs.groupby(["condition", "sample", "population"], observed=True).size()
              .rename("n_cells").reset_index())
    counts["frac_of_sample"] = counts["n_cells"] / counts.groupby("sample")["n_cells"].transform("sum")
    counts.insert(0, "dataset", gse)

    pdir = root / cfg["processed_dir"]
    pdir.mkdir(parents=True, exist_ok=True)
    f1, f2 = pdir / f"03_{gse}.h5ad", pdir / f"03_{gse}_myeloid.h5ad"
    for k in list(ad.obs.columns):
        if k.startswith("mk_"):
            del ad.obs[k]
    ad.write_h5ad(f1, compression="gzip")
    my.write_h5ad(f2, compression="gzip")
    des["species_check"] = str(sp_chk)
    return des, qc, ann, myann, markers, counts, [f1, f2]


def main():
    cfg = repro.load_config("config/03_single_cell.yaml")
    repro.set_global_seed(cfg["seed"])
    sc.settings.n_jobs = cfg.get("n_jobs", 16)
    root = repro.project_root()
    out = repro.results_dir(STAGE)
    only = sys.argv[1:] or list(cfg["datasets"])
    parts = {k: [] for k in ["design", "qc_summary", "cluster_annotation",
                             "myeloid_subcluster_annotation", "markers", "cell_counts"]}
    h5 = []
    for gse in only:
        res = process(gse, cfg, root, out)
        for k, v in zip(parts, res[:6]):
            v.to_csv(out / f"_part_{k}_{gse}.tsv", sep="\t", index=False)
        h5 += res[6]
    outputs = []
    for k in parts:
        fs = sorted(out.glob(f"_part_{k}_*.tsv"))
        df = pd.concat([pd.read_csv(f, sep="\t") for f in fs])
        outputs.append(repro.write_table(df, out / f"{k}.tsv"))
    # h5ad checksums; stage provenance (inputs, outputs, seed) is written by the analysis step
    for p in h5:
        (out / f"_part_h5_{p.stem}.tsv").write_text(f"{p.name}\t{repro.md5_or_sha(p)}\n")
    rows = "".join(f.read_text() for f in sorted(out.glob("_part_h5_*.tsv")))
    (out / "processed_objects.tsv").write_text("file\tmd5\n" + rows)
    print("done", outputs)


if __name__ == "__main__":
    main()
