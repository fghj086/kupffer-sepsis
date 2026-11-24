"""Loaders for the human sepsis blood cohorts: gene x sample matrix (log2) and a harmonised sample table.

Sample table columns: patient, day, group (survivor / nonsurvivor / control / sepsis),
died (0/1, NaN when unknown), age, sex (M/F), plus cohort-specific extras.
"""
from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from lib import blood_geo as geo


def _platform_from_full(path: Path, id_col: str, sym_col: str) -> pd.Series:
    with open(path, errors="replace") as fh:
        hdr = next(i for i, line in enumerate(fh) if line.startswith("ID\t"))
    ann = pd.read_csv(path, sep="\t", skiprows=hdr, dtype=str, low_memory=False,
                      usecols=[id_col, sym_col])
    s = ann.dropna().drop_duplicates(id_col).set_index(id_col)[sym_col].str.strip()
    return s[(s != "") & ~s.str.contains("///", regex=False)]


# ------------------------------------------------------------------ GAinS
def _gains_ids(names: pd.Series) -> pd.DataFrame:
    m = names.str.extract(r"^((?:CAP|FP|CON)\d+)(?:\.B)?(?:\.(\d))?$")
    return pd.DataFrame({"patient": m[0].values,
                         "day": pd.to_numeric(m[1]).fillna(1).values}, index=names.values)


def gains_5273(raw: Path, platform_full: Path):
    d = pd.read_csv(raw / "E-MTAB-5273/E-MTAB-5273.sdrf.txt", sep="\t")
    x = pd.read_csv(raw / "E-MTAB-5273/Burnham_sepsis_discovery_normalised_231.txt", sep="\t", index_col=0)
    x.index = x.index.astype(str)
    p2g = _platform_from_full(platform_full, "Array_Address_Id", "Symbol")
    p2g.index = p2g.index.astype(str)
    g = geo.collapse_probes(x, p2g, "max_mean")
    meta = d.drop_duplicates("Source Name").set_index("Source Name")
    s = _gains_ids(pd.Series(g.columns))
    s["died"] = meta["Characteristics[clinical information]"].reindex(s.index).map(
        {"alive at 28 day survival": 0, "dead at 28 day survival": 1})
    s["age"] = pd.to_numeric(meta["Characteristics[age]"].reindex(s.index), errors="coerce")
    s["sex"] = meta["Characteristics[sex]"].reindex(s.index).map({"male": "M", "female": "F"})
    s["source"] = meta["Characteristics[disease]"].reindex(s.index)
    s["group"] = np.where(s.patient.str.startswith("CON"), "control",
                          np.where(s.died == 1, "nonsurvivor", np.where(s.died == 0, "survivor", "sepsis")))
    return g, s


def gains_4421(raw: Path, platform_full: Path):
    d = pd.read_csv(raw / "E-MTAB-4421/E-MTAB-4421.sdrf.txt", sep="\t")
    x = pd.read_csv(raw / "E-MTAB-4421/Davenport_sepsis_Jan2016_normalised_265.txt", sep="\t", index_col=0)
    p2g = _platform_from_full(platform_full, "ID", "Symbol")
    g = geo.collapse_probes(x, p2g, "max_mean")
    meta = d.drop_duplicates("Comment[beadchiparray_id]").set_index("Comment[beadchiparray_id]")
    meta = meta.reindex(g.columns)
    s = pd.DataFrame({
        "patient": meta["Source Name"].values,
        "srs": meta["Characteristics[sepsis response signature group]"].map(
            {"group 1": "SRS1", "group 2": "SRS2"}).values,
        "died": meta["Characteristics[28 day survival]"].map({"non-survivor": 1, "survivor": 0}).values,
    }, index=g.columns)
    return g, s


# ------------------------------------------------------------------ MARS
def mars(raw: Path):
    expr, meta = geo.parse_series_matrix(raw / "GSE65682/GSE65682_series_matrix.txt.gz")
    sym = _platform_from_full(raw / "platforms/GPL13667.txt", "ID", "Gene Symbol")
    g = geo.collapse_probes(expr.astype(float), sym, "max_mean")
    meta = meta.reindex(g.columns)
    s = pd.DataFrame({"endotype": meta["endotype_class"].replace("NA", np.nan),
                      "died": pd.to_numeric(meta["mortality_event_28days"], errors="coerce")},
                     index=g.columns)
    return g, s


# ------------------------------------------------------------------ GEO microarray cohorts
def harmonise_gse57065(md: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=md.index)
    out["cohort"] = "GSE57065"
    is_ctrl = md["collection_time"].str.strip() == ""
    pid = md["title"].str.extract(r"_(P\d+)_")[0]
    out["patient"] = np.where(is_ctrl, "HV_" + md["title"].str.replace(
        r"\W+", "", regex=True), pid)
    hours = pd.to_numeric(md["collection_time"].str.extract(r"(\d+)")[0],
                          errors="coerce")
    out["hour"] = np.where(is_ctrl, np.nan, hours)
    out["day"] = np.where(is_ctrl, np.nan, hours / 24.0 + 1.0)  # H0 -> day 1
    out["group"] = np.where(is_ctrl, "control", "septic_shock")
    out["age"] = pd.to_numeric(md["age"], errors="coerce")
    out["sex"] = md["gender"].map({"FEMALE": "F", "MALE": "M"})
    out["severity"] = md["sapsii"].replace("NA", np.nan)
    out["severity_metric"] = "SAPS_II_class"
    out["time_unit"] = "hour"
    return out


def harmonise_gse95233(md: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=md.index)
    out["cohort"] = "GSE95233"
    is_ctrl = md["time_point"] == "NA"
    pid = md["title"].str.extract(r"Blood-([A-Z]+_\d+)_")[0]
    out["patient"] = pid
    day = pd.to_numeric(md["time_point"].str.extract(r"D0?(\d)")[0],
                        errors="coerce")
    out["day"] = np.where(is_ctrl, np.nan, day)
    surv = md["survival"].map({"Survivor": "survivor",
                               "Non Survivor": "nonsurvivor"})
    out["group"] = np.where(is_ctrl, "control", surv)
    out["age"] = pd.to_numeric(md["age"], errors="coerce")
    out["sex"] = md["gender"].map({"F": "F", "M": "M"})
    out["severity"] = np.nan
    out["severity_metric"] = "not_reported"
    out["time_unit"] = "day"
    return out


def series_cohort(raw: Path, cohort: str, gpl: str, harmonise):
    sm = raw / cohort / f"{cohort}_series_matrix.txt.gz"
    expr, md = geo.parse_series_matrix(sm)
    if np.nanmax(expr.values) > 50:
        expr = np.log2(expr.clip(lower=1) + 1)
    g = geo.collapse_probes(expr, geo.load_platform_annotation(raw / "platforms" / f"{gpl}.annot.gz"))
    return g, harmonise(md).loc[g.columns]


# ------------------------------------------------------------------ GSE222393 (RNA-seq)
def _characteristics(path: Path) -> pd.DataFrame:
    d: dict[str, list[str]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            v = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_title"):
                d["title"] = v
            elif line.startswith("!Sample_characteristics_ch1"):
                k = next((x.split(": ")[0] for x in v if ": " in x), None)
                if k and k not in d:
                    d[k] = [x.split(": ", 1)[-1] for x in v]
    return pd.DataFrame(d)


def _ensembl_to_symbol(features: Path) -> pd.Series:
    """Ensembl gene -> symbol from the 10x feature table of GSE279452 (same GENCODE annotation)."""
    f = pd.read_csv(features, sep="\t", header=None)
    sym = f.drop_duplicates(0).set_index(0)[1]
    return sym[~sym.duplicated(keep=False)]


def gse222393(raw_root: Path, min_mean_cpm: float):
    """Counts of GSE222393 plus earlier admission samples of the same patients (GSE185263)."""
    raw = raw_root / "GSE222393"
    sym = _ensembl_to_symbol(raw_root / "GSE279452/GSE279452_features.tsv.gz")
    s = _characteristics(raw / "GSE222393_series_matrix.txt.gz")
    s["patient"] = s.title.str.extract(r"(sepcv\d+)")[0]
    s["day"] = s.title.str.extract(r"T(\d+)$")[0].astype(int)
    cols = {}
    with tarfile.open(raw / "GSE222393_RAW.tar") as t:
        for m in t.getmembers():
            title = m.name.split("_", 1)[1].replace(".count.txt.gz", "")
            cols[title] = pd.read_csv(io.BytesIO(gzip.decompress(t.extractfile(m).read())),
                                      sep="\t", header=None, index_col=0)[1]
    C = pd.DataFrame(cols)
    s["deposit"] = "GSE222393"
    old = pd.read_csv(raw / "GSE185263_raw_counts.csv.gz", index_col=0)
    o = _characteristics(raw / "GSE185263_series_matrix.txt.gz")
    o = o[o.title.str.match(r"sepcv\d+T\d+$")]
    o["patient"] = o.title.str.extract(r"(sepcv\d+)")[0]
    o["day"] = o.title.str.extract(r"T(\d+)$")[0].astype(int)
    o = o[o.patient.isin(s.patient) & ~o.title.isin(s.title) & o.title.isin(old.columns)]
    first = s.groupby("patient").day.min()
    o = o[o.day < o.patient.map(first)]
    meta = s.set_index("patient")[["age", "Sex", "mortality", "covid"]].groupby(level=0).first()
    o = o[["title", "patient", "day"]].join(meta, on="patient").assign(deposit="GSE185263")
    C = C.join(old[o.title], how="inner")
    C = C[~C.index.str.startswith("__")]
    S = pd.concat([s[["title", "patient", "day", "age", "Sex", "mortality", "covid", "deposit"]], o],
                  ignore_index=True).set_index("title")
    S = S.rename(columns={"Sex": "sex"})
    S["sex"] = S["sex"].map(lambda v: {"male": "M", "female": "F", "m": "M", "f": "F"}.get(str(v).lower(), np.nan))
    S["age"] = pd.to_numeric(S["age"], errors="coerce")
    S["died"] = (S.mortality == "Deceased").astype(int)
    S["covid"] = (S.covid == "Positive").astype(int)
    S["group"] = np.where(S.died == 1, "nonsurvivor", "survivor")
    C.index = C.index.str.split(".").str[0]
    gs = C.index.map(sym)
    C = C[gs.notna()].groupby(gs[gs.notna()]).sum()[S.index]
    cpm = C / C.sum(axis=0) * 1e6
    cpm = cpm.loc[cpm.mean(axis=1) >= min_mean_cpm]
    return np.log2(cpm + 1), S
