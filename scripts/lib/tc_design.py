"""Comparisons for stage 04: which liver groups are contrasted with which reference.

Every comparison stays inside one count file (one sequencing batch), so no counts or
scores are pooled across batches. Descriptions of the perturbations follow the GEO
series design and the authors' sample metadata for GSE224127.
"""
from __future__ import annotations

import re

import pandas as pd

# file_key -> reference group within that file (used to centre scores)
REFERENCE = {
    "LPS_timecourse": "control", "CLP_d0.5": "sham", "CLP_d0.25_d1": "sham",
    "Cytokine_combo": "control", "Cytokine_combo2": "control", "LPS_vs_control": "control",
    "LPS_KO": "LPS", "LPS_KO2": "LPS", "CLP_CytoBlock": "isotype",
    "CLP_CytoBlock2": "isotype", "CLP_antiTNF": "isotype",
}

PERTURBATION_DESCRIPTION = {
    "LPS": "LPS injection, wild type",
    "TNFKO": "CLP (severe) in TNF-deficient mice",
    "antiTNF": "CLP (severe) with TNF-neutralising antibody, wild type",
    "IFNgKOantiTNF": "CLP (severe) in IFN-gamma-deficient mice plus TNF-neutralising antibody",
    "IL18KOantiTNF": "CLP (severe) in IL-18-deficient mice plus TNF-neutralising antibody",
    "IL1bKOantiTNF": "CLP (severe) in IL-1beta-deficient mice plus TNF-neutralising antibody",
    "LPSantiTNF": "LPS with TNF-neutralising antibody, wild type",
    "LPSTNFKO": "LPS in TNF-deficient mice",
    "LPSIRF1KO": "LPS in IRF1-deficient mice",
    "LPSSTAT1KO": "LPS in STAT1-deficient mice",
    "LPSIFNgKOantiTNF": "LPS in IFN-gamma-deficient mice plus TNF-neutralising antibody",
    "LPSIL1bKOantiTNF": "LPS in IL-1beta-deficient mice plus TNF-neutralising antibody",
    "LPSIL18KOantiTNF": "LPS in IL-18-deficient mice plus TNF-neutralising antibody",
    "severe": "CLP, severe grade, wild type",
}


def describe(group: str) -> str:
    if group in PERTURBATION_DESCRIPTION:
        return PERTURBATION_DESCRIPTION[group]
    # recombinant cytokine singles and pairs, e.g. TNFIL18 -> TNF + IL18
    toks = re.findall(r"TNF|IFNg|IL1b|IL6|IL10|IL18", group)
    if toks and "".join(toks) == group:
        return "recombinant " + " + ".join(toks) + " injection, wild type"
    return group


def _cols(design: pd.DataFrame, **kw) -> list[str]:
    m = pd.Series(True, index=design.index)
    for k, v in kw.items():
        m &= design[k] == v
    return design.loc[m, "column"].tolist()


def timecourse_comparisons(design: pd.DataFrame) -> list[dict]:
    comps = []
    d = design[design.file_key == "LPS_timecourse"]
    ref = _cols(d, group="control")
    for t in sorted(d.loc[d.group == "LPS", "time_d"].unique()):
        comps.append({"model": "LPS", "arm": "LPS", "time_d": t, "file_key": "LPS_timecourse",
                      "reference": "untreated (day 0)",
                      "test_cols": _cols(d, group="LPS", time_d=t), "ref_cols": ref})
    for fk in ("CLP_d0.25_d1", "CLP_d0.5"):
        d = design[design.file_key == fk]
        for t in sorted(d.time_d.unique()):
            ref = _cols(d, group="sham", time_d=t)
            for arm in ("mild", "moderate", "severe"):
                tc = _cols(d, group=arm, time_d=t)
                if tc:
                    comps.append({"model": "CLP", "arm": arm, "time_d": t, "file_key": fk,
                                  "reference": "time-matched sham",
                                  "test_cols": tc, "ref_cols": ref})
    return comps


def perturbation_comparisons(design: pd.DataFrame) -> list[dict]:
    comps = []
    for fk, ref_group in REFERENCE.items():
        if fk in ("LPS_timecourse", "CLP_d0.25_d1"):
            continue
        d = design[design.file_key == fk]
        ref = _cols(d, group=ref_group)
        for g in sorted(d.group.unique()):
            if g == ref_group or (fk == "CLP_d0.5" and g != "severe"):
                continue
            kind = ("stimulus_vs_untreated" if ref_group in ("control", "sham")
                    else "blockade_vs_stimulus")
            comps.append({"file_key": fk, "perturbation": g, "comparison": kind,
                          "description": describe(g), "reference": ref_group,
                          "test_cols": _cols(d, group=g), "ref_cols": ref})
    return comps
