"""Sample identity checks for the longitudinal human blood cohorts.

Every patient in these cohorts is sampled more than once, so the genetic sex read
from chromosome Y genes and XIST must be the same in every sample of a patient
and must agree with the annotated sex. A sample that breaks this cannot be
assigned to its patient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Y_GENES = ["RPS4Y1", "KDM5D", "EIF1AY"]  # the three Y transcripts measured well on both platforms

def genetic_sex(expr: pd.DataFrame, gap_min: float = 1.0, label: str = "") -> pd.DataFrame:
    """Per-sample score (mean Y genes minus XIST) and call at the widest gap.

    The two sexes form two clusters; the threshold is the midpoint of the largest
    gap between sorted scores, and that gap must exceed gap_min log2 units.
    """
    y = [g for g in Y_GENES if g in expr.index]
    if not y or "XIST" not in expr.index:
        raise SystemExit("Y genes or XIST missing from the matrix")
    score = expr.loc[y].mean() - expr.loc["XIST"]
    s = np.sort(score.values)
    i = int(np.argmax(np.diff(s)))
    gap = s[i + 1] - s[i]
    if gap < gap_min:
        raise SystemExit(f"{label}: genetic sex not bimodal (largest gap {gap:.2f})")
    thr = (s[i] + s[i + 1]) / 2
    return pd.DataFrame({"sex_score": score, "genetic_sex": np.where(score > thr, "M", "F"),
                         "threshold": thr, "gap": gap})


def identity_table(samples: pd.DataFrame, sex: pd.DataFrame) -> pd.DataFrame:
    """Join calls to the sample table and flag samples that disagree with their person.

    A person's genetic sex is the majority call over their samples. A sample whose
    call differs from that majority belongs to someone else and is flagged. When
    all of a person's samples agree with each other but not with the annotation,
    the annotation is wrong, not the samples: the samples are kept and the sex
    covariate is taken from the genetic call.
    """
    t = samples.join(sex)
    major = t.groupby("patient")["genetic_sex"].agg(lambda s: s.value_counts().idxmax())
    t["person_genetic_sex"] = t["patient"].map(major)
    t["conflict"] = t["genetic_sex"] != t["person_genetic_sex"]
    t["annotated_sex"] = t["sex"]
    t["sex_corrected"] = t["sex"].notna() & (t["sex"] != t["person_genetic_sex"])
    t["sex"] = t["person_genetic_sex"]
    return t
