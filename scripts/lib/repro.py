"""Reproducibility helpers shared by every analysis stage.

Contract:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    ...
    write_provenance("01_perturbation", inputs=[...], outputs=[...], cfg=cfg)

A stage without results/<stage>/PROVENANCE.json is considered not run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def load_config(path: str | Path = "config/run.yaml") -> dict:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(p)
    cfg["_config_sha256"] = md5_or_sha(p, "sha256")
    return cfg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def md5_or_sha(path: str | Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: str | Path) -> str:
    """Path relative to the project (data files may be symbolic links to another disk)."""
    p = Path(p)
    for q in (p.absolute(), p.resolve()):
        try:
            return str(q.relative_to(ROOT))
        except ValueError:
            continue
    return p.name


def results_dir(stage: str) -> Path:
    d = ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index, float_format="%.6g")
    return path


def read_table(path: str | Path, index_col=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=index_col)


def _git_commit() -> str:
    for d in (ROOT,):
        try:
            out = subprocess.run(
                ["git", "-C", str(d), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    return "not-a-git-repo"


def _env_snapshot() -> dict:
    try:
        import importlib.metadata as md
        pkgs = {d.metadata["Name"]: d.version for d in md.distributions()
                if d.metadata.get("Name")}
    except Exception:
        pkgs = {}
    keep = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
            "xgboost", "scanpy", "anndata", "matplotlib", "seaborn", "PyYAML"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": Path(os.environ.get("CONDA_PREFIX", sys.prefix)).name,
        "packages": {k: pkgs.get(k) for k in keep if pkgs.get(k)},
    }


def write_provenance(stage: str, inputs, outputs, cfg: dict,
                     notes: str = "", extra: dict | None = None) -> Path:
    d = results_dir(stage)
    prov = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.get("seed"),
        "config_path": _rel(cfg.get("_config_path", "config/run.yaml")),
        "config_sha256": cfg.get("_config_sha256"),
        "git_commit": _git_commit(),
        "script": _rel(sys.argv[0]) if sys.argv and sys.argv[0] else "",
        "inputs": [{"path": _rel(p), "md5": md5_or_sha(p)} for p in inputs
                   if Path(p).exists()],
        "outputs": [{"path": _rel(p), "md5": md5_or_sha(p)} for p in outputs
                    if Path(p).exists()],
        "environment": _env_snapshot(),
        "notes": notes,
    }
    if extra:
        prov["extra"] = extra
    out = d / "PROVENANCE.json"
    with open(out, "w") as fh:
        json.dump(prov, fh, indent=2)
    return out


def append_provenance(stage: str, name: str, inputs, outputs, cfg: dict, notes: str = "") -> Path:
    """Record an add-on analysis of an existing stage under prov["addenda"][name] without
    overwriting the stage's own provenance (inputs/outputs md5, script, config hash)."""
    out = results_dir(stage) / "PROVENANCE.json"
    prov = json.loads(out.read_text()) if out.exists() else {"stage": stage}
    prov.setdefault("addenda", {})[name] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.get("seed"),
        "config_path": _rel(cfg.get("_config_path", "config/run.yaml")),
        "config_sha256": cfg.get("_config_sha256"),
        "git_commit": _git_commit(),
        "script": _rel(sys.argv[0]) if sys.argv and sys.argv[0] else "",
        "inputs": [{"path": _rel(p), "md5": md5_or_sha(p)} for p in inputs if Path(p).exists()],
        "outputs": [{"path": _rel(p), "md5": md5_or_sha(p)} for p in outputs if Path(p).exists()],
        "notes": notes,
    }
    with open(out, "w") as fh:
        json.dump(prov, fh, indent=2)
    return out
