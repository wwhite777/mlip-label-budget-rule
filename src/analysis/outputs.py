"""Output writers. Every result file is byte-deterministic given the inputs and seeds: npz members are written with a
fixed zip timestamp (numpy.savez stamps the current time), JSON with sorted keys, CSV via pandas.
Timing / environment facts go only into run_manifest.json and the run_record, never into result files."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import REPO, sha256_file

_ZDATE = (1980, 1, 1, 0, 0, 0)


def save_npz(path, compress: bool = False, **arrays) -> str:
    """numpy-loadable .npz (np.load(allow_pickle=False)) with fixed member timestamps; returns sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comp = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=comp) as zf:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(v), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=_ZDATE)
            zi.compress_type = comp
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())
    return sha256_file(path)


def write_json(path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=_default) + "\n",
                    encoding="utf-8")
    return sha256_file(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_csv(path, rows: list, columns=None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def write_unit_csv(units_dir, rid: str, recs: list) -> dict:
    rows = [{"run_id": rid, "menu": r["menu"], "backbone": r["backbone"], "type": r["type"],
             "grid_index": r["grid_index"], "params": json.dumps(r["params"], sort_keys=True),
             "inner_val_ap": r["inner_val_ap"], "best_epoch": r.get("best_epoch", ""),
             "chosen_for_type": r["chosen_for_type"], "selected_best": r["selected_best"]} for r in recs]
    p = Path(units_dir) / f"{rid}.csv"
    sha = write_csv(p, rows)
    return {"path": str(p), "sha256": sha, "rows": len(rows), "metric_columns": ["inner_val_ap"]}


def write_scores(path, part, reps: list, oof: dict, ref_rows: dict) -> str:
    methods = list(oof) + list(ref_rows)
    S = np.empty((len(methods), len(reps), part.n), dtype=np.float64)
    for i, m in enumerate(oof):
        S[i] = oof[m][reps]
    for j, m in enumerate(ref_rows):
        S[len(oof) + j] = np.broadcast_to(ref_rows[m], (len(reps), part.n))
    return save_npz(path, compress=True, material_id=part.ids, fold=part.fold, replicate=np.asarray(reps),
                    method=np.asarray(methods), score=S)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def write_bundle(out_dir, cfg, stage: str, look: str, evals: list, decision: dict) -> dict:
    """src/gates/README_bundle.md schema p3_results_bundle_v1 (family contrasts only; primary split only)."""
    if not cfg.readings_allowed:
        raise ValueError(f"split role {cfg.role}: no bundle (descriptive split)")
    out_dir = Path(out_dir)
    contrasts, draws, families = {}, {}, {}
    for ev in evals:
        for r in ev["readings"]:
            cid = r["bundle_id"]
            claim, tag, bb = cid.split(":")
            tests = {"up": {"family": r["family_up"], "margin": r["margin"], "p_raw": r["p_up"],
                            "p_holm": r["p_up_holm"]}}
            families.setdefault(r["family_up"], []).append(cid)
            if r["family_down"]:
                tests["down"] = {"family": r["family_down"], "margin": r["margin"], "p_raw": r["p_down"],
                                 "p_holm": r["p_down_holm"]}
                families.setdefault(r["family_down"], []).append(cid)
            contrasts[cid] = {"claim": claim, "contrast": tag, "backbone": bb, "estimate": r["estimate"],
                              "interval": {"low": r["interval"][0], "high": r["interval"][1]}, "tests": tests,
                              "reading": r["reading"]}
            draws[cid] = np.asarray(ev["dstar"][r["contrast"]], dtype=np.float64)
    dsha = save_npz(out_dir / "draws.npz", **draws)
    reps = evals[0]["replicates"]
    bundle = {"schema": "p3_results_bundle_v1", "stage": stage, "look": look,
              "prereg": {"path": _rel(cfg.prereg_path), "sha256": sha256_file(cfg.prereg_path)},
              "n": int(evals[0]["n"]), "replicates": list(reps), "bootstrap": {"B": cfg.B, "seed": cfg.boot_seed},
              "alpha": cfg.alpha, "draws": {"path": "draws.npz", "sha256": dsha}, "contrasts": contrasts,
              "families": {k: sorted(v) for k, v in families.items()}, "decision": decision,
              "mode": cfg.mode, "labels": "SYNTHETIC" if cfg.synthetic else "REAL"}
    bsha = write_json(out_dir / "bundle.json", bundle)
    return {"bundle": str(out_dir / "bundle.json"), "bundle_sha256": bsha, "draws_sha256": dsha}
