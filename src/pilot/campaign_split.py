"""Deterministic WBM substitution-round campaigns, built within a single part."""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import pandas as pd

from pilot_data import HULL, load_metadata, load_pilot_labels
from analysis.config import resolve

EXPECTED_ROUND_COUNTS = {1: 54209, 2: 45979, 3: 66528, 4: 34531, 5: 14241}


def build_campaign_split(partition: pd.DataFrame, part: str, variant: str) -> pd.DataFrame:
    """Return every id of part once, with train/test/excluded_shared_system status.

    S-a: train rounds 1/2, test 3/4/5. S-b: same train, exclude test rows
    whose chemical system occurs anywhere in the train pool of this part.
    The generator reads only material_id, chemsys and part, and never uses labels.
    """
    if part not in ("PILOT", "CONFIRM") or variant not in ("S-a", "S-b"):
        raise ValueError("expected part PILOT/CONFIRM and variant S-a/S-b")
    p = partition.loc[partition.part == part, ["material_id", "chemsys", "part"]].copy()
    if p.material_id.duplicated().any():
        raise ValueError("duplicate material_id in part")
    p["round"] = p.material_id.map(lambda mid: int(mid.split("-")[1]))
    if not set(p["round"]) <= set(EXPECTED_ROUND_COUNTS):
        raise ValueError("WBM round outside 1..5")
    train = p["round"].isin([1, 2])
    p["variant"] = variant
    p["assignment"] = "test"
    p.loc[train, "assignment"] = "train"
    if variant == "S-b":
        shared = p.chemsys.isin(set(p.loc[train, "chemsys"]))
        p.loc[~train & shared, "assignment"] = "excluded_shared_system"
    return p.sort_values("material_id", kind="mergesort").reset_index(drop=True)


def write_assignments(frame, path):
    """CSV bytes and gzip metadata are deterministic, independent of output filename."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False, lineterminator="\n")


def describe_campaign(assignments, labels=None):
    train = assignments.loc[assignments.assignment == "train"]
    test = assignments.loc[assignments.assignment == "test"]
    kept = assignments.loc[assignments.assignment.isin(["train", "test"])]
    info = {"structures": len(kept), "chemical_systems": int(kept.chemsys.nunique()),
            "shared_systems": len(set(train.chemsys) & set(test.chemsys))}
    for name in ("train", "test", "excluded_shared_system"):
        sub = assignments.loc[assignments.assignment == name]
        row = {"structures": len(sub), "chemical_systems": int(sub.chemsys.nunique()),
               "round_counts": {str(k): int((sub["round"] == k).sum()) for k in EXPECTED_ROUND_COUNTS}}
        if labels is not None and name in ("train", "test"):
            values = labels.lookup(sub.material_id.to_numpy())
            row["stable_structures"] = int((values[HULL] <= 0).sum())
            row["stable_prevalence"] = float((values[HULL] <= 0).mean()) if len(sub) else None
        info[name] = row
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["PILOT", "CONFIRM", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg, universe, partition = load_metadata()
    part_of = dict(zip(partition.material_id, partition.part))
    counts = {"definition": "within part: train rounds 1,2; test 3,4,5; S-b removes train-system test rows",
              "universe_structures": len(universe), "parts": {}}
    combined = partition.material_id.map(lambda mid: int(mid.split("-")[1])).value_counts().to_dict()
    if combined != EXPECTED_ROUND_COUNTS:
        raise AssertionError(f"unique-prototype round counts {combined} differ from expected")
    counts["combined_round_counts"] = {str(k): combined[k] for k in EXPECTED_ROUND_COUNTS}
    for part in ("PILOT", "CONFIRM") if args.part == "both" else (args.part,):
        rows = partition.loc[partition.part == part]
        rounds = rows.material_id.map(lambda mid: int(mid.split("-")[1]))
        info = {"structures": len(rows), "chemical_systems": int(rows.chemsys.nunique()),
                "round_counts": {str(k): int((rounds == k).sum()) for k in EXPECTED_ROUND_COUNTS}}
        labels = None
        if part == "PILOT":
            labels, label_manifest = load_pilot_labels(
                resolve(cfg.y["population"]["summary_file"]["path"]),
                rows.material_id.to_numpy(), part_of)
            info["label_access"] = label_manifest
        for variant in ("S-a", "S-b"):
            assignments = build_campaign_split(partition, part, variant)
            filename = f"campaign_{part.lower()}_{variant.lower()}.csv.gz"
            write_assignments(assignments, args.out_dir / filename)
            info[variant] = dict(describe_campaign(assignments, labels), assignments_file=filename)
        counts["parts"][part] = info
    path = args.out_dir / "campaign_split_counts.json"
    path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts_file": str(path), "round_counts": counts["combined_round_counts"],
                      "parts": {p: {v: counts["parts"][p][v]["test"]["structures"] for v in ("S-a", "S-b")}
                                for p in counts["parts"]}}, indent=2))


if __name__ == "__main__":
    main()
