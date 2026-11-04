"""Write result/membership/MPTRJ_MEMBERSHIP.md from mptrj_membership_summary.json plus the checkpoint training-data
table (statements quoted from the pinned sources fetched 2026-10-02 into cache/membership/).

python3 src/membership/write_mptrj_report.py
Exit 1 if a self-check fails (stream md5 != supplied md5, or a parse check counter is non-zero where zero is required).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "cache/membership"
SUMMARY = ROOT / "result/membership/mptrj_membership_summary.json"
OUT = ROOT / "result/membership/MPTRJ_MEMBERSHIP.md"
C = "e3d7966144e9653fb28c17c66a74dbc5686ff063"
MBD = f"https://raw.githubusercontent.com/janosh/matbench-discovery/{C}"

# (checkpoint as used in P3, training data, partition statement, sources); every statement is from the cited file
CHECKPOINTS = [
    ("MACE-MP-0 small (2023-12-10-mace-128-L0_energy_epoch-249) - feature backbone", "MPtrj",
     "not documented in the sources read",
     "ACEsuit/mace-foundations README table: MACE-MP-0a, Training Dataset 'MPTrj', sizes small/medium/large "
     "(https://raw.githubusercontent.com/ACEsuit/mace-foundations/main/README.md)"),
    ("MACE-MP-0 medium (2023-12-03-mace-128-L1_epoch-199) - feature backbone + MBD prediction file", "MPtrj",
     "not documented in the sources read",
     f"MBD models/mace/mace-mp-0.yml: training_sets [MPtrj]; 'uses the same MPtrj-trained medium checkpoint' "
     f"({MBD}/models/mace/mace-mp-0.yml); mace-foundations README (as above)"),
    ("CHGNet 0.3.0 - feature backbone + MBD prediction file", "MPtrj",
     "'MPtrj dataset with 9-0.5-0.5 train-val-test splitting' (split membership not published in the sources read)",
     f"CederGroupHub/chgnet chgnet/pretrained/0.3.0/README.md "
     f"(https://raw.githubusercontent.com/CederGroupHub/chgnet/main/chgnet/pretrained/0.3.0/README.md); MBD "
     f"models/chgnet/chgnet-0.3.0.yml: training_sets [MPtrj], 'MPtrj ... pulled in September 2022 from the v2021.11.10 "
     f"MP release' ({MBD}/models/chgnet/chgnet-0.3.0.yml)"),
    ("SevenNet-0 (11July2024) - feature backbone + MBD prediction file", "MPtrj",
     "'for SevenNet-0 (11July2024), we did not split the dataset' (only a Train MAE row)",
     "MDIL-SNU/SevenNet sevenn/pretrained_potentials/SevenNet_0__11July2024/README.md at afb56e10b6: 'pre-trained on "
     "the MPTrj(CHGNet) dataset' (https://raw.githubusercontent.com/MDIL-SNU/SevenNet/afb56e10b6/sevenn/"
     f"pretrained_potentials/SevenNet_0__11July2024/README.md); MBD models/sevennet/sevennet-0.yml: training_sets [MPtrj]"),
    ("MatterSim v1.0.0-1M - feature backbone", "MatterSim proprietary data (unreleased); model card: training data "
     "size 3M for 1M", "not documented in the sources read",
     "microsoft/mattersim MODEL_CARD.md at d0a52e64fc, 'Available Models' table "
     "(https://raw.githubusercontent.com/microsoft/mattersim/d0a52e64fc/MODEL_CARD.md); MBD data/datasets.yml "
     f"MatterSim: access 'unreleased' ({MBD}/data/datasets.yml)"),
    ("MatterSim v1.0.0-5M - MBD prediction file only", "MatterSim (6M per model card); MBD notes list sources "
     "'Materials Project, Alexandria dataset, newly generated structures and MD trajectories'",
     "not documented in the sources read", f"{MBD}/models/mattersim/mattersim-v1-5m.yml; MODEL_CARD.md as above"),
    ("ORB v3 conservative-inf-mpa (orb-v3-conservative-inf-mpa-20250404) - feature backbone + MBD prediction file",
     "MBD yml: training_sets [MPtrj, Alex, OMat24]; 'pretrained diffusion model backbone which is fine-tuned on "
     "MPtrj + Alexandria + OMAT'; fine-tuned < 1 epoch (10k steps)", "not documented in the sources read",
     f"{MBD}/models/orb/orb-v3.yml; orbital-materials/orb-models MODELS.md: suffix Z = training dataset 'omat' or "
     "'mpa' (https://raw.githubusercontent.com/orbital-materials/orb-models/main/MODELS.md). The sources read say "
     "Alexandria, not sAlex, for this checkpoint (MBD datasets.yml only recommends sAlex)."),
    ("MACE-MPA-0 medium - MBD prediction file only", "MPtrj + sAlex", "not documented in the sources read",
     "mace-foundations README table row MACE-MPA-0, Training Dataset 'MPTrj + sAlex'"),
]


def main() -> int:
    s = json.loads(SUMMARY.read_text())
    ds = yaml.safe_load((M / "data_datasets_e3d7966.yml").read_text())["MPtrj"]
    cat = yaml.safe_load((M / "data-files_e3d7966.yml").read_text())["mp_trj_extxyz"]
    src = s["source"]
    ck = s["checks"]
    doc_frames, doc_mats = int(ds["n_structures"]), int(ds["n_materials"])
    must_be_zero = ["frames_without_material_id", "frames_without_formula", "frames_formula_vs_species_mismatch",
                    "frames_chemsys_differs_within_material", "members_with_not_one_material_id",
                    "members_material_id_ne_filename"]
    fails = [k for k in must_be_zero if ck.get(k, 0)]
    if not src["md5_match"]:
        fails.append("stream md5 != supplied md5")

    L = ["# MPtrj membership manifest (chemical-system level)", "",
         f"Generated by src/membership/write_mptrj_report.py from result/membership/mptrj_membership_summary.json "
         f"(created {s['created_utc']}, code sha256 `{s['code']['sha256']}`).", "",
         "## Source and access", "",
         f"- Figshare article {src['figshare_article']}, file {src['file_id']} `{src['file_name']}`, "
         f"{src['size_bytes']} B; supplied md5 `{src['supplied_md5']}`; catalogue entry `mp_trj_extxyz` "
         f"(md5 `{cat['md5']}`): \"{cat['description']}\"",
         f"- Read through HTTP Range requests ({src['range_requests']} block requests, {src['bytes_fetched']} B "
         "fetched; ndownloader URL followed to the signed storage URL on every request); every response was "
         "required to be HTTP 206 with the exact Content-Range, else abort. No local copy was stored.",
         f"- Whole-stream md5 `{src['stream_md5']}` (match with supplied: **{src['md5_match']}**), sha256 "
         f"`{src['stream_sha256']}`. zipfile verified each member's CRC-32.", "",
         "## Counts", "",
         "| quantity | value |", "|---|---|",
         f"| zip members (central directory) | {s['n_zip_members_central_directory']} |",
         f"| frames | {s['n_frames']} |",
         f"| atoms (sum over frames) | {s['n_atoms_total']} |",
         f"| materials (distinct material_id) | {s['n_materials']} |",
         f"| chemical systems | {s['n_chemical_systems']} |",
         f"| systems by arity | {s['n_systems_by_arity']} |", "",
         "## Keys parsed", "",
         f"- Comment-line keys seen (frame counts): {s['comment_line_keys_frame_counts']}",
         f"- Used: {s['keys_parsed']}",
         f"- Not parsed / not stored: {s['not_parsed']}", "",
         "## Self-checks", "",
         f"- Frames: {s['n_frames']} here vs {doc_frames} = MBD data/datasets.yml MPtrj n_structures (original "
         f"MPtrj release, Figshare 23713842); difference {s['n_frames'] - doc_frames:+d}"
         + (" (MATCH)" if s["n_frames"] == doc_frames else " (MISMATCH; the 2024-09-03 extxyz is described as "
            "'subsampled to reduce redundancy and faulty data', so a difference can be by construction)") + ".",
         f"- Materials: {s['n_materials']} here vs {doc_mats} = datasets.yml MPtrj n_materials; difference "
         f"{s['n_materials'] - doc_mats:+d}" + (" (MATCH)" if s["n_materials"] == doc_mats else " (MISMATCH)") + ".",
         f"- Zip members {s['n_zip_members_central_directory']} vs materials {s['n_materials']}: one member per "
         "material" + (" (MATCH)" if s["n_zip_members_central_directory"] == s["n_materials"] else " (MISMATCH)") + ".",
         f"- Parse counters (must be 0 unless noted): {ck if ck else '{} (all zero)'}; examples: "
         f"{s['check_examples'] if s['check_examples'] else 'none'}.",
         "- `frames_formula_differs_within_material` (if present) counts frames whose formula key differs from the "
         "material's first frame (e.g. a different cell multiple); the chemical system is checked separately and "
         "must not differ.",
         f"- Gate result: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}", "",
         "## Output", "",
         f"- `{s['outputs']['csv']}` ({s['outputs']['csv_bytes']} B, sha256 `{s['outputs']['csv_sha256']}`; "
         "gitignored cache): material_id, formula (first frame's formula key), chemical_system (species set), n_frames.",
         "",
         "## Training data per checkpoint (as documented; no guesses)", "",
         "| checkpoint (P3 use) | training data | train/validation partition | source |", "|---|---|---|---|"]
    for row in CHECKPOINTS:
        L.append("| " + " | ".join(row) + " |")
    L += ["", "Membership here is the full MPtrj file. Where a checkpoint held out part of MPtrj (CHGNet 0.3.0: "
          "9-0.5-0.5) the held-out materials are not identified in the sources read, so `sys_in_mptrj` is an upper "
          "bound on that checkpoint's training exposure at chemical-system level.", ""]
    OUT.write_text("\n".join(L))
    print("\n".join(L))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
