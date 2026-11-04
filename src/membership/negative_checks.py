"""Prove the gates can fail: (1) the HTTP Range reader rejects a response whose Content-Range does not match the
expected total size; (2) the label-blind loader rejects a label column in its allow-list before reading any data.
python3 src/membership/negative_checks.py   (exit 0 iff both gates fail as intended)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src/membership"))
sys.path.insert(0, str(ROOT / "src"))
from mptrj_membership import HTTPRangeReader  # noqa: E402
import splits.make_splits as ms  # noqa: E402

URL, SIZE = "https://ndownloader.figshare.com/files/49034296", 1521713089
bad = 0
ok = HTTPRangeReader(URL, SIZE, 1 << 20)._fetch(0, 63)
print("positive control: HTTP 206 + exact Content-Range, first bytes", ok[:4])
try:
    HTTPRangeReader(URL, SIZE - 1, 1 << 20)._fetch(0, 63)
    print("NEGATIVE CHECK 1 FAILED: wrong total size accepted")
    bad += 1
except RuntimeError as e:
    print("negative check 1 (wrong total size) raises as intended:", e)
ms.ALLOWED_COLUMNS["wbm"] = ("material_id", "formula", "unique_prototype", "e_form_per_atom_mp2020_corrected")
try:
    ms.load_label_blind(ROOT / "cache/mbd/2023-12-13-wbm-summary_v64706751.csv.gz", "wbm")
    print("NEGATIVE CHECK 2 FAILED: label column accepted")
    bad += 1
except AssertionError as e:
    print("negative check 2 (label column in allow-list) raises before reading:", str(e)[:60])
sys.exit(1 if bad else 0)
