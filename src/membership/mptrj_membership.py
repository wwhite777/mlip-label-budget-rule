"""MPtrj membership manifest by HTTP Range streaming (no local copy of the 1.52 GB zip).

python3 src/membership/mptrj_membership.py [--block-mb 16] [--max-members N]

Source: Matbench Discovery Figshare article 22715158, file 49034296 "2024-09-03-mp-trj.extxyz.zip"
(supplied md5 7f433171e4e5f2ef9304dccd42d5488f). The download URL is resolved through the Figshare API; every block
request goes to the ndownloader URL and follows its redirect to the (short-lived, signed) storage URL. Every response
must be HTTP 206 with a Content-Range equal to the requested range and the expected total size, otherwise the run
aborts (Range not honoured -> stop, never fall back to a full download).

The zip's central directory and members are read with Python's zipfile through a seekable reader (16 MB blocks, small
LRU cache, read-ahead of the next --prefetch blocks in worker threads, nothing written to disk). Members are visited in header-offset order, so the stream is fetched almost
sequentially; md5 and sha256 of the whole byte stream are accumulated over in-order blocks and compared with the
Figshare supplied md5 (whole-file integrity check). zipfile verifies each member's CRC-32 when read to EOF.

Each extxyz member is parsed line by line: natoms line, comment line, natoms atom lines. From the comment line only
key NAMES (all frames) and the values of material_id, formula and frame_id are taken; energies/stress/forces are never
parsed or stored. Composition is taken from the species column (first token of each atom line) and cross-checked
against the formula key. chemical_system = sorted set of species symbols joined by "-" (same rule as
src/splits/make_splits.py).

Outputs:
  cache/membership/mptrj_frames_by_material.csv.gz   material_id, formula, chemical_system, n_frames (gzip mtime=0)
  result/membership/mptrj_membership_summary.json    counts, keys, self-checks, stream md5/sha256, output sha256
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import sys
import threading
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ARTICLE_ID = 22715158
FILE_ID = 49034296
FILE_NAME = "2024-09-03-mp-trj.extxyz.zip"
SUPPLIED_MD5 = "7f433171e4e5f2ef9304dccd42d5488f"
OUT_CSV = ROOT / "cache/membership/mptrj_frames_by_material.csv.gz"
OUT_JSON = ROOT / "result/membership/mptrj_membership_summary.json"
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
KEY_RE = re.compile(r'(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=')
MID_RE = re.compile(r'(?:^|\s)material_id=("?)([^"\s]+)\1')
FORMULA_RE = re.compile(r'(?:^|\s)formula=(?:"([^"]*)"|(\S+))')
FRAME_RE = re.compile(r'(?:^|\s)frame_id=("?)([^"\s]+)\1')


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}", flush=True)


class HTTPRangeReader(io.RawIOBase):
    """Read-only seekable file over HTTP Range requests with block cache and in-order stream hashing."""

    def __init__(self, url: str, size: int, block: int, cache_blocks: int = 4, prefetch: int = 0):
        super().__init__()
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self.cache_blocks = cache_blocks
        self.n_blocks = (size + block - 1) // block
        self.prefetch = prefetch  # read-ahead of the next N blocks in worker threads (0 = off)
        self.pool = cf.ThreadPoolExecutor(prefetch) if prefetch else None
        self.futures: dict[int, cf.Future] = {}
        self.md5, self.sha256 = hashlib.md5(), hashlib.sha256()
        self.hashed_upto = 0
        self.lock = threading.Lock()
        self.n_requests = 0
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        want = f"bytes {start}-{end}/{self.size}"
        last_err = None
        for attempt in range(8):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=180,
                                 allow_redirects=True)
                if r.status_code != 206:
                    raise RuntimeError(f"Range NOT honoured: HTTP {r.status_code} for bytes={start}-{end}")
                got = r.headers.get("Content-Range", "")
                if got != want:
                    raise RuntimeError(f"Content-Range mismatch: got {got!r}, want {want!r}")
                data = r.content
                if len(data) != end - start + 1:
                    raise IOError(f"short body {len(data)} != {end - start + 1}")
                with self.lock:
                    self.n_requests += 1
                    self.bytes_fetched += len(data)
                return data
            except RuntimeError:
                raise
            except Exception as e:  # network hiccup: retry with backoff
                last_err = e
                log(f"retry {attempt + 1} for bytes={start}-{end}: {e!r}")
                time.sleep(min(60, 2 ** attempt))
        raise IOError(f"giving up on bytes={start}-{end}: {last_err!r}")

    def _get_block(self, idx: int) -> bytes:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            data = self.cache[idx]
        else:
            fut = self.futures.pop(idx, None)
            data = fut.result() if fut is not None else self._fetch(*self._span(idx))
            self.cache[idx] = data
            if len(self.cache) > self.cache_blocks:
                self.cache.popitem(last=False)
            if self.pool is not None:
                for j in range(idx + 1, min(self.n_blocks, idx + 1 + self.prefetch)):
                    if j not in self.cache and j not in self.futures:
                        self.futures[j] = self.pool.submit(self._fetch, *self._span(j))
        if idx * self.block == self.hashed_upto:
            self.md5.update(data)
            self.sha256.update(data)
            self.hashed_upto += len(data)
        return data

    def _span(self, idx: int) -> tuple[int, int]:
        start = idx * self.block
        return start, min(self.size, start + self.block) - 1

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        out = memoryview(b)
        done = 0
        while done < n:
            idx = (self.pos + done) // self.block
            blk = self._get_block(idx)
            off = (self.pos + done) - idx * self.block
            k = min(n - done, len(blk) - off)
            out[done:done + k] = blk[off:off + k]
            done += k
        self.pos += n
        return n

    def finish_hash(self) -> tuple[str, str]:
        while self.hashed_upto < self.size:
            self._get_block(self.hashed_upto // self.block)
        return self.md5.hexdigest(), self.sha256.hexdigest()


def formula_counts(formula: str) -> dict:
    c: dict = collections.Counter()
    for el, n in FORMULA_TOKEN_RE.findall(formula):
        c[el] += float(n) if n else 1.0
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--prefetch", type=int, default=4, help="blocks fetched ahead in parallel threads")
    ap.add_argument("--max-members", type=int, default=0, help="debug: stop after N members (no outputs written)")
    args = ap.parse_args()
    t0 = time.time()

    api = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files?page_size=1000"
    meta = next(f for f in requests.get(api, timeout=60).json() if f["id"] == FILE_ID)
    assert meta["name"] == FILE_NAME, meta["name"]
    assert meta["supplied_md5"] == SUPPLIED_MD5, meta["supplied_md5"]
    url, size = meta["download_url"], int(meta["size"])
    log(f"figshare file {FILE_ID} {meta['name']} size={size} supplied_md5={meta['supplied_md5']} url={url}")

    rdr = HTTPRangeReader(url, size, args.block_mb << 20, prefetch=args.prefetch)
    # explicit Range probe: 64 bytes at offset 0 must come back as 206 with the right Content-Range
    probe = rdr._fetch(0, 63)
    assert probe[:4] == b"PK\x03\x04", probe[:4]
    log("Range probe OK (HTTP 206, Content-Range matches)")

    zf = zipfile.ZipFile(rdr)
    infos = sorted(zf.infolist(), key=lambda z: z.header_offset)
    n_members_cd = len(infos)
    log(f"central directory: {n_members_cd} members; first={infos[0].filename} last={infos[-1].filename}")

    mats: dict[str, dict] = {}
    key_counter: collections.Counter = collections.Counter()
    checks = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    n_frames_total = 0
    n_atoms_total = 0
    uncompressed_total = 0

    for i, zi in enumerate(infos):
        if args.max_members and i >= args.max_members:
            break
        if zi.is_dir():
            checks["dir_entries"] += 1
            continue
        uncompressed_total += zi.file_size
        stem = zi.filename.rsplit("/", 1)[-1].rsplit(".extxyz", 1)[0]
        member_mids = set()
        with zf.open(zi) as fh:
            lines = io.TextIOWrapper(fh, encoding="utf-8")
            while True:
                head = lines.readline()
                if not head:
                    break
                if not head.strip():
                    checks["blank_lines_between_frames"] += 1
                    continue
                nat = int(head.strip())
                comment = lines.readline().rstrip("\n")
                species = collections.Counter()
                for _ in range(nat):
                    ln = lines.readline()
                    if not ln:
                        raise IOError(f"truncated frame in {zi.filename}")
                    species[ln.split(None, 1)[0]] += 1
                n_frames_total += 1
                n_atoms_total += nat
                for k in KEY_RE.findall(comment):
                    key_counter[k] += 1
                m = MID_RE.search(comment)
                if m is None:
                    checks["frames_without_material_id"] += 1
                    mid = stem
                else:
                    mid = m.group(2)
                member_mids.add(mid)
                fm = FORMULA_RE.search(comment)
                formula = (fm.group(1) if fm.group(1) is not None else fm.group(2)) if fm else None
                if formula is None:
                    checks["frames_without_formula"] += 1
                elif formula_counts(formula) != {k: float(v) for k, v in species.items()}:
                    checks["frames_formula_vs_species_mismatch"] += 1
                    if len(examples["formula_vs_species"]) < 5:
                        examples["formula_vs_species"].append([zi.filename, formula, dict(species)])
                if FRAME_RE.search(comment) is None:
                    checks["frames_without_frame_id"] += 1
                csys = "-".join(sorted(species))
                rec = mats.get(mid)
                if rec is None:
                    mats[mid] = {"formula": formula, "chemical_system": csys, "n_frames": 1}
                else:
                    rec["n_frames"] += 1
                    if rec["chemical_system"] != csys:
                        checks["frames_chemsys_differs_within_material"] += 1
                        if len(examples["chemsys_within_material"]) < 5:
                            examples["chemsys_within_material"].append([mid, rec["chemical_system"], csys])
                    if rec["formula"] != formula:
                        checks["frames_formula_differs_within_material"] += 1
                        if len(examples["formula_within_material"]) < 5:
                            examples["formula_within_material"].append([mid, rec["formula"], formula])
        if len(member_mids) != 1:
            checks["members_with_not_one_material_id"] += 1
        elif next(iter(member_mids)) != stem:
            checks["members_material_id_ne_filename"] += 1
            if len(examples["mid_ne_filename"]) < 5:
                examples["mid_ne_filename"].append([zi.filename, sorted(member_mids)])
        if (i + 1) % 5000 == 0:
            mb = rdr.bytes_fetched / 2**20
            log(f"{i + 1}/{n_members_cd} members, {n_frames_total} frames, {mb:.0f} MiB fetched, "
                f"{rdr.n_requests} requests, {time.time() - t0:.0f} s")

    if args.max_members:
        log(f"debug stop: {len(mats)} materials, {n_frames_total} frames, keys={dict(key_counter)}, "
            f"checks={dict(checks)}")
        return 0

    md5, sha = rdr.finish_hash()
    log(f"stream md5={md5} (supplied {SUPPLIED_MD5}) sha256={sha}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    buf.write("material_id,formula,chemical_system,n_frames\n")
    for mid in sorted(mats):
        r = mats[mid]
        f = (r["formula"] or "").replace('"', "")
        buf.write(f'{mid},"{f}",{r["chemical_system"]},{r["n_frames"]}\n')
    with open(OUT_CSV, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    out_sha = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()

    systems = {r["chemical_system"] for r in mats.values()}
    arity = collections.Counter(len(s.split("-")) for s in systems)
    summary = {
        "source": {"figshare_article": ARTICLE_ID, "file_id": FILE_ID, "file_name": FILE_NAME,
                   "download_url": url, "size_bytes": size, "supplied_md5": SUPPLIED_MD5,
                   "stream_md5": md5, "stream_sha256": sha, "md5_match": md5 == SUPPLIED_MD5,
                   "range_requests": rdr.n_requests, "bytes_fetched": rdr.bytes_fetched,
                   "stored_locally": False},
        "n_zip_members_central_directory": n_members_cd,
        "uncompressed_bytes": uncompressed_total,
        "n_frames": n_frames_total,
        "n_atoms_total": n_atoms_total,
        "n_materials": len(mats),
        "n_chemical_systems": len(systems),
        "n_systems_by_arity": {str(k): v for k, v in sorted(arity.items())},
        "comment_line_keys_frame_counts": dict(key_counter.most_common()),
        "keys_parsed": {"material_id": "comment key material_id (cross-checked with member file name)",
                        "formula": "comment key formula (first frame per material; cross-checked with species)",
                        "chemical_system": "species column of the atom lines",
                        "frame_id": "presence only"},
        "not_parsed": "energy, mp2020_corrected_energy, stress, forces, magmoms (never read into variables)",
        "checks": dict(checks),
        "check_examples": dict(examples),
        "outputs": {"csv": str(OUT_CSV.relative_to(ROOT)), "csv_sha256": out_sha,
                    "csv_bytes": OUT_CSV.stat().st_size},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    log(f"wrote {OUT_CSV} ({OUT_CSV.stat().st_size} B, sha256 {out_sha}) and {OUT_JSON}")
    log(f"frames={n_frames_total} materials={len(mats)} systems={len(systems)} checks={dict(checks)}")
    if md5 != SUPPLIED_MD5:
        log("ERROR: stream md5 does not match the supplied md5")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
