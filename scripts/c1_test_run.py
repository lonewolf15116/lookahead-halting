"""C1 ONE-TIME TEST RUN (docs/C1_PROTOCOL.md C1.7 step 3). Run ONLY after the reviewer approves this script.

    python scripts/c1_test_run.py --approved-code-sha <sha256 of the reviewed TEST code bundle>

Order, enforced:
  1. Checks, before anything else (any failure exits with code 2 and touches nothing):
     protocol docs/C1_PROTOCOL.md, results/c1/frozen.json, results/c1/pi1x_models.npz, the phase3 manifest and the
     committed TEST extraction check equal their pinned sha256; every code file recorded in frozen.json at freeze is
     unchanged; the TEST code bundle (this script + EVERY .py file of src/lookahead) equals --approved-code-sha;
     results/c1/TEST_OPENED.json, test_report.json and TEST_FAILED.json do not exist; and a METADATA-ONLY check of the
     TEST extraction (c1_test.metadata_preflight: id lists vs manifest, per-part done.json counts/ids/statuses, raw-file
     checksums; no outcome is read) passes.
  2. results/c1/TEST_OPENED.json is created with EXCLUSIVE create (it can never be overwritten) BEFORE any TEST file is
     opened. From then on TEST counts as opened, even if the run fails; the marker is never deleted by this script. If
     anything fails after it, results/c1/TEST_FAILED.json records the stage and exception type (no message, no values)
     and the run is never retried.
  3. TEST is read once (c1_test.load_test inside c1_test.unsealed(root), which admits only resolved paths inside
     data/extracted_phase3_test), the frozen policies are applied, and
     results/c1/test_report.json is written with exclusive create.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from lookahead import c1_test as T
from lookahead import gate1 as g1
from lookahead.config import REPO_ROOT

PINNED = {
    "docs/C1_PROTOCOL.md": "4f38b8b71479bdf2236f3887eac4dd56026c2b20b7edac10ed581529d00a2050",
    "results/c1/frozen.json": "65728e16113e94114554fb28a951eee2b93af4b3fb5964bf32cf12845560b25f",
    "results/c1/pi1x_models.npz": "72f1550813e12f242128345b4b20097eca0c83ebe843b358f5649a8ed8938bad",
    "data/manifests/phase3_questions.json": "b2df19303860c0d543d49a06454a1268addfee6fada8a7465c5dd71da3619cec",
    "results/phase3/extraction_check_phase3_test.json": "c8d9c20206209ae2fafae46c8cb5c420657f10698170ade205e9e646ca9318b2",
}
BUNDLE_SCRIPT = "scripts/c1_test_run.py"
PACKAGE_DIR = "src/lookahead"             # EVERY module of the package is in the bundle: all code the run can execute
MARKER = "results/c1/TEST_OPENED.json"
REPORT = "results/c1/test_report.json"
FAILED = "results/c1/TEST_FAILED.json"


def _sha_norm(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def bundle_files(root: Path) -> list:
    """The executable code of the run: this script plus every .py file of the lookahead package (c1_test, c1, e1,
    phase3_g3a, phase3_readouts, gate1, gate2, phase3_build, ... — everything an import can reach), sorted. A stray or
    changed file in the package changes the hash."""
    pkg = sorted(p.relative_to(root).as_posix() for p in (root / PACKAGE_DIR).glob("*.py"))
    return [BUNDLE_SCRIPT] + pkg


def bundle_sha(root: Path) -> str:
    """sha256 over the bundle files (path, NUL, LF-normalised bytes, NUL), in bundle_files order."""
    import hashlib

    h = hashlib.sha256()
    for f in bundle_files(root):
        h.update(f.encode() + b"\0" + (root / f).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def preflight(root: Path, approved_code_sha: str) -> dict:
    """Every check that must pass before TEST is touched. Raises SystemExit (nothing is written)."""
    for rel, exp in PINNED.items():
        p = root / rel
        if not p.exists():
            raise SystemExit(f"missing frozen artifact {rel}")
        if exp not in (g1.sha256_file(p), _sha_norm(p)):
            raise SystemExit(f"{rel} is not the frozen file")
    frozen = json.loads((root / "results/c1/frozen.json").read_text(encoding="utf-8"))
    for rel, recorded in frozen["provenance"]["code"].items():
        if _sha_norm(root / rel) != recorded:
            raise SystemExit(f"{rel} changed since the freeze")
    if bundle_sha(root) != approved_code_sha:
        raise SystemExit("the TEST code bundle is not the approved one (--approved-code-sha)")
    for rel in (MARKER, REPORT, FAILED):
        if (root / rel).exists():
            raise SystemExit(f"refusing: {rel} exists; the C1 TEST run is one-time")
    frozen["_test_metadata"] = T.metadata_preflight(root)        # metadata only; no TEST outcome is read
    return frozen


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def write_failure(root: Path, stage: str, exc: BaseException) -> Path:
    """Durable failure status after the marker (reviewer, TEST script reviews): stage, exception type, timestamp and the
    code locations of the traceback; deliberately NO exception message and no values, so nothing from TEST leaks into it.

    Published by EXCLUSIVE create only (open mode 'x'): an existing record is never overwritten, not even in a race. If
    TEST_FAILED.json already exists, the record goes to TEST_FAILED.<UTC timestamp>.<n>.json, again by exclusive create.
    The content is flushed and fsync'ed before the file is closed. The marker stays; no retry is possible. Returns the
    path written.

    Partial writes: if the disk write itself fails after the exclusive create, the (possibly truncated) record is left as
    is, never deleted or replaced, and the caller re-raises the ORIGINAL exception (main()). Directory fsync is POSIX only;
    on Windows the file-level fsync is the durability step."""
    import os
    import traceback

    now = datetime.datetime.now(datetime.timezone.utc)
    rec = {"failed_utc": now.isoformat(), "stage": stage, "exception_type": type(exc).__name__,
           "traceback_locations": [f"{Path(fr.filename).name}:{fr.lineno} in {fr.name}" for fr in traceback.extract_tb(exc.__traceback__)],
           "note": "TEST counts as opened; the C1 run is one-time and is NOT retried. Report this file to the reviewer."}
    data = json.dumps(rec, indent=1, sort_keys=True) + "\n"
    first = root / FAILED
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    candidates = [first] + [first.with_name(f"TEST_FAILED.{stamp}.{i}.json") for i in range(100)]
    for path in candidates:
        try:
            f = open(path, "x", encoding="utf-8", newline="\n")        # exclusive create: never replaces anything
        except FileExistsError:
            continue
        with f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:                                                           # best effort: persist the directory entry (POSIX)
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except (OSError, AttributeError):
            pass
        return path
    raise RuntimeError("could not create a failure record without overwriting")


def main(argv=None, *, root: Path | None = None, loader=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved-code-sha", required=True)
    args = ap.parse_args(argv)
    root = Path(root or REPO_ROOT)
    try:
        frozen = preflight(root, args.approved_code_sha)
    except SystemExit as e:
        print(f"PREFLIGHT FAILED (TEST untouched): {e}", file=sys.stderr)
        return 2
    marker = {"opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pinned": PINNED,
              "test_metadata": frozen.get("_test_metadata"), "test_code_bundle_files": bundle_files(root),
              "test_code_bundle_sha256": args.approved_code_sha, "git_commit": git_commit(root),
              "note": "TEST counts as opened from this moment; this file is never removed"}
    (root / MARKER).parent.mkdir(parents=True, exist_ok=True)
    with open(root / MARKER, "x", encoding="utf-8", newline="\n") as f:          # exclusive create, before any TEST read
        f.write(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    stage = "load models"
    try:
        with np.load(root / "results/c1/pi1x_models.npz") as npz:
            models = T.load_models(npz)
        stage = "load TEST"
        with T.unsealed(root):
            test = (loader or T.load_test)(root)
        stage = "evaluate"
        frozen.pop("_test_metadata", None)
        rep = T.evaluate_c1(frozen, models, test)
        rep["provenance"] = {"marker": marker, "test_metadata": marker.get("test_metadata"), "test_data_files": test.provenance,
                             "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        stage = "write report"
        with open(root / REPORT, "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rep, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    except BaseException as e:
        try:                                 # recording the failure must never hide the original exception
            write_failure(root, stage, e)
        except BaseException as rec_err:     # noqa: BLE001
            print(f"WARNING: the failure record could not be written ({type(rec_err).__name__}); "
                  f"the original error follows. TEST counts as opened.", file=sys.stderr)
        raise
    p = rep["primary"]
    print(f"VERDICT: {rep['verdict']}")
    print(f"primary: delta (gross, pi1-x - native-pb) = {p['delta_gross']:+.4f}, 95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
          f"  (delta_min {rep['delta_min']}; L > 0.02: {p['L_above_delta_min']})")
    for n, t in rep["budget_table"].items():
        print(f"budget {n:10s} " + "  ".join(f"{b}: loss {v['loss_pp']:.2f} pp (UB {v['upper_bound_95_pp']:.2f}"
                                             f"{' >1!' if v['upper_bound_over_1pp'] else ''})" for b, v in t.items()))
    if "secondary_net" in rep:
        s = rep["secondary_net"]
        print(f"net (estimated deployment cost): delta {s['delta_net']:+.4f}, 95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]")
    print(f"report: {root / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
