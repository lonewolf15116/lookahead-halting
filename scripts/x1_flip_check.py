"""Exploratory study X1 (docs/X1_FLIP_PROTOCOL.md rev 2): correctness flips at matched saving, DEVELOPMENT data only.

    python scripts/x1_flip_check.py                      # real development data -> results/exploratory/x1_flips.json
    python scripts/x1_flip_check.py --synthetic OUT.json # synthetic dry run of the whole script path (no real data)

Real data is loaded only through phase3_g3a.load_g3a (validation not requested), after the G3a script's own check_frozen.
Any path containing "extracted_phase3_test" is refused. The report is created exclusively (never overwritten) and records the
git commit, the protocol sha256, the code sha256s and the fold fingerprints (to be checked against E1's).
"""
import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from lookahead import gate1 as g1
from lookahead import phase3_g3a as G
from lookahead import x1_flips as X
from lookahead.config import REPO_ROOT

REPORT = "results/exploratory/x1_flips.json"
PROTOCOL_FILE = "docs/X1_FLIP_PROTOCOL.md"
CODE_FILES = ["src/lookahead/x1_flips.py", "scripts/x1_flip_check.py", "src/lookahead/e1.py", "src/lookahead/phase3_g3a.py",
              "src/lookahead/phase3_readouts.py"]
SEALED = "extracted_phase3_test"


def _sha(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def refuse_sealed(*paths) -> None:
    """SystemExit(2) if any path mentions the sealed TEST directory."""
    for p in paths:
        if SEALED in str(p).replace("\\", "/"):
            print(f"refusing: {p} refers to {SEALED}", file=sys.stderr)
            raise SystemExit(2)


def code_identity(root: Path) -> dict:
    out = {"protocol_sha256": _sha(root / PROTOCOL_FILE), **{f: _sha(root / f) for f in CODE_FILES}}
    try:
        out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                           check=True).stdout.strip()
        out["git_dirty_code"] = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "docs"], cwd=root,
                                                    capture_output=True, text=True, check=True).stdout.strip())
    except Exception as e:
        out["git_commit"] = f"unavailable ({type(e).__name__})"
    return out


def g3a_check_frozen(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location("phase3_g3a_script", root / "scripts" / "phase3_g3a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_frozen(root)


def summary_lines(rep: dict) -> list:
    r = rep["reading"]
    lines = [f"X1 ({rep['status']}); {rep['protocol']}; reading: {r['reading']} (F {r['F_levels']}, K {r['K_levels']}, "
             f"L {r['L_levels']}, losses sentence allowed: {r['losses_sentence_allowed']})"]
    for l, e in rep["contrasts"]["primary"][X.PRIMARY_CONTRAST]["levels"].items():
        if e:
            d = e["delta"]["phi"]
            lines.append(f"  s={l} dphi {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n_bench={len(e['included'])}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", metavar="OUT", help="dry run on synthetic data; writes OUT (no real data is read)")
    args = ap.parse_args(argv)
    root = REPO_ROOT
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out_path = Path(args.synthetic) if args.synthetic else root / REPORT
    refuse_sealed(out_path)
    if out_path.exists():
        print(f"refusing: {out_path} already exists", file=sys.stderr)
        return 2
    if args.synthetic:
        sys.path.insert(0, str(root / "tests"))
        from test_phase3_g3a import synth

        dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
        inputs = {"synthetic": True}
    else:
        inputs = g3a_check_frozen(root)
        dev, _ = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=False)
        refuse_sealed(json.dumps(dev.provenance, default=str))     # keys AND values (file paths), not just keys
        inputs["data_files"] = dev.provenance
    rep = X.run_x1(dev)
    rep["provenance"] = {"inputs": inputs, "code": code_identity(root), "started_utc": started,
                         "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "x", encoding="utf-8", newline="\n") as f:      # exclusive creation
            f.write(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    except FileExistsError:
        print(f"refusing: {out_path} already exists", file=sys.stderr)
        return 2
    for ln in summary_lines(rep):
        print(ln)
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
