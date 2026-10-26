"""Exploratory study E1 (docs/EXPLORATORY_E1_NATIVE_GATE.md rev 1): diagnostics on DEVELOPMENT data only. No confirmatory
claim. NOT TO BE RUN on real data before the reviewer's implementation review.

    python scripts/e1_run.py                      # real development data -> results/exploratory/e1_report.json
    python scripts/e1_run.py --synthetic OUT.json # synthetic dry run of the whole script path (no real data)

Real data is loaded only through phase3_g3a.load_g3a (its sealed-path guard, id allow-list and deny-list). Before anything is
loaded, the frozen G3a inputs are hash-checked by the G3a script's own check_frozen. The report records the code identity
(sha256 of the E1 and G3a files, git commit) and the input provenance. Refuses to overwrite its report.
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

from lookahead import e1 as E
from lookahead import gate1 as g1
from lookahead import phase3_g3a as G
from lookahead.config import REPO_ROOT

REPORT = "results/exploratory/e1_report.json"
CODE_FILES = ["src/lookahead/e1.py", "scripts/e1_run.py", "src/lookahead/phase3_g3a.py", "src/lookahead/phase3_readouts.py"]


def _sha(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def g3a_script(root: Path):
    """The G3a entry script as a module (for its check_frozen); importing it runs nothing."""
    spec = importlib.util.spec_from_file_location("phase3_g3a_script", root / "scripts" / "phase3_g3a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_identity(root: Path) -> dict:
    out = {f: _sha(root / f) for f in CODE_FILES}
    try:
        import subprocess

        out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                           check=True).stdout.strip()
        out["git_dirty_code"] = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=root,
                                                    capture_output=True, text=True, check=True).stdout.strip())
    except Exception as e:
        out["git_commit"] = f"unavailable ({type(e).__name__})"
    return out


def summary_lines(rep: dict) -> list:
    lines = [f"E1 ({rep['status']}); plan {rep['plan']}"]
    lad = rep["A1"]["ladder"]
    for n, v in lad.items():
        ci = rep["A1"]["paired_vs_native_pb"].get(n, {}).get("saving_difference_primary")
        extra = "" if ci is None else f"  paired diff {ci['point']:+.4f} CI95 [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
        lines.append(f"A1  {n:12s} held-out primary saving {v['heldout_saving_primary']:.4f}  "
                     f"vs native-pb {v['improvement_vs_native_pb']:+.4f}{extra}")
    for n in ("pi1-x@c_t", "pi1-x@safe"):
        lines.append(f"A2  {n:12s} vs native-pb {rep['A2']['stopping_utility'][n.split('@')[1]]['point']['improvement_vs_native_pb']:+.4f}")
    for n in ("pi1-x+cummax", "pi1-x+rank"):
        lines.append(f"A3  {n:12s} vs native-pb {rep['A3'][n]['improvement_vs_native_pb']:+.4f}")
    lines.append(f"A5  combo-max vs native-pb {rep['A5']['combo_max']['improvement_vs_native_pb']:+.4f}")
    for rule in ("point", "risk"):
        a = rep["A6"][rule]["native-pb"]
        lines.append(f"A6  {rule:5s} native-pb saving {a['heldout_saving_primary']:.4f}, violations {a['n_violations_primary']}, "
                     f"worst loss {a['worst_loss_primary']:.3f} pp")
    if rep["A7"]:
        lines.append("A7  (descriptive) validation vs native-pb: " + ", ".join(
            f"{n} {v['improvement_vs_native_pb']:+.4f}" for n, v in rep["A7"]["methods"].items() if n != "native-pb"))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", metavar="OUT", help="dry run on synthetic data; writes OUT (no real data is read)")
    args = ap.parse_args(argv)
    root = REPO_ROOT
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.synthetic:
        sys.path.insert(0, str(root / "tests"))
        from test_phase3_g3a import synth

        out_path = Path(args.synthetic)
        if out_path.exists():
            print(f"refusing: {out_path} already exists", file=sys.stderr)
            return 2
        dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
        val = synth(n=50, seed=12, source="p1_validation", prefix="v/")
        inputs = {"synthetic": True}
    else:
        out_path = root / REPORT
        if out_path.exists():
            print(f"refusing: {REPORT} already exists", file=sys.stderr)
            return 2
        inputs = g3a_script(root).check_frozen(root)
        dev, val = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=True)
        inputs["data_files"] = {**dev.provenance, **val.provenance}
    rep = E.run_e1(dev, val)
    rep["provenance"] = {"inputs": inputs, "code": code_identity(root), "started_utc": started,
                         "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    for ln in summary_lines(rep):
        print(ln)
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
