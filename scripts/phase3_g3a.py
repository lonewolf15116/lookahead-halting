"""Phase 3 G3a: D1-D4 feasibility diagnostics on development data (Phase 1 train + DEV+; Phase 1 validation exposed,
reported only). TEST is never read. NOT TO BE RUN on real data before the reviewer's implementation sign-off.

    python scripts/phase3_g3a.py                      # real development data -> results/phase3/g3a_report.json
    python scripts/phase3_g3a.py --synthetic OUT.json # synthetic dry run of the whole script path (no real data)

Before anything is loaded, the frozen inputs are hash-checked: plan v1.1, A1, the frozen readout module (A2), the DEV+
extraction check report (passed), and the Gate 2 frozen C values. The report records the code identity and every input
hash. Refuses to overwrite its report. Exit 0 = report written (the decision is in the report and printed).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

from lookahead import gate1 as g1
from lookahead import phase3_g3a as G
from lookahead.config import REPO_ROOT

REPORT = "results/phase3/g3a_report.json"
PINNED = {
    "docs/PHASE3_PLAN.md": "e599f07c688f040467494a42d7efcaa36e13f437502da860ec8952944053b28f",
    "docs/PHASE3_AMENDMENT_A1.md": "6b7e73284f40e8fe93854dfe6097bb4017b3a63b3ab9c3e5d0665908965231bf",
    "src/lookahead/phase3_readouts.py": "1b3a88a3519df5fd913340c828b88d158db59f7c18df1111fd3b5fdd0e372389",
    "results/phase3/extraction_check_phase3_dev.json": "ff6e2e34a037ee73dc806f991dee3e1f8381c7eea8d956a52ed14541075a837c",
}


def _sha(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def check_frozen(root: Path) -> dict:
    out = {}
    for rel, exp in PINNED.items():
        p = root / rel
        raw = g1.sha256_bytes(p.read_bytes())
        if exp not in (raw, _sha(p)):
            raise SystemExit(f"{rel} is not the frozen file ({raw[:12]} != {exp[:12]})")
        out[rel] = exp
    chk = json.loads((root / "results/phase3/extraction_check_phase3_dev.json").read_text(encoding="utf-8"))
    if not chk.get("passed"):
        raise SystemExit("the DEV+ extraction check did not pass")
    fz = json.loads((root / "results/gate2/frozen_selection.json").read_text(encoding="utf-8"))
    for key, variant in (("P", "hx"), ("Q", "x")):
        if fz[key]["config"]["variant"] != variant or fz[key]["config"]["family"] != 1:
            raise SystemExit(f"Gate 2 frozen {key} is not family-1 {variant}")
        Cs = {v["C"] for v in fz[key]["config"]["hyper"].values()}
        if Cs != {G.FROZEN_C[variant]}:
            raise SystemExit(f"Gate 2 frozen C for {variant} is {Cs}, expected {G.FROZEN_C[variant]}")
    out["results/gate2/frozen_selection.json"] = g1.sha256_file(root / "results/gate2/frozen_selection.json")
    sec = json.loads((root / "results/gate2/validation_report.json").read_text(encoding="utf-8"))["secondary"]
    for (target, v), Cs in G.GATE2_DIRECTION_C.items():
        rec = tuple(sec[f"{target}/{v}"]["hyper"][str(t)]["C"] for t in range(3))
        if rec != tuple(Cs):
            raise SystemExit(f"Gate 2 recorded C for {target}/{v} is {rec}, expected {Cs}")
    out["results/gate2/validation_report.json"] = g1.sha256_file(root / "results/gate2/validation_report.json")
    return out


def code_identity(root: Path) -> dict:
    files = ["src/lookahead/phase3_g3a.py", "scripts/phase3_g3a.py", "src/lookahead/phase3_readouts.py",
             "src/lookahead/gate1.py", "src/lookahead/gate2.py", "src/lookahead/phase3_build.py"]
    out = {f: _sha(root / f) for f in files}
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
    d1, d2, d3 = rep["D1"], rep["D2"], rep["D3"]
    lines = [f"D1  ceiling(ii) {d1['primary']['ceiling_ii']:.4f} - B_pb {d1['primary']['B_pb']:.4f} = "
             f"{d1['primary']['headroom_ii']:.4f}  (stop if < {G.D1_MIN})  -> {'STOP' if d1['stop'] else 'pass'}"]
    for c, v in d2["candidates"].items():
        lines.append(f"D2  [gross-saving screen] {c}: nested improvement over the A2 comparator {v['improvement_vs_comparator']:+.4f} "
                     f"(vs B_pb {v['improvement_vs_B_pb']:+.4f})")
    lines.append(f"D2  comparators per fold {d2['comparator_per_fold']}  (stop if both < {G.D2_MIN}) -> "
                 f"{'STOP' if d2['stop'] else 'pass'}")
    lines.append(f"D3  ({d3['decision_method']}) h > x in {d3['n_h_better']} of {d3['n_counted']} counted cells -> hx variants "
                 f"{'kept' if d3['keep_hx'] else 'dropped'}")
    lines.append(f"DECISION: {rep['decision']}")
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

        dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
        val = synth(n=50, seed=12, source="p1_validation", prefix="v/")
        out_path, inputs = Path(args.synthetic), {"synthetic": True}
    else:
        out_path = root / REPORT
        if out_path.exists():
            print(f"refusing: {REPORT} already exists", file=sys.stderr)
            return 2
        inputs = check_frozen(root)
        dev, val = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=True)
        inputs["data_files"] = {**dev.provenance, **val.provenance}
    if out_path.exists():
        print(f"refusing: {out_path} already exists", file=sys.stderr)
        return 2
    rep = G.run_g3a(dev, val)
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
