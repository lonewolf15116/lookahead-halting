"""Gate 1, step 2: the one-time TEST evaluation with the frozen selection. Do not run until validation is final.

    python scripts/gate1_test.py --confirm-one-time-test-evaluation [--allow-rerun] [--root PATH]

Reads only results/gate1/frozen_selection.json (thresholds are used exactly, never reselected) and the test split.
Writes results/gate1/gate1_test_report.json, docs/GATE1_REPORT.md and, last, results/gate1/TEST_EVALUATED.json.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lookahead import gate1 as g
from lookahead.config import REPO_ROOT

SPLIT = "test"


def evaluate_one(data, frozen_set: dict, n_boot: int) -> dict:
    benches, baselines = frozen_set["benchmarks"], frozen_set["baselines"]
    ev = g.analyze(data, benches, baselines, n_boot)
    ev["R_unc"] = g.r_unc(data, benches, baselines, ev["S_or"])
    ev["constraint_violations"] = [n for n, b in ev["baselines"].items() if b["qualified"] and b["constraint_violation"]]
    ev["frozen_thresholds_used"] = {n: baselines[n]["threshold"] for n in g.CHEAP_BASELINES}
    return ev


def render(report: dict) -> str:
    p, s = report["primary"], report["sensitivity"]
    out = ["# Gate 1 report (test split, frozen selection)\n",
           f"Generated {report['timestamp_utc']}. Thresholds frozen on validation (frozen_selection.json sha256 "
           f"`{report['frozen_selection_sha256']}`); nothing was reselected on test.\n"]
    if report["rerun"]["is_rerun"]:
        out.append("**This is a RERUN (--allow-rerun); a previous test evaluation exists.**\n")
    out.append("**S_or is the final-answer-preserving headroom, not the maximum possible accuracy-preserving saving.**\n")
    out.append(f"## Gate 1 verdict (primary): **{p['label']}**\n")
    out.append(g.md_evaluation(p["evaluation"], "Primary (six benchmarks)"))
    if p["evaluation"]["constraint_violations"]:
        out.append(f"**Constraint violation (test macro loss > 1 pp):** {', '.join(p['evaluation']['constraint_violations'])}\n")
    ru = p["evaluation"]["R_unc"]
    out.append(f"- R_unc (diagnostic, best test-feasible grid point per baseline): {g.f(ru['R_unc'])} ({ru['argmax']})\n")
    out.append(g.md_reference(p["evaluation"]))
    out.append("### Transition tables (test)\n")
    out.append(g.md_transitions(report["transitions"]))
    out.append(f"## Sensitivity verdict (no openbookqa): **{s['label']}**\n")
    out.append(g.md_evaluation(s["evaluation"], "Sensitivity (five benchmarks)"))
    if s["evaluation"]["constraint_violations"]:
        out.append(f"**Constraint violation (test macro loss > 1 pp):** {', '.join(s['evaluation']['constraint_violations'])}\n")
    ru = s["evaluation"]["R_unc"]
    out.append(f"- R_unc (diagnostic): {g.f(ru['R_unc'])} ({ru['argmax']})\n")
    out.append(g.md_reference(s["evaluation"]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--confirm-one-time-test-evaluation", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--n-boot", type=int, default=g.N_BOOT)
    args = ap.parse_args(argv)
    if not args.confirm_one_time_test_evaluation:
        print("refusing: pass --confirm-one-time-test-evaluation to evaluate the test split (one time only)",
              file=sys.stderr)
        return 2
    root = Path(args.root)
    out_dir = root / "results" / "gate1"
    marker = out_dir / "TEST_EVALUATED.json"
    previous = None
    if marker.exists():
        if not args.allow_rerun:
            print("refusing: results/gate1/TEST_EVALUATED.json exists (use --allow-rerun to record a rerun)",
                  file=sys.stderr)
            return 2
        previous = json.loads(marker.read_text(encoding="utf-8"))

    frozen_path = out_dir / "frozen_selection.json"
    raw = frozen_path.read_bytes()
    frozen_sha = g.sha256_bytes(raw)
    frozen = json.loads(raw.decode("utf-8"))
    for key in ("primary", "sensitivity"):
        if set(frozen[key]["baselines"]) != set(g.CHEAP_BASELINES):
            raise ValueError(f"frozen {key} baselines are malformed")

    benches = list(frozen["primary"]["benchmarks"])
    if benches != list(g.GATE1_BENCHMARKS) or list(frozen["sensitivity"]["benchmarks"]) != list(g.SENSITIVITY_BENCHMARKS):
        raise ValueError("frozen benchmark lists differ from the Gate 1 sets")
    prov = {}
    data = g.load_split(root, benches, SPLIT, provenance=prov)
    for b, d in data.items():
        assert (d["split"] == SPLIT).all()

    stamp = datetime.now(timezone.utc).isoformat()
    ev_p = evaluate_one(data, frozen["primary"], args.n_boot)
    ev_s = evaluate_one(data, frozen["sensitivity"], args.n_boot)
    tr = g.transition_tables(data, benches, args.n_boot)
    code = g.code_provenance()
    frozen_code = frozen.get("provenance", {}).get("code_sha256", {})
    report = {"label": "Gate 1 test evaluation (frozen thresholds)", "timestamp_utc": stamp, "split": SPLIT,
              "frozen_selection_sha256": frozen_sha,
              "rerun": {"is_rerun": previous is not None, "previous_test_evaluated": previous},
              "provenance": {"test_parquet_part_sha256": prov, "code_sha256": code,
                             "code_changed_since_freeze": {k: code.get(k) != frozen_code.get(k) for k in code}},
              "primary": {"benchmarks": benches, "evaluation": ev_p, "label": ev_p["label"]},
              "sensitivity": {"benchmarks": list(frozen["sensitivity"]["benchmarks"]), "evaluation": ev_s,
                              "label": ev_s["label"]},
              "transitions": tr}
    g.write_json(out_dir / "gate1_test_report.json", report)
    doc = root / "docs" / "GATE1_REPORT.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(report), encoding="utf-8")
    g.write_json(marker, {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "frozen_selection_sha256": frozen_sha,
                          "is_rerun": previous is not None, "previous": previous})
    print(f"Gate 1 primary: S_or={ev_p['S_or']:.4f} R={ev_p['R']:.4f} -> {ev_p['label']}")
    print(f"Gate 1 sensitivity: S_or={ev_s['S_or']:.4f} R={ev_s['R']:.4f} -> {ev_s['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
