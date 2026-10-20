"""Gate 1, step 1: threshold selection and headroom on the VALIDATION split only (not the Gate 1 verdict).

    python scripts/gate1_validation.py [--root PATH] [--n-boot 2000]

Reads only validation rows (test rows are filtered out of every part as it is read). Writes
results/gate1/frozen_selection.json, results/gate1/validation_report.json and docs/GATE1_VALIDATION_REPORT.md.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from lookahead import gate1 as g
from lookahead.config import REPO_ROOT

SPLIT = "validation"
LABEL = "validation (not the Gate 1 verdict)"


def render(report: dict, sel_p: dict, sel_s: dict) -> str:
    p, s = report["primary"], report["sensitivity"]
    out = [f"# Gate 1 validation report ({LABEL})\n",
           f"Generated {report['timestamp_utc']}. All numbers below are from the **validation** split only. "
           "No test-split outcome has been read: test rows were filtered out of every parquet part at read time.\n",
           "**S_or is the final-answer-preserving headroom** (earliest loop after which the answer never changes before "
           "loop 4), **not the maximum possible accuracy-preserving saving**: a policy could stop at an earlier answer that "
           "differs from the final one and still match fixed-4 accuracy. The earliest-correct oracle (reference row below) "
           "is a different, label-using reference: it raises accuracy but runs never-correct questions to loop 4, so its "
           "saving can be lower than S_or.\n",
           "Macro = unweighted mean over benchmarks; question-weighted numbers are secondary (in the JSON report).\n",
           "## Selection rules\n", sel_p["rules"] + "\n",
           f"## Primary ({', '.join(p['benchmarks'])})\n",
           "### Qualification and frozen thresholds (validation)\n", g.md_selection(sel_p),
           g.md_evaluation(p["evaluation"], f"Headroom on validation, primary - {LABEL}"),
           "### Reference rows, wall-clock (mean ms/question) and per-policy numbers\n",
           g.md_reference(p["evaluation"], sel_p),
           "### Transition tables (validation)\n", g.md_transitions(p["transitions"]),
           f"## Sensitivity (no openbookqa: {', '.join(s['benchmarks'])})\n",
           "### Qualification and frozen thresholds (validation)\n", g.md_selection(sel_s),
           g.md_evaluation(s["evaluation"], f"Headroom on validation, sensitivity - {LABEL}"),
           "### Reference rows and wall-clock\n", g.md_reference(s["evaluation"], sel_s)]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--n-boot", type=int, default=g.N_BOOT)
    args = ap.parse_args(argv)
    root = Path(args.root)
    out_dir = root / "results" / "gate1"
    if (out_dir / "TEST_EVALUATED.json").exists():
        print("refusing: TEST_EVALUATED.json exists; the frozen selection must not be rewritten", file=sys.stderr)
        return 2

    prov_parts = {}
    data = g.load_split(root, g.GATE1_BENCHMARKS, SPLIT, provenance=prov_parts)
    for b, d in data.items():
        assert (d["split"] == SPLIT).all(), f"{b}: loaded a row whose split is not {SPLIT}"

    splits_csv = root / "data" / "manifests" / "splits.csv"
    if not splits_csv.exists():
        raise FileNotFoundError(splits_csv)
    stamp = datetime.now(timezone.utc).isoformat()
    sel_p = g.run_selection(data, g.GATE1_BENCHMARKS)
    sel_s = g.run_selection(data, g.SENSITIVITY_BENCHMARKS)
    fr_p, fr_s = g.freeze(sel_p), g.freeze(sel_s)
    ev_p = g.analyze(data, g.GATE1_BENCHMARKS, fr_p["baselines"], args.n_boot)
    ev_s = g.analyze(data, g.SENSITIVITY_BENCHMARKS, fr_s["baselines"], args.n_boot)
    tr = g.transition_tables(data, g.GATE1_BENCHMARKS, args.n_boot)

    provenance = {"parquet_part_sha256": prov_parts, "splits_csv_sha256": g.sha256_file(splits_csv),
                  "code_sha256": g.code_provenance(), "timestamp_utc": stamp}
    frozen = {"label": "frozen on validation only", "rules": g.SELECTION_RULES, "primary": fr_p, "sensitivity": fr_s,
              "provenance": provenance}
    report = {"label": LABEL, "timestamp_utc": stamp, "split": SPLIT, "provenance": provenance,
              "primary": {"benchmarks": list(g.GATE1_BENCHMARKS), "selection": sel_p, "evaluation": ev_p,
                          "transitions": tr},
              "sensitivity": {"benchmarks": list(g.SENSITIVITY_BENCHMARKS), "selection": sel_s, "evaluation": ev_s}}
    g.write_json(out_dir / "frozen_selection.json", frozen)
    g.write_json(out_dir / "validation_report.json", report)
    doc = root / "docs" / "GATE1_VALIDATION_REPORT.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(report, sel_p, sel_s), encoding="utf-8")
    print(f"[{LABEL}] primary S_or={ev_p['S_or']:.4f} R={ev_p['R']:.4f} -> {ev_p['label']}")
    print(f"[{LABEL}] sensitivity S_or={ev_s['S_or']:.4f} R={ev_s['R']:.4f} -> {ev_s['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
