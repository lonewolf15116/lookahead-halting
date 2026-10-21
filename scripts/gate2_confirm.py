"""Gate 2: guarded ONE-TIME confirmation evaluation of the frozen policies (plan v3 section 7 + amendment v3.1).

Real run (only after the reviewer approves; refuses to repeat):
    python scripts/gate2_confirm.py --confirm-one-time-gate2-evaluation --frozen-sha <12+ hex of frozen_selection.json> \
        --baseline-sha <12+ hex of baseline_B_frozen.json> --reviewer-approval-ref "<who/when/where>"

Rehearsal (touches NO confirmation data; runs the same code on VALIDATION and checks that every frozen validation number
is reproduced):
    python scripts/gate2_confirm.py --rehearsal-on-validation

The real run: pins the frozen files by sha256 prefix (typed by the person running it); refuses if a STARTED or EVALUATED
marker exists (there is no override: a repeat would need a new commit and a reviewer decision); refuses with any modified
tracked file (no override); requires the shuffled control; BEFORE opening the confirmation set it reloads train and
validation, refits the shuffled control deterministically on train only, and checks that P, Q and the control reproduce
their frozen/reported validation saving and loss at the frozen thresholds (exit 3 otherwise); then writes the STARTED
marker, reads only the frozen files, applies thresholds exactly (never reselects), reports the original section 7 verdict
first and the v3.1 analyses separately, and writes the EVALUATED marker last.
"""
import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lookahead import gate1 as g
from lookahead import gate2 as g2
from lookahead import gate2_eval as ev

B_APPROVED = (2, 2, 2, 3, 3, 2)
PLAN_VERSION = "v3"
MIN_SHA = 12


def git_state(root: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    modified = [ln for ln in run("status", "--porcelain").splitlines() if not ln.startswith("??")]
    return {"head": run("rev-parse", "HEAD"), "modified_tracked": modified}


def check_prefix(name: str, given: str | None, actual: str) -> None:
    if not given or len(given) < MIN_SHA or not actual.startswith(given.lower()):
        raise SystemExit(f"refusing: --{name} must be at least {MIN_SHA} hex characters matching the file's sha256 "
                         f"(file is {actual[:12]}...)")


def load_frozen(root: Path):
    fpath, bpath = root / "results/gate2/frozen_selection.json", root / "results/gate2/baseline_B_frozen.json"
    frozen_raw, base_raw = fpath.read_bytes(), bpath.read_bytes()
    frozen, base = json.loads(frozen_raw), json.loads(base_raw)
    sha_f, sha_b = g.sha256_bytes(frozen_raw), g.sha256_bytes(base_raw)
    if frozen["plan_version"] != PLAN_VERSION:
        raise SystemExit(f"frozen plan_version {frozen['plan_version']} != {PLAN_VERSION}")
    if tuple(base["B"]["depths"]) != B_APPROVED:
        raise SystemExit(f"B vector {base['B']['depths']} is not the reviewer-approved {list(B_APPROVED)}")
    if base["benchmark_order"] != frozen["benchmarks"] or list(frozen["benchmarks"]) != list(g.GATE1_BENCHMARKS):
        raise SystemExit("benchmark order mismatch between the frozen files")
    if base["frozen_selection_sha256"] != sha_f:
        raise SystemExit("baseline_B_frozen.json was made against a different frozen_selection.json")
    if not base.get("constructions_agree"):
        raise SystemExit("baseline B constructions did not agree at freeze time")
    for key in ("model_file", "standardisation_file"):
        for who in ("P", "Q"):
            if "confirm" in str(frozen[who][key]).lower():
                raise SystemExit(f"{who}.{key} refers to confirmation data")
    return frozen, base, sha_f, sha_b


def rehearsal_checks(root, frozen, base, benchmarks, analysis, ps_eval, gclip) -> list:
    """On validation the evaluation must reproduce every frozen validation number. Returns a list of failures."""
    bad = []
    pt = analysis["point"]

    def close(name, got, want, tol=1e-9):
        if want is None or abs(got - want) > tol:
            bad.append(f"{name}: got {got!r}, frozen {want!r}")

    close("P saving", pt["P"]["macro_saving"], frozen["P"]["validation_saving"])
    close("P loss", pt["P"]["macro_loss"], frozen["P"]["validation_loss"])
    close("Q saving", pt["Q"]["macro_saving"], frozen["Q"]["validation_saving"])
    close("Q loss", pt["Q"]["macro_loss"], frozen["Q"]["validation_loss"])
    for n in g.CHEAP_BASELINES:
        if frozen["baselines"][n]["qualified"]:
            close(f"{n} saving", pt[n]["macro_saving"], frozen["baselines"][n]["validation_saving"])
            close(f"{n} loss", pt[n]["macro_loss"], frozen["baselines"][n]["validation_loss"])
    close("B saving", pt["B"]["macro_saving"], base["B"]["validation_macro_saving"])
    close("B loss", pt["B"]["macro_loss"], base["B"]["validation_macro_loss"])
    close("B_cap saving", pt["B_cap"]["macro_saving"], base["sensitivity_per_benchmark_cap"]["macro_saving"])
    ref = g2.load_probe_data(root, "validation", benchmarks, g_clip=gclip)
    same = (np.array_equal(ref.doc_id, ps_eval.doc_id) and np.array_equal(ref.h, ps_eval.h)
            and np.array_equal(ref.x, ps_eval.x) and np.array_equal(ref.ok, ps_eval.ok)
            and all(np.array_equal(ref.targets[k], ps_eval.targets[k]) for k in ref.targets))
    if not same:
        bad.append("eval_loader differs from gate2.load_probe_data on validation")
    return bad


def validation_preflight(root, benchmarks, gclip, entries) -> list:
    """Before the confirmation set is opened: each (name, probe, threshold, saving, loss) must reproduce its frozen/reported
    validation macro saving and loss at its fixed threshold. Reads train/validation data only."""
    val = g2.load_probe_data(root, "validation", benchmarks, g_clip=gclip)
    bad = []
    for name, probe, thr, want_s, want_l in entries:
        m = g.summarize(g2.probe_policy_data(val, probe.predict(val)), benchmarks, "native_gate", thr)["macro"]
        for what, got, want in (("saving", m["saving"], want_s), ("loss", m["loss"], want_l)):
            if want is None or abs(got - want) > 1e-9:
                bad.append(f"{name} {what}: got {got!r}, frozen/reported {want!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--confirm-one-time-gate2-evaluation", action="store_true")
    ap.add_argument("--rehearsal-on-validation", action="store_true")
    ap.add_argument("--frozen-sha")
    ap.add_argument("--baseline-sha")
    ap.add_argument("--reviewer-approval-ref")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--n-boot", type=int, default=g.N_BOOT)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    real = args.confirm_one_time_gate2_evaluation
    if real == args.rehearsal_on_validation:
        print("refusing: pass exactly one of --confirm-one-time-gate2-evaluation or --rehearsal-on-validation",
              file=sys.stderr)
        return 2

    frozen, base, sha_f, sha_b = load_frozen(root)
    benchmarks = list(frozen["benchmarks"])
    out_dir = root / "results" / ("gate2" if real else "gate2_rehearsal")
    started, evaluated = root / "results/gate2/CONFIRMATION_STARTED.json", root / "results/gate2/CONFIRMATION_EVALUATED.json"
    git = git_state(root)
    if real and args.skip_control:
        print("refusing: the real run requires the shuffled control (no --skip-control)", file=sys.stderr)
        return 2
    if real:
        check_prefix("frozen-sha", args.frozen_sha, sha_f)
        check_prefix("baseline-sha", args.baseline_sha, sha_b)
        if not args.reviewer_approval_ref or len(args.reviewer_approval_ref.strip()) < 8:
            print("refusing: --reviewer-approval-ref must say where/when the reviewer approved this run", file=sys.stderr)
            return 2
        if started.exists() or evaluated.exists():
            print("refusing: a STARTED/EVALUATED marker exists (the confirmation set was already opened); there is no "
                  "override", file=sys.stderr)
            return 2
        if git["modified_tracked"]:
            print("refusing: tracked files are modified (no override):\n  " + "\n  ".join(git["modified_tracked"]),
                  file=sys.stderr)
            return 2

    gclip = tuple(frozen["provenance"]["g_clip_train"])
    P = ev.load_frozen_probe(root, frozen["P"])
    Q = ev.load_frozen_probe(root, frozen["Q"])
    control_thr, shuf = None, None
    if not args.skip_control:
        vr = json.loads((root / "results/gate2/validation_report.json").read_text(encoding="utf-8"))
        rec = [r for r in vr["control_records"] if r["name"].startswith("control-c")][0]
        control_thr = rec["threshold"]
        train = g2.load_probe_data(root, "train", benchmarks)          # training data only
        shuf = g2.fit_probe(train, 1, "h", "S", labels=g2.shuffle_labels_within(train, train.targets["S"]))
    if shuf is not None:
        entries = [("P", P, frozen["P"]["threshold"], frozen["P"]["validation_saving"], frozen["P"]["validation_loss"]),
                   ("Q", Q, frozen["Q"]["threshold"], frozen["Q"]["validation_saving"], frozen["Q"]["validation_loss"]),
                   ("shuffled_control", shuf, control_thr, rec["saving"], rec["loss"])]
        bad = validation_preflight(root, benchmarks, gclip, entries)
        if bad:
            print("PRE-FLIGHT FAILED; the confirmation set was NOT opened:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 3
        print("pre-flight: P, Q and the refitted shuffled control reproduce their frozen validation numbers", flush=True)
    if real:
        stamp0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        g.write_json(started, {"timestamp_utc": stamp0, "frozen_selection_sha256": sha_f, "baseline_B_sha256": sha_b,
                               "git": git, "reviewer_approval_ref": args.reviewer_approval_ref,
                               "note": "written BEFORE any confirmation outcome is opened"})
        print("STARTED marker written; opening the confirmation set", flush=True)

    subdir, split = ("extracted_confirm", "confirmation") if real else ("extracted", "validation")
    prov = {}
    ctx = ev.allow_confirmation_paths() if real else contextlib.nullcontext()
    with ctx:
        ps = ev.eval_loader(root, benchmarks, subdir, split, gclip, provenance=prov)
    print(f"loaded {ps.N} questions from data/{subdir} (split {split})", flush=True)

    p_sc, q_sc = P.predict(ps), Q.predict(ps)
    datasets = {"base": ps.arrays, "P": g2.probe_policy_data(ps, p_sc), "Q": g2.probe_policy_data(ps, q_sc)}
    scores = {"P": p_sc, "Q": q_sc}
    if shuf is not None:
        c_sc = shuf.predict(ps)
        datasets["control"] = g2.probe_policy_data(ps, c_sc)
        scores["shuffled_control"] = c_sc

    pols = ev.build_policies(frozen, base["B"]["depths"], base["sensitivity_per_benchmark_cap"]["depths"], benchmarks,
                             control_thr)
    C, E = ev.outcomes(pols, datasets, benchmarks)
    analysis = ev.analyse(pols, C, E, benchmarks, args.n_boot, g.SEED)
    auroc = ev.auroc_report(ps, scores)

    if not real:
        bad = rehearsal_checks(root, frozen, base, benchmarks, analysis, ps, gclip)
        if bad:
            print("REHEARSAL FAILED; the pipeline does not reproduce the frozen validation numbers:\n  " + "\n  ".join(bad),
                  file=sys.stderr)
            return 3
        print("rehearsal: every frozen validation number (P, Q, baselines, B, B_cap) and the loader are reproduced")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    banner = ("**REHEARSAL on VALIDATION data. This is not the Gate 2 verdict.** It checks the pipeline against the frozen "
              "validation numbers." if not real else
              "Gate 2 confirmation: frozen policies applied once to the never-scored confirmation set.")
    report = {"label": "gate2 confirmation" if real else "gate2 rehearsal on validation (not a verdict)",
              "timestamp_utc": stamp, "benchmarks": benchmarks, "banner": banner, "split": split,
              "frozen_selection_sha256": sha_f, "baseline_B_sha256": sha_b, "git": git,
              "n_failed": ps.n_failed, "input_hashes": prov, "auroc": auroc, "analysis": analysis,
              "code": g.code_provenance()}
    g.write_json(out_dir / ("gate2_confirmation_report.json" if real else "rehearsal_report.json"), report)
    md = ev.render(report, "Gate 2 report (confirmation, frozen selection)" if real else "Gate 2 rehearsal (validation)")
    doc = root / "docs" / ("GATE2_REPORT.md" if real else "GATE2_REHEARSAL.md")
    doc.write_text(md, encoding="utf-8")
    if real:
        g.write_json(evaluated, {"timestamp_utc": stamp, "frozen_selection_sha256": sha_f, "baseline_B_sha256": sha_b,
                                 "verdict": analysis["original"]["verdict"]})
    o = analysis["original"]
    print(f"{'ORIGINAL VERDICT' if real else 'rehearsal (not a verdict)'}: {o['verdict']}  L={o['L_budget']} E={o['E_effect']} "
          f"H={o['H_hidden_state']}  P saving {o['P_macro_saving']:.4f} loss {o['P_macro_loss']:.3f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
