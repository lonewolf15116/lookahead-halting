"""D-t2: GPU determinism check (docs/DT_TRUNCATION_RESULT.md, reviewer GO 2026-10-27). Diagnostic [E]; development data only.

    python scripts/dt2_determinism_check.py --mode default
    python scripts/dt2_determinism_check.py --mode deterministic     # CUBLAS_WORKSPACE_CONFIG must be set BEFORE this process starts
    python scripts/dt2_determinism_check.py --summarize

Frozen procedure:
- Questions: the first 20 HellaSwag doc_ids of the D-t selection order (sha256 of "lh-dt-v1:" + doc_id, ascending).
- Per mode, in its own process: 2 untimed warm-up runs, then each question 5 times at full depth (4 loops) through the same per-loop
  bookkeeping as D-t (dt_truncation.run_full / loop_record).
- Compared bit for bit, repeats 2-5 against repeat 1 of the same question: option log-likelihoods, conf, ent, vent, gate logit and
  CDF at every loop. A question "differs" if any value at any loop differs in any bit.
- Deterministic mode: CUBLAS_WORKSPACE_CONFIG must already be ":4096:8" when the process starts (checked before torch is imported),
  then torch.use_deterministic_algorithms(True) (warn_only=False). If it raises, for example because a kernel has no deterministic
  implementation, the exception is recorded as the outcome. Settings are never changed silently.
- Outputs, each created exclusively and never overwritten: results/diagnostics/dt2_default.json, dt2_deterministic.json and dt2_summary.json.
- Summary reading (reviewer table): default differs & deterministic agrees -> evidence supports GPU nondeterminism as a cause;
  both differ -> deterministic settings did not eliminate the differences; both agree -> nondeterminism was not reproduced in this
  diagnostic (it is NOT ruled out); default agrees & deterministic differs -> investigate configuration or execution-path effects.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = {"default": "results/diagnostics/dt2_default.json", "deterministic": "results/diagnostics/dt2_deterministic.json",
       "summary": "results/diagnostics/dt2_summary.json"}
N_Q, N_REP, BENCH, CUBLAS = 20, 5, "hellaswag", ":4096:8"
KEYS = ("opt_ll", "conf", "ent", "vent", "gate", "cdf")


def _write_x(rel: str, obj: dict) -> Path:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8", newline="\n") as f:          # exclusive create: never overwrites
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    return p


def values(trace) -> dict:
    """{key: float64 array [loops, ...]} from a dt_truncation.Trace."""
    import numpy as np

    out = {k: np.array([r[k] for r in trace.rows], dtype=np.float64) for k in KEYS}
    return out


def bitwise_diff(a: dict, b: dict) -> dict:
    """Per key: (any bit differs, max |a - b|)."""
    import numpy as np

    res = {}
    for k in KEYS:
        x, y = np.ascontiguousarray(a[k]), np.ascontiguousarray(b[k])
        same = x.shape == y.shape and bool((x.view(np.int64) == y.view(np.int64)).all())
        res[k] = {"differs": not same, "max_abs": 0.0 if same else float(np.nanmax(np.abs(x - y)))}
    return res


def run_mode(mode: str) -> int:
    rel = OUT[mode]
    if (ROOT / rel).exists():
        print(f"refusing: {rel} exists (never overwritten)", file=sys.stderr)
        return 2
    env_before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if mode == "deterministic" and env_before != CUBLAS:
        print(f"refusing: CUBLAS_WORKSPACE_CONFIG must be {CUBLAS} before the process starts (got {env_before!r})", file=sys.stderr)
        return 2
    if mode == "default" and env_before is not None:
        print(f"refusing: default mode must run without CUBLAS_WORKSPACE_CONFIG (got {env_before!r})", file=sys.stderr)
        return 2
    import importlib.util

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from lookahead import dt_truncation as D

    spec = importlib.util.spec_from_file_location("dtc", ROOT / "scripts" / "dt_truncation_check.py")
    dtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dtc)
    D.check_pinned(ROOT)
    rep = {"mode": mode, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "cublas_workspace_config": env_before, "plan": "D-t2 (docs/DT_TRUNCATION_RESULT.md)", "n_questions": N_Q,
           "n_repeats": N_REP, "git_commit": dtc.git_commit(ROOT)}
    det_error = None
    if mode == "deterministic":
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception as e:                                       # recorded, never worked around
            det_error = f"{type(e).__name__}: {e}"
    dev, qs, _ = dtc.real_loader(ROOT)
    ids = D.select_ids(dev, D.N_PER_BENCH)[BENCH][:N_Q]
    sub = D.G.subset(dev, np.isin(dev.doc_id, ids) & (dev.source == "devplus"))
    qmap = D.match_questions(sub, {BENCH: ids}, qs)
    scorer, model = dtc.real_scorer("cuda")
    rep["environment"] = dtc.environment(model)
    rep["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    rep["deterministic_setup_error"] = det_error
    rep["selected_ids"] = ids
    sync = torch.cuda.synchronize
    import time

    per_q, run_error = [], None
    try:
        for _ in range(2):
            D.run_full(scorer, qmap[ids[0]], BENCH, ids[0], sync, time.perf_counter)
        stored = D.load_stored_trace(ROOT, BENCH, ids)
        for i in ids:
            reps = [values(D.run_full(scorer, qmap[i], BENCH, i, sync, time.perf_counter)[0]) for _ in range(N_REP)]
            diffs = [bitwise_diff(reps[0], r) for r in reps[1:]]
            per_q.append({"id": i,
                          "repeats_differ": any(d[k]["differs"] for d in diffs for k in KEYS),
                          "by_key": {k: {"n_repeats_differ": sum(d[k]["differs"] for d in diffs),
                                         "max_abs": max(d[k]["max_abs"] for d in diffs)} for k in KEYS},
                          "repeat1_vs_stored_opt_ll_max_abs": float(max(
                              np.abs(np.asarray(reps[0]["opt_ll"])[t] - np.asarray(stored[i]["opt_ll"][t])).max()
                              for t in range(len(reps[0]["opt_ll"]))))})
    except Exception as e:                                           # e.g. a kernel without a deterministic implementation
        run_error = f"{type(e).__name__}: {e}"
    rep["run_error"] = run_error
    rep["per_question"] = per_q
    rep["n_questions_completed"] = len(per_q)
    rep["n_questions_repeats_differ"] = sum(q["repeats_differ"] for q in per_q)
    rep["max_abs_by_key"] = {k: max((q["by_key"][k]["max_abs"] for q in per_q), default=None) for k in KEYS}
    rep["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = _write_x(rel, rep)
    print(f"{mode}: deterministic enabled={rep['deterministic_algorithms_enabled']} setup_error={det_error} run_error={run_error}")
    print(f"{mode}: {rep['n_questions_repeats_differ']}/{len(per_q)} questions with any bit difference across {N_REP} repeats; "
          f"max |diff| {rep['max_abs_by_key']}")
    print(f"report: {p}")
    return 0


def summarize() -> int:
    if (ROOT / OUT["summary"]).exists():
        print(f"refusing: {OUT['summary']} exists", file=sys.stderr)
        return 2
    r = {m: json.loads((ROOT / OUT[m]).read_text(encoding="utf-8")) for m in ("default", "deterministic")}
    ok = {m: r[m]["run_error"] is None and r[m]["deterministic_setup_error"] is None and r[m]["n_questions_completed"] == N_Q
          for m in r}
    differ = {m: r[m]["n_questions_repeats_differ"] > 0 for m in r}
    if not all(ok.values()):
        reading = "INCOMPLETE: a mode did not complete (setup or run error recorded); report as a diagnostic outcome"
    elif differ["default"] and not differ["deterministic"]:
        reading = "default differs, deterministic agrees: evidence supports GPU nondeterminism as a cause"
    elif differ["default"] and differ["deterministic"]:
        reading = "both differ: deterministic settings did not eliminate the differences"
    elif not differ["default"] and not differ["deterministic"]:
        reading = "both agree: nondeterminism was not reproduced in this diagnostic; it is NOT ruled out"
    else:
        reading = "default agrees, deterministic differs: investigate configuration or execution-path effects"
    s = {"reading": reading, "completed": ok, "questions_differ": {m: r[m]["n_questions_repeats_differ"] for m in r},
         "max_abs_by_key": {m: r[m]["max_abs_by_key"] for m in r},
         "deterministic_algorithms_enabled": {m: r[m]["deterministic_algorithms_enabled"] for m in r},
         "errors": {m: {"setup": r[m]["deterministic_setup_error"], "run": r[m]["run_error"]} for m in r}}
    _write_x(OUT["summary"], s)
    print("D-t2 reading:", reading)
    print(json.dumps({k: s[k] for k in ("questions_differ", "completed", "errors")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=("default", "deterministic"))
    g.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    return summarize() if a.summarize else run_mode(a.mode)


if __name__ == "__main__":
    sys.exit(main())
