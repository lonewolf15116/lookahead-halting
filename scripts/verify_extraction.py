"""Acceptance checks for a Phase 1 extraction (reviewer, 2026-10-20), run on the stored files.

    python scripts/verify_extraction.py --out data/extracted_smoke --limit 40
    python scripts/verify_extraction.py                         # data/extracted, everything

1 schema       every question once, 4 loop rows with every field, hidden [n, 4, hidden_size]
2 targets      every option-level value and target recomputed from the stored option log-likelihoods with independent
               code; on a sample per benchmark, the same packed input is run again and the full-vocabulary entropy and KL
               targets, gate and state recomputed with separate code (must reproduce the stored values); the bf16 gap to
               a context-only pass is reported, and the float32 equivalence from diagnose_short_context.py is required
3 numerics     finite values, probabilities sum to 1, KL >= 0, entropies in range; failures classified, <= 1% each
4 splits       every question has exactly one split, equal to splits.csv
5 resume       part checksums match, no duplicate questions, no leftover temporary files
6 reproducible meta.json records code, model and dataset revisions, seeds, versions, split checksum

Exit code 0 only if every check passes. Writes results/phase1/extraction_check[_smoke].json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from lookahead.config import REPO_ROOT, results_dir
from lookahead.extract import LOOPS, load_questions, load_splits, read_benchmark
from lookahead.questions import BENCHMARKS, SYNTHETIC

ALL = list(BENCHMARKS) + list(SYNTHETIC)
FIELDS = ["benchmark", "stratum", "doc_id", "split", "target", "n_options", "n_tokens", "metric", "status", "loop",
          "opt_ll", "opt_lln", "p", "pred", "correct", "conf", "ent", "vent", "gate", "cdf", "ms", "loss",
          "D", "D_vocab", "F", "T", "G", "K", "K_vocab", "F_final", "T_final"]
META_KEYS = ["code_sha256", "git_commit", "model", "datasets", "splits_sha256", "seeds", "versions", "device"]
TOL = 1e-6  # option-level recomputation is float64 on the same stored inputs
RERUN_TOL = 1e-4  # re-running the same packed input: same shapes, same kernels; fixed 2026-10-20 before any such run
RERUN_HIDDEN_TOL = 1e-3  # stored state is float16 of a bf16 value, so the re-run must reproduce it to well under 1 ulp
FP32_TOL = 1e-4  # float32 packed vs context-only (scripts/diagnose_short_context.py); measured ~2e-5


def recompute_options(opt_ll, choice_lens, metric, target):
    """Independent re-derivation of p_t and the option-level values for one question's loops."""
    out = []
    for ll in opt_ll:
        ll = np.array(ll, dtype=np.float64)
        s = ll / np.array(choice_lens, dtype=np.float64) if metric == "acc_norm" else ll
        z = s - s.max()
        logp = z - np.log(np.sum(np.exp(z)))
        p = np.exp(logp)
        pred = int(np.argmax(s))
        out.append({"logp": logp, "p": p, "pred": pred, "correct": int(pred == target), "conf": p.max(),
                    "ent": -np.sum(p * logp), "loss": -logp[target]})
    return out


def kl(a, b):
    return float(np.sum(np.exp(a) * (a - b)))


def label(c0, c1):
    return ("C" if c0 else "W") + ("C" if c1 else "W")


def check_targets(q, rows):
    """Max abs error per quantity between stored and recomputed values for one ok question."""
    rec = recompute_options([r["opt_ll"] for r in rows], [len(c) for c in q.choice_text], q.metric, q.target)
    errs, mismatches = {}, []

    def err(name, a, b):
        errs[name] = max(errs.get(name, 0.0), abs(float(a) - float(b)))

    fin = rec[-1]
    for t, (r, x) in enumerate(zip(rows, rec)):
        err("p", 0, np.abs(np.array(r["p"]) - x["p"]).max())
        for k in ("conf", "ent", "loss"):
            err(k, r[k], x[k])
        err("K", r["K"], kl(x["logp"], fin["logp"]))
        if r["pred"] != x["pred"] or r["correct"] != x["correct"]:
            mismatches.append(f"loop {t + 1} pred/correct")
        if r["F_final"] != int(x["pred"] != fin["pred"]) or r["T_final"] != label(x["correct"], fin["correct"]):
            mismatches.append(f"loop {t + 1} final flip/transition")
        if t + 1 < len(rec):
            nx = rec[t + 1]
            err("D", r["D"], kl(x["logp"], nx["logp"]))
            err("G", r["G"], x["loss"] - nx["loss"])
            if r["F"] != int(x["pred"] != nx["pred"]) or r["T"] != label(x["correct"], nx["correct"]):
                mismatches.append(f"loop {t + 1} next flip/transition")
        elif not (math.isnan(r["D"]) and math.isnan(r["D_vocab"]) and math.isnan(r["G"]) and r["F"] == -1 and r["T"] == ""):
            mismatches.append("last loop has next-loop targets")
    return errs, mismatches


def model_sample_check(model, qs, df, hidden, n):
    """Re-run n questions per benchmark and compare with the stored values.

    rerun   gated: the SAME packed input run again, with the entropy, KL targets, gate and state recomputed by code that
            does not go through extract_question. Same shapes mean the same kernels, so this checks the target and storage
            path, not bf16 kernel noise.
    ctx     reported only: a context-only pass. In bf16 its shapes differ from the packed pass, and on very short prompts
            that changes the matmul kernels (diagnose_short_context.py: float32 agrees to ~2e-5, bf16 does not).
    """
    import torch

    from lookahead.ouro_adapter import NativeGate, OuroLoopRunner
    from lookahead.packed import PackedScorer, build_packed, score_per_option

    runner, scorer = OuroLoopRunner(model), PackedScorer(model)
    dev = next(model.parameters()).device
    by_id = {q.doc_id: q for q in qs}
    ok_ids = [d for d in df.loc[(df.status == "ok") & (df.loop == 1), "doc_id"]]
    pos = {d: i for i, d in enumerate(df.loc[df.loop == 1, "doc_id"])}
    keys = ("vent", "D_vocab", "K_vocab", "gate", "cdf", "hidden")
    rerun = {k: 0.0 for k in keys + ("opt_ll",)}
    ctx = {k: 0.0 for k in keys}
    ctx["hidden_rel_l2"] = 0.0

    def vocab_stats(logps):
        out = [{"vent": -float(np.sum(np.exp(lp) * lp))} for lp in logps]
        for t, lp in enumerate(logps):
            out[t]["K_vocab"] = kl(lp, logps[-1])
            out[t]["D_vocab"] = kl(lp, logps[t + 1]) if t + 1 < len(logps) else None
        return out

    def compare(worst, stats, gates, cdfs, hs, rows, stored_h):
        for t in range(LOOPS):
            for k in ("vent", "K_vocab", "D_vocab"):
                if stats[t][k] is not None:
                    worst[k] = max(worst[k], abs(stats[t][k] - rows[t][k]))
            worst["gate"] = max(worst["gate"], abs(gates[t] - rows[t]["gate"]))
            worst["cdf"] = max(worst["cdf"], abs(cdfs[t] - rows[t]["cdf"]))
            h = hs[t].float().cpu().numpy().astype(np.float16).astype(np.float64)
            st = stored_h[t].astype(np.float64)
            worst["hidden"] = max(worst["hidden"], float(np.abs(h - st).max()))
            if "hidden_rel_l2" in worst:
                worst["hidden_rel_l2"] = max(worst["hidden_rel_l2"], float(np.linalg.norm(h - st) / np.linalg.norm(st)))

    for doc_id in ok_ids[:: max(1, len(ok_ids) // n)][:n]:
        q = by_id[doc_id]
        rows = df[df.doc_id == doc_id].sort_values("loop").to_dict("records")
        stored_h = hidden[pos[doc_id]]
        last = torch.tensor([len(q.context_enc) - 1], device=dev)
        # rerun: the same packed input again
        gate, logps, gates, cdfs, hs = NativeGate(runner.total_loops, 1), [], [], [], []
        for out in scorer.loops(build_packed(q.context_enc, q.cont_encs)):
            logps.append(out.vocab_logp.double().cpu().numpy())
            gates.append(out.gate_logit)
            cdfs.append(float(gate.update(torch.tensor([out.gate_logit]))[0]))
            hs.append(out.hidden)
        compare(rerun, vocab_stats(logps), gates, cdfs, hs, rows, stored_h)
        # ctx: a context-only pass (reported)
        state = runner.start(torch.tensor([q.context_enc], device=dev))
        gate, logps, gates, cdfs, hs = NativeGate(runner.total_loops, 1), [], [], [], []
        for _ in range(LOOPS):
            runner.step(state)
            logps.append(torch.log_softmax(runner.readout(state.hidden, last)[0].double(), -1).cpu().numpy())
            g = float(runner.gate_logit(state.hidden, last)[0])
            gates.append(g)
            cdfs.append(float(gate.update(torch.tensor([g]))[0]))
            hs.append(state.hidden[0, -1])
        compare(ctx, vocab_stats(logps), gates, cdfs, hs, rows, stored_h)
        ref_ll, _, _ = score_per_option(model, q.context_enc, q.cont_encs)
        for t in range(LOOPS):
            rerun["opt_ll"] = max(rerun["opt_ll"], max(abs(a - b) for a, b in zip(ref_ll[t], rows[t]["opt_ll"])))
    return {"rerun_same_input": rerun, "vs_context_only_reported": ctx}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=ALL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=10, help="questions per benchmark re-run through the model")
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-model", action="store_true", help="skip the model re-run (CPU-only check of stored files)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else REPO_ROOT / "data" / ("extracted_smoke" if args.limit else "extracted")
    splits = load_splits(REPO_ROOT / "data" / "manifests" / "splits.csv")

    model = None
    if not args.no_model:
        from lookahead.ouro_adapter import load_ouro

        model, _ = load_ouro()
        hidden_size = model.config.hidden_size
    else:
        hidden_size = None

    report, ok_all = {"out": str(out), "benchmarks": {}}, True
    for b in args.benchmarks:
        qs = load_questions(REPO_ROOT / "data" / "questions" / f"{b}.jsonl.gz")
        qs = qs[: args.limit] if args.limit else qs
        df, hid, dones = read_benchmark(out, b)  # 5: raises on any checksum mismatch
        checks, notes = {}, {}
        if df is None:
            report["benchmarks"][b] = {"checks": {"present": False}}
            ok_all = False
            continue
        firsts = df[df.loop == 1]
        ids = list(firsts.doc_id)
        # 1 schema
        checks["all_fields"] = all(f in df.columns for f in FIELDS)
        checks["every_question_once"] = sorted(ids) == sorted(q.doc_id for q in qs) and len(set(ids)) == len(ids)
        checks["four_loops_each"] = bool((df.groupby("doc_id").loop.apply(lambda s: sorted(s) == list(range(1, LOOPS + 1)))).all())
        checks["hidden_shape"] = hid.shape == (len(ids), LOOPS, hidden_size or hid.shape[2])
        # 3 numerics, failures
        status = firsts.status.value_counts().to_dict()
        notes["status_counts"] = status
        okdf = df[df.status == "ok"]
        num = okdf[["conf", "ent", "vent", "gate", "cdf", "ms", "loss", "K", "K_vocab"]].to_numpy(dtype=float)
        nxt = okdf[okdf.loop < LOOPS][["D", "D_vocab", "G"]].to_numpy(dtype=float)
        ok_pos = np.flatnonzero((firsts.status == "ok").to_numpy())
        checks["finite"] = bool(np.isfinite(num).all() and np.isfinite(nxt).all() and np.isfinite(hid[ok_pos]).all())
        checks["p_sums_to_1"] = bool(all(abs(sum(p) - 1) < 1e-9 for p in okdf.p))
        checks["kl_nonnegative"] = bool((okdf[["K", "K_vocab"]].to_numpy() >= -1e-6).all() and (nxt[:, :2] >= -1e-6).all())
        checks["entropy_in_range"] = bool(((okdf.ent >= -1e-9) & (okdf.ent <= np.log(okdf.n_options) + 1e-9)).all()
                                          and (okdf.vent >= -1e-6).all())
        checks["failures_le_1pct"] = (len(ids) - status.get("ok", 0)) <= 0.01 * len(ids)
        checks["cdf_monotone_ends_at_1"] = bool(okdf.groupby("doc_id").cdf.apply(
            lambda s: bool(np.all(np.diff(s.to_numpy()) >= -1e-9) and abs(s.iloc[-1] - 1) < 1e-9)).all())
        # 2 targets from stored option scores
        by_id = {q.doc_id: q for q in qs}
        worst, mism = {}, []
        for doc_id, g in okdf.groupby("doc_id"):
            errs, m = check_targets(by_id[doc_id], g.sort_values("loop").to_dict("records"))
            for k, v in errs.items():
                worst[k] = max(worst.get(k, 0.0), v)
            mism += [f"{doc_id}: {x}" for x in m]
        notes["recompute_max_abs_err"] = worst
        notes["recompute_mismatches"] = mism[:20]
        checks["targets_recomputed"] = not mism and all(v <= TOL for v in worst.values())
        if model is not None and args.sample:
            sample = model_sample_check(model, qs, df, hid, args.sample)
            notes["model_sample"] = sample
            re = sample["rerun_same_input"]
            checks["vocab_targets_gate_hidden_reproduced_on_same_input"] = (
                all(re[k] <= RERUN_TOL for k in ("vent", "D_vocab", "K_vocab", "gate", "cdf")) and re["hidden"] <= RERUN_HIDDEN_TOL)
            # option scores against per-option scoring: bf16 differences up to ~0.45 nats were seen in the pilot (P5)
            checks["option_ll_vs_per_option_le_1nat"] = re["opt_ll"] <= 1.0
        # 4 splits
        checks["splits_match_manifest"] = all(splits.get((b, r.doc_id), (None, None))[1] == r.split for r in firsts.itertuples())
        checks["one_split_per_question"] = bool((df.groupby("doc_id").split.nunique() == 1).all())
        # 5 resume
        checks["no_leftover_tmp"] = not list((out / b).glob("*.tmp"))
        checks["done_records_match_rows"] = sum(d["n"] for d in dones) == len(ids)
        report["benchmarks"][b] = {"checks": checks, **notes}
        ok_all &= all(checks.values())
        print(f"{b:15s} {'PASS' if all(checks.values()) else 'FAIL'}  {len(ids)} q  status {status}  "
              + ", ".join(k for k, v in checks.items() if not v), flush=True)

    # cross-benchmark split integrity: no (benchmark, doc_id) assigned twice is enforced by load_splits
    diag_path = REPO_ROOT / "results" / "phase1" / "diagnose_short_context.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))["float32"]
        worst = {b_: max(v[k]["maxabs_max"] for k in ("packed_vs_ctx", "packed_vs_ctxmask")) for b_, v in diag.items()}
        report["float32_packed_vs_context_only_maxabs"] = worst
        report["float32_equivalence_recorded"] = bool(worst) and max(worst.values()) <= FP32_TOL
    else:
        report["float32_equivalence_recorded"] = False
    ok_all &= report["float32_equivalence_recorded"]
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"runs": []}
    report["meta_complete"] = bool(meta["runs"]) and all(all(k in run for k in META_KEYS) for run in meta["runs"])
    report["code_identical_across_runs"] = len({run.get("code_sha256") for run in meta["runs"]}) <= 1
    ok_all &= report["meta_complete"] and report["code_identical_across_runs"]
    report["passed"] = bool(ok_all)
    name = args.report or ("extraction_check_smoke.json" if args.limit else "extraction_check.json")
    rdir = results_dir("phase1")
    (rdir / name).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"float32 packed vs context-only recorded and within {FP32_TOL}: {report['float32_equivalence_recorded']}")
    print(f"meta complete: {report['meta_complete']}, same code across runs: {report['code_identical_across_runs']}")
    print("ALL CHECKS PASSED" if ok_all else "CHECKS FAILED: see " + str(rdir / name))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
