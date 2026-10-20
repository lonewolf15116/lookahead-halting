"""Acceptance checks for a Phase 1 extraction (reviewer, 2026-10-20), run on the stored files.

    python scripts/verify_extraction.py --out data/extracted_smoke --limit 40
    python scripts/verify_extraction.py                         # data/extracted, everything

1 schema       every question once, 4 loop rows with every field, hidden [n, 4, hidden_size]
2 targets      every option-level value and target recomputed from the stored option log-likelihoods with independent
               code; on a sample per benchmark, the model is re-run (context-only pass, per-option scoring) to recompute
               the full-vocabulary entropy and KL targets, the gate and the hidden state
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
VOCAB_TOL = 2e-3  # sample re-run: bf16 model, float32 log-softmax; the shared state is bit-identical in practice


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
    """Re-run n questions through independent paths and compare vocabulary targets, gate, hidden state, option scores."""
    import torch

    from lookahead.ouro_adapter import NativeGate, OuroLoopRunner
    from lookahead.packed import score_per_option

    runner = OuroLoopRunner(model)
    dev = next(model.parameters()).device
    by_id = {q.doc_id: q for q in qs}
    ok_ids = [d for d in df.loc[(df.status == "ok") & (df.loop == 1), "doc_id"]]
    pos = {d: i for i, d in enumerate(df.loc[df.loop == 1, "doc_id"])}
    worst = {"vent": 0.0, "D_vocab": 0.0, "K_vocab": 0.0, "gate": 0.0, "cdf": 0.0, "hidden": 0.0, "opt_ll": 0.0}
    for doc_id in ok_ids[:: max(1, len(ok_ids) // n)][:n]:
        q = by_id[doc_id]
        rows = df[df.doc_id == doc_id].sort_values("loop").to_dict("records")
        state = runner.start(torch.tensor([q.context_enc], device=dev))
        last = torch.tensor([len(q.context_enc) - 1], device=dev)
        gate = NativeGate(runner.total_loops, 1)
        logps = []
        for t in range(LOOPS):
            runner.step(state)
            lp = torch.log_softmax(runner.readout(state.hidden, last)[0].double(), -1).cpu().numpy()
            logps.append(lp)
            g = float(runner.gate_logit(state.hidden, last)[0])
            worst["gate"] = max(worst["gate"], abs(g - rows[t]["gate"]))
            worst["cdf"] = max(worst["cdf"], abs(float(gate.update(torch.tensor([g]))[0]) - rows[t]["cdf"]))
            h = state.hidden[0, -1].float().cpu().numpy().astype(np.float16).astype(np.float32)
            worst["hidden"] = max(worst["hidden"], float(np.abs(h - hidden[pos[doc_id], t].astype(np.float32)).max()))
        for t in range(LOOPS):
            worst["vent"] = max(worst["vent"], abs(-float(np.sum(np.exp(logps[t]) * logps[t])) - rows[t]["vent"]))
            worst["K_vocab"] = max(worst["K_vocab"], abs(kl(logps[t], logps[-1]) - rows[t]["K_vocab"]))
            if t + 1 < LOOPS:
                worst["D_vocab"] = max(worst["D_vocab"], abs(kl(logps[t], logps[t + 1]) - rows[t]["D_vocab"]))
        ref_ll, _, _ = score_per_option(model, q.context_enc, q.cont_encs)
        for t in range(LOOPS):
            worst["opt_ll"] = max(worst["opt_ll"], max(abs(a - b) for a, b in zip(ref_ll[t], rows[t]["opt_ll"])))
    return worst


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
            notes["model_sample_max_abs_err"] = sample
            checks["vocab_targets_gate_hidden_recomputed"] = all(
                sample[k] <= VOCAB_TOL for k in ("vent", "D_vocab", "K_vocab", "gate", "cdf", "hidden"))
            # option scores against per-option scoring: bf16 differences up to ~0.45 nats were seen in the pilot (P5)
            checks["option_ll_vs_per_option_le_1nat"] = sample["opt_ll"] <= 1.0
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
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"runs": []}
    report["meta_complete"] = bool(meta["runs"]) and all(all(k in run for k in META_KEYS) for run in meta["runs"])
    report["code_identical_across_runs"] = len({run.get("code_sha256") for run in meta["runs"]}) <= 1
    ok_all &= report["meta_complete"] and report["code_identical_across_runs"]
    report["passed"] = bool(ok_all)
    name = args.report or ("extraction_check_smoke.json" if args.limit else "extraction_check.json")
    rdir = results_dir("phase1")
    (rdir / name).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"meta complete: {report['meta_complete']}, same code across runs: {report['code_identical_across_runs']}")
    print("ALL CHECKS PASSED" if ok_all else "CHECKS FAILED: see " + str(rdir / name))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
