"""Phase 1, step 4: packed-scoring pilot on ARC-Challenge (plan section 4, tests P4-P8).

Runs packed scoring on all 1,172 ARC-C questions (25-shot, all 4 loops) and compares it with the Gate 0 per-option
scores in results/gate0/arc_cache.jsonl, which must cover every request (that also proves the prompts are identical).

    python scripts/pilot_packed.py
    python scripts/pilot_packed.py --limit 50     # quick check

Writes results/phase1/pilot_packed.json (summary, pass/fail) and pilot_packed_questions.jsonl (per question).
Exit code 0 only if every hard criterion (P0, P6, P7, P8 flip agreement, no failures) passes.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time

import numpy as np
import torch

from lookahead.config import REPO_ROOT, results_dir
from lookahead.lm_eval_ouro import LoopedOuroLM
from lookahead.ouro_adapter import OuroLoopRunner, load_ouro
from lookahead.packed import PackedScorer, build_packed
from lookahead.questions import load_lm_eval_questions, predictions, primary_scores

LOOPS, N_HIDDEN, N_ISOLATION, TIMEOUT_S = 4, 200, 100, 60.0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(np.clip(p, 1e-30, None)) - np.log(np.clip(q, 1e-30, None)))))


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def summarize(xs):
    xs = sorted(xs)
    return {"median": xs[len(xs) // 2], "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = results_dir("phase1")

    cache = {}
    for line in (REPO_ROOT / "results" / "gate0" / "arc_cache.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["k"]] = row["ll"]

    model, tok = load_ouro()
    qs = load_lm_eval_questions("arc_challenge", LoopedOuroLM(None, tok), limit=args.limit)
    keys = [[hashlib.sha1(f"{q.context}\x00{c}".encode("utf-8")).hexdigest() for c in q.continuations] for q in qs]
    missing = sum(k not in cache for ks in keys for k in ks)
    print(f"{len(qs)} questions, {sum(map(len, keys))} requests; {missing} missing from the Gate 0 cache", flush=True)

    scorer, runner = PackedScorer(model), OuroLoopRunner(model)
    for q in qs[:3]:  # warm-up, not timed
        list(scorer.loops(build_packed(q.context_enc, q.cont_encs)))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows, failures = [], {}
    started = time.time()
    for i, q in enumerate(qs):
        rec = {"doc_id": q.doc_id, "target": q.target, "packed_len": q.packed_len, "status": "ok"}
        try:
            p = build_packed(q.context_enc, q.cont_encs)
            lls, hid, gates, ms = [], [], [], []
            sync()
            t_prev = time.perf_counter()
            for out in scorer.loops(p):
                sync()
                now = time.perf_counter()
                ms.append(1000 * (now - t_prev))
                t_prev = now
                lls.append(out.option_ll)
                hid.append(out.hidden)
                gates.append(out.gate_logit)
            if not all(math.isfinite(x) for ll in lls for x in ll):
                rec["status"] = "nonfinite"
            elif sum(ms) / 1000 > TIMEOUT_S:
                rec["status"] = "timeout"
            rec.update(packed_ll=lls, ref_ll=[[cache[k][t] for k in keys[i]] for t in range(LOOPS)], loop_ms=ms, gate=gates)
            if i < N_ISOLATION and len(q.cont_encs) >= 2:
                # P2 on the real backend: replace the last option's tokens (same length, then 3 tokens longer);
                # every other option's score and the shared hidden state must not move
                gen = torch.Generator().manual_seed(i)
                d_ll, d_h = 0.0, 0.0
                for extra in (0, 3):
                    alt = torch.randint(0, model.config.vocab_size, (len(q.cont_encs[-1]) + extra,), generator=gen).tolist()
                    for t, o in enumerate(scorer.loops(build_packed(q.context_enc, q.cont_encs[:-1] + [alt]))):
                        d_ll = max(d_ll, max(abs(x - y) for x, y in zip(o.option_ll[:-1], lls[t][:-1])))
                        d_h = max(d_h, float((o.hidden.float() - hid[t].float()).abs().max()))
                rec.update(isolation_ll_maxabs=d_ll, isolation_hidden_maxabs=d_h)
            if i < N_HIDDEN:  # P4: shared hidden state against a context-only pass, in bf16
                state = runner.start(torch.tensor([q.context_enc], device=scorer.device))
                diffs, coss = [], []
                for t in range(LOOPS):
                    runner.step(state)
                    a, b = hid[t].float(), state.hidden[0, -1].float()
                    diffs.append(float((a - b).abs().max()))
                    coss.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
                rec.update(hidden_maxabs=diffs, hidden_cos=coss)
        except torch.cuda.OutOfMemoryError:
            rec["status"] = "oom"
            torch.cuda.empty_cache()
        except KeyError:
            rec["status"] = "missing_reference"
        failures[rec["status"]] = failures.get(rec["status"], 0) + 1
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} questions ({(time.time() - started) / (i + 1):.2f} s each)", flush=True)
    elapsed = time.time() - started

    ok = [(q, r) for q, r in zip(qs, rows) if r["status"] == "ok"]
    res = {"n_questions": len(qs), "n_requests": sum(map(len, keys)), "missing_from_gate0_cache": missing,
           "status_counts": failures, "elapsed_min": round(elapsed / 60, 2), "per_loop": []}
    for t in range(LOOPS):
        ll_d, conf_d, same_norm, same_acc, same_rank = [], [], 0, 0, 0
        acc = {"packed_acc": 0, "packed_acc_norm": 0, "ref_acc": 0, "ref_acc_norm": 0}
        for q, r in ok:
            a, b = r["packed_ll"][t], r["ref_ll"][t]
            ll_d += [abs(x - y) for x, y in zip(a, b)]
            pa, pb = predictions(q, a), predictions(q, b)
            same_norm += pa["pred_acc_norm"] == pb["pred_acc_norm"]
            same_acc += pa["pred_acc"] == pb["pred_acc"]
            sa, sb = primary_scores(q, a), primary_scores(q, b)
            same_rank += list(np.argsort(-sa, kind="stable")) == list(np.argsort(-sb, kind="stable"))
            conf_d.append(abs(softmax(sa).max() - softmax(sb).max()))
            for name, pr in (("packed", pa), ("ref", pb)):
                acc[f"{name}_acc"] += pr["pred_acc"] == q.target
                acc[f"{name}_acc_norm"] += pr["pred_acc_norm"] == q.target
        n = max(1, len(ok))
        row = {"loop": t + 1, "ll_absdiff": summarize(ll_d), "conf_absdiff": summarize(conf_d),
               "top_identical_acc_norm": same_norm / n, "top_identical_acc": same_acc / n, "ranking_identical": same_rank / n,
               **{k: 100 * v / len(qs) for k, v in acc.items()}}
        row["acc_norm_gap"] = row["packed_acc_norm"] - row["ref_acc_norm"]
        row["acc_gap"] = row["packed_acc"] - row["ref_acc"]
        loop_ms = [r["loop_ms"][t] for _, r in ok]
        row["ms_mean"] = statistics.mean(loop_ms) if loop_ms else None
        if any("hidden_maxabs" in r for _, r in ok):
            h = [r["hidden_maxabs"][t] for _, r in ok if "hidden_maxabs" in r]
            c = [r["hidden_cos"][t] for _, r in ok if "hidden_cos" in r]
            row["hidden_maxabs"] = summarize(h)
            row["hidden_cos_min"] = min(c)
        res["per_loop"].append(row)

    targets = []
    for t in range(LOOPS - 1):  # P8: D_t and F_t from packed vs per-option scores
        flip_same, d_p, d_r = 0, [], []
        for q, r in ok:
            pp = [softmax(primary_scores(q, r["packed_ll"][k])) for k in (t, t + 1)]
            pr = [softmax(primary_scores(q, r["ref_ll"][k])) for k in (t, t + 1)]
            flip_same += (pp[0].argmax() != pp[1].argmax()) == (pr[0].argmax() != pr[1].argmax())
            d_p.append(kl(*pp))
            d_r.append(kl(*pr))
        targets.append({"step": f"{t + 1}->{t + 2}", "flip_agreement": flip_same / max(1, len(ok)),
                        "D_spearman": spearman(d_p, d_r) if len(ok) > 2 else None,
                        "D_absdiff": summarize([abs(x - y) for x, y in zip(d_p, d_r)])})
    res["targets"] = targets

    # timing model: ms per loop = a + b * tokens, fitted on this run
    xs = np.array([r["packed_len"] for _, r in ok], dtype=float)
    ys = np.array([statistics.mean(r["loop_ms"]) for _, r in ok])
    b, a = np.polyfit(xs, ys, 1) if len(ok) > 2 else (float("nan"), float("nan"))
    res["timing"] = {"ms_per_loop_intercept": float(a), "ms_per_loop_per_token": float(b),
                     "mean_s_per_question": float(np.mean(ys) * LOOPS / 1000) if len(ok) else None}
    iso_ll = [r["isolation_ll_maxabs"] for _, r in ok if "isolation_ll_maxabs" in r]
    iso_h = [r["isolation_hidden_maxabs"] for _, r in ok if "isolation_hidden_maxabs" in r]
    res["P2_isolation_gpu"] = {"n": len(iso_ll), "ll_maxabs": max(iso_ll, default=None),
                               "hidden_maxabs": max(iso_h, default=None)}
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
    tc = out_dir / "token_counts.json"
    if tc.exists():
        rows_tc = json.loads(tc.read_text(encoding="utf-8"))["rows"]
        est = {r["benchmark"]: round(LOOPS * (a * r["questions"] + b * r["packed_tokens_total"]) / 3.6e6, 2) for r in rows_tc}
        res["runtime_estimate_h"] = {**est, "total": round(sum(est.values()), 2)}

    hard = {
        "P0_prompts_match_gate0": missing == 0,
        "no_failures": failures.get("ok", 0) == len(qs),
        "P6_top_identical_ge_99pct": all(r["top_identical_acc_norm"] >= 0.99 for r in res["per_loop"]),
        "P7_accuracy_within_0.5": all(abs(r["acc_norm_gap"]) <= 0.5 and abs(r["acc_gap"]) <= 0.5 for r in res["per_loop"]),
        "P8_flip_agreement_ge_98pct": all(t["flip_agreement"] >= 0.98 for t in targets),
        # the untouched options run the same computation, so this should be exact; 1e-3 only allows kernel noise
        "P2_isolation_gpu": bool(iso_ll) and max(iso_ll) <= 1e-3 and max(iso_h) <= 1e-3,
    }
    res["hard_criteria"] = hard
    res["passed"] = all(hard.values())
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    (out_dir / f"pilot_packed{suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (out_dir / f"pilot_packed_questions{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nPacked pilot: {len(qs)} questions in {elapsed / 60:.1f} min, status {failures}")
    for r in res["per_loop"]:
        print(f"loop {r['loop']}: acc_norm {r['packed_acc_norm']:.2f} vs {r['ref_acc_norm']:.2f}  acc {r['packed_acc']:.2f} vs "
              f"{r['ref_acc']:.2f}  top same {100 * r['top_identical_acc_norm']:.2f}%  rank same {100 * r['ranking_identical']:.2f}%  "
              f"|dll| p99 {r['ll_absdiff']['p99']:.4f} max {r['ll_absdiff']['max']:.4f}  {r['ms_mean'] or 0:.0f} ms")
    for t in targets:
        print(f"{t['step']}: flip agreement {100 * t['flip_agreement']:.2f}%  D spearman {t['D_spearman']:.4f}")
    iso = res["P2_isolation_gpu"]
    print(f"P2 isolation ({iso['n']} q): max |dll| {iso['ll_maxabs']}, max |dh| {iso['hidden_maxabs']}; "
          f"peak VRAM {res['peak_vram_gb']} GB")
    if "runtime_estimate_h" in res:
        print(f"Runtime estimate (packed, measured): {res['runtime_estimate_h']}")
    print(f"Hard criteria: {hard}\n{'PASSED' if res['passed'] else 'NOT PASSED'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
