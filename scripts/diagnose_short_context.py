"""Why does the packed pass differ from a context-only pass on OpenBookQA (very short prompts)?

For each sampled question and loop it compares the hidden state at the last context token from
  packed      the packed pass (context + all option tokens, custom mask)
  ctx         a context-only pass (the pass the verify script re-runs)
  ctx_mask    a context-only pass through the same explicit-mask code path as the packed pass
in bf16 and then in float32. In float32 packed and ctx must agree to ~1e-5 if the mask is right; a bf16-only gap that
depends on length points to shape-dependent kernels, not leakage.

    python scripts/diagnose_short_context.py
Writes results/phase1/diagnose_short_context.json.
"""
import json
import sys

import numpy as np
import torch

from lookahead.config import REPO_ROOT, results_dir
from lookahead.extract import load_questions
from lookahead.ouro_adapter import OuroLoopRunner
from lookahead.packed import PackedScorer, build_packed, segment_mask_function

SAMPLES = {"openbookqa": 100, "modarith": 30, "arc_easy": 30, "commonsense_qa": 20}
LOOPS = 4


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    return {"maxabs": float((a - b).abs().max()), "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "rel_l2": float((a - b).norm() / b.norm())}


@torch.no_grad()
def compare(model, qs) -> list[dict]:
    dev = next(model.parameters()).device
    runner, scorer = OuroLoopRunner(model), PackedScorer(model)
    rows = []
    for q in qs:
        ids = torch.tensor([q.context_enc], device=dev)
        n = len(q.context_enc)
        zeros = segment_mask_function(torch.zeros(n, dtype=torch.long, device=dev))
        s_plain, s_mask = runner.start(ids), runner.start(ids, and_mask_function=zeros)
        packed = [o.hidden for o in scorer.loops(build_packed(q.context_enc, q.cont_encs))]
        for t in range(LOOPS):
            runner.step(s_plain)
            runner.step(s_mask)
            ref, ref_m = s_plain.hidden[0, -1], s_mask.hidden[0, -1]
            rows.append({"doc_id": q.doc_id, "loop": t + 1, "n_ctx": n, "n_packed": q.packed_len,
                         "scale": float(ref.double().abs().max()),
                         **{f"packed_vs_ctx_{k}": v for k, v in metrics(packed[t], ref).items()},
                         **{f"ctxmask_vs_ctx_{k}": v for k, v in metrics(ref_m, ref).items()},
                         **{f"packed_vs_ctxmask_{k}": v for k, v in metrics(packed[t], ref_m).items()}})
    return rows


def summarize(rows) -> dict:
    out = {}
    for key in ("packed_vs_ctx", "ctxmask_vs_ctx", "packed_vs_ctxmask"):
        m = np.array([r[f"{key}_maxabs"] for r in rows])
        out[key] = {"maxabs_median": float(np.median(m)), "maxabs_max": float(m.max()),
                    "rel_l2_max": float(max(r[f"{key}_rel_l2"] for r in rows)),
                    "cos_min": float(min(r[f"{key}_cos"] for r in rows)),
                    "share_exact": float((m == 0).mean())}
    out["hidden_scale_max"] = float(max(r["scale"] for r in rows))
    out["n_ctx_range"] = [min(r["n_ctx"] for r in rows), max(r["n_ctx"] for r in rows)]
    return out


def main() -> int:
    from lookahead.ouro_adapter import load_ouro

    model, _ = load_ouro()
    qdir = REPO_ROOT / "data" / "questions"
    sets = {b: load_questions(qdir / f"{b}.jsonl.gz")[:n] for b, n in SAMPLES.items()}
    report = {}
    for dtype in ("bfloat16", "float32"):
        model = model.to(getattr(torch, dtype))
        report[dtype] = {}
        for b, qs in sets.items():
            rows = compare(model, qs)
            report[dtype][b] = summarize(rows)
            s = report[dtype][b]
            print(f"{dtype:9s} {b:15s} ctx {s['n_ctx_range']}  packed vs ctx: max {s['packed_vs_ctx']['maxabs_max']:.4g} "
                  f"(exact {100 * s['packed_vs_ctx']['share_exact']:.0f}%, rel L2 {s['packed_vs_ctx']['rel_l2_max']:.2g})  "
                  f"explicit-mask ctx vs ctx: max {s['ctxmask_vs_ctx']['maxabs_max']:.4g}  "
                  f"packed vs explicit-mask ctx: max {s['packed_vs_ctxmask']['maxabs_max']:.4g}", flush=True)
        if dtype == "bfloat16":
            report["bfloat16"]["openbookqa_rows"] = compare(model, sets["openbookqa"])[:400]
    (results_dir("phase1") / "diagnose_short_context.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
