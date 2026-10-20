"""Check the loop-by-loop adapter against Ouro's own forward on the real model, and profile the GPU.

Writes results/gate0/adapter_check.json and results/gate0/gpu_profile.json. Exit code is 0 only when the
adapter reproduces the stock forward: same predictions, logits within tolerance.
"""
import json
import sys
import time
import traceback

import torch
import transformers

from lookahead.config import results_dir
from lookahead.ouro_adapter import OuroLoopRunner, load_ouro

PROMPTS = [
    "Question: What is 17 times 8?\nAnswer:",
    "Question: Which gas do plants absorb from the air during photosynthesis?\nAnswer:",
    "Question: A student mixes baking soda and vinegar in a flask and a balloon over the flask inflates. "
    "Which best explains why the balloon inflated?\nAnswer:",
]
# Identical kernels should give exactly 0.0. A bf16 logit near 16-32 moves in steps of 0.06-0.12, so these
# allow a couple of steps without letting a real discrepancy through.
LOGIT_TOL = 0.25
PROFILE_SHAPES = [(1, 512), (1, 1536), (4, 1536)]  # (batch, tokens); 1536 is about a 25-shot ARC-C prompt


def max_diff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def check_against_stock(model, runner, ids):
    """Per loop: hidden state, gate logit and answer-position logits against Ouro's own forward."""
    positions = torch.tensor([ids.shape[1] - 1], device=ids.device)
    out = {"tokens": ids.shape[1], "loops": []}
    ref_hidden = ref_gate = None
    try:
        with torch.no_grad():
            _, ref_hidden, ref_gate = model.model(input_ids=ids, use_cache=False)
    except Exception:  # noqa: BLE001
        out["stock_internals_error"] = traceback.format_exc()
    state = runner.start(ids)
    for k in range(runner.total_loops):
        runner.step(state)
        mine = runner.readout(state.hidden, positions)
        with torch.no_grad():
            stock = model(input_ids=ids, use_cache=False, exit_at_step=k).logits[:, -1].float()
        row = {
            "loop": k + 1,
            "logit_max_abs_diff": max_diff(mine, stock),
            "argmax_match": bool((mine.argmax(-1) == stock.argmax(-1)).all()),
        }
        if ref_hidden is not None:
            row["hidden_max_abs_diff"] = max_diff(state.hidden, ref_hidden[k])
            with torch.no_grad():
                row["gate_max_abs_diff"] = max_diff(runner.core.early_exit_gate(state.hidden), ref_gate[k])
        out["loops"].append(row)
    return out


def check_padding(runner, tok, device):
    """A right-padded batch must give the same answer-position logits as each prompt alone."""
    a = tok(PROMPTS[0], return_tensors="pt").input_ids[0]
    b = tok(PROMPTS[2], return_tensors="pt").input_ids[0]
    length = max(len(a), len(b))
    batch = torch.full((2, length), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros(2, length, dtype=torch.long)
    batch[0, : len(a)], batch[1, : len(b)] = a, b
    mask[0, : len(a)], mask[1, : len(b)] = 1, 1
    positions = torch.tensor([len(a) - 1, len(b) - 1])
    batched, _ = runner.all_loops(batch.to(device), positions.to(device), mask.to(device))
    alone_a, _ = runner.all_loops(a[None].to(device), positions[:1].to(device))
    alone_b, _ = runner.all_loops(b[None].to(device), positions[1:].to(device))
    rows = []
    for k in range(runner.total_loops):
        rows.append({
            "loop": k + 1,
            "logit_max_abs_diff": max(max_diff(batched[k, 0], alone_a[k, 0]), max_diff(batched[k, 1], alone_b[k, 0])),
            "argmax_match": bool(batched[k, 0].argmax() == alone_a[k, 0].argmax()
                                 and batched[k, 1].argmax() == alone_b[k, 0].argmax()),
        })
    return rows


def profile(runner):
    device = next(runner.model.parameters()).device
    vocab = runner.config.vocab_size
    runner.step(runner.start(torch.randint(0, vocab, (1, 64), device=device)))  # warm-up: CUDA init, kernel setup
    rows = []
    for batch, length in PROFILE_SHAPES:
        row = {"batch": batch, "tokens": length}
        try:
            ids = torch.randint(0, vocab, (batch, length), device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            state = runner.start(ids)
            times = []
            for _ in range(runner.total_loops):
                sync(device)
                start = time.perf_counter()
                runner.step(state)
                sync(device)
                times.append(round((time.perf_counter() - start) * 1000, 1))
            row["loop_ms"] = times
            if device.type == "cuda":
                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        except torch.cuda.OutOfMemoryError:
            row["out_of_memory"] = True
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()
        rows.append(row)
    return rows


def main() -> int:
    model, tok = load_ouro()
    device = next(model.parameters()).device
    runner = OuroLoopRunner(model)
    weights_gb = round(torch.cuda.memory_allocated() / 1024**3, 3) if device.type == "cuda" else None

    report = {"prompts": [], "padding": None}
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        report["prompts"].append(check_against_stock(model, runner, ids))
    report["padding"] = check_padding(runner, tok, device)

    stock_ok = all(
        loop["argmax_match"] and loop["logit_max_abs_diff"] <= LOGIT_TOL
        for p in report["prompts"] for loop in p["loops"]
    )
    # Padding is informational: batched and single-sequence bf16 kernels differ numerically (0.69 seen on 2026-10-20) and
    # this project runs batch size 1, so only the top answer must agree. Exact invariance is in the float32 unit test.
    pad_ok = all(r["argmax_match"] for r in report["padding"])
    # The loop computation itself should match bit for bit. Logits can differ in the last bits because Ouro projects
    # every position through lm_head and the adapter only the answer position, a different matmul shape.
    internals = [loop for p in report["prompts"] for loop in p["loops"] if "hidden_max_abs_diff" in loop]
    report["loop_computation_identical"] = bool(internals) and all(
        loop["hidden_max_abs_diff"] == 0.0 and loop["gate_max_abs_diff"] == 0.0 for loop in internals
    )
    report["tolerances"] = {"logit": LOGIT_TOL}
    report["all_ok"] = stock_ok and pad_ok
    out_dir = results_dir("gate0")
    (out_dir / "adapter_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    gpu = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "dtype": str(next(model.parameters()).dtype),
        "weights_gb": weights_gb,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "profile": profile(runner),
    }
    (out_dir / "gpu_profile.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")

    print(json.dumps({"stock_ok": stock_ok, "padding_ok": pad_ok, "loop_computation_identical": report["loop_computation_identical"]}))
    print(json.dumps(gpu, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
