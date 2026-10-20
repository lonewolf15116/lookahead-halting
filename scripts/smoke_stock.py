"""Load Ouro through the stock Hugging Face path and see what it exposes.

Each step is recorded on its own, so one failure does not hide the others. This only observes the
model; it assumes nothing about Ouro-specific outputs. Writes results/gate0/stock_smoke.json.
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lookahead.config import REPO_ROOT, load_config, results_dir

PROMPT = "Question: What is 17 times 8?\nAnswer:"


def describe(obj, depth=0):
    if obj is None:
        return None
    if hasattr(obj, "shape"):
        return {"tensor": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        first = describe(obj[0], depth + 1) if obj and depth < 2 else None
        return {"sequence_length": len(obj), "first_item": first}
    if hasattr(obj, "keys"):
        return {k: describe(obj[k], depth + 1) for k in obj.keys()} if depth < 3 else "mapping"
    return type(obj).__name__


def main() -> int:
    cfg = load_config("model")
    local = str(REPO_ROOT / cfg["local_dir"])
    report, state = {}, {}

    def step(name, fn):
        start = time.perf_counter()
        try:
            result = fn()
            report[name] = {"ok": True, "seconds": round(time.perf_counter() - start, 3), "result": result}
        except Exception:  # noqa: BLE001
            report[name] = {"ok": False, "error": traceback.format_exc()}

    def load():
        torch.cuda.reset_peak_memory_stats()
        state["tok"] = AutoTokenizer.from_pretrained(local, trust_remote_code=cfg["trust_remote_code"])
        state["model"] = AutoModelForCausalLM.from_pretrained(
            local, trust_remote_code=cfg["trust_remote_code"], torch_dtype=getattr(torch, cfg["dtype"])
        ).to("cuda").eval()
        model = state["model"]
        return {
            "class": type(model).__name__,
            "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "config_total_ut_steps": getattr(model.config, "total_ut_steps", None),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        }

    def forward():
        inputs = state["tok"](PROMPT, return_tensors="pt").to("cuda")
        state["inputs"] = inputs
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = state["model"](**inputs)
        torch.cuda.synchronize()
        return {"output": describe(out), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}

    def forward_hidden():
        with torch.no_grad():
            out = state["model"](**state["inputs"], output_hidden_states=True)
        return describe(getattr(out, "hidden_states", None))

    def generate():
        with torch.no_grad():
            ids = state["model"].generate(**state["inputs"], max_new_tokens=8, do_sample=False)
        return state["tok"].decode(ids[0], skip_special_tokens=True)

    step("load", load)
    if report["load"]["ok"]:
        step("forward", forward)
        step("forward_with_hidden_states", forward_hidden)
        step("generate_8_tokens", generate)

    out = results_dir("gate0") / "stock_smoke.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["ok"] for k, v in report.items()}, indent=2))
    print(f"Wrote {out}")
    return 0 if all(v["ok"] for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
