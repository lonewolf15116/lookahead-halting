"""Phase 3 D-a2 (reviewer D-a Failure Review, 2026-10-26): investigate the PIQA float32 D-a failure (1.17e-4 > 1e-4).

    python scripts/diagnose_short_context_ood_a2.py

The D-a report (results/phase3/diagnose_short_context_ood.json, sha256 pinned) is read, never written; its FAIL stands.
Rows: exactly the D-a rows (same selection rule; the fingerprints and question-set hashes must equal the D-a report's).
1 conditions  the settings actually in effect: GPU, torch/CUDA/cuDNN/transformers versions, TF32 flags, float32 matmul
              precision, attention implementation, SDPA backends, deterministic-algorithm flag, CPU threads.
2 localise    float32 on the GPU, all 100 PIQA and 100 Social IQa rows: packed vs context-only (plain and explicit-mask)
              state differences per question and per loop. Also checks that D-a's float32 summary is reproduced.
3 reference   float64 on the CPU for the 10 PIQA questions with the largest float32 packed-vs-ctx difference (a numerical
              quantity, not an outcome; ties by doc id). Ouro's forced-float32 steps (RMSNorm, RoPE, eager softmax) are
              replaced by the same formulas in float64 (src/lookahead/f64_reference.py) and verified at run time; a dtype
              audit records any module output that is not float64. Same weights (bf16 values, exactly representable),
              tokens, masks, positions and code path as D-a.
              Criterion (fixed before running): float64 packed-vs-ctx and packed-vs-explicit-mask-ctx max |diff| <= 1e-9 on
              every selected question and loop -> "numerical"; > 1e-9 -> "differs" (HOLD, investigate); any reference check
              failing or the run incomplete -> "unresolved" (HOLD). Also reported: each float32 path's distance to the
              float64 context-only state.
No option score, prediction, label or correctness is computed or written. Writes
results/phase3/diagnose_short_context_ood_a2.json (refuses to overwrite). Exit 0 iff the interpretation is "numerical".
"""
import importlib.util
import json
import platform
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT, load_config

DA_REPORT = "results/phase3/diagnose_short_context_ood.json"
DA_REPORT_SHA256 = "7783daa870cf9028d87ee9dae69a1c8bb0e3e9b6516a7246125a454965d59947"
REPORT = "results/phase3/diagnose_short_context_ood_a2.json"
K_WORST = 10
F64_TOL = 1e-9
MIN_FREE_RAM_GB = 16.0
LOOPS = 4


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def worst_k(rows: list[dict], k: int = K_WORST) -> list[str]:
    """doc ids of the k questions with the largest float32 packed-vs-ctx max |diff| over loops (ties: doc id)."""
    per_q: dict = {}
    for r in rows:
        per_q[r["doc_id"]] = max(per_q.get(r["doc_id"], 0.0), r["packed_vs_ctx_maxabs"], r["packed_vs_ctxmask_maxabs"])
    return [d for d, _ in sorted(per_q.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def per_loop(rows: list[dict]) -> dict:
    out = {}
    for t in range(1, LOOPS + 1):
        rs = [r for r in rows if r["loop"] == t]
        out[str(t)] = {"maxabs_max": max(r["packed_vs_ctx_maxabs"] for r in rs),
                       "maxabs_median": sorted(r["packed_vs_ctx_maxabs"] for r in rs)[len(rs) // 2],
                       "rel_l2_max": max(r["packed_vs_ctx_rel_l2"] for r in rs),
                       "share_exact": sum(r["packed_vs_ctx_maxabs"] == 0 for r in rs) / len(rs)}
    return out


def interpret(f64_rows: list[dict], checks_ok: bool, complete: bool) -> tuple[str, float]:
    if not complete or not checks_ok or not f64_rows:
        return "unresolved", float("nan")
    worst = max(max(r["packed_vs_ctx_maxabs"], r["packed_vs_ctxmask_maxabs"]) for r in f64_rows)
    return ("numerical" if worst <= F64_TOL else "differs"), worst


def free_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]

            s = MS()
            s.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.ap / 2**30
        return float("nan")


def conditions(torch, model) -> dict:
    cuda = torch.cuda.is_available()
    out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if cuda else None,
           "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "attn_implementation": getattr(model.config, "_attn_implementation", None),
           "cpu": platform.processor(), "cpu_threads": torch.get_num_threads(), "python": platform.python_version()}
    if cuda:
        b = torch.backends.cuda
        out["sdpa_backends_enabled"] = {"flash": b.flash_sdp_enabled(), "mem_efficient": b.mem_efficient_sdp_enabled(),
                                        "math": b.math_sdp_enabled(),
                                        "cudnn": getattr(b, "cudnn_sdp_enabled", lambda: None)()}
    import transformers

    out["transformers"] = transformers.__version__
    out["lm_eval"] = C.lm_eval_version()
    out["rmsnorm_class_forward_module"] = type(model.model.norm).forward.__module__
    return out


def main(argv=None) -> int:
    root = REPO_ROOT
    out_path = root / REPORT
    if out_path.exists():
        print(f"refusing: {REPORT} already exists", file=sys.stderr)
        return 2
    if not P._sha_ok(root / DA_REPORT, DA_REPORT_SHA256):
        raise SystemExit("the D-a report is not the committed one (6efe327)")
    da = json.loads((root / DA_REPORT).read_text(encoding="utf-8"))
    DA = _load("da", "scripts/diagnose_short_context_ood.py")
    D = _load("diag", "scripts/diagnose_short_context.py")
    report = {"diagnostic": "D-a2", "da_report_sha256": DA_REPORT_SHA256, "k_worst": K_WORST, "f64_tol": F64_TOL,
              "criterion": "float64 packed-vs-ctx and packed-vs-explicit-mask-ctx max |diff| <= 1e-9 on every selected "
                           "question and loop -> numerical; > 1e-9 -> differs; incomplete or a reference check failing -> "
                           "unresolved", "failures": []}
    ram = free_ram_gb()
    report["free_ram_gb_at_start"] = ram
    if not ram >= MIN_FREE_RAM_GB:  # nothing computed yet: no report, so it can be re-run after freeing memory
        print(f"only {ram:.1f} GB RAM free; {MIN_FREE_RAM_GB} GB needed for the float64 reference. Close other programs "
              "and re-run. Nothing was computed or written.", file=sys.stderr)
        return 3

    # same rows and questions as D-a
    alloc = P.read_allocation(root)
    allocated = {r["text_hash"] for role in alloc.values() for _, rows in role.values() for r in rows}
    import torch
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    from lookahead import f64_reference as F
    from lookahead.lm_eval_ouro import LoopedOuroLM
    from lookahead.ouro_adapter import load_ouro

    cfg = load_config("model")
    tm = TaskManager()
    lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(root / cfg["local_dir"])))
    sets = {}
    for b in DA.BENCHES:
        chosen, fp = DA.select(DA.eligible_rows(root, b, allocated), b)
        qs, rep = P.build_set(b, lm, chosen, DA.SPLIT, tm, P.OOD_SPECS[b], allocated, DA._text_hash(), "diag_da2", set())
        same = fp == da["benchmarks"][b]["selection_sha256"] and rep["question_set_sha256"] == da["benchmarks"][b]["question_set_sha256"]
        report.setdefault("benchmarks", {})[b] = {"selection_sha256": fp, "question_set_sha256": rep["question_set_sha256"],
                                                  "same_as_da": same, "n": len(qs)}
        if not same:
            report["failures"].append(f"{b}: rows or questions differ from D-a")
        sets[b] = qs
    if report["failures"]:
        report["interpretation"] = "unresolved"
        return _finish(out_path, report)

    # 1 + 2: float32 on the GPU
    model, _ = load_ouro()
    model = model.to(torch.float32)
    report["conditions_gpu_float32"] = conditions(torch, model)
    f32_rows = {}
    for b, qs in sets.items():
        rows = D.compare(model, qs)
        f32_rows[b] = rows
        s = D.summarize(rows)
        da_max = da["benchmarks"][b]["float32"]["packed_vs_ctx"]["maxabs_max"]
        report["benchmarks"][b].update({
            "float32_summary": s, "float32_per_loop": per_loop(rows),
            "float32_reproduces_da": s["packed_vs_ctx"]["maxabs_max"] == da_max,
            "float32_per_question": [{k: r[k] for k in ("doc_id", "loop", "n_ctx", "n_packed", "scale", "packed_vs_ctx_maxabs",
                                                        "packed_vs_ctx_rel_l2", "packed_vs_ctxmask_maxabs")} for r in rows]})
        print(f"float32  {b:11s} max {s['packed_vs_ctx']['maxabs_max']:.4g} (D-a {da_max:.4g})  per loop "
              + "  ".join(f"L{t} {v['maxabs_max']:.2g}" for t, v in report["benchmarks"][b]["float32_per_loop"].items()), flush=True)
    sel_ids = worst_k(f32_rows["piqa"])
    sel = [q for q in sets["piqa"] if q.doc_id in set(sel_ids)]
    report["float64_selected_doc_ids"] = sel_ids
    # keep the float32 states of the selected questions for the cross-precision comparison
    f32_states = {q.doc_id: _states(model, q, D, torch) for q in sel}
    del model
    torch.cuda.empty_cache()

    # 3: float64 reference on the CPU
    model, _ = load_ouro(device="cpu")
    model = model.to(torch.float64)
    patched = F.patch_float64(model)
    rot = model.model.rotary_emb
    ref_checks = {"patched": patched, "rmsnorm_rel_err": F.rmsnorm_check(model.model.norm), "rotary": F.rotary_check(rot)}
    ref_checks["ok"] = (bool(patched["rmsnorm"]) and bool(patched["rotary"]) and ref_checks["rmsnorm_rel_err"] <= 1e-14
                        and ref_checks["rotary"]["ok"])
    report["conditions_cpu_float64"] = conditions(torch, model)
    audit = F.DtypeAudit(model)
    f64_rows, cross, complete = [], [], True
    try:
        for q in sel:
            rows = D.compare(model, [q])
            f64_rows += rows
            st = _states(model, q, D, torch)
            for t in range(LOOPS):
                ref = st["ctx"][t]
                cross.append({"doc_id": q.doc_id, "loop": t + 1,
                              "f32_packed_vs_f64_ctx_maxabs": float((f32_states[q.doc_id]["packed"][t] - ref).abs().max()),
                              "f32_ctx_vs_f64_ctx_maxabs": float((f32_states[q.doc_id]["ctx"][t] - ref).abs().max())})
            print(f"float64  {q.doc_id}: packed vs ctx max {max(r['packed_vs_ctx_maxabs'] for r in rows):.3g}", flush=True)
    except Exception as e:  # recorded; the interpretation becomes "unresolved"
        complete = False
        report["failures"].append(f"float64 run incomplete: {type(e).__name__}: {e}")
    finally:
        audit.close()
    ref_checks["dtype_audit_non_float64_outputs"] = audit.bad
    ref_checks["dtype_audit_calls"] = audit.n_calls
    ref_checks["ok"] = ref_checks["ok"] and not audit.bad
    report["float64_reference_checks"] = ref_checks
    report["float64_rows"] = f64_rows
    report["cross_precision"] = cross
    report["float64_summary"] = D.summarize(f64_rows) if f64_rows else None
    interp, worst = interpret(f64_rows, ref_checks["ok"], complete and len({r["doc_id"] for r in f64_rows}) == K_WORST)
    report["interpretation"], report["float64_maxabs"] = interp, worst
    if interp != "numerical":
        report["failures"].append(f"interpretation {interp} (float64 max {worst})")
    report["provenance"] = {**DA.code_identity(root),
                            "scripts/diagnose_short_context_ood_a2.py": C.sha256_bytes(
                                (root / "scripts/diagnose_short_context_ood_a2.py").read_bytes().replace(b"\r\n", b"\n")),
                            "model": cfg["repo_id"], "model_revision": cfg["revision"]}
    return _finish(out_path, report)


def _states(model, q, D, torch) -> dict:
    """Packed and context-only states at the last context token, all loops, as float64 CPU tensors (same code as D-a)."""
    from lookahead.ouro_adapter import OuroLoopRunner
    from lookahead.packed import PackedScorer, build_packed

    dev = next(model.parameters()).device
    runner, scorer = OuroLoopRunner(model), PackedScorer(model)
    with torch.no_grad():
        packed = [o.hidden.detach().to("cpu", torch.float64) for o in scorer.loops(build_packed(q.context_enc, q.cont_encs))]
        s = runner.start(torch.tensor([q.context_enc], device=dev))
        ctx = []
        for _ in range(LOOPS):
            runner.step(s)
            ctx.append(s.hidden[0, -1].detach().to("cpu", torch.float64))
    return {"packed": packed, "ctx": ctx}


def _finish(out_path: Path, report: dict) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"D-a2 interpretation: {report.get('interpretation')}  float64 max {report.get('float64_maxabs')}  "
          f"failures {report['failures']}")
    return 0 if report.get("interpretation") == "numerical" else 2


if __name__ == "__main__":
    sys.exit(main())
