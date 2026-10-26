"""Phase 3 D-a (reviewer Numerical Fidelity Review, 2026-10-26): float32 packed-vs-context-only equivalence on PIQA and
Social IQa, the two 0-shot OOD benchmarks the Phase 1 diagnostic (scripts/diagnose_short_context.py) did not cover.

    python scripts/diagnose_short_context_ood.py

Rows: only from each benchmark's TRAIN split, taken from the frozen inventory's clean list (never used before: not a
Phase 1 or Gate 2 evaluation or exemplar row, not a duplicate), minus every frozen A1 allocation row (DEV+ and TEST), by text
hash. N = 100 per benchmark, chosen by sha256 rank under a fixed salt; the selection is fingerprinted. If fewer than N
eligible rows exist the script stops (exit 2) and does not change the sampling rule.
Questions are built exactly as in the Phase 3 build (lm_eval format, 0-shot, the frozen OOD metric), so no exemplar is drawn.
Comparison: the hidden state at the last context token, packed pass vs context-only pass (plain and explicit-mask), all 4
loops, in bfloat16 (reported) and float32 (gated). PASS per benchmark iff the float32 max |diff| <= 1e-4 (FP32_TOL in
scripts/verify_extraction.py; unchanged). No option score, prediction, label or correctness is computed or written.
Writes results/phase3/diagnose_short_context_ood.json (refuses to overwrite); exit 0 only if both benchmarks pass.
"""
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT, load_config

SALT = "lh-phase3-da-v1"
N = 100
FP32_TOL = 1e-4
BENCHES = ("piqa", "social_iqa")
SPLIT = "train"
INVENTORY_SHA256 = "7ec501aee8e2f1b1ca078f14af969f887ec947fd0b2ec9e778151fd93ce517f1"
REPORT = "results/phase3/diagnose_short_context_ood.json"


class Insufficient(SystemExit):
    pass


def eligible_rows(root: Path, benchmark: str, allocated_texts: set) -> list[dict]:
    """Clean, never-used TRAIN rows of `benchmark` from the frozen inventory, minus any allocated text."""
    inv_path = root / "results/phase3/inventory.json"
    if not P._sha_ok(inv_path, INVENTORY_SHA256):
        raise SystemExit("inventory.json is not the frozen inventory")
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    csv = root / "results/phase3/inventory_clean_ids.csv"
    if not P._sha_ok(csv, inv["inputs"]["clean_ids_csv_sha256"]):
        raise SystemExit("inventory_clean_ids.csv does not match the frozen inventory")
    lines = csv.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        r = dict(zip(head, ln.split(",")))
        if r["benchmark"] == benchmark and r["source_split"] == SPLIT and r["text_hash"] not in allocated_texts:
            r["row"] = int(r["row"])
            out.append(r)
    return out


def rank(benchmark: str, r: dict) -> str:
    return hashlib.sha256(f"{SALT}|{benchmark}|{r['task']}|{r['row']}|{r['text_hash']}".encode()).hexdigest()


def select(rows: list[dict], benchmark: str, n: int = N) -> tuple[list[dict], str]:
    """First n by salted sha256 rank, returned in row order, with a fingerprint of the exact selection."""
    if len(rows) < n:
        raise Insufficient(f"{benchmark}: only {len(rows)} eligible unused {SPLIT} rows, {n} required; stopping "
                           "without changing the sampling rule")
    chosen = sorted(sorted(rows, key=lambda r: rank(benchmark, r))[:n], key=lambda r: (r["task"], r["row"]))
    fp = hashlib.sha256("".join(f"{r['task']},{r['row']},{r['text_hash']}\n" for r in chosen).encode()).hexdigest()
    return chosen, fp


def verdict(fp32_summary: dict) -> tuple[float, bool]:
    worst = max(fp32_summary[k]["maxabs_max"] for k in ("packed_vs_ctx", "packed_vs_ctxmask"))
    return worst, worst <= FP32_TOL


def code_identity(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location("extract_phase1_script", root / "scripts" / "extract_phase1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = m.code_identity(root)
    for f in ("scripts/diagnose_short_context_ood.py", "scripts/diagnose_short_context.py"):
        out[f] = C.sha256_bytes((root / f).read_bytes().replace(b"\r\n", b"\n"))
    return out


def main(argv=None) -> int:
    root = REPO_ROOT
    out_path = root / REPORT
    if out_path.exists():
        print(f"refusing: {REPORT} already exists", file=sys.stderr)
        return 2
    alloc = P.read_allocation(root)
    allocated = {r["text_hash"] for role in alloc.values() for _, rows in role.values() for r in rows}
    inv = json.loads((root / "results/phase3/inventory.json").read_text(encoding="utf-8"))
    report = {"diagnostic": "D-a", "salt": SALT, "n_per_benchmark": N, "split": SPLIT, "fp32_tol": FP32_TOL,
              "inventory_sha256": INVENTORY_SHA256, "allocation_json_sha256": P.PINNED["allocation_json"],
              "benchmarks": {}, "failures": []}
    selections = {}
    for b in BENCHES:
        rows = eligible_rows(root, b, allocated)
        exp = inv["provenance"]["revisions"][b]
        now = C.dataset_revision(exp["hf_dataset"])
        report["benchmarks"][b] = {"n_eligible": len(rows), "hf_dataset": exp["hf_dataset"],
                                   "hf_revision_inventory": exp["hf_revision"], "hf_revision_now": now}
        if now != exp["hf_revision"]:
            report["failures"].append(f"{b}: dataset revision {now} differs from the inventory's {exp['hf_revision']}")
        try:
            chosen, fp = select(rows, b)
        except Insufficient as e:
            report["failures"].append(str(e))
            continue
        selections[b] = chosen
        report["benchmarks"][b].update({"selection_sha256": fp, "rows": [f"{r['task']}/{SPLIT}@r{r['row']}" for r in chosen]})
    if report["failures"]:
        return _finish(out_path, report)

    import torch
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    from lookahead.lm_eval_ouro import LoopedOuroLM

    cfg = load_config("model")
    tm = TaskManager()
    lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(root / cfg["local_dir"])))
    sets = {}
    for b, chosen in selections.items():
        qs, rep = P.build_set(b, lm, chosen, SPLIT, tm, P.OOD_SPECS[b], allocated, _text_hash(), "diag_da", set())
        bad = (rep["exemplar_draws"] + rep["exemplar_is_allocated_row"] + rep["exemplar_not_previously_used"]
               + rep["self_exemplar"] + rep["live_exemplar_failures"])
        report["benchmarks"][b].update({"n_built": len(qs), "question_set_sha256": rep["question_set_sha256"],
                                        "shots": P.OOD_SPECS[b]["shots"], "exemplar_draws": rep["exemplar_draws"]})
        if bad or len(qs) != N:
            report["failures"].append(f"{b}: {bad} exemplar draws or violations, {len(qs)} questions built")
        sets[b] = qs
    if report["failures"]:
        return _finish(out_path, report)

    spec = importlib.util.spec_from_file_location("diag", root / "scripts" / "diagnose_short_context.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    from lookahead.ouro_adapter import load_ouro

    model, _ = load_ouro()
    report["provenance"] = {**code_identity(root), "model": cfg["repo_id"], "model_revision": cfg["revision"],
                            "dtypes": ["bfloat16 (reported)", "float32 (gated)"], "torch": torch.__version__,
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                            "python": platform.python_version(), "lm_eval_version": C.lm_eval_version()}
    for dtype in ("bfloat16", "float32"):
        model = model.to(getattr(torch, dtype))
        for b, qs in sets.items():
            s = D.summarize(D.compare(model, qs))
            report["benchmarks"][b][dtype] = s
            print(f"{dtype:9s} {b:11s} n {len(qs)}  ctx {s['n_ctx_range']}  packed vs ctx max {s['packed_vs_ctx']['maxabs_max']:.4g} "
                  f"(exact {100 * s['packed_vs_ctx']['share_exact']:.0f}%, rel L2 {s['packed_vs_ctx']['rel_l2_max']:.2g})  "
                  f"packed vs explicit-mask ctx max {s['packed_vs_ctxmask']['maxabs_max']:.4g}", flush=True)
    for b in sets:
        worst, ok = verdict(report["benchmarks"][b]["float32"])
        report["benchmarks"][b]["float32_maxabs"] = worst
        report["benchmarks"][b]["passed"] = ok
        if not ok:
            report["failures"].append(f"{b}: float32 max |diff| {worst:.3g} > {FP32_TOL}")
    return _finish(out_path, report)


def _text_hash():
    spec = importlib.util.spec_from_file_location("phase3_inventory_script", REPO_ROOT / "scripts" / "phase3_inventory.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S.inv_text_hash


def _finish(out_path: Path, report: dict) -> int:
    report["passed"] = not report["failures"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for b, v in report["benchmarks"].items():
        print(f"{b}: {'PASS' if v.get('passed') else 'FAIL'}  eligible {v['n_eligible']}  "
              f"selection {v.get('selection_sha256', '-')[:12]}  float32 max {v.get('float32_maxabs', 'n/a')}")
    print("D-a PASSED" if report["passed"] else f"D-a FAILED: {report['failures']}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
