"""Phase 3: are HellaSwag rows excluded only by a repeated native id real overlaps or identifier collisions? CPU only.

    python scripts/phase3_hellaswag_id_audit.py [--tokenizer]

Reviewer request (Phase 3 inventory decision): the inventory excluded 820 HellaSwag train rows by native id alone. This rebuilds
the same used-set (identical code path to scripts/phase3_inventory.py, verified against the committed manifests), then compares
each id-only row with the used rows sharing its id (same context? same source video/article? same activity?).
Reads the HellaSwag dataset and manifests only. No model outputs, labels of any model or confirmation predictions. Writes
results/phase3/hellaswag_id_audit.json and refuses to overwrite. It changes nothing in the inventory.
"""
import importlib.util
import json
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_inventory as inv
from lookahead.config import REPO_ROOT, load_config
from lookahead.questions import BENCHMARKS

_spec = importlib.util.spec_from_file_location("phase3_inventory_script", REPO_ROOT / "scripts" / "phase3_inventory.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

OUT = "results/phase3/hellaswag_id_audit.json"


def main(argv=None, *, task_manager=None, lm=None, root=None, specs=None, split_plan=None, take=None, benchmarks=None, name="hellaswag",
         salt=None) -> int:
    use_tok = bool(argv and "--tokenizer" in argv) or (argv is None and "--tokenizer" in sys.argv[1:])
    root = Path(root or REPO_ROOT)
    out = root / OUT
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    benchmarks = list(benchmarks or BENCHMARKS)
    specs, split_plan = specs or BENCHMARKS, split_plan or C.SPLIT_PLAN
    take = take or {b: C.DEFAULT_TAKE.get(b) for b in benchmarks}
    if lm is None:
        if use_tok:
            from transformers import AutoTokenizer

            from lookahead.lm_eval_ouro import LoopedOuroLM

            lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
        else:
            lm = S.StubLM()
    print("rebuilding used-set (same path as the inventory) ...", flush=True)
    used, checks, failures = S.build_used_set(root, benchmarks, lm, tm, specs, split_plan, take, salt or C.SALT, use_tok and lm is not None)
    if failures:
        print("STOP: used-set verification failed:", failures, file=sys.stderr)
        return 3
    loaded = tm.load([name])["tasks"]
    rows = []
    for task_name in sorted(loaded):
        task = loaded[task_name]
        C.switch_eval_split(task, "train")
        for i, doc in enumerate(task.eval_docs):
            nid, th = S.inv_native_id(name, doc, "train"), S.inv_text_hash(name, doc, task)
            id_kinds, tx_kinds = used.kinds_for(name, nid, th)
            rows.append({"row": i, "nid": nid, "text_hash": th, "used_text": bool(tx_kinds), "id_used": bool(id_kinds),
                         "source_id": str(doc.get("source_id", "")), "ctx_a": str(doc.get("ctx_a", "")),
                         "activity": str(doc.get("activity_label", ""))})
    rep = inv.id_collision_report(rows)
    rep["pool"] = f"{name}/train"
    rep["audit_rule_version"] = inv.AUDIT_RULE_VERSION
    rep["verification"] = checks
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(inv.canonical_json(rep), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "verification"}, indent=1))
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
