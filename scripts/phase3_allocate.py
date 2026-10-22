"""Phase 3 amendment A1: HellaSwag cross-split audit (rule v2) and the TEST / DEV+ allocation. CPU only.

    python scripts/phase3_allocate.py [--tokenizer]

Reads the committed inventory (sha256-checked), the HellaSwag dataset and the manifests. No model outputs, labels of any model,
hidden states or confirmation predictions. Writes, refusing to overwrite:
  results/phase3/allocation.json           counts, rule versions, HellaSwag v2 audit, list hashes, precision at final sizes
  results/phase3/allocation_test_ids.csv   one-time TEST rows (in-distribution and OOD), sealed after extraction
  results/phase3/allocation_dev_ids.csv    DEV+ rows (development only)
If HellaSwag v2 leaves fewer than the requested TEST rows, only allocation.json is written, with status RETURN_FOR_REVIEW and
no id lists (exit 3): the target is never reduced silently. Exit 0 = allocation written, ready for review and freeze.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_allocation as A
from lookahead import phase3_inventory as inv
from lookahead.config import REPO_ROOT, load_config
from lookahead.questions import BENCHMARKS

_spec = importlib.util.spec_from_file_location("phase3_inventory_script", REPO_ROOT / "scripts" / "phase3_inventory.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

INVENTORY_SHA256 = "7ec501aee8e2f1b1ca078f14af969f887ec947fd0b2ec9e778151fd93ce517f1"  # commit 848d1b9
# A1.3 (reviewer-accepted sizes): benchmark -> (source split, TEST rows, group key)
PLAN = {"openbookqa": ("train", 3000, None), "commonsense_qa": ("train", 1000, None), "hellaswag": ("train", 750, "source_id")}
OOD = {"piqa": "validation", "social_iqa": "validation"}  # A1.4: whole split is TEST; WinoGrande excluded
OUT_JSON, OUT_TEST, OUT_DEV = ("results/phase3/allocation.json", "results/phase3/allocation_test_ids.csv",
                               "results/phase3/allocation_dev_ids.csv")
COLUMNS = "role,benchmark,source_split,task,row,native_id,text_hash,group"
MIN_PER_5K_MIN = 35  # measured extraction speed, minutes per 5,000 questions


def read_clean_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        d = dict(zip(head, ln.split(",")))
        d["row"] = int(d["row"])
        out.append(d)
    return out


def csv_rows(role, benchmark, split, rows, group_key=None):
    return [f"{role},{benchmark},{split},{r['task']},{r['row']},{r.get('native_id') or ''},{r['text_hash']},"
            f"{(r.get(group_key) or '') if group_key else ''}" for r in rows]


def main(argv=None, *, task_manager=None, lm=None, root=None, inventory_sha=INVENTORY_SHA256, plan=None, ood=None,
         hs_name="hellaswag", benchmarks=None, specs=None, split_plan=None, take=None, salt=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true")
    args = ap.parse_args(argv)
    root = Path(root or REPO_ROOT)
    plan, ood = dict(PLAN if plan is None else plan), dict(OOD if ood is None else ood)
    outs = [root / OUT_JSON, root / OUT_TEST, root / OUT_DEV]
    if any(p.exists() for p in outs):
        print("refusing: an allocation output already exists", file=sys.stderr)
        return 2
    inv_path, csv_path = root / "results/phase3/inventory.json", root / "results/phase3/inventory_clean_ids.csv"
    got = C.sha256_bytes(inv_path.read_bytes())
    if got != inventory_sha:
        print(f"STOP: inventory sha256 {got} is not the committed {inventory_sha}", file=sys.stderr)
        return 2
    report = json.loads(inv_path.read_text(encoding="utf-8"))
    if C.sha256_bytes(csv_path.read_bytes()) != report["inputs"]["clean_ids_csv_sha256"]:
        print("STOP: inventory_clean_ids.csv does not match the inventory report", file=sys.stderr)
        return 2
    clean = read_clean_csv(csv_path)
    by_pool: dict = {}
    for r in clean:
        by_pool.setdefault((r["benchmark"], r["source_split"]), []).append(r)

    # 1. used-set, rebuilt and verified exactly as in the inventory
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    benchmarks = list(benchmarks or BENCHMARKS)
    specs, split_plan = specs or BENCHMARKS, split_plan or C.SPLIT_PLAN
    take = take or {b: C.DEFAULT_TAKE.get(b) for b in benchmarks}
    with_hashes = bool(args.tokenizer and lm is None)
    if lm is None:
        if args.tokenizer:
            from transformers import AutoTokenizer

            from lookahead.lm_eval_ouro import LoopedOuroLM

            lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
        else:
            lm = S.StubLM()
    print("rebuilding used-set (same path as the inventory) ...", flush=True)
    used, checks, failures = S.build_used_set(root, benchmarks, lm, tm, specs, split_plan, take, salt or C.SALT, with_hashes)
    if failures:
        print("STOP: used-set verification failed:", failures, file=sys.stderr)
        return 3

    # 2. HellaSwag rule v2, cross-split: sources and contexts of every used HellaSwag row (train and validation)
    hs_rows = []
    loaded = tm.load([hs_name])["tasks"]
    for task_name in sorted(loaded):
        task = loaded[task_name]
        for split in ("train", "validation"):
            C.switch_eval_split(task, split)
            for i, doc in enumerate(task.eval_docs):
                hs_rows.append({"split": split, "task": task_name, "row": i, "native_id": S.inv_native_id(hs_name, doc, split),
                                "text_hash": S.inv_text_hash(hs_name, doc, task), "labeled": S.is_labeled(task, doc, hs_name),
                                "source_id": str(doc.get("source_id") or ""), "ctx_a": str(doc.get("ctx_a") or "")})
    other_texts = {r["text_hash"] for r in clean if r["benchmark"] != hs_name}
    hs = A.hellaswag_v2(hs_rows, set(used.texts), other_texts)
    hs_clean = hs.pop("_clean")
    print("hellaswag v2:", json.dumps(hs), flush=True)

    result = {"amendment": "A1", "allocation_salt": A.SALT, "hellaswag_rule": hs, "inventory_sha256": inventory_sha,
              "used_set_fingerprint": used.fingerprint(), "verification": checks,
              "plan": {b: {"source_split": s, "n_test": n, "group_key": g} for b, (s, n, g) in plan.items()},
              "ood": ood, "tokenizer_mode": "real" if with_hashes else "stub"}
    n_hs = plan.get(hs_name, (None, 0, None))[1]
    if hs["clean"] < n_hs:
        result["status"] = "RETURN_FOR_REVIEW"
        result["reason"] = f"HellaSwag v2 leaves {hs['clean']} clean rows < requested TEST {n_hs}; no allocation written"
        outs[0].parent.mkdir(parents=True, exist_ok=True)
        outs[0].write_text(inv.canonical_json(result), encoding="utf-8", newline="\n")
        print(result["reason"])
        return 3

    # 3. allocation
    test_lines, dev_lines, per = [], [], {}
    for b, (split, n, gk) in plan.items():
        rows = hs_clean if b == hs_name else by_pool.get((b, split), [])
        al = A.allocate(rows, b, n, gk)
        test_lines += csv_rows("test", b, split, al["test"], gk)
        dev_lines += csv_rows("dev", b, split, al["dev"], gk)
        per[b] = {k: al[k] for k in ("n_pool", "n_test_requested", "n_test", "n_dev", "n_groups")}
        per[b]["precision_test"] = inv.precision_table(al["n_test"])
        print(f"{b}: pool {al['n_pool']} TEST {al['n_test']} DEV+ {al['n_dev']}", flush=True)
    for b, split in ood.items():
        rows = sorted(by_pool.get((b, split), []), key=lambda r: A.rank_key(b, r["task"], r["row"], r["text_hash"]))
        test_lines += csv_rows("test_ood", b, split, rows)
        per[b] = {"n_pool": len(rows), "n_test": len(rows), "n_dev": 0, "precision_test": inv.precision_table(len(rows))}
        print(f"{b} (OOD): TEST {len(rows)}", flush=True)
    test_text = "\n".join([COLUMNS] + test_lines) + "\n"
    dev_text = "\n".join([COLUMNS] + dev_lines) + "\n"
    n_extract = len(test_lines) + len(dev_lines)
    result.update({"status": "ALLOCATED (pending reviewer freeze)", "per_benchmark": per,
                   "n_test_rows": len(test_lines), "n_dev_rows": len(dev_lines),
                   "test_ids_sha256": C.sha256_bytes(test_text.encode("utf-8")),
                   "dev_ids_sha256": C.sha256_bytes(dev_text.encode("utf-8")),
                   "extraction_estimate": {"questions": n_extract,
                                           "gpu_minutes": round(n_extract / 5000 * MIN_PER_5K_MIN)}})
    outs[0].parent.mkdir(parents=True, exist_ok=True)
    outs[1].write_text(test_text, encoding="utf-8", newline="\n")
    outs[2].write_text(dev_text, encoding="utf-8", newline="\n")
    outs[0].write_text(inv.canonical_json(result), encoding="utf-8", newline="\n")
    print("written", *map(str, outs), sep="\n  ")
    print("allocation sha256", C.sha256_bytes(outs[0].read_bytes()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
