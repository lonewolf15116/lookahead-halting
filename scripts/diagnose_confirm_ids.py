"""Evidence for the confirmation-id failure: are dataset-native ids unique? CPU only, reads no outcomes.

    python scripts/diagnose_confirm_ids.py

1. For every benchmark with a native id, on the Phase 1 split(s) and the confirmation pool split: number of docs, distinct
   native ids, duplicate groups (with rows, source fields and a text snippet), and whether duplicates share text.
2. Replays the `--limit 20` smoke selection (Phase 1 rebuilt on the first 20 docs per benchmark, as the smoke run did) and
   lists the 20 selected HellaSwag docs with their native id and row, flagging every native id that occurs more than once.
Writes results/gate2/confirm_id_diagnostic.json. Does not write any manifest or question file.
"""
import json
import sys
from collections import defaultdict

from lookahead.config import REPO_ROOT, load_config, results_dir
from lookahead.confirm import (SALT, SPLIT_PLAN, Exclusion, build_confirmation_questions, native_id, record_exemplars,
                               switch_eval_split, text_hash)
from lookahead.questions import BENCHMARKS, load_lm_eval_questions

PHASE1_SPLIT = {"arc_challenge": "test", "arc_easy": "test", "openbookqa": "test", "hellaswag": "validation",
                "commonsense_qa": "validation"}
SOURCE_FIELDS = ("ind", "id", "source_id", "split", "split_type", "activity_label")


def survey(tm, benchmark, split):
    loaded = tm.load([benchmark])["tasks"]
    groups, n, texts = defaultdict(list), 0, defaultdict(set)
    for task_name in sorted(loaded):
        task = loaded[task_name]
        switch_eval_split(task, split)
        for i, doc in enumerate(task.eval_docs):
            n += 1
            nid = native_id(benchmark, doc, split)
            if nid is None:
                continue
            snippet = " ".join(str(doc.get("ctx") or doc.get("question") or doc.get("question_stem") or "").split())[:80]
            groups[nid].append({"row": i, **{k: doc.get(k) for k in SOURCE_FIELDS if k in doc}, "text": snippet})
            texts[nid].add(text_hash(benchmark, doc, task))
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    return {"split": split, "n_docs": n, "n_with_native_id": sum(len(v) for v in groups.values()),
            "n_distinct_native_ids": len(groups), "n_duplicate_groups": len(dup),
            "n_docs_in_duplicate_groups": sum(len(v) for v in dup.values()),
            "max_group_size": max((len(v) for v in dup.values()), default=1),
            "n_duplicate_groups_with_differing_text": sum(len(texts[k]) > 1 for k in dup),
            "example_groups": dict(list(sorted(dup.items()))[:10])}


def main() -> int:
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    from lookahead.lm_eval_ouro import LoopedOuroLM

    tm = TaskManager()
    out = {"native_id_survey": {}, "smoke_replay": {}}
    for b in ("hellaswag", "commonsense_qa", "arc_challenge", "arc_easy", "openbookqa"):
        out["native_id_survey"][b] = [survey(tm, b, PHASE1_SPLIT[b]), survey(tm, b, SPLIT_PLAN[b])]
        for s in out["native_id_survey"][b]:
            print(f"{b:15s} {s['split']:10s} docs {s['n_docs']:6d}  distinct native ids {s['n_distinct_native_ids']:6d}  "
                  f"duplicate groups {s['n_duplicate_groups']:5d} (docs {s['n_docs_in_duplicate_groups']}, max size "
                  f"{s['max_group_size']}, with differing text {s['n_duplicate_groups_with_differing_text']})", flush=True)

    lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
    recorders = {}
    for b in BENCHMARKS:
        with record_exemplars() as rec:
            load_lm_eval_questions(b, lm, limit=20, task_manager=tm, spec=BENCHMARKS[b], on_task=rec.begin_task,
                                   on_doc=rec.on_doc)
        recorders[b] = rec
    build = build_confirmation_questions("hellaswag", lm, 20, Exclusion.from_recorders(recorders), salt=SALT,
                                         task_manager=tm, spec=BENCHMARKS["hellaswag"])
    count = defaultdict(int)
    for c in build.selected:
        count[c.native_id] += 1
    rows = [{"doc_id": q.doc_id, "native_id": c.native_id, "row": c.index, "native_id_repeated": count[c.native_id] > 1}
            for q, c in zip(build.questions, build.selected)]
    out["smoke_replay"] = {"selected": rows, "repeated_native_ids": {k: v for k, v in count.items() if v > 1},
                           "doc_ids_unique": len({r["doc_id"] for r in rows}) == len(rows)}
    print("\nsmoke replay (hellaswag, 20 selected):")
    for r in rows:
        print(f"  {r['doc_id']:40s} row {r['row']:6d} {'REPEATED native id' if r['native_id_repeated'] else ''}")
    print(f"repeated native ids among the 20: {out['smoke_replay']['repeated_native_ids'] or 'none'}; "
          f"doc_ids unique now: {out['smoke_replay']['doc_ids_unique']}")

    path = results_dir("gate2") / "confirm_id_diagnostic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
