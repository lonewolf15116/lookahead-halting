"""Phase 3 amendment A1: HellaSwag audit rule v2 and the TEST / DEV+ allocation. Pure logic, no dataset access.

HellaSwag rule v2 (A1.3, reviewer-accepted conditionally): a HellaSwag train row is clean when the dataset ships a gold answer,
its text hash was never used (any benchmark, any role), its `source_id` and its context are not those of ANY used HellaSwag row
(train or validation split), and its text is not already a clean row of another candidate pool. Native ids are not used
(the repeated-id audit showed they collide). Internal text duplicates keep one row, first in allocation-rank order.

Allocation: rows are ranked by sha256(`SALT|benchmark|task|row|text_hash`). Without groups, the first n_test rows are TEST and
the rest DEV+. With groups (HellaSwag `source_id`), whole groups are taken in the rank order of their best-ranked row until
TEST holds at least n_test rows; a group is never split between TEST and DEV+.
"""
import hashlib
from collections import Counter

SALT = "lh-phase3-alloc-v1"
HS_RULE_VERSION = "hellaswag-audit-v2"


def rank_key(benchmark: str, task: str, row: int, text_hash: str) -> str:
    return hashlib.sha256(f"{SALT}|{benchmark}|{task}|{row}|{text_hash}".encode("utf-8")).hexdigest()


def hellaswag_v2(rows: list[dict], used_texts: set, other_clean_texts: set) -> dict:
    """rows: every HellaSwag row of the train AND validation splits, dicts with keys split, task, row, text_hash, labeled,
    source_id, ctx_a. Returns counts by first exclusion reason (train rows only) and the clean train rows.

    Used rows (text hash in used_texts) of BOTH splits define the excluded sources and contexts. Reason order:
    UNLABELED, USED_TEXT, SHARED_SOURCE, SHARED_CONTEXT, OTHER_POOL_TEXT, INTERNAL_DUPLICATE.
    """
    used = [r for r in rows if r["text_hash"] in used_texts]
    used_src = {r["source_id"] for r in used if r["source_id"]}
    used_ctx = {r["ctx_a"] for r in used if r["ctx_a"]}
    src_by_split = {s: {r["source_id"] for r in used if r["source_id"] and r["split"] == s} for s in {r["split"] for r in used}}
    order = ("UNLABELED", "USED_TEXT", "SHARED_SOURCE", "SHARED_CONTEXT", "OTHER_POOL_TEXT", "INTERNAL_DUPLICATE")
    excl = Counter()
    only_via_validation_sources = 0
    survivors = []
    train = [r for r in rows if r["split"] == "train"]
    for r in train:
        if not r["labeled"]:
            excl["UNLABELED"] += 1
        elif r["text_hash"] in used_texts:
            excl["USED_TEXT"] += 1
        elif r["source_id"] and r["source_id"] in used_src:
            excl["SHARED_SOURCE"] += 1
            if r["source_id"] not in src_by_split.get("train", set()):
                only_via_validation_sources += 1  # what the within-train audit could not see
        elif r["ctx_a"] and r["ctx_a"] in used_ctx:
            excl["SHARED_CONTEXT"] += 1
        elif r["text_hash"] in other_clean_texts:
            excl["OTHER_POOL_TEXT"] += 1
        else:
            survivors.append(r)
    seen, clean = set(), []
    for r in sorted(survivors, key=lambda r: rank_key("hellaswag", r["task"], r["row"], r["text_hash"])):
        if r["text_hash"] in seen:
            excl["INTERNAL_DUPLICATE"] += 1
        else:
            seen.add(r["text_hash"])
            clean.append(r)
    if sum(excl.values()) + len(clean) != len(train):
        raise AssertionError("hellaswag v2 counts do not add up")
    return {"rule_version": HS_RULE_VERSION, "n_train_rows": len(train), "n_used_rows_all_splits": len(used),
            "n_used_rows_by_split": dict(sorted(Counter(r["split"] for r in used).items())),
            "n_used_sources": len(used_src), "n_used_contexts": len(used_ctx),
            "excluded": {k: excl.get(k, 0) for k in order},
            "excluded_by_source_only_through_validation_split": only_via_validation_sources,
            "clean": len(clean), "_clean": clean}


def allocate(rows: list[dict], benchmark: str, n_test: int, group_key: str | None = None) -> dict:
    """rows: dicts with task, row, text_hash (and group_key if given). Returns TEST and DEV+ lists (rank order) and counts.
    Raises ValueError when the pool has fewer than n_test rows (the caller must return for review, not shrink the target)."""
    if len(rows) < n_test:
        raise ValueError(f"{benchmark}: pool {len(rows)} < requested TEST {n_test}")
    keyed = sorted(rows, key=lambda r: rank_key(benchmark, r["task"], r["row"], r["text_hash"]))
    if group_key is None:
        test, dev = keyed[:n_test], keyed[n_test:]
        n_groups = None
    else:
        groups: dict = {}
        for r in keyed:  # rank order: a group's first member is its best-ranked row
            g = r.get(group_key) or f"__row:{r['task']}:{r['row']}"  # no source id: the row is its own group
            groups.setdefault(g, []).append(r)
        test, dev = [], []
        for g in groups.values():  # dict keeps insertion order = rank of each group's best row
            (test if len(test) < n_test else dev).extend(g)
        n_groups = len(groups)
    if {(r["task"], r["row"]) for r in test} & {(r["task"], r["row"]) for r in dev}:
        raise AssertionError("TEST and DEV+ overlap")
    return {"benchmark": benchmark, "n_pool": len(rows), "n_test_requested": n_test, "n_test": len(test), "n_dev": len(dev),
            "n_groups": n_groups, "test": test, "dev": dev}
