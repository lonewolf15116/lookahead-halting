"""Phase 3 fresh-data inventory: pure audit logic (docs/PHASE3_PLAN.md section 9). CPU only, read-only.

This module holds the parts that need no dataset access, so they can be tested on synthetic records:
  * `Rec`       identity of one candidate question (benchmark, source split, native id, template-independent text hash)
  * `UsedSet`   everything earlier stages touched, by native id and by text hash, with the reason it was touched
  * `audit_pool` exclusion accounting for one (benchmark, source split) pool
  * `precision_table`, `build_report`, `canonical_json`

It never sees model outputs, correctness labels of any model, hidden states, thresholds or confirmation predictions. The only
label-related fact it receives is whether the dataset itself ships a gold answer for a row (`labeled`), because a pool row
without a gold answer cannot be scored.

Exclusion reasons (a row can have several; `primary` is the first in this priority order, so counts add up):
  UNLABELED, P1_EVAL (Gate 1 / Gate 2 train+validation / Phase 1 scored question), CONF_EVAL (Gate 2 confirmation question),
  P1_EXEMPLAR, CONF_EXEMPLAR (used as a few-shot exemplar for one of those), then INTERNAL_DUPLICATE (same text hash as an
  earlier row of the same pool, kept first in hash order), then CROSS_POOL_DUPLICATE (same text hash as a row in another
  candidate pool of this inventory that sorts earlier).
A row is excluded on a native-id match within the same benchmark OR a text-hash match anywhere (any benchmark).
"""
import hashlib
import json
from dataclasses import dataclass

AUDIT_RULE_VERSION = "phase3-inventory-v1"
MIN_CLEAN = 300  # eligibility filter only (reviewer: it is not evidence of statistical adequacy)
REASON_ORDER = ("UNLABELED", "P1_EVAL", "CONF_EVAL", "P1_EXEMPLAR", "CONF_EXEMPLAR", "INTERNAL_DUPLICATE",
                "CROSS_POOL_DUPLICATE")
USED_KINDS = ("P1_EVAL", "CONF_EVAL", "P1_EXEMPLAR", "CONF_EXEMPLAR")
DISCORDANCE_GRID = (0.05, 0.10, 0.20)  # illustrative only: no model output is available to the inventory
SALT = "lh-phase3-inventory-v1"


@dataclass(frozen=True)
class Rec:
    benchmark: str  # lm_eval benchmark name, e.g. "hellaswag"
    source_split: str  # split of the dataset the row came from, e.g. "train"
    task: str  # lm_eval (sub)task name, e.g. an MMLU subject
    row: int  # row index within (task, source_split)
    native_id: str | None
    text_hash: str
    labeled: bool = True


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    """Byte-deterministic JSON: sorted keys, fixed separators, LF, trailing newline, no NaN."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def rank_key(benchmark: str, r: Rec) -> int:
    """Deterministic order inside a pool, independent of load order (used for duplicate resolution only)."""
    return int(hashlib.sha256(f"{SALT}|{benchmark}|{r.native_id or r.text_hash}|{r.task}|{r.source_split}".encode()).hexdigest(), 16)


class UsedSet:
    """What earlier stages touched. ids are per benchmark; texts are shared across benchmarks."""

    def __init__(self):
        self.ids: dict[tuple, set] = {}
        self.texts: dict[str, set] = {}
        self.counts = {k: 0 for k in USED_KINDS}

    def add(self, kind: str, benchmark: str, native_id, text_hash: str) -> None:
        if kind not in USED_KINDS:
            raise ValueError(f"unknown used-kind {kind!r}")
        self.counts[kind] += 1
        if native_id is not None:
            self.ids.setdefault((benchmark, native_id), set()).add(kind)
        self.texts.setdefault(text_hash, set()).add(kind)

    def kinds_for(self, benchmark: str, native_id, text_hash: str) -> tuple[set, set]:
        by_id = self.ids.get((benchmark, native_id), set()) if native_id is not None else set()
        return set(by_id), set(self.texts.get(text_hash, set()))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for (b, i), ks in sorted(self.ids.items()):
            h.update(f"id|{b}|{i}|{','.join(sorted(ks))}\n".encode())
        for t, ks in sorted(self.texts.items()):
            h.update(f"tx|{t}|{','.join(sorted(ks))}\n".encode())
        return h.hexdigest()


def pool_fingerprint(recs) -> str:
    """Identity of a pool's raw content: sorted (task, row, native_id, text_hash, labeled)."""
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: (r.task, r.row)):
        h.update(f"{r.task}|{r.row}|{r.native_id}|{r.text_hash}|{int(r.labeled)}\n".encode())
    return h.hexdigest()


def audit_pool(recs: list, used: UsedSet, *, benchmark: str, source_split: str, earlier_pool_texts: set | None = None,
               min_clean: int = MIN_CLEAN) -> dict:
    """Exclusion accounting for one pool. `earlier_pool_texts`: text hashes of clean rows of pools processed before this
    one (cross-pool duplicate rule). Returns a JSON-able dict plus the clean rows under key '_clean' (stripped by build_report).
    """
    earlier = earlier_pool_texts or set()
    any_reason = {k: 0 for k in REASON_ORDER}
    by_id_only = by_text_only = 0
    primary = {k: 0 for k in REASON_ORDER}
    flagged: dict[int, list[str]] = {}
    eligible_after_used = []
    for r in recs:
        reasons = []
        if not r.labeled:
            reasons.append("UNLABELED")
        id_kinds, tx_kinds = used.kinds_for(benchmark, r.native_id, r.text_hash)
        for k in USED_KINDS:
            if k in id_kinds or k in tx_kinds:
                reasons.append(k)
        if id_kinds and not tx_kinds:
            by_id_only += 1
        if tx_kinds and not id_kinds:
            by_text_only += 1
        if reasons:
            flagged[id(r)] = reasons
        else:
            eligible_after_used.append(r)
    # duplicates among rows that survived the used-set and label checks: keep the first in rank order
    seen: set[str] = set()
    clean = []
    for r in sorted(eligible_after_used, key=lambda r: (rank_key(benchmark, r), r.task, r.row)):
        if r.text_hash in seen:
            flagged[id(r)] = ["INTERNAL_DUPLICATE"]
        elif r.text_hash in earlier:
            flagged[id(r)] = ["CROSS_POOL_DUPLICATE"]
        else:
            seen.add(r.text_hash)
            clean.append(r)
    n_dup_native_id = 0
    if clean:
        ids = [r.native_id for r in clean if r.native_id is not None]
        n_dup_native_id = len(ids) - len(set(ids))
    for rs in flagged.values():
        for k in rs:
            any_reason[k] += 1
        primary[next(k for k in REASON_ORDER if k in rs)] += 1
    n = len(recs)
    if sum(primary.values()) + len(clean) != n:  # internal invariant: counts must add up
        raise AssertionError(f"{benchmark}/{source_split}: exclusion counts do not add up")
    return {
        "benchmark": benchmark, "source_split": source_split, "raw": n,
        "excluded_primary": primary, "excluded_any_reason": any_reason,
        "excluded_by_id_only": by_id_only, "excluded_by_text_only": by_text_only,
        "clean": len(clean), "eligible": len(clean) >= min_clean, "min_clean": min_clean,
        "clean_rows_sharing_native_id": n_dup_native_id,
        "pool_fingerprint": pool_fingerprint(recs),
        "clean_set_fingerprint": pool_fingerprint(clean),
        "_clean": clean,
    }


def precision_table(n: int) -> dict:
    """Illustrative precision of a one-time test of size n. No outcome data is used.

    single_accuracy_halfwidth: 95% half-width of a proportion at p=0.5 (worst case).
    paired_difference_halfwidth[d]: 95% half-width of a paired difference when a fraction d of questions is discordant
    between the two policies (se = sqrt(d / n), the largest se for that discordance). The discordance values are assumptions
    for planning only; the real value is measured after the evaluation, never before.
    """
    if n <= 0:
        return {"n": n, "single_accuracy_halfwidth": None, "paired_difference_halfwidth": {}}
    return {"n": n, "single_accuracy_halfwidth": round(1.96 * (0.25 / n) ** 0.5, 4),
            "paired_difference_halfwidth": {str(d): round(1.96 * (d / n) ** 0.5, 4) for d in DISCORDANCE_GRID},
            "note": "eligibility (>= min_clean) is not statistical adequacy; compare these widths with the effect sizes "
                    "the plan must detect (0.03 saving, 0.01 loss) before spending a pool"}


def build_report(*, pools: list[dict], groups: dict, provenance: dict, inputs: dict, used_summary: dict,
                 unresolved: list[dict]) -> dict:
    """Assemble the deterministic report. `pools` are audit_pool results; `groups` maps pool key -> 'in_distribution' | 'new_benchmark'."""
    out_pools = {}
    for p in pools:
        key = f"{p['benchmark']}/{p['source_split']}"
        q = {k: v for k, v in p.items() if k != "_clean"}
        q["group"] = groups[key]
        q["precision"] = precision_table(p["clean"])
        out_pools[key] = q
    unresolved_keys = {u["pool"] for u in unresolved}

    def is_unresolved(key):
        return "*" in unresolved_keys or key in unresolved_keys or key.split("/")[0] + "/*" in unresolved_keys

    summary = {"in_distribution": {}, "new_benchmark": {}}
    for key, q in out_pools.items():
        status = "unresolved" if is_unresolved(key) else ("eligible" if q["eligible"] else "ineligible")
        summary[q["group"]][key] = {"clean": q["clean"], "status": status}
    for key, grp in groups.items():  # pools that could not be enumerated at all
        if key not in out_pools:
            summary[grp][key] = {"clean": None, "status": "unresolved"}
    return {
        "audit_rule_version": AUDIT_RULE_VERSION, "min_clean": MIN_CLEAN,
        "purpose": "Phase 3 fresh-data inventory (inventory only: no model outputs, labels of any model, hidden states, "
                   "thresholds or confirmation predictions were read)",
        "provenance": provenance, "inputs": inputs, "used_set": used_summary,
        "pools": dict(sorted(out_pools.items())),
        "summary": {g: dict(sorted(v.items())) for g, v in summary.items()},
        "unresolved": sorted(unresolved, key=lambda u: (u["pool"], u["reason"])),
    }


# --------------------------------------------------------------------------- repeated-id audit (HellaSwag)

def id_collision_report(rows: list[dict]) -> dict:
    """Are rows excluded only because their native id repeats genuinely the same example, or identifier collisions?

    rows: one dict per pool row with keys row, nid (str|None), text_hash, used_text (text hash is in the used set),
    id_used (native id is in the used set), source_id, ctx_a, activity. Pure function; no dataset access.

    A row is `id_only` when its id matches a used row but its text does not. For each such row the relation to the used
    rows that share its id is graded from strongest to weakest: same_context (ctx_a equal), same_source (source_id
    equal), same_activity, unrelated. `would_be_clean_*` counts are what the pool would hold if the id rule were dropped,
    first as it is, then also excluding rows that share a context or a source with a used row of the same pool.
    Nothing here changes the inventory; any rule change is a reviewer decision.
    """
    ids = [r["nid"] for r in rows if r["nid"] is not None]
    from collections import Counter

    cnt = Counter(ids)
    used_by_id: dict[str, list] = {}
    for r in rows:
        if r["used_text"] and r["nid"] is not None:
            used_by_id.setdefault(r["nid"], []).append(r)
    used_sources = {r["source_id"] for r in rows if r["used_text"] and r["source_id"] not in (None, "")}
    used_ctx = {r["ctx_a"] for r in rows if r["used_text"] and r["ctx_a"] not in (None, "")}
    rel = {"same_context": 0, "same_source": 0, "same_activity": 0, "unrelated": 0, "no_used_partner_in_pool": 0}
    id_only = [r for r in rows if r["id_used"] and not r["used_text"]]
    for r in id_only:
        partners = used_by_id.get(r["nid"], [])
        if not partners:
            rel["no_used_partner_in_pool"] += 1  # the colliding used row lives in another split
        elif any(p["ctx_a"] and p["ctx_a"] == r["ctx_a"] for p in partners):
            rel["same_context"] += 1
        elif any(p["source_id"] not in (None, "") and p["source_id"] == r["source_id"] for p in partners):
            rel["same_source"] += 1
        elif any(p["activity"] and p["activity"] == r["activity"] for p in partners):
            rel["same_activity"] += 1
        else:
            rel["unrelated"] += 1

    def dedup_count(rs):
        return len({r["text_hash"] for r in rs})

    free = [r for r in rows if not r["used_text"]]  # not used by text: the id rule is the only thing excluding id_only rows
    strict = [r for r in free if r["source_id"] not in used_sources and r["ctx_a"] not in used_ctx]
    return {
        "n_rows": len(rows), "n_distinct_native_ids": len(cnt), "n_rows_in_repeated_id_groups": sum(c for c in cnt.values() if c > 1),
        "largest_id_group": max(cnt.values()) if cnt else 0,
        "n_rows_used_by_text": sum(r["used_text"] for r in rows), "n_id_only": len(id_only), "id_only_relation_to_used_partner": rel,
        "n_fresh_rows_sharing_source_with_a_used_row": sum(r["source_id"] in used_sources for r in free),
        "n_fresh_rows_sharing_context_with_a_used_row": sum(r["ctx_a"] in used_ctx for r in free),
        "would_be_clean_if_id_rule_dropped": dedup_count(free),
        "would_be_clean_if_id_rule_dropped_and_shared_source_or_context_excluded": dedup_count(strict),
        "note": "descriptive only; the inventory's exclusion rule is unchanged until the reviewer amends it",
    }
