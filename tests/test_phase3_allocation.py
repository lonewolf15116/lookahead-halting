"""Phase 3 amendment A1: HellaSwag rule v2 and TEST / DEV+ allocation (synthetic only; no downloads, no model)."""
import hashlib
import importlib.util
import json

import pytest

from lookahead import phase3_allocation as A
from lookahead.config import REPO_ROOT


def hs(i, split="train", *, src="", ctx="", used=False, labeled=True, th=None):
    return {"split": split, "task": "hellaswag", "row": i, "text_hash": th or f"{split}-{i}", "labeled": labeled,
            "source_id": src, "ctx_a": ctx, "_used": used}


def run_v2(rows, other=frozenset()):
    used = {r["text_hash"] for r in rows if r["_used"]}
    return A.hellaswag_v2(rows, used, set(other))


# ---------------------------------------------------------------- rule v2

def test_v2_excludes_shared_source_and_context_across_splits_and_ignores_ids():
    rows = [
        hs(0, src="vidA", used=True),                       # used train row
        hs(1, src="vidA"),                                   # same source as a used train row
        hs(2, src="vidB"),                                   # same source as a used VALIDATION row only
        hs(3, ctx="the same context"),                      # same context as a used row
        hs(4, src="vidC"), hs(5, src="vidC"),               # fresh pair from one source: both clean
        hs(6, labeled=False), hs(7, th="other-pool"), hs(8, th="validation-0"),  # unlabeled; other pool's text; used text
        hs(0, "validation", src="vidB", ctx="the same context", used=True),
    ]
    r = run_v2(rows, other={"other-pool"})
    assert r["n_train_rows"] == 9 and r["n_used_rows_by_split"] == {"train": 2, "validation": 1}  # row 8 repeats a used validation text
    assert r["excluded"] == {"UNLABELED": 1, "USED_TEXT": 2, "SHARED_SOURCE": 2, "SHARED_CONTEXT": 1, "OTHER_POOL_TEXT": 1,
                             "INTERNAL_DUPLICATE": 0}
    assert r["excluded_by_source_only_through_validation_split"] == 1  # row 2: invisible to the within-train audit
    assert r["clean"] == 2 and {x["row"] for x in r["_clean"]} == {4, 5}


def test_v2_keeps_one_of_internal_text_duplicates_deterministically():
    rows = [hs(i, th="dup") for i in range(4)] + [hs(9)]
    a, b = run_v2(rows), run_v2(list(reversed(rows)))
    assert a["clean"] == 2 and a["excluded"]["INTERNAL_DUPLICATE"] == 3
    assert {x["row"] for x in a["_clean"]} == {x["row"] for x in b["_clean"]}


# ---------------------------------------------------------------- allocation

def rows_(n, groups=None):
    return [{"task": "t", "row": i, "text_hash": f"h{i}", "source_id": None if groups is None else groups(i)} for i in range(n)]


def test_allocation_is_disjoint_complete_deterministic_and_order_independent():
    rs = rows_(50)
    a, b = A.allocate(rs, "x", 20), A.allocate(list(reversed(rs)), "x", 20)
    assert a["n_test"] == 20 and a["n_dev"] == 30
    assert [r["row"] for r in a["test"]] == [r["row"] for r in b["test"]]
    assert {r["row"] for r in a["test"]} | {r["row"] for r in a["dev"]} == set(range(50))
    assert A.allocate(rs, "y", 20)["test"] != a["test"]  # the benchmark name is part of the key


def test_grouped_allocation_never_splits_a_source_and_reaches_the_target():
    rs = rows_(60, groups=lambda i: f"src{i // 3}")  # 20 sources of 3 rows each
    a = A.allocate(rs, "hellaswag", 10, "source_id")
    t_src, d_src = {r["source_id"] for r in a["test"]}, {r["source_id"] for r in a["dev"]}
    assert not t_src & d_src and a["n_groups"] == 20
    assert 10 <= a["n_test"] <= 12 and a["n_test"] % 3 == 0  # whole groups: 4 groups = 12 rows
    rs2 = rows_(5, groups=lambda i: "")  # no source id: each row is its own group
    assert A.allocate(rs2, "hellaswag", 2, "source_id")["n_test"] == 2


def test_allocation_refuses_a_pool_smaller_than_the_target():
    with pytest.raises(ValueError, match="< requested TEST"):
        A.allocate(rows_(5), "x", 6)


def test_rank_key_uses_the_frozen_salt():
    assert A.SALT == "lh-phase3-alloc-v1"
    assert A.rank_key("b", "t", 1, "h") == hashlib.sha256(b"lh-phase3-alloc-v1|b|t|1|h").hexdigest()


# ---------------------------------------------------------------- end to end on local fake tasks

pytest.importorskip("lm_eval")
pytest.importorskip("torch")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load(REPO_ROOT / "tests" / "test_phase3_inventory.py", "inv_tests")
TAKE = {"fake_arc": None, "fake_hs": 10}


@pytest.fixture()
def world2(tmp_path):
    from lm_eval.tasks import TaskManager

    d = tmp_path / "tasks"
    d.mkdir()
    arc_train = [T.mkdoc(i, id=f"arc-train-{i}") for i in range(1, 61)]
    arc_test = [T.mkdoc(i, id=f"arc-test-{i}") for i in range(200, 208)]
    arc_val = [T.mkdoc(i, id=f"arc-val-{i}") for i in range(100, 112)]
    hs_train = [T.mkdoc(300 + i, ind=i, split="train", source_id=f"s{i // 2}") for i in range(160)]  # pairs share a source
    hs_val = [T.mkdoc(400 + i, ind=i, split="val", source_id=f"s{15 + i // 2}") for i in range(8)]  # = train rows 30-37
    new_train = [T.mkdoc(500 + i, id=f"n{i}") for i in range(12)]
    new_val = [T.mkdoc(600 + i, id=f"nv{i}") for i in range(6)]
    T.write_task(d, "fake_arc", {"train": arc_train, "validation": arc_val, "test": arc_test}, "test_split: test")
    T.write_task(d, "fake_hs", {"train": hs_train, "validation": hs_val})
    T.write_task(d, "fake_new", {"train": new_train, "validation": new_val})
    return TaskManager(include_path=str(d))


def setup_root(tmp_path, world):
    inv_script = T._load_script()
    root = T.make_root(tmp_path, world, inv_script, take=TAKE)
    assert T.run(inv_script, world, root, take=TAKE) == 0
    return root, hashlib.sha256((root / "results/phase3/inventory.json").read_bytes()).hexdigest()


def alloc(root, world, sha, n_hs):
    mod = _load(REPO_ROOT / "scripts" / "phase3_allocate.py", "alloc_script")
    return mod, mod.main([], task_manager=world, root=root, inventory_sha=sha, hs_name="fake_hs",
                         plan={"fake_new": ("train", 5, None), "fake_hs": ("train", n_hs, "source_id")},
                         ood={"fake_new": "validation"}, benchmarks=["fake_arc", "fake_hs"],
                         specs={"fake_arc": T.SPEC, "fake_hs": T.SPEC}, split_plan=T.PLAN, take=TAKE)


def test_end_to_end_allocation_cross_split_and_lists(tmp_path, world2):
    root, sha = setup_root(tmp_path, world2)
    _, rc = alloc(root, world2, sha, 1)
    assert rc == 0
    rep = json.loads((root / "results/phase3/allocation.json").read_text(encoding="utf-8"))
    h = rep["hellaswag_rule"]
    assert h["clean"] > 0 and sum(h["excluded"].values()) + h["clean"] == 160 and h["n_used_rows_by_split"]["validation"] == 8
    test = (root / "results/phase3/allocation_test_ids.csv").read_text(encoding="utf-8")
    dev = (root / "results/phase3/allocation_dev_ids.csv").read_text(encoding="utf-8")
    assert hashlib.sha256(test.encode()).hexdigest() == rep["test_ids_sha256"]
    assert hashlib.sha256(dev.encode()).hexdigest() == rep["dev_ids_sha256"]
    t_rows = [ln.split(",") for ln in test.splitlines()[1:]]
    d_rows = [ln.split(",") for ln in dev.splitlines()[1:]]
    hs_rows = [r for r in t_rows + d_rows if r[1] == "fake_hs"]
    # rows 30-37 share a source with a used VALIDATION row: never allocated
    assert not {int(r[4]) for r in hs_rows} & set(range(30, 38))
    # TEST and DEV+ share no row and no HellaSwag source
    assert not {tuple(r[1:5]) for r in t_rows} & {tuple(r[1:5]) for r in d_rows}
    assert not {r[7] for r in t_rows if r[1] == "fake_hs"} & {r[7] for r in d_rows if r[1] == "fake_hs"}
    assert rep["extraction_estimate"]["questions"] == len(t_rows) + len(d_rows)
    _, rc2 = alloc(root, world2, sha, 1)
    assert rc2 == 2  # refuses to overwrite


def test_too_small_hellaswag_pool_returns_for_review_without_lists(tmp_path, world2):
    root, sha = setup_root(tmp_path, world2)
    _, rc = alloc(root, world2, sha, 999)
    assert rc == 3
    rep = json.loads((root / "results/phase3/allocation.json").read_text(encoding="utf-8"))
    assert rep["status"] == "RETURN_FOR_REVIEW"
    assert not (root / "results/phase3/allocation_test_ids.csv").exists()


def test_wrong_inventory_hash_is_refused(tmp_path, world2):
    root, _ = setup_root(tmp_path, world2)
    _, rc = alloc(root, world2, "0" * 64, 1)
    assert rc == 2 and not (root / "results/phase3/allocation.json").exists()
