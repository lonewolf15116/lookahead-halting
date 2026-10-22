"""Phase 3 A1: DEV+ / sealed TEST question build and the sealed-path guard (synthetic only; stub tokenizer, no downloads)."""
import hashlib
import importlib.util
import json

import pytest

from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT


# ---------------------------------------------------------------- sealed-path guard

@pytest.mark.parametrize("path", ["data/extracted_phase3_test/hellaswag", r"C:\x\data\QUESTIONS_PHASE3_TEST\a.gz",
                                  "data/extracted_confirm/arc", "data/questions_confirm/x.jsonl.gz"])
def test_guard_refuses_sealed_paths(path):
    with pytest.raises(PermissionError):
        P.assert_not_sealed(path)


@pytest.mark.parametrize("path", ["data/extracted_phase3_dev/hellaswag", "data/extracted/arc_easy", "data/questions/mmlu.jsonl.gz"])
def test_guard_allows_development_paths(path):
    P.assert_not_sealed(path)


def test_sealed_and_development_directories_are_named_so_the_guard_separates_them():
    with pytest.raises(PermissionError):
        P.assert_not_sealed(P.XDIR["test"])
    with pytest.raises(PermissionError):
        P.assert_not_sealed(P.QDIR["test"])
    P.assert_not_sealed(P.XDIR["dev"])
    P.assert_not_sealed(P.QDIR["dev"])


def test_pinned_hashes_are_the_frozen_a1_values():
    a1 = (REPO_ROOT / "docs/PHASE3_AMENDMENT_A1.md").read_text(encoding="utf-8")
    for v in P.PINNED.values():
        assert v in a1
    root = REPO_ROOT
    for name, key in (("allocation.json", "allocation_json"), ("allocation_test_ids.csv", "test_csv"),
                      ("allocation_dev_ids.csv", "dev_csv")):
        assert P._sha_ok(root / "results/phase3" / name, P.PINNED[key])


def test_frozen_allocation_reads_back_with_the_a1_counts():
    alloc = P.read_allocation(REPO_ROOT)
    n = {role: {b: len(rows) for b, (_, rows) in d.items()} for role, d in alloc.items()}
    assert n["test"] == {"openbookqa": 3000, "commonsense_qa": 1000, "hellaswag": 750, "piqa": 1837, "social_iqa": 1930}
    assert n["dev"] == {"openbookqa": 1952, "commonsense_qa": 435, "hellaswag": 248}
    assert alloc["test"]["piqa"][0] == "validation" and alloc["dev"]["hellaswag"][0] == "train"


# ---------------------------------------------------------------- end to end on local fake tasks

pytest.importorskip("lm_eval")
pytest.importorskip("torch")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load(REPO_ROOT / "tests" / "test_phase3_inventory.py", "inv_tests_b")
SPEC = {"shots": 2, "metric": "acc_norm"}
SPECS = {"fake_hs": SPEC, "fake_new": {"shots": 0, "metric": "acc"}}
TEST_ROWS, DEV_ROWS = list(range(0, 10)), list(range(10, 20))


@pytest.fixture()
def world(tmp_path):
    from lm_eval.tasks import TaskManager

    d = tmp_path / "tasks"
    d.mkdir()
    # HellaSwag-like: few-shot exemplars are drawn from train, where the allocated rows live
    T.write_task(d, "fake_hs", {"train": [T.mkdoc(300 + i, ind=i, split="train") for i in range(40)],
                                "validation": [T.mkdoc(400 + i, ind=i, split="val") for i in range(8)]})
    T.write_task(d, "fake_new", {"train": [T.mkdoc(500 + i, id=f"n{i}") for i in range(6)],
                                 "validation": [T.mkdoc(600 + i, id=f"nv{i}") for i in range(6)]})
    return TaskManager(include_path=str(d))


def hashes(world, bench, split):
    from lookahead import confirm as C

    task = world.load([bench])["tasks"][bench]
    C.switch_eval_split(task, split)
    return {i: C.text_hash(bench, doc, task) for i, doc in enumerate(task.eval_docs)}


def make_root(tmp_path, world, *, tamper_hash=False):
    root = tmp_path / "root"
    (root / "results/phase3").mkdir(parents=True)
    (root / "data/manifests").mkdir(parents=True)
    hs, new = hashes(world, "fake_hs", "train"), hashes(world, "fake_new", "validation")
    head = "role,benchmark,source_split,task,row,native_id,text_hash,group"
    test = [f"test,fake_hs,train,fake_hs,{i},,{hs[i]}," for i in TEST_ROWS]
    test += [f"test_ood,fake_new,validation,fake_new,{i},,{new[i]}," for i in range(6)]
    dev = [f"dev,fake_hs,train,fake_hs,{i},,{hs[i]}," for i in DEV_ROWS]
    if tamper_hash:
        dev[0] = dev[0].rsplit(",", 2)[0] + "," + "0" * 64 + ","
    t_text, d_text = "\n".join([head] + test) + "\n", "\n".join([head] + dev) + "\n"
    (root / "results/phase3/allocation_test_ids.csv").write_text(t_text, encoding="utf-8", newline="\n")
    (root / "results/phase3/allocation_dev_ids.csv").write_text(d_text, encoding="utf-8", newline="\n")
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    a = json.dumps({"test_ids_sha256": sha(t_text), "dev_ids_sha256": sha(d_text)})
    (root / "results/phase3/allocation.json").write_text(a, encoding="utf-8", newline="\n")
    (root / "results/phase3/inventory.json").write_text(json.dumps({"provenance": {"revisions": {
        "fake_hs": {"hf_dataset": "ds/hs", "hf_revision": "r1"}, "fake_new": {"hf_dataset": "ds/new", "hf_revision": "r2"}}}}),
        encoding="utf-8")
    pinned = {"allocation_json": sha(a), "test_csv": sha(t_text), "dev_csv": sha(d_text)}
    return root, pinned, hs


REV = {"ds/hs": "r1", "ds/new": "r2"}


def run(root, world, pinned, rev=REV):
    mod = _load(REPO_ROOT / "scripts" / "prepare_phase3.py", "prep3")
    return mod.main([], task_manager=world, lm=T.Stub(), root=root, pinned=pinned, specs=SPECS,
                    fetch_revision=lambda r: rev[r], ood=("fake_new",))


def test_build_strips_allocated_rows_from_the_fewshot_pool_and_writes_frozen_sets(tmp_path, world):
    root, pinned, hs = make_root(tmp_path, world)
    assert run(root, world, pinned) == 0
    man = json.loads((root / P.MANIFEST).read_text(encoding="utf-8"))
    rep = json.loads((root / "results/phase3/build_report.json").read_text(encoding="utf-8"))
    assert set(man["sets"]) == {"dev/fake_hs", "test/fake_hs", "test/fake_new"}
    assert man["sets"]["test/fake_new"]["split_label"] == "test_ood" and man["sets"]["test/fake_new"]["shots"] == 0
    for k in ("dev/fake_hs", "test/fake_hs"):
        r = rep["sets"][k]
        assert r["exemplar_is_allocated_row"] == 0 and r["self_exemplar"] == 0 and r["live_exemplar_failures"] == 0
        assert r["pool_removed"] == 20 and r["pool_size_after"] == 20 and r["exemplar_draws"] == 2 * 10
    # independent check on the built prompts: no allocated question's text occurs in another question's context
    from lookahead.extract import load_questions

    allocated = {f"What is {300 + i} plus {300 + i}?" for i in TEST_ROWS + DEV_ROWS}
    for role in ("dev", "test"):
        for q in load_questions(root / "data" / P.QDIR[role] / "fake_hs.jsonl.gz"):
            own = f"What is {300 + int(q.doc_id.rsplit('@r', 1)[1])} plus"
            others = [t for t in allocated if t in q.context and not t.startswith(own)]
            assert not others, (q.doc_id, others)
            assert q.doc_id.startswith("p3:fake_hs/train@r")
    dev_ids = (root / P.IDS["dev"]).read_text(encoding="utf-8").splitlines()
    test_ids = (root / P.IDS["test"]).read_text(encoding="utf-8").splitlines()
    assert dev_ids[0] == test_ids[0] == "benchmark,stratum,doc_id,split"
    assert len(dev_ids) == 1 + 10 and all(ln.endswith(",dev") for ln in dev_ids[1:])
    assert len(test_ids) == 1 + 10 + 6 and sum(ln.endswith(",test_ood") for ln in test_ids) == 6
    # extraction-side frozen-input check accepts it, per role
    _, _, splits, qs = P.check_phase3_inputs(root, ["fake_hs"], None, "dev", allocation_sha=pinned["allocation_json"])
    assert len(qs["fake_hs"]) == 10 and {v[1] for v in splits.values()} == {"dev"}
    _, _, splits_t, qs_t = P.check_phase3_inputs(root, ["fake_hs", "fake_new"], None, "test", allocation_sha=pinned["allocation_json"])
    assert len(qs_t["fake_new"]) == 6 and {v[1] for v in splits_t.values()} == {"test", "test_ood"}
    with pytest.raises(SystemExit):  # a manifest built from another allocation is refused
        P.check_phase3_inputs(root, ["fake_hs"], None, "dev")
    assert run(root, world, pinned) == 2  # refuses to overwrite


def test_build_is_deterministic(tmp_path, world):
    outs = []
    for k in range(2):
        sub = tmp_path / f"r{k}"
        sub.mkdir()
        root, pinned, _ = make_root(sub, world)
        assert run(root, world, pinned) == 0
        outs.append(((root / P.MANIFEST).read_bytes(), (root / P.IDS["dev"]).read_bytes(), (root / P.IDS["test"]).read_bytes()))
    assert outs[0] == outs[1]


def test_tampered_question_file_is_refused_at_extraction(tmp_path, world):
    root, pinned, _ = make_root(tmp_path, world)
    assert run(root, world, pinned) == 0
    from lookahead import confirm as C
    from lookahead.extract import load_questions

    p = root / "data" / P.QDIR["dev"] / "fake_hs.jsonl.gz"
    qs = load_questions(p)
    qs[0].target = (qs[0].target + 1) % len(qs[0].cont_encs)
    p.unlink()
    C.write_questions(p, qs)
    with pytest.raises(SystemExit, match="do not match the manifest"):
        P.check_phase3_inputs(root, ["fake_hs"], None, "dev", allocation_sha=pinned["allocation_json"])


def test_revision_change_writes_report_only(tmp_path, world):
    root, pinned, _ = make_root(tmp_path, world)
    assert run(root, world, pinned, rev={"ds/hs": "CHANGED", "ds/new": "r2"}) == 2
    rep = json.loads((root / "results/phase3/build_report.json").read_text(encoding="utf-8"))
    assert not rep["passed"] and "revision" in rep["failures"][0]
    assert not (root / P.MANIFEST).exists() and not (root / "data" / P.QDIR["test"]).exists()


def test_allocation_hash_mismatch_and_text_hash_mismatch_are_refused(tmp_path, world):
    root, pinned, _ = make_root(tmp_path, world)
    with pytest.raises(SystemExit, match="frozen"):
        run(root, world, {**pinned, "test_csv": "0" * 64})
    sub = tmp_path / "t"
    sub.mkdir()
    root2, pinned2, _ = make_root(sub, world, tamper_hash=True)
    with pytest.raises(RuntimeError, match="text hash differs"):
        run(root2, world, pinned2)


def test_extractor_routes_phase3_sets_through_the_frozen_check(tmp_path, world):
    """extract_phase1.check_inputs must use check_phase3_inputs with the PINNED A1 allocation: a fixture built from another
    allocation is refused, for both roles, before any model is loaded."""
    root, pinned, _ = make_root(tmp_path, world)
    assert run(root, world, pinned) == 0
    ex = _load(REPO_ROOT / "scripts" / "extract_phase1.py", "extract_p3")
    assert {"phase3_dev", "phase3_test"} <= set(ex.SETS)
    for which in ("phase3_dev", "phase3_test"):
        with pytest.raises(SystemExit, match="frozen A1 allocation"):
            ex.check_inputs(root, ["fake_hs"], None, which)
    ver = (REPO_ROOT / "scripts" / "verify_extraction.py").read_text(encoding="utf-8")
    assert '"phase3_dev", "phase3_test"' in ver and "check_phase3_inputs(REPO_ROOT" in ver
