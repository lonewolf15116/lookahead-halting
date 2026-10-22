"""Phase 3 fresh-data inventory: synthetic tests only (no downloads, no model, no tokenizer)."""
import hashlib
import importlib.util
import json
import re
import shutil

import pytest

from lookahead import phase3_inventory as inv
from lookahead.phase3_inventory import id_collision_report
from lookahead.config import REPO_ROOT

H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def rec(i, *, split="train", bench="b", nid="auto", text=None, labeled=True, task="t"):
    return inv.Rec(bench, split, task, i, f"id{i}" if nid == "auto" else nid, H(text if text is not None else f"q{i}"), labeled)


def used_with(**kw):
    u = inv.UsedSet()
    for kind, items in kw.items():
        for bench, nid, text in items:
            u.add(kind, bench, nid, H(text))
    return u


# ---------------------------------------------------------------- overlap detection

def test_overlap_by_native_id_is_per_benchmark_and_by_text_is_global():
    u = used_with(P1_EVAL=[("b", "id1", "other text")], CONF_EVAL=[("zzz", None, "q2")])
    pool = [rec(1), rec(2), rec(3), rec(4, nid="id1x")]
    # id match within benchmark b (text differs) -> excluded; text match from another benchmark -> excluded
    r = inv.audit_pool(pool, u, benchmark="b", source_split="train", min_clean=1)
    assert r["clean"] == 2 and r["excluded_primary"]["P1_EVAL"] == 1 and r["excluded_primary"]["CONF_EVAL"] == 1
    assert r["excluded_by_id_only"] == 1 and r["excluded_by_text_only"] == 1
    # the same native id in ANOTHER benchmark's pool is not an id match
    r2 = inv.audit_pool([rec(1, bench="c")], u, benchmark="c", source_split="train", min_clean=1)
    assert r2["clean"] == 1


def test_exemplar_use_excludes_and_is_counted_separately_from_eval_use():
    u = used_with(P1_EXEMPLAR=[("b", "id5", "x5")], CONF_EXEMPLAR=[("b", None, "q6")], P1_EVAL=[("b", "id5", "x5")])
    r = inv.audit_pool([rec(5), rec(6), rec(7)], u, benchmark="b", source_split="train", min_clean=1)
    assert r["clean"] == 1
    assert r["excluded_primary"]["P1_EVAL"] == 1 and r["excluded_primary"]["CONF_EXEMPLAR"] == 1
    assert r["excluded_any_reason"]["P1_EXEMPLAR"] == 1  # id5 is both: counted under each reason, once as primary


def test_unknown_used_kind_is_rejected():
    with pytest.raises(ValueError):
        inv.UsedSet().add("TEST", "b", "1", H("x"))


def test_counts_always_add_up_and_priority_is_fixed():
    u = used_with(P1_EVAL=[("b", "id1", "zz")], CONF_EXEMPLAR=[("b", "id2", "zz2")])
    pool = [rec(1, labeled=False), rec(2), rec(3, text="dup"), rec(4, text="dup"), rec(5)]
    r = inv.audit_pool(pool, u, benchmark="b", source_split="train", min_clean=1)
    assert r["raw"] == 5 and sum(r["excluded_primary"].values()) + r["clean"] == 5
    assert r["excluded_primary"]["UNLABELED"] == 1  # row 1 is unlabeled AND used: the first reason in the order wins
    assert r["excluded_any_reason"]["P1_EVAL"] == 1
    assert r["excluded_primary"]["CONF_EXEMPLAR"] == 1 and r["excluded_primary"]["INTERNAL_DUPLICATE"] == 1
    assert r["clean"] == 2


# ---------------------------------------------------------------- duplicates

def test_internal_duplicates_keep_exactly_one_independent_of_input_order():
    rows = [rec(i, text="same") for i in range(5)] + [rec(9, text="uniq")]
    a = inv.audit_pool(rows, inv.UsedSet(), benchmark="b", source_split="train", min_clean=1)
    b = inv.audit_pool(list(reversed(rows)), inv.UsedSet(), benchmark="b", source_split="train", min_clean=1)
    assert a["clean"] == 2 and a["excluded_primary"]["INTERNAL_DUPLICATE"] == 4
    assert a["clean_set_fingerprint"] == b["clean_set_fingerprint"]
    assert {r.row for r in a["_clean"]} == {r.row for r in b["_clean"]}


def test_cross_pool_duplicate_uses_earlier_clean_pools_only():
    first = inv.audit_pool([rec(1, text="shared"), rec(2)], inv.UsedSet(), benchmark="b", source_split="train", min_clean=1)
    earlier = {r.text_hash for r in first["_clean"]}
    second = inv.audit_pool([rec(10, split="validation", text="shared"), rec(11, split="validation")], inv.UsedSet(),
                            benchmark="b", source_split="validation", earlier_pool_texts=earlier, min_clean=1)
    assert second["clean"] == 1 and second["excluded_primary"]["CROSS_POOL_DUPLICATE"] == 1


def test_duplicate_native_ids_among_clean_rows_are_reported_not_dropped():
    r = inv.audit_pool([rec(1, nid="same"), rec(2, nid="same"), rec(3, nid=None)], inv.UsedSet(), benchmark="b",
                       source_split="train", min_clean=1)
    assert r["clean"] == 3 and r["clean_rows_sharing_native_id"] == 1


# ---------------------------------------------------------------- eligibility and precision

@pytest.mark.parametrize("n,eligible", [(299, False), (300, True), (301, True)])
def test_eligibility_boundary_at_300(n, eligible):
    r = inv.audit_pool([rec(i) for i in range(n)], inv.UsedSet(), benchmark="b", source_split="train")
    assert r["clean"] == n and r["eligible"] is eligible and r["min_clean"] == 300


def test_precision_table_is_illustrative_and_handles_empty():
    p = inv.precision_table(1000)
    assert p["single_accuracy_halfwidth"] == pytest.approx(0.031, abs=1e-3)
    assert p["paired_difference_halfwidth"]["0.1"] == pytest.approx(1.96 * (0.1 / 1000) ** 0.5, abs=1e-4)
    assert "not statistical adequacy" in p["note"]
    assert inv.precision_table(0)["single_accuracy_halfwidth"] is None


# ---------------------------------------------------------------- determinism and report

def test_canonical_json_is_order_independent_and_lf():
    a, b = {"z": 1, "a": {"y": [1, 2], "x": 0.5}}, {"a": {"x": 0.5, "y": [1, 2]}, "z": 1}
    assert inv.canonical_json(a) == inv.canonical_json(b) and "\r" not in inv.canonical_json(a)
    with pytest.raises(ValueError):
        inv.canonical_json({"x": float("nan")})


def test_used_fingerprint_independent_of_insertion_order():
    items = [("b", f"id{i}", f"t{i}") for i in range(6)]
    a, b = used_with(P1_EVAL=items), used_with(P1_EVAL=list(reversed(items)))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != used_with(P1_EXEMPLAR=items).fingerprint()


def _pool(bench, split, n=3, **kw):
    return inv.audit_pool([rec(i, bench=bench, split=split) for i in range(n)], inv.UsedSet(), benchmark=bench,
                          source_split=split, min_clean=1, **kw)


def test_report_marks_unresolved_pools_and_keeps_groups_separate():
    pools = [_pool("a", "train"), _pool("n", "train")]
    groups = {"a/train": "in_distribution", "n/train": "new_benchmark", "m/*": "new_benchmark"}
    base = dict(pools=pools, groups=groups, provenance={}, inputs={}, used_summary={})
    ok = inv.build_report(**base, unresolved=[])
    assert ok["summary"]["in_distribution"]["a/train"]["status"] == "eligible"
    assert list(ok["summary"]["new_benchmark"]) == ["m/*", "n/train"]  # m could not be enumerated
    assert ok["summary"]["new_benchmark"]["m/*"]["status"] == "unresolved"
    one = inv.build_report(**base, unresolved=[{"pool": "n/*", "reason": "r"}])
    assert one["summary"]["new_benchmark"]["n/train"]["status"] == "unresolved"
    assert one["summary"]["in_distribution"]["a/train"]["status"] == "eligible"
    allr = inv.build_report(**base, unresolved=[{"pool": "*", "reason": "r"}])
    assert allr["summary"]["in_distribution"]["a/train"]["status"] == "unresolved"
    assert "\"_clean\"" not in json.dumps(ok)


# ---------------------------------------------------------------- end to end on local fake tasks (stub tokenizer)

lm_eval = pytest.importorskip("lm_eval")
pytest.importorskip("torch")

SPEC = {"shots": 2, "metric": "acc_norm"}
PLAN = {"fake_arc": "validation", "fake_hs": "train"}


def _load_script():
    spec = importlib.util.spec_from_file_location("phase3_inventory_script", REPO_ROOT / "scripts" / "phase3_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mkdoc(n, **extra):
    return {"question": f"What is {n} plus {n}?", "answerKey": "B",
            "choices": {"text": [str(2 * n), f"the number {2 * n + 1}", "x" * (n % 5 + 1), "none of these"], "label": list("ABCD")},
            **extra}


def write_task(tmp, name, splits, extra=""):
    files = {}
    for sp, docs in splits.items():
        p = tmp / f"{name}_{sp}.json"
        p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
        files[sp] = p.as_posix()
    data_files = "\n".join(f'    {sp}: "{p}"' for sp, p in files.items())
    (tmp / f"{name}.yaml").write_text(f"""task: {name}
dataset_path: json
dataset_kwargs:
  data_files:
{data_files}
output_type: multiple_choice
training_split: train
validation_split: validation
{extra}
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{choices.label.index(answerKey)}}}}"
doc_to_choice: "{{{{choices.text}}}}"
metric_list:
  - metric: acc
  - metric: acc_norm
metadata:
  version: 1.0
""", encoding="utf-8")


@pytest.fixture()
def world(tmp_path):
    from lm_eval.tasks import TaskManager

    d = tmp_path / "tasks"
    d.mkdir()
    arc_train = [mkdoc(i, id=f"arc-train-{i}") for i in range(1, 61)]
    arc_test = [mkdoc(i, id=f"arc-test-{i}") for i in range(200, 208)]
    arc_val = [mkdoc(i, id=f"arc-val-{i}") for i in range(100, 112)]
    arc_val[1] = mkdoc(101, id="arc-test-201")  # native id of a Phase 1 test doc, different text
    arc_val[2] = mkdoc(203, id="arc-val-dup203")  # text of a Phase 1 test doc, different id
    hs_train = [mkdoc(i, ind=i, split="train") for i in range(300, 330)]
    hs_val = [mkdoc(i, ind=i, split="val") for i in range(400, 408)]
    new_train = [mkdoc(500, id="n0", answerKey=""),  # no gold answer in the dataset
                 mkdoc(501, id="n1"), mkdoc(501, id="n2"),  # internal duplicate
                 mkdoc(200, id="n3"),  # text of a Phase 1 ARC test doc: used across benchmarks
                 *[mkdoc(510 + i, id=f"n{4 + i}") for i in range(8)]]
    new_val = [mkdoc(600 + i, id=f"nv{i}") for i in range(6)]
    write_task(d, "fake_arc", {"train": arc_train, "validation": arc_val, "test": arc_test}, "test_split: test")
    write_task(d, "fake_hs", {"train": hs_train, "validation": hs_val})
    write_task(d, "fake_new", {"train": new_train, "validation": new_val})
    return TaskManager(include_path=str(d))


class Stub:
    def _encode_pair(self, context, continuation):
        return [0], [0]


def make_root(tmp, world, script, *, revisions=None):
    """A repo-like folder whose manifests are what the Phase 1 / Gate 2 code would have written for the fake world."""
    from lookahead import confirm as C
    from lookahead.questions import load_lm_eval_questions

    root = tmp / "root"
    (root / "data/manifests").mkdir(parents=True)
    (root / "results/gate2").mkdir(parents=True)
    (root / "docs").mkdir()
    rows, qinfo, recs = ["benchmark,stratum,doc_id,split"], {}, {}
    for b in ("fake_arc", "fake_hs"):
        with C.record_exemplars() as rec_:
            qs = load_lm_eval_questions(b, Stub(), task_manager=world, spec=SPEC, on_task=rec_.begin_task, on_doc=rec_.on_doc)
        recs[b] = rec_
        rows += [f"{b},{q.subtask},{q.doc_id},train" for q in qs]
        qinfo[b] = {"n_questions": len(qs)}
    (root / "data/manifests/splits.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    (root / "data/manifests/questions.json").write_text(json.dumps({"benchmarks": qinfo}), encoding="utf-8")
    excl = C.Exclusion.from_recorders(recs)
    builds = {b: C.build_confirmation_questions(b, Stub(), None, excl, salt=C.SALT, task_manager=world, spec=SPEC, split_plan=PLAN)
              for b in ("fake_arc", "fake_hs")}
    ids = C.ids_csv_text(builds)
    (root / "data/manifests/confirmation_ids.csv").write_text(ids, encoding="utf-8", newline="\n")
    (root / "data/manifests/confirmation_ids.sha256").write_text(C.sha256_bytes(ids.encode()) + "\n", encoding="utf-8")
    revs = revisions or {"fake_arc": "revA", "fake_hs": "revH"}
    (root / "data/manifests/confirmation.json").write_text(
        json.dumps({"benchmarks": {b: {"hf_revision": revs[b]} for b in revs}}), encoding="utf-8")
    (root / "results/gate2/confirmation_audit.json").write_text("{}", encoding="utf-8")
    (root / "docs/PHASE3_PLAN.md").write_text("plan\n", encoding="utf-8")
    return root


def run(script, world, root, **kw):
    kwargs = dict(task_manager=world, root=root, specs={"fake_arc": SPEC, "fake_hs": SPEC}, split_plan=PLAN,
                  take={"fake_arc": None, "fake_hs": None}, benchmarks=["fake_arc", "fake_hs"], new_benchmarks=["fake_new"],
                  hf_datasets={"fake_arc": "ds/arc", "fake_hs": "ds/hs", "fake_new": "ds/new"},
                  fetch_revision=lambda repo: {"ds/arc": "revA", "ds/hs": "revH", "ds/new": "revN"}[repo], min_clean=5)
    kwargs.update(kw)
    return script.main([], **kwargs)


def report(root):
    return json.loads((root / "results/phase3/inventory.json").read_text(encoding="utf-8"))


def test_end_to_end_exclusions_counts_and_separation(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    assert run(script, world, root) == 0
    rep = report(root)
    P = rep["pools"]
    # every pool: primary exclusions + clean == raw
    for q in P.values():
        assert sum(q["excluded_primary"].values()) + q["clean"] == q["raw"]
    # used splits are empty after exclusion
    assert P["fake_arc/test"]["raw"] == 8 and P["fake_arc/test"]["clean"] == 0 and P["fake_arc/test"]["excluded_primary"]["P1_EVAL"] == 8
    v = P["fake_arc/validation"]
    assert v["clean"] == 0 and v["excluded_primary"]["P1_EVAL"] == 2 and v["excluded_primary"]["CONF_EVAL"] == 10
    assert v["excluded_by_id_only"] >= 1 and v["excluded_by_text_only"] >= 1  # the two injected collisions
    t = P["fake_arc/train"]
    assert t["raw"] == 60 and 0 < t["clean"] < 60
    assert t["excluded_primary"]["P1_EXEMPLAR"] + t["excluded_primary"]["CONF_EXEMPLAR"] == 60 - t["clean"]
    # new benchmark: unlabeled, internal duplicate and a cross-benchmark text match are excluded; kept in its own group
    n = P["fake_new/train"]
    assert n["group"] == "new_benchmark" and P["fake_arc/train"]["group"] == "in_distribution"
    assert n["excluded_primary"]["UNLABELED"] == 1 and n["excluded_primary"]["INTERNAL_DUPLICATE"] == 1
    assert n["excluded_primary"]["P1_EVAL"] == 1 and n["clean"] == 9  # 12 - unlabeled - duplicate - used
    assert set(rep["summary"]["new_benchmark"]) == {"fake_new/train", "fake_new/validation"}
    assert all(k.startswith("fake_new") for k in rep["summary"]["new_benchmark"])
    assert rep["summary"]["new_benchmark"]["fake_new/train"]["status"] == "eligible"  # min_clean=5 in this test
    assert rep["provenance"]["fewshot_overlap_new_benchmarks"]["fake_new"]["validation"]["source"] == "train"
    assert rep["inputs"]["verification"]["confirmation"]["matches"] is True
    assert all(c["doc_ids_equal_splits_csv"] for c in rep["inputs"]["verification"]["phase1"].values())
    assert rep["unresolved"] == [] and rep["audit_rule_version"] == inv.AUDIT_RULE_VERSION
    # the clean-id file is exactly the clean rows and its hash is in the report
    ids = (root / "results/phase3/inventory_clean_ids.csv").read_text(encoding="utf-8")
    assert len(ids.strip().splitlines()) - 1 == sum(q["clean"] for q in P.values()) == rep["inputs"]["n_clean_ids"]
    assert hashlib.sha256(ids.encode()).hexdigest() == rep["inputs"]["clean_ids_csv_sha256"]
    # no clean id appears among confirmation ids
    conf = (root / "data/manifests/confirmation_ids.csv").read_text(encoding="utf-8")
    assert "arc-test" not in ids and "confirm:" not in ids
    assert "Unresolved contamination risks" in (root / "docs/PHASE3_INVENTORY.md").read_text(encoding="utf-8") and conf


def test_output_is_deterministic_and_inputs_are_untouched(tmp_path, world):
    script = _load_script()
    roots = []
    for k in range(2):
        sub = tmp_path / f"r{k}"
        sub.mkdir()
        roots.append(make_root(sub, world, script))
    before = {p.relative_to(roots[0]): p.read_bytes() for p in roots[0].rglob("*") if p.is_file()}
    assert run(script, world, roots[0]) == 0 and run(script, world, roots[1]) == 0
    for rel in ("results/phase3/inventory.json", "results/phase3/inventory_clean_ids.csv", "docs/PHASE3_INVENTORY.md"):
        assert (roots[0] / rel).read_bytes() == (roots[1] / rel).read_bytes()
    after = {p.relative_to(roots[0]): p.read_bytes() for p in roots[0].rglob("*") if p.is_file()}
    new = set(after) - set(before)
    assert {str(p).replace("\\", "/") for p in new} == {"results/phase3/inventory.json", "results/phase3/inventory_clean_ids.csv",
                                                        "docs/PHASE3_INVENTORY.md"}
    assert all(after[p] == before[p] for p in before)  # every pre-existing file is byte-identical


def test_refuses_to_overwrite_existing_outputs(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    assert run(script, world, root) == 0
    first = (root / "results/phase3/inventory.json").read_bytes()
    assert run(script, world, root) == 2
    assert (root / "results/phase3/inventory.json").read_bytes() == first


def test_phase1_split_mismatch_makes_every_pool_unresolved(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    sp = root / "data/manifests/splits.csv"
    lines = sp.read_text(encoding="utf-8").splitlines()
    sp.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # one Phase 1 doc missing from the committed manifest
    assert run(script, world, root) == 3
    rep = report(root)
    assert rep["unresolved"] and rep["unresolved"][0]["pool"] == "*"
    assert {s["status"] for g in rep["summary"].values() for s in g.values()} == {"unresolved"}


def test_confirmation_identity_mismatch_makes_every_pool_unresolved(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    (root / "data/manifests/confirmation_ids.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    assert run(script, world, root) == 3
    assert {s["status"] for g in report(root)["summary"].values() for s in g.values()} == {"unresolved"}


def test_changed_or_unavailable_revision_marks_only_that_benchmark_unresolved(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    assert run(script, world, root, fetch_revision=lambda r: {"ds/arc": "NEW", "ds/hs": "revH", "ds/new": "unavailable (URLError)"}[r]) == 3
    s = report(root)["summary"]
    assert all(v["status"] == "unresolved" for k, v in s["in_distribution"].items() if k.startswith("fake_arc"))
    assert all(v["status"] != "unresolved" for k, v in s["in_distribution"].items() if k.startswith("fake_hs"))
    assert all(v["status"] == "unresolved" for v in s["new_benchmark"].values())  # revision of a new benchmark not established


def test_unloadable_new_benchmark_is_unresolved_not_clean(tmp_path, world):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    rc = run(script, world, root, new_benchmarks=["fake_missing"], hf_datasets={"fake_arc": "ds/arc", "fake_hs": "ds/hs", "fake_missing": "ds/m"},
             fetch_revision=lambda r: {"ds/arc": "revA", "ds/hs": "revH", "ds/m": "revM"}[r])
    assert rc == 3
    rep = report(root)
    assert any(u["pool"] == "fake_missing/*" and "could not be loaded" in u["reason"] for u in rep["unresolved"])
    assert rep["summary"]["new_benchmark"]["fake_missing/*"]["status"] == "unresolved"


def test_script_never_names_outcome_or_frozen_artifacts():
    """Static guard: the inventory code has no path to model outputs, labels of any model, thresholds or frozen files."""
    banned = ["frozen_selection", "baseline_B", "gate2_confirmation_report", "CONFIRMATION_", "results/gate1", "extracted",
              ".npz", ".npy", "probe", "threshold", "hidden", "targets"]
    for path in (REPO_ROOT / "scripts/phase3_inventory.py", REPO_ROOT / "src/lookahead/phase3_inventory.py"):
        code = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines())
        # docstrings and comments may mention the words in the negative; look only at string literals used as paths
        for lit in re.findall(r"\"([^\"\n]*)\"", code):
            if "/" in lit or "." in lit:
                assert not any(b in lit for b in banned if b not in ("probe", "threshold", "hidden", "targets")), (path.name, lit)
    for bad in ("np.load", "numpy", "load_probe_data", "gate2_eval", "import gate1", "from lookahead import gate"):
        for path in (REPO_ROOT / "scripts/phase3_inventory.py", REPO_ROOT / "src/lookahead/phase3_inventory.py"):
            assert bad not in path.read_text(encoding="utf-8"), (path.name, bad)
    shutil  # (kept: tests above copy nothing, but the import documents that the guard is file-based)


# ---------------------------------------------------------------- regressions from the first real run

def test_winogrande_label_is_read_from_the_answer_field_not_from_doc_to_target():
    script = _load_script()

    class T:  # doc_to_target is the suffix text, as in lm_eval's winogrande: it is never an option index
        def doc_to_target(self, doc):
            return "after the blank"

        def doc_to_choice(self, doc):
            return ["prefix A", "prefix B"]

    assert script.is_labeled(T(), {"answer": "1"}, "winogrande") and script.is_labeled(T(), {"answer": "2"}, "winogrande")
    assert not script.is_labeled(T(), {"answer": ""}, "winogrande")  # the test split ships no answers
    assert not script.is_labeled(T(), {}, "winogrande")


def test_generic_label_check_handles_index_string_and_errors():
    script = _load_script()

    class T:
        def __init__(self, gold, choices=("a", "b")):
            self.gold, self.choices = gold, list(choices)

        def doc_to_target(self, doc):
            if isinstance(self.gold, Exception):
                raise self.gold
            return self.gold

        def doc_to_choice(self, doc):
            return self.choices

    assert script.is_labeled(T(1), {}) and script.is_labeled(T("b"), {})
    assert not script.is_labeled(T(-1), {}) and not script.is_labeled(T(2), {}) and not script.is_labeled(T("z"), {})
    assert not script.is_labeled(T(ValueError("empty label")), {})


def test_one_unloadable_split_is_unresolved_but_the_other_splits_are_still_audited(tmp_path, world, monkeypatch):
    script = _load_script()
    root = make_root(tmp_path, world, script)
    real = script.C.switch_eval_split

    def flaky(task, split):  # like HellaSwag's test split, whose process_docs cannot cast an empty label
        if split == "test" and task.config.task == "fake_arc":
            raise ValueError("invalid literal for int() with base 10: ''")
        return real(task, split)

    monkeypatch.setattr(script.C, "switch_eval_split", flaky)
    assert run(script, world, root) == 3
    rep = report(root)
    assert "fake_arc/test" not in rep["pools"]  # not audited: a partial pool would look clean
    bad = [u for u in rep["unresolved"] if u["pool"] == "fake_arc/test"]
    assert bad and "split could not be loaded" in bad[0]["reason"] and "ValueError" in bad[0]["reason"]
    assert rep["summary"]["in_distribution"]["fake_arc/test"]["status"] == "unresolved"
    assert rep["pools"]["fake_arc/train"]["clean"] > 0 and rep["summary"]["in_distribution"]["fake_arc/train"]["status"] != "unresolved"
    assert [u["pool"] for u in rep["unresolved"]] == ["fake_arc/test"]  # nothing else became unresolved


# ---------------------------------------------------------------- repeated-id audit

def _row(i, nid, *, used=False, id_used=None, src="s", ctx="c", act="a", th=None):
    return {"row": i, "nid": nid, "text_hash": th or f"h{i}", "used_text": used, "id_used": used if id_used is None else id_used,
            "source_id": src, "ctx_a": ctx, "activity": act}


def test_id_collision_report_grades_relations_and_counts_what_would_be_clean():
    rows = [
        _row(0, "7", used=True, src="vid1", ctx="ctx-1", act="act1"),  # the used row
        _row(1, "7", id_used=True, src="vid9", ctx="ctx-9", act="act9"),  # id-only, unrelated to it
        _row(2, "7", id_used=True, src="vid1", ctx="ctx-2", act="actX"),  # id-only, same source
        _row(3, "7", id_used=True, src="vid3", ctx="ctx-1", act="actY"),  # id-only, same context
        _row(4, "7", id_used=True, src="vid4", ctx="ctx-4", act="act1"),  # id-only, same activity
        _row(5, "8", id_used=True, src="vid5", ctx="ctx-5"),  # id collides with a used row in ANOTHER split
        _row(6, "9", src="vid6", ctx="ctx-6"),  # fresh: no id or text hit, no shared source/context
        _row(7, "9", src="vid6", ctx="ctx-6", th="h6"),  # internal text duplicate of row 6
    ]
    r = id_collision_report(rows)
    assert r["n_rows"] == 8 and r["n_distinct_native_ids"] == 3 and r["n_rows_in_repeated_id_groups"] == 7
    assert r["n_id_only"] == 5 and r["n_rows_used_by_text"] == 1
    assert r["id_only_relation_to_used_partner"] == {"same_context": 1, "same_source": 1, "same_activity": 1, "unrelated": 1,
                                                     "no_used_partner_in_pool": 1}
    assert r["would_be_clean_if_id_rule_dropped"] == 6  # rows 1-7 are not used by text, minus one internal duplicate
    # stricter view: rows sharing the used row's source (2) or context (3) are also excluded
    assert r["would_be_clean_if_id_rule_dropped_and_shared_source_or_context_excluded"] == 4
    assert r["n_fresh_rows_sharing_source_with_a_used_row"] == 1 and r["n_fresh_rows_sharing_context_with_a_used_row"] == 1


def test_id_collision_report_handles_missing_ids_and_is_descriptive_only():
    r = id_collision_report([_row(0, None), _row(1, None)])
    assert r["n_distinct_native_ids"] == 0 and r["n_id_only"] == 0 and r["largest_id_group"] == 0
    assert "unchanged" in r["note"]


def test_hellaswag_id_audit_script_end_to_end_on_the_fake_world(tmp_path, world):
    spec = importlib.util.spec_from_file_location("hs_audit", REPO_ROOT / "scripts" / "phase3_hellaswag_id_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    script = _load_script()
    root = make_root(tmp_path, world, script)
    rc = mod.main([], task_manager=world, root=root, specs={"fake_arc": SPEC, "fake_hs": SPEC}, split_plan=PLAN,
                  take={"fake_arc": None, "fake_hs": None}, benchmarks=["fake_arc", "fake_hs"], name="fake_hs")
    assert rc == 0
    rep = json.loads((root / "results/phase3/hellaswag_id_audit.json").read_text(encoding="utf-8"))
    assert rep["pool"] == "fake_hs/train" and rep["n_rows"] == 30 and rep["audit_rule_version"] == inv.AUDIT_RULE_VERSION
    assert rep["verification"]["confirmation"]["matches"] is True
    assert mod.main([], task_manager=world, root=root) == 2  # refuses to overwrite
