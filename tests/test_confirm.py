"""Gate 2 confirmation set and contamination audit (docs/GATE2_PLAN.md 3, 3a), on local fake tasks only."""
import importlib.util
import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lookahead import confirm  # noqa: E402
from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.lm_eval_ouro import LoopedOuroLM  # noqa: E402
from lookahead.questions import load_lm_eval_questions, question_set_hash  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")
SPEC = {"shots": 2, "metric": "acc_norm"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mkdoc(n, **extra):
    return {"question": f"What is {n} plus {n}?", "answerKey": "B",
            "choices": {"text": [str(2 * n), f"the number {2 * n + 1}", "x" * (n % 5 + 1), "none of these"],
                        "label": list("ABCD")}, **extra}


def write_task(tmp, name, splits, yaml_extra):
    """A json-backed ARC-style task; splits maps split name -> list of docs."""
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
{yaml_extra}
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
    """fake_arc: ARC-like (Phase 1 evaluates test, exemplars from train, confirmation = validation).
    fake_hs: HellaSwag-like (no test split; Phase 1 evaluates validation, confirmation = train; few-shot split == eval split).
    Injected: an ARC validation doc with a Phase 1 test doc's native id (different text), one with a Phase 1 test doc's text
    (different id), and a HellaSwag train doc with an ARC test doc's text (cross-benchmark)."""
    from lm_eval.tasks import TaskManager

    arc_train = [mkdoc(i, id=f"arc-train-{i}") for i in range(1, 15)]
    arc_test = [mkdoc(i, id=f"arc-test-{i}") for i in range(200, 208)]
    arc_val = [mkdoc(i, id=f"arc-val-{i}") for i in range(100, 112)]
    arc_val[1] = mkdoc(101, id="arc-test-201")  # same native id as a Phase 1 test doc, different text
    arc_val[2] = mkdoc(203, id="arc-val-dup203")  # same text as Phase 1 test doc 203, different id
    hs_train = [mkdoc(i, ind=i, split="train") for i in range(300, 330)]
    hs_train[3] = mkdoc(204, ind=303, split="train")  # text equal to ARC test doc 204: excluded across benchmarks
    hs_val = [mkdoc(i, ind=i, split="val") for i in range(400, 408)]
    write_task(tmp_path, "fake_arc", {"train": arc_train, "validation": arc_val, "test": arc_test}, "test_split: test")
    write_task(tmp_path, "fake_hs", {"train": hs_train, "validation": hs_val}, "")
    # few-shot split == eval split (test): the sampler draws n + 1 and drops the doc itself
    write_task(tmp_path, "fake_same", {"train": arc_train, "validation": arc_val, "test": arc_test},
               "test_split: test\nfewshot_split: test")
    return TaskManager(include_path=str(tmp_path))


@pytest.fixture()
def lm(tok):
    return LoopedOuroLM(None, tok)


PLAN = {"fake_arc": "validation", "fake_hs": "train", "fake_same": "validation"}


def phase1(world, lm, benches=("fake_arc", "fake_hs")):
    out, recs = {}, {}
    for b in benches:
        with confirm.record_exemplars() as rec:
            qs = load_lm_eval_questions(b, lm, task_manager=world, spec=SPEC, on_task=rec.begin_task, on_doc=rec.on_doc)
        out[b], recs[b] = (qs, rec), rec
    return out, recs


# ---------------------------------------------------------------- recording

@pytest.mark.parametrize("bench", ["fake_arc", "fake_same", "fake_hs"])
def test_recording_captures_exactly_the_exemplars_in_each_context(world, lm, bench):
    with confirm.record_exemplars() as rec:
        qs = load_lm_eval_questions(bench, lm, task_manager=world, spec=SPEC, on_task=rec.begin_task, on_doc=rec.on_doc)
    assert len(rec.by_doc_id) == len(qs) and not rec.live_failures
    dropped_extra = same_split_doc_drawn = 0
    for q in qs:
        r = rec.by_doc_id[q.doc_id]
        assert len(r.inserted) == SPEC["shots"]
        assert q.context.count("Answer:") == SPEC["shots"] + 1  # exemplars + the question: nothing else got in
        pos = [q.context.index(d.text) for d in r.inserted]
        assert pos == sorted(pos) and len(set(pos)) == len(pos)
        for d in r.superset:
            if d not in r.inserted:
                if d.doc != r.eval_doc:  # (the eval doc's own text is of course in its context, as the question)
                    assert d.text not in q.context  # drawn but dropped: not in the context, still on record
                dropped_extra += 1
        assert all(d.doc != r.eval_doc for d in r.inserted)
        same_split_doc_drawn += any(d.doc == r.eval_doc for d in r.superset)
        assert r.native_id is not None and r.text_hash
    if bench == "fake_arc":
        assert dropped_extra == 0  # fewshot split != eval split: n drawn, n inserted
    else:
        assert dropped_extra == len(qs)  # n + 1 drawn, the extra one dropped
    if bench == "fake_same":
        assert same_split_doc_drawn > 0  # the eval doc itself was drawn at least once and filtered out
    assert confirm.verbatim_failures(rec, {q.doc_id: q.context for q in qs}) == []


def test_recorder_does_not_change_questions_and_restores_sampler(world, lm):
    from lm_eval.api import samplers

    before = {c: c.__dict__["sample"] for c in set(samplers.SAMPLER_REGISTRY.values()) if "sample" in c.__dict__}
    for bench in ("fake_arc", "fake_hs", "fake_same"):
        plain = load_lm_eval_questions(bench, lm, task_manager=world, spec=SPEC)
        with confirm.record_exemplars() as rec:
            hooked = load_lm_eval_questions(bench, lm, task_manager=world, spec=SPEC, on_task=rec.begin_task,
                                            on_doc=rec.on_doc)
        assert question_set_hash(plain) == question_set_hash(hooked)
        assert [q.context for q in plain] == [q.context for q in hooked]
    assert before == {c: c.__dict__["sample"] for c in before}


def test_zero_shot_records_no_exemplars(world, lm):
    with confirm.record_exemplars() as rec:
        qs = load_lm_eval_questions("fake_arc", lm, task_manager=world, spec={"shots": 0, "metric": "acc"},
                                    on_task=rec.begin_task, on_doc=rec.on_doc)
    assert qs and all(r.inserted == [] and r.superset == [] for r in rec.by_doc_id.values())


def test_verbatim_check_flags_a_tampered_context(world, lm):
    with confirm.record_exemplars() as rec:
        qs = load_lm_eval_questions("fake_arc", lm, task_manager=world, spec=SPEC, on_task=rec.begin_task, on_doc=rec.on_doc)
    ctx = {q.doc_id: q.context for q in qs}
    assert confirm.verbatim_failures(rec, ctx) == []
    victim = qs[2]
    ex = rec.by_doc_id[victim.doc_id].inserted[0]
    ctx[victim.doc_id] = ctx[victim.doc_id].replace(ex.text, "Question: something else\nAnswer:")
    fails = confirm.verbatim_failures(rec, ctx)
    assert [f["doc_id"] for f in fails] == [victim.doc_id]
    del ctx[qs[0].doc_id]
    assert {f["doc_id"] for f in confirm.verbatim_failures(rec, ctx)} == {victim.doc_id, qs[0].doc_id}


def test_native_id_and_text_hash():
    assert confirm.native_id("arc_easy", {"id": "Mercury_1"}) == "Mercury_1"
    assert confirm.native_id("hellaswag", {"ind": 5, "split": "val"}) == "val:5"
    assert confirm.native_id("hellaswag", {"ind": 5, "split": "train"}) != confirm.native_id("hellaswag", {"ind": 5, "split": "val"})
    assert confirm.native_id("mmlu", {"question": "q"}) is None

    class T:
        def doc_to_text(self, d):
            return d["t"]

        def doc_to_choice(self, d):
            return d["c"]

    a = confirm.text_hash("x", {"t": " Hello   World ", "c": ["A b", "c"]}, T())
    assert a == confirm.text_hash("y", {"t": "hello world", "c": ["a  B", "C "]}, T())
    assert a != confirm.text_hash("x", {"t": "hello world", "c": ["a b", "d"]}, T())
    assert a != confirm.text_hash("x", {"t": "hello world a b", "c": ["c"]}, T())  # the separator matters


# ---------------------------------------------------------------- eligibility, selection

def test_eligibility_excludes_by_id_by_text_and_across_benchmarks(world, lm):
    p1, recs = phase1(world, lm)
    ex = confirm.Exclusion.from_recorders(recs)

    arc = confirm.build_confirmation_questions("fake_arc", lm, None, ex, task_manager=world, spec=SPEC, split_plan=PLAN)
    assert arc.stats["pool_size"] == 12 and arc.stats["split"] == "validation"
    assert arc.stats["n_excluded_id"] == 1 and arc.stats["n_excluded_text"] == 1
    assert arc.stats["excluded_id"][0]["item"].endswith("arc-test-201")
    assert arc.stats["excluded_text"][0]["item"].endswith("arc-val-dup203")
    assert "fake_arc eval" in arc.stats["excluded_text"][0]["phase1"][0]
    assert arc.stats["n_eligible"] == arc.stats["n_taken"] == 10
    assert all(q.doc_id.startswith("confirm:fake_arc/arc-val-") for q in arc.questions)
    assert all(q.subtask == "fake_arc" and q.benchmark == "fake_arc" and q.metric == "acc_norm" for q in arc.questions)

    hs = confirm.build_confirmation_questions("fake_hs", lm, None, ex, task_manager=world, spec=SPEC, split_plan=PLAN)
    used = {d.native_id for r in recs["fake_hs"].by_doc_id.values() for d in r.superset + r.inserted}
    assert used and hs.stats["split"] == "train" and hs.stats["pool_size"] == 30
    # excluded by native id: every train doc the sampler drew (including the dropped n+1th), counted from the raw record
    assert hs.stats["n_excluded_id"] == len(used)
    # the doc with ARC test doc 204's text is excluded by text, whatever its id (listed with the Phase 1 item it matches)
    dup = [e for e in hs.stats["excluded_text"] if e["item"].endswith("train:303")]
    assert (not dup and "train:303" in used) or ("fake_arc" in dup[0]["phase1"][0])
    assert "train:303" not in {c.native_id for c in hs.selected}
    sel_ids = {c.native_id for c in hs.selected}
    assert sel_ids and not sel_ids & used
    assert hs.stats["n_eligible"] + len(used) + (0 if "train:303" in used else 1) == 30
    assert hs.stats["n_taken"] == hs.stats["n_eligible"]
    # no selected item shares a text hash with any Phase 1 eval doc or exemplar, in any benchmark
    assert all(c.text_hash not in ex.texts for c in arc.selected + hs.selected)
    assert all(("fake_hs", c.native_id) not in ex.ids for c in hs.selected)


def test_text_collision_across_benchmarks_is_excluded(world, lm):
    """Only the ARC benchmark's Phase 1 is recorded; the HellaSwag-like train doc with ARC test text is still excluded."""
    p1, recs = phase1(world, lm, benches=("fake_arc",))
    ex = confirm.Exclusion.from_recorders(recs)
    hs = confirm.build_confirmation_questions("fake_hs", lm, None, ex, task_manager=world, spec=SPEC, split_plan=PLAN)
    assert hs.stats["n_excluded_id"] == 0  # no HellaSwag-like exemplars on record
    assert hs.stats["n_excluded_text"] == 1 and "fake_arc" in hs.stats["excluded_text"][0]["phase1"][0]
    assert hs.stats["excluded_text"][0]["item"].endswith("train:303")
    assert hs.stats["n_taken"] == 29


def test_selection_is_deterministic_salt_dependent_and_never_self_exemplar(world, lm):
    _, recs = phase1(world, lm)
    ex = confirm.Exclusion.from_recorders(recs)

    def build(salt, n):
        return confirm.build_confirmation_questions("fake_hs", lm, n, ex, salt=salt, task_manager=world, spec=SPEC,
                                                    split_plan=PLAN)

    a, b, c = build("lh-confirm-v1", 4), build("lh-confirm-v1", 4), build("other-salt", 4)
    ids = lambda bd: [q.doc_id for q in bd.questions]  # noqa: E731
    assert ids(a) == ids(b) and len(ids(a)) == 4
    assert [q.context for q in a.questions] == [q.context for q in b.questions]
    assert ids(a) != ids(c)
    full = build("lh-confirm-v1", None)
    assert ids(full)[:4] == ids(a)  # the take is a prefix of the hash order
    # context built for the selected docs only
    assert len(a.recorder.by_doc_id) == 4 and a.stats["n_taken"] == 4
    for q, cand in zip(full.questions, full.selected):
        r = full.recorder.by_doc_id[q.doc_id]
        assert len(r.inserted) == SPEC["shots"]
        assert all(d.doc != cand.doc and d.text_hash != cand.text_hash for d in r.inserted)
    assert full.stats["self_exemplar_violations"] == []
    # the key is documented: sha256 of salt|benchmark|native id (or text hash)
    first = min(full.selected, key=lambda c: c.key)
    assert first is full.selected[0] and first.key == confirm.rank_key("lh-confirm-v1", "fake_hs", first.native_id, first.text_hash)


def test_a_doc_drawn_as_its_own_exemplar_is_dropped_and_replaced(tmp_path, lm):
    """Tiny few-shot pool == eval pool, so the sampler often draws the eval doc itself; it must never reach the context."""
    from lm_eval.tasks import TaskManager

    train = [mkdoc(i, ind=i, split="train") for i in range(1, 7)]
    write_task(tmp_path, "fake_tiny", {"train": train, "validation": [mkdoc(50, ind=50, split="val")]}, "")
    tm = TaskManager(include_path=str(tmp_path))
    bd = confirm.build_confirmation_questions("fake_tiny", lm, None, confirm.Exclusion(), task_manager=tm, spec=SPEC,
                                              split_plan={"fake_tiny": "train"})
    assert len(bd.questions) == 6
    drawn_self = 0
    for q, cand in zip(bd.questions, bd.selected):
        r = bd.recorder.by_doc_id[q.doc_id]
        assert len(r.inserted) == SPEC["shots"] and q.context.count("Answer:") == SPEC["shots"] + 1
        assert all(d.doc != cand.doc for d in r.inserted)
        assert q.context.count(cand.doc["question"]) == 1  # only as the question itself
        drawn_self += any(d.doc == cand.doc for d in r.superset)
    assert drawn_self > 0 and bd.stats["self_exemplar_violations"] == []


def test_internal_duplicates_in_the_pool_are_kept_once(tmp_path, lm):
    from lm_eval.tasks import TaskManager

    docs = [mkdoc(i, id=f"v{i}") for i in range(100, 106)] + [mkdoc(100, id="v-copy")]
    write_task(tmp_path, "fake_dup", {"train": [mkdoc(i, id=f"t{i}") for i in range(1, 8)], "validation": docs,
                                      "test": [mkdoc(200, id="x")]}, "test_split: test")
    tm = TaskManager(include_path=str(tmp_path))
    bd = confirm.build_confirmation_questions("fake_dup", lm, None, confirm.Exclusion(), task_manager=tm, spec=SPEC,
                                              split_plan={"fake_dup": "validation"})
    assert bd.stats["n_internal_duplicates"] == 1 and len(bd.questions) == 6


# ---------------------------------------------------------------- audit

def make_audit(world, lm, **over):
    p1, recs = phase1(world, lm)
    ex = confirm.Exclusion.from_recorders(recs)
    builds = {b: confirm.build_confirmation_questions(b, lm, None, ex, task_manager=world, spec=SPEC, split_plan=PLAN)
              for b in ("fake_arc", "fake_hs")}
    kw = dict(exclusion=ex, manifest_hashes={b: question_set_hash(p1[b][0]) for b in p1}, stored_contexts={},
              phase1_doc_ids={q.doc_id for b in p1 for q in p1[b][0]}, take={"fake_arc": None, "fake_hs": None})
    kw.update(over)
    return p1, builds, ex, kw


def test_audit_passes_on_a_clean_setup(world, lm):
    p1, builds, ex, kw = make_audit(world, lm)
    rep = confirm.audit(p1, builds, **kw)
    assert rep["passed"], rep["failed_checks"]
    assert all(rep["checks"].values()) and rep["failed_checks"] == []
    assert rep["benchmarks"]["fake_arc"]["phase1"]["hash_matches"] is True
    assert rep["benchmarks"]["fake_hs"]["phase1"]["n_exemplar_docs_superset"] >= rep["benchmarks"]["fake_hs"]["phase1"]["n_exemplar_docs_inserted"]
    assert rep["fingerprints"]["lm_eval_version"] and len(rep["fingerprints"]["confirmation_ids_sha256"]) == 64
    json.dumps(rep, default=str)


def test_audit_fails_on_each_injected_defect(world, lm):
    p1, builds, ex, kw = make_audit(world, lm)
    # text overlap with a Phase 1 eval doc
    eval_hash = next(iter(ex.eval_texts))
    builds["fake_arc"].selected[0].text_hash = eval_hash
    rep = confirm.audit(p1, builds, **kw)
    assert not rep["passed"] and {"no_text_overlap", "no_leakage_into_probe_data"} <= set(rep["failed_checks"])
    # identity overlap with a Phase 1 exemplar / eval doc
    p1, builds, ex, kw = make_audit(world, lm)
    builds["fake_arc"].selected[0].native_id = "arc-test-200"
    assert "no_identity_overlap" in confirm.audit(p1, builds, **kw)["failed_checks"]
    # a confirmation doc_id that is also a Phase 1 doc_id
    p1, builds, ex, kw = make_audit(world, lm)
    kw["phase1_doc_ids"] = {builds["fake_arc"].questions[0].doc_id}
    assert "no_leakage_into_probe_data" in confirm.audit(p1, builds, **kw)["failed_checks"]
    # the Phase 1 rebuild no longer reproduces the frozen hash
    p1, builds, ex, kw = make_audit(world, lm, manifest_hashes={"fake_arc": "0" * 64, "fake_hs": None})
    assert confirm.audit(p1, builds, **kw)["failed_checks"] == ["phase1_rebuild_matches_manifest"]
    # tampered stored context: an exemplar is not in what the model saw
    p1, builds, ex, kw = make_audit(world, lm)
    q = p1["fake_arc"][0][2]
    stored = {x.doc_id: x.context for x in p1["fake_arc"][0]}
    stored[q.doc_id] = "Question: nothing\nAnswer: x"
    rep = confirm.audit(p1, builds, **{**kw, "stored_contexts": {"fake_arc": stored}})
    assert rep["failed_checks"] == ["exemplars_verbatim_in_contexts"]
    assert rep["benchmarks"]["fake_arc"]["phase1"]["verbatim_failures"][0]["doc_id"] == q.doc_id
    # own text among exemplars; too few eligible; pool size differs from the plan
    p1, builds, ex, kw = make_audit(world, lm)
    builds["fake_hs"].stats["self_exemplar_violations"] = [{"doc_id": "x"}]
    assert "no_self_exemplar" in confirm.audit(p1, builds, **kw)["failed_checks"]
    p1, builds, ex, kw = make_audit(world, lm, take={"fake_arc": 500, "fake_hs": None})
    builds["fake_arc"].stats["n_requested"] = 500
    assert "enough_eligible" in confirm.audit(p1, builds, **kw)["failed_checks"]
    p1, builds, ex, kw = make_audit(world, lm, expected_pool={"fake_arc": 299})
    assert "pool_sizes_match_plan" in confirm.audit(p1, builds, **kw)["failed_checks"]


def test_task_yaml_fingerprints_include_includes(world, tmp_path):
    (tmp_path / "base.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("include: base.yaml\n", encoding="utf-8")
    seen = set()
    confirm._yaml_closure(tmp_path / "child.yaml", seen)
    assert {p.name for p in seen} == {"child.yaml", "base.yaml"}
    h = confirm.task_yaml_hashes(world, ["fake_arc", "fake_hs"])
    assert set(h) == {"fake_arc.yaml", "fake_hs.yaml"} and all(len(v) == 64 for v in h.values())


# ---------------------------------------------------------------- the script and the confirmation inputs

@pytest.fixture()
def root(tmp_path_factory, world, lm):
    """A miniature repo with Phase 1 frozen for fake_arc and fake_hs."""
    import gzip

    root = tmp_path_factory.mktemp("repo")
    (root / "data" / "questions").mkdir(parents=True)
    (root / "data" / "manifests").mkdir(parents=True)
    manifest, rows = {"benchmarks": {}}, []
    for b in ("fake_arc", "fake_hs"):
        qs = load_lm_eval_questions(b, lm, task_manager=world, spec=SPEC)
        with gzip.open(root / "data" / "questions" / f"{b}.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q.__dict__) + "\n")
        manifest["benchmarks"][b] = {"question_set_sha256": question_set_hash(qs)}
        rows += [f"{b},{q.subtask},{q.doc_id},train" for q in qs]
    (root / "data" / "manifests" / "questions.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "data" / "manifests" / "splits.csv").write_text("benchmark,stratum,doc_id,split\n" + "\n".join(rows) + "\n",
                                                           encoding="utf-8", newline="\n")
    return root


def run_prepare(prep, root, world, lm, *extra, plan=PLAN):
    return prep.main(["--root", str(root), "--benchmarks", "fake_arc", "fake_hs", "--take", *extra], task_manager=world, lm=lm,
                     specs={"fake_arc": SPEC, "fake_hs": SPEC}, split_plan=plan, fetch_revision=lambda repo: "rev-test",
                     expected_pool={})


def test_script_writes_everything_on_pass_and_inputs_check_out(root, world, lm):
    prep = load_script("prepare_confirmation")
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 0
    m = json.loads((root / "data" / "manifests" / "confirmation.json").read_text())
    assert m["benchmarks"]["fake_hs"]["n_questions"] == 5 and m["benchmarks"]["fake_hs"]["source_split"] == "train"
    assert m["benchmarks"]["fake_arc"]["n_questions"] == 10 and m["benchmarks"]["fake_arc"]["hf_revision"] == "rev-test"
    audit = json.loads((root / "results" / "gate2" / "confirmation_audit.json").read_text())
    assert audit["passed"] and audit["fingerprints"]["confirmation_ids_sha256"] == m["ids_sha256"]
    assert audit["fingerprints"]["task_yaml_sha256"] and audit["fingerprints"]["n_take"]["fake_hs"] == 5
    ids = (root / "data" / "manifests" / "confirmation_ids.csv").read_text().splitlines()
    assert ids[0] == "benchmark,stratum,doc_id,split" and len(ids) == 16 and all(r.endswith(",confirmation") for r in ids[1:])
    assert b"\r" not in (root / "data" / "manifests" / "confirmation_ids.csv").read_bytes()
    manifest, sha, splits, questions = confirm.check_confirmation_inputs(root, ["fake_arc", "fake_hs"])
    assert sha == m["ids_sha256"] and len(questions["fake_hs"]) == 5 and len(splits) == 15
    assert all(v[1] == "confirmation" for v in splits.values())
    # the written questions are the audited ones and carry the Phase 1 prompt format
    assert all(q.context.count("Answer:") == SPEC["shots"] + 1 for qs in questions.values() for q in qs)
    # running again reproduces the same ids (fixed before extraction)
    first = (root / "data" / "manifests" / "confirmation_ids.csv").read_text()
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 0
    assert (root / "data" / "manifests" / "confirmation_ids.csv").read_text() == first


def test_script_writes_no_manifest_when_an_overlap_is_injected(root, world, lm, monkeypatch):
    prep = load_script("prepare_confirmation")
    real = prep.build_confirmation_questions

    def leaky(benchmark, lm_, n_take, exclusion, **kw):
        bd = real(benchmark, lm_, n_take, exclusion, **kw)
        bd.selected[0].text_hash = next(iter(exclusion.eval_texts))  # a Phase 1 eval doc sneaks into the selection
        return bd

    monkeypatch.setattr(prep, "build_confirmation_questions", leaky)
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 2
    audit = json.loads((root / "results" / "gate2" / "confirmation_audit.json").read_text())
    assert audit["passed"] is False and "no_text_overlap" in audit["failed_checks"]
    for p in ("confirmation.json", "confirmation_ids.csv", "confirmation_ids.sha256"):
        assert not (root / "data" / "manifests" / p).exists()
    assert not (root / "data" / "questions_confirm").exists()


def test_script_refuses_when_phase1_hash_differs_and_limit_writes_limit_files(root, world, lm):
    prep = load_script("prepare_confirmation")
    qj = root / "data" / "manifests" / "questions.json"
    good = qj.read_text()
    qj.write_text(good.replace(json.loads(good)["benchmarks"]["fake_arc"]["question_set_sha256"], "0" * 64), encoding="utf-8")
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 2
    assert "phase1_rebuild_matches_manifest" in json.loads(
        (root / "results" / "gate2" / "confirmation_audit.json").read_text())["failed_checks"]
    assert not (root / "data" / "manifests" / "confirmation.json").exists()
    # --limit: the comparison is skipped and only *_limitN outputs appear
    assert run_prepare(prep, root, world, lm, "fake_hs=5", "--limit", "3") == 0
    names = {p.name for p in (root / "data" / "manifests").iterdir()}
    assert {"confirmation_limit3.json", "confirmation_ids_limit3.csv", "confirmation_ids_limit3.sha256"} <= names
    assert "confirmation.json" not in names and (root / "results" / "gate2" / "confirmation_audit_limit3.json").exists()
    audit = json.loads((root / "results" / "gate2" / "confirmation_audit_limit3.json").read_text())
    assert audit["benchmarks"]["fake_arc"]["phase1"]["hash_matches"] is None and audit["benchmarks"]["fake_arc"]["phase1"]["n_questions"] == 3
    assert audit["benchmarks"]["fake_hs"]["confirmation"]["n_taken"] == 3


def test_confirmation_input_checks_reject_sha_mismatches(root, world, lm):
    prep = load_script("prepare_confirmation")
    ext = load_script("extract_phase1")
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 0
    mdir = root / "data" / "manifests"
    benches = ["fake_arc", "fake_hs"]
    ext.check_inputs(root, benches, None, "confirmation")  # passes
    ids, ids_sha, man = mdir / "confirmation_ids.csv", mdir / "confirmation_ids.sha256", mdir / "confirmation.json"
    orig = {p: p.read_bytes() for p in (ids, ids_sha, man)}

    def expect_refusal(match):
        with pytest.raises(SystemExit, match=match):
            ext.check_inputs(root, benches, None, "confirmation")
        for p, b in orig.items():
            p.write_bytes(b)

    ids.write_bytes(orig[ids].replace(b"confirmation", b"confirmatioN", 1))  # edited ids file
    expect_refusal("does not match its recorded sha256")
    m = json.loads(orig[man])
    m["ids_sha256"] = "0" * 64  # manifest points at another ids file
    man.write_text(json.dumps(m), encoding="utf-8")
    expect_refusal("does not match its recorded sha256")
    ids_sha.write_text("0" * 64 + "\n", encoding="utf-8")
    expect_refusal("does not match its recorded sha256")
    m = json.loads(orig[man])
    m["audit_sha256"] = "0" * 64
    man.write_text(json.dumps(m), encoding="utf-8")
    expect_refusal("confirmation_audit.json")
    m = json.loads(orig[man])
    m["benchmarks"]["fake_hs"]["question_set_sha256"] = "0" * 64
    man.write_text(json.dumps(m), encoding="utf-8")
    expect_refusal("do not match the manifest hash")
    ext.check_inputs(root, benches, None, "confirmation")  # restored files pass again


def test_extract_and_verify_run_on_the_confirmation_set(root, world, lm, tok, monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM

    import lookahead.ouro_adapter as oa
    from lookahead.questions import load_synthetic_questions

    prep, ext, ver = load_script("prepare_confirmation"), load_script("extract_phase1"), load_script("verify_extraction")
    assert run_prepare(prep, root, world, lm, "fake_hs=5") == 0
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=4,
        layer_types=["full_attention"] * 2, max_position_embeddings=2048)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        model.model.early_exit_gate.weight.normal_(0, 1.0)
    for mod in (ext, ver):
        monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(ver, "results_dir", lambda sub: (root / "results" / sub).mkdir(parents=True, exist_ok=True)
                        or root / "results" / sub)
    monkeypatch.setattr(oa, "load_ouro", lambda device="cuda": (model, tok))
    monkeypatch.setattr(ext, "code_identity", lambda r: {"code_sha256": "test", "git_commit": "test"})
    diag = load_script("diagnose_short_context")
    rows = diag.compare(model, load_synthetic_questions("parity", tok)[:3])
    (root / "results" / "phase1").mkdir(parents=True, exist_ok=True)
    (root / "results" / "phase1" / "diagnose_short_context.json").write_text(
        json.dumps({"float32": {"parity": diag.summarize(rows)}}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["extract_phase1.py", "--set", "confirmation", "--benchmarks", "fake_arc", "fake_hs",
                                      "--chunk", "4"])
    assert ext.main() == 0
    out = root / "data" / "extracted_confirm"
    assert (out / "fake_hs").is_dir() and not (root / "data" / "extracted").exists()
    assert ver.main(["--set", "confirmation", "--benchmarks", "fake_arc", "fake_hs", "--sample", "2"]) == 0
    rep = json.loads((root / "results" / "gate2" / "extraction_check_confirm.json").read_text())
    assert rep["passed"] and rep["out"] == str(out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["runs"][0]["splits_sha256"] == json.loads((root / "data" / "manifests" / "confirmation.json").read_text())["ids_sha256"]


def test_text_hash_is_template_independent_across_benchmarks():
    from lookahead.confirm import text_hash

    class T:  # two tasks with different prompt templates; the hash must not depend on them
        def __init__(self, fmt):
            self.fmt = fmt

        def doc_to_text(self, d):
            return self.fmt.format(d.get("question", d.get("question_stem")))

        def doc_to_choice(self, d):
            return list("ABCD")

    arc = {"id": "a1", "question": "Which  gas do plants absorb?", "choices": {"text": ["O2", "CO2", "N2", "He"]}}
    obqa = {"id": "o9", "question_stem": "which gas do plants absorb?", "choices": {"text": ["o2", "co2", "n2", "he"]}}
    assert text_hash("arc_challenge", arc, T("Question: {}\nAnswer:")) == text_hash("openbookqa", obqa, T("{}"))
    mmlu = {"question": "Which gas do plants absorb?", "choices": ["O2", "CO2", "N2", "He"]}
    assert text_hash("mmlu", mmlu, T("{}\nA.")) == text_hash("openbookqa", obqa, T("{}"))
    hs = {"ctx": "A man is", "endings": ["running.", "sitting."]}
    assert text_hash("hellaswag", hs, T("{}")) != text_hash("openbookqa", obqa, T("{}"))



def test_confirmation_ids_unique_when_native_ids_repeat(tmp_path, lm):
    """HellaSwag's `ind` repeats within train: two different pool docs with one native id must get distinct doc_ids,
    and the collision is counted in the audit stats."""
    from lm_eval.tasks import TaskManager

    train = [mkdoc(i, ind=i, split="train") for i in range(500, 520)]
    train[5] = mkdoc(777, ind=503, split="train")  # different text, same native id as train[3]
    write_task(tmp_path, "fake_rep", {"train": train, "validation": [mkdoc(i, ind=i, split="val") for i in range(600, 604)]},
               "")
    build = confirm.build_confirmation_questions("fake_rep", lm, None, confirm.Exclusion(),
                                                 task_manager=TaskManager(include_path=str(tmp_path)), spec=SPEC,
                                                 split_plan={"fake_rep": "train"})
    ids = [q.doc_id for q in build.questions]
    assert len(ids) == len(set(ids)) == 20
    assert sum(i.startswith("confirm:fake_rep/train:503@r") for i in ids) == 2
    assert build.stats["n_native_id_collisions"] == 2
