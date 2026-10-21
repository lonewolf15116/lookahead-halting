"""Gate 2 confirmation set and its contamination audit (docs/GATE2_PLAN.md sections 3 and 3a).

Phase 1 questions are rebuilt through `load_lm_eval_questions` with lm_eval's few-shot sampler instrumented, so every
exemplar that really went into a context is on record. Confirmation questions are then drawn from the labelled pool Phase 1
never scored, excluding anything Phase 1 touched (as an eval doc or an exemplar) by native id or by normalised text, and are
built with the same prompt format, shots and seed. Written against lm_eval 0.4.13 (requirements.txt): its sampler is
`ContextSampler.sample(n, eval_doc=...)`, which draws n + 1 and drops eval_doc only when the few-shot split equals the test
split (HellaSwag and CommonsenseQA, where both are None).
"""
import functools
import gzip
import hashlib
import json
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lookahead.questions import BENCHMARKS, FEWSHOT_SEED, Question, load_lm_eval_questions, question_set_hash  # noqa: F401

SALT = "lh-confirm-v1"
# Eval split of the confirmation set per benchmark: the labelled split Phase 1 did not score (plan section 3).
SPLIT_PLAN = {"arc_challenge": "validation", "arc_easy": "validation", "mmlu": "validation", "openbookqa": "validation",
              "hellaswag": "train", "commonsense_qa": "train"}
DEFAULT_TAKE = {"hellaswag": 1000, "commonsense_qa": 1000}  # others: all
EXPECTED_POOL = {"arc_challenge": 299, "arc_easy": 570, "mmlu": 1531, "openbookqa": 500, "hellaswag": 39905,
                 "commonsense_qa": 9741}  # plan section 3; checked against the loaded split at run time
HF_DATASETS = {"arc_challenge": "allenai/ai2_arc", "arc_easy": "allenai/ai2_arc", "hellaswag": "Rowan/hellaswag",
               "mmlu": "cais/mmlu", "commonsense_qa": "tau/commonsense_qa", "openbookqa": "allenai/openbookqa"}
NATIVE_ID_KIND = {"arc_challenge": "id", "arc_easy": "id", "openbookqa": "id", "commonsense_qa": "id",
                  "hellaswag": "hellaswag", "mmlu": None}  # unknown benchmark names (tests): "auto"
MAX_LISTED = 50  # collisions listed in the audit; the counts are always complete
UNIT = "\x1f"

Desc = namedtuple("Desc", "native_id text_hash text doc")
Call = namedtuple("Call", "eval_doc inserted raw")


# --------------------------------------------------------------------------- identities

def native_id(benchmark, doc, split=None):
    """Dataset-native id of a doc, or None (MMLU: text hash only). The split is part of the HellaSwag id."""
    kind = NATIVE_ID_KIND.get(benchmark, "auto")
    if kind == "auto":
        kind = "hellaswag" if "ind" in doc else ("id" if "id" in doc else None)
    if kind == "id":
        return None if doc.get("id") is None else str(doc["id"])
    if kind == "hellaswag":
        if doc.get("ind") is None:
            return None
        return f"{doc.get('split') or split}:{doc['ind']}"
    return None


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def _raw_stem_choices(doc):
    """Template-independent question content from the dataset's own fields, or None if the doc has none of them.

    ARC/CSQA: question + choices.text; OpenBookQA: question_stem + choices.text; MMLU: question + choices (list);
    HellaSwag: ctx + endings. Using raw fields (not doc_to_text, which adds each benchmark's prompt template, nor
    doc_to_choice, which is letters for CSQA/MMLU) lets the same question be caught across benchmarks.
    """
    stem = next((doc[k] for k in ("question_stem", "question", "ctx") if isinstance(doc.get(k), str)), None)
    ch = doc.get("choices", doc.get("endings"))
    if isinstance(ch, dict):
        ch = ch.get("text")
    if stem is None or not isinstance(ch, (list, tuple)) or not all(isinstance(c, str) for c in ch):
        return None
    return stem, list(ch)


def text_hash(benchmark, doc, task) -> str:
    """sha256 of normalised question content and choices, independent of the benchmark's prompt template.

    Falls back to the task's own doc_to_text / doc_to_choice when the doc has no recognised raw fields.
    """
    raw = _raw_stem_choices(doc)
    if raw is not None:
        parts = [_norm(raw[0])] + [_norm(c) for c in raw[1]]
    else:
        parts = [_norm(task.doc_to_text(doc))] + [_norm(c) for c in task.doc_to_choice(doc)]
    return hashlib.sha256(UNIT.join(parts).encode("utf-8")).hexdigest()


def _desc(cache, benchmark, task, doc) -> Desc:
    d = cache.get(id(doc))
    if d is None:  # the Desc keeps doc alive, so its id stays unique
        d = Desc(native_id(benchmark, doc), text_hash(benchmark, doc, task), str(task.doc_to_text(doc)).strip(), doc)
        cache[id(doc)] = d
    return d


# --------------------------------------------------------------------------- recording lm_eval's sampler

class _RecordingRandom:
    def __init__(self, real, draws):
        self._real, self._draws = real, draws

    def sample(self, population, k, **kw):
        res = self._real.sample(population, k, **kw)
        self._draws.extend(res)
        return res

    def __getattr__(self, name):
        return getattr(self._real, name)


@dataclass
class QRecord:
    doc_id: str  # as questions.py forms it: f"{task_name}/{doc_id}"
    task_name: str
    native_id: str | None
    text_hash: str
    eval_doc: dict
    inserted: list  # Desc of the exemplars that ended up in the context, in order
    superset: list  # Desc of every doc the sampler drew (includes the extra n+1 draw that was dropped)


class ExemplarRecorder:
    """Collects, per question, the few-shot exemplars lm_eval inserted. Use via `record_exemplars()`.

    Pass `begin_task` / `on_doc` to load_lm_eval_questions as on_task / on_doc. lm_eval calls sampler.sample exactly once per
    eval doc with shots > 0, in doc order, so call k belongs to the doc built k-th; this is cross-checked against the doc.
    """

    def __init__(self):
        self.by_doc_id: dict[str, QRecord] = {}
        self.live_failures: list[dict] = []  # exemplar text missing from the context it was drawn for
        self.task_names: list[str] = []
        self._sampler = None
        self._calls: list[Call] = []
        self._cache: dict = {}

    def begin_task(self, task):
        self._sampler = getattr(task, "sampler", None)
        self._calls = []
        self.task_names.append(task.config.task)

    def on_doc(self, q: Question, task, doc):
        idx = int(q.doc_id.rsplit("/", 1)[1])
        shots = task.config.num_fewshot or 0
        if shots > 0:
            if idx >= len(self._calls):
                raise RuntimeError(f"{q.doc_id}: no sampler call recorded (calls: {len(self._calls)})")
            call = self._calls[idx]
            if call.eval_doc is not None and call.eval_doc != doc:
                raise RuntimeError(f"{q.doc_id}: sampler call {idx} was made for a different doc")
            inserted = [_desc(self._cache, q.benchmark, task, d) for d in call.inserted]
            raw = [_desc(self._cache, q.benchmark, task, d) for d in call.raw]
        else:
            inserted, raw = [], []
        rec = QRecord(q.doc_id, q.subtask, native_id(q.benchmark, doc), text_hash(q.benchmark, doc, task), doc, inserted,
                      raw)
        if q.doc_id in self.by_doc_id:
            raise RuntimeError(f"{q.doc_id}: recorded twice; use a fresh recorder per rebuild")
        self.by_doc_id[q.doc_id] = rec
        for d in inserted:
            if d.text not in q.context:
                self.live_failures.append({"doc_id": q.doc_id, "exemplar": d.native_id or d.text_hash[:16]})

    def rekey(self, old: str, new: str):
        rec = self.by_doc_id.pop(old)
        rec.doc_id = new
        self.by_doc_id[new] = rec


@contextmanager
def record_exemplars():
    """Instrument lm_eval's few-shot samplers while Phase 1 (or confirmation) questions are built; yields the recorder."""
    from lm_eval.api import samplers

    rec = ExemplarRecorder()
    patched = []

    def wrap(orig):
        @functools.wraps(orig)
        def sample(self, *args, **kwargs):
            if rec._sampler is not self:
                return orig(self, *args, **kwargs)
            draws, real = [], getattr(self, "rnd", None)
            if real is not None:
                self.rnd = _RecordingRandom(real, draws)
            try:
                res = orig(self, *args, **kwargs)
            finally:
                if real is not None:
                    self.rnd = real
            rec._calls.append(Call(kwargs.get("eval_doc", args[1] if len(args) > 1 else None), list(res),
                                   list(draws) or list(res)))
            return res
        return sample

    for cls in set(samplers.SAMPLER_REGISTRY.values()):
        if "sample" in cls.__dict__:
            patched.append((cls, cls.__dict__["sample"]))
            cls.sample = wrap(cls.__dict__["sample"])
    try:
        yield rec
    finally:
        for cls, orig in patched:
            cls.sample = orig


def verbatim_failures(rec: ExemplarRecorder, contexts: dict) -> list[dict]:
    """Every recorded exemplar's doc_to_text must occur verbatim in the stored context of its question."""
    out = []
    for doc_id, r in rec.by_doc_id.items():
        ctx = contexts.get(doc_id)
        if ctx is None:
            out.append({"doc_id": doc_id, "problem": "no stored context"})
            continue
        for d in r.inserted:
            if d.text not in ctx:
                out.append({"doc_id": doc_id, "exemplar": d.native_id or d.text_hash[:16], "problem": "not in context"})
    return out


# --------------------------------------------------------------------------- exclusion sets

class Exclusion:
    """Everything Phase 1 touched: native ids per benchmark, normalised text hashes across all benchmarks."""

    def __init__(self):
        self.ids: dict[tuple, list] = {}
        self.texts: dict[str, list] = {}
        self.eval_texts: set[str] = set()

    def add(self, benchmark, label, nid, thash, is_eval=False):
        if nid is not None:
            self.ids.setdefault((benchmark, nid), [])
            if len(self.ids[(benchmark, nid)]) < 3:
                self.ids[(benchmark, nid)].append(label)
        self.texts.setdefault(thash, [])
        if len(self.texts[thash]) < 3:
            self.texts[thash].append(label)
        if is_eval:
            self.eval_texts.add(thash)

    @classmethod
    def from_recorders(cls, recorders: dict) -> "Exclusion":
        ex = cls()
        for b, rec in recorders.items():
            for r in rec.by_doc_id.values():
                ex.add(b, f"{b} eval {r.doc_id}", r.native_id, r.text_hash, is_eval=True)
                for d in r.superset + r.inserted:  # superset is conservative: it includes the dropped extra draw
                    ex.add(b, f"{b} exemplar of {r.doc_id}", d.native_id, d.text_hash)
        return ex

    def hit(self, benchmark, nid, thash):
        """(id labels or None, text labels or None)"""
        return (self.ids.get((benchmark, nid)) if nid is not None else None), self.texts.get(thash)


# --------------------------------------------------------------------------- confirmation questions

@dataclass
class Cand:
    task_name: str
    index: int
    doc: dict
    native_id: str | None
    text_hash: str
    key: int


@dataclass
class ConfirmBuild:
    benchmark: str
    split: str
    questions: list
    selected: list  # Cand, in the same order as questions
    recorder: ExemplarRecorder
    stats: dict = field(default_factory=dict)
    task_names: list = field(default_factory=list)


def switch_eval_split(task, split: str) -> None:
    """Point the task's eval docs at `split`, touching nothing else (the few-shot pool, format and sampler stay).

    A task with a test split evaluates on it, so that is switched; otherwise (HellaSwag, CommonsenseQA) it evaluates on
    the validation split and that is switched. The test-split comparison that decides whether lm_eval passes eval_doc to
    the sampler is unchanged for the Phase 1 pattern in both cases.
    """
    task.set_config("test_split" if task.config.test_split is not None else "validation_split", split)
    if split not in task.dataset:
        raise ValueError(f"{task.config.task}: dataset has no split {split!r}")


def rank_key(salt, benchmark, nid, thash) -> int:
    return int(hashlib.sha256(f"{salt}|{benchmark}|{nid or thash}".encode("utf-8")).hexdigest(), 16)


def build_confirmation_questions(benchmark, lm, n_take, phase1_exclusion: Exclusion, salt=SALT, task_manager=None, spec=None,
                                 split_plan=None) -> ConfirmBuild:
    from lm_eval.tasks import TaskManager

    spec = spec or BENCHMARKS[benchmark]
    split = (split_plan or SPLIT_PLAN)[benchmark]
    tm = task_manager or TaskManager()

    # 1. the pool: every doc of the unused split, with identities, ranked
    pool, stats = {}, {"split": split, "pool_size": 0, "n_excluded_id": 0, "n_excluded_text": 0,
                       "excluded_id": [], "excluded_text": [], "n_internal_duplicates": 0}
    eligible = []
    loaded = tm.load([benchmark])["tasks"]
    for task_name in sorted(loaded):
        task = loaded[task_name]
        switch_eval_split(task, split)
        for i, doc in enumerate(task.eval_docs):
            nid, th = native_id(benchmark, doc, split), text_hash(benchmark, doc, task)
            c = Cand(task_name, i, doc, nid, th, rank_key(salt, benchmark, nid, th))
            stats["pool_size"] += 1
            id_hit, text_hit = phase1_exclusion.hit(benchmark, nid, th)
            if id_hit:
                stats["n_excluded_id"] += 1
                stats["excluded_id"].append({"item": f"{task_name}/{nid}", "phase1": id_hit})
            if text_hit:
                stats["n_excluded_text"] += 1
                stats["excluded_text"].append({"item": f"{task_name}/{nid or th[:16]}", "phase1": text_hit})
            if not id_hit and not text_hit:
                eligible.append(c)
    for k in ("excluded_id", "excluded_text"):
        stats["n_" + k + "_listed"] = min(len(stats[k]), MAX_LISTED)
        stats[k] = stats[k][:MAX_LISTED]
    eligible.sort(key=lambda c: (c.key, c.task_name, c.index))
    seen, unique = set(), []
    for c in eligible:  # two identical pool items would share a doc_id; keep the first in hash order
        if c.text_hash in seen:
            stats["n_internal_duplicates"] += 1
        else:
            seen.add(c.text_hash)
            unique.append(c)
    nid_counts = {}
    for c in unique:
        if c.native_id is not None:
            nid_counts[c.native_id] = nid_counts.get(c.native_id, 0) + 1
    stats["n_native_id_collisions"] = sum(n for n in nid_counts.values() if n > 1)  # docs sharing a native id
    stats["n_eligible"] = len(unique)
    stats["n_requested"] = n_take
    selected = unique if n_take is None else unique[:n_take]
    stats["n_taken"] = len(selected)

    # 2. contexts for the selected docs only (lm_eval's `samples=` skips every other doc)
    by_task = {}
    for c in selected:
        by_task.setdefault(c.task_name, []).append(c)
    index_of = {t: sorted(cs, key=lambda c: c.index) for t, cs in by_task.items()}
    violations = []
    renamed = {}
    built = []
    with record_exemplars() as rec:
        def on_task(task):
            switch_eval_split(task, split)
            rec.begin_task(task)

        def on_doc(q, task, doc):
            cand = index_of[q.subtask][int(q.doc_id.rsplit("/", 1)[1])]  # lm_eval renumbers built docs 0..k-1 in order
            if cand.doc != doc:
                raise RuntimeError(f"{q.doc_id}: built doc is not the selected pool doc {cand.task_name}[{cand.index}]")
            rec.on_doc(q, task, doc)
            r = rec.by_doc_id[q.doc_id]
            if any(d.doc == doc for d in r.inserted):
                raise AssertionError(f"{q.doc_id}: the question is among its own few-shot exemplars")
            if any(d.text_hash == cand.text_hash for d in r.inserted):
                violations.append({"doc_id": q.doc_id, "problem": "own text among exemplars under another id"})
            # native ids are not always unique (HellaSwag `ind` repeats within train), so the row index in the
            # source split is appended; the native id (or text hash) stays in the id for traceability
            new = f"confirm:{cand.task_name}/{cand.native_id or cand.text_hash[:16]}@r{cand.index}"
            rec.rekey(q.doc_id, new)
            renamed[new] = cand
            built.append((new, cand))  # kept as a list: a repeated id must not hide which docs shared it
            q.doc_id = new

        qs = load_lm_eval_questions(benchmark, lm, task_manager=tm, spec=spec, on_task=on_task, on_doc=on_doc,
                                    samples_for=lambda name: [c.index for c in by_task.get(name, [])])
        task_names = list(rec.task_names)
    order = {id(c): n for n, c in enumerate(selected)}
    qs.sort(key=lambda q: order[id(renamed[q.doc_id])])
    if len(qs) != len(selected) or len({q.doc_id for q in qs}) != len(qs):
        groups = {}
        for new_id, c in built:
            groups.setdefault(new_id, []).append(f"{c.task_name}[row {c.index}] native_id={c.native_id}")
        dups = {k: v for k, v in groups.items() if len(v) > 1}
        raise RuntimeError(f"{benchmark}: built {len(qs)} questions for {len(selected)} selected docs, or ids not unique; "
                           f"duplicate doc_id groups: {dups}")
    for q in qs:
        if "," in q.doc_id or "\n" in q.doc_id:
            raise ValueError(f"unsafe doc_id {q.doc_id!r}")
    stats["self_exemplar_violations"] = violations
    return ConfirmBuild(benchmark, split, qs, [renamed[q.doc_id] for q in qs], rec, stats, task_names)


# --------------------------------------------------------------------------- fingerprints

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p) -> str:
    return sha256_bytes(Path(p).read_bytes().replace(b"\r\n", b"\n"))


def _yaml_closure(path: Path, seen: set):
    import re

    path = Path(path)
    if path in seen or not path.exists():
        return
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"^include:\s*(\S+)", text, re.M):
        _yaml_closure(path.parent / m.group(1), seen)
    for m in re.finditer(r"!function\s+([\w.]+)\.\w+", text):
        mod = path.parent / (m.group(1).replace(".", "/") + ".py")
        if mod.exists():
            seen.add(mod)


def task_yaml_hashes(tm, task_names) -> dict:
    """sha256 of every task YAML used (plus the YAMLs it includes and the python modules it references)."""
    files = set()
    for name in dict.fromkeys(task_names):
        try:
            entry = tm.task_index[name]
            path = entry["yaml_path"] if isinstance(entry, dict) else entry.yaml_path
            _yaml_closure(Path(path), files)
        except Exception as e:
            files.add(Path(f"unavailable:{name}:{type(e).__name__}"))
    out = {}
    for f in sorted(files):
        out[f.name if not f.name.startswith("unavailable") else str(f)] = sha256_path(f) if f.exists() else "unavailable"
    return out


def dataset_revision(repo):
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception as e:  # recorded, not fatal: the question-set hash is the real identity check
        return f"unavailable ({type(e).__name__})"


def lm_eval_version():
    import importlib.metadata

    try:
        return importlib.metadata.version("lm_eval")
    except Exception:
        return "unavailable"


# --------------------------------------------------------------------------- audit

def ids_csv_text(builds: dict) -> str:
    rows = sorted((b, c.task_name, q.doc_id) for b, bd in builds.items() for q, c in zip(bd.questions, bd.selected))
    return "\n".join(["benchmark,stratum,doc_id,split"] + [f"{b},{s},{d},confirmation" for b, s, d in rows]) + "\n"


def audit(phase1: dict, builds: dict, *, exclusion: Exclusion, manifest_hashes: dict, stored_contexts: dict,
          phase1_doc_ids: set, salt=SALT, take=None, fingerprints=None, expected_pool=None, limit=None) -> dict:
    """Assemble the audit record. phase1: {benchmark: (questions, recorder)}; builds: {benchmark: ConfirmBuild}.

    manifest_hashes[b] is the frozen question_set_sha256 (None = comparison skipped); stored_contexts[b] maps doc_id to the
    context stored in data/questions (None = use the rebuilt context).
    """
    rep = {"limit": limit, "salt": salt, "benchmarks": {}, "overlaps": {"identity": [], "text": [], "doc_id": []}}
    checks = {k: True for k in ("phase1_rebuild_matches_manifest", "exemplars_verbatim_in_contexts", "no_identity_overlap",
                                "no_text_overlap", "no_leakage_into_probe_data", "confirmation_exemplars_verbatim",
                                "no_self_exemplar", "enough_eligible", "pool_sizes_match_plan")}
    probe_text = exclusion.eval_texts
    for b, (qs, rec) in phase1.items():
        h = question_set_hash(qs)
        want = manifest_hashes.get(b)
        ctx = stored_contexts.get(b) or {q.doc_id: q.context for q in qs}
        vf = verbatim_failures(rec, ctx) + [dict(f, problem="not in rebuilt context") for f in rec.live_failures]
        inserted = {id(d.doc): d for r in rec.by_doc_id.values() for d in r.inserted}
        sup = {id(d.doc): d for r in rec.by_doc_id.values() for d in r.superset + r.inserted}
        entry = {"rebuild_question_set_sha256": h, "manifest_question_set_sha256": want,
                 "hash_matches": None if want is None else h == want, "n_questions": len(qs),
                 "n_exemplar_draws_inserted": sum(len(r.inserted) for r in rec.by_doc_id.values()),
                 "n_exemplar_docs_inserted": len(inserted), "n_exemplar_docs_superset": len(sup),
                 "verbatim_checked_against": "stored data/questions" if stored_contexts.get(b) else "rebuilt questions",
                 "verbatim_failures": vf[:MAX_LISTED], "n_verbatim_failures": len(vf)}
        rep["benchmarks"].setdefault(b, {})["phase1"] = entry
        checks["phase1_rebuild_matches_manifest"] &= entry["hash_matches"] is not False
        checks["exemplars_verbatim_in_contexts"] &= not vf
    for b, bd in builds.items():
        st = dict(bd.stats)
        vf = verbatim_failures(bd.recorder, {q.doc_id: q.context for q in bd.questions}) + bd.recorder.live_failures
        ident, txt = [], []
        for q, c in zip(bd.questions, bd.selected):  # independent re-check of the final selection
            id_hit, text_hit = exclusion.hit(b, c.native_id, c.text_hash)
            if id_hit:
                ident.append({"item": q.doc_id, "phase1": id_hit})
            if text_hit:
                txt.append({"item": q.doc_id, "phase1": text_hit})
            if q.doc_id in phase1_doc_ids:
                rep["overlaps"]["doc_id"].append(q.doc_id)
            if c.text_hash in probe_text:
                rep["overlaps"]["doc_id"].append(f"{q.doc_id} (text hash of a Phase 1 eval doc)")
        rep["overlaps"]["identity"] += ident
        rep["overlaps"]["text"] += txt
        st.update({"verbatim_failures": vf[:MAX_LISTED], "n_verbatim_failures": len(vf),
                   "expected_pool": (expected_pool or {}).get(b)})
        rep["benchmarks"].setdefault(b, {})["confirmation"] = st
        checks["confirmation_exemplars_verbatim"] &= not vf
        checks["no_self_exemplar"] &= not st["self_exemplar_violations"]
        want_n = st["n_requested"]
        checks["enough_eligible"] &= st["n_taken"] == st["n_eligible"] if want_n is None else st["n_taken"] == want_n
        if expected_pool and b in expected_pool:
            checks["pool_sizes_match_plan"] &= st["pool_size"] == expected_pool[b]
    checks["no_identity_overlap"] = not rep["overlaps"]["identity"]
    checks["no_text_overlap"] = not rep["overlaps"]["text"]
    checks["no_leakage_into_probe_data"] = not rep["overlaps"]["doc_id"]
    rep["checks"] = checks
    rep["failed_checks"] = [k for k, v in checks.items() if not v]
    rep["passed"] = not rep["failed_checks"]
    rep["fingerprints"] = {"lm_eval_version": lm_eval_version(), "salt": salt, "n_take": take or {},
                           "confirmation_ids_sha256": sha256_bytes(ids_csv_text(builds).encode("utf-8")),
                           "confirmation_question_set_sha256": {b: question_set_hash(bd.questions) for b, bd in builds.items()},
                           **(fingerprints or {})}
    return rep


# --------------------------------------------------------------------------- inputs for extraction / verification

def write_questions(path: Path, qs) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q.__dict__) + "\n")


def check_confirmation_inputs(root: Path, benches, limit=None):
    """Frozen-input checks for `--set confirmation`, same pattern as extract_phase1.check_inputs.

    Returns (manifest, ids_sha, splits, questions). Raises SystemExit on any mismatch.
    """
    from lookahead.extract import load_questions, load_splits

    mdir = Path(root) / "data" / "manifests"
    manifest = json.loads((mdir / "confirmation.json").read_text(encoding="utf-8"))
    raw = (mdir / "confirmation_ids.csv").read_bytes()
    recorded = (mdir / "confirmation_ids.sha256").read_text(encoding="utf-8").strip()
    if recorded not in (sha256_bytes(raw), sha256_bytes(raw.replace(b"\r\n", b"\n"))) or manifest["ids_sha256"] != recorded:
        raise SystemExit("confirmation_ids.csv does not match its recorded sha256: inputs are not the frozen ones")
    audit_path = Path(root) / "results" / "gate2" / "confirmation_audit.json"
    if not audit_path.exists() or sha256_path(audit_path) != manifest["audit_sha256"]:
        raise SystemExit("results/gate2/confirmation_audit.json is missing or does not match the manifest")
    if not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
        raise SystemExit("the contamination audit did not pass")
    splits = load_splits(mdir / "confirmation_ids.csv", allowed=("confirmation",))
    questions = {}
    for b in benches:
        qs = load_questions(Path(root) / "data" / "questions_confirm" / f"{b}.jsonl.gz")
        if question_set_hash(qs) != manifest["benchmarks"][b]["question_set_sha256"]:
            raise SystemExit(f"{b}: confirmation questions do not match the manifest hash; re-run scripts/prepare_confirmation.py")
        n_split = sum(1 for (bb, _) in splits if bb == b)
        if n_split != len(qs):
            raise SystemExit(f"{b}: {len(qs)} questions but {n_split} id rows")
        questions[b] = qs[:limit] if limit else qs
    return manifest, recorded, splits, questions
