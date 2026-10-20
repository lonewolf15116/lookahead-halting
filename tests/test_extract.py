"""Extraction-writer acceptance checks 1-6 (reviewer, 2026-10-20), end to end on CPU with a tiny Ouro."""
import gzip
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")
pd = pytest.importorskip("pandas")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.extract import load_splits, read_benchmark, run_benchmark  # noqa: E402
from lookahead.lm_eval_ouro import LoopedOuroLM  # noqa: E402
from lookahead.packed import PackedScorer  # noqa: E402
from lookahead.questions import load_lm_eval_questions, load_synthetic_questions, question_set_hash  # noqa: E402
from lookahead.splits import assign_splits  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny4():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=4,
        layer_types=["full_attention"] * 2, max_position_embeddings=2048,
    )
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        model.model.early_exit_gate.weight.normal_(0, 1.0)
    return model


@pytest.fixture()
def root(tmp_path, tok, fake_task):
    """A miniature repo: frozen questions, manifest and splits for 'fake_arc' (8 q) and 'parity' (first 30 q)."""
    benches = {
        "fake_arc": load_lm_eval_questions("fake_arc", LoopedOuroLM(None, tok), task_manager=fake_task,
                                           spec={"shots": 3, "metric": "acc_norm"}),
        "parity": load_synthetic_questions("parity", tok)[:30],
    }
    (tmp_path / "data" / "questions").mkdir(parents=True)
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    rows = []
    manifest = {"benchmarks": {}}
    for b, qs in benches.items():
        with gzip.open(tmp_path / "data" / "questions" / f"{b}.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q.__dict__) + "\n")
        manifest["benchmarks"][b] = {"question_set_sha256": question_set_hash(qs), "hf_dataset": "test",
                                     "hf_revision": None, "shots": 3, "metric": qs[0].metric}
        rows += [(b, q.subtask, q.doc_id) for q in qs]
    sp = assign_splits(rows)
    text = "benchmark,stratum,doc_id,split\n" + "".join(f"{b},{s},{d},{sp[(b, d)]}\n" for b, s, d in sorted(rows))
    (tmp_path / "data" / "manifests" / "splits.csv").write_text(text, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(text.encode()).hexdigest()
    (tmp_path / "data" / "manifests" / "splits.sha256").write_text(sha + "\n", encoding="utf-8")
    manifest["splits_sha256"] = sha
    (tmp_path / "data" / "manifests" / "questions.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, benches


def patch_scripts(monkeypatch, root_dir, model, tok):
    import lookahead.ouro_adapter as oa

    ext, ver = load_script("extract_phase1"), load_script("verify_extraction")
    for mod in (ext, ver):
        monkeypatch.setattr(mod, "REPO_ROOT", root_dir)
    monkeypatch.setattr(ver, "results_dir", lambda sub: (root_dir / "results" / sub).mkdir(parents=True, exist_ok=True)
                        or root_dir / "results" / sub)
    monkeypatch.setattr(oa, "load_ouro", lambda device="cuda": (model, tok))
    monkeypatch.setattr(ext, "code_identity", lambda r: {"code_sha256": "test", "git_commit": "test"})
    return ext, ver


def run_extract(ext, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["extract_phase1.py", "--benchmarks", "fake_arc", "parity", "--chunk", "7", *args])
    return ext.main()


def test_end_to_end_extraction_passes_all_checks(root, tiny4, tok, monkeypatch):
    root_dir, benches = root
    ext, ver = patch_scripts(monkeypatch, root_dir, tiny4, tok)
    assert run_extract(ext, monkeypatch) == 0
    out = root_dir / "data" / "extracted"
    assert ver.main(["--out", str(out), "--benchmarks", "fake_arc", "parity", "--sample", "3"]) == 0
    report = json.loads((root_dir / "results" / "phase1" / "extraction_check.json").read_text())
    for b, r in report["benchmarks"].items():
        assert all(r["checks"].values()), (b, r["checks"])
        assert r["model_sample_max_abs_err"]["D_vocab"] < 1e-5 and r["model_sample_max_abs_err"]["opt_ll"] < 1e-4
    # 1 schema: 4 rows per question, hidden [n, 4, d]
    df, hid, _ = read_benchmark(out, "parity")
    assert len(df) == 4 * 30 and hid.shape == (30, 4, 64) and hid.dtype == np.float16
    # 6 reproducibility
    meta = json.loads((out / "meta.json").read_text())
    run = meta["runs"][0]
    for k in ("code_sha256", "git_commit", "model", "datasets", "splits_sha256", "seeds", "versions", "device", "ended"):
        assert k in run, k
    assert run["model"]["revision"] == load_config("model")["revision"]


def test_resume_after_interruption_neither_duplicates_nor_rewrites(root, tiny4, tok, monkeypatch):
    root_dir, benches = root
    ext, ver = patch_scripts(monkeypatch, root_dir, tiny4, tok)
    out = root_dir / "data" / "extracted"
    # 8 fake_arc questions finish (parts of 7 + 1), then parity stops 2 questions into its second part
    assert run_extract(ext, monkeypatch, "--stop-after", "17") == 3
    pdir = out / "parity"
    part0 = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in pdir.glob("part-00000*")}
    assert len(part0) == 3 and not (pdir / "part-00001.done.json").exists()
    (pdir / "part-00001.parquet.tmp").write_bytes(b"half-written junk")  # what a crash could leave behind
    assert run_extract(ext, monkeypatch) == 0
    after = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in pdir.glob("part-00000*")}
    assert after == part0  # the completed part is neither recomputed nor rewritten
    meta = json.loads((out / "meta.json").read_text())
    assert meta["runs"][1]["benchmarks"]["fake_arc"]["computed"] == 0
    assert meta["runs"][1]["benchmarks"]["parity"]["computed"] == 30 - 7  # only the unfinished parts
    df, _, _ = read_benchmark(out, "parity")
    assert len(df) == 4 * 30 and df.groupby("doc_id").size().eq(4).all()  # no duplicates
    assert ver.main(["--out", str(out), "--benchmarks", "fake_arc", "parity", "--sample", "2"]) == 0


def test_targets_check_catches_wrong_values(root, tiny4, tok, monkeypatch):
    root_dir, benches = root
    ver = load_script("verify_extraction")
    q = benches["fake_arc"][0]
    scorer = PackedScorer(tiny4)
    from lookahead.extract import extract_question

    rows, _, status = extract_question(scorer, q)
    assert status == "ok"
    errs, mism = ver.check_targets(q, rows)
    assert not mism and max(errs.values()) < 1e-9
    bad = [dict(r) for r in rows]
    bad[1]["D"] += 1e-3
    bad[2]["T"] = "CW" if bad[2]["T"] != "CW" else "WC"
    errs, mism = ver.check_targets(q, bad)
    assert errs["D"] >= 0.99e-3 and any("next flip/transition" in m for m in mism)  # 1e-3 minus float rounding


def test_vocab_targets_match_context_only_pass(root, tiny4):
    from lookahead.extract import extract_question
    from lookahead.ouro_adapter import OuroLoopRunner

    _, benches = root
    q = benches["parity"][3]
    rows, hid, _ = extract_question(PackedScorer(tiny4), q)
    runner = OuroLoopRunner(tiny4)
    state = runner.start(torch.tensor([q.context_enc]))
    lps = []
    for _ in range(4):
        runner.step(state)
        lps.append(torch.log_softmax(runner.readout(state.hidden, torch.tensor([len(q.context_enc) - 1]))[0].double(), -1))
    for t in range(4):
        assert abs(rows[t]["vent"] + float((lps[t].exp() * lps[t]).sum())) < 1e-4
        assert abs(rows[t]["K_vocab"] - float((lps[t].exp() * (lps[t] - lps[3])).sum())) < 1e-4
        if t < 3:
            assert abs(rows[t]["D_vocab"] - float((lps[t].exp() * (lps[t] - lps[t + 1])).sum())) < 1e-4
    assert rows[3]["K_vocab"] == 0.0 and rows[3]["F"] == -1


def test_split_integrity_is_enforced(root, tiny4, tmp_path):
    root_dir, benches = root
    text = (root_dir / "data" / "manifests" / "splits.csv").read_text()
    dup = tmp_path / "dup.csv"
    first = text.splitlines()[1]
    dup.write_text(text + first.rsplit(",", 1)[0] + ",test\n")
    with pytest.raises(ValueError, match="more than one split"):
        load_splits(dup)
    splits = load_splits(root_dir / "data" / "manifests" / "splits.csv")
    splits.pop(("parity", benches["parity"][0].doc_id))
    with pytest.raises(ValueError, match="no split"):
        run_benchmark(PackedScorer(tiny4), "parity", benches["parity"], splits, tmp_path / "x")


def test_changed_questions_or_splits_are_refused(root, monkeypatch):
    root_dir, benches = root
    ext = load_script("extract_phase1")
    ext.check_inputs(root_dir, ["fake_arc", "parity"], None)  # frozen inputs pass
    p = root_dir / "data" / "questions" / "parity.jsonl.gz"
    lines = gzip.open(p, "rt").read().splitlines()
    d = json.loads(lines[0])
    d["target"] = 1 - d["target"]
    with gzip.open(p, "wt") as f:
        f.write("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit, match="do not match the manifest"):
        ext.check_inputs(root_dir, ["fake_arc", "parity"], None)
    s = root_dir / "data" / "manifests" / "splits.csv"
    s.write_text(s.read_text().replace(",train\n", ",test\n", 1), newline="\n")
    with pytest.raises(SystemExit, match="splits.csv"):
        ext.check_inputs(root_dir, ["fake_arc"], None)
