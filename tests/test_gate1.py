"""Gate 1 headroom analysis: synthetic data in the real parquet / done.json format (no torch, no real data)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lookahead import gate1 as g

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


# ------------------------------------------------------------------------------------------------ builders
def _trans(a, b):
    return {(1, 1): "CC", (1, 0): "CW", (0, 1): "WC", (0, 0): "WW"}[(a, b)]


def make_rows(bench, split, questions):
    """questions: list of dicts (doc_id, status, pred, correct, conf, ent, cdf, D[3], ms) -> long DataFrame."""
    rows = []
    for q in questions:
        c = q["correct"]
        for t in range(4):
            rows.append({
                "benchmark": bench, "doc_id": q["doc_id"], "split": split, "status": q.get("status", "ok"),
                "loop": t + 1, "pred": q["pred"][t], "correct": c[t], "conf": q["conf"][t], "ent": q["ent"][t],
                "cdf": q["cdf"][t], "D": q["D"][t] if t < 3 else np.nan, "ms": q["ms"][t],
                "T": _trans(c[t], c[t + 1]) if t < 3 else "", "T_final": _trans(c[t], c[3]), "extra": 1.5})
    return pd.DataFrame(rows)


def default_q(doc_id, **kw):
    q = {"doc_id": doc_id, "status": "ok", "pred": [0, 0, 0, 0], "correct": [1, 1, 1, 1], "conf": [0.5] * 4,
         "ent": [1.0] * 4, "cdf": [0.5] * 4, "D": [0.5] * 3, "ms": [10.0] * 4}
    q.update(kw)
    return q


def random_questions(rng, prefix, n, mode="normal", fail_rate=0.05):
    qs = []
    for i in range(n):
        did = f"{prefix}{i:04d}"
        if mode == "extreme":
            qs.append(default_q(did, pred=[int(x) for x in rng.integers(0, 4, 4)], correct=[0] * 4,
                                conf=[12345.678] * 4, ent=[0.0] * 4, cdf=[1e6] * 4, D=[1e-9] * 3, ms=[987654.0] * 4))
            continue
        ans = int(rng.integers(0, 4))
        pred = [int(x) for x in rng.integers(0, 4, 4)]
        if rng.random() < 0.7:
            pred[3] = ans if rng.random() < 0.65 else pred[3]
            k = int(rng.integers(1, 4))
            for t in range(k, 4):
                pred[t] = pred[3]
        correct = [int(p == ans) for p in pred]
        conf = sorted(float(x) for x in rng.random(4))
        cdf = sorted(float(x) for x in rng.random(4))
        ent = sorted((float(x) for x in rng.random(4) * 1.4), reverse=True)
        D = sorted((float(x) for x in rng.random(3)), reverse=True)
        q = default_q(did, pred=pred, correct=correct, conf=conf, ent=ent, cdf=cdf, D=D,
                      ms=[float(x) for x in 5 + rng.random(4) * 10])
        if rng.random() < fail_rate:
            q.update(status="timeout", pred=[np.nan] * 4, conf=[np.nan] * 4, ent=[np.nan] * 4, cdf=[np.nan] * 4,
                     D=[np.nan] * 3, ms=[np.nan, 3.0, np.nan, np.nan])
        qs.append(q)
    return qs


def sha(path):
    return g.sha256_file(path)


def write_bench(root, bench, df, nparts=2, seed=0):
    bdir = Path(root) / "data" / "extracted" / bench
    bdir.mkdir(parents=True, exist_ok=True)
    ids = sorted(df["doc_id"].unique())
    np.random.default_rng(seed).shuffle(ids)
    for p, chunk in enumerate(np.array_split(np.array(ids, dtype=object), nparts)):
        part = df[df["doc_id"].isin(set(chunk))].sample(frac=1.0, random_state=p)
        stem = bdir / f"part-{p:05d}"
        part.to_parquet(str(stem) + ".parquet", index=False)
        (bdir / f"part-{p:05d}.done.json").write_text(
            json.dumps({"parquet_sha256": sha(str(stem) + ".parquet"), "finished_at": 0.0}), encoding="utf-8")


def write_dataset(root, test_mode="normal", with_test=True, n_val=70, n_test=50, seed=1):
    root = Path(root)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "data" / "manifests" / "splits.csv").write_text("benchmark,stratum,doc_id,split\n", encoding="utf-8")
    for bi, bench in enumerate(g.GATE1_BENCHMARKS):
        rng = np.random.default_rng(seed + bi)
        frames = [make_rows(bench, "validation", random_questions(rng, "v", n_val + 10 * bi))]
        frames.append(make_rows(bench, "train", random_questions(rng, "r", 15)))
        if with_test:
            trng = np.random.default_rng(1000 + bi + (0 if test_mode == "normal" else 77))
            frames.append(make_rows(bench, "test", random_questions(trng, "t", n_test, mode=test_mode)))
        write_bench(root, bench, pd.concat(frames, ignore_index=True))


def arrays(questions, split="validation"):
    return g.build_arrays(make_rows("b", split, questions), split)


def run_script(name, root, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, str(SCRIPTS / name), "--root", str(root), "--n-boot", "100", *extra],
                          capture_output=True, text=True, env=env)


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def strip(d):
    d = dict(d)
    d.pop("provenance", None)
    d.pop("timestamp_utc", None)
    return d


# ------------------------------------------------------------------------------------------------ 1 policies
@pytest.fixture()
def hand():
    q0 = default_q("a", pred=[0, 1, 1, 1], correct=[0, 1, 1, 1], conf=[.9, .95, .99, 1.0], ent=[2, 1, .5, .1],
                   cdf=[.1, .5, .9, 1], D=[.5, .05, .01], ms=[10, 20, 30, 40])
    q1 = default_q("b", pred=[2, 3, 3, 3], correct=[1, 0, 0, 0], conf=[.6, .7, .8, .9], ent=[3, 2.5, 2, 1],
                   cdf=[.2, .3, .4, .5], D=[.4, .3, .2], ms=[5, np.nan, 7, 8])
    q2 = default_q("c", status="timeout", pred=[1, 1, 1, 1], correct=[1, 1, 1, 1], conf=[.99] * 4, ent=[0.0] * 4,
                   cdf=[1.0] * 4, D=[0.0] * 3)
    return arrays([q0, q1, q2])


def check(d, kind, param, e, c):
    ee, cc = g.run_policy(d, kind, param)
    assert ee.tolist() == e, (kind, param)
    assert cc.tolist() == c, (kind, param)


def test_policy_semantics_hand_example(hand):
    d = hand
    check(d, "fixed", 1, [1, 1, 4], [0, 1, 0])
    check(d, "fixed", 2, [2, 2, 4], [1, 0, 0])
    check(d, "fixed", 4, [4, 4, 4], [1, 0, 0])
    check(d, "confidence", 0.9, [1, 4, 4], [0, 0, 0])
    check(d, "confidence", 0.7, [1, 2, 4], [0, 0, 0])
    check(d, "confidence", 0.95, [2, 4, 4], [1, 0, 0])
    check(d, "entropy", 2.0, [1, 3, 4], [0, 0, 0])
    check(d, "entropy", 1.0, [2, 4, 4], [1, 0, 0])
    check(d, "native_gate", 0.5, [2, 4, 4], [1, 0, 0])
    check(d, "native_gate", 0.3, [2, 2, 4], [1, 0, 0])
    # deployable: fires after running t+1, answers loop t+1, earliest exit is loop 2
    check(d, "deployable_kl", 0.5, [2, 2, 4], [1, 0, 0])
    check(d, "deployable_kl", 0.05, [3, 4, 4], [1, 0, 0])
    check(d, "deployable_kl", 0.01, [4, 4, 4], [1, 0, 0])
    assert g.run_policy(d, "deployable_kl", 1e9)[0].min() >= 2
    # retrospective: answers loop t at cost t
    check(d, "retrospective_kl", 0.5, [1, 1, 4], [0, 1, 0])
    check(d, "retrospective_kl", 0.05, [2, 4, 4], [1, 0, 0])
    check(d, "stable_oracle", None, [2, 2, 4], [1, 0, 0])
    check(d, "earliest_correct", None, [2, 1, 4], [1, 1, 0])


def test_wall_clock_nan_safe_and_deployable_pays_extra_loop(hand):
    e, _ = g.run_policy(hand, "deployable_kl", 0.5)
    assert g.wall_clock(hand, e)[:2].tolist() == [30.0, 5.0 + 0.0]  # q1: loops 1..2, NaN counts as 0
    e3, _ = g.run_policy(hand, "fixed", 3)
    assert g.wall_clock(hand, e3)[1] == 12.0
    assert np.isfinite(g.wall_clock(hand, np.full(3, 4))).all()


# ------------------------------------------------------------------------------------------------ 2 stable oracle
def test_stable_final_oracle_random():
    rng = np.random.default_rng(5)
    qs = []
    for i in range(300):
        pred = [int(x) for x in rng.integers(0, 3, 4)]
        qs.append(default_q(f"q{i:03d}", pred=pred, correct=[int(x) for x in rng.integers(0, 2, 4)]))
    qs.append(default_q("zfail", status="nonfinite", pred=[1, 1, 1, 1]))
    d = arrays(qs)
    e, c = g.run_policy(d, "stable_oracle")
    for i in range(d["n"]):
        p = d["pred"][i]
        if not d["ok"][i]:
            assert e[i] == 4
            continue
        want = next(t for t in range(1, 5) if all(p[j] == p[3] for j in range(t - 1, 4)))
        assert e[i] == want
        assert p[e[i] - 1] == p[3]
        assert c[i] == d["correct"][i, e[i] - 1]
        # chain definition from the spec: pred[t-1]==pred[t]==...==pred[3]
        assert all(p[j] == p[j + 1] for j in range(e[i] - 1, 3))
    assert e[d["doc_id"].tolist().index("zfail")] == 4


# ------------------------------------------------------------------------------------------------ 3 macro vs pooled
def test_macro_not_pooled():
    big = [default_q(f"b{i:05d}", pred=[0, 0, 0, 0], correct=[1, 1, 1, 1]) for i in range(4000)]
    small = [default_q(f"s{i:03d}", pred=[0, 1, 2, 3], correct=[0, 0, 0, 1]) for i in range(20)]
    data = {"big": arrays(big), "small": arrays(small)}
    sel = g.run_selection(data, ["big", "small"])
    f1, f3 = sel["baselines"]["fixed-1"], sel["baselines"]["fixed-3"]
    assert f1["grid"][0]["pooled_loss"] < 1.0       # question-weighted would qualify ...
    assert f3["grid"][0]["pooled_loss"] < 1.0
    assert not f1["qualified"] and not f3["qualified"]  # ... but selection is on the macro loss (50 pp)
    assert f1["grid"][0]["macro_loss"] == pytest.approx(50.0)
    ev = g.evaluate_frozen(data, ["big", "small"], g.freeze(sel)["baselines"])
    assert ev["S_or"] == pytest.approx((0.75 + 0.0) / 2)
    assert ev["S_or_pooled"] == pytest.approx(1 - (4000 + 80) / (4 * 4020))
    assert ev["S_or"] != pytest.approx(ev["S_or_pooled"])


# ------------------------------------------------------------------------------------------------ 4 qualification
def test_fixed_k_qualifies_iff_macro_loss_at_most_one():
    qs = []
    for i in range(100):
        corr = [1, 1, 1, 1]
        if i == 0:
            corr = [1, 1, 0, 1]   # fixed-3 loses nothing vs fixed-4 only when... see below
        qs.append(default_q(f"q{i:03d}", pred=[i, i, i, i], correct=corr))
    # fixed-4 acc 100. fixed-3: q0 wrong -> loss 1.0 exactly (qualifies). fixed-2: all right -> loss 0.
    qs[1]["correct"] = [0, 0, 1, 1]   # fixed-1 and fixed-2 now lose q1 (and q0 fine at 1,2) -> loss 1.0 each
    qs[2]["correct"] = [0, 1, 1, 1]   # fixed-1 loses a second question
    d = {"b": arrays(qs)}
    sel = g.run_selection(d, ["b"])
    b = sel["baselines"]
    assert b["fixed-3"]["grid"][0]["macro_loss"] == pytest.approx(1.0)
    assert b["fixed-3"]["qualified"]
    assert b["fixed-2"]["grid"][0]["macro_loss"] == pytest.approx(1.0) and b["fixed-2"]["qualified"]
    assert b["fixed-1"]["grid"][0]["macro_loss"] == pytest.approx(2.0)
    assert not b["fixed-1"]["qualified"] and b["fixed-1"]["selected"] is None and b["fixed-1"]["threshold"] is None
    d["b"]["correct"][:, 2][3] = 0   # one more loss at fixed-3 -> 2.0 > 1
    assert not g.run_selection(d, ["b"])["baselines"]["fixed-3"]["qualified"]


def test_thresholded_selects_max_saving_among_qualified():
    rng = np.random.default_rng(3)
    d = {"x": arrays(random_questions(rng, "q", 300, fail_rate=0.0)), "y": arrays(random_questions(rng, "r", 120))}
    sel = g.run_selection(d, ["x", "y"])
    for kind in g.THRESHOLDED:
        b = sel["baselines"][kind]
        q = [r for r in b["grid"] if r["macro_loss"] <= 1.0 + 1e-12]
        assert len(b["grid_thresholds"]) == len(set(b["grid_thresholds"])) <= 50
        if not q:
            assert not b["qualified"]
            continue
        assert b["qualified"]
        assert b["selected"]["macro_saving"] == pytest.approx(max(r["macro_saving"] for r in q), abs=1e-12)
        assert b["threshold"] == b["selected"]["threshold"]
    vals = g.pooled_values(d, ["x", "y"], "conf")
    assert sel["baselines"]["confidence"]["grid_thresholds"][0] == vals.min()
    assert g.pooled_values(d, ["x", "y"], "D").size == sum(int(v["ok"].sum()) for v in d.values()) * 3


def row(thr, sav, loss):
    return {"threshold": thr, "macro_saving": sav, "macro_loss": loss}


def test_tie_break_rules():
    # saving tie -> smaller loss
    assert g.select_point([row(.1, .5, .9), row(.2, .5, .3), row(.3, .4, 0.0)], "confidence")["threshold"] == .2
    # saving and loss tie -> most conservative threshold
    tie = [row(.1, .5, .3), row(.2, .5, .3 + 1e-14), row(.3, .5 - 1e-14, .3)]
    assert g.select_point(tie, "confidence")["threshold"] == .3
    assert g.select_point(tie, "native_gate")["threshold"] == .3
    assert g.select_point(tie, "entropy")["threshold"] == .1
    assert g.select_point(tie, "deployable_kl")["threshold"] == .1
    # unqualified points never selected; boundary 1.0 qualifies
    assert g.select_point([row(.1, .9, 1.1)], "entropy") is None
    assert g.select_point([row(.1, .9, 1.0 + 1e-13), row(.2, .1, 0.0)], "entropy")["threshold"] == .1


def test_end_to_end_tie_picks_conservative_threshold():
    # Few questions -> the 50 grid points fall several to a gap between order statistics, so thresholds tie exactly.
    q0 = default_q("q0", correct=[0, 1, 1, 1], conf=[0.1, 0.5, 0.6, 0.7], ent=[3.0, 2.0, 1.5, 1.0])
    others = [default_q(f"q{i}", conf=[0.8 + 0.01 * i, 0.85 + 0.01 * i, 0.9 + 0.01 * i, 0.95],
                        ent=[0.2 + 0.01 * i, 0.15 + 0.01 * i, 0.1 + 0.01 * i, 0.05]) for i in (1, 2, 3)]
    d = {"b": arrays([q0] + others)}
    sel = g.run_selection(d, ["b"])
    c = sel["baselines"]["confidence"]
    best = [r for r in c["grid"] if r["macro_saving"] == c["selected"]["macro_saving"]]
    assert len(best) == 4 and c["selected"]["macro_loss"] == 0.0
    assert c["threshold"] == max(r["threshold"] for r in best)
    e = sel["baselines"]["entropy"]
    best = [r for r in e["grid"] if r["macro_saving"] == e["selected"]["macro_saving"]]
    assert len(best) == 4 and e["threshold"] == min(r["threshold"] for r in best)


def test_headroom_and_decision():
    assert g.headroom_R(0.4, {}) == (0.0, None)
    R, arg = g.headroom_R(0.4, {"a": 0.1, "b": 0.2})
    assert R == pytest.approx(0.5) and arg == "b"
    assert g.decide(0.25, 0.7499) == "PASS"
    assert g.decide(0.2499, 0.1) == "FAIL"
    assert g.decide(0.3, 0.75) == "FAIL"


# ------------------------------------------------------------------------------------------------ 5 blindness
def test_validation_blind_to_test_rows(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    write_dataset(a)
    write_dataset(b, with_test=False)
    write_dataset(c, test_mode="extreme")
    for r in (a, b, c):
        res = run_script("gate1_validation.py", r)
        assert res.returncode == 0, res.stderr
    out = {r.name: (load_json(r / "results/gate1/validation_report.json"),
                    load_json(r / "results/gate1/frozen_selection.json")) for r in (a, b, c)}
    for other in ("b", "c"):
        assert strip(out["a"][0]) == strip(out[other][0])
        assert strip(out["a"][1]) == strip(out[other][1])
    text = (c / "results/gate1/frozen_selection.json").read_text() + (c / "results/gate1/validation_report.json").read_text()
    text += (c / "docs/GATE1_VALIDATION_REPORT.md").read_text()
    assert "12345.678" not in text and "987654" not in text
    assert "No test-split outcome has been read" in (c / "docs/GATE1_VALIDATION_REPORT.md").read_text()
    assert "final-answer-preserving" in (c / "docs/GATE1_VALIDATION_REPORT.md").read_text()
    # the loader never keeps test rows, and the n matches the validation questions only
    data = g.load_split(a, g.GATE1_BENCHMARKS, "validation")
    for bi, bench in enumerate(g.GATE1_BENCHMARKS):
        assert data[bench]["n"] == 70 + 10 * bi
        assert (data[bench]["split"] == "validation").all()
        assert not any(str(x).startswith(("t", "r")) for x in data[bench]["doc_id"])
    frozen = out["a"][1]
    assert set(frozen) >= {"primary", "sensitivity", "provenance", "rules"}
    assert frozen["primary"]["benchmarks"] == list(g.GATE1_BENCHMARKS)
    assert frozen["sensitivity"]["benchmarks"] == list(g.SENSITIVITY_BENCHMARKS)
    prov = frozen["provenance"]
    assert len(prov["parquet_part_sha256"]) == 12 and prov["splits_csv_sha256"]
    assert set(prov["code_sha256"]) == {"src/lookahead/gate1.py", "scripts/gate1_validation.py", "scripts/gate1_test.py"}


def test_load_split_rejects_excluded_benchmarks(tmp_path):
    for bench in ("modarith", "parity"):
        with pytest.raises(ValueError):
            g.load_split(tmp_path, [bench], "validation")


# ------------------------------------------------------------------------------------------------ 6 gate1_test
def test_gate1_test_guards_and_frozen_thresholds(tmp_path):
    write_dataset(tmp_path)
    assert run_script("gate1_validation.py", tmp_path).returncode == 0
    out = tmp_path / "results" / "gate1"
    frozen_path = out / "frozen_selection.json"
    res = run_script("gate1_test.py", tmp_path)
    assert res.returncode != 0 and not (out / "TEST_EVALUATED.json").exists()
    assert not (out / "gate1_test_report.json").exists()

    res = run_script("gate1_test.py", tmp_path, "--confirm-one-time-test-evaluation")
    assert res.returncode == 0, res.stderr
    rep = load_json(out / "gate1_test_report.json")
    frozen = load_json(frozen_path)
    assert rep["frozen_selection_sha256"] == sha(frozen_path)
    assert load_json(out / "TEST_EVALUATED.json")["frozen_selection_sha256"] == sha(frozen_path)
    assert load_json(out / "TEST_EVALUATED.json")["timestamp_utc"]
    for key in ("primary", "sensitivity"):
        used = rep[key]["evaluation"]["frozen_thresholds_used"]
        for name, ent in frozen[key]["baselines"].items():
            assert used[name] == ent["threshold"]  # exact equality
    assert rep["primary"]["evaluation"]["benchmarks"] == list(g.GATE1_BENCHMARKS)
    assert rep["rerun"]["is_rerun"] is False
    assert (tmp_path / "docs" / "GATE1_REPORT.md").exists()
    ev = rep["primary"]["evaluation"]
    assert ev["R_unc"]["R_unc"] >= 0 and "constraint_violations" in ev
    # test split sizes are the test rows (50 each), not validation rows
    assert ev["fixed-4"]["per_benchmark"].keys() == set(g.GATE1_BENCHMARKS)

    # second run refused
    before = (out / "gate1_test_report.json").read_text()
    res = run_script("gate1_test.py", tmp_path, "--confirm-one-time-test-evaluation")
    assert res.returncode != 0 and (out / "gate1_test_report.json").read_text() == before

    # rerun allowed, recorded, and uses whatever the frozen file says exactly (no reselection)
    frozen["primary"]["baselines"]["confidence"].update(qualified=True, threshold=0.123456789012345)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    res = run_script("gate1_test.py", tmp_path, "--confirm-one-time-test-evaluation", "--allow-rerun")
    assert res.returncode == 0, res.stderr
    rep2 = load_json(out / "gate1_test_report.json")
    assert rep2["rerun"]["is_rerun"] is True and rep2["rerun"]["previous_test_evaluated"]["timestamp_utc"]
    assert rep2["primary"]["evaluation"]["frozen_thresholds_used"]["confidence"] == 0.123456789012345
    assert rep2["frozen_selection_sha256"] == sha(frozen_path) != rep["frozen_selection_sha256"]
    # validation script refuses to rewrite the frozen file after the test evaluation
    assert run_script("gate1_validation.py", tmp_path).returncode != 0


# ------------------------------------------------------------------------------------------------ 7 failures
def test_failed_question_wrong_and_four_loops_everywhere():
    good = default_q("g0", pred=[1, 1, 1, 1], correct=[1, 1, 1, 1], conf=[.1, .2, .3, .4])
    bad = default_q("g1", status="timeout", pred=[1, 1, 1, 1], correct=[1, 1, 1, 1], conf=[.99] * 4, ent=[0.0] * 4,
                    cdf=[1.0] * 4, D=[0.0] * 3, ms=[np.nan, 2.0, np.nan, 1.0])
    d = arrays([good, bad])
    i = 1
    assert (d["correct"][i] == 0).all() and not d["ok"][i]
    assert d["conf"][i].max() == -np.inf and d["ent"][i].min() == np.inf and d["D"][i].min() == np.inf
    cases = [("fixed", k) for k in (1, 2, 3, 4)] + [("confidence", -1e9), ("entropy", 1e9), ("native_gate", -1e9),
                                                     ("deployable_kl", 1e9), ("retrospective_kl", 1e9),
                                                     ("confidence", np.inf), ("stable_oracle", None),
                                                     ("earliest_correct", None)]
    for kind, p in cases:
        e, c = g.run_policy(d, kind, p)
        assert e[i] == 4 and c[i] == 0, (kind, p)
    assert g.wall_clock(d, np.full(2, 4))[i] == 3.0
    # fixed-1 on [good, bad]: mean cost (1 + 4) / 2
    assert g.summarize({"b": d}, ["b"], "fixed", 1)["macro"]["saving"] == pytest.approx(1 - 2.5 / 4)
    assert (d["T"][i] == "WW").all() and (d["Tfin"][i] == "WW").all()
    # grids come from ok questions only
    assert np.isfinite(g.pooled_values({"b": d}, ["b"], "conf")).all()


def test_loader_structure_errors(tmp_path):
    base = [default_q("x1"), default_q("x2")]
    df = make_rows("b", "validation", base)
    with pytest.raises(ValueError, match="exactly"):
        g.build_arrays(df.iloc[:-1], "validation")
    with pytest.raises(ValueError, match="twice"):
        g.build_arrays(pd.concat([df, df.iloc[:4]]), "validation")
    with pytest.raises(AssertionError):
        g.build_arrays(df, "test")
    d = g.build_arrays(make_rows("b", "validation", [default_q("z"), default_q("a")]), "validation")
    assert d["doc_id"].tolist() == ["a", "z"]
    # same doc in two parts
    write_bench(tmp_path, "arc_easy", pd.concat([make_rows("arc_easy", "validation", base)] * 2), nparts=2)
    with pytest.raises(ValueError):
        g.load_split(tmp_path, ["arc_easy"], "validation")


# ------------------------------------------------------------------------------------------------ 8 bootstrap
def _frozen_for(d, benches):
    return g.freeze(g.run_selection(d, benches))["baselines"]


def test_bootstrap_deterministic_and_ordered():
    rng = np.random.default_rng(11)
    d = {"x": arrays(random_questions(rng, "q", 150)), "y": arrays(random_questions(rng, "r", 90))}
    fr = _frozen_for(d, ["x", "y"])
    b1 = g.bootstrap_headroom(d, ["x", "y"], fr, n_boot=300)
    b2 = g.bootstrap_headroom(d, ["x", "y"], fr, n_boot=300)
    assert b1 == b2
    assert b1["S_or_ci"][0] <= b1["S_or_ci"][1] and b1["R_ci"][0] <= b1["R_ci"][1]
    pe = g.evaluate_frozen(d, ["x", "y"], fr)
    assert b1["S_or_ci"][0] <= pe["S_or"] + 0.2 and b1["S_or_ci"][1] >= pe["S_or"] - 0.2
    for v in b1["baselines"].values():
        assert v["saving_ci"][0] <= v["saving_ci"][1] and v["loss_ci"][0] <= v["loss_ci"][1]
    t1 = g.transition_tables(d, ["x", "y"], n_boot=200)
    assert t1 == g.transition_tables(d, ["x", "y"], n_boot=200)
    assert list(t1) == ["1->2", "2->3", "3->4", "1->4", "2->4", "3->4 (T_final)"]
    for tab in t1.values():
        for v in tab["per_benchmark"].values():
            assert v["net"] == v["WC"] - v["CW"] and v["net_pts_ci"][0] <= v["net_pts_ci"][1]
            assert v["CC"] + v["CW"] + v["WC"] + v["WW"] == v["n"]
        assert tab["macro_net_pts"] == pytest.approx(np.mean([v["net_pts"] for v in tab["per_benchmark"].values()]))
    # bootstrap rng scheme: seed 0, per resample per benchmark in order
    rng0 = np.random.default_rng(0)
    v = ((d["x"]["T"][:, 0] == "WC").astype(float) - (d["x"]["T"][:, 0] == "CW"))
    first = 100 * v[rng0.integers(0, len(v), len(v))].mean()
    one = g.transition_tables({"x": d["x"]}, ["x"], n_boot=1)["1->2"]["per_benchmark"]["x"]["net_pts_ci"]
    assert one[0] == pytest.approx(first)


def test_marginal_logic():
    assert g.is_marginal([0.2, 0.3], [0.1, 0.2])        # 0.25 inside S_or interval
    assert g.is_marginal([0.3, 0.4], [0.7, 0.8])        # 0.75 inside R interval
    assert g.is_marginal([0.25, 0.4], [0.1, 0.2])       # inclusive
    assert g.is_marginal([0.3, 0.4], [0.1, 0.75])       # inclusive
    assert not g.is_marginal([0.3, 0.4], [0.1, 0.5])
    assert not g.is_marginal([0.3, 0.4], [0.8, 0.9])
    assert not g.is_marginal([0.1, 0.2], [0.8, 0.9])


# ------------------------------------------------------------------------------------------------ 9 checksum
def test_checksum_mismatch_raises_and_incomplete_parts_ignored(tmp_path):
    write_dataset(tmp_path, with_test=False)
    bench = g.GATE1_BENCHMARKS[0]
    bdir = tmp_path / "data" / "extracted" / bench
    # a parquet without a done.json is not a part (even if it would collide)
    pd.read_parquet(bdir / "part-00000.parquet").to_parquet(bdir / "part-00009.parquet", index=False)
    g.load_split(tmp_path, [bench], "validation")
    done = bdir / "part-00001.done.json"
    meta = json.loads(done.read_text())
    meta["parquet_sha256"] = "0" * 64
    done.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum"):
        g.load_split(tmp_path, [bench], "validation")
    res = run_script("gate1_validation.py", tmp_path)
    assert res.returncode != 0 and not (tmp_path / "results/gate1/frozen_selection.json").exists()
