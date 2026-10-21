"""Gate 2 probes: synthetic data in the real on-disk layout (parquet parts + hidden.npy + done.json); no real data."""
import builtins
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lookahead import gate1 as g
from lookahead import gate2 as g2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import gate2_select as sel  # noqa: E402

D_HID = 16
BENCHES = ("arc_easy", "openbookqa")
SIGNAL = 4.0


# ------------------------------------------------------------------------------------------- synthetic writer
def _trans(a, b):
    return {(1, 1): "CC", (1, 0): "CW", (0, 1): "WC", (0, 0): "WW"}[(a, b)]


def make_question(rng, doc_id, split, bench, d, fail=False):
    """Stable from loop k (uniform 1..4): S_t = 1[t >= k]. h_t[0] carries S_t; cheap features are pure noise."""
    k = int(rng.integers(1, 5))
    ans = int(rng.integers(0, 4))
    final = ans if rng.random() < 0.7 else int((ans + 1) % 4)
    pred = [int((final + 1 + rng.integers(0, 3)) % 4) if t + 1 < k else final for t in range(4)]
    correct = [int(p == ans) for p in pred]
    S = [int(t + 1 >= k) for t in range(4)]
    h = rng.normal(size=(4, d)).astype(np.float32)
    h[:, 0] += SIGNAL * (2 * np.array(S) - 1)
    loss = rng.random(4) * 2
    rows = []
    for t in range(4):
        rows.append({
            "benchmark": bench, "stratum": "s0", "doc_id": doc_id, "split": split, "target": ans, "n_options": 4,
            "n_tokens": int(rng.integers(50, 400)), "metric": "acc", "status": "ok", "loop": t + 1, "pred": pred[t],
            "correct": correct[t], "conf": float(rng.random()), "ent": float(rng.random() * 1.3),
            "vent": float(rng.random() * 3), "cdf": float(rng.random()), "ms": float(5 + rng.random() * 5),
            "loss": float(loss[t]), "D": float(rng.random()) if t < 3 else np.nan, "F": int(pred[t] != pred[min(t + 1, 3)]),
            "G": float(loss[t] - loss[t + 1]) if t < 3 else np.nan,
            "T": _trans(correct[t], correct[t + 1]) if t < 3 else "", "T_final": _trans(correct[t], correct[3])})
    if fail:
        for r in rows:
            for key in ("pred", "correct", "conf", "ent", "vent", "cdf", "ms", "loss", "D", "G"):
                r[key] = np.nan
            r["status"] = "timeout"
            r["T"] = r["T_final"] = ""
        h[:] = np.nan
    return rows, h.astype(np.float16)


def write_probe_bench(root, bench, split_counts, d=D_HID, seed=0, nparts=2, fail_rate=0.04):
    rng = np.random.default_rng(seed)
    qs = []
    for split, n in split_counts.items():
        for i in range(n):
            fail = rng.random() < fail_rate
            did = f"{split[:2]}-{bench}-{i:04d}"
            qs.append(make_question(rng, did, split, bench, d, fail))
    order = rng.permutation(len(qs))
    bdir = Path(root) / "data" / "extracted" / bench
    bdir.mkdir(parents=True, exist_ok=True)
    for p, idx in enumerate(np.array_split(order, nparts)):
        rows = [r for i in idx for r in qs[i][0]]
        hid = np.stack([qs[i][1] for i in idx])
        stem = bdir / f"part-{p:05d}"
        pd.DataFrame(rows).to_parquet(str(stem) + ".parquet", index=False)
        np.save(str(stem) + ".hidden.npy", hid)
        ids = [qs[i][0][0]["doc_id"] for i in idx]
        done = {"doc_ids": ids, "statuses": [qs[i][0][0]["status"] for i in idx], "n": len(idx),
                "parquet_sha256": g.sha256_file(str(stem) + ".parquet"),
                "hidden_sha256": g.sha256_file(str(stem) + ".hidden.npy"), "finished_at": 0.0}
        Path(str(stem) + ".done.json").write_text(json.dumps(done), encoding="utf-8")


def write_dataset(root, benches=BENCHES, counts=None, d=D_HID, seed=0):
    counts = counts or {"train": 260, "validation": 130, "test": 50}
    (Path(root) / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(benches):
        write_probe_bench(root, b, counts, d=d, seed=seed + i)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("probe_root")
    write_dataset(root)
    return root


@pytest.fixture(scope="module")
def train(dataset):
    return g2.load_probe_data(dataset, "train", BENCHES)


@pytest.fixture(scope="module")
def val(dataset, train):
    return g2.load_probe_data(dataset, "validation", BENCHES, g_clip=train.g_clip)


def pooled_auroc(ps, pr, target="S"):
    """Mean of the per-loop AUROCs (the pooled one also rewards the loop-dependent base rate of S)."""
    return float(np.mean(list(g2.predictive_metrics(ps, pr.predict(ps), target)["per_loop"].values())))


# ----------------------------------------------------------------------------------------------- loading
def test_rejects_other_splits(dataset, train):
    for bad in ("test", "confirmation", "all"):
        with pytest.raises(ValueError):
            g2.load_probe_data(dataset, bad, BENCHES, g_clip=(0, 1))
    with pytest.raises(ValueError):
        g2.load_probe_data(dataset, "validation", BENCHES)          # needs the train clip
    with pytest.raises(ValueError):
        g2.load_probe_data(dataset, "train", ["modarith"])
    with pytest.raises(ValueError):
        g2.fit_probe(g2.load_probe_data(dataset, "validation", BENCHES, g_clip=train.g_clip), 1, "h")


def test_loaded_data_is_split_pure_and_hidden_aligned(dataset, train, val):
    assert train.d == D_HID and train.h.dtype == np.float32 and train.h.shape == (train.N, 3, D_HID)
    assert all(d.startswith("tr-") for d in train.doc_id) and all(d.startswith("va-") for d in val.doc_id)
    assert train.N == 2 * 260 and val.N == 2 * 130
    assert sum(v["failed"] for v in train.n_failed.values()) == int((~train.ok).sum()) > 0
    # the stored hidden state of a known question comes back for that question
    b = BENCHES[0]
    part = np.load(dataset / "data" / "extracted" / b / "part-00000.hidden.npy")
    done = json.loads((dataset / "data" / "extracted" / b / "part-00000.done.json").read_text())
    i = next(i for i, (d, s) in enumerate(zip(done["doc_ids"], done["statuses"])) if d.startswith("tr-") and s == "ok")
    row = int(np.flatnonzero(train.doc_id == done["doc_ids"][i])[0])
    assert np.array_equal(train.h[row], part[i, :3].astype(np.float32))
    # S is built from the known stable-from-k structure: S_t is monotone in t for ok questions
    S = train.targets["S"][train.ok]
    assert (np.diff(S, axis=1) >= 0).all()
    # the signal coordinate of h_t reflects S_t
    assert np.corrcoef(train.h[train.ok][:, :, 0].ravel(), S.ravel())[0, 1] > 0.9
    # failed questions: zeros in h and x, not used for fitting
    assert (train.h[~train.ok] == 0).all()


def test_misaligned_hidden_or_parquet_order_raises(tmp_path):
    for variant in ("swap_done_ids", "hidden_rows_permuted", "statuses"):
        root = tmp_path / variant
        write_dataset(root, benches=("arc_easy",), counts={"train": 60, "validation": 30, "test": 10}, seed=3)
        bdir = root / "data" / "extracted" / "arc_easy"
        dp = bdir / "part-00000.done.json"
        done = json.loads(dp.read_text())
        if variant == "swap_done_ids":
            done["doc_ids"][0], done["doc_ids"][1] = done["doc_ids"][1], done["doc_ids"][0]
        elif variant == "statuses":
            i = done["statuses"].index("ok")
            done["statuses"][i] = "nonfinite"
        else:                      # hidden file rewritten in another order (checksums updated, as a buggy writer would)
            hp = bdir / "part-00000.hidden.npy"
            hid = np.load(hp)
            bad = done["statuses"].index("timeout") if "timeout" in done["statuses"] else None
            assert bad is not None, "fixture needs a failed question in part 0"
            ok_i = done["statuses"].index("ok")
            hid[[bad, ok_i]] = hid[[ok_i, bad]]
            np.save(hp, hid)
            done["hidden_sha256"] = g.sha256_file(hp)
        dp.write_text(json.dumps(done))
        with pytest.raises(ValueError):
            g2.load_probe_data(root, "train", ["arc_easy"])
            g2.load_probe_data(root, "validation", ["arc_easy"], g_clip=(0, 1))


def test_checksum_mismatch_raises(tmp_path):
    write_dataset(tmp_path, benches=("arc_easy",), counts={"train": 20, "validation": 10, "test": 5}, seed=4)
    hp = tmp_path / "data" / "extracted" / "arc_easy" / "part-00000.hidden.npy"
    hid = np.load(hp)
    hid[0, 0, 0] += 1
    np.save(hp, hid)
    with pytest.raises(ValueError, match="checksum"):
        g2.load_probe_data(tmp_path, "train", ["arc_easy"])


def test_targets_on_hand_made_questions():
    # preds per question over loops 1..4; T from loop t to t+1
    pred = np.array([[0, 1, 1, 1], [2, 2, 0, 2], [3, 3, 3, 3], [0, 1, 2, 3]])
    correct = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]])
    T = np.array([["WC", "CC", "CC"], ["WW", "WC", "CW"], ["CC", "CC", "CC"], ["WW", "WW", "WC"]])
    D = np.array([[0.5, 0.0, 0.1], [0.2, 0.3, 0.4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    a = {"n": 4, "pred": pred, "correct": correct, "T": T, "D": D, "ok": np.array([True] * 4)}
    G = np.arange(12, dtype=float).reshape(4, 3)
    t = g2.targets_from_arrays(a, G)
    assert t["S"].tolist() == [[0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]
    assert t["F"].tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]]
    assert t["C"].tolist() == correct[:, :3].tolist()
    assert t["WC"].tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert t["CW"].tolist() == [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert t["logD"][0, 1] == pytest.approx(np.log(1e-12)) and t["logD"][0, 0] == pytest.approx(np.log(0.5))
    assert t["G_raw"].tolist() == G.tolist()


def test_g_clip_from_train_applied_to_validation(train, val):
    lo, hi = np.percentile(train.targets["G_raw"][train.ok], [1, 99])
    assert train.g_clip == (lo, hi) and val.g_clip == (lo, hi)
    assert train.targets["G"].min() >= lo and val.targets["G"].max() <= hi


def test_cheap_feature_columns(train):
    assert train.x.shape == (train.N, 3, 9)
    ok = np.flatnonzero(train.ok)[0]
    a = train.arrays[train.bench[ok]]
    j = ok - train.slices[train.bench[ok]][0]
    assert train.x[ok, 1, 0] == pytest.approx(a["conf"][j, 1]) and train.x[ok, 1, 3] == pytest.approx(a["cdf"][j, 1])
    assert train.x[ok, 0, 4] == 4 and train.x[ok, 0, 5] > np.log(49)
    assert train.x[ok, 2, 6:].tolist() == [0, 0, 1]


# ---------------------------------------------------------------------------------------------- folds / CV
def test_fold_and_es_definitions(train):
    import hashlib

    did = "va-arc_easy-0007"
    assert g2.fold_of(did) == int(hashlib.sha256(("lh-fold-v1|" + did).encode()).hexdigest(), 16) % 5
    assert g2.in_es_slice(did) == (int(hashlib.sha256(("lh-es-v1|" + did).encode()).hexdigest(), 16) % 10 == 0)
    assert set(train.fold) == {0, 1, 2, 3, 4}
    assert 0.05 < train.es.mean() < 0.16
    assert (train.fold == np.array([g2.fold_of(d) for d in train.doc_id])).all()


def test_shared_model_folds_keep_a_questions_loops_together(train, monkeypatch):
    seen = []
    real = g2.cv_scores

    def spy(X, y, folds, grid, task):
        seen.append(folds.copy())
        return real(X, y, folds, grid, task)

    monkeypatch.setattr(g2, "cv_scores", spy)
    g2.fit_probe(train, 2, "h")
    n_ok = int(train.ok.sum())
    f = seen[0]
    assert len(f) == 3 * n_ok
    assert np.array_equal(f[:n_ok], f[n_ok:2 * n_ok]) and np.array_equal(f[:n_ok], f[2 * n_ok:])
    assert np.array_equal(f[:n_ok], train.fold[train.ok])


def test_cv_tie_goes_to_smaller_C_and_larger_alpha():
    assert g2.pick_param({1e-4: 0.5, 1e-3: 0.5, 1e-2: 0.7}, "smaller") == 1e-4
    assert g2.pick_param({1e-4: 0.6, 1e-3: 0.5, 1e-2: 0.5 + 1e-12}, "smaller") == 1e-3
    assert g2.pick_param({1.0: 0.4, 10.0: 0.4, 100.0: 0.9}, "larger") == 10.0
    # a constructed real tie: all-zero features -> every C (and alpha) predicts the same constant
    rng = np.random.default_rng(0)
    X = np.zeros((100, 4), np.float32)
    y = (rng.random(100) < 0.3).astype(float)
    folds = np.arange(100) % 5
    sc = g2.cv_scores(X, y, folds, g2.C_GRID, "clf")
    assert len(set(round(v, 12) for v in sc.values())) == 1
    assert g2.pick_param(sc, "smaller") == min(g2.C_GRID)
    sr = g2.cv_scores(X, rng.normal(size=100), folds, g2.ALPHA_GRID, "reg")
    assert g2.pick_param(sr, "larger") == max(g2.ALPHA_GRID)


def test_cv_picks_by_logloss_not_by_accuracy():
    # informative feature: with a tiny C the model is under-fit, so the CV log-loss must prefer a larger C
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    y = (X[:, 0] * 3 + rng.normal(size=400) > 0).astype(float)
    sc = g2.cv_scores(X, y, np.arange(400) % 5, g2.C_GRID, "clf")
    assert sc[1e-4] > sc[1.0]
    assert g2.pick_param(sc, "smaller") >= 1e-2


def test_standardisation_uses_train_stats_and_floor():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], np.float32)
    sc = g2.fit_scaler(X)
    assert sc["sd"][1] == g2.SD_FLOOR and sc["mean"].tolist() == [3.0, 5.0]
    Z = g2.apply_scaler(np.array([[3.0, 5.0], [5.0, 5.0]], np.float32), sc)
    assert Z[0].tolist() == [0.0, 0.0] and Z[1, 0] == pytest.approx(2 / np.std([1, 3, 5])) and Z[1, 1] == 0


# ------------------------------------------------------------------------------------------------- probes
def test_hidden_state_probe_beats_cheap_features_and_shuffle_control_is_chance(train, val):
    ph, px = g2.fit_probe(train, 1, "h"), g2.fit_probe(train, 1, "x")
    phx = g2.fit_probe(train, 1, "hx")
    shuf = g2.fit_probe(train, 1, "h", labels=g2.shuffle_labels_within(train, train.targets["S"]))
    a_h, a_x, a_hx, a_sh = (pooled_auroc(val, p) for p in (ph, px, phx, shuf))
    assert a_h > 0.9 and a_hx > 0.9 and a_x < 0.6 and 0.4 < a_sh < 0.65
    assert ph.hyper["chosen"]["0"]["C"] in g2.C_GRID
    ops = {n: g2.select_operating_point(g2.probe_policy_data(val, p.predict(val)), BENCHES)
           for n, p in (("h", ph), ("x", px))}
    sav_x = ops["x"]["selected"]["macro_saving"] if ops["x"]["qualified"] else 0.0
    assert ops["h"]["qualified"] and ops["h"]["selected"]["macro_saving"] > sav_x + 0.05


def test_shuffle_labels_preserves_group_counts(train):
    y = train.targets["S"]
    s = g2.shuffle_labels_within(train, y)
    for b in BENCHES:
        a, e = train.slices[b]
        ok = train.ok[a:e]
        assert np.array_equal(y[a:e][ok].sum(0), s[a:e][ok].sum(0))
    assert (s != y).any() and (s[~train.ok] == y[~train.ok]).all()
    assert np.array_equal(s, g2.shuffle_labels_within(train, y))      # seed 0, reproducible


def test_failed_questions_are_not_used_for_fitting(train, monkeypatch):
    shapes = []
    real = g2.cv_scores
    monkeypatch.setattr(g2, "cv_scores", lambda X, y, f, gr, t: (shapes.append(len(y)), real(X, y, f, gr, t))[1])
    g2.fit_probe(train, 1, "h")
    assert set(shapes) == {int(train.ok.sum())} and int((~train.ok).sum()) > 0


def test_shared_and_mlp_families(train, val):
    p2 = g2.fit_probe(train, 2, "hx")
    p3 = g2.fit_probe(train, 3, "h")
    assert set(p2.units) == {0} and set(p3.units) == {0, 1, 2}
    assert pooled_auroc(val, p2) > 0.9 and pooled_auroc(val, p3) > 0.9
    for key, c in p3.hyper["chosen"].items():
        assert c["weight_decay"] in g2.WD_GRID and 0 <= c["best_epoch"] < g2.MLP_MAX_EPOCHS
        assert c["n_fit"] + c["n_es"] == int(train.ok.sum())
    assert p3.units[0]["model"].W1.shape == (g2.MLP_HIDDEN, D_HID)
    assert p3.macs_per_loop() == {1: D_HID * 256 + 256, 2: D_HID * 256 + 256, 3: D_HID * 256 + 256}
    assert p2.macs_per_loop()[1] == D_HID + 9      # h_t + the 9 cheap columns (hx)
    # deterministic: the same fit twice gives identical weights (seed 0)
    p3b = g2.fit_probe(train, 3, "h")
    assert np.array_equal(p3.units[1]["model"].W1, p3b.units[1]["model"].W1)


def test_mlp_weight_decay_tie_goes_to_larger_and_es_slice_is_train_only(train, monkeypatch):
    calls = []

    def fake(Xtr, ytr, Xes, yes, wd, device="cpu"):
        calls.append((len(Xtr), len(Xes), wd))
        return g2.MLPModel(np.zeros((g2.MLP_HIDDEN, Xtr.shape[1])), np.zeros(g2.MLP_HIDDEN), np.zeros((1, g2.MLP_HIDDEN)),
                           np.zeros(1)), 0.5, 3          # identical ES BCE for every wd -> tie

    monkeypatch.setattr(g2, "fit_mlp", fake)
    pr = g2.fit_probe(train, 3, "h")
    assert all(c["weight_decay"] == 1e-2 for c in pr.hyper["chosen"].values())
    n_es = int((train.es & train.ok).sum())
    assert all(c[1] == n_es and c[0] == int(train.ok.sum()) - n_es for c in calls) and len(calls) == 9


def test_ridge_targets_and_secondary_report(train, val):
    pr = g2.fit_probe(train, 1, "h", "logD")
    assert pr.task == "reg" and pr.hyper["chosen"]["0"]["alpha"] in g2.ALPHA_GRID
    with pytest.raises(ValueError):
        g2.fit_probe(train, 2, "h", "logD")
    sec = g2.secondary_report(train, val)
    for key in ("F/h", "C/h", "WC/x", "CW/h", "logD/h", "G/x", "WCvsCW/h"):
        assert key in sec
    assert sec["logD/h"]["metric"] == "spearman" and sec["F/h"]["metric"] == "auroc"
    assert sec["WCvsCW/h"]["n_pairs"] == sec["WCvsCW/h"]["n_wc"] + sec["WCvsCW/h"]["n_cw"]


def test_save_and_reload_probe_round_trip(train, val, tmp_path):
    for fam in (1, 3):
        pr = g2.fit_probe(train, fam, "hx")
        sha_a = g2.save_npz_deterministic(tmp_path / "m.npz", pr.to_arrays())
        g2.save_npz_deterministic(tmp_path / "s.npz", pr.scaler_arrays())
        assert sha_a == g2.save_npz_deterministic(tmp_path / "m2.npz", pr.to_arrays())      # byte-identical
        back = g2.Probe.from_arrays(dict(np.load(tmp_path / "m.npz")), dict(np.load(tmp_path / "s.npz")))
        assert np.allclose(back.predict(val), pr.predict(val), atol=1e-6)


# ------------------------------------------------------------------------------------------------ policies
def test_policy_stops_at_first_threshold_crossing(val):
    scores = np.full((val.N, 3), 0.1)
    ok_rows = np.flatnonzero(val.ok)
    r0, r1, r2 = ok_rows[:3]
    scores[r0] = [0.2, 0.9, 0.95]       # crosses 0.8 first at loop 2
    scores[r1] = [0.95, 0.1, 0.9]       # loop 1
    scores[r2] = [0.1, 0.2, 0.3]        # never -> loop 4
    failed = np.flatnonzero(~val.ok)[0]
    scores[failed] = [0.99, 0.99, 0.99]  # a failed question never stops early
    data = probe_data = g2.probe_policy_data(val, scores)
    b = val.bench[r0]
    a, _ = val.slices[b]
    e, _ = g.run_policy(data[b], "native_gate", 0.8)
    assert e[r0 - a] == 2
    if val.bench[r1] == b:
        assert e[r1 - a] == 1
    if val.bench[r2] == b:
        assert e[r2 - a] == 4
    bf = val.bench[failed]
    af, _ = val.slices[bf]
    ef, cf = g.run_policy(probe_data[bf], "native_gate", 0.8)
    assert ef[failed - af] == 4 and cf[failed - af] == 0


def test_operating_point_is_gate1s_rule(val):
    """Feeding the native gate's own cdf through the probe path reproduces Gate 1's baseline selection exactly."""
    scores = np.concatenate([val.arrays[b]["cdf"][:, :3] for b in BENCHES])
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    op = g2.select_operating_point(g2.probe_policy_data(val, scores), BENCHES)
    ref = g.run_selection(val.arrays, BENCHES)["baselines"]["native_gate"]
    assert op["qualified"] == ref["qualified"]
    assert op["threshold"] == ref["threshold"]
    if op["qualified"]:
        assert op["selected"]["macro_saving"] == ref["selected"]["macro_saving"]
        assert op["selected"]["macro_loss"] == ref["selected"]["macro_loss"]
    assert [r["threshold"] for r in op["grid"]] == ref["grid_thresholds"]


def test_config_choice_tie_breaks():
    def rec(fam, var, q, s, l):
        return {"family": fam, "variant": var, "qualified": q, "saving": s, "loss": l, "name": f"{fam}{var}"}

    recs = [rec(1, "h", True, 0.40, 0.9), rec(2, "h", True, 0.45, 0.8), rec(3, "h", False, 0.9, 5.0),
            rec(3, "hx", True, 0.45, 0.5)]
    assert g2.choose_config(recs)["name"] == "3hx"                                # tie on saving -> smaller loss
    recs[3]["loss"] = 0.8
    recs.append(rec(1, "hx", True, 0.45, 0.8))
    assert g2.choose_config(recs)["name"] == "1hx"                                # tie on both -> simpler family
    recs.append(rec(1, "h", True, 0.45, 0.8 + 1e-13))
    assert g2.choose_config(recs)["name"] == "1h"                                 # then h before hx
    assert g2.choose_config([rec(1, "h", False, 0.5, 2.0)]) is None


def test_overhead_helpers(train):
    pr = g2.fit_probe(train, 3, "h")
    macs = g2.probe_macs(pr)
    assert macs[1] == D_HID * 256 + 256 and g2.loop_equivalents(macs[1], 1e6) == pytest.approx(macs[1] / 1e6)
    ms = g2.measure_probe_ms(pr.units[0]["model"], D_HID, n_repeats=20)
    assert ms > 0
    lin = g2.fit_probe(train, 1, "h").units[0]["model"]
    assert g2.measure_probe_ms(lin, D_HID, n_repeats=20) > 0


# ---------------------------------------------------------------------------------- script / freeze (E2E)
@pytest.fixture(scope="module")
def full_run(dataset, tmp_path_factory):
    root = tmp_path_factory.mktemp("full_root")
    shutil.copytree(dataset / "data", root / "data")
    assert sel.main(["--root", str(root), "--benchmarks", *BENCHES]) == 0
    return root


def test_full_run_outputs_and_hashes(full_run):
    root = full_run
    fz = json.loads((root / "results" / "gate2" / "frozen_selection.json").read_text())
    assert fz["plan_version"] == "v3" and fz["seeds"]["global"] == 0 and fz["benchmarks"] == list(BENCHES)
    assert not fz["chosen_none_qualified"] and fz["P"]["qualified"]
    assert fz["P"]["validation_loss"] <= 1.0 + 1e-9
    assert {"fixed-1", "fixed-2", "fixed-3", "confidence", "entropy", "native_gate", "deployable_kl"} == set(fz["baselines"])
    assert all("validation_saving" in e for e in fz["baselines"].values())
    for tag in ("P", "Q"):
        for key in ("model_file", "standardisation_file"):
            f = root / fz[tag][key]
            assert f.exists() and g.sha256_file(f) == fz[tag][key.replace("file", "sha256")]
    assert fz["standardisation_sha256"]["P"] == fz["P"]["standardisation_sha256"]
    assert len(fz["code"]["code_sha256"]) == 64 and len(fz["all_configs"]) == 8
    assert fz["P"]["config"]["family"] in (1, 2, 3)
    # Q is family 1 on cheap features and does worse than P on this synthetic data
    assert fz["Q"]["config"] == {"family": 1, "variant": "x", "hyper": fz["Q"]["config"]["hyper"]}
    assert (fz["Q"]["validation_saving"] or 0) < fz["P"]["validation_saving"]
    rep = json.loads((root / "results" / "gate2" / "validation_report.json").read_text())
    assert rep["secondary"]["F/h"]["metric"] == "auroc" and "S/Q:family1-x" in rep["secondary"]
    assert rep["n_failed"]["train"]["arc_easy"]["n"] == 260 and rep["n_failed"]["validation"]["openbookqa"]["n"] == 130
    md = (root / "docs" / "GATE2_VALIDATION_REPORT.md").read_text()
    for needle in ("Failed questions", "Probe configurations", "Controls", "Secondary predictive", "WC vs CW"):
        assert needle in md
    assert not (root / "results" / "gate2_dryrun").exists()


def test_frozen_probe_reproduces_its_validation_point(full_run, dataset, train, val):
    root = full_run
    fz = json.loads((root / "results" / "gate2" / "frozen_selection.json").read_text())
    pr = g2.Probe.from_arrays(dict(np.load(root / fz["P"]["model_file"])), dict(np.load(root / fz["P"]["standardisation_file"])))
    s = g2.evaluate_probe_on(val, pr.predict(val), fz["P"]["threshold"])
    assert s["macro"]["saving"] == pytest.approx(fz["P"]["validation_saving"]) and s["macro"]["loss"] <= 1.0 + 1e-9


def test_freeze_refuses_overwrite_and_refreeze_is_reproducible(full_run, capsys):
    root = full_run
    path = root / "results" / "gate2" / "frozen_selection.json"
    before = path.read_text()
    assert sel.main(["--root", str(root), "--benchmarks", *BENCHES]) == 2          # refuses without --force-refreeze
    assert path.read_text() == before
    with pytest.raises(FileExistsError):
        g2.freeze_gate2(root, chosen_rec=None, chosen_probe=None, q_rec=None, q_probe=None,
                        baseline_selection={}, benchmarks=BENCHES, all_records=[])
    assert sel.main(["--root", str(root), "--benchmarks", *BENCHES, "--force-refreeze"]) == 0
    archived = list((root / "results" / "gate2" / "archive").glob("frozen_selection.*.json"))
    assert len(archived) == 1 and archived[0].read_text() == before

    def strip(text):
        d = json.loads(text)
        d.pop("timestamp_utc")
        return json.dumps(d, sort_keys=True)

    assert strip(path.read_text()) == strip(before)                               # identical apart from the timestamp
    fz = json.loads(before)
    assert g.sha256_file(root / fz["P"]["model_file"]) == fz["P"]["model_sha256"]


def test_dry_run_end_to_end_reads_only_train_and_validation(tmp_path, monkeypatch):
    write_dataset(tmp_path, counts={"train": 120, "validation": 60, "test": 30}, seed=9)
    # sealed data that must never be opened, and test rows that sit in the same parquet parts
    conf = tmp_path / "data" / "extracted_confirm" / "arc_easy"
    conf.mkdir(parents=True)
    (conf / "part-00000.parquet").write_bytes(b"not parquet")
    (conf / "part-00000.done.json").write_text("{}")
    qc = tmp_path / "data" / "questions_confirm"
    qc.mkdir(parents=True)
    (qc / "arc_easy.jsonl.gz").write_bytes(b"x")
    opened, read_filters = [], []
    real_open, real_read, real_load = builtins.open, g2.pq.read_table, np.load

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_read(path, *a, **k):
        opened.append(str(path))
        read_filters.append(k.get("filters"))
        return real_read(path, *a, **k)

    def spy_load(f, *a, **k):
        opened.append(str(f))
        return real_load(f, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(g2.pq, "read_table", spy_read)
    monkeypatch.setattr(np, "load", spy_load)
    assert sel.main(["--root", str(tmp_path), "--benchmarks", *BENCHES, "--dry-run", "--families", "1", "2", "3"]) == 0
    monkeypatch.undo()
    assert opened and not [p for p in opened if "confirm" in p.replace("\\", "/").lower()]
    assert read_filters and all(f in ([("split", "==", "train")], [("split", "==", "validation")]) for f in read_filters)
    assert (tmp_path / "results" / "gate2_dryrun" / "frozen_selection.json").exists()
    assert not (tmp_path / "results" / "gate2").exists()                      # dry run never takes the real frozen path
    assert (tmp_path / "docs" / "GATE2_VALIDATION_REPORT_DRYRUN.md").exists()
    rep = json.loads((tmp_path / "results" / "gate2_dryrun" / "validation_report.json").read_text())
    assert rep["families"] == [1] and {c["family"] for c in rep["configs"]} == {1}
    assert rep["n_failed"]["train"]["arc_easy"]["n"] == 120                   # test rows (30 per benchmark) never counted


def test_forbidden_paths_and_benchmarks_are_rejected(tmp_path):
    with pytest.raises(AssertionError):
        g2._assert_allowed_path(tmp_path / "data" / "extracted_confirm" / "arc_easy")
    with pytest.raises(AssertionError):
        g2._assert_allowed_path("/x/data/questions_confirm/a.jsonl.gz")
    with pytest.raises(AssertionError):
        sel.assert_inputs_allowed(tmp_path, ["modarith"])
    with pytest.raises(AssertionError):
        sel.assert_inputs_allowed(tmp_path / "data" / "extracted_confirm", ["arc_easy"])
    with pytest.raises(AssertionError):
        sel.assert_inputs_allowed(tmp_path / "data_confirm_copy", ["arc_easy"])
    sel.assert_inputs_allowed(tmp_path, list(sel.BENCHMARKS))
