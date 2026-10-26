"""C1 one-time TEST script, on synthetic data only: verdict table, paired bootstrap, frozen selections applied unchanged,
preflight refusals, marker-before-read ordering, no reruns, and the scoped unsealing of the G3a loader."""
import copy
import importlib.util
import json
import math
import shutil
import sys

import numpy as np
import pytest

from lookahead import c1 as C1
from lookahead import c1_test as T
from lookahead import gate1 as g1
from lookahead import phase3_build as P
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R
from lookahead.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_phase3_g3a import synth  # noqa: E402

_spec = importlib.util.spec_from_file_location("c1_test_run", REPO_ROOT / "scripts" / "c1_test_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)


def _test_data(seed=40, n=60):
    return synth(benches=G.PRIMARY + T.OOD, n=n, seed=seed, source="test", prefix="t:")


@pytest.fixture(scope="module")
def real_frozen():
    fz = json.loads((REPO_ROOT / "results/c1/frozen.json").read_text(encoding="utf-8"))
    with np.load(REPO_ROOT / "results/c1/pi1x_models.npz") as npz:
        models = T.load_models(npz)
    return fz, models


# ------------------------------------------------------------------------------------------------ verdict table
def _s(loss):
    return {b: {"loss_pp": loss} for b in G.PRIMARY}


def test_verdict_table_is_exhaustive_ordered_and_has_the_right_boundaries():
    assert T.verdict({**_s(0.5), "hellaswag": {"loss_pp": 1.01}}, 0.10, 0.08, 0.12) == "FAIL (budget)"   # veto first
    assert T.verdict(_s(1.0), 0.02, 0.001, 0.04) == "PASS"                    # observed exactly 0.02, L > 0
    assert T.verdict(_s(0.5), 0.02, 0.0, 0.04) == "INCONCLUSIVE"              # L must be > 0
    assert T.verdict(_s(0.5), 0.0199, 0.01, 0.03) == "INCONCLUSIVE"           # observed below delta, U >= delta
    assert T.verdict(_s(0.5), 0.005, -0.01, 0.0199) == "FAIL (no meaningful advantage)"
    assert T.verdict(_s(0.5), 0.01, 0.0, 0.02) == "INCONCLUSIVE"              # U exactly 0.02 does not rule out delta
    assert T.verdict(_s(0.5), -0.05, -0.06, -0.04) == "FAIL (no meaningful advantage)"


# ------------------------------------------------------------------------------------------------ bootstrap
def test_paired_macro_point_value_pairing_and_determinism():
    rng = np.random.default_rng(0)
    mk = lambda e: {"e": e, "halted_by_score": e < 4, "m": 4, "ids": np.arange(len(e)).astype(str)}   # noqa: E731
    a = {b: mk(rng.integers(1, 5, 50)) for b in G.PRIMARY}
    c = {b: mk(rng.integers(1, 5, 50)) for b in G.PRIMARY}
    idx = T.boot_indices({b: 50 for b in G.PRIMARY}, 500, 2026)
    r = T.paired_macro(a, c, G.PRIMARY, idx)
    manual = np.mean([((1 - a[b]["e"] / 4) - (1 - c[b]["e"] / 4)).mean() for b in G.PRIMARY])
    assert r["delta"] == pytest.approx(manual) and r["L"] <= r["delta"] <= r["U"]
    r2 = T.paired_macro(a, c, G.PRIMARY, T.boot_indices({b: 50 for b in G.PRIMARY}, 500, 2026))
    assert (r2["L"], r2["U"]) == (r["L"], r["U"])
    bad = copy.deepcopy(c)
    bad["hellaswag"]["ids"] = bad["hellaswag"]["ids"][::-1].copy()
    with pytest.raises(AssertionError):
        T.paired_macro(a, bad, G.PRIMARY, idx)


def test_net_bootstrap_uses_the_c1_net_accounting():
    e = np.array([2, 3, 4, 1])
    p = {"openbookqa": {"e": e, "halted_by_score": np.array([True, False, False, True]), "m": 3, "ids": np.array(list("abcd"))}}
    z = {"openbookqa": {"e": e, "halted_by_score": np.zeros(4, bool), "m": 3, "ids": np.array(list("abcd"))}}
    rho = {"openbookqa": {"rho_score": 0.1, "rho_rest": 0.02}}
    idx = T.boot_indices({"openbookqa": 4}, 50, 1)
    got = T.paired_macro(p, z, ["openbookqa"], idx, key="net", rho_a=rho)["delta"]
    net = C1.net_saving(e, p["openbookqa"]["halted_by_score"], 3, 0.1, 0.02)
    assert got == pytest.approx(net - (1 - e.mean() / 4))


# ------------------------------------------------------------------------------------------------ evaluation
def test_frozen_selections_are_applied_unchanged_and_the_frozen_dict_is_not_modified(real_frozen):
    fz, models = real_frozen
    before = json.dumps(fz, sort_keys=True, default=str)
    d = _test_data()
    rep = T.evaluate_c1(fz, models, d, n_boot=300)
    assert json.dumps(fz, sort_keys=True, default=str) == before
    assert rep["verdict"] in T.VERDICTS
    S = T.pi1x_test_stat(d, models)
    for b in G.PRIMARY:
        s = T._sel(fz["primary"]["pi1-x"])[b]
        r = np.flatnonzero(d.bench == b)
        e = R.exit_loops(S[r], d.ok[r], s["tau"], s["m"])
        assert rep["per_benchmark"]["pi1-x"][b]["saving"] == pytest.approx(1 - e.mean() / 4)
    assert set(rep["secondary_ood"]) == set(T.OOD)
    assert rep["secondary_all_baselines"]["strongest_on_test_point"] in G.BASELINES
    assert sum(rep["secondary_all_baselines"]["selection_frequency"].values()) == 300
    assert "secondary_net" in rep and rep["primary"]["pass_meaning"].startswith("L > 0")
    assert T.evaluate_c1(fz, models, d, n_boot=300)["primary"] == rep["primary"]          # deterministic


def test_models_load_with_nine_finite_coefficients_per_loop(real_frozen):
    _, models = real_frozen
    assert len(models) == 3 and all(m.coef.shape == (9,) and np.isfinite(m.coef).all() for _, m in models)


# ------------------------------------------------------------------------------------------------ unsealing scope
def test_unsealed_admits_only_the_canonical_test_directory(tmp_path):
    base = tmp_path / "data" / T.TEST_DIR
    (base / "openbookqa").mkdir(parents=True)
    (tmp_path / "data" / "extracted_phase3_dev").mkdir(parents=True)
    with pytest.raises(PermissionError):
        G._guard(base / "openbookqa")                                       # sealed outside the context
    with T.unsealed(tmp_path):
        G._guard(base)
        G._guard(base / "openbookqa" / "part-00000.done.json")
        for bad in (tmp_path / "data" / "extracted_phase3_dev" / "openbookqa",            # another extraction
                    tmp_path / "data" / "extracted" / "arc_easy",                         # Phase 1
                    tmp_path / "data" / "extracted_confirm" / "arc_easy",                 # Gate 2 confirmation
                    base / ".." / "extracted_phase3_dev" / "x",                           # traversal out
                    base / ".." / ".." / "results",                                       # traversal further out
                    tmp_path / "elsewhere" / T.TEST_DIR / "openbookqa",                   # same name, wrong place
                    "data/extracted_phase3_test/openbookqa"):                             # relative: not under root
            with pytest.raises(PermissionError):
                G._guard(bad)
    with pytest.raises(PermissionError):
        G._guard(base / "openbookqa")                                       # restored afterwards
    with pytest.raises(RuntimeError):
        with T.unsealed(tmp_path):
            raise RuntimeError("boom")
    with pytest.raises(PermissionError):
        G._guard(base / "x")                                                # restored even after an exception


def test_unsealed_refuses_a_symlink_that_escapes(tmp_path):
    base = tmp_path / "data" / T.TEST_DIR
    base.mkdir(parents=True)
    outside = tmp_path / "data" / "extracted_phase3_dev"
    outside.mkdir(parents=True)
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with T.unsealed(tmp_path):
        with pytest.raises(PermissionError):
            G._guard(base / "link" / "openbookqa")


# ------------------------------------------------------------------------------------------------ the guarded script
@pytest.fixture()
def root(tmp_path):
    fz = json.loads((REPO_ROOT / "results/c1/frozen.json").read_text(encoding="utf-8"))
    files = set(RUN.PINNED) | set(fz["provenance"]["code"]) | set(RUN.BUNDLE) | {P.IDS["test"], P.IDS["dev"]}
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, tmp_path / rel)
    _fake_test_extraction(tmp_path)
    return tmp_path


def _fake_test_extraction(root):
    """Metadata-valid placeholder parts (real TEST ids and manifest counts; the part files are random bytes)."""
    lines = (root / P.IDS["test"]).read_text(encoding="utf-8").splitlines()[1:]
    by_b = {}
    for ln in lines:
        b, _, doc, _ = ln.split(",")
        by_b.setdefault(b, []).append(doc)
    for b, ids in by_b.items():
        d = root / "data" / T.TEST_DIR / b
        d.mkdir(parents=True, exist_ok=True)
        (d / "part-00000.parquet").write_bytes(b"not a real parquet " + b.encode())
        (d / "part-00000.hidden.npy").write_bytes(b"not a real array " + b.encode())
        done = {"parquet_sha256": g1.sha256_file(d / "part-00000.parquet"),
                "hidden_sha256": g1.sha256_file(d / "part-00000.hidden.npy"),
                "doc_ids": ids, "statuses": ["ok"] * len(ids), "n": len(ids)}
        (d / "part-00000.done.json").write_text(json.dumps(done), encoding="utf-8")


def test_preflight_refuses_a_changed_artifact_and_touches_nothing(root):
    sha = RUN.bundle_sha(root)
    p = root / "results/c1/frozen.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    called = []
    assert RUN.main(["--approved-code-sha", sha], root=root, loader=lambda r: called.append(1)) == 2
    assert not called and not (root / RUN.MARKER).exists()


def test_preflight_refuses_unapproved_code_and_changed_frozen_code(root):
    assert RUN.main(["--approved-code-sha", "0" * 64], root=root, loader=lambda r: None) == 2
    sha = RUN.bundle_sha(root)
    f = root / "src/lookahead/e1.py"
    f.write_text(f.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert RUN.main(["--approved-code-sha", sha], root=root, loader=lambda r: None) == 2
    assert not (root / RUN.MARKER).exists()


def test_marker_exists_before_any_test_read_and_reruns_are_refused(root):
    sha = RUN.bundle_sha(root)
    seen = {}

    def loader(r):
        seen["marker_before_read"] = (r / RUN.MARKER).exists()
        seen["inside_unsealed"] = True
        G._guard(r / "data" / T.TEST_DIR / "openbookqa")                # allowed only inside unsealed(root)
        with pytest.raises(PermissionError):
            G._guard(r / "data" / "extracted_phase3_dev" / "openbookqa")   # and nothing else
        return _test_data(seed=41)

    assert RUN.main(["--approved-code-sha", sha], root=root, loader=loader) == 0
    assert seen == {"marker_before_read": True, "inside_unsealed": True}
    rep = json.loads((root / RUN.REPORT).read_text(encoding="utf-8"))
    assert rep["verdict"] in T.VERDICTS and rep["n_boot"] == 10_000 and rep["seed"] == 2026
    m = json.loads((root / RUN.MARKER).read_text(encoding="utf-8"))
    assert m["test_code_bundle_sha256"] == sha and m["pinned"] == RUN.PINNED
    calls = []
    assert RUN.main(["--approved-code-sha", sha], root=root, loader=lambda r: calls.append(1)) == 2
    assert not calls                                                       # second run refused before any read


def test_a_failed_run_keeps_the_marker(root):
    sha = RUN.bundle_sha(root)

    def boom(r):
        raise RuntimeError("disk error")

    with pytest.raises(RuntimeError):
        RUN.main(["--approved-code-sha", sha], root=root, loader=boom)
    assert (root / RUN.MARKER).exists() and not (root / RUN.REPORT).exists()
    fail = json.loads((root / RUN.FAILED).read_text(encoding="utf-8"))
    assert fail["stage"] == "load TEST" and fail["exception_type"] == "RuntimeError"
    assert "disk error" not in json.dumps(fail)                              # no message, no values
    assert not (root / (RUN.FAILED + ".tmp")).exists()
    assert RUN.main(["--approved-code-sha", sha], root=root, loader=lambda r: None) == 2


def test_test_code_never_selects_or_fits():
    src = open(T.__file__, encoding="utf-8").read()
    for forbidden in ("select_all(", "select_risk(", "fit_e1(", "crossfit", "fit_logistic"):
        assert forbidden not in src


# ------------------------------------------------------------------------------------------------ metadata preflight
@pytest.mark.parametrize("damage", ["missing_part", "count", "foreign_id", "checksum", "status", "manifest"])
def test_metadata_preflight_refuses_before_the_marker(root, damage):
    sha = RUN.bundle_sha(root)
    d = root / "data" / T.TEST_DIR / "hellaswag"
    done = json.loads((d / "part-00000.done.json").read_text(encoding="utf-8"))
    if damage == "missing_part":
        (d / "part-00000.done.json").unlink()
    elif damage == "count":
        done["doc_ids"], done["n"], done["statuses"] = done["doc_ids"][:-1], done["n"] - 1, done["statuses"][:-1]
    elif damage == "foreign_id":
        done["doc_ids"][0] = "p3:hellaswag/train@r999999999"
    elif damage == "checksum":
        (d / "part-00000.parquet").write_bytes(b"changed")
    elif damage == "status":
        done["statuses"][0] = "failed"
    elif damage == "manifest":
        m = root / "data/manifests/phase3_questions.json"
        m.write_text(m.read_text(encoding="utf-8") + " ", encoding="utf-8")
    if damage in ("count", "foreign_id", "status"):
        (d / "part-00000.done.json").write_text(json.dumps(done), encoding="utf-8")
    calls = []
    assert RUN.main(["--approved-code-sha", sha], root=root, loader=lambda r: calls.append(1)) == 2
    assert not calls and not (root / RUN.MARKER).exists()


def test_metadata_preflight_passes_on_the_fixture_and_reads_no_outcome(root, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(pq, "read_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parquet read in preflight")))
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("array read in preflight")))
    meta = T.metadata_preflight(root)
    assert meta == {"openbookqa": {"n": 3000, "parts": 1}, "commonsense_qa": {"n": 1000, "parts": 1},
                    "hellaswag": {"n": 750, "parts": 1}, "piqa": {"n": 1837, "parts": 1}, "social_iqa": {"n": 1930, "parts": 1}}


def test_strongest_baseline_is_labelled_test_selected(real_frozen):
    fz, models = real_frozen
    rep = T.evaluate_c1(fz, models, _test_data(seed=42), n_boot=100)
    assert "TEST-selected" in rep["secondary_all_baselines"]["label"]
