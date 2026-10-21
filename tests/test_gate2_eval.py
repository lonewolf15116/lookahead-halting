"""Gate 2 confirmation evaluation: verdict table, paired bootstrap, and the guards of scripts/gate2_confirm.py.
No real data is opened: the guard tests stop before any data is read."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from lookahead import gate1 as g
from lookahead import gate2_eval as ev

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import gate2_confirm as cf  # noqa: E402


@pytest.mark.parametrize("L,E,H,want", [
    (True, True, True, "PASS"), (True, True, False, "PARTIAL"), (False, True, True, "FAIL"), (False, True, False, "FAIL"),
    (False, False, True, "FAIL"), (False, False, False, "FAIL"), (True, False, True, "FAIL"), (True, False, False, "FAIL")])
def test_verdict_table_is_exhaustive(L, E, H, want):
    assert ev.verdict(L, E, H) == want


def _toy(n=300, seed=0):
    """Two benchmarks, policies: fixed-4 (reference), P, Q, one baseline, B, B_cap."""
    rng = np.random.default_rng(seed)
    benches = ["arc_easy", "openbookqa"]
    C, E = {}, {}
    for b in benches:
        c4 = (rng.random(n) < 0.7).astype(float)
        cs = [c4, c4, c4, c4, c4, c4]
        es = [np.full(n, 4.0), np.full(n, 2.0), np.full(n, 3.0), np.full(n, 3.0), np.full(n, 3.0), np.full(n, 3.0)]
        C[b], E[b] = np.stack(cs, 1), np.stack(es, 1)
    pols = [("fixed-4", "reference", "base", "fixed", 4), ("P", "candidate", "P", "native_gate", 0.5),
            ("Q", "control_Q", "Q", "native_gate", 0.5), ("fixed-3", "baseline", "base", "fixed", 3),
            ("B", "amendment_B", "base", "fixed", 3), ("B_cap", "sensitivity", "base", "fixed", 3)]
    return pols, C, E, benches


def test_analysis_on_toy_policies_is_exact_and_paired():
    pols, C, E, benches = _toy()
    a = ev.analyse(pols, C, E, benches, n_boot=200, seed=0)
    pt = a["point"]
    assert pt["P"]["macro_saving"] == pytest.approx(0.5) and pt["P"]["macro_loss"] == pytest.approx(0.0)
    assert pt["Q"]["macro_saving"] == pytest.approx(0.25)
    # P exits at 2 for everyone, Q at 3, fixed-3 at 3: delta is exactly constant -> a degenerate paired CI at that value
    d = a["original"]["delta_base"]
    assert d["delta"] == pytest.approx(0.25) and d["ci"][0] == pytest.approx(0.25) and d["ci"][1] == pytest.approx(0.25)
    assert a["original"]["delta_ctrl_P_minus_Q"]["ci"][0] == pytest.approx(0.25)
    assert a["original"]["verdict"] == "PASS"            # same accuracy for all, P saves 0.25 more than everyone
    assert a["amendment"]["P_minus_B"]["delta"] == pytest.approx(0.25)
    assert a["amendment"]["per_benchmark_cap_sensitivity"]["policies_over_cap"]["P"] == []


def test_margin_and_ci_both_required_for_E():
    pols, C, E, benches = _toy()
    for b in benches:                      # P now exits at 3 like the baseline: delta 0 -> E false -> FAIL
        E[b][:, 1] = 3.0
    a = ev.analyse(pols, C, E, benches, n_boot=100, seed=0)
    assert not a["original"]["E_effect"] and a["original"]["verdict"] == "FAIL"


def test_budget_failure_gives_fail_even_with_large_saving():
    pols, C, E, benches = _toy()
    for b in benches:
        C[b][:, 1] = 0.0                   # P is always wrong -> big loss
    a = ev.analyse(pols, C, E, benches, n_boot=100, seed=0)
    assert not a["original"]["L_budget"] and a["original"]["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------- script guards
@pytest.fixture()
def fake_root(tmp_path):
    (tmp_path / "results/gate2").mkdir(parents=True)
    for f in ("frozen_selection.json", "baseline_B_frozen.json"):
        shutil.copy(REPO / "results/gate2" / f, tmp_path / "results/gate2" / f)
    return tmp_path


def _shas(root):
    return (g.sha256_file(root / "results/gate2/frozen_selection.json")[:12],
            g.sha256_file(root / "results/gate2/baseline_B_frozen.json")[:12])


def _args(root, *extra):
    f, b = _shas(root)
    return ["--root", str(root), "--confirm-one-time-gate2-evaluation", "--frozen-sha", f, "--baseline-sha", b,
            "--reviewer-approval-ref", "reviewer message 2026-10-21", *extra]


def test_refuses_without_exactly_one_mode(fake_root):
    assert cf.main(["--root", str(fake_root)]) == 2
    assert cf.main(["--root", str(fake_root), "--confirm-one-time-gate2-evaluation", "--rehearsal-on-validation"]) == 2


def test_refuses_wrong_or_missing_sha_and_writes_no_marker(fake_root):
    f, b = _shas(fake_root)
    with pytest.raises(SystemExit):
        cf.main(["--root", str(fake_root), "--confirm-one-time-gate2-evaluation", "--frozen-sha", "0" * 12,
                 "--baseline-sha", b, "--reviewer-approval-ref", "reviewer message"])
    with pytest.raises(SystemExit):
        cf.main(["--root", str(fake_root), "--confirm-one-time-gate2-evaluation", "--baseline-sha", b,
                 "--reviewer-approval-ref", "reviewer message"])
    assert not (fake_root / "results/gate2/CONFIRMATION_STARTED.json").exists()


def test_refuses_without_approval_ref(fake_root):
    f, b = _shas(fake_root)
    rc = cf.main(["--root", str(fake_root), "--confirm-one-time-gate2-evaluation", "--frozen-sha", f, "--baseline-sha", b])
    assert rc == 2 and not (fake_root / "results/gate2/CONFIRMATION_STARTED.json").exists()


@pytest.mark.parametrize("marker", ["CONFIRMATION_STARTED.json", "CONFIRMATION_EVALUATED.json"])
def test_refuses_when_a_marker_exists(fake_root, marker):
    (fake_root / "results/gate2" / marker).write_text("{}", encoding="utf-8")
    assert cf.main(_args(fake_root)) == 2


def test_refuses_a_changed_B_vector(fake_root):
    p = fake_root / "results/gate2/baseline_B_frozen.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    j["B"]["depths"] = [2, 2, 2, 3, 3, 3]
    p.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit):
        cf.main(_args(fake_root))


def test_refuses_baseline_made_against_another_frozen_file(fake_root):
    p = fake_root / "results/gate2/baseline_B_frozen.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    j["frozen_selection_sha256"] = "0" * 64
    p.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit):
        cf.main(["--root", str(fake_root), "--confirm-one-time-gate2-evaluation", "--frozen-sha",
                 g.sha256_file(fake_root / "results/gate2/frozen_selection.json")[:12], "--baseline-sha",
                 g.sha256_file(p)[:12], "--reviewer-approval-ref", "reviewer message"])


def test_confirmation_paths_are_only_open_inside_the_context_manager():
    from lookahead import gate2 as g2
    with pytest.raises(AssertionError):
        g2._assert_allowed_path("data/extracted_confirm/mmlu")
    with ev.allow_confirmation_paths():
        g2._assert_allowed_path("data/extracted_confirm/mmlu")
    with pytest.raises(AssertionError):
        g2._assert_allowed_path("data/extracted_confirm/mmlu")


# ------------------------------------------------------------------------ no overrides (reviewer request)
@pytest.mark.parametrize("flag", ["--allow-rerun", "--allow-dirty"])
def test_override_flags_do_not_exist(fake_root, flag):
    with pytest.raises(SystemExit):
        cf.main(_args(fake_root, flag))


def test_real_run_requires_the_shuffled_control(fake_root):
    assert cf.main(_args(fake_root, "--skip-control")) == 2
    assert not (fake_root / "results/gate2/CONFIRMATION_STARTED.json").exists()


def test_refuses_with_modified_tracked_files_and_no_override(fake_root, monkeypatch):
    monkeypatch.setattr(cf, "git_state", lambda root: {"head": "x", "modified_tracked": [" M src/lookahead/gate2.py"]})
    assert cf.main(_args(fake_root)) == 2
    assert not (fake_root / "results/gate2/CONFIRMATION_STARTED.json").exists()


# --------------------------------------------------------------------- confirmation identity pre-flight
BENCH = ["arc_easy", "openbookqa"]
COUNTS = {"arc_easy": 5, "openbookqa": 3}


def _build_confirm_tree(root: Path):
    """Synthetic confirmation artifact: ids csv, manifest, audit, evidence, check report, two parts per benchmark."""
    import csv
    import hashlib
    (root / "data/manifests").mkdir(parents=True)
    (root / "results/gate2").mkdir(parents=True)
    ids, parts, comb = {}, {}, hashlib.sha256()
    for b in BENCH:
        ids[b] = [f"confirm:{b}/{i}" for i in range(COUNTS[b])]
        bdir = root / "data/extracted_confirm" / b
        bdir.mkdir(parents=True)
        parts[b] = []
        for k, chunk in enumerate((ids[b][:2], ids[b][2:])):
            stem = bdir / f"part-{k:05d}"
            (Path(str(stem) + ".parquet")).write_bytes(f"pq-{b}-{k}".encode())
            (Path(str(stem) + ".hidden.npy")).write_bytes(f"hid-{b}-{k}".encode())
            ph, hh = g.sha256_file(str(stem) + ".parquet"), g.sha256_file(str(stem) + ".hidden.npy")
            (Path(str(stem) + ".done.json")).write_text(json.dumps(
                {"n": len(chunk), "doc_ids": chunk, "statuses": ["ok"] * len(chunk), "parquet_sha256": ph,
                 "hidden_sha256": hh}), encoding="utf-8")
            parts[b].append({"part": f"part-{k:05d}", "n": len(chunk), "parquet_sha256": ph, "hidden_sha256": hh})
            comb.update(ph.encode())
            comb.update(hh.encode())
    with open(root / "data/manifests/confirmation_ids.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "stratum", "doc_id", "split"])
        for b in BENCH:
            for d in ids[b]:
                w.writerow([b, b, d, "confirmation"])
    (root / "results/gate2/confirmation_audit.json").write_text("{}", encoding="utf-8")
    (root / "results/gate2/extraction_check_confirm.json").write_text('{"ok": true}', encoding="utf-8")
    pin = {"ids_sha256": g.sha256_file(root / "data/manifests/confirmation_ids.csv"),
           "audit_sha256": g.sha256_file(root / "results/gate2/confirmation_audit.json"),
           "check_report_sha256": g.sha256_file(root / "results/gate2/extraction_check_confirm.json"),
           "combined_parts_sha256": comb.hexdigest()}
    manifest = {"ids_sha256": pin["ids_sha256"], "audit_sha256": pin["audit_sha256"],
                "benchmarks": {b: {"n_questions": COUNTS[b]} for b in BENCH}}
    (root / "data/manifests/confirmation.json").write_text(json.dumps(manifest), encoding="utf-8")
    pin["manifest_sha256"] = g.sha256_file(root / "data/manifests/confirmation.json")
    evidence = {"ids_sha256": pin["ids_sha256"], "manifest_sha256": pin["manifest_sha256"],
                "check_report_sha256": pin["check_report_sha256"], "combined_parts_sha256": pin["combined_parts_sha256"],
                "outcomes_inspected": False, "total_records": sum(COUNTS.values()), "parts": parts}
    (root / "results/gate2/confirmation_extraction_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return pin


def test_identity_preflight_accepts_the_audited_artifact(tmp_path):
    pin = _build_confirm_tree(tmp_path)
    assert ev.confirmation_identity_preflight(tmp_path, BENCH, COUNTS, pin) == []


def test_identity_preflight_rejects_wrong_counts(tmp_path):
    pin = _build_confirm_tree(tmp_path)
    assert ev.confirmation_identity_preflight(tmp_path, BENCH, {"arc_easy": 6, "openbookqa": 3}, pin)


def test_identity_preflight_rejects_a_swapped_document_id(tmp_path):
    pin = _build_confirm_tree(tmp_path)
    dp = tmp_path / "data/extracted_confirm/arc_easy/part-00000.done.json"
    d = json.loads(dp.read_text(encoding="utf-8"))
    d["doc_ids"][0] = "confirm:arc_easy/INTRUDER"
    dp.write_text(json.dumps(d), encoding="utf-8")
    bad = ev.confirmation_identity_preflight(tmp_path, BENCH, COUNTS, pin)
    assert any("doc-id set differs" in b for b in bad)


def test_identity_preflight_rejects_a_failed_record(tmp_path):
    pin = _build_confirm_tree(tmp_path)
    dp = tmp_path / "data/extracted_confirm/openbookqa/part-00001.done.json"
    d = json.loads(dp.read_text(encoding="utf-8"))
    d["statuses"][0] = "timeout"
    dp.write_text(json.dumps(d), encoding="utf-8")
    assert any("failed records" in b for b in ev.confirmation_identity_preflight(tmp_path, BENCH, COUNTS, pin))


def test_identity_preflight_rejects_changed_file_bytes_and_a_wrong_pinned_hash(tmp_path):
    pin = _build_confirm_tree(tmp_path)
    (tmp_path / "data/extracted_confirm/arc_easy/part-00001.hidden.npy").write_bytes(b"tampered")
    assert any("bytes differ" in b for b in ev.confirmation_identity_preflight(tmp_path, BENCH, COUNTS, pin))
    pin2 = _build_confirm_tree(tmp_path / "other")
    pin2["ids_sha256"] = "0" * 64
    assert ev.confirmation_identity_preflight(tmp_path / "other", BENCH, COUNTS, pin2)


def test_the_real_pinned_constants_match_the_committed_manifests():
    pin = ev.PINNED_CONFIRMATION_HASHES
    assert sum(ev.EXPECTED_CONFIRMATION_COUNTS.values()) == 4869
    assert g.sha256_file(REPO / "data/manifests/confirmation_ids.csv") == pin["ids_sha256"]
    assert g.sha256_file(REPO / "data/manifests/confirmation.json") == pin["manifest_sha256"]
    assert g.sha256_file(REPO / "results/gate2/confirmation_audit.json") == pin["audit_sha256"]
    assert g.sha256_file(REPO / "results/gate2/extraction_check_confirm.json") == pin["check_report_sha256"]


def test_script_modes_are_exclusive_and_identity_mode_writes_nothing(fake_root):
    assert cf.main(["--root", str(fake_root), "--check-confirmation-identity", "--rehearsal-on-validation"]) == 2
    before = sorted(p.name for p in (fake_root / "results/gate2").iterdir())
    cf.main(["--root", str(fake_root), "--check-confirmation-identity"])      # fails (no data in the fake root), exit 3
    assert sorted(p.name for p in (fake_root / "results/gate2").iterdir()) == before
