"""Phase 3 D-a: row eligibility and selection on the frozen artifacts, the pass rule, and refusals (no model, no data download)."""
import importlib.util
import json

import pytest

from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("da", REPO_ROOT / "scripts" / "diagnose_short_context_ood.py")
DA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DA)


@pytest.fixture(scope="module")
def allocated():
    alloc = P.read_allocation(REPO_ROOT)
    return {r["text_hash"] for role in alloc.values() for _, rows in role.values() for r in rows}


@pytest.mark.parametrize("bench", DA.BENCHES)
def test_eligible_rows_are_unused_train_rows_outside_every_allocation(bench, allocated):
    rows = DA.eligible_rows(REPO_ROOT, bench, allocated)
    assert len(rows) >= DA.N
    assert all(r["benchmark"] == bench and r["source_split"] == "train" for r in rows)
    assert not {r["text_hash"] for r in rows} & allocated
    # the frozen allocation of these benchmarks is validation-only (TEST OOD); no selected row is in a sealed or DEV+ list
    alloc = P.read_allocation(REPO_ROOT)
    keys = {(b, s, r["row"]) for role in alloc.values() for b, (s, rs) in role.items() for r in rs}
    assert not {(bench, "train", r["row"]) for r in rows} & keys


@pytest.mark.parametrize("bench", DA.BENCHES)
def test_selection_is_deterministic_sized_and_fingerprinted(bench, allocated):
    rows = DA.eligible_rows(REPO_ROOT, bench, allocated)
    a, fa = DA.select(rows, bench)
    b, fb = DA.select(list(reversed(rows)), bench)
    assert a == b and fa == fb and len(a) == DA.N and len({(r["task"], r["row"]) for r in a}) == DA.N


def test_insufficient_rows_stop_instead_of_resampling():
    rows = [{"task": "piqa", "row": i, "text_hash": f"h{i}"} for i in range(DA.N - 1)]
    with pytest.raises(DA.Insufficient):
        DA.select(rows, "piqa")


def test_pass_rule_uses_the_unchanged_verifier_tolerance_on_both_packed_comparisons():
    vspec = importlib.util.spec_from_file_location("v", REPO_ROOT / "scripts" / "verify_extraction.py")
    V = importlib.util.module_from_spec(vspec)
    vspec.loader.exec_module(V)
    assert DA.FP32_TOL == V.FP32_TOL == 1e-4
    ok = {"packed_vs_ctx": {"maxabs_max": 2e-5}, "packed_vs_ctxmask": {"maxabs_max": 1e-4}}
    bad = {"packed_vs_ctx": {"maxabs_max": 2e-5}, "packed_vs_ctxmask": {"maxabs_max": 1.1e-4}}
    assert DA.verdict(ok) == (1e-4, True) and DA.verdict(bad)[1] is False


def test_frozen_inputs_are_hash_checked(tmp_path, allocated):
    for f in ("inventory.json", "inventory_clean_ids.csv"):
        (tmp_path / "results/phase3").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results/phase3" / f).write_bytes((REPO_ROOT / "results/phase3" / f).read_bytes())
    csv = tmp_path / "results/phase3/inventory_clean_ids.csv"
    csv.write_text(csv.read_text(encoding="utf-8") + "piqa,train,piqa,999999,x,y\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        DA.eligible_rows(tmp_path, "piqa", allocated)


def test_report_path_is_development_side_and_never_sealed():
    P.assert_not_sealed(DA.REPORT)
    src = (REPO_ROOT / "scripts" / "diagnose_short_context_ood.py").read_text(encoding="utf-8")
    for word in ("correct", "pred", "accuracy", "score_per_option", "opt_ll"):
        assert f'"{word}"' not in src and f"'{word}'" not in src
