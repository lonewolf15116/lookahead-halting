import re
from collections import Counter
from pathlib import Path

import pytest

from lookahead import synthetic as S

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return S.generate_all()


def test_determinism(data):
    again = S.generate_all()
    for t in data:
        assert [i.to_dict() for i in data[t]] == [i.to_dict() for i in again[t]]


def test_counts(data):
    for t, levels in (("modarith", S.MOD_LEVELS), ("parity", S.PARITY_LEVELS)):
        c = Counter(i.level for i in data[t])
        assert dict(c) == {l: 500 for l in levels}
        assert len({i.doc_id for i in data[t]}) == 2000


def test_modarith_grammar_and_answers(data):
    for it in data["modarith"]:
        assert re.fullmatch(r"[0-9+\-*() ]+", it.expr)
        assert it.expr.count("(") == it.level - 1
        assert len(re.findall(r"\d+", it.expr)) == it.level
        assert it.choices == [str(k) for k in range(7)]
        assert 0 <= it.target <= 6
        assert it.target == eval(it.expr) % 7
        assert it.context.endswith(f"{it.expr} mod 7 = ")
        assert not it.context.endswith("  ")


def test_modarith_exemplars_and_uniqueness(data):
    items = data["modarith"]
    assert len({i.expr for i in items}) == len(items)
    evals = {i.expr for i in items}
    for it in items:
        lines = it.context.split("\n")
        assert len(lines) == 6
        shots = [l[:-len(" mod 7 = X")] for l in lines[:5]]
        assert len(set(shots)) == 5
        for l, s in zip(lines[:5], shots):
            assert s not in evals
            assert int(l[-1]) == eval(s) % 7
            assert s.count("(") == it.level - 1


def test_parity(data):
    items = data["parity"]
    for L in S.PARITY_LEVELS:
        lv = [i for i in items if i.level == L]
        assert Counter(i.target for i in lv) == {0: 250, 1: 250}
        assert len({i.expr for i in lv}) == 500
    for it in items:
        assert len(it.expr) == it.level
        assert it.choices == [" even", " odd"]
        assert it.choices[it.target].strip() == ("even" if it.expr.count("1") % 2 == 0 else "odd")
        assert it.context.endswith("is")
        blocks = it.context.split("\n\n")
        assert len(blocks) == 6 and blocks[-1] == f"Bits: {it.expr}\nNumber of 1s is"
        for b in blocks[:5]:
            m = re.fullmatch(r"Bits: ([01]+)\nNumber of 1s is (even|odd)", b)
            assert m and len(m[1]) == it.level
            assert m[2] == ("even" if m[1].count("1") % 2 == 0 else "odd")
            assert m[1] != it.expr


def test_parity_exemplars_disjoint_when_dedupe(data):
    for L in S.PARITY_LEVELS:
        lv = [i for i in data["parity"] if i.level == L]
        evals = {i.expr for i in lv}
        for it in lv:
            shots = [b.split("\n")[0][6:] for b in it.context.split("\n\n")[:5]]
            assert len(set(shots)) == 5
            assert not set(shots) & evals


def test_dedupe_flags_all_true():
    assert all(v for d in S.DEDUPE.values() for v in d.values())
    assert S.PARITY_LEVELS == (10, 12, 16, 20)


def test_token_boundaries(data):
    pytest.importorskip("transformers")
    path = ROOT / "hf_cache" / "ouro-1.4b"
    if not path.exists():
        pytest.skip("tokenizer dir missing")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    for t, items in data.items():
        assert S.check_token_boundaries(items, tok) == []
