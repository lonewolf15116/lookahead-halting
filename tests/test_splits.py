from collections import Counter

from lookahead.splits import assign_splits


def rows():
    return [("b", f"s{k % 3}", f"d{k}") for k in range(1000)] + [("c", "c", f"d{k}") for k in range(7)]


def test_proportions_exact_per_stratum():
    out = assign_splits(rows())
    for s in ("s0", "s1", "s2"):
        n = Counter(out[("b", f"d{k}")] for k in range(1000) if f"s{k % 3}" == s)
        total = sum(n.values())
        assert n["train"] == round(0.6 * total) and n["validation"] == round(0.2 * total)


def test_order_independent_and_disjoint():
    a = assign_splits(rows())
    b = assign_splits(list(reversed(rows())))
    assert a == b
    assert len(a) == 1007  # every question in exactly one split
