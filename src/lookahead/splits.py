"""Question-disjoint 60/20/20 splits, exactly stratified, independent of load order (plan section 3)."""
import hashlib

SALT = "lh-split-v1"
FRACTIONS = (("train", 0.6), ("validation", 0.2))  # the rest is test


def split_key(benchmark: str, doc_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{benchmark}|{doc_id}".encode("utf-8")).hexdigest()


def assign_splits(rows):
    """rows: iterable of (benchmark, stratum, doc_id). Returns {(benchmark, doc_id): split}.

    Within each stratum, questions are sorted by split_key; the first round(0.6 n) go to train, the next round(0.2 n)
    to validation, the rest to test.
    """
    strata = {}
    for bench, stratum, doc_id in rows:
        strata.setdefault((bench, stratum), []).append(doc_id)
    out = {}
    for (bench, _), ids in strata.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate doc_id in {bench}")
        ids = sorted(ids, key=lambda d: split_key(bench, d))
        n_train = round(FRACTIONS[0][1] * len(ids))
        n_val = round(FRACTIONS[1][1] * len(ids))
        for i, d in enumerate(ids):
            out[(bench, d)] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
    return out
