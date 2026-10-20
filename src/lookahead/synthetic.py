"""Deterministic synthetic benchmarks (modular arithmetic, parity), 5-shot.

Dedupe rule (all levels): eval items are unique within a level, no exemplar
equals any eval expression/bit string of its level, and the 5 exemplars of a
prompt are distinct. Parity levels start at L=10 so 2**L >= 500 holds.
"""
import random
from dataclasses import asdict, dataclass

N_ITEMS = 500
N_SHOTS = 5
POOL_SIZE = 100
MOD_LEVELS = (2, 3, 4, 5)
PARITY_LEVELS = (10, 12, 16, 20)
SEEDS = {"modarith": (20261020, 20261021), "parity": (20261022, 20261023)}
OPS = ("+", "-", "*")


@dataclass
class Item:
    benchmark: str
    doc_id: str
    level: int
    context: str
    choices: list
    target: int
    expr: str

    def to_dict(self):
        return asdict(self)


# ---- modular arithmetic ----------------------------------------------------
def _mod_expr(rng, n):
    nums = [rng.randint(0, 99) for _ in range(n)]
    ops = [rng.choice(OPS) for _ in range(n - 1)]
    expr = str(nums[0])
    for op, x in zip(ops, nums[1:]):
        expr = f"({expr} {op} {x})"
    return expr


def _mod_answer(expr):
    return eval(expr, {"__builtins__": {}}, {}) % 7  # only digits, + - *, parens


def _gen_modarith():
    ev, ex = (random.Random(s) for s in SEEDS["modarith"])
    evals = {}
    for n in MOD_LEVELS:
        seen = []
        s = set()
        while len(seen) < N_ITEMS:
            e = _mod_expr(ev, n)
            if e not in s:
                s.add(e)
                seen.append(e)
        evals[n] = seen
    all_eval = {e for v in evals.values() for e in v}
    items = []
    for n in MOD_LEVELS:
        pool = []
        while len(pool) < POOL_SIZE:
            e = _mod_expr(ex, n)
            if e not in all_eval and e not in pool:
                pool.append(e)
        for i, e in enumerate(evals[n]):
            shots = ex.sample(pool, N_SHOTS)
            lines = [f"{s} mod 7 = {_mod_answer(s)}" for s in shots]
            ctx = "\n".join(lines) + "\n" + f"{e} mod 7 = "
            items.append(Item("modarith", f"modarith-n{n}-{i:05d}", n, ctx,
                              [str(k) for k in range(7)], _mod_answer(e), e))
    return items


# ---- parity ----------------------------------------------------------------
def _bits(rng, L):
    return "".join(rng.choice("01") for _ in range(L))


def _par(bits):
    return "even" if bits.count("1") % 2 == 0 else "odd"


def _gen_parity():
    ev, ex = (random.Random(s) for s in SEEDS["parity"])
    items = []
    for L in PARITY_LEVELS:
        half = N_ITEMS // 2
        need = {"even": half, "odd": half}
        chosen, seen = [], set()
        while len(chosen) < N_ITEMS:
            b = _bits(ev, L)
            p = _par(b)
            if need[p] == 0 or b in seen:
                continue
            need[p] -= 1
            seen.add(b)
            chosen.append(b)
        ev.shuffle(chosen)
        pool = []
        while len(pool) < POOL_SIZE:
            b = _bits(ex, L)
            if b not in pool and b not in seen:
                pool.append(b)
        for i, b in enumerate(chosen):
            shots = ex.sample(pool, N_SHOTS)
            blocks = [f"Bits: {s}\nNumber of 1s is {_par(s)}" for s in shots]
            ctx = "\n\n".join(blocks) + "\n\n" + f"Bits: {b}\nNumber of 1s is"
            items.append(Item("parity", f"parity-L{L}-{i:05d}", L, ctx,
                              [" even", " odd"], 0 if _par(b) == "even" else 1, b))
    return items


_GEN = {"modarith": _gen_modarith, "parity": _gen_parity}
DEDUPE = {"modarith": {n: True for n in MOD_LEVELS},
          "parity": {L: True for L in PARITY_LEVELS}}


def generate(task: str):
    return _GEN[task]()


def generate_all():
    return {t: generate(t) for t in _GEN}


def check_token_boundaries(items, tokenizer):
    """Mimic lm-eval `_encode_pair`; return failure messages (empty = ok)."""
    fails = []
    for kw in ({}, {"add_special_tokens": False}):
        for it in items:
            c = tokenizer.encode(it.context, **kw)
            for ch in it.choices:
                w = tokenizer.encode(it.context + ch, **kw)
                if w[:len(c)] != c or len(w) - len(c) != 1:
                    fails.append(f"{it.doc_id} choice={ch!r} kw={kw}: "
                                 f"prefix_ok={w[:len(c)] == c} cont_len={len(w) - len(c)}")
    return fails
