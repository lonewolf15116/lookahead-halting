"""D-t: truncated-execution check of the retrospective C1 evaluation (docs/DT_TRUNCATION_PLAN.md, rev 1).

Diagnostic on DEVELOPMENT data only (DEV+ of the three primary benchmarks); TEST is never touched. Every statistic and decision
is computed by the SAME functions the C1 evaluation uses; nothing is reimplemented here:

  pi1-x       c1_test.pi1x_test_stat(d, models)   (e1.predict_e1 with the frozen models of results/c1/pi1x_models.npz)
  native-pb   phase3_g3a.baseline_stat(d, "native-pb")   (Ouro's cumulative exit probability, the stored `cdf`)
  r-kl        phase3_g3a.baseline_stat(d, "r-kl")        (phase3_readouts.question_stats on the option scores)
  decision    phase3_readouts.exit_loops(S, ok, tau, m) with the frozen (tau, m) of results/c1/frozen.json (c1_test._sel)
  retrospective reference   c1_test.apply_policy on the stored DEV+ arrays

The only new code is the ADAPTER that turns freshly computed loops into a single-question `DevData` holding loops 1..t and NaN
for later loops (`Trace.devdata`), plus the bookkeeping of the check. The per-loop quantities (option record, vocabulary entropy,
native cumulative exit probability, hidden state) are produced by the same functions as `extract.extract_question`
(`extract.option_record`, `ouro_adapter.NativeGate`, the same `vent` expression).

Things that cannot be bit-identical to C1 by construction are reported, not hidden: the policy statistic is evaluated one
question at a time here (batch of 1) and over the whole selected set in C1 (`apply_policy`); the P-2 reference therefore is the
stored full row evaluated in the same batch shape, and the C1 batch statistic is compared as well (see `run_check`).
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lookahead import c1_test as T
from lookahead import extract as X
from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_build as P
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R
from lookahead.ouro_adapter import NativeGate
from lookahead.packed import build_packed

PLAN = "docs/DT_TRUNCATION_PLAN.md rev 1"
STATUS = "diagnostic [E]; changes no frozen artifact and no C1 result"
SALT = "lh-dt-v1:"
N_PER_BENCH = 200
LOOPS = 4
POLICIES = ("pi1-x", "native-pb", "r-kl")
PRIMARY = G.PRIMARY
REPORT = "results/diagnostics/dt_truncation_report.json"
PINNED = {
    "results/c1/frozen.json": "65728e16113e94114554fb28a951eee2b93af4b3fb5964bf32cf12845560b25f",
    "results/c1/pi1x_models.npz": "72f1550813e12f242128345b4b20097eca0c83ebe843b358f5649a8ed8938bad",
}
SEALED_DIR = "data/extracted_phase3_test"
# where each policy's frozen per-benchmark (tau, m) lives in results/c1/frozen.json
FROZEN_SELECTION = {"pi1-x": ("primary", "pi1-x"), "native-pb": ("primary", "native-pb"),
                    "r-kl": ("secondary_risk_all_baselines", "r-kl")}


# ------------------------------------------------------------------------------------------------------- guards
def refuse_sealed(root, *paths) -> None:
    """Refuse any path under <root>/data/extracted_phase3_test (after resolving '..' and symlinks), and any path the
    development loaders already treat as sealed (phase3_build.assert_not_sealed: 'phase3_test', 'confirm')."""
    base = Path(root) / SEALED_DIR
    for p in paths:
        s = str(p).replace("\\", "/").lower()
        if "extracted_phase3_test" in s or T._within(p, base):
            raise PermissionError(f"D-t never touches TEST: refusing {p}")
        P.assert_not_sealed(p)


def check_pinned(root, pinned: dict = PINNED) -> None:
    """sha256 of the frozen artifacts (raw bytes or LF-normalised, as c1_test_run does). SystemExit on any mismatch."""
    for rel, exp in pinned.items():
        p = Path(root) / rel
        if not p.exists():
            raise SystemExit(f"missing frozen artifact {rel}")
        raw = p.read_bytes()
        if exp not in (g1.sha256_bytes(raw), g1.sha256_bytes(raw.replace(b"\r\n", b"\n"))):
            raise SystemExit(f"{rel} is not the frozen file")


def frozen_selections(frozen: dict) -> dict:
    """{policy: {benchmark: {'tau', 'm'}}} for the three policies, through c1_test._sel (tau may be the string 'inf')."""
    out = {}
    for pol, (sect, key) in FROZEN_SELECTION.items():
        sel = T._sel(frozen[sect][key])
        missing = [b for b in PRIMARY if b not in sel]
        if missing:
            raise ValueError(f"{pol}: no frozen (tau, m) for {missing}")
        out[pol] = {b: sel[b] for b in PRIMARY}
    return out


# ------------------------------------------------------------------------------------------------------ selection
def question_key(doc_id: str) -> str:
    return hashlib.sha256((SALT + doc_id).encode("utf-8")).hexdigest()


def select_ids(dev: G.DevData, n: int = N_PER_BENCH, benches=PRIMARY) -> dict:
    """{benchmark: [doc_id]}: the n lowest sha256("lh-dt-v1:" + doc_id) among stored-status-ok DEV+ questions."""
    if (dev.source == "test").any() or (dev.source == "p1_validation").any():
        raise PermissionError("D-t uses development data only")
    out = {}
    for b in benches:
        m = (dev.bench == b) & (dev.source == "devplus") & dev.ok
        ids = dev.doc_id[m].tolist()
        if len(ids) < n:
            raise ValueError(f"{b}: only {len(ids)} ok DEV+ questions, need {n}")
        out[b] = sorted(ids, key=lambda i: (question_key(i), i))[:n]
    return out


def match_questions(sub: G.DevData, ids: dict, questions: dict) -> dict:
    """{doc_id: Question}. Every selected id must match exactly one DEV+ question of its benchmark, and that question must be
    the one the stored trace was extracted from (n_options and log n_tokens of the stored cheap features)."""
    by = {}
    for b, qs in questions.items():
        for q in qs:
            by.setdefault((b, q.doc_id), []).append(q)
    out = {}
    for b, lst in ids.items():
        for i in lst:
            c = by.get((b, i), [])
            if len(c) != 1:
                raise AssertionError(f"{b}: {len(c)} DEV+ questions with doc_id {i} (expected exactly 1)")
            out[i] = c[0]
    pos = {d: k for k, d in enumerate(sub.doc_id.tolist())}
    for i, q in out.items():
        k = pos[i]
        if not q.context_enc or not q.cont_encs:
            raise AssertionError(f"{i}: question has no token encodings")
        if sub.x[k, 0, 4] != len(q.cont_encs) or sub.x[k, 0, 5] != np.float32(np.log(max(float(q.packed_len), 1.0))):
            raise AssertionError(f"{i}: the DEV+ question does not match the stored trace (n_options / n_tokens differ)")
    return out


# ------------------------------------------------------------------------------------------------ per-loop quantities
def loop_record(q, out, gate: NativeGate) -> dict:
    """Per-loop quantities from one LoopOutput, computed exactly as extract.extract_question does."""
    o = X.option_record(q, out.option_ll)
    vent = float(-(out.vocab_logp.exp() * out.vocab_logp).sum())
    hidden = out.hidden.float().cpu().numpy().astype(np.float16)
    gate_logit = out.gate_logit
    cdf = float(gate.update(torch.tensor([gate_logit]))[0])
    nums = [o["conf"], o["ent"], vent, gate_logit, cdf, o["loss"], *o["opt_ll"]]
    return {"loop": out.loop, "opt_ll": o["opt_ll"], "opt_lln": o["opt_lln"], "z": o["opt_lln"] if q.metric == "acc_norm" else o["opt_ll"],
            "pred": o["pred"], "correct": o["correct"], "conf": o["conf"], "ent": o["ent"], "vent": vent, "gate": gate_logit,
            "cdf": cdf, "hidden": hidden, "finite": bool(all(math.isfinite(x) for x in nums) and np.isfinite(hidden).all())}


class Trace:
    """The loops of one question seen so far, and the adapter to a single-question DevData (loops 1..t real, later NaN)."""

    def __init__(self, q, doc_id: str, bench: str):
        self.q, self.doc_id, self.bench = q, doc_id, bench
        self.rows, self.failed = [], False

    @property
    def t(self) -> int:
        return len(self.rows)

    def add(self, rec: dict) -> bool:
        """Append a loop; False once any loop was non-finite (the question then counts as failed, as in extract)."""
        self.rows.append(rec)
        if not rec["finite"]:
            self.failed = True
        return not self.failed

    def devdata(self, t: int | None = None) -> G.DevData:
        t = self.t if t is None else t
        if not 1 <= t <= self.t:
            raise ValueError(f"loop {t} not available (have {self.t})")
        rows = self.rows[:t]
        nan4 = lambda: np.full((1, LOOPS), np.nan)   # noqa: E731
        conf, ent, cdf, vent = nan4(), nan4(), nan4(), np.full((1, 3), np.nan)
        for k, r in enumerate(rows):
            conf[0, k], ent[0, k], cdf[0, k] = r["conf"], r["ent"], r["cdf"]
            if k < 3:
                vent[0, k] = r["vent"]
        a = {"n": 1, "ok": np.array([True]), "conf": conf, "ent": ent, "cdf": cdf}
        # the same feature builder as the stored data (gate2.cheap_features); no nan_to_num on vent, so a future loop stays NaN
        x = g2.cheap_features(a, vent, np.array([float(len(self.q.cont_encs))]), np.array([float(self.q.packed_len)]))
        x[:, t:, :] = np.nan                                       # whole feature rows of later loops are NaN
        d = len(rows[0]["hidden"])
        h = np.full((1, 3, d), np.nan, dtype=np.float32)
        for k, r in enumerate(rows[:3]):
            h[0, k] = r["hidden"].astype(np.float32)
        K = len(rows[0]["z"])
        z = np.full((LOOPS, K), np.nan)
        pred, correct = np.full((1, LOOPS), -1, dtype=np.int64), np.zeros((1, LOOPS), dtype=np.int64)
        for k, r in enumerate(rows):
            z[k] = r["z"]
            pred[0, k], correct[0, k] = r["pred"], r["correct"]
        # T / Tfin / pred / correct beyond what is observed are placeholders: no policy statistic reads them
        lab = np.full((1, 3), "WW")
        return G.DevData(bench=np.array([self.bench]), doc_id=np.array([self.doc_id]), source=np.array(["devplus"]),
                         ok=np.array([True]), correct=correct, pred=pred, cdf=cdf, T=lab, Tfin=lab.copy(), x=x, h=h, z=[z])


def stored_prefix(d: G.DevData, i: int, t: int) -> G.DevData:
    """Row i of stored development data with everything after loop t set to NaN (model-free emulation of the online view)."""
    r = G.subset(d, np.arange(d.N) == i)
    r.x, r.h, r.cdf, z = r.x.copy(), r.h.copy(), r.cdf.astype(float).copy(), r.z[0].copy()
    r.x[:, t:, :] = np.nan
    r.h[:, t:, :] = np.nan
    r.cdf[:, t:] = np.nan
    z[t:] = np.nan
    r.z = [z]
    return r


# ------------------------------------------------------------------------------------------------ statistic + decision
def policy_stat(policy: str, d: G.DevData, models: list) -> np.ndarray:
    """[N, 3] statistic, by the code C1 used (c1_test.evaluate_c1)."""
    if policy == "pi1-x":
        return T.pi1x_test_stat(d, models)
    if policy in ("native-pb", "r-kl"):
        return G.baseline_stat(d, policy)
    raise ValueError(policy)


def decide(policy: str, d: G.DevData, models: list, sel_b: dict):
    """(statistic row [3], exit loop) for a one-question DevData, through phase3_readouts.exit_loops."""
    S = policy_stat(policy, d, models)
    return S[0].copy(), int(R.exit_loops(S, d.ok, sel_b["tau"], sel_b["m"])[0])


def exit_by_prefixes(policy: str, d: G.DevData, i: int, models: list, sel_b: dict) -> int:
    """Model-free adapter check: the exit loop obtained by stopping at the first loop t whose NaN-future view says exit <= t."""
    for t in range(1, LOOPS + 1):
        _, e = decide(policy, stored_prefix(d, i, t), models, sel_b)
        if e <= t:
            return e
    raise AssertionError("no exit by loop 4 (cannot happen)")


def bits_equal(a, b, policy: str) -> bool:
    """P-2 comparison of two statistic prefixes: bit-for-bit equal, where the only admissible NaN is a statistic undefined by
    definition (r-kl at loop 1, in BOTH). Any other NaN, or any bit difference, is a failure."""
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    undefined = np.array([policy == "r-kl" and c == 0 for c in range(len(a))])
    if (np.isnan(a) != undefined).any() or (np.isnan(b) != undefined).any():
        return False
    return bool((a.view(np.int64)[~undefined] == b.view(np.int64)[~undefined]).all())


def _maxabs(a, b) -> float:
    """max |a - b|; 0 for equal entries (NaN with NaN, inf with the same inf); inf if the NaN patterns or shapes differ."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or (np.isnan(a) != np.isnan(b)).any():
        return math.inf
    keep = ~np.isnan(a)
    a, b = a[keep], b[keep]
    if not len(a):
        return 0.0
    fin = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(a[~fin], b[~fin]):
        return math.inf
    return float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0


# ------------------------------------------------------------------------------------------------------------ runs
def run_truncated(scorer, q, bench: str, doc_id: str, policy: str, sel_b: dict, models: list, sync, clock) -> dict:
    """Iterate PackedScorer.loops one loop at a time; after loop t decide from loops 1..t only; on exit close the generator
    (no further loop is computed). Timed from the start of packing to the closed generator, CUDA-synchronised."""
    sync()
    t0 = clock()
    packed = build_packed(q.context_enc, q.cont_encs)
    gate = NativeGate(scorer.runner.total_loops, 1)
    trace = Trace(q, doc_id, bench)
    gen = scorer.loops(packed)
    stats, e = {}, None
    try:
        for out in gen:
            if not trace.add(loop_record(q, out, gate)):
                continue                      # non-finite loop: failed question, runs all loops and counts wrong (extract rule)
            S, e_t = decide(policy, trace.devdata(), models, sel_b)
            stats[trace.t] = S
            if e_t <= trace.t:
                e = e_t
                break
    finally:
        gen.close()
    sync()
    ms = 1000.0 * (clock() - t0)
    if trace.failed:
        e = LOOPS
    if e is None:
        raise AssertionError("the policy did not exit within the available loops")
    last = trace.rows[e - 1]
    return {"e": e, "pred": last["pred"], "correct": 0 if trace.failed else last["correct"], "loops_run": trace.t, "ms": ms,
            "stats": stats, "nonfinite": trace.failed}


def run_full(scorer, q, bench: str, doc_id: str, sync, clock):
    """All loops of the question with the same per-loop bookkeeping as the truncated run but no policy statistic."""
    sync()
    t0 = clock()
    packed = build_packed(q.context_enc, q.cont_encs)
    gate = NativeGate(scorer.runner.total_loops, 1)
    trace = Trace(q, doc_id, bench)
    for out in scorer.loops(packed):
        trace.add(loop_record(q, out, gate))
    sync()
    return trace, 1000.0 * (clock() - t0)


# ----------------------------------------------------------------------------------------------- stored raw traces
def load_stored_trace(root, bench: str, ids, dirname: str = G.DEVPLUS_DIR, split: str = "dev") -> dict:
    """Raw stored per-loop values (gate logit, option log-likelihoods, conf, ent, vent, cdf) of the given DEV+ questions, for
    P-3. Reads only the completed, checksum-verified parts of the development extraction (guarded like G.load_source)."""
    import pyarrow.parquet as pq

    base = Path(root) / "data" / dirname / bench
    refuse_sealed(root, base)
    G._guard(base)
    want, out = set(ids), {}
    for dp in sorted(base.glob("part-*.done.json")):
        G._guard(dp)
        done = json.loads(dp.read_text(encoding="utf-8"))
        if not want & set(done["doc_ids"]):
            continue
        pq_path = Path(str(dp)[: -len(".done.json")] + ".parquet")
        G._guard(pq_path)
        if g1.sha256_file(pq_path) != done["parquet_sha256"]:
            raise ValueError(f"checksum mismatch for {pq_path}")
        df = pq.read_table(pq_path, columns=["doc_id", "loop", "split", "status", "gate", "opt_ll", "conf", "ent", "vent", "cdf"],
                           filters=[("split", "==", split)]).to_pandas()
        df = df[df["split"] == split]
        for doc, g in df[df["doc_id"].isin(want)].groupby("doc_id"):
            g = g.sort_values("loop", kind="stable")
            if g["loop"].tolist() != list(range(1, LOOPS + 1)):
                raise ValueError(f"{doc}: stored loops are not 1..{LOOPS}")
            out[doc] = {"status": g["status"].iloc[0], "gate": g["gate"].to_numpy(float), "conf": g["conf"].to_numpy(float),
                        "ent": g["ent"].to_numpy(float), "vent": g["vent"].to_numpy(float), "cdf": g["cdf"].to_numpy(float),
                        "opt_ll": [np.asarray(v, dtype=float) for v in g["opt_ll"]]}
    missing = want - set(out)
    if missing:
        raise ValueError(f"{bench}: {len(missing)} selected questions have no stored trace, e.g. {sorted(missing)[:3]}")
    return out


P3_KEYS = ("gate_logit", "opt_ll", "conf", "ent", "vent", "cdf")


def fresh_vs_stored(trace: Trace, raw: dict) -> dict:
    """P-3: max |fresh - stored| over the 4 loops (and options) for one question."""
    f = trace.rows
    d = {"gate_logit": _maxabs([r["gate"] for r in f], raw["gate"])}
    for k in ("conf", "ent", "vent", "cdf"):
        d[k] = _maxabs([r[k] for r in f], raw[k])
    d["opt_ll"] = max(_maxabs(r["opt_ll"], raw["opt_ll"][k]) for k, r in enumerate(f))
    return d


# ------------------------------------------------------------------------------------------------------- the check
def _pct(a, b) -> float:
    return float(100.0 * np.mean(np.asarray(a) == np.asarray(b))) if len(a) else float("nan")


def run_check(root, dev: G.DevData, questions: dict, models: list, frozen: dict, scorer, stored_loader=load_stored_trace,
              n_per_bench: int = N_PER_BENCH, log=None, clock=time.perf_counter) -> dict:
    """The whole check on in-memory inputs. Returns the report dict (nothing is written here)."""
    log = log or (lambda *_: None)
    if scorer.runner.total_loops != LOOPS:
        raise ValueError(f"the model has {scorer.runner.total_loops} loops, the check needs {LOOPS}")
    sel = frozen_selections(frozen)
    ids = select_ids(dev, n_per_bench)
    chosen = set(i for lst in ids.values() for i in lst)
    sub = G.subset(dev, np.isin(dev.doc_id, list(chosen)) & (dev.source == "devplus"))
    if sub.N != len(chosen):
        raise AssertionError(f"selected {len(chosen)} ids but matched {sub.N} stored DEV+ rows")
    qmap = match_questions(sub, ids, questions)
    pos = {d: k for k, d in enumerate(sub.doc_id.tolist())}

    # stored (retrospective) reference: the C1 code path on the stored arrays, for exactly these questions
    S_batch = {p: policy_stat(p, sub, models) for p in POLICIES}
    retro = {p: T.apply_policy(sub, S_batch[p], sel[p], PRIMARY) for p in POLICIES}
    retro_at = {p: {b: {i: (int(r["e"][k]), int(r["c"][k]), int(r["c4"][k])) for k, i in enumerate(r["ids"].tolist())}
                    for b, r in retro[p].items()} for p in POLICIES}
    S_single = {p: np.stack([policy_stat(p, G.subset(sub, np.arange(sub.N) == k), models)[0] for k in range(sub.N)])
                for p in POLICIES}                    # stored full rows, evaluated one question at a time (batch shape of the online run)
    raw = {b: stored_loader(root, b, ids[b]) for b in PRIMARY}
    for b in PRIMARY:
        bad = [i for i, r in raw[b].items() if r["status"] != "ok"]
        if bad:
            raise AssertionError(f"{b}: stored status of selected questions is not ok: {bad[:3]}")

    sync = torch.cuda.synchronize if scorer.device.type == "cuda" else (lambda: None)
    order = [(b, i) for b in PRIMARY for i in ids[b]]
    b0, i0 = order[0]
    for _ in range(2):                                # untimed warm-up (CUDA kernels, allocator, caches)
        run_full(scorer, qmap[i0], b0, i0, sync, clock)
    for p in POLICIES:
        run_truncated(scorer, qmap[i0], b0, i0, p, sel[p][b0], models, sync, clock)

    per_q, p3 = [], {b: [] for b in PRIMARY}
    for n, (b, i) in enumerate(order):
        q = qmap[i]
        trunc, full = {}, None
        for step in ("full", *POLICIES) if n % 2 == 0 else (*POLICIES, "full"):   # alternate order: no systematic warm-cache bias
            if step == "full":
                full = run_full(scorer, q, b, i, sync, clock)
            else:
                trunc[step] = run_truncated(scorer, q, b, i, step, sel[step][b], models, sync, clock)
        ftrace, full_ms = full
        k = pos[i]
        p3[b].append(fresh_vs_stored(ftrace, raw[b][i]))
        rec = {"id": i, "bench": b, "full_ms": full_ms, "full_nonfinite": ftrace.failed, "policy": {}}
        fd = ftrace.devdata(LOOPS) if not ftrace.failed else None
        for p in POLICIES:
            tr = trunc[p]
            e_ret, c_ret, c4 = retro_at[p][b][i]
            S_fresh = policy_stat(p, fd, models)[0] if fd is not None else np.full(3, np.nan)
            chk = {"checks": 0, "fail_vs_stored": 0, "fail_vs_stored_batch": 0, "fail_vs_fresh_full": 0, "fail_stored_masked": 0}
            for t, S_t in tr["stats"].items():
                kk = min(t, 3)
                chk["checks"] += 1
                chk["fail_vs_stored"] += not bits_equal(S_t[:kk], S_single[p][k, :kk], p)
                chk["fail_vs_stored_batch"] += not bits_equal(S_t[:kk], S_batch[p][k, :kk], p)
                chk["fail_vs_fresh_full"] += not bits_equal(S_t[:kk], S_fresh[:kk], p)
                masked = policy_stat(p, stored_prefix(sub, k, t), models)[0]
                chk["fail_stored_masked"] += not bits_equal(masked[:kk], S_single[p][k, :kk], p)
            rec["policy"][p] = {
                "e": tr["e"], "correct": tr["correct"], "pred": tr["pred"], "loops_run": tr["loops_run"], "ms": tr["ms"],
                "nonfinite": tr["nonfinite"], "e_ret": e_ret, "c_ret": c_ret, "c4_ret": c4, "p2": chk,
                "stat_fresh_full_vs_stored_single": _maxabs(S_fresh, S_single[p][k]),
                "stat_batch_vs_single": _maxabs(S_batch[p][k], S_single[p][k]),
                "online": {str(t): S.tolist() for t, S in tr["stats"].items()},
                "stored": S_single[p][k].tolist(), "fresh_full": S_fresh.tolist()}
        per_q.append(rec)
        if (n + 1) % 25 == 0:
            log(f"  {n + 1}/{len(order)} questions")
    return {"plan": PLAN, "status": STATUS, "n_per_bench": n_per_bench, "salt": SALT, "selected_ids": ids,
            "selected_ids_sha256": hashlib.sha256("\n".join(i for b in PRIMARY for i in ids[b]).encode()).hexdigest(),
            "frozen_selections": {p: {b: {"tau": str(s["tau"]) if math.isinf(s["tau"]) else s["tau"], "m": s["m"]}
                                      for b, s in sel[p].items()} for p in POLICIES},
            "policies": aggregate(per_q, p3, n_per_bench), "p3": aggregate_p3(p3),
            "timing_caveats": ["batch size 1", "Python loop overhead included", "policy statistic computed on the CPU after each loop",
                               "per-loop bookkeeping (option record, vocabulary entropy, gate, hidden to CPU) is in BOTH runs",
                               "order of full/truncated runs alternates by question; 2 untimed warm-up full runs first"]}


def aggregate_p3(p3: dict) -> dict:
    out = {}
    for b, lst in p3.items():
        out[b] = {k: float(max(x[k] for x in lst)) for k in P3_KEYS}
    out["all"] = {k: float(max(out[b][k] for b in p3)) for k in P3_KEYS}
    return out


def aggregate(per_q: list, p3: dict, n_per_bench: int) -> dict:
    out = {}
    for p in POLICIES:
        by = {}
        for b in (*PRIMARY, "all"):
            rs = [r for r in per_q if b == "all" or r["bench"] == b]
            pr = [r["policy"][p] for r in rs]
            e, er = np.array([x["e"] for x in pr]), np.array([x["e_ret"] for x in pr])
            c, cr = np.array([x["correct"] for x in pr]), np.array([x["c_ret"] for x in pr])
            c4 = np.array([x["c4_ret"] for x in pr])
            tr_ms, full_ms = sum(x["ms"] for x in pr), sum(r["full_ms"] for r in rs)
            p2 = {k: int(sum(x["p2"][k] for x in pr)) for k in ("checks", "fail_vs_stored", "fail_vs_stored_batch",
                                                                "fail_vs_fresh_full", "fail_stored_masked")}
            p2["pass_vs_stored_arrays"] = p2["fail_vs_stored"] == 0
            p2["pass_code_level_causality"] = p2["fail_vs_fresh_full"] == 0 and p2["fail_stored_masked"] == 0
            dis = [{"id": r["id"], "bench": r["bench"], "e_truncated": x["e"], "e_retrospective": x["e_ret"],
                    "correct_truncated": x["correct"], "correct_retrospective": x["c_ret"], "pred_truncated": x["pred"],
                    "statistic_online_by_loop": x["online"], "statistic_stored_full": x["stored"],
                    "statistic_fresh_full": x["fresh_full"]}
                   for r, x in zip(rs, pr) if x["e"] != x["e_ret"] or x["correct"] != x["c_ret"]]
            by[b] = {
                "n": len(rs), "exit_agreement_pct": _pct(e, er), "correctness_agreement_pct": _pct(c, cr),
                "n_exit_disagree": int((e != er).sum()), "n_correct_disagree": int((c != cr).sum()), "disagreements": dis,
                "P1_pass": bool((e == er).all() and (c == cr).all()), "P2": p2,
                "loops_run_equals_exit": int(sum(x["loops_run"] == x["e"] for x in pr)),
                "n_nonfinite": int(sum(x["nonfinite"] for x in pr)),
                "accuracy_truncated": float(c.mean()), "accuracy_retrospective": float(cr.mean()), "accuracy_loop4_stored": float(c4.mean()),
                "simulated_gross_saving": float(1 - e.mean() / LOOPS), "simulated_gross_saving_retrospective": float(1 - er.mean() / LOOPS),
                "exit_hist_truncated": {str(k): int((e == k).sum()) for k in range(1, 5)},
                "total_truncated_ms": float(tr_ms), "total_full_ms": float(full_ms),
                "measured_time_saving": float(1 - tr_ms / full_ms) if full_ms > 0 else float("nan"),
                "stat_max_abs_diff_fresh_full_vs_stored": float(max(x["stat_fresh_full_vs_stored_single"] for x in pr)),
                "stat_max_abs_diff_batch_vs_single": float(max(x["stat_batch_vs_single"] for x in pr))}
        out[p] = by
    return out


def verdict(report: dict) -> dict:
    """Pre-declared outcomes: P-1 (exit loop and correctness, 100%) and P-2 (causality), overall and per policy."""
    pol = report["policies"]
    return {"P1_pass": all(pol[p]["all"]["P1_pass"] for p in pol),
            "P2_pass": all(pol[p]["all"]["P2"]["pass_vs_stored_arrays"] for p in pol),
            "P2_code_level_causality_pass": all(pol[p]["all"]["P2"]["pass_code_level_causality"] for p in pol)}
