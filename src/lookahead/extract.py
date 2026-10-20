"""Phase 1 extraction: per-question, per-loop records and targets (plan sections 4, 5 and 7).

Each benchmark is written as fixed-size parts under <out>/<benchmark>/:

    part-00000.parquet    one row per (question, loop): scores, p_t, prediction, confidence, entropies, gate, targets
    part-00000.hidden.npy float16 [n_questions, 4, d]: h_t at the last context token, in the order of the parquet
    part-00000.done.json  written last: doc_ids, statuses, sha256 of both files. A part is complete iff this exists.

Parts are written to a temporary name and renamed into place, so an interruption leaves no half-written part that
looks complete. On resume, completed parts are never recomputed or rewritten; an incomplete part is recomputed whole.

Full-vocabulary log-probabilities at the last context token are kept in memory for the 4 loops of one question, reduced
to entropy and KL targets, and then discarded.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from lookahead.ouro_adapter import NativeGate
from lookahead.packed import PackedScorer, build_packed
from lookahead.questions import Question

LOOPS = 4
TIMEOUT_S = 60.0
TRANSITION = {(1, 1): "CC", (1, 0): "CW", (0, 1): "WC", (0, 0): "WW"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def option_record(q: Question, option_ll: list[float]) -> dict:
    ll = np.asarray(option_ll, dtype=np.float64)
    lln = ll / np.asarray([float(len(c)) for c in q.choice_text])
    scores = lln if q.metric == "acc_norm" else ll
    logp = log_softmax_np(scores)
    p = np.exp(logp)
    pred = int(np.argmax(scores))  # np.argmax: first index wins ties, as in lm_eval
    return {"opt_ll": ll.tolist(), "opt_lln": lln.tolist(), "logp": logp, "p": p.tolist(), "pred": pred,
            "correct": int(pred == q.target), "conf": float(p.max()), "ent": float(-(p * logp).sum()),
            "loss": float(-logp[q.target])}


def kl_logp(lp: np.ndarray, lq: np.ndarray) -> float:
    return float((np.exp(lp) * (lp - lq)).sum())


def vocab_kl(lp: torch.Tensor, lq: torch.Tensor) -> float:
    return float((lp.exp() * (lp - lq)).sum())


def extract_question(scorer: PackedScorer, q: Question):
    """Run all loops for one question. Returns (rows, hidden [LOOPS, d] float16 numpy, status)."""
    sync = torch.cuda.synchronize if scorer.device.type == "cuda" else (lambda: None)
    gate = NativeGate(scorer.runner.total_loops, 1)
    opts, vocab, hidden, gates, cdfs, ms = [], [], [], [], [], []
    sync()
    t0 = time.perf_counter()
    for out in scorer.loops(build_packed(q.context_enc, q.cont_encs)):
        sync()
        now = time.perf_counter()
        ms.append(1000 * (now - t0))
        t0 = now
        opts.append(option_record(q, out.option_ll))
        vocab.append(out.vocab_logp)
        hidden.append(out.hidden.float().cpu().numpy().astype(np.float16))
        gates.append(out.gate_logit)
        cdfs.append(float(gate.update(torch.tensor([out.gate_logit]))[0]))
    vent = [float(-(v.exp() * v).sum()) for v in vocab]
    T = len(opts)
    rows = []
    for t in range(T):
        o = opts[t]
        nxt = opts[t + 1] if t + 1 < T else None
        fin = opts[-1]
        rows.append({
            "loop": t + 1, "opt_ll": o["opt_ll"], "opt_lln": o["opt_lln"], "p": o["p"], "pred": o["pred"],
            "correct": o["correct"], "conf": o["conf"], "ent": o["ent"], "vent": vent[t], "gate": gates[t],
            "cdf": cdfs[t], "ms": ms[t], "loss": o["loss"],
            # next-loop targets (undefined at the last loop)
            "D": kl_logp(o["logp"], nxt["logp"]) if nxt else float("nan"),
            "D_vocab": vocab_kl(vocab[t], vocab[t + 1]) if nxt else float("nan"),
            "F": int(o["pred"] != nxt["pred"]) if nxt else -1,
            "T": TRANSITION[(o["correct"], nxt["correct"])] if nxt else "",
            "G": o["loss"] - nxt["loss"] if nxt else float("nan"),
            # final-loop targets
            "K": kl_logp(o["logp"], fin["logp"]),
            "K_vocab": vocab_kl(vocab[t], vocab[-1]),
            "F_final": int(o["pred"] != fin["pred"]),
            "T_final": TRANSITION[(o["correct"], fin["correct"])],
        })
    numbers = [v for r in rows for k, v in r.items() if isinstance(v, float) and k not in ("D", "D_vocab", "G")]
    numbers += [r[k] for r in rows[:-1] for k in ("D", "D_vocab", "G")]
    numbers += [x for r in rows for x in r["opt_ll"]]
    status = "ok"
    if not all(math.isfinite(x) for x in numbers) or not all(np.isfinite(h).all() for h in hidden):
        status = "nonfinite"
    elif sum(ms) / 1000 > TIMEOUT_S:
        status = "timeout"
    return rows, np.stack(hidden), status


def load_questions(path: Path) -> list[Question]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f]


def load_splits(path: Path) -> dict:
    out = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines[0] != "benchmark,stratum,doc_id,split":
        raise ValueError("unexpected splits header")
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"bad splits row: {line!r}")
        bench, stratum, doc_id, split = fields
        key = (bench, doc_id)
        if key in out:
            raise ValueError(f"question in more than one split row: {key}")
        if split not in ("train", "validation", "test"):
            raise ValueError(f"bad split {split!r} for {key}")
        out[key] = (stratum, split)
    return out


def save_npy(path, array) -> None:
    with open(path, "wb") as f:  # closed before the rename (Windows will not rename an open file)
        np.save(f, array)


def write_atomic(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def run_benchmark(scorer: PackedScorer, bench: str, questions: list[Question], splits: dict, out_dir: Path,
                  chunk: int = 256, stop_after: int | None = None, log_every: int = 500, hidden_size: int | None = None):
    """Extract one benchmark into parts. Returns (questions computed in this call, finished?)."""
    import pandas as pd

    bdir = out_dir / bench
    bdir.mkdir(parents=True, exist_ok=True)
    missing = [q.doc_id for q in questions if (bench, q.doc_id) not in splits]
    if missing:
        raise ValueError(f"{bench}: {len(missing)} questions have no split, e.g. {missing[:3]}")
    d = hidden_size or scorer.runner.config.hidden_size
    computed, started = 0, time.time()
    for k in range(0, len(questions), chunk):
        stem = bdir / f"part-{k // chunk:05d}"
        if Path(f"{stem}.done.json").exists():
            continue
        part = questions[k:k + chunk]
        rows, hid, statuses = [], np.full((len(part), LOOPS, d), np.nan, dtype=np.float16), []
        for i, q in enumerate(part):
            if stop_after is not None and computed >= stop_after:
                return computed, False  # simulated interruption: this part is left incomplete
            stratum, split = splits[(bench, q.doc_id)]
            base = {"benchmark": bench, "stratum": stratum, "doc_id": q.doc_id, "split": split, "target": q.target,
                    "n_options": len(q.cont_encs), "n_tokens": q.packed_len, "metric": q.metric}
            try:
                qrows, h, status = extract_question(scorer, q)
                hid[i, : len(h)] = h
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                qrows, status = [{"loop": t + 1} for t in range(LOOPS)], "oom"
            statuses.append(status)
            rows += [{**base, "status": status, **r} for r in qrows]
            computed += 1
            if log_every and computed % log_every == 0:
                rate = (time.time() - started) / computed
                print(f"  {bench}: {k + i + 1}/{len(questions)} ({rate:.2f} s per question)", flush=True)
        write_atomic(Path(f"{stem}.parquet"), lambda p: pd.DataFrame(rows).to_parquet(p, index=False))
        write_atomic(Path(f"{stem}.hidden.npy"), lambda p: save_npy(p, hid))
        done = {"doc_ids": [q.doc_id for q in part], "statuses": statuses, "n": len(part),
                "parquet_sha256": sha256_file(Path(f"{stem}.parquet")),
                "hidden_sha256": sha256_file(Path(f"{stem}.hidden.npy")), "finished_at": time.time()}
        write_atomic(Path(f"{stem}.done.json"), lambda p: Path(p).write_text(json.dumps(done), encoding="utf-8"))
    return computed, True


def read_benchmark(out_dir: Path, bench: str):
    """Load every completed part: (DataFrame of rows, hidden [n, LOOPS, d], list of done records)."""
    import pandas as pd

    bdir = out_dir / bench
    frames, hiddens, dones = [], [], []
    for done_path in sorted(bdir.glob("part-*.done.json")):
        stem = str(done_path)[: -len(".done.json")]
        done = json.loads(done_path.read_text(encoding="utf-8"))
        for name, key in ((".parquet", "parquet_sha256"), (".hidden.npy", "hidden_sha256")):
            if sha256_file(Path(stem + name)) != done[key]:
                raise ValueError(f"checksum mismatch: {stem + name}")
        frames.append(pd.read_parquet(stem + ".parquet"))
        hiddens.append(np.load(stem + ".hidden.npy"))
        dones.append(done)
    if not frames:
        return None, None, []
    return pd.concat(frames, ignore_index=True), np.concatenate(hiddens), dones
