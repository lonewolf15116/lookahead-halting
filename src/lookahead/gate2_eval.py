"""Gate 2 confirmation evaluation (plan v3 section 7 plus amendment v3.1, section 7b).

Pure evaluation of FROZEN policies on one evaluation set. Nothing here selects or tunes anything: thresholds, probe
weights and the benchmark depth vector B come from committed frozen files. The guarded command line lives in
scripts/gate2_confirm.py; this module has no knowledge of whether the set is the confirmation set or a rehearsal on
validation, except that `eval_loader` can read the confirmation directory (and only does so inside
`allow_confirmation_paths()`).

Original verdict (section 7, unchanged): L (budget), E (effect vs the best validation-qualified baseline, re-determined
inside each bootstrap resample), H (hidden-state benefit vs Q). Amendment v3.1 analyses are reported separately and never
change the verdict.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lookahead import gate1 as g
from lookahead import gate2 as g2

LOOPS = g.LOOPS
MARGIN = 0.03                      # section 7 practical-significance margin
PER_BENCH_CAP = 1.0                # pp, sensitivity only
FIXED4 = "fixed-4"


# ------------------------------------------------------------------------------------------------ verdict
def verdict(L: bool, E: bool, H: bool) -> str:
    """Section 7 table, exhaustive: L no -> FAIL; L yes, E no -> FAIL; L yes, E yes, H yes -> PASS; H no -> PARTIAL."""
    if not L or not E:
        return "FAIL"
    return "PASS" if H else "PARTIAL"


# ------------------------------------------------------------------------------------------------ loading
@contextlib.contextmanager
def allow_confirmation_paths():
    """The only place the Gate 2 'never open confirmation data' guard is relaxed. Used by the guarded confirm script
    after every pre-flight check; selection code never calls it."""
    saved = g2.FORBIDDEN_MARKERS
    g2.FORBIDDEN_MARKERS = ()
    try:
        yield
    finally:
        g2.FORBIDDEN_MARKERS = saved


def eval_loader(root, benchmarks, subdir: str, split: str, g_clip, provenance: dict | None = None) -> g2.ProbeSet:
    """Same construction as gate2.load_probe_data, but for data/<subdir> and any split label. A rehearsal on validation
    (subdir 'extracted', split 'validation') must reproduce gate2.load_probe_data exactly (checked by the script)."""
    root = Path(root)
    prov = {} if provenance is None else provenance
    arrays, per, n_failed = {}, {}, {}
    for bench in benchmarks:
        if bench not in g.GATE1_BENCHMARKS:
            raise ValueError(f"{bench!r} is not a natural Gate 1 benchmark")
        bdir = root / "data" / subdir / bench
        g2._assert_allowed_path(bdir)
        done_paths = sorted(bdir.glob("part-*.done.json"))
        if not done_paths:
            raise FileNotFoundError(f"no completed parts for {bench} under {bdir}")
        frames, hiddens = [], []
        for dp in done_paths:
            df, h, _ = g2._read_part_with_hidden(root, bench, dp, split, prov)
            if len(df):
                frames.append(df)
                hiddens.append(h)
        if not frames:
            raise ValueError(f"{bench}: no {split} rows")
        df = pd.concat(frames, ignore_index=True)
        if not (df["benchmark"] == bench).all():
            raise ValueError(f"{bench}: rows of another benchmark")
        hid = np.concatenate(hiddens)
        qids_order = df["doc_id"].to_numpy().reshape(-1, LOOPS)[:, 0]
        if len(set(qids_order)) != len(qids_order):
            raise ValueError(f"{bench}: a doc_id appears twice")
        order = np.argsort(qids_order, kind="stable")
        df = df.iloc[(order[:, None] * LOOPS + np.arange(LOOPS)).ravel()].reset_index(drop=True)
        hid = hid[order]
        a = g.build_arrays(df, split)
        assert np.array_equal(a["doc_id"], qids_order[order]), "doc_id order mismatch between hidden and arrays"
        n = a["n"]

        def col(name):
            return df[name].to_numpy().astype(float).reshape(n, LOOPS)

        vent = np.nan_to_num(col("vent")[:, :3], nan=0.0)
        Graw = np.nan_to_num(col("G")[:, :3], nan=0.0)
        n_opt = np.nan_to_num(df["n_options"].to_numpy().astype(float).reshape(n, LOOPS)[:, 0], nan=0.0)
        n_tok = np.nan_to_num(df["n_tokens"].to_numpy().astype(float).reshape(n, LOOPS)[:, 0], nan=1.0)
        strat = df["stratum"].to_numpy().astype(str).reshape(n, LOOPS)[:, 0]
        ok = a["ok"]
        h32 = hid[:, :g2.PROBE_LOOPS].astype(np.float32)
        h32[~ok] = 0.0
        arrays[bench] = a
        n_failed[bench] = {"n": int(n), "failed": int((~ok).sum())}
        per[bench] = {"h": h32, "x": g2.cheap_features(a, vent, n_opt, n_tok), "strat": strat,
                      "targets": g2.targets_from_arrays(a, Graw)}
    names = list(benchmarks)
    sizes = [arrays[b]["n"] for b in names]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    slices = {b: (int(bounds[i]), int(bounds[i + 1])) for i, b in enumerate(names)}
    doc_id = np.concatenate([arrays[b]["doc_id"] for b in names]).astype(str)
    ps = g2.ProbeSet(
        split=split, benchmarks=names, slices=slices, arrays=arrays, doc_id=doc_id,
        bench=np.concatenate([[b] * arrays[b]["n"] for b in names]).astype(str),
        stratum=np.concatenate([per[b]["strat"] for b in names]),
        ok=np.concatenate([arrays[b]["ok"] for b in names]),
        h=np.concatenate([per[b]["h"] for b in names]), x=np.concatenate([per[b]["x"] for b in names]),
        targets={k: np.concatenate([per[b]["targets"][k] for b in names]) for k in per[names[0]]["targets"]},
        fold=np.array([g2.fold_of(d) for d in doc_id]), es=np.array([g2.in_es_slice(d) for d in doc_id]),
        n_failed=n_failed, provenance=prov)
    ps.g_clip = tuple(g_clip)
    ps.targets["G"] = np.clip(ps.targets["G_raw"], ps.g_clip[0], ps.g_clip[1])
    return ps


def load_frozen_probe(root, entry: dict) -> g2.Probe:
    """Load a frozen probe and check both files against the sha256 recorded at freeze time."""
    root = Path(root)
    for key_file, key_sha in (("model_file", "model_sha256"), ("standardisation_file", "standardisation_sha256")):
        sha = g.sha256_file(root / entry[key_file])
        if sha != entry[key_sha]:
            raise ValueError(f"{entry[key_file]}: sha256 {sha} differs from the frozen {entry[key_sha]}")
    arrs = dict(np.load(root / entry["model_file"], allow_pickle=False))
    scal = dict(np.load(root / entry["standardisation_file"], allow_pickle=False))
    return g2.Probe.from_arrays(arrs, scal)


# ------------------------------------------------------------------------------------------------ policies
def build_policies(frozen: dict, depthsB, depths_cap, benchmarks, control_threshold) -> list:
    """Ordered list of (name, role, dataset_key, kind, params); params is a scalar or {benchmark: scalar}."""
    pol = [(FIXED4, "reference", "base", "fixed", {b: 4 for b in benchmarks})]
    if not frozen["P"]["qualified"]:
        raise ValueError("frozen P is not qualified; the Gate 2 verdict is FAIL by the plan before any evaluation")
    pol.append(("P", "candidate", "P", "native_gate", frozen["P"]["threshold"]))
    pol.append(("Q", "control_Q", "Q", "native_gate", frozen["Q"]["threshold"]))
    for n in g.CHEAP_BASELINES:
        f = frozen["baselines"][n]
        if not f["qualified"]:
            continue
        kind, param = g.policy_of(n, f)
        if param is None:
            raise ValueError(f"{n} is qualified but has no threshold")
        pol.append((n, "baseline", "base", kind, param))
    pol.append(("B", "amendment_B", "base", "fixed", dict(zip(benchmarks, depthsB))))
    pol.append(("B_cap", "sensitivity", "base", "fixed", dict(zip(benchmarks, depths_cap))))
    if control_threshold is not None:
        pol.append(("shuffled_control", "control_c", "control", "native_gate", control_threshold))
    return pol


def outcomes(pols: list, datasets: dict, benchmarks):
    """C[b], E[b]: [n_b, n_policies] correctness and executed loops of every policy on every question."""
    C, E = {}, {}
    for b in benchmarks:
        cs, es = [], []
        for name, role, key, kind, params in pols:
            p = params[b] if isinstance(params, dict) else params
            e, c = g.run_policy(datasets[key][b], kind, p)
            cs.append(c)
            es.append(e)
        C[b] = np.stack(cs, axis=1).astype(float)
        E[b] = np.stack(es, axis=1).astype(float)
    return C, E


def point_estimates(C, E, benchmarks):
    """acc [P, nb] (pp) and sav [P, nb]."""
    acc = np.stack([100.0 * C[b].mean(axis=0) for b in benchmarks], axis=1)
    sav = np.stack([1.0 - E[b].mean(axis=0) / LOOPS for b in benchmarks], axis=1)
    return acc, sav


def bootstrap(C, E, benchmarks, n_boot: int, seed: int):
    """Paired (same question draw for every policy), benchmark-stratified bootstrap: acc, sav of shape [R, P, nb]."""
    P = C[benchmarks[0]].shape[1]
    nb = len(benchmarks)
    rng = np.random.default_rng(seed)
    acc = np.empty((n_boot, P, nb))
    sav = np.empty((n_boot, P, nb))
    for r in range(n_boot):
        for bi, b in enumerate(benchmarks):
            n = C[b].shape[0]
            idx = rng.integers(0, n, n)
            acc[r, :, bi] = 100.0 * C[b][idx].mean(axis=0)
            sav[r, :, bi] = 1.0 - E[b][idx].mean(axis=0) / LOOPS
    return acc, sav


def _ci(x):
    return g.ci95(x)


# ------------------------------------------------------------------------------------------------ analysis
def analyse(pols: list, C, E, benchmarks, n_boot: int, seed: int) -> dict:
    names = [p[0] for p in pols]
    ix = {n: i for i, n in enumerate(names)}
    nb = len(benchmarks)
    acc_pt, sav_pt = point_estimates(C, E, benchmarks)
    acc_bs, sav_bs = bootstrap(C, E, benchmarks, n_boot, seed)
    f4 = ix[FIXED4]

    def msav(a, i, mask=None):          # macro saving of policy i
        return a[..., i, :].mean(axis=-1) if mask is None else a[..., i, :][..., mask].mean(axis=-1)

    def mloss(a, i, mask=None):         # macro loss vs fixed-4
        d = a[..., f4, :] - a[..., i, :]
        return d.mean(axis=-1) if mask is None else d[..., mask].mean(axis=-1)

    # pooled (question-weighted) point estimates
    pooled = {}
    N = sum(C[b].shape[0] for b in benchmarks)
    for n, i in ix.items():
        c = sum(C[b][:, i].sum() for b in benchmarks)
        e = sum(E[b][:, i].sum() for b in benchmarks)
        c4 = sum(C[b][:, f4].sum() for b in benchmarks)
        pooled[n] = {"saving": float(1 - e / (LOOPS * N)), "loss": float(100.0 * (c4 - c) / N)}

    point = {}
    for n, i in ix.items():
        point[n] = {
            "role": pols[i][1],
            "macro_saving": float(sav_pt[i].mean()), "macro_loss": float((acc_pt[f4] - acc_pt[i]).mean()),
            "macro_acc": float(acc_pt[i].mean()),
            "saving_ci": _ci(msav(sav_bs, i)), "loss_ci": _ci(mloss(acc_bs, i)),
            "pooled": pooled[n],
            "per_benchmark": {b: {"saving": float(sav_pt[i, bi]), "loss": float(acc_pt[f4, bi] - acc_pt[i, bi]),
                                  "acc": float(acc_pt[i, bi]),
                                  "loss_ci": _ci(acc_bs[:, f4, bi] - acc_bs[:, i, bi])}
                              for bi, b in enumerate(benchmarks)},
            "exit_distribution": {b: {str(k): float((E[b][:, i] == k).mean()) for k in range(1, LOOPS + 1)}
                                  for b in benchmarks},
            "per_benchmark_loss_over_cap": [b for bi, b in enumerate(benchmarks)
                                            if acc_pt[f4, bi] - acc_pt[i, bi] > PER_BENCH_CAP + g.TOL]}

    def diff(a_name, b_name, mask=None):
        i, j = ix[a_name], ix[b_name]
        pt = float(msav(sav_pt, i, mask) - msav(sav_pt, j, mask))
        bs = msav(sav_bs, i, mask) - msav(sav_bs, j, mask)
        return {"delta": pt, "ci": _ci(bs)}

    # ------------------------------------------------------------ original verdict (section 7, unchanged)
    qb = [p[0] for p in pols if p[1] == "baseline"]       # all validation-qualified baselines
    qi = [ix[n] for n in qb]

    def best_baseline(sav, idxs, mask=None):
        return np.max(np.stack([msav(sav, i, mask) for i in idxs], axis=-1), axis=-1)

    S_B_pt = float(best_baseline(sav_pt, qi))
    arg_B = qb[int(np.argmax([sav_pt[i].mean() for i in qi]))]
    d_base_bs = msav(sav_bs, ix["P"]) - best_baseline(sav_bs, qi)
    d_base = {"delta": float(sav_pt[ix["P"]].mean() - S_B_pt), "ci": _ci(d_base_bs), "comparator": arg_B,
              "comparator_saving": S_B_pt}
    d_ctrl = diff("P", "Q")
    L = bool(point["P"]["macro_loss"] <= g.LOSS_LIMIT + g.TOL)
    Ecnd = bool(d_base["delta"] >= MARGIN and d_base["ci"][0] > 0)
    Hcnd = bool(d_ctrl["ci"][0] > 0)
    original = {"L_budget": L, "E_effect": Ecnd, "H_hidden_state": Hcnd, "verdict": verdict(L, Ecnd, Hcnd),
                "P_macro_loss": point["P"]["macro_loss"], "P_macro_saving": point["P"]["macro_saving"],
                "delta_base": d_base, "delta_ctrl_P_minus_Q": d_ctrl, "margin": MARGIN,
                "baselines_confirmation": {n: {"saving": point[n]["macro_saving"], "loss": point[n]["macro_loss"]}
                                           for n in qb},
                "note": "B is the largest confirmation saving among all validation-qualified baselines, including any "
                        "that exceed 1 pp loss on confirmation; it is re-determined inside every resample."}
    comp = [n for n in qb if point[n]["macro_loss"] <= g.LOSS_LIMIT + g.TOL]
    secondary = {"loss_compliant_baselines": comp}
    if comp:
        ci_ = [ix[n] for n in comp]
        bs = msav(sav_bs, ix["P"]) - best_baseline(sav_bs, ci_)
        pt = float(sav_pt[ix["P"]].mean() - max(sav_pt[i].mean() for i in ci_))
        secondary["P_minus_best_loss_compliant_baseline"] = {"delta": pt, "ci": _ci(bs)}
        bsq = msav(sav_bs, ix["Q"]) - best_baseline(sav_bs, ci_)
        secondary["Q_minus_best_loss_compliant_baseline"] = {
            "delta": float(sav_pt[ix["Q"]].mean() - max(sav_pt[i].mean() for i in ci_)), "ci": _ci(bsq)}
    bsq = msav(sav_bs, ix["Q"]) - best_baseline(sav_bs, qi)
    secondary["Q_minus_best_baseline_all_qualified"] = {
        "delta": float(sav_pt[ix["Q"]].mean() - S_B_pt), "ci": _ci(bsq)}
    no_obqa = [i for i, b in enumerate(benchmarks) if b != "openbookqa"]
    if len(no_obqa) < nb:
        mask = np.array([b != "openbookqa" for b in benchmarks])
        d_nb = msav(sav_bs, ix["P"], mask) - best_baseline(sav_bs, qi, mask)
        secondary["sensitivity_without_openbookqa"] = {
            "P_loss": float(mloss(acc_pt, ix["P"], mask)),
            "P_minus_best_baseline": {"delta": float(msav(sav_pt, ix["P"], mask) - best_baseline(sav_pt, qi, mask)),
                                      "ci": _ci(d_nb)},
            "P_minus_Q": diff("P", "Q", mask)}

    # ------------------------------------------------------------ amendment v3.1 (separate; never changes the verdict)
    amend = {
        "task_aware_note": "B uses benchmark identity: a task-aware control, not a deployable policy. B was chosen on "
                           "validation among 1,024 vectors, so its validation numbers were optimistic.",
        "P_minus_B": diff("P", "B"), "Q_minus_B": diff("Q", "B"),
        "P_minus_fixed3": diff("P", "fixed-3"), "Q_minus_fixed3": diff("Q", "fixed-3"),
        "B": {"saving": point["B"]["macro_saving"], "loss": point["B"]["macro_loss"],
              "saving_ci": point["B"]["saving_ci"], "loss_ci": point["B"]["loss_ci"]},
        "fixed3": {"saving": point["fixed-3"]["macro_saving"], "loss": point["fixed-3"]["macro_loss"]},
        "B_meets_macro_budget": bool(point["B"]["macro_loss"] <= g.LOSS_LIMIT + g.TOL),
        "per_benchmark_cap_sensitivity": {
            "cap_pp": PER_BENCH_CAP,
            "policies_over_cap": {n: point[n]["per_benchmark_loss_over_cap"] for n in names if n != FIXED4},
            "B_cap_vector": {"saving": point["B_cap"]["macro_saving"], "loss": point["B_cap"]["macro_loss"]}},
        "interpretation_guard": "The original verdict is reported separately above. B outperforming P or Q does not "
                                "alter it."}
    if "shuffled_control" in ix:
        amend["shuffled_control_minus_B"] = diff("shuffled_control", "B")
    return {"policies": names, "n_boot": n_boot, "seed": seed, "n_questions": {b: int(C[b].shape[0]) for b in benchmarks},
            "point": point, "original": original, "secondary": secondary, "amendment": amend}


def auroc_report(ps: g2.ProbeSet, score_sets: dict) -> dict:
    """Secondary: AUROC for S_t (loops 1-3), pooled over ok questions and within each benchmark."""
    S = ps.targets["S"]
    out = {}
    for name, sc in score_sets.items():
        out[name] = {"pooled": [g2._auroc(S[ps.ok, t], sc[ps.ok, t]) for t in range(g2.PROBE_LOOPS)], "within": {}}
        for b in ps.benchmarks:
            a, e = ps.slices[b]
            rows = a + np.flatnonzero(ps.ok[a:e])
            out[name]["within"][b] = [g2._auroc(S[rows, t], sc[rows, t]) for t in range(g2.PROBE_LOOPS)]
    return out


# ------------------------------------------------------------------------------------------------ rendering
def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci_s(c, nd=3):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def render(rep: dict, title: str) -> str:
    a, o = rep["analysis"], rep["analysis"]["original"]
    pt = a["point"]
    L = [f"# {title}\n", f"{rep['banner']}\n",
         f"Generated {rep['timestamp_utc']}. Frozen selection sha256 `{rep['frozen_selection_sha256']}`; baseline B file "
         f"sha256 `{rep['baseline_B_sha256']}`. Nothing was reselected or retuned on this set.\n",
         "## 1. Original Gate 2 verdict (plan v3 section 7, unchanged)\n",
         f"**{o['verdict']}**  (L budget: {o['L_budget']}, E effect: {o['E_effect']}, H hidden state: {o['H_hidden_state']})\n",
         f"- P: macro saving {_f(o['P_macro_saving'])}, macro loss {_f(o['P_macro_loss'], 2)} pp "
         f"(budget 1.00 pp, point estimate).",
         f"- Δ_base = S_P − S_best-baseline = {_f(o['delta_base']['delta'])} "
         f"(comparator {o['delta_base']['comparator']}, saving {_f(o['delta_base']['comparator_saving'])}); "
         f"95% CI {_ci_s(o['delta_base']['ci'])}; needs ≥ {o['margin']} and CI lower bound > 0.",
         f"- Δ_ctrl = S_P − S_Q = {_f(o['delta_ctrl_P_minus_Q']['delta'])}; 95% CI {_ci_s(o['delta_ctrl_P_minus_Q']['ci'])}; "
         "needs CI lower bound > 0.\n",
         "Baselines on this set (validation-qualified, frozen thresholds):\n",
         "| baseline | macro saving | macro loss (pp) |", "|---|---|---|"]
    for n, v in o["baselines_confirmation"].items():
        L.append(f"| {n} | {_f(v['saving'])} | {_f(v['loss'], 2)} |")
    L += ["", "## 2. All policies, macro (equal weight over benchmarks)\n",
          "| policy | role | saving [95% CI] | loss pp [95% CI] | pooled saving | pooled loss |", "|---|---|---|---|---|---|"]
    for n in a["policies"]:
        v = pt[n]
        L.append(f"| {n} | {v['role']} | {_f(v['macro_saving'])} {_ci_s(v['saving_ci'])} | {_f(v['macro_loss'], 2)} "
                 f"{_ci_s(v['loss_ci'], 2)} | {_f(v['pooled']['saving'])} | {_f(v['pooled']['loss'], 2)} |")
    L += ["", "## 3. Amendment v3.1 (separate from the verdict)\n", a["amendment"]["task_aware_note"] + "\n",
          "| comparison | Δ saving | 95% CI |", "|---|---|---|"]
    am = a["amendment"]
    for k in ("P_minus_B", "Q_minus_B", "P_minus_fixed3", "Q_minus_fixed3", "shuffled_control_minus_B"):
        if k in am:
            L.append(f"| {k} | {_f(am[k]['delta'])} | {_ci_s(am[k]['ci'])} |")
    L += ["", f"B meets the macro budget on this set: {am['B_meets_macro_budget']}. {am['interpretation_guard']}\n",
          "### Per-benchmark saving / loss (pp), point estimates\n"]
    benches = list(rep["benchmarks"])
    L.append("| policy | " + " | ".join(benches) + " |")
    L.append("|---|" + "---|" * len(benches))
    for n in a["policies"]:
        L.append(f"| {n} | " + " | ".join(f"{_f(pt[n]['per_benchmark'][b]['saving'], 2)} / {_f(pt[n]['per_benchmark'][b]['loss'], 1)}"
                                          for b in benches) + " |")
    cap = am["per_benchmark_cap_sensitivity"]
    L += ["", f"### Sensitivity: per-benchmark loss cap ({cap['cap_pp']} pp; not a Gate 2 condition)\n",
          "| policy | benchmarks over the cap |", "|---|---|"]
    for n, over in cap["policies_over_cap"].items():
        L.append(f"| {n} | {', '.join(over) if over else 'none'} |")
    L += ["", f"B_cap vector on this set: saving {_f(cap['B_cap_vector']['saving'])}, loss {_f(cap['B_cap_vector']['loss'], 2)} pp.\n",
          "## 4. Secondary analyses (never change the verdict)\n"]
    sec = a["secondary"]
    for k, v in sec.items():
        if isinstance(v, dict) and "delta" in v:
            L.append(f"- {k}: Δ = {_f(v['delta'])}, 95% CI {_ci_s(v['ci'])}")
    if "sensitivity_without_openbookqa" in sec:
        s = sec["sensitivity_without_openbookqa"]
        L.append(f"- without OpenBookQA: P loss {_f(s['P_loss'], 2)} pp; P − best baseline {_f(s['P_minus_best_baseline']['delta'])} "
                 f"{_ci_s(s['P_minus_best_baseline']['ci'])}; P − Q {_f(s['P_minus_Q']['delta'])} {_ci_s(s['P_minus_Q']['ci'])}")
    L.append("- Not run here (retrospective, to be added only after the verdict is recorded): the accuracy-budget frontier "
             "and leave-one-benchmark-out probes (plan 7a).")
    if rep.get("auroc"):
        L += ["", "AUROC for S_t (loops 1-3), pooled: " + "; ".join(
            f"{k}: " + ", ".join(_f(x) for x in v["pooled"]) for k, v in rep["auroc"].items())]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- confirmation identity (metadata only)
EXPECTED_CONFIRMATION_COUNTS = {"arc_challenge": 299, "arc_easy": 570, "hellaswag": 1000, "mmlu": 1500,
                                "commonsense_qa": 1000, "openbookqa": 500}
# Hashes recorded in docs/DECISIONS.md (confirmation prep, audit pass and extraction evidence entries).
PINNED_CONFIRMATION_HASHES = {
    "ids_sha256": "4cc7aac3febca19163aee8c9d554ac805f94390f2f4c48f1b2d52885cd752dbc",
    "manifest_sha256": "eac4680d39abf0c4d418117b88648896d912f69d5f6615602dfc1d77da35e0a0",
    "audit_sha256": "72e3b57f8a1f9d554a6814a18457cd0bb53c959a42b1c6f95c5189d5c5d3ddce",
    "check_report_sha256": "8efafa5fcc7de0f899d4a740b9d0e7fdb87d3b9ce3c8479c19aa9a95cb8ef575",
    "combined_parts_sha256": "98f7cbd6a7495d5e3d7dacead7b186c5092fd70ad2aa71c9d945bde0b44d6213"}


def confirmation_identity_preflight(root, benchmarks, expected_counts=None, pinned=None) -> list:
    """Is the extracted confirmation data exactly the artifact that was audited and verified? Uses committed manifests,
    done.json metadata (doc ids, statuses, counts, hashes) and file byte hashes only: no correctness, no outcomes.
    Returns a list of failures (empty = identity confirmed)."""
    root = Path(root)
    exp = EXPECTED_CONFIRMATION_COUNTS if expected_counts is None else expected_counts
    pin = PINNED_CONFIRMATION_HASHES if pinned is None else pinned
    bad = []

    def sha(path):
        return g.sha256_file(path)

    man_p, ids_p = root / "data/manifests/confirmation.json", root / "data/manifests/confirmation_ids.csv"
    aud_p = root / "results/gate2/confirmation_audit.json"
    evd_p = root / "results/gate2/confirmation_extraction_evidence.json"
    chk_p = root / "results/gate2/extraction_check_confirm.json"
    for name, path, key in (("ids csv", ids_p, "ids_sha256"), ("manifest", man_p, "manifest_sha256"),
                            ("audit", aud_p, "audit_sha256"), ("extraction check report", chk_p, "check_report_sha256")):
        if not path.exists():
            bad.append(f"{name} missing: {path}")
        elif sha(path) != pin[key]:
            bad.append(f"{name} sha256 {sha(path)} != pinned {pin[key]}")
    if bad:
        return bad
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    evidence = json.loads(evd_p.read_text(encoding="utf-8"))
    for k_m, k_p in (("ids_sha256", "ids_sha256"), ("audit_sha256", "audit_sha256")):
        if manifest[k_m] != pin[k_p]:
            bad.append(f"manifest.{k_m} != pinned")
    for k in ("ids_sha256", "manifest_sha256", "check_report_sha256", "combined_parts_sha256"):
        if evidence[k] != pin[k]:
            bad.append(f"extraction evidence {k} != pinned")
    if evidence.get("outcomes_inspected") is not False:
        bad.append("extraction evidence does not say outcomes_inspected is false")
    if evidence["total_records"] != sum(exp.values()):
        bad.append(f"evidence total_records {evidence['total_records']} != {sum(exp.values())}")

    # ids from the committed csv, grouped by benchmark
    import csv
    ids = {b: [] for b in benchmarks}
    with open(ids_p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["benchmark"] not in ids:
                bad.append(f"csv has benchmark {row['benchmark']!r} outside the six")
                continue
            if row["split"] != "confirmation":
                bad.append(f"csv row with split {row['split']!r}")
            ids[row["benchmark"]].append(row["doc_id"])
    comb = hashlib.sha256()
    for b in benchmarks:
        if len(ids[b]) != exp[b] or len(set(ids[b])) != len(ids[b]):
            bad.append(f"{b}: csv has {len(ids[b])} ids ({len(set(ids[b]))} distinct), expected {exp[b]}")
        if manifest["benchmarks"][b]["n_questions"] != exp[b]:
            bad.append(f"{b}: manifest n_questions {manifest['benchmarks'][b]['n_questions']} != {exp[b]}")

    # extracted parts: ids, statuses, hashes
    for b in benchmarks:
        bdir = root / "data" / "extracted_confirm" / b
        done_paths = sorted(bdir.glob("part-*.done.json"))
        ev_parts = evidence["parts"][b]
        if len(done_paths) != len(ev_parts):
            bad.append(f"{b}: {len(done_paths)} parts on disk, {len(ev_parts)} in the evidence")
            continue
        got_ids, n_failed = [], 0
        for dp, evp in zip(done_paths, ev_parts):
            d = json.loads(dp.read_text(encoding="utf-8"))
            stem = str(dp)[: -len(".done.json")]
            if dp.name[: -len(".done.json")] != evp["part"] or d["n"] != evp["n"]:
                bad.append(f"{b}/{dp.name}: part name or n differs from the evidence")
            for key, ext in (("parquet_sha256", ".parquet"), ("hidden_sha256", ".hidden.npy")):
                if d[key] != evp[key]:
                    bad.append(f"{b}/{dp.name}: done.json {key} differs from the evidence")
                elif sha(Path(stem + ext)) != evp[key]:
                    bad.append(f"{b}/{dp.name}: file {ext} bytes differ from the verified hash")
            got_ids += list(d["doc_ids"])
            n_failed += sum(1 for s_ in d["statuses"] if s_ != "ok")
        if n_failed:
            bad.append(f"{b}: {n_failed} failed records (expected 0)")
        if len(got_ids) != exp[b] or set(got_ids) != set(ids[b]) or len(set(got_ids)) != len(got_ids):
            bad.append(f"{b}: extracted doc-id set differs from the committed id list "
                       f"({len(got_ids)} extracted, {len(ids[b])} listed, {len(set(got_ids) ^ set(ids[b]))} differ)")
    # combined hash over the parts, in the evidence's order (as when it was recorded)
    for b in evidence["parts"]:
        for evp in evidence["parts"][b]:
            comb.update(evp["parquet_sha256"].encode())
            comb.update(evp["hidden_sha256"].encode())
    if comb.hexdigest() != pin["combined_parts_sha256"]:
        bad.append("combined parts hash differs from the pinned value")
    return bad
