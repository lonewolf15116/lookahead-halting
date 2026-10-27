"""Step 4 reproducibility rerun on DEVELOPMENT data only: recompute a committed artifact and compare it leaf by leaf.

    python scripts/step4_rerun_dev.py c1freeze --out DIR   # C1.freeze(dev): selections, thresholds, pooled OOD, expected precision, probe weights
    python scripts/step4_rerun_dev.py e1       --out DIR   # E1 (run_e1 on dev + Phase 1 validation)
    python scripts/step4_rerun_dev.py x1       --out DIR   # X1 (run_x1 on dev)

Nothing is written to results/ and no committed artifact is touched: outputs go to DIR (created exclusively). Real data is loaded only
through phase3_g3a.load_g3a (after the G3a script's own frozen-input check); any provenance path that mentions the sealed TEST directory
aborts the run. GPU-dependent parts of the freeze (rho timings) are NOT rerun and are excluded from the comparison.
Writes DIR/<name>.json (the recomputed artifact) and DIR/<name>_compare.json (leaf-by-leaf comparison with the committed artifact).
"""
import argparse
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from lookahead import c1 as C1
from lookahead import e1 as E
from lookahead import gate1 as g1
from lookahead import phase3_g3a as G
from lookahead import x1_flips as X
from lookahead.config import REPO_ROOT

SEALED = "extracted_phase3_test"
IGNORE = {"provenance", "started_utc", "finished_utc", "rho", "model_file", "model_file_sha256"}


def roundtrip(o):
    return json.loads(json.dumps(o, indent=1, sort_keys=True, default=str))


def leaves(o, path=""):
    if isinstance(o, dict):
        for k in sorted(o):
            if k in IGNORE:
                continue
            yield from leaves(o[k], f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def compare(new: dict, old: dict) -> dict:
    a, b = dict(leaves(new)), dict(leaves(old))
    only_new, only_old = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    n = exact = 0
    nonnum = []
    worst = []                                    # (abs diff, path, new, old)
    for p in sorted(set(a) & set(b)):
        x, y = a[p], b[p]
        n += 1
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                exact += 1
            else:
                worst.append((abs(x - y), p, x, y))
        elif x == y:
            exact += 1
        else:
            nonnum.append((p, x, y))
    worst.sort(reverse=True)
    mx = worst[0][0] if worst else 0.0
    return {"n_common_leaves": n, "n_bit_identical": exact, "n_numeric_differ": len(worst), "max_abs_diff": mx,
            "n_only_in_recomputed": len(only_new), "n_only_in_committed": len(only_old),
            "only_in_recomputed": only_new[:20], "only_in_committed": only_old[:20],
            "non_numeric_mismatches": [dict(path=p, new=str(x)[:80], old=str(y)[:80]) for p, x, y in nonnum[:20]],
            "n_non_numeric_mismatches": len(nonnum),
            "largest_numeric_differences": [dict(path=p, new=x, old=y, abs_diff=d) for d, p, x, y in worst[:15]]}


def g3a_check(root):
    spec = importlib.util.spec_from_file_location("g3a_script", root / "scripts/phase3_g3a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_frozen(root)


def abort_if_sealed(prov):
    if SEALED in json.dumps(prov, default=str).replace("\\", "/"):
        raise SystemExit("refusing: provenance mentions the sealed TEST directory")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["c1freeze", "e1", "x1"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = REPO_ROOT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)                       # exclusive
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("scipy", "sklearn", "pandas", "pyarrow"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:
            env[mod] = f"unavailable ({type(e).__name__})"
    g3a_check(root)
    extra = {}
    if args.what == "c1freeze":
        dev, _ = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=False)
        abort_if_sealed(dev.provenance)
        frozen, models = C1.freeze(dev)
        new = roundtrip({k: v for k, v in frozen.items()})
        old = json.loads((root / "results/c1/frozen.json").read_text(encoding="utf-8"))
        # probe weights vs the committed npz
        npz = np.load(root / "results/c1/pi1x_models.npz")
        wd = []
        for t, (sc, m) in enumerate(models):
            for nm, arr in ((f"loop{t + 1}_mean", sc["mean"]), (f"loop{t + 1}_sd", sc["sd"]), (f"loop{t + 1}_coef", m.coef),
                            (f"loop{t + 1}_intercept", np.array(m.intercept))):
                ref = npz[nm]
                wd.append({"array": nm, "max_abs_diff": float(np.max(np.abs(np.asarray(arr, dtype=float) - ref))),
                           "bit_identical": bool(np.array_equal(np.asarray(arr), ref))})
        extra["probe_weights_vs_committed_npz"] = wd
    elif args.what == "e1":
        dev, val = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=True)
        abort_if_sealed({**dev.provenance, **val.provenance})
        new = roundtrip(E.run_e1(dev, val))
        old = json.loads((root / "results/exploratory/e1_report.json").read_text(encoding="utf-8"))
    else:
        dev, _ = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=False)
        abort_if_sealed(dev.provenance)
        new = roundtrip(X.run_x1(dev))
        old = json.loads((root / "results/exploratory/x1_flips.json").read_text(encoding="utf-8"))
    with open(out / f"{args.what}.json", "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(new, indent=1, sort_keys=True) + "\n")
    cmp_ = compare(new, old)
    rep = {"what": args.what, "environment_of_rerun": env, "comparison": cmp_, **extra}
    with open(out / f"{args.what}_compare.json", "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1) + "\n")
    c = cmp_
    print(f"{args.what}: {c['n_common_leaves']} common leaves, {c['n_bit_identical']} bit-identical, {c['n_numeric_differ']} numeric differences "
          f"(max |diff| {c['max_abs_diff']:.3e}), {c['n_non_numeric_mismatches']} non-numeric mismatches, "
          f"{c['n_only_in_recomputed']} only in recomputed, {c['n_only_in_committed']} only in committed")
    for w in c["largest_numeric_differences"][:5]:
        print("  ", w)
    for w in extra.get("probe_weights_vs_committed_npz", [])[:12]:
        print("  weights", w)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
