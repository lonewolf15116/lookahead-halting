"""D-t truncated-execution check (docs/DT_TRUNCATION_PLAN.md rev 1). Diagnostic [E]; DEVELOPMENT data only; TEST is never opened.

    python scripts/dt_truncation_check.py                 # laptop GPU, about 600 questions x (1 full + 3 truncated) runs

Order, enforced:
  1. every path is checked against data/extracted_phase3_test (refused), then results/c1/frozen.json and
     results/c1/pi1x_models.npz must equal their pinned sha256 -- before any data or model is loaded;
  2. results/diagnostics/dt_truncation_report.json must not exist (it is created exclusively at the end; never overwritten);
  3. DEV+ stored traces and questions are loaded through the same frozen-input checks as scripts/c1_freeze.py
     (scripts/phase3_g3a.py check_frozen, phase3_g3a.load_g3a, phase3_build.check_phase3_inputs);
  4. 600 questions are selected (sha256 of "lh-dt-v1:" + doc_id, 200 lowest per primary benchmark among status-ok DEV+),
     run truncated under the three frozen policies and in full, and compared with the retrospective C1 code path.
Nothing is tuned, and nothing is re-run to make a disagreement go away.
"""
import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from lookahead import dt_truncation as D
from lookahead import gate1 as g1
from lookahead import phase3_build as P
from lookahead import phase3_g3a as G
from lookahead.config import REPO_ROOT

CODE = ["src/lookahead/dt_truncation.py", "scripts/dt_truncation_check.py", "src/lookahead/c1_test.py",
        "src/lookahead/phase3_readouts.py", "src/lookahead/extract.py", "src/lookahead/packed.py"]


def _sha(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def real_loader(root: Path):
    """(dev, questions, provenance) exactly as scripts/c1_freeze.py obtains them: frozen-input check, DEV+ stored traces, DEV+
    questions with token encodings (hash-checked against the manifest)."""
    spec = importlib.util.spec_from_file_location("g3a_script", root / "scripts/phase3_g3a.py")
    g3a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g3a)
    inputs = g3a.check_frozen(root)
    dev, _ = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=False)
    _, _, _, qs = P.check_phase3_inputs(root, G.PRIMARY, None, "dev")
    return dev, qs, {"inputs": inputs, "data_files": dev.provenance}


def real_scorer(device: str):
    from lookahead.ouro_adapter import load_ouro
    from lookahead.packed import PackedScorer

    model, _ = load_ouro(device)
    return PackedScorer(model), model


def environment(model) -> dict:
    import platform

    import torch
    import transformers

    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "device": str(next(model.parameters()).device), "dtype": str(next(model.parameters()).dtype),
           "attn_implementation": getattr(model.config, "_attn_implementation", None), "platform": platform.platform()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def main(argv=None, *, root=None, loader=None, scorer_factory=None, pinned=None, report_rel=D.REPORT) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-bench", type=int, default=D.N_PER_BENCH, help="plan value is 200; anything else is recorded as a deviation")
    args = ap.parse_args(argv)
    root = Path(root or REPO_ROOT)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_path = root / report_rel
    try:
        D.refuse_sealed(root, root / "data" / G.DEVPLUS_DIR, report_path)
        D.check_pinned(root, pinned or D.PINNED)                           # hashes before anything else is read
    except (SystemExit, PermissionError) as e:
        print(f"REFUSED (nothing read, nothing written): {e}", file=sys.stderr)
        return 2
    if report_path.exists():
        print(f"refusing: {report_path} already exists (never overwritten)", file=sys.stderr)
        return 2
    frozen = json.loads((root / "results/c1/frozen.json").read_text(encoding="utf-8"))
    from lookahead import c1_test as T

    with np.load(root / "results/c1/pi1x_models.npz") as npz:
        models = T.load_models(npz)
    dev, qs, prov = (loader or real_loader)(root)
    scorer, model = (scorer_factory or real_scorer)(args.device)
    rep = D.run_check(root, dev, qs, models, frozen, scorer, n_per_bench=args.n_per_bench,
                      log=lambda s: print(s, flush=True))
    rep["verdict"] = D.verdict(rep)
    rep["plan_conformant_n"] = args.n_per_bench == D.N_PER_BENCH
    rep["provenance"] = {"pinned": pinned or D.PINNED, "inputs": prov, "environment": environment(model),
                         "code": {f: _sha(root / f) for f in CODE if (root / f).exists()}, "git_commit": git_commit(root),
                         "started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "x", encoding="utf-8", newline="\n") as f:          # exclusive create: never overwrites
        f.write(json.dumps(rep, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    v = rep["verdict"]
    print(f"P-1 (exit loop and correctness agree on every question): {'PASS' if v['P1_pass'] else 'FAIL'}")
    print(f"P-2 (NaN-future statistic == stored full statistic, bit for bit): {'PASS' if v['P2_pass'] else 'FAIL'}"
          f"   code-level causality only: {'PASS' if v['P2_code_level_causality_pass'] else 'FAIL'}")
    for p, by in rep["policies"].items():
        for b, r in by.items():
            print(f"{p:10s} {b:15s} n={r['n']:3d}  exit agree {r['exit_agreement_pct']:6.2f}%  correct agree {r['correctness_agreement_pct']:6.2f}%  "
                  f"P-2 fails {r['P2']['fail_vs_stored']}/{r['P2']['checks']}  saving sim {r['simulated_gross_saving']:.3f} "
                  f"measured {r['measured_time_saving']:.3f}")
    mx = rep["p3"]["all"]
    print("P-3 max |fresh - stored|: " + "  ".join(f"{k} {mx[k]:.3g}" for k in D.P3_KEYS))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
