"""C1 freeze (docs/C1_PROTOCOL.md C1.7 step 2): fit and select everything on DEVELOPMENT data, measure rho on the GPU.
NOT TO BE RUN before the reviewer approves the final protocol and this implementation. TEST is never read.

    python scripts/c1_freeze.py                       # real: results/c1/frozen.json + results/c1/pi1x_models.npz
    python scripts/c1_freeze.py --synthetic OUTDIR    # synthetic dry run (no real data, no GPU)

Refuses to overwrite. Before loading data it runs the G3a frozen-input check and pins the protocol file's sha256 (given on
the command line as --protocol-sha so the frozen file records exactly which text was approved).
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from lookahead import c1 as C1
from lookahead import gate1 as g1
from lookahead import phase3_build as P
from lookahead import phase3_g3a as G
from lookahead.config import REPO_ROOT

OUT = "results/c1"
CODE = ["src/lookahead/c1.py", "src/lookahead/c1_rho.py", "scripts/c1_freeze.py", "src/lookahead/e1.py",
        "src/lookahead/phase3_g3a.py", "src/lookahead/phase3_readouts.py"]


def _sha(p: Path) -> str:
    return g1.sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def save_models(path: Path, models: list) -> str:
    arrs = {}
    for t, (sc, m) in enumerate(models):
        arrs[f"loop{t + 1}_mean"], arrs[f"loop{t + 1}_sd"] = sc["mean"], sc["sd"]
        arrs[f"loop{t + 1}_coef"], arrs[f"loop{t + 1}_intercept"] = m.coef, np.array(m.intercept)
    with open(path, "wb") as f:
        np.savez(f, **arrs)
    return g1.sha256_file(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", metavar="OUTDIR")
    ap.add_argument("--protocol-sha", help="sha256 of docs/C1_PROTOCOL.md as approved by the reviewer (real run)")
    args = ap.parse_args(argv)
    root = REPO_ROOT
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.synthetic:
        out = Path(args.synthetic)
        sys.path.insert(0, str(root / "tests"))
        from test_phase3_g3a import synth

        dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
        inputs, questions = {"synthetic": True}, None
    else:
        out = root / OUT
        proto = _sha(root / "docs/C1_PROTOCOL.md")
        if args.protocol_sha != proto:
            raise SystemExit(f"docs/C1_PROTOCOL.md sha256 {proto} does not match --protocol-sha (the approved text)")
        spec = importlib.util.spec_from_file_location("g3a_script", root / "scripts/phase3_g3a.py")
        g3a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g3a)
        inputs = {**g3a.check_frozen(root), "docs/C1_PROTOCOL.md": proto}
        dev, _ = G.load_g3a(root, g1.GATE1_BENCHMARKS, with_validation=False)
        inputs["data_files"] = dev.provenance
        _, _, _, qs = P.check_phase3_inputs(root, G.PRIMARY, None, "dev")   # DEV+ questions for timing (hash-checked)
        questions = {b: sorted(qs[b], key=lambda q: q.doc_id) for b in G.PRIMARY}
    if (out / "frozen.json").exists():
        print(f"refusing: {out / 'frozen.json'} already exists", file=sys.stderr)
        return 2
    P.assert_not_sealed(out)
    frozen, models = C1.freeze(dev)
    out.mkdir(parents=True, exist_ok=True)
    frozen["pi1x"]["model_file"] = f"{OUT}/pi1x_models.npz"
    frozen["pi1x"]["model_file_sha256"] = save_models(out / "pi1x_models.npz", models)
    if questions is not None:
        from lookahead.c1_rho import measure_rho
        from lookahead.ouro_adapter import load_ouro

        model, _ = load_ouro()
        frozen["rho"] = measure_rho(model, questions, models)
    else:
        frozen["rho"] = "not measured (synthetic dry run)"
    frozen["provenance"] = {"inputs": inputs, "code": {f: _sha(root / f) for f in CODE}, "started_utc": started,
                            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out / "frozen.json").write_text(json.dumps(frozen, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8",
                                     newline="\n")
    p = frozen["primary"]
    for name in ("pi1-x", "native-pb"):
        print(f"{name:10s} " + "  ".join(f"{b}: m={s['m']} tau={s['tau']:.4g} sav={s['saving']:.3f} UB={s['upper_bound']:.2f}"
                                         for b, s in p[name].items()))
    print(f"development selection macro saving: pi1-x {frozen['selection_macro_saving_dev']['pi1-x']:.4f}  "
          f"native-pb {frozen['selection_macro_saving_dev']['native-pb']:.4f}  (in-sample; not an estimate of TEST)")
    print(f"expected 95% half-width at TEST sizes: {frozen['expected_precision']['expected_half_width_95']:.4f}")
    if isinstance(frozen["rho"], dict):
        print("rho: " + "  ".join(f"{b}: {v['rho']:.4f} ({v['probe_ms']:.2f} / {v['loop_ms']:.2f} ms)" for b, v in frozen["rho"].items()))
    print(f"frozen: {out / 'frozen.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
