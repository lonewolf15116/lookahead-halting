"""Phase 1 extraction (plan sections 4-5, 7): all loops of every question, packed scoring, resumable.

    python scripts/extract_phase1.py --limit 40      # small end-to-end run -> data/extracted_smoke/
    python scripts/extract_phase1.py                 # everything -> data/extracted/ (about 4 h on the RTX 5050)

Inputs are frozen: data/questions/*.jsonl.gz must reproduce the question-set sha256 in data/manifests/questions.json, and
data/manifests/splits.csv must match its recorded sha256. Re-running continues where it stopped; completed parts are
never recomputed. Run metadata (code commit, revisions, seeds, versions, runtime) goes to <out>/meta.json.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from lookahead.config import REPO_ROOT, load_config
from lookahead.extract import load_questions, load_splits, run_benchmark, sha256_file
from lookahead.questions import BENCHMARKS, SYNTHETIC, question_set_hash

ALL = list(BENCHMARKS) + list(SYNTHETIC)
SETS = ("phase1", "confirmation", "phase3_dev", "phase3_test")  # confirmation: Gate 2; phase3_*: A1 (phase3_test is sealed)


def code_identity(root: Path) -> dict:
    files = sorted((root / "src" / "lookahead").glob("*.py")) + [root / "scripts" / "extract_phase1.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    out = {"code_sha256": h.hexdigest()}
    try:
        out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        out["git_dirty_code"] = bool(dirty)
    except Exception as e:  # git may not be on PATH on the laptop; code_sha256 still identifies the code
        out["git_commit"] = f"unavailable ({type(e).__name__})"
    return out


def check_inputs(root: Path, benches, limit, which="phase1"):
    if which == "confirmation":
        from lookahead.confirm import check_confirmation_inputs

        return check_confirmation_inputs(root, benches, limit)
    if which.startswith("phase3_"):
        from lookahead.phase3_build import check_phase3_inputs

        return check_phase3_inputs(root, benches, limit, which.split("_", 1)[1])
    mdir = root / "data" / "manifests"
    manifest = json.loads((mdir / "questions.json").read_text(encoding="utf-8"))
    split_text_sha = sha256_file(mdir / "splits.csv")
    recorded = (mdir / "splits.sha256").read_text(encoding="utf-8").strip()
    # the manifest hash is over LF text; a CRLF checkout would differ, so compare on normalised bytes too
    normalised = hashlib.sha256((mdir / "splits.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if recorded not in (split_text_sha, normalised) or manifest["splits_sha256"] != recorded:
        raise SystemExit("splits.csv does not match its recorded sha256: inputs are not the frozen ones")
    splits = load_splits(mdir / "splits.csv")
    questions = {}
    for b in benches:
        qs = load_questions(root / "data" / "questions" / f"{b}.jsonl.gz")
        if question_set_hash(qs) != manifest["benchmarks"][b]["question_set_sha256"]:
            raise SystemExit(f"{b}: questions do not match the manifest hash; re-run scripts/prepare_phase1.py")
        n_split = sum(1 for (bb, _) in splits if bb == b)
        if n_split != len(qs):
            raise SystemExit(f"{b}: {len(qs)} questions but {n_split} split rows")
        questions[b] = qs[:limit] if limit else qs
    return manifest, recorded, splits, questions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", choices=SETS, default="phase1", help="question set (confirmation -> data/extracted_confirm)")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N questions per benchmark (smoke run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--stop-after", type=int, default=None, help="testing: stop after N questions, mid-part")
    args = ap.parse_args()

    root = REPO_ROOT
    conf = args.which == "confirmation"
    p3 = args.which.startswith("phase3_")
    if p3:
        from lookahead.phase3_build import MANIFEST, XDIR

        role = args.which.split("_", 1)[1]
        p3_sets = json.loads((root / MANIFEST).read_text(encoding="utf-8"))["sets"]
        args.benchmarks = args.benchmarks or sorted(k.split("/", 1)[1] for k in p3_sets if k.startswith(role + "/"))
        default_out = XDIR[role] + ("_smoke" if args.limit else "")
    else:
        args.benchmarks = args.benchmarks or (list(BENCHMARKS) if conf else ALL)
        default_out = ("extracted_confirm_smoke" if args.limit else "extracted_confirm") if conf else (
            "extracted_smoke" if args.limit else "extracted")
    out = Path(args.out) if args.out else root / "data" / default_out
    out.mkdir(parents=True, exist_ok=True)
    manifest, split_sha, splits, questions = check_inputs(root, args.benchmarks, args.limit, args.which)

    from lookahead.ouro_adapter import load_ouro
    from lookahead.packed import PackedScorer

    model, _ = load_ouro()
    scorer = PackedScorer(model)
    cfg = load_config("model")
    import transformers

    run = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), **code_identity(root),
        "model": {"repo_id": cfg["repo_id"], "revision": cfg["revision"], "dtype": cfg["dtype"]},
        "datasets": {b: {k: manifest["benchmarks"][b].get(k) for k in ("hf_dataset", "hf_revision", "question_set_sha256",
                                                                        "shots", "metric")} for b in args.benchmarks},
        "splits_sha256": split_sha,
        "seeds": {"fewshot": 1234, "split_salt": "lh-split-v1", "synthetic": {"modarith": [20261020, 20261021],
                                                                            "parity": [20261022, 20261023]}},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "chunk": args.chunk, "benchmarks": {},
    }
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"runs": []}
    meta["runs"].append(run)

    def save_meta():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_meta()
    remaining = args.stop_after
    finished_all = True
    for b in args.benchmarks:
        t0 = time.time()
        computed, finished = run_benchmark(scorer, b, questions[b], splits, out, chunk=args.chunk, stop_after=remaining)
        run["benchmarks"][b] = {"computed": computed, "finished": finished, "minutes": round((time.time() - t0) / 60, 2)}
        save_meta()
        print(f"{b}: {computed} questions computed this run, {'complete' if finished else 'INCOMPLETE'} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if remaining is not None:
            remaining -= computed
            if not finished:
                finished_all = False
                break
    run["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
    save_meta()
    return 0 if finished_all else 3


if __name__ == "__main__":
    sys.exit(main())
