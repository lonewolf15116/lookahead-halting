"""Write data/synthetic/{modarith,parity}.jsonl and manifest.json."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from lookahead import synthetic as S
from lookahead.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "data" / "synthetic"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(
        str(REPO_ROOT / load_config("model")["local_dir"]), trust_remote_code=True)
    manifest = {"seeds": {t: {"eval": a, "exemplar": b} for t, (a, b) in S.SEEDS.items()},
                "n_shots": S.N_SHOTS, "benchmarks": {}}
    total_fail = 0
    for task, items in S.generate_all().items():
        path = OUT / f"{task}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it.to_dict()) + "\n")
        fails = S.check_token_boundaries(items, tok)
        total_fail += len(fails)
        levels = sorted({i.level for i in items})
        manifest["benchmarks"][task] = {
            "levels": levels,
            "counts_per_level": {l: sum(i.level == l for i in items) for l in levels},
            "answer_distribution": {l: dict(sorted(Counter(
                i.choices[i.target].strip() for i in items if i.level == l).items()))
                for l in levels},
            "majority_class_rate": {l: max(Counter(
                i.target for i in items if i.level == l).values()) / S.N_ITEMS
                for l in levels},
            "dedupe": S.DEDUPE[task],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "token_boundary_failures": len(fails),
        }
        for m in fails[:5]:
            print("FAIL", m)
    manifest["token_boundary_failures_total"] = total_fail
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
