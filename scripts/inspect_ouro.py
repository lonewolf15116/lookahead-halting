"""Download Ouro (weights and its own modeling code) into hf_cache/ and record what is there.

Needs no GPU. The files stay readable in the repo folder, so the adapter can be written against
the real source rather than a description of it. Writes results/gate0/ouro_inspect.json.
"""
import json
import sys

from huggingface_hub import HfApi, snapshot_download

from lookahead.config import REPO_ROOT, load_config, results_dir

KEYWORDS = ("total_ut_steps", "early_exit", "hidden_states_list", "gate_list", "exit_threshold", "ut_steps")


def main() -> int:
    cfg = load_config("model")
    sha = HfApi().model_info(cfg["repo_id"], revision=cfg["revision"]).sha
    print(f"{cfg['repo_id']} resolves to commit {sha}")
    local = REPO_ROOT / cfg["local_dir"]
    snapshot_download(cfg["repo_id"], revision=sha, local_dir=local)

    files = {
        p.relative_to(local).as_posix(): p.stat().st_size
        for p in sorted(local.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }
    hits = {}
    for name in files:
        if name.endswith(".py"):
            lines = (local / name).read_text(encoding="utf-8").splitlines()
            found = [f"{i}: {line.strip()[:200]}" for i, line in enumerate(lines, 1)
                     if any(k in line for k in KEYWORDS)]
            hits[name] = found[:80]

    config_path = local / "config.json"
    report = {
        "repo_id": cfg["repo_id"],
        "resolved_commit": sha,
        "total_gb": round(sum(files.values()) / 1024**3, 2),
        "files": files,
        "config_json": json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None,
        "keyword_hits": hits,
    }
    out = results_dir("gate0") / "ouro_inspect.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(files)} files, {report['total_gb']} GB. Wrote {out}")
    print(f"Pin this in configs/model.yaml: revision: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
