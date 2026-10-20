"""Repo paths and config loading."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    path = REPO_ROOT / "configs" / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(sub: str) -> Path:
    path = REPO_ROOT / "results" / sub
    path.mkdir(parents=True, exist_ok=True)
    return path
