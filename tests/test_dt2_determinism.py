import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dt2", ROOT / "scripts" / "dt2_determinism_check.py")
dt2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt2)


def _trace(opt_shift=0.0):
    rows = [{"opt_ll": [-1.0 + opt_shift, -2.0], "conf": 0.7, "ent": 0.6, "vent": 3.0, "gate": 0.1, "cdf": 0.2 * t}
            for t in range(4)]
    return SimpleNamespace(rows=rows)


def test_bitwise_identical_and_one_ulp():
    a = dt2.values(_trace())
    assert not any(v["differs"] for v in dt2.bitwise_diff(a, dt2.values(_trace())).values())
    b = dt2.values(_trace(np.spacing(1.0)))
    d = dt2.bitwise_diff(a, b)
    assert d["opt_ll"]["differs"] and d["opt_ll"]["max_abs"] > 0
    assert not d["gate"]["differs"]


def test_mode_env_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(dt2, "ROOT", tmp_path)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert dt2.run_mode("deterministic") == 2                      # env must be set before the process starts
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert dt2.run_mode("default") == 2                            # default must run without it


def test_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(dt2, "ROOT", tmp_path)
    (tmp_path / "results/diagnostics").mkdir(parents=True)
    (tmp_path / dt2.OUT["default"]).write_text("{}")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert dt2.run_mode("default") == 2


@pytest.mark.parametrize("d_def,d_det,expect", [(3, 0, "supports GPU nondeterminism"), (3, 2, "did not eliminate"),
                                                (0, 0, "NOT ruled out"), (0, 1, "investigate configuration")])
def test_summary_readings(monkeypatch, tmp_path, d_def, d_det, expect):
    monkeypatch.setattr(dt2, "ROOT", tmp_path)
    (tmp_path / "results/diagnostics").mkdir(parents=True)
    for m, n in (("default", d_def), ("deterministic", d_det)):
        (tmp_path / dt2.OUT[m]).write_text(json.dumps({"run_error": None, "deterministic_setup_error": None,
                                                        "n_questions_completed": dt2.N_Q, "n_questions_repeats_differ": n,
                                                        "max_abs_by_key": {}, "deterministic_algorithms_enabled": m == "deterministic"}))
    assert dt2.summarize() == 0
    assert expect in json.loads((tmp_path / dt2.OUT["summary"]).read_text())["reading"]


def test_summary_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(dt2, "ROOT", tmp_path)
    (tmp_path / "results/diagnostics").mkdir(parents=True)
    for m in ("default", "deterministic"):
        (tmp_path / dt2.OUT[m]).write_text(json.dumps({"run_error": "RuntimeError: x" if m == "deterministic" else None,
                                                        "deterministic_setup_error": None, "n_questions_completed": 3,
                                                        "n_questions_repeats_differ": 0, "max_abs_by_key": {},
                                                        "deterministic_algorithms_enabled": True}))
    dt2.summarize()
    assert json.loads((tmp_path / dt2.OUT["summary"]).read_text())["reading"].startswith("INCOMPLETE")
