# Lookahead Halting

Can a looped transformer forecast how much one more loop would change its answer, and does that forecast help it decide when to stop?

Working title: *Predicting the Value of Another Thought: Future-State Forecasting for Adaptive Depth in Looped Language Models.*

- Model: frozen Ouro-1.4B (4 trained loops), run on an RTX 5050 laptop (8 GB VRAM)
- Protocol: [docs/PROTOCOL.md](docs/PROTOCOL.md), v1.1, frozen 2026-10-20 before any extraction
- Decision log: [docs/DECISIONS.md](docs/DECISIONS.md)

## Status

Phase 0 done: **Gate 0 passed** on 2026-10-20 (ARC-C acc_norm within 2 points of the paper at all four loops; see docs/GATE0_REPORT.md). Next: Phase 1, extraction and the Gate 1 headroom report.

- Loop-by-loop adapter written and checked against Ouro's own code on a tiny random model (CPU). The check on the real weights is next.
- Gate 0 reference: Ouro paper Table 10, ARC-C 25-shot, loops 1-4 = 37.63 / 54.86 / 59.47 / 60.92 (see docs/DECISIONS.md).

## Finish Gate 0 on the laptop

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_arc_windows.ps1"
```

Runs the tests, re-records the probes, then ARC-Challenge 25-shot at loops 1-4 (smoke test, then the full test set, about an hour). Resumable.

## Run Phase 0 on the laptop

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\setup_windows.ps1"
```

Creates `.venv`, installs a CUDA 12.8 PyTorch build, runs the tests, downloads Ouro into `hf_cache/`, then records the environment, the model files and a stock-load smoke test in `results/gate0/`. Safe to re-run.

`trust_remote_code` is on because Ouro ships its own modeling code; the code is downloaded into `hf_cache/` and is readable there.
