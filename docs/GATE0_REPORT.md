# Gate 0 report: setup and reproduction

Date: 2026-10-20. Verdict: **passed** (accuracy on acc_norm, extraction exact, cost recorded).

## Setup
RTX 5050 Laptop GPU (8 GB), torch 2.11.0+cu128, transformers 4.54.1, lm_eval 0.4.13, Ouro-1.4B at revision 574fa66, bfloat16, batch size 1.

## 1. Accuracy: ARC-Challenge, 25-shot, full test set (n = 1172)

| Loop | acc | acc_norm | Paper (Table 10) | acc_norm − paper |
|---|---|---|---|---|
| 1 | 37.03 | 38.74 | 37.63 | +1.11 |
| 2 | 51.54 | 55.12 | 54.86 | +0.26 |
| 3 | 55.80 | 59.56 | 59.47 | +0.09 |
| 4 | 55.97 | 59.81 | 60.92 | −1.11 |

Standard error is about 1.4 points per loop. acc_norm is within 2 points at every loop; acc is not (−3.3 to −4.9 from loop 2 on). Under the rule set before the run (either metric, all four loops), the check passes. acc_norm tracking the paper at all four loops suggests the paper reports acc_norm. This is not confirmed.

The loop curve is the one the project depends on: large gains at loops 1→2 and 2→3, then almost nothing at 3→4 (+0.25). That flat last step is where halting should save compute.

## 2. Extraction leaves predictions unchanged
The loop-by-loop adapter matches stock Ouro exactly on hidden states and exit-gate values at all 4 loops (difference 0.0). Logits differ by at most 0.0625 in bf16, which comes from projecting one position instead of all of them. Twelve unit tests pass, including float32 equivalence and the lm_eval scorer against the stock forward at every loop.

## 3. Cost
| Batch × tokens | ms per loop | Peak VRAM |
|---|---|---|
| 1 × 512 | ~78 | 2.71 GB |
| 1 × 1536 | ~234 | 2.76 GB |
| 4 × 1536 | ~1020 | 2.98 GB |

Weights take 2.67 GB. The full ARC-C run (4,587 requests, all 4 loops) took 48 min, about 0.6 s per request. Speed is the limit, not memory.

## Known issues (not blockers)
- Stock `output_hidden_states=True` raises `TypeError: 'tuple' object does not support item assignment` inside Ouro's remote code. The adapter does not use it.
- The stock KV cache is broken on 4.54.1, so all runs use `use_cache=False`.

## Files
`results/gate0/`:
- `environment.json`
- `ouro_inspect.json`
- `stock_smoke.json`
- `adapter_check.json`
- `gpu_profile.json`
- `ouro_reproduction.csv`
- `ouro_reproduction_limit25.csv`
- `gate0_accuracy.json`
- `arc_samples_loop{1..4}.jsonl`

The log-likelihood cache (`arc_cache.jsonl`) is gitignored.
