# Gate 0 report: setup and reproduction

Date: 2026-10-20. Verdict: **accepted provisionally** (reviewer, 2026-10-20). The accuracy check passes on acc_norm, extraction is exact, and cost is recorded, subject to the three qualifications below. The reviewer accepted these results as reported and did not re-run them.

## Setup
RTX 5050 Laptop GPU (8 GB), torch 2.11.0+cu128, transformers 4.54.1, lm_eval 0.4.13, Ouro-1.4B at revision 574fa66, bfloat16, batch size 1.

## 1. Accuracy: ARC-Challenge, 25-shot, full test set (n = 1172)

| Loop | acc | acc_norm | Paper (Table 10) | acc_norm − paper |
|---|---|---|---|---|
| 1 | 37.03 | 38.74 | 37.63 | +1.11 |
| 2 | 51.54 | 55.12 | 54.86 | +0.26 |
| 3 | 55.80 | 59.56 | 59.47 | +0.09 |
| 4 | 55.97 | 59.81 | 60.92 | −1.11 |

Standard error is about 1.4 points per loop. acc_norm is within 2 points at every loop; acc is not (−3.3 to −4.9 from loop 2 on). Under the rule set before the run (either metric, all four loops), the check passes. Which metric the paper reports is unresolved (see Qualification 1).

### Loop-to-loop transitions (C = correct, W = wrong)

This is a descriptive count over all 1,172 questions. It sets no threshold. The 95% interval comes from a paired question-level bootstrap (2,000 resamples, seed 0).

| Metric | Step | C→C | C→W | W→C | W→W | Net (pts) | 95% CI |
|---|---|---|---|---|---|---|---|
| acc_norm | 1→2 | 399 | 55 | 247 | 471 | +16.38 | [+13.65, +19.20] |
| acc_norm | 2→3 | 607 | 39 | 91 | 435 | +4.44 | [+2.56, +6.40] |
| acc_norm | **3→4** | **673** | **25** | **28** | **446** | **+0.26** | **[−0.94, +1.45]** |
| acc | 3→4 | 623 | 31 | 33 | 485 | +0.17 | [−1.11, +1.54] |

The small 3→4 gain hides two-way traffic. On acc_norm, 53 questions (4.5%) change correctness: 28 are fixed and 25 are broken, which nearly cancels. On acc, the top answer changes on 77 questions at 3→4 (6.6%), compared with 179 at 2→3 and 381 at 1→2. So a fixed 3-loop budget is not a free lunch. It gives up 28 fixes to avoid 25 breaks. A halting rule can only gain at loop 4 by telling these two groups apart, which is the question Gate 2 tests. These ARC-C test questions return in Phase 1, so the counts above are a description of the data and are not used for any tuning.

## 2. Extraction leaves predictions unchanged
The loop-by-loop adapter matches stock Ouro exactly on hidden states and exit-gate values at all 4 loops (difference 0.0). Logits differ by at most 0.0625 in bf16, which comes from projecting one position instead of all of them. Twelve unit tests pass, including float32 equivalence and the lm_eval scorer against the stock forward at every loop.

## 3. Cost
| Batch × tokens | ms per loop | Peak VRAM |
|---|---|---|
| 1 × 512 | ~78 | 2.71 GB |
| 1 × 1536 | ~234 | 2.76 GB |
| 4 × 1536 | ~1020 | 2.98 GB |

Weights take 2.67 GB. The full ARC-C run (4,587 requests, all 4 loops) took 48 min, about 0.6 s per request. Speed is the limit, not memory.

## Qualifications (kept open)

1. **Metric ambiguity.** acc_norm matches the paper within 1.2 points at every loop, while acc is 3.3 to 4.9 points low from loop 2. That fits the paper reporting acc_norm, but it is not proof, and the paper does not say. Treat the question as unresolved. Gate 0 passes under the rule set before the run (either metric, all loops), and that rule is a choice between two metrics.

2. **Headroom.** The aggregate +0.26 at 3→4 says nothing on its own about how much one could save. The transitions above show changes in both directions, so Gate 1 measures all four transition categories at every step, on every benchmark. It does not rely on aggregate gains.

3. **Stock-model failure.** Calling the stock model with `output_hidden_states=True` raises an error on transformers 4.54.1:
   ```
   modeling_ouro.py:696  outputs, hidden_states_list, gate_list = self.model(...)
   transformers/utils/generic.py:1083 (check_model_inputs wrapper)
       outputs[key] = collected_outputs[key]
   TypeError: 'tuple' object does not support item assignment
   ```
   The cause is that `OuroModel.forward` is wrapped in `@check_model_inputs` (line 517) but returns a plain tuple (line 610). When hidden states are requested, the wrapper tries to write them into that tuple and fails. The workaround is that `src/lookahead/ouro_adapter.py` re-runs Ouro's loop itself and reads each loop's hidden state directly. It matches the stock model with a difference of 0.0 on hidden states and gate values at all 4 loops (`results/gate0/adapter_check.json`). That is strong evidence the adapter is correct. Even so, anyone reproducing this cannot get per-loop states from the stock API on this pinned version, so this is a reproducibility limitation.

## Other known issues
- The stock KV cache is broken on 4.54.1 (`UniversalTransformerCache.key_cache` has no setter), so all runs use `use_cache=False`.

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
