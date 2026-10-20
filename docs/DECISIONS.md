# Decision log

| Date | Decision |
| --- | --- |
| 2026-10-20 | Protocol v1.1 adopted after four external reviews (research critique, design audit, implementation plan, explainer). |
| 2026-10-20 | Gate thresholds in PROTOCOL.md confirmed and frozen before any data is extracted. Any later change is logged here with a reason. |
| 2026-10-20 | Pinned transformers==4.54.1 (huggingface_hub<1.0). Ouro's model card says use transformers<4.56 and recommends 4.54.1; transformers 5.20.2 failed to build the model (`OuroConfig` has no `pad_token_id`). |
| 2026-10-20 | Pinned Ouro-1.4B to commit 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. |
| 2026-10-20 | Finding: Ouro's stock forward always runs all `total_ut_steps` loops and only then selects which loop's output to read, so it cannot save compute. Real early exit needs the loop-by-loop adapter (`src/lookahead/ouro_adapter.py`). |
| 2026-10-20 | Gate 0 accuracy reference: Ouro paper Table 10, ARC-C 25-shot, loops 1-4 = 37.63 / 54.86 / 59.47 / 60.92, evaluated with lm-eval-harness (acc vs acc_norm not stated). Retrieved through a summarising fetch, so unverified until our run lands near it. |
| 2026-10-20 | Clarification, not a threshold change: the Gate 0 "within 2 points" accuracy check is judged on the full ARC-C test set (n=1172, standard error about 1.4 points). A 200-question subset (about 3.5 points) is only a pipeline smoke test. |
| 2026-10-20 | Real-weights check: the adapter's per-loop hidden states and exit gates equal Ouro's own forward exactly (diff 0.0, all loops, 3 prompts); answer-position logits differ by at most 0.0625 (bf16 rounding from projecting one position instead of all). |
| 2026-10-20 | Finding: Ouro's stock KV-cache path is broken on transformers 4.54.1 (`UniversalTransformerCache.key_cache` has no setter), so stock `generate` only works with use_cache=False. Nothing here needs the cache. |
| 2026-10-20 | Decision: all extraction and halting runs use batch size 1, no padding. A padded bf16 batch differed from single-sequence logits by up to 0.69 (top answer unchanged), and batching was slower per sequence anyway (1.02 s vs 0.94 s at 1536 tokens, all 4 loops). The padding check no longer gates. |
| 2026-10-20 | Measured on the RTX 5050: one 1536-token sequence, all 4 loops, 0.94 s and 2.76 GB peak VRAM (2.67 GB weights). Plenty of headroom; speed is the limit, not memory. |
| 2026-10-20 | The ARC-C reproduction runs through lm-evaluation-harness 0.4.13 (the paper's framework). The harness builds prompts, few-shot examples (seed 1234) and metrics; `LoopedOuroLM` only supplies log-likelihoods, scoring all 4 loops in one cached pass at batch size 1. One difference from its HF backend: log-softmax in float32 instead of the model dtype. |
| 2026-10-20 | Gate 0 accuracy rule: the paper does not say whether its figures are acc or acc_norm, so both are reported, and the check passes if either metric is within 2 points of the paper at all four loops. That is a choice between two metrics, noted here so it is not mistaken for a single pre-specified test. |
| 2026-10-20 | Gate 0 passed. ARC-C 25-shot acc_norm 38.74 / 55.12 / 59.56 / 59.81 vs paper 37.63 / 54.86 / 59.47 / 60.92, all within 2 points (acc alone is not). Adapter exact on hidden states and gates; cost recorded. See docs/GATE0_REPORT.md. Phase 1 may start. |
| 2026-10-20 | Finding: stock `output_hidden_states=True` raises a TypeError in Ouro's remote code on 4.54.1. Not needed; the adapter reads hidden states directly. |
| 2026-10-20 | Gate 0 status amended to "accepted provisionally" after review, with three open qualifications recorded in GATE0_REPORT.md: (1) which metric the paper reports is unresolved; (2) the aggregate +0.26 at 3→4 hides 53 changes in correctness (28 W→C, 25 C→W), so Gate 1 measures transitions in both directions; (3) the stock `output_hidden_states` failure is diagnosed (the `@check_model_inputs` wrapper on a tuple-returning forward) and is a reproducibility limitation. |
| 2026-10-20 | Phase 1: design approved; full extraction on HOLD until docs/PHASE1_PLAN.md is reviewed and frozen. |
