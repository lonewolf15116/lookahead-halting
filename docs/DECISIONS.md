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
