"""An lm-evaluation-harness model that scores Ouro at every loop in a single pass.

The Ouro paper evaluated with lm-evaluation-harness, so the harness builds the prompts, the few-shot examples and the
metrics, and this class only supplies log-likelihoods. Tokenisation comes from the harness's own TemplateLM, so
context/continuation splitting matches its Hugging Face backend.

Every (context, continuation) request is run through all loops once (batch size 1, no padding) and the per-loop results
are cached on disk, so evaluating loops 1-4 costs one pass, and an interrupted run resumes where it stopped.
`exit_loop` (1-based) picks which loop's scores the harness sees.

One difference from the harness's HF backend: log-softmax is taken in float32, where the harness uses the model dtype.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from lm_eval.api.model import TemplateLM

from lookahead.ouro_adapter import OuroLoopRunner


class LoopedOuroLM(TemplateLM):
    def __init__(self, model, tokenizer, cache_path=None, max_length=4096, log_every=100):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.runner = OuroLoopRunner(model)
        self.exit_loop = self.runner.total_loops
        self.max_length = max_length
        self.log_every = log_every
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.newly_scored = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["k"]] = (row["ll"], row["g"])

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def tokenizer_name(self) -> str:
        return str(self.tokenizer.name_or_path)

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        kw = {} if add_special_tokens is None else {"add_special_tokens": add_special_tokens}
        # As in the harness: do not add a BOS if the string already starts with one.
        if add_special_tokens is None and string.startswith(self.tokenizer.decode([self.prefix_token_id])):
            kw["add_special_tokens"] = False
        return self.tokenizer.encode(string, **kw)

    @torch.no_grad()
    def score_all_loops(self, context_enc, continuation_enc):
        """Log-likelihood of the continuation, and whether it is the greedy continuation, at every loop."""
        ids = (context_enc + continuation_enc)[-(self.max_length + 1):]
        inp = ids[:-1]
        cont = len(continuation_enc)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([inp], device=device)
        positions = torch.arange(len(inp) - cont, len(inp), device=device)[None]  # the positions that predict the continuation
        targets = torch.tensor(continuation_enc, device=device)
        state = self.runner.start(input_ids)
        lls, greedy = [], []
        for _ in range(self.runner.total_loops):
            self.runner.step(state)
            logp = torch.log_softmax(self.runner.readout(state.hidden, positions)[0], dim=-1)  # [cont, vocab], float32
            lls.append(float(logp.gather(1, targets[:, None]).sum()))
            greedy.append(bool((logp.argmax(-1) == targets).all()))
        return lls, greedy

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        out = []
        started = time.time()
        for (context, continuation), context_enc, continuation_enc in requests:
            key = hashlib.sha1(f"{context}\x00{continuation}".encode("utf-8")).hexdigest()
            if key not in self.cache:
                self.cache[key] = self.score_all_loops(context_enc, continuation_enc)
                self.newly_scored += 1
                if self.cache_path:
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": key, "ll": self.cache[key][0], "g": self.cache[key][1]}) + "\n")
                if self.newly_scored % self.log_every == 0:
                    rate = (time.time() - started) / max(1, self.newly_scored)
                    print(f"  scored {self.newly_scored} new requests ({rate:.2f} s each)", flush=True)
            ll, greedy = self.cache[key]
            out.append((ll[self.exit_loop - 1], greedy[self.exit_loop - 1]))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("only multiple-choice log-likelihood is needed")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("only multiple-choice log-likelihood is needed")
