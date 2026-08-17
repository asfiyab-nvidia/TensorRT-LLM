# TRTLLM-14874 — EAGLE3 one-model non-greedy sampling is captured as argmax

## Symptom

With EAGLE3 one-model speculative decoding and CUDA graphs enabled, non-greedy
requests decode **greedily**. Repeating an identical `temperature=0.9` request
against one live `LLM` returns the same completion every time, equal to the
greedy result.

## Root cause

`_capture_generation_cuda_graphs` runs across two warmup rounds (grow-workspace
pass, then real capture), and the advanced-sampling pass sets
`_force_non_greedy_for_capture` on the **live** `spec_metadata` to decide the
graph **key**. The graph **body**, however, is decided by a cached **copy** of
`spec_metadata` that `populate_sampling_params_for_one_model` re-scans every
iteration:

1. Round 1 — no copy exists yet, so the copy inherits the flag from the live
   object. Body is non-greedy. Correct.
2. End of round 1 — `clear_capture_only_spec_state()` wipes the flag from all
   cached copies (`teardown cleared=34`).
3. Round 2 (the real capture) — copies are reused, now flagless. The live
   object still carries the flag, so the key is still `is_all_greedy_sample=False`,
   but the flagless copy re-scans the parameter-less dummy requests and comes
   out **greedy**. `_sample_tokens_for_batch` takes the argmax branch, and
   **argmax gets recorded under the advanced key** (`teardown cleared=0`).
4. Serving — a non-greedy request selects the advanced key and replays the
   argmax graph. No `sample_from_logits_op`, no `self.seed += 1`.
