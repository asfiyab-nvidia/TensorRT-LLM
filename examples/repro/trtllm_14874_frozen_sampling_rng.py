"""Repro: EAGLE3 one-model non-greedy sampling is captured as argmax.

Under CUDA graphs, the advanced-sampling graph for one-engine speculative
decoding ends up holding the argmax body. Non-greedy requests select that graph
by key and are then decoded greedily, and because no sampling kernel runs, the
spec worker's RNG counter never advances on replay.

Add TRTLLM_REPRO_14874=1 to also print the in-tree hooks:

    TRTLLM_REPRO_14874=1 python3 examples/repro/trtllm_14874_frozen_sampling_rng.py \
        --model /home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct \
        --draft-model /home/scratch.trt_llm_data/llm-models/EAGLE3-LLaMA3.1-Instruct-8B

  [repro-14874] teardown cleared=N ...          the flag wipe that causes it
  [repro-14874] key key_greedy=False live_spec_metadata.is_all_greedy_sample=... seed=...
                                                 the CUDA graph key context
  [repro-14874] branch=argmax spec_metadata.is_all_greedy_sample=...
                                                 the actual sample/argmax
                                                 decision in
                                                 _sample_tokens_for_batch

Expected on this branch: during the advanced-sampling capture pass, a
`teardown cleared=N` line is immediately followed by `branch=argmax` lines
(instead of `branch=sample`) for the key's non-greedy graph, then a frozen
`seed` across every non-greedy generation forward at serving time.
"""

import argparse

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import Eagle3DecodingConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig

PROMPT = "Once upon a time, in a kingdom far away,"
REPEATS = 4
GREEDY = SamplingParams(temperature=0, max_tokens=48)
NON_GREEDY = SamplingParams(temperature=0.9, top_k=30, top_p=0.95, max_tokens=48)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Target model (e.g. Llama-3.1-8B-Instruct)")
    parser.add_argument("--draft-model", required=True, help="EAGLE3 draft model")
    parser.add_argument("--max-draft-len", type=int, default=4)
    args = parser.parse_args()

    spec_config = Eagle3DecodingConfig(
        max_draft_len=args.max_draft_len,
        speculative_model=args.draft_model,
        eagle3_one_model=True,
    )
    with LLM(
        model=args.model,
        speculative_config=spec_config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
        cuda_graph_config=CudaGraphConfig(),
    ) as llm:
        greedy_text = llm.generate([PROMPT], GREEDY)[0].outputs[0].text
        texts = [llm.generate([PROMPT], NON_GREEDY)[0].outputs[0].text for _ in range(REPEATS)]

    print("\n" + "=" * 72)
    print(f"Prompt: {PROMPT!r}")
    print(f"Greedy (T=0): {greedy_text!r}")
    print("-" * 72)
    for i, text in enumerate(texts):
        print(f"[non-greedy rep {i}] {text!r}")
    print("=" * 72)


if __name__ == "__main__":
    main()
