# Qwen3.8-27B for ARC Prize 2026 (ARC-AGI-2): public resources sweep (2026-10-17)

**Method:**
- Kaggle kernel search, about 230 competition notebooks matching qwen, qwen3.8 or q38. Notebooks were pulled to `D:\ARC AGI 2\_scratch\qwen38_search\pull\`.
- Web search, plus arcprize.org (see `arcprize-qwen38-effort-scores` in memory).
- The Kaggle discussion API returned 403 for our token, and the discussion pages render with JS, so the threads themselves were not read.

## New Qwen3.8-27B notebooks (not analysed before)

| Notebook | What it does | Score |
|---|---|---|
| `khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning` ("Discovery by Accumulation of Observation: Community Edition", 11 votes, run 2026-10-10) | Multi-turn **agent with a Python workspace**: it records observations, tests candidate laws on every train pair, revises, then commits. **2 independent branches per task** give attempt_1 and attempt_2. 8 tasks x 2 branches = 16 concurrent. Settings: **T 0.9, top_p 0.95, presence_penalty 1.5**, `preserve_thinking: True`, context 262,144, up to 65,536 output per turn, up to 256 turns, 4 h per task. At 80 % of context it **compresses** the conversation into a summary of up to 32k tokens. vLLM TP4, MTP-3, fp8 KV, TRITON_ATTN, util 0.96, solving stops at 11.5 h. Model: `mikedan7/qwen3-8-27b-fp8-official`. | No score in the notebook. It calls itself a smaller "community edition" of a full exploration notebook. A team "Nguyen" sits at LB 50.14, but **no link to this author is proven**. |
| `yjsh2003/hyperqwen-v7-fast-arc-submission-20261009` | "HyperQwen v7": **Qwen3.8 W4A16 (INT4) + MTP-3**, TP4, concurrency 4, fp8 KV, 131,072 context, **60K / xhigh**. All code sits in a private base64 dataset; the public wrapper is only 1.7k characters. | Unknown |
| `dynamo14324/arc-agi-2-q38-taaf-baseline` | Tufa Labs TAAF harness for **ARC-AGI-3**, with a Qwen3.8 FP8 snapshot. | Not relevant to ARC-AGI-2 |

## Known before

- **delongyang:** medium effort, direct prompting, temperature 0. Its timing was measured on the placeholder file, and its submission format looks wrong.
- **madarshbb q38 direct b98k c16:** LB 21.5, then 13 on a rerun.

## Related (not Qwen3.8)

- **`johntaylorai/swarm-arc2-budget-curve` (NVARC LB 33.89 baseline):** 20–41 % of hidden tasks are never attempted, with a 1,200 s cap against 705 s available per task. NVARC also has a coverage problem, which matters for any NVARC + Qwen hybrid.
- **arXiv 2606.31543 "Modality-Driven Search with Holistic Trace Judging":** 72.9 % on semi-private at $39 per task. It searches across text, image and code candidates with a long-context judge. No Qwen; code at github.com/beetree/ARC-AGI.

## Token-usage resources

- **ARC Prize (verified), v2 semi-private:** xhigh 42.4 %, medium 13.2 %, low 22.8 %. From cost I estimate xhigh at about 60k output tokens per request.
- **insiderllm, HumanEval, xhigh (llama.cpp):** median 790 thinking tokens, mean 2,515, p99 30k, with a 320–540x spread. The top 10 problems produce 45 % of all output, a heavy tail like our cap hits. Medium injects no instruction, and "high" is silently mapped to xhigh.
- **Terminal-Bench 2.1:** about 45.6k / 46.8k / 63.5k output tokens for low / medium / xhigh.

## Ideas to consider for us (not decided)

1. **presence_penalty 1.5:** Qwen's recommended range against repetition loops. Our runs use 0, and hit the cap or loop on hard tasks. It is a sampling change, so measure it first.
2. **Agentic multi-turn with code execution plus context compression:** an alternative to Poetiq single-shot programs. We are already on the Exp4/Exp5 agent track; the compression step is new.
3. **INT4 W4A16 + MTP at 60K xhigh:** another team runs this, which confirms the E11 direction.
