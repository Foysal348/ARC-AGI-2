I wrote the design doc, a pure-Python prototype of the checkpoint logic, and 11 CPU tests, which all pass. I did not write a Merlin diff: none of the three workable options is small enough to call clearly correct without GPU gates. This was CPU and source work only: no GPU, Kaggle, git or push.

Files:
- Design doc: `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Review/DESIGN_generated_tail_checkpoint_2026-11-01.md` (the write was not blocked)
- `D:/ARC AGI 2/_scratch/gentail_ckpt/gentail_logic.py`: position math, the PP fence and producer rules, a fence queue, and a toy model of PP0's commit ring. It does not import vLLM.
- `D:/ARC AGI 2/_scratch/gentail_ckpt/test_gentail_logic.py`: run with `python -I test_gentail_logic.py`.

Paths below are relative to `_scratch/e23/p2/merlin-src/vllm/`.

## Headline numbers

I replayed all 677 K100Q later turns through each option. The perfect saving is 2,430,527 tokens.

| Option | What it is | Tokens saved | Share | Run value | Change needed |
|---|---|---|---|---|---|
| B | publish the 4096-boundary state the decode commit already writes | 1,702,336 (309 turns helped) | 70.0 % | about 4.2-5.7 % | scheduler and KV manager only, no kernel change |
| C | ring of R tail pages, one state saved at every 64-token crossing | 2,429,440 | 99.96 % with R=3 | about 6.0-8.2 % | B plus 4 Triton kernels |
| D | park the finished turn as a resumable session | 2,494,563 | 102.6 % (also removes the 64-127 tail) | about 6.2-8.4 % | API routing plus park, timeout and evict logic |

Ring coverage in simulation: R=3 covers 99.9 % of finishes, R=2 covers 96.9 %.

## Q1. Why the hit stops 64-127 tokens before the previous prompt's end

Both groups stop it, but the Mamba group is the one that binds.

- **Generated tokens are hashed.** Every output token updates the 64-token hash chain (`v1/request.py:272-288`, `v1/core/kv_cache_utils.py:942-1001`).
- **Attention blocks of generated tokens are hashed, but only whole 4096 blocks.** This confirms that part of your premise.
  - The per-step caching path is `kv_cache_manager.py:703-707` → `kv_cache_coordinator.py:376-399` → `single_type_kv_cache_manager.py:496-553` → `block_pool.py:229-302`.
  - The sub-block key is registered only at the prompt tail, floor(P/64)·64 (`single_type_kv_cache_manager.py:889-924`).
  - So over generated tokens, the attention hit ends at the last full generated block (lookup at `:822-868`).
- **The Mamba group never publishes past the prompt.**
  - Under RecoverSSM + PP, `publish_decode_checkpoints` is set to False (`sched/scheduler.py:88-110`, called at `:383-391`).
  - `MambaManager.cache_blocks` then caps at `num_prompt_tokens − 1` (`single_type_kv_cache_manager.py:2323-2330`).
  - Even without PP, retention interval 0 keeps only prompt-derived boundaries (`:1868-1925`, `kv_cache_coordinator.py:67-88`). The partial tail is keyed only at the prompt boundary (`:2453-2486`).
- **Where the 64-127 comes from.**
  - The prompt checkpoint sits at floor(P/64)·64 − 64 (`v1/kv_cache_interface.py:1111-1123`). The eagle drop is on for MTP (`kv_cache_coordinator.py:186-190`).
  - The combined lookup drops one unit from the attention hit, then lets Mamba search only up to that (`kv_cache_coordinator.py:827-962`, `single_type_kv_cache_manager.py:1813-1839`).
- **Proof:** this formula reproduces `cached_tokens` exactly in 677 of 677 turns.

## Q2. The final SSM state at request finish

- **The final running state cannot be published as is.**
  - It sits in the block of the last committed token (`model_executor/layers/mamba/ops/recoverssm.py:80-92`; `v1/worker/gpu/model_states/recoverssm.py:200-212`).
  - Its position S is an arbitrary number, not a multiple of 64.
- **The block-floored state already exists.**
  - When a verify window crosses a multiple of 4096, the commit writes the exact state at the boundary (`ops/recoverssm.py:93-102`).
  - It is stored by `gdn/recoverssm_gdn.py:436-442`, for conv state at `:657-676`, and for PLE per `ple_recoverssm.py:10`.
  - It is then retired unhashed and freed (`single_type_kv_cache_manager.py:1673-1702, 1947-1978`).
- **MTP lookahead and the reprefillable tokens.**
  - Reprefillable tokens = lookahead − 1, which is 2 for multi-module MTP K3 (`kv_cache_coordinator.py:119`, `config/vllm.py:674-694`). The eagle drop removes one more 64-unit on top.
  - Stop truncation trims output tokens but not `num_computed_tokens` (`scheduler.py:2800-2817`). The committed state can therefore include up to 2 discarded draft tokens.
  - So the valid prefix is min(num_computed, num_tokens) − reprefillable, and option C's target is floor(valid/64)·64 − 64.
- **Commit timing.** The verify forward does not advance the state; only the commit after sampling does (`recoverssm_gdn.py:1-15`, `model_states/recoverssm.py:114-193`). On PP1 this happens in the same step.
- **Under PP2, PP0 commits step T only at its call T+2.**
  - This is the ring of depth 2 (`v1/worker/gpu/pp_utils.py:81-90, 130-160`).
  - Inside that call, the commit runs before the frees, the copy-on-write copies and the forward (`v1/worker/gpu/model_runner.py:2196-2201, 1491-1497`).
  - The commit is skipped if the request's slot was freed first (`pp_utils.py:142-147`).
- **On an idle engine, PP0 loses the commit of the request's last step.** The finish can arrive at call L+1, before that commit runs. The toy-model tests show both the busy case and this idle case.
- **Rule 1 (producer):** publish only states produced by a step strictly before the request's last scheduled step.
- **Rule 2 (fence):** publish only once `processed_step_seq ≥ T + pp_size`.
  - The step counters advance only for non-empty steps (`scheduler.py:1845-1847, 2352-2355`).
  - They only advance at all while `defer_block_free` is on (`:224-233`). The patch should assert this.
- **Publish at finish only, not while decoding.** A rolling publish during decode would also need a "not freed before T+2" check.

## Q3. The minimal correct design

**Option B, stage 1:**
- Record the newest grid crossing per Mamba group after the rejection adjustment: position b, its block, and step T.
- Hold that block through retirement, reusing the existing one-checkpoint-per-request hold. This also stops PP0's late write from landing in a reused block.
- At finish, apply rule 1 and check b ≤ floor(valid/64)·64 − 64.
- At finish, register an attention partial key at A = floor(valid/64)·64. This is required: otherwise the eagle drop discards b.
- Queue the Mamba key behind the fence and drain it next to `_drain_deferred_frees` (`scheduler.py:3097-3111`).
- After the fence, hand the block to the grace pin.
- Readers need no change.
- **Memory:** about zero extra pages in steady state; the newer state replaces the prompt-tail hold (`:1693-1700`).
- **Eviction:** use the grace pin (`kv_cache_manager.py:763-835`), which holds the block until the next turn is admitted and gives it up first under pressure. Do not use evict-first junctions: at the 99.4 % pool peak they would be gone within the 0.3-0.5 s tool gap.
- Raise `MAX_GRACE_PINS` from 16 to 21 (`kv_cache_manager.py:735`), since 21 workers can be between turns at once.

**Option C, stage 2:**
- Same as B, but the Mamba entry is a ring page.
- The plan kernel gets a second "64-unit crossing" target. A window is at most 4 tokens, so there is at most one crossing per window.
- The GDN state kernel gets a second masked store; conv compaction and the PLE window get the same second destination.
- The reader uses the existing prompt-tail partial-hit / copy-on-write path.
- **Memory:** R × 3 groups pool blocks per running request. At R=3 that is 9 per stream, about 189 at 21 streams.

## Q4. The four gates

1. **Token-ID identity.** Add `return_token_ids` (chat `protocol.py:423`) to any Duck run, with no engine change. Check prompt_ids_k[:P+G−1] == prompt_ids_{k−1} + output_ids_{k−1}[:−1] and record where the first mismatch falls. Only tokens at least 66 before the end must match. A mismatch costs gain, never correctness, because the chained hash guarantees identical IDs on any hit.
2. **Checkpoint position == prefix length.** `cached_tokens` per turn must equal the predicted hit from the prototype functions.
3. **KV + state equivalence.** In an in-process `LLM`, compare the published blocks against a cold prefill at the same position under another `cache_salt`, using `collective_rpc` on all 4 workers. This covers GDN state, conv state, the PLE window and the attention KV. Also check that the first 64 greedy warm and cold tokens are identical.
4. **PP0 / PP1 fence.** Run busy (2 filler streams) and idle (single stream) variants, plus an abort right after a grid crossing. Add a negative control that switches the fence off; it must show a PP0 state difference in the idle case.

## Q5. Client-directed alternative

- **Per-request flag:** `session_id` (`protocol.py:406` → `request.py:197`) and `vllm_xargs` (`protocol.py:511, 746`) already reach the engine, so a flag is about 10 lines. It does not make B or C simpler: the hard part, an aligned state at the end of decode under PP, is inside the engine. Duck also cannot know in advance which turn is its last.
- **`kv_hints`:** only the offload connector reads it, and the chat API does not expose it.
- **Option D, session park, is the real client path.** The resumable streaming-input machinery already exists (`scheduler.py:2772-2798, 1957-1997, 2924-2943`).
  - The worker slot stays live, so PP0's commit is never skipped. The state at S is exact, so no alignment, ring or fence is needed.
  - Today it is reachable only through `AsyncLLM.generate` and the realtime speech websocket (`async_llm.py:580-692`), and it rejects stop strings.
  - It needs routing in the chat server, the output processor finishing the turn while the engine request stays parked, a timeout and pressure eviction.
  - It needs a guard against the stop-as-accepted-draft case.
- **Token identity by construction:** Duck could send token-ID prompts (previous prompt IDs + output IDs + new IDs) to `/v1/completions`. That guarantees gate 1 for B, C or D, at the cost of parsing tool calls client-side.
- **Upstream:**
  - RFC #55697 (open) uses a marker for prompt positions only, so it does not fit generated tokens.
  - PR #52789 (merged) adds internal prefill checkpoints; at merge it was FlashKDA-only with spec decoding excluded.
  - Draft PR #53803 is the same idea as option B. Its author reports the decode-end part "showed no effect" on their append benchmark, it has no PP fence, and its grid only exists at chunk ends. In Merlin the commit itself writes the grid state.

## Q6. Effort, risk and the cheapest GPU test

**Effort and risk:**
- **B:** 1-2 days plus 1 GPU commit, medium risk. Main risks are holding the grid block through retirement and the grace-pin interaction.
- **C:** 3-5 more days, high risk. Main risks are the kernels on both PP ranks and pool pressure.
- **D:** 2-4 days, medium-high risk. The streaming path has never run with Mamba align + RecoverSSM + MTP + PP.

**Cheapest GPU test:** one Kaggle GPU commit of about 30-40 minutes with fail-fast gates. It needs your go.
1. Replay one recorded 10-turn K100Q chain, greedy, including at least 3 turns with generations over 4k tokens.
2. Run a cold pass (fresh `cache_salt` per turn) and a warm pass (shared salt), both with `return_token_ids` and logprobs.
3. Gates 1 and 2 come from the API, gate 3 from `collective_rpc` block reads, gate 4 from the busy/idle/abort variants.

## Not verified yet (listed in the doc)

- That `defer_block_free` is on in the production config. If not, the fence needs its own counter.
- Whether today's code can retire and reuse a decode grid block before PP0's late boundary write lands in it (retirement runs on the last-rank count, `kv_cache_manager.py:590-596`). Option B removes this for the held block; gate 3 on PP0 checks it.
- Whether the klee100 MTP is multi-module (2 reprefillable tokens, else 0). It moves option C's target by at most one unit.
- Grace pins only hold hashed blocks (`kv_cache_manager.py:774`), so B must pin the block after the fence, not at free.