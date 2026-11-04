# GPU test spec: GENTAIL_CKPT (Option B) prototype, one fail-fast Kaggle GPU commit (~30-40 min)

- **Status:** spec only. Nothing here has run. A GPU commit needs the user's explicit go.
- **Patch under test:** `gentail_seam.py` (E30), applied to the Merlin v3.0.0 tree the payload installs (commit ad42e37, tree 07c16a79). The seam's pre-shas must match, or the notebook stops.
- **Config:** the K100Q server config, unchanged:
  - TP2 × PP2, MTP K3, `block_size 4096`, `prefix_match_unit 64`, `mamba_cache_mode align`, `use_replayssm`;
  - `max_num_batched_tokens 1024`, `max_num_seqs 21`, `prefix_cache_retention_interval` at its default 0;
  - E25 NB4 Q-defer as in K100Q (`NB4_PPDEFER=1`). It changes only when PP0 posts the receive; consumption stays at call T + 2, so the fence is unchanged (nb4_ppack.py docstring).
- **Dual mode (notebook rule):** the notebook carries the proven eval + private-format cells (E18 R2 CELL 0 verbatim, E15B writer), as every notebook does. This measurement runs in the interactive part only.

---

## 0. Before the GPU: Kaggle CPU kit (same pinned image, Py3.12)

1. Run `test_gentail_seam.py` and `test_gentail_logic.py` on the kit. Expected: 31 / 31 and 11 / 11. Every test fails if either Duck CSV is missing, so ship both CSVs with the kit.
2. Real-class import check (the local venvs lack vLLM's dependencies):
   - apply the seam to the payload tree;
   - `import vllm.v1.core.sched.scheduler, vllm.v1.core.kv_cache_manager, vllm.v1.core.gentail_ckpt`.
   - Then `maybe_install_gentail` on a real `KVCacheManager` built from the real K100Q `kv_cache_config` (CPU, no model). Assert:
     - the `install` line lists the Mamba groups (2 after the PLE fold) and the attention groups (target and MTP draft; no QSA ring);
     - `reprefillable` is 2 (multi-module MTP) or 0, and the value is recorded;
     - `mamba_block` is 4096 and `unit` is 64;
     - `MAX_GRACE_PINS` is 21.
3. Revert the seam: the files must be byte-identical to the originals.

## 1. Boots (two server boots; total ≈ 2 × 5-6 min from a Kaggle Model)

| Boot | Env (`gentail_seam.server_env(...)`) | Used by |
|---|---|---|
| **A** | `GENTAIL_CKPT=1`, `GENTAIL_CKPT_LOG_PUBLISH=1`, `GENTAIL_CKPT_STATS_EVERY=1` (strict truncation is the default) | gates 1-4 |
| **B** | the same plus `GENTAIL_CKPT_NOFENCE=1` | the gate-4 negative control only |

Run the engine in-process: `VLLM_ENABLE_V1_MULTIPROCESSING=0`, with `LLM(...)` and the server's engine args. Gate 3 needs the scheduler objects and `collective_rpc` on all 4 workers. Gates 1, 2 and 4 use `llm.chat(...)` with the Duck chat template and tools, the same rendering path as the server.

**Boot gates (fail-fast; on any failure: stop and save the logs):**
- `E30_GENTAIL {"event": "install", ...}`: `pp_size` 2, `spec_query_len` 4, `nofence` false (boot A) / true (boot B).
- The line "RecoverSSM + pipeline parallel: Mamba prefix checkpoints are published for prompt positions only (3 groups)" is still there. The per-step guard is untouched.
- No `inactive` line, and no `GENTAIL_CKPT=1 needs defer_block_free` error. That error means `defer_block_free` is off in this config; stop.
- `readback(log)["label"] == "INSTALLED"`.

## 2. Workload

- **Chain:** one recorded 10-turn Duck chain from K100Q, picked from `reprefill_per_turn.csv`. Messages are replayed from the saved trace with greedy decoding (`temperature 0`).
  - Requirements: at least 3 turns with a generation over 4 k tokens whose previous turn ended with `tool_calls`; every turn's predicted Option-B checkpoint is above the previous prompt.
  - Prefer a chain with turns over 90 k tokens (the user's rule: long bands decide).
- **Per request:** `return_token_ids`, `logprobs: 1`, `max_tokens` from the trace.
- **Passes:**
  - **cold:** each turn with a fresh `cache_salt`;
  - **warm:** one shared salt.

## 3. Gates

### Gate 1: token-ID identity (no engine dependency)

- For every warm turn k: `prompt_ids_k[:P_{k−1} + G_{k−1} − 1]` vs `prompt_ids_{k−1} + output_ids_{k−1}[:−1]`.
- Record the first mismatch index m_k and the token kinds around it (tool-call formatter, reasoning, stop truncation).
- **Pass:** m_k ≥ b_k (the predicted checkpoint) for every turn the patch publishes. A turn with m_k < b_k must show a miss (`cached_tokens` ≤ the prompt checkpoint), never a hit. This is the chained-hash guarantee, now measured.
- Also record the share of turns with m_k ≥ P_{k−1} + G_{k−1} − 66. That is the achievable gain for options C and D.

### Gate 2: checkpoint position

- **Predicted hit per warm turn:** the `test_k100q_savings_under_patch_rules` arithmetic, applied to the real P / G and the finish status. Only `FINISHED_STOPPED` publishes. Under strict truncation, a turn whose `stop_truncated` counter rose publishes nothing.
- **Pass:**
  - `usage.prompt_tokens_details.cached_tokens` equals the prediction on every turn;
  - every `publish` line's `position` equals the next turn's `cached_tokens` when that turn hit through the tail;
  - the counters agree: `adopted` = the number of tail hits, `adopted_tokens` = the sum of their positions.
- Any turn with `cached_tokens` > prediction is a hard fail (it would mean a key the rules do not allow).

### Gate 3: KV + recurrent-state equivalence vs cold

**3a, state level, on all 4 workers.** For each `publish` line (block_ids per Mamba group):
1. Read the GDN SSM state, the conv state and the PLE window of that block for every Mamba layer on each rank (`collective_rpc`: index the Mamba layer caches by block id).
2. **Cold reference:** a cold request with prompt = the first b + 64 tokens of the same token sequence under a fresh salt. Its prompt checkpoint is at floor((b + 64)/64)·64 − 64 = b. Find its block id by wrapping `BlockPool.cache_partial_block` / `cache_full_blocks` in-process during that request: log (request, group, block_id, num_tokens == b). Read the same tensors.
3. **Noise baseline, measured first:** the same comparison between two cold prefills of different lengths (b + 64 and b + 1024) whose checkpoints both sit at b. This is the chunking noise.
4. **Pass, per layer per rank:** max-abs(warm vs cold) ≤ max(4 × baseline, 1e-6 relative). A difference confined to PP0 is the fence hazard: hard fail.
5. **Attention:** compare the K/V of the last 64-token unit before b (target and MTP-draft groups) between the warm and cold blocks. Expect bit-identity.

**3b, API level.** The next turn warm (hit at b) vs the same turn cold:
- the first 64 greedy tokens are identical;
- the top-1 logprob max-abs is within the E19 parity tolerance, comparing like with like (prefix ON on both sides).

### Gate 4: PP fence (busy, idle, abort, negative control)

- **(a) Busy:** the chain plus 2 filler streams (long generations started 1 step apart, so a second microbatch is in flight).
  - Expect `published` ≥ 1, `deferred` == 0 (pp 2: b ≤ limit puts the producer ≥ 16 windows before the end), and gate 3 passes.
- **(b) Idle:** the same chain, single stream.
  - Expect the same publishes and gate 3 passes. The hit is used by the next turn.
- **(c) Abort at a boundary:** stream a request and abort it in the step right after `output tokens` crosses (4096·k − P).
  - Expect `skip_abort` += 1, no `publish` line for it, and a later `stats` line with `queue` back to 0.
  - Check that no block leaks: `num_free_blocks` returns to its value before the request, once the grace pins are released.
- **(d) Negative control (boot B, idle):**
  - Pick `max_tokens` so the generation ends 1-3 tokens after a 4096 crossing. NOFENCE allows a length-capped finish and skips the producer rule and the limit.
  - Read the published block (gate 3a) on PP0 and PP1 against the cold reference.
  - **Expected:** a PP0 difference ≫ baseline (PP0 never committed the final window), PP1 within baseline. This shows the test can see the hazard.
  - If PP0 is clean, gate 4 is inconclusive: report it, and do not claim the fence was proven.
- **(e) Strict truncation A/B (optional, if time is left):** rerun the chain with `GENTAIL_CKPT_STRICT_TRUNC=0`. Record `stop_truncated`, `published` and gate 2/3.

### Counters to save every run

`recorded, superseded, queued, deferred, published (+_pinned/_lru/_tokens), skipped (+ every skip_<reason>), released, adopted (+_tokens), evicted, stop_truncated, final_window, pin_budget_released, index_reuse, plan_mismatch, seq_mismatch, no_block, sched_overlap`

- **Hard fail:** `plan_mismatch` or `seq_mismatch` > 0 on a pure decode stream. The commit-plan port disagrees with the engine.

## 4. Outputs

Save all of:
- per turn: `cached_tokens` predicted vs actual;
- the m_k identity table;
- every `publish` line;
- state max-abs per layer, per rank, per group (3a);
- the 3b token / logprob diffs;
- TTFT warm vs cold;
- all counters;
- the full server and worker logs;
- the full thinking of every request (standing rule).

Record the time budget per phase. The run stops at the first hard fail, after saving what it has.

## 5. What it does not prove

- **Throughput:** that needs a Duck run with gentail on vs off.
- **Pin pressure at 21 workers:** the G-3 simulation estimates the peak at 10.7-12.7 % of the pool under grace pins. The budget defaults to 25 % of `num_gpu_blocks`.
- **The real share of stop-truncated finishes:** this run measures it on one chain only.
