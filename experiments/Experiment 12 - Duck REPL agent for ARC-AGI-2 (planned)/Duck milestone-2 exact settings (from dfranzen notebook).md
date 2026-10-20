# Duck harness: the exact milestone-2 settings (copied from the pulled notebook, not from memory)

**Source:** Kaggle `dfranzen/arc-agi-3-milestone-2-solution`, pulled 2026-10-19 to `_scratch/ext/kaggle_dfranzen/`.
- Cell 2: the harness patch (= the da-fr repo source).
- Cell 4: `setup_env`.
- Cell 12: the SGLang launch.
- Cell 16: solver overrides.

## Harness environment (cell 4), verbatim

| Group | Setting | Value | ARC-AGI-2 port |
|---|---|---|---|
| serving | LOCAL_ANALYZER_PROVIDER / OPENAI_PROVIDER | `vllm` (OpenAI-compatible request mode) | same; base URL = our SGLang `/v1` |
| analyzer | ARC3_REASONING_HISTORY_KEY | `reasoning_content` | same |
| | LOCAL_ANALYZER_ENABLE_THINKING | true (effort not set → template default **xhigh**) | same |
| | LOCAL_ANALYZER_TOOL_TIMEOUT | 30 s | same |
| | LOCAL_ANALYZER_YIELD_TOKENS | **2048** (YIELD_SECONDS 0) | same |
| | TEMPERATURE / TOP_P / TOP_K | **0.7 / 0.95 / 20** | same |
| context | LOCAL_ANALYZER_CONTEXT_WINDOW | (116+12)·1024 = 131,072 | same (server context 155,648 ≥ this) |
| | LOCAL_ANALYZER_MAX_OUTPUT | **12,288** | same (approved exception to the 60k floor, user 2026-10-19) |
| | ARC3_CONTEXT_DRAIN_TOKENS | 58·1024 | same |
| | ARC3_HISTORY_ASSISTANT_TURNS / TURN_DRAIN / DRAIN_COALESCE | 150 / 30 / 1 | same |
| tool output | LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS | 3072 | same |
| | ARC3_MIDDLE_TRUNCATION | 1 | same |
| | ARC3_ACTION_ECHO | 1 | n/a (game actions) |
| prompts | EXPLAIN_GAMEPLAY_CHANGED, NEW_CHANGED_PROMPTS, LEVEL_TRANSFER_GUIDANCE, ACTION_INFO, DEDUPE_MULTICALL_LINE, FRAME_DIFF_HINT | 1 | game-only → off |
| vision | MULTIMODAL_CONTEXT current_grid, UPSCALE 10, DIFF_IMAGE, GAMEOVER_DIFF_IMAGE, ANIMATION | on | phase 1 off; phase 3: grid images (Qwen3.8 is a VLM; Tufa credits multimodality) |
| guards | GUARDS_FROM_LEVEL 2, BATCH_NOOP_BLOCK, STALE_STATE_BLOCK | on | game-only → off |
| world model | ARC3_MEMORY_SECTIONS | off | same |
| functions | PERSISTENT_FUNCTIONS 1, SCOPE game, IMPORTS 1 | on | same (scope = task) |
| scheduling | MAX_ACTIVE_STREAMS 10, PRIORITY_REFRESH_QUEUE 1, PACE 0, TAIL_FADE 1 (fraction 0.2), TAIL_LOOKUP remaining, SCORE_NORMALIZATION 1 | on | MAX_ACTIVE_STREAMS = 12 (our server); the priority formula needs an ARC-2 progress signal (best train score) |
| other | ARC3_WARMUP_ACTION_GAMES 1, ARC3_HTTP_RETRY_INITIAL_SECONDS 900 | | same |

## Solver (cell 16)

- **Hidden rerun:** concurrency 120 (all games admitted, the priority gate limits to 10 active), `max_runtime_s_per_game` 532·60, analyzer_timeout 900 s, 1 pass.

## SGLang launch (cell 12): Pennyroyal v2.5.3 on an RTX PRO 6000 (sm120)

- **Flags that matter for us (all verified to exist in SGLang 0.5.21):**
  - `--reasoning-parser qwen3`
  - `--tool-call-parser qwen3_coder`
  - `--default-chat-template-kwargs '{"preserve_thinking":true}'`
  - `--schedule-policy lpm`
  - `--enable-cache-report`
  - `--mamba-track-interval 64`
  - `--page-size 64`
  - `--watchdog-timeout 1800`
  - `--enable-metrics`
- **Their engine settings, not ours:** TP1, auto-round W4A16 Flash-Next, NEXTN MTP + FR-Spec, PLE offload. We keep W1: TP4, RedHat INT4 27B, DFlash2, extra_buffer_lazy, fp16 SSM, server 13 / client 12.
- **Open:**
  - Do `--mamba-track-interval 64` and `--page-size 64` help multi-turn prefix reuse on 0.5.21 + DFLASH? Measure; it may change the mamba cache size.
  - `lpm` (longest-prefix-match) scheduling suits multi-turn reuse; check it is compatible with our admission.
