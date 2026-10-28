# Experiment 21 — Duck config B on vLLM 0.19 + vLLM 0.32 (two halves per notebook)

**Status (2026-10-28):** notebooks built and under validation (local CPU harness, then Kaggle CPU kit, then 2× agent
validation). Nothing has been pushed.

## Why

Duck config B (Experiment 15, R3K1/R3K2) has only ever run on SGLang 0.5.21 + RedHat INT4 + DFlash2. On EVAL120, in the
same 6-h window, it scored task-norm 33.47 % (R3K1) and 39.03 % (R3K2).

On the Poetiq method, vLLM beat SGLang on the LB by about 4-7 points:
- vLLM: S1-60K 34.58 / 31.94, S1-49K 32.08, S1-55K 31.81;
- SGLang: P1-60K 29.17, E15P 26.94.

Per-request quality on eval looked equal. Iteration 1 of R5 vs E15P, on the 26 tasks where R5 closed naturally under
60k: 23 vs 22 train-verified. Nobody had measured Duck on vLLM. E12-V ran only a 1-task smoke.

## What runs

| Notebook | Weights / streams | Half A (0-~6 h) | Half B (~6-12 h) | Fallback per half |
|---|---|---|---|---|
| `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` | RedHat INT4 W4A16 + MTP-3, **20** streams | vLLM 0.19.0 | vLLM 0.32.0 | FP8 MTP-3, 16 streams |
| `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb` | FP8 + MTP-3 + `--mamba-ssm-cache-dtype float16`, **16** streams | vLLM 0.19.0 | vLLM 0.32.0 | the same without the SSM flag (= S1-60K's engine) |

**Both halves:**
- the same 120 evaluation tasks;
- fresh Duck state;
- own folder `q27x/half<H>/`;
- a solve window of 5.5 h from each half's own solve start (`E21_HALF_SOLVE_SEC`). R3K had about 5.7 h.

Half A must end by `HARD_STOP − 180 − window − 25 min`, so that half B fits.

The hidden rerun (`RERUN_ALL`) runs **half A only**: all 240 tasks, 12 h.

**Report:** `q27x/e21_report.json` gives, per half:
- the full score;
- the **attempt-1-only score**, which can be compared across halves whatever attempt 2 reached;
- the config actually used, the capacity check, tok/s, prefix-cache hits, preemptions and the timeline.

## Donor map (build_e21.py takes and asserts every piece)

| New cell | Source | Change |
|---|---|---|
| CELL 0 | identical in E15B R3K2, E12-V, E18 (asserted by the builder); also identical in S1 and VMS (checked, not asserted) | none |
| CELL 1 | E15B R3K2 (`c058d58b`) CELL 1 | default profile `E21X2EVAL120` (new row, 12 h − 3 min, `solve_window_sec` 5.5 h); E21 block appended (variant, half specs, `ENGINE_BACKEND = "vllm"`, staging list, window, ports, served names, 0.32 wheelhouse, SGLang cache diagnostic off) |
| CELL 1b | E15B CELL 1b | staging also for `vllm`, of the variant's weights (`E21_STAGE_PATHS`); the read probe on them; the submit-arm label |
| CELL 2 | E15B CELL 2, verbatim | (with `ENGINE_BACKEND = "vllm"` it installs vllm==0.19.0 into the kernel, the S1 path) |
| CELL 3 | E15B CELL 3 (Duck package `q27x_e15b_duck_9e4b89285136.bin`) | + `ARC3_REASONING_HISTORY_KEY=reasoning` (see Findings 1) |
| order cell | identical in all donors (asserted) | none |
| CELL 5 (engine) | verbatim: E12-V CELL 5 (`e85b69c2`, the run 2 Duck server: libcuda shim, request helper, Prometheus reader + aliases, stats-line summary, kill); E18 R2 CELL 5 (`116d2d41`: `e18_install`, `E18Server`, `_e18_kill_gpu_pids`, `_wait_gpu_release` = S1) | new `e21_engine_new.py`: vLLM 0.19 flags (= the E12-V command + `--max-num-seqs` per half + optional SSM flag + INT4 path), vLLM 0.32 flags (= VMS `vms_flags` arm + Duck serving flags + 135,168 context), capacity check, launch + smoke, begin / end half, watchdog relaunch, staging join |
| CELL 6 | E15B CELL 6, wrapped as `e21_cell6()` (every module-level name global) | `_e12_capacity_probe` → vLLM gauges; `_e15b_gen_tokens_total` → `vllm:generation_tokens_total`; `_e15b_server_log_summary` → vLLM stats lines; watchdog → `e21_relaunch()`; the solve end capped by the half's end. The scheduler call (`solve_all_r6` + every config-B parameter) is unchanged. |
| CELL 7 | E15B CELL 7, wrapped as `e21_cell7()` | per-half bundle name; cache-note text |
| CELL 8 | new | runs the halves: begin → CELL 6 → CELL 7 → end, each step isolated |
| CELL 9 | new | the two-half report + bundle |

**Dropped from E15B:**
- the SGLang engine cell;
- the E14-A1/A2 SGLang measurement arms;
- the E14 JIT-cache cell.

**Dead code kept:** `_e12_parity` (parity profile only, never called).

## Findings while building (2026-10-28)

1. **Reasoning history key.** E15B's Duck sends prior thinking as `reasoning_content` (DUCK_ENV).
   - vLLM 0.19 reads only `message["reasoning"]` on input: `vllm/entrypoints/chat_utils.py:1512`.
   - vLLM 0.32 does the same: `vllm/renderers/chat_utils.py:2083`.
   - So on vLLM the history would lose the thinking, and prefix-cache hits would drop.
   - The E12-V run 2 Duck sent `reasoning`, and its prompts grew by the generated tokens (6,882 + 12,288 → 19,215).
   - Fix: `ARC3_REASONING_HISTORY_KEY=reasoning` (CELL 3 and each half).
2. **Duck package correction.** R3K2 ran `q27x_e15b_duck_9e4b89285136.bin` (CELL 3 line 9), not `d6faf44ec0d5` (the R3J notebooks, Exp 15 Notes/14).
3. **SSM fp16 is not new for us.** SGLang production already runs `--mamba-ssm-dtype float16`. vLLM 0.19's `CacheConfig` has `mamba_ssm_cache_dtype` (`config/cache.py:105`).
4. **INT4 vs FP8 template and tokenizer.** Checked on the E13 fixture copies of both model folders:
   - `chat_template.jinja` is identical (sha c3cf9e34), and equal to the FP8 tokenizer_config's embedded template;
   - vocab and added tokens are equal;
   - encode and decode are identical on grid, code, thinking and template text.
   - To be re-checked on the real Kaggle model files in the CPU kit.
5. **RedHat INT4 quantization config:** compressed-tensors pack-quantized W4A16 g128. It ignores `re:^mtp.*`, `lm_head`, the vision tower and, in each GDN layer, only `linear_attn` itself, `in_proj_a`, `in_proj_b` and the norm. The GDN `in_proj_qkv`, `in_proj_z` and `out_proj` ARE INT4 (correction from VAL_engine_A2 L8). The MTP head is BF16. The INT4 checkpoint also carries calibrated fp8 KV scales; FP8 uses 1.0. vLLM 0.32 ran it (VMS, diagnostic). On vLLM 0.19 it is untested: NB1's fallback covers that.
6. **R3K2 really ran 16 concurrent streams:** server running mean 15.62; 3,109 lines at 16. Its `E12_LIVE_BRANCHES = 12` is unused by the rev6 scheduler.
7. **Duck on vLLM prefix cache (E12-V run 2):** turn 2 hit 4,800 / 19,215; later turns 67-94 %. SGLang gives 93 %. vLLM's mamba `align` mode caches in 1,600-token blocks.

## Pre-run capacity check (per half, printed by CELL 5)

- Effective KV = vLLM's "Maximum concurrency" factor × max-model-len.
- Duck streams = min(seqs, effective KV // 50,000), at least 8.
- R3K2 per-attempt peak context: p50 57k, p90 65k, max 79k.
- **NB2 (FP8, 16 streams):** needs 16 × 57k ≈ 0.91 M at the p50 peak. FP8 effective KV is about 1.03-1.11 M.
- **NB1 (INT4, 20 streams):** needs about 1.14 M. INT4 frees about 3.5 GB per GPU.
- Both are checked against the logged values at launch.

## rev 2 (2026-10-28): fixes from the 10 validation reports

Notebooks: NB1 `c980f651`, NB2 `d87f687f`. The reports are `Experiment Review/VAL_{engine_A1,engine_A2,state_B1,state_B2,timing_C1,timing_C2,api_D1,api_D2,fidelity_E1,fidelity_E2}.md`. Each claim was checked against the source before it was fixed.

| # | Fix | Found by |
|---|---|---|
| 1 | **Duck's priority gate is reset per half.** `tool_agent._PRIORITY_GATE` is a process-wide singleton built once with `ARC3_MAX_ACTIVE_STREAMS` and a min() clock, so half B ran with half A's slots and an expired tail-fade clock. | B1, B2, C2, E1, E2 |
| 2 | **`finish_reason` "length" is restored for a tool call cut at max_tokens.** vLLM 0.19/0.32 report "tool_calls" (verified: `chat_completion/serving.py` 0.19 L1550-1560, 0.32 L1099-1116). Duck's cut handling keys on "length". Done by a CELL 3 patch; the package bytes are unchanged. | D1 |
| 3 | **Launch gate made strict:** the smoke answer must contain "4" and not be degenerate; a two-turn tools + thinking contract probe must pass (history prompt growth checked); and the capacity check is block-based and fail-closed. Any failure → fallback. | A2, C2, D1, A1, D2, E1 |
| 4 | **Capacity = blocks.** Every request holds cdiv(L, B) + 15 fixed Mamba blocks. Checked: E12-V log 8.46 × (85 + 15) = 846 blocks; 846 // 4 × 1,600 = 337,600 = its KV line. Streams = min(seqs, num_blocks // (cdiv(57k, B) + 15)); refused below 8. | A1, A2, D2, E1 |
| 5 | **`--language-model-only` on vLLM 0.32**, as half A. E18 R2 rung s1 ran with it on 0.32: KV 1,110,415 vs 1,006,517 without. | E2, A1, A2 |
| 6 | **`--disable-uvicorn-access-log` on both versions.** Our /metrics polls kept the log growing, so the watchdog never reached /health. The flag exists in 0.19 `openai/cli_args.py:226` and 0.32 `launchers/cli_args.py:309`. | D2 |
| 7 | **NB1 stages its FP8 fallback too**, so the fallback does not read 30 GB cold from GCSFUSE. All R3K2 inputs are attached + the 0.32 wheelhouse. | A2, A1, C1, E2 |
| 8 | **Equal windows:** half B gets half A's achieved window when that was shorter. `windows_equal` is in the report. | C1, C2, B1, B2, E1, A1, A2 |
| 9 | **If half A ran its fallback, half B runs the same config** (`E21_B_FOLLOWS_A`). Recommended option (ka); the user has not yet answered. | C2 |
| 10 | **The last half kills fast and does not wait for GPU release.** The hidden-rerun tail was ~53 s from 12 h at worst. | C1, C2 |
| 11 | **Watchdog relaunch timeout follows the solve end:** min(30 min, end − 5 min); no relaunch with < 12 min. | B1, A1, E1 |
| 12 | **Smaller fixes:** see the list below. | A1, A2, B1, B2, D2, E2 |
| 13 | **Report:** see the list below. | various |

Smaller fixes (row 12):
- the half-A watchdog thread is joined;
- a straggler relaunch never kills another half's server;
- the 0.32 install failure is cached;
- `e21_end_half` kills in `finally`;
- GPU release is checked before every launch;
- per-half state is reset first;
- the model-id fallback is the served name;
- the stats regex handles Preemptions / Deferred;
- server counters are snapshot before every kill;
- the `wrap()` guard is fail-closed (any multi-line token, Match / Delete / exec / locals, a symtable check that only comprehension locals are left);
- the PUSHED guard matches 8-hex prefixes and checks the size before writing;
- the profile is asserted;
- CONFIG records what really runs;
- the flagcheck import works on 0.32, and parsing works on a CPU box (stub platform).

Report additions (row 13), per half:
- window;
- recomputed prompt tokens;
- Duck gate stats;
- server counters (preemptions, whole-run prefix hits);
- cut calls restored;
- contract probe;
- GPU checks.

**Structural, not fixable at the HTTP level (D2 F1, D1 F3).**
- vLLM's mamba `align` mode never caches a previous reply. Each Duck turn re-prefills it, plus about 1-2 blocks (exact on 10/10 E12-V turns).
- On R3K2's request pattern: about 74.7 % cached on vLLM vs 93.4 % on SGLang, which is about +11 M prefill tokens per EVAL120 half (~1-2 h).
- Measured per half as `recomputed_prompt_tokens`. NB2's SSM fp16 halves the block (832 tokens; seen in the 0.32 VMS log).

## rev 2b (2026-10-28): the findings rev 2 skipped

Notebooks: NB1 `84876a8b`, NB2 `9b6e7359`. Every finding of the 10 reports, with its status: `Experiment Review/E21_FINDINGS_TRACKER.md`.

Added in 2b:
- a prefix-cache probe in the launch gate (D2 F5);
- B-side setup risk:
  - window 5.45 h, reserve 30 min;
  - vLLM 0.32 venv built in the background during half A;
  - 0.32 ready timeout 20 min;
  - preflight request timeout 60 s;
- the second relaunch uses the half's other config;
- separate JIT caches for 0.32;
- Duck's tokenizer falls back to FP8 when INT4 is missing (identical encode);
- per-half stage-1 completion + strict pass1 (the attempt-1-only score is comparable only when stage 1 finished);
- counter of tool calls with visible content (the 0.32 parser difference);
- MTP-acceptance warning;
- the `accept_len_median` key;
- the Duck env base URL per half;
- the kit's transformers-4.57.6 / 0.32-venv tokenizer check of the real INT4 and FP8 folders (`src/e21_tokcheck.py`).

**Donor-map notes (VAL_fidelity_E1 LOW-7):**
- The E12-V `e11_kill_server`, `_E12V_STAT_RE` and `e11_decode_summary` are inserted verbatim, then **overridden** by E21 code (signature fix, the Preemptions / Deferred regex, the counters).
- In CELL 6, the R3K2 SGLang-specific relaunch path is replaced by `e21_relaunch`. That path covered: the 12-stream degraded relaunch on OOM, `_e11_gpu_free`, and `E11Server` / `e11_flags`.
- The SGLang startup split in the log summary is dropped.
- The preflight request timeout is 60 s. The model-id fallback is the served name.
- CELL 1 also adds `E21_B_SETUP_RESERVE_SEC`, `E21_LAUNCH_TIMEOUT_SEC` / `_V032_SEC`, `E21_B_FOLLOWS_A`, `E21_VLLM_MAX_MODEL_LEN`, `E18_SILENCE_SEC`, `E21_FAKE` / `E18_FAKE`, two asserts, the tokenizer fallback and the CONFIG refresh.
- R3K2's `hidden_equiv` (gate 4b) window is not used.
- Dropped with R3K2 CELL 7b: `E14_STATE.end("done")`, `e14_report.md`, `Q27X_E14A_BUNDLE.zip`.
- **Dead code kept:** `_e12_parity`, `_e12_no_spec`, `_e12_cache_diag` (switched off) and `_e12_branch_done`.

**Report-only fields that are SGLang-shaped:** `_r6e_concurrency` (empty on vLLM), the preflight's drafter config (not applicable) and the heartbeat's keep filter. They are kept verbatim.

**Inputs each GPU notebook must attach (both notebooks):**
- the competition data;
- the payload `esfersami50/q27-e15b-payload` (or the hosen42 copy);
- FP8 `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`;
- INT4 `foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1`: NB1 weights, NB2 Duck tokenizer. If it is missing, Duck falls back to FP8's tokenizer;
- DFlash2 `foysalemonshanto/qwen3-8-27b-dflash2/PyTorch/bf16/1` and the SGLang wheelhouse: not used, but attached as in R3K2, so CELL 1's `_e11_discover` does not rglob-scan `/kaggle/input`;
- `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (vLLM 0.19);
- `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` (vLLM 0.32);
- the pinned image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`, NvidiaL4, internet off.

## Validation plan

1. Local CPU harness (`src/harness/`): fake vLLM Duck server and modes commit / rerun / fallbackA / halfAfail / death / reasoningkey.
2. Kaggle CPU kit on the pinned image (Python 3.12). It also hashes the real model files from Finding 4.
3. 10 agents validating twice.
4. GPU commit / push only after the user's go.
