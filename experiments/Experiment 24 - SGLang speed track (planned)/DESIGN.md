# Experiment 24: build design (rev 2, 2026-10-30)

This file turns the six-section plan in `README.md` into notebooks. It records every arm, how each number is measured, the run order with its time budget, and every place where the build departs from the README, with the reason.

- **Rev 1** was based on eight research notes (2026-10-29/30) covering:
  - the E23 template;
  - the EMS/E15B engine layer;
  - the SGLang 0.5.21 source (PyPI wheel sha256 `ac300998…`);
  - the section-6 backports;
  - the research PDF v5;
  - prior accuracy work (E11/E15P/E17/EMS);
  - the vLLM oracle arms;
  - the Kaggle inputs and kits.
- **Rev 2** applies an independent four-lens review of rev 1: 71 findings, 6 of them blockers. The main changes:
  - SGLang logprob chunking against an out-of-memory crash;
  - correct census plateaus;
  - a starvation measure for section 5;
  - aligned and A/A-calibrated oracle comparisons;
  - FP8-generated teacher-forcing traces reaching 120k;
  - production-sampled decode-vs-prefill tests;
  - per-arm cache policy;
  - section-6 trigger evidence;
  - the README's 600 s gate applied to every launch.

## 1. What the build delivers

One builder (`Experiment Notebooks/Build/build_e24_notebook.py`) writes three notebooks with identical code. Only the baked `MODE` / `PART` defaults differ, because Kaggle commits cannot set env.

| Notebook | Sections | Budget (hard stop) | Planned work | Quota (2x on 4xL4) |
|---|---|---|---|---|
| `Q27-XHIGH-E24_SGLANG_SPEED_SMOKE.ipynb` | README SMOKE list (subsets of 1-6) | 210 min | ~170 min | ~6 h |
| `Q27-XHIGH-E24_SGLANG_SPEED_FULL_A.ipynb` | 1 accuracy decomposition, 6 backports, multi-boot A/A | 480 min | ~420 min | ~14 h |
| `Q27-XHIGH-E24_SGLANG_SPEED_FULL_B.ipynb` | 2 width × streams, 3 NCCL, 4 capacity, 5 interference, ≤ 3 A/A rounds | 690 min | ~570 min + conditionals | ~21 h |

**Why FULL is split.** At measured unit costs the README's FULL is 12-15 h before A/A repeats:
- a cold SGLang launch takes 480-570 s, of which about 380 s is target-verify graph capture, and a warm one 55-76 s;
- a cold vLLM 0.32 launch takes 487-560 s;
- distinct-prompt prefill runs at about 1,250-1,650 tok/s (one 90k prompt ≈ 60-70 s, one 116k prompt ≈ 85-95 s).

A session ends when its plan ends, so quota equals the real wall time.

**Quota.** E24 needs about 40 quota-h, more than one account's 30 h per week. The suggested split is SMOKE + FULL_A on one account and FULL_B on another. The private inputs must be shared with whichever account runs them.

**Not in these notebooks.** The paired ARC sample (§5.1).

## 2. Engine baseline, arm deltas, inputs

**BASE** is the EMS `dflash_w6_s20` argv as the engine cell built it. Every value was replayed and matched against the logged `server_args`.

```
<venv>/bin/python -m sglang.launch_server --model-path <INT4 staged> --tp-size 4 --kv-cache-dtype fp8_e4m3
 --context-length 155648 --chunked-prefill-size 4096 --mamba-ssm-dtype float16 --enable-metrics
 --disable-flashinfer-autotune --random-seed 20261005 --host 127.0.0.1 --port <p> --log-level info
 --watchdog-timeout 1800 --decode-log-interval 40 --speculative-algorithm DFLASH
 --speculative-draft-model-path <DFlash2 staged> --speculative-draft-model-quantization unquant
 --disable-prefill-cuda-graph --mamba-radix-cache-strategy extra_buffer_lazy --mem-fraction-static 0.83
 --incremental-streaming-output --stream-interval 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
 --default-chat-template-kwargs '{"preserve_thinking": true}' --schedule-policy lpm --enable-cache-report
 --speculative-num-draft-tokens 6 --min-free-slots-delay 1 --max-running-requests 21
 --cuda-graph-bs-decode 1 … 21 --max-mamba-cache-size 84
```

- The README's argv block lists two resolved defaults (`--attention-backend`, `--linear-attn-backend`) that are not passed, and it omits about 12 real flags. The builder therefore generates argv from the engine-cell logic (`e11_flags` plus the EMS arm edits) and asserts it against this token list.
- The real log prints the resolved args as a dict (`server_args={...}`). `/server_info` returns them too.

**Rules every SGLang arm keeps:**
- **Streams set slots and graphs together** (`_streams(n)`): `--max-running-requests n+1`, `--cuda-graph-bs-decode 1..n+1`, `--max-mamba-cache-size 4·(n+1)`.
  - With `--min-free-slots-delay 1`, a graph list shorter than the running count crashes every TP rank (E14A).
  - In 0.5.21, `extra_buffer_lazy` with the overlap scheduler costs 4 slots per request, and `max_running_requests` is capped at `max_mamba_cache_size // 4`. That cap is the source-derived minimum.
  - The non-lazy `extra_buffer` needs 5 slots per request. Those section-6 arms use explicit recipes, never `_streams()`.
- **Target-only** drops the six speculative flags and `--min-free-slots-delay`.
- **Section-1 and section-6 production arms use S16** (E15B R3K2: max-running 17, Mamba 68, graphs 1..17). The speed sections use S20 unless the arm changes it.
- **Process and GPU hygiene:** one fresh server process per arm, and a GPU release gate between arms (< 1024 MiB on every GPU).
- **Fixed env:**
  - `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`;
  - `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`, set explicitly as in E17;
  - `SGLANG_LOGPROB_CHUNK_SIZE=128` on every SGLang arm, in all sections, so GEMM shapes match across arms (§3.1).
  - The env record asserts that `SGLANG_TEST_*`, `SGLANG_SIMULATE_*` and `E24_INJECT_*` are absent, except in the arms that set them.
- **Sampling.** Every T>0 request on either engine passes `temperature 1.0, top_p 0.95, top_k 20` explicitly. SGLang's APIs do not apply `generation_config`; vLLM does.

**vLLM 0.32 oracle arms** use E23's `stock_flags(model, port, seqs=16, spec=0, prefix=False)`: TP4, fp8_e4m3 KV, max-model-len 131072, max-num-batched-tokens 4096, gpu-memory-utilization 0.90, disable-custom-all-reduce, seed 20261005, FLASHINFER, no prefix caching, no speculation.
- `prompt_logprobs` is computed in 1,024-row bf16 chunks (about 1.1 GB peak), so teacher forcing fits.
- SMOKE checks it with a scoring call of at least 4,096 rows.

**Inputs** (Kaggle mount paths; FP8 is attached to all three notebooks, because prompt ids come from its `tokenizer.json`):

| Input | Path |
|---|---|
| SGLang wheelhouse | `/kaggle/input/datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` |
| vLLM 0.32 wheelhouse | `/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` (SMOKE, FULL_A) |
| INT4 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` |
| DFlash2 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1` |
| FP8 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` |
| ARC tasks | `/kaggle/input/competitions/arc-prize-2026-arc-agi-2` |

- **Staging.** INT4 and DFlash2 are staged to `/dev/shm` by the E14 load path, whose full sha256 values are recorded as the checkpoint pins.
- **FP8 weights** are read from NFS. A background page-cache prefetch starts after the staging join, and only when the plan has `s1_a`.
- **The DFlash2 variation** (`bf16`) was found by the kit agent from the EMS mount path. A wrong variation drops the mount silently (the E23 kit v1 lesson), so the notebook fails closed on a missing input.

## 3. Shared machinery

### 3.1 The lab, `src/e24_lab.py` (stdlib only)
E23's `e23_lab.py` rev 3b plus an engine adapter. vLLM stays, because section 1 needs it. All SGLang facts below were verified in the 0.5.21 source.

**Counters:**
- **Decode tokens.** `sglang:generation_tokens_total` / `prompt_tokens_total` move only when a request **finishes**. Windows therefore use `sglang:realtime_tokens_total{mode="decode"}`: accepted plus bonus tokens every decode step, TP0.
- **Prefill tokens.** Prefill uses `{mode="prefill_compute"}`, with `prefill_cache` reported too.
- **Steps.** Verify steps are `sglang:cuda_graph_passes_total{mode=decode_*}` on `tp_rank="0"`. A missing series counts as 0.
- **Label handling.** `prom()` keeps labels and takes tp_rank 0 or the max, never the sum.
- **Retractions.** `num_retracted_requests_total` does not exist before the first retraction. It counts as 0 when the scheduler series are live, no retraction line appears in the TP0 log slice, and the `num_retracted_reqs` gauge is 0. It is `None` only when the scheduler series are missing.

**Gauges:**
- `num_running_reqs`, `num_queue_reqs`, `full_token_usage`, `kv_available_tokens` / `kv_evictable_tokens`, Mamba available and `spec_accept_length` refresh every `--decode-log-interval` (40 steps, about 6 s), and are republished unchanged during prefill.
- Settle and plateau rules therefore use the **prefill counter plus the TP0 log lines**, not the gauges alone.

**Load and windows:**
- **Load streams.** Load uses streaming `/v1/completions` with `return_token_ids` on both engines.
  - Per-chunk arrival times are recorded. With `--stream-interval 1` and incremental streaming, there is one chunk per verify step per stream, so client inter-chunk intervals give a step-time distribution with no profiler.
- **Window alignment.** Every steady window records the median tokens already generated per stream at its start, and the mean accept length. Compared arms start their windows at the same generation depth (≥ 1,000 tokens per stream), because accept length and tok/s drift by 2-8 % within minutes. Windows report tok/s, tokens per step and step ms together.

**Greedy rows and teacher forcing (SGLang native `/generate`).**
- Fields: `input_ids`, `return_logprob`, `top_logprobs_num 2`, `return_text_in_logprobs false`.
- SGLang's OpenAI API drops `prompt_logprobs` and keys top logprobs by text, so it is not used for these.
- **Teacher-forcing convention.** `logprob_start_len = len(prompt) - 1`, and entry 0 is dropped. Entry k is then lp(cont[k] | prompt + cont[:k]).
  - Every response is asserted: length == len(cont), and the ids match. Otherwise the result is `INCONCLUSIVE_TF_ALIGN`.
  - vLLM's `prompt_logprobs[p]` = lp(ids[p] | ids[:p]), with entry 0 null.
- **Memory.** With the default `SGLANG_LOGPROB_CHUNK_SIZE=2048`, scoring a 4,096-token extend needs 3-4 GB of fp32 logits per rank, and only 0.9-2.6 GB are free. That would crash the server. At 128 the peak is about 0.33 GB.
  - Scoring requests run one at a time.
  - The first SGLang arm runs a 4,096-row pre-flight before M3.
- **Under DFlash.** Output logprobs are the target's verify-time logprobs (`compute_spec_logprobs`).

**Server control:**
- **Readiness.** Poll `GET /ready` until it returns 200, then read `GET /server_info` (resolved args, raw argv, `effective_max_running_requests_per_dp`). `/health` runs a 1-token generate per call, so it is never polled.
- **Flag check (enforced).**
  - In the SGLang venv, with the arm's cache env, every `--flag` must be an exact option string of 0.5.21's parser. Prefixes are refused: the parser accepts unique prefixes, and `--cuda-graph-bs` is ambiguous.
  - `prepare_server_args(argv)` must parse the argv. On the GPU box, `resolve_once()` must also pass; a failure is `NO_GO_FLAG_RESOLUTION`.
  - Results are cached by argv hash. Runtime-built (conditional) argvs are checked at launch.
- **Profile (breakdown only, never overlapping a measured window).**
  - Request: `POST /start_profile {output_dir, num_steps, activities:[CPU,GPU], with_stack:false, record_shapes:false}`.
  - Output: one `<id>-TP-<r>.trace.json.gz` per rank.
  - Per-step wall is the interval between consecutive `step[VERIFY` (or `step[DECODE`) span starts on TP0. Intervals between graph launches would not work, because DFlash launches two graphs per step.

### 3.2 Oracle comparisons (aligned, A/A-calibrated)
**Alignment.** Rows are aligned by **prompt key** and truncated to the common token count. No divergence within N tokens gives `EXACT_TO_N`.
- The c16 prompt set is a superset of the serial 8k prompts (its first 4).
- The same alignment applies to pass 1 ↔ pass 2 and to cross-arm pairs.
- Every result reports the positions examined (the sum of first-difference indices) as the denominator of "0 flips".

**Classification of the first differing token k:**

| Class | Rule |
|---|---|
| NEAR_TIE | m_ref + m_cand < T_pair |
| FLIP | T_pair ≤ m_ref + m_cand < 4 |
| STATE_FLIP | ≥ 4 |

- **T_pair** = max(0.5, q99.9 of the pair's |Δ(top1−top2)| distribution on identical prefixes), taken from the pair's teacher-forced top-2 scores (M3).
- **Why calibrate.** Margins are bf16-quantized: about 90-99 % are multiples of 1/16-1/8 nat, and about 1 % are exact ties. Cross-kernel pairs had no calibration in rev 1.
- **Swap.** The row records a swap when the candidate token equals the reference top-2.
- **FLIP attribution.** Each FLIP is attributed by prefill re-scoring of prefix[:k] in both arms. If the prefills agree, the flip is decode/verify-path specific; otherwise it is engine-level.

**Verdicts:**
- **Same-weights pair:** a **finding** (`FAIL_HIGH_MARGIN`) only if its FLIP + STATE_FLIP count exceeds its own A/A count + 1 (the EMS rule). The A/A comes from c16 ↔ c16 repeats, pass 1 ↔ pass 2, cold ↔ warm, and the r2 boots.
- **Cross-weights pair** (a ↔ b, which is "weights + KV scale" because FP8 uses scale 1.0 and INT4 calibrated scales): only STATE_FLIP alarms. Otherwise the result is reported as distributions plus ΔNLL.
- Without a usable A/A, a same-weights FLIP is `BOUNDARY`, not FAIL.
- **Garbage** (empty, a 200-char run, `!!!!`, low variety) counts only when the reference is clean.
- **Decisive pair.** The chain is a → b → c32 → c → d6 / d4. The decisive pair is the earliest link whose A/A-normalised high-margin count is above 0; downstream links are reported too.
- **Report-cell recompute.** The report cell recomputes every section-1 verdict with all A/A data present.

### 3.3 Cache policy and isolation
- **Section 1: fresh roots.** Each arm gets `/tmp/e24/cache/<arm>/` holding `SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `CUDA_CACHE_PATH` (+ `CUDA_CACHE_DISABLE=0`), `TILELANG_CACHE_DIR` and `TORCH_EXTENSIONS_DIR`, plus `VLLM_CACHE_ROOT` for vLLM.
  - `import sglang` only uses `setdefault` under `~/.cache/sglang`, and `SGLANG_JIT_CACHE_DIR` is outside that root, so all of these are set explicitly.
  - The flag check runs with the arm's cache env too.
  - Exceptions: `s1_c_warm` and `s1_c_nocache` relaunch on `s1_c`'s root (the cold/warm control and the cache-OFF reference).
- **Section 6 and SMOKE speed arms.** They run on a copy of the matching section-1 root: `s1_c` for target-only and NGRAM arms, `s1_d6` for DFlash and combined arms. Patched kernels need no fresh cache, because Triton and Inductor caches are content-keyed and the patcher drops `__pycache__`.
- **FULL_B.** The first launch builds one shared cache, budgeted as cold. A "warm" arm whose launch exceeds 300 s is flagged `CACHE_MISS`.

**Isolation proof (section 1 arms only):**
1. **Every server process writes an env record.** The E24 hook (`sitecustomize`) writes `env_<pid>.json` with the cache variables, every `SGLANG_*`, `E24_*` and `NCCL_*` variable, and the role (argv, process name, ppid).
   - Forked children (vLLM EngineCore and workers fork by default) write theirs through `os.register_at_fork`. The cache env is re-snapshotted periodically, because vLLM rewrites Inductor and Triton dirs after import.
   - The expected sets are: SGLang = main + 4 schedulers + detokenizer; vLLM = API server + EngineCore + 4 workers.
2. **Every logged path lies under the arm root.**
3. **The engine's expected dirs are non-empty after the arm.**
   - SGLang: Triton, plus Inductor on DFlash arms.
   - vLLM: `VLLM_CACHE_ROOT/torch_compile_cache` and Triton.
   - JIT, CUDA_CACHE_PATH and FlashInfer dirs are informational.
4. **The default locations received no new files.** These are `~/.cache/sglang`, `~/.cache/flashinfer`, `~/.triton`, `~/.nv/ComputeCache`, `~/.cache/vllm` and `/tmp/torchinductor_*`.

Any failure → `INVALID_CACHE_ISOLATION`, and the arm's comparisons are not used.

### 3.4 Patches and the hook
**`src/e24_sglang_patches.py`:**
- **Base.** The E14 guarded text patcher (check / apply / revert, file sha256 guards, compile check, drops `__pycache__`), plus five E24 hunks:
  - `43567` applies clean;
  - `43559` is **ported**: the upstream text uses `req.lock`, which 0.5.21 lacks;
  - `43029`, `41736` and `41737` apply clean.
- **Rules.**
  - `apply()` refuses `43559` without `43567`.
  - Patch state is checked for all seven known names (the five above plus `44401` and `41872`) against each arm's expected set before every launch.
  - Revert is verified before any unpatched arm.
- **Ordering.** Section 1 runs unpatched first, then section 6.

**`src/e24_hook.py`** (sitecustomize on the server `PYTHONPATH` only):
- **Always:** the env record. The hook thread reads only `sys.modules` / `os.environ`. Triton and Inductor dirs are resolved in the main thread, inside a wrapper on a late method (SGLang `Scheduler.__init__`), to avoid import-lock deadlocks.
- **`E24_HOOK_COUNTERS=1`:**
  - E17's counters (optional-slot failures, in-place fallbacks, stale-donation candidates and confirmations);
  - the `_alloc_mamba_slot` and `_mamba_check_track_boundary` wrappers, with a **synchronous flush before re-raising** on any assert path;
  - a rid context from `DFlashWorkerV2.forward_batch_generation`.
- **`E24_HOOK_TARGETS`:** the arm's expected target set. A missing target on any rank → `HOOK_MISSING`.
- **`E24_HOOK_PROBE=1`:** per rank, the lm_head dtype, the Mamba state dtype, the FP8 k/v scales, and the sha256 of the imported patched module files.
- **Injection** (`E24_INJECT_DRAFT_NAN`, `E24_INJECT_SELECTOR_NAN`) wraps `_grouped_conv` / `CandidateSelector.sample_path`, but only when injection is requested **and** the arm disables decode graphs. The wrapper:
  - returns `orig()` while a CUDA stream is capturing;
  - injects only in the draft-decode forward;
  - keeps the compiled variant;
  - is rank-deterministic by rid. Unequal per-rank injection counts → `INVALID_INJECTION_DESYNC`.
- **Safety.** The hook never changes computation unless injection is set. It chain-executes the image's own `sitecustomize`, whose missing `wrapt` dependency made it fail, and records the result.

### 3.5 Prompts
- **Tokenizer.** Ids come from the FP8 `tokenizer.json`, and the INT4 tokenizer must give identical ids on every prompt. Vocab, merges, added tokens and chat template match; the pre-tokenizer regex differs only for `\p{M}`.
- **Bands.** Distinct ARC prompts per stream at 8k / 40k / 90k / 116k ("120k"), each exactly the band length (E23 CELL 3).
- **EMS replica sets.** These copy EMS's construction for the baseline comparison: the short set (one task per stream, p50 about 3.9k), long (36k shared block + task) and XL (85k shared block + task, at EMS's level).
- **Oracle set.** FULL uses 8k × 4, 40k × 2, 90k × 2 and 116k × 1; SMOKE uses 8k × 3 and 40k × 1.
- **c16 set.** 16 distinct 8k prompts; the first 4 are the serial 8k prompts.
- **Section 5 residents.** 8k distinct prompts.
- **Section 6 recipes.** As in E17 / the backports design. **P255** = [42]*128+[43]*127 for every DFlash forced-fail or natural arm.

### 3.6 Statuses and gates (fail-closed)
**Arm status vocabulary:** `GO`, `NO_GO_<reason>`, `INVALID_<reason>`, `INCONCLUSIVE_<reason>`, `SKIPPED_BUDGET`, `SKIPPED_DEPENDENCY`, `SKIPPED_CONDITION` and `ERROR_<x>`. Nothing defaults to GO, and `INCOMPLETE` / `NOT_EXERCISED` never count as PASS.

**Gates (README):**
- **Provenance:** patch state and shas, the enforced flag check, offline env, checkpoint identity (INT4/DFlash2 shard sha256 from staging; FP8 config/tokenizer identity), and no unexpected `SGLANG_TEST_*` / `SIMULATE` / `INJECT` env.
- **Launch:** every launch, cold or warm, must start within **600 s**. Otherwise it gets `NO_GO_STARTUP_600`, which blocks adoption. Its measurements are kept and labelled "diagnostic: startup gate failed", since startup time does not change numerics. The wait timeout is 900 s. An invalid SM89 kernel (`no kernel image`, `invalid device function`) is NO-GO.
- **Correctness:** stale donation or restore, high-margin drift (§3.2), `STATE_DRIFT` (§4.1 M4), grammar corruption, a tool-call probe failure, repeated punctuation or empty output, unreported truncation (`finish_reason` length), missing cold/warm controls, and the DFlash greedy-fallback warning on any T>0 measurement.
- **Not measured in E24:** coverage, adoption ("correct ARC outputs per GPU-hour") and the agent end-to-end denominator. These are reported as `NOT_MEASURED` with the reason.

**Throughput denominators (Y11), reported separately:**
- saturated decode (steady windows);
- the mixed serving window (section 5 fixed windows);
- agent end-to-end (`NOT_MEASURED`).

**Outputs.**
- Raw files (hook records, rows, traces) go under `/tmp/e24/<arm>/` and are packed into one `e24_raw.tar.gz`.
- `/kaggle/working` keeps `e24_results.json`, `e24_report.md`, the server logs and per-arm summaries. The report asserts it holds fewer than 300 files.

## 4. The sections

### 4.1 Section 1: accuracy-gap decomposition (FULL_A; SMOKE subset)

**Arms**, in run order (each has a fresh isolated cache unless marked):

| ID | Engine / weights | Change | Cache |
|---|---|---|---|
| `s1_a` | vLLM 0.32 FP8 | target-only (SSM fp32 from config, KV scale 1.0) | fresh |
| `s1_b` | vLLM 0.32 INT4 | target-only (SSM fp32, calibrated KV scales) | fresh |
| `s1_d6` | SGLang INT4 | DFlash W6, S16 production (fp16 SSM, fp8 KV) | fresh |
| `s1_d6fr` | SGLang INT4 | `s1_d6` + `SGLANG_SIMULATE_ACC_LEN=1`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` (every draft rejected, target argmax emitted). **Greedy only**: forced reject turns T>0 into argmax | fresh |
| `s1_d4` | SGLang INT4 | DFlash W4 | fresh |
| `s1_c` | SGLang INT4 | target-only, fp16 SSM | fresh |
| `s1_c_warm` | SGLang INT4 | `s1_c` relaunched on its cache (cold vs warm, SG-CACHE-CLEAN) | `s1_c` root |
| `s1_c_nocache` | SGLang INT4 | `s1_c` + `--disable-radix-cache` (the cache-OFF reference, S-P0-01) | `s1_c` root |
| `s1_c32` | SGLang INT4 | target-only, `--mamba-ssm-dtype float32` | fresh |
| `s1_d6_r2` | SGLang INT4 | multi-boot A/A of `s1_d6` (#38009); runs before section 6 | fresh |
| `s1_c_r2` | SGLang INT4 | multi-boot A/A of `s1_c`; last, if time is left | fresh |

**Pairs.** The reference arm is first.

| Pair | Kind | What it isolates |
|---|---|---|
| a ↔ b | cross-weights | weights + KV scale |
| b ↔ c32 | same weights | engine, both fp32 SSM |
| c32 ↔ c | same weights | SSM dtype |
| c ↔ d6, c ↔ d4 | same weights | speculation |
| c ↔ d6fr | same weights | target-verify path vs plain decode |
| c ↔ c_warm | same weights | cold vs warm cache |
| c ↔ c_nocache | same weights | radix / Mamba checkpoint reuse |
| b ↔ c | mixed engine + SSM | the README's literal (b) vs (c) pair, from existing data |

Inside each SGLang arm there are two more pairs:
- **pass 1 ↔ pass 2:** the same prompts again, a radix hit through a Mamba checkpoint restore. Each row records `cached_tokens` and the restore depth; 0 → `NOT_EXERCISED`. NEAR_TIE is expected.
- **c16 ↔ c16:** two batches, an A/A.

**Measurements:**
- **M1 greedy rows.** The oracle set, serial: 512 tokens in FULL, 256 in SMOKE, top-2 margins, ignore_eos. Pass 2 repeats it on SGLang arms (not on `s1_d6fr` / `s1_c_warm`).
- **M2 concurrent rows.** The 16-prompt c16 set at 256 tokens, run twice (the A/A), on `s1_a`, `s1_b`, `s1_d6`, `s1_d4`, `s1_c` and `s1_c32`.
  - Plus one T=1.0 production-sampling batch of 2,048 tokens, labelled a **smoke garbage check**. It has no pair verdict; 16 rows cannot detect EOS-rate differences.
- **M3 teacher forcing** (engine and SSM numerics at long context; prefill path only).
  - **Traces.** `s1_a` generates three reference traces first, at production sampling (T=1, top_p 0.95, top_k 20, fixed seeds, ignore_eos, natural EOS position recorded):
    - 2 × (8k prompt + 30,000 tokens);
    - 1 × (90k prompt + 30,000 tokens), reaching 120k context.
    - SMOKE uses 2 × (8k + 4,000) only.
  - **Why FP8 makes them.** A trace generated on an INT4/SGLang arm would sit on that arm's own argmax path and favour the INT4 arms.
  - **Scoring.** Every later arm scores all three, one request at a time, with top-2.
  - **Reported per pair**, in bins 0-2k / 2-8k / 8-16k / 16-30k of the continuation and 90-100k / 100-110k / 110-120k of context:
    - mean and max |Δ logprob|;
    - rank-1 agreement;
    - ΔNLL;
    - the |Δ(top1−top2)| distribution, which feeds T_pair in §3.2.
  - **Limit.** M3 never runs speculative code. For speculation pairs it is a server-equivalence sanity check (`NOT_APPLICABLE_PREFILL_ONLY` for DFlash state). The DFlash long-context test is M4.
- **M4 decode vs prefill** (state drift in the real decode / verify path).
  - **Generation.** `s1_d6`, `s1_d4`, `s1_c` and `s1_c32` each run the same protocol: production sampling at c16, 16 × (8k prompt + 15,000 tokens), fixed seeds, ignore_eos, with output logprobs. Decode logprobs at T=1 are raw. SMOKE: `s1_d6` and `s1_c`, 16 × (8k + 4,000).
  - **Re-score.** The same server re-scores each trace by prefill (teacher forcing).
  - **Per bin:** mean |Δ| and its SE, counts of |Δ| ≥ 1 / 2 / 4 nat, and entropy.
  - **`STATE_DRIFT` on an arm when either holds:**
    - **gross:** |Δ| ≥ 4 nat at ≥ 3 positions;
    - **excess** over the length-matched target-only baseline (`s1_c`): bin mean > baseline mean + 3 SE and > 0.02 nat, in a bin where the baseline itself is clean.
  - **Comparisons:**
    - c vs c32 is the **decode-time SSM fp16 accumulation** effect;
    - d6 / d4 vs c is the speculation effect;
    - `s1_d6fr` is greedy only and gets M1.
- **M5 rank probe.** Per rank: lm_head dtype, Mamba state dtype, FP8 k/v scales and patched-module shas (S-P0-01 item 4, S-P2-02). SGLang loads the calibrated scales (k 0.028-0.051, v 0.021-0.254) even though its log says 1.0.
- **M6 tool-call probe.** One `/v1/chat/completions` tool-call probe per arm (about 10 s, parser validity).
- **M7 cache-isolation verdict** (§3.3).

**Decisive number.** The decisive pair of the chain (§3.2), with its first high-margin divergences, plus the M3 and M4 drift bins up to 120k.

**STOP / NO-GO:**
- cache not honoured → `INVALID`;
- recurrent-state corruption: STATE_FLIP or STATE_DRIFT in a same-weights pair → NO-GO for the arm under test;
- repeated garbage.

**SMOKE subset.** `s1_a` (traces), `s1_b`, `s1_d6` (pre-flight, M1, M2, M4), `s1_d6fr` (M1), `s1_d4` (M1), `s1_c` (M1, M2, M4); 8k × 3 + 40k × 1 at 256 tokens.

### 4.2 Section 2: DFlash width × streams (FULL_B; SMOKE: W4 S20 at 8k)

**Arms** (shared warm cache):

| ID | Width | Max-running / slots / graphs | Bands |
|---|---|---|---|
| `s2_w6s20` (control) | 6 | 21 / 84 / 1..21 | 8k, 40k, 90k, 116k distinct; EMS replica short / long / XL; parity rows |
| `s2_w6s20_aa` | 6 | same | 8k, 40k: the A/A before any section-2 decision |
| `s2_w4s20` | 4 | 21 / 84 / 1..21 | 8k, 40k, EMS-replica long; parity rows |
| `s2_w6s16` | 6 | 17 / 68 / 1..17 | 8k, 40k; parity rows |
| `s2_w4s16` | 4 | 17 / 68 / 1..17 | 8k, 40k; parity rows |
| `s2_w5s20` or `s2_w7s20` | 5 / 7 | 21 / 84 | 8k, 40k; parity rows |
| `s2_best_long` | the best challenger | as that arm | 90k, 116k (if it is not the control) |

**Branch rule.** W7 runs if W6 S20 beats W4 S20 by ≥ max(5 %, 2× the A/A spread) on the mean of 8k and 40k; otherwise W5 runs.

**Equal-N windows.** For each band, N = floor((pool − margin) / (band + growth + 2.9k reservation)). Here growth is the tokens generated during settle + window + 60 s, and pool is the control's resolved `max_total_num_tokens`. Every compared arm uses the same N; an arm whose pool cannot hold it records `NOT_EQUAL_N`.
- At 8k, N = 20 for S20 and 16 for S16. At 40k, N ≈ 16.
- A retraction in a window → `INCONCLUSIVE_RETRACTION` (numbers kept).
- The "offered 20 at 40k" admission question belongs to the section 4 census.

**Parity.** Every challenger runs 8k × 3 greedy rows (256 tokens), compared with the control's rows in the same notebook (same weights, §3.2 rule). Without it, W5/W7 would have parity `NOT_EXERCISED`.

**Decisive numbers:** accepted-output tok/s (the realtime decode counter), tokens per step, verify-step ms and actual running p50.

**STOP:** XL active collapses, net tok/s not better (beyond the A/A spread), or parity fails.

**Expected:** The PDF's step model (26 ms + 0.94 ms per verify row) caps width and stream changes at about 745 (W6) and 880 (W4) tok/s. Reaching 1500 needs section 3 and the later host-staged all-reduce branch.

### 4.3 Section 3: TP4 collective critical path (FULL_B; SMOKE: one CTA arm + profile)

**Arms:** fresh launches on the warm cache, each with `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV`.

| ID | NCCL env |
|---|---|
| `s3_ctrl` | none (control) |
| `s3_cta2`, `s3_cta4`, `s3_cta8` | `NCCL_MIN_CTAS = NCCL_MAX_CTAS` = 2 / 4 / 8 (CTA 2 may equal the default; then it doubles as an A/A) |
| `s3_ring`, `s3_tree` | `NCCL_ALGO=allreduce:ring` / `allreduce:tree` |
| `s3_simple`, `s3_ll` | `NCCL_PROTO=Simple` / `LL` |
| `s3_nthreads512` | `NCCL_NTHREADS=512` (last) |

- ALGO uses the per-collective form, so the three 11.9 MB all-gathers stay unconstrained.
- LL128 is never used.
- NTHREADS already defaults to 512 on recent GPUs.

**Per arm:**
- **8k window** at N = 20.
  - Decisive step statistics are p50 / p95 / mean of the unprofiled **client inter-chunk intervals** over the whole window (about 400 steps × 20 streams).
  - Ranking uses mean step ms at the fixed running count, and tok/s normalised by tokens per step.
- **Separate 30-step profile** for the breakdown: NCCL, Marlin, GDN, attention and drafter GPU time.
- **Env proof.** The NCCL INFO lines must show the variable was read ("set by environment", or the channel count for CTAs). The trace's NCCL kernel names (algorithm and protocol in the name, e.g. `…_RING_LL`) and grid must match. Otherwise → `INVALID_ENV_NOT_APPLIED`. SMOKE's `s3_cta4` confirms which lines exist.

**Confirmation.** Gains are claimed only from independent confirmation windows: the two best arms at 40k and 90k, plus a control re-run at 40k in the same session. The control records cold-prefill seconds too.

**STOP:** slower end to end, a hang or graph timeouts. Never forced P2P, PCIe-IPC, `NCCL_SHM_USE_CUDA_MEMCPY` or `NCCL_SHM_DISABLE`.

### 4.4 Section 4: capacity at 90k / 120k (FULL_B; SMOKE: 90k census at 84)

**Arms:**

| ID | Slots / max-running | mem-fraction-static | Notes |
|---|---|---|---|
| `s4_m84` (control) | 84 / 21 | 0.83 | |
| `s4_m68` | 68 / 17 | 0.83 | |
| `s4_mem080` | 84 / 21 | 0.80 | |
| `s4_mem086` | 84 / 21 | 0.86 | Gated by an 8k probe first: TP0 was already near OOM at 0.83, so an OOM is an expected-negative result, not ERROR |
| `s4_m88`, `s4_m92` | 88 / 22, 92 / 23 | | Only when the control is **Mamba-bound** by the direct gauges |
| `s4_lowcap` | | | Optional negative test (`--max-total-tokens 300000`): the census must report the cap |

- **Mamba-bound** means Mamba available slots < the per-request need. **KV-bound** means `kv_available_tokens + kv_evictable_tokens` < next prompt + reservation.

**Census loads.** Between loads: wait_idle, `/flush_cache`, and a baseline snapshot.
- distinct 90k, offered 24;
- distinct 116k, offered 24;
- on the control, shared groups: 4 groups × 4 streams on one ~85k prefix each.

The census offers the full logical count. It never caps streams on the client, as EMS's `kv // 100k` did; that cap is the source of the "7 running at 90k" number.

**Plateau.** A plateau needs all of the following, held for ≥ max(60 s, 1.5 × band / 1300 tok/s): about 105 s at 90k and 135 s at 116k.
- the TP0 `prefill_compute` counter flat;
- no new `Prefill batch` line;
- the queue unchanged;
- a blocking reason recorded from the gauges (KV-bound / Mamba-bound / all admitted).

`max_wait` = offered × band / 1300 + 120 s.

**Recorded:**
- resolved max-running and the KV pool;
- actual running p10 / p50 / p90 (from log lines) and the queue;
- KV and Mamba usage, available and evictable;
- retractions;
- "Can not alloc mamba cache" and OOM;
- tok/s at the plateau (labelled **hook-on**: the counter hook runs on section-4 arms for natural #43567 exposure, i.e. optional-alloc failures and stale candidates per 1,000 boundaries).

**STOP:** "Can not alloc mamba cache", OOM, or repeated prefix recompute. Slots are never cut only to gain KV.

### 4.5 Section 5: prefill / decode interference (FULL_B; SMOKE: one D2 load)

**Arms:** `s5_c2048`, `s5_c4096` (control), `s5_c8192` (`--chunked-prefill-size`), plus `s5_pdi1` and `s5_pdi4` (`--prefill-decode-interval` 1 / 4, chunk 4096).

**Why the PDI arms.** In 0.5.21, prefill always pre-empts decode, a chunked prompt's chunks run back to back, and `enable_mixed_chunk` is False. A cold 90k prompt therefore freezes every resident for about 60-70 s. Chunk size alone cannot trade prefill against decode, while the interval can. `--enable-mixed-chunk` is optional, if time remains. Both values are recorded from `/server_info`.

**Loads** (residents are distinct 8k prompts). Before each load: `/flush_cache`, idle, snapshot.

| Load | Residents | Injected | Notes |
|---|---|---|---|
| D0 | 20 decoding | none | |
| D1 | 19 | 1 warm 90k | Order: flush, warm-up, then residents. The injected request must show `#cached-token` ≥ 0.99 × prompt (64-aligned), otherwise `INVALID_WARM_MISS` |
| D2 | 19 | 1 cold 90k | |
| D3 | 16 | 4 cold 90k | |

**Windows.** One fixed wall window per load, identical across arms: injection → the latest completion seen on the 4096 control + 30 s (or a fixed amount of work), preset and recorded.

**Measured:**
- **Starvation (decisive).** Per-resident max inter-chunk gap, stall seconds (gaps > 5 × the D0 median step), resident tokens decoded during the prefill phase, and seconds the TP0 decode counter stays flat. Starvation = max gap > 5 s **or** stall > 20 % of the window → the README's STOP.
- **Also:** decode TPOT p50 / p95 (secondary), decode tok/s over the fixed window (the Y11 mixed denominator), prefill seconds per injected request, waiting max and retractions.

**STOP:** worse mixed-window throughput, or decode starvation.

### 4.6 Section 6: correctness backports (FULL_A; SMOKE subset)

**Rules for every arm** (from E17 rev 2 and its audits):
- every arm is a fresh server;
- the patch state is checked before launch;
- the hook self-test must show the arm's target set live on all four TP ranks, with identical counters;
- KL uses full-vocab next-token scoring (`token_ids_logprob` over the whole vocab), float64, fixed tolerance 5e-3; the reference is valid only if its A/A ≤ 1e-3 and its flushes are valid;
- the evidence chain comes from one cycle and one seed rid, persisted atomically per cycle.

**Verdicts:**
- **PASS** = the unpatched chain **and** the patched arm clean **and** the trigger evidence present in the patched arm itself (same arm, same rid). Missing trigger evidence → `INCONCLUSIVE_NOT_EXERCISED`.
- **FAIL** = any wrong restore, assert, slot leak, scheduler traceback, or `Mamba release hit lock_ref=0` on a patched arm.

| ID | Patch | What it shows | Trigger evidence required in the patched arm |
|---|---|---|---|
| `s6_unit` | overlay copies | GPU unit tests with no server and no venv change. The script copies `site-packages/sglang` to `/tmp/e24/overlay_{none,41736,41737}`, patches the copies, puts each first on `sys.path`, and runs with the server env, a timeout and killpg. **#41736:** a NaN in request 0's last block row must not reach request 1, through `DFlashGroupedConv._convolve`. **#41737:** the upstream NaN-score selector test. It also reports the max \|Δ\| between versions on finite inputs. | the imported module `__file__` + sha recorded |
| `s6_ffu` / `s6_ffp` | none / 43567+43559 | Small-slot target-only (`--max-mamba-cache-size 4 --max-running-requests 1 --mamba-track-interval 128 --chunked-prefill-size 2048 --cuda-graph-bs-decode 1`), `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1`, upstream class-1 KL recipe (W44, W42, P255). Unpatched: optional-alloc failure → stale donation → probe hit at 192 → KL > tol. | the seed rid has `optional_alloc_failed ≥ 1` in every valid cycle; the probe hits `cached 128`; E17 safety cases (EOS, abort, max_new, 5× reuse, cold after flush), each with an allocator snapshot equal to the baseline |
| `s6_np` | 43567+43559 | The natural 4-slot path (no forced failure). The unpatched half is E17 run 1's recorded crash (`AssertionError: Can not alloc mamba cache`); the README lists it as a measured negative, so it is not repeated. | `prefix_lock_released ≥ 1`; no assert; safety cases; KL ≤ tol |
| `s6_ffud` / `s6_ffpd` | none / 43567+43559 | DFlash W6, interval 64, prompt **P255** (any accept ≥ 1 crosses 256): the speculative in-place fallback chain. | as FF-P, on the spec path |
| `s6_gu` / `s6_gp` | none / 43029 | NGRAM: `--speculative-algorithm NGRAM --speculative-num-draft-tokens 4 --speculative-ngram-max-bfs-breadth 1 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 32 --max-running-requests 4 --cuda-graph-bs-decode 1 2 3 4 --random-seed 1 --grammar-backend xgrammar`, with the draft-model flags and `--min-free-slots-delay` dropped. xgrammar regex truncation where accepted > committed; upstream recipe; delta ≥ 3 nat at `cached == I` unpatched. Budgets the one-time NGRAM C++ JIT build. **Fallback** if NGRAM fails to launch: DFLASH W6 under non-lazy `extra_buffer`, same recipe. **G-L** (the production lazy strategy) runs when time remains. | `grammar_truncated_steps ≥ 1` and `false_crossing_possible ≥ 1`, `false_crossing_returned == 0`, all delta < 3, `resumed_at_checkpoint > 0` |
| `s6_sent_u` / `s6_sent_p` | 41737 / 41737+41736 | Eager DFlash W6 (`--disable-decode-cuda-graph`; the log must show no capture lines). Every request runs at T=0 (top_k 1). The sentinel is submitted first and streams its first token before the other 7 / 15 are submitted, so it sits at batch index 0. NaN is injected into the sentinel's last draft row. Each concurrency (1 / 8 / 16) also runs without injection. #41737 is in both arms because an unpatched #41736 leak creates all-NaN neighbour rows that the unpatched selector would walk out of. | injection count ≥ 1, identical on 4 ranks, a non-sentinel neighbour logged |
| `s6_nan_u` / `s6_nan_p` | none / 41737 | Eager DFlash W6, T=0, NaN injected into slot 0 of greedy selector rows. Unpatched: an out-of-row pick. Patched: none, and greedy rows match the uninjected run under `compare()`. | injection count ≥ 1 on 4 ranks, greedy rows |
| `s6_combined` | all five | Production S16 config, after `s1_d6_r2`. | **PASS** = every M1 row EXACT / NEAR_TIE against `s1_d6` (A/A-calibrated), 0 error-regex hits, and the tool probe OK; missing rows → INCONCLUSIVE |

**SMOKE subset:** `s6_unit`, `s6_ffu`, `s6_ffp`.

## 5. Run order and budgets

`step(est_min)` skips a step with `SKIPPED_BUDGET` unless `est_min + 5` minutes remain. Each estimate includes the launch (cold about 9-10 min, warm about 1.5 min), prefill at about 1,300 tok/s, settle, windows and scoring. Each arm records its actual minutes.

**SMOKE (210 min).**

| Step | Minutes |
|---|---|
| Setup | 8 |
| `s1_a` (traces) | 19 |
| `s1_b` | 16 |
| `s1_d6` | 21 |
| `s1_d6fr` | 12 |
| `s1_d4` | 11 |
| `s1_c` | 21 |
| `s6_unit` | 6 |
| `s6_ffu` + `s6_ffp` | 16 |
| W6 S20 control (from the `s1_d6` cache): 8k window, profile, 90k census at 84, one D2 load | 25 |
| `s2_w4s20` (8k + parity) | 7 |
| `s3_cta4` with profile | 8 |
| Report | |

The planned total is about 170 min.

**FULL_A (480 min).**

| Step | Minutes |
|---|---|
| Setup | 8 |
| `s1_a` (three traces) | 42 |
| `s1_b` | 24 |
| `s1_d6` (incl. M4) | 38 |
| `s1_d6fr` | 17 |
| `s1_d4` (incl. M4) | 35 |
| `s1_c` (incl. M4) | 40 |
| `s1_c_warm` | 13 |
| `s1_c_nocache` | 19 |
| `s1_c32` (incl. M4) | 38 |
| `s1_d6_r2` | 19 |
| `s6_unit` | 6 |
| `s6_ffu` + `s6_ffp` | 16 |
| `s6_np` | 8 |
| `s6_ffud` + `s6_ffpd` | 15 |
| `s6_gu` + `s6_gp` (+16 if the fallback is needed) | 17 |
| `s6_sent_u` + `s6_sent_p` | 16 |
| `s6_nan_u` + `s6_nan_p` | 12 |
| `s6_combined` | 10 |
| `s1_c_r2` | 17 |
| G-L (if time remains) | |
| Report | |

The planned total is about 420 min.

**FULL_B (690 min).**

| Step | Minutes |
|---|---|
| Setup | 8 |
| `s2_w6s20` (first launch, cold) | 57 |
| `s2_w6s20_aa` | 18 |
| `s2_w4s20` | 23 |
| `s2_w6s16` | 19 |
| `s2_w4s16` | 19 |
| W5 / W7 | 19 |
| `s2_best_long` | 24 |
| Section 3 (incl. confirmations) | 139 |
| Section 4: `s4_m84` 35, `s4_m68` 26, `s4_mem080` 26, `s4_mem086` 29, conditionals | 116 |
| Section 5 (5 arms × about 25) | 125 |
| A/A rounds, at most 3 | |
| Report | |

The planned total is about 570 min.

### 5.1 Paired ARC sample: a separate session, built after FULL_A

The README asks for "a paired ARC sample (24-64 tasks, same budget)" after the oracle. It is not in these notebooks, for three reasons:
- **Cost.** It takes 1-2 h per arm per 24 tasks (R3K2 rate: 174.5 s per task).
- **Power.** At 20 % discordance, McNemar on 24-64 tasks detects only gaps of about 15-20 points; a 6.5-point gap needs about 370 tasks.
- **Order.** The README orders it after the divergence is located.

**When it is built.** Once FULL_A shows the decisive pair, using:
- the S1 Poetiq solver (E15P SGLang shim / E18 vLLM 0.32 shim);
- `E10_POETIQ_MAX_ITERATIONS=1`;
- offline scoring with the evaluation solutions.

**The gap itself needs checking:**
- "about 32" is the mean of five vLLM FP8 S1 LBs (32.39);
- E11-06's 29.17 changed the prompt and the width at the same time;
- E15P's 26.94 is still logged as pending (submission 57644341) and equals v1.4.4's LB exactly.

So the gap is -3.2 to -5.5 points, not 6-7, if 26.94 holds.

## 6. Departures from the README, and why

1. **FULL is two notebooks (PART A / B).** It does not fit 11.5 h (§1).
2. **Section 1 has extra arms.**
   - `s1_c32` and the b ↔ c32 / c32 ↔ c split: vLLM 0.32 runs SSM fp32 (checkpoint config) and SGLang production runs fp16, so b ↔ c mixes engine and SSM precision. b ↔ c is still reported.
   - `s1_c_warm` and `s1_c_nocache` (cold/warm and cache-OFF, as the PDF asks) use `s1_c`'s cache by design, not a fresh one.
   - `s1_d6_r2` and `s1_c_r2` are multi-boot A/A runs.
3. **Calibrated margin rule** (§3.2), instead of a single 0.5-nat FAIL for every pair. 0.5 nat is only 4 bf16 quanta, and the cross-kernel pairs need their own noise floor.
4. **The ≥ 30k test is two tests: M3 and M4.**
   - M3 scores FP8-generated traces up to 120k context.
   - M4 checks production-sampled c16 decode against prefill.
   - Generating 30k tokens on every arm would cost about 20 min per arm, and first-divergence greedy is blind after the first near tie.
5. **Forced reject uses SGLang's own switch** (`SGLANG_SIMULATE_ACC_LEN=1` + `real-draft-token`) and is greedy only.
6. **Distinct prompts by default, equal-N windows, EMS replica windows on the control.** The README baseline (597 / 536 / 364) used shared reference blocks, with an "8k" band median of 3,852 tokens. Offering 20 distinct 40k streams has no steady state, because retraction starts in about 2 min.
7. **Screening for time.**
   - Section 2 runs 90k / 116k on the control and the best challenger only.
   - Section 3 runs 40k / 90k on the two best arms only.
   - Section 4 offers 24, which subsumes 12 / 16 / 20 when KV binds first, plus 4 × 4 shared groups.
8. **Section 3 uses `allreduce:`-scoped ALGO; NTHREADS runs last.**
9. **Section 4 runs 88 / 92 only when the control is Mamba-bound** by the direct gauges. At distinct 90k, KV is expected to bind first.
10. **Section 5 adds `--prefill-decode-interval` arms and a starvation measure.** Chunk size cannot trade prefill against decode in 0.5.21, and TPOT p95 does not see a single 60 s freeze.
11. **Section 6.**
    - The natural unpatched #43559 arm is not repeated (README measured negative); E17 run 1 is the unpatched evidence.
    - The sentinel arms carry #41737 in both arms (§4.6).
    - #43029 has an NGRAM recipe with a DFLASH fallback.
12. **The paired ARC sample is a follow-up session** (§5.1).
13. **Declared `NOT_MEASURED`:** coverage, adoption per GPU-hour, and the agent end-to-end denominator. They need the Duck / Poetiq harness.

## 7. Files

**`src/`:**
- `e24_lab.py`;
- `e24_sglang_patches.py`;
- `e24_hook.py`;
- `e24_unit_41736_41737.py`;
- `patches/pr*_0521.diff` (provenance of the five ports);
- `harness/` (self-contained fake SGLang and vLLM servers, mock inputs, the local runner);
- `test_*.py`.

**`Experiment Notebooks/`:**
- `Build/build_e24_notebook.py`, which writes the three notebooks, a `kernel-metadata.json` per variant, and `plan_argvs_e24.json`;
- `Build/build_cpu_kit_e24.py` plus `Build/cpu_kernel_e24/`, the Kaggle CPU kit:
  - Linux unit tests;
  - the offline SGLang and vLLM installs through the notebook's own CELL 2;
  - flag checks of every runtime argv, both conditional branches forced;
  - patch apply / revert on the installed package;
  - hook target AST checks and environ checks;
  - FAKE notebooks against the real mounts;
  - fault modes;
  - kit size < 400 kB.

**`Experiment Results/`:** the local build and validation report.

Every notebook stays under 400 kB; commits above about 407-444 kB never start. Every GPU run needs the user's go.
