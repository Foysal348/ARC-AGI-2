# Audit: Q27-XHIGH-v1.ipynb (Qwen3.8-27B-FP8, xhigh, 4×L4, vLLM 0.19.0)

Audited file: `D:\ARC AGI 2\Research Base Experiment\Q27-XHIGH-v1.ipynb`. It is byte-identical to a fresh build from `make_q27x.py` (sha256 `cde10269…89acc`).

How locations are cited: `CELL n` is the notebook's own cell header. The notebook cell index is `CELL n` + 1, because index 0 is the markdown header. `L` is the line within that cell's source; `B` is the line in `make_q27x.py`.

Scratch tests are in `…\scratchpad\audit_q27x\`: `t_prompts.py`, `t_parse.py`, `t_fence.py`, `xh_harness.py`, `t_kv.py`, and their `.out` files.

## Verdict

- **12-task commit run (EVAL12): GO-with-fixes.** Nothing in the code will crash or hang a normal run. But the answer parser silently drops correct programs and grids for some answer layouts (F1). Without F1 fixed, the 12-task score understates the method. Without F4 and F5, the run cannot measure the things it is meant to measure (MTP acceptance, KV pressure, why tasks failed). All four fixes take under 40 lines.
- **Hidden-test submission: NO-GO for now.** It becomes GO once these three conditions hold:
  1. F2 is fixed. Today any exception in the 11-hour loop ends the run with "Notebook Threw Exception" and a score of zero, even though submission.json is valid.
  2. F1 is fixed.
  3. The commit run shows that MTP works and that the projected 240-task time is ≤ ~10 h. The rerun needs about 250 tok/s sustained (section 7).

## Findings (ranked)

### P0 — will crash or waste the run

No defect in the notebook code will crash a normal run. One operational precondition, which is not in the notebook itself:

**P0-ops. The docker image must be pinned to the Python 3.12 build.** `PIV_Q27_V31_S12_RESULT_2026-10-04.md` says: "Kaggle's default moved to Python 3.13 on 2026-10-03; the cp312 wheelhouse cannot install vLLM there". CELL 2 L15 would then fail with `CalledProcessError`. The notebook carries no image pin and no Python-version check.
- Status: VERIFIED from our own result document.
- Fix:
  - Push with `docker_image` pinned to the `sha256:37c64f7d…` image, `machine_shape` NvidiaL4, and internet off.
  - Attach the competition, the model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` and the wheelhouse dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`.
  - Optionally add `assert sys.version_info[:2] == (3, 12)` at the top of CELL 2.

### P1 — wrong results or large inefficiency

**F1. The fenced-block regexes desynchronise and silently drop the program or the grids.**

Where: CELL 4 L7–8 and L13–17 (B243–244, B249–253).

```python
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_JSON_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.S)
```

Mechanism: a fence whose label is not allowed (for example `json`, `text`, `plaintext`, `Python` or `python3`) does not match as an opener. Its closing fence then matches as an opener, and every later fence is paired the wrong way. Results from `t_fence.out`, which runs the notebook's own `parse_answer`:

| answer layout | code | grids |
|---|---|---|
| code, then json (the instructed order) | found | found, but **only through the greedy fallback**: `_JSON_RE` returns `['\n']` |
| a ```` ```text ```` grid shown first, then code, then json | **MISSED** | found |
| json first, then code | **MISSED** | found |
| code, json, then a trailing sentence containing `{1: 2}` | found | **MISSED** |
| a ```` ```plaintext ```` example, then code | **MISSED** | found |
| ```` ```Python ```` or ```` ```python3 ```` label (`t_parse.out` B, C) | **MISSED** | — |

When the code is missed, a train-verified program is lost and the task falls back to the direct grid or to the test input.

In the instructed order the JSON fence is never matched by `_JSON_RE`. It is found only by `\{\s*"test_outputs".*\}`, which is greedy up to the last `}` in the text. Any later `}` therefore loses the grids.

The harness cannot see this because its answers always follow the instructed order.

- Status: VERIFIED by execution.
- Fix: tokenise fences once, then filter by label.

```python
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
blocks = [(lab.strip().lower(), body) for lab, body in _FENCE_RE.findall(text)]
py = [b for l, b in blocks if l in ("", "python", "py", "python3")]
js = [b for l, b in blocks if l in ("", "json")]
```

Also try `json.loads` on each balanced `{…}` that starts at `"test_outputs"` (use a brace counter, not a greedy regex).

**F2. There is no exception containment in CELL 6 or CELL 7. In the rerun, any runtime error scores zero.**

Where: CELL 6 L147 `run_scheduler()` (B562); CELL 7.

`xh_harness.py crash` makes `engine.step()` raise `EngineDeadError`. Result:
- CELL 6 raises.
- submission.json exists and is valid (identity, plus whatever was written before the crash).
- No run_report or bundle is produced.

On Kaggle the run stops at that cell. In a competition rerun, a notebook that raises is scored as a failure regardless of submission.json. Our own SUB v1.3 notes this in CELL 0: "a notebook that raises scores nothing whether or not this file exists". The design summary's claim that the "submission.json is rewritten after every task" therefore protects nothing on the hidden test.

Over about 11 hours, plausible triggers include:
- the engine-core process dying (OOM in a rare long prefill, an NCCL timeout, an MTP draft fault);
- `add_request` raising.

- Status: VERIFIED (crash scenario); the Kaggle scoring behaviour is INFERRED.
- Fix:

```python
try:
    run_scheduler()
except BaseException as e:
    SCHED["fatal"] = repr(e)[:500]; traceback.print_exc()
    try: collect_verifications(block=True)
    except Exception: pass
finally:
    write_submission()
```

Make CELL 7 tolerant as well (wrap it in try/except). Do not re-raise when `RERUN` is set.

### P2 — minor, measurement gaps, inefficiency

**F3. Concurrency and KV capacity: the design summary's claim is wrong, and the first wave will be preempted.**

Where: CELL 5 L32–39 (B400–407); CELL 1 L29 and L35.

The summary says "KV capacity cannot be read from the front-end process, so concurrency is a fixed 16". That is false:
- vLLM copies `num_gpu_blocks` into the front-end's `cache_config` during the READY handshake: `v1/engine/utils.py` 1121–1123, `num_gpu_blocks += msg["num_gpu_blocks"]; cache_config.num_gpu_blocks = num_gpu_blocks`.
- The hybrid `block_size` is set in the front-end by `HybridAttentionMambaModelConfig`, called from `VllmConfig.__post_init__` (config/vllm.py 1612–1613).

So the `CONCURRENCY` formula is live. `t_kv.py` reproduces vLLM's formulas, calibrated on a real log of this model (v2_4G journal: block 784, 10.21 GiB, "GPU KV cache size: 166,992 tokens", "20.80x"; the script reproduces 166,992 and 20.8× exactly). It gives:

| | block size | pool blocks | cell-5 `kv_tokens` | `CONCURRENCY` | real attention capacity |
|---|---|---|---|---|---|
| no MTP | 784 | 853 | 668,752 | 16 | 631k tokens at 16 running |
| MTP (2 spec tokens) | 800 | ~778 | ~622k | **15** | **~514k at 15 running** (~34k per sequence) |

What MTP costs in capacity:
- It adds one attention layer, so the group size rises to 17.
- With 2 spec tokens, the mamba page grows from 801,792 to 812,032 bytes, which sets the block size to 800.
- Every request needs 3 mamba groups × (1 + 2) = 9 pool blocks.

`num_gpu_blocks × block_size` counts blocks that the mamba groups consume as if they held attention tokens.

Peak demand:
- RERUN first wave (all admitted together and growing in lockstep): 15 × (1.4k prompt + 45k) ≈ 697k > 514k.
- EVAL12: 12 × (2.9k + 45k) ≈ 576k against ~536k.

So preemption will happen in the first wave. Because prefix caching is off, each preempted request is recomputed in full (~35k-token prefill). Re-admission waits for full-length fit (`scheduler_reserve_full_isl=True`, scheduler.py 733), so there is no thrashing. The estimated cost is a few percent of throughput.

- Status: VERIFIED (vLLM code and calibrated arithmetic); MTP memory deltas INFERRED.
- Fix:
  - Correct the documentation.
  - Optionally stagger the first admissions (for example, admit 4 every ~60 s), or accept the preemption.
  - fp8 KV (as NguyenThanhNhan used) would double capacity. On sm89 it needs a non-FlashAttention backend, which is untested here.

**F4. No vLLM statistics, so the commit run cannot measure MTP or KV pressure.**

Where: CELL 5 L2–8. `LLM.__init__` forces `disable_log_stats=True` unless it is passed (entrypoints/llm.py 272–273). Without it there are no acceptance-rate, KV-usage, running/waiting or preemption logs. Russell Kirk's comment in the discussion ("I set it up incorrectly and didn't get any gain") is exactly the failure this would hide.

- Status: VERIFIED.
- Fix: add `"disable_log_stats": False` to ENGINE_CONFIG. The `LoggingStatLogger` then prints every 10 s, including "SpecDecoding metrics", "GPU KV cache usage" and "Running/Waiting".

**F5. The answer text and code are never saved.**

Where: CELL 6 L53–59 and L73–79. Records store only `has_code`/`has_grids`, so F1-type misses, truncations and NameErrors cannot be diagnosed after the run.

- Fix: write `OUT/answers/<tid>.txt` containing `text` plus the last ~3k characters of thinking. That is about 30 KB per task.

**F6. Every phase transition re-prefills the whole context.**

Where: CELL 6 L30–41 (B445–456). The forced-close, continue and drain paths each submit a new request of up to ~57k tokens, with prefix caching off.
- On 4×L4, each re-prefill is about 7–10 s of engine time (≈3e15 FLOPs) and stalls decode for every other sequence (a 32,768-token chunk per step).
- With about 60–70% of tasks force-closed at xhigh (median ~64k > 45k), the rerun incurs ~150–200 re-prefills, roughly 20–30 min of its ~10.5 h.

- Status: INFERRED (cost); the mechanism is VERIFIED.
- Fix: accept it for v1, or force the stop tokens inside the same request with a V1 logits processor. Prefix caching would need mamba "align" mode; `Qwen3_5MTP` rejects "all" (qwen3_5_mtp.py `__init__`).

**F7. The MTP retry does not wait for GPU memory to be released.**

Where: CELL 5 L19–29 (B387–397). If speculative init fails after the workers have allocated memory, the immediate `LLM(**ENGINE_CONFIG)` can hit vLLM's start-up free-memory check. The engine-core and worker shutdown is asynchronous. `torch.cuda.empty_cache()` in the front-end does nothing, because CUDA is never initialised there.

- Status: INFERRED.
- Fix: poll `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` until every GPU is below 1 GB, or for up to 120 s, before retrying. Also record `spec_error` to a file.

**F8. Phase 1 ending in EOS before `</think>` produces a malformed phase-2 prompt.**

Where: CELL 6 L33 (B448) and L127–133. With finish reason 'stop' and no `</think>`, phase 2 starts the forced close. vLLM includes the EOS token in the final delta (scheduler.py 1634–1642 keeps the stop token). `xh_harness.py eos_unclosed` shows the phase-2 prompt tail:

`' thinking', '<|im_end|>', '\n\n', 'Considering', …, '</think>', '\n\n'`

This is rare.

- Status: VERIFIED.
- Fix: strip trailing ids in `{248046, 248044}` from `p1_ids` before building `prefix`.

**F9. Smaller parsing and verification gaps.**

Where: CELL 4 L13–15 and L87–88. All VERIFIED in `t_parse.out`.
- (a) `{"test_outputs": [[7, 6]]}` (one unwrapped grid for one test input) is rejected as a direct answer. Fix: if `n == 1 and valid_grid(test_outputs)`, wrap it.
- (b) Helper functions in an earlier python block cause `NameError` (case D). Fix: concatenate all python blocks up to and including the one with `def solution`.
- (c) If two `solution` blocks appear, only the last is tried (case J). Fix: verify each candidate block and prefer the first that passes training.

**F10. The sandbox environment is untested on Kaggle.**

Where: CELL 4 L53–65 (B289–301).
- The self-test does not import numpy. Locally, `-I` hid a user-site numpy (case G: `ModuleNotFoundError`). On Kaggle numpy should be in system site-packages (INFERRED), but nothing checks this.
- RLIMIT_AS is 4 GiB and OpenBLAS uses its default thread count on a machine with many cores.
- `preexec_fn` runs in a process that has threads.
- The child's cwd is `/kaggle/working`, so any file the model's code writes lands in the outputs.

- Fix:
  - Add a numpy self-test that asserts `train_pass`.
  - Pass `env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}`.
  - Call `setrlimit` inside RUNNER instead of through `preexec_fn`.
  - Use `cwd=tempfile.mkdtemp()`.

**F11. The drain answer budget is too small for large grids.**

Where: CELL 1 L22 (B102), `DRAIN_ANSWER_MAX_TOKENS = 4096`. A 30×30 grid via `json.dumps` costs 2,700 tokens (1,801 compact). 9 of 120 evaluation tasks need more than 4,096 tokens for the JSON alone; the maximum is 5,410 (`e3721c99`).

- Status: VERIFIED with the real tokenizer.
- Fix: raise it to 6–8k, and ask for compact JSON in the instructions.

**F12. Rerun time margin and adaptivity.**

Where: CELL 1 L43–45. The hard stop is 15 min before 12 h, measured from cell 0 rather than from kernel start. After it come blocking verification (≤ ~3 min) and the bundle, so the margin is about 10 min. There is also no budget adaptation: if throughput is low, the tail of the alphabetically sorted queue simply gets identity answers.

- Fix: hard stop at 20–25 min before the limit. Optionally lower `THINK_BUDGET` for new admissions once (remaining tasks × expected tokens) exceeds (remaining time × measured rate).

**F13. Reproducibility and documentation details.**
- Phase-2 seeds use `SEED + 7919 + len(STATE)`, which depends on timing (CELL 6 L40).
- The summary says `n_predict = mtp_num_hidden_layers (=1)`. But `speculative.py` 327 reads `getattr(hf_config, "mtp_num_hidden_layers", None)` from the top-level `Qwen3_5Config`, where the field lives in `text_config`. It is likely `None`. This is harmless, because `num_speculative_tokens=2` is explicit and the divisibility check (speculative.py ~566) is then skipped. INFERRED.
- The checkpoint's `generation_config.json` has temperature 1.0; the notebook uses 0.6, as russcore does. This is a design choice that the sources cannot settle.

## Checked and found correct (with evidence)

**vLLM API (0.19.0 source, identical to the extracted wheel)**
- `LLM(**ENGINE_CONFIG, speculative_config=dict)` is valid:
  - `EngineArgs.speculative_config: dict | None` (arg_utils 547);
  - `create_speculative_config` mutates the dict it receives (`.update`, 1508), and the notebook passes a copy, so `SPECULATIVE_CONFIG` stays clean;
  - the other kwargs are identical to V3.1, which is proven on Kaggle.
- `add_request(id, {"prompt_token_ids": …}, params)` follows the raw-prompt path (input_processor 234–247; still present in 0.19, with a deprecation warning). The prompt-length check is `prompt_len >= max_model_len` (input_processor 351–375); the maximum here is 57,305.
- Request ids: the internal id is `f"{external}-{uuid8}"` (input_processor 185–193), and `RequestOutput.request_id = external_req_id` (output_processor 340–352). Reusing an id would be harmless, but the notebook never reuses one.
- DELTA outputs: `token_ids = new_token_ids` in delta mode (`_new_completion_output`). `finish_reason` is the string from `FINISH_REASON_STRINGS` ("stop", "length", "abort", "error", "repetition"). `stream_interval` defaults to 1 (scheduler.py 150), so nothing is buffered. With `detokenize=False` the plain `IncrementalDetokenizer` is used (detokenizer 56–58), and it does not conflict with DELTA or spec decoding. Spec tokens are trimmed per token at stop, max_tokens and max_model_len (scheduler 1634–1642; `check_stop` 112–117).
- `abort_request(external ids)` maps them to internal ids (output_processor `abort_requests`). With no queue there is no final abort output, and any later engine output for the request is dropped ("Ignore output for already-aborted request").
- EOS: `generation_config` `eos_token_id` [248046, 248044] is merged into `stop_token_ids` (sampling_params 539–553). Explicit sampling parameters are not overridden.
- MTP for this checkpoint:
  - `qwen3_5` maps to `Qwen3_5MTP` (speculative.py 324–332).
  - The draft is built with the target `vllm_config` and prefix "" (eagle.py `_get_model`, model_loader `initialize_model`), so the module prefix is `mtp.*`.
  - `Fp8Config` reads `modules_to_not_convert` (fp8.py 161–166). Exact-prefix matching skips `mtp.fc` (quant_utils `is_layer_skipped`), and the target's `hf_to_vllm_mapper` (`model.visual.`, `lm_head.`, `model.language_model.`) does not rename `mtp.*` entries.
  - `mtp.safetensors` has 22 tensors; q/k/v/o and gate/up/down carry `weight_scale_inv`; `fc` is bf16.
  - Weight remap `mtp.` → `model.`; embeddings and lm_head are shared with the target (eagle.py `_maybe_share_*`).
  - Prefix caching off gives mamba mode "none", which `Qwen3_5MTP` accepts.
  - Async scheduling stays on for MTP (vllm.py 755–788).
  - `num_speculative_tokens=2` is accepted, with vLLM's warning about lower acceptance.
- `engine.vllm_config.cache_config.num_gpu_blocks` is set in the front-end (see F3).

**Template and tokens (real tokenizer and template)**
- xhigh renders the system line "Reasoning effort is set to xhigh…". The prompt ends `<|im_start|>assistant\n<think>\n`.
- `</think>` is a single token, 248069, and is not special.
- STOP_IDS is 25 tokens ending `[198, 248069, 271]`, the same as the template's `'\n</think>\n\n'`. The forced-close and continue prefixes are token-exact concatenations.
- Prompt lengths: evaluation max 8,712, test max 6,829, training max 8,712. No task exceeds `PROMPT_TOKEN_LIMIT` (12,224). EVAL12 prompts are 1,310–4,988 tokens.

**Scheduler (CELL 6)**
- No double finish; ids are unique; there is no busy loop. The `not active` branch exits only when the queue is empty, after the admission deadline, or after the drain, and pending verifications are then collected with `block=True`.
- The drain with thinking already closed continues the answer with cap 4,096 (`xh_harness drain_closed`).
- The hard stop aborts before finishing.
- `max_model_len` is never exceeded: phase 1 ≤ 57,280; phase-2 cap = `min(cap, 65536 - len(prefix) - 8)`.
- Missing outputs (preempted or waiting requests) are handled (`xh_harness preempt`: 12/12 finished).

**Verification and scoring**
- Train and test checks are exact; timeouts and `SystemExit` are caught; ranking and dedup are correct.
- `score_task` is the ARC per-task fraction of queries where either attempt is exact.

**Submission**
- The identity fallback covers all 240 tasks and 259 queries; writes are atomic (tmp + `os.replace`).
- The rerun overlay works: with non-identity model answers, 259/259 queries carried the model grid (`xh_harness rerun_nonid`, with `KAGGLE_IS_COMPETITION_RERUN="True"`).
- A commit run writes an identity submission, so it is not eval-only.

**Parity with V3.1 and SUB v1.3**
- Same pins (vllm 0.19.0, transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2), same wheelhouse and model paths, same offline and `HF_*` environment variables, same engine settings except context length, concurrency and MTP.
- `CUDA_VISIBLE_DEVICES` is no longer set explicitly; this is harmless.
- `chat_template.jinja` is identical to the template embedded in `tokenizer_config`.

## Time and resource feasibility

- **Memory.** `max_model_len` 65,536 at 0.90 utilisation fits: one full sequence needs ~82 + 9 of ~778 pool blocks. Weights take 7.04 GiB per GPU (real log), plus ~0.11 GiB for MTP (INFERRED).
- **Throughput anchor.** Measured C1 throughput was 129 tok/s with 8 long sequences, no MTP (v2_4G journal). Estimated at 12–16 sequences: ~150–210 tok/s without MTP and ~200–350 tok/s with MTP (INFERRED). Anil's estimate of 350 is the optimistic end.
- **Commit run (EVAL12).**
  - Generation: ~45k thinking tokens per task at 13–22 tok/s per sequence, about 35–60 min.
  - Set-up: ~10–15 min (install, plus 42 s load and 374 s profile/compile in the real log).
  - Total: about 1.0–1.5 h, inside the 2.75 h hard stop.
- **Rerun (240 tasks, 259 queries).** About 40k generated tokens per task (mostly force-closed at 45k + 3k), so ~9.6M tokens. That needs about 250 tok/s sustained over ~10.5 h.
  - With working MTP it is borderline-feasible.
  - Without MTP, about 150–200 tasks get model answers and the rest get identity at the admission deadline.

## The discussion versus the notebook

| point | discussion | notebook | comment |
|---|---|---|---|
| reasoning effort | xhigh | xhigh | — |
| token budget | median ~64k; NguyenThanhNhan forced a commit at 49k | 45k thinking + 8k answer | most tasks will be force-closed, as in that submission |
| KV cache dtype | NguyenThanhNhan used fp8_e4m3 KV | bf16 KV | half the KV capacity, hence F3 |
| prefill chunking | SGLang chunked prefill 2,048 | 32,768-token chunks | longer decode stalls during re-prefill; total work is similar |
| speculative decoding | EAGLE, 3 steps | MTP, 2 tokens | reasonable choice, but unmeasured (F4) |
| passes | one | one generation per task, covering all queries | better than one per query |

## Harness gaps (`harness_q27x.py`)

1. The fake engine sets `num_gpu_blocks=None`, so the live `CONCURRENCY` path (15 with MTP) is never exercised. My `kvblocks` scenario covers it: kv_tokens 654,400 gives `CONCURRENCY` 16 at that block count.
2. Every request emits output on every step. There is no preemption, no prefill latency, no blocking `step()`, no EOS in the final delta, and no 'error' finish.
3. In the rerun scenario the "good" answers equal the test inputs ("non-identity first attempts: 0"), so the scenario cannot detect a broken submission overlay. I tested the overlay separately and it works.
4. Answers always follow the instructed order (code, then JSON), so F1 is invisible.
5. No solution imports numpy, and the harness runs on Windows, so RLIMIT_AS, `preexec_fn` and `-I` never run.
6. "MTP refused" is an instant exception raised before any GPU allocation, so F7 is invisible.
7. No exception is ever raised mid-loop, so F2 is invisible.
8. The harness runs with `THINK_BUDGET` 3000 and `ANSWER_MAX` 40000, which gives `PROMPT_TOKEN_LIMIT` 22,472 instead of 12,224. This is harmless, because the largest real prompt is 8,712.

## Extra tests run (all local, with the real tokenizer and template)

| test | result |
|---|---|
| `t_prompts.py` | prompt-length statistics as above; 0 tasks skipped; template, `</think>` and STOP_IDS checks pass |
| `t_parse.py` | 13 answer formats; failures in B, C, D, E, J and A(direct); `-I` hid numpy locally (G) |
| `t_fence.py` | F1 demonstrated (table above) |
| `xh_harness.py kvblocks` | `CONCURRENCY` path works |
| `xh_harness.py eos_unclosed` | F8 shown |
| `xh_harness.py crash` | F2 shown |
| `xh_harness.py preempt` | 12/12 finished |
| `xh_harness.py drain_closed` | drained tasks continue with cap 4,096 |
| `xh_harness.py rerun_nonid` | 259/259 queries carry the model answer |
| `t_kv.py` | capacity numbers above; reproduces the real vLLM log exactly |
| original harness, mixed and drain | still pass |

## Minimum change list before the commit run

1. F1: generic fence parser and balanced JSON search.
2. F2: try/except/finally around `run_scheduler()`, and a tolerant CELL 7.
3. F4: `"disable_log_stats": False`.
4. F5: save the answer texts.
5. Cheap extras: F8 (strip EOS), F9a (wrap a single grid), F9b (concatenate code blocks), and a numpy self-test (F10).
