# Experiment 12: the Duck REPL agent for ARC-AGI-2. Plan (2026-10-19)

**User decision (2026-10-19):**
- **Option A:** Duck-style short requests with preserved, accumulating thinking. This is an approved exception to the 60k-per-request floor; the floor still holds for Poetiq/DIRECT and every rescue.
- **Adapt verbatim** whatever can be adapted verbatim. Write new code only where ARC-AGI-2 differs from ARC-AGI-3.

## 0. Sources (all Apache-2.0; keep LICENSE + attribution)

| Source | What we take | Status |
|---|---|---|
| `da-fr/arc-agi-3-solution`, Daniel Franzen's fork of Tufa's Duck harness at upstream 7652836 | `ARC3-Inference/inference/` package (`tool_agent.py` 7.3k lines, `python_tool_sandbox.py`, `retained_functions`, `priority_scheduler.py`, `prompts.py`, utils), CONFIGURATION.md, WRITEUP | **cloned** to `_scratch/ext/dafr` |
| `Tufalabs/duck-harness` @ 7652836 (the upstream) | for diffing: which parts are Tufa's originals | to clone |
| Kaggle `dfranzen/arc-agi-3-milestone-2-solution` | exact env switches, server args, the "harness patch" as submitted | to pull (Kaggle API) |
| Kaggle `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` + thread 717133 | Tufa's own settings and write-up | to pull |
| Kaggle dataset `keithtyser/taaf-duck-qwen38-serving-v1` (ARC Raiders, 68.33) | Duck serving for Qwen3.8-27B on 4×L4 | to pull and read |
| `madarshbb/arc-qwen-3-8-27b-agent` (Duck-modelled ARC-2 agent, 15.69) | reference only: how someone mapped train pairs to the REPL; what not to do | local copy exists |

## 1. What we adopt verbatim (no edits; imported as a vendored package)

| Piece | Where | Why verbatim |
|---|---|---|
| Chat request builder + retries | `ToolAgent._chat_completion`, `_post_with_retries`, `_http_retry_delay_seconds` | tools + `chat_template_kwargs` (`preserve_thinking`, `reasoning_effort`), 5xx/timeout recovery (the bugs that killed our Q27-AGENT) |
| Tool-call parsing + recovery | `_contains_tool_call_markup`, `_recover_tool_calls_from_markup`, `_extract_leaked_tool_calls`, `_extract_reasoning_text` | Qwen3.x tool calls leaking into content / reasoning |
| Yield / resume | `_yield_resume_prompt*`, yield at N generated tokens | Duck: yield at 2,048, max output 12,288 per request |
| Context estimation + trimming | `_estimate_tokens`, `_calibrate_from_usage`, `_trim_messages_for_context`, `_drop_oldest_history_block`, hysteresis / drain switches | 93 % prefix reuse in Duck; trimming in large blocks keeps the prefix cache-hot |
| Python sandbox | `python_tool_sandbox.run_sandboxed_python` (subprocess `-I -S`, SAFE_MODULES, timeouts, output caps, middle truncation, function retention) | proven isolation and retention; kills the empty-code / hang classes of bugs |
| Admission + priority gate | `_PriorityGate`, `_max_active_streams`, Duck's C-term (token fade, `t/62000`) | slots = server limit, cache-aware handover at eviction |
| Transcripts / request logs / viewer | `_append_transcript_section`, request snapshots, `viewer/` | full trace of every request (the save-all-thinking rule) |
| Sampling + limits | T 0.7, top_p 0.95, top_k 20; max output 12,288; context 128 Ki with drain to ~57 Ki; tool output ~3,072 tokens | exactly as in the milestone notebook (verify against the pulled notebook) |

Game-only features are switched **off with their own env switches**, not deleted:
- animations;
- frame_diff / auto diff;
- UNDO / RESET;
- noop, stale and terminal guards;
- death ledger;
- world-model sections;
- level-transfer guidance;
- images (phase 1; the image path is kept for phase 3).

## 2. The only new code: a thin ARC-AGI-2 adapter (target ≤ 400 lines)

1. **Subclass `ToolAgent` → `Arc2ToolAgent`.** Override only:
   - `_build_system_prompt`: ARC-2 objective. Duck's tool-usage wording is kept verbatim where it is not game-specific.
   - `_build_user_prompt`: the task as text, using the Duck/Poetiq digit-row format.
   - `_run_python_tool`: `initial_state` = the task, and `action_handler` = submit.
   
   Everything else is inherited.
2. **REPL globals (initial_state):**
   - `train` = a list of {input, output} grids;
   - `test_inputs`;
   - helper `check(transform)` → per-pair pass/fail, shape and a wrong-cell diff. The diff format is Poetiq's `poetiq_array_diff`, which we already run.
   - The helpers are model-visible source, kept with Duck's function-retention mechanism.
3. **`submit(transform)` = the host action.** It runs the program on every train pair and every test input in the sandbox:
   - all train pairs pass → record a candidate and end the branch;
   - otherwise → return the failure diff and continue.
   
   The episode cap is set by tokens and time (Duck's budgets), not by a fixed iteration count.
4. **Two independent branches per task** → attempt_1 / attempt_2. Diversity comes from T 0.7 sampling: SGLang 0.5.21 ignores a per-request seed without deterministic mode.
   - Selection: (a) agreement of the test outputs across branches; (b) train pass.
   - Fallback: if no branch passes train, use the best partial program; else the test input. This is our existing `poetiq_two_attempts` logic.
5. **Notebook frame from our proven notebooks:**
   - CELL0 fallback submission;
   - RERUN = hidden;
   - commit = eval subset + score;
   - 12 h clock (admit / drain / hard stop);
   - the SGLang W1 engine (rev 3c: server 13 / client 12; `MAX_MODEL_LEN` 155,648);
   - the full-trace bundle.

## 2a. Rev 1: approved by the user on 2026-10-19 (blockers verified in the source; vendored code still unchanged)

The GPT reviews were checked against the code. Only real blockers were taken, and every fix stays in the adapter.

| # | Verified blocker | Source | Fix (adapter only) |
|---|---|---|---|
| 1 | `_build_system_prompt` is a module-level function called in `ToolAgent.__init__`, so a subclass method would never run | `tool_agent.py:2653, 3502` | `Arc2ToolAgent.__init__`: after `super().__init__()`, set `self._system_prompt = ARC2_SYSTEM_PROMPT` |
| 2 | The sandbox's `_refresh_state` exposes only fixed game keys; extra `initial_state` keys never reach the REPL | `python_tool_sandbox.py` `_refresh_state` | Prepend a trusted pure-Python prelude (task grids as literals + `check` / `submit` helpers) to the model's code, then call the verbatim `run_sandboxed_python` |
| 3 | The host callback is JSON IPC (`action_handler(actions, stale_after)`), so a function object cannot be submitted | `python_tool_sandbox.py:931` | `submit` runs the transform inside the sandbox and sends JSON (train results + test grids) through the existing action channel. The host re-verifies the program in a clean sandbox call (verbatim runner) and never trusts the self-report. |
| 4 | No numpy in the sandbox (SAFE_MODULES whitelist, `python -I -S`) | `SAFE_MODULES` | The sandbox stays unchanged. The system prompt requires a pure-Python list `transform(grid)`. |
| 5 | `analyze()` loads a game state file (`load_runtime_state`), and the python-tool description is game text | `tool_agent.py:6563+`, `runtime_state.py:113`, `_tools()` | Write a minimal state file with Duck's own `write_runtime_state` (session bridge); override `_tools()` with an ARC-2 tool description |

| 8 | `analyze()` writes an inline game-text nudge (world model, `current_frame`, `action(actions)`) when a reply has no tool call (user-approved 2026-10-19) | `tool_agent.py:7019-7037` | Override `_append_context_message`: for messages Duck marks as control kind "nudge", keep Duck's generic prefix (truncation / malformed JSON / markup sentences) and replace only the game tail with the ARC-2 instruction (`transform` → `check` → `submit`) + `TOOL_CALL_FORMAT_GUIDANCE` |

**Override points therefore become:** `__init__`, `_build_user_prompt`, `_run_python_tool`, `_tools`, plus the state-file bridge. The adapter is expected at about 400–600 lines (soft target).

**6. Selection order corrected:**
1. The program passes all train pairs (host-verified) with valid test grids.
2. Then agreement across branches.
3. Then distinct second grids for attempt_2.

Agreement never outranks train pass (the same logic as our Poetiq voting).

**7. Added step 3.5, a GPU smoke test** (1 task, 2 branches, 10–15 min) before EVAL30. It must show:
- structured tool calls;
- preserved thinking replayed in the next turn;
- function retention across tool calls;
- `cached_tokens` rising across turns;
- no server errors.

**Not adopted (would drift from the plan):**
- KV-aware admission (the server queue + Duck's gate stay);
- a numpy execution backend;
- forcing `reasoning_effort` per request (the template default is xhigh; the ladder is off by default);
- porting the prefix-cache patch before measuring;
- a new priority-formula design or a coverage-first branch policy beyond Duck's own all-admitted + priority gate.

**Checked, no change needed:**
- `ARC3_REASONING_HISTORY_KEY=reasoning_content` is already in the notebook settings.
- The yield is evaluated between completions; that is copied verbatim.

## 2b. Step 2 status (2026-10-19): done, gate passed

- `duck_arc2/arc2_duck.py` (538 lines): `Arc2ToolAgent` + prelude + host re-verification + driver (`run_branch` / `solve_all`, a line-by-line mirror of `play()` / `_play_inner()`) + `select_attempts`. **pyflakes clean.** The vendored `inference/` (20 files) **still hash-matches** da-fr @ 10882e33.
- **Offline smoke** (`_scratch/session_1c6db7aa/e12_smoke/smoke_adapter.py`; real Duck sandbox, our scripted code, synthetic flip task): **26/26 PASS**. It covers: task in the system prompt, no game words, globals, `check`, retention, wrong submit, the `train`-reading cheat refused, numpy refused, line numbers rebased, lambda refused, accept + host test outputs, forged marker ignored, nudge rewrite, resume opener, selection.
- **Small fixes found while coding, all inside the adapter's own text or helpers (no plan change):**
  - **Resume opener:** the milestone uses the 'full' resume mode, so `_build_user_prompt` runs after every 2,048-token yield. It now returns a continuation (Duck's own "nothing has been executed… your tool results above are still valid" wording) instead of "Start with…".
  - **Output noise:** `check` and `submit` no longer echo their return value through Duck's last-expression print. `check(..., verbose=False)` returns the list.
  - **Retention prompt was untrue:** a retained function cannot depend on the prelude globals, because the sandbox computes `_available_names` before the code runs. The prompt now says so.
  - **Wording:** "in this game" in retention notes → "on this task".
  - **Driver alignment with Duck:** the agent now receives `timeout=analyzer_timeout` (as Duck's solver passes it); the request timeout is `max(0.1, min(...))` as in Duck.
  - **Optional `max_generated_tokens` per branch:** this is Duck's `max_generated_tokens_per_game`. The default is None, as in the milestone hidden run.
- **Local-only note:** Duck passes its 65 KB sandbox bootstrap as `python -c`. Windows caps a command line at 32,767 characters (WinError 206); Linux/Kaggle allows 128 KiB per argument. The local harness therefore uses a test shim (the bootstrap written to a file). Vendored code is unchanged.
- **Open, measure in EVAL30:** in Duck a branch frees its gate slot only on accept or at its first context trim (~128k context). Unsolved branches may therefore each consume ~80–100k generated tokens before yielding. That is roughly 20M for 240 branch-0s, against ~14M per 12 h at 329 tok/s. If EVAL30 confirms this, the per-branch cap (`max_generated_tokens`) is the Duck-native lever, and its value is a user decision.

## 2c. Step 3 status (2026-10-19): CPU harness done, 10/10 scenarios (113 checks) pass

**Harness.** `_scratch/session_1c6db7aa/e12_harness/`:
- `fake_openai.py`: a scripted `/v1/chat/completions` server. It never generates or runs model code; it records every request and the in-flight count.
- `harness_e12.py`: real Duck `ToolAgent` + real sandbox; each scenario runs in a fresh subprocess, because Duck reads its settings at import.

**Scenarios and what each one checks:**

| Scenario | What it checks |
|---|---|
| solve | — |
| wrong_then_fix | the rejection opener is sent; 2 submissions |
| empty_code | + submit-before-define → NameError |
| markup_leak | leaked `<tool_call>` markup is recovered and executed |
| nudge | ARC-2 nudge text |
| http500 | two 500s retried in place; no driver failure |
| yield | continuation opener; several turns |
| trim | history trimmed; task kept; retained `transform` survives; accepted |
| deadline | stops at the deadline |
| gate | 2 slots, 3 tasks × 2 branches; ≤ 2 in flight; all first branches start before any second branch; the token cap stops unsolved branches; submission schema; agreement, distinct attempt_2, fallback |

**Common checks in every scenario:**
- `preserve_thinking` + `enable_thinking` are sent;
- sampling 0.7 / 0.95 / 20 and `max_tokens` 12,288;
- the system prompt is the ARC-2 task;
- thinking is replayed (`reasoning_content`);
- no game words reach the model;
- the transcript holds the full thinking.

**Adapter fixes found by the harness (all in our adapter):**
- **Wording:** Duck's opener note "retained functions … in this game" now reads "on this task" (`_retained_function_context`).
- **Deterministic first admission:** Duck's gate admits whoever arrives first while slots are free, so the first 12 admissions were a thread race. `solve_all` now holds every slot until all branches are queued, then releases them, so the first admissions follow the queue priority too.
- **Setup guard:** a branch that fails during setup no longer takes `solve_all` down.
- **New helper:** `build_submission` writes the full schema with fallbacks.

**Duck behaviour seen (verbatim, kept):** trimming drops whole earlier turns. With 2,048-token yields the turns are small, so trimming is gradual: message counts 22 → 9 → 7 …

## 2d. Step 3.5 build (2026-10-19): notebooks built, CPU-verified, audit zip ready; not pushed

**Builder.** `Experiment Notebooks/Build/build_e12_duck.py`:
- Source: E11-06 P1-60K EVAL14 (sha f0575ff3).
- Kept unchanged: cell 0 / 1 / 2, task load, engine.
- Added: SMOKE1 / EVAL30 profiles, the E12 block, Duck serving flags.
- New cells: CELL 3 (package + self-test), CELL 6 (solver), CELL 7 (report + bundle).

**Notebooks.**
- `Q27-XHIGH-E12_DUCK_REPL_SGLANG_SMOKE1.ipynb`: sha ac3c706c… (SMOKE1 = 2ba387bc, 2 branches, 15 min).
- `..._EVAL30.ipynb`: sha 22e09634….

**Whole-notebook CPU harness** (`_scratch/session_1c6db7aa/e12_harness/harness_e12_nb.py`). It uses a fake SGLang server that refuses to start without the Duck flags.

| Scenario | Checks | Result |
|---|---|---|
| smoke | 15/15 | eval 1/1, cached 88.9 % |
| rerun | 13/13 | the solved grids are in submission.json |
| launchfail | 4/4 | identity fallback |

**Adapter additions:**
- Per-request usage (`prompt` / `completion` / `cached`) is recorded via `_accumulate_usage_tokens` (Duck's counting unchanged).
- `difflib` was added to the prompt's module list (it is in Duck's SAFE_MODULES).
- **Template check:** the INT4 template = HF pin c0c686f9 = the FP8 template (xhigh default, `preserve_thinking` honoured).

**Audit and push.**
- Audit zip: `Experiment Review/E12_STEP3_5_BUILD_AUDIT_FOR_GPT_2026-10-19.zip`.
- Push script: `_scratch/session_1c6db7aa/push_e12/push_e12_duck.py` (`--esfer`, `--commit`, `--eval30`, `--sha`, `--check-only`).

**Open (user):**
1. Sandbox RLIMIT_AS 4 GB: Duck has none; adapter-only proposal.
2. KV 881k vs 12 × ~93k average Duck context: decide after the smoke.
3. Coverage cap `E12_MAX_GEN`.

## 2e. Rev 2 (2026-10-19): GPT audits validated, all fixes applied (user: "fix all problems")

Validation: `Experiment Review/E12 step 3.5 - GPT audits validated (2026-10-19).md`. Vendored Duck is still unchanged.

| # | Problem (validated) | Fix (adapter / builder only) | New test |
|---|---|---|---|
| 1 | `submission.json` written by concurrent branch callbacks through one `.tmp` (reproduced) | CELL 6: `_E12_WRITE_LOCK` around snapshot + write; a unique `mkstemp` temp file per write | 12 writers × 20 + a reader: no error, every read parses, no temp left (nb rerun) |
| 2 | A wrong program could forge the verification through `print` (reproduced) | One clean sandbox **per grid**: only the retained sources + that input; the value is read from the sandbox's structured `result`; the host validates and compares. Test grids always run (best-partial fallback kept). Budget 2 × 30 s; stop after a timeout | 6 attacks rejected (forged marker, patched print, forged result, silenced print, non-grid, raise); slow function bounded; tuple output accepted (smoke) |
| 3 | 900 s first-request HTTP grace ignored the deadline (reproduced: > 90 s at a 5 s deadline) | `Arc2ToolAgent.__init__`: `_http_initial_grace_used = True` (server already passed /health + smoke); Duck's normal retries kept | `deadserver`: returns in ~10 s (Windows connect cost) |
| 4 | Held gate slots released without try/finally; a setup failure made `solve_all` wait 120 s | try/finally release; failed setups are counted | `setupfail`: 2.6 s, healthy task still solves |
| 5 | No sandbox memory limit | `install_sandbox_memory_limit(4096)`: `resource.prlimit` (RLIMIT_AS) on each sandbox child right after Popen, before its code is sent (no `preexec_fn`). CELL 3 self-test: read-back, then 5 GiB rejected + 128 MiB ok; reported, the run continues if it is unavailable | Linux-only check in smoke / nb (Windows: reported unavailable) |
| 6 | Duck request snapshots off | `E12_REQUEST_LOGS` (default on for SMOKE1 only); bundled from `duck/*/requests.jsonl` | smoke: snapshots bundled and contain the replayed `reasoning_content` |
| 7 | Mirror fallback was not in the plan | Removed: next distinct candidate, else the test input | smoke |
| 8 | "Different seeds" wording | No code: SGLang 0.5.21 ignores the per-request seed without deterministic mode; branch diversity = T 0.7 sampling | — |
| 9 | Harness not self-contained | `harness/fixtures/` (data, both tokenizers, `e12_fixtures.py`); paths relative to the zip layout | — |
| — | 480 worker threads (GPT 2 B2) | **Not a problem**: 240 × 2 ran with 495 threads, ≤ 12 in flight, order kept | `validate_gpt_findings.py scale` |

## 2f. E12-V (2026-10-19, user): the same Duck notebook on our best vLLM engine

**Only the engine cell differs.** `build_e12_duck.py --vllm` keeps CELL 0/1/2/3/4/6/7 the same with three changes:
- CELL 1 default `E11_ENGINE=vllm`, so CELL 2 installs vllm 0.19.0 exactly as for S1-60K.
- CELL 3 adds `ARC3_REASONING_HISTORY_KEY=reasoning`.
- CELL 5 becomes a vLLM OpenAI-server cell.

The SGLang notebooks are byte-identical to before (SMOKE1 still sha 0fea4d0c).

**The engine cell.** It is **Duck's own Kaggle vLLM launcher** (`inference/framework/kaggle.py`: `start_vllm_server` / `wait_for_vllm_server` / `run_vllm_api_smoke_test`), with verbatim flags:
`--enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm --enable-prefix-caching --default-chat-template-kwargs {"preserve_thinking": true} --reasoning-parser qwen3`.
keithtyser's Duck launch on our FP8 repack is the same command plus `--kv-cache-dtype fp8`.

**Added from S1-60K (LB 34.58):**
- TP4, `fp8_e4m3` KV;
- MTP-3;
- FlashInfer when the libcuda probe passes (S49's shim, verbatim), else TRITON_ATTN;
- 4096 batched tokens, 16 seqs, 0.90 GPU memory;
- `--language-model-only --disable-custom-all-reduce`;
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

**Added for Duck:**
- `--max-model-len 135168` (the 128 Ki context);
- `--enable-prompt-tokens-details` (`cached_tokens`);
- 12 active streams (= the SGLang run).

There is no launch ladder: a failed launch leaves the CELL 0 fallback.

**Checked in the vLLM v0.19.0 source (GitHub raw):**
- **`reasoning` key.** The server reads prior reasoning only from `reasoning` (`entrypoints/chat_utils.py:1512`). That is Duck's default key; the milestone's `reasoning_content` is the SGLang value.
- **Truncated thinking.** Output that stops without `</think>` (with `enable_thinking`) is all reasoning (`reasoning/qwen3_reasoning_parser.py`).
- **No HTTP 500 for Duck.** The Q27-AGENT 500 (`assert content is not None`, `engine/serving.py:899`) fires only for a *named* `tool_choice`; Duck sends `"auto"`, whose path tolerates empty content.
- **Flags.** Every flag exists in `engine/arg_utils.py` / `entrypoints/openai/cli_args.py`.

**CPU harness.** `fake_vllm_duck_server.py` refuses to start without every Duck + S1-60K flag and records the history key.

| Scenario | Checks |
|---|---|
| vLLM smoke | 21/21 |
| vLLM rerun | 20/20 |
| vLLM launchfail | 6/6 |
| SGLang smoke (unchanged) | 18/18 |

**Notebooks and zip.**
- `Q27-XHIGH-E12_DUCK_REPL_VLLM_SMOKE1.ipynb`: sha d6f305fe.
- `..._VLLM_EVAL30.ipynb`: sha d8fb7940.
- Zip: `Experiment Review/E12V_VLLM_ENGINE_BUILD_2026-10-19.zip`.
- Push: `push_e12_duck.py --vllm`.

### E12-V rev 2 (2026-10-19): GPT audits validated, fixes applied (user: "do the fixes")

Validation: `Experiment Review/E12-V - GPT audits validated (2026-10-19).md`. The fixes are in the vLLM build only; the SGLang notebooks are unchanged (0fea4d0c / b65fb9dc).

**Fixes:**
- **Blocker (B1):** vLLM 0.19 omits `prompt_tokens_details` when 0 tokens were cached (`chat_completion/serving.py:1601-1603`, https://github.com/vllm-project/vllm/issues/44377), so the old smoke rejected every cold start. The smoke now passes on content + integer usage counts and records `cache_details_present`.
- **CELL 6:** the model-id fallback is `qwen` (the served name).
- **CELL 7:** a missing cached value is no longer a measured 0. The report adds `cache_detail_requests`, `cache_detail_coverage` and `cached_pct_reported_only`; the vLLM log/metrics rate stays in the report.
- **Harness:**
  - the fake server follows vLLM 0.19 usage (no details when nothing is cached);
  - new scenarios `thinkmarkup` (tool-call XML only in `reasoning`, https://github.com/vllm-project/vllm/issues/39056; Duck recovers it, runs it once and strips it) and `modelsfail`;
  - the capacity values are labelled simulated.

**Not valid:** "337,600 tokens → ~28k per stream". The real logs on this model give 12.63× at 81,920 and 10.32× at 106,496, i.e. ~1.03–1.10 M effective.

**GPU checks still open (upstream reports):**
- https://github.com/vllm-project/vllm/issues/50188: MTP + prefix cache + tool calls;
- https://github.com/vllm-project/vllm/issues/53504;
- https://github.com/vllm-project/vllm/issues/38979.

**Tests:**

| Scenario | Checks |
|---|---|
| vLLM smoke | 22/22 |
| vLLM rerun | 21/21 |
| vLLM launchfail | 6/6 |
| vLLM thinkmarkup | 22/22 |
| vLLM modelsfail | 23/23 |
| SGLang smoke | 18/18 |

**Notebooks and zip.**
- `VLLM_SMOKE1`: sha 6af38987.
- `VLLM_EVAL30`: sha 975fa8e4.
- Zip: `Experiment Review/E12V_VLLM_ENGINE_REV2_2026-10-19.zip`.

### E12-V rev 3 (2026-10-19): vLLM SMOKE1 run 1 failed → root cause found, validated, fixed

**What happened in vLLM SMOKE1 run 1** (esfersami50 v1, sha 6af38987):
- The engine came up fine: 450 s, FlashInfer, MTP-3, **KV 337,600 tokens, 8.46× at 135,168 → ~1.14 M effective**.
- The smoke passed with `prompt_tokens_details: null` (the rev-2 fix worked).
- **Every Duck request got 404:** "The model `pending` does not exist."

**Root cause.** Duck treats `"qwen"` (also `local` / `local-qwen` / `qwen-local`) as a preset alias and then sends `LOCAL_ANALYZER_MODEL_ID` instead, which CELL 3 sets to the placeholder `pending` (`tool_agent._resolve_analyzer_model`). The vLLM cell had chosen exactly `qwen` as the served name. SGLang never hit this: its model id is the path, not an alias, and SGLang does not check the name. The fakes did not check it either.

**Validated before fixing** (`validate_e12v_findings.py model_name`, strict fake = 404 unless the request names the served model):

| Served name | Duck sends | Result |
|---|---|---|
| `qwen` | `pending` | 404, "analyzer failed 11 times consecutively" (the Kaggle failure, reproduced) |
| `Qwen/Qwen3.8-27B-FP8` | the same name | 200, accepted |

**Fix (vLLM build only).**
- `VLLM_SERVED_NAME = "Qwen/Qwen3.8-27B-FP8"` (keithtyser's Duck served name).
- `fake_vllm_duck_server.py` now answers 404 to any request that does not name the served model, exactly as vLLM does.
- **Negative control:** the old notebook on the strict fake gives 14/22 (accepted 0, usage 0), i.e. the bug is caught.

**Harness gap found and fixed.** `harness_e12_nb.py` read the fake's request log after CELL 7 had stopped the server, so every log-based check (model name, history key, think-markup recovery, in-flight, branch order) had been skipped. The log is now read right after CELL 6. The branch-order check now applies only when branches outnumber slots.

**Tests:**

| Scenario | Checks |
|---|---|
| vLLM smoke | 28/28 |
| vLLM rerun | 27/27 |
| vLLM launchfail | 6/6 |
| vLLM thinkmarkup | 29/29 |
| vLLM modelsfail | 29/29 |
| SGLang smoke | 22/22 |
| SGLang rerun | 21/21 |
| SGLang launchfail | 6/6 |

**Notebooks and zip.**
- SGLang notebooks: unchanged.
- `VLLM_SMOKE1`: sha e85b69c2.
- `VLLM_EVAL30`: sha 0b152164.
- Zip: `Experiment Review/E12V_VLLM_ENGINE_REV3_2026-10-19.zip`.

## 3. Serving (our validated engine plus what Duck needs)

- **Server flags:** SGLang 0.5.21 W1 (INT4 + DFlash2, extra_buffer_lazy, fp16 SSM) **+ `--tool-call-parser qwen3_coder --reasoning-parser qwen3`**. Verify both exist in 0.5.21 and that the template honours `preserve_thinking`.
- **Request path:** the harness uses the OpenAI-compatible `/v1/chat/completions` (Duck provider mode "vllm" = an OpenAI-compatible request; it does not start vLLM).
- **Prefix reuse across turns is the critical risk on 0.5.21:** re-tokenization, and lazy mamba checkpoints at track intervals.
  - Measure `cached_tokens` per request in the first GPU run.
  - If it is low, port Franzen's patch #4 (keep the end-of-prefill Mamba checkpoint, LRU refresh) from Pennyroyal v2.5.3 to 0.5.21.
  - Patches #1–#3 (sm120 GEMM, sm120 SSM budget, Marlin MoE dtype) are sm120/MoE-specific and not needed on L4 with 27B.

## 4. Steps and gates

| # | Step | Gate (must pass before the next step) |
|---|---|---|
| 1 | Pull the remaining sources (§0); diff Franzen vs Tufa upstream; record the exact milestone env/config | the settings table is filled from the notebook, not from memory |
| 2 | Vendor `inference/` unchanged; write `Arc2ToolAgent` + REPL globals + submit | pyflakes clean; the vendored files hash-match the upstream clone |
| 3 | CPU harness: a fake OpenAI server that replays scripted tool calls (our own scripted code, never model code), with the real sandbox | scenarios: solve, wrong-then-fix, empty code, tool markup leak, HTTP 500 retry, yield/resume, context trim, deadline, 2-branch selection, rerun submission schema |
| 3.5 | GPU smoke: 1 task, 2 branches, 10–15 min (Rev 1 #7) | tool calls, thinking replay, function retention, rising `cached_tokens`, no server errors |
| 4 | GPU EVAL30 (the same 30 tasks as E11-05/E11-06) on one account | report: hits, tokens/task, turns, tool calls, submits, `cached_tokens` %, running 12, trace bundle |
| 5 | Decide the Flash-Next question separately (user); then the LB submission | — |

**Deadlines:** rosters lock 2026-10-26; final submission 2026-11-02. Target: step 4 on GPU by 2026-10-22.

## 5. What is not changed, and the open user decisions

- **Not changed:** xhigh; Qwen3.8-27B (Apache-2.0); the 60k floor for every non-agent path.
- **Open (user):**
  - Flash-Next (speed vs licence risk);
  - whether to run P1/S60 EVAL30 in parallel as the baseline.
