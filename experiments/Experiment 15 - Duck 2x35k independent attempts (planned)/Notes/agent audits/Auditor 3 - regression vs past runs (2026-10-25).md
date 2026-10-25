# Auditor 3: regression audit of E15B (config B) against every past run and analysis (2026-10-25)

**Scope.** This audit compares the built E15B code with every bug, crash and weakness found in earlier runs and analyses. It covers:
- a full ledger of all past issues;
- the status of each one in the current build;
- new risks for config B.

**Read-only.** No project file was changed. My scratch scripts and their outputs are in `D:\ARC AGI 2\_scratch\e15b\agent3\`.

## 0. What was audited

| Item | Value |
|---|---|
| SUBMIT notebook | `Q27-XHIGH-E15B_SUBMIT_2X35K_S16_DUCK_SGLANG.ipynb`, sha256 `c471a3f6ff2e…` (03:16), 255 kB |
| EVAL120G4B notebook | `Q27-XHIGH-E15B_EVAL120G4B_2X35K_S16_DUCK_SGLANG.ipynb`, sha256 `fa54669a27e6…` (03:16) |
| Duck payload | `q27x_e15b_duck_4831ed01f2a8.bin`. Its manifest matches the notebook (CELL 3). It contains `arc2_duck.py` `2941e8cfd45e`, `tool_agent.py` `f7454201484f` and `python_tool_sandbox.py` `d29d807eef10` |
| Do the two notebooks differ? | Only in the default commit profile: E15B vs E15BEVAL120G4B. Every other cell is byte-identical, so the rerun path is the same |
| Unit tests (`src/test_duck_b.py`, run by me) | 67/67 PASS, 0 SKIP |
| `src/verify_items.py` (run by me) | 55 OK, 7 NOTE, 0 MISS, manifest mismatches 0. Both notebooks are below 400 kB |

**The working copy changed during this audit.** `src/duck_b/arc2_duck.py` was modified at 03:44 (sha `e0a449e6a0a8`), after the build.
- The changes are Auditor 1's F1-F3:
  - the partial test now runs before the reasoning harvest, and the kept functions are restored;
  - the sandbox timeout is clamped to the deadline;
  - a rejected submit inside the commit zone now gets the commit text.
- These changes are **not** in the payload or the notebooks.
- Everything below refers to the **built** code (`2941e8cf`). Line numbers are `grep -n` numbers in the payload copy, which is identical to `src` before 03:44. Where a line moved by 6-20 lines in the new working copy, the function name is also given.

**Sources mined, read in full.**
- Exp 14 Notes 22-46.
- `EVAL120 analysis 2026-10-24/` A, B and C.
- Exp 12 `Experiment Results/*.md` (6 files).
- Exp 15 Notes 01-07.
- I delegated the reading of five long groups to sub-agents, each of which returned a full ledger:
  - Speed review A1/A2;
  - Speed review B/C;
  - GPT Notes 26-28 and 32-33, with Claude's 29/34;
  - GPT Notes 38-40 and 44-46;
  - Exp 15 Notes 04 and 06, with 05/07.
- I read every other file myself. I checked every status below myself against the built code.

**Status key.**
- **FIXED**: evidence in the built code.
- **STILL PRESENT**: a failure scenario is given.
- **N/A in config B**: the reason is given.
- **CANNOT VERIFY ON CPU**: the metric in the run report that will show it is given.

---

## 1. Ledger

### A. Attempt structure and scheduler

| # | Past issue | Source | Status | Evidence in the build / failure scenario / metric |
|---|---|---|---|---|
| L1 | Branch 1 (indexed view) gave 0 correct programs in 244 runs and used 22-26 % of the tokens | A §2/§4; B §3.1; GPT 44-46; N42 A1; N43 #1 | FIXED (structure). The yield is CANNOT VERIFY | CELL 1 sets `"E12_R6_B1_VIEW": "facts"`. `sess()` gives b ≥ 1 its own BranchSession with `view=b1_view` (arc2_duck `solve_all_r6.sess`). **Metric:** `eval_b_attempts.b1_only_acc`, `b1_acc` |
| L2 | The attempts must be independent: own agent, history and run_dir, no shared seed | N02 #1; P1/P2 #1; U28 | FIXED | One `BranchSession` per (task, branch), `run_dir = {tid}_b{b}` (`_ensure`). The Duck sends no seed (`LOCAL_ANALYZER_SEED` -1) |
| L3 | Attempt 1 on every task before any attempt 2 | N01/02 #2; user D-A | FIXED (start barrier, per the user's D-A) | arc2_duck L3309: `if prio[0] == 1 and state["stage1_left"] > 0: push(...); wait`. Each task's attempt 2 is queued by `stage2_items` only when its own attempt 1 ends |
| L4 | The 28k pause and stage-2 resumes caused 58-66 % of all prefill and a −18-30 % throughput drop after hour 2; 156-170 branches were paused and never resumed | C §3.4-3.5; A §4; N42 A2 | FIXED | CELL 1: `"E12_R6_FIRST_EXPOSURE": 0`. L3345: `pause = first_exposure_tokens or None` gives None. `run()` sets `paused` only when `pause_at_tokens` is set, so the status is always `done` (L2935). Harness: one run per attempt |
| L5 | The fair share chunked stage 2 (12,288 minimum chunk) | N02 #45 | FIXED | CELL 1: `"E12_R6_FAIR_SHARE": 0`. `share_tokens()` returns None when `not fair_share` (L3161-3163) |
| L6 | EVAL30: 30 time-sliced live branches thrashed the KV (8.39 M re-prefill, prefill 55 % of the window) | Exp 12 EVAL30 §2; C #15 | FIXED | Fixed 16 worker threads. No pause, so there are at most 16 resident contexts. The `n_live = len(jobs)` trap is in the old `solve_all` only, which is not used |
| L7 | EVAL30: branch 1 starved (29 of 30 never started) | Exp 12 EVAL30 §1 | FIXED | Start barrier plus class rank (`rank`, `stage2_items`). Attempt 2 is queued for every class |
| L8 | Rev 6d false capacity step-down 12 → 6 on queue = 1 | Exp 12 rev 6d F1; C #21 | FIXED | `probe_once` only feeds the KV gate and gauges. There is no step-down (L3213-3243) |
| L9 | The KV gate must read `sglang:full_token_usage`, not `token_usage` | C #63 | FIXED | CELL 6 `_e12_capacity_probe` (sub cell 9, L161-172) |
| L10 | Two KV gates side by side (0.85 and 0.95) | C #64 | N/A | The Duck uses only `E12_R6_KV_ADMIT` 0.95. The 0.85 gate (`SGL_KV_ADMIT_MAX`) is on the Poetiq path, which does not run |
| L11 | Attempt-2 order: progress first, then nothing, then accepted last (D2) | N02 #19/#52; P1/P2 | FIXED | `stage2_class` + `rank()` (1, rounds, class, flag, idx, sub); `stage2_items` pushes start1 for every class. **Gap:** no starvation count for classes that never start (U19); report only, low |
| L12 | A worker exception killed the worker silently (code audit M5) | N02 #18; N37 #10 | FIXED, with residuals | `worker()` restart wrapper (L3268-3279); `publish(s.write())` is inside a try (L3363). **Residual (low):** an exception in the novelty block (L3357-3362, before `running -= 1`) leaks one running slot. An item that raises after its pop is dropped (counted only in `worker_errors`) |
| L13 | KV-gate deadlock when nothing runs | rev 6e G4-1 | FIXED (unchanged) | `kv_idle_overrides` path in `_worker_body` |
| L14 | `min_admit_sec`: 1,300 s would stop workers about 22 min early | N02 #6/#49; P1/P2 #6/#49; A5 | FIXED | CELL 1: `"E12_R6_MIN_ADMIT_SEC": 300`. `admission_stopped_at_sec` is not logged (U25); low |
| L15 | ORDER2 + cells coverage order | A §4; GPT 44 | FIXED (unchanged) | CELL 6 L19 `E12_ORDER` |
| L16 | p90 and last-admission cutoff (built, off) | C A32 | N/A | `E12_R6_P90_CUTOFF` 0 |
| L17 | Stragglers; one absolute join cutoff (C6-4) | EVAL30 overrun 679 s; C #33 | FIXED | L3385 `join_until = deadline + join_grace_sec` (90). Freeze + `closed` make late writes no-ops |
| L18 | Low-occupancy tail at the end of the run | A2 Tail; C #19 | FIXED (design), occupancy CANNOT VERIFY | The attempt-2 queue fills the tail and requests are deadline-clamped. **Metric:** `b_summary.server_log.running_hist`, gauges |

### B. Budget, request caps, context window, commit turn

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L19 | 12,288 / 24,576 / 32,768 caps: 84-98 tool-less length cuts per run | B §3.1; C §6; GPT 44 #24; N02 #4 | FIXED | `_arc2_apply_cap` L1715: `cap = left - reserve if left > reserve + 1024 else max(256, left)` |
| L20 | Soft caps overshot (100k → 159k; the 28k pause overshot by p50 5.1k, max 15.8k); a hard per-attempt clamp is needed | Exp 12 rev 5 §sim; C #34 | FIXED (residual ≤ 255 tokens) | max_tokens follows the remaining budget. The `max(256, left)` floor overshoots by at most 256 − left (unit test "cap at 34,900: 256"). Negligible |
| L21 | No request when the remaining budget is 0 (U7) | P2 #3 | FIXED | `done_now()` stops at `generated_tokens >= max_generated_tokens` (L2852-2856), before every request (`should_stop`) |
| L22 | Context trimming inside an attempt: the trim at 51,396; trimmed branches were correct 16 % vs 34 % of the time | B §5.1b; GPT 44 #32; N02 #5 | FIXED (arithmetic + data) | `budget = 100,000 − cap − 512 = 68,584 + G`, where G is the attempt's generated tokens. So a trim fires only if the prompt plus all tool outputs exceeds 68.6k. **Measured** on EVAL120 b0 (`ctx_growth.py`): prompt − cumulative generated at 30-35k generated is p50 7.6-9.9k, p90 13.4-16.0k. **Metric:** `b_summary.trimmed_requests` |
| L23 | The window formula KV / streams would give about 54k at 16 streams | N02 #46 | FIXED | CELL 6 L442: `rev6_window = min(131072, E15B["E12_B_WINDOW"])` = 100,000 |
| L24 | The window must stay at or below the server context | P2 #5 | FIXED | `SGL_CONTEXT_LEN = 155648` |
| L25 | Request timeout too short for long replies (EVAL30: 40 timeouts at 900 s; 35k at 16 streams ≈ 1,400-1,900 s) | Exp 12 W6; C #32; N02 #47 | FIXED; the margin is CANNOT VERIFY | `E12_DUCK_REQUEST_TIMEOUT` 3000. The first request cap of 30,904 needs ≥ 10.3 tok/s per stream to finish. **Metric:** `requests[].sec` per attempt (no p99 in `b_summary`, see L126) |
| L26 | A ReadTimeout loses the in-flight reply (`tool_agent` :628 / :7106) | C #32 | STILL PRESENT (by design, low) | The timeout path yields with history kept, but the reply body is lost. With 3,000 s this happens only at the deadline clamp (see R8) |
| L27 | Per-request deadline clamp (R6-S6) | Exp 12 rev 6; C #33 | FIXED (unchanged) | `run()` L2873: `request_timeout_seconds = max(0.1, min(analyzer_timeout, remaining))` |
| L28 | The last request ends mid-thinking at the budget with no chance to write code. Qwen emits tools only after `</think>` | N02 #48; GPT 44 #24 | FIXED (built). Effectiveness CANNOT VERIFY | Commit opener L1583-1585, tool note L2104-2108, reserve 4,096 (+1,024 zone). Duck yields after a 30k reply (`control_yield_reason` before the nudge path), so the commit opener is the one that is used. **Metric:** `b_summary.commit_with_tool_call / commit_turns` (the fake harness gave 21/32) |
| L29 | No commit after an accept; the commit is recorded once (U6) | P2 #48 | FIXED | An accept ends the attempt (`done_now`). `_arc2_b_commit_sent` records the event once |
| L30 | The budget is not stated to the model | N02 #48 | FIXED | `B_PROMPT_EDITS` #48 (L1121-1135); `first_budget` in the first opener (L1577) |
| L31 | Code written only in the reasoning is never run (db0c5428) | GPT 44 Cause D; N02 #50 | FIXED, with a new weakness | `b_code_blocks`, `b_harvest_source`, `arc2_b_reasoning_harvest` (L1234-1283, L2224-2264). **Weakness (R9):** `Import` nodes are kept, so one blocked import (`import numpy as np` in a draft) raises at the first line and none of the definitions run. Also Auditor 1 F1 (harvest before partial test) is being fixed in the working copy, not yet built |
| L32 | First check comes late (median 27-31k); 21/46 and 17/41 correct programs first appeared after 35k under rev 6 | B §1.1; GPT 44-46 #23; N43 §0 | N/A (the user fixed 35k, D1 = B); the loss is CANNOT VERIFY | **Metrics:** `eval_b_attempts.b0_budget_partial`, `b0_budget_partial_b1_acc`, `rev6_events.first_check_tokens` |
| L33 | Late-correct tasks (135a2760, 898e7135) must not be cut early | GPT 45 P1-2; N43 #75 | N/A (design) | As L32 |
| L34 | The E16 nudge is neutral; turn it off | B §1; GPT 44-46 #33; N43 D3 | FIXED | `build_e15b.py` L45: `assert … not G["E16"]` |
| L35 | The rev 6 24k provisional nudge | rev 6f H6-1 | N/A (consistent with B) | `E12_R6_NUDGE_TOKENS` 24000 fires before the commit zone. Its text is neutral (L1216-1218) |
| L36 | Effort mislabelled (the LOW run's top level said xhigh) | GPT 46 Fix 4; N43 #87 | FIXED (label). Partial | `E12_REASONING_EFFORT` "" (template default) and `b_summary.effective_effort`. Not read from the request payload (P1 #9); low |
| L37 | `yield_tokens` 2,048 caused the rollback | N02 #51 | N/A (kept on purpose) | Harmless after L38 |

### C. Agent, tools and sandbox

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L38 | Turn rollback when a yield fires after the first of several tool calls (56 / 47 / 6 events; 0.52 / 0.46 M context tokens lost) | B §5.1a; N02 #10 | FIXED | `tool_agent.py` L7108-7112: with `ARC3_YIELD_AFTER_BATCH` (set in B by `duck_env`, arc2_duck L80) the loop continues to the remaining calls. Residual path L7103 (an accepted submit that is not the last call) is harmless because the attempt ends. The unit test passes |
| L39 | Rollback of a whole turn on a non-timeout request error | `tool_agent.py` L7156-7157 (found in this audit while checking L38) | STILL PRESENT (low) | On `requests.RequestException`, `preserve_history = False` and the tokens are reset. During a server crash or relaunch, the completed exchanges of the current turn (≤ ~2k tokens and their tool calls) are discarded while the sandbox keeps the functions they defined. History and sandbox then disagree (the NameError class). **Fix:** treat it like the ReadTimeout path and keep the completed exchanges |
| L40 | `submit` / `check` called as tools (22 / 27 / 23 branches) | C §4; GPT 46 P0-3; N02 #11 | FIXED | `_dispatch_tool` L1504-1527 (B only, identifier regex, code argument run in the sandbox). Tested |
| L41 | Other unknown tool names: `code` 1/0/6, `function` 0/2/1, `function_calls` 1/0/0; "Parsed parameter 'python'" 0/1/0 | C §4 | STILL PRESENT (low) | Only submit / check / check_many are routed; the others still get "Unknown tool" and cost a request. **Fix:** route any unknown tool whose arguments hold a `code` string to python |
| L42 | No tool contract in the prompt | GPT 46; N02 #12 | FIXED | `B_PROMPT_EDITS` #12, tested ("matches once") |
| L43 | Autocheck misses `trans` / `candidate` (cbebaa4b) | GPT 44 Cause B; N02 #13 | FIXED | `_R6_AUTOCHECK_RE_B` L1227; tests (transpose / translate not matched) |
| L44 | Sandbox blocked `sys` (21/21/13), `time`, helper imports, numpy message | N43 §0; N02 #14 | FIXED | Sandbox L73 (`time`), L280-315 (sys shim, recursion clamp to 20,000, helper and numpy messages), L808 (env to the child). 8 sandbox tests pass. Relative imports and the `sys.exit` test (U9): base behaviour, low |
| L45 | Unbounded search timed out `check()` (e8686506) | GPT 44 Cause C; N02 #22 | FIXED (prompt only) | `B_PROMPT_EDITS` "Bound any search…". No `SearchBudget` helper (rejected in N07) |
| L46 | NameError and retained-state confusion (19/22/23 branches) | GPT 46; N02 #21 | FIXED (partial) | L2104: `payload["defined_functions"]` (40 names). No "fresh-call contract" text; low |
| L47 | Repetition loops (rev 4: 29.4 %); penalties do nothing under DFlash | Exp 12 W3; C #37 | FIXED (unchanged rev 5); CANNOT VERIFY at B | T 1.0 (`E12_DUCK_TEMPERATURE`); no penalties sent; the degenerate-reply stub is in Duck. **Metric:** `rev6_events.reasoning_repeat_chars` |
| L48 | A fresh long reply was dropped from the next prompt (35 % of tokens in rev 4) | Exp 12 W2; C #28 | FIXED (rev 5 B3) | `E12_TOKENIZER_JSON` = the served model's tokenizer (CELL 1) |
| L49 | Blank-line normalization lost the reply prefix (59-91 % of requests) | C #23; N23 | FIXED; residual causes CANNOT VERIFY | CELL 1b L80: `E12_RAW_REASONING_HISTORY = "1"`. EVAL120 showed 1569 kept / 0 mismatch. **Metric:** cache hit % (`usage_all`) |
| L50 | Grid-row guard gap `print(r, row)` | Exp 12 rev 6d F2 | FIXED | `_GRID_ROW_RE` row-index alternatives (L997-1000) |
| L51 | Facts cap on larger hidden grids | N02 #55 | FIXED (Notes/03 measurement; not re-measured by me) | `E12_R6_FACTS_MAX_TOKENS` 800 |
| L52 | Hand perception errors; the grid-view choice | Exp 12 W4; GPT 44 #41 | N/A | Solver design, held (no redesign) |
| L53 | `for i,(a,b) in enumerate(train)` (train items are dicts) | Exp 12 W6 | NOT VERIFIED (low) | No pair-list alias found. The helpers and `check()` are the intended path |
| L54 | A tool call cut by the cap was executed as a stub | Exp 12 W6 | FIXED (rev 5 R5-2) | `arc2-cut-placeholder` with invalid JSON |
| L55 | A dead-reasoning stub would remove a 30k thinking block before the commit turn | `tool_agent` L888-894 (checked in this audit) | N/A (off) | `ARC3_STUB_DEAD_REASONING` defaults to False and is not set in `DUCK_ENV` |

### D. Selection and submission

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L56 | Check-exact functions were never harvested (W1: EVAL6 would have been 5/6) | Exp 12 W1 | FIXED (rev 5 B1, unchanged) | `_arc2_harvest_r6` |
| L57 | attempt_2 was the test input (56/56; 58 of 60) | Exp 12 W5; B §3.2; N02 #7 | FIXED (structure); the yield is CANNOT VERIFY | The second attempt's best plus `arc2_b_partial_test` (L2284-2300) and rules (L2701). **Metric:** `eval_pass2_hits`, `b_summary.partial_test_runs` |
| L58 | Distinct grids; two verified predictions both kept | GPT 44 #26; N43 #76 | FIXED | `select_attempts_r5` groups branch bests by identical output, then explicit submit, then latest; `a2 = next distinct` (L2657-2711) |
| L59 | The selector must not lose a correct candidate (46 = 46) | N43 §0 | FIXED (unchanged logic) | — |
| L60 | `overfit_flags` must not be used in selection | B §4; N02 #23 | FIXED | Not referenced in `select_attempts_r5` |
| L61 | A rule-library exception lost the task's program candidates | N07 A6 | FIXED | L2677-2681 try. The measured value is about 0 on eval (0/120) |
| L62 | CPU-rule deadline and memory bound (U10) | P1 #20 | N/A | Pure-Python O(cells) rules on ≤ 30×30 grids |
| L63 | Repeated submission writes; schema validation (U13) | N02 #26; P2 #26 | FIXED | CELL 6 `e12_write_submission`: an identity base for **every** test task, rows replaced only on a length match, unique tmp + `os.replace`, a write lock |
| L64 | The fallback must be written before any risky work | GPT 32/33 B3 | FIXED (unchanged) | CELL 0 writes the identity file right after reading the test JSON. If that JSON is unreadable there is no file (unchanged; low) |
| L65 | SGLang failure means an identity submission (LB ≈ 0); the "vLLM fallback" does not exist | N37; GPT 32/33 B3 | STILL PRESENT (reduced) | The fallback chain is now 16 → 12 → oom / lazy / prefill → W1 (`e11_next`, CELL 5). If every launch fails, the submission is still the identity file |
| L66 | No gold leakage into runtime decisions | GPT 44-46 #45 | FIXED | Gold is read only in CELL 7 scoring (A2 `eval_b_attempts`) |

### E. Engine: concurrency, KV, mamba, graphs, flags

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L67 | DFlash min-free-slots delayer held 11 of 12 (2.2-2.5 %) | C §2; N30; N02 #36 | FIXED (flag + source); occupancy CANNOT VERIFY | `E14_SUBMIT_ARM["--min-free-slots-delay"] = ["1"]`. **Source-checked:** `resolve_min_free_slots(1, 17, True)` returns None (`threshold > 1` else None). **Metrics:** `server_log.running_hist_queue_gt0`, `server_at_streams_share_client_full` |
| L68 | Nodelay crash at 13 running (a buffer sized 12) | N30; GPT 32/33 A3 | FIXED (source-checked) | `dflash_worker_v2.py` L844/884/919: `max_bs = max(cuda_graph_config.decode.bs)` = 17 = mrr. `--cuda-graph-max-bs-decode` is dropped. The 12-stream fallback uses mrr 13 and bs 1..13 (consistent). The W1 fallback keeps the delayer on |
| L69 | Scheduler latch needs mrr ≥ streams + 1 | A1 §1.5 D1 | FIXED | `SGL_SERVER_MAX_RUNNING = 17` |
| L70 | Mamba slots cap concurrency (56 // 4 = 14) | B/C #13; A2 #18 | FIXED | `W1["mamba_cache"] = 68` gives 17 |
| L71 | Running sizes above the graph list run eager | A1 #16; C #22 | FIXED | `--cuda-graph-bs-decode 1..17` |
| L72 | The client gate capped concurrency at 12 (KV // 70k) | P2 #38 (U1) | FIXED; the margin is CANNOT VERIFY | `SGL_KV_PER_REQUEST = 50_000`. 16 needs KV ≥ 800,000; predicted 852k (N35 table). Below that the gate still passes (min 8): conc 15 → 15 streams, a graceful fallback. **Metric:** `engine.kv_tokens`, `CONFIG.e11` |
| L73 | `/get_server_info` misreports mrr | A2 #20 | FIXED | `e11_caps` parses "capped to" and the KV / mamba / graph lines |
| L74 | Default mrr 48 when the flag is omitted | A2 #21 | FIXED | Explicit flag |
| L75 | KV fit at 16 × B contexts | N02 #40; P1/P2 #40; D66 precedent | CANNOT VERIFY ON CPU (see **R2**) | My measurement: at 30-36k generated, prompt p50 40.9-42.9k, p90 46.0-47.9k, max 49.8k. So 16 × ~48k ≈ 0.90 of ~852k, before in-flight output. **Metrics:** `server_log.retract_lines` / `preempt_lines`, gauge KV max |
| L76 | Free GPU memory after more graphs and a larger mamba pool; late Triton OOM (0.79-0.98 GiB free at runtime with graphs 1..13) | GPT 32/33 A7; C #17 | CANNOT VERIFY ON CPU | Gate `SGL_MIN_FREE_GB = 1.0` on `avail_mem_gb_last`. **Metrics:** `server_log.startup.graph_capture[*].avail_gb_after`, `oom_lines` |
| L77 | Int8 mamba OOM | N30 | N/A | fp16 SSM |
| L78 | SSM dtype (fp16, not bf16) | A2 #42 | FIXED | `--mamba-ssm-dtype float16` |
| L79 | The drafter inherited INT4 (#39087) | A1 #35 | FIXED | `--speculative-draft-model-quantization unquant` |
| L80 | Stream interval with DFlash | C #47 | FIXED | `--incremental-streaming-output --stream-interval 1` |
| L81 | Unsafe flags: mixed chunk (#43514), grammar / `tool_choice` required, HiCache, penalties, logprobs, NCCL overrides, forced custom AR, ReplaySSM + DFlash, torch.compile, adaptive spec | A1/A2 #32 #36 #44 #47-49; B/C #18 #46 #50 #56 | FIXED | None of these is in `e11_flags` (CELL 5) or in the Duck request kwargs |
| L82 | Width 6 unmeasured above ~52k per request | C §5 | N/A for B; CANNOT VERIFY at 16 | B requests stay ≤ ~50k context (L75 data) |
| L83 | Spec gain shrinks at 16 concurrency (E11-03: DFlash fp16 1.10× AR at c16 vs 1.39× at c12) | A2 #27 | CANNOT VERIFY ON CPU (see **R6**) | `b_summary.server_log.gen_tok_s_p50` is the only figure; there is no per-level tok/s (L126) |
| L84 | #36548 cross-request contamination under concurrency | GPT 27/28/32/33; A1 #38; C #44 | CANNOT VERIFY ON CPU | Only the 16-marker isolation preflight (CELL 6 `_e12_r6_preflight`). The parity gate was rejected (N05 #44). **Metric:** `rev6_preflight.isolation_*` |
| L85 | Accept length swings between boots (#38009); log it per boot, including relaunches | A1 #39; C #45 | STILL PRESENT (report, low) | `_e15b_server_log_summary` parses no accept length, and it reads only `E11_SERVER.log_text()`, i.e. the last server's log |
| L86 | FP8 KV scales default to 1.0 | B/C #54 | N/A (accepted, unchanged) | — |
| L87 | Chunked prefill 4096 stalls decode | B #28; GPT 26-28 | N/A (held) | B has no resumes; first prompts are 3-8k |
| L88 | lpm cold-request starvation | C #60 | N/A | The client offers ≤ 16 against mrr 17 |
| L89 | SGLang self-kill during long ops | C #59 | FIXED | `--watchdog-timeout 1800` |
| L90 | DFlash `target_layer_ids` silently wrong (#37475) | GPT 27; A2 #41 | FIXED (logged, not asserted; low) | The preflight records `drafter_target_layer_config` |

### F. Startup and platform

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L91 | A fixed 30-min launch timeout gives LB 0 on slow storage | A1/A2 #2-3; C #1, #4 | FIXED (unchanged E14) | `E11Server.wait_ready` → `E14L.StallMonitor` (900 / 720 s stalls); give-up `min(T0 + 75 min, admit − 5 min)` |
| L92 | GCSFUSE slow load | A1 #1; C #2-3 | FIXED (E14 staging) | Parallel `/dev/shm` copy, carried |
| L93 | The staging copy ran past the launch join | N02 #27 | FIXED | CELL 1b L76: `deadline = max(T0 + 10 min, min(T0 + 45 min, give_up − 20 min))` |
| L94 | A stale NV / JIT cache corrupts results (#42917) | A1 #7; C #6 | N/A | No cache is shipped; fresh cache dirs |
| L95 | JIT tar IndexError (`splitlines()[-1]`) | N30; GPT 32/33 A6 | N/A (code unchanged but not run) | CELL 7b L17 is still unguarded but runs only under `E14_DO_MEASURE` = False (CELL 1b L40) |
| L96 | FlashInfer sampling JIT crash (#43412) | A1 #8 | FIXED (carried; not re-verified) | Sampled smoke request (T 0.6, top_k 20) in `e11_smoke` |
| L97 | Commit-notebook size limit (407-444 kB) | N23 | FIXED | 255 kB; builder `NB_LIMIT` assert |
| L98 | pip / venv hangs | N02 #24 | FIXED | Timeouts 300 / 1,800 / 120 s |
| L99 | Large output or wheels under `/kaggle/working` mark the run ERROR | A1 #12 | FIXED (venv in `/tmp`). Bundle size CANNOT VERIFY | 480 transcripts (about 70-100 MB estimated); low |
| L100 | Python 3.12 `Thread._stop` shadowing crash | N22 | FIXED | No Thread subclass in the new code (grep) |
| L101 | Wrong model name (404 "pending") | B §1.12 | FIXED | `/v1/models` lookup (CELL 6 L413) |
| L102 | Self-test and tokenizer probe on the first hidden task (M4) | N02 #17 | FIXED | CELL 3 `E15B_SELFTEST_TASK`; CELL 2 fixed probe task |
| L103 | 12-h margin in the rerun | N43 #66; A1 #11 | FIXED (as E14S, +30 s) | Deadline 42,720 s, join ≤ 42,810, budget 43,020 vs 43,200. The submission is written at the end of CELL 6, before the report cells |
| L104 | Platform ERROR before our code ran / mount stalls | A1 #10; C #7 | N/A | Platform |
| L105 | Kaggle CPU kit on the exact sha before any GPU run | memory rule; N07; U23 | STILL PRESENT (process) | Not run for `c471a3f6` / `fa54669a`. The README lists it as "not yet done" |
| L106 | The payload dataset in every account; keep the old `.bin` files | N02 #25 | N/A (process) | — |

### G. Watchdog and launch fallback

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L107 | The server dies mid-run and the rest of the 12 h is lost (H1) | N37 #5; N02 #15 | FIXED (fault-kill harness 28/28) | CELL 6 `_e15b_watchdog` (L486-535); `SERVER_DOWN` wait in `run()` (L2884-2896). Duck retries connection errors in place 3 times, and a new attempt's first request gets a 900 s grace |
| L108 | Zero-throughput stall with the process alive (EVAL30: running 11, queue 8, gen 0.0) | Exp 12 EVAL30; C #58 | FIXED (source-checked) | SGLang `/health` (`http_server.py` L701-777) sends a 1-token request and passes only if a detokenizer output arrives within 20 s. 180 s of failures leads to a relaunch |
| L109 | A slow but healthy server must not be killed | P2 #15; N07 | FIXED | 180 s of consecutive failures (fault-slow harness 26/26). The client's 10 s urlopen timeout is shorter than the server's 20 s wait, so single checks can fail during a ≥ 10 s output gap. Harmless at 180 s |
| L110 | The relaunch must reuse the same flags (71 s path); pin the ports | A1/A2 #6, #55 | FIXED (flags); the NCCL port is not pinned (low) | `E11Server(list(_srv.cmd), _srv.env, …)` |
| L111 | Orphan processes after a crash | A2 #55 ("kill the old tree, confirm GPU freed") | STILL PRESENT (low-medium) | `E11Server.kill()` does nothing when the leader has already exited (`if self.alive():`). The orphaned TP workers keep the GPUs, `_e11_gpu_free` waits 180 s, then runs `pkill -9`, sleeps 15 s and relaunches. That is about **+3.3 min per crash**. **Fix:** call `os.killpg(pid, SIGKILL)` unconditionally (catch `ProcessLookupError`) |
| L112 | The relaunch repeats a config-caused crash | Derived from N30 (nodelay crash), GPT 38 HIGH-1, U3 | STILL PRESENT → **R1** | The watchdog relaunches with the identical 16-stream command twice. It never uses the 12-stream fallback that `e11_next` uses at launch |
| L113 | Launch fallback (H2) | N02 #16 | FIXED | `e11_next` → streams12 first for any 16-stream failure, then oom / lazy / prefill / W1. The gate fallback (CELL 5 `e11_bring_up`). No 14 tier (the user: smoke-driven) |
| L114 | Double-driven agent; a straggler after shutdown | Exp 12 rev 6c traps | FIXED (unchanged) | `active` set; `freeze()`; `state["closed"]` |
| L115 | A request after the deadline (`max(0.5, …)`) | P1 #47 | N/A (accepted in N05) | — |
| L116 | Silent re-run of a long generation on retry | A2 #55 | STILL PRESENT (low) | `_post_with_retries` re-sends on connection errors and on 500 / 502 / 503 / 504 up to 3 times (`tool_agent` L521, L5293-5301). After a client-side reset the server may still run the old request (an orphan). Rare |

### H. Reporting

| # | Past issue | Source | Status | Evidence / scenario / metric |
|---|---|---|---|---|
| L117 | Retract counter false positive (`retraction_policy`) | A §9; N42 R2 | FIXED | CELL 5 counter and `_e15b_server_log_summary` exclude `server_args` / `retraction_policy` |
| L118 | Deadline-cut tokens are missing from the count | N42 R1; N02 #32 | FIXED; wrong after a relaunch (low) | `server_gen_tokens` = Prometheus delta. The counter resets on a watchdog relaunch, so the delta becomes wrong or negative |
| L119 | Accepted vs harvested | N02 #34 | FIXED | `accepted_submit`, `harvested_only` |
| L120 | Per-attempt record and candidate lifecycle | N02 #29/#31 | FIXED | `b_events`, `b_budget`, `b_candidate`. There is no explicit `not_admitted` (U16); low |
| L121 | B-risk metric (attempts at the budget with progress; rescued by attempt 2) | N02 #30; A2 | FIXED | `at_budget_with_progress`; `eval_b_attempts`. It uses gold, but only offline in CELL 7 |
| L122 | Real concurrency | N02 #39; A1 | FIXED | `running_hist`, `running_hist_queue_gt0`, `server_running_mean_client_full`. The decode-line histogram is line-weighted, not time-weighted |
| L123 | Startup split | A3 | FIXED | `server_log.startup` (load, graph capture, KV, ready span) |
| L124 | KV gate ran on stale samples | A4 | FIXED (counts checks, not samples, so it is inflated; low) | `kv_stale_checks` |
| L125 | Engine block after a relaunch | This audit | STILL PRESENT (low) | Only the last server's log is parsed (same root as L85 / L118) |
| L126 | Request latency p99, decode tok/s per running level, step ms (N02 #42, #47) | N02 | STILL PRESENT (report only, low) | Not in `b_summary` |
| L127 | Stale documents | — | STILL PRESENT (docs) | Notes/03 still describes the 45-min / 16-task smoke and the rev 2 shas. The README notes table ends at 07. The E15B profile comment cites "Exp 15 Notes/08", which does not exist |

### I. Solver-design weaknesses (held by the user: no redesign)

| # | Past issue | Source | Status | Note / metric |
|---|---|---|---|---|
| L128 | No executable candidate on 64-67 tasks (63-65 % of the tokens); program-first protocol | GPT 44-46 #21-22; N43 #30/#74/#84 | N/A (held) | `b_summary.never_wrote_candidate`, `wrote_never_checked` |
| L129 | Train-exact but test-wrong on 10-16 tasks; soft sanity score | GPT 44-46 #35; N43 #31/#86 | N/A (held); attempt 2 partly addresses it | `eval_b_attempts` |
| L130 | Poetiq-style attempt 2 / hybrid | GPT 44-46; N36; N43 D8 | N/A (held; the user chose an independent Duck attempt) | — |
| L131 | Facts vs grid / object views | GPT 44/45 | N/A (held) | — |
| L132 | E15 MTP-engine notebook bugs: `_gated` passes without a probe, `compare_passes` reports 0 flips on all errors, unprobed arms adopted, `_fix` skips the long gate, latch at level 12, extra probe rows ignored, window errors ignored, the final-launch return value ignored | GPT 38-40 #1-#13, #19 | N/A (E15 engine off; the CELL 5b arms run only under `E14_DO_MEASURE`) | The analog in B: the watchdog checks `_ok` before it swaps `E11_SERVER` (FIXED) |
| L133 | MTP-only risks (#44145 KV shrink, head presence) | GPT 32/33 C1-C2; N35 | N/A | DFlash2 W6 |
| L134 | Upgrade, 2×TP2 and Flash-Next risks | GPT 26-28; Speed B/C | N/A | TP4, 0.5.21 pinned |
| L135 | CPU starvation and L4 power throttling not logged | A1 #51-52 | N/A (monitoring only; low) | — |

---

## 2. Count by status

Each of the 135 rows is counted once, under its primary status (the first status in its row).

| Status | Count | Items |
|---|---|---|
| FIXED, including "FIXED (partial)" and "FIXED, with residual" | **91** | — |
| STILL PRESENT | **12** | L26, L39, L41, L65, L85, L105, L111, L112, L116, L125, L126, L127 |
| N/A in config B | **27** | — |
| CANNOT VERIFY ON CPU, as the primary status | **4** | L75, L76, L83, L84 |
| NOT VERIFIED (low) | **1** | L53 |

Some FIXED rows still carry a residual weakness that is noted in the row:
- L12: a slot leaks or an item is dropped on an exception;
- L20: an overshoot of at most 255 tokens;
- L31: an import aborts the harvest (see R9);
- L36: the effort label is not read from the payload;
- L44: relative imports;
- L46: no fresh-call contract;
- L90: the layer ids are logged, not asserted;
- L110: the NCCL port is not pinned;
- L118: the token count is wrong after a relaunch;
- L120: no `not_admitted` record;
- L124: the stale count is inflated.

Other FIXED rows need the GPU run to confirm the effect: L1, L18, L25, L28, L47, L49, L57, L67, L72, L82 and L99. The metric to read is given in each row.

---

## 3. NEW RISKS for config B, ranked by impact

| Rank | Risk | Where | Failure scenario | Minimal fix |
|---|---|---|---|---|
| **R1 (high)** | The watchdog relaunch repeats a config-caused crash | CELL 6 `_e15b_watchdog`: `E11Server(list(_srv.cmd), …)` | The 16-stream configuration is new on GPU: mrr 17, graphs 1..17, mamba 68, the delayer off, less free memory (L76). If the server dies because of that configuration (an OOM during a long prefill or a late Triton load, or a runtime shape error), both allowed relaunches use the same command and die the same way. The rest of the run is then lost; on the hidden run that can be hours | On relaunch #2, or as soon as the dead server's log tail has `out of memory`, `RuntimeError` or `size of tensor`, relaunch with `launch + e11_flags(dict(E14_VENV["cfg"], streams12=True, mamba_cache=56))` (mrr 13, bs 1..13, the delayer still off). Keep the 16 client workers: the server queues up to 3, with no crash. Save the old log tail in the record |
| **R2 (high-medium)** | KV peak from 16 long contexts that grow in step | The scheduler: all 16 attempt-1s start at t = 0; the KV gate only blocks new admissions | The EVAL120 data (L75) put a 35k attempt's context at p90 ~48k, max ~50k, and hidden tasks are sized like training (larger grids, larger tool outputs). The first wave reaches its peak together after ~20 min: 16 × 48k ≈ 0.90 of a ~852k pool, plus in-flight output. Possible results: SGLang retraction (re-prefill thrash, the D66 precedent: 66 preemptions at 16), or the gate stuck near 0.95 | Measure in the smoke (`retract_lines`, `preempt_lines`, gauge KV max). If retract > 0 or KV max > 0.92: `E12_R6_KV_ADMIT=0.85` (env), or 14 streams. Optionally stagger the first wave (no code change needed if it is not seen) |
| **R3 (medium)** | The commit turn produces no tool call | `_arc2_r6_opener` commit branch; cap = left (≤ 5,120) | After a 30,904-token first request is cut, the model has about 4,096 tokens, with xhigh thinking on, to close its thinking and emit the call. When it runs out, the attempt ends with no executed candidate unless the reasoning harvest finds code. Under rev 6 the same situation (84-98 tool-less cuts per run) was the main waste | Read `commit_with_tool_call / commit_turns` and `reasoning_harvest_new_accepted` in the smoke. If the rate is below ~50 %: `E12_B_RESERVE=8192` (env, no code change) |
| **R4 (medium)** | The 20-task commit window is marginal for "both attempts finishing" | Builder profile `E15B`: `solve_window_sec` 65 min | My schedule simulation (`commit_window_sim.py`: the real ORDER2 order, start barrier, D2, class rank; accept tokens from the A per-task table): the last attempt ends at 53.0 min at 27 tok/s per stream, 58.5 min at 25, and 67.4 min at 22, when **4 of 40 attempts are cut**. The per-stream rate at 16 is unmeasured (L83). The profile comment's arithmetic is also inconsistent: it says "3 waves × ~22 min" but "worst ~58 min" | Raise `solve_window_sec` to 78 min. It fits: the solve deadline is `HARD_STOP − 180 s` = 95 min, the solve starts ~16-17 min after T0, and the 100-min budget stays |
| **R5 (medium-low)** | Recovery from a crash is slow, and its report is wrong | `E11Server.kill()`; `_e15b_gen_tokens_total`; `_e15b_server_log_summary` | When the leader process dies first, the orphans are killed only by the 180 s `_e11_gpu_free` + pkill path (+3.3 min of 16 idle streams per crash; L111). After a relaunch, `server_gen_tokens` is wrong (the counter resets) and the engine block covers only the new server (L118, L125) | `kill()`: `os.killpg` always. Keep a list of the server logs and sum the counters per server |
| **R6 (medium-low)** | The 16-stream throughput gain may be small | Engine | A modelled step of +5.2 ms per stream and the c16 spec-gain shrink (L83) would put the per-stream rate at ~25 tok/s; aggregate e2e +12 % is predicted but unmeasured. This sets R4 and the hidden coverage (240 × 2 × ~30k ≈ 14.4 M tokens needs ≥ ~345 tok/s over 11.6 h) | Report decode tok/s per running level (L126). If 16 is not above 12 (E14S 423-434 tok/s decode), use the 12-stream flags |
| **R7 (low-medium)** | The 16-marker isolation preflight can lock 6 workers | CELL 6: `workers_for_isolation` (D4) | 16 concurrent 24-token replies; one empty or odd reply twice (INCONCLUSIVE, no leak) sets 6 workers for the whole 12 h. That costs more at 16 than it did at 12 | Needs the user's approval (D4): fall back to 12 instead of 6 on INCONCLUSIVE, or retry twice |
| **R8 (low)** | End-of-run cut | `min_admit_sec` 300 (A5); non-streaming requests | Attempts started in the last ~20 min mostly lose their in-flight request at the deadline (the reply body is lost, L26), about 1-2 % of tokens | Accepted (A5). Optional: stage-2 starts need ≥ 900 s left |
| **R9 (low)** | A blocked import aborts the reasoning harvest | `b_harvest_source` keeps `ast.Import` nodes | `import numpy as np` (or any blocked module) in a draft raises at the first line, so no definition runs and `new_accepted` = 0 | Drop Import / ImportFrom nodes whose root is not an allowed module (or wrap each in try/except) |
| **R10 (low)** | Turn rollback on a request error during a relaunch | `tool_agent` L7156 | See L39: history and sandbox disagree after a crash | Keep the completed exchanges on `RequestException`, as the ReadTimeout path does |
| **R11 (process, high before any GPU run)** | The working copy is ahead of the build | `src/duck_b/arc2_duck.py` `e0a449e6` (03:44) vs payload `2941e8cf` | The audited notebooks and payload do not contain Auditor 1's F1-F3. A push of the current notebooks would run without them. A rebuild changes the shas, so the CPU kit must run on the new sha | Rebuild, then re-run `test_duck_b.py`, `verify_items.py`, the harness in 3 modes plus the fault modes, and the Kaggle CPU kit on the final sha (L105) |

**Checked and found safe** (the example areas from the brief):
- Preemption at the 16-running buffer: L68 is source-checked.
- Delayer semantics: L67 is source-checked.
- Watchdog false kill of a busy server: L109.
- Stage-2 start before attempt 1 finishes: per the user's D-A.
- Selection between the two attempts (L58): each attempt's best is placed before any alternate; agreement wins attempt_1. With exactly one best each, both reach attempt_1 and attempt_2.
- The hidden deadline (L103).

---

## 4. Scratch evidence (my scripts; read-only use of past data)

- `D:\ARC AGI 2\_scratch\e15b\agent3\ctx_growth.py`: EVAL120 XH / XH+E16 b0 prompt tokens vs cumulative generated tokens (untrimmed requests). At 30-36k generated:
  - prompt p50 42,899 / 40,890; p90 47,881 / 45,969; max 49,800 / 47,861;
  - overhead (prompt − generated) p90 16.0k / 13.4k.
- `D:\ARC AGI 2\_scratch\e15b\agent3\commit_window_sim.py`: the E15B commit schedule (see R4).
- `D:\ARC AGI 2\_scratch\e15b\agent3\test_out.txt` (67 PASS) and `verify_out.txt` (55 OK / 7 NOTE): my runs on the built notebooks.
- `D:\ARC AGI 2\_scratch\e15b\agent3\arc2_duck.diff`, `tool_agent.diff`, `sandbox.diff`: rev 6g → B diffs.
- `D:\ARC AGI 2\_scratch\e15b\agent3\payload_arc2_duck.py`: the built `arc2_duck.py` taken out of the payload, for the diff against the working copy.
- `D:\ARC AGI 2\_scratch\e15b\agent3\sub_cellNN.py` and `e120_cellNN.py`: the notebook cells.
