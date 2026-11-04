## E33 Poetiq solver module: built, 32/32 local tests pass (Python 3.12.10 and 3.14.6)

The module copies the Poetiq core from the E18 R2 notebook unchanged. The transport is copied unchanged from E29. The per-call driver is new and follows E18's cell 7 logic; every place it departs from E18 carries a `# E33` comment.

### 1. Files and sha256
All files are in `D:\ARC AGI 2\experiments\Experiment 33 - Duck + Poetiq + Direct attempt chain (planned)\Build\`. All are LF only (0 CR bytes, checked).

| File | sha256 | Size |
|---|---|---|
| `e33_poetiq.py` (generated, do not edit) | `397acb4c7578045ac057ec1f85b1e74a876a6613cb9119ecbfc6390329a418fd` | 87,749 B |
| `e33_poetiq_build.py` | `1a2595585cfe75d4f940062958356a5db5e2d5e327e86cafabf0631d12a20c33` | |
| `e33_poetiq_tests.py` | `4857e397159735c3bbc3cf691d4508ee65a403378cc13e1385db86ebb1fb3340` | |

- **Builder commands:**
  - `py -3.12 e33_poetiq_build.py` writes the module.
  - `--check` rebuilds it in memory and compares; it currently reports `check: true`.
- **Pinned sources (the builder refuses to run if any sha differs):**
  - E18 R2 notebook `Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb`: `116d2d41…79a1d`.
  - Its cells: index 2 (CELL 1) `02bdec93…`, index 5 (Poetiq core) `21d72950…`, index 7 (solver, read only) `59f619b6…`.
  - `Experiment 29/Build/e29_direct.py`: `8b42802c…5feb93c`.
- I did not touch any `e33_sched_*` file or anything outside these three files. No Kaggle calls, no git. The `__pycache__` in Build is not mine (`e33_kit_switch_tests`).

### 2. The API
```python
poetiq_run(tid, task, *, iteration, prior=None, deadline, base_url, model_id, out_dir, tok, cfg=None, task_index=None,
           health=None, server_down=None, harness=False, log=print, attempt=None) -> record   # never None
```
- **`iteration`:** 1 (stage C) or 2 (stage D).
- **`prior`:** the iteration-1 record or its `poetiq_state`. It is ignored when `iteration == 1`.
- **`deadline`:** `time.monotonic()` of the solve end T, as in E29.
- **`base_url`:** with or without `/v1`; both forms are accepted.
- **`tok`:** E29's `{"render", "encode", "decode", "eos_ids"}`, i.e. what `e29d_tok()` returns.
- **`task_index`:** an int or a `{task: index}` dict (the sorted full task set, E18's seed index).
- **`health` / `server_down`:** as in E29: a callable returning bool, and the `SERVER_DOWN` Event.
- **`harness=True`:** tests only. It allows `think` below 60k and a `cfg["harness_floor"]`.
- **`attempt`:** an extra optional keyword (or `cfg["attempt"]`), 1 or 2. Attempt 2 = Poetiq expert 1, using E18's own seed and message formula with `expert_idx = 1`.
  - **The scheduler must pass `attempt=2` for the C′ and D′ stages. Otherwise attempt 2 repeats attempt 1's seeds.**
- **`cfg` defaults:**
  - budgets: think 60000, answer 8192, drain_answer 8192, temperature 1.0;
  - timing: drain_min 9.0, tok_s 18.0, tok_s_measured None, admit_margin_sec 60;
  - sandbox: sandbox_workers 2, sandbox_timeout 5.0;
  - other: max_error_retries 2, max_model_len 131072, attempt 1, branch None.

**Record (the E29 DIRECT record shape):**
- Common fields: `task_id`, `branch` (= `cfg["branch"]`), `status`, `accepted`, `generated_tokens`, `total_tokens`, `t0`, `t1`, `note`, `turns`, `requests`, `failures`, `b_budget`, `view="e33_poetiq"`, `usage`, `e33_poetiq=True`, `attempt`, `iteration`.
- `status` is `"done"`, `"not_admitted"` or `"not_run"`.
- `submissions = [{"test_outputs", "accepted", "train_exact", "t", "branch", "origin", "name"}]`:
  - `test_outputs` = this iteration's program run on every test input in the sandbox, with None for a failed test;
  - the list is empty when no request was sent.
- `accepted` = `train_pass` = Poetiq score 1.0.
- `poetiq`:
  - iteration and identity: `iteration`, `attempt`, `expert`;
  - admission: `admitted`, `reason` (`time` / `server_down_until_late` / `insufficient_context` / `cfg_think_below_floor` / `FAIL_CLOSED_TOKEN_IDS` / `sandbox_unavailable`), `admission` (the rule's numbers);
  - result: `score` (None if there was no program), `train_score` (0 to 1), `train_pass`, `test_usable`, `program`, `parse_status` (`ok` / `no_code_block` / `no_think_close`), `feedback`, `test_errors`;
  - phases: `mode` (`natural` / `continue` / `forced_close`), `end_reason`, `think_tokens`, `answer_tokens`, `think_closed`, `p1_tokens`, `p2_tokens`, `p1_finish`, `p2_finish`, `p1_cap`, `p2_cap`, `p2_kept`, `drained`;
  - seeds: `seed_message`, `seed_p1`, `seed_p2`;
  - errors: `request_errors`, `error_retries`;
  - feedback in the prompt: `feedback_solutions`, `feedback_trimmed`;
  - E18's ranked attempts over this attempt's programs: `e18_two_attempts`.
- `poetiq_state` (what iteration 2 needs): E18's per-(task, expert) state in JSON form:
  - `history` = `[{code, feedback, score}]`;
  - `candidate_history` = the full verification results;
  - `best_index`, `best_score`, `iteration`, `status`, `iterations`.

**Helpers for the scheduler:**
- `can_admit(deadline, cfg, now=None, harness=False) -> (ok, info)`: the same admission rule, usable as a start-time hook like E29D_CAN_START.
- `sandbox_ready(force=False)`: the E18 self-test, run once and cached. Poetiq sends no request unless it passed.
- `set_sandbox_workers(n)`.

### 3. E18 code copied verbatim
**From the Poetiq core (notebook cell index 5), every line except 673-679:**
- Prompts:
  - the adapted `POETIQ_SOLVER_PROMPTS` (10-18) and `POETIQ_FEEDBACK_PROMPT` (19);
  - the donor originals `SOLVER_PROMPT_1/2/3` (270-632) and `FEEDBACK_PROMPT` (633-639);
  - their override (642-643).
- Problem format and code parse:
  - `poetiq_diagram` (21-22);
  - `poetiq_format_problem` (24-34);
  - `poetiq_parse_code` (36-49);
  - `poetiq_create_examples` (51-55);
  - `poetiq_build_message` (57-66).
- Grid helpers:
  - `poetiq_normalize_grid` (68-92);
  - `poetiq_valid_grid` (95-96);
  - `poetiq_grid_from_output` (99-103).
- Sandbox:
  - `_POETIQ_TRUSTED_RUNNER` (110);
  - `poetiq_sandbox_run` (114-166; UID drop, rlimits, seccomp, 5 s timeout).
- Verification and voting:
  - `poetiq_array_diff` (168-170);
  - `poetiq_verify_code` (172-214);
  - `poetiq_test_key` (216-217);
  - `poetiq_mean_soft` (219-221);
  - `poetiq_vote` (223-244);
  - `poetiq_two_attempts` (246-265).
- Self-test: `poetiq_sandbox_selftest` (647-671).

**From CELL 1 (notebook cell index 2):**
- the `POETIQ_*` settings, lines 109-125;
- the asserts, lines 129-131.

The E10_KV gate lines 126-128 are left out. The E28 reservation hook replaces them.

**From `e29_direct.py`, copied verbatim:**
- constants, lines 36-55: SEED, MAX_MODEL_LEN 131072, CONTEXT_SAFETY, THINK_FLOOR 60000, ADMIT/HARD offsets, PER_STREAM_TOK_S, HEALTH_EVERY_SEC, TOP_P/TOP_K/MIN_P, STOP_THINKING_TEXT, FN_STOP_IDS, FN_THINK_CLOSE_ID, FN_EOS_IDS;
- functions and classes: `drain_need_min` (59-61), `phase1_cap` (143-145), `_Completion` / `_Output` / `request_body` / `StreamEngine` (315-485), `_health` (494-499).

**Builder checks:**
- E18's STOP_THINKING_TEXT and SEED are byte-identical to E29's.
- E18's top_p 0.95 / top_k 20 equal E29's TOP_P / TOP_K.
- The tests confirm that every copied region appears exactly once in the module.

**Re-implemented for one unit, following E18 cell 7 (index 7):**
- `e10_render_request` (40-50) → `poetiq_message`;
- `e10_seed` (52-53) → `poetiq_seed`;
- `e10_make_params` (55-60) → E29 `request_body`;
- `e10_admit` (65-96);
- `e10_start_phase2` (98-110);
- `e10_answer_text` (112-117: the text after the FIRST `</think>`);
- `e10_after_decode` (119-150);
- `e10_commit_iteration` (152-185);
- `e10_rank_task` (198-218) → `poetiq_rank_state`.

### 4. Every "# E33" change (module line numbers)
1. **Line 34: generation budget as literals.** REASONING_EFFORT xhigh, THINK_BUDGET 60000, ANSWER_MAX_TOKENS 8192, DRAIN_ANSWER_MAX_TOKENS 8192 are literals instead of E18's Q27X_* environment reads. The values pass through `cfg` (CELL 1 switches). xhigh is carried by `tok["render"]`.
2. **Line 65: two constants follow the E29 server.**
   - SEED is E18's own value, read from the E29 block.
   - MAX_MODEL_LEN = 131072, the Merlin server's limit. E18 used 106,496 on its own server.
3. **Line 771: import-time self-test removed.** E18's self-test block (cell lines 673-679) no longer runs at import. `E10_SANDBOX_SELFTEST = None`, and `sandbox_ready()` runs that exact block on demand and caches the result.
4. **Lines 776-821: sandbox CPU bound.** E18's runner sits unchanged behind one process-wide BoundedSemaphore, default 2 concurrent children (4 vCPUs shared with the server). E18 used a 4-thread pool. Timeouts and rlimits are unchanged.
5. **Lines 1014-1021: admission settings and rate samples.**
   - `drain_min` 9.0: covers one 8,192-token answer at 18 tok/s (7.6 min) plus margin.
   - `tok_s`, `tok_s_measured`, `admit_margin_sec` 60 s.
   - Measured-rate samples: phase-1 requests with at least 1,000 tokens, the last 16.
6. **Line 1103: best deduplicated by index.** `poetiq_rank_state` identifies the best by its index, because the state is JSON (E18 used `id()`).
7. **Line 1237: trim floor raised.** The feedback-trim floor is THINK_FLOOR 60,000; E18 used 49,152.
8. **Line 1449: no sandbox after the solve end.** No sandbox verification starts after T; the record says "not verified: past the solve end".
9. **Line 1470: best stored as an index.** `st["best_index"]` replaces E18's `st['best'] = result`.

**Other driver changes, all marked in place:**
- one iteration per call instead of E18's loop over iterations 1-10;
- attempt 2 = expert 1;
- `base_url` with or without `/v1`;
- E29 behaviour on this transport:
  - stop / `</think>` / EOS ids fail closed before any request;
  - EOS comes from `tok["eos_ids"]`;
  - phase-1 and phase-2 resends keep the streamed ids (E29 N1);
  - a request error is never a finish (wait for health or SERVER_DOWN, at most 2 resends, then `end_reason` error);
  - drain at T - drain_min: phase 1 aborted, forced close with STOP_IDS, answer of 8,192 tokens;
  - hard stop at T - 60 s, with the partial answer parsed;
- a below-floor `think` without `harness` returns a "not_run" record instead of E29's assert;
- thinking traces in `<out_dir>/<tid>_a<A>_i<I>/`: `transcript.txt` (header, PROMPT, GENERATED with special tokens, ANSWER), `token_ids.json`, `program.py`, `verification.json`, `branch_result.json`, plus `e33_poetiq_records.jsonl` and `e33_poetiq_requests.jsonl`.

### 5. The admission rule
- **Context:** `phase1_cap(prompt, think, answer 8192, 131072)` must be at least 60,000.
  - Otherwise E18's trim runs: drop low-score feedback examples, still in score order.
  - If it still does not fit: not admitted, `insufficient_context`.
  - So a prompt may be at most 62,624 tokens.
- **Time:** admit only if `now < ADMIT` and `now + admit_margin_sec (60) + 60000 / rate <= DRAIN`, where:
  - `DRAIN = T - drain_min * 60` (default 9 min);
  - `ADMIT = min(T - 12 min, DRAIN)`, as in E29;
  - `rate` = the lowest of `cfg["tok_s"]` (18), `cfg["tok_s_measured"]`, and the median of this module's recently measured per-stream phase-1 rates (once there are 3 or more samples).
- **Result at 18 tok/s:** a Poetiq item starts only if at least 3,393 s (56.6 min) remain before its drain point, i.e. about 65.6 min before T.
- **Server down:** the rule is re-evaluated while the unit waits for the server. If it becomes too late: `server_down_until_late`.
- A thinking cap below 60,000 is never admitted.

### 6. Test results
`e33_poetiq_tests.py` (stdlib unittest + numpy) uses the E29 fake-server pattern: a streaming token-id `/v1/completions` server, where the request seed selects the behaviour, and a stub tokenizer with the Flash-Next stop / `</think>` ids. The E18 reference is the notebook's cell 5 exec'd into its own namespace.

- **Results:**
  - Python 3.12.10: 32/32 OK in 42-46 s, run three times.
  - Python 3.14.6: 32/32 OK in 57 s.
- **Iteration 1:** a correct program gives train_pass, the test grids filled, and `e18_two_attempts` set. A wrong program gives feedback and score below 1, byte-equal to E18's `poetiq_verify_code`.
- **Iteration 2:** its phase-1 prompt is byte-identical to `render(E18.poetiq_build_message(task, 1, 0, history, …, seed=task_index*1000))`, both with `prior` = the record and with `prior` = the `poetiq_state`. The iteration-1 prompt is also identical. Iteration 2 after a parse failure carries no feedback.
- **Failure paths:** parse failure; sandbox timeout ("sandbox timeout" in the feedback); sandbox exception (feedback byte-equal to E18); multi-test with 3 tests, one failing test giving None for that test only.
- **Phases and continuation:**
  - forced close at the think cap: phase-2 prompt = prompt + p1 ids + STOP_IDS, seed +2;
  - an EOS before `</think>` is stripped;
  - continue mode;
  - drain gives a forced close with the kept ids;
  - hard stop.
- **Errors and server:**
  - a phase-1 error is resent with prompt + the 24 kept ids, cap reduced by 24, same seed;
  - a phase-2 error is resent with the 5 kept ids, cap 256 - 5;
  - retries exhausted gives `end_reason` error;
  - health down mid-run gives a wait, then a resend;
  - SERVER_DOWN set, then clear: the unit is admitted;
  - SERVER_DOWN until late: not admitted.
- **Admission:** not admitted for time (no request sent); not admitted for context; the production floor (60,000; `can_admit` boundary at ±0.5 s; `tok_s_measured` lowers the rate; `harness_floor` is ignored outside the harness).
- **Fail-closed and sandbox off:** fail-closed token ids or empty EOS sends no request; sandbox unavailable sends no request.
- **Sandbox bound:** 6 parallel runs with 2 workers show at most 2 live at once.
- **Seeds:**
  - the same `task_index` gives the same seeds and prompt;
  - seeds follow `SEED + ti*1000 + 1/2`;
  - attempt 2 adds 100 (expert 1);
  - iteration 2 adds 2.
- **Records and traces:** all trace files written, including one for a not-admitted record; record shape; JSON round trip.
- **Build:** the rebuild equals the file; every copied region appears exactly once; the E18 functions behave the same as the module's; LF only.

**Sandbox used by the tests:** on Linux as root with libseccomp (the Kaggle CPU kit), the REAL E18 runner. Elsewhere (Windows, here), a plain subprocess runner with E18's input / output / timeout contract but no UID drop, used for both the module and the exec'd E18 functions.

### 7. Risks, and what I could not do
1. **Real sandbox not run here.** The real E18 sandbox (UID drop, seccomp) cannot run on Windows. It needs `e33_poetiq_tests.py` in a Kaggle CPU kit, where it switches to the real runner automatically. No kit, audit zip or GPU test was done (no Kaggle calls allowed).
2. **Attempt 2 seeds.** The scheduler must pass `attempt=2` (or `cfg["attempt"]=2`) for C′/D′, or attempt 2 duplicates attempt 1's seeds.
3. **`accepted` = train_pass, as the spec says, even when a test input fails.** That test's `test_outputs` entry is None and `test_usable` is False. E18 would have kept iterating in that case (`train_pass_test_unusable`), while the chain here stops.
4. **Admission rate.** The default 18 tok/s comes from the E28 K100 Duck run, not from the E29 DFlash2 engine. Set `cfg["tok_s"]` or `tok_s_measured` from the E29 smoke. A higher rate admits later Poetiq items.
5. **Late verification.** Verification after a hard stop can take up to (train + test) × 5 s, plus semaphore waits when many units end together (2 workers). New verifications are blocked only after T.
6. **Feedback trim in iteration 2.** If the iteration-2 prompt is too long for the floor, E18's trim keeps 0 examples. Iteration 2 then runs without feedback, as E18 did. This needs a prompt over about 62k tokens, which is unlikely.
7. **Environment variables.** The verbatim `POETIQ_*` lines still read `E10_POETIQ_*` variables at import. The notebook must not set them.
8. **Source pins.** The builder pins `e29_direct.py` sha 8b42802c. If E29 regenerates that file, the build refuses until the pin is updated.
9. **Answer text.** Taken from the FIRST `</think>`, as E18 does; E29 Direct uses the last one.
10. **No LIVE line from this module.** It only logs `[E33 POETIQ] …`; the scheduler prints LIVE.
