# Hybrid solver plan: Duck rev 6g stage 1 + S1-60K plain request in stage 2 (design only, 2026-10-24)

Base: `Q27-XHIGH-E14S_SUBMIT_W6G13_DUCK_SGLANG.ipynb` (sha 3dde5cbd, 13 cells, adapter arc2_duck.py sha 758bddb5).
S1 source: `Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb` (sha ed530071 = the LB 34.58 build).
Cell extracts used for this plan: `D:/ARC AGI 2/_scratch/hybrid_design/extract/{e14,s1}/cellNN_*.txt`.

## 0. Premise correction
S1-60K is NOT a direct-grid answer. It is Poetiq program synthesis:
- the model writes `def transform(grid: np.ndarray)`;
- the program runs in S1's own sandbox on every train pair and every test input;
- attempts are program outputs.
Details: donor SOLVER_PROMPT_1, 1 expert, up to 10 iterations, T 1.0. LB 34.58 came mostly from iteration 1.

What the hybrid's "plain request" is: S1 iteration 1, exactly.
- 60,000 max_tokens in phase 1.
- If the thinking is not closed: forced close (STOP_THINKING_TEXT) + 8,192 answer tokens.
- Then the last ```python block that defines transform is taken, and verified on train + test.

Duck's sandbox cannot run it: SAFE_MODULES has no numpy. So the S1 sandbox has to be ported:
- uid drop + seccomp + rlimits (`_POETIQ_TRUSTED_RUNNER`);
- it needs root and libseccomp.so.2;
- both were proven on Kaggle in S1 and E11-06.

## 1. Stage-2 policy (decided)
1. **Stage 1 (unchanged rev 6g):**
   - Duck branch 0 for every task, in E12_ORDER (ORDER2 + cells).
   - It pauses at 28,000 tokens.
   - 12 workers; KV admit gate 0.95.
   - Stage 2 waits for `stage1_left == 0`.
2. **Hybrid class per task** = rev 6g `stage2_class`, but with the plain-verified signatures added. Plain-verified means passed train and every test grid is valid.
3. **Items per class:**

   | Class | Meaning | Items, in order | Plain? |
   |---|---|---|---|
   | 0 | Duck partial progress, nothing verified | resume0 (one fair-share chunk), then plain | yes |
   | 1 | nothing | plain, then resume0 | yes |
   | 2 / 3 | verified | resume0 only | no (`HY_PLAIN_FOR_VERIFIED=0`) |

   - **Duck branch 1 is OFF:** E12_BRANCHES defaults to 1. With `HY_DUCK_B1=1` it is queued after plain.
   - Why: in R2, b1 gave 0 new outputs; 4 of 5 b1 starts produced 0 tokens before the deadline.
4. **Rank of a plain item:** `(1, plain_iterations_done, 0, 1, ORDER2 idx, sub)`.
   - The class slot is fixed to 0, so plain coverage runs in ORDER2 order across ALL unverified tasks. Class-0 resume chunks interleave with it.
   - Class-1 resumes and every resume round ≥ 1 come after all round-0 plain items.
   - At pop: if the task has become verified, the item is dropped (counted).
5. **One plain request per task** (`HY_PLAIN_MAX_ITER=1`).
   - With 2, Poetiq iteration 2 is queued, with S1 feedback, at round 1.
6. **Admission:**
   - global KV < 0.95 (unchanged);
   - plain also needs KV < 0.85 (`HY_PLAIN_KV_ADMIT`; this is S1's SGLang gate). If blocked, the worker takes the best NON-plain heap entry; if there is none, it waits. No spin.
   - plain live ≤ `HY_PLAIN_MAX_LIVE` (default = workers, i.e. 12; 6 if isolation is not PASS);
   - plain only if time left ≥ `HY_PLAIN_MIN_ADMIT = max(240, 1.05*((60000+2500)/r_stream + 30))`:
     - r_stream = the median per-stream tok/s of finished plain requests, default 32 → about 2,070 s;
     - so no plain is PLANNED below 60k thinking;
     - otherwise the item is dropped (`HY_PLAIN_LATE_FILL=0`; **user decision**: S1 itself admitted late, drained requests).
7. **Deadlines:**
   - Phase 1 is aborted at deadline − 420 s, i.e. HARD_STOP − 10 min; then forced close + 8,192.
   - Phase 2 is aborted at deadline − 5 s.
   - Verification runs only while deadline + 40 − now ≥ 5 s.
   - The runner returns before the 60-s join grace. It polls `eng.step()` (0.05 s) and never blocks on the socket.
8. **Prompt / sampling = S1 byte-for-byte:**
   - `poetiq_build_message(task, 0, 0, [], variant=1, selection_probability=1.0, max_solutions=5, improving_order=True, shuffle=True, seed=0+task_index*1000)`;
   - one user message, `apply_chat_template(enable_thinking=True, reasoning_effort=REASONING_EFFORT="xhigh")`, also in --effort low/medium builds;
   - T 1.0, top_p 0.95, top_k 20;
   - `phase1_cap(len(ids))`, which is 60,000 at MAX_MODEL_LEN 155,648; a cap < E11_THINK_FLOOR → record `insufficient_context`, never sent.
   - Use S1's values (shuffle=1, max_solutions=5), NOT E14 CELL 1's E11-06 P1 values (shuffle 0, max_solutions 2).

## 2. Merge into submission (pass@2), per test input qi
```
D = Duck candidates in exactly select_attempts_r5 order, tagged (grid, verified)
P = plain candidates: passed & grid valid at qi = verified; others by (score desc, t desc)
a1 = Dv[0] | Pv[0] | Pu[0] | Du[0] | test input          (first that exists)
a2 = first g != a1 in [other-method verified, same-method verified[1:], other-method unverified,
                       same-method unverified, test input]; else a1
```
- **Regression property:** with no plain rows, the result is identical to select_attempts_r5. It is tested.
- Two distinct verified grids from the two methods → both are submitted.
- Plain verification is per input (S1 F3); Duck's "accepted" still needs every test grid valid.

## 3. Throughput / time arithmetic (hidden 240 tasks)

**Clock:**
- RERUN_ALL budget 43,020 s; HARD_STOP = T0 + 42,900 s; Duck deadline = 42,720 s.
- READY is at about 783 s and the preflight takes about 30 s, so the window is W ≈ 41,900 s (11.64 h).

**Rates:**

| Rate | Value | Source |
|---|---|---|
| Duck agent rate | 287 tok/s | R2: 430,062 tok / 1,500 s |
| Plain decode | 380 tok/s | E11-06 P1-60K on SGLang W1: 379.9. Width 6 adds +5 %, but only measured at 6k context |
| Mixed stage 2 | 340 tok/s | assumption |

**Cost of one plain request:**
- R5 iteration-1 records re-capped at 60k: mean 58.7k tokens. Only 26/120 close naturally ≤ 60k; the forced answer median is 2,075 tokens.
- On Duck-unverified (harder) tasks: about 60-61k.
- Prompt median 6,250 tokens, max 17,617, so the resident peak is about 68k, at most about 80k.
- Wall time: one stream at about 32 tok/s → about 31-33 min.
- Machine time: 60k/380 = 158 s (2.6 min of the whole 12-stream server); at 290 tok/s it is 207 s.

**All 240 tasks plain-only:**
- 14.1-14.6M tokens = 10.3-10.7 h at 380 tok/s. It fits in W.
- At 290 tok/s it would need 13.5-14 h, which does not fit.

**Hybrid:**
- **Stage 1:** 240 × 31.5k (R2: 377,802 tok / 12 stage-1 runs) = 7.56M = 26,340 s (7.3 h). At 28k per task: 6.72M = 6.5 h.
- **Stage 2:** 15,560-18,500 s × 340 = 5.3-6.3M tokens.
  - About 85 % goes to plain → **75-89 plain requests**.
  - The rest is about 0.6-0.9M of class-0 resume chunks.

**Unverified tasks after stage 1 (U):**

| Stage-1 verify rate | U |
|---|---|
| 17 % (R2, hard G4B tasks: 2 / 12) | 199 |
| 25 % | 180 |
| 35 % | 156 |

- **So plain reaches only about 38-57 % of the unverified tasks.**
- Plain for all of U needs 9.4-11.9M tokens, i.e. 6.9-8.7 h of stage 2 at 380 tok/s; only 4.3-5.1 h exist.

**Formula** (plug in the E14S EVAL120 G4B commit's measured stage-1 tokens and verify rate):
`N_plain ≈ (W − 240·b0/r_duck) · r_mix · f_plain / c_plain`

Each −4k of Duck first exposure frees about 0.96M tokens, i.e. about 16 plain requests. **Not recommended:** in R2, the first check() of b0 came at a median of 27.7k tokens, and solved G4B branches defined transform at 18.4k and 26.2k.

**Evidence for the method mix** (12 G4B tasks, 19 inputs):
- Duck R2 (25 min): 6/19.
- Poetiq R5 (77.8k, eval run): 13/19. Caveat: G4B was chosen to include tasks that S49/R5/DIRECT solved.
- On the 8 tasks Duck left unverified, R5 got 8 of 13 inputs (136b0064, 898e7135, 4c3d4a41 ×2, 62593bfd ×2, f931b4a8 ×2).
- R5 on all of EVAL120: "Poetiq if verified, else + DIRECT" = 55.1 % vs 48.6 %. The second method pays where the first failed train.

## 4. Implementation (exact anchors)

### 4a. arc2_duck.py → the builder applies the hybrid patch in memory to the chosen base and overlays it
- The base is 758bddb5, or the E16 adapter: E16 edits lines 1284-1974 only, and these edits are 2271-3001, so they are disjoint.
- The overlay uses the E16 mechanism: zlib + b64 + sha assert, before `import arc2_duck as E12A`.

1. `def select_attempts_r5(task: dict, branch_records: list[dict]) -> list[dict]:` stays unchanged.
   - Add after it: `_r5_tagged(task, rows)` (the same logic, returning [(grid, verified)] per qi), `_plain_tagged(task, rows)` and `select_attempts_hy(task, duck_rows, plain_rows)` (§2), plus `LAST_SOURCES` (provenance per task/qi).
2. In `def build_submission(tasks: dict, records: dict[str, list[dict]]) -> dict[str, list[dict]]:`, replace `            sub[tid] = select(task, records.get(tid) or [])` with:
   - split the rows by `r.get("kind") == "plain"`;
   - `select_attempts_hy(...)` if there are plain rows, else `select(task, duck_rows)`.
3. Keep `def stage2_class(sessions: list[BranchSession]) -> int:`.
   - Add an inner `task_class(tid)` inside solve_all_r6 = the same rules over the Duck sigs ∪ `plain_sigs[tid]`.
   - Used only when `plain_runner` is set.
4. Signature: append `, plain_runner: Callable[..., dict] | None = None, plain_opts: dict | None = None` to the end of `kv_poll_sec: float = 5.0, kv_stale_sec: float = 30.0) -> dict[str, list[dict]]:`.
   - `plain_opts` keys: kv_admit, max_live, max_iter, for_verified, late_fill, drain_sec, think_budget, r_stream_default, b1.
5. `"publish_seq": 0, "task_sigs": {}}` → add `"plain": {...counters...}, "plain_tokens": 0`, plus `plain_state = {}` and `plain_sigs = {}`.
6. `def rank(tid: str, b: int, sub: int) -> tuple:`
   - first line: `if b == "plain": return (1, len(plain_state.get(tid, {}).get("iterations", [])), 0, 1, idx[tid], sub)`;
   - use `task_class` instead of `stage2_class` when hybrid is on.
7. `share_tokens()`, at the line `gen = sum(s.agent.generated_tokens for s in sessions.values() if s.agent is not None)`:
   - add `+ state["plain_tokens"]`;
   - subtract the committed plain budget (60k × (queued + live plain)) before dividing by the non-plain items. The min_chunk floor stays.
8. `stage2_items()`, at the line `order_ = {0: ("resume0", "start1"), 1: ("start1", "resume0"), 2: ("start1", "resume0"), 3: ("resume0", "start1")}[cls]`:
   - in hybrid mode use `{0: ("resume0","plain","start1"), 1: ("plain","start1","resume0"), 2: ("start1","resume0")(+"plain" if for_verified), 3: ("resume0","start1")}[task_class(tid)]`;
   - add a `kind == "plain"` arm: push `(tid,"plain")` if it is not queued/active, iterations < max_iter, and the task has no plain pass.
   - With E12_BRANCHES=1, the existing guard `elif branches > 1 and (tid, 1) not in sessions and (tid, 1) not in queued:` keeps start1 off.
9. Worker pop, at `prio, _, tid, b, kind = heapq.heappop(heap)` (after `hold_end(time.monotonic())`):
   - if `heap[0][4] == "plain"` and the plain KV / live gate says wait → take `alt = min(e for e in heap if e[4] != "plain")` (`heap.remove(alt); heapq.heapify(heap)`); if there is none → `cond.wait(min(5, kv_poll_sec)); continue`;
   - after the existing stage-1 wait block, for a plain item: drop if the task is verified (unless for_verified) or `left < plain_min_admit()` (unless late_fill);
   - put this before `if prio[0] == 1 and kind == "start1" and b == 1:`;
   - in front of `s = sess(tid, b)          # created under the lock (stage2_class iterates the dict)`, branch `if kind == "plain": state["plain"]["live"] += 1; tag = {"stage": 2, "kind": "plain"}` and skip sess().
10. Worker run, at `t0 = time.monotonic()` / `try:` / `if s.status != "done":`: a plain branch first.
    - `rec = _run_plain(tid, tag)`, which calls `plain_runner(tid, iteration, history, deadline, drain_at=deadline-drain_sec)` under try/except → error record;
    - it updates plain_state / plain_sigs / task_sigs novelty, `state["plain_tokens"]` and the r_stream samples;
    - then `publish(rec)`, the bookkeeping (running−1, live−1, active.discard, durations / dur2), `stage2_items(tid)`, `notify_all`, and `continue`.
11. `SCHED_LAST.update(...)`: before `gauge_fields=[...]` add `plain={run, passed, usable, dropped_verified, dropped_time, kv_waits, live_max, tokens, r_stream_p50, min_admit_last}`.
12. The plain row is a record dict with these fields:
    - `{task_id, branch:"plain", kind:"plain", status:"done", t0, t1, accepted:passed&usable, generated_tokens, total_tokens, usage:[{prompt,cached,completion}], runs:[{stage:2, kind:"plain", tokens, sec, status, new_in_branch, new_for_task}], plain:{iterations:[...S1 base_rec fields + test_grids/train_soft/score/passed/test_usable], candidates:[...]}, note}`;
    - CELL 7's `r.get` paths handle it.
    - Hybrid OFF (`plain_runner=None`) → the code paths are byte-identical to rev 6g.

### 4b. Notebook, via build_e14.py `--hybrid`
- **B1 (CLI flag)** after `E16 = "--e16" in sys.argv`:
  - add `HYBRID = "--hybrid" in sys.argv`;
  - NAME `...E14HY...`;
  - `S1_NB` = the S1-60K path, sha-asserted ed530071.
- **B2 (overlay):** generalize the `if E16:` overlay block so the overlaid file = (E16 adapter or the base) + the hybrid patches (once() each); sha pinned and printed.
- **CELL 1 (c1):**
  - `E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))` → default 1 when HYBRID;
  - append the `HY = {...}` env-overridable block (HY_ON 1, KV_ADMIT 0.85, MAX_LIVE 12, MAX_ITER 1, FOR_VERIFIED 0, LATE_FILL 0, DRAIN_SEC 420, R_STREAM 32, DUCK_B1 0, EARLY "" (smoke only; `assert not RERUN or not EARLY`));
  - add `assert THINK_BUDGET >= E11_THINK_FLOOR`;
  - add the profiles E14HYS (12 G4B tasks, budget 45 min, solve_window 20 min, HY_EARLY = first 2 ORDER2 tasks: exercises phase 1 → drain → forced close → verify → merge) and E14HYEVAL120G4B (= E14SEVAL120G4B, hidden_equiv, 6.5 h);
  - put these in the same once() that adds E14A/E14S.
- **New CELL 5c (c_hy)**, inserted between `code(c_meas1),` and `code(c_solve)` in the `nb = {"cells": [...]}` list:
  - The VERBATIM S1 cell 5 source, read from the sha-checked S1 notebook, with `E10_SANDBOX_SELFTEST` → `HY_SANDBOX_SELFTEST` (3 occurrences, asserted).
  - **MANDATORY rename:** otherwise a failing S1 self-test would overwrite Duck's self-test and disable the whole solver.
  - Then `hy_plain_run(tid, iteration, history, deadline, drain_at)`:
    - it renders the S1 prompt (§1.8) and encodes it;
    - asserts the cap;
    - creates a PRIVATE `E11SglangEngine(E11_SERVER)`: its own queue, its own stats = per-request prompt/cached/completion;
    - sends rid `hy~{tid}~i{n}|1`, `SamplingParams(max_tokens=cap, temperature=1.0, top_p=0.95, top_k=20)`;
    - polls `step()`; at drain_at → `abort_request([rid], wait_sec=min(30, left))`, `p1_finish='drained'`;
    - phase 2 = S1's `e10_start_phase2` rule (strip EOS; prefix + STOP_IDS if not closed; cap min(8192, MAX_MODEL_LEN−len−8));
    - text after THINK_CLOSE_ID → `poetiq_parse_code`;
    - saves the FULL generation to `OUT/hybrid_plain/thinking/{tid}_i{n}.txt` (S1 R4T header) and the program;
    - runs `poetiq_verify_code(code, CHALLENGES[tid], 5.0)`, deadline-clamped;
    - returns the iteration record + candidate + usage.
  - `HY_PLAIN_OK = HY["on"] and HY_SANDBOX_SELFTEST["ok"] and ENGINE_READY`. A failure disables plain only; the Duck runs.
- **CELL 6 (c_solve)**, at `                          stage2_p90_cutoff=bool(E12_R6["E12_R6_P90_CUTOFF"]))`:
  - add `, plain_runner=hy_plain_run if HY_PLAIN_OK else None, plain_opts=dict(HY, think_budget=THINK_BUDGET, max_live=min(HY["max_live"], _workers))`;
  - E12_RUN["hybrid"] = {ok, selftest, opts}.
- **CELL 7 (c_rep):**
  - after `"harvested_programs_correct": ...})` add `plain_ran`, `plain_verified`, `plain_correct`, `a1_source`, `a2_source` (from `E12A.LAST_SOURCES`);
  - `report["hybrid"]` = SCHED_LAST["plain"] + pass1/pass2 by source + union;
  - in the bundle loop `for fp in sorted(E12_OUT.rglob("*")):`, also add `OUT/hybrid_plain/**`;
  - `_gen = E12A.build_submission(...)` needs no change.

## 5. Risks
1. **Coverage binds:** plain reaches about 40-57 % of the unverified tasks (§3). The tail of the ORDER2 order gets no plain.
2. **KV:** a plain stream peaks at 68-80k against Duck's per-stream window of 72,228. 12 near-peak plain streams could exceed 866,744 → retractions. The 0.85 plain gate is the guard; HY_PLAIN_MAX_LIVE is the fallback.
3. Long plain streams evict paused Duck histories → more re-prefill on resume (watch stage2_resume_uncached).
4. Width 6 is unmeasured at long context. The plain rate of 380 tok/s is an assumption.
5. **Late-tail idle:** with LATE_FILL=0, the last ~34 min run only Duck chunks.
6. Plain verified-but-wrong is about 21-31 % (R5: 69 % of verified fully right). It matters only with 3 or more candidates.
7. **Notebook size:** about 227.8 + 72 (overlay) + 45 (S1 cell 5) + 15 (driver) ≈ 360 kB, against the 400 kB limit. The fallback is to move the S1 core into the payload dataset.
8. **E10_SANDBOX_SELFTEST collision** (above) and the vars E14 shares with E11-06 P1 (POETIQ_* in CELL 1): use the HY / S1 constants explicitly.
9. A hung stream: the shim thread is a daemon and the runner returns on its own clock (no straggler). abort_request wait_sec must be clamped to the time left.
10. SGLang ignores seeds, so runs are not reproducible (the same as E11-06).
11. The 45-min smoke cannot test the natural ≤ 60k close path. The harness covers it.

## 6. CPU-harness checks (local Py3.12 + Kaggle CPU kernel, pinned image, root)

**Unit tests:**
- (a) `select_attempts_hy` == `select_attempts_r5` with no plain rows: on R2's real `q27x/duck/*/branch_result.json` plus 1,000 random cases.
- (b) A merge table: Duck-only, plain-only, both same, both distinct, verified + unverified, both unverified, plain unusable at one qi of a multi-test task, empty → test input; a2 != a1 unless only one grid exists.
- (c) **Prompt parity:** hy ids == S1 `e10_render_request` ids on 20 eval tasks (S1 cell 5 + config exec'd from the sha-checked S1 notebook).
- (d) parse_code on R5's saved answers.
- (e) The phase machine (natural / length-unclosed → forced / closed-but-length → continue / drain / hard stop / stream error) with a scripted engine, against S1 `e10_after_decode`.
- (f) The scheduler with a fake agent + fake plain_runner:
  - no plain before stage1_left == 0;
  - plain only for class 0/1;
  - dropped when verified before pop;
  - the 0.85 gate skips to a non-plain item (no spin) and waits when only plain items remain;
  - the time guard; the live cap;
  - publish_seq order; no stragglers;
  - the runner returns by deadline + 15 s;
  - **hybrid OFF: events/records identical to rev 6g** (timestamps masked).
- (g) share_tokens with plain tokens (no ≤ 0 / div0).
- (h) Every sent plain request has cap == 60,000; a cap < 60k is recorded and never sent.

**Notebook harness** (`harness_e14hy_nb.py`, from harness_e14s_nb.py) + fake server:
- `hy~` rids stream scripted token ids, pre-tokenized with the FP8 fixture tokenizer: natural + correct, natural + wrong, unclosed → phase 2 code, drain-aborted, error.
- Modes:
  - commit E14HYS;
  - rerun (RERUN_ALL, 5 test tasks: submission = hy selection of the records);
  - eval120 (the gate-4b formula);
  - hybrid-off (= E14S results);
  - forced S1-sandbox failure (`HY_FORCE_SANDBOX_FAIL=1`: Duck runs, plain is disabled, the submission is valid).
- Also check: thinking files saved, programs saved, provenance counts, NB < 400 kB, every once() matched.

**Kaggle CPU kernel:**
- the REAL Poetiq self-test (uid drop / seccomp) + `poetiq_verify_code` on R5 programs (passing and failing), from 12 threads concurrently, with Duck's prlimit proxy installed;
- plus the full harness rerun mode.

**GPU:**
- the E14HYS 45-min smoke (≈1.5 h quota);
- then either submit or run E14HYEVAL120G4B (6.5 h) for the measured coverage.
