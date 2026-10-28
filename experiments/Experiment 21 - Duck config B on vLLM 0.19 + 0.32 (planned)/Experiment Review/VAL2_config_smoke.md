# VAL2 config + smoke: the E21 clock (3 profiles), CELL 1 / 1b / 3, the smoke push metadata (rev 2d, 2026-10-28)

**Scope:** a read-only review. I did not run the harness, the fake server or anything on Kaggle.

**What was read:**
- the four rev 2d notebooks, split into cells. The shas match the brief: NB1 `dddb5d0c`, NB2 `3ce943a2`, NB1_SMOKE `f063fea7`, NB2_SMOKE `18285c9c`. The two push-folder `.ipynb` files are byte-identical to the SMOKE notebooks (`f063fea7`, `18285c9c`).
  - SMOKE vs full notebook: only CELL 1 line 2 differs (the default profile `E21SMOKE` vs `E21X2EVAL120`).
  - NB1 vs NB2: only the markdown, `E21_VARIANT`, `E21_HALF_SPECS` and `E21_STAGE_PATHS` differ.
- `Build/e21_cell3_patch.py`, identical to the CELL 3 patch block;
- `donor_map/e14mods/e14_loadpath.py` (sha `7c66c179`, = `_E14_MOD_SHA`);
- `donor_map/duck_9e4b/arc2_duck.py` (sha `b9f67010`, = the manifest) and `vendor/inference/agent/tool_agent.py` (sha `f7454201`), `openai_compat.py`;
- both push `kernel-metadata.json` files; the earlier push metadata (E15B foysal / hosen42 / esfersami50, E18, E20);
- the payload folders `_scratch/e21/payload_foysal`, `_scratch/e15b/foysal_payload`, `_scratch/e15b/hamid_payload`;
- the E15B foysal run logs, the R3K2 run folder, the E12-V run 2 vLLM 0.19 log, the E18 v1 vLLM 0.32 log and the E15B SUBMIT foysal `e14_results.json` (Kaggle machine facts);
- the harness env block (`src/harness/harness_e21_nb.py`), the README, `E21_FINDINGS_TRACKER.md` and `VAL_timing_C1.md` (headings and findings).

**Notation:** T0 = `NOTEBOOK_T0` (anchored at kernel start). Every time is in seconds after T0.

---

## Summary of findings

| # | Severity | Finding |
|---|---|---|
| F1 | **BLOCKER (until verified)** | The smoke kernels attach `foysalemonshanto/q27-e15b-payload`. The last version on record (10-25, `_scratch/e15b/foysal_payload`, used by both foysal E15B runs) holds the Duck packages `19192e…` and `b2080…`, **not** `q27x_e15b_duck_9e4b89285136.bin`, which CELL 3 requires. `_scratch/e21/payload_foysal` (built 23:23 today) adds it, but nothing shows it was uploaded as a new version. |
| F2 | **HIGH (smoke only)** | In E21SMOKE, half A's fallback is starved. A's end cap is T0+3,900, and a launch's ready timeout is `cap − now − 30 min`. A primary that fails after its ready point (probe failure, ~T0+1,000-1,700) leaves the FP8 fallback 360-1,100 s. If the primary times out at T0+2,100, the fallback is skipped outright. NB1's untested INT4-on-0.19 primary is exactly the case the fallback exists for. |
| F3 | MEDIUM | The prefix-cache probe runs only when contract-probe turn 1 returns a tool call. "No call within 3,072 tokens" returns `ok` early, so the #53912 MTP + prefix-cache check is silently skipped. |
| F4 | MEDIUM (misleading smoke) | In a 20-min window a stream generates ~20-26k tokens: below every attempt budget (50k / 45k − 4,096 reserve) and the 57k sizing context. The smoke never exercises the CELL 3 `length` restore, the commit turn, the budget end / harvest, attempt 2's cap, KV pressure / preemption, or a watchdog relaunch (min-left rule 20 min > window). `length_restored = 0`, `preempt_lines = 0`, `relaunches = []` and the scores are therefore not evidence. |
| F5 | MEDIUM (affects the hidden rerun, not the smoke) | The contract probe and the cache probe are hard launch gates in **every** profile, `RERUN_ALL` included. A flaky "equal" (cold vs cached greedy reply) or marker miss on both configs gives no engine in the hidden run, so the identity submission ≈ 0. |
| F6 | LOW-MEDIUM | The cache probe's "grid digits" prefix is 9,000 copies of the digit `5`: `Random(7)` is re-created for every digit. That is ~18k tokens of one repeated digit, a repetition-attractor prompt for a hard gate, and not representative of Duck prompts. |
| F7 | LOW | The probe and install timeouts are not tied to the half end: the contract probe 2 × 600 s, the cache probe 4 × 300 s (2,700 s worst after ready), and the 0.32 install join 2,400 s. Pathological hangs can eat a whole smoke half or push the smoke past 2 h (the 12-h limit is not at risk). |
| F8 | LOW | CELL 9 reads `prefix_cache_hit_pct_last`, but the server-log summary key is `prefix_cache_hit_pct_last_rolling`. The report's prefix-hit field is therefore always `None`. |
| F9 | LOW | The harness always sets `E21_B_SETUP_RESERVE_SEC=300`, also in SMOKE mode, so the real smoke clock (reserve 1,800 s → A's cap T0+3,900) was never exercised. That is why F2 was not caught. |
| F10 | LOW (info) | E21SMOKE's 12 tasks are the old EVAL12 random sample (same seed, same code; 22 queries: 136b0064, 20a9e565, 21897d95, 247ef758, 2b83f449, 4c3d4a41, 4c7dc4dd, 6e4f6532, 88bcf3b4, 898e7135, 8b7bacbf, b6f77b65). v1.3.5 scored 1/22 on them. A 20-min score difference between halves is noise. |
| F11 | LOW | The README input list still names `esfersami50/q27-e15b-payload` (or hosen42); the push uses the foysal copy. The equal-window rule and `comparison_valid` ignore server-down time inside a half. |

Nothing else blocks the smoke. Everything listed under "Verified correct" at the end was checked against the code.

---

## 1. The clock, all three profiles (rev 2d values)

### 1.1 Constants and derived deadlines

| | E21SMOKE (commit) | E21X2EVAL120 (commit) | RERUN_ALL (hidden) |
|---|---|---|---|
| `TOTAL_BUDGET_SEC` | 7,200 | 43,020 (12 h − 3 min) | 43,020 |
| `HARD_STOP` (− `hard_margin` 120) | 7,080 | 42,900 | 42,900 |
| `DRAIN` / `ADMIT` (−480 / −120) | 6,600 / 6,480 | 42,420 / 42,300 | 42,420 / 42,300 |
| `VERIFY` (−60) | 7,140 | 42,960 | 42,960 |
| `_stop` = HARD_STOP − 180 | 6,900 | 42,720 | 42,720 |
| window W (`solve_window_sec`) | 1,200 | 19,620 (5.45 h) | none (absolute end) |
| reserve R (`E21_B_SETUP_RESERVE_SEC`) | 1,800 | 1,800 | not used (one half) |
| `E21_HALF_END_PC` half A = `_stop − W − R` | **3,900** | **21,300** | 42,720 (`len(E21_HALVES)==1` → `_stop`) |
| `E21_HALF_END_PC` half B = `_stop` | 6,900 | 42,720 | — |
| A gets a full window iff solve start ≤ | 2,700 (45 min) | 1,680 (28 min) | — |
| `SGL_GIVE_UP_AT` = min(T0+75 min (rerun 240), ADMIT − 5 min) | 4,500 | 4,500 | 14,400 |
| staging deadline = max(600, min(2,700, GIVE_UP − 1,200)) | 2,700 | 2,700 | 2,700 |
| staging join (CELL 5) = min(GIVE_UP − 1,200, 3,300) | 3,300 | 3,300 | 3,300 |
| 0.19 ready timeout = max(60, min(1,800, END_PC − now − 1,800)) | min(1,800, 2,100 − now) | 1,800 | 1,800 |
| 0.32 ready timeout = max(60, min(1,200, END_PC − now − 1,800)) | 1,200 while now ≤ 3,900 | 1,200 | — |
| CELL 1 assert `2W + R < HARD_STOP − T0` | 4,200 < 7,080 **true** (spare 2,880) | 41,040 < 42,900 **true** (spare 1,860) | skipped (`not RERUN`) |
| CELL 1 line 197 order assert | 0 < 6,480 < 6,600 < 7,080 < 7,140 < 7,200 **true** | true | true |

**How the window is placed (CELL 6):**
1. `_end = min(_stop, END_PC, now + W)`.
2. After the isolation preflight the end shifts by the preflight time, bounded by `min(_stop, END_PC)` (commit only).
3. `deadline_in_sec` ends up = W ± 1 s when not capped.

**Half B:**
- B gets `min(W, A's deadline_in_sec)` if A had an engine (`_a.get("used")`).
- `E21_B_FOLLOWS_A`: B copies A's fallback if A used it.

**Teardown (`e21_end_half`):**
- Non-last half: SIGTERM, ≤ 30 s, SIGKILL, ≤ 30 s, kill the GPU pids, `_wait_gpu_release` ≤ 180 s, then the scheduler-thread handoff wait ≤ 120 s.
- Last half: `kill(fast=True)` (0.1 s + SIGKILL + ≤ 30 s), no GPU wait, no handoff.

**The 0.32 venv** is built in a daemon thread, started at the end of half A's launch loop (`e21_begin_half`). It is never started in `RERUN` (no half B).

### 1.2 Setup-time model

**Install, staging and launch:**
- CELL 2: vLLM 0.19 install 160 s (E18: 161 s). Staging runs in parallel from CELL 1b.
- Staging copy at 680 MB/s, then verify. E15B foysal: 19.47 GB copied in 25.8 s, total 91.7 s, so verify ≈ 3.4 s/GB (a blake2b re-read plus a sequential sha256).
  - NB1: INT4 19.5 GB + FP8 ~30 GB = ~49 GB, so ~72 s copy + ~165 s verify ≈ **240 s**. The CELL 5 join returns at ~T0+250-300. Launch start **L0 ≈ 300**.
  - NB2: FP8 ~30 GB, so ~44 + ~100 ≈ 145 s. It finishes inside CELL 2. **L0 ≈ 180**.
- Server ready R: 550-1,200 s (0.32 / E18: 550 s; 0.19 / S1: 668-733 s).

**Probes:**
- strict smoke: ≤ 5 s;
- contract probe: turn 1 ≤ 3,072 tokens at a single stream (~45-60 tok/s), so ~50-70 s; turn 2 ≤ 512 tokens, ~10 s;
- cache probe: 4 requests of ~18k prompt tokens (see F6), ≤ 96 output tokens, ~20-40 s.
- Probes P total ≈ 90-180 s.

**CELL 6 isolation preflight:** 16/20 concurrent short requests plus 2 parity pairs, 10-40 s (one retry possible).

**Teardown of a non-last half:**
- the Duck join grace (≤ 90 s; requests are clamped to the deadline, so usually ≤ 30 s);
- CELL 7: report + zip (R3K2 EVAL120 Duck bundle: 10 MB, so seconds);
- kill + GPU release + handoff.
- Typical 60-150 s, worst ~465 s.

### 1.3 E21SMOKE timelines (both notebooks)

| Step | NB1 fast | NB1 slow | NB2 fast | NB2 slow |
|---|---|---|---|---|
| A launch start L0 | 300 | 300 | 180 | 180 |
| A ready (+R) | 850 | 1,500 | 730 | 1,380 |
| probes done (+P) | 940 | 1,680 | 820 | 1,560 |
| A solve start S_A (+preflight) | 955 | 1,720 | 835 | 1,600 |
| A solve end E_A = S_A + 1,200 (cap 3,900) | 2,155 | 2,920 | 2,035 | 2,800 |
| B begin (+teardown 90 / 465) | 2,245 | 3,385 | 2,125 | 3,265 |
| 0.32 venv done (started ≈ probes end, 175 s) | ~1,115 | ~1,855 | ~1,000 | ~1,735 |
| B ready timeout | 1,200 | 1,200 | 1,200 | 1,200 |
| B solve start S_B (+R_B + P + pre) | 2,900 | 4,805 | 2,780 | 4,685 |
| B solve end E_B = S_B + 1,200 (cap 6,900) | 4,100 | 6,005 | 3,980 | 5,885 |
| notebook end (+fast kill, CELL 9) | ~4,150 (69 min) | ~6,060 (101 min) | ~4,030 | ~5,940 |

**Can the smoke fit 2 h?**
- **Yes.** Every solve end is bounded by 6,900, and the tail after it is at most ~150 s (join grace 90 + CELL 7 + fast kill ≤ 31 + CELL 9).
- The only exits past 2 h are the unbounded probe and install timeouts of F7: pathological hangs. Even then the Kaggle 12-h limit is far away; only GPU quota is spent.

**Do both smoke halves get their 20-min window?**
- **Yes, in every non-failure case.** S_A ≤ ~1,720 < 2,700, and S_B ≤ ~4,805 < 5,700.
- B's window = `min(1,200, A's deadline_in_sec)` = 1,199-1,200 (rounding), so `windows_equal` / `comparison_valid` hold.

**Failure cases (smoke):**
- **A primary crashes during load** (~L0+300 = 600):
  - fallback timeout = min(1,800, 3,900 − 600 − 1,800) = 1,500;
  - FP8 ready 550-1,200, so S_A ≈ 1,300-2,000 and A still gets its full window.
- **A primary is ready, then fails a probe** at t_f ≈ 940-1,680 (+ kill / GPU ~30-60 s):
  - fallback timeout = max(60, 2,100 − t_f) = **1,100 … 360 s**;
  - FP8 needs 550-1,200, so it usually fails. **Half A is lost** (F2).
- **A primary hits its 1,800-s timeout** (t_f ≈ 2,130):
  - `time > END_PC − 1,800 = 2,100`, so the loop breaks and the fallback never starts. **Half A is lost.**
  - A lost half A has no `used`, so B runs its primary with the full window (no equal-window or follow rule).
- **B primary times out** (B begin + 1,200 = 3,445-4,585):
  - fallback timeout = min(1,200, 6,900 − t − 1,800) = 1,200 … 515;
  - so the fallback is fine on the fast path and marginal on the slow path.

### 1.4 E21X2EVAL120 commit timeline

| Step | Typical (NB1) | Slow (NB1) | Worst that still ends in time |
|---|---|---|---|
| S_A | 955 | 1,720 | 1,720 |
| E_A = min(S_A + 19,620, 21,300) | 20,575 (full) | 21,300 (A window 19,580, −40 s) | 21,300 |
| B begin | ~20,700 | ~21,450 | 21,765 (teardown 465) |
| B ready timeout | 1,200 | 1,200 | 1,200 |
| S_B | ~21,355 | ~22,870 | 23,185 |
| B window = min(19,620 or A's, 42,720 − S_B) | 19,620 | 19,580 (= A) | 19,535 vs A 19,620 (−85 s, 0.43 % < 1 %: valid) |
| E_B | 40,975 | 42,450 | 42,720 |
| notebook end | ~41,050 | ~42,520 | ~42,800 (< 43,200) |

**What the reserve covers:** the 30-min reserve covers teardown + B ready + probes + preflight up to 1,800 s. The all-worst sum is ~1,925 s, which shortens B by ≤ 125 s, still within the 1 % validity rule.

**B fallback** (primary timed out, fallback 1,200 s): B's window is ~18,275 (−6.9 %), giving `CONFOUNDED_FALLBACK` / `comparison_valid = False`. That is by design.

**A's launch and fallback timeouts** are always 1,800 here (A's cap is 21,300), so F2 does not apply to the full commit.

### 1.5 RERUN_ALL timeline (half A only, 240 tasks)

**Setup:**
- staging: NB1 49 GB, NB2 30 GB (same as above);
- launch timeout 1,800, fallback 1,800;
- no window shift (RERUN): the solve ends at absolute 42,720 (11:52:00);
- no 0.32 install thread.

**Tail:**
1. Duck join grace ≤ 90 → 42,810.
2. The final `e12_write_submission()` at the end of CELL 6, before CELL 7.
3. CELL 7 (report + a Duck bundle of ~20 MB, seconds).
4. Fast kill ≤ 31 s, no GPU wait, no handoff.
5. CELL 9.

**End:** ≈ 42,830-42,900, leaving 300-370 s before 43,200. The tail is no longer than R3K2's (the same margins).

**Risk specific to the rerun:** F5. The probes are hard gates here too.

### 1.6 Answers to the clock questions

- **SMOKE fits 2 h:** yes (69-101 min with realistic setup). Bounded by 6,900 + ~150 s, except F7 hangs.
- **Both smoke halves get 20 min:** yes, unless a config fails (then F2 for A).
- **Can anything exceed the budget?** Only via F7 (probe timeouts of up to 2,700 s after ready; the 0.32 install join of up to 2,400 s), which are not bound to the half end. In the 12-h profiles these can only shorten a window, never pass 12 h (every solve end is absolute and the tail is ≤ ~180 s).
- **CELL 1 assert for E21SMOKE:** true (4,200 < 7,080). It is static: it only proves the windows fit if A started at T0 (the C1 L2 note still applies). The real A setup budget is 2,700 s.
- **E21SMOKE task selection:**
  - `n_tasks` 12, no `task_list`, so the order cell takes the `random.Random(TASK_SAMPLE_SEED=20261005).sample(sorted(eval), 12)` branch, sorted.
  - That is exactly the `EVAL12` logic and sample (the same seed, code and file). It is the set E10 R4 ran ("12 tasks, 22 queries"; ids in F10).
  - It is **not** the EVAL12R6 / G4B hand-picked list.

---

## 2. CELL 1 / CELL 1b / CELL 3 as built

### 2.1 CELL 1

**Profile row.** `E21SMOKE`: `source` eval, `n_tasks` 12, `time_budget_sec` 7,200, admit/drain fractions 0.90/0.94, hard 120 / verify 60 / drain 480 / admit 120, `solve_window_sec` 1,200. That is identical to the builder's row (build_e21.py L165). `E21X2EVAL120` is unchanged: 43,020, W 19,620.

**Asserts:**
- the profile allow-list (`E21X2EVAL120`, `E21SMOKE`, `RERUN_ALL`, or `E21_ANY_PROFILE=1`);
- "two halves need a window";
- `2W + R < HARD_STOP − T0`.
- All three are true for both commit profiles. RERUN takes the `RERUN` branches.

**Window.** `E12_SOLVE_WINDOW_SEC` is re-derived after the E21 block from `E21_HALF_SOLVE_SEC` (env) or the profile, and is `None` for RERUN. `E21_HALF_SOLVE_SEC = W or 0`. Correct.

**Tokenizer fallback.**
- L326 sets `E12_TOKENIZER_JSON = SGL_MODEL_PATH/tokenizer.json` (the INT4 path, after `_e11_discover`).
- L421-422 replace it with the FP8 `tokenizer.json` only when the INT4 file is missing and the FP8 one exists.
- With the INT4 model attached (both smoke kernels attach it), Duck counts with the INT4 file, as R3K2 did.
- `Path("")` is not a file, so an unset variable is also handled. Correct.

**CONFIG refresh.**
- `CONFIG["e11"]["ENGINE_BACKEND"] = "vllm"`;
- `E12_SOLVE_WINDOW_SEC` is re-recorded;
- `rev6.E12_R6_CACHE_DIAG = 0` (and the env `"0"`, so the SGLang cache diagnostic after solving never runs);
- `E12_STREAMS` is marked "per half";
- `CONFIG["e21"]` holds the variant, halves, specs, window, reserve, max_model_len, served names, ports, stage list, the 0.32 wheelhouse and the history key.
- Correct. Cosmetic: the SGLang-only fields in `CONFIG["e11"]` stay (harmless, already DEFERRED in the tracker).

### 2.2 CELL 1b: staging

**NB1 stages two folders.** `E21_STAGE_PATHS = [SGL_MODEL_PATH (INT4), MODEL_PATH (FP8)]`.

`E14L.Staging` → `stage_models`:
- **Sizing.** It sums both sizes (≈ 49 GB) and calls `choose_dest(total)`.
- **The /dev/shm rule:** free ≥ 1.02 × 49 + 2 GiB ≈ 52.1 GB, and `host_headroom()` ≥ 49 GB + 24 GiB (25.8 GB) = 74.8 GB.
- **On the Kaggle 4×L4 box** (E15B foysal `e14_results.json`): /dev/shm total / free 91.8 GB; cgroup limit 185.8 GB with 0.4 GB used; MemAvailable 194.6 GB. So headroom ≈ 185 GB: **both rules pass**, and the destination is /dev/shm.
- **After staging:** 42 GB are left in /dev/shm and ~136 GB in the cgroup for 4 TP workers + the Duck sandboxes (4 GiB address-space cap each). Fine. Half B reuses the same staged files.
- **Names are unique** (`m0-w4a16-g128-1-<sha1>`, `m1-hf-fp8-1-<sha1>`), and resolved paths are distinct, so there is no collision refusal.
- **Order.** The models are copied one after another (INT4 first). If the deadline (T0+2,700) hits during the FP8 copy:
  - the verified INT4 copy is kept;
  - the FP8 partial copy is deleted;
  - `warm_page_cache` runs for FP8 with the same, already-expired deadline;
  - FP8 then loads from `/kaggle/input`.
  - Fail-open, correct. At ~240 s expected, the deadline is far.
- **CELL 5** joins until min(3,300, 3,300) = T0+55 min and fills `E21_STAGED` by matching `str(SGL_MODEL_PATH)` / `str(MODEL_PATH)` against the returned map. `e21_model_path` uses that map, else the original paths. Correct for both halves, both configs.
- `e14_after_install` (SGLang only) is never called. The CELL 5 join replaces it. Correct.

**NB2** stages only FP8 (~30 GB; the same rules pass easily). Its fallback uses the same weights. Correct.

**Read probe:** 64 MiB of `E21_STAGE_PATHS[0]` (INT4 for NB1, FP8 for NB2), about 1 s.

**CELL 2's `cat` prefetch** is skipped (`E14_STAGING is not None`). CELL 2 builds its tokenizer from `MODEL_PATH` on `/kaggle/input` (FP8 attached). Correct.

### 2.3 CELL 3: the `finish_reason` patch

**Does patching the base class reach `Arc2ToolAgent`?** Yes.
- `make_agent_class()` builds `class Arc2ToolAgent(TA.ToolAgent)`, where `TA` comes from `arc2_duck._import_duck()` → `from inference.agent import tool_agent as TA`. That is the same `sys.modules` object as CELL 3's `_E12_TA` (no reload anywhere in arc2_duck).
- `Arc2ToolAgent._chat_completion` (arc2_duck.py 1914) calls `super()._chat_completion(messages, **kwargs)` (1940). `super()` resolves through the MRO at call time, so it lands on the patched `TA.ToolAgent._chat_completion`, i.e. `_e21_chat_completion`. That in turn calls the saved original.
- Order of effects: Arc2's pre-request code, then the HTTP call (original), then the patch restores `"length"`, then back in Arc2's override (1950-2003) the rev 6 `length` / no-call accounting, and the rev 5 / E15B cut-call handling (the B cut-code harvest, invalid-JSON arguments) sees `"length"`. This is the SGLang behaviour R3K2 had.

**Is `res.usage` a dict with the right keys?** Yes.
- `_ChatCompletionResult(message=…, finish_reason=str(...), usage=payload.get("usage"), served_by=…)` is a mutable dataclass, so `res.finish_reason = "length"` works.
- Duck's payload is non-streaming (`build_chat_payload(stream=False)`; streaming only for openrouter, and the provider is `vllm`). vLLM's non-stream response carries `usage` with `prompt_tokens` / `completion_tokens` / `prompt_tokens_details`.
- The patch guards against a non-dict `usage` (it then does nothing).

**`max_tokens` kwarg name:** correct.
- The original signature is `(self, messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None)`.
- The main turn passes no `max_tokens` (tool_agent.py 6785: `tools` + `request_timeout_seconds`), so the request used `self._max_output_tokens`. The patch falls back to the same attribute.
- `_max_output_tokens` is set by Arc2's cap function before the request, and nothing on that agent changes it during the call (one thread per agent).
- The rolling-summary call passes `max_tokens=`, which the patch reads first.

**Can a correct, complete tool call be falsely turned into "length"?**
- The patch fires only on `finish == "tool_calls"` and `completion_tokens ≥ min(cap, 135,168 − prompt)`. That is vLLM's own length-cap condition (`num_output_tokens ≥ max_tokens` or `num_tokens ≥ max_model_len`; MTP output is truncated to `max_tokens`, so ct == cap exactly when capped). A reply that ends naturally below the cap is never touched.
- **Edge case 1, EOS exactly at the cap.** I could not open vLLM's `v1/core/sched/utils.py check_stop` (not in `_scratch/vllm_src`), so I could not confirm whether EOS or length is checked first. Either way, a complete call ending with EOS exactly on token `cap` (probability ~1/cap per request) becomes "length", which is what SGLang (length checked first) also reports.
- **Edge case 2, a complete call followed by more text cut at the cap.** It becomes "length", so arc2 rev 5 drops **all** calls of that reply, including the complete one. This is again exactly the R3K2 / SGLang behaviour, so there is no regression versus the donor.
- **Conclusion:** the patch is correct and has no false positives beyond donor parity.

**Not exercised by the smoke:** F4. The request cap in B mode is the remaining budget − 4,096 (≈ 45.9k / 40.9k), and a 20-min stream generates ~20-26k, so no request is cut at `max_tokens`. Requests at the deadline end by the clamped timeout, not by length.

---

## 3. Push metadata (both smoke folders)

| Input | Notebook default path | Metadata source | Mount (Kaggle rule) | OK? |
|---|---|---|---|---|
| FP8 `MODEL_PATH` | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` | `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1` | framework lower-cased: same | yes |
| INT4 `SGL_MODEL_PATH` | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` | `…/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1` | same (also seen mounted in the E15B foysal env report) | yes |
| DFlash2 `SGL_DRAFT_PATH` | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1` | `…/qwen3-8-27b-dflash2/PyTorch/bf16/1` | same. Unused, but attached so `_e11_discover` finds its marker and does not rglob `/kaggle/input` | yes |
| `SGL_WHEELHOUSE` | `/kaggle/input/datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` | `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` | same (seen mounted, fuse.gcsfuse) | yes |
| `WHEELHOUSE` (vLLM 0.19) | `/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` | `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` | same; proven by S1 / E18 / E20 | yes |
| `E18_VLLM_WHEELHOUSE` (0.32) | `/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` | `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` | same; proven by E18 (`rglob install_wheelhouse.py`) | yes |
| Competition `ARC_ROOT` | `/kaggle/input/competitions/arc-prize-2026-arc-agi-2` | `arc-prize-2026-arc-agi-2` | same | yes |
| Payload `q27x_e14_modules_55c1eb7911c6.bin` | searched by name + sha up to `/kaggle/input/*/*/*/` (= `datasets/<owner>/<slug>/<file>`) | `foysalemonshanto/q27-e15b-payload` | — | yes (in the 10-25 foysal version; used by both foysal E15B runs) |
| Payload `q27x_e15b_duck_9e4b89285136.bin` | same search | `foysalemonshanto/q27-e15b-payload` | — | **not in the last recorded foysal version → F1** |

**Other metadata:**
- `docker_image` `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…bf3fc647d461` is the same pinned image as every earlier E15B / E18 / E20 push. CELL 2 asserts Python 3.12. Correct.
- `machine_shape` `NvidiaL4`, `enable_gpu` true, `enable_internet` **false** (all installs are offline from the wheelhouses), `is_private` true. Correct.
- `id` slug = the slugified `title` = `code_file` stem (`q27-xhigh-e21-duckb-vllm-nb1-int4-s20-smoke`, `…-nb2-fp8-ssm16-s16-smoke`). Correct.
- **Notebook size:** 250,696 / 250,650 bytes, < 400 kB (the builder asserts < 400 × 1024). Correct.

**Kaggle quick-save:** the memory rule says to push with `QUICK_SAVE` and then commit. That is not visible in the metadata; it is the push command's job.

---

## 4. Findings in detail

### F1: BLOCKER (until verified). The foysal payload may not contain the R3K2 Duck package

**Location:**
- the push `kernel-metadata.json` `dataset_sources`: `foysalemonshanto/q27-e15b-payload`;
- CELL 3 L9: `q27x_payload("q27x_e15b_duck_9e4b89285136.bin", "9e4b8928…")`.

**Evidence:**
- `_scratch/e15b/foysal_payload` (10-25, the foysal dataset folder) holds `q27x_e14_modules_55c1eb7911c6.bin`, `q27x_e15b_duck_19192e8006e0.bin` and `q27x_e15b_duck_b20804213563.bin`.
- The foysal E15B runs used those: 0f0d6f67 → `19192e`, 560ed8b8 → `b2080`.
- `9e4b` was only in `hamid_payload` (hosen42) and the esfersami50 payload (R3K1 / R3K2).
- `_scratch/e21/payload_foysal` (10-28 23:23) has all four bins and `dataset-metadata.json` id `foysalemonshanto/q27-e15b-payload`, which means a new version is intended. Nothing here shows it was uploaded and processed.
- The E21 CPU kit kernels attach their own kit dataset, not this payload, so the kit never tested this input.

**Scenario:**
- CELL 1b loads the e14 modules (present), then CELL 3 raises `FileNotFoundError: payload q27x_e15b_duck_9e4b89285136.bin … not found`.
- Run All stops at ~T0+3-4 min, so both smoke notebooks are wasted (small GPU cost, one day's slot).

**Fix:**
- Upload `_scratch/e21/payload_foysal` as a new version of `foysalemonshanto/q27-e15b-payload`.
- Wait until Kaggle has processed it (the kernel attaches the latest version at commit time).
- Then push. The version's file list should show `q27x_e15b_duck_9e4b89285136.bin`.
- Also update the README input list (F11).

### F2: HIGH (smoke only). Half A's fallback is starved by the 30-min ready margin under a 20-min window

**Location:**
- `e21_launch`: `timeout = max(60, min(cap, E21_HALF_END_PC − now − 30*60))`;
- `e21_begin_half`: `if time.perf_counter() > E21_HALF_END_PC − 30*60: break`;
- `E21_HALF_END_PC(A) = _stop − W − R` = 3,900 for E21SMOKE.

**Evidence:** §1.3.
- A probe failure after the ready point (t_f ≈ 1,000-1,740 with the kill and GPU release) gives the FP8 fallback 1,100-360 s, against the 550-1,200 s it needs.
- A primary ready timeout (t_f ≈ 2,130) skips the fallback (break).
- In the full commit A's cap is 21,300, so the fallback always gets 1,800 s. The `−30 min` margin was sized for the 5.45-h window.

**Scenario:**
- NB1 primary = INT4 on vLLM 0.19, untested (README Finding 5; GPT 2 #58807).
- If it loads and then fails the contract or cache probe (or its capacity is < 8), the fallback has no time and half A has no engine. The smoke reports nothing for 0.19.
- B then runs the INT4 primary (A has no `used`, so no follow rule). The smoke never shows that NB1's FP8 fallback path works, which is the one thing the full run might need.

**Fix (any one of):**
- **(a)** E21SMOKE `time_budget_sec` 2.5 h (9,000). Then A's cap = 8,700 − 3,000 = 5,700, and the fallback timeout = min(1,800, 5,700 − 1,740 − 1,800) = 1,800. A normal smoke still ends at ~70-100 min (the budget is a cap).
- **(b)** Make the margin `min(30 min, E12_SOLVE_WINDOW_SEC)` in both places. Then the A fallback gets 3,900 − t_f − 1,200 ≈ 960-1,700 s: better, not always enough.
- **(c)** Smoke-only: `E21_B_SETUP_RESERVE_SEC` 30 min stays, but the A cap is computed from `_stop − W_B_min`. This is more invasive.
- (a) is the simplest and changes nothing in the full notebooks.

### F3: MEDIUM. The cache probe is skipped whenever turn 1 has no tool call

**Location:** `e21_contract_probe`: `if not calls: out.update(ok=True, turn2="inconclusive…"); return out`. This returns before `out["cache"] = e21_cache_probe(served)`.

**Scenario:**
- At `reasoning_effort` low and T 0, Qwen3.8 may think past 3,072 tokens, or answer without calling the tool.
- The launch is then accepted without the MTP + prefix-cache check (#53912), the very check rev 2d strengthened to 96 greedy tokens.
- `contract.cache` is absent from the report, which is easy to misread as "passed".

**Fix:** run `e21_cache_probe` unconditionally (before the early return, or in `e21_launch` after the contract probe). Record `turn2="inconclusive"` separately.

### F4: MEDIUM (misleading smoke results). What a 20-min half cannot test

**Evidence:**
- ~19-23 tok/s per stream (E18: 382 tok/s at 20 streams; R3K2 ~22) × 1,200 s gives ~23-28k generated per stream.
- Attempt budgets are 50,000 / 45,000, with a 4,096 commit reserve and a request cap = remaining − 4,096.
- `_WD_MIN_LEFT` is 20 min and `E21_RELAUNCH_MIN_SEC` 12 min.

**Not exercised:**
- (i) the CELL 3 `length` restore (no request reaches `max_tokens`);
- (ii) the commit turn / weak commit / reasoning harvest / cut-code harvest;
- (iii) the end of attempt 1 and the stage-2 / stage-3 scheduling in a meaningful way (12 tasks × a1 occupy 12 of 16/20 slots at once; some a2 start immediately);
- (iv) KV pressure. Contexts stay ≤ ~35k (sizing assumes a 57k peak), so `preempt_lines = 0` and `kv_usage_pct_max` are not evidence for 5.45 h;
- (v) a watchdog relaunch (the left time is always < 20 min);
- (vi) the scores: see F10.

**Exercised, and valuable:**
- real loads of the INT4 / FP8 configs on 0.19 and 0.32;
- the capacity lines, `lmo_in_log`, the strict smoke, the contract + cache probes;
- the Duck tool-call contract at 16-20 real streams;
- `reasoning` history growth; prefix-cache hit rates; tok/s at full concurrency;
- the handoff (kill, GPU release, scheduler threads), the 0.32 background install, the equal-window rule, the per-half report.

**Fix:**
- State in the smoke result note that (i)-(vi) are not covered.
- Optionally, add to the contract probe one forced cut: tools + `max_tokens` ≈ 300, asking for a long python call. Check that the raw finish is `tool_calls` and that the Duck path sees `length` (call `Arc2ToolAgent`'s patched base once, or assert `ct ≥ cap` → restored).

### F5: MEDIUM (hidden-rerun risk). The probes are hard gates in RERUN_ALL

**Location:** `e21_launch` calls `e21_contract_probe` → `e21_cache_probe` for every launch in every profile. Any `RuntimeError` rejects the config.

**Scenario:**
- In the hidden rerun, both configs of half A must pass "own marker in both replies", "cold == cached at T 0 for ≤ 96 tokens", "cached ≥ 1,024", and the turn-2 shortfall rule.
- A numerics flip (E19 measured flips with MTP when prefix caching is off; on is believed fine) or a marker miss on both configs gives `ENGINE_READY = False`. The identity submission scores ≈ 0 on the LB.
- In a commit the same failure costs only a half.

**Fix (decision for the user):**
- in `RERUN`, keep the hard failures (HTTP error, markup in content, degenerate output, non-JSON arguments, missing capacity lines);
- record `equal` / `own_marker` / `cached` / shortfall as warnings, unless the failure repeats on a retry.
- The GPU smoke will show how stable the probes are on both versions.

### F6: LOW-MEDIUM. The cache-probe prefix is one repeated digit

**Location:** `e21_cache_probe`: `rows = " ".join(str(_rnd.Random(7).randrange(10)) for _ in range(9000))` re-seeds for every element, so every element is `5`.

**Evidence:** reproduced locally. 17,999 chars, 1 distinct digit; with Qwen's digit / space pre-tokenisation that is ~18k tokens (the docstring says ~5k, the brief ~9k).

**Scenario:**
- An 18k run of `5 5 5 …` followed by "Reply with exactly this code" is a repetition-attractor prompt at T 0 with thinking off.
- A degenerate reply fails `own_marker` and rejects the config (F5 makes this a hard gate). It is also a poor stand-in for Duck's real prompts (block alignment is the same, the content is not).

**Fix:** `_r = _rnd.Random(7); rows = " ".join(str(_r.randrange(10)) for _ in range(9000))` (as `_e12_cache_diag._blob` does). Optionally use rows of 30 to look like grids.

### F7: LOW. Timeouts not bound to the half end

**Location:**
- `e21_contract_probe`: `timeout=600` × 2;
- `e21_cache_probe`: `timeout=300` × 4;
- `e21_launch` 0.32: `_th.join(timeout=2400)`, then `e18_install` `timeout=2400`.
- The launch ready timeout is computed after the install join.

**Scenario:**
- A server that is "ready" but generates pathologically slowly can take up to 2,700 s in probes. In SMOKE that is longer than A's whole window, so A starts solving with ≤ 0 s left; in B it can push the notebook past 2 h.
- A hung 0.32 install delays B by up to 40 min. The B window is then cut by the absolute 6,900 / 42,720 end.
- 12 h is never at risk.

**Fix:**
- clamp each probe timeout to `max(30, E21_HALF_END_PC − now − W_min)`;
- compute the 0.32 install join timeout from `E21_HALF_END_PC − now − 30 min`.

### F8: LOW. A CELL 9 key mismatch

**Location:** CELL 9 `"prefix_cache_hit_pct_last": _sl.get("prefix_cache_hit_pct_last")`. `_e15b_server_log_summary` returns `"prefix_cache_hit_pct_last_rolling"`.

**Scenario:** the E21 table always prints `prefix hit None %`. The other cache numbers (`cached_pct_requests`, the counters' `prefix_hit_pct_whole_run`) are still correct.

**Fix:** read `prefix_cache_hit_pct_last_rolling` (and add `counters.prefix_hit_pct_whole_run` to the table).

### F9: LOW. The harness never runs the real smoke clock

**Location:** `harness_e21_nb.py` L146 sets `E21_B_SETUP_RESERVE_SEC="300"` for every mode. The SMOKE branch (L161-) only pops `E21_ANY_PROFILE`.

**Effect:** A's cap in the harness is `_stop − W − 300`, not `− 1,800`, so F2's starvation cannot appear there.

**Fix:** in SMOKE mode, keep the notebook's reserve and scale it with `E14_TIME_SCALE` like the window. Add a `halfAfail`-late variant: the primary fails after the ready point, and the test asserts that the fallback gets ≥ 550 s × scale.

### F10: LOW (info). The smoke task set and scores

**Tasks:** the E21SMOKE sample = the EVAL12 random sample (136b0064, 20a9e565, 21897d95, 247ef758, 2b83f449, 4c3d4a41, 4c7dc4dd, 6e4f6532, 88bcf3b4, 898e7135, 8b7bacbf, b6f77b65; 22 queries; v1.3.5 got 1/22; E10 R4 ran the same set).

**Implications:**
- With 20-min windows expect 0-3 query hits per half.
- The A-vs-B difference and the attempt-1-only score are noise.
- `stage1.attempt1_all_done_h` will be `None` or cut.
- If a smoke accuracy signal were wanted, the hand-picked EVAL12R6 / G4B list (fast solvers like 2ba387bc, bf45cf4b) would give more. That is not needed for a mechanics smoke.

### F11: LOW. Documentation and validity details

- **README input list.** README "Inputs each GPU notebook must attach" names `esfersami50/q27-e15b-payload` (or hosen42). The smoke push uses the foysal copy (correct for the foysal account; private datasets of other users are not attachable). Update it together with F1.
- **Server-down time.** `comparison_valid` and the equal-window rule use `deadline_in_sec` (the planned window). Server-down time inside a half (watchdog relaunch, `SERVER_DOWN` waits) is not subtracted or flagged. `relaunches` / `server_down_waits` are in the report, but not in `comparison_valid`.
- **Fix:** add `no relaunch in either half` (or the down-seconds within 1 % of the window) to `comparison_valid`.

---

## 5. Verified correct

**Clock:**
- All derived deadlines for E21SMOKE / E21X2EVAL120 / RERUN_ALL are in §1.1, and the order assert holds for all three.
- Half A's cap (`_stop − W − R`), half B's cap (`_stop`), and RERUN's single half capped at `_stop`.
- The window is placed after the preflight (shift bounded by the half cap). B gets `min(W, A's achieved)` only when A had an engine. `E21_B_FOLLOWS_A` copies A's fallback.
- The 0.32 venv builds in the background during half A (started after A's launch loop, also when A failed) and is never started in RERUN. The install error is cached.
- The 0.32 ready timeout is 20 min; the 0.19 timeout is 30 min.
- The handoff wait is ≤ 120 s (non-last half only, after the kill and GPU release).
- The last half kills fast with no GPU wait or handoff.
- `_wait_gpu_release` ≤ 180 s (`Q27X_GPU_WAIT_MAX`).
- **Both smoke halves get their full 20-min window in every non-failure case. The smoke fits in 2 h (≈ 69-101 min).** The full commit keeps the halves within 1 % in all realistic cases. The rerun tail leaves ≥ 300 s before 12 h.

**Task selection:** the E21SMOKE selection uses the EVAL12 sampling code and seed (12 tasks, 22 queries).

**CELL 1:**
- the profile row equals the builder's;
- the three asserts are true;
- the tokenizer fallback logic;
- the CONFIG refresh;
- `E12_R6_CACHE_DIAG` off;
- the ports (A 1234, B 8011, distinct);
- the served names are never `"qwen"`;
- the history key `reasoning` (CELL 3 and per half).

**CELL 1b / CELL 5 staging:**
- NB1 copies INT4 and FP8, both into /dev/shm;
- the `choose_dest` free-space and headroom rules pass with the measured Kaggle values (91.8 GB shm, 185 GB cgroup headroom vs 52.1 / 74.8 GB needed);
- names are unique, the publish is atomic, and a partial copy is deleted;
- the deadline T0+45 min and the join T0+55 min (all three profiles) are far beyond the ~4 min expected;
- `E21_STAGED` maps both weights; launches use the staged paths, else the originals.

**Capacity parsing:** `e21_capacity` matches the real log lines.
- 0.19: "Setting attention block size to 1600 tokens …" and "Maximum concurrency for 135,168 tokens per request" (E12-V log).
- 0.32: "Setting attention block size to 1600 tokens" and "Maximum concurrency for 106,496 tokens per request: 10.43x" (E18 log).
- E18 FP8 on 0.32 gives 855 blocks, so 855 // (36 + 15) = 16 streams at 57k.

**CELL 3 patch:**
- reaches `Arc2ToolAgent` through `super()` (the same module object);
- `usage` is a dict with `completion_tokens` / `prompt_tokens` (non-streaming vLLM response);
- the `max_tokens` kwarg name matches Duck's signature, and the fallback to `_max_output_tokens` equals what the payload sent;
- the 0.32 clamp is handled;
- no false "length" beyond SGLang / R3K2 parity;
- the Build file and the notebook block are identical.

**Push metadata:**
- every model and wheelhouse path in CELL 1 matches an attached source under Kaggle's mount rule (framework lower-cased);
- the competition is attached; the image is pinned (the same sha as every earlier push);
- NvidiaL4, GPU on, internet off, private; the slugs are consistent;
- the notebooks are 250.7 / 250.6 kB (< 400 kB);
- the push notebooks are byte-identical to the built SMOKE notebooks.
- The only open input is the payload's `9e4b` file (F1).
