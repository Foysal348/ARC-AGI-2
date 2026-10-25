# Experiment 14 — Optimized engine on 4xL4 (planned, opened 2026-10-23)

**Goal:** raise Qwen3.8-27B throughput on Kaggle 4xL4 toward the 600–800 tok/s that top teams say is needed. Throughput must not cost correctness.
- **Decode today:** SGLang W1 (INT4 + DFlash2) does ~330–380 tok/s over whole runs. A decode step at 12 running takes ~115 ms below 10k context and ~133–140 ms at 50–80k.
- **Effective today:** the rate inside the Duck agent is only ~167–280 tok/s.

**Why:** 240 hidden tasks in 12 h are bound by coverage.
- At about 355 tok/s, each task gets about 60k tokens. At 1.3× that is ~78k; at 1.6×, ~96k (Notes/00).
- The deadline is 2026-11-02.

**Scope:**
- Mainly the serving engine: launch flags, NCCL, verify width, and the load path.
- Changes to the Duck/solver code (prompt re-render, Mamba checkpoint handling, sticky routing) need the user's approval, under the rev 6g "last version" rule.
- Every GPU run needs:
  - the user's explicit go;
  - a written estimate of wall time, quota (2×) and stop point;
  - a check of the account's quota.

## Notes (everything is kept; nothing is dropped)

| File | What |
|---|---|
| `Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md` | Step time from our logs; the two suspects; what 1.3×/1.6×/2.4× would buy. Corrected in 02/05 |
| `Notes/01 GPT - GPU bottleneck research and validation (2026-10-23).md` | GPT note 1, verbatim |
| `Notes/02 Claude validation of GPT notes 01, 03 and 04 (= final report, 2026-10-23).md` | Copy of the final report. Verdict tables for about 95 GPT claims, Claude's errors, 24 conflicts, and the priority list |
| `Notes/03 GPT - E12 GPU bottleneck validation, second note (2026-10-23).md` | GPT note 2, verbatim |
| `Notes/04 GPT - GPU throughput bottleneck, independent validation, third note (2026-10-23).md` | GPT note 3, verbatim |
| `Notes/05 Claude - gaps found after full re-read (2026-10-23).md` | My 6 errors and 10 skipped items after the full re-read; why 5 more agents were launched |
| `Notes/06 Claude - E14-P0 offline token-diff result (2026-10-23).md` | Template and canonical tokenization round-trip exactly (170/170): the reply re-prefill is not a template bug; 3 remaining causes, GPU replay planned in E14-A |
| `Notes/07 Claude - E14-0 load path module built (2026-10-23).md` | `src/e14_loadpath.py` + 34 CPU tests; 6 problems found and fixed while building; the integration plan per G4B cell; the honest untested list |
| `Notes/08 GPT - review of the final report and P0 decision, fourth note (2026-10-23).md` | GPT note 4, verbatim: GO for both CPU P0 tasks, NO-GO for GPU; mandatory constraints |
| `Notes/09 GPT - independent assessment with five gaps, fifth note (2026-10-23).md` | GPT note 5, verbatim: 5 gaps (exact ids, copy integrity, recovery state machine, #44401 attribution, McNemar) + NCCL matrix, quota, account rules |
| `Notes/10 Claude - validation of GPT notes 08 and 09, changed plan (2026-10-23).md` | 26 GPT points checked. 5 new facts (server flags for exact ids; same-text-different-ids in our tokenizer; preserve_thinking verified; narrower token-diff scope; a short-write bug in my copy code). Redesigned correctness gate; what changes in the plan |
| `Notes/11 Claude - rev 2 load path and the re-prefill root cause (2026-10-23).md` | Load path rev 2 (56 tests, 6/6 runs, diff in Experiment Review). **Re-prefill cause: Duck deletes blank lines from the reasoning before storing it** (0/180 THINKING blocks have one). Server usage over 150 requests: 45 % of requests lose the reply (217.5k tokens); 18 deeper misses (66.8k) have a separate, earlier-message cause |
| `Notes/12 Claude - E14-P raw reasoning fix and the fake-server prefix probe (2026-10-23).md` | Fix built in arc2_duck as `E12_RAW_REASONING_HISTORY` (off by default; 15/15 unit checks; E12 smoke results identical old vs new). Fake server with the real template: switch off, 24/24 replies diverge at the first blank line; switch on, 24/24 full prefix; **0 client edits of earlier content**, so the deeper misses point to Mamba checkpoint eviction (E14-M) |
| `Notes/13 Claude - E14-A full plan (2026-10-23).md` | **Full E14-A plan.** New E14 builder (base untouched; proven fixes on by default; engine changes as measured arms). 9 decisions with adopt rules. Run A1 ≈ 61 min / ≈ 2 h quota (cap 95 min); run A2 ≈ 80–93 min (cap 110). Stop rules, quota order, E14-C scoring pipeline, calendar to 11-02. **The two-run shape was replaced on 10-23 05:50 by the user's "one notebook": one E14A profile of 200 min (Notes/16 §4e)** |
| `Notes/14 Claude agent - #44401 and #41872 backport for 0.5.21 (2026-10-23).md` | Agent note: the 6-hunk guarded backport (36 tests) and the finding that 11-of-12 is the DFlash min-free-slots delayer, not #44401 |
| `Notes/15 Claude agent - NCCL bench and trace summarizer (2026-10-23).md` | Agent note: `e14_nccl_bench.py` (arms, 983,040 B in CUDA graphs, adopt rule) and `e14_trace_summary.py` (33 tests) |
| `Notes/16 Claude - 3-day conversation insights and notebook fixes (2026-10-23).md` | **All rules, facts, failure classes and open items from the 10-20 → 10-23 conversation**, and the full E14-A notebook check list: 13 fixes applied (4 new today, including the `/start_profile` plain-text bug that would have broken the GPU profiler phase), 15 items checked against the 0.5.21 source with no change needed, GPU-only questions, items deliberately not built, plan reconciliation |
| `Notes/17`, `18`, `19 GPT - E14-A pre-GPU audit 1/2/3 of 3 (2026-10-23).md` | Three GPT audits of `a9dad9fc`, verbatim; all NO-GO with targeted fixes |
| `Notes/20 Claude - validation of GPT audits 17-19 (2026-10-23).md` | Every claim checked: **all true**. A P0 staging collision was reproduced (both Kaggle model dirs end in `/1`, so target and draft were staged into the same dir). The consolidated 13-item patch set awaits the user's go |
| `Notes/Conversation insights 2026-10-20 to 23/` | The 12 full reader reports (`insights_00` … `insights_11`), one per transcript part, 7 sections each |
| `Notes/Research - Qwen 27B fastest serving setup/` | First research (5 agents) and its final report |
| `Notes/Research - 4xL4 engine bottleneck validation/` | Second research (10 agents) and its final report (`00_FINAL_REPORT.md`) |
| `Notes/Kaggle discussions 2026-10-22 summary.md` | Discussion re-read: GCSFUSE slowdown, other teams' Qwen3.8 results |

## Key findings (final report, 2026-10-23)

- **Step anatomy (source + logs):**
  - Each step pushes 96 verify rows through the target.
  - Collectives per step: 140 all-reduces of 983,040 B, one 47.7 MB logits all-gather, and small gathers.
  - TP4 custom all-reduce is disabled on PCIe, so all of them go through NCCL. The transport is almost surely SHM: we measured 4.82 GB/s busbw at 64 MiB.
  - **Arithmetic budget for the step (only weights and KV are measured):**

    | Component | ms |
    |---|---|
    | All-reduce | 35–55 |
    | Logits all-gather + FP32 sampling | 6–14 |
    | GDN state snapshots | 8–10 |
    | Drafter | 6–7.5 |
    | Weights | 13–16 |
    | KV at 50k | 16–20 |
    | **Total** | 87–131, centre ≈108 (measured 115–133) |

- **Agent loss:**
  - Prefill runs at only ~1,500 tok/s and blocks decode, so re-prefill takes 13–55 % of the solve window.
  - In 72–91 % of later requests, the previous reply is prefilled again. A template re-render is suspected.
  - Prefix reuse is worth 3–4× more here than on an RTX PRO 6000.
- **Settled questions:**
  - accept len includes the bonus token.
  - Width 4 caps accept len at 4, and it is not a truncation of width 8 (the drafter's attention within the block is non-causal).
  - The GDN projections are INT4; only `in_proj_a/b`, norm, lm_head, MTP and vision stay BF16.
  - ReplaySSM + DFLASH is blocked in code (`kv_cache_configurator.py` L1129-1136).
  - #44401 is not in 0.5.22, only in nightly ≥ 10-16.
  - PCIe-IPC all-reduce needs FlashInfer ≥ 0.7; our 0.6.18 lacks it.
  - 2×TP2 holds only ~170k KV per replica, so it is ~0.7–1.0× of TP4: nearly rejected.
- **G4B ERROR on 10-22:** our code never ran (empty log, no runner warning). The cause is open; ranked likeliest first: attach/mount stall > pinned image > quota.

## Experiment list by priority (final, 2026-10-23)

| Priority | ID | Lever | Expected gain (status) | Cost | Gate |
|---|---|---|---|---|---|
| **P0** | E14-0 | **Load path** (module built, Notes/07; revised after GPT 08/09, Notes/10): 64 MiB ranged copy to `/dev/shm`/`/tmp`. Specifics:<br>• **adaptive 4 → 8 → 16 workers (cap 24)**<br>• **`.partial` + atomic rename**<br>• **per-range hash + destination re-read**<br>• safetensors no-hole check<br>• write loop (bug fix)<br>• cgroup memory check<br>• fail open, plus an off switch<br>• stall watchdog (log + rchar + page faults; 75-min give-up; never kill a progressing server)<br>• `E14_STATE` phase record<br>• JIT cache without `nv/` + fixed version/wheel-hash manifest + known-answer gate<br>• first-line instrumentation | Avoids a 0.00 run on slow storage. Load 669 s → copy ~3.6 min (if ~107 MB/s holds). JIT cache ~360 s per cold start (M/E). Does **not** help when the runner never reaches the notebook (10-22 case) | Module ~570 lines + tests; revision pending | `ready_sec`, MB/s per worker level, mount type, cgroup headroom, greedy known-answer match |
| **P0** | E14-P0 | **Offline token-diff** (done, Notes/06 + correction): **reconstructed divergence 0/170 on tool-call turns**; not proof about generated ids (the same text has several id sequences). Extension pending: every turn class (plain / malformed / last / after-trim), a same-text-different-ids test, and the after-trim share counted | Diagnosis only. A fix could give G4B ~+10 % | CPU only | Per-class counts; the exact ids come from the E14-A replay (`return_output_ids_in_sglext`) |
| **P1** | E14-A | **Measurement run (Tier A):**<br>• topology/P2P<br>• NCCL sweep incl. 983,040 B × 128 graph replay and the 11.9 MB/rank all-gather<br>• **NCCL first round 3 arms only: baseline / `NCCL_P2P_LEVEL=SYS` (if P2P possible) / `NCCL_SHM_USE_CUDA_MEMCPY=1`**<br>• TP4 **W1 unchanged** steady windows (8k/40k) with device timer<br>• profiler window by category<br>• **replay with exact ids** (`return_output_ids_in_sglext`) | Decides between the engine levers | ~35–45 min wall ≈ 1.2–1.5 h quota | ms per category per step, transport, µs at 983 KB, first divergent id at the reply boundary |
| **P1** | E14-S | **#44401 + #41872 backport** (monkey-patch on 0.5.21, applied only if the source hash matches, else fail loudly) **+ `--cuda-graph-bs-decode 1..13`**. Arms in order, after the unchanged baseline: +#44401, then +graphs | +3–8 % whole run (11 → 12 running is ~+8–9 % (M/D)) | Small patch; same session as E14-A | Held-row count, no negative free slots, real running == 12 during chunked prefill, latch-case replay |
| **P1** | E14-G | **Correctness gate (Tier B), redesigned (Notes/10 §C):**<br>• exact greedy parity D vs T with first divergence + margin<br>• A/A repeatability (T vs T, D vs D)<br>• #36548 probe = state contamination vs the A/A rate<br>• paired known-answer scoring on distinct tasks<br>• failure taxonomy<br>• CPU test that a worse arm fails | Risk removal; helps explain 29.17 vs 34.58 | +10 min; the answer probe ≈ 1–1.5 h quota at reduced size | Paired 90 % CI lower bound > −δ (δ preset, proposal 5 points) **and** parity at the A/A level. Never a p-value alone |
| **P2** | E14-W | **DFlash2 width 8/6/4 A/B (Tier C)** | W4 −10 % to +12 %, W6 −4 % to +7 % (model only) | 15–30 min ≈ 0.5–1 h quota | tok/s at equal context; accept len; answer probe |
| **P2** | E14-N | **NCCL transport env** (`NCCL_P2P_LEVEL=SYS/PHB` if P2P is OK, or the winning knob) | +2–16 % (consumer analog) | env; measured in E14-A | ≥ 20 % faster at 983 KB, sum check, greedy parity |
| **P3** | E14-U | **SGLang 0.5.22 upgrade** (+ #44401 backport); alternatives: nightly retag or vLLM 0.32 | Enables PCIe-IPC; hardening fixes | Wheelhouse rebuild (~220 wheels / 5 GB) | Paired tok/s by context band, greedy gate |
| **P3** | E14-I | **PCIe-IPC all-reduce** (needs FlashInfer ≥ 0.7 + P2P) | ~+20–45 % if all-reduce is 40–55 ms and can be cut; ~0 if small | After E14-U; one relaunch | Enabled-log line, ms/step, greedy parity |
| **P3** | E14-L | **Logits vocab-parallel verify sampling** | −6–14 ms → +5–12 % | Code patch + exactness test | Distribution equivalence |
| **P3** | E14-K | **GDN verify kernel config sweep** | −5–10 ms if GDN ≥ 15 ms | Monkeypatch + microbench | Bit-exact vs default |
| **P1 (built 2026-10-23, switch off; Notes/12)** | E14-P | **Duck reasoning-normalization fix** (approved; `E12_RAW_REASONING_HISTORY=1` turns it on): store the server's raw `reasoning_content` in history; keep the normalized text for logs only. Cause found in Notes/11 (`tool_agent.py` L3414-3418 deletes blank lines). Plus: find the earlier-message edit behind the 18 deeper misses (CPU, fake-server harness) | Removes up to 217.5k (+66.8k) re-prefilled tokens per G4B-size run, ≈ 146 (+45) s of prefill; ~+10 % effective (working estimate) | One override in arc2_duck; verify in the E14-A replay first | Raw vs normalized replay: cached ≈ prev prompt + reply |
| **P3** | E14-M | **Mamba checkpoint protection** (Franzen refresh-on-unlock, bigger mamba cache, int8 ckpt) | Up to ~+14 % (G4B S3 → S4) | Server patch port | Zero/partial hit share at low KV |
| **P4** | E14-C | **Confirm on a real Duck workload** (bundled winners) | Effective 279 → 330–400 (D/E) | G4B-style ≈ 1.5–2 h quota | Effective tok/s, solved, cached fraction |
| **P5** | E14-T | **2×TP2** (nearly rejected): only a boot test of one replica's memory | 0.7–1.0× | < 10 min optional | KV tokens per replica |
| later | E14-H | HiCache | ~0 at KV max 0.44; hybrid host-tier bugs | — | — |
| later | E14-V | Token-efficient variant (Swift-1.5 INT4) | −10–25 % tokens → 1.18–1.33× if accuracy holds | Paired EVAL | **Licence probably not prize-eligible** |

**Run plan (estimate):**
- **Full E14-A run (current, one notebook, 2026-10-23):**
  - Normal ≈ 145–165 min wall; worst = the 200 min budget plus Kaggle start/save (≈ 210 min).
  - Quota ≈ 4.9–5.5 h normal, ≈ 7.0 h worst (2×).
  - Only hosen42 (8.63 h at the last reading, 10-22) can hold it.
  - (The earlier 60–70 min estimate was for Tiers A–C before the Duck run and the A2 arms were added.)
- **Today, no GPU:** the E14-0 build and the E14-P0 token-diff.
- **Bundling:** submissions bundle winners (no single-change ablations).

## Status log

- **2026-10-23 (E14A first GPU commits, Notes/21-22):**
  - **Embedded notebook a5318dda (745 kB) never started:** INT4 mount error on 3 of 3 commits.
  - **Cause:** the commit notebook-size limit, between 407 kB (ran) and 444 kB (never ran). This probably also explains the 10-22 rev 6g G4B loss.
  - **Fix:** the SLIM build. Payloads moved to the private dataset `foysalemonshanto/q27-e14a-payload` (sha-named `.bin`); the notebook is 224 kB.
  - **SLIM run1** (3c67794e, foysal, stopped at about 25 min, about 0.8 quota-hours): it started, the payload loaded and staging ran at 518 MB/s. Further results:
    - NCCL: baseline is best (345 µs per 983 KB, about 56 ms of communication per step; P2P is 5.5× slower).
    - First launch 502 s, about 7 min of it compile; **the relaunch took 71 s**.
    - A1-2 crashed: `Sampler._stop` shadows `Thread._stop` on Python 3.12.
  - **R2** (2298d827) fixes the crash, adds 2 regression tests and passes 167/167 on Python 3.12 and 3.14.
  - **New gate:** the Kaggle CPU harness kit (`Build/build_cpu_kit.py`, kernel `q27-e14-cpu-harness`).

- **2026-10-23 (build 2 after GPT audits 17-19, Notes/20 §6):**
  - **Notebook:** `a5318dda`.
  - **Fixes (category A + B):**
    - staging collision fixed;
    - Duck reserve;
    - strict deadlines with one shared cleanup bound;
    - NCCL `graph_correct`;
    - gate-first correctness gate on every arm;
    - latch metrics 11 → 12;
    - fresh launch before the Duck run;
    - `E14_KEPT`;
    - 24 independent-review findings handled.
  - **Tests:** unit 165/165 ×3; harness 31/31 in both the default and corrupt scenarios.
  - **GPT bundle:** `Experiment Review/E14A_BUILD2_AUDIT_FOR_GPT_2026-10-23.zip`.
  - **Not pushed.**

- **2026-10-23 (E14-A notebook built and CPU-tested; 3-day insight review, Notes/16):**
  - **The notebook:** `Experiment Notebooks/Q27-XHIGH-E14A_OPTIMIZED_ENGINE_SGLANG.ipynb`, sha **a9dad9fc**, built by `Experiment Notebooks/Build/build_e14.py` with profile E14A and a 200 min budget.
  - **Fixes from the review:**
    - (a) the `/start_profile` plain-text reply, which would have broken the GPU profiler phase;
    - (b) per-phase server-log sign counts;
    - (c) `/flush_cache` before the Duck run;
    - (d) graphs13 adopted only if the server captured 12 and 13;
    - (e) a literal-CR build bug;
    - (f) peak watchdog stats.
  - **Tests:** fake-server harness 26/26; unit tests 125/125 three runs in a row; raw reasoning 15/15.
  - E12 notebooks unchanged.
  - **Not pushed;** waiting for the user's go and the written pre-run sheet.

- **2026-10-23 (E14-P built, Notes/12):**
  - arc2_duck switch `E12_RAW_REASONING_HISTORY` (sha 7644aa06 → 758bddb5, default off). Built notebooks are unchanged.
  - Fake-server probe confirms the cause and the fix on CPU, and rules out client edits for the deeper misses.
  - Next: the E14-A replay (token level) and E14-M.
- **2026-10-23 (rev 2 + root cause, Notes/11):**
  - Accounts confirmed as separate team members.
  - Load path rev 2 done: 56 tests, 6/6 runs.
  - Token-diff extended with server usage data. The reply re-prefill comes from Duck's reasoning normalization (blank lines deleted).
  - Analysis scripts are in `src/analysis/`.
  - A Duck fix needs approval.
- **2026-10-23 (GPT notes 4 and 5):**
  - Saved verbatim (Notes/08, 09). Claude's validation and the changed plan are in Notes/10; Notes/06 got a correction header.
  - Code changes are listed but not made yet: write loop; atomic `.partial`; range hashes; worker ramp; cgroup check; state record; JIT manifest; more failure tests; token-diff turn classes.
  - **Open user question:** account ownership vs the competition's multiple-accounts rule (Notes/10 B25).
- **2026-10-23 (later):**
  - **E14-P0 token-diff done (Notes/06):** A/B/C 170/170 each, so the template is not the cause.
  - **E14-0 module built (Notes/07):** `src/e14_loadpath.py` + `src/test_e14_loadpath.py`, 34 passed in 5 of 5 runs.
  - Not yet integrated into a notebook. Integration needs a NEW variant, the harness rerun and the user's go.
- **2026-10-23:**
  - Folder opened. Notes 00–05 saved.
  - Research 1 (5 agents) and research 2 (10 agents) done; final reports copied into Notes.
  - Priority list finalized.
  - Claude verified two report claims in the source: ReplaySSM is blocked for DFLASH, and the GDN projections are INT4.
- **2026-10-22:** the rev 6g G4B commit on esfersami50 ended in ERROR after ~114 min, with an empty log and no outputs. esfersami50 quota is exhausted (30.21/30 h). Remaining: foysal 4.71 h, hosen42 8.63 h.
