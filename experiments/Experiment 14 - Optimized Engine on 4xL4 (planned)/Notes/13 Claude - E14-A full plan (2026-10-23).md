# E14-A: full plan (Claude, 2026-10-23)

**User, 2026-10-23:**
- Give the full E14-A plan.
- Change the whole pipeline if needed.
- Do not hold the experiment back with "the base will break"; the base is preserved.

**Consequence for this plan:**
- **The base stays untouched.** rev 6g notebooks, `build_e12_duck.py` and the E12 notebooks are not edited or rebuilt.
- E14 gets **its own builder and notebooks**. Inside E14 every change that is proven on CPU is **on by default**:
  - the load path rev 2;
  - the stall watchdog;
  - the raw-reasoning fix (`E12_RAW_REASONING_HISTORY=1`);
  - the JIT cache save.
- Engine changes that can affect speed or correctness (#44401, graph sizes, NCCL env, DFlash width, int8 Mamba checkpoints) are **arms**: measured in the run, adopted by a written rule. They are not refused in advance.

## 0. What E14-A must decide

| # | Decision | Measured by | Adopt if |
|---|---|---|---|
| D1 | Where the ~115–140 ms decode step goes (all-reduce vs GEMM vs attention vs GDN vs sampling) | profiler window by kernel category, at 12 running, 8k and 40k context | (measurement; ranks P2/P3 levers) |
| D2 | NCCL transport and the cost of one 983,040 B all-reduce | `nccl-bench` with `NCCL_DEBUG=INFO`: 140× graph replay, 3 env arms | an arm ≥ 20 % faster at 983 KB with the right sum, then ≥ 5 % tok/s in the server window |
| D3 | #44401 backport (11-of-12 admission) | latch test: 11 long decodes + new arrivals during chunked prefill; real running from /metrics | 12 real running during chunked prefill, no negative free slots, tok/s ≥ baseline − 2 % |
| D4 | `--cuda-graph-bs-decode 1..13` | the same windows at 12 and 13 running | capture OK, no OOM, tok/s at 13 ≥ at 12 |
| D5 | E14-P at token level (raw vs normalized reasoning) | replay client with `return_output_ids_in_sglext` / `return_input_ids_in_sglext` | raw follow-up cached ≥ prompt₁ + output₁ − 64 and normalized ≈ prompt₁ end; first divergent id reported |
| D6 | DFlash2 correctness (concurrency contamination #36548, thinking divergence #38009) | Notes/10 §C gate: greedy parity D vs T with first divergence + margin, A/A, c12-vs-serial | c12-vs-serial divergence ≤ A/A rate + preset allowance; D vs T parity at the A/A level |
| D7 | DFlash2 width 8 / 6 / 4 | windows at equal context; accept length | ≥ +5 % tok/s **and** D6 passes for that width |
| D8 | int8 Mamba checkpoints (`--enable-int8-mamba-checkpoint`, ~2× checkpoint capacity) against the deeper misses (E14-M) | agent-like multi-session replay: share of zero/deeper hits; parity of resumed requests | the deeper-miss share drops, tok/s not lower, resumed output parity at the A/A level (int8 states are lossy) |
| D9 | Load path on real Kaggle storage | `env_report`, `read_probe`, copy MB/s per worker level, ready time | copy verified and ready ≤ 15 min; else the fallback path is logged |

## 1. The new E14 pipeline (built on CPU first)

**New builder:** `experiments/Experiment 14 …/Experiment Notebooks/Build/build_e14.py`.
- It starts from a copy of the rev 6g builder logic, so G4B cells 2–7 and the solver come in unchanged.
- It writes **only** to the E14 `Experiment Notebooks/` folder, and it refuses to overwrite a notebook that was pushed (sha registry, like the E12 `_REBUILD` guard).

### Cells

**C0 – first line**
- `env_report` (mounts / GCSFUSE, df, RAM, cgroup, CPU, nproc);
- `read_probe`;
- `E14State` → `/kaggle/working/e14_state.json`;
- a heartbeat thread (one line per minute to `e14_heartbeat.log`).

**C1 – mode, config**
- Commit vs rerun (dual-mode rule).
- `E14_PLAN` = the list of phases with per-phase deadlines.
- `HARD_STOP`.

**C2 – staging + install, in parallel**
- `Staging([model, draft])` starts first: ramp 4 → 8 → 16 (cap 24); verify; `.partial` + atomic rename; fail open.
- The venv install from the wheelhouse runs alongside, as now (208 s).
- `jit_cache_env()` is added to the venv env.

**C3 – NCCL bench**
- GPUs are free while the copy runs. A `torch.distributed` 4-rank script (venv python) measures:
  - all-reduce at 983,040 B, eager and **140× CUDA-graph replay**;
  - all-gather of 11.9 MB per rank;
  - 64 MiB busbw.
- Three env arms: baseline / `NCCL_P2P_LEVEL=SYS` (only if `can_device_access_peer` is true) / `NCCL_SHM_USE_CUDA_MEMCPY=1`.
- Transport is parsed from the `NCCL_DEBUG=INFO` lines; `nvidia-smi topo -m` is logged.

**C4 – engine**
- **Launch.** `e11_flags(W1)` with the staged paths. `wait_ready` = `StallMonitor` (load 900 s, compile 720 s stall limits, give-up = `SGL_GIVE_UP_AT`).
- **Relaunch helper** `e14_relaunch(cfg, env, tag)`:
  1. kill;
  2. wait until GPU memory is free;
  3. launch;
  4. `StallMonitor`;
  5. smoke test;
  6. greedy known-answer check;
  7. a record in `E14State`.
- **#44401 / #41872 backport**:
  - applied to the venv's installed `sglang` source **before** an arm's launch;
  - a source-hash guard per patched function;
  - on a mismatch: fail loudly and skip the arm, never a silent monkey-patch.

**C5 – measurement module `e14_measure.py`** (stdlib + requests; embedded)
- **Workload generator** with real Duck-shaped prompts:
  - the arc2 r6 system prompt + facts of eval tasks;
  - 12 streams;
  - context bands 8k and 40k (a shared 40k prefix + a 1k unique tail per stream, so prefill stays ~30 s);
  - fixed `max_tokens`, greedy and T 1.0 variants.
- **Real running is asserted from `/metrics`** (`num_running_reqs` == level) before any window counts (memory rule "verify real concurrency"). Server caps are read from the launch log.
- **Window metrics** from the server's decode lines (every 40 forwards: running, gen throughput, accept len) and the `/metrics` gauges:
  - step time = window wall / forwards;
  - tok/s;
  - accept length;
  - KV and Mamba usage.
- **Profiler.** `POST /start_profile {num_steps: 20, activities: [GPU, CPU], output_dir}` during the 8k window. The trace is summarised in the notebook into ms per step by category: all-reduce, all-gather, Marlin GEMM, attention, GDN/triton, sampling/top-k, drafter, other. Only the summary + top-30 kernels go to the output; the raw trace only if < 200 MB.
- **Replay client (D5).** Per conversation:
  1. request 1 (a Duck-shaped prompt, thinking on, cap 3,000), with `return_input_ids_in_sglext` + `return_output_ids_in_sglext`;
  2. follow-up A with the **normalized** reasoning;
  3. follow-up B with the **raw** reasoning.

  Record `cached_tokens`, the input ids of A and B, and the first id where they differ from `input_ids₁ + output_ids₁`. Three conversations. All generated text is saved (thinking-trace rule).
- **Latch test (D3).**
  1. 11 decodes of 6k tokens;
  2. then 3 arrivals of 20k prompts (chunked prefill 4096);
  3. sample `/metrics` every 0.5 s;
  4. report the share of time at 12 running while a prefill chunk is in flight.
- **Parity probes (D6).**
  - 12 fixed prompts, greedy, 768 tokens, serial;
  - the same 12 at concurrency 12;
  - A/A = the same serial set twice;
  - first-divergence position + top-2 logprob margin there.
- **Agent-like replay (D8).**
  - 24 sessions × 6 turns at 12 admitted. Each turn = the previous prompt + a real generated reply (raw reasoning) + a ~600-token tool result.
  - Measures zero-hit / prefill-end / full shares as in Notes/11.

**C6 – report**
- `e14_report.json` + `e14_report.md`: every phase, every arm, decision rules applied, raw numbers.
- `E14State` terminal reason.
- In rerun mode no measurement runs; the solver path runs (see §4).

### Local checks before any GPU run
- The CPU harness (fake server) runs the whole notebook end to end: every phase, deadlines, the report, the rerun path.
- New unit tests for: the trace summarizer (a synthetic trace), the decode-line parser (real G4B lines), the patch applier (against the local 0.5.21 source tree: hash match → patch applies; a changed file → refusal), and the replay client logic (fake ids).
- Re-run the load-path tests (56) and the raw-reasoning tests (15).
- sha check: no E12 notebook changed.

## 2. Phase plan and time (wall clock)

**Basis for the times (measured):**
- install 208 s;
- weights 669 s from slow storage (≈ 13–60 s from /dev/shm, estimated);
- graph capture + startup ≈ 6–8 min, which **every relaunch pays**;
- prefill ~1,500 tok/s;
- decode step 115–140 ms.

### Run A1 (core, the bigger decisions first)

| Phase | What | Wall (normal) | Worst case |
|---|---|---|---|
| A1-0 | C0 + C2 staging/install ‖ C3 NCCL bench (GPUs free during the copy) | 9 min | 38 min (copy fails → in-place load) |
| A1-1 | Launch W1 baseline (unchanged flags), ready | 8 min | 12 min |
| A1-2 | Baseline windows 8k + 40k at 12 running + profiler (D1) | 8 min | 10 min |
| A1-3 | Replay client E14-P (D5) | 3 min | 5 min |
| A1-4 | D-side probes on the baseline: parity serial, A/A, c12 (D6 part 1) | 5 min | 6 min |
| A1-5 | Relaunch + #44401 backport: latch test + windows (D3) | 13 min | 16 min |
| A1-6 | Relaunch + #44401 + graph bs 1..13 (max running 13): windows at 12 and 13 (D4) | 13 min | 16 min |
| A1-7 | Report, JIT cache save, outputs | 2 min | 3 min |
| **Total** | | **≈ 61 min** | **≈ 106 min → capped by HARD_STOP 95 min** |

**Account quota (4×L4 is charged at 2×):** ≈ 2.0 h normal; ≤ 3.2 h at the HARD_STOP.

### Run A2 (after the A1 results; arms chosen by A1)

| Phase | What | Wall | Worst |
|---|---|---|---|
| A2-0/1 | Boot with the A1 winners (ready) | 15 min | 40 min |
| A2-2 | Agent-like replay at default Mamba settings (D8 baseline) | 8 min | 10 min |
| A2-3 | Relaunch + `--enable-int8-mamba-checkpoint`: replay + resumed parity (D8) | 17 min | 20 min |
| A2-4 | Relaunch target-only (no spec): T parity set (D6 part 2) | 12 min | 14 min |
| A2-5 | Relaunch width 6: windows + parity subset (D7) | 13 min | 15 min |
| A2-6 | Relaunch width 4: windows + parity subset (D7) | 13 min | 15 min |
| A2-7 | (only if D2 found a ≥ 20 % NCCL arm) server window with that env | 13 min | 15 min |
| A2-8 | Report | 2 min | 3 min |
| **Total** | | **≈ 80 min (93 with A2-7)** | **capped by HARD_STOP 110 min** |

**Account quota:** ≈ 2.7–3.1 h normal; ≤ 3.7 h at the HARD_STOP.

### Stop rules (written into the notebook, not left to judgment)

- **HARD_STOP:** at `T0 + 95` (A1) / `T0 + 110` (A2) min the notebook stops every phase, writes the report and ends.
- **Per-phase deadline:** a phase that overruns its worst-case time by 50 % is ended, recorded as `timeout`, and the next phase starts.
- **No ready server:** if no server is ready by `T0 + 45` min, only C0/C3 results and the logs are written, and the run ends (≈ 1.5 h quota max).
- **Failed arm:** an arm that fails to launch twice is skipped. The previous arm's config is relaunched only if a later phase needs it.
- **Correctness:** any parity or known-answer failure marks the arm `REJECTED` regardless of its speed.

## 3. Quota, accounts and order

**Quota now (10-22 readings; re-read just before launch):**
- hosen42 8.63 h;
- foysal 4.71 h;
- esfersami50 0 (weekly reset; check).

**Order:**
- **A1 on hosen42** (≤ 3.2 h) → about 5.4 h left.
- **A2 on hosen42 or foysal**, after A1 (≤ 3.7 h).
- **E14-C** (below) on whichever account still has ≥ 3 h, or esfersami50 after its reset.

**Before each launch, the user gets in writing:**
- the account;
- its quota at that moment;
- normal and worst wall time;
- the quota cost at 2×;
- the HARD_STOP time;
- the notebook sha.

**No push or run without the user's explicit go.**

## 4. After E14-A: from measurement to score

**E14-C (the scoring pipeline):**
- the rev 6g Duck solver + every adopted lever: load path, raw reasoning, A1/A2 winners;
- built by `build_e14.py` as a dual-mode notebook:
  - **commit** = an eval subset (EVAL12 or EVAL24) + score;
  - **rerun** = all hidden tasks with full-schema `submission.json`.

**Commit run:** ≈ 80 min (G4B scale) → ≈ 2.7 h quota. It measures:
- effective tok/s (the number that matters);
- cached share (replies reused);
- solved;
- real running.

**Submission:** QUICK_SAVE push + submit (no GPU quota; 12–16 h scoring; one pending at a time).

**Proposed calendar (deadline 2026-11-02; 10 days left):**

| Date | Step |
|---|---|
| 10-23 / 10-24 | Build `build_e14.py` + E14-A1 + local harness on CPU; written estimate → user go → A1 on hosen42 |
| 10-24 / 10-25 | Read A1 (report in full) → build A2 with the chosen arms → A2 |
| 10-25 / 10-26 | E14-C build with the winners → commit run (EVAL subset) |
| 10-26 | Submission 1 (E14-C) |
| 10-27 – 11-01 | Submissions 2–4: bundled next levers (bundle-changes rule) |
| 11-02 | Deadline |

**Option to save a day (the user's call):** fold A2-2/A2-3 (Mamba) and the A2 width arms into the E14-C commit run as extra phases. That costs ≈ +45 min wall in that run instead of a separate A2.

## 5. Risks and what each costs

- **Slow storage beyond the fallback:** the 10-22 empty-log ERROR happened before our code ran. Nothing in the notebook can fix that. The cost is bounded by the platform; our part is bounded by HARD_STOP.
- **The #44401 diff may not apply to 0.5.21 as is:** the arm is skipped loudly (A1 still delivers D1, D2, D4, D5, D6 part 1).
- **Profiler overhead or trace size:** 20 steps only. If `/start_profile` fails, D1 falls back to the step-time model + NCCL bench.
- **int8 Mamba checkpoints can change outputs** (lossy states): gated by resumed-request parity, never adopted on speed alone.
- **Width 4/6 changes the drafter's block semantics** (non-causal inside the block): gated by D6 for that width.
- **Relaunch failures** (GPU memory not freed): wait for free memory (`nvidia-smi`); 2 tries; then skip.

## 6. What must be built before the first GPU minute (CPU, now)

1. `build_e14.py` (base copy; E14 cells C0–C6; sha registry; no E12 writes).
2. `e14_measure.py`: workload, `/metrics` asserts, decode-line parser, profiler trigger + trace summarizer, replay client, latch test, parity probes, agent-like replay, report.
3. NCCL bench script (`torch.distributed`, CUDA graph replay of 140 all-reduces).
4. #44401 (+ #41872) backport patch: fetch the merged diff, port it to the 0.5.21 sources, write the hash guard and apply/rollback, test against the local source tree.
5. Embedding of `e14_loadpath.py` + the raw-reasoning switch on in E14.
6. CPU harness run of E14-A1 end to end (fake server and fake NCCL) + all unit tests.
7. The written run estimate for the user (§3).
