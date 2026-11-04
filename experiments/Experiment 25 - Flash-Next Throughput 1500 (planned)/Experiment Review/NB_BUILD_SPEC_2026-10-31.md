# E25 notebook build spec (2026-10-31): NB1 Capacity, NB2 Decode, NB3 Kernel bench, NB5 FN-Duck

Read this file first, then `PLAN_1500.md` (v3.1, same folder's parent) in full. User go for BUILDING (2026-10-31): NB1, NB2, NB3
builders + the NB5 Duck port, in parallel. **No Kaggle push, no Kaggle run, no submission** — that needs a separate
explicit go from the user. NB4 (engine patch) and NB6 (production) are NOT built now.

## 1. RC2 result = the base (S0 PASSED, 2026-10-31, foysalemonshanto/q27-xhigh-e13-flashnext-rc2 v1, 83 min)

Local output: `D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/` (`rc2/rc2_results.json`, `rc2/logs/server_merlin_{bf16,fp8}.log`,
`rc2/telemetry.jsonl`, `rc2/traces/`). Builder: `experiments/Experiment 13 - .../Experiment Notebooks/Build/build_e13_rc2.py`
(1081 lines; reuses E23 cells via `cut()/once()`; `PUSHED` sha guard -> `_REBUILD`; unit tests `rc2_unit_tests.py`; CPU kit
`build_e13_rc2_cpu_kit.py`). Merlin source: `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/` (launch: `tools/merlin/launch.py`,
profile `models/flash-next/...`, `flash-next.json`); the bundled copies in `_scratch/rc2_final_bundle/merlin/`.

| Fact | Value |
|---|---|
| Correctness | BF16 CLEAN, FP8 CLEAN (fp8_gate PASS, oracle vs BF16 NEAR_TIE first diff 19-77 tok), probes/sampled OK, template MATCH |
| Long | FP8 natural xhigh 116,217 tok, finish stop, think closed, PASS_LONG; forced 60k PASS_LONG; ~57-59 tok/s per stream at concurrency 2 |
| **Open flag** | Teacher-forced 40k FP8 vs BF16: mean abs dlogprob **1.88**, p99 5.46, rank-1 agreement **0.367** (A/A = 0.0). Large; not explained yet. Must be isolated before FP8 is trusted for production (see NB1 boot 0) |
| KV pool | BF16 269,042 tok (78 blocks, conc 2.05 at 131k); FP8 513,940 tok (149 blocks, conc 3.92) |
| Boot | BF16 ready 411 s; FP8 ready 999 s — one PP1 worker "Model loading took 873 s" vs 125 s on the others: host page-cache pressure (cgroup cur 176.9 / max 177.1 GiB; file 112 GiB, anon 63 GiB, shmem 4.2 GiB while serving). Merlin warned "available 68 GiB < 69 GiB floor" (its rule: max − (current − inactive_file)) |
| Steady FP8 (90 s windows, T 0.6, MTP K3) | 40k@1 70.0 / @2 107.4 / @4 161.0 / @5 190.5; 8k@8 **261.8**; 90k@4 159.8; 116k@3 140.5 / @4 160.3 tok/s |
| Acceptance | accepted per draft 2.0-2.25 (mean acceptance length 3.0-3.25); step_ms 30-48; 0 preemptions |
| KV usage | 8k@8 0.973, 90k@4 0.978, 116k@4 0.957, 40k@5 0.652 of 149 blocks → fixed per-request blocks dominate (F); measure F exactly |
| Server args | TP2×PP2+EP, `--max-num-seqs 8` (FP8) / 3 (BF16), `--max-num-batched-tokens 1024`, capture [1,2,3,4,6,8,12,16,24,32], block 4096, align, `--use-replayssm`, `--prefix-match-unit 64`, `--kv-cache-memory-bytes 2479838700` (FP8) / 2464415744 (BF16), custom AR off, P2P off, `VLLM_PP_LAYER_PARTITION=25,23`, MTP K3 probabilistic |
| Prefill | 40k TTFT 9.1 s (4,409 tok/s single); 116k@3 admission prefill 13,316 tok/s aggregate |

Decision rubric (PLAN §4): FP8 steady at 5 streams = 190 (< 250), at 8 streams = 262 (250-450) → Tier C capacity first,
then D5/D4; profile one boot (trace classifier fix, A7).

## 2. Rules every builder follows (user rules; non-negotiable)

- Flash-Next only. Thinking ≥ 60k, up to 120k; `--max-model-len 131072`. Never import ARC-AGI-3 caps (12,288 / 28k pause).
- Few arms per notebook (2-3 boots + at most one reference boot). Each boot records its exact config **diff vs RC2 FP8**.
- Fail closed: boot readiness gate (≤ 30 min per boot), per-boot watchdog, global budget, hard stop; a failed boot never
  starts the next dependent arm; GPU release check (< 150 MiB/card) before every boot; SHA verification of the merge.
- "real running == level" must be asserted in every steady window (memory rule); read server caps (`max-num-seqs`) from the log.
- Save the FULL text/trace of every long generation (memory rule) and the full server logs.
- Host RAM: before every boot drop the page cache of the checkpoint files (`os.posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)` on
  every file under the merged dir and the Model mounts' resolved targets) and gate on Merlin's own rule
  (`max − (current − inactive_file)` ≥ 69 GiB + 4 GiB margin); record before/after. (RC2: 873 s load + 68 GiB warning.)
- Reuse, don't rewrite: take RC2's cells/functions via `cut()/once()` from `build_e13_rc2.py` (as RC2 did from E23) so
  Kaggle-proven code stays verbatim; put new code in clearly marked new functions. Same input paths as RC2 (exact Model
  paths in `rc2_results.json` → `inputs`).
- `PUSHED` sha guard + `_REBUILD` redirect in every builder (memory: never overwrite a run notebook).
- Builder writes `../<NOTEBOOK>.ipynb` into `experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/`;
  builder + tests in `.../Experiment Notebooks/Build/`. Notebook < 400 kB (Kaggle commit limit 407-444 kB).
- Dual-mode not needed for NB1-NB3 (speed notebooks; they write the fallback submission like RC2). NB5 is a real eval
  notebook and must be dual-mode (rerun = all hidden tasks; commit = eval subset + score; full-schema submission.json).
- Python 3.12 on Kaggle (pinned image). Local Windows: CRLF appears on Edit/Write; normalize to LF in generated notebooks.
- Local tests: a fault-injection unit test file per builder (pattern: `rc2_unit_tests.py`) + the fake-server mode
  (`E23_FAKE_SERVER`) end-to-end. Everything must pass locally. The Kaggle CPU kit is built later by the lead as ONE kit.
- Security: never open `C:\Users\hp\hamid vai`, `~/.kaggle/*.json`, access_token files, never print Kaggle keys. No Kaggle API
  calls at all in this task. No git commits (the lead commits).

## 3. Notebooks

### NB1 FN-S1 Capacity (`Q27-XHIGH-E25_NB1_CAPACITY.ipynb`, builder `build_e25_nb1.py`)
Goal: more resident streams with no correctness change. Boots are cumulative; each records its diff and is measured alone.
- **Boot 0 (reference, BF16 KV, short)**: BF16 teacher-forced references at 2k / 8k / 40k / 90k (last 256 tokens, same
  prompts as RC2 `TF_P` plus the new bands). Saved to file for the FP8 boots.
- **Boot 1 (FP8 = RC2 config + text-only C4 + #60490 D9)**: #60490 is a memory-accounting/correctness backport (QSA
  workspace profiling; may REDUCE the reported pool ~0.4-0.5 GiB) — apply it as a source patch to the installed Merlin tree
  (fetch the upstream PR diff; vLLM PR 60490, issue 60489), verify it applied (grep/marker), and measure free VRAM per card,
  KV pool, F per request BEFORE any pin change. FP8 quality isolation: TF vs boot-0 refs at 2k/8k/40k/90k (does the 1.88 nat
  divergence grow with context?), plus the same 40k TF with MTP-independent prompt-logprob path noted. Census: F from
  `kv_cache_usage` at known lengths (1, 2, 4 requests at 8k/40k) and the block rule; prefix continuation test (2-turn replay:
  turn 2 = turn-1 prompt + reply + new user message; measure cached tokens via `--enable-prompt-tokens-details`) and the
  same after an aborted request; PP2 stage-overlap + per-stage idle + target/draft/RecoverSSM-commit graph mode + verify-rows
  per microbatch from a short torch-profiler window (A7: add `execute_context_*` scopes, `gdn_*`/`qsa_*`/`marlin_moe`/
  `persistent_topk` patterns); steady ladder 8k@8, 40k@5 and the highest level the pool admits.
- **Boot 2 (+ SSM state bf16 C3 + `MERLIN_SHARE_DRAFT_WORKSPACE`)**: same census + steady; correctness oracle vs boot 1.
- **Boot 3 (+ KV pin step, guarded)**: KV_BYTES + 0.5 GiB ONLY if boots 1-2 measured ≥ 0.9 GiB free per card after
  #60490; otherwise skip and record why. CHUNK (W6, `--max-num-batched-tokens` 2048) is NOT mixed in: measure it in a
  separate window only if a runtime override exists; otherwise leave for a later boot.
- `max-num-seqs` above 8 (needed for 12/16 streams): check the profile allows it and the capture list covers seqs×(1+K).
- Budget ~2 h + boot time; hard stop.

### NB2 FN-S2 Decode (`Q27-XHIGH-E25_NB2_DECODE.ipynb`, builder `build_e25_nb2.py`)
Control = RC2 FP8 K3 / BF16 draft head (numbers in §1; no re-run of the control unless budget allows one 8k@8 window).
- Boot A: K3 + DRAFT_HEAD int8. Boot B: K2 + int8. Boot C: K1 + int8. (K0 = target-only only if budget remains; a
  proposal, flagged.) Knobs: `MTP`/`SPEC_K`/`DRAFT_HEAD` in Merlin `launch.py` + `flash-next.json`.
- Per boot: real xhigh ARC prompts (RC2 prompt builder); streams 5 and 8 at 40k; 12 and 16 ONLY at 8k-16k and only after a
  check that `max-num-seqs` > 8 is accepted by the profile (else record). Per-position acceptance, accepted/draft, step_ms,
  tokens/step, F per K (hybrid-state blocks change with K), graph coverage (FULL/PIECEWISE per capture size), sampled
  correctness + oracle vs RC2 FP8 refs, acceptance per 10-min window (decay check) in one long-ish window per boot.
- Same PP2/verify-row trace window as NB1 (one per boot, short).

### NB3 FN-S3a Kernel bench (`Q27-XHIGH-E25_NB3_KERNELS.ipynb`, builder `build_e25_nb3.py`)
GPU notebook, no model server. Install Merlin (RC2's merlin_install) to get its kernels.
- D5 Marlin MoE: real Flash-Next expert weights (one MoE layer from the merged checkpoint: INC W4A16, 512 experts top-10),
  token counts T = seqs×(1+K) for seqs ∈ {1,2,4,5,8,12,16,24,32}, K ∈ {1,2,3}; distinct experts D(T)=512(1−(1−10/512)^T);
  EP split as in the profile (TP2 EP); correctness vs a dequantized reference on a slice; CUDA-event timing, warmup,
  synchronized; also the Merlin tuning knobs if present (configs). 
- D7 world-2 all-reduce: NCCL (SHM, P2P off as in RC2) vs P5 host-staged (E23 prep `prep/P5_host_allreduce`) vs
  `NCCL_PROTO=LL128` at Flash-Next message sizes (hidden × T × dtype for the T above), both GPU pairs (0,1) and (2,3)
  concurrently (as PP2 runs them), numeric check of the reduced tensor.
- D4 sm_89 GEMM plans: does Merlin's sm_89 GEMM plan path compile/run for Flash-Next shapes (E23 prep `P4_sm89_fp8_gemm`
  for the method); report missing plans (A7) and timing vs the fallback.
- Output: one JSON of tables + a ranked "patch candidates for NB4" list with measured µs and projected tick share.

### NB5 FN-Duck (port only now; `Q27-XHIGH-E25_NB5_FNDUCK.ipynb`, builder `build_e25_nb5.py`)
Port the Duck harness to the Merlin Flash-Next server (RC2 FP8 boot path). Sources: E13's own Flash-Next Duck builder
`experiments/Experiment 13 - .../Build/build_e13_duck.py` (rev 6, SGLang era), Duck config B (E15B R3K2, sha c058d58b,
`experiments/Experiment 15 .../Build/build_e15b.py`), Duck-on-vLLM gotchas from E21 (`build_e21.py`; memory: reasoning key,
finish_reason, gate singleton, block capacity, align cache). Requirements:
- Thinking ≥ 60k, up to 120k; long uninterrupted thinking (no Duck 12,288 cap, no 28k pause, no chunks).
- Commit-turn variant C (measured on CPU: `Experiment 25/Experiment Results/commit_turn_prefix_test_2026-10-31.json`):
  keep the thinking-on render and replace the trailing `<think>\n` by `<think>\n\n</think>\n\n`, sent as token ids → 98.3 %
  prefix reuse, header identical. Never `enable_thinking=False` (0 % reuse). Count commit turns without a tool call.
- Stream count / KV gate from the measured FP8 pool (149 blocks, block rule ceil(L/4096)+F, F to be measured by NB1);
  controller replay test on REAL recorded RC2 metrics (memory rule), never invented fixtures.
- Modes: SMOKE (short) → metric reset (or timestamp-delta) + workload reset → EVAL; paired EVAL30 MTP on/off option;
  counters: prefix hit tokens, recompute, per-turn prefill, tool-call rate, solve-vs-tokens bands; measure pure decode,
  steady mixed, solver-window average and whole-notebook average separately. 11:50 hard stop; dual-mode; full traces.
- Not run now. Deliverables: builder, unit tests, local fake-server end-to-end pass.

## 3a. Scorecard (added 2026-10-31, PLAN §4.1)
NB1, NB2 and NB3 each write `lever_scorecard.json` (incrementally, after each boot/bench): lever, measured,
projected_run_tok_s_gain (with inputs), correctness, confidence, adoption (config | patch | research), cost_gpu_h, notes.
After the screening round only the top 2-3 levers get more GPU; the rest are parked.

## 4. What each agent returns
Paths of every file written; a FULL list of design decisions and every deviation from this spec with the reason; local
test commands + their full pass/fail output; open questions; notebook sha256 and size.
