# 19 — E15P rev 2: every finding of agents A and B and GPT audits 1-2, verified, and what was done

**Date:** 2026-10-25.

**User request:** fix everything, then the final notebook.

**Notebook:** `Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb` (renamed from S16, because rev 2 runs 15 streams). Sha: see the build output / the final message. It has 11 code cells and is 219 kB.

**Builder:** `Experiment Notebooks/Build/build_e15p.py`. Every change is marked "E15P rev 2" in the cells.

**Verification:** every claim was checked by Claude against:
- the notebook code;
- the SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src`);
- the past run logs.

## Fixes

| Id | Finding | Sources | Verified | Fix in rev 2 |
|---|---|---|---|---|
| F1 | A failed tokenizer check sets `ENGINE_BACKEND="vllm"`, but there was no vLLM cell. SGLang is then skipped and the score is ~0. A failed SGLang start also scored ~0. | A HIGH-1, GPT2 E15P-1, B M2 | yes: c3 L106, c6 L728 | S60's vLLM FP8 fallback cell, unchanged, after the engine cell. GPT2 proposed raising instead; rejected, because a raise in the rerun means no score at all. |
| F15 | A 16-stream server that is READY and passes the capacity gate, but fails the smoke test, gets no 12-stream retry. | GPT2 | yes: `e11_bring_up` ends with `raise E11Fail("smoke failed")` | One `e15p_relaunch_s12("smoke")`, then the gate and the smoke again; if that fails too, F1 takes over. |
| F2 | A server death mid-run is not recovered. | A MED-3, GPT2 | yes | `e15p_run_solver_with_restart()`: the running units go back to the front of the queue (iteration not advanced, thinking saved as `.restartN.partial.txt`), then a 12-stream relaunch with the same venv / env / staged model. At most 2 restarts, only with ≥ 45 min before the admission deadline. One verification pool serves the whole run, so queued verifications survive the restart. |
| F2b | (found by Claude while building F2) When the server dies, the streams end with `error`; phase 2 is then sent to a dead server, and the per-output handler closes the unit as `unit_error`. Those tasks would be lost even with a restart. | Claude | yes | A stream `error` while the server is gone (polled up to 5 s), or a "SGLang server" RuntimeError, now goes to the restart path with the unit still active. |
| F3 | At the drain, deltas queued after the last `step()` were left out of the phase-2 prefix. An unconfirmed abort still started phase 2. | GPT1 HIGH-2 (B L1: 5-25 tokens, harmless) | yes: `step()` skips phase-1 output once `req.phase` is 2 | Shim `take_pending(rids)` returns the aborted rids' queued deltas in order and keeps every other item, in order, for the next `step()`. The drain appends those deltas. A phase 1 that had closed its thinking (`stop` + `</think>`) finishes as natural. An abort that is not confirmed starts no phase 2. |
| F4 | The gate read only `full_token_usage` and failed open when `/metrics` failed. | GPT1 HIGH-1, GPT2 E15P-2, B L2 | yes: `token_usage` = max(full, swa, mamba), `pool_stats_observer.py` L69-73, 128; mamba counts active slots only (L272-276) | The shim's admission metric = `token_usage`, else max(full, mamba). A missing metric is None, and the gate holds new admissions. The floor of 10 stays (S1 design: 9 × 76k < the pool). GPT1's "remove the floor" was rejected: below 10 running the pool cannot fill, and without the floor a missing metric could stall the solver. |
| F5 | The sandbox self-test runs once (8-s probes) while the 24-thread model copy is reading; a failure means no engine. | A MED-2 | yes: staging joins inside `e14_after_install`, after the donor cell | Up to 3 attempts, 10 s apart. The attack probes stay strict. |
| **F6** | **16 Poetiq streams peak at KV 0.97 of the 852,152-token pool.** The E11-06 decode-line peak was 0.73 of 866,744 = 51.9k per request; Poetiq units reach the 60k cap as a cohort. The client gate cannot stop running units from growing. | A MED-1, GPT1 §5, GPT2 §3, **B H1 (simulation)** | numbers checked against the logs | **`SGL_KV_PER_REQUEST` = 56,000, giving 15 streams.** Agent B simulated SGLang retraction: at 16 streams, 2 of 8 six-hour and 2 of 4 twelve-hour runs; at 15, 0 of 28. Cost about −1 % tokens. The retraction path has never run on this stack (no `Retract` line in any log). The server keeps mrr 17 / graphs 1..17. |
| F7 | Leaving `with ThreadPoolExecutor` waits for running verifications. | GPT1 HIGH-3 | yes (a local test waited 3.0 s), but bounded: ≤ 4 jobs × ~10 grids × (5 s + 2 s) ≈ 70 s, against a 5-min margin. LOW. | `_E15PPool`: a normal exit shuts the pool without waiting (queued jobs cancelled); an error exit leaves it open for a restart; the final block closes it. |
| F8 | No payload means CELL 1b raises, and the notebook throws. | A MED-4 | yes | Push checklist: `esfersami50/q27-e15b-payload` attached, with `q27x_e14_modules_55c1eb7911c6.bin` at its root (markdown inputs list). |
| F9 | The docs claimed the E14 backports 44401 / 41872 were applied. | A LOW-1, B M1 | yes: `E14P.apply` exists only in R3K1's measurement cell | Builder docstring, markdown and this note corrected. **Notes/18 and the rev 1 audit README were wrong.** E15P (like every E15B production run) runs unpatched 0.5.21. |
| F10 | Stale Duck / E14 items. | A LOW-2..4 | yes | `E14_DO_MEASURE = False`; `E12_RAW_REASONING_HISTORY` and the Duck sha removed from CELL 1b; `EVAL120` row = 6 h (S60); bundle renamed `Q27X_E15P_ENGINE_BUNDLE.zip`. |
| F11 | The builder's `PUSHED` set is empty. | A LOW-5 | yes | Add the sha right after the push (process). |
| F12 | The report lacks coverage and memory evidence. | A INFO-1, GPT1 §7, GPT2 §5 | — | `report['e15p']`: iteration-1 tasks started / completed / train-verified, not started, mode / finish histograms, phase-1 / phase-2 tokens, sampled peaks (token / full / mamba usage, running, queue), decode-log maxima across all server logs (full / mamba usage, running, queue, retract lines, gen tok/s, accept length), kv gate blocks, missing metric count, drain tail tokens, unconfirmed aborts, restarts, relaunches, final concurrency, backend. |
| F13 | DFlash width 6 or 8. | GPT1, GPT2 | **B §3:** W6/W8 step ratio 0.83 from 10k to 40k context; a geometric acceptance model gives W6 +5-12 % for Poetiq | Keep width 6. Check `by_context` after the run. |
| — | Seed not forwarded. | GPT2 E15P-3 | true, but SGLang builds the seed tensor only `if enable_deterministic` (`sampling_batch_info.py` L143-157) | No change (no effect without `--enable-deterministic-inference`). |
| — | SGLang issue #43204 (mamba exhaustion crash). | GPT1 | not verifiable offline | Covered by F2 if it happens. |
| — | Drain margin 12 min could be 5-8 min (+1.5 % rerun tokens). | B L3 | aborted prefixes are cached: 14 of 16 hit ≥ 99 % | Not changed (S60 parity, the user's choice). |
| — | Coverage projection. | B §4 | — | **6-h commit:** iteration 1 on 119-120 / 120 tasks (central; pessimistic 108-112). **12-h rerun:** 240 / 240 plus some iteration 2 (pessimistic 229-238 at 15 streams). Break-even about 350-359 tok/s. |

## Harness

**Files:** `src/harness/harness_e15p_nb.py` and the fake server (`make_e15p_fake_server.py`), which now has:
- `token_usage` = max(full, mamba);
- a mamba override;
- smoke fail and server death on the 17-mrr server.

**Modes (logs in `_scratch/e15p/rev2/`):**

| Mode | What it tests |
|---|---|
| commit | 120 eval tasks, 15 streams |
| kvgate | KV 0.97: the gate holds at 10 |
| mambagate | mamba 0.97 with KV low: the gate holds at 10 |
| drain | the drain path |
| rerun | the rerun path |
| tokfail | S60's vLLM fallback on the fake vllm |
| smokefail | 16-stream smoke fail → 12-stream relaunch, run at 12 |
| death | the server crashes after 30 requests → requeue, relaunch at 12, run completes |

**Unit tests:**
- `take_pending` ordering;
- the admission metric (token_usage / max(full, mamba) / none);
- the gate truth table: (16, None) closed, (16, 0.97) closed, (16, 0.50) open, (9, None) open, (9, 0.97) open, (10, 0.97) closed.

**A local artefact, not a notebook fault:** with 8 harness processes in parallel, the Windows fake server sometimes crashed at start (`rc=3221226505`). The notebook then went 16 → 12 streams, as designed. Those modes were rerun two at a time.

## Still to do

- The Kaggle CPU kit on the final sha (Python 3.12 pinned image).
- The GPU run, on the user's go.
- After a push: add the sha to `PUSHED`.
