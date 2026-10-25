# E14A SLIM run1: findings and the Thread._stop crash (2026-10-23)

## The run

| Item | Value |
|---|---|
| Kernel | `foysalemonshanto/q27-xhigh-e14a-slim-optimized-engine-sglang` v1 (commit) |
| Notebook | sha 3c67794e, 224 kB |
| Start | RUN_ID 20261023T150946Z |
| End | the user stopped it at about 25 min (15:35 UTC), in A1-5 |
| Quota | 26.92 → 27.75 (about 0.8 h) |
| Outputs | `Experiment Results/Kaggle Runs/E14A SLIM run1 (foysal, 2026-10-23, stopped at A1-5)/`: kernel_log, e14_results.json, e14_state.json, heartbeat, submission |

## 1. The size fix works

The slim notebook committed and started normally, and the payload dataset loaded:
- `E14 modules`;
- `E12 package OK: 23 files`;
- sandbox memory limit applied 3, over-limit rejected.

## 2. The crash: `Sampler._stop` shadows `Thread._stop()` on Python 3.12

- **Symptom:** `E14 phase A1-2_baseline_W1: TypeError: 'Event' object is not callable` at `e14_measure.py:126` (`stop` → `self.join(10)` → `threading._wait_for_tstate_lock` → `self._stop()`).
- **Cause:** `Sampler(threading.Thread)` stored its Event as `self._stop`. On Python ≤ 3.12, `join()` calls the Thread's own `_stop()` method.
- **Why it was missed:** the local harness ran on Python 3.14, whose `join()` no longer calls it.
- **Effect:** every throughput window, the latch test and every arm window would fail. No baseline, so no adopt decision (nodelay included) was possible.
- **Fix:** `self._halt`. Two regression tests were added:
  - an AST check that no Thread subclass of the e14 modules assigns a Thread-internal name (3.12 + 3.14 lists);
  - a Sampler start/stop test.
  - Both fail on the old code with the exact Kaggle error, and pass on the fix.
  - Unit tests: 167/167 on Python 3.12.10 and on 3.14.6.
- **New rule (user):** CPU validation runs on a **Kaggle CPU kernel** with the same pinned image. Kit: `Build/build_cpu_kit.py`, see memory `kaggle-cpu-validation-before-gpu`.

## 3. Measurements this run did produce

**Staging (E14-0 load path, `/dev/shm`):**

| Model | Size | Copy | Total with verify re-read | Rate |
|---|---|---|---|---|
| INT4 | 19.47 GB | 37.6 s | 103.2 s | 518 MB/s |
| DFlash2 | 3.85 GB | 7.6 s | 19.4 s | 506 MB/s |

- Worker ramp: 4 → 216, 8 → 562, 16 → **786**, 24 → 786 MB/s, so it plateaus at 16. No retries.
- For comparison, the single-stream read probe gave **34.6 MB/s**.

**NCCL bench (topology: all GPU pairs PHB):**

| Arm | 983,040 B all-reduce in a graph (per call) | 64 MiB bus bandwidth | All-gather | Est. comm per step |
|---|---|---|---|---|
| **baseline** | **345 µs** | 4.9 GB/s | 8.0 ms | **≈ 56 ms** |
| `NCCL_P2P_LEVEL=SYS` (P2P/IPC) | 1,887 µs (5.5× slower) | 0.41 GB/s | 81-84 ms | ≈ 348 ms |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | graph part: timeout (eager 500 µs) | 4.6 GB/s | – | – |

- So baseline is best; P2P over PHB is a large loss. No NCCL env arm is a candidate.
- The bench took 308 s.

**Launch (W1 from `/dev/shm`): `ready in 502s`, against 507-562 s on 10-20 from `/kaggle/input`.**
- Phase 1 (weights) ended at 65 s.
- Phase 2 then ran to 502 s. The server log stayed at 33,501 B from 90 s to 420 s, with rchar flat at 6.6 GB.
- So about 7 min of the first launch is compile / JIT / graph capture, not the disk. Staging barely changes the first launch.

**Relaunch (p44401 arm, same machine): `ready in 71s`** (76.8 s with the patch step).
- The compile caches from the first launch are reused within the session.
- **Lever:** carry those caches into the next session as a dataset (the E14 JIT-cache tar). The first launch of a submission could then drop from about 500 s to about 70-100 s. That gives about 7 min more solving per 12 h run and removes most of the 30-min launch-timeout risk on a slow day.
- To verify: which cache directories hold the state (flashinfer JIT, torch inductor / triton, CUDA graphs are not cached).

**Capacity:** KV 866,744, mamba 56 / 4 slots, verify graphs [1-8, 10, 12], effective running 12, gate PASS, smoke OK.

**A1-3 replay (E14-P):**
- Raw `full_reply_reused` 1/3, normalized 2/3, text_match 3/3.
- The three short probe prompts had **no blank lines** in reasoning, so they cannot show the E14-P effect. A probe with blank-line reasoning is needed.

**A1-4 parity:**

| Comparison | Identical | Divergences | Large-margin flips |
|---|---|---|---|
| A/A serial | 10/12 | 2, at tokens 43 and 94, both with margin 0.0 (exact ties) | 0 |
| c12 vs c12 (batching A/A) | 2/12 | p50 at token 34, margins ≤ 0.125 | 0 |
| contamination (serial vs c12) | 0/12 | margins ≤ 0.25 | 0 |

- Verdict: PASS (flips decide).

## 4. Next

- R2 notebook (`Q27-XHIGH-E14A_SLIM_R2_…`, sha 2298d827): the Sampler fix only, plus the new module payload `q27x_e14_modules_78029e146cd3.bin`.
- First the Kaggle CPU harness on the pinned image, then the GPU commit.
