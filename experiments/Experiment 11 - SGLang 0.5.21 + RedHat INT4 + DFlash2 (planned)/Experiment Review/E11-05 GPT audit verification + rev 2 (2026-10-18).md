# E11-05: verification of the GPT audit (2026-10-18) and rev 2

Every finding was checked against the SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src`), the notebook and the E11-04 results before any change. The GPT verdict was NO-GO for the LB submission and GO for a patched EVAL validation.

## Findings

| # | GPT finding | Check | Decision |
|---|---|---|---|
| 1 | Capacity caps parsed but not used as a gate | **True.** rev 1 always set CONCURRENCY = 12 (my reasoning was that the server queues the extra requests). | **Fixed.** `e11_capacity_gate`, fail-closed: every limit must be in the log. CONCURRENCY = min(12, mamba // slots, logged cap, largest verify graph ≤ 12, KV // 70k). Below 8, or with missing evidence → S49 vLLM. |
| 2 | Drain in 8 min not proven | **True.** E11-04: an aborted 30k prefix had 0 cached tokens, 24.5 s. Source: an aborted running request is released through the normal path with `is_insert` (batch_result_processor.py:1347-1352), yet E11-04 measured 0, so the reuse cannot be relied on. | **Fixed (bounded).** On SGLang the drain is 12 min before the hard stop (was 8). It costs 4 min of thinking at the end. |
| 3 | `--stream-interval 8` can hold back deltas | **True.** output_streamer.py:427-436: with N > 1 the rule is `len % N == 1`; DFLASH accepts several tokens per step, so the residue can be skipped for long stretches. | **Fixed.** Interval 1: a chunk every step. |
| 4 | KV gate inert (floor 12 = CONCURRENCY 12) | **True.** Same as S49 on 16 slots with a floor of 12 (active there between 12 and 16). Here the worst case is 12 × (≤ 24k + 49k + 8k) > 881k. | **Fixed.** On SGLang the floor is 10 and the threshold 0.85 (full_token_usage, evictable cache excluded). |
| 5 | Abort is asynchronous; phase 2 started at once | **True.** tokenizer_manager.py:2114-2140 only dispatches; scheduler.py:5558-5572 finishes a running request after one more decode pass. | **Fixed in the shim** (the scheduler is unchanged). `abort_request` sends all POSTs in parallel, then waits (one 30 s deadline) until each stream has ended. It returns a per-rid outcome. |
| 6 | 12 sequential aborts, worst case 360 s | True (worst case). | **Fixed** by #5 (parallel POSTs, one deadline). |
| 7 | Transport error treated as a completion | Partly. An `error` finish goes on to forced close, or to code if `</think>` is already there. The code is still checked by train verification, so a truncated program fails train. A dead server raises in `step()` before this happens. | **Not changed.** The scheduler stays S49's. Errors are counted in shim stats and in `sglang_requests.jsonl`. |
| 8 | Partial thinking lost on a solver error | **True** (breaks the save-every-thinking rule). | **Fixed.** On a solver error the scheduler writes `<task>_e<expert>_i<iter>.partial.txt` for every unit still running. This is the only scheduler change (asserted < 900 chars). |
| 9 | Tokenizer: one probe only | Fair. | **Fixed.** Vocab + **merges** (normalized; locally equal: 247,587) + probe ids on a real ARC prompt (all train pairs of the first task) + EOS. |
| 10 | Give-up deadline vs. smoke | Minor. | Smoke waits 300 s per part (was 600). |
| 11 | rid prefix match in abort | Not raised by GPT. The source matches with `rid.startswith` (scheduler.py:5460, 5572). | **Safe.** Our rids end in `\|1` / `\|2`, so no other live rid starts with an aborted rid. |
| 12 | Upstream issues #36876, #43204, #34149, #24221 | Not verified offline. | Treated as hypotheses. We do not disconnect clients. A mamba allocation crash would end the solver; the submission and the partial thinking are still written. |
| 13 | Speed at 40–60k, quality vs FP8, real 12-way drain | Need a GPU run. | **Open:** the EVAL validation run (below). |

GPT says the fake drain lost 64 tokens per request. The harness log shows 0 and 32 (one chunk in flight).

## rev 2 harness (12 scenarios)

The 10 scenarios of rev 1, plus:
- **lowcap:** mamba cache 40 → 10 effective → concurrency 10.
- **nocaplog:** the mamba line is missing → the gate fails → fallback.

New checks:
- interval 1;
- the gate passes for W1;
- KV gate 10 / 0.85;
- drain: the fake abort lands 0.5 s after the 200, and phase 2 is sent only after the old stream ended;
- serverdeath: the partial thinking files are written;
- rerun: drain 720 s before the hard stop.

## Next step (user's decision)

GPT and I agree: before the scarce LB submission, run one EVAL validation on GPU, because it measures what no CPU test can:
- 40–60k decode speed;
- 12-way drain;
- forced close and iteration-2 cache;
- accuracy against S49 FP8 on the same tasks.

A commit run with the EVAL120 profile does all of this, and the same version is then submittable (its rerun uses RERUN_ALL).
