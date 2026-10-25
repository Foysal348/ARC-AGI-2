# E12 rev 4 EVAL6 (Duck REPL agent, Qwen3.8-27B): audit request (2026-10-20)

## Ask

Find **blockers and bugs** (file:line) in the rev 4 changes that would waste the GPU run or give a wrong conclusion. Please:
- separate *certain* from *possible*;
- **do not** propose new features, tests or a different plan. The user chose this small accuracy test.

## Why this run (measured, E12 SGLang EVAL30 run 1, 2026-10-19)

Result: 7/48, task-norm 13.3 %, against R5 Poetiq 16/48 on the same 30 tasks.

**1. Thrash.**
- 30 live branch-0 contexts competed for 867k KV; 144 of 423 later requests lost their prefix, giving 9.6 M uncached prefill tokens.
- Prefill ran at 1,459 tok/s and decode at 431 tok/s, so more than half of the solve time went to re-prefill.
- The offline replay (`analysis/rev4_simulator.py`) reproduces the solve window (12,018 s modelled vs 12,634 s real). Without eviction it gives 5,706 s.

**2. Starvation.** Branch 1 never started in 29 of 30 tasks: no per-branch cap, and the queue held all branch 0s first.

**3. Accuracy.**
- With about 76k tokens per branch, Duck solved 4/30 tasks; Poetiq R5 solved about 12.
- 29 of 30 **first** requests were cut at the 12,288 output cap. Qwen3.8-27B xhigh reasoning is about 60k per request.

## The rev 4 bundle (only `duck_arc2/arc2_duck.py` + the builder; the vendored Duck files are unchanged)

| Change | Where | Value for EVAL6 |
|---|---|---|
| Output per request | `duck_env()` reads `E12_DUCK_MAX_OUTPUT` → `LOCAL_ANALYZER_MAX_OUTPUT` | 60,000 (the user's 60k floor) |
| Live-branch pool | `solve_all(..., live_branches=N)`: a ThreadPoolExecutor with N workers, so at most N branches alive; the rest wait before the gate. Task-major job order, so both branches of a task start together. Queue position `i*branches+b`. The startup gate-hold waits for N, not for all jobs. | 6 (= KV 867k / 128k context) |
| Streams | `E12_STREAMS` default for EVAL6 | 6 |
| Per-branch cap | `E12_MAX_GEN` (Duck's `max_generated_tokens_per_game`, unchanged mechanism) | 100,000 |
| Profile EVAL6 | 6 tasks that S49 / R5 / DIRECT solved and EVAL30 Duck did not: 136b0064, 898e7135, 135a2760, b5ca7ac4, 4c3d4a41, 62593bfd (8 test inputs). Budget 100 min; solve window 60 min. | — |
| Builder guard | SMOKE1 / EVAL30 builds now go to `*_REV4.ipynb`; the Kaggle-run notebooks are never overwritten | — |

Defaults keep the old behaviour when the knobs are unset (SMOKE1 / EVAL30 regression passes).

## Pre-run capacity check

| Item | Number |
|---|---|
| Branches | 6 tasks x 2 = 12; 6 live, 6 streams |
| Tokens | 12 x cap 100k = 1.2 M ≤ about 1.32 M (60 min x 368 tok/s at 6 running, EVAL30 decode summary) |
| KV | 6 x up to 128k context = 768k ≤ 867k |
| Time | about 10 min load (E11-04 W1 567 s first launch) + 60 min solve + margins → 100 min budget → ≤ about 3.3 h quota (hosen42: 33 h left) |
| Decision after | ≥ 2-3 of 6 solved → full rev 4 EVAL30; 0-1 → stop Duck-27B, back to the Poetiq line |

## Local evidence (CPU, the real notebook cells against a fake server)

- **EVAL6 harness: 24/24.** Every Duck request asks `max_tokens` 60000; at most 6 requests are in flight; the first 6 conversations are the first 3 tasks x both branches; all 12 branches finish (by the cap in the harness); scoring and fallback work.
- **SMOKE1 regression** (rev-4 build, knobs unset): 22/22.
- **Adapter unit checks:** ALL OK (39).

## Questions

1. Does `solve_all` with a 6-worker pool interact badly with Duck's priority gate? Look at the startup hold of `ARC3_MAX_ACTIVE_STREAMS` slots, the wait for 6 waiting branches, and queue positions. Can it deadlock or starve when a branch fails setup?
2. With `LOCAL_ANALYZER_MAX_OUTPUT` 60000 and `LOCAL_ANALYZER_CONTEXT_WINDOW` 131,072 (drain to 58k), is Duck's prompt budget still positive at every step? Are there hidden assumptions in Duck about the 12,288 output: yield at 2,048 tokens, resume, trimming, `max_generated_tokens_per_game`?
3. Is the server side safe for 60k outputs: SGLang W1 context 155,648; the request length check; the 12 CUDA-graph / 13 server limits with 6 streams?
4. Does the per-branch cap of 100k cut a branch mid-request cleanly (Duck counts generated tokens), and is the branch then recorded and selected correctly?
5. Anything that makes the result uninterpretable, i.e. a reason that a solve or a non-solve would not mean what the decision rule assumes?
