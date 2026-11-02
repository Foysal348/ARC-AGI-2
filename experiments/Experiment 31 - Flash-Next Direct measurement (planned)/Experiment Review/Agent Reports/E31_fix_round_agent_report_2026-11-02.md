## E31 fix round (the merge list, one round): all 5 items done, nothing pushed

Rebuilt and retested:
- local unit tests: **86/86 PASS**
- the units kit run on its own files in an empty folder: **81/81 PASS**, kit verdict OK
- the extended fake server smoke: passes, including the server kill and one relaunch

The e2e-commit and real kits are built but have not run on Kaggle.

### The five fixes
1. **A phase-1 request error no longer force-closes** (`e31_direct.py`, new `resend_phase1`).
   - Before the drain: phase 1 is resent with the prompt ids plus the kept phase-1 ids (trailing EOS stripped), cap = phase-1 cap − kept ids, and the same seed.
   - If fewer than 64 tokens are left, it counts as a length end and goes to the normal phase 2.
   - At or after the drain: forced close, as before.
   - The bounded retry (2) and "an error is never a finish" are unchanged; there is a new `phase1_resent` counter.
   - The fake servers (the unit-test one and the e2e kit's) now tell phase 1 from phase 2 by the seed, so a resend is served as phase 1.
   - New tests:
     - an HTTP 500 with 0 tokens → phase 1 resent with the same prompt, cap and seed, no phase 2, ends natural;
     - a kill after 24 streamed tokens → the resend prompt is the original prompt + those 24 ids exactly, cap − 24, same seed, ends natural;
     - a server that recovers only after the drain → forced close with the stop ids, no resend;
     - phase 1 always failing → resent twice, then end_reason `error`, with no forced close and no phase 2.
2. **The phase-1 cap reserves answer × N tests**: `phase1_cap(len(ids), think, answer * NTEST[tid], max_model_len)`.
   - Test: b6f77b65 (prompt 4,006 tokens) → **102,234**.
   - Every single-test task's cap is unchanged.
   - All 30 tasks still keep a cap of at least 60k.
3. **CELL 7b workers** = `min(int(E31_CFG["workers"]), int(E12_RUN["streams"]))`. Test: Duck's isolation-gated `workers=6` is ignored and the run uses 24; with streams 16 it uses 16. The Duck argument is still recorded.
4. **Fail closed**: before any request, the solver checks STOP_IDS against the 25-id spec list and the `</think>` id against 248069. On a mismatch it writes `e31_direct_report.json` with status `FAIL_CLOSED_TOKEN_IDS` and the reason, then raises. Test: a wrong stop text raises, and the fake server saw no request.
5. **Units-kit fix**: the Duck contrast test's exec namespace now starts with `from __future__ import annotations` and `from typing import Any, Callable`. I checked it on Python 3.12.10: the snippet runs and Duck's CPU rule lands in attempt_2.

### Files (`D:\ARC AGI 2\experiments\Experiment 31 - Flash-Next Direct measurement (planned)\`)

| File | sha256 |
|---|---|
| `Experiment Notebooks\Q27-XHIGH-E31_FN_DIRECT30.ipynb` (**395,079 B**, < 400 kB, 10 cells) | `d57a7916f9ee8c737f23270f5918d92f5c63fb1d9c08edab1ccc19e74de13072` |
| `Build\e31_direct.py` | `9967a6e0c83067e9d4a4bb3cf262614d356b693acc26a573c946faa630362528` |
| `Build\e31_code_export.py` | `b029821bc8652a2fe2182e117b4ea8c19d3ddad51aeb01dc8e299e231fcac14e` |
| `Build\build_e31_direct.py` | `fc60aaf7…` |
| `Build\build_e31_cpu_kit.py` | `e640b008…` |
| `Build\e31_unit_tests.py` | `5ddd614a…` |
| `Build\e31_build_facts.json` | `0bc4d2ed…` |
| `Build\e31_kit_facts.json` | `4fa8bfd4…` |

The audit zip `Experiment Review\Audit Zips\E31_FN_DIRECT30_audit_FLAT_2026-11-02.zip` (415,081 B) was refreshed with the new files and logs.

The E31_CFG line is unchanged:
```
E31_CFG = {'think': 104000, 'answer': 8192, 'window_min': 170, 'workers': 24, 'temperature': 0.6, 'drain_min': 23}
```

### Push folders (written, not pushed)
Both hold the same notebook (`d57a7916…`) and the same datasets, models, image and NvidiaL4 machine:
- `D:\ARC AGI 2\_scratch\push_e25\q27-xhigh-e31-fn-direct30__esfersami50\` — id `esfersami50/q27-xhigh-e31-fn-direct30`
- **new:** `D:\ARC AGI 2\_scratch\push_e25\q27-xhigh-e31-fn-direct30__foysalemonshanto\` — id `foysalemonshanto/q27-xhigh-e31-fn-direct30`

### Kit folders (`D:\ARC AGI 2\_scratch\push_e25\`, all on esfersami50, not pushed)

| Kit | Folder | Size | sha256 |
|---|---|---|---|
| units | `q27-e31-fn-direct30-kit-units__esfersami50` | 211,201 B | `abea820d752bc3cc…` |
| e2e-commit | `q27-e31-fn-direct30-kit-e2e-commit__esfersami50` | 212,932 B | `8639d436613172e3…` |
| real | `q27-e31-fn-direct30-kit-real__esfersami50` | 282,469 B | `ad2a087afa78df23…` |

The kits are still built for esfersami50. If they should also run on foysalemonshanto, only `OWNER` in the kit builder needs to change.

### Fake-server smoke (30 tasks, 24 workers)
- **Normal:** 30 finished in 4.7 s (23 natural, 7 forced close); all 7 forced-close prefixes token-exact; 24 streams at once on the fake.
- **Kill + one relaunch:** 30 finished; 9 request errors and 9 retries; 0 error finishes; returned early; 6 forced-close prefixes recorded, all exact.

### Still open
- The e2e-commit and real kits have only been built and compiled locally; they have not run on Kaggle.
- The bounded retry of 2 remains my own addition from audit E31-6.
