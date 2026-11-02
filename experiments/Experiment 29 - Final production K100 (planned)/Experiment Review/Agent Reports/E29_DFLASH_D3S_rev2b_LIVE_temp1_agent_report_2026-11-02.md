## E29_DFLASH_D3S rev 2b: LIVE progress lines + DIRECT temperature 1.0 (nothing pushed or uploaded)

Rev 2b is built on top of rev 2, and everything from rev 2 is kept. DEFAULT, MTP_V64 and ALL_OFF were rebuilt and are byte-identical to the audited versions (asserted against the on-disk notebook and push copies):
- DEFAULT `5c74abf8…`
- MTP_V64 `13419f69…`
- ALL_OFF `d7c39120…`

All tests pass, including from the built kits' own files in an empty folder on Py 3.12 and Py 3.14.

The payload dataset must still be uploaded as a new version before any GPU commit or kit.

### New notebook
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S.ipynb`, also in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s__foysalemonshanto/`
- sha `beebb252bdea9ecab22815fec5c5dfa90a0e558f5c0e09900627a41e462a6c77`
- 398,949 B, which is 1,051 B under 400 kB. That margin is thin.
- To fit, CELL 3's DIRECT apply step moved into the dataset module (`e29d_sched_patch.cell3`, same logic) and a few comments were shortened.
- It supersedes rev 2 `814fc0f0`.

### DIRECT temperature 1.0
- The first line of CELL 1 is now `E29_DIRECT2 = {'on': True, 'think': 104000, 'answer': 8192, 'temperature': 1.0, 'drain_min': 26}`.
- Every DIRECT request takes its temperature from that switch, so it sends 1.0. top_p 0.95, top_k 20 and min_p 0.0 are still sent explicitly, unchanged.
- Tests, facts and README are updated.

### LIVE lines (print only)
- **One line per finished attempt**, printed after the original callback with flush=True. Fields:
  - minutes since notebook start, task id, phase (A / B1 / B2 / C-DIRECT / supply), and train pass k/n (always);
  - exact vs gold only when SOLUTIONS are loaded, so the hidden rerun shows no gold fields;
  - running totals per phase (done / accepted / exact);
  - queries correct and task-norm over the finished tasks, computed from the current attempt_1 / attempt_2 selection by the notebook's own `build_submission`;
  - for DIRECT: mode, generated tokens, grids found / N, exact k/N and running attempt_2 hits.
- **Every 15 min:** a `[LIVE 15-min]` line with server tok/s (from the generation counter), running streams and the totals.
- **How it is wired:** the logic is `e29_lib.LiveLog`; `e29_live_setup` wraps `E12A.solve_all_r6` once so that `on_branch_done` also prints. It covers Duck attempts and DIRECT task ends, since both go through the scheduler's publish callback. The original callback runs first, every LIVE exception is caught and counted, and records, submission, scheduler and solver are untouched.

### Payload dataset (upload as a new version)
Folder `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` → `foysalemonshanto/q27-e29-final-payload`:

| File | sha | Note |
|---|---|---|
| `dflash_seam.py` | 341d859b | |
| `e29_direct.py` | 8b42802c | unchanged |
| `e29d_sched_patch.py` | cfb5834c | new: adds `cell3` |
| `e29_nbs1_log_fixture.json` | 46ef4839 | kit tests only |
| `k100q_fixture.json` | f59cee51 | new, kit tests only; moved out of the units kit for size |
| `draft_vocab_k100_arc_n32768.pt` | a164d1a7 | unchanged |
| `draft_vocab_k100_arc_n65536.pt` | a915fa6d | unchanged |
| `SHA256SUMS` | | now covers every file in the folder |

### Tests (all local, all pass)
- **Retargeted E28 units:** 44/44. The A6 check (every changed hunk marked E29) now also accepts the case where every hunk is marked.
- **e29 kit-lib:** 31/31. New in 2b:
  - R15: lines with and without solutions, exact totals on a fake run, `[LIVE 15-min]` format.
  - R16: an exception inside LIVE never propagates; the original callback gets every record first and the return value is unchanged.
  - R17: the LIVE wiring.
  - R11: the temperature 1.0 literal; R14: CELL 3 calls `cell3`.
- **e29d_direct_tests:** 12/12. New in 2b: `cell3` only acts on the E29 scheduler, sets E29_DIRECT only after verify, and on failure reverts and never writes an unknown file.
- **Kit-layout runs in an empty folder** (Py 3.12 and Py 3.14): units 44/44 and 31/31 with `python -I`; direct 12/12 run the way the kit runs it (no `-I`, because Duck needs `requests` from site-packages).

### Kits (rebuilt for this variant only, not pushed)
| Kit | Folder (`_scratch/push_e25/`) | sha | Size |
|---|---|---|---|
| units-dflash-d3s | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | 516a1f7d | 390,586 B |
| real-dflash-d3s | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | 97647fd0 | 354,878 B |
| direct | `q27-e29-kit-direct__foysalemonshanto` | 5903d0e4 | 80,914 B |

The units kit now loads the K100Q fixture from the mounted payload dataset.

### Files changed in 2b
All in the Build folder unless noted; all LF.

| File | sha |
|---|---|
| `build_e29_final.py` | 3c218ca7 |
| `e29_lib.py` | 353461be |
| `e29_direct_build.py` | f80cb5c4 |
| `e29d_sched_patch.py` | cfb5834c |
| `build_e29_cpu_kit.py` | 9ae950ad |
| `e29_kit_lib_tests.py` | eaf78207 |
| `e29d_direct_tests.py` | 00628af6 |
| `e29_dflash_d3s_code_export.py` | 3ce8efd2 |
| `README.md` (experiment folder) | rev 2b entry added |

The build facts files are updated too.

Audit zip: `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Review\Audit Zips\E29_DFLASH_D3S_REV2B_LIVE_DIRECT_audit_FLAT_2026-11-02.zip`

### Open risks
- **Size margins:** 1,051 B on the notebook; the units kit is 390,586 B.
- **LIVE cost:** each line runs `build_submission` for that one task (CPU only, cheap).
- **The 15-min tok/s** is the change in the server's generation counter, so the first summary line shows "?".
- **Not run on GPU yet:** DFlash in Duck, DIRECT inside the scheduler, and LIVE.
