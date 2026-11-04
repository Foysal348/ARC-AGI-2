## E29_DFLASH_D3S rev 2d: every item assigned to me is done (nothing pushed or uploaded)

All of my rev 2d items are in, each with a test. SCHED's regenerated patch is pinned. All tests pass locally and in an empty folder unpacked from the built kits (Py 3.12 and 3.14), and the three kits are rebuilt.

DEFAULT, MTP_V64 and ALL_OFF were rebuilt and are byte-identical to the audited builds (asserted against the on-disk notebook and push copies):
- DEFAULT `5c74abf8…`
- MTP_V64 `13419f69…`
- ALL_OFF `d7c39120…`

One thing is not done: TEST's `e29_rev2d_adversarial_tests.py` did not exist when I finished, so I have not run it.

**The payload dataset changes and must be uploaded as a new version.** It is listed at the end.

### Notebook
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S.ipynb`, also in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s__foysalemonshanto/`.
- sha `80d4aeff3127d650cb38653a189caf328bf26dcd95185ea4140e0f7bfa8542c2`, 393,048 B, which is 6.9 kB under 400 kB. It supersedes 2c `ed2e23a7`.
- Pinned in the notebook: `e29d_sched_patch.py` 4cc807f2 (SCHED's F1 build; the builder asserts this sha), `e29_nbx.py` 1388aa90, `e29_direct.py` 8b42802c (unchanged).

### Items and their tests
Test IDs are checks in `Build\e29_kit_lib_tests.py`.

| Item | What I did | Test |
|---|---|---|
| **F1 notebook hook** | `e29_direct_setup` sets `E12A.E29D_CAN_START = lambda deadline: time.monotonic() + 60 + drain_min*60 < deadline` next to `E29D_RUN`. Both are cleared on switch-off, on setup failure and when the payload module is missing. | R27: 31 min and 27.1 min left → True; 20 min and 1 min → False; cleared on every off path. SCHED's DIRECT tests 20/20 cover the routing. |
| **F2** | On every DFlash try, before the phases: one forced 130,000-token first prefill, then each worker's own free memory via RPC (`nb1_cache_info`), plus the ledger. | R20, below. |
| **F3** | The 4-rank QSA env read-back is now required before the first prefill. It reads `/proc`, and falls back to the worker RPC (`nb4s_env`) when `/proc` gives fewer than 4 workers. A mismatch or missing rank relaunches the same rung without the QSA env, i.e. Merlin's 256/256 plus EMBED_HOST, recorded. DFlash is never dropped for this. | R21: same pin, QSA env gone, EMBED_HOST kept, nothing dropped. R2: the read-back gate. |
| **F4** | The in-run DFlash → MTP relaunch now runs C-11, the method map and the I8 read-back on the new server. | R23, below. |
| **F10** | The in-run relaunch plan checks for a device fault first: XID alone or XID with an OOM → MTP, recorded as `GPU_DEVICE_FAULT`. A plain OOM → next rung; last rung → MTP; any other death → same config. | R22 (all five cases, plus the full path: top pin / 24 / I8 / no DFlash or QSA env / the F4 gates armed). R10 updated. |
| **F11** | Before the seam is applied, every seam-touched file is snapshotted (bytes + sha). After `finish()` (also when it raised) the files are verified and rewritten from the snapshot if needed. If they are still not pristine, the next MTP boot needs the `q_seam_inert` read-back. I did not make this fail-closed: per your rule the production run is never killed. | R24, below. |
| **F5 / F14** | An `effective_algorithm` record (drafter, DIRECT on/off + why, QSA label, EMBED_HOST) is printed at the start and goes into the report and the LIVE start and 15-min lines. In EVAL, DIRECT off → `experiment_validity: DIRECT_OFF`; the run continues. QSA labels: `QSA_VERIFIED` / `UNVERIFIED_QSA` / `QSA_ENV_MISMATCH` / `QSA_OFF_MERLIN_256_256`. | R25, R21 |
| **F6 / F13** | Ledger, record only: host cgroup memory (current / max / avail), KV usage, running / waiting, preemptions, the reservation (held / reclaimable / attempts / waiting), F and the pool. Recorded at ready, after the F2 probe, and in every LIVE 15-min line. | R28 |
| **F12** | `submission_check`, record only: every task id, one row per test input, valid 0–9 grids for both attempts. Runs after every writer callback (LIVE) and at the end (report). | R26: ok / missing task / short rows / bad grid / no file. |
| **F9** | New code went into the packed block and the payload module, and the verbatim "E21 donor" functions (which no test anchors on) became a second sha-pinned packed literal. The builder, kits and tests now unpack every packed block. | Size above. The retargeted E28 units 44/44 pass, including the check that every changed hunk is marked E29. |

- **R20 (F2):** the probe is 130,000 tokens and sits before the phases.
  - An OOM, dead server, error or no token → `E29_DFLASH_PROBE_FAILED`. With an OOM line the D ladder steps a rung; any other error drops DFlash and the next try is the E28 top rung (24 seqs).
  - Any worker under 512 MiB → `E29_DFLASH_LOW_FREE`, which steps one rung.
  - No RPC answer → recorded only.
- **R23 (F4):**
  - If the INT8 head is not engaged → E28's own I8 rule. The server is killed and the watchdog's retry boots the same pin with the BF16 head.
  - If C-11 or the method map fails → this relaunch fails and is retried once.
- **R24 (F11):** clean → `SEAM_PRISTINE`; a dirty file is restored; still dirty → `SEAM_REVERT_UNVERIFIED`. On the MTP boot, inert → continue (`SEAM_REVERT_UNVERIFIED_INERT`); not inert → that try fails and the E28 ladder steps.

**F2's 512 MiB margin is my choice from the NB-S1 data** (NB-S1's own gate was 512; the lowest seen was 577 MiB at 256/256 with EMBED_HOST). Please confirm it.

### Tests run
- Retargeted E28 units: 44/44.
- e29 kit-lib: 42/42 (R20–R28 new; R2, R10, R16, R19 updated).
- SCHED's `e29d_direct_tests.py`: 20/20.
- The same in an empty folder from the kits' own files, Py 3.12 and 3.14: units 44/44 and 42/42 with `python -I`; direct 20/20 run the way the kit runs it.

### Kits (rebuilt, not pushed)
| Kit | Folder (`_scratch/push_e25/`) | sha | Size |
|---|---|---|---|
| units | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | ad8b9d73 | 371,470 B |
| real | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | a552c139 | 361,183 B |
| direct | `q27-e29-kit-direct__foysalemonshanto` | 6f26ec7f | 84,724 B |

The units kit now loads `k100q.ipynb` and `k100q_fixture.json` from the payload dataset, to stay under 400 kB.

### Payload dataset to upload (new version)
Folder `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` → `foysalemonshanto/q27-e29-final-payload`:

| File | sha | Status |
|---|---|---|
| `e29d_sched_patch.py` | 4cc807f2 | changed (SCHED) |
| `e29_nbx.py` | 1388aa90 | changed |
| `k100q.ipynb` | 4506a71f | new, kit tests only |
| `dflash_seam.py` | 341d859b | unchanged |
| `e29_direct.py` | 8b42802c | unchanged |
| `e29_nbs1_log_fixture.json` | 46ef4839 | unchanged, kit tests only |
| `k100q_fixture.json` | f59cee51 | unchanged, kit tests only |
| the two vocab tables | a164d1a7 / a915fa6d | unchanged |
| `SHA256SUMS` | | updated |

### Files I changed
All in `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\` unless noted; all LF.

| File | sha |
|---|---|
| `build_e29_final.py` | d4fb0513 |
| `e29_lib.py` | 99f97226 |
| `e29_live.py` | 88f713e6 |
| `e29_nbx.py` (generated) | 1388aa90 |
| `build_e29_cpu_kit.py` | acaf0511 |
| `e29_kit_lib_tests.py` | c1ba5546 |
| `e29_dflash_d3s_code_export.py` | 01787a21 |
| `README.md` (experiment folder) | rev 2d entry |

The build facts files are updated too. I did not touch SCHED's or TEST's files. I made no audit zip, since TEST owns the packer this round.

### Open risks
- **Not run on Kaggle yet:** the probe, the QSA gate and the relaunch gates need the real kit and the GPU smoke.
- **F3 RPC fallback:** the E29 worker extension may not have `nb4s_env`. If `/proc` ever gives fewer than 4 workers, the result is `UNVERIFIED_QSA` and the same rung relaunches with 256/256 — safe, but it would run Merlin's defaults.
- **F4:** if the MTP relaunch is already the watchdog's 2nd relaunch, there is no retry left for the BF16 boot (the watchdog limit is 2).
