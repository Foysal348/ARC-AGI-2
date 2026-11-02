## NB-S1 round 2: moved to the foysal account, real kit built, DYN kept

All four items are done except the optional profile parse. The NB-S1 notebook is rebuilt as **0c432f93** (gate still closed, `gpu_ready: false`). Units pass **56/56** locally. Both kits are ready for you to push together, and both are on the foysal account.

Nothing was pushed and git was not run. No NBDF / K100 file changed: the frozen-file hash assert (13 files) runs before and after both the notebook build and the real-kit build.

### Kit folders and new outputs (all `__foysalemonshanto`, private)

| Item | Folder / path | sha256 | Bytes |
|---|---|---|---|
| **Real kit** | `D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs1-kit-real__foysalemonshanto/` | **f760371f19026d40b83482ab721409ac64412f8ea377e6c90816b2340880b7d3** | 352,995 |
| **Units kit** | `D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs1-kit-units__foysalemonshanto/` | **cf71d5e89007e46931cb27f37c953cfef38f45ece2c5dbfd3589f1cae7e7fe86** | 380,393 |
| Notebook | `Experiment 25/Experiment Notebooks/Q27-XHIGH-E25_NBS1.ipynb` | **0c432f93acef40f6d11401be1ecbc973ed8044558ffd09c45be45896f4493aac** | 381,954 |
| GPU push folder | `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbs1__foysalemonshanto/` | id `foysalemonshanto/q27-xhigh-e25-nbs1` | — |
| Audit zip | `Experiment 25/Experiment Review/Audit Zips/E25_NBS1_audit_FLAT_2026-11-01.zip` (now also holds the kit builder and the real-kit notebook) | 7c721a2cafce138970409576f34d0a72469bbb39f91a640e5a0d4e481a0e578a | 907,449 |

- **Real kit inputs:** the 4 Models (klee100, PLE overlay, Merlin env, DFlash2 drafter) and the dataset `foysalemonshanto/q27-e25-nb0-tf-ref`. The units kit needs no inputs.
- The old `__hosen42` folders are deleted.

### Files (in `Experiment 25/Experiment Notebooks/Build/`)

| File | sha256 (prefix) | Note |
|---|---|---|
| `build_e25_nbs1.py` | 0b66edf2 | owner set to foysalemonshanto; NBDF's own push folder is kept |
| `nbs1_lib.py` | 9668adca | r3: only the DYN source note changed |
| `nbs1_unit_tests.py` | 34e4f1f8 | unchanged |
| `nbs1_cell7_body.py` | 9aac6ab2 | generated |
| `nbs1_code_export.py` | b3f2f28c | generated |
| `build_e25_nbs1_units_kit.py` | e86e5a7e | owner set to foysalemonshanto |
| `build_e25_nbs1_cpu_kit.py` | b03f606f | new: the real-kit builder |

All five source files are LF.

### 1. Moved to the foysal account
The NBDF builder already writes `<slug>__foysalemonshanto`, so NB-S1 now uses that folder directly. The kernel id is asserted to be `foysalemonshanto/q27-xhigh-e25-nbs1`, private.

### 2. How the real kit is built
`build_e25_nbs1_cpu_kit.py` reads the text of `build_e27_nbdf_cpu_kit.py` and, with `NBDF_VARIANT=k100`, points it at the NB-S1 notebook, facts, export and body, with the NB-S1 arms and slug. It runs that text in memory and builds only the real kit.

- **Kept from K100:** every K100 substitution stays: the `k100_setup` / scales-sidecar step, `VLLM_TARGET_DEVICE=cpu` around the notebook's own `--help`, `PSEQ.before_arm(ARMS["Q"])`, per-arm `SSEQ.check` + `PSEQ.before_arm`, `arm_boot`'s first try, and `launch_check` at `K100_CFG q_pin`.
- **Kit-must-mirror:** every arm goes through the notebook's own `merlin_fn_server` chain (K6, Merlin build + host_phase, the `--help` flag gate, the NBDF launch check, `VLLM_QSA_KV_SCALES`) and `nbdf_lib.launch_check`. The argv fields are compared with the arm's own spec, captures and `want_sp`. There is no kit-only spec expectation.
- **NB-S1 edits to the in-memory template:**
  - **KV probe:** a new `K3_fp8_sw2048` run (fp8 drafter KV, SW block 2048, TRITON_ATTN) for F8. B, G1, G2 and EH use `K3_swblock1024`; EH changes no KV spec at the same pin, so its gain shows only as free memory, which is recorded. DYN uses `K4_swblock1024`.
  - **KV probe argv:** taken from B (key Q), which has the D3s config.
  - **Env change per plan:** each plan records the arm's own change (GDN / EMBED_HOST / SW block) from the final launch env.
  - **Spawned-child check:** it now reads all of the arm's `expect` keys, GDN keys included. Without this, the spawn check would have failed for every NB-S1 arm.
  - **Captures item:** it covers all 6 arms, B plus one change each. DYN coverage is tier-aware (b×5 for b ≤ 8, b×4 for 9-24).
- **New verdict keys:**
  - the three the build gate needs: `nbs1_arm_plan_ok`, `nbs1_launch_path_all_arms_ok`, `nbs1_help_flag_gate_real_ok`;
  - plus `nbs1_kv_probes_ok`;
  - plus `checks.NBS1` with the step-coverage table.
- **Local test of that block:** I ran the kit's NB-S1 block with fakes built on the real NB-S1 arms and `launch_check`:
  - All 4 keys are True and every step is covered.
  - With the F8 KV run removed and G1's env change lost, `nbs1_launch_path_all_arms_ok` and `nbs1_kv_probes_ok` both turn False.

**Step-coverage table** (`checks.NBS1.steps`). The kit fills the "covered" column at run time.

| Step | Notebook function | Kit equivalent | Covered |
|---|---|---|---|
| merge | CELL 3 merge | R1 merge | yes |
| install | `merlin_install()` | R2 | yes |
| #60490 + worker extension | `apply_60490()` | R3 | yes |
| k100_setup + scales sidecar | `k100_setup()` | K100_SETUP (+ sidecar exists) | yes |
| sha gate | `sha_gate()` | R1 sha | yes |
| nb4_ppack | `pseq.before_arm(ARMS["Q"])` | R3P, the same call | yes |
| dflash_seam apply + ring | `sseq.apply()` | R3S | yes |
| B base gate | `S1.base_gate` | R1 drafter pins + R3S seam (this kit is the kit gate) | yes |
| arm plan | `s1_arms` / `install` / `check_arms` | `check_arms` on the notebook's ARMS | yes |
| seam / ppack revert | `finish()` | R3S / R3P pristine | yes |
| per arm ×6: D-loop checks | `SSEQ.check` + `PSEQ.before_arm` | the same calls in `plan_one` | yes |
| per arm ×6: tree check | `tree_check` | the same call | yes |
| per arm ×6: arm_boot knobs | NB1_EXTRA / NB1_ENV / NB1_KV at q_pin, ARM_OF, NB4T_BUILD_ENV | the same lines | yes |
| per arm ×6: K6 | K100 wrapper | through `merlin_fn_server` | yes |
| per arm ×6: build + host_phase | CELL 2 launcher | through `merlin_fn_server` | yes |
| per arm ×6: `--help` gate | `L.help_check` + `KL.cli_flags_check` | through `merlin_fn_server`, real help text on CPU | yes |
| per arm ×6: launch check | NBDF launch check + `launch_check` at the pin | the same | yes |
| per arm ×6: QSA scales env | K100 wrapper | launch env recorded and checked | yes |
| per arm ×6: env in spawned child | worker spawn | R4 spawn, the arm's full `expect` | yes |
| per arm ×6: the arm's own env change | arm env | launch env vs plan | yes |
| per arm ×6: argv spec / captures / seqs / KV bytes | `launch_check` `want_sp` | R5 + captures item | yes |
| per arm ×6: KV probe | engine KV config | R9 run for that arm | yes |
| boot_gate, LoadMonitor / readiness | GPU only | none | **no** |
| `S1.readback` (worker env RPC, dflash read-back, DYN fallback line), `k100_boot_gates` | GPU only | none | **no** |
| windows / oracle / TF / profile | GPU only | none | **no** |

### 3. DYN source check: DYN stays
- `v1/worker/gpu/model_runner.py:2786-2803` passes `num_speculative_tokens=<this batch's K>` to `speculator.propose()`, and sets `draft_tokens[:, K:] = -1` when K is below the max.
- `DFlashSpeculator.propose` (`spec_decode/dflash/speculator.py:343-369`, the DFlash2 base) accepts that argument but always drafts its full fixed block (`num_query_per_req = 1 + num_speculative_steps`, line 62).
- So a per-batch K works: the drafter costs K4 at every batch size, and only the target's verify length shrinks at 9-24 running. It is not a "cannot".
- The real-failure read-back stays (an "Overriding cudagraph_mode" or "Disabling num_speculative_tokens_per_batch_size" line). The source note is in `nbs1_lib.DYN_SOURCE_CHECK` and the notebook flags.

### 4. Draft / verify ms per token from B's profile: not done
It was not quick. B's profile (20k@24, 30 s of load, trace bounded at 40 iterations) still records only what NBDF's existing trace analysis gives.

### Tests

| Run | Result |
|---|---|
| `nbs1_unit_tests.py` (local) | **56/56 PASS** |
| NB-S1 units kit (local run) | the tests pass; the kit line itself requires Python 3.12, which only the Kaggle image has (local is 3.14) |
| Real kit | builds and compiles; it runs only on Kaggle (Merlin and the mounts) |

### Next
After you push both kits and the real kit prints ALL OK, run the gated rebuild from Exp 25 `Build/`:
```
python build_e25_nbs1.py --kv-gate-report "<real kit out>/nbdf_kit_real_report.json"
```
The gate opens only if the K100 gate parts (D3s per-K success included), the kit's ALL_OK and the three `nbs1_*` keys are all True.
