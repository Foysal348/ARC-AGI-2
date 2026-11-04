## E29_DFLASH_D3S rev 2c: top-rung MTP fallback + QSA workspace switch (nothing pushed or uploaded)

Both approved fixes are in. All tests pass locally, the notebook is 396,772 B (3.2 kB under 400 kB), and the three kits are rebuilt. DEFAULT, MTP_V64 and ALL_OFF were rebuilt and are byte-identical (asserted against the on-disk notebook and push copies): `5c74abf8…`, `13419f69…`, `d7c39120…`.

**The payload dataset changes.** It has a new file, `e29_nbx.py`, and `SHA256SUMS` is updated. Upload it as a new version before any GPU commit or kit. With the old version mounted, DIRECT and LIVE turn off; nothing else is affected.

Everything from rev 2b is kept: seam 341d859b, EMBED_HOST from the first D try, the OOM ladder at any stage, real-failure-only gates, indexer_fp8 off, DIRECT at T 1.0, LIVE lines.

### New notebook
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S.ipynb`, also in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s__foysalemonshanto/`
- sha `ed2e23a72c7fcbb9a15a9758544d499e59c5b528681387cad4bc23a4710d871c`, 396,772 B (supersedes 2b `beebb252`)
- To make room, the DIRECT and LIVE notebook functions moved into the payload module `e29_nbx.py`. The notebook loads it by name + sha into its own namespace (`e29_extras()`). If it is missing, DIRECT and LIVE turn off and phase C runs today's Duck attempt 2; the run does not stop.

### 1. Blocker: the MTP fallback after a DFlash drop
**Cause, two parts:**
- In the notebook, after the D ladder it jumped to "the first non-D try", which is whatever rung E28's own ladder starts at. In production, with #60490 verified, that is the top rung.
- In the kit, the namespace never got CELL 8's settings before `e29_setup`, so #60490 looked unverified and I8 was off. E28's fail-safe then started its ladder at the RC2 pin: 2,479,838,700 B, 6 seqs, I8 off — exactly what the rev 2b real kit saw.

**Fix in the notebook:**
- Whenever DFlash is dropped (ladder end, XID, a gate, GRD), the next try is now always the E28 production top rung: MTP K3, 7,933,526,016 B, `--max-num-seqs 24`, INT8 head restored, E28 captures (`e29_lib.mtp_top`).
- Only if that try fails does E28's own ladder take over: first E28's I8 rule (same pin, BF16 head), then its own rungs.
- The in-run switch to MTP (OOM with no rung left, or the zero-accept canary) relaunches on the same top rung with I8 on (`e29_mtp_boot`). Rev 2b used a BF16 head there.
- One deliberate deviation: if #60490 were ever unverified in production, this one fallback try still uses the top pin, where E28 alone would start at the RC2 pin. The D tries already use that pin anyway.

**Fix in the kit:** before `e29_setup`, the real kit now sets #60490 from its own real apply and I8 from the profile, as CELL 8 does. It also checks that the first MTP try after the drop is top rung / 24 seqs / INT8. The existing check that every built argv parses to 24 seqs covers that try.

**Tests:** R18 (new) covers ladder end and XID, including when E28's list starts at the RC2 pin. R7 and R10 were updated for the fallback and the in-run switch.

**E29_MTP_V64:** the notebook bug does not apply there, because it has no D tries and its lever drops retry the same pin. In production its ladder starts at the top rung. Its real kit, which I did not rebuild, has the same kit gap and would show the same pin0 / 6-seqs failure. Fixing that is a kit-only rebuild; MTP_V64's bytes do not change.

### 2. QSA indexer workspace (your decision)
- **Switch:** CELL 1 line 2, a plain editable literal: `E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}`. A bad value means Merlin's defaults, recorded, never a stop.
- **Env, DFlash tries only:** `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` and `VLLM_E2_QSA_LOGITS_GRANULE_MB=64`. They travel with the other boot-try env and are applied after Merlin's `build()`, which sets both to 256 (`launch.py:419-420`). So our values are what the server process gets.
- **MTP fallback and in-run MTP relaunch:** I kept Merlin's 256 / 256 there; the QSA values are stripped. Reason: 256 / 256 is the GPU-proven E28 MTP config, the OOM was a DFlash PP1 problem, and this keeps one unproven change per run. The real kit asserts 256 on the MTP tries.
- **Read-back (recorded only, not a gate):** after ready and again after the warm-up, the spawn env of the 4 `VLLM::Worker` processes under the server pid is read from `/proc/<pid>/environ`. It is recorded together with the granule Merlin logs at the first prefill and the backing bound. A mismatch only means Merlin's proven 256 / 256 is in effect, so I did not make it able to drop DFlash.
- **Allocation size:** Merlin has no per-allocation log line without an OOM, so the proof is the formula. Merlin rounds each chunk up to the granule but caps it at max_logits (`qsa_indexer.py:510` and `:776`; the fixed backing is used only when granule is 0, `:789`). R19 checks the bound stays ≤ 128 MiB over widths 64–32,768 and 1–131,072 rows. The real allocation size is only shown on GPU.

### Payload dataset (new version needed)
Folder `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` → `foysalemonshanto/q27-e29-final-payload`:

| File | sha | Status |
|---|---|---|
| `e29_nbx.py` | cb42b7ff | **new** |
| `dflash_seam.py` | 341d859b | unchanged |
| `e29_direct.py` | 8b42802c | unchanged |
| `e29d_sched_patch.py` | cfb5834c | unchanged |
| `e29_nbs1_log_fixture.json` | 46ef4839 | unchanged (kits only) |
| `k100q_fixture.json` | f59cee51 | unchanged (kits only) |
| the two vocab tables | a164d1a7 / a915fa6d | unchanged |
| `SHA256SUMS` | — | updated |

### Tests (all local, all pass)
- Retargeted E28 units: 44/44.
- e29 kit-lib: 33/33. New: R18 (top-rung fallback) and R19 (QSA: switch line, env on D tries only, fake `/proc` worker-env read-back, logged granule, Merlin formula bound). Updated: R2, R4, R7, R10, R12, R13, R16, R17.
- DIRECT tests: 12/12.
- Kit-layout runs in an empty folder from the kits' own files, on Py 3.12 and 3.14: units 44/44 and 33/33 with `python -I`; direct 12/12 run the way the kit runs it.

### Kits (rebuilt, not pushed)
| Kit | Folder (`_scratch/push_e25/`) | sha | Size |
|---|---|---|---|
| units | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | 810147c3 | 389,897 B |
| real | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | c6ffea7b | 354,011 B |
| direct | `q27-e29-kit-direct__foysalemonshanto` | 682606ce | 81,117 B |

### Files changed in 2c
All in `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\` unless noted; all LF.

| File | sha | Note |
|---|---|---|
| `build_e29_final.py` | 545326b0 | |
| `e29_lib.py` | b3e72044 | adds `mtp_top`, the QSA helpers and the `/proc` read-back; LiveLog moved out |
| `e29_live.py` | cbc244f6 | new: the LiveLog class |
| `e29_nbx.py` | cb42b7ff | generated |
| `build_e29_cpu_kit.py` | 7f684f7b | |
| `e29_kit_lib_tests.py` | b2c4ff3a | |
| `e29_dflash_d3s_code_export.py` | 6322308c | |
| `README.md` (experiment folder) | — | rev 2c entry added |

The build facts files are updated too. Audit zip: `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Review\Audit Zips\E29_DFLASH_D3S_REV2C_audit_FLAT_2026-11-02.zip`.

### Open risks
- The QSA 128 / 64 setting and the in-run MTP relaunch with the INT8 head (no INT8 read-back gate on a relaunch) are not GPU-measured yet.
- The real kit's new checks have not run on Kaggle; they need the new dataset version.
- DFlash in Duck, DIRECT inside the scheduler, and LIVE have still not run on GPU.
