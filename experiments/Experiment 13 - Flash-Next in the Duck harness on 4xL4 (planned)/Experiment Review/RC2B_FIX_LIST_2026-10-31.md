# E13 RC2b: fixes from the three pre-GPU GPT audits (2026-10-31)

**Input:** three independent GPT audits of `E13_RC2_AUDIT_FOR_GPT_2026-10-31.zip` (notebook 2fb12343).
- All three said **NO-GO for the artifact, GO for the architecture**: Merlin `build("flash-next")`, the TP2×PP2+EP profile, P2P off, the BF16/FP8 selection, the trailing 131072 / xhigh, the checkpoint merge design, TF last, the sampled check first, graceful shutdown, the budget and the rerun.
- One audit ran `build()` from the attached Merlin source on CPU and confirmed the profile, NCCL_P2P_DISABLE=1, BF16 3 seqs / FP8 8 seqs, the scales path, and the 131072 / xhigh overrides.

**Result:**
- Notebook `Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb`, sha256 **fb02980b**, 136,302 bytes.
- `rc2_unit_tests.py` **34/34 PASS**.
- Local fake flows: main, no-Merlin fallback and rerun, all OK.

| # | Audits | Severity | Item | Fix |
|---|---|---|---|---|
| 1 | A1, A2, A3 (reproduced by two) | HIGH | A missing oracle or a failed sampled check gave CLEAN | `evidence_gate()` runs before E23's `oracle_gate`. Probes, the sampled check and a COMPLETE oracle are mandatory. Each missing or failed piece has its own non-CLEAN verdict (INCONCLUSIVE_PROBES_MISSING, NO_GO_SHORT_PROBE, INCONCLUSIVE_NO_SAMPLED_TEST, NO_GO_SAMPLED_PATH, INCONCLUSIVE_NO_ORACLE, NO_GO_ORACLE_REQUEST_ERROR, NO_GO_ORACLE_GARBAGE, INCONCLUSIVE_ORACLE_INCOMPLETE) |
| 2 | A1, A3 | HIGH | Oracle completeness was not recorded | `oracle()` now records `row_count`, `expected_rows`, per-row `tokens` and `complete` (every expected row, no error, exactly NTOK tokens; greedy runs with ignore_eos). A reference is stored only when complete and garbage-free. The stock oracle is parameterized to 3 rows |
| 3 | A1, A2, A3 | HIGH | FP8 gate INCONCLUSIVE still ran the ladder, census and long rows | `fp8_gate()` returns FAIL_SAMPLED, FAIL_PROBES, FAIL (error, garbage or FAIL_HIGH_MARGIN), PASS (complete + EXACT/NEAR_TIE) or INCONCLUSIVE. **Only PASS runs the load tests.** Otherwise `load_tests_skipped` is recorded and the status becomes NO_GO_FP8_KV or INCONCLUSIVE_FP8_KV. BF16-load runs only after a FAIL* |
| 4 | A1, A3 | HIGH | Merlin `host_phase()` failure was ignored | It is recorded in `R.setup.host_phase_error` and raised as `MERLIN_HOST_PHASE_FAILED` before any server starts. The step is then ERROR, and A1's failure selects the stock fallback |
| 5 | A2 | HIGH | Any server death in TF was TF_OOM, which let A3 start after a CUDA fault | It is TF_OOM only with an OOM line in the log tail AND no Xid/device fault. Otherwise it is TF_SERVER_DIED, and A3 does not start (A1 must be DONE or TF_OOM, and CLEAN) |
| 6 | A1, A2, A3 | HIGH | The stock fallback crashed in CELL3 without the Merlin tokenizer | Prompts come from the Merlin checkpoint when the merge is OK, otherwise from the Intel checkpoint (`chat_template.jinja`, or `tokenizer_config.json`'s `chat_template`). The Merlin hash assert applies only to the Merlin path; Intel hashes are recorded. `prompt_render.source` is merlin or intel_fallback |
| 7 | A1, A2, A3 | HIGH | The stock boot gate was not enforced; the Merlin gate accepted None | Both launch only on `gate["ok"] is True`; otherwise GPU_NOT_EMPTY. Stock also returns FALLBACK_UNAVAILABLE without the Intel checkpoint or the wheelhouse |
| 8 | A3 | MED | census_long garbage was not gated | `_windows()` covers `steady` and `census_long` |
| 9 | A3 | MED | TF not strictly last | `census_end` now runs before TF in both arms |
| 10 | A3 | MED | Long rows run concurrently | Kept concurrent for the budget, but labelled `concurrency: 2`, never single-stream latency. Cap raised from 2,700 s to 3,000 s. A cap cut is recorded as incomplete, not as a correctness failure |
| 11 | A3 | MED | /dev/shm cleanup was too aggressive | Only entries that appeared during THIS boot (absent from the pre-boot snapshot) and match the engine prefixes are unlinked. Everything else is recorded and kept |
| 12 | A3 | MED | Memory before the second boot | `boot_gate()` records host memory (cgroup, inactive_file, MemAvailable, /dev/shm) next to GPU memory before every boot |
| 13 | A2 | (own) | BF16-load has no oracle of its own | `certification: DIAGNOSTIC_ONLY`; gated without an oracle requirement (probes and sampled still required); load runs only after a passing sampled check |
| 14 | A1 | note | Both boots run MTP K3 | `R.correctness_meaning`: CLEAN means BF16_MTP3-vs-FP8_MTP3 parity, not target-only correctness. A production GO needs an MTP-off oracle and a forced-rejection test |
| 15 | (own, found by the tests) | MED | `merge_ckpt()` kept a stale sha result from an earlier merge | `SHA` is reset at the start of every merge |

## Regression tests (`Experiment Notebooks/Build/rc2_unit_tests.py`, 34/34 PASS)

The tests run the exported notebook with `RC2_UNIT=1` (definitions, A0 and prompts only; no arm) on a fake input tree, then inject each fault.

**Evidence gate:**
- missing oracle and absent oracle key;
- sampled ok False / missing / malformed / garbage;
- probe failed / missing;
- 3 of 4 rows; short-token rows; HTTP error;
- a clean 4-row oracle gives CLEAN; a stock 3-row oracle with expected_rows=3 gives CLEAN;
- FP8 with vs_bf16 missing;
- census_long garbage.

**`oracle()`:** a missing row, short rows, and complete rows (reference stored only for complete rows).

**FP8:**
- INCONCLUSIVE_NO_ORACLE, FAIL_HIGH_MARGIN, incomplete-but-EXACT and a sampled failure: 0 levels and 0 long_rows calls;
- PASS: the load tests run.

**host_phase:** a LaunchError raises, no server is started, and the error is recorded.

**Stock:** no Intel/wheelhouse gives FALLBACK_UNAVAILABLE; a boot gate of False or None gives GPU_NOT_EMPTY with no server started.

**TF death:** an OOM line gives TF_OOM; an illegal memory access + Xid gives TF_SERVER_DIED; a segfault gives TF_SERVER_DIED.

**/dev/shm:** a pre-existing `torch_` entry is kept; the boot's `torch_`/`psm_` entries are removed; unknown entries are kept.

**Merge (FAKE off):** a wrong size gives a failed merge; a wrong sha256 is refused by the sha gate; correct sizes and sha pass.

## Flow tests (E23 fake vLLM server)

| Flow | Result |
|---|---|
| Main | A0, A1 DONE CLEAN (oracle complete, 4 rows), A3 DONE CLEAN, fp8_gate PASS |
| No Merlin (part 3 removed, Intel tokenizer from `tokenizer_config.chat_template`, fake wheelhouse) | `prompt_render.source` = intel_fallback; F stock_fn DONE CLEAN (oracle complete, 3 rows) |
| Rerun | 120-task fallback submission |

## Still not testable off-GPU

- The real Merlin `host_phase()` on the Kaggle image: `_verify_products`, `_toolchain` (cc, c++, ninja, Python.h, nvcc, libcudart) and the marker. E23 never called `host_phase()`, so a false refusal here would cost the Merlin arms. This is a Kaggle CPU kit check.
- The real vLLM CLI parse of the final argv.
- The full sha256 of the mounted 124 GB.
- Real CUDA, PLE, PP or MTP correctness.
