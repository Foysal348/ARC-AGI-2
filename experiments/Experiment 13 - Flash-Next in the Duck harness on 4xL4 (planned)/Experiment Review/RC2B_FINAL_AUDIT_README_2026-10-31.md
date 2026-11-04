# E13 RC2b: FINAL pre-GPU audit (2026-10-31)

**Who this is for:** an independent auditor (GPT), for the last check before the first GPU commit of this notebook.

**Your job:** check that every finding of the three earlier audits is fixed correctly, and look for anything new that would crash the run, waste the about 4 h of 4×L4, or give a wrong verdict.

**Answer format:**
- The first line is exactly **GO** or **NO-GO**.
- **If you find no HIGH problem, say GO.** MEDIUM/LOW items may follow as notes; they do not block.
- For each finding: severity, file:line, failure scenario, and the fix.
- Mark guesses as GUESS. Say "verified" only for what you checked in the attached files.

## What changed since your last review (notebook 2fb12343 → 16fbc7ef)

The full table is in `docs/RC2B_FIX_LIST_2026-10-31.md`. In short:

| Your finding | Fix (in `notebook/rc2_code_export.py`) |
|---|---|
| Missing oracle or failed sampled check gave CLEAN | `evidence_gate()` runs before E23's `oracle_gate`. Probes, the sampled check and a COMPLETE oracle are mandatory. `oracle()` records `row_count`, `expected_rows`, per-row `tokens` and `complete` (all rows, no error, exactly NTOK tokens); a reference is stored only when complete. The stock oracle is parameterized to 3 rows |
| FP8 INCONCLUSIVE still ran the load tests | `fp8_gate()` returns FAIL_SAMPLED / FAIL_PROBES / FAIL / PASS / INCONCLUSIVE. **Only PASS** runs the ladder, census and long rows; otherwise the status is NO_GO_FP8_KV or INCONCLUSIVE_FP8_KV |
| `host_phase()` failure ignored | It is recorded and raised as MERLIN_HOST_PHASE_FAILED before any server starts |
| Any TF server death was TF_OOM | TF_OOM requires an OOM line AND no Xid / device fault; otherwise TF_SERVER_DIED, and A3 does not start |
| Stock fallback crashed in CELL3; its boot gate was not enforced | The Intel tokenizer/template is used when the merge fails, and the Merlin hash assert applies only on the Merlin path. Both boot gates require `ok is True`. A missing Intel checkpoint or wheelhouse gives FALLBACK_UNAVAILABLE |
| MED items | `census_long` is gated; `census_end` runs before TF; the long rows are labelled concurrency 2 (cap 3,000 s); `/dev/shm` cleanup touches only this boot's entries; host memory is recorded before every boot; BF16-load is DIAGNOSTIC_ONLY; the CLEAN meaning is BF16_MTP3/FP8_MTP3 parity; the stale sha result is reset per merge |
| (new, 16fbc7ef) | `RC2_CPU_KIT` env skips the 4-GPU preflight. It is set only in the CPU kit; the GPU run does not set it |

## Evidence

**1. Local regression `notebook/rc2_unit_tests.py`: 34/34 PASS.** Every fault injection you asked for:
- missing, absent, partial (3/4), short-token and HTTP-error oracles;
- sampled ok False, missing, malformed and garbage;
- probe failed or missing;
- census_long garbage;
- FP8 INCONCLUSIVE / FAIL_HIGH_MARGIN / incomplete / sampled failure: 0 levels and 0 long_rows calls; PASS runs them;
- host_phase LaunchError: raises and starts no server;
- stock without Intel/wheelhouse, and boot gate False or None;
- TF death by OOM / illegal memory access + Xid / segfault;
- /dev/shm ownership;
- merge with a wrong size, a wrong sha and correct files.

**2. Local flows (E23 fake vLLM server):**

| Flow | Result |
|---|---|
| Main | A1 + A3 CLEAN, oracle complete 4/4, FP8 PASS |
| No-Merlin | intel_fallback, stock CLEAN, 3/3 |
| Rerun | 120-task submission |

**3. Kaggle CPU kit `foysalemonshanto/q27-e13-rc2-cpu-kit` v1** (pinned image, Py3.12, real mounts): **ALL OK**, raw outputs in `cpu_kit_v1/`.
- inputs at the exact paths;
- merge: 38 files, the index's 27, every real size matches;
- full sha256 of about 124 GB in 275 s, 0 mismatches;
- tokenizer/template hashes;
- Merlin offline install, 394 s;
- **`build()` + `host_phase()` = ok for BF16 and FP8**;
- no server started;
- units 34/34 on Py3.12;
- the 3 fake flows.

`vllm serve --help=all` on CPU reads every flag as missing, even `--port`, so that CPU output is not an argument list. It is reported only and never gates.

**Not testable without a GPU:** real CUDA, PLE, PP, MTP correctness and speed. That is what this GPU run measures.

## Files

| Path | What |
|---|---|
| `notebook/Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb` | The notebook to run, sha256 16fbc7ef…, 7 cells |
| `notebook/rc2_code_export.py` | Same cells as one .py (cite lines from here) |
| `notebook/build_e13_rc2.py`, `notebook/rc2_unit_tests.py`, `notebook/build_e13_rc2_cpu_kit.py` | Builder, tests, kit builder |
| `push/kernel-metadata.json`, `push/model_files_api_2026-10-31.txt` | GPU run inputs (foysalemonshanto) and API file listing |
| `e23/e23_lab.py`, `merlin/launch.py`, `merlin/registry.py`, `merlin/flash-next.json` | Embedded lab module and the Merlin launcher |
| `docs/` | RC2B fix list, RC2 build notes, past-experiment review, Merlin synthesis, RC1 GPU result |
| `cpu_kit_v1/` | Kaggle CPU kit report and logs |
| `previous_audit/README_rc2.md` | The brief of your previous round (9 questions; still apply) |
