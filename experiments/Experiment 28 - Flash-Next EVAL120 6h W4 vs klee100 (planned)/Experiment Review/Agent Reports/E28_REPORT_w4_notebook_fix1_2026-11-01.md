# E28 W4 notebook agent: fix 1 report, verbatim

## E28 W4: your three follow-ups are in; new notebook sha 809cb2ad…

I applied all three fixes, rebuilt, and ran the units locally: 77 / 77 PASS, 1 skip (the deadman killpg path is POSIX-only; the Linux kit runs it). The 4 kit folders and the audit zip are refreshed. Nothing is pushed.

**Notebook:** `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Q27-XHIGH-E28_W4_EVAL120_6H.ipynb`
- sha256 **`809cb2adf6ac269a42589a30bed079fa4a17781edc381a3792df2070924eb1e8`**, 391,281 B (under 400 kB), 9 cells, LF, no CR.
- Embedded scheduler: `47ab9874…`; arc2_duck.py post sha `da2c8682…`; API file sha `421ab22c…`.
- N1-N4 are applied by label, with admission "slots".
- Review diff vs NB6Q: +285 / −12 lines.

## The three fixes (line numbers in `Build/e28w4_code_export.py`)
1. **A scheduler failure in a commit now stops the run.** In CELL 3 (L564-L609):
   - After any apply, verify or env failure, the original arc2_duck.py is restored and the env is cleared.
   - **Commit:** it raises `RuntimeError("NO_GO_E28_SCHED: …")` before any boot. CELL 0's fallback submission stays.
   - **Hidden rerun:** it falls back to NB6Q's scheduler, so the submission is still written.
   - New unit test (T6): a commit-mode failure raises NO_GO_E28_SCHED, with the file restored and the env unset. The rerun-mode failure cases still fall back.
2. **The B2 request timeout is set explicitly to 12,000 s.**
   - New profile key `b2_request_timeout=12000` in E28_LEVERS. CELL 3 L585 sets `E28_B2_REQUEST_TIMEOUT=12000` in the notebook-process env.
   - The comparison against the module's `server_env()` ignores that one key.
   - T1 stays 7,200 s through `E12_DUCK_REQUEST_TIMEOUT`.
   - The unit test now asserts `E28_B2_REQUEST_TIMEOUT` = 12000.
   - The e2e now asserts the notebook env is 7200 / 12000 and the scheduler config is request_timeout 7200 / b2_request_timeout 12000. The harness value is 12000 too, no longer 10800.
3. **A second GPU push folder for esfersami50.** `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__esfersami50/`
   - id `esfersami50/q27-xhigh-e28-w4-eval120-6h`, the same notebook (sha 809cb2ad…), kernel-metadata.json sha `4367ee6c…`.
   - Every other field equals the foysal folder: L4, GPU, private, 7 model sources, dataset `foysalemonshanto/q27-e15b-payload`, the same image.
   - A local T7 check verifies this.
   - The foysal folder is kept: `…__foysalemonshanto/`, metadata `e3d0e671…`, same notebook.

## Files changed (sha256, bytes)
- `Build/build_e28_w4.py`: cad61c52cb1d000cbb8bda77837cea815fcd9607e2b555e33a0a07a4cd45d819 (45,676)
- `Build/build_e28_w4_cpu_kit.py`: 4834062afd7d5488ba0fa099b67eeae2ceb51b553db30b781ead61101c6fbafb (22,937)
- `Build/e28w4_unit_tests.py`: a7dc421f5cc9a329787895dd4accba6349402f034bdc00f1e17ab0d6863070e9 (47,323)
- `Build/e28w4_build_facts.json`: b778f4386d9ac9deaf051742914278253d3aedc18d9b0c4f591c1788e0d4d70f (4,630)
- `Build/e28w4_review_diff_vs_nb6q.diff`: e66664fdd692db1fd74c2172d7d0ba1101f6762674ff0340b02e48b5003d5c11 (51,199)
- `Build/e28w4_code_export.py`: 0b70af651f1ef244221b3b4ea19220326e4e2603f2f3dcd3921a133d25ebed88 (363,195)
- Unchanged: `e28w4_lib.py` 5dcbd0cf… (15,660), `e28w4_unpack.py` 0b6c6703…, `e28w4_kit_inject.py` bb618fae…, `e28w4_nb4t_log_fixture.json` 36db3b19….

## Tests (local)
- `e28w4_unit_tests.py`: **77 / 77 PASS**, 1 POSIX-only skip.
- NB6Q's `nb6q_unit_tests.py` sections S2 + S4 on this notebook: **31 / 31 PASS**, 1 POSIX-only skip.

## Kit folders (refreshed, on foysal, not pushed), in `D:/ARC AGI 2/_scratch/push_e25/`

| Folder | Size | sha256 (prefix) |
|---|---|---|
| `q27-e28-w4-kit-units__foysalemonshanto` | 349,843 B | 002dfdcc… |
| `q27-e28-w4-kit-e2e-rerun__foysalemonshanto` | 311,578 B | 28a868e8… |
| `q27-e28-w4-kit-e2e-commit__foysalemonshanto` | 311,569 B | bf8a25ff… |
| `q27-e28-w4-kit-real__foysalemonshanto` | 393,568 B | 70dcc6aa… |

The kits are only ready, none has run on Kaggle yet. In the e2e commit kit, a scheduler failure would now raise in CELL 3; the "every code cell ran" check would report it.

## Audit zip
- `…/Experiment Review/Audit Zips/E28_W4_audit_FLAT_2026-11-01.zip`, 30 files, sha256 `c48bca7d5caf60169aeb0f06d20c1d73e6c8ecc70364fbddfd3350a548c516c5`.
- `00_GPT_PROMPT.md` is updated for slots admission, the explicit B2 timeout, commit NO_GO vs rerun fallback, and both push folders.

## Open risks (from the last report)
1. **C12 + I8 together, never measured.** CAPS16 captures go up to 64 (NB4T used 32) and max-model-len is 262144, so a boot OOM is possible. The ladder is 0.75 → 0.5 → 0 GiB.
2. **Slots + 12 workers on a pool of about 197 blocks.** Preemptions are likely above a mean context of about 60k and with 200k B2s. They are recorded at run level and per attempt, not prevented.
3. **200k context on Flash-Next has not been validated.**
4. **`--language-model-only` is new against NB6Q.** It comes from NB4T's measured flags.
5. **Long T1 requests.** A 115,904-token request at about 25 tok/s per stream takes about 77 min, close to the 7,200-s timeout.
6. **The NB5 QUALITY regression is no longer in the units kit**, because of the 400-kB limit.
7. **The watchdog relaunch uses the RC2 pin.** This is NB6 behaviour, unchanged.
8. **The kits have not run on Kaggle yet.**

