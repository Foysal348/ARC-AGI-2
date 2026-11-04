# E28 W4 notebook agent: kits v1 analysis + v2 fixes, verbatim

## E28 W4 kits v2: every v1 failure was a kit defect; the notebook is unchanged

All four kit failures from v1 came from the kit itself, not the notebook. The notebook stays **`809cb2adf6ac269a42589a30bed079fa4a17781edc381a3792df2070924eb1e8`** (391,281 B), so the sami GPU folder (same sha) needed no refresh. I fixed the kit, rebuilt the e2e-rerun, e2e-commit and real kits, and refreshed the audit zip. The units kit did not change. Nothing is pushed; the v2 kits have not run yet.

I read all the logs and reports in `D:/ARC AGI 2/_scratch/e28_w4_kits_v1/`.

## Each failure: cause and fix

**1. e2e-commit / e2e-rerun / rerun-retry: "capacity of the booted try"**
- Cause: the fake server prints the max-concurrency factor with 2 decimals, as vLLM does. At `--max-model-len 262144` that factor is small (197 / 69 = 2.86), so F_log reads 4.881 at pin 0.75 and 5.084 at pin 0.5. The old check required |F_log − 5| < 0.06 and F_eff == 5. At pin 0.5 the notebook's own rule ceil(F_log − 0.05) correctly gives 6.
- Real-run effect: none that matters. F can be over-estimated by 1 at most (conservative), and in slots mode F only feeds the C-7 check.
- Fix (V2-1, `Build/e28w4_kit_inject.py` L122-127): the check now accepts F_log within 0.15 of 5 and requires F_eff == ceil(F_log − 0.05).

**2. e2e (all three modes): "E28_SCHED never in a server env"**
- Cause: the server process inherits the notebook's env, like every NB5 env, so `E28_SCHED=1` reaches it. vLLM never reads it. Asserting it was absent was my wrong expectation.
- Fix (V2-5, `Build/build_e28_w4_cpu_kit.py` L122-125): the inherited env is recorded, not asserted absent.

**3. real: `vllm_argparse_fields_all_plans` and `e28_specconfig_accepts_draft_lm_head_quant_int8`** (one root cause)
- This is **not** the same "torch circular import on CPU" artifact NB6Q saw. The NB6 kit's argparse probe was written as `K/argparse.py`, which shadowed the stdlib `argparse`.
  - The argparse probe itself failed: torch's own `import argparse` re-ran the probe, giving "partially initialized module torch … no attribute nn".
  - My SpeculativeConfig probe in the same folder hit the same shadow twice:
    - plain CPU attempt: `IndexError` (the shadow read `sys.argv[1]`);
    - fake-L4 attempt: `GradBucket already defined`.
  - This is the "never name a probe after a stdlib module" lesson. NB6Q's kit had this defect too.
- Fix (V2-2, `Build/e28w4_kit_inject.py` L135-167):
  - The probe is now named `vllm_argparse_probe`, and any old `argparse.py(c)` in K is removed first.
  - If the plain CPU import still fails, the kit falls back to E27's fake-L4 `PLATFORM` (+ the libcuda stub, via a new `e28_l4_env()` helper). The SpeculativeConfig probe uses the same helper.

**4. real: `rerun_plans_equal_commit_plans`**
- Same artifact as NB6Q: the argv differ only by port. I confirmed this in the report: the only differences are at argv positions 11 and 66, values 8100 vs 8105 (and 8110 / 8115, 8120 / 8125). There are two `--port` flags, Merlin's own and the notebook's, and the old strip only replaced the first.
- Fix (V2-3, `Build/e28w4_kit_inject.py` L128-134): every `--port` value is stripped.

**5. real: `submission_commit_strict_full_schema_identity`**
- Cause: the NB6 kit expected profile NB6COMMIT. The report shows strict schema True, identity True, profile E28EVAL120, so the submission itself is fine.
- Fix (V2-4, `Build/e28w4_kit_inject.py` L168-169): the check expects E28EVAL120.

In the same real-kit run, the scheduler round trip, the hand-over and the checkpoint max positions (262144) passed: both namespaces APPLIED (post sha `da2c8682…`), verify → revert → pre sha → re-apply → post sha.

## Tests (local)
- `e28w4_unit_tests.py`: 77 / 77 PASS, 1 POSIX-only skip.
- `nb6q_unit_tests.py` S2 + S4 on the notebook: 31 / 31 PASS.

## Changed files
- `Build/build_e28_w4_cpu_kit.py`: sha `6c3893848c527db25156edbcedb83d29ff61b793cac23c69da9879cd249a39cf`
- `Build/e28w4_kit_inject.py`: sha `499aff76c1bb01eeb156d320e978cd62e2bcaa6eb1ab84b615a52d0601cb6374`
- The notebook, the GPU builder, the lib and the unit tests are unchanged.

## Kit folders v2 (on foysal, in `D:/ARC AGI 2/_scratch/push_e25/`, not pushed)

| Kit folder | Bytes | sha256 |
|---|---|---|
| `q27-e28-w4-kit-units__foysalemonshanto` | 349,843 | `002dfdcc2e5977a0b1f3b2e05cbe284f48067e0e667114686a4c7b7b62630492` (same as v1, ALL OK) |
| `q27-e28-w4-kit-e2e-rerun__foysalemonshanto` | 311,588 | `94570e0cc5f751601b6c5498d8625aaef77c52880771fde00876f2aaf773b56f` |
| `q27-e28-w4-kit-e2e-commit__foysalemonshanto` | 311,579 | `698a5717d37ca7d0b5b78dc7f61f27dc430d8dbc996250a75d2cff456301f774` |
| `q27-e28-w4-kit-real__foysalemonshanto` | 396,122 | `c84f5364d0b8f4990cf6f965e577da95011113bdb3a2cad82699ed91586b1800` (under 400 kB) |

## GPU push folders (unchanged, notebook `809cb2ad…`)
- `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__esfersami50/`
- `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__foysalemonshanto/`

## Audit zip
- `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Review/Audit Zips/E28_W4_audit_FLAT_2026-11-01.zip`
- 34 files, sha `078a76199a9554600f5221a1647cf04266a9f656625f64da0433e01ddf944f45`.
- It now includes the four v1 kit reports (`kits_v1_*_report.json`), and `00_GPT_PROMPT.md` has a v1 / v2 section asking the auditor to confirm that no fix hides a real defect.

## Open risks (unchanged)
1. C12 + I8 together have never been measured; CAPS16 captures up to 64 and max-model-len 262144 add boot OOM risk (ladder 0.75 / 0.5 / 0 GiB).
2. Slots with 12 workers on a pool of about 197 blocks will likely preempt above a mean context of about 60k; this is recorded, not prevented.
3. 200k context on Flash-Next is untested.
4. `--language-model-only` is new against NB6Q.
5. A 115,904-token request at about 25 tok/s takes about 77 min against the 7,200 s timeout.
6. The NB5 QUALITY regression is no longer in the units kit.
7. The watchdog relaunch uses the RC2 pin.

