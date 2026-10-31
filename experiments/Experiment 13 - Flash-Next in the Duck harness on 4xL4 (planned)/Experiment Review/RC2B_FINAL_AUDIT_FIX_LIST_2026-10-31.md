# E13 RC2b: fix list from the three FINAL GPT audits (2026-10-31)

**Notebook under audit:** 16fbc7ef.

| Audit | Verdict | Blocking items |
|---|---|---|
| **A** | **GO** | MED/LOW notes only |
| **B** | **NO-GO** | 1 HIGH, reproduced by CPU fault injection |
| **C** | **NO-GO** | 2 HIGH, both reproduced |

All three confirm:
- every earlier RC2 blocker is fixed;
- the Merlin TP2×PP2+EP design, the checkpoint staging and the CPU-kit evidence are fine;
- no redesign is needed.

B and C each give GO once their HIGH items and their negative tests pass.

**Status:** H1, H2, H3 APPLIED (user 2026-10-31); M1-M3 not applied. Notebook 12eaa96c; rc2_unit_tests 57/57; flows main (A3 PASS_LONG) / no-Merlin / rerun OK; foysal access 10/10.

## HIGH (must fix before the GPU run)

| # | Audit | Problem (reproduced) | Fix |
|---|---|---|---|
| H1 | B | **No fail-fast after a failed short check.** In A1, a failed probe AND a failed sampled request still run the 40k prefill, the greedy oracle and the teacher-forced A/A. A3 and BF16-load behave the same way | `short_gate_failed(o)`: probes ok (skipped on the fake server, as in `evidence_gate`) and sampled ok and not garbage. It runs right after the short checks in **A1, A3, BF16-load and stock**. On failure it sets `early_stop = SHORT_CORRECTNESS_GATE_FAILED` and returns; `gated()` then gives NO_GO_SHORT_PROBE / NO_GO_SAMPLED_PATH (status never forced to DONE). Also: an incomplete A1 oracle skips the TF A/A (`early_stop = ORACLE_INCOMPLETE`). When the A3 fp8_gate is not PASS, the TF diagnostic is **skipped by default**; an explicit `RC2_TF_ON_FP8_FAIL=1` turns it on |
| H2 | C | **NaN/Inf logprob read as NEAR_TIE; `compare_tf` gives MEASURED on NaN.** `L.compare()` with a NaN reference margin and differing tokens returns NEAR_TIE | (a) **`Stream` layer:** the RC2 `Stream.run` wrapper sets `error = NONFINITE_LOGPROB` when any streamed margin is NaN or Inf. Greedy rows then carry an error, which gives NO_GO_ORACLE_REQUEST_ERROR. (b) **`compare()`:** an RC2 wrapper returns INCONCLUSIVE_NONFINITE for a row pair with any non-finite margin, before the token comparison, so no later branch can overwrite it. (c) **`compare_tf()`:** an RC2 wrapper rejects any non-finite position value (INCONCLUSIVE_NONFINITE) before the numbers are computed. The E23 lab text stays verbatim; the overrides live in the RC2 cells |
| H3 | C (B: as a MED "long_quality") | **Garbage in the 60k/120k rows still gives CLEAN.** `_windows()` sees `steady` and `census_long` only, so `long_rows()` errors and garbage never reach the gate | (a) `long_rows()` checks **every consecutive 2,000-token window** of each row, recording the first bad window and its verdict. The current checks look only at 8k/32k/60k/100k marks. A whole-text `degenerate()`, as proposed, would give false LOW_VARIETY on 100k tokens of healthy reasoning, because unique words fall under the 8 % rule. (b) New `long_quality` per row: PASS_LONG, FAIL_LONG_GARBAGE, FAIL_LONG_REQUEST, INCOMPLETE_LONG (cut by the cap; the forced row not at 60k). (c) `long_gate()` is propagated to the top-level status: a natural row failure gives NO_GO_LONG_GARBAGE / NO_GO_LONG_REQUEST_ERROR; a forced-row failure gives NO_GO_FORCED60K_GARBAGE (labelled synthetic past its first EOS); incomplete gives INCONCLUSIVE_LONG_INCOMPLETE. `correctness` (the short parity verdict) is kept separately, so short PASS and long PASS are never merged |

## MED / LOW (audit A; cheap, proposed together with the HIGH fixes)

| # | Item | Fix |
|---|---|---|
| M1 | Notebook vs export differ in blank lines | The builder asserts that each notebook code cell equals the exported string (byte-exact equivalence test) |
| M2 | `help_check` proves nothing on CPU | On LAUNCH_FAILED / SERVER_DIED, also record the argparse rejection lines (`unrecognized arguments`, `error: argument`, `invalid choice`) next to the existing `argv_tail` and `log_tail`. It stays record-only |
| M3 | A1 vs A3 same short answer | Record `probes_match_bf16`: A3's 7×6 / 2+2 / 9−3 answers against A1's (record-only; both must be right anyway) |
| M4 | MTP K3 in both boots: CLEAN = MTP3 parity | No change now. It stays as written in `R.correctness_meaning`. An MTP-off short boot is a separate later test; every auditor calls it a non-blocker |
| L1 | Long rows run concurrently | Already labelled `concurrency: 2`. No change |

## Tests to add (`rc2_unit_tests.py`)

**H1:**
- probe failure: 0 prefill, 0 oracle, 0 TF calls;
- sampled failure: the same;
- incomplete A1 oracle: TF skipped;
- FP8 INCONCLUSIVE: 0 ladder, 0 long rows, TF skipped by default;
- all short gates pass: the order is unchanged (probes → sampled → prefill → oracle → census_end → TF).

**H2:**
- NaN margin and Inf margin in `compare`: never NEAR_TIE or EXACT;
- NaN teacher-forced logprob: never MEASURED;
- a valid near-tie is still NEAR_TIE;
- a `Stream` with a NaN margin gets error NONFINITE_LOGPROB.

**H3:**
- a clean 60k row gives PASS_LONG;
- garbage at 8k (natural) gives NO_GO_LONG_GARBAGE;
- garbage at 60k (forced) gives NO_GO_FORCED60K_GARBAGE;
- a request failure gives NO_GO_LONG_REQUEST_ERROR;
- a cap cut gives INCONCLUSIVE_LONG_INCOMPLETE;
- 100k tokens of varied text give no false garbage.

**M2:** a log tail with `unrecognized arguments: --foo` is captured.

**Then:**
1. Rebuild and record the new sha.
2. Run all units (34 old + about 20 new) and the three local flows.
3. A new CPU kit is **not needed**: inputs, the merge, the install, `build` and `host_phase` are unchanged, and only gating and recording code changes. If wanted, it can run in parallel.
