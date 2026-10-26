# EMS / VMS rev 2g: disposition of the rev 2f audits (2026-10-26)

**Inputs.**
- GPT strict audit 1 (pasted, R1-R6).
- GPT audit 2: `README_EMS_VMS_REV2F_INDEPENDENT_AUDIT_2026-10-26.md` and `ENGINE_MATRIX_REV2F_STRICT_INDEPENDENT_AUDIT_BN.pdf` (R2F-01..10).
- Four read-only agents:
  1. SGLang items of the pre-GPU audit vs EMS;
  2. vLLM items vs VMS;
  3. the rev 2e disposition vs the code;
  4. a regression review of the split.

**Final builds.** EMS SAFE `9ed2b035` (206 kB), EMS RISKY `5b8b91f2` (206 kB), VMS `6c2a39c5` (128 kB).

**Patch scripts.** All are assert-guarded; each replacement must match exactly once.
- `_scratch/ems/patch_ems_rev2g.py`
- `_scratch/vms/patch_vms_rev2g.py`
- `_scratch/vms/patch_vms_rev2g_b.py`
- `_scratch/ems/patch_rev2g_c.py`
- `_scratch/vms/patch_harness_rev2g.py`

**Rule.** Every change keeps an arm's speed data. It only stops adoption when the evidence is missing. Every claim was checked in the code first.

## Fail-open adoption (an invalid arm could be adopted)

| ID | Finding | Fix |
|---|---|---|
| R1 / R2F-01 | VMS: missing XL and missing canary counted as success; DP 2 x TP2 could be adoptable | `vms_classify`: XL gate must be PASS whenever XL prompts exist. The canary must be present, at the arm's level, with 0 errors and 0 leaks. DP rows are `diagnostic_only`. A PCIe-IPC arm without its init line is `backend_unverified`. |
| R6 / R2F-02 | EMS: XL None accepted; oracle canary errors ignored | `_gated`: XL PASS required when `_X` exists. The oracle canary must be a dict with errors 0, leaks 0 and an idle server. A PCIe arm needs `workspace_built`. |
| R2F-10 | VMS oracle promoted without a complete XL A/A or with canary errors | The oracle predicate now requires a complete XL A/A and a clean canary. |
| R2 / R2F-03 | VMS healthy window with a stream that never got a token | `vms_window_healthy` requires: `no_first_token == 0`, joined threads, idle after, 0 preemptions, `finished_empty == 0`. |
| R3 / R2F-06 | The digit-share grid exemption let `"0"*250` through | Structural rule in both builders: only digits, brackets, commas and whitespace, and every row has at most 30 cells (the ARC maximum). Tested: degenerate inputs (`"0"*250`, `"0"*40+"!"*60+"0"*60`, `"0 0 0..."`x200, `"!"*250`, empty) are flagged; a 30x30 grid, a JSON 30x30 grid, rows without separators, prose with a grid and plain prose are not. Agent 3 measured 0 of 210 real greedy rows and 0 of 77 real thinking traces flagged under the old rule. The new rule only narrows the exemption. |
| R4 | INT4 MTP static check: packed MTP covered by the ignore list passed | Packed modules matched by the ignore list (vLLM would build them dense) fail. Tests: dense + ignored = ok; dense, not ignored = FAIL; packed + ignored = FAIL; packed, not ignored = ok. |
| R5 | `vms_judge` PASS on 1 row of 6 | `vms_judge` and `vms_aa` require the planned row counts (6 serial; all-at-once = every prompt). |
| agent 3 B5 | EMS tail check judged the whole row (a clean head masked a degenerate tail) | `_post_div_tails`: only the continuation after the first divergence from the matching oracle row is judged. |
| agent 3 B6 | EMS oracle A/A had no degenerate check; an incomplete oracle XL was still used | The oracle needs 0 degenerate rows. The oracle XL counts only when complete; otherwise `EMSR["oracle_xl"] = NO_ORACLE_XL (nothing adoptable)`. |
| agent 3 B2 | VMS silent-empty stream (`[DONE]` with 0 tokens) not gated | `finished_empty` recorded; the healthy window requires 0. |
| agent 3 B4 / agent 1 A1 | EMS canary at 12, not at the arm's level | The canary runs at `max(12, level)`. The 12-prompt gate stays; the canary covers cross-request bleed at 16-24. |
| agent 2 #3 | `wait_ready` trusted any `/health` 200 (an old process on the port) | A launch counts only when this server's own log has the `GPU KV cache size` line. |
| agent 2 V1-d | Reference's warm (prefix-hit) rows never judged against the oracle | Both cold and warm rows are judged; the worse verdict counts. |

## A valid arm could be wrongly rejected

| ID | Finding | Fix |
|---|---|---|
| agent 3 A1 (real E14A data: 1 of 12 streams ended early, share 0.881 at 505 tok/s) | EMS window streams stopped on EOS, so the running share fell below 0.80. Arms at about 900 tok/s and above would be rejected. | A notebook-local `Gen` subclass sets `ignore_eos` for window streams (`max_new >= 16384`). `max_new` goes from 4096 to 16384. Greedy, probe and canary rows keep EOS. The shared E14 module (Kaggle payload, also used by E15P) is untouched. |
| GPT note | VMS: a stream finishing 8,192 tokens inside the window lowers `running_mean` | Window streams get `max_tokens = min(32768, 106496 - prompt - 512)`. |
| agent 3 A4 | The 1,024-token probe ran after the optional XL window, so a slow XL could cost the probe | The probe runs before the XL window. |
| agent 3 A5 | A whole-log `Preempt` scan blocked adoption after one preemption in the diagnostic XL window | `Preempt` is no longer in `log_errors`; the 8k / 40k windows require 0 preemptions (per-window delta). |
| kit v4 ems_all | `target_only` XL skipped on the phase cap | Reference and target window phases 11 -> 14 min; arm cap 12 -> 14 min. XL is now required for adoption, so a skip would be a false reject. |
| agent 4 #4 | The VMS wait-loop exited at once when one stream died | It exits only when even `level - 1` is unreachable. |

## Crash isolation and robustness

| ID | Finding | Fix |
|---|---|---|
| R2F-05, agents N1 / N2 | The best-MTP-at-20 call and both `ems_tp2` calls were outside the per-arm try/except. A crash there skipped the remaining arms and the adoption pass. | `_safe()` around every arm call. `_ems_adopt_all()` runs at the end and from the outer except. |
| agents N3 | After the split, the best MTP at 20 could never pick a RISKY MTP (#45151 / PCIe / NTN) | It runs per part, after the whole plan. |
| agent 4 #3 / agent 2 #2 | A half-started server could keep the GPUs after a crash (EMS and VMS) | A crash inside `wait_ready` kills that server, then re-raises. |
| agent 4 #5 | VMS crash in post-processing lost the measured windows | The arm record is stored as soon as it is created. |
| R2F-08 | Unbounded TP2 worker joins | Joins are bounded by the arm deadline. Still-alive workers mean both servers are killed and the result is `TP2_WORKER_DEADLINE`. |
| N4 / N5 / N6 / N7 / N8 | Small defects | FP8 prefetch runs only in RISKY. Topology is recorded in SAFE. `EMS_PART` is validated. The #45151 locator runs with the venv env. `custom_allreduce_disabled` is read from the flags. |

## Evidence, labels, provenance

| ID | Finding | Fix |
|---|---|---|
| R2F-04 | The zip had only the old rev 2c CPU-kit reports | Kit v6 runs on the final three shas (`esfersami50` account, 4 kernels in parallel). Its reports go into the rev 2g zip. |
| R2F-07 | Canary with 0 own hits is non-informative | Records `informative` and a `classification` (FAIL / INCOMPLETE / INCONCLUSIVE / PASS_NO_OBSERVED_LEAK). It is not a gate; a paraphrasing model is not a leak. |
| GPT R-4 | BOUNDARY policy | Three fields kept separate: `candidate_for_arc_validation`, `certified_exact`, and `production_approved` (always False here; it needs the 60k+ generation and ARC task gates). |
| R2F-09 | `end_to_end_tok_s` wording | Labelled as a submission-to-window-end rate, not completed-request throughput. |
| checklist 9 (agent 3) | No pip freeze; no prompt provenance | `pip freeze` of each venv and a sha per prompt are saved. `topo -p2p p` is recorded next to `-p2p r`. Per-engine running max is recorded (DP skew). |
| agent 2 | `top_k=-1` at temperature 0 could not be verified in the 0.32.0 snapshot | Greedy requests send no `top_k`. |
| docs | Stale text: VMS docstring BOUNDARY, `200 min`, `--linear-backend marlin`, TP2 skip label | Fixed. The TP2 skip now uses the `NOT_TESTED_BUDGET` prefix. |

## Not changed, with the reason

- **B7: BOUNDARY accepts any number of near-tie divergences.**
  - This is the documented tolerance. Requiring exact identity would reject every arm on batch-size numerics, without evidence of a wrong answer.
  - `production_approved` stays False until the ARC task test passes.
- **The EMS 12-prompt concurrent gate is not run at 16-24 streams.** The canary covers cross-request bleed at the arm's level. More gate prompts would cost GPU time on every arm.
- **Not covered by this matrix** (open for production, as before):
  - server restart x3;
  - forced all-reject / partial-accept;
  - streaming vs assembled output;
  - thinking-off parity;
  - more than 100k positions;
  - 60k+ generated tokens;
  - an MTP-width x prefix-off cross;
  - a peak HBM sample.
- **U8 (#43029):** cannot be checked offline; it stays open.

## Run cost (three notebooks)

- **Wall time:** EMS SAFE, EMS RISKY and VMS each have a 260-min cap. Each repeats install, oracle and reference (about 25-35 min).
- **Quota:** about 9-9.5 h of GPU in total, about 18-19 h of quota.
