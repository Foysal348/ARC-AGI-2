# 34 — Validation of the GPT R2 reports (Notes/32, 33) and whether the notebook needs an update (Claude, 2026-10-24)

Both reports reviewed the zip `GPT_VALIDATE_E14A_R2_2026-10-24.zip`. That zip holds the **R2** notebook (2298d827) and was built **before** E14S (Notes/31). Each claim below was checked against the raw files.

## Claim-by-claim

| # | GPT claim | Check | Verdict | Notebook change? |
|---|---|---|---|---|
| 1 | **NO-GO (both reports):** the rerun runs W1 (width 8, graphs up to 12), not width 6 + graphs13 | True for R2. **Already fixed in E14S** (Notes/31): `e11_flags` sets width 6 + `--cuda-graph-bs-decode 1..13` and drops max-bs whenever no measurement runs. The flags were diffed against R2's real width6 launch (identical). The harness asserts the argv in rerun mode (13/13 local, Kaggle CPU) | Correct, already done | No |
| 2 | Width 6 gain is +5.23 % vs W1 (505); +6.26 % is vs graphs13 (500.14) | 531.43 / 505.0 = 1.052 | **Correct: my "+6.3 %" against W1 was wrong** | No |
| 3 | graphs13 raised no tok/s: short −0.96 %, long 449.84 → 436.89 (−2.9 %); adopted on a per-step criterion | Confirmed in `e14_results.json` | Correct | Decision (below) |
| 4 | Width 6 at long context is unmeasured; only one launch per arm, no A/B/A; the gate is short (256 tokens) | Confirmed: width arms ran only the short window (~5.9k context) | Correct | Decision (below) |
| 5 | Triton kernels are loaded after serving starts, with 0.89 GiB free (width6 log) | True, but **not width-6 specific**: W1 try0 shows 0.96 GiB, width4 0.94, and the 25-min Duck run on graphs13 0.79-0.80 GiB with **no OOM**. graphs13 / width 6 leave ~0.1-0.2 GiB less headroom than W1 | Partly correct | Decision (below) |
| 6 | JIT tar IndexError: `_q.stdout.strip().splitlines()[-1]` on empty stdout (rc / stderr not kept) | Confirmed at `build_e14.py` CELL 7b | Correct | No: E14S skips the tar outside measurements. The rc/stderr fix goes into E15 |
| 7 | nodelay crash: `stage_sampling_params` at line 278 (warning at 273), sampling buffer 12 vs batch 13 | Confirmed in `sglang_server_e14_nodelay.log` | Correct | No (nodelay stays off) |
| 8 | Latch 114/114 at 11; nodelay 30 of 83 samples at 12; real Duck cost 0-10 %, not measured | Confirmed | Correct | No |
| 9 | Profiler: communication 59.03 ms of 116.04 (TP0 trace re-summed); busy 112.96 vs wall 116.66; NCCL time includes sync waits | Confirmed | Correct | No |
| 10 | "First server READY ≈ 1,074 s" (report 33) | R2 READY was 1,058 s, but **301 s of that was the NCCL bench (254 → 555 s), which only runs in measurements**. E14S launches at ~255 s, so READY ≈ 760 s. The E14S give-up is 1,140 s | The number is right for R2, wrong for E14S | No |
| 11 | ~287 agent tok/s in the Duck window (430,062 / 1,500 s) → ~11.8M tokens in 11.5 h ≈ 49k / task; 60k for all would need ~351 tok/s | Arithmetic confirmed | Correct (a scenario) | No (coverage = the EVAL120 G4B run) |
| 12 | Attempt 2 / branch 1 added 0 (pass2 0; b1 10,976 tokens, 0 new predictions); 12/17 branches never called check(); 3 of the 6 correct outputs came from check-only harvesting | Confirmed in `e12_run_report.json` (pass2_hits 0; branches) | Correct; **the largest accuracy lever** | Not tonight: a solver change, next Duck revision |
| 13 | Rev 5 LOW 10/19 > rev 6g 6/19 | 10/19 was a **150-min** window (EVAL12R5_LOW) vs 25 min here | Not comparable | No |
| 14 | Hidden failure-path test with 240 synthetic tasks, missing model / draft / payload, launch timeout | Our rerun harness uses 5 tasks; the CELL 0 identity fallback is unchanged from rev 5 (LB 17.50 scored through this path) | Useful, not a blocker | Optional harness extension |
| 15 | MTP head in the RedHat INT4 checkpoint: check `config.json` + the safetensors index offline before E15 | Speed-review notes say `model_mtp.safetensors` ~0.85 GB, `mtp.*` ignored by quantization (BF16). Not verified on the Kaggle copy | Correct | E15 preflight |
| 16 | E15 order: W8 reference → MTP 3/1/4 → MTP 2/1/3 → ReplaySSM → C14/16 → chunked 2048 → NCCL (optional) → 2×TP2 | Agrees with Notes/29 except that NCCL moves to the end | Adopt | E15 |
| 17 | FP8 KV scales absent (defaults 1.0); "Error in sitecustomize" Traceback strings are not real exceptions | Both are present in every log, including W1 runs that scored on the LB | Correct; not new | No |

## Does the E14S notebook (3dde5cbd, committing on hosen42) need an update?

**No mandatory update.**
- Every blocker the reports found in R2 (items 1 and 6) is already handled in E14S.

**One open decision: width 6 + graphs13 or W8.**
- **For W6 + g13:**
  - +5.2 % short-context throughput;
  - gate PASS;
  - the R2 Duck run on graphs13 ran 25 min at 12 streams with no OOM.
- **Against:**
  - long context is unmeasured for width 6;
  - graphs13 gave no gain and leaves about 0.1-0.2 GiB less free memory;
  - DFlash correctness issues are public (#36548, #38009).
- **Evidence coming now:** the running E14S commit is the first width-6 Duck run (8 min, 12 streams, growing contexts).
  - If it finishes with no OOM, no exception, normal accept and normal Duck progress → submit as it is.
  - Otherwise → rebuild with W8 (width 8, graphs up to 12, i.e. the R2 / rev 5 engine): about 10 min of build + a 36-min commit.

**Later (not this notebook):**
- the solver: first check() earlier, branch-1 diversity;
- the JIT probe rc/stderr fix;
- the MTP-head preflight and E15.
