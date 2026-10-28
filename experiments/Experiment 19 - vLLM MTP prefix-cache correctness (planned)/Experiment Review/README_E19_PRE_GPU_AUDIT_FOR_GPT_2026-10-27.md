# E19 pre-GPU audit request: does vLLM 0.19's MTP + prefix caching (+ async scheduling, fp8 KV) silently corrupt our best notebook's outputs?

**What we want from you:** an independent audit before we spend about 3 GPU-hours (3 Kaggle accounts × 60 min).
1. Will this experiment actually expose the suspected bugs on our exact stack? Or can it "PASS" without having tested them, which is how E17 failed?
2. Are the verdict rules sound? Could they produce false FAILs (normal MTP / batch numeric noise) or false PASSes?
3. Any BLOCKER / HIGH bug in the notebooks, the launcher, the client or the judge.

Give a GO / NO-GO with exact fixes. Mark anything you cannot verify as UNVERIFIED.

## Why E19 exists (context from your three previous audits and our follow-up checks, 2026-10-27)

**Our best notebook.** S1-60K (LB 34.58 and 31.94) runs Qwen3.8-27B with this setup:
- vLLM 0.19.0, in-process `LLM()`;
- FP8 block-128 weights, fp8_e4m3 KV with scale 1.0 (the checkpoint has no KV scales);
- MTP num_speculative_tokens 3;
- prefix caching ON (mamba_cache_mode 'align');
- async scheduling ON by default;
- TP4 on 4x L4.

**vLLM fix check.** We checked the code at the release tags.
- v0.19.0 contains none of these fixes: #50729 (Mamba state-copy overlap race), #51113 (align chunk split, the cause of issue #43559's ~20 pp drop), #53919 (async race in the accepted-token counts, model runner V1), #43650 / #48375 / #57128 (drop the trailing Mamba block on a cache hit).
- Issue #53912 (Qwen3.5/3.8-27B FP8 + MTP + prefix caching: '!!!' / punctuation / empty outputs) is still open.
- v0.32.0 contains #50729 and #51113, and runs Model Runner V2 (confirmed in our E18 log).

**Our scan.** 679 saved thinking traces from vLLM 0.19 runs (R5, D66, L66, W3H) plus the E18 and E11-05 runs show **0 visible corruption**: no symbol runs, no CJK bursts, no empty answers. Any corruption would therefore be subtle.

**Confirmed real losses, outside E19's scope.**
- At the final drain and at forced close, the phase-2 re-prefill misses the prefix cache.
- D66 lost 14 of 15 drained answers. The E18 smoke run lost 16 of 18.
- In-run re-prefill took 4.6 % (R5) and 11 % (D66) of the run.
- The S1 rerun-path audit found no bug that only appears in the hidden rerun.

**What E19 tests.** E19 tests only engine correctness on S1's stack. If an arm is corrupted, a workaround flag is the cheap fix (`--no-async-scheduling` or prefix caching off), or we move to vLLM 0.32.

## Design

| Notebook | Account | First arm | Second arm |
|---|---|---|---|
| P1 | foysal | D = S1 exactly | C = MTP off |
| P2 | sami | E = S1 + `--no-async-scheduling` | A = MTP off + prefix off |
| P3 | hamid | B = prefix off | K = bf16 KV |

- Each notebook has a 60-min budget and a hard stop at 57 min.
- Engine: vLLM 0.19 OpenAI server from the kernel's Python with S1's `ENGINE_CONFIG` as flags, one server per arm, killed between arms. Not S1's in-process `LLM()`; please check whether that difference matters for the async race.
- Prompts: 12 real Poetiq prompts (3.3-9.4k tokens), rendered by S1's own code. Greedy decoding.

**Probes per arm:**
- serial: cold, warm (prefix hit), cold repeat;
- churn: 24 requests, up to 14 concurrent, max_tokens staggered 256-2,048, arrivals refilled as requests finish. The purpose is to reproduce #53919's row-move race;
- long continuation at about 8k and about 30k: prompt + filler + the server's own first 512 tokens, which must reproduce tokens 512-1,024;
- MTP acceptance from /metrics every 10 s.

**Verdict.** An arm FAILS on any of:
- a corrupt row whose partner row is clean;
- more large-margin first-divergence flips (> 1 nat) than its own serial A/A;
- an MTP acceptance window below 0.10;
- churn flips or corrupt rows above those of the A/B calibrators.

An unfinished arm is NOT_TESTED_BUDGET. Cross-account calibration happens offline in `src/e19_compare.py`.

**Local CPU harness**, against a fake vLLM 0.19 server. All 10 scenarios pass:

| Scenario | Checks | Notes |
|---|---|---|
| normal | 47 | |
| launchfail | 22 | |
| death | 22 | |
| corrupt | 21 | only D fails |
| race | 22 | D, B and K fail; E, A and C pass |
| collapse | 21 | |
| kvretry | 21 | |
| budget | 10 | |
| rerun | 3 | |
| split | 11 | |

normal and death also pass on Python 3.12 with transformers 4.57.6. The Kaggle CPU kit has not been run.

**Known limits (from the builder):**
- The flip rule judges only the first divergence of a pair.
- MTP arms rarely get cache hits over generated tokens, because align mode + MTP rarely lands on a block boundary. Arm C covers that path.
- Greedy decoding, while S1 uses temperature 1.0.
- The timing of the second arm is an estimate (vLLM 0.19 load takes about 9-11 min).

## Contents of the zip

- `notebooks/`: the three notebooks (P1 ad7f1eb5, P2 f17eac9e, P3 e46c1430), `build_e19.py`, `make_push_e19.py`.
- `src/`: `e19_core.py` (launcher, streaming client), `e19_run.py` (prompts, stages, report), `e19_judge.py` (verdict), `e19_compare.py`.
- `src/harness/`: the fake server, the harness, and all scenario logs.
- `push/`: the three `kernel-metadata.json` files.
- `README.md`: the experiment README.
- `context/`: your three previous audits; our reports on vLLM fix presence, re-prefill / drain loss and the S1 rerun path; the S1-60K notebook (ed530071).
