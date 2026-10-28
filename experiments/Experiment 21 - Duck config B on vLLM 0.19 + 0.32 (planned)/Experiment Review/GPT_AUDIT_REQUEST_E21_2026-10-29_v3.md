# E21 audit request v3: the last check before the GPU smoke (rev 2g, 2026-10-29)

## What this is

E21 runs the proven Duck config B (R3K2: 50K45K, Duck package `9e4b89285136`) on vLLM instead of SGLang. Each notebook has two halves of 5.45 h:

| Half | Engine | NB1 primary | NB2 primary | Fallback |
|---|---|---|---|---|
| A | vLLM 0.19, in the kernel | RedHat INT4, MTP 3, 20 streams | FP8, MTP 3, SSM fp16, 16 streams | FP8, MTP 3, 16 streams |
| B | vLLM 0.32, `/tmp/vllm-venv` | same as A (+ `--language-model-only`) | same as A (+ `--language-model-only`) | same as A's fallback |

The hidden rerun runs half A only. You audited v2 (the rev 1 shas `ad5b2d1f` / `b968e322`). Since then there were six fix passes (rev 2b to 2g), driven by your three audits, 13 agent reports and two Kaggle CPU kits.

**Plan:** this audit, then the GPU smoke (`*_SMOKE` notebooks: 12 eval tasks, 20 min per half, 2.5 h, on the user's account), then the full runs.

## The question

**Is there anything left that would make the GPU smoke or the full run break, or write a wrong `submission.json`, or make the half-A vs half-B comparison false?**

Please report only those: BLOCKER and HIGH, each with the file and line, a concrete failure scenario and a fix. Metric, naming and hardening items are already logged (tracker `LOGGED`) and will not be changed before the run.

## Notebooks (rev 2g)

| Notebook | sha256 [:8] |
|---|---|
| `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` | `8a957b7c` |
| `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb` | `14e034c8` |
| `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20_SMOKE.ipynb` | `a0b10da7` |
| `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16_SMOKE.ipynb` | `daec777f` |

The SMOKE notebooks are the same code; only the default profile differs (`E21SMOKE` instead of `E21X2EVAL120`). See `changes_rev1_to_rev2g/NB1_vs_NB1_SMOKE.diff`.

## What changed since v2

The full list, with sources and status, is in `reviews/E21_FINDINGS_TRACKER.md`. Each fix pass is in `fix_passes/patch_*.py`. The per-cell diffs against what you audited are in `changes_rev1_to_rev2g/`.

**The cross-half isolation items you raised (BLOCKER / CRITICAL in all three audits):**
- **The branch callback** is bound to its half through default arguments (`_recs`, `_run`, `_lock`, `_half`). It returns before any write when `_half != E21_HALF`.
- **The heartbeat and the watchdog** are bound the same way (rev 2f: the rev 2c module-level binding did nothing after `wrap()`). The watchdog re-checks the half right before acting. Both are named and joined in `e21_end_half`.
- **The handoff:** after half A's server is killed, there is a bounded wait (≤ 120 s) for `solve_all_r6` threads. `handoff_scheduler_threads_left` is recorded. It is not a hard block (see the tracker for why).
- **The Duck priority gate** (a process singleton) is reset per half.

**Engine:**
- **Capacity** is block-based: `cdiv(57k, block) + 15` Mamba blocks per stream. It fails closed, and refuses fewer than 8 streams.
- **`--language-model-only`** is set on 0.32 and checked in the log (`lmo_in_log`).
- **GPU release** is checked before every launch.
- **Smoke:** strict, needing "4" and non-degenerate text (rev 2g: "4." is accepted).
- **The 2-turn contract probe** uses Duck's real tool schema. The cache probe compares 96 greedy tokens, cold vs cached (#53912).
- **Hidden rerun (rev 2f):** the history and cache probes become warnings, so a false probe failure cannot leave the rerun without an engine. Real breakage stays fatal: HTTP errors, markup in content, degenerate text and broken tool arguments.
- **finish_reason:** CELL 3 restores "length" for a tool call cut at max_tokens (vLLM returns "tool_calls"; `e21_cell3_patch.py`).
- **Relaunch:**
  - its timeout follows the solve end;
  - the second relaunch switches config only between configs with the same weights, so the served name stays valid (rev 2g, VAL2_engine F1);
  - nothing is launched with less than 12 min left.
- **/metrics outage:** admissions hold for ≤ 120 s, then fall back to Duck's own fail-open. The clock is per half.

**Timing:**
- equal windows: half B gets half A's achieved window;
- the B setup reserve is 30 min;
- the 0.32 venv is installed in the background during half A;
- the last half kills its server fast, without waiting for the GPUs;
- `solve_overrun_sec`, `deadline_violation` and `stragglers` are recorded (the 9,187-s NB2 rerun was a host suspend, as the heartbeat gap shows).

**Report:**
- `config_status` (PRIMARY_BOTH, FALLBACK_BOTH, CONFOUNDED_FALLBACK, SWITCHED_MIDHALF or NOT_RUN) compares configs;
- `comparison_valid` requires equal windows (within 1 %), the same config, equal streams and no deadline violation;
- `stage1` / `pass1` are reported per half.

## Tests

| Test | Code | Result |
|---|---|---|
| Kaggle CPU flag kernel (pinned image, Py 3.12): every vLLM 0.19 / 0.32 command parses; the tokenizer check | rev 2c | ALL OK (8/8 + 8/8; tokcheck OK with transformers 4.57.6 and 5.19.0) |
| Kaggle CPU harness kit H1-H4 (fake vLLM server, all 11 modes: commit, rerun, fallbackA, halfAfail, death, reasoningkey, gateAB, wdhold, overrun, kvoutage, gpuheldB + smoke commits) | rev 2e | report-only failures, fixed in 2e/2f (`cpu_kit_kaggle_rev2e/`) |
| The same kit | rev 2f | 20 / 24 runs fully PASS. The 4 others each had one stale harness expectation, the harness was not updated after rev 2e / 2f: gateAB expected `comparison_valid` True with 10 vs 20 streams (rev 2e makes unequal streams invalid); smoke expected a 2-h budget (rev 2f: 2.5 h). Both expectations were corrected (`cpu_kit_kaggle_rev2f/`) |
| Local harness on rev 2g, with the corrected harness | rev 2g | NB1 death 75/75, NB1 SMOKE commit 71/71. NB2 death had a Windows port-reuse flake (WinError 10048 on 8011 from the previous run; the notebook fell back correctly); its rerun, NB1 gateAB and NB2 SMOKE commit were still running at zip time (`harness_runs_local_rev2g/`) |
| Local harness, the latest run per mode | rev 2e/2f | `harness_runs_local/` |

The rev 2g change is two expressions (`fix_passes/patch_rev2g.py`). The smoke predicate was checked on "4", "4.", "2 + 2 equals 4.", "Four." (all accept) and "4444444444", "!!!!!!!!" (both reject).

## Known and accepted (do not re-report unless you think it breaks the run)

- vLLM's align-mode prefix cache never reuses a previous reply: about 75 % cached vs SGLang's 93 %, about +11 M prefill per half. It is measured.
- Capacity is sized at 57k per stream (the R3K2 p50 peak), not the p90 of 65k. The 0.95 KV gate and preemption cover the tail.
- The commit's final `submission.json` is half B's. The commit is for eval scoring only, and per-half scores are in the reports.
- CELL 2 install failure is the same path as S1 / R3K2 (the proven 0.19 wheelhouse).
- Everything marked LOGGED / DEFERRED / NOTE in the tracker.

## Files in this zip

| Path | Contents |
|---|---|
| `notebooks/` | the 4 rev 2g notebooks |
| `build/` | builder, engine cell source, CELL 3 patch, log summary, CPU kit builder |
| `src/` | flagcheck, tokcheck, the flags json, the fake vLLM server, the harness |
| `reviews/` | all 14 agent reports (`VAL_*`, `VAL2_*`, including `VAL2_dryrun.md` and `VAL2_engine.md`), the tracker, the v2 request and provenance |
| `fix_passes/` | every patch script, rev 2 to 2g |
| `changes_rev1_to_rev2g/` | per-cell diffs against the v2 notebooks; NB1 vs NB1_SMOKE |
| `harness_runs_local/`, `cpu_kit_kaggle_rev2e/` | test outputs |
| `donors/`, `evidence/`, `provenance_rev1/` | unchanged from v2: R3K2, E12-V, E18, VMS, S1 donor cells, the Duck package, vLLM source excerpts, real server logs, rev 1 provenance diffs |
| `previous/` | the v2 request and provenance file |
