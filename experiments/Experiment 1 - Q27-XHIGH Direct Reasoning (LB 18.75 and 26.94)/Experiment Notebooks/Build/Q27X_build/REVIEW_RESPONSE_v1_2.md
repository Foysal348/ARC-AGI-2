# Q27-XHIGH v1.2: response to the agent audit (on v1.0) and the GPT review (on v1.0), with a true / false ledger

I checked every claim against the code, the vLLM 0.19 source or the model files before changing anything.

**Scope rule (from the user):** stay on our goal, which is a full-thinking Qwen3.8-27B submission following the
discussion recipe. Ideas that would change direction are listed under "deferred", not implemented.

- Notebook: `notebooks/Q27-XHIGH-v1_2.ipynb`.
- Builder: `build/make_q27x.py`.
- Local tests: `build/harness_output_v1_2.txt`. Seven scenarios pass: mixed, drain, rerun, crash, eos, vdeadline, and
  engine-ladder.
- The full agent report is in `AGENT_AUDIT_REPORT_v1_0.md`.

## Agent audit (F1–F13)
| # | claim | verdict | action |
|---|---|---|---|
| P0-ops | A notebook on the default Python 3.13 image cannot install the cp312 wheelhouse | TRUE (our S12 failure) | Push with the pinned image (as always); cell 2 raises a clear error on a Python ≠ 3.12 |
| F1 | The code/JSON regexes mis-pair fences: answers with JSON first, a `text` / `plaintext` block, or a `Python` / `python3` label lose the program | **TRUE, reproduced** (JSON-first and text-first answers gave `code found: False`) | One fence scanner plus label classification; brace-counting `test_outputs` search; 8 layouts covered by the cell-4 self-test |
| F2 | Any runtime error in the scheduler ends the notebook with an exception, which scores 0 | TRUE | try/except/finally around the scheduler, an exception-safe report cell; harness `crash` scenario |
| F3 | KV capacity IS readable in the front end; concurrency becomes 15 with MTP | TRUE (`v1/engine/utils.py` 1121–1123); my earlier claim was wrong | Documentation fixed; harness uses real block counts |
| F4 | `LLM()` disables stats unless `disable_log_stats` is passed, so MTP acceptance and KV usage are invisible | TRUE (`entrypoints/llm.py` 272–273) | `disable_log_stats=False` |
| F5 | Answer texts are not saved | TRUE | `q27x/answers/<task>.txt`: thinking tail + answer |
| F6 | Phase-2 requests re-prefill up to ~57k tokens | TRUE (cost ~20–30 min of ~10.5 h, estimated) | Accepted for this version |
| F7 | The retry after a failed engine does not wait for GPU memory | TRUE (inferred) | Waits until nvidia-smi shows < 1 GB per GPU (≤ 180 s) |
| F8 | An EOS before `</think>` stays in front of the stop text | TRUE (reproduced) | Trailing EOS ids stripped; harness `eos` scenario |
| F9 | A lone grid is rejected; helper blocks are not joined; only one `solution` block is tried | TRUE | Lone grid wrapped; helper blocks joined; up to 4 variants tried |
| F10 | The sandbox does not test numpy; thread / limit / working-directory issues | TRUE | numpy self-test; BLAS threads = 1; limits set inside the child; temporary working directory |
| F11 | A 4,096-token drain answer is too small for large grids | TRUE (9/120 evaluation tasks need more) | 8,192 |
| F12 | The hard-stop margin is thin | TRUE | Hard stop 25 min before the limit; verification deadline 8 min before |
| F13 | Seeds depended on timing | TRUE | Seeds derived from the task index |

## GPT review (12 points)
| # | claim | verdict | action |
|---|---|---|---|
| 1 | Prompts > 12,224 tokens were skipped (identity answers) | The gate was REAL. "Public prompts reach 16.4k" is russcore's grid format; with ours the longest is 8,712 tokens (evaluation) and 6,829 (test file) | Fixed anyway: a per-task thinking cap `min(49,152, context − prompt − answer − 256)`; skipped only if < 4k is left |
| 2 | EVAL12 averaged tasks; the Kaggle metric is per test query | Valid (the official metric is not in the sources offline) | Report now gives the per-query score (primary) and the task-normalised score (diagnostic) |
| 3 | Verification could run past the 12 h limit | TRUE | Train and test in one process; per-call timeout clipped to the verification deadline; unfinished verifications fall back to the direct grid; harness `vdeadline` scenario |
| 4 | Budget below the discussion's recipe | TRUE | Thinking 49,152 (NguyenThanhNhan's forced-commit point) + answer 12,288; context 81,920 |
| 5 | BF16 KV with 16 long sequences is tight | Consistent with the agent's KV calculation | `kv_cache_dtype="fp8"` (discussion recipe). VERIFIED: on sm89 vLLM 0.19 then chooses FlashInfer, because FlashAttention-2 does not support fp8 KV (`flash_attn.py` 165–170, `flashinfer.py` 320). This attention path is new for us, so the engine falls back to BF16 if it does not start (reported) |
| 6 | The temperature 0.6 comment ("Qwen recommendation") was wrong | TRUE: the checkpoint's `generation_config.json` says 1.0 | temperature 1.0, top_p 0.95, top_k 20 |
| 7 | No verifier-feedback repair turn | Valid idea | **Deferred** (scope): decide after the 12-task measurement |
| 8 | attempt_2 is rarely an independent hypothesis; fuse with the NVARC baseline | Valid idea | **Deferred** (a different direction: combining with the NVARC notebook) |
| 9 | MTP 3 tokens is the vLLM recipe | Plausible; one MTP layer; vLLM only warns | 3 tokens; acceptance is visible now that stats are on (F4) |
| 10 | Retrying the engine in the same process is risky | TRUE (same as F7) | **Kept the fallback** with a GPU-release wait and full reporting: in the rerun an exception scores 0, so falling back is better than raising. Every attempt is in `run_report.json` → `engine.attempts` (not silent) |
| 11 | The sandbox is unsafe | Partly | Added CPU-seconds and file-size limits and process-group kill. **Not** added an import/AST blocklist (it would reject legitimate code; Kaggle is already a container) |
| 12 | The projection was optimistic ("normally higher") | TRUE | Wording fixed; generation-only and end-to-end projections reported |
| — | The harness was not reproducible from the bundle | TRUE | `fixtures/fake_input/` added (competition JSON + tokenizer + template) |

## Deferred, to stay on our goal
1. Verifier-feedback repair turn (GPT 7).
2. Second attempt fused with the NVARC baseline (GPT 8).
3. Import blocklist in the sandbox (GPT 11).
4. Logits-processor forced close to avoid the phase-2 re-prefill (agent F6).
