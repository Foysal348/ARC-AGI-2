# Handoff — v2.5A fixes, PIV-SUB v1.0 submission notebook, and the competitor baseline (2026-10-02)

Audience: ChatGPT (notebook author) and the project owner. Everything below was produced and verified locally with the
fake-engine harness (real Qwen tokenizer + chat template, real Kaggle competition data, real verifier subprocesses,
scripted vLLM). No GPU time was used.

---

## 1. The Kaggle notebook that uses our model

`phamtruongdat/notebookfa6c76ce70` (the link had a trailing character; the real slug has none). Pulled through the
Kaggle API and read in full.

**What it is:** the direct non-thinking baseline, i.e. the Q1 recipe that already scored 0% for us.
* prompt = `{"examples": train pairs, "test_input": grid}` → `{"grid": [[...]]}`, `enable_thinking=False`,
  `n=2`, temp 0.7 / top_p 0.8 / top_k 20, `max_output_tokens=4096`, `max_model_len=16384`, batch 4, TP=4,
  `gpu_memory_utilization=0.85`, `enforce_eager=True`, prefix caching off.
* No train verification, no program synthesis, no REPL, no repair. Parse failure ⇒ identity copy of the test input.
* Expected ARC-AGI-2 score: ~0–2%. It is not a recipe to copy; our PIV pipeline is strictly ahead of it in method.

**Three things worth taking:**
1. **Self-prepared offline runtime.** A separate internet-ON CPU notebook runs
   `pip download --only-binary=:all: --extra-index-url https://flashinfer.ai/whl/cu129 vllm==0.19.1 transformers==5.8.0
   flashinfer-jit-cache==0.6.6`, writes a `runtime_manifest.json` with per-wheel sha256, and is attached to the
   submission notebook, which verifies every checksum before installing. We currently depend on a third party's
   wheelhouse dataset (`driessmit1/arc3-vllm-h100-wheelhouse-v3`) and on vLLM 0.19.0. Owning the runtime removes a
   single point of failure and would let us move to 0.19.1 deliberately.
2. **`language_model_only=True`** in `LLM(...)`. For this checkpoint it skips the non-text tower. Our prefix-caching
   experiment OOM'd; this flag is a cheap thing to test for KV headroom.
3. **Submission-side engineering:** supervisor process with a wall-clock limit that kills the whole process group and
   keeps the last valid checkpoint, atomic validated `submission.json` writes after every batch, an identity fallback
   for every test input from the start. We had no submission path at all; PIV-SUB v1.0 (below) now covers this with
   the same guarantees, implemented inside our wave scheduler instead of an external supervisor.

---

## 2. PIV-Q27-C1 v2.5A — the four audited fixes

`Research Base Experiment/PIV-Q27-C1-v2_5A-CANDIDATE-FIRST-VERIFIER-FEEDBACK-REPAIR.ipynb`
sha256 `2188b999efeb6e286fc914bb8c1d876a5c9bdab99537842cd8c94360cd47d5ff`

| # | Fix | What it was |
|---|---|---|
| P0-1 | repair-helper self-test now asserts `dims_mismatch` on a pred whose dims differ and `cells_wrong` on a same-dims identity pred | the old assertion demanded `dims_mismatch` from a same-dims pair, so cell 6 raised `AssertionError` in **every** profile |
| P0-2 | `_mk_params` / `_mk_finalizer_params` hoisted to module level, plus `assert callable(...)` in the repair cell (runs in PREFLIGHT too) | cell 8B called `_mk_finalizer_params` at module scope while cell 8 defined it inside `run_real_piv` → `NameError` after the full 78-minute scheduler, before scoring |
| P1-2 | when the verifier returns no per-pair results (`train_timeout`, `train_code_error`) the feedback now reads `VERIFIER: <status>; <last error line>` | it claimed `pair i: EXCEPTION missing verifier pair result` for every pair and dropped the real error |
| P1-1 | `REPAIR_BUDGET_PREREGISTRATION` in cell 1 and in the manifest | the repair cap is the leftover of the same 12,288-token budget; for ACT-LOW 271d71e2 / 2c181942 / 16b78196 it is smaller than candidate #1's own JSON size, so those repairs are structurally truncated |

Harness status: **PREFLIGHT, TERMINAL_PILOT, ADAPTIVE_SMOKE and ADAPTIVE_SMOKE + over-cap injection all complete
end-to-end.** A scripted verified repair is selected as candidate #2 and reaches scoring (`train_perfect 0.125`,
tier `usable_quality`), and the over-cap injection is still caught (`repair_output_exceeded_max_tokens`, candidate #1
preserved). v2.5A is cleared for GPU.

Also confirmed now that the harness uses the **real** Kaggle competition files (downloaded through the API rather than
GitHub): the Proxy-48 source-EDA contract reproduces exactly, including `multi_query: 8`. The earlier 7-vs-8 mismatch
was a GitHub-data artefact, not a notebook defect.

---

## 3. PIV-Q27-SUB v1.0 — the submission notebook (new)

`Research Base Experiment/PIV-Q27-SUB-v1_0-HIDDEN-TEST-SUBMISSION.ipynb`

Derived from v2.5A. Cells 1–8B (config → data → EDA → prompts → context → parser/REPL/verifier/repair helpers →
engine → scheduler → repair round) are the audited solver, unchanged in substance. Every measurement/scoring cell was
replaced by submission machinery.

**Guarantees, in execution order**
1. A competition-legal `submission.json` (identity copy of each test input) is written **before** the model is
   allocated, after schema self-tests. Nothing downstream can leave the file missing or malformed.
2. The task set is the attached `arc-agi_test_challenges.json` — no task id is hard-coded, so the file Kaggle swaps in
   at scoring time is what gets solved. The local placeholder has **240 tasks / 259 queries**.
3. A task whose rendered prompt exceeds the context budget is dropped with a recorded reason instead of raising
   (v2.5A would have killed the run).
4. The frozen v2.4G adaptive controller is reused: terminal positive control → 2-task interface rehearsal → gate →
   full sweep. A failed gate stops the run and keeps the fallback submission.
5. `append_record` ingests every terminal record into the submission and rewrites the file atomically (throttled to
   30 s), so a session that dies mid-sweep keeps the best file reached.
6. The existing wave deadline logic stops scheduling instead of overrunning; the repair round is now chunked by
   `AGENT_BATCH_SIZE` with a per-chunk deadline projection (a 200-job single `generate()` call could not be
   interrupted).
7. CPU pass: programs that failed the train gate never had their test inputs executed. They are executed here in
   fresh sandboxes (guarded, 10 s per input, 20-minute budget) so they can still fill `attempt_2`.
8. `attempt_1` = highest-ranked program output (train-verified first, then more train-exact pairs, then newer),
   `attempt_2` = best *different* output, identity copy wherever no program produced a legal grid. Every grid passes
   an explicit 1–30 × 1–30 / ints 0–9 / rectangular sanitizer before it can enter the file.
9. `run_report.json` + evidence bundle: coverage, train-verified count, stop-reason histogram, repair summary,
   interface gate, ingest errors, and the pre-registration block.

**One P0 found and fixed while testing this notebook:** the terminal positive control only ran for
`RUN_PROFILE in {TERMINAL_PILOT, ADAPTIVE_SMOKE}`, so in a submission profile `positive_control_ok` was `False`, the
interface gate failed, and the entire sweep was skipped — a wasted session that would still have produced only the
identity submission. The control now runs in the submission profiles too.

**Profiles**

| profile | per-task tokens | finalizer reserve | concurrency | tool calls / turns | purpose |
|---|---|---|---|---|---|
| `SUBMIT_SMOKE` | 8,192 | 2,048 | 4 | 6 / 12 | 4-task interface rehearsal (~25–35 min incl. load). Never submit it. |
| `SUBMIT_FAST` | 4,096 | 2,048 | 12 | 4 / 8 | whole file inside one working day |
| `SUBMIT` (default) | 8,192 | 2,048 | 12 | 6 / 12 | better-quality full sweep |

**Pre-registered deviations from the v2.4G/v2.5 measurement runs** (stated in the notebook header and the run report):
smaller per-task budget, smaller finalizer reserve, concurrency 12 instead of 8, single arm `ACT-LOW`. Reason is
wall-clock only — v2.4G measured ~4.25–4.9 min per task-run at batch 8 with the 12,288-token budget, i.e. ~8.5 h for
120 tasks and ~17 h for 240. These runs are therefore **not comparable** with the arm measurements.

**Runtime projection** (scaled from the v2.4G measurement, 0.75 concurrency efficiency, plus ~10 min load and ~20 min
post-passes):

| | 120 hidden tasks | 240 tasks |
|---|---|---|
| `SUBMIT_FAST` | ≈ 3.5–4 h | ≈ 7–7.5 h |
| `SUBMIT` | ≈ 7–7.5 h | truncated by the deadline (~75–85 % coverage) |

**Local verification (fake engine):** `SUBMIT_SMOKE` and the full `SUBMIT` profile both run every cell to completion,
including all 240 tasks in 20 batches of 12, the chunked repair round, the CPU test-execution pass, the run report and
the bundle. The produced `submission.json` was validated independently of the notebook: ids identical to
`sample_submission.json`, one answer per test input, only `attempt_1`/`attempt_2` keys, zero illegal grids.

---

## 4. Suggested order today

1. `RUN_PROFILE = "SUBMIT_FAST"` → **Save & Run All** (it contains its own positive control + 2-task gate, so a
   separate smoke run is optional). Submit `/kaggle/working/submission.json` from that saved version.
2. Then `RUN_PROFILE = "SUBMIT"` for the better sweep, and `PIV-Q27-C1 v2.5A ADAPTIVE_SMOKE` (78 min) for the repair
   measurement on the public evaluation set with known answers.

Expect a near-zero score: v2.4G produced 0/16 train-exact programs, and nothing since then has changed the solver's
reasoning. The purpose of today's run is to prove the submission path end-to-end and to measure the
**train-verified rate on the real test file** — the leading indicator for any future score.

## 5. Open items for the author

* Own the runtime bundle (item 1 in section 1) instead of the third-party wheelhouse; decide deliberately on
  vLLM 0.19.0 vs 0.19.1.
* Test `language_model_only=True` for KV headroom.
* The 768-token phase-2 truncation in `ACT-BUDGET3072` is still frozen and still wastes ~2.5k tokens per affected
  task; `ACT-LOW` is used for submissions partly to avoid it.
* Next quality lever after v2.5A: K independent candidates per task with verified pick, not a longer single session.
