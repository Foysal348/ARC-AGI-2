# Experiment 5 — Q27-AGENT v1.9 Long-Think Synthesis (planned)

**Status (2026-10-12):**
- Plan validated by GPT.
- GPT built **v1.9.0**. Claude validated it and fixed 8 findings in **v1.9.1**; see `Experiment Review/v1.9.0 validation and v1.9.1 changes.md`.
- All local gates PASS: static, 17/17 unit tests, whole-notebook dryrun with DRYCHECK 17/17.
- Kaggle CPU DRYRUN: 17/17 PASS.
- **GPU EVAL12 (foysal v3, about 56 min):** agent **1/22**, DIRECT 3/22, hybrid 3/22, so no gain.
- **Root cause:** `thinking_token_budget` was not enforced. 10 of 12 tasks thought until `max_tokens` (57,344) with no answer.
- Details: `Experiment Results/Analysis/GPU EVAL12 v1.9.1 analysis (2026-10-12).md`.

**Open decision for the user:** the Phase 3 (repair) thinking budget. The rule is no budget below 49,152 without approval.

## Architecture (chosen by the user; proposed in GPT's root-cause audit of Experiment 4)

**Long-Think → Program → Verify → Repair**

1. **Phase 1 — long-think synthesis.**
   - One request per task, xhigh, **49,152 thinking + 8,192 answer**.
   - **No tools, no named `tool_choice`.** The model writes one complete ```python program defining `transform(grid)`.
2. **Phase 2 — deterministic verification** on every training pair (sandbox + guard). No LLM tokens.
3. **Phase 3 — repair** (if rejected).
   - Input: the previous program + the exact cell diff, with the REPL available.
   - No forced submit, no effort ladder.
   - Budget: TBD (user).

**Merge with DIRECT** (the LB-26.94 method): attempt_1 = DIRECT's answer, attempt_2 = the verified program's output (proposed).

**First GPU step:** EVAL12, Phase 1 only (about 58 min). Early abort if each of the first 4 tasks yields no program.

## Files

- `v1.9 implementation plan (for GPT review).md`: the full plan (modules, tests T1–T15, GPU sequence, questions).
- **Experiment Review/:** `Q27_AGENT_v1_9_PLAN_FOR_GPT_2026-10-12.zip`.
- **Experiment Results/:** put GPT's answer here.
- **Experiment Notebooks/:**
  - `Q27_AGENT_v1_9_0_SYNTH_MICROPROBE.ipynb`: GPT's original, unchanged.
  - **`Q27_AGENT_v1_9_1_SYNTH_EVAL12.ipynb`**: the version to run.
  - `Build/Q27A_v19_build/`: the builder (`build_notebook.py`), `modules/`, `tests/test_v19.py` and GPT's delivery notes.
  - Rebuild with `python build_notebook.py`.
