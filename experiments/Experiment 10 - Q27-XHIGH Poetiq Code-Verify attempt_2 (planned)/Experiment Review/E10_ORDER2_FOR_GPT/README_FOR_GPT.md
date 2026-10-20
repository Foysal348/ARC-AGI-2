# E10 ORDER2: pre-GPU check before a 10-hour full evaluation run (2026-10-14)

**What we ask:**
- Report **blockers or real bugs only**: anything that would crash the run, waste the GPU time, produce a wrong score or an invalid `submission.json`, or break the hidden rerun.
- **Focus on the puzzle-ordering code.**
- Please do not propose new features, refactors or design changes. The design is the user's decision and is fixed.
- If you find nothing, say "GO".

## The run

| | |
|---|---|
| Notebook | `notebook/Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb`, sha256 `b62ae567…` |
| Base | `notebook/Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb`, sha256 `530cb119…`. This is v1.4.4 DIRECT, the LB-26.94 code, with an 86k budget and no tail resample. The same code ran on GPU as Experiment 9 (EVAL12: 0 errors, KV max 91.4 %, 0 preemptions, 4/22) |
| Commit run | profile **EVAL120**: all 120 evaluation tasks / 172 test inputs, scored |
| Clock | the hidden-run clock: 11 h 50 min budget; admit until 694 min, drain at 696, hard stop at 704 |
| Budget | 77,824 thinking + 8,192 answer, `MAX_MODEL_LEN` 106,496, 16 concurrent, Qwen3.8-27B-FP8, vLLM 0.19.0 TP4 on 4×L4 |
| Hidden rerun | profile `RERUN_ALL`, unchanged apart from the queue order |

## The only changes versus the base (`BASE_TO_ORDER2.diff`, built by `build/build_e10_order2.py`)

1. **Cell 4 (ordering):**
   - `order_score(task) = int(len(task["train"]) >= 3) + int(len(task["test"]) == 1)`.
   - `ORDERED_TASKS` sorts by score, highest first; ties keep the original sorted-id order.
   - `ORDERED_UNITS` keeps each task's units together (query 0, 1, …). `ORDER_ADMIT_RANK` is built from it.
   - Asserts check that the reordering keeps the same unit set.
2. **Cell 7 (scheduler):**
   - `queue = [u for u in ORDERED_UNITS if u in PROMPT_IDS]` replaces `UNITS`.
   - Records gain `order_score` and `admit_rank`.
3. **Cell 2 (profiles):** profiles `MIX20` and `EVAL120` are added, and the commit default is `EVAL120`.
4. **Cell 0:** the banner text.

## Questions on the ordering (please check)

- Does the ordering apply in **both** modes (eval commit and hidden rerun), and **only** change the admission order? Nothing else should change: prompts, seeds (`UNIT_INDEX` is unchanged), scoring and the submission.
- Can any unit be lost or duplicated? Consider skipped long prompts (not in `PROMPT_IDS`), drain, the hard stop, and the not-admitted path.
- Are `order_score` / `admit_rank` safe for every record path (`_record` and any other path that writes records)?
- Is the rule computed **only** from the visible task (training pairs and number of test inputs)? It must never read solutions.
- Is anything wrong with EVAL120 on the hidden-run clock? For example, the report or scoring with 172 units, or the 12 h Kaggle session limit.

**Evidence for the rule (context only; the rule is fixed):** `ordering_evidence/`.
- `derive_ordering_rules.py` runs over 120 evaluation tasks.
- Label: solved by Qwen3.8-27B in any of our runs or in ARC Prize's verified run.
- AUC of the 2-rule score under nested CV: 0.57, against 0.51–0.53 for the NVARC tier/fine order.
- Solve rate by score 2/1/0: 58 / 43 / 20 %.

## CPU validation already done (`cpu_validation/`)

`build/harness_e10.py` runs the real notebook cells with the real Qwen tokenizer and chat template, the real competition files, and a fake vLLM engine.

| Scenario | Result |
|---|---|
| mixed (EVAL120) | 172/172 finished; admission follows `admit_rank`; scores never increase along the queue; `order_score` / `admit_rank` in every record |
| drain | drain and hard stop handled |
| rerun (placeholder hidden file) | 259/259 finished; admitted in rank order; scores 2/1/0 = 182 / 69 / 8 units; submission written |

**Prompt tokens:**
- EVAL120: median 4,336, max 16,400. No thinking budget is reduced; full 77,824 holds up to a 20,224-token prompt.
- **Real deadlines** (no harness override): EVAL120 and RERUN_ALL both give admit 694, drain 696, hard stop 704 minutes.

## User rules (fixed)

- At least 49,152 thinking tokens per request.
- A valid `submission.json` is always produced.
- Keep changes simple.
- Nothing is run or pushed by you.
