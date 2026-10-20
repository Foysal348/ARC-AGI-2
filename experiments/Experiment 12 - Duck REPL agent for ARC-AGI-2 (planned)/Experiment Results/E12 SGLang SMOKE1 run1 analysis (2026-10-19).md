# E12 SGLang SMOKE1, run 1 (foysalemonshanto, 2026-10-19)

**Run:** https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-smoke1?scriptVersionId=359922347 (notebook sha 0fea4d0c).
**Outputs:** `Experiment Results/Kaggle Runs/E12 SGLang SMOKE1 run1 (foysal, 2026-10-19)/`.

## Result: every step-3.5 gate passed

| Gate (PLAN Rev 1 #7) | Evidence |
|---|---|
| Structured tool calls | 10/10 replies have `finish_reason=tool_calls` |
| Preserved thinking replayed | Each next request carries the earlier `reasoning_content` (b0: 6,563 / 760 / 503 / 736 chars; b1: 11,984 …) |
| Function retention | "Retained your function analyze(grid) / blocks_of(grid) / transform(grid) …"; b1 later calls `blocks_of(test_inputs[0])` from a previous call |
| `cached_tokens` rising | b0: 4,096 → 6,867 → 10,515 → 11,776 → 11,968; b1: none → 6,867 → 13,056 → 13,120 → 13,376. Overall 80.8 %, per request up to 99 % |
| No server errors | none, apart from the harmless Kaggle-image `sitecustomize: No module named 'wrapt'` line |

**Task result:**
- Task **2ba387bc solved by both branches** (eval 1/1, task-normalised 100 %).
- Both branches were accepted at their first submit (train 4/4, host-verified).
- b0: 2 turns, 4,905 generated tokens, 60 s. b1: 2 turns, 5,907 tokens, 73 s.
- For comparison, the Poetiq DIRECT solve of this task needed 5,355 thinking tokens in one request.

**What the model did:**
- **b0:** inspected objects with BFS → its first `transform` returned an invalid grid (`check` 0/4) → it debugged the output shapes → fixed it → `check` 4/4 → `submit(transform)`.
- **b1:** `blocks_of` → `transform` (`check` 4/4) → looked at the test input → called a tool named `submit` (Duck: "Unknown tool: submit") → corrected itself next turn with `submit(transform)` in python.

**Infrastructure:**
- **Engine:** W1 up in 677 s, KV 866,744 tokens.
- **Memory limit verified on Kaggle Linux:** RLIMIT_AS applied to every child, read-back ok, 5 GiB rejected, 128 MiB ok.
- **Decode:** 204 tok/s with 2 running (DFlash acceptance median 5.2).

## For EVAL30

- KV pressure and coverage were not exercised: only 2 conversations, ≤ 13.6k context. EVAL30 (12 streams) has to measure these.
- Watch how often the model calls `submit` as a tool name (1 of 10 replies here). It cost one turn; Duck's error made the model correct itself.
- SGLang also omits `prompt_tokens_details` when nothing is cached (b1's first reply: `None`). The SGLang smoke does not depend on it.
