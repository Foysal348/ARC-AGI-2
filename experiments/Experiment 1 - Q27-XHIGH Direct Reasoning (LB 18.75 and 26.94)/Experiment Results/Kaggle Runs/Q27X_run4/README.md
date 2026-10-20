# Q27-XHIGH v1.4: Kaggle EVAL12 run 4 (2026-10-06)

## The goal and the request to the reviewer
We compete in Kaggle "ARC Prize 2026 – ARC-AGI-2" on 4×L4 GPUs with a 12-hour limit. Our goal is **LB ~60** using Qwen3.8-27B-FP8. Another competitor reportedly reached 60 with this same model; our team's best is 32.64, from a different system (NVARC).

**Please analyse this bundle independently and tell us:**
1. Why our score is low. Rank the causes by expected score impact, with evidence from the files below.
2. What to change, as concrete and testable steps. Say which ones fit in the 12 h, 4×L4 budget.
3. Which of our own conclusions in `OUR_ANALYSIS.md` are wrong or missing something. Read that file **after** forming your own view.

## Contents
| path | what |
|---|---|
| `notebook/Q27-XHIGH-v1_4.ipynb` | Notebook as run (Kaggle foysalemonshanto/q27-xhigh, version 6). sha256 `aa27d6e1ac27320f676337d6a8ca05bdb8ac3f1123ba60c9ac9d93e0c5669137` |
| `run4/run4_executed.ipynb` | The same notebook after the Kaggle interactive run, with all outputs and the vLLM log |
| `run4/output/q27x/run_report.json` | Run summary |
| `run4/output/q27x/records.jsonl` | One record per test query (unit = task#query): mode, tokens, grid parsed, attempt sources, hit |
| `run4/output/q27x/answers/<task>_q<i>.txt` | Per query: the last ~1,500 thinking tokens and the full answer after `</think>` |
| `run4/output/q27x/vllm_metrics.json` | vLLM metric snapshots (spec decode, KV usage, preemptions) |
| `previous_run3/` | Previous version v1.3.5 on the same 12 tasks: README, report, records, answers |
| `reference/russcore_LB10.14.ipynb` | Public notebook russcore/arcagi2-qwen38-freeform-direct, the version that scored LB 10.14 (same model) |
| `reference/kaggle_discussion_745787.md` | Discussion thread "Qwen 3.8 token generation speed", fetched via the Kaggle API, plus two screenshots |
| `OUR_ANALYSIS.md` | Our own diagnosis and proposed fixes |

## How v1.4 works
**Engine**
- vLLM 0.19.0 with Qwen3.8-27B-FP8 (hybrid model: 48 GDN layers plus 16 attention layers).
- Tensor parallel 4, fp8_e4m3 KV cache.
- MTP speculative decoding with 3 tokens, FlashInfer attention.
- `max_num_batched_tokens` 4096; prefix caching off; up to 16 queries at once.

**Per test input (query)**
- One prompt per test input. The prompt is russcore's text verbatim: grids as rows of space-separated digits, "give ONLY the final output grid … in a fenced ```grid block".
- Thinking at xhigh effort, temperature 0.6, top_p 0.95, top_k 20.
- If thinking has not ended after 49,152 tokens, Qwen's stop sentence plus `</think>` is appended, keeping the full context, and the grid is generated in a second request of ≤ 8,192 tokens. This is the commit point NguyenThanhNhan described in the discussion.
- attempt_1 is the model's grid; attempt_2 is the test input.

**Scheduler:** continuous admission on vLLM's step API, with admit / drain / hard-stop deadlines.

## Run 4 results (12 public evaluation tasks, 22 queries)
**Score:** **2/22 queries (9.09 %)**; task-normalised 12.5 %. The previous version v1.3.5 scored 1/22 on the same tasks.

**How queries ended**

| how the query ended | queries | exact | 1 cell wrong | share of generated tokens |
|---|---|---|---|---|
| finished thinking naturally (26k–42k thinking tokens) | 4 | 2 | 2 | 13 % |
| force-closed at 49,152 thinking tokens | 18 | 0 | 0 (closest: 6/323 cells wrong) | 87 % |

**Grids:** a grid was parsed for 20/22 queries. The 2 that failed had ragged rows written by the model (one row had an extra cell).

**Throughput**
- 373 tok/s while 16 queries were running.
- 248.9 tok/s over the whole run.
- In 33 % of the logging windows ≤ 6 queries were running (the tail of a 22-query run).
- Generation stalled in 12 % of the time: each force-close re-prefills about 50k tokens because prefix caching is off.

**Other engine measurements**
- MTP acceptance 65.7 %.
- KV-cache usage peaked at 74 %; 0 preemptions.
- Setup took 14.3 min.

**Projection for the hidden test** (240 tasks / 259 queries): the report gives 13.86 h, computed from the overall rate.

## Reference points
| who | score | how |
|---|---|---|
| arcprize.org | ARC-AGI-2 semi-private 42.4 % | Qwen3.8-27B xhigh, ~175k output tokens per task (2 attempts) |
| NguyenThanhNhan (discussion) | LB 23.33 / 22.50 | SGLang, EAGLE, radix cache on, xhigh, 65k, forced commit at 49k with a grid, 1 pass, pure reasoning ("2x pure 48-12-4 partial fallback") |
| russcore free-form (same model) | LB 10.14 / 7.78 / 8.61 | Per query, grid-only, temperature 0.6, 57,344 tokens, no forced commit, recovery call when truncated |
| ours, v1.3.5 (code + JSON per task) | EVAL12 1/22 | — |
| ours, v1.4 | EVAL12 2/22 | not submitted yet |
