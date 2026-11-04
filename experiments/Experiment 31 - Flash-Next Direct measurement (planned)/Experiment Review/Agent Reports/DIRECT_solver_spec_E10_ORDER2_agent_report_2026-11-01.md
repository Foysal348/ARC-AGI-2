## DIRECT solver specification (v1.4.4 DIRECT, E10 ORDER2 86k variant)

**Which file this covers.** `D:\ARC AGI 2\experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Notebooks\Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb`. Its sha256 is `b62ae567aa4d8af2…`, the same as the README's "Update 3 / EVAL120" build, so this is the notebook that produced the GPU result (hosen42/q27-xhigh-e10-order2: 58/172 = 33.72 %, task-normalised 36.94 %).

**Cell numbering.** The notebook has 9 cells. Index 0 is markdown. Indices 1–8 are code cells whose headers say "CELL 0" to "CELL 7". Below I cite **nb[i]** (the JSON index) and give the header label in brackets.

**Where it differs from the LB-26.94 notebook.** I diffed it against `Build/source/Q27-XHIGH-v1_4_4_LB26.94.ipynb`. These are the only differences:
- THINK_BUDGET 49,152 → 77,824
- MAX_MODEL_LEN 81,920 → 106,496
- EXPECTED_LIVE_TOKENS_PER_SEQ 30,000 → 43,000
- extra profiles (MIX20, EVAL120, PROXY56) plus fixed task-list support
- the ORDER2 queue order
- record fields t_admit_min, t_done_min, order_score, admit_rank
- default commit profile EVAL12 → EVAL120

Prompt, parser, attempts, sampling, engine and scheduler logic are otherwise identical.

---

### 1. Prompt

**Grid serialization** (nb[4] [CELL 3] lines 17–18): one row per line, cells as integers separated by single spaces, rows joined by `\n`. There are no brackets and no commas.
```python
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)
```

**User message, verbatim** (nb[4] lines 20–32). The code calls this "russcore's LB-10.14 free-form prompt, verbatim". Each prompt contains all training pairs and exactly one test input.
```python
def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)
```
- Every part already ends in `\n`, and the parts are joined with `\n`, so there is one blank line between sections.
- Within an example, the INPUT grid and the OUTPUT header are separated by a single newline only.
- Examples are numbered from 1.
- There is no system prompt written by us and there are no few-shot examples.
- Self-test (nb[4] line 72): the message must not contain "xhigh", must contain "FINAL INPUT:", and must end with "Output nothing after the block."

**Chat template call** (nb[4] lines 34–37):
```python
def render_prompt(train, test_input):
    msgs = [{"role": "user", "content": build_user_message(train, test_input)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True, reasoning_effort=REASONING_EFFORT)
```
- `REASONING_EFFORT = "xhigh"` is set at nb[2] [CELL 1] line 33.
- The prompt is then tokenized with `tokenizer.encode(text, add_special_tokens=False)` (nb[4] line 62). Everything downstream works on token ids.

**How "xhigh" is expressed.** It is not a sampling or engine parameter. It is a Jinja variable of the model's `chat_template.jinja`.
- I read the 27B template at `experiments/Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)/Experiment Notebooks/Build/Q27X_build/sources/model_qwen3-8-27b-fp8-repacked-v1/chat_template.jinja`.
- With `enable_thinking` true or undefined, `reasoning_effort|default('xhigh')` must be one of `xhigh`, `medium` or `low`.
- xhigh inserts this system message: `Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.`
- low inserts a different sentence. medium inserts no system message at all.
- The generation prompt ends with `<|im_start|>assistant\n<think>\n`.

**Exact rendered output.** I rendered a tiny 2-example task with the Flash-Next tokenizer, whose template is identical (see section 9):
```
<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nYou are solving an abstract reasoning puzzle. Each example has an INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually background). Infer the ONE rule that maps every input to its output, then apply it to the FINAL INPUT.\n\nExample 1 INPUT:\n1 0\n0 1\nExample 1 OUTPUT:\n0 1\n1 0\n\nExample 2 INPUT:\n2 2\nExample 2 OUTPUT:\n3 3\n\nFINAL INPUT:\n5 0 5\n\nReason step by step however you like. Then give ONLY the final output grid as rows of space-separated digits inside a fenced block:\n```grid\n<rows>\n```\nOutput nothing after the block.<|im_end|>\n<|im_start|>assistant\n<think>\n
```
Omitting `reasoning_effort` gives byte-identical output, because the default is xhigh.

---

### 2. Sampling

**Settings** (nb[2] line 49):
```python
SAMPLING = {"temperature": float(os.environ.get("Q27X_TEMPERATURE", 0.6)), "top_p": 0.95, "top_k": 20}
```
- **min_p is not set** (vLLM default 0.0).
- No repetition, presence or frequency penalty is set (defaults).
- No stop strings and no `min_tokens`. `ignore_eos` is at its default (False).

**SamplingParams** (nb[6] [CELL 5] lines 208–210):
```python
def make_params(max_tokens, seed):
    return SamplingParams(max_tokens=int(max_tokens), seed=int(seed), detokenize=False, skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA, **SAMPLING)
```

**Seeds.**
- `SEED = 20261005` (nb[2] line 26). The engine is also started with `seed=SEED` (nb[6] line 72).
- Per request: phase 1 uses `SEED + 10*UNIT_INDEX[uid] + 1` (nb[7] line 29). Phase 2 uses `SEED + 10*UNIT_INDEX[uid] + 2` (nb[7] line 45).
- `UNIT_INDEX` follows the sorted-task-id unit order, not the ORDER2 order (nb[4] line 45, comment "seeds do not depend on timing").

**Samples.**
- n = 1 per test input. One generation per unit, where a unit is one test input of one task.
- There is no voting, no resampling and no retry.

**max_tokens.**
- Phase 1: `P1_CAPS[uid] = phase1_cap(prompt_len)` (see section 3).
- Phase 2: `min(8192, MAX_MODEL_LEN - len(prefix) - 8)`.

**Note on the checkpoint defaults.** The checkpoint's `generation_config.json` has temperature 1.0, top_k 20, top_p 0.95. In-process SamplingParams overrides this with 0.6. Behind an OpenAI server, vLLM fills any field the request leaves out from generation_config, so every field must be sent explicitly.

---

### 3. Thinking budget and forced close

**Budget constants** (nb[2] lines 34–43):
- `THINK_BUDGET = 77824`
- `ANSWER_MAX_TOKENS = 8192`
- `DRAIN_ANSWER_MAX_TOKENS = 8192`
- `MAX_MODEL_LEN = 106496`
- `CONTEXT_SAFETY = 256`
- `MIN_THINK_BUDGET = 4096`
- `PROMPT_TOKEN_LIMIT = 106496-8192-256-4096 = 93,952`

**Phase-1 cap:**
```python
def phase1_cap(prompt_len):
    return max(0, min(THINK_BUDGET, MAX_MODEL_LEN - prompt_len - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
```
- The full 77,824 applies for prompts up to 20,224 tokens. The asserts check `phase1_cap(20000)==77824` and `phase1_cap(30000)==68048`.
- A unit is skipped as `"insufficient_context"` only if the cap is below 4096 (nb[4] lines 63–66). In EVAL120 the prompts had median 4,336 and max 16,400 tokens, and no budget was reduced.

**What "86k" means.**
- Phase 1 is a single request with `max_tokens = 77,824`. It covers thinking and, when thinking closes naturally, the answer as well, all in one request. Thinking does not stop at `</think>`.
- Phase 2 adds up to 8,192 tokens. 77,824 + 8,192 = 86,016.

**Forced-close string** (nb[2] line 37):
```python
STOP_THINKING_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
```
- It is tokenized once (nb[3] [CELL 2] lines 40–44) as `STOP_IDS = tokenizer.encode(STOP_THINKING_TEXT, add_special_tokens=False)`.
- `</think>` must be a single token: the code asserts `len(THINK_CLOSE_IDS)==1` and `THINK_CLOSE_ID in STOP_IDS`.
- With the Flash-Next tokenizer this gives 25 ids: `[271, 79939, 279, 6973, 854, 539, 279, 1156, 11, 353, 599, 310, 2873, 279, 6093, 3018, 383, 279, 7047, 5774, 1381, 13, 198, 248069, 271]`. `</think>` = 248069 sits at index 23, and the last token 271 = "\n\n".

**Phase-1 end handling** (nb[7] [CELL 6] lines 131–141):
```python
if phase == 1:
    st["p1_finish"] = fr
    closed = THINK_CLOSE_ID in st["p1_ids"]
    if closed and fr != "length":
        finish(st, "natural")
    elif not start_phase2(st, ANSWER_MAX_TOKENS):
        finish(st, "no_room_for_answer")
```

**Phase 2** (nb[7] lines 32–46):
```python
def start_phase2(st, answer_cap):
    closed = THINK_CLOSE_ID in st["p1_ids"]
    if not closed:
        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
            st["p1_ids"].pop()
    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
    room = MAX_MODEL_LEN - len(prefix) - 8
    cap = min(int(answer_cap), room)
    if cap < 64:
        return False
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    ...
    engine.add_request(_req(st["uid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
```
- The phase-2 prompt is token-exact: the rendered prompt ids (ending `<think>\n`) + all phase-1 ids + STOP_IDS.
- There is no chat-template re-render and no new user turn.
- **"forced_close"** happens when thinking never closed. That is usually because the length cap was hit; an EOS before `</think>` also counts. Trailing EOS ids are stripped first, and `EOS_IDS` comes from generation_config `eos_token_id` = {248046, 248044}.
- **"continue"** happens when thinking closed but the answer was cut at the length cap. The same prefix is used without the stop text and simply continues.
- Phase 2 uses the same sampling with up to 8,192 tokens. It ends on EOS or on length.
- There is no further continuation or resume beyond phase 2.
- With vLLM prefix caching on (mamba "align" mode), phase 2 reuses the phase-1 KV. The EVAL120 hit rate was 0.834.

**EVAL120 outcome** (records):

| Mode | Phase-1 / phase-2 finish | Units |
|---|---|---|
| natural | (stop, –) | 97 |
| forced_close | (length, stop) | 73 |
| forced_close | (length, length) | 1 |
| continue | (length, stop) | 1 |

Other run figures:
- mean think tokens 59,863.5; mean generated tokens 60,738.7
- generated p50 / p90 = 70,292 / 79,287
- live context p50 / p75 / p90 / max = 75,394 / 83,449 / 85,377 / 89,922

---

### 4. Answer parsing (nb[5] [CELL 4])

**Which text is parsed** (nb[7] lines 48–53):
```python
def answer_text(st):
    ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
    if THINK_CLOSE_ID not in ids:
        return "", 0, False
    k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[k + 1:], skip_special_tokens=True), k, True
```
- The text after the **last** `</think>` is decoded with `skip_special_tokens=True`, which removes `<|im_end|>`.
- If there is no `</think>` at all, the answer text is "" and the grid is None.

**Validation** (lines 2–5): the grid must be a list of 1–30 rows, each row a list of 1–30 cells, all rows the same length, and every cell an int (not a bool) in 0–9.

**Regexes** (lines 7–9):
```python
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")
```

**Row parser** (`_parse_row`, lines 11–19):
- a line of contiguous digits becomes one cell per digit ("102" → [1,0,2])
- otherwise, a line fully matching `_SEP_ROW` (single digits separated by space or comma, optional brackets and trailing comma) becomes its digits
- anything else gives None

**`parse_grid(text)`** (lines 72–92):
- **Step 1.** Take the fenced blocks, any label, from **last to first**. The first block that yields a valid grid wins.
  - Inside a block (`_grid_from_block`, lines 59–70): if the body starts with `[` or `{`, try `json.loads`. The `_grid_from_obj` helper takes dict keys in the order "output", "grid", "test_output", "answer", "test_outputs", or a grid, or the last grid in a list of grids.
  - If JSON fails, `_json_rows_grid` reads `[d, d, …]` rows one at a time. A `]` after a row closes a grid, and the last valid grid wins. This handles a missing bracket.
  - Otherwise the block uses `_first_row_run`: the **first** run of consecutive parseable rows, skipping header lines before it and stopping at the first non-row line after it.
- **Step 2.** If no block works, split the whole answer text into runs of consecutive parseable lines and return the **last** valid run. This covers unfenced answers and unclosed fences.
- Self-tests (lines 113–141) cover 12 layouts plus the ragged, two-digit-cell, empty and None cases.

**On parse failure** there is no retry and no re-ask. The grid is None, and the attempts fall back to the test input and its mirror (section 5). In EVAL120, 165 of 172 units had a grid and 7 used the test-input fallback.

---

### 5. attempt_1 / attempt_2, multiple test inputs, submission

**Attempts** (nb[5] lines 94–104):
```python
def make_attempts(uid, grid):
    inp = CHALLENGES[UNIT_TASK[uid]]["test"][UNIT_Q[uid]]["input"]
    cands = ([("model", grid)] if valid_grid(grid) else []) + [("test_input", inp), ("mirror", [list(reversed(r)) for r in inp])]
    uniq = []
    for src, g in cands:
        if all(g != u[1] for u in uniq):
            uniq.append((src, g))
    while len(uniq) < 2:
        uniq.append(uniq[-1])
    return {"attempt_1": uniq[0][1], "attempt_2": uniq[1][1], "src_1": uniq[0][0], "src_2": uniq[1][0]}
```
- **attempt_1** is the model grid.
- **attempt_2** is the test input. If the model grid equals the test input, attempt_2 is the left-right mirror instead (each row reversed).
- With no grid, the attempts are the test input and its mirror.
- In EVAL120, pass@1 was 58 and hits were 58, so attempt_2 never scored.

**Multiple test inputs.**
- Each test input is its own unit, `uid = f"{tid}#{qi}"` (nb[4] lines 40–44).
- Each unit gets a separate prompt containing all train pairs and only that test input, and the units are fully independent.
- Request ids are `f"{uid}|{phase}"` (nb[7] line 22).

**Submission** (nb[1] [CELL 0] lines 25–35; nb[7] lines 11–19):
- Format: `{task_id: [{"attempt_1": grid, "attempt_2": grid}, … one per test input in order]}`, written atomically (`.tmp` then `os.replace`) to `/kaggle/working/submission.json`.
- An identity fallback (test input in both attempts) is written in cell 0, before anything else runs.
- It is rewritten after every finished unit (`_record` → `write_submission`) and once more at the end of the scheduler.
- **Important:** in commit/eval mode (`not RERUN`), `write_submission()` writes only the identity submission of `arc-agi_test_challenges.json`. Model grids go into submission.json only in `KAGGLE_IS_COMPETITION_RERUN` mode. Eval answers live only in records.jsonl and run_report.json.

---

### 6. Scheduling

**Queue (ORDER2)** (nb[4] lines 46–53; queue at nb[7] line 97):
```python
def order_score(task):
    return int(len(task["train"]) >= 3) + int(len(task["test"]) == 1)
ORDER_SCORE = {tid: order_score(CHALLENGES[tid]) for tid in TASK_IDS}
ORDERED_TASKS = sorted(TASK_IDS, key=lambda tid: (-ORDER_SCORE[tid], TASK_IDS.index(tid)))
ORDERED_UNITS = [f"{tid}#{qi}" for tid in ORDERED_TASKS for qi in range(len(CHALLENGES[tid]["test"]))]
ORDER_ADMIT_RANK = {u: i for i, u in enumerate(ORDERED_UNITS)}
...
queue = [u for u in ORDERED_UNITS if u in PROMPT_IDS]   # ORDER2
```
- Higher score goes first. Ties keep the sorted task-id order, and a task's units stay together.
- The rank is computed at runtime from the task structure; there is no stored rank list.
- The rule's evidence script is `D:\ARC AGI 2\resources\Puzzle Capability Ledger\derive_ordering_rules.py`. The builder is `Experiment Notebooks/Build/build_e10_order2.py`, which patches cells 0, 2, 4 and 7 of `Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb` (sha `530cb119…`).
- I recomputed the order: it matches every `admit_rank` in the run's records.

**Concurrency** (nb[6] lines 202–204):
```python
CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
```
- `MAX_NUM_SEQS = 16`, `EXPECTED_LIVE_TOKENS_PER_SEQ = 43000`, and `kv_tokens = num_gpu_blocks * block_size` from `engine.vllm_config.cache_config`.
- In the run: kv_tokens 1,353,600 // 43,000 = 31, so the max_num_seqs cap of **16** applied (max_active 16, 0 preemptions).
- Concurrency counts units in phase 1 or phase 2.

**Admission** (nb[7] lines 116–117):
```python
while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
    admit(queue.pop(0))
```
Admission is continuous: a slot is refilled the moment a unit finishes, and there are no waves.

**Deadlines** (nb[2] lines 72–81). The EVAL120 and RERUN_ALL profiles have the same rows. `TOTAL_BUDGET_SEC = 12*3600 - 600 = 42,600 s`, measured from `NOTEBOOK_T0` (cell 0):

| Deadline | Formula | Time | Effect |
|---|---|---|---|
| HARD_STOP | T0 + 42600 − 360 | 704 min | abort everything and finish with whatever text exists |
| DRAIN_DEADLINE | HARD_STOP − 480 | 696 min | — |
| ADMIT_DEADLINE | DRAIN − 120 | 694 min | no new admissions |
| VERIFY_DEADLINE | T0 + 42600 − 180 | 707 min | reporting only |

The code asserts `T0 < ADMIT < DRAIN < HARD_STOP < VERIFY < T0 + budget`.

**Drain** (nb[7] lines 108–115):
- Every phase-1 request is aborted (`engine.abort_request`). The tokens already streamed are kept, and `p1_finish="drained"`.
- Then `start_phase2(s, DRAIN_ANSWER_MAX_TOKENS=8192)` runs: a forced close, or "continue" if `</think>` was already emitted. If there is no room, the unit ends as "drain_no_room".
- No admission happens after the drain.

**Hard stop** (lines 102–107): all active requests are aborted and `finish(s, "hard_stop")` parses whatever partial answer exists.

**Other details.**
- There is no per-request timeout; only the global deadlines exist.
- Stale outputs after an abort are ignored (`if st["phase"] != phase: continue`).
- Scheduler exceptions are caught (lines 156–180): finished units are kept and the submission is still written.
- EVAL120 run: all 172 units admitted, 0 drained, 0 hard-stop aborts, last unit done at 534 min, scheduler wall 31,343 s, 333.3 tok/s, MTP acceptance 0.669.

---

### 7. Scoring and report

**Per unit** (nb[5] lines 106–110):
```python
sol = SOLUTIONS[UNIT_TASK[uid]][UNIT_Q[uid]]
return {"hit": int(att["attempt_1"] == sol or att["attempt_2"] == sol), "pass1": int(att["attempt_1"] == sol)}
```
The comparison is exact equality of Python lists. Solutions come from `arc-agi_evaluation_solutions.json` (nb[4] line 4).

**Records:** `/kaggle/working/q27x/records.jsonl`, one JSON line per finished unit (nb[7] lines 75–93). Fields:
- unit, task_id, query, mode (natural / continue / forced_close), end_reason
- prompt_tokens, p1_tokens, p2_tokens, think_tokens, think_closed
- p1_finish, p2_finish, sec, answer_chars, has_grid
- src [src_1, src_2], score {hit, pass1} (None in rerun mode), p1_cap
- t_admit_min, t_done_min, order_score, admit_rank

**Report:** `q27x/run_report.json` (nb[8] [CELL 7]). It contains config, engine and scheduler, plus:
- planned and finished task/query counts, modes, think_closed_naturally
- mean_think_tokens, mean_total_gen_tokens, with_grid, first_attempt_source
- `eval_query_hits` and `eval_query_total` (all planned units; unfinished counts as a miss)
- `eval_score_pct` = 100 × hits / units
- `eval_task_normalized_pct` = 100 × Σ_task (hits_t / n_test_t) / n_tasks (lines 22–26)
- `eval_pass1_queries`, `eval_hit_units`
- the live-token and gen-token percentiles
- MTP draft and accepted tokens with acceptance rate, num_preemptions, prefix-cache queries, hits and hit rate
- projection fields and `deadlines_min`

**Other outputs:**
- `q27x/answers/{task}_q{qi}.txt`: end_reason/mode header, the last 1,500 thinking tokens, then the answer text. The original saves only the tail, not the full trace.
- `q27x/vllm_metrics.json` (Prometheus snapshots every 120 s plus a final one) and `q27x/vllm_frontend.log`
- `/kaggle/working/Q27X_BUNDLE.zip`, which holds all of q27x plus submission.json

---

### 8. Engine interface (what must be replaced)

**Current interface: in-process vLLM 0.19.0, token ids in and out** (nb[6]):
- `llm = vllm.LLM(**ENGINE_CONFIG, attention_config={"backend": …}, enable_prefix_caching=…, kv_cache_dtype=…, speculative_config=…)`, then `engine = llm.llm_engine`.
- `engine.add_request(request_id, {"prompt_token_ids": [...]}, SamplingParams)` (nb[7] lines 28, 45).
- `engine.step()` returns RequestOutputs. With `output_kind=DELTA` and `detokenize=False`, the loop reads `o.request_id`, `o.outputs[0].token_ids` (new ids only), `o.finished` and `o.outputs[0].finish_reason` ("stop" or "length").
- `engine.abort_request([ids])` is used at drain, hard stop and on error.
- `llm.get_metrics()` provides spec_decode, preemption, kv_cache, prefix_cache and token counters.
- `engine.vllm_config.cache_config.num_gpu_blocks/block_size` sizes the concurrency.
- `llm.generate(...)` runs a 16-token greedy smoke decode after each engine-ladder attempt.

**ENGINE_CONFIG** (lines 67–74): `tensor_parallel_size 4`, `distributed_executor_backend "mp"`, `dtype "auto"`, `load_format "safetensors"`, `language_model_only True`, `max_model_len 106496`, `max_num_seqs 16`, `max_num_batched_tokens 4096`, `gpu_memory_utilization 0.90`, `enforce_eager False`, `disable_custom_all_reduce True`, `seed 20261005`, `disable_log_stats False`.

**Engine ladder** (lines 134–141):
1. fp8_e4m3 KV + MTP3 + FLASHINFER (only if the libcuda link probe passes) + prefix caching
2. the same without prefix caching
3. fp8 + TRITON_ATTN
4. bf16 + FLASH_ATTN + MTP
5. bf16 without MTP

After a failed attempt, child processes are killed and the code waits for the GPUs to release memory.

**Environment:** `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, and a libcuda.so shim on LIBRARY_PATH.

**What an OpenAI-server port must provide.** The loop is token-exact, so the port needs:
- **Token-id prompts.** This means `/v1/completions` with `prompt` = list of ints, not `/v1/chat/completions`. The phase-2 prefix is prompt ids + phase-1 ids + STOP_IDS, which is a raw continuation.
- **Token-id outputs.** Re-tokenizing the decoded text is not guaranteed to reproduce the same ids. That would change the forced-close context and break prefix-cache reuse.
- Per-request `seed`, `top_k` (vLLM extra param), `skip_special_tokens=False`, and every sampling field sent explicitly.
- `finish_reason` that distinguishes "length" from "stop".
- **Streaming**, so that at drain the partial phase-1 tokens are in hand when the request is cancelled. With the in-process engine, DELTA outputs provided this.
- **Cancellation**: closing the HTTP stream or an abort.
- **KV capacity from the server**, to replace `kv_tokens // 43000`, plus metrics from the server's `/metrics` (spec decode, prefix cache, preemptions).

None of this server-side support exists in the notebook. Whether Merlin's OpenAI server accepts each of these parameters is not checked by any code here.

---

### 9. 27B-specific items that need a decision for Flash-Next

**Hard asserts that will fail on Flash-Next** (nb[3] lines 38–39):
```python
assert _model_cfg.get("model_type") == "qwen3_5"
assert (_model_cfg.get("quantization_config") or {}).get("quant_method") == "fp8"
```
The Flash-Next Merlin config is `model_type "qwen4_exp"`, `Qwen4ExpForConditionalGeneration`, max_position_embeddings 262,144, and it is W4A16 rather than fp8.

**Chat template.** The Flash-Next template is **byte-identical** to the 27B one; `diff` showed no differences against `_scratch/e13_rc2_local/input/models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1/chat_template.jinja`. So the xhigh system sentence and the `<think>\n` header are the same.

**Special token ids.** They are identical in both tokenizer configs: `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, `<think>` 248068, `</think>` 248069. `<think>` and `</think>` are not marked special, so `skip_special_tokens` does not remove them; this is why `answer_text` slices by id. The Flash-Next generation_config has the same eos [248046, 248044], temperature 1.0, top_k 20, top_p 0.95.

**Stop sentence.** `STOP_THINKING_TEXT` is the Qwen stop sentence. It tokenizes to 25 ids with the Flash-Next tokenizer, `</think>` is one token, and so the asserts would pass.

**Values tuned for the 27B that need a decision:**
- `EXPECTED_LIVE_TOKENS_PER_SEQ = 43000`, and the reading of `kv_tokens = num_gpu_blocks*block_size`. The 27B pool is shared by 1 attention group and 3 GDN groups with block 1,600 (nb[2] comments lines 62–65). Flash-Next's KV and state layout is different.
- `MAX_NUM_SEQS 16` and `MAX_NUM_BATCHED_TOKENS 4096`.
- MTP `{"method":"mtp","num_speculative_tokens":3}`, fp8_e4m3 KV, the FLASHINFER/libcuda shim, mamba "align" prefix caching, and `language_model_only=True`.
- The vLLM 0.19.0 / transformers 4.57.6 / tokenizers 0.22.2 / hf-hub 0.36.2 parity pins, the Python 3.12 gate, the wheelhouse, and `MODEL_PATH` (nb[2] lines 29–30; nb[3] lines 7–33).
- Budgets: `MAX_MODEL_LEN 106496` / `THINK_BUDGET 77824` / 8192. The asserts at nb[2] lines 45–48 are tied to these exact triples.

---

### Eval task ids (EVAL120)

The list is **not** embedded in the notebook. EVAL120 is `sorted(random.Random(20261005).sample(sorted(_ch), 120))` over `arc-agi_evaluation_challenges.json`. That file has exactly 120 tasks and 172 test inputs, so the sample is all of them; I confirmed it equals the 120 task ids in the run's records.

In ORDER2 admission order, by score (xN = number of test inputs, default 1):

**Score 2 (52 tasks):** 0934a4d8 136b0064 1818057f 271d71e2 28a6681f 291dc1e1 2ba387bc 2c181942 31f7f899 332f06d7 35ab12c3 3dc255db 409aa875 4c416de3 5545f144 58490d8a 58f5dbd5 5961cc34 6e453dd6 6ffbe589 7491f3cf 7b0280bc 7b3084d4 7c66cb00 800d221b 89565ca0 8b9c3697 8f215267 8f3a5a89 97d7923e 981571dc 9aaea919 9bbf930d a395ee82 a47bf94d a6f40cea aa4ec2a5 b5ca7ac4 b99e7126 b9e38dc0 bf45cf4b c4d067a0 d59b0160 da515329 db0c5428 db695cfb dbff022c dfadab01 e376de54 e87109e9 eee78d87 fc7cae8d

**Score 1 (53 tasks):** 135a2760 13e47133(x2) 142ca369(x2) 16b78196 16de56c4(x2) 195c6913(x2) 1ae2feb7(x3) 20270e3b(x2) 20a9e565(x2) 21897d95(x2) 247ef758(x2) 269e22fb(x2) 2b83f449 2d0172a1(x2) 36a08778(x2) 3e6067c3(x2) 45a5af55 4a21e3da(x2) 53fb4810 581f7754(x2) 64efde09 65b59efc(x2) 67e490f4 71e489b6(x2) 7666fa5d 78332cb0(x2) 7b5033c1 7b80bb43 80a900e0 8698868d 88bcf3b4(x2) 88e364bc(x2) 898e7135 8b7bacbf(x2) 9385bd28(x2) a251c730 a25697e4(x2) a32d8b75(x2) abc82100(x2) b0039139(x2) b10624e5 b6f77b65(x3) c7f57c3e cb2d8a2c(x2) d35bdbdc(x3) d8e07eb2(x2) dd6b8c4b(x2) de809cff e12f9a14(x2) e8686506 edb79dae f931b4a8(x2) faa9f03d(x2)

**Score 0 (15 tasks):** 221dfab4(x2) 38007db0(x2) 3a25b0d8(x2) 446ef5d2(x2) 4c3d4a41(x2) 4c7dc4dd(x2) 4e34c42c(x2) 5dbc8537(x2) 62593bfd(x2) 6e4f6532(x2) 7ed72f31(x2) 8e5c0c38(x2) cbebaa4b(x2) e3721c99(x2) f560132c(x2)

The notebook does embed fixed lists for the other profiles: MIX20 and PROXY56 at nb[2] lines 7 and 14.

---

### Files
- Notebook: `D:\ARC AGI 2\experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Notebooks\Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb`
- Builders: `...\Experiment Notebooks\Build\build_e10_order2.py` and `...\Build\build_e10_base.py`
- LB-26.94 source: `...\Build\source\Q27-XHIGH-v1_4_4_LB26.94.ipynb`
- Run outputs (I read only aggregates and records, no traces): `...\Experiment Results\Kaggle Runs\GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)\q27x\run_report.json` and `records.jsonl`
- ORDER2 rule evidence: `D:\ARC AGI 2\resources\Puzzle Capability Ledger\derive_ordering_rules.py`
- 27B template: `D:\ARC AGI 2\experiments\Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)\Experiment Notebooks\Build\Q27X_build\sources\model_qwen3-8-27b-fp8-repacked-v1\chat_template.jinja`
- Render and tokenization check script (scratch): `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\render_check.py`

No project files were edited. No thinking traces or `answers/*.txt` files were opened.