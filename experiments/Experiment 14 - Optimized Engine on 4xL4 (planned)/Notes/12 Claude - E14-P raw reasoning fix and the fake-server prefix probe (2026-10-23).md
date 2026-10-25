# E14-P: the raw-reasoning fix in arc2_duck, and the fake-server prefix probe (Claude, 2026-10-23)

**What the user approved (2026-10-23, "শুরু করো" / "start"):**
1. The Duck fix, as a switch.
2. Searching for the "earlier message edited" cause on CPU with a fake server.

**Scope:** no GPU, no Kaggle push, no notebook built or rebuilt.

## 1. The fix (arc2_duck.py, off by default)

**Files:**
- arc2_duck.py sha `7644aa06…` (rev 6g, = the G4B notebook manifest) → `758bddb5…`.
- Diff: `Experiment Review/E14-P arc2_duck raw reasoning switch.diff` (100 lines).
- Backup: `duck_arc2/arc2_duck.py.pre_e14p_backup`.
- Notebooks already built carry their own embedded copy and are unchanged.

**Switch:** `E12_RAW_REASONING_HISTORY=1`. Default `0`: rev 6g behaviour; the counters are not even written to `branch_result.json`.

**How it works:**
- `_chat_completion` keeps the server's raw reasoning of the latest reply. It uses the same key precedence as Duck's `_extract_reasoning_text`: `reasoning`, then `reasoning_content`.
- `_append_context_message` (already overridden by arc2) first calls `_arc2_raw_reasoning_message`. That function replaces the reasoning keys of the **assistant** message with the raw text only if all of these hold:
  - the message is an assistant message (a user or tool message never consumes the capture);
  - `normalize(raw) ==` the stored text (it is the same reply); otherwise nothing changes and `mismatch` is counted;
  - Duck did not strip tool-call markup out of it; otherwise its text is kept and `markup` is counted.
- If the raw text equals the normalized text, nothing changes (`same`).
- Transcript, metrics and the token estimate still use Duck's normalized text.
- **Counters:** `branch_result.json` → `raw_reasoning` {kept, same, mismatch, markup}, only when the switch is on.

**Unit checks:** `src/analysis/test_raw_reasoning.py`, **15/15 PASS**:
- off / unset = unchanged;
- on = raw stored; content and tool calls untouched; the original dict not mutated;
- the capture is consumed once; a user message does not consume it;
- a mismatch, markup, identical text or a message without reasoning is left alone;
- the real `_append_context_message` path works;
- `raw|trim` keeps the inner blank lines.

**Regression (E12 smoke tests):**
- The tests were run on copies of the **old** (7644aa06) and **new** (758bddb5) code, because the bundle's own `duck_arc2` would shadow the workspace file.
- **Identical PASS/FAIL lines:**

  | Test | Result (old and new) |
  |---|---|
  | smoke_adapter | 41 PASS / 0 FAIL |
  | smoke_rev5 | 56 PASS / 0 FAIL |
  | smoke_rev6 | 66 PASS / 1 FAIL + TypeError |

- The smoke_rev6 FAIL ("V4: show() over the limit refused") and `TypeError: solve_all_r6() got an unexpected keyword argument 'min_workers'` are on the **old** code too. That test file is from the rev 6c era and was not updated for the later rev 6 changes. They are not caused by this fix.

## 2. The fake-server prefix probe

**Files:**
- `src/analysis/fake_prefix_server.py`
- `src/analysis/run_prefix_probe.py`

**How it works:**
- **The server renders every request with the real INT4 chat template**: transformers `apply_chat_template`; `preserve_thinking=True` as the server default; the request's `chat_template_kwargs`; tools; tool-call arguments parsed to dicts, as SGLang does.
- It compares the rendered text, character by character, with **previous rendered prompt + the generated reply text**. The generated text is reasoning with paragraph breaks, then `\n</think>\n\n` + the qwen3_coder tool-call markup + `<|im_end|>`.
- The response carries `reasoning_content` = reasoning + "\n", as SGLang's parser returns it.
- So this checks **everything** that goes into the prompt, not only the message fields the old harness compared: tools, arguments, template kwargs, reasoning effort and the system prompt.

**What the driver runs:**
- Real `arc2_duck.BranchSession`s: rev 5 + rev 6, Duck's own loop, yields every 2,048 tokens (new turns, `_persistent_history_messages`), openers, the cut and no-call paths.
- 2 tasks × 2 branches (facts and grids views).
- 8 requests per branch.
- Program: inspect, cut at the output limit without a call, check(transform), inspect, a plain reply without a call, inspect, check, inspect.

**Result:**

| Switch | Follow-up requests | full | in_reply | before_reply (client edit) | cut_reply |
|---|---|---|---|---|---|
| 0 (as run on Kaggle) | 28 | 0 | **24**, all at reply offset 57–58 inside the reasoning = the first blank line (expected `"\nThe first grid…"`, got `"The first grid…"`) | **0** | 4 (match stops at 57) |
| 1 (fix) | 28 | **24** | 0 | **0** | 4 (the whole cut reasoning, 217 chars, matches; the template then adds the `</think>` the model never wrote) |

Counters with the switch on: `kept` 8 per branch, `mismatch` 0, `markup` 0.

**Conclusions:**
1. **The normalization is the cause of the reply re-prefill, confirmed on CPU end to end.** Every normal reply diverged exactly at its first paragraph break. With the switch on, the next prompt contains the whole previous prompt + reply as an exact text prefix.
2. **Duck does not edit earlier content** in these paths: 0 `before_reply` in 56 follow-ups. That covers the system prompt, tools, tool arguments, template kwargs, openers and new turns.
   - So the 18 G4B deeper misses (66,771 tokens) are **not** a client edit.
   - Together with Notes/11 (10 of 18 land within 64 tokens of an older request's checkpoint; the 5 k=1 cases land on the shared system prefix), they point to **Mamba checkpoint eviction** (README E14-M, Franzen's refresh-on-unlock).
   - Not covered by the probe: history trims (a prefix change by design; 1 in G4B) and the rev 5 opener rewrite after a trim (4 events in G4B).
3. **What is still not proven:** the token-id level. Text equality is necessary; the server also needs the generated ids to equal the re-tokenized ids (canonical tokenization; candidate 1 of Notes/06). The cut reply is still a miss by design.
   - **E14-A replay check (a few minutes, our own client, `return_output_ids_in_sglext`):** raw vs normalized reasoning → `cached_tokens` and the first divergent id.

## 3. Status and next decisions

- **Switch in rev 6g notebooks:** off. Enabling it means a new notebook variant (never a rebuild of a run notebook) with `E12_RAW_REASONING_HISTORY=1`. Best after the E14-A replay confirms the server-side gain.
- **Expected gain (G4B scale):** up to 217.5k re-prefilled tokens (≈ 146 s of prefill) removed. Working estimate: ~+10 % effective throughput.
- **The deeper misses** belong to E14-M (Mamba checkpoint protection), a server-side change, measured in E14-A.
