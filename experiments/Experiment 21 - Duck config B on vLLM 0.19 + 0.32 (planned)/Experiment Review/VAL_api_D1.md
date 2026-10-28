# VAL_api_D1 — Duck <-> vLLM HTTP contract (vLLM 0.19.0 and 0.32.0)

Date: 2026-10-28. Read-only validation. No project file was edited and Kaggle was not touched.

## Scope and method

- **Duck package under test:** `D:/ARC AGI 2/_scratch/donor_map/duck_9e4b/`. The sha256 of `arc2_duck.py` (b9f67010…), `tool_agent.py` (f7454201…), `openai_compat.py` (fea0a750…) and `prompts.py` (14965a99…) equals the `_E12_MANIFEST` in NB1 CELL 3. This is the package that R3K2 ran and that E21 ships.
- **E21 code:** the generated NB1 notebook (`Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb`, cells extracted), `build_e21.py`, `e21_engine_new.py`, `src/e21_flags.json`, `src/e21_flagcheck.py` and `src/harness/fake_vllm_duck_server.py`.
- **vLLM source:** fetched raw from GitHub at the tags `v0.19.0` and `v0.32.0`. The copies in `_scratch/vllm_src/*/full/` are byte-identical: `entrypoints/chat_utils.py` (0.19) and `renderers/chat_utils.py` (0.32).
- **Evidence runs:**
  - E12-V vLLM 0.19 SMOKE1 run 2 (`requests.jsonl` of both branches);
  - E15B R3K2 and R3K1 EVAL120 transcripts on SGLang. There were 2,303 and 2,167 `MODEL RESPONSE META` blocks, all counted.
- **Template check:** I rendered every smoke request locally with the Qwen3.8 `chat_template.jinja` (sha c3cf9e34, the INT4 and FP8 copies are identical) and the FP8 `tokenizer.json`. The rendering emulates vLLM: `reasoning` is copied to `reasoning_content`, tool-call argument strings become dicts, and content uses the openai list form.
  - Result: **12 of 12 local token counts equal the server's `usage.prompt_tokens`** (6,882 / 19,215 / 26,342 / 27,725 / 30,138 / 30,527 / 30,684 and 6,882 / 19,965 / 20,872 / 21,099 / 21,280).
  - So the emulation is exact for vLLM 0.19.
  - Script: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\render_check.py`.

---

## Findings

### F1 — HIGH — vLLM reports `finish_reason: "tool_calls"` for a tool call cut at max_tokens. All of arc2_duck's cut-call handling keys on `"length"`, so it is silently disabled on both vLLM versions

**Evidence (vLLM, both versions).** The serving layer replaces the engine's finish reason with `"tool_calls"` whenever the parser extracted any call:
- `vllm/entrypoints/openai/chat_completion/serving.py` v0.19.0 L1550-1560;
- the same file in v0.32.0, L1099-1115.

```python
is_finish_reason_tool_calls = auto_tools_called or (...)
...
finish_reason="tool_calls" if is_finish_reason_tool_calls else output.finish_reason if output.finish_reason else "stop",
```

`auto_tools_called = tool_calls is not None and len(tool_calls) > 0`. Neither version checks `output.finish_reason == "length"`.

**Evidence (the parsers accept unterminated calls).**
- **v0.19.0** `tool_parsers/qwen3coder_tool_parser.py` L59-68. Every regex has an end-of-text branch, so a cut call is parsed with the partial code as valid JSON:
  - `<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$`
  - `<function=(.*?)</function>|<function=(.*)$`
  - `<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)`
  - The parameter values lose one wrapping `\n` (L262-266) and are `json.dumps`-ed (L270-276).
- **v0.32.0** `parser/engine/streaming_parser_engine.py` L310-328: `finish()` emits `TOOL_CALL_END` for a still-open call.
  - `parser/engine/parser_engine.py` L1119 then builds its arguments with `converter(raw_body, False)`.
  - `parser/qwen3.py` L51-56: `_PARAM_RE` needs `</parameter>` or a following `<parameter=`, so the unclosed last parameter is dropped.
  - The call becomes `python` with arguments `"{}"`.
- I reproduced both outcomes with the verbatim regexes (scratchpad `cut_call_test.py`):
  - 0.19 gives `{"name":"python","arguments":"{\"code\": \"def transform(grid):\\n ... out[r][0] = \"}"}`;
  - 0.32 gives `{}`.

**What SGLang 0.5.21 did.**
- R3K2 transcripts: 30 replies had `finish_reason: length` together with a cut tool call; all 30 have `tool_call_argument_issues`, i.e. they were turned into a placeholder by B6.
- `cut_tool_calls_dropped` totals 30, and `cut_code_saved` is 30 over the branches.
- R3K1: 14 such replies.
- Out of 2,303 replies in R3K2: 217 `length` (all without a call after B6), 2,085 `tool_calls` and 1 `stop`.

**The code that keys on "length":**
- `arc2_duck.py` L1911-1997 (rev 5 B6 / R5-2): makes cut calls invalid JSON, saves `b_cut_call_code` for the rev 3h harvest (L1991), and sets `_arc2_cut_pending`, which gives the R6 `"cut"` opener (L1665-1676).
- L1952-1971 (`length_cut_no_call` / `_arc2_nocall_cuts`, the rescue cap at L1794).
- `tool_agent.py` L6836 (effort ladder, unused) and L6895 (`truncated_tool_call`, which gives the "CUT OFF" follow-up at L6992).

**Failure scenario on vLLM.** About 1.3 % of replies (~30 per EVAL120 run) end at the request cap inside a tool call. In config B this is typically the last request of an attempt, where the cap is the remaining budget minus the reserve.
- **0.19:** the partial code is dispatched as a normal call. Most often it gives `Python syntax error` (arc2 L2079-2082). A cut that lands on a line boundary runs a truncated program, which is exactly the "3-line stub" B6 was built to stop. Either way the model gets the syntax error or the stub's output instead of R6 `"cut"` ("…that call did NOT run…"). `_arc2_b_cut_code` stays empty, `cut_tool_calls_dropped` stays 0, and the cut call with its partial code is committed to history.
- **0.32:** the call arrives as `python({})`, giving `"python requires a non-empty code string."` (arc2 L2076-2078). The cut code is lost entirely: it is not in history, not in the harvest, and not in the transcript.
- **Impact measured on R3K2:** the harvest used cut code in 24 branches and produced 0 new accepts, so the direct score loss is probably small. But the attempt-end turn flow differs from the tuned R3K2 config, and the counters that would reveal it stay at 0.

**Why the local harness cannot see it.** `src/harness/fake_vllm_duck_server.py` L523-524 models SGLang semantics, not vLLM:

```python
msg = {"role": "assistant", "content": None, "reasoning": R + "\n", "tool_calls": calls}
fin = "length" if code == CUT_CODE else "tool_calls"
```

**Fix.** No package change is needed, so the manifest hashes stay valid. In CELL 3, right after `_E12_TA, _E12_DP, _e12_run_sandboxed, _ = E12A._import_duck()`, restore "length" from the token count:

```python
# E21 (VAL_api_D1 F1): vLLM 0.19/0.32 return finish_reason "tool_calls" for a call cut at max_tokens (serving.py
# 0.19 L1550-1560, 0.32 L1099-1115); SGLang 0.5.21 kept "length", which arc2_duck's cut handling keys on.
_e21_cc_orig = _E12_TA.ToolAgent._chat_completion
E21_LENGTH_RESTORED = {"n": 0}
def _e21_chat_completion(self, messages, **kw):
    res = _e21_cc_orig(self, messages, **kw)
    try:
        u = res.usage if isinstance(res.usage, dict) else {}
        cap = kw.get("max_tokens")
        cap = self._max_output_tokens if cap is None else cap
        ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        if str(res.finish_reason or "") == "tool_calls" and cap and isinstance(ct, int):
            lim = int(cap) if not isinstance(pt, int) else min(int(cap), E21_VLLM_MAX_MODEL_LEN - pt)  # 0.32 clamps
            if ct >= lim:
                res.finish_reason = "length"
                E21_LENGTH_RESTORED["n"] += 1
    except Exception:
        pass
    return res
_E12_TA.ToolAgent._chat_completion = _e21_chat_completion
```

- arc2's override calls `super()._chat_completion(...)` and only then inspects the result, so the patched base runs first and B6 / R5-2 / harvest / opener all work again.
- A reply that ended exactly at the cap with a complete call is also treated as cut. SGLang did the same.
- Record `E21_LENGTH_RESTORED` in the report.
- **Harness fix:** for `CUT_CODE`, the fake server should return `finish_reason "tool_calls"` with `completion_tokens == max_tokens` (now `n_out = min(2100, mt)`, L532). Add an assertion that `cut_tool_calls_dropped > 0` in that mode.
- **Residual on 0.32:** the cut code text is already gone from the response, because vLLM dropped the unclosed parameter. The harvest of cut code cannot be restored there. It contributed 0 accepts in R3K2.

### F2 — HIGH — No real-server check of the Duck tool/reasoning contract on vLLM 0.32 before a ~5.5-h half

**Evidence.**
- vLLM 0.32 serves `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` with a new unified parser engine:
  - `reasoning/__init__.py` L155-156 and `tool_parsers/__init__.py` L193-196 both map to `parser/engine/registered_adapters.Qwen3Parser*`, which wraps `parser/qwen3.py`;
  - it replaces the 0.19 `Qwen3ReasoningParser` + `Qwen3CoderToolParser`.
- None of our runs exercised it with tools and thinking:
  - VMS / E18 are Poetiq (no tools);
  - E12-V run 2 is vLLM 0.19.
- The launch smoke in `e21_engine_new.py` L117-125 sends no tools and `enable_thinking: false`. It checks only `content` and the integer usage. So a 0.32 contract failure would pass the smoke and run the whole half. The fallback config only triggers on launch or smoke failure. Possible failures:
  - `message.reasoning` empty;
  - calls not parsed;
  - history `reasoning` dropped by request validation;
  - a template error on the history.
- The local and Kaggle CPU harnesses use the fake server, so they cannot detect a real-parser difference either. F1 is one such difference that they already miss.

**Fix.** Add a two-turn contract probe to `e21_launch` after the existing smoke (about 1 minute at 512 tokens; temperature 0; the real Duck tools via `E12A.make_agent_class()`'s `_tools`, or a one-tool copy of `{"type":"function","function":{"name":"python",...,"parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}`; `tool_choice:"auto"`; `chat_template_kwargs {"enable_thinking": true, "preserve_thinking": true}`):
1. **Turn 1** asks for "call python with print(1)". Assert all of:
   - `message.reasoning` is a non-empty str;
   - `finish_reason` is in {`tool_calls`, `length`};
   - when there are calls, `json.loads(arguments)` is a dict with a str `code`.
2. **Turn 2** appends `{"role":"assistant","reasoning":R,"tool_calls":calls}` + `{"role":"tool","tool_call_id":id,"content":"1"}`. Assert:
   - `usage.prompt_tokens >= turn1.prompt_tokens + turn1.completion_tokens - 8`. This proves that the history thinking and the call rendered. A dropped `reasoning` shows up as a shortfall of about the reasoning length, the E12-V method.
3. A failure counts as a smoke failure, which triggers the next config, and is recorded in `info["contract"]`.
- It is the same check for 0.19, so add it to both versions.

### F3 — MEDIUM — vLLM 0.19 hybrid prefix cache never reuses the previous reply. The prompt prefix itself is byte-stable

**Evidence (the prefix is stable).** In both smoke branches, every request's rendered prompt is an exact character and token prefix of the next request's prompt. In my render, `prev_prompt_is_prefix: true` and common tokens equal the previous prompt length in all 11 follow-ups, with token counts equal to the server's.
- **Template:** with `preserve_thinking: true` every assistant turn renders `'<|im_start|>assistant\n<think>\n' + reasoning_content|trim + '\n</think>\n\n' + content` (template, assistant branch). vLLM supplies `reasoning_content` from the request's `reasoning`:
  - 0.19 `entrypoints/chat_utils.py` L1512 and L1537-1541;
  - 0.32 `renderers/chat_utils.py` L2083 and L2110-2114.
- **Tools block:** rendered once in the system turn and identical across turns.
- **Tool-call ids:** never rendered.
- So Duck's prompts are append-only, and history thinking renders.

**Evidence (what the cache reuses).** Over all 11 follow-up requests, `cached_tokens = (floor(P_prev / 1600) - 1) * 1600` exactly, where P_prev is the previous request's prompt length:

| prev prompt | cached (server) | formula |
|---|---|---|
| 6,882 | 4,800 | 3 × 1600 |
| 19,215 | 17,600 | 11 × 1600 |
| 26,342 | 24,000 | 15 × 1600 |
| 27,725 | 25,600 | 16 × 1600 |
| 30,138 | 27,200 | 17 × 1600 |
| 30,527 | 28,800 | 18 × 1600 |
| 19,965 | 17,600 | 11 × 1600 |
| 20,872 | 19,200 | 12 × 1600 |
| 21,099 | 19,200 | 12 × 1600 |

(Branch b1 request 1 is the first request, null. The b0 request 0 hit of 3,200 is the shared system prefix.)

- None of the generated reply tokens are ever hit, and the last full prompt block is also lost.
- This matches `mamba_cache_mode` "align" (0.19 `config/cache.py` L109-117: "only cache the mamba state of the last token of each scheduler step and when the token is at position i * block_size").

**Consequences.**
- Every Duck turn re-prefills the whole previous reply, the tool result and about one block. A 30k-token thinking reply is prefilled again on the next request. SGLang's mamba radix cache reached about 93 %.
- E14-P (`E12_RAW_REASONING_HISTORY=1`) cannot raise the hit rate on vLLM 0.19: its purpose was to make the history match the generated tokens. It still keeps what the model reads identical to R3K2, so keep it on.
- `e21_capacity` sizes streams by KV only. It does not count the extra prefill compute per turn.
- On 0.32 (`cache.py` L189-202, "align" default, plus an opt-in `enable_mamba_shared_prefix_checkpoint` for EAGLE/MTP siblings) this is unmeasured.

**Fix.** There is nothing to change at the HTTP level. Report, per half, the measured `cached/prompt` and the same formula fit from `arc2_requests` (prompt / cached / prev). Treat throughput differences versus R3K2 in that light.

### F4 — LOW (rare) — vLLM 0.32 ends reasoning at a `<tool_call>` emitted inside `<think>`. SGLang and vLLM 0.19 keep it inside reasoning

**Evidence.**
- 0.32 `parser/qwen3.py` L140-144: `(ParserState.REASONING, "TOOL_START") -> TOOL_PREAMBLE, (REASONING_END, TOOL_CALL_START)`, "Tool call directly from reasoning (implicit end)". `<tool_call>` (id 248058, `special=False`) is a token-id terminal (L118-123).
- Text after that call's `</tool_call>` is content, and a later `</think>` is dropped silently (L134-138).
- 0.19 `reasoning/qwen3_reasoning_parser.py` `extract_reasoning`: everything before `</think>` is reasoning. Tools are parsed from the content only (0.19 `chat_completion/serving.py` L1375-1376: "tool calls are extracted exclusively from the content").

**Failure scenario on 0.32.** A draft call written while thinking is executed as a real call, and so is any real call after `</think>`. On SGLang and 0.19, Duck recovers markup calls only when no parsed call exists (`tool_agent.py` L6855-6857). The rest of the thinking moves to `content`: Duck stores it as visible assistant text and feeds it to `_update_summarized_knowledge_from_assistant`. A cut inside such a call also falls under F1.

**Frequency.** R3K2 on SGLang had `tool_call_markup_in_text: yes` in 1 of 2,303 replies.

**Fix.** None needed now. Count, on half B, the replies that have `tool_calls` and also non-empty content after normalization.

### F5 — LOW — prompt + max_tokens > max-model-len: 0.19 rejects with 400, 0.32 clamps silently

**Evidence.**
- **0.19:** `renderers/params.py` L188-193 (`max_input_tokens = max_total_tokens - max_output_tokens`) and L407-428 reject with "This model's maximum context length is … Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, …)". Duck routes this to `force_drain` (`tool_agent.py` L6812-6834), which trims history even when only the cap was too large.
- **0.32:** `renderers/params.py` L209-220 ("max_output_tokens is *not* subtracted here … clamping … get_max_tokens()"). `entrypoints/serve/utils/api_utils.py` L169-195 then gives max_tokens = min(cap, max_model_len − prompt). That ends in finish "length" below the requested cap. With F1's fix, the `min(cap, max_model_len - prompt)` limit covers this case.

**Reachability.** Not reached in practice:
- arc2 sets the trim budget to `window - cap - margin` (`arc2_duck.py` L1803, window 100,000 from `E15B["E12_B_WINDOW"]`), so prompt + cap ≤ ~100k against `--max-model-len 135168`;
- it would need an estimator error above 35k.
- The fake server (L438-446) applies the 0.19 rule to both versions.

**Fix.** None needed.

### F6 — LOW — `prompt_tokens_details` presence differs (0.19: only when cached > 0; 0.32: whenever enabled)

**Evidence.**
- 0.19 `chat_completion/serving.py` L1601: `if self.enable_prompt_tokens_details and final_res.num_cached_tokens:`. The smoke shows `"prompt_tokens_details": null` on cold requests.
- 0.32 `serving.py` L90-109: the object is returned whenever `num_cached_tokens` is not None, and it also carries `created_cache_tokens`.
- arc2 reads it defensively (`arc2_duck.py` L1554-1557 and L2054-2059: `cached = 0`, `cached_reported` flag). The comment there names SGLang, but the handling is generic.

**Fix.** None. Optionally, have the fake server emit the 0.32 shape for half B.

### F7 — LOW (out of the HTTP scope, found while checking flags) — `e21_flagcheck.py` cannot import the 0.32 argument parser

**Evidence.** `src/e21_flagcheck.py` L21: `from vllm.entrypoints.openai.cli_args import make_arg_parser`. The git tree of v0.32.0 has no `vllm/entrypoints/openai/cli_args.py`; the parser is now `vllm/entrypoints/launchers/cli_args.py` (`make_arg_parser` at L421). `vllm/utils/argparse_utils.py` exists in both versions.

**Failure scenario.** The 0.32 run of the kit's flagcheck ImportErrors before any command is parsed, so `v032.summary.all_ok` is false. The kit fails loudly, not silently.

**Fix.**
```python
try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
except ImportError:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
```

### F8 — LOW — Duck's comment about kwargs "replacement" is wrong for vLLM (harmless)

`tool_agent.py` L1325-1333 says a request-level `chat_template_kwargs` replaces `--default-chat-template-kwargs` on vLLM. In both versions the two are merged, and the request wins per key:
- `ChatParams.with_defaults` → `merge_kwargs(default_chat_template_kwargs, self.chat_template_kwargs)`: 0.19 `renderers/params.py` L90-118 and L28-41; 0.32 L104-134 and L36-48;
- the parser's kwargs: 0.19 `engine/serving.py` L798-807 (`default | request`); 0.32 `chat_completion/serving.py` L192-202.

Duck sends `preserve_thinking: true` itself, so the result is the same.

### F9 — LOW / info — SGLang-only serving flags have no vLLM counterpart

- `--schedule-policy lpm` (longest-prefix-match scheduling) does not exist in vLLM, which schedules FCFS. Branch admission order on a shared prefix can differ.
- `--enable-cache-report` ↔ `--enable-prompt-tokens-details`: equivalent.
- The `--default-chat-template-kwargs`, `--reasoning-parser` and `--tool-call-parser` flags have the same meaning on both engines.
- vLLM additionally needs `--enable-auto-tool-choice` for `tool_choice:"auto"` (0.19 `entrypoints/serve/render/serving.py` L204-221). It is set in all 8 commands of `e21_flags.json`.

---

## Verified (no issue)

### Request (`openai_compat.build_chat_payload` L38-98 + `tool_agent._chat_completion` L5228-5290)

The provider is `"vllm"` on SGLang (R3K2) and on vLLM alike, so the body is byte-identical to R3K2 apart from the history key.

| Field Duck sends | Value in E21 | vLLM 0.19 / 0.32 behaviour |
|---|---|---|
| `model` | `/v1/models` `data[0].id` (NB1 CELL 6) = `E21_SERVED[...]` | a match is required (`_check_model`); correct by construction |
| `messages` | system / user / assistant(`reasoning`, `content` or none, `tool_calls`) / tool(`tool_call_id`, `content`) | `CustomChatCompletionMessageParam` has `reasoning`, `tool_calls` and `tool_call_id` (0.19 `chat_utils.py` L274-296; 0.32 L361-383). `reasoning` is copied to `reasoning`+`reasoning_content` for the template (0.19 L1537-1541; 0.32 L2110-2114). Tool-call args str → dict via `json.loads` (0.19 L1555-1580; 0.32 L2153-2215, where 0.32 coerces malformed args to `{}` instead of a 400). Duck never commits malformed args (`tool_agent.py` L6870-6896). Empirically: the smoke token counts match with reasoning rendered (19,215 = 6,882 + 12,288 + 45) |
| history key | `ARC3_REASONING_HISTORY_KEY=reasoning` (CELL 3 + `e21_begin_half`); read at call time (`tool_agent.py` L1262-1278) | 0.19 `chat_utils.py` L1512 and 0.32 `renderers/chat_utils.py` L2083 read only `message["reasoning"]`; the E21 README Finding 1 line numbers are correct. The harvest (`arc2_duck.py` L2335), E14-P (L1742) and rev 5 counting (L1859) all read via `_reasoning_history_keys()` |
| `tools` | one `python` tool, `code: string`, required | rendered once in the system turn; no structural-tag / JSON-schema constraint for `"auto"` (0.32 `abstract_tool_parser.py` L122-160: `get_json_schema_from_tools` only for required/named; the unified parser's `adjust_request` only sets `skip_special_tokens=False`, `parser_engine.py` L227-231). `validate_tool_names` defaults to False (0.32 `parser_engine_config.py` L110), so `submit`/`check` called as tools still reach arc2's router (`arc2_duck.py` L1575-1610); 0.19 does no name validation |
| `tool_choice` | `"auto"` (`_request_tool_choice`) | needs `--enable-auto-tool-choice` + parser: set |
| `max_tokens` | per-request cap (rev 6 caps / B budget) | deprecated alias of `max_completion_tokens` and still honoured (0.19 `protocol.py` L159-164 + L380-396; 0.32 L657-675) |
| `temperature` / `top_p` / `top_k` | 1.0 / 0.95 / 20 | passed to `SamplingParams` (0.19 `protocol.py` L425-520). `--generation-config vllm` gives neutral defaults (`config/model.py` L1391-1408), while the eos list [248046, 248044] is still loaded from `generation_config.json` (`try_get_generation_config` L1370-1377; `input_processor.py` L274). The model's `generation_config.json` holds only temperature 1.0 / top_k 20 / top_p 0.95, so no sampling default differs from SGLang's |
| `seed` | not sent (`LOCAL_ANALYZER_SEED` = -1, `tool_agent.py` L204; payload L95) | per-request RNG; the attempts of a task are not correlated |
| `priority` | not sent (`build_chat_payload` is called without it; only one POST site, `tool_agent.py` L5274) | a non-zero value would error without priority scheduling (0.19 `protocol.py` L285-293); not applicable |
| `chat_template_kwargs` | `{"enable_thinking": true, "preserve_thinking": true}` (+ `reasoning_effort` only if `E12_REASONING_EFFORT` is set, which it is not) | merged over the server default (F8). Top-level `reasoning_effort` stays None, so `merge_kwargs` skips None and the kwarg survives (0.19 `protocol.py` L360-377; 0.32 L621-655; 0.32 adds `enable_thinking` only when it is absent from the user kwargs). Template vars are filtered (`renderers/hf.py` 0.19 L407-435): `enable_thinking`, `preserve_thinking` and `reasoning_effort` are template variables and pass. Unset effort gives the template default `xhigh` (template L46-56); only xhigh / medium / low are legal there (0.32 turns an illegal one into a 400, `hf.py` L834-838) |
| `stream` | `false`; no `stream_options` | non-streaming full response; the SSE assembler is unused |
| client timeout / disconnect | read timeout clamped to the deadline (arc2 L1916-1925) | `@with_cancellation` aborts the generation when the client disconnects (0.19 `chat_completion/api_router.py` L51-53, `entrypoints/utils.py` L56-90; 0.32 `api_router.py` L52-54), so no orphan decode after a clamp |

### Response (what Duck reads)

- **`choices[0].message.reasoning`:** the field is named `reasoning` in both versions (0.19 `protocol.py` L54-64; 0.32 L69-86). Duck and E14-P read `reasoning` first, then `reasoning_content` (`tool_agent.py` L3439-3443; `arc2_duck.py` L1946-1950).
  - 0.19: reasoning = the text before `</think>`, raw. Without `</think>`, all of it is reasoning (`qwen3_reasoning_parser.py` `extract_reasoning`).
  - 0.32: `strip_trailing_reasoning_whitespace=False` (`parser/qwen3.py` L200); the template trims anyway.
  - `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are added tokens with `special=False` (`tokenizer.json`), so they survive `skip_special_tokens=True` in 0.19.
- **`content`:** 0.19 returns the text before `<tool_call>` (often `"\n\n"`, `qwen3coder_tool_parser.py` L335 content slice); 0.32 strips or drops whitespace-only content. Duck normalizes either way (`_normalize_message_content`), so for a call reply the history has no `content` key, as on SGLang.
- **`tool_calls`:**
  - 0.19: `[{"id":"chatcmpl-tool-…","type":"function","function":{"name","arguments": JSON str}}]` (smoke `requests.jsonl`), and `[]` when there is none;
  - 0.32: the key is omitted when empty (`chat_completion/protocol.py` L81-86, `ChatMessage._serialize`).
  - Duck uses `.get("tool_calls") or []`.
  - Multiple calls per reply are parsed by both: R3K2 had 214 multi-call replies; in 0.19 `parallel_tool_calls` defaults to True.
  - String parameters round-trip exactly: one leading and one trailing `\n` are stripped (0.19 L262-266; 0.32 `parser/qwen3.py` L60-66, `_trim_wrapping_newlines`), and the template re-adds them.
- **`finish_reason`:** `stop` / `length` / `tool_calls`. See F1 for `length` with a call. `length` without a call is reported as `length` (smoke b0 request 0: 12,288 = cap).
- **`usage`:** `prompt_tokens` / `completion_tokens` / `total_tokens` are integers in both (Duck counting, `tool_agent.py` L3640-3680). For `prompt_tokens_details.cached_tokens` see F6.
- **Errors (prompt too long):** Duck's matcher (`tool_agent.py` L3446-3456) is a lowercase substring test over `"{HTTPError} | response: {body}"`. Every vLLM overflow message contains `"maximum context length"`:
  - 0.19 `renderers/params.py` L327-335 (char pre-check), L419-426 (tokens), `engine/serving.py` L752-771, `entrypoints/utils.py` L182-185 ("Input length (X) exceeds model's maximum context length (Y).");
  - 0.32 `renderers/params.py` L389-395, L528-534, `serve/utils/api_utils.py` L184-187.
  - The 0.19 token message also matches `"reduce the length of the input prompt"` and `"parameter=input_tokens"` (`VLLMValidationError.__str__`, `exceptions.py` L30-35).
  - HTTP status is 400 (0.32 `exception_handling/error_response.py` `VLLMValidationError` → BadRequestError / 400).
  - The only vLLM length error without the phrase is `max_tokens > max_model_len` (0.19 `params.py` L201-213), which Duck's caps (≤ 50k) never reach.
  - The fake server's wording (L444-446) differs from vLLM's but contains the same key phrase.

### Template and prefix (see F3)

- History thinking renders on vLLM; the prompts are append-only and byte-stable (12/12 exact token counts, 11/11 exact prefixes).
- The `is_reasoning_end` scans of the prompt stop at the last `<think>`, so a history full of `</think>` does not switch output parsing to content:
  - 0.19 `basic_parsers.py` L64-73;
  - 0.32 `parser_engine.py` L658-680;
  - 0.32 `adjust_initial_state_from_prompt` is a no-op (L208-210).

### Raw-reasoning path (E14-P, `E12_RAW_REASONING_HISTORY=1`, set in NB1 CELL 1b)

- **Capture** (`arc2_duck.py` L1943-1950): reads `msg["reasoning"]`, which vLLM supplies.
- **Substitution** (L1728-1758): checks that the stored text (key `reasoning`) equals `normalize(raw)`, then puts the raw text back. This works unchanged on vLLM.
- F4 and F1 reduce it on 0.32 only in rare cases.
- Its cache purpose does not apply on vLLM 0.19 (F3). It still keeps the model's view equal to R3K2.

### SGLang-specific branches in the Duck code

There is no code branch on the engine; the provider is `"vllm"` everywhere. The comments that mention SGLang:

| Location | What it says | Status on vLLM |
|---|---|---|
| `arc2_duck.py` L1547 | `/health` + smoke | E21 does both |
| L1911 | B6 | F1 |
| L2058 | `prompt_tokens_details` | F6, handled |
| L3385 | queue = 1 is steady state | gauges are mapped in E21 CELL 5 |
| L3657 | `SERVER_DOWN` | set by the E21 watchdog relaunch |
| `tool_agent.py` L3450 | SGLang wording | the vLLM wording is also matched |
