# E14-P0 offline token-diff: the chat template is NOT the cause of the reply re-prefill (Claude, 2026-10-23)

> **Correction (2026-10-23, after GPT notes 08/09; details in Notes/10):**
> - The result below is a **reconstructed divergence of 0/170 on tool-call turns**. It is not proof about the generated token ids.
> - The INT4 tokenizer has different id sequences for the same text. For example, `"\n\n"` = `[271]` or `[198, 198]`, and re-encoding always gives the canonical `[271]`. So check C cannot show the model emitted canonical ids.
> - The scope was narrower than stated: malformed calls, plain or last replies, and turns after a history trim were skipped. **After a trim, re-prefill is unavoidable by design**, which is a fourth cause.
> - Exact ids: the E14-A replay with `return_output_ids_in_sglext` (SGLang 0.5.21 supports it). Duck does not keep raw responses.
>
> **Second correction (Notes/11):** the cause found is on the client side.
> - Duck's `_normalize_message_content` (`tool_agent.py` L3414-3418) deletes every blank line of the reasoning before storing it in history.
> - The transcripts hold the same normalized text: 0/180 THINKING blocks have a blank line. That is why this check, which compared normalized with normalized, saw 0 divergence.
> - Server usage numbers: 68/150 requests lose the reply at the end of the previous prompt (217,541 tokens).

**Question.** In 72–91 % of later Duck requests, the cache hit stops at the end-of-prefill checkpoint of the previous prompt, so the whole previous reply is prefilled again (agent_effective_throughput.md). Is the cause the chat template re-rendering the reply differently from what the model emitted?

**Script:** `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/token_diff/token_diff.py`
- Data: the G4B rev 6d transcripts (`_scratch/g4b_bundle_verify/runs/rev6d_EVAL12R6G4B/duck/*/transcript.txt`).
- Tokenizer and template: the RedHat INT4 tokenizer and `chat_template.jinja` (fixture `tok_int4`), with `preserve_thinking=True`, as in production.
- Per assistant turn k (170 turns):
  - A: render(msgs[:k], add_generation_prompt) is a text prefix of render(msgs[:k] + assistant k + tool result).
  - B: its token ids are a token prefix of the history's ids.
  - C: the reply tokenized on its own equals the same text tokenized inside the history.

**Result:**

| Check | Result |
|---|---|
| A (text prefix) | 170 / 170 |
| B (token prefix) | 170 / 170 |
| C (reply tokens equal) | 170 / 170 |

**Source checks (SGLang 0.5.21, local):**
- `function_call/qwen3_coder_detector.py` L215-219 and L344-347 strip exactly one leading and one trailing `\n` from each parameter value. The template re-adds exactly one on each side (`'<parameter=' + name + '>\n' + value + '\n</parameter>\n'`). So values that keep an extra trailing `\n` (54 / 170) still round-trip exactly.
- `parser/reasoning_parser.py` L185-190 returns the text before `</think>` without stripping. Duck stores it as `reasoning_content` (`ARC3_REASONING_HISTORY_KEY=reasoning_content`, arc2_duck.py L41). The template applies `reasoning_content|trim`, then adds `'\n</think>\n\n'`. This is exact when the model wrote a single `\n` before `</think>`.
- Duck rev 6 does not modify the last assistant message: only trims of whole old blocks, and the opener sentinel rewrite after a trim.

**Conclusion:**
- With canonical text and canonical tokenization the round trip is exact, so the re-prefill is **not** a template bug.
- **Remaining candidates** (not separable offline, because no token ids were logged):
  1. **Non-canonical sampled tokens at T 1.0.**
     - The model can emit a valid but non-canonical token split, for example `\n` + `\n` instead of `\n\n`, or a split word.
     - Re-tokenizing the same text then gives different ids, so the radix match stops at the first such token.
     - The match falls back to the last Mamba checkpoint, which is the end-of-prefill one.
     - Long replies (5–30k tokens) almost surely contain at least one such token. That fits "misses rise after long replies".
  2. **Whitespace the parsers drop and the template cannot restore.** Examples: two newlines before `</think>` (trim), or text outside the tool call.
  3. **Cache mechanics:** the decode-end Mamba checkpoint was evicted or never kept (lazy mode has no evict-retry for track slots).

**Next (GPU, a few minutes inside E14-A):**
- Replay a recorded conversation through `/generate` with `input_ids` = tok(P_k) and `return_output_ids`.
- Then compare tok(P_{k+1}) with input_ids + output_ids at the reply boundary. This separates candidate 1/2 (ids differ) from candidate 3 (ids equal but no hit).

**If 1/2 is confirmed:** the fix is token-level continuation. Two options:
- SGLang streaming sessions: append-only, a 300 s idle reaper, supported for Mamba models.
- Duck sending token ids through the native API.

Either one is a Duck-side change that needs the user's approval (rev 6g rule).
