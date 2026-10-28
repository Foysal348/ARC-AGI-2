"""VAL_engine_A2 M6: the INT4 tokenizer_config.json is in transformers-5 format (backend, processor_class Qwen3VLProcessor,
no added_tokens_decoder / chat_template inside). Check, with THIS python's transformers (the kernel's 4.57.6 = what the
vLLM 0.19 server in the kernel uses; /tmp/vllm-venv's = vLLM 0.32's), that both model folders load and render a
Duck-like conversation (system, user, assistant with reasoning + a tool call, tool result, tools, preserve_thinking)
to the SAME token ids. usage: <python> e21_tokcheck.py <fp8 dir> <int4 dir>  -> one JSON line "TOKCHECK {...}"."""
import json
import sys
import traceback

fp8, int4 = sys.argv[1], sys.argv[2]
out = {"python": sys.version.split()[0]}
try:
    import transformers
    out["transformers"] = transformers.__version__
    from transformers import AutoTokenizer
    tools = [{"type": "function", "function": {"name": "python", "description": "Run Python code.",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    msgs = [{"role": "system", "content": "You are a coding agent solving an ARC-AGI-2 grid puzzle.\n0 1 2\n3 4 5"},
            {"role": "user", "content": "Find the rule."},
            {"role": "assistant", "content": "", "reasoning_content": "The grid flips left-right.",
             "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "python",
                             "arguments": {"code": "def transform(g):\n    return [r[::-1] for r in g]\nprint(check(transform))"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "[ok, ok, ok]"}]
    ids, txt = {}, {}
    for tag, d in (("fp8", fp8), ("int4", int4)):
        rec = {}
        try:
            tok = AutoTokenizer.from_pretrained(d, trust_remote_code=False)
            rec["class"] = type(tok).__name__
            rec["has_chat_template"] = bool(getattr(tok, "chat_template", None))
            t = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=True, preserve_thinking=True)
            txt[tag] = t
            ids[tag] = tok(t, add_special_tokens=False)["input_ids"]
            rec["n_tokens"] = len(ids[tag])
            rec["reasoning_rendered"] = "The grid flips left-right." in t
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["tb"] = traceback.format_exc()[-1200:]
        out[tag] = rec
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(int4, trust_remote_code=False)
        out["int4_processor"] = "ok"
    except Exception as e:
        out["int4_processor"] = f"{type(e).__name__}: {e}"[:300]
    out["text_equal"] = txt.get("fp8") is not None and txt.get("fp8") == txt.get("int4")
    out["ids_equal"] = ids.get("fp8") is not None and ids.get("fp8") == ids.get("int4")
    out["ok"] = bool(out["ids_equal"] and out["fp8"].get("reasoning_rendered") and out["int4"].get("reasoning_rendered"))
except Exception as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:])
print("TOKCHECK " + json.dumps(out), flush=True)
