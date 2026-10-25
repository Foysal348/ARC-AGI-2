"""E14-P0 offline token-diff (2026-10-23).

Question: when Duck sends request k+1, does the re-rendered history of assistant turn k tokenize to the same token
sequence the server saw at the end of request k (prompt_k + generated reply)? If not, the radix/mamba cache can only
match up to the end-of-prefill checkpoint of prompt_k, and the whole reply is prefilled again.

No token ids were logged, so this checks what CAN be checked offline, per assistant turn:
  A. template self-consistency: render(msgs[:k], add_generation_prompt) is a text prefix of render(msgs[:k+1] + tool)
  B. token-prefix stability: tok(P_k) is a token prefix of tok(H_k)
  C. reply tokenization: tok(reply_raw) (the reply as the model emits it after '<think>\\n') equals the slice of
     tok(H_k) that follows P_k. A mismatch here = re-tokenization divergence even with identical text.
  D. whitespace lost by the template: reasoning|trim and the parameter-value newlines.
Prints counts only (no transcript text).
"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4")
tok = AutoTokenizer.from_pretrained(str(FIX))
tmpl = (FIX / "chat_template.jinja").read_text(encoding="utf-8")

BLOCK = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|TOOL CALL: python|TOOL RESULT: python|ASSISTANT|"
                   r"MODEL RESPONSE META|ANALYZER STATUS)\]\s*$", re.M)


def parse(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    # drop the "--- analysis_step=... ---" separator lines
    text = re.sub(r"^--- analysis_step=.*---\s*$", "", text, flags=re.M)
    parts = BLOCK.split(text)
    out = []  # (kind, body)
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


def raw_calls(meta):
    m = re.search(r"raw_tool_calls:\s*(\[.*\])", meta, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def build_turns(blocks):
    """Returns the message list and, per assistant message index, the raw reply text guess."""
    msgs, replies = [], {}
    pend_meta = None
    pend_reason = ""
    pend_rc = None
    for kind, body in blocks:
        b = body.strip("\n")
        if kind == "SYSTEM PROMPT":
            if not msgs:
                msgs.append({"role": "system", "content": b})
        elif kind == "USER PROMPT":
            msgs.append({"role": "user", "content": b})
        elif kind == "MODEL RESPONSE META":
            pend_meta = body
            pend_reason = ""
            mm = re.search(r"reasoning_chars:\s*(\d+)", body)
            pend_rc = int(mm.group(1)) if mm else None
        elif kind == "THINKING":
            pend_reason = body[1:] if body.startswith("\n") else body
            if pend_rc is not None:
                pend_reason = pend_reason[:pend_rc]   # exact server reasoning_content length (META)
            else:
                pend_reason = pend_reason.rstrip("\n")
        elif kind == "TOOL CALL: python":
            calls = raw_calls(pend_meta or "") or []
            tcs = []
            for c in calls:
                f = c.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except Exception:
                    args = {}
                tcs.append({"type": "function", "function": {"name": f.get("name"), "arguments": args}})
            if not tcs:
                continue
            m = {"role": "assistant", "content": "", "reasoning_content": pend_reason, "tool_calls": tcs}
            msgs.append(m)
            replies[len(msgs) - 1] = (pend_reason, b)
            pend_meta, pend_reason = None, ""
        elif kind == "TOOL RESULT: python":
            msgs.append({"role": "tool", "content": b})
    return msgs, replies


def render(msgs, gen):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, chat_template=tmpl,
                                   preserve_thinking=True)


def main(roots):
    stats = {"turns": 0, "A_text_prefix_ok": 0, "B_token_prefix_ok": 0, "C_reply_tokens_equal": 0,
             "D_reason_ws_lost": 0, "D_param_ws_lost": 0, "markup_equal": 0, "errors": 0}
    first_div = []
    for root in roots:
        for tp in sorted(Path(root).rglob("transcript.txt")):
            try:
                msgs, replies = build_turns(parse(tp))
            except Exception:
                stats["errors"] += 1
                continue
            for ai, (reason, markup) in replies.items():
                if ai + 1 >= len(msgs) or msgs[ai + 1]["role"] != "tool":
                    continue
                try:
                    P = render(msgs[:ai], True)
                    H = render(msgs[:ai + 2], False)
                except Exception:
                    stats["errors"] += 1
                    continue
                stats["turns"] += 1
                if H.startswith(P):
                    stats["A_text_prefix_ok"] += 1
                tp_ids = tok(P, add_special_tokens=False)["input_ids"]
                th_ids = tok(H, add_special_tokens=False)["input_ids"]
                if th_ids[:len(tp_ids)] == tp_ids:
                    stats["B_token_prefix_ok"] += 1
                # the reply as emitted: reasoning, '\n</think>\n\n', tool markup, '<|im_end|>'
                # text-identical to the history by construction (parser strips exactly one \n around each value and
                # the template re-adds it), so C isolates TOKENIZATION: the reply tokenized on its own (how the model
                # emits it, token by token after '<think>\n') vs the same text tokenized inside the full history
                calls_txt = ""
                for j, c in enumerate(msgs[ai]["tool_calls"]):
                    f = c["function"]
                    calls_txt += ("" if j == 0 else "\n") + "<tool_call>\n<function=" + f["name"] + ">\n"
                    for pk, pv in (f["arguments"] or {}).items():
                        pv = pv if isinstance(pv, str) else json.dumps(pv, ensure_ascii=False)
                        calls_txt += "<parameter=" + pk + ">\n" + pv + "\n</parameter>\n"
                    calls_txt += "</function>\n</tool_call>"
                reply_raw = reason.strip() + "\n</think>\n\n" + calls_txt + "<|im_end|>"
                rr = tok(reply_raw, add_special_tokens=False)["input_ids"]
                seg = th_ids[len(tp_ids):len(tp_ids) + len(rr)]
                if seg == rr:
                    stats["C_reply_tokens_equal"] += 1
                else:
                    k = next((i for i, (a, b) in enumerate(zip(seg, rr)) if a != b), min(len(seg), len(rr)))
                    first_div.append((k, len(rr)))
                if reason != reason.strip():
                    stats["D_reason_ws_lost"] += 1
                for c in msgs[ai]["tool_calls"]:
                    v = (c["function"]["arguments"] or {}).get("code")
                    if isinstance(v, str) and v != v.strip("\n"):
                        stats["D_param_ws_lost"] += 1
                hist_markup = H[len(P):].split("</think>\n\n", 1)[-1].split("<|im_end|>", 1)[0]
                if hist_markup == markup.strip("\n"):
                    stats["markup_equal"] += 1
    print(json.dumps(stats, indent=1))
    if first_div:
        ks = sorted(k for k, _ in first_div)
        print("first divergence token index in reply: n", len(ks), "p10/p50/p90",
              ks[len(ks) // 10], ks[len(ks) // 2], ks[9 * len(ks) // 10])


if __name__ == "__main__":
    main(sys.argv[1:])
