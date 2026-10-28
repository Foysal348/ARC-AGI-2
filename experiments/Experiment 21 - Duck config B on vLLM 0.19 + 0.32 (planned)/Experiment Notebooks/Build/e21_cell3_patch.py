# E21 rev 2 (VAL_api_D1 F1, verified in the vLLM source): vLLM 0.19 / 0.32 report finish_reason "tool_calls" whenever
# the parser found a call, also for a reply cut at max_tokens (chat_completion/serving.py 0.19 L1550-1560, 0.32
# L1099-1116); SGLang 0.5.21 kept "length", which arc2_duck's cut-call handling (rev 5 B6 / R5-2, the cut-code harvest,
# the R6 "cut" opener) keys on. Restore "length" from the token count (Duck's package files stay byte-identical).
_E21_CC_ORIG = _E12_TA.ToolAgent._chat_completion
E21_LENGTH_RESTORED = {"n": 0}
E21_CALL_WITH_CONTENT = {"n": 0}     # rev 2b (VAL_api_D1 F4): a tool call + visible text


def _e21_chat_completion(self, messages, **kw):
    res = _E21_CC_ORIG(self, messages, **kw)
    try:
        u = res.usage if isinstance(res.usage, dict) else {}
        cap = kw.get("max_tokens")
        cap = self._max_output_tokens if cap is None else cap
        ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        _m = res.message if isinstance(res.message, dict) else {}
        if _m.get("tool_calls") and str(_m.get("content") or "").strip():
            E21_CALL_WITH_CONTENT["n"] += 1
        if str(res.finish_reason or "") == "tool_calls" and cap and isinstance(ct, int):
            lim = int(cap) if not isinstance(pt, int) else min(int(cap), E21_VLLM_MAX_MODEL_LEN - int(pt))   # 0.32 clamps
            if ct >= lim:
                res.finish_reason = "length"
                E21_LENGTH_RESTORED["n"] += 1
    except Exception:
        pass
    return res


_E12_TA.ToolAgent._chat_completion = _e21_chat_completion
