"""Fake vllm 0.19 for the E11-05 harness fallback scenarios: LLM(**kw) with llm_engine (add_request / step /
abort_request, DELTA token ids), generate() for the smoke, get_metrics(). Token streams follow the harness behaviours."""
import os, sys, types
from .sampling_params import SamplingParams, RequestOutputKind

__version__ = "0.19.0-harness-fake"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import beh_e11_05 as B                       # noqa: E402

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(B.TOKDIR_INT4))
    return _TOK


class _Out:
    def __init__(self, rid, ids, finished, reason):
        self.request_id, self.finished = rid, finished
        self.outputs = [types.SimpleNamespace(token_ids=ids, finish_reason=reason)]


class _Engine:
    def __init__(self):
        _ch, _sol, multi, single = B.load()
        self.beh = B.beh_map(multi, single, os.environ.get("FAKE_SCEN", ""))
        self.active, self.added = {}, []
        cc = types.SimpleNamespace(num_gpu_blocks=781, block_size=1600, cache_dtype="fp8_e4m3", mamba_cache_mode="align",
                                   enable_prefix_caching=True)
        self.vllm_config = types.SimpleNamespace(cache_config=cc, attention_config=types.SimpleNamespace(backend="TRITON_ATTN"))

    def add_request(self, rid, prompt, params):
        t = _tok()
        tid, it, phase = B.parse_rid(rid)
        b = self.beh.get(tid, "fail_same")
        fill = t.encode(" thinking", add_special_tokens=False)
        close = t.encode("</think>", add_special_tokens=False)[0]
        code = t.encode("\n\n" + B.answer_code(b, it), add_special_tokens=False)
        cap = params.max_tokens
        if phase == 1 and b == "forced_then_pass" and it == 0:
            s, r = (fill * (cap // len(fill) + 1))[:cap], "length"
        elif phase == 1:
            s, r = fill * 30 + [close] + code, "stop"
        else:
            s, r = code, "stop"
        self.active[rid] = [s, 0, r]
        self.added.append(rid)

    def abort_request(self, ids):
        for i in ids:
            self.active.pop(i, None)

    def step(self):
        outs = []
        for rid, r in list(self.active.items()):
            chunk = r[0][r[1]:r[1] + 4000]; r[1] += len(chunk)
            done = r[1] >= len(r[0])
            outs.append(_Out(rid, chunk, done, r[2] if done else None))
            if done:
                self.active.pop(rid, None)
        return outs


class LLM:
    def __init__(self, **kw):
        self.kw = kw
        self.llm_engine = _Engine()

    def generate(self, prompts, params, use_tqdm=False):
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[1, 2, 3, 4])]) for _ in prompts]

    def get_metrics(self):
        return [types.SimpleNamespace(name="vllm:kv_cache_usage_perc", value=0.3)]
