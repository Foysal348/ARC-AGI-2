# -*- coding: utf-8 -*-
"""Execute PIV-Q27-C1 v2.4E cells in-process on Windows with a fake vLLM/torch and real ARC-AGI-2 public data.
Usage: python harness_c1.py ADAPTIVE_SMOKE|PREFLIGHT [notebook_path]
Everything Kaggle/GPU-specific is stubbed; every other line is the notebook's own code."""
import sys, os, json, types, time, threading, collections, importlib, importlib.metadata, subprocess, traceback, hashlib, random
from pathlib import Path

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "ADAPTIVE_SMOKE"
NB = Path(sys.argv[2])   # notebook path is required: python harness_c1.py <PROFILE> <notebook.ipynb>
# Scratch root holding the fake Kaggle tree (fake_input/, fake_working_c1/).
# Override on another machine: PIV_HARNESS_SCRATCH=/path/to/scratch python harness_c1.py SUBMIT <notebook>
S = Path(os.environ.get("PIV_HARNESS_SCRATCH", r"C:\Users\hp\AppData\Local\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad"))
FAKE_INPUT = S / "fake_input"
MODEL_DIR = FAKE_INPUT / "models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
WORKING = S / "fake_working_c1"
WORKING.mkdir(exist_ok=True)
(MODEL_DIR / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}), encoding="utf-8")
(MODEL_DIR / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}), encoding="utf-8")

# ---------------------------------------------------------------- package-version pins
_EXPECTED = {"vllm": "0.19.0", "transformers": "4.57.6", "tokenizers": "0.22.2", "huggingface-hub": "0.36.2", "torch": "2.10.0+cu128"}
_real_version = importlib.metadata.version
def _fake_version(name):
    if name in _EXPECTED: return _EXPECTED[name]
    return _real_version(name)
importlib.metadata.version = _fake_version

# ---------------------------------------------------------------- transformers stub (real Qwen tokenizer + template)
import tokenizers, jinja2
class _Tok:
    def __init__(self, d):
        self.t = tokenizers.Tokenizer.from_file(str(Path(d) / "tokenizer.json"))
        self.chat_template = (Path(d) / "chat_template.jinja").read_text(encoding="utf-8")
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, **kw)
        def _raise(msg): raise ValueError(msg)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.tpl = env.from_string(self.chat_template)
    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids
    def decode(self, ids, skip_special_tokens=False):
        return self.t.decode(list(ids), skip_special_tokens=skip_special_tokens)
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None, **kw):
        return self.tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **kw)
    def __len__(self): return self.t.get_vocab_size()
class _AutoTokenizer:
    @staticmethod
    def from_pretrained(d, **kw): return _Tok(d)
_tf = types.ModuleType("transformers"); _tf.AutoTokenizer = _AutoTokenizer; _tf.__version__ = "4.57.6"
sys.modules["transformers"] = _tf
_ipy = types.ModuleType("IPython"); _ipyd = types.ModuleType("IPython.display")
_ipyd.display = lambda *a, **k: print(*[str(x)[:2000] for x in a])
_ipy.display = _ipyd; sys.modules["IPython"] = _ipy; sys.modules["IPython.display"] = _ipyd

# ---------------------------------------------------------------- torch stub
_torch = types.ModuleType("torch"); _torch.__version__ = "2.10.0+cu128"
_torch.version = types.SimpleNamespace(cuda="12.8")
class _Props: total_memory = 24 * 1024**3
_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 4, get_device_capability=lambda i: (8, 9),
                                    get_device_properties=lambda i: _Props(), get_device_name=lambda i: "NVIDIA L4")


# ---------------------------------------------------------------- selectors stub + REPL stdout wrapper (Windows pipes)
class _StdoutWrap:
    def __init__(self, raw): self.raw = raw; self.pending = collections.deque()
    def readline(self): return self.pending.popleft() if self.pending else self.raw.readline()
    def read(self, *a): return self.raw.read(*a)
    def fileno(self): return self.raw.fileno()
    def close(self): return self.raw.close()
    @property
    def closed(self): return self.raw.closed
class _FakeSelector:
    def __init__(self): self.obj = None
    def register(self, fileobj, events, data=None): self.obj = fileobj
    def select(self, timeout=None):
        obj = self.obj
        if getattr(obj, "pending", None): return [(None, 1)]
        raw = getattr(obj, "raw", obj); box = []
        t = threading.Thread(target=lambda: box.append(raw.readline()), daemon=True); t.start(); t.join(timeout)
        if t.is_alive(): return []
        if hasattr(obj, "pending"): obj.pending.append(box[0]); return [(None, 1)]
        return [(None, 1)]
    def close(self): pass
_sel = types.ModuleType("selectors"); _sel.DefaultSelector = _FakeSelector; _sel.EVENT_READ = 1
sys.modules["selectors"] = _sel
_RealPopen = subprocess.Popen
class _PatchedPopen(_RealPopen):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        args = a[0] if a else k.get("args")
        if isinstance(args, list) and args and "REPL initialized" in str(args[-1]) and self.stdout is not None:
            self.stdout = _StdoutWrap(self.stdout)
subprocess.Popen = _PatchedPopen

# ---------------------------------------------------------------- vLLM stub with a scripted model
FAKE_LOG = []
class _SP:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        assert "max_tokens" in kw and int(kw["max_tokens"]) >= 1, kw
class _SOP:
    def __init__(self, **kw): self.__dict__.update(kw)
class _Out:
    def __init__(self, tok, text, n, fr, overcap=False):
        ids = list(tok.encode(text, add_special_tokens=False)) or [0]
        if len(ids) > int(n) and not overcap:        # real vLLM never exceeds max_tokens; overcap=True simulates a violation
            ids = ids[:int(n)]; text = tok.decode(ids, skip_special_tokens=False); fr = "length"
        self.text = text; self.finish_reason = fr; self.stop_reason = None   # vLLM: None on EOS / length
        self.token_ids = ids + [ids[-1]] * max(0, int(n) - len(ids))
class _Req:
    def __init__(self, o, pids): self.outputs = [o]; self.prompt_token_ids = pids

TOOL = lambda code, think="inspect": f"{think}\n</think>\n<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
FINAL = "done\n</think>\n```python\ndef solution(grid):\n    return [row[:] for row in grid]\n```"
GOOD_JSON = json.dumps({"solution_code": "def solution(grid):\n    return [row[:] for row in grid]"}) + "<|im_end|>"
BAD_JSON = '{"solution_code":"def solution(grid):\\n    return [row[:] for row in gr'   # truncated by max_tokens
LONG_THINK = "still reasoning about the rule; row 0: 1 2 3; row 1: 4 5 6; alternatively maybe rotation. " * 400

class _FakeLLM:
    """Scenario per SMOKE8 position (task order fixed in cell 4):
    0 pilot 0934a4d8 : tool, then exploration blow-up (length at cap) -> terminal -> good JSON
    1 pilot cb2d8a2c : tool, tool, final program in exploration (no finalizer)
    2 : six tool calls -> tool_budget_exhausted -> terminal -> good JSON
    3 : malformed tool xml every turn -> retries exhausted -> terminal -> good JSON
    4 : prose / no action every turn -> source retries exhausted -> terminal -> BAD (truncated) JSON, finish=length
    5 : tool with 15s sleep (REPL timeout + restart) then final
    6 : tool call, then guard-rejected code, then final
    7 : long think every turn (forced arm: phase-2 emits tool then final; natural arm: length at cap -> terminal)
    """
    def __init__(self, **kw): self.kw = kw; FAKE_LOG.append(("LLM_init", sorted(kw)))
    def _tid(self, text):
        G = _G
        for tid in G["ACTIVE_TASK_IDS"]:
            if G["puzzle_to_source_text"](G["eval_challenges"][tid]) in text: return tid
        return None
    def generate(self, prompts, params, use_tqdm=False):
        G = _G; tok = G["tokenizer"]; outs = []
        assert len(prompts) == len(params)
        _evo_short = False     # HARNESS_EVO_ENGINE=p1|p2|short: inject an engine failure into the first EVO batch of that kind
        _emode = os.environ.get("HARNESS_EVO_ENGINE")
        if _emode and prompts and isinstance(prompts[0], dict) and "prompt_token_ids" in prompts[0]:
            _t0 = tok.decode(list(prompts[0]["prompt_token_ids"]), skip_special_tokens=False)
            if "ARC-EVO v1" in _t0:
                _ph2 = _t0.rstrip().endswith("</think>")
                if (_emode == "p1" and not _ph2) or (_emode == "p2" and _ph2):
                    raise RuntimeError(f"injected vLLM engine failure ({_emode})")
                _evo_short = (_emode == "short" and not _ph2)
        smoke = G["choose_smoke8_from_proxy"]()
        for p, sp in zip(prompts, params):
            if isinstance(p, dict) and "prompt_token_ids" in p:
                pids = list(p["prompt_token_ids"]); text = tok.decode(pids, skip_special_tokens=False)
            else:
                text = p if isinstance(p, str) else p["prompt"]; pids = tok.encode(text, add_special_tokens=False)
            cap = int(sp.max_tokens)
            tid = self._tid(text)
            if "ARC-EVO v1" in text and tid is not None:
                _act = list(G["ACTIVE_TASK_IDS"]); pos = (_act.index(tid) % 8) if tid in _act else 0
                phase2 = text.rstrip().endswith("</think>")
                repair = "A previous attempt at this puzzle" in text
                _task = G["eval_challenges"][tid]
                _key = lambda g: (len(g), len(g[0]), sum(map(sum, g)), g[0][0], g[-1][-1])
                import zlib as _z, base64 as _b   # compact lookup so large-grid tasks fit the 2,048-token answer cap
                _blob = _b.b64encode(_z.compress(json.dumps({repr(_key(ex["input"])): ex["output"] for ex in _task["train"]}).encode(), 9)).decode()
                cheat = ("## Rule\nlook up the answer\n## Code\n```python\nimport json, zlib, base64\nT = json.loads(zlib.decompress(base64.b64decode('" + _blob + "')))"
                         + "\ndef solution(grid):\n    return T.get(repr((len(grid), len(grid[0]), sum(map(sum, grid)), grid[0][0], grid[-1][-1])), grid)\n```")
                # pos 6 initial answers behave like a truly correct program: the lookup also covers the test inputs
                _full = {repr(_key(ex["input"])): ex["output"] for ex in _task["train"]}
                _full.update({repr(_key(tx["input"])): sol for tx, sol in zip(_task["test"], G["eval_solutions"].get(tid, []))})
                _blob2 = _b.b64encode(_z.compress(json.dumps(_full).encode(), 9)).decode()
                correct = cheat.replace(_blob, _blob2)
                ident = "## Rule\ncopy the grid\n## Code\n```python\ndef solution(grid):\n    return [row[:] for row in grid]\n```"
                alternative = "This program reproduces EVERY training example exactly" in text
                if repair and phase2 is False and "TEST INPUT FAILURE" in text:
                    print("[HARNESS] exact-partial repair prompt carries TEST INPUT FAILURE:", tid)
                if not phase2:
                    if pos == 1:
                        o = _Out(tok, LONG_THINK, cap, "length")
                    elif pos == 7:     # EOS inside thinking: must be a forced close, not a natural one
                        _br = "short thought<|im_end|>"; o = _Out(tok, _br, len(tok.encode(_br, add_special_tokens=False)), "stop")
                    elif pos == 5:     # stopped on the </think> id, token not echoed in ids/text
                        _br = "brief reasoning"; o = _Out(tok, _br, len(tok.encode(_br, add_special_tokens=False)), "stop")
                        o.stop_reason = tok.encode("</think>", add_special_tokens=False)[0]
                    else:
                        _br = "brief reasoning about the rule\n</think>"; o = _Out(tok, _br, len(tok.encode(_br, add_special_tokens=False)), "stop")
                else:
                    if alternative:    # pos 2: the second, genuinely different program is the correct one
                        body = correct if pos == 2 else ident
                    elif repair:
                        body = cheat if pos in (2, 4, 6) else ident
                    else:
                        body = {0: ident, 1: ident,
                                2: "## Rule\nkeep the first row\n## Code\n```python\ndef solution(grid):\n    return [grid[0][:]]\n```",
                                3: "I am not sure what the rule is.",
                                4: "## Rule\nx\n## Code\n```python\ndef solution(grid)\n    return grid\n```",
                                # pos 5: train-exact lookup that raises KeyError on the test input (exact-partial)
                                5: cheat.replace("return T.get(repr((len(grid), len(grid[0]), sum(map(sum, grid)), grid[0][0], grid[-1][-1])), grid)",
                                                 "return T[repr((len(grid), len(grid[0]), sum(map(sum, grid)), grid[0][0], grid[-1][-1]))]"),
                                6: correct, 7: ident}[pos]
                    _n = len(tok.encode(body, add_special_tokens=False)) + 1
                    o = _Out(tok, body + "<|im_end|>", min(cap, _n), "stop" if _n <= cap else "length")
                FAKE_LOG.append(("evo", tid, "p2" if phase2 else "p1", pos, o.finish_reason, len(o.token_ids)))
                outs.append(_Req(o, pids)); continue
            if tid is None:
                if "identity ARC program" in text:      # cell 7 positive control
                    o = _Out(tok, GOOD_JSON, 60, "stop")
                else:                                    # cell 7 warmup (32-token greedy probe)
                    o = _Out(tok, "Let me look at the puzzle first.", min(cap, 8), "stop")
                FAKE_LOG.append(("control_or_warmup", None, None, None, o.finish_reason, len(o.token_ids)))
                outs.append(_Req(o, pids)); continue
            if tid in smoke:
                pos = smoke.index(tid)
            else:
                # Submission profiles: spread the eight scripted failure scenarios over the active task list.
                _active = list(G["ACTIVE_TASK_IDS"])
                pos = (_active.index(tid) % 8) if tid in _active else 0
            terminal = "[TERMINAL COMMIT]" in text and text.rstrip().endswith("</think>")
            tail = text.rstrip()
            forced_p2 = tail.endswith("</think>") and (G["STOP_THINKING_SENTENCE"] in text[-600:]) and not terminal
            nat_cont = tail.endswith("<parameter=code>")
            n_tool = text.count("<tool_response>")
            arm = G["CURRENT_ARM"]; forced = G["ARM_CONFIG"][arm]["think_policy"] == "forced_close"
            repair = "[V25_REPAIR_FEEDBACK_SHA256=" in text
            if repair:
                task = G["eval_challenges"][tid]
                cheat = "T=" + repr({repr(ex["input"]): ex["output"] for ex in task["train"]}) + "\ndef solution(grid):\n    return T.get(repr(grid), grid)\n"
                cheat_json = json.dumps({"solution_code": cheat}) + "<|im_end|>"
                cl = len(tok.encode(cheat_json, add_special_tokens=False)); fits = cl <= cap - 8
                _oc = bool(os.environ.get('HARNESS_OVERCAP'))
                def J(code, n, fr="stop", overcap=False): return _Out(tok, json.dumps({"solution_code": code}) + "<|im_end|>", n, fr, overcap=overcap)
                if pos == 7 and fits and not _oc: o = _Out(tok, cheat_json, cl, "stop")          # verified repair -> candidate #2 selected
                elif pos == 0:                  o = _Out(tok, GOOD_JSON, 120, "stop")            # identity again -> delta 0
                elif pos == 1:                  o = J("def solution(grid):\n    return grid[:1]", 80)                     # dims mismatch
                elif pos == 2:                  o = _Out(tok, BAD_JSON, cap, "length")            # truncated JSON -> parse failed
                elif pos == 3:                  o = J("def solution(grid):\n    return grid[99][99]", 80)                 # exception
                elif pos == 5:                  o = J("import time\ndef solution(grid):\n    time.sleep(15)\n    return grid", 80)  # verifier timeout
                elif pos == 6:                  o = J("import os\ndef solution(grid):\n    return grid", 80)              # guard reject
                else:                           o = J("def solution(grid):\n    return grid[:1]", (cap + 5 if _oc else 80), overcap=_oc)  # overcap injection
                FAKE_LOG.append((arm, tid, "repair", pos, o.finish_reason, len(o.token_ids), "cheat" if (pos == 7 and fits and not _oc) else "other"))
                outs.append(_Req(o, pids)); continue
            if terminal:
                if pos == 4: o = _Out(tok, BAD_JSON, cap, "length")
                else: o = _Out(tok, GOOD_JSON, 120, "stop")
            elif nat_cont:
                o = _Out(tok, "puzzle.keys()\n</parameter>\n</function>\n</tool_call>", 30, "stop")
            elif forced and not forced_p2 and pos != 1:
                # phase 1 of the forced arm: never closes </think>
                o = _Out(tok, LONG_THINK, (cap + 5 if (os.environ.get("HARNESS_OVERCAP") and pos == 7) else cap), "length", overcap=bool(os.environ.get("HARNESS_OVERCAP") and pos == 7))
            else:
                pre = "" if forced_p2 else None
                def act(s):   # forced phase 2 emits only the post-</think> part
                    return s.split("</think>", 1)[1].lstrip("\n") if forced_p2 and "</think>" in s else s
                if pos == 0:
                    o = _Out(tok, act(TOOL("print(puzzle.keys())")), 300, "stop") if n_tool == 0 else _Out(tok, LONG_THINK, cap, "length")
                elif pos == 1:
                    seq = [TOOL("print(puzzle.keys())"), TOOL("rows=len(puzzle['train']); print(rows)"), FINAL]
                    o = _Out(tok, act(seq[min(n_tool, 2)]), 300, "stop")
                elif pos == 2:
                    _c2 = ("def solution(grid):\n    return [row[:] for row in grid]" if n_tool == 1
                           else f"x{n_tool}={n_tool}; print(x{n_tool})")
                    o = _Out(tok, act(TOOL(_c2)), 200, "stop")
                elif pos == 3:
                    o = _Out(tok, act("r\n</think>\n<tool_call><function=python><parameter=code>x=1</function></tool_call>"), 100, "stop")
                elif pos == 4:
                    o = _Out(tok, act("thinking\n</think>\nI believe the rule is a rotation but I cannot commit yet."), 80, "stop")
                elif pos == 5:
                    seq = [TOOL("import time\ntime.sleep(15)"), FINAL]
                    o = _Out(tok, act(seq[min(n_tool, 1)]), 300, "stop")
                elif pos == 6 and os.environ.get("HARNESS_V3_COMMIT", "1") == "1" and n_tool == 0:
                    _task = G["eval_challenges"][tid]
                    _key = lambda g: (len(g), len(g[0]), sum(map(sum, g)), g[0][0], g[-1][-1])
                    _cheat = ("T = " + repr({_key(ex["input"]): ex["output"] for ex in _task["train"]})
                              + "\ndef solution(grid):\n    return T.get((len(grid), len(grid[0]), sum(map(sum, grid)), grid[0][0], grid[-1][-1]), grid)")
                    if os.environ.get("HARNESS_V31_TESTFAIL"):
                        _cheat = _cheat.replace("return T.get(", "return T[").replace("grid[-1][-1]), grid)", "grid[-1][-1])]")
                    _ct = act(TOOL(_cheat)); o = _Out(tok, _ct, min(cap, len(tok.encode(_ct, add_special_tokens=False))), "stop")
                elif pos == 6:
                    seq = [TOOL("print(puzzle.keys())"), TOOL("import os\nprint(os.getcwd())"), FINAL]
                    o = _Out(tok, act(seq[min(n_tool, 2)]), 300, "stop")
                else:
                    _oc1 = bool(os.environ.get('HARNESS_OVERCAP')) and not forced_p2
                    o = _Out(tok, act(TOOL("print(len(puzzle['train']))")) if forced_p2 else LONG_THINK, (300 if forced_p2 else cap), ("stop" if forced_p2 else "length"), overcap=_oc1)
            FAKE_LOG.append((arm, tid, "terminal" if terminal else ("p2" if forced_p2 else ("nat" if nat_cont else "p1")), n_tool, o.finish_reason, len(o.token_ids)))
            outs.append(_Req(o, pids))
        return outs[:-1] if _evo_short else outs

_vllm = types.ModuleType("vllm"); _vllm.__version__ = "0.19.0"; _vllm.LLM = _FakeLLM; _vllm.SamplingParams = _SP
_vsp = types.ModuleType("vllm.sampling_params"); _vsp.StructuredOutputsParams = _SOP; _vsp.SamplingParams = _SP
_vllm.sampling_params = _vsp; sys.modules["vllm"] = _vllm; sys.modules["vllm.sampling_params"] = _vsp

# ---------------------------------------------------------------- run the notebook cells
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
_G = {"__name__": "__main__"}
rep = {
    'Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1")': f'Path(r"{MODEL_DIR}")',
    'Path("/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3")': f'Path(r"{FAKE_INPUT / "datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"}")',
    'Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")': f'Path(r"{FAKE_INPUT / "competitions/arc-prize-2026-arc-agi-2"}")',
    'Path("/kaggle/working/piv_q27_c1_v24")': f'Path(r"{WORKING / "piv_q27_c1_v24"}")',
    'Path("/kaggle/working/piv_q27_eval120")': f'Path(r"{WORKING / "piv_q27_eval120"}")',
    'Path("/kaggle/working/piv_q27_c1_v25")': f'Path(r"{WORKING / "piv_q27_c1_v25"}")',
    'Path("/kaggle/working/piv_q27_sub_v10")': f'Path(r"{WORKING / "piv_q27_sub_v10"}")',
    'RUN_PROFILE = "ADAPTIVE_SMOKE"   # default': f'RUN_PROFILE = "{PROFILE}"   # harness',
    'RUN_PROFILE = "SUBMIT"   # default': f'RUN_PROFILE = "{PROFILE}"   # harness',
}
t_all = time.perf_counter(); failed = None
for i, src in enumerate(codes, 1):
    if 'MODEL_PATH = Path("/kaggle/input/models/' in src or 'RUN_PROFILE = "' in src:
        for a, b in rep.items():
            if src.count(a) == 0 and ("piv_q27_" in a or "RUN_PROFILE = " in a): continue   # differs per notebook
            assert src.count(a) == 1, a
            src = src.replace(a, b)
    if i == 7: sys.modules["torch"] = _torch   # only after sklearn (cell 3) has been imported
    if '_Path("/kaggle/working/submission.json")' in src:
        src = src.replace('_Path("/kaggle/working/submission.json")', '_Path(r"' + str(WORKING / "submission.json") + '")')
        src = src.replace('_Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")',
                          '_Path(r"' + str(FAKE_INPUT / "competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json") + '")')
        src = src.replace('_Path("/kaggle/input")', '_Path(r"' + str(FAKE_INPUT) + '")')
    if 'Path("/kaggle/working/submission.json")' in src:
        src = src.replace('Path("/kaggle/working/submission.json")', 'Path(r"' + str(WORKING / "submission.json") + '")')
    for _bp in ('Path("/kaggle/working")/BUNDLE_NAME', 'Path("/kaggle/working") / BUNDLE_NAME'):
        if _bp in src: src = src.replace(_bp, 'Path(r"' + str(WORKING) + '")/BUNDLE_NAME')
    if os.environ.get("HARNESS_SHRINK_TOTAL") and "TOTAL_RUNTIME_SEC = 12 * 60 * 60" in src:
        src = src.replace("TOTAL_RUNTIME_SEC = 12 * 60 * 60",
                          'TOTAL_RUNTIME_SEC = float(os.environ["HARNESS_SHRINK_TOTAL"])')
    if os.environ.get("HARNESS_SHRINK_RESERVE") and "SAFETY_RESERVE_SEC = (75 * 60) if SUBMIT_MODE else (60 * 60)" in src:
        src = src.replace("SAFETY_RESERVE_SEC = (75 * 60) if SUBMIT_MODE else (60 * 60)",
                          'SAFETY_RESERVE_SEC = float(os.environ["HARNESS_SHRINK_RESERVE"])')
    if os.environ.get("HARNESS_SHRINK_MARGIN") and "SESSION_HARD_MARGIN_SEC = 10 * 60" in src:
        src = src.replace("SESSION_HARD_MARGIN_SEC = 10 * 60",
                          'SESSION_HARD_MARGIN_SEC = float(os.environ["HARNESS_SHRINK_MARGIN"])')
    if os.environ.get("HARNESS_SHRINK_PRIOR") and "DEFAULT_SEC_PER_OUTPUT_TOKEN = 0.20" in src:
        src = src.replace("DEFAULT_SEC_PER_OUTPUT_TOKEN = 0.20",
                          'DEFAULT_SEC_PER_OUTPUT_TOKEN = float(os.environ["HARNESS_SHRINK_PRIOR"])')
    _evo_v = 'v = fresh_verify_solution(cand["code"], task, VERIFIER_TIMEOUT_SEC)\n'
    if os.environ.get("HARNESS_EVO_INFRA") and _evo_v in src:   # inject one verifier infrastructure failure
        src = src.replace(_evo_v, _evo_v + '        if str(cand.get("cand_id", "")).endswith(":r0:s3"): v = dict(v, infrastructure_failure=True, infrastructure_errors=["injected"])\n')
    if os.environ.get("HARNESS_SHRINK_DEADLINE") and "deadline = notebook_start" in src:
        src = src.replace("deadline = notebook_start + TOTAL_RUNTIME_SEC - SAFETY_RESERVE_SEC",
                          'deadline = notebook_start + float(os.environ["HARNESS_SHRINK_DEADLINE"])')
    if os.environ.get("HARNESS_SHRINK_SESSION") and "SESSION_HARD_MARGIN_SEC = 10 * 60" in src:
        src = src.replace("SESSION_HARD_MARGIN_SEC = 10 * 60",
                          'SESSION_HARD_MARGIN_SEC = TOTAL_RUNTIME_SEC - float(os.environ["HARNESS_SHRINK_SESSION"])')
    print(f"\n################ CELL {i} ################", flush=True); t0 = time.perf_counter()
    try:
        exec(compile(src, f"cell{i}", "exec"), _G)
    except BaseException as e:
        failed = (i, e); print(f"!!!!!!!! CELL {i} FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); break
    print(f"---- cell {i} ok in {time.perf_counter()-t0:.1f}s", flush=True)
print(f"\nHARNESS {'FAILED at cell %d' % failed[0] if failed else 'COMPLETED'} in {time.perf_counter()-t_all:.1f}s; profile={PROFILE}")
(Path(__file__).parent / f"fake_llm_log_{PROFILE}.json").write_text(json.dumps(FAKE_LOG, indent=0), encoding="utf-8")
