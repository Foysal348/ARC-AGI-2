"""E25 NB5 fake Merlin / vLLM 0.32 OpenAI server for the local end-to-end (no model, no GPU). Launched by the REAL NB5
engine cell (RC2's merlin_fn_server FAKE branch): argv = [<ckpt>-w4a16, --tensor-parallel-size 2, --max-num-seqs N,
--port P, --max-model-len M] + the boot's extra args (e.g. --speculative-config None).
Derived from Exp 21 src/harness/fake_vllm_duck_server.py (scripted Duck program: task_of / lookup programs / CONV
routing; block-hash prefix cache; 2-s stats lines; usage omission rules) and Exp 23's fake (SSE /v1/completions for
the e23 lab). New for NB5:
- the REAL Flash-Next tokenizer + chat template (<ckpt> without '-w4a16', else NB5_FAKE_TOKDIR) render every chat
  prompt and /tokenize (vLLM semantics: assistant 'reasoning' -> reasoning_content, tool-call arguments JSON -> dict,
  the server default kwargs {enable_thinking: true, reasoning_effort: xhigh} under the request's own);
- Merlin's startup lines (non-default args with max_num_seqs / block_size / speculative_config, KV size + max
  concurrency for the 149-block FP8 pool, F = 6, RecoverSSM prompt-only checkpoints) and cache_config_info;
- prefix cache in 64-token units (--prefix-match-unit 64), prompt positions only (RecoverSSM + PP: the reply is not
  checkpointed), cache_salt = cold;
- retained finished turns counted in the KV gauge (Merlin: up to 16), so the raw gauge reads high with few running;
- /v1/completions non-streaming with prompt ids: the commit turn variant C (prompt ends '<think>\\n\\n</think>\\n\\n')
  answers the conversation's scripted step as tool-call markup (no reasoning), the NB5 commit probe a print call;
- chat continue_final_message (the probe's server-parser path); JIT-during-inference lines on the first requests;
- latency histograms (TTFT / queue / prefill / e2e / decode _sum/_count).
Fault injection (env): FAKE_NO_TOKENIZE=1 (/tokenize 404 -> variant C falls back), FAKE_C_NOCALL=1 (variant C replies
without a call), FAKE_PREEMPT_AFTER=<n> (a preemption is counted after n tool requests), FAKE_SPEED (tok/s per
request, default 3000), FAKE_REPLY_TOKENS (tokens per Duck reply, default 2100), FAKE_STATE_DIR (launches.jsonl,
stats_<pid>.json). Scripted replies only (our own text and code)."""
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
if argv and argv[0] == "serve":
    argv = argv[1:]
T_START = time.time()
STATE_DIR = os.environ.get("FAKE_STATE_DIR") or ""


def arg(name, default=None):
    i = len(argv) - 1 - argv[::-1].index(name) if name in argv else -1     # the LAST occurrence wins (as argparse)
    return argv[i + 1] if 0 <= i < len(argv) - 1 else default


def log(line, who="APIServer"):
    print(f"({who} pid={os.getpid()}) INFO {time.strftime('%m-%d %H:%M:%S')} {line}", flush=True)


def launch_record(outcome, **kw):
    if STATE_DIR:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(os.path.join(STATE_DIR, "launches.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "pid": os.getpid(), "outcome": outcome, "argv": argv,
                                # NB5Q-ENV-1: the lever env as THIS server process sees it (the e2e asserts it)
                                "env_levers": {k: os.environ.get(k) for k in ("MERLIN_SHARE_DRAFT_WORKSPACE",)}, **kw}) + "\n")


MODEL = argv[0] if argv and not argv[0].startswith("--") else "?"
PORT = int(arg("--port", 8001))
SEQS = int(arg("--max-num-seqs", 8))
MAXLEN = int(arg("--max-model-len", 131072))
_spec_raw = arg("--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}')
SPEC = None if _spec_raw in ("None", "") else json.loads(_spec_raw)
SPEED = float(os.environ.get("FAKE_SPEED", 3000))
REPLY_TOKENS = int(os.environ.get("FAKE_REPLY_TOKENS", 2100))
BLOCK, POOL, FIXED, UNIT = 4096, 149, 6, 64
# NB5 R2: the KV dtype / pin from the argv (RC2 per-card block bytes: FP8 16,515,072, BF16 31,195,136; usable = blocks - 1)
KV_DTYPE = arg("--kv-cache-dtype", "fp8_e4m3")
KV_BYTES = int(arg("--kv-cache-memory-bytes", 0) or 0)
if KV_BYTES:
    POOL = KV_BYTES // (16515072 if KV_DTYPE.startswith("fp8") else 31195136) - 1
if os.environ.get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1":
    FIXED = 5                      # FAKE ONLY: a smaller F so the measured-F path is exercised (no real-engine claim)
_oom = int(os.environ.get("FAKE_OOM_ABOVE_BYTES", "0") or 0)
if _oom and KV_BYTES > _oom:       # fault injection: the pin ladder's launch-failure step-down
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB (fake Merlin, KV pin "
          f"{KV_BYTES} > {_oom})", flush=True)
    sys.exit(1)
if os.environ.get("FAKE_LEVER_ERROR") == "1" and os.environ.get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1":
    print("RuntimeError: MERLIN_SHARE_DRAFT_WORKSPACE=1 is not supported by this fake build", flush=True)
    sys.exit(1)
SERVED = "q27"
if MAXLEN != 131072:
    print(f"ValueError: fake Merlin: --max-model-len {MAXLEN} (NB5 needs 131072)", flush=True)
    launch_record("refused")
    sys.exit(2)
print(f"fake merlin argv: {' '.join(argv)}", flush=True)
_nd = {"model": MODEL, "max_model_len": MAXLEN, "max_num_seqs": SEQS, "block_size": BLOCK, "port": PORT,
       "tensor_parallel_size": 2, "pipeline_parallel_size": 2, "speculative_config": SPEC, "kv_cache_dtype": KV_DTYPE,
       **({"kv_cache_memory_bytes": KV_BYTES} if KV_BYTES else {}),
       "enable_prefix_caching": True, "prefix_match_unit": UNIT, "mamba_cache_mode": "align", "use_replayssm": True}
log(f"[api_utils.py:286] non-default args: {_nd!r}")

# ---- the real Flash-Next tokenizer + template
_tokdir = MODEL[:-len("-w4a16")] if MODEL.endswith("-w4a16") else MODEL
if not os.path.isfile(os.path.join(_tokdir, "tokenizer.json")):
    _tokdir = os.environ.get("NB5_FAKE_TOKDIR", _tokdir)
from tokenizers import Tokenizer            # noqa: E402
from transformers import AutoTokenizer      # noqa: E402
TOK = Tokenizer.from_file(os.path.join(_tokdir, "tokenizer.json"))
HF = AutoTokenizer.from_pretrained(_tokdir)
TPL = open(os.path.join(_tokdir, "chat_template.jinja"), encoding="utf-8").read()
DEFAULT_KW = {"enable_thinking": True, "reasoning_effort": "xhigh"}       # Merlin's --default-chat-template-kwargs (RC2)
TAIL_C = TOK.encode("<think>\n\n</think>\n\n", add_special_tokens=False).ids


def enc(text):
    return TOK.encode(text, add_special_tokens=False).ids


def dec(ids):
    return TOK.decode(list(ids), skip_special_tokens=False)


def vllm_messages(msgs):
    """vLLM 0.32 chat preprocessing as the template sees it: 'reasoning' -> reasoning_content (input
    'reasoning_content' is dropped), tool-call arguments JSON -> dict, None content -> ''."""
    out = []
    for m in msgs or []:
        m2 = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
        if m2.get("content") is None:
            m2["content"] = ""
        if m.get("role") == "assistant":
            m2["reasoning_content"] = m.get("reasoning") or ""
            tcs = []
            for tc in m.get("tool_calls") or []:
                f = dict(tc.get("function") or {})
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        f["arguments"] = json.loads(a) if a.strip() else {}
                    except Exception:
                        f["arguments"] = {}
                tcs.append(dict(tc, function=f))
            if tcs:
                m2["tool_calls"] = tcs
        out.append(m2)
    return out


def render(body):
    kw = dict(DEFAULT_KW)
    kw.update(body.get("chat_template_kwargs") or {})
    cfm = bool(body.get("continue_final_message"))
    agp = body.get("add_generation_prompt", True) and not cfm
    text = HF.apply_chat_template(vllm_messages(body.get("messages")), tools=body.get("tools") or None, tokenize=False,
                                  add_generation_prompt=agp, continue_final_message=cfm, chat_template=TPL, **kw)
    return enc(text)


log(f"[kv_cache_utils.py:2361] GPU KV cache size: {POOL * 3449:,} tokens, Maximum concurrency for {MAXLEN:,} tokens per request: "
    f"{POOL / (-(-MAXLEN // BLOCK) + FIXED):.2f}x", "EngineCore")
log(f"[kv_cache_utils.py:855] kv lcm block sizes {BLOCK}", "EngineCore")
log("[scheduler.py:105] RecoverSSM + pipeline parallel: Mamba prefix checkpoints are published for prompt positions only (3 groups)",
    "EngineCore")

# ---- the scripted Duck program (E21 fake, adapted: the commit turn reaches the solving lookup)
ARC = os.environ.get("NB5_FAKE_ARC", "")
CH = json.load(open(os.path.join(ARC, "arc-agi_evaluation_challenges.json"))) if ARC else {}
SOL = json.load(open(os.path.join(ARC, "arc-agi_evaluation_solutions.json"))) if ARC and os.path.exists(
    os.path.join(ARC, "arc-agi_evaluation_solutions.json")) else {}
CONV_STEP, CONV_META, TASK_ROLES = {}, {}, {}
CONV_RE = re.compile(r"CONV=(\w+)")


def _counts(g):
    d = {}
    for r in g:
        for v in r:
            d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items()))


def _r6_sig(t):
    out_ = []
    for p in t["train"][:2]:
        for tag, g in (("in ", p["input"]), ("out", p["output"])):
            out_.append(f"  {tag} {len(g)}x{len(g[0])} colors {_counts(g)}")
    return out_


def task_of(system_text):
    if "ARC TASK FACTS" in system_text:
        hits = [tid for tid, t in CH.items() if all(s in system_text for s in _r6_sig(t))]
        if len(hits) == 1:
            return hits[0]
        hits = [tid for tid, t in CH.items() if all(
            f"TRAIN {i}: in {len(p['input'])}x{len(p['input'][0])} {_counts(p['input'])} -> out {len(p['output'])}x"
            f"{len(p['output'][0])} {_counts(p['output'])}" in system_text for i, p in enumerate(t["train"][:2]))]
        if len(hits) == 1:
            return hits[0]
    return None


CUT_CODE = "print('CUT_CODE_RAN')"
INSPECT_R6 = "print(grid_shape(train[0]['input']), len(components(train[0]['input'])))\ninspect_pair(0)"


def lookup_r6(tid, name="transform"):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    for qi, q in enumerate(t["test"]):
        if tid in SOL:
            table[json.dumps(q["input"])] = SOL[tid][qi]
    return (f"def {name}(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), flip_h(flip_h(g)))\nprint(check(" + name + "))")


def lookup_train_only_r6(tid):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    return ("def transform_alt(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), [list(r) for r in g])\nprint(check(transform_alt))")


def program(tid, role):
    """role 0 (attempt 1): two exploring replies (the budget reaches the commit zone), then the solving lookup in the
    commit turn; role 1: a cut call, then a train-only lookup (accepted, test-wrong); later roles: explore, lookup."""
    if role == 0:
        return [INSPECT_R6, INSPECT_R6, lookup_r6(tid)]
    if role == 1:
        return [CUT_CODE, lookup_train_only_r6(tid)]
    return [INSPECT_R6, lookup_r6(tid)]


def call_markup(calls):
    s = ""
    for j, c in enumerate(calls or []):
        f = c.get("function") or {}
        a = f.get("arguments")
        a = json.loads(a) if isinstance(a, str) and a.strip().startswith("{") else (a or {})
        s += ("" if j == 0 else "\n") + "<tool_call>\n<function=" + str(f.get("name")) + ">\n"
        for k, v in (a or {}).items() if isinstance(a, dict) else []:
            s += f"<parameter={k}>\n{v}\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s


def conv_step(tid, text_for_marker):
    """(conv, tid, role, step) for a Duck conversation: the CONV marker of its history, else a new conversation."""
    f = CONV_RE.findall(text_for_marker or "")
    conv = f[-1] if f else None
    with LOCK:
        if conv is None or conv not in CONV_META:
            role = TASK_ROLES.get(tid, 0)
            TASK_ROLES[tid] = role + 1
            conv = f"{tid}r{role}"
            CONV_META[conv] = (tid, role)
            CONV_STEP[conv] = 0
        tid, role = CONV_META[conv]
        i = CONV_STEP[conv]
        CONV_STEP[conv] = i + 1
    return conv, tid, role, i


# ---- prefix cache (64-token units, prompt positions only), counters, admission
LOCK = threading.RLock()        # re-entrant: account() is called with LOCK held from the stream path
CACHE = set()
RETAINED = []            # the last 16 finished prompts' block counts (Merlin keeps finished turns referenced)
RUN, WAIT = {}, []
ST = {"requests": 0, "chat": 0, "completions": 0, "tool_requests": 0, "max_running": 0, "gen": 0, "prompt": 0, "pc_q": 0,
      "pc_h": 0, "drafts": 0, "acc": 0, "dtok": 0, "preempt": 0, "success": 0, "tokenize": 0, "tokenize_cfm": 0,
      "commit_c": 0, "commit_c_tail_ok": 0, "commit_c_cached": [], "cold_salted": 0, "cfm_chat": 0, "probe_turns": 0,
      "stream": 0, "hist": {}, "jit_lines": 0, "contract": 0, "max_inflight": 0, "non_leading_system": 0,
      "prefix_chain_ok": 0, "prefix_chain_checked": 0}
WIN = {"prompt": 0, "gen": 0, "pc_q": 0, "pc_h": 0, "busy": False}
INFLIGHT = [0]
LAST_PROMPT = {}         # conv -> its previous prompt ids (the turn-N+1-extends-turn-N check)


def unit_hashes(ids):
    out, prev = [], 0
    for k in range(len(ids) // UNIT):
        prev = hash((prev, tuple(ids[k * UNIT:(k + 1) * UNIT])))
        out.append(prev)
    return out


def cached_len(ids, salt=None):
    if salt:
        return 0
    hit = 0
    for k, hh in enumerate(unit_hashes(ids)):
        if (k + 1) * UNIT > len(ids) - 1 or hh not in CACHE:
            break
        hit = (k + 1) * UNIT
    return hit


def hist_add(name, v):
    h = ST["hist"].setdefault(name, [0.0, 0])
    h[0] += float(v)
    h[1] += 1


def admit_and_run(rid, ctx, n_tokens):
    t_in = time.time()
    with LOCK:
        WAIT.append(rid)
        INFLIGHT[0] += 1
        ST["max_inflight"] = max(ST["max_inflight"], INFLIGHT[0])
    try:
        while True:
            with LOCK:
                if WAIT and WAIT[0] == rid and len(RUN) < SEQS:
                    WAIT.pop(0)
                    RUN[rid] = ctx
                    ST["max_running"] = max(ST["max_running"], len(RUN))
                    break
            time.sleep(0.005)
        t_run = time.time()
        time.sleep(n_tokens / SPEED)
        with LOCK:
            hist_add("vllm:request_queue_time_seconds", t_run - t_in)
            hist_add("vllm:time_to_first_token_seconds", t_run - t_in + 0.01)
            hist_add("vllm:request_prefill_time_seconds", 0.01)
            hist_add("vllm:e2e_request_latency_seconds", time.time() - t_in)
            hist_add("vllm:request_decode_time_seconds", time.time() - t_run)
    finally:
        with LOCK:
            RUN.pop(rid, None)
            if rid in WAIT:
                WAIT.remove(rid)
            INFLIGHT[0] -= 1


def account(p_ids, hit, n_out, cache_prompt=True):
    with LOCK:
        ST["pc_q"] += len(p_ids)
        ST["pc_h"] += hit
        ST["prompt"] += len(p_ids) - hit
        WIN.update(pc_q=WIN["pc_q"] + len(p_ids), pc_h=WIN["pc_h"] + hit, prompt=WIN["prompt"] + len(p_ids) - hit)
        if cache_prompt:
            for hh in unit_hashes(p_ids):
                CACHE.add(hh)
            RETAINED.append(-(-len(p_ids) // BLOCK) + 3)
            del RETAINED[:-16]
        ST["gen"] += n_out
        WIN["gen"] += n_out
        ST["success"] += 1
        if SPEC:
            k = int(SPEC.get("num_speculative_tokens", 3))
            d = int(n_out / (1 + 0.693 * k))
            ST["drafts"] += d
            ST["dtok"] += d * k
            ST["acc"] += n_out - d
        ST["requests"] += 1
        if ST["requests"] <= 2:          # Merlin's jit_monitor on the first decodes (RC2 lines 441, 551-555)
            ST["jit_lines"] += 1
            log("[jit_monitor.py:140] Triton kernel JIT compilation during inference: _fill_num_accepted_kernel. This causes a "
                "latency spike; consider extending warmup to cover this shape/config.", "Worker_PP0_TP0_EP0")
        pa = os.environ.get("FAKE_PREEMPT_AFTER")
        if pa and ST["tool_requests"] == int(pa) and not ST.get("_preempted"):
            ST["_preempted"] = True
            ST["preempt"] += 1


def usage(p_ids, n_out, cached):
    u = {"prompt_tokens": len(p_ids), "total_tokens": len(p_ids) + n_out, "completion_tokens": n_out}
    if cached > 0:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return u


PLAIN = (("7 * 6 =", "42"), ("2 + 2 =", "4"), ("9 - 3 =", "6"), ("Complete the sequence with the next number only: ", "10"))


def chat(body):
    msgs = body.get("messages") or []
    tools = body.get("tools") or None
    with LOCK:
        ST["non_leading_system"] += int(any(m.get("role") == "system" for m in msgs[1:]))
    try:
        p_ids = render(body)
    except Exception as e:
        return 400, {"object": "error", "message": f"template error: {e}", "code": 400}
    mt = body.get("max_tokens")
    mt = int(mt) if mt is not None else MAXLEN - len(p_ids)
    if len(p_ids) + mt > MAXLEN:
        return 400, {"object": "error", "code": 400, "message": f"This model's maximum context length is {MAXLEN} tokens. "
                     f"However, you requested {len(p_ids) + mt} tokens ({len(p_ids)} in the messages, {mt} in the completion)."}
    hit = cached_len(p_ids, body.get("cache_salt"))
    last = str((msgs[-1].get("content") if msgs else "") or "")
    first_user = next((str(m.get("content") or "") for m in msgs if m.get("role") == "user"), "")
    rid = f"chatcmpl-{time.time_ns()}"
    base = {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": body.get("model")}
    reasoning, content, calls, fin = None, "", [], "stop"
    if body.get("continue_final_message"):              # the probe's server-parser path (closed think in the prompt)
        with LOCK:
            ST["cfm_chat"] += 1
        calls = [{"id": f"chatcmpl-tool-cfm-{time.time_ns()}", "type": "function",
                  "function": {"name": "python", "arguments": json.dumps({"code": "print(7 * 6)"})}}]
        fin = "tool_calls"
        n_out = 24
    elif not tools:
        text = "OK"
        for k, v in PLAIN:
            if k in last:
                text = v
        for k in ("Reply with exactly this code and nothing else: ", "Reply with exactly this code: ", "Repeat exactly, nothing else: "):
            if k in last:
                text = last.split(k, 1)[1].strip()
        if "what is 2 + 2" in last:
            text = "2 + 2 = 4."
        content, n_out = text, len(enc(text))
    elif "Use the python tool exactly once" in first_user or "NB5 commit probe" in first_user:
        turn2 = any(m.get("role") == "assistant" for m in msgs)
        with LOCK:
            ST["contract" if "Use the python" in first_user else "probe_turns"] += 1
        if not turn2:
            reasoning = "The user wants print(6 * 7) run once with the python tool. I will call it now."
            calls = [{"id": f"chatcmpl-tool-probe-{time.time_ns()}", "type": "function",
                      "function": {"name": "python", "arguments": json.dumps({"code": "print(6 * 7)"})}}]
            fin = "tool_calls"
        else:
            reasoning, content = "The tool printed 42.", "The result is 42."
        n_out = len(enc((reasoning or "") + "\n</think>\n\n" + content + call_markup(calls)))
    else:
        with LOCK:
            ST["tool_requests"] += 1
        tid = task_of(str(msgs[0].get("content") or "")) if msgs else None
        hist_text = " ".join(str(m.get("reasoning") or "") for m in msgs if m.get("role") == "assistant")
        conv, tid, role, i = conv_step(tid, hist_text)
        steps = program(tid, role) if tid in CH else [INSPECT_R6]
        code = steps[min(i, len(steps) - 1)]
        with LOCK:                                        # turn N+1's prompt must extend turn N's prompt (prefix check)
            prev = LAST_PROMPT.get(conv)
            if prev is not None:
                ST["prefix_chain_checked"] += 1
                ST["prefix_chain_ok"] += int(p_ids[:len(prev)] == prev)
            LAST_PROMPT[conv] = list(p_ids)
        reasoning = f"CONV={conv} THINK step {i} for {tid} role {role}.\n\nA second paragraph (a blank line above, as models write).\n"
        calls = [{"id": f"chatcmpl-tool-{conv}-{i}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        fin = "tool_calls"
        n_out = max(1, min(REPLY_TOKENS, mt - 1))
        if code == CUT_CODE:
            n_out = mt                                    # a cut call: vLLM says 'tool_calls' with completion == max_tokens
    admit_and_run(rid, len(p_ids), min(n_out, 300))
    account(p_ids, hit, n_out)
    msg = {"role": "assistant", "content": content or None, "reasoning": reasoning, "tool_calls": calls}
    ch = {"index": 0, "message": msg, "logprobs": None, "finish_reason": fin, "stop_reason": None}
    return 200, dict(base, choices=[ch], usage=usage(p_ids, n_out, hit))


def completion(body):
    """Non-streaming /v1/completions: the commit turn variant C, the probe's C request, anything else plain text."""
    ids = body.get("prompt")
    if isinstance(ids, str):
        ids = enc(ids)
    ids = list(ids or [])
    mt = int(body.get("max_tokens") or 16)
    hit = cached_len(ids, body.get("cache_salt"))
    text_in = dec(ids)
    tail_ok = ids[-len(TAIL_C):] == TAIL_C
    with LOCK:
        ST["completions"] += 1
        ST["cold_salted"] += int(bool(body.get("cache_salt")))
    if "NB5 commit probe" in text_in:
        out_text = "<tool_call>\n<function=python>\n<parameter=code>\nprint(7 * 6)\n</parameter>\n</function>\n</tool_call>"
    elif tail_ok:
        with LOCK:
            ST["commit_c"] += 1
            ST["commit_c_tail_ok"] += 1
            ST["commit_c_cached"].append([len(ids), hit])
            ST["tool_requests"] += 1
        sys_text = text_in.split("<|im_end|>", 1)[0]
        tid = task_of(sys_text)
        conv, tid, role, i = conv_step(tid, text_in)
        steps = program(tid, role) if tid in CH else [INSPECT_R6]
        code = steps[min(i, len(steps) - 1)]
        if os.environ.get("FAKE_C_NOCALL") == "1":
            out_text = "I would submit my best transform now."
        else:
            out_text = ("Committing the best version.\n" + call_markup([{"function": {"name": "python",
                                                                                      "arguments": json.dumps({"code": code})}}]))
    else:
        out_text = "plain completion"
    n_out = min(mt, len(enc(out_text)))
    admit_and_run(f"cmpl-{time.time_ns()}", len(ids), n_out)
    account(ids, hit, n_out, cache_prompt=not body.get("cache_salt"))
    return {"id": f"cmpl-{time.time_ns()}", "object": "text_completion", "model": body.get("model"),
            "choices": [{"index": 0, "text": out_text, "finish_reason": "stop", "logprobs": None}],
            "usage": usage(ids, n_out, hit)}


_SYL = ["ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa", "qu", "ri", "bel", "dor", "fen", "gul"]
_VOCAB = [a + b + c for a in _SYL for b in _SYL for c in _SYL]


def stream_completion(handler, body):
    """E23 fake semantics for the lab's Stream: SSE chunks with token_ids, ignore_eos to max_tokens, running capped."""
    import random
    ids = list(body.get("prompt") or [])
    max_t = int(body.get("max_tokens", 16))
    rid = f"s{time.time_ns()}"
    with LOCK:
        ST["stream"] += 1
        WAIT.append(rid)
    while True:
        with LOCK:
            if WAIT and WAIT[0] == rid and len(RUN) < SEQS:
                WAIT.pop(0)
                RUN[rid] = len(ids)
                break
        time.sleep(0.005)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    rng = random.Random(hashlib.sha256((json.dumps(ids[-64:]) + str(time.time_ns())).encode()).hexdigest())
    sent = 0
    try:
        first = True
        while sent < max_t:
            n = min(8, max_t - sent)
            chunk = [_VOCAB[rng.randrange(len(_VOCAB))] + " " for _ in range(n)]
            sent += n
            d = {"choices": [{"text": "".join(chunk), "index": 0, "finish_reason": "length" if sent >= max_t else None,
                              "token_ids": [1000 + rng.randrange(4000) for _ in chunk]}]}
            b = b"data: " + json.dumps(d).encode() + b"\n\n"
            handler.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
            handler.wfile.flush()
            with LOCK:
                if first:
                    account(ids, cached_len(ids), 0)
                    first = False
                ST["gen"] += n
                WIN["gen"] += n
                RUN[rid] = RUN.get(rid, 0) + n
            time.sleep(n / SPEED)
        b = b"data: [DONE]\n\n"
        handler.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" + b"0\r\n\r\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    finally:
        with LOCK:
            RUN.pop(rid, None)
            if rid in WAIT:
                WAIT.remove(rid)


def stats_json():
    with LOCK:
        d = {k: (list(v) if isinstance(v, list) else v) for k, v in ST.items() if not k.startswith("_")}
        d.update(running=len(RUN), waiting=len(WAIT), argv=argv, port=PORT, seqs=SEQS, spec=SPEC, pid=os.getpid())
    return d


def dump_stats():
    if STATE_DIR:
        try:
            p = os.path.join(STATE_DIR, f"stats_{os.getpid()}.json")
            with open(p + ".tmp", "w", encoding="utf-8") as f:
                json.dump(stats_json(), f)
            os.replace(p + ".tmp", p)
        except Exception:
            pass


def printer():
    while True:
        time.sleep(2.0)
        with LOCK:
            n, nq = len(RUN), len(WAIT)
            ctx_blocks = sum(-(-c // BLOCK) + FIXED for c in RUN.values()) + sum(RETAINED)
            w = dict(WIN)
            WIN.update(prompt=0, gen=0, pc_q=0, pc_h=0, busy=bool(RUN or WAIT))
            pq, ph = ST["pc_q"], ST["pc_h"]
        if n or nq or w["busy"] or w["gen"] or w["prompt"]:
            line = (f"[loggers.py:335] Engine 000: Avg prompt throughput: {w['prompt'] / 2.0:.1f} tokens/s, Avg generation "
                    f"throughput: {w['gen'] / 2.0:.1f} tokens/s, Running: {n} reqs, Waiting: {nq} reqs")
            if nq:
                line += f", Deferred: {nq} reqs"
            line += f", GPU KV cache usage: {100.0 * min(1.0, ctx_blocks / POOL):.1f}%"
            if pq:
                line += f", Prefix cache hit rate: {100.0 * ph / pq:.1f}%"
            log(line)
            if SPEC and w["gen"]:
                log("[metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.08, Accepted throughput: 1.00 tokens/s, "
                    "Drafted throughput: 1.00 tokens/s, Accepted: 1 tokens, Drafted: 1 tokens, Per-position acceptance rate: "
                    "0.880, 0.700, 0.500, Avg Draft acceptance rate: 69.4%")
        dump_stats()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, code=200, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            return self._send("", 200)
        if p == "/v1/models":
            return self._send(json.dumps({"object": "list", "data": [{"id": SERVED, "object": "model", "max_model_len": MAXLEN}]}))
        if p == "/metrics":
            lab = f'{{engine="0",model_name="{SERVED}"}}'
            with LOCK:
                n, nq = len(RUN), len(WAIT)
                blocks = sum(-(-c // BLOCK) + FIXED for c in RUN.values()) + sum(RETAINED)
                lines = [f"vllm:num_requests_running{lab} {float(n)}", f"vllm:num_requests_waiting{lab} {float(nq)}",
                         f"vllm:kv_cache_usage_perc{lab} {min(1.0, blocks / POOL):.6f}",
                         f"vllm:generation_tokens_total{lab} {float(ST['gen'])}",
                         f"vllm:prompt_tokens_total{lab} {float(ST['prompt'])}",
                         f"vllm:num_preemptions_total{lab} {float(ST['preempt'])}",
                         f"vllm:request_success_total{lab} {float(ST['success'])}",
                         f"vllm:prefix_cache_queries_total{lab} {float(ST['pc_q'])}",
                         f"vllm:prefix_cache_hits_total{lab} {float(ST['pc_h'])}",
                         f'vllm:cache_config_info{{block_size="{BLOCK}",cache_dtype="{KV_DTYPE}",engine="0",num_gpu_blocks="{POOL + 1}",'
                         f'enable_prefix_caching="True"}} 1.0']
                for h, (s_, c_) in ST["hist"].items():
                    lines += [f"{h}_sum{lab} {s_}", f"{h}_count{lab} {float(c_)}"]
                if SPEC:
                    lines += [f"vllm:spec_decode_num_accepted_tokens_total{lab} {float(ST['acc'])}",
                              f"vllm:spec_decode_num_drafts_total{lab} {float(ST['drafts'])}",
                              f"vllm:spec_decode_num_draft_tokens_total{lab} {float(ST['dtok'])}"]
            return self._send("\n".join(lines) + "\n", ctype="text/plain; version=0.0.4")
        if p == "/fake_stats":
            return self._send(json.dumps(stats_json()))
        return self._send(json.dumps({"detail": "Not Found"}), 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(json.dumps({"object": "error", "message": "bad json", "code": 400}), 400)
        p = self.path.split("?")[0]
        try:
            if p == "/tokenize":
                if os.environ.get("FAKE_NO_TOKENIZE") == "1":
                    return self._send(json.dumps({"detail": "Not Found"}), 404)
                with LOCK:
                    ST["tokenize"] += 1
                    ST["tokenize_cfm"] += int(bool(body.get("continue_final_message")))
                ids = render(body) if "messages" in body else enc(str(body.get("prompt") or ""))
                return self._send(json.dumps({"count": len(ids), "max_model_len": MAXLEN, "tokens": ids}))
            if p == "/v1/chat/completions":
                code, out = chat(body)
                return self._send(json.dumps(out), code)
            if p == "/v1/completions":
                if body.get("stream"):
                    return stream_completion(self, body)
                return self._send(json.dumps(completion(body)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as e:
            return self._send(json.dumps({"object": "error", "message": f"fake error: {type(e).__name__}: {e}", "code": 500}), 500)
        return self._send(json.dumps({"detail": "Not Found"}), 404)


threading.Thread(target=printer, daemon=True).start()
ThreadingHTTPServer.request_queue_size = 256
if os.name == "nt":
    ThreadingHTTPServer.allow_reuse_address = False
try:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
except OSError as e:
    print(f"OSError: fake Merlin cannot bind 127.0.0.1:{PORT}: {e}", flush=True)
    launch_record("port_in_use")
    sys.exit(1)
srv.daemon_threads = True
launch_record("ready", spec=SPEC, seqs=SEQS)
log(f"[entry.py:140] Starting vLLM server on http://127.0.0.1:{PORT}")
print("INFO:     Application startup complete.", flush=True)
srv.serve_forever()
