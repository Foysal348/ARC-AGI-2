"""E21 fake vLLM OpenAI server for the CPU notebook harness (no model, no GPU). Launched by the REAL E21 engine cell as
[sys.executable, fake_vllm_duck_server.py] + flags, in either argv form:
  vLLM 0.19  `python -m vllm.entrypoints.openai.api_server --model <path> --flag v ...`  -> argv = --model <path> ...
  vLLM 0.32  `vllm serve <path> --flag v ...`                                           -> argv = [serve] <path> ...
Derived from Exp 14 e14_fake_server.py (the E12 rev 6 scripted Duck program, copied verbatim below) and the vLLM-shaped
parts of Exp 14 fake_vllm_server.py / Exp 19 fake_vllm019_server.py.

Emulated vLLM behaviour
- refuses to start (exit 2) without Duck's serving flags (--enable-auto-tool-choice, --tool-call-parser qwen3_coder,
  --reasoning-parser qwen3, --default-chat-template-kwargs with preserve_thinking true, --enable-prefix-caching,
  --max-model-len >= 131072) or with a served name Duck aliases (local, local-qwen, qwen-local, qwen)
- refuses unknown flags ("unrecognized arguments", exit 2): the accepted set = the flags the E21 notebooks pass;
  prints the "non-default args: {...}" line (e.g. 'language_model_only': True), as vLLM does
- startup lines (rev 2, the vLLM hybrid layout): "Setting attention block size to B tokens ..." (B = 1,600, or 832 with
  --mamba-ssm-cache-dtype float16), "GPU KV cache size: num_blocks // 4 x B tokens", "Maximum concurrency for M tokens
  per request: num_blocks / (cdiv(M, B) + 3 x (2 + spec tokens))x", num_blocks 846 FP8 / 1,069 int4|w4a16 / 1,628 at
  B 832; then the port is bound and /health answers
- every 2 s while anything ran: "Engine 000: Avg prompt throughput ... Running: N reqs, Waiting: M reqs[, Deferred: D
  reqs (0.32)][, Preemptions: P], GPU KV cache usage: K%[, Prefix cache hit rate: Q% (after the first prefix query)]"
  (every 4th busy line reports one simulated preemption, also in vllm:num_preemptions_total) and, with
  --speculative-config, "SpecDecoding metrics: Mean acceptance length: 3.08, ..."
- a tool call cut at max_tokens (the scripted CUT_CODE step) answers finish_reason "tool_calls" with completion_tokens
  == max_tokens (vLLM 0.19 / 0.32 serving.py; SGLang said "length")
- the launch cache probe ("Reply with exactly this code: QZ...K<i>", ~9,000 digits) echoes the code; the repeat
  request hits the 1,600-block cache
- the notebook's contract probe ("Use the python tool exactly once ...") gets reasoning + one python call
  {"code": "print(6 * 7)"} in turn 1 and a plain answer in turn 2; completion tokens = the rendered reply's tokens
- GET /health, /v1/models, /metrics (vllm:* names with {model_name="..."} labels), /fake_stats (JSON)
- POST /v1/chat/completions (non-streaming; stream=true is answered as SSE): the reply carries the thinking under
  `reasoning` (vLLM 0.19+), tool calls in `tool_calls`; usage.prompt_tokens_details.cached_tokens only when > 0 and
  --enable-prompt-tokens-details is set (vLLM omits it otherwise); prompt + max_tokens > max-model-len -> HTTP 400
- INPUT: only message["reasoning"] of an assistant history message reaches the prompt (vLLM 0.19
  entrypoints/chat_utils.py:1512, 0.32 renderers/chat_utils.py:2083); `reasoning_content` is ignored and counted
- prefix cache: cached = the longest token prefix shared with any earlier prompt + output, in whole 1,600-token blocks
  (vLLM mamba 'align' block size on this model), capped at prompt - 1 (a block-hash chain, as vLLM)
- running requests are capped at --max-num-seqs (the rest wait, FIFO); gauges and counters follow

Fault injection (env, all optional; FORM = 0.19 | 0.32 limits a fault to one argv form)
  FAKE_REJECT_FLAG=<flag>            exit 1 with a traceback if that exact argv element is present (any form)
  FAKE_NUM_BLOCKS=<FORM>:<n>[,...]   log n KV blocks for that form (fewer streams fit: the gate A/B test)
  FAKE_DROP_BLOCK_LINE=<FORM>        omit the "Setting attention block size" line (the notebook must refuse)
  FAKE_REJECT=<FORM>:<substr>[,...]  exit 1 if an argv element contains <substr> in that form ("*" = any form)
  FAKE_REJECT_UNLESS=<FORM>:<substr> exit 1 if NO argv element contains <substr> in that form (rev 2i)
  FAKE_NEVER_READY=1 | <FORM>        print the loading lines, then never bind the port (no /health, no KV line)
  FAKE_DIE_AFTER_SEC=<s>             exit (os._exit 1) s seconds after the first tool request (mid-solve)
  FAKE_DIE_FORM=<FORM>               limit the death to one form; FAKE_DIE_TIMES=<n> (default 1): the first n
                                     servers of that form die (marker files in FAKE_STATE_DIR), the next survives
  FAKE_ACCEPT_LEN=<FORM>:<x>         the MTP mean acceptance length that form logs (default 1 + 0.693 x k = 3.08)
  FAKE_QUIET=<FORM>                  no periodic stats lines (a silent log: the watchdog probes /health)
  FAKE_HOLD_ONE=<FORM>:<n>:<sec>     the n-th Duck tool request is held <sec> s in its running slot (deadline test)
  FAKE_METRICS_FAIL=<FORM>:<a>:<d>   /metrics answers 500 from (first tool request + a) for d s (KV probe outage)
  FAKE_STATE_DIR=<dir>               launches.jsonl (one row per start: argv, form, outcome) + stats_<pid>.json
  FAKE_SPEED (tokens/s per request, default 2000), E14_FIXTURES (fixture dir)
Scripted replies only (our own text and code)."""
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIX = os.environ.get("E14_FIXTURES", r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
sys.path.insert(0, FIX)
import e12_fixtures as B  # noqa: E402

argv = sys.argv[1:]
if argv and argv[0] == "serve":
    argv = argv[1:]
FORM = "0.32" if argv and not argv[0].startswith("--") else "0.19"
STATE_DIR = os.environ.get("FAKE_STATE_DIR") or ""
T_START = time.time()


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default


def ts():
    return time.strftime("%m-%d %H:%M:%S")


def log(line, who="APIServer"):
    print(f"({who} pid={os.getpid()}) INFO {ts()} {line}", flush=True)


def launch_record(outcome, **kw):
    if not STATE_DIR:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(os.path.join(STATE_DIR, "launches.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "pid": os.getpid(), "form": FORM, "outcome": outcome,
                                "argv": argv, **kw}) + "\n")
    except Exception:
        pass


def form_match(spec):
    s = (spec or "").strip()
    return s in ("1", "*", "true", "yes") or s == FORM


MODEL = argv[0] if FORM == "0.32" else (arg("--model") or "?")
PORT = int(arg("--port", 8000))
SEQS = int(arg("--max-num-seqs", 256))
SERVED = arg("--served-model-name", MODEL)
MAXLEN = int(arg("--max-model-len", 262144))
SSM = arg("--mamba-ssm-cache-dtype")
KVD = arg("--kv-cache-dtype", "auto")
SPEC = json.loads(arg("--speculative-config", "null") or "null")
PTD = "--enable-prompt-tokens-details" in argv
SPEED = float(os.environ.get("FAKE_SPEED", 2000))
INT4 = any(k in MODEL.lower() for k in ("int4", "w4a16"))
# vLLM hybrid (Mamba) KV layout, rev 2: the attention block size is raised so one attention page >= one Mamba page
# (1,600 tokens with the default SSM dtype, 832 with --mamba-ssm-cache-dtype float16); the pool has num_blocks blocks
# (E12-V log: 846 for FP8 at 0.90); "Maximum concurrency for M tokens" = num_blocks / (cdiv(M, B) + F), F = the fixed
# Mamba blocks per request (align mode: 3 groups x (2 + spec tokens) = 15 with MTP-3); "GPU KV cache size" = num_blocks
# // 4 x B (the pool is shared by 1 attention + 3 GDN groups)
BLOCK = 832 if SSM == "float16" else 1600
_NB_BASE = (1628 if BLOCK == 832 else 846)
NUM_BLOCKS = int(round(_NB_BASE * (1069 / 846))) if INT4 else _NB_BASE
for _r in [x for x in os.environ.get("FAKE_NUM_BLOCKS", "").split(",") if ":" in x]:   # FORM:N (gate tests: fewer blocks)
    if (_r.split(":", 1)[0].strip() in ("*", FORM)):
        NUM_BLOCKS = int(_r.split(":", 1)[1])
SPEC_K = int((SPEC or {}).get("num_speculative_tokens", 0) or 0)


def _form_spec(name):
    """FORM:a:b env values -> [a, b] floats for this form (or None)."""
    for _r in [x for x in os.environ.get(name, "").split(",") if x.count(":") >= 2]:
        f_, a_, b_ = _r.split(":")[:3]
        if f_.strip() in ("*", FORM):
            return [float(a_), float(b_)]
    return None


HOLD_ONE = _form_spec("FAKE_HOLD_ONE")            # [n, sec]: the n-th Duck tool request is held sec seconds
METRICS_FAIL = _form_spec("FAKE_METRICS_FAIL")    # [start, dur]: /metrics 500 from first tool request + start, dur s
FIXED_BLOCKS = 3 * (2 + SPEC_K)
KV_TOKENS = NUM_BLOCKS * BLOCK                 # token capacity used for the KV-usage gauge
KV_LINE_TOKENS = NUM_BLOCKS // 4 * BLOCK
print(f"fake vllm {FORM} argv: {' '.join(argv)}", flush=True)
log(f"[api_server.py:1] vLLM API server version {'0.19.0' if FORM == '0.19' else '0.32.0'}")

# ------------------------------------------------------------------ refusals (argparse-like, exit 2)
problems = []
# the flags the E21 notebooks pass (0.19 api_server and 0.32 serve accept the same set here); anything else is refused
# as argparse would ("unrecognized arguments"); the Kaggle CPU kit checks the real parsers
VALUE_FLAGS = {"--model", "--served-model-name", "--host", "--port", "--tool-call-parser", "--generation-config",
               "--default-chat-template-kwargs", "--reasoning-parser", "--max-model-len", "--tensor-parallel-size",
               "--kv-cache-dtype", "--speculative-config", "--attention-backend", "--max-num-batched-tokens",
               "--max-num-seqs", "--gpu-memory-utilization", "--seed", "--mamba-ssm-cache-dtype"}
BOOL_FLAGS = {"--enable-auto-tool-choice", "--enable-prefix-caching", "--enable-prompt-tokens-details",
              "--disable-uvicorn-access-log", "--language-model-only", "--disable-custom-all-reduce", "--trust-remote-code"}
_i = 1 if FORM == "0.32" else 0
_unknown = []
while _i < len(argv):
    a = argv[_i]
    if a in VALUE_FLAGS:
        _i += 2
    elif a in BOOL_FLAGS:
        _i += 1
    else:
        _unknown.append(a)
        _i += 1
if _unknown:
    problems.append(f"unrecognized arguments: {' '.join(_unknown)}")
_nd, _i = {"model": MODEL} if FORM == "0.32" else {}, 1 if FORM == "0.32" else 0
while _i < len(argv):
    a = argv[_i]
    if a in VALUE_FLAGS and _i + 1 < len(argv):
        v = argv[_i + 1]
        _nd[a[2:].replace("-", "_")] = int(v) if v.isdigit() else v
        _i += 2
    else:
        if a in BOOL_FLAGS:
            _nd[a[2:].replace("-", "_")] = True
        _i += 1
log(f"[utils.py:233] non-default args: {_nd!r}")
if "--enable-auto-tool-choice" not in argv:
    problems.append("--enable-auto-tool-choice missing")
if arg("--tool-call-parser") != "qwen3_coder":
    problems.append(f"--tool-call-parser {arg('--tool-call-parser')!r} (need qwen3_coder)")
if arg("--reasoning-parser") != "qwen3":
    problems.append(f"--reasoning-parser {arg('--reasoning-parser')!r} (need qwen3)")
try:
    _kw = json.loads(arg("--default-chat-template-kwargs", "{}") or "{}")
except Exception:
    _kw = {}
if not (isinstance(_kw, dict) and _kw.get("preserve_thinking") is True):
    problems.append(f"--default-chat-template-kwargs {arg('--default-chat-template-kwargs')!r} (need preserve_thinking true)")
if "--enable-prefix-caching" not in argv:
    problems.append("--enable-prefix-caching missing")
if MAXLEN < 131072:
    problems.append(f"--max-model-len {MAXLEN} < 131072 (Duck context)")
if str(SERVED).strip().lower() in {"local", "local-qwen", "qwen-local", "qwen"}:
    problems.append(f"--served-model-name {SERVED!r} is a Duck alias (E12-V run 1 bug)")
if FORM == "0.19" and "--model" not in argv:
    problems.append("0.19 form without --model")
if problems:
    print("usage: vllm serve [model_tag] [options]\nerror: fake vLLM (E21 harness) refuses the launch: "
          + "; ".join(problems), flush=True)
    launch_record("refused", problems=problems)
    sys.exit(2)


def _reject(why):
    print("Traceback (most recent call last):\n"
          '  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 1987, in <module>\n'
          "    uvloop.run(run_server(args))\n"
          '  File "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 702, in run_engine_core\n'
          "    raise e\n"
          f"RuntimeError: Engine core initialization failed. See root cause above. (fake: {why})", flush=True)
    launch_record("rejected", why=why)
    sys.exit(1)


if os.environ.get("FAKE_REJECT_FLAG") and os.environ["FAKE_REJECT_FLAG"] in argv:
    time.sleep(0.3)
    _reject(f"FAKE_REJECT_FLAG {os.environ['FAKE_REJECT_FLAG']}")
for _r in [x for x in os.environ.get("FAKE_REJECT_UNLESS", "").split(",") if ":" in x]:
    _f, _sub = _r.split(":", 1)
    if form_match(_f) and not any(_sub in str(a) for a in argv):
        time.sleep(0.3)
        _reject(f"FAKE_REJECT_UNLESS {_r}")
for _r in [x for x in os.environ.get("FAKE_REJECT", "").split(",") if ":" in x]:
    _f, _sub = _r.split(":", 1)
    if form_match(_f) and any(_sub in str(a) for a in argv):
        time.sleep(0.3)
        _reject(f"FAKE_REJECT {_r}")

log(f"[model.py:1] Resolved architecture: Qwen3_5ForConditionalGeneration (quant {'compressed-tensors' if INT4 else 'fp8'})",
    "EngineCore")
if SSM:
    log(f"[config.py:1] mamba_ssm_cache_dtype={SSM}", "EngineCore")
if not form_match(os.environ.get("FAKE_DROP_BLOCK_LINE", "")):     # dropping it tests the notebook's fail-closed check
    log(f"[config.py:2] Setting attention block size to {BLOCK} tokens to ensure that attention page size is >= mamba "
        "page size.", "EngineCore")
log("[config.py:3] Mamba cache mode is set to 'align' for Qwen3_5ForCausalLM by default when prefix caching is enabled",
    "EngineCore")
time.sleep(0.3)
if form_match(os.environ.get("FAKE_NEVER_READY", "")):
    log("[gpu_model_runner.py:1] Starting to load model ...", "EngineCore")
    launch_record("never_ready")
    while True:                     # silent forever: no KV line, no port (the notebook's stall / timeout decides)
        time.sleep(60)
log(f"[kv_cache_utils.py:1087] GPU KV cache size: {KV_LINE_TOKENS:,} tokens", "EngineCore")
log(f"[kv_cache_utils.py:1091] Maximum concurrency for {MAXLEN:,} tokens per request: "
    f"{NUM_BLOCKS / (-(-MAXLEN // BLOCK) + FIXED_BLOCKS):.2f}x", "EngineCore")

# ------------------------------------------------------------------ the E12 rev 6 scripted Duck program (from e14_fake_server.py)
CH, SOL, _, _ = B.load()
CONV_STEP, CONV_META, TASK_ROLES, NEW_CONVS = {}, {}, {}, []
CONV_RE = re.compile(r"CONV=(\w+)")


def rows(g):
    return "\n".join(" ".join(str(v) for v in r) for r in g)


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
    for marker in ("Training pair 1 input:\n", "Training pair 0 input:\n"):
        if marker in system_text:
            head = system_text.split(marker, 1)[-1].split("\nTraining pair " + marker[14] + " output:", 1)[0]
            for tid, t in CH.items():
                if rows(t["train"][0]["input"]) == head:
                    return tid
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
NOCALL = "@@NOCALL@@"
INSPECT_R6 = "print(grid_shape(train[0]['input']), len(components(train[0]['input'])))\ninspect_pair(0)"


def lookup_r6(tid, name="transform"):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    for qi, q in enumerate(t["test"]):
        table[json.dumps(q["input"])] = SOL[tid][qi]
    return (f"def {name}(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), flip_h(flip_h(g)))")


def lookup_train_only_r6(tid):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    return ("def transform_alt(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), [list(r) for r in g])")


def program(tid, role):
    return ([NOCALL, INSPECT_R6, lookup_r6(tid), INSPECT_R6] if role == 0
            else [CUT_CODE, lookup_train_only_r6(tid), INSPECT_R6])


# ------------------------------------------------------------------ prompt render + tokens (vLLM input semantics)
TOK_RE = re.compile(r"\d|[A-Za-z]+|\s+|[^\sA-Za-z\d]")     # digits one by one (as Qwen), words, spaces, symbols


def tok(text):
    return TOK_RE.findall(text)


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


def _content_text(c):
    if isinstance(c, list):
        return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def render(msgs, tools):
    """Stand-in for the Qwen template as vLLM feeds it: tools first; an assistant message renders <think> + its
    `reasoning` (vLLM copies message["reasoning"] to reasoning_content for the template; message["reasoning_content"]
    on input is dropped) + content + tool calls."""
    s = ("<|im_start|>system\n# Tools\n" + json.dumps(tools, sort_keys=True) + "<|im_end|>\n") if tools else ""
    for m in msgs:
        r = m.get("role")
        if r == "assistant":
            reasoning = m.get("reasoning")
            s += "<|im_start|>assistant\n<think>\n" + (str(reasoning).strip() if reasoning is not None else "") + \
                 "\n</think>\n\n" + _content_text(m.get("content")) + call_markup(m.get("tool_calls")) + "<|im_end|>\n"
        else:
            s += f"<|im_start|>{r}\n" + _content_text(m.get("content")) + "<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n"


def h64(*xs):
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), "little")


def block_hashes(ids):
    out, prev = [], 0
    for k in range(len(ids) // BLOCK):
        prev = h64(prev, tuple(ids[k * BLOCK:(k + 1) * BLOCK]))
        out.append(prev)
    return out


LOCK = threading.Lock()
CACHE = set()
RUN, WAIT = {}, []              # rid -> context tokens (admitted = running); FIFO of waiting rids
ST = {"requests": 0, "chat": 0, "tool_requests": 0, "max_running": 0, "max_inflight": 0, "gen": 0, "prompt": 0,
      "pc_q": 0, "pc_h": 0, "drafts": 0, "acc": 0, "dtok": 0, "context_rejections": 0, "preflight": 0, "smoke": 0,
      # assistant history messages by the reasoning key they carried (vLLM reads only `reasoning`)
      "hist_msgs_reasoning": 0, "hist_msgs_reasoning_content_only": 0, "hist_msgs_both": 0, "hist_msgs_neither": 0,
      "requests_with_history": 0, "requests_history_reasoning": 0, "requests_history_reasoning_content_only": 0,
      "ignored_reasoning_content_chars": 0, "stream_requests": 0, "seen_max_tokens": [], "seen_kwargs": [],
      "first_tool_t": None, "cached_reported": 0, "cached_zero_omitted": 0, "preempt": 0, "success": 0,
      "cut_tool_calls": 0, "cut_delivered": 0, "contract_turn1": 0, "contract_turn2": 0, "cache_probe": 0, "held": [], "metrics_500": 0,
      "metrics_fail_window": None, "conv_starts": []}
WIN = {"prompt": 0, "gen": 0, "pc_q": 0, "pc_h": 0, "busy": False}
INFLIGHT = [0]


def stats_json():
    with LOCK:
        d = {k: (list(v) if isinstance(v, list) else v) for k, v in ST.items()}
        d.update(running=len(RUN), waiting=len(WAIT), form=FORM, argv=argv, port=PORT, max_num_seqs=SEQS, served=SERVED,
                 max_model_len=MAXLEN, ssm=SSM, kv_tokens=KV_LINE_TOKENS,
                 num_blocks=NUM_BLOCKS, block=BLOCK, fixed_blocks=FIXED_BLOCKS, int4=INT4, pid=os.getpid(), new_convs=len(NEW_CONVS),
                 uptime=round(time.time() - T_START, 1))
    d["seen_max_tokens"] = d["seen_max_tokens"][-50:]
    d["seen_kwargs"] = d["seen_kwargs"][-20:]
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
    acc_len = 1 + 0.693 * int((SPEC or {}).get("num_speculative_tokens", 0))
    for _r in [x for x in os.environ.get("FAKE_ACCEPT_LEN", "").split(",") if ":" in x]:   # FORM:x (an MTP mismatch)
        if _r.split(":", 1)[0].strip() in ("*", FORM):
            acc_len = float(_r.split(":", 1)[1])
    k_line = 0
    while True:
        time.sleep(2.0)
        with LOCK:
            n, nq = len(RUN), len(WAIT)
            ctx = sum(RUN.values())
            w = dict(WIN)
            WIN.update(prompt=0, gen=0, pc_q=0, pc_h=0, busy=bool(RUN or WAIT))
            pq, ph = ST["pc_q"], ST["pc_h"]
        if (n or nq or w["busy"] or w["gen"] or w["prompt"]) and not form_match(os.environ.get("FAKE_QUIET", "")):
            k_line += 1
            # vLLM 0.19 inserts ", Preemptions: N" (N in this interval) when there were any, 0.32 also ", Deferred: N
            # reqs"; the prefix-cache field is absent until the first prefix query. Here every 4th busy line carries a
            # (reported, simulated) preemption, counted in vllm:num_preemptions_total too
            pre = 1 if (n and k_line % 4 == 0) else 0
            with LOCK:
                ST["preempt"] += pre
            line = (f"[loggers.py:257] Engine 000: Avg prompt throughput: {w['prompt'] / 2.0:.1f} tokens/s, Avg generation "
                    f"throughput: {w['gen'] / 2.0:.1f} tokens/s, Running: {n} reqs, Waiting: {nq} reqs")
            if FORM == "0.32" and pre:
                line += f", Deferred: {pre} reqs"
            if pre:
                line += f", Preemptions: {pre}"
            line += f", GPU KV cache usage: {100.0 * min(1.0, ctx / KV_TOKENS):.1f}%"
            if pq:
                line += f", Prefix cache hit rate: {100.0 * ph / pq:.1f}%"
            log(line)
            if SPEC and w["gen"]:
                k = int(SPEC.get("num_speculative_tokens", 3))
                dr = int(w["gen"] / acc_len)
                log(f"[metrics.py:96] SpecDecoding metrics: Mean acceptance length: {acc_len:.2f}, Accepted throughput: "
                    f"{(w['gen'] - dr) / 2.0:.2f} tokens/s, Drafted throughput: {dr * k / 2.0:.2f} tokens/s, Accepted: "
                    f"{w['gen'] - dr} tokens, Drafted: {dr * k} tokens, Per-position acceptance rate: 0.880, 0.700, 0.500, "
                    f"Avg Draft acceptance rate: {100.0 * (acc_len - 1) / k:.1f}%")
        dump_stats()


def die_watch():
    s = os.environ.get("FAKE_DIE_AFTER_SEC")
    if not s or not form_match(os.environ.get("FAKE_DIE_FORM", "*")):
        return
    # FAKE_DIE_TIMES (default 1; FAKE_DIE_ONCE=0 = every server): the first N servers of this form die, the next survives
    times = int(os.environ.get("FAKE_DIE_TIMES", "1")) if os.environ.get("FAKE_DIE_ONCE", "1") == "1" else 10 ** 9
    died = [f for f in os.listdir(STATE_DIR) if f.startswith(f"died_{FORM}_")] if STATE_DIR and os.path.isdir(STATE_DIR) else []
    if STATE_DIR and len(died) >= times:
        return
    marker = os.path.join(STATE_DIR, f"died_{FORM}_{len(died)}") if STATE_DIR else None
    while True:
        time.sleep(0.2)
        with LOCK:
            t1 = ST["first_tool_t"]
        if t1 is not None and time.time() - t1 >= float(s):
            break
    if marker:
        open(marker, "w").write(str(os.getpid()))
    log("[core.py:1] fake: FAKE_DIE_AFTER_SEC reached -> the engine core dies", "EngineCore")
    print("RuntimeError: fake engine death (E21 harness)", flush=True)
    launch_record("died", running=len(RUN))
    dump_stats()
    os._exit(1)


def admit_and_run(rid, ctx, n_tokens):
    """Hold the request: wait for a free max-num-seqs slot (FIFO), then decode for n_tokens / FAKE_SPEED seconds."""
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
                    assert len(RUN) <= SEQS
                    break
            time.sleep(0.01)
        time.sleep(n_tokens / SPEED)
    finally:
        with LOCK:
            RUN.pop(rid, None)
            if rid in WAIT:
                WAIT.remove(rid)
            INFLIGHT[0] -= 1


def history_keys(msgs):
    """Count the reasoning keys of the assistant history messages (what the notebook sends back)."""
    r = rc = both = neither = 0
    ign = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        a = isinstance(m.get("reasoning"), str) and m.get("reasoning") != ""
        b = isinstance(m.get("reasoning_content"), str) and m.get("reasoning_content") != ""
        if a and b:
            both += 1
        elif a:
            r += 1
        elif b:
            rc += 1
            ign += len(m["reasoning_content"])
        else:
            neither += 1
    return r, rc, both, neither, ign


def usage(p_ids, n_out, cached):
    u = {"prompt_tokens": len(p_ids), "total_tokens": len(p_ids) + n_out, "completion_tokens": n_out}
    with LOCK:
        if cached > 0 and PTD:
            u["prompt_tokens_details"] = {"cached_tokens": cached}
            ST["cached_reported"] += 1
        else:
            ST["cached_zero_omitted"] += 1
    return u


def chat(body):
    msgs = body.get("messages") or []
    tools = body.get("tools") or None
    p_ids = tok(render(msgs, tools))
    mt = body.get("max_tokens") or body.get("max_completion_tokens")
    mt = int(mt) if mt is not None else MAXLEN - len(p_ids)
    if len(p_ids) + mt > MAXLEN:
        with LOCK:
            ST["context_rejections"] += 1
        return 400, {"object": "error", "type": "BadRequestError", "param": None, "code": 400,
                     "message": f"This model's maximum context length is {MAXLEN} tokens. However, you requested "
                                f"{len(p_ids) + mt} tokens ({len(p_ids)} in the messages, {mt} in the completion). "
                                "Please reduce the length of the messages or completion."}
    r, rc, both, neither, ign = history_keys(msgs)
    with LOCK:
        ST["requests"] += 1
        ST["chat"] += 1
        ST["seen_max_tokens"].append(mt)
        ST["seen_kwargs"].append(body.get("chat_template_kwargs"))
        if r or rc or both or neither:
            ST["requests_with_history"] += 1
            ST["hist_msgs_reasoning"] += r
            ST["hist_msgs_reasoning_content_only"] += rc
            ST["hist_msgs_both"] += both
            ST["hist_msgs_neither"] += neither
            ST["ignored_reasoning_content_chars"] += ign
            if r or both:
                ST["requests_history_reasoning"] += 1
            elif rc:
                ST["requests_history_reasoning_content_only"] += 1
        bh = block_hashes(p_ids)
        hit = 0
        for k, hh in enumerate(bh):
            if (k + 1) * BLOCK > len(p_ids) - 1 or hh not in CACHE:
                break
            hit = (k + 1) * BLOCK
        ST["pc_q"] += len(p_ids)
        ST["pc_h"] += hit
        ST["prompt"] += len(p_ids) - hit
        WIN["pc_q"] += len(p_ids)
        WIN["pc_h"] += hit
        WIN["prompt"] += len(p_ids) - hit
    last = _content_text(msgs[-1].get("content")) if msgs else ""
    model = body.get("model")
    rid = f"chatcmpl-{time.time_ns()}"
    base = {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model}
    if not tools:      # preflight markers, the engine smoke, any plain request: a plain content reply
        if "Reply with exactly this code and nothing else: " in last:
            text = last.split("Reply with exactly this code and nothing else: ", 1)[1].strip()
            with LOCK:
                ST["preflight"] += 1
        elif "Reply with exactly this code: " in last:            # the launch cache probe (e21_cache_probe)
            text = last.split("Reply with exactly this code: ", 1)[1].strip()
            with LOCK:
                ST["cache_probe"] += 1
        elif "7 * 6 =" in last:                    # rev 2i content probes
            text = "42"
        elif "Complete the sequence with the next number only: " in last:
            text = "10"
        elif "Repeat exactly, nothing else: " in last:
            text = last.split("Repeat exactly, nothing else: ", 1)[1].strip()
        elif "what is 2 + 2" in last:
            text = "2 + 2 = 4."
            with LOCK:
                ST["smoke"] += 1
        else:
            text = "OK"
        o_ids = tok(text)
        admit_and_run(rid, len(p_ids), min(len(o_ids), mt))
        msg = {"role": "assistant", "content": text, "reasoning": None, "tool_calls": []}
        fin, n_out = "stop", len(o_ids)
    elif msgs and "Use the python tool exactly once" in _content_text(msgs[0].get("content")):
        # the notebook's launch contract probe (e21_contract_probe): turn 1 = short reasoning + one python call with
        # JSON args {"code": ...}; turn 2 (the call + its result in history) = a plain answer. Completion tokens are the
        # rendered reply's own tokens, so turn 2's prompt = turn 1 prompt + completion + the tool message (vLLM)
        turn2 = any(m.get("role") == "assistant" for m in msgs)
        if not turn2:
            R = "The user wants print(6 * 7) run once with the python tool. I will call it now."
            calls = [{"id": f"chatcmpl-tool-probe-{time.time_ns()}", "type": "function",
                      "function": {"name": "python", "arguments": json.dumps({"code": "print(6 * 7)"})}}]
            msg = {"role": "assistant", "content": None, "reasoning": R, "tool_calls": calls}
            gen_text = R + "\n</think>\n\n" + call_markup(calls) + "<|im_end|>"
            fin = "tool_calls"
        else:
            R = "The tool printed 42."
            msg = {"role": "assistant", "content": "The result is 42.", "reasoning": R, "tool_calls": []}
            gen_text = R + "\n</think>\n\nThe result is 42.<|im_end|>"
            fin = "stop"
        with LOCK:
            ST["contract_turn2" if turn2 else "contract_turn1"] += 1
        o_ids = tok(gen_text)
        n_out = min(len(o_ids), mt)
        admit_and_run(rid, len(p_ids), n_out)
    else:
        with LOCK:
            ST["tool_requests"] += 1
            _nth = ST["tool_requests"]               # this request's number (atomic: FAKE_HOLD_ONE holds exactly one)
            if ST["first_tool_t"] is None:
                ST["first_tool_t"] = time.time()
        tid = task_of(_content_text(msgs[0].get("content"))) if msgs else None
        conv = None
        for m in msgs:
            if m.get("role") == "assistant":     # route on either key (the routing is the harness's, not vLLM's)
                f = CONV_RE.search(str(m.get("reasoning") or "") + str(m.get("reasoning_content") or ""))
                if f:
                    conv = f.group(1)
        with LOCK:
            if conv is None or conv not in CONV_META:
                role = TASK_ROLES.get(tid, 0)
                TASK_ROLES[tid] = role + 1
                conv = f"{tid}r{role}"
                CONV_META[conv] = (tid, role)
                CONV_STEP[conv] = 0
                NEW_CONVS.append([tid, role, time.time()])
                ST["conv_starts"].append([round(time.time(), 2), len(RUN), f"{tid}r{role}"])
            tid, role = CONV_META[conv]
            i = CONV_STEP[conv]
            CONV_STEP[conv] = i + 1
        steps = program(tid, role) if tid else [INSPECT_R6]
        code = steps[min(i, len(steps) - 1)]
        R = f"CONV={conv} THINK step {i} for {tid} role {role}.\n\nA second paragraph (a blank line above, as models write)."
        calls = [{"id": f"chatcmpl-tool-{conv}-{i}", "type": "function",
                  "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        msg = {"role": "assistant", "content": None, "reasoning": R + "\n", "tool_calls": calls}
        fin = "tool_calls"
        gen_text = R + "\n</think>\n\n" + call_markup(calls) + "<|im_end|>"
        # the e14 fake's per-reply token count (the attempt budgets rely on it); a complete reply stops BEFORE max_tokens
        # (completion == max_tokens means the cap was hit, which the notebook's CELL 3 patch reads as a cut)
        n_out = max(1, min(2100, mt - 1))
        if code == CUT_CODE:
            # a tool call cut at max_tokens: vLLM 0.19 / 0.32 still answer finish_reason "tool_calls" whenever the parser
            # found a call (chat_completion/serving.py), with completion_tokens == max_tokens (SGLang said "length")
            n_out = mt
            with LOCK:
                ST["cut_tool_calls"] += 1
        if code == NOCALL:
            msg = {"role": "assistant", "content": None,
                   "reasoning": R + " (cut at the output limit, no tool call).", "tool_calls": []}
            fin = "length"
            gen_text = msg["reasoning"]
        o_ids = tok(gen_text)
        if HOLD_ONE and _nth == int(HOLD_ONE[0]):
            with LOCK:
                ST["held"].append({"t0": round(time.time(), 2), "sec": HOLD_ONE[1], "conv": conv})
            dump_stats()
            admit_and_run(rid, len(p_ids), HOLD_ONE[1] * SPEED)     # held in its running slot for HOLD_ONE[1] s
            with LOCK:
                ST["held"][-1]["t1"] = round(time.time(), 2)
        else:
            admit_and_run(rid, len(p_ids), min(n_out, 200))
    with LOCK:
        for hh in block_hashes(p_ids + o_ids):  # prompt + output blocks become cacheable
            CACHE.add(hh)
        ST["gen"] += n_out
        ST["success"] += 1
        WIN["gen"] += n_out
        if SPEC:
            k = int(SPEC.get("num_speculative_tokens", 3))
            d = int(n_out / (1 + 0.693 * k))
            ST["drafts"] += d
            ST["dtok"] += d * k
            ST["acc"] += n_out - d
    ch = {"index": 0, "message": msg, "logprobs": None, "finish_reason": fin, "stop_reason": None}
    res = dict(base, choices=[ch], usage=usage(p_ids, n_out, hit))
    if tools and fin == "tool_calls" and n_out == mt and msg.get("tool_calls"):
        res["_e21_cut"] = True                   # counted as delivered only once the reply was written (do_POST)
    return 200, res


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
            return self._send(json.dumps({"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "vllm",
                                                                      "root": MODEL, "max_model_len": MAXLEN}]}))
        if p == "/metrics":
            with LOCK:
                t1 = ST["first_tool_t"]
            if METRICS_FAIL and t1 is not None:
                a_, b_ = t1 + METRICS_FAIL[0], t1 + METRICS_FAIL[0] + METRICS_FAIL[1]
                with LOCK:
                    ST["metrics_fail_window"] = [round(a_, 2), round(b_, 2)]
                if a_ <= time.time() < b_:
                    with LOCK:
                        ST["metrics_500"] += 1
                    return self._send("Internal Server Error", 500, "text/plain")
            lab = f'{{engine="0",model_name="{SERVED}"}}'
            with LOCK:
                n, nq, ctx = len(RUN), len(WAIT), sum(RUN.values())
                lines = ["# HELP vllm:num_requests_running Number of requests in model execution batches.",
                         "# TYPE vllm:num_requests_running gauge",
                         f"vllm:num_requests_running{lab} {float(n)}",
                         f"vllm:num_requests_waiting{lab} {float(nq)}",
                         f"vllm:kv_cache_usage_perc{lab} {min(1.0, ctx / KV_TOKENS):.6f}",
                         f"vllm:generation_tokens_total{lab} {float(ST['gen'])}",
                         f"vllm:generation_tokens_created{lab} 1.7e9",
                         f"vllm:prompt_tokens_total{lab} {float(ST['prompt'])}",
                         f"vllm:num_preemptions_total{lab} {float(ST['preempt'])}",
                         f"vllm:request_success_total{{engine=\"0\",finished_reason=\"stop\",model_name=\"{SERVED}\"}} "
                         f"{float(ST['success'])}",
                         f"vllm:prefix_cache_queries_total{lab} {float(ST['pc_q'])}",
                         f"vllm:prefix_cache_hits_total{lab} {float(ST['pc_h'])}"]
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
        if self.path.split("?")[0] != "/v1/chat/completions":
            return self._send(json.dumps({"detail": "Not Found"}), 404)
        code, out = chat(body)
        cut = bool(out.pop("_e21_cut", False)) if isinstance(out, dict) else False
        if code != 200 or not body.get("stream"):
            try:
                self._send(json.dumps(out), code)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return                            # the client gave up (deadline / timeout): not delivered
            if cut:
                with LOCK:
                    ST["cut_delivered"] += 1
            return
        with LOCK:
            ST["stream_requests"] += 1
        ch = out["choices"][0]
        m = ch["message"]
        chunks = [{"role": "assistant", "content": ""}]
        if m.get("reasoning"):
            chunks.append({"reasoning": m["reasoning"]})
        if m.get("content"):
            chunks.append({"content": m["content"]})
        for j, tc in enumerate(m.get("tool_calls") or []):
            chunks.append({"tool_calls": [dict(tc, index=j)]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for j, d in enumerate(chunks):
            c = {"id": out["id"], "object": "chat.completion.chunk", "model": out["model"],
                 "choices": [{"index": 0, "delta": d, "finish_reason": ch["finish_reason"] if j == len(chunks) - 1 else None}]}
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
        if (body.get("stream_options") or {}).get("include_usage"):
            self.wfile.write(b"data: " + json.dumps({"id": out["id"], "choices": [], "usage": out["usage"]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


threading.Thread(target=printer, daemon=True).start()
threading.Thread(target=die_watch, daemon=True).start()
ThreadingHTTPServer.request_queue_size = 256      # Windows: the default 5 drops simultaneous connects
if os.name == "nt":      # Windows SO_REUSEADDR lets a second server bind a port another one listens on: refuse instead
    ThreadingHTTPServer.allow_reuse_address = False
try:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
except OSError as e:
    print(f"OSError: fake vLLM cannot bind 127.0.0.1:{PORT}: {e}", flush=True)
    launch_record("port_in_use", error=repr(e)[:200])
    sys.exit(1)
srv.daemon_threads = True
launch_record("ready", kv_tokens=KV_LINE_TOKENS, num_blocks=NUM_BLOCKS, block=BLOCK, max_num_seqs=SEQS, model=MODEL, served=SERVED, port=PORT, ssm=SSM)
log(f"[api_server.py:2] Starting vLLM API server 0 on http://127.0.0.1:{PORT}")
print("INFO:     Application startup complete.", flush=True)
dump_stats()
srv.serve_forever()
