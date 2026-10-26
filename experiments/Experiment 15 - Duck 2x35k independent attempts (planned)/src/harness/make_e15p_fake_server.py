"""Writes e15p_fake_server.py = the E14 fake SGLang server (Exp 14 src/harness/e14_fake_server.py, unchanged launch log,
/metrics, decode lines, chat path) with the streaming /generate path replaced by the E11-05 Poetiq behaviours
(E11-05 harness fake_sgl_stream_server.py): phase 1 = thinking + </think> + a ```python block + EOS, or a forced close
(length) for chosen tasks; phase 2 = the code + EOS; aborts honoured; every streamed request counts as running (server
cap from --max-running-requests) with its growing context, so /metrics and the KV gate see it.
Behaviour per task: FAKE_POETIQ_BEH = JSON {task: behaviour}; other tasks by a hash of the id. FAKE_KV_USAGE overrides
full_token_usage (the KV-gate scenario). FAKE_SPEED = tokens per second per request (the thinking is streamed at it)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src/harness/e14_fake_server.py")
t = SRC.read_text(encoding="utf-8")


def rep(a, b):
    global t
    assert t.count(a) == 1, (t.count(a), a[:80])
    t = t.replace(a, b)


start = t.index('        if req.get("stream"):                                   # the E11 shim\'s streaming path (smoke)')
end = t.index("        greedy = float(sp.get(\"temperature\", 1.0)) == 0.0")
t = t[:start] + '''        if req.get("stream"):                                   # E15P: the Poetiq streaming path (+ the smoke)
            return poetiq_stream(self, rid, ids, sp, total)
''' + t[end:]
rep('''srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)''', '''# ------------------------------------------------------------------ E15P: Poetiq streaming (E11-05 behaviours)
import hashlib as _hl  # noqa: E402
_PT = []          # the tokenizer is loaded only for streamed (Poetiq / smoke) requests: EMS's servers never need it,
                  # and two servers loading it at once ran out of memory on a 16 GB machine (EMS harness run 1)


def _ptok():
    if not _PT:
        from transformers import AutoTokenizer
        _t = AutoTokenizer.from_pretrained(os.environ["E11_INT4"])
        _PT.extend([_t, _t.encode(" thinking", add_special_tokens=False), _t.encode("</think>", add_special_tokens=False)[0]])
    return _PT
PEOS = 248046
PBEH = json.loads(os.environ.get("FAKE_POETIQ_BEH", "{}"))
KV_OVERRIDE = os.environ.get("FAKE_KV_USAGE")
MAMBA_OVERRIDE = os.environ.get("FAKE_MAMBA_USAGE")
SMOKE_FAIL_MRR = os.environ.get("FAKE_SMOKE_FAIL_MRR")      # this server ignores the smoke's abort -> smoke FAIL
DIE_MRR, DIE_AFTER = os.environ.get("FAKE_DIE_MRR"), int(os.environ.get("FAKE_DIE_AFTER", "0"))
N_SOLVE = [0]


def _full_u(ft):
    return float(KV_OVERRIDE) if KV_OVERRIDE else min(0.99, ft / 881336)


def _mamba_u(n):
    return float(MAMBA_OVERRIDE) if MAMBA_OVERRIDE else min(1.0, (3 * min(n, CAP)) / K)
POETIQ_LOG = {}       # rid -> input, sent, max_new, finish, t_start, t_finish, temperature, top_k
SEQS_P = []


def p_beh(tid):
    if tid in PBEH:
        return PBEH[tid]
    h = _hl.sha256(tid.encode()).digest()[0] % 3
    return ("pass", "fail_distinct", "forced_then_pass")[h]


def p_code(b, it):
    if b in ("pass", "slow_pass"):
        return "```python\\ndef transform(grid):\\n    return grid  # MARK:PASS_ALL\\n```"
    if b == "partial_then_fail":
        return ("```python\\ndef transform(grid):\\n    return grid  # MARK:PASS_PARTIAL\\n```" if it == 0 else
                f"```python\\ndef transform(grid):\\n    return grid  # MARK:FAIL_ITER {it}\\n```")
    if b == "forced_then_pass":
        return (f"```python\\ndef transform(grid):\\n    return grid  # MARK:FAIL_ITER {it}\\n```" if it == 0 else
                "```python\\ndef transform(grid):\\n    return grid  # MARK:PASS_ALL\\n```")
    if b == "fail_distinct":
        return f"```python\\ndef transform(grid):\\n    return grid  # MARK:FAIL_ITER {it}\\n```"
    return "```python\\ndef transform(grid):\\n    return grid  # MARK:ZERO\\n```"


def p_plan(rid, max_new):
    """(tokens, finish 'stop' | 'length', tokens per second)"""
    PTOK, PFILL, PCLOSE = _ptok()
    if rid == "e11-smoke-a":
        return (PFILL * 20)[:min(16, max_new)], "length", 0
    if rid == "e11-smoke-b":
        return (PFILL * 5000)[:max_new], "length", 400
    tid, rest = rid.split("~e", 1)
    it, phase = int(rest.split("~i")[1].split("|")[0]), int(rid[-1])
    b = p_beh(tid)
    code = PTOK.encode("\\n\\n" + p_code(b, it), add_special_tokens=False)
    n_think = 200 + _hl.sha256(rid.encode()).digest()[1] * 4          # 200..1220 thinking tokens
    if phase == 1 and ((b == "forced_then_pass" and it == 0) or b == "slow_pass"):
        # never closes: the whole phase-1 cap streamed (fast for the forced close, slow for the drain / hard stop)
        return (PFILL * (max_new // len(PFILL) + 1))[:max_new], "length", (SPEED if b == "slow_pass" else 200 * SPEED)
    s = (PFILL * n_think + [PCLOSE] + code + [PEOS]) if phase == 1 else (code + [PEOS])
    s = s[:max_new]
    return s, ("stop" if s and s[-1] == PEOS else "length"), SPEED


def poetiq_stream(h, rid, ids, sp, total):
    assert sp.get("no_stop_trim") is True and sp.get("skip_special_tokens") is False, sp
    if DIE_AFTER and DIE_MRR == str(CAP) and not rid.startswith("e11-smoke"):
        with LOCK:
            N_SOLVE[0] += 1
            die = N_SOLVE[0] > DIE_AFTER
        if die:
            out(" TP0] Scheduler hit an exception: simulated crash (FAKE_DIE_AFTER)")
            os._exit(3)
    stream, fin, speed = p_plan(rid, int(sp.get("max_new_tokens", total)))
    with LOCK:
        cached = max((min(len(ids), len(s_)) for s_ in SEQS_P if s_[:64] == ids[:64]), default=0)
        POETIQ_LOG[rid] = {"input": ids, "sent": [], "max_new": sp.get("max_new_tokens"), "finish": None,
                           "t_start": time.time(), "temperature": sp.get("temperature"), "top_k": sp.get("top_k")}
        WAIT.append(rid)
    while True:                      # the server cap (--max-running-requests) like decode_wait
        with LOCK:
            if WAIT and WAIT[0] == rid and CAP - len(RUN) > 0:
                WAIT.pop(0)
                RUN[rid] = {"ctx": len(ids), "t0": time.time()}
                INFLIGHT[0] = len(RUN)
                INFLIGHT[1] = max(INFLIGHT[1], INFLIGHT[0])
                break
            if rid in ABORT:
                WAIT.remove(rid)
                break
        time.sleep(0.01)
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Transfer-Encoding", "chunked")
    h.send_header("Connection", "close")
    h.end_headers()
    h.close_connection = True

    def emit(b):
        h.wfile.write(f"{len(b):x}\\r\\n".encode() + b + b"\\r\\n")
        h.wfile.flush()
    pos, finish, step = 0, None, 64
    try:
        while finish is None:
            with LOCK:
                ab = rid in ABORT and not (rid == "e11-smoke-b" and SMOKE_FAIL_MRR == str(CAP))
            if ab:
                finish, chunk = {"type": "abort", "message": "Abort requested"}, []
            else:
                chunk = stream[pos:pos + step]
                pos += len(chunk)
                if pos >= len(stream):
                    finish = {"type": "stop", "matched": PEOS} if fin == "stop" else {"type": "length", "length": len(stream)}
            with LOCK:
                POETIQ_LOG[rid]["sent"] += chunk
                if finish is not None:
                    POETIQ_LOG[rid]["t_finish"] = time.time()
                if rid in RUN:
                    RUN[rid]["ctx"] = len(ids) + len(POETIQ_LOG[rid]["sent"])
            meta = {"id": rid, "finish_reason": finish, "prompt_tokens": len(ids),
                    "completion_tokens": len(POETIQ_LOG[rid]["sent"]), "cached_tokens": cached,
                    "spec_accept_length": (1 + 0.47 * (WIDTH - 1)) if SPEC else None}
            emit(b"data: " + json.dumps({"output_ids": chunk, "meta_info": meta}).encode() + b"\\n\\n")
            if speed and chunk:
                time.sleep(len(chunk) / speed)
        emit(b"data: [DONE]\\n\\n")
        h.wfile.write(b"0\\r\\n\\r\\n")
        h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        finish = {"type": "abort", "message": "client disconnected"}
    with LOCK:
        POETIQ_LOG[rid]["finish"] = finish["type"]
        RUN.pop(rid, None)
        INFLIGHT[0] = len(RUN)
        SEQS_P.append(ids + POETIQ_LOG[rid]["sent"])


srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)''')
# /metrics: the KV / mamba overrides; token_usage = max(full, mamba) as SGLang 0.5.21 (pool_stats_observer.py:69-73)
rep('''f'sglang:full_token_usage{{model_name="int4"}} {min(0.99, ft / 881336):.4f}',''',
    '''f'sglang:full_token_usage{{model_name="int4"}} {_full_u(ft):.4f}',''')
rep('''f'sglang:mamba_usage{{model_name="int4"}} {min(1.0, (3 * min(n, CAP)) / K):.4f}',''',
    '''f'sglang:mamba_usage{{model_name="int4"}} {_mamba_u(n):.4f}',''')
rep(''''sglang:token_usage{model_name="int4"} 0.1',''',
    '''f'sglang:token_usage{{model_name="int4"}} {max(_full_u(ft), _mamba_u(n)):.4f}',''')
rep('''                return self._send(json.dumps(dict(LOG, new_convs=NEW_CONVS, max_inflight=INFLIGHT[1],''',
    '''                return self._send(json.dumps(dict(LOG, new_convs=NEW_CONVS, max_inflight=INFLIGHT[1], poetiq=POETIQ_LOG,''')
t = t.replace('"""E14 fake SGLang 0.5.21 server', '"""E15P fake (generated by make_e15p_fake_server.py; do not edit). E14 fake SGLang 0.5.21 server', 1)
_d = "    if SPEC and CAP >= 8:"
assert t.count(_d) == 1
t = t.replace(_d, '    if SPEC and str(arg("--speculative-algorithm", "")).upper() == "DFLASH" and CAP >= 8:   # 0.5.21: DFlash only')
(HERE / "e15p_fake_server.py").write_text(t, encoding="utf-8", newline="\n")
print("written", len(t))
