"""Experiment 12 notebook builder: the Duck REPL agent (vendored verbatim) on the E11-06 SGLang W1 engine.

Source = the E11-06 P1-60K EVAL14 notebook (sha below). Kept verbatim: CELL 0 (mode + fallback), CELL 1 (config; two
profiles + an E12 block appended), CELL 2 (install + tokenizer), the task-load cell, CELL 5 (SGLang engine; only the Duck
serving flags are added to e11_flags). Replaced: the Poetiq donor/solver/report cells and the vLLM fallback cell (Duck
needs an OpenAI server; without SGLang the CELL 0 fallback submission stays).
    python build_e12_duck.py            # SMOKE1 (step 3.5: 1 task, 2 branches, 15 min)
    python build_e12_duck.py --eval30   # EVAL30 (step 4: the E11-05 EVAL30 tasks, 4 h)
"""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E12 = HERE.parents[1]
DUCK = E12 / "duck_arc2"
SRC_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Notebooks/"
              r"Q27-XHIGH-v1_4_4_E11_06_P1_60K_SGLANG_INT4_DFLASH2_EVAL14.ipynb")
SRC_SHA = "f0575ff38e85cbfbb6d1da133b5cdf53307a49b42ca9120e13459ceb23e6f28e"   # = the pushed E11-06 EVAL14 commit
VARIANT = "EVAL30" if "--eval30" in sys.argv else "SMOKE1"
VLLM = "--vllm" in sys.argv   # E12-V (user 2026-10-19): the same Duck notebook, engine = our best vLLM (S1-60K) as a server
NAME = f"Q27-XHIGH-E12_DUCK_REPL_{'VLLM' if VLLM else 'SGLANG'}_{VARIANT}.ipynb"
DUCK_COMMIT = "10882e332d883af3e61616df8c8a17d03a328a75"   # da-fr/arc-agi-3-solution (fork of Tufalabs/duck-harness @ 7652836)


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"patch '{label}': expected 1 match, found {n}")
    return text.replace(old, new)


raw = SRC_NB.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "source notebook changed"
src = json.loads(raw.decode("utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
assert cells[1].startswith("# ==================================================================== CELL 0"), cells[1][:80]
assert cells[2].startswith("# ==================================================================== CELL 1"), cells[2][:80]
assert cells[3].startswith("# ==================================================================== CELL 2"), cells[3][:80]
assert cells[4].startswith("# E10: original task-order"), cells[4][:80]
assert cells[6].startswith("# ==================================================================== CELL 5 — engine"), cells[6][:80]

# ------------------------------------------------------------------ the embedded package (verbatim Duck + adapter)
files = {}
for p in sorted((DUCK / "vendor").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["vendor/" + p.relative_to(DUCK / "vendor").as_posix()] = p.read_bytes()
files["arc2_duck.py"] = (DUCK / "arc2_duck.py").read_bytes()
assert len([k for k in files if k.startswith("vendor/inference/")]) == 20, sorted(files)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 19, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
ZIP_B64 = base64.b64encode(buf.getvalue()).decode("ascii")
MANIFEST = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}

# ------------------------------------------------------------------ CELL 1: profiles + E12 block
c1 = cells[2]
c1 = once(c1, 'os.environ.get("Q27X_PROFILE", "EVAL14")', f'os.environ.get("Q27X_PROFILE", "{VARIANT}")', "default profile")
c1 = once(c1, '    "PROXY56":', '''    # E12 step 3.5: GPU smoke, 1 fast task (2ba387bc: 5,355 thinking tokens in E10 ORDER2), 2 branches, 15 min solving
    "SMOKE1":    {"source": "eval", "task_list": ["2ba387bc"], "time_budget_sec": 50 * 60, "admit_frac": 0.90,
                  "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60,
                  "admit_margin_sec": 2 * 60, "solve_window_sec": 15 * 60},
    # E12 step 4: the E11-05 EVAL30 tasks (same seed sample), 4 h
    "EVAL30":    {"source": "eval", "n_tasks": 30, "time_budget_sec": 4.0 * 3600,
                  "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                  "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},
    "PROXY56":''', "profiles")
c1 += '''

# ---- E12: Duck REPL agent (Experiment 12, user option A 2026-10-19). Duck's own settings are in arc2_duck.DUCK_ENV
# (= dfranzen milestone-2 cell 4): 12,288 output per request, yield at 2,048 generated tokens, preserved thinking,
# 128 Ki context with drain to 58 Ki. Approved exception to the 60k-per-request floor (thinking accumulates in context).
E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))                 # independent branches per task -> attempt_1/2
E12_STREAMS = int(os.environ.get("E12_STREAMS", 0))                   # Duck ARC3_MAX_ACTIVE_STREAMS; 0 = engine concurrency
E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", 0)) or None   # Duck max_generated_tokens_per_game (milestone: none)
E12_SOLVE_WINDOW_SEC = P.get("solve_window_sec")                      # SMOKE1 only
E12_STOP_BEFORE_HARD_SEC = 180   # branches stop; outputs are written after it. rev 3 (GPT re-audit): a submit at the
                                 # deadline can still verify for <= 60 s, then CELL 6/7 write; 60 s left no margin
E12_MAMBA_TRACK_64 = os.environ.get("E12_MAMBA_TRACK_64", "0") == "1"  # Duck's --mamba-track-interval/--page-size 64 (measure)
# rev 2 (GPT audits, validated 2026-10-19): Duck request snapshots (the exact messages sent; large) for the smoke only,
# and a per-sandbox-process address-space limit (Duck sets CPU / file size / open files only)
E12_REQUEST_LOGS = os.environ.get("E12_REQUEST_LOGS", "1" if PROFILE == "SMOKE1" else "0") == "1"
E12_SANDBOX_MEM_MB = int(os.environ.get("E12_SANDBOX_MEM_MB", 4096))
CONFIG["e12"] = {k: globals()[k] for k in ("E12_BRANCHES", "E12_STREAMS", "E12_MAX_GEN_PER_BRANCH", "E12_SOLVE_WINDOW_SEC",
                                           "E12_STOP_BEFORE_HARD_SEC", "E12_MAMBA_TRACK_64", "E12_REQUEST_LOGS",
                                           "E12_SANDBOX_MEM_MB")}
CONFIG["e12"]["duck_commit"] = "''' + DUCK_COMMIT + '''"
print("E12:", json.dumps(CONFIG["e12"]))
'''

# ------------------------------------------------------------------ CELL 3: package + sandbox self-test
c_pkg = '''# ==================================================================== CELL 3 — E12: Duck package (verbatim) + ARC-2 adapter
# Tufa Labs' Duck harness as forked by Daniel Franzen (da-fr/arc-agi-3-solution @ ''' + DUCK_COMMIT[:10] + ''', Apache-2.0,
# LICENSE + NOTICE inside). The 20 files of inference/ are byte-identical to that commit (sha256 manifest below);
# arc2_duck.py is the only new code (ARC-2 prompt, prelude with check/submit, host re-verification, driver, selection).
import base64 as _b64, io as _io, logging as _logging
E12_DIR = Path(os.environ.get("E12_DIR", "/tmp/e12_duck"))
_E12_ZIP = "''' + ZIP_B64 + '''"
_E12_MANIFEST = ''' + json.dumps(MANIFEST, indent=0) + '''
if E12_DIR.exists():
    shutil.rmtree(E12_DIR)
with zipfile.ZipFile(_io.BytesIO(_b64.b64decode(_E12_ZIP))) as _z:
    _z.extractall(E12_DIR)
_bad = [p for p, h in _E12_MANIFEST.items() if hashlib.sha256((E12_DIR / p).read_bytes()).hexdigest() != h]
assert not _bad, ("E12 package hash mismatch", _bad)
sys.path[:0] = [str(E12_DIR), str(E12_DIR / "vendor")]
import arc2_duck as E12A
# Duck reads its settings at import: set them first. The base URL is the engine cell's (same port); the stream count is
# set again from the engine's concurrency before the gate is first used (read at call time).
os.environ.update(E12A.duck_env(f"http://127.0.0.1:{int(os.environ.get('E11_PORT', 30000))}/v1", "pending", SGL_MAX_RUNNING))
_E12_TA, _E12_DP, _e12_run_sandboxed, _ = E12A._import_duck()
E12_SANDBOX_LIMIT = E12A.install_sandbox_memory_limit(E12_SANDBOX_MEM_MB)   # rev 2: every sandbox child, before its code


class _E12LevelCountFilter(_logging.Filter):   # Duck's per-game level-count warning has no ARC-2 meaning (priority uses 10)
    def filter(self, record):
        return "total level count unavailable" not in record.getMessage()


_logging.getLogger(_E12_TA.__name__).addFilter(_E12LevelCountFilter())

# sandbox self-test with our own scripted code (the engine cell runs only if it passes)
_t = time.perf_counter()
try:
    _st_task = test_challenges[sorted(test_challenges)[0]]
    _st = _e12_run_sandboxed(
        code=E12A.task_prelude(_st_task) + "def _e12_ident(g):\\n    return g\\nres = check(_e12_ident, verbose=False)\\nprint(len(res), sum(1 for r in res if 'ok' in r))",
        timeout_seconds=30, initial_state={"current_frame": None, "history": [], "valid_actions": [], "last_action_call_result": {}},
        action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[], retain_imports=True)
    _n = len(_st_task["train"])
    E10_SANDBOX_SELFTEST = {"ok": (_st.get("stdout") or "").strip() == f"{_n} {_n}" and not _st.get("error"),
                            "stdout": (_st.get("stdout") or "")[:200], "error": (_st.get("error") or "")[:500],
                            "sec": round(time.perf_counter() - _t, 2), "python": sys.version.split()[0]}
except Exception as _e:
    E10_SANDBOX_SELFTEST = {"ok": False, "error": f"{type(_e).__name__}: {_e}"[:500]}
# rev 2: memory limit evidence (reported; the run continues without it rather than not at all). The 5 GiB request is
# made only after the read-back confirmed the limit, so it can never really allocate.
_mem = {"limit": dict(E12_SANDBOX_LIMIT)}
try:
    _st_kw = dict(timeout_seconds=30, initial_state={"current_frame": None, "history": [], "valid_actions": [],
                                                     "last_action_call_result": {}},
                  action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[],
                  retain_imports=True)
    if E12_SANDBOX_LIMIT.get("readback_ok"):
        _big = _e12_run_sandboxed(code="x = bytearray(5 * 1024 ** 3)\\nprint(len(x))", **_st_kw)
        _small = _e12_run_sandboxed(code="x = bytearray(128 * 1024 ** 2)\\nprint(len(x))", **_st_kw)
        _mem.update(over_limit_rejected="MemoryError" in (_big.get("error") or ""),
                    small_ok=(_small.get("stdout") or "").strip() == str(128 * 1024 ** 2))
        _mem["ok"] = bool(_mem["over_limit_rejected"] and _mem["small_ok"])
    else:
        _mem["ok"] = False
except Exception as _e:
    _mem.update(ok=False, error=f"{type(_e).__name__}: {_e}"[:300])
_mem["limit"] = dict(E12_SANDBOX_LIMIT)
E10_SANDBOX_SELFTEST["memory_limit"] = _mem
if E12_SANDBOX_MEM_MB > 0 and not _mem.get("ok"):
    print("E12 WARNING: sandbox memory limit NOT verified:", json.dumps(_mem), flush=True)
    if os.name == "posix":   # rev 3 (GPT re-audit): on Kaggle (Linux) never run model code without the limit
        E10_SANDBOX_SELFTEST["ok"] = False
        E10_SANDBOX_SELFTEST["error"] = "memory limit not verified (fail closed): " + json.dumps(_mem)[:300]
        print("E12: solver disabled (fail closed); CELL 0 fallback submission stays", flush=True)
CONFIG["e12"]["sandbox_selftest"] = E10_SANDBOX_SELFTEST
print("E12 package OK:", len(_E12_MANIFEST), "files | sandbox self-test:", json.dumps(E10_SANDBOX_SELFTEST), flush=True)
'''

# ------------------------------------------------------------------ CELL 5: engine, + Duck serving flags only
c_eng = cells[6]
c_eng = once(c_eng, '''          "--incremental-streaming-output", "--stream-interval", "1"]
    return f''', '''          "--incremental-streaming-output", "--stream-interval", "1"]
    # E12: Duck milestone-2 serving flags (dfranzen notebook cell 12; each verified in SGLang 0.5.21): OpenAI tool calls
    # and reasoning split for /v1/chat/completions, preserved thinking by default, prefix-aware scheduling, cached_tokens
    # in usage. /generate (the smoke below) is unaffected.
    f += ["--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
          "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
          "--schedule-policy", "lpm", "--enable-cache-report"]
    if E12_MAMBA_TRACK_64:
        f += ["--mamba-track-interval", "64", "--page-size", "64"]
    return f''', "duck serving flags")
c_eng = "# E12: = E11-06 CELL 5 unchanged except the Duck serving flags in e11_flags().\n" + c_eng

if VLLM:
    # ---- E12-V: CELL 1 default engine = vLLM, so CELL 2 (verbatim) installs vllm==0.19.0 exactly as for S1-60K
    c1 = once(c1, 'ENGINE_BACKEND = os.environ.get("E11_ENGINE", "sglang")', 'ENGINE_BACKEND = os.environ.get("E11_ENGINE", "vllm")',
              "engine default vllm")
    # ---- CELL 3: Duck's own default history key. vLLM 0.19 reads prior reasoning only from `reasoning`
    # (entrypoints/chat_utils.py:1512); the milestone's `reasoning_content` is the SGLang setting and vLLM would drop it.
    c_pkg += ('\n# E12-V: vLLM reads prior reasoning from `reasoning` only (vllm/entrypoints/chat_utils.py:1512, v0.19.0); this is\n'
              '# Duck\'s own default (tool_agent._reasoning_history_keys). The milestone value reasoning_content is for SGLang.\n'
              'os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"\n'
              'print("E12-V: ARC3_REASONING_HISTORY_KEY =", os.environ["ARC3_REASONING_HISTORY_KEY"], flush=True)\n')
    # ---- CELL 5: S49's libcuda shim function, verbatim (dedented from the E11 fallback cell)
    import textwrap
    _fb = cells[7]
    _a = _fb.index("    import shutil as _shutil\n")
    _b = _fb.index('    os.environ.setdefault("MAX_JOBS", "4")')
    _shim_src = textwrap.dedent(_fb[_a:_b])
    assert "def ensure_libcuda_linkable" in _shim_src and "q27x_probe" in _shim_src
    c_eng = r'''# ==================================================================== CELL 5 — engine: vLLM 0.19 OpenAI server (E12-V)
# Duck's own Kaggle vLLM launcher (inference/framework/kaggle.py: start_vllm_server / wait_for_vllm_server /
# run_vllm_api_smoke_test), verbatim flags, plus the S1-60K engine settings (LB 34.58: TP4, fp8_e4m3 KV, MTP-3,
# FlashInfer, prefix caching, 4096 batched tokens, 16 seqs, 0.90 GPU memory, libcuda shim). vllm==0.19.0 is installed by
# CELL 2 (the S1-60K path). The names CELL 6/7 use (SGL_URL, ENGINE_READY, CONCURRENCY, metrics_snapshot, E11_SERVER,
# e11_kill_server, e11_decode_summary, engine) are kept. No launch ladder: if the server does not come up, CELL 0's
# fallback submission stays.
import urllib.request as _urlreq, signal as _signal, statistics as _statistics
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only: fake server, no GPU
VLLM_PORT = int(os.environ.get("E12_VLLM_PORT", 1234))       # Duck's DEFAULT_VLLM_PORT
SGL_URL = f"http://127.0.0.1:{VLLM_PORT}"                      # name kept for CELL 6/7
VLLM_SERVED_NAME = "Qwen/Qwen3.8-27B-FP8"   # E12-V rev 3: keithtyser's Duck served name. NOT "qwen": Duck treats
# "local"/"local-qwen"/"qwen-local"/"qwen" as a preset alias and then sends LOCAL_ANALYZER_MODEL_ID (CELL 3: "pending")
# instead (tool_agent._resolve_analyzer_model); vLLM answered every Duck request with 404 (vLLM SMOKE1 run 1, 2026-10-19).
VLLM_MAX_MODEL_LEN = 135168          # Duck's 128 Ki context window + margin (S1-60K: 106,496)
VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60    # Duck waits 900 s; our 4xL4 load + JIT + CUDA graphs took ~8-9 min (Q27-AGENT run 4)
E12_VLLM_CONCURRENCY = 12            # = the SGLang W1 run (Duck active streams); the server admits up to MAX_NUM_SEQS
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0; E11_SERVER = None
ENGINE_INFO = {"backend": None, "vllm": {"stage": "not started"}, "kv_tokens": None, "concurrency": None, "fatal": None,
               "load_sec": None}

''' + _shim_src + r'''

class _E12VllmServer:
    def __init__(self, cmd, env):
        self.cmd, self.env, self.proc, self.log_f = cmd, env, None, None
        self.log_path = OUT / "vllm_openai_server.log"

    def start(self):
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     cwd=None if E11_HARNESS else "/tmp", start_new_session=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def kill(self, wait=True):
        if self.alive():
            for sig, w in ((_signal.SIGTERM, 30 if wait else 0), (getattr(_signal, "SIGKILL", _signal.SIGTERM), 30)):
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(self.proc.pid, sig)          # the whole group: API server, engine core, TP workers
                    else:
                        self.proc.kill()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=w if w else 0.1)
                    break
                except Exception:
                    continue
        if self.log_f:
            try:
                self.log_f.close()
            except Exception:
                pass
            self.log_f = None


def e11_kill_server(wait=True):
    if E11_SERVER is not None:
        E11_SERVER.kill(wait=wait)


def _e12v_request_json(url, payload=None, timeout=30):   # = Duck's request_json
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


_E12V_PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
# vLLM metric names -> the names CELL 6's heartbeat prints (SGLang wording)
_E12V_ALIAS = {"vllm:num_requests_running": "num_running_reqs", "vllm:num_requests_waiting": "num_queue_reqs",
               "vllm:kv_cache_usage_perc": "token_usage"}


def metrics_snapshot():
    if not ENGINE_READY:
        return []
    try:
        txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
        rows = []
        for line in txt.splitlines():
            m = _E12V_PROM_RE.match(line) if line and not line.startswith("#") else None
            if not m or m.group(1).endswith(("_bucket", "_created")):
                continue
            name = m.group(1)
            if name in _E12V_ALIAS or any(k in name for k in ("prefix_cache", "spec_decode", "generation_tokens_total",
                                                               "prompt_tokens_total", "num_preemptions")):
                try:
                    rows.append({"name": _E12V_ALIAS.get(name, name), "labels": m.group(2), "value": float(m.group(3))})
                except ValueError:
                    pass
        return rows
    except Exception as e:
        return [{"error": repr(e)[:300]}]


_E12V_STAT_RE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                           r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")


def e11_decode_summary():
    """vLLM's periodic engine stats lines: generation throughput by running level, KV usage, prefix-cache hit rate."""
    if E11_SERVER is None:
        return None
    rows = [tuple(float(x) for x in m.groups()) for m in _E12V_STAT_RE.finditer(E11_SERVER.log_text())]
    med = lambda v: round(_statistics.median(v), 1) if v else None
    acc = re.findall(r"Mean acceptance length: ([\d.]+)", E11_SERVER.log_text())
    out = {"n_lines": len(rows),
           "running_hist": {str(int(k)): sum(1 for r in rows if r[2] == k) for k in sorted({r[2] for r in rows})},
           "gen_tok_s_median_at_concurrency": med([r[1] for r in rows if r[2] >= E12_VLLM_CONCURRENCY]),
           "gen_tok_s_median_all_busy": med([r[1] for r in rows if r[2] > 0]),
           "waiting_max": max((r[3] for r in rows), default=None), "kv_usage_pct_max": max((r[4] for r in rows), default=None),
           "prefix_cache_hit_pct_last": rows[-1][5] if rows else None,
           "spec_mean_acceptance_median": med([float(a) for a in acc]),
           "preemption_lines": sum(1 for l in E11_SERVER.log_text().splitlines() if "preempt" in l.lower())}
    try:
        write_json_atomic(OUT / "sglang_decode_summary.json", out)   # bundled under this name by CELL 7
    except Exception:
        pass
    return out


def e12v_bring_up():
    global E11_SERVER
    info = ENGINE_INFO["vllm"]
    info["stage"] = "inputs"
    if not (MODEL_PATH / "config.json").is_file():
        raise RuntimeError(f"model missing: {MODEL_PATH}")
    if not E11_HARNESS:
        assert pkg_version("vllm") == "0.19.0", pkg_version("vllm")
        os.environ.setdefault("MAX_JOBS", "4")              # bound FlashInfer JIT build parallelism (S1-60K)
        info["cuda_link"] = ensure_libcuda_linkable()       # S1-60K: makes -lcuda linkable for FlashInfer's JIT
    else:
        info["cuda_link"] = {"probe": "PASS", "harness": True}
    backend = "FLASHINFER" if info["cuda_link"].get("probe") == "PASS" else "TRITON_ATTN"   # = the S1-60K ladder's first
    env = os.environ.copy()
    env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "PYTHONUNBUFFERED": "1"})   # Duck's vllm_env + S1-60K
    launch = ([sys.executable, "-m", "vllm.entrypoints.openai.api_server"] if not E11_HARNESS
              else [sys.executable, os.environ["E12_FAKE_VLLM_SERVER"]])
    flags = [
        # Duck's start_vllm_server, verbatim
        "--model", str(MODEL_PATH), "--served-model-name", VLLM_SERVED_NAME, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "4", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--generation-config", "vllm", "--enable-prefix-caching", "--default-chat-template-kwargs",
        json.dumps({"preserve_thinking": True}), "--reasoning-parser", "qwen3", "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        # the S1-60K engine (CELL 1 values); keithtyser's Duck launch adds --kv-cache-dtype fp8 the same way
        "--kv-cache-dtype", KV_CACHE_DTYPE, "--speculative-config", json.dumps(SPECULATIVE_CONFIG),
        "--attention-backend", backend, "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs", str(MAX_NUM_SEQS), "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--language-model-only", "--disable-custom-all-reduce", "--seed", str(SEED),
        # cached_tokens in usage (prefix-cache measurement; SGLang: --enable-cache-report)
        "--enable-prompt-tokens-details"]
    info.update(stage="launch", flags=flags, attention_backend=backend)
    srv = _E12VllmServer(launch + flags, env)
    srv.start()
    E11_SERVER = srv
    print("E12-V: starting vLLM OpenAI server:", " ".join(flags), flush=True)
    deadline = time.perf_counter() + min(VLLM_LAUNCH_TIMEOUT_SEC, max(60, ADMIT_DEADLINE - time.perf_counter()))
    while True:                                          # = Duck's wait_for_vllm_server (/v1/models)
        if not srv.alive():
            raise RuntimeError(f"vLLM server exited rc={srv.proc.returncode}: " + srv.log_text()[-3000:])
        try:
            models = _e12v_request_json(SGL_URL + "/v1/models", timeout=5)
            info["models"] = [m.get("id") for m in models.get("data", [])]
            break
        except Exception:
            if time.perf_counter() > deadline:
                raise RuntimeError("vLLM server not ready before the deadline: " + srv.log_text()[-3000:])
            time.sleep(2 if E11_HARNESS else 5)
    info["ready_sec"] = round(time.perf_counter() - srv.t0, 1)
    log = srv.log_text()
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    mc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log)
    info["kv_tokens_logged"] = int(kv[-1].replace(",", "")) if kv else None
    info["max_concurrency_logged"] = {"tokens": int(mc[-1][0].replace(",", "")), "x": float(mc[-1][1])} if mc else None
    info["stage"] = "smoke"
    # = Duck's run_vllm_api_smoke_test + valid usage counters. E12-V rev 2 (GPT audits, validated): vLLM 0.19 sets
    # usage.prompt_tokens_details only when num_cached_tokens is truthy (chat_completion/serving.py:1601-1603; issue
    # https://github.com/vllm-project/vllm/issues/44377), so this first, cold request has none: record it, never fail on it.
    r = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
        "model": VLLM_SERVED_NAME, "messages": [{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}],
        "temperature": 0.0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    gen = (r["choices"][0]["message"].get("content") or "").strip()
    usage = r.get("usage") or {}
    _det = usage.get("prompt_tokens_details")
    info["smoke"] = {"generated": gen[:200], "usage": usage, "cache_details_present": isinstance(_det, dict),
                     "ok": bool(gen) and isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
                     and isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] > 0}
    print("E12-V smoke:", json.dumps(info["smoke"]), flush=True)
    if not info["smoke"]["ok"]:
        raise RuntimeError("vLLM smoke failed: " + json.dumps(info["smoke"])[:600])
    return info


_t_engine = time.perf_counter()
if not E10_SANDBOX_SELFTEST.get("ok", False):
    ENGINE_INFO["vllm"]["stage"] = "skipped: sandbox self-test failed"
elif ENGINE_BACKEND != "vllm":
    ENGINE_INFO["vllm"]["stage"] = f"skipped: ENGINE_BACKEND={ENGINE_BACKEND}"
else:
    try:
        e12v_bring_up()
        ENGINE_READY = True
        CONCURRENCY = E12_VLLM_CONCURRENCY
        ENGINE_INFO.update(backend="vllm-0.19.0-fp8-mtp3-openai-server", kv_tokens=ENGINE_INFO["vllm"].get("kv_tokens_logged"),
                           max_model_len=VLLM_MAX_MODEL_LEN)
        ENGINE_INFO["vllm"]["stage"] = "ready"
    except Exception as _e:
        ENGINE_INFO["vllm"]["error"] = f"{type(_e).__name__}: {_e}"[:3000]
        ENGINE_INFO["fatal"] = ENGINE_INFO["vllm"]["error"][:300]
        print("E12-V vLLM engine NOT ready:", ENGINE_INFO["vllm"]["stage"], "|", ENGINE_INFO["vllm"]["error"][:1500], flush=True)
        e11_kill_server(); E11_SERVER = None
        ENGINE_READY = False; CONCURRENCY = 0
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
ENGINE_INFO["concurrency"] = int(CONCURRENCY)
print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)
'''

# ------------------------------------------------------------------ CELL 6: the Duck solver
c_solve = r'''# ==================================================================== CELL 6 — E12: Duck REPL agent, every task, E12_BRANCHES branches
# Every branch is admitted at once; Duck's priority gate keeps ARC3_MAX_ACTIVE_STREAMS conversations active (first
# branches in ORDER2 order before any second branch). A branch ends when the host verifies a submitted program on every
# train pair, at the deadline, or at the optional token cap. Full thinking: Duck's transcript.txt per branch.
_E12_INDEX = {t: i for i, t in enumerate(TASK_IDS)}


def _e12_order2_score(tid):   # = S49 ORDER2 ([>=3 train] + [1 test]) with the cells tie-break, sorted-id order last
    _t = CHALLENGES[tid]
    return int(len(_t["train"]) >= 3) + int(len(_t["test"]) == 1)


def _e12_total_cells(tid):
    _t = CHALLENGES[tid]
    _c = lambda g: len(g) * len(g[0]) if g and g[0] else 0
    return sum(_c(p["input"]) + _c(p["output"]) for p in _t["train"]) + sum(_c(p["input"]) for p in _t["test"])


E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))
E12_RECORDS = {t: [] for t in TASK_IDS}
E12_RUN = {"started_sec": None, "model_id": None, "streams": None, "deadline_in_sec": None, "error": None,
           "branches_done": 0, "branches_accepted": 0, "finished_sec": None}
_E12_LOCK = threading.Lock()
E12_OUT = OUT / "duck"
E12_OUT.mkdir(parents=True, exist_ok=True)


import tempfile as _tempfile
_E12_WRITE_LOCK = threading.Lock()   # rev 2 (GPT audits, reproduced): branches finish concurrently


def _e12_write_json_atomic(path, obj):
    """A unique temp file per write + os.replace (CELL 0's helper reuses one '<name>.tmp' for every writer)."""
    fd, tmp = _tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".tmp", dir=str(Path(path).parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def e12_write_submission():
    with _E12_WRITE_LOCK:                          # one writer at a time; the snapshot is taken inside, so it is fresh
        sub = identity_submission(test_challenges)
        if RERUN:
            with _E12_LOCK:
                recs = {t: list(v) for t, v in E12_RECORDS.items() if v}
            for t, rows in E12A.build_submission({t: CHALLENGES[t] for t in recs}, recs).items():
                if t in sub and len(sub[t]) == len(rows):
                    sub[t] = rows
        _e12_write_json_atomic(SUBMISSION_PATH, sub)   # evaluation: keeps the competition-format test fallback


def _e12_branch_done(rec):
    with _E12_LOCK:
        E12_RECORDS.setdefault(rec["task_id"], []).append(rec)
        E12_RUN["branches_done"] += 1
        E12_RUN["branches_accepted"] += int(bool(rec.get("accepted")))
    try:
        e12_write_submission()
    except Exception as _e:
        print("E12 submission write failed:", repr(_e)[:200], flush=True)


_E12_STOP = threading.Event()


def _e12_heartbeat(every=300):
    while not _E12_STOP.wait(every):
        snap = {r.get("name"): r.get("value") for r in metrics_snapshot() if isinstance(r, dict) and "name" in r}
        keep = {k: v for k, v in snap.items() if any(s in k for s in ("num_running_reqs", "num_queue_reqs", "gen_throughput",
                                                                        "token_usage", "cache_hit_rate"))}
        print(f"E12 [{(time.perf_counter() - NOTEBOOK_T0) / 60:.0f} min] branches done {E12_RUN['branches_done']}"
              f"/{len(TASK_IDS) * E12_BRANCHES}, accepted {E12_RUN['branches_accepted']} | {json.dumps(keep)[:400]}", flush=True)


if ENGINE_READY and E10_SANDBOX_SELFTEST.get("ok"):
    try:
        try:
            E12_RUN["model_id"] = json.loads(_urlreq.urlopen(SGL_URL + "/v1/models", timeout=30).read())["data"][0]["id"]
        except Exception as _e:
            E12_RUN["model_id"] = str(SGL_MODEL_PATH)
            print("E12: /v1/models failed, using the model path:", repr(_e)[:200], flush=True)
        E12_RUN["streams"] = min(int(CONCURRENCY), E12_STREAMS or int(CONCURRENCY))
        os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(E12_RUN["streams"])
        _end = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
        if E12_SOLVE_WINDOW_SEC:
            _end = min(_end, time.perf_counter() + E12_SOLVE_WINDOW_SEC)
        E12_RUN["deadline_in_sec"] = round(_end - time.perf_counter())
        E12_RUN["started_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
        print("E12 solving:", len(E12_ORDER), "tasks x", E12_BRANCHES, "branches | streams", E12_RUN["streams"],
              "| model", E12_RUN["model_id"], "| stops in", E12_RUN["deadline_in_sec"], "s | order head", E12_ORDER[:5], flush=True)
        threading.Thread(target=_e12_heartbeat, daemon=True).start()
        E12A.solve_all({t: CHALLENGES[t] for t in E12_ORDER}, E12_ORDER, out_dir=E12_OUT,
                       deadline=time.monotonic() + max(0.0, _end - time.perf_counter()), base_url=SGL_URL + "/v1",
                       model_id=E12_RUN["model_id"], branches=E12_BRANCHES, max_generated_tokens=E12_MAX_GEN_PER_BRANCH,
                       save_request_logs=E12_REQUEST_LOGS, on_branch_done=_e12_branch_done)
    except BaseException:
        E12_RUN["error"] = traceback.format_exc()[-3000:]
        print("E12 solver error (submission.json keeps what was written):", E12_RUN["error"], flush=True)
else:
    E12_RUN["error"] = f"not run: ENGINE_READY={ENGINE_READY}, sandbox self-test={E10_SANDBOX_SELFTEST.get('ok')}"
    print("E12:", E12_RUN["error"], "-> CELL 0 fallback submission", flush=True)
_E12_STOP.set()
E12_RUN["finished_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
try:
    e12_write_submission()
except Exception:
    print("E12 final submission write failed:", traceback.format_exc()[-1000:], flush=True)
'''

# ------------------------------------------------------------------ CELL 7: report + bundle
c_rep = r'''# ==================================================================== CELL 7 — E12: report, score (evaluation) and bundle
try:
    with _E12_LOCK:
        _recs = {t: list(v) for t, v in E12_RECORDS.items()}
    _gen = E12A.build_submission({t: CHALLENGES[t] for t in TASK_IDS}, _recs)
    write_json_atomic(OUT / "e12_generated_attempts.json", _gen)

    def _usage_summary(u):
        p = sum(int(x.get("prompt") or 0) for x in u); c = sum(int(x.get("cached") or 0) for x in u)
        return {"requests": len(u), "prompt_tokens": p, "cached_tokens": c, "cached_pct": round(100 * c / p, 1) if p else None,
                "completion_tokens": sum(int(x.get("completion") or 0) for x in u),
                "cached_pct_by_request": [round(100 * int(x.get("cached") or 0) / int(x["prompt"]), 1) if x.get("prompt") else None
                                          for x in u][:400]}
    branches = []
    for t in TASK_IDS:
        for r in _recs.get(t, []):
            subs = r.get("submissions") or []
            branches.append({"task": t, "branch": r.get("branch"), "accepted": bool(r.get("accepted")), "turns": r.get("turns"),
                             "failures": r.get("failures"), "note": r.get("note"), "generated_tokens": r.get("generated_tokens"),
                             "total_tokens": r.get("total_tokens"), "sec": round((r.get("t1") or 0) - (r.get("t0") or 0), 1),
                             "submissions": len(subs), "train_exact_per_submission": [sum(map(bool, s.get("train_exact") or []))
                                                                                       for s in subs],
                             "usage": _usage_summary(r.get("usage") or [])})
    _all_u = [x for t in TASK_IDS for r in _recs.get(t, []) for x in (r.get("usage") or [])]
    report = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "configuration": CONFIG, "engine": ENGINE_INFO,
              "e12_run": E12_RUN, "order": E12_ORDER, "branches": branches, "usage_all": {k: v for k, v in _usage_summary(_all_u).items()
                                                                                        if k != "cached_pct_by_request"},
              "tasks_planned": len(TASK_IDS), "queries_planned": sum(len(CHALLENGES[t]["test"]) for t in TASK_IDS)}
    if SOLUTIONS is not None:
        hits = pass1 = pass2 = total = 0; task_norm = 0.0; recs_eval = []
        for t in TASK_IDS:
            th = 0
            for qi, att in enumerate(_gen[t]):
                gold = SOLUTIONS[t][qi]; a = att["attempt_1"] == gold; b = att["attempt_2"] == gold
                hits += a or b; pass1 += a; pass2 += b; total += 1; th += a or b
                acc = [s for r in _recs.get(t, []) for s in (r.get("submissions") or []) if s.get("accepted")]
                recs_eval.append({"task": t, "query": qi, "hit": bool(a or b), "pass1": bool(a), "pass2": bool(b),
                                  "accepted_programs": len(acc),
                                  "accepted_programs_correct": sum(1 for s in acc if qi < len(s.get("test_outputs") or [])
                                                                   and s["test_outputs"][qi] == gold)})
            task_norm += th / max(1, len(_gen[t]))
        report.update(eval_query_hits=hits, eval_query_total=total, eval_pass1_hits=pass1, eval_pass2_hits=pass2,
                      eval_task_norm_pct=round(100 * task_norm / max(1, len(TASK_IDS)), 2), eval_unit_records=recs_eval)
    try:
        report["e11"] = {"decode_summary": e11_decode_summary(),
                         "shim": engine.stats if isinstance(getattr(engine, "stats", None), dict) else None,
                         "server_log": str(E11_SERVER.log_path) if E11_SERVER is not None else None}
    except Exception as _e11e:
        report["e11"] = {"error": repr(_e11e)[:300]}
    write_json_atomic(OUT / "e12_run_report.json", report)
    print("E12 RUN REPORT:", OUT / "e12_run_report.json", "| eval", report.get("eval_query_hits"), "/", report.get("eval_query_total"),
          "| task-norm", report.get("eval_task_norm_pct"), "| accepted branches", E12_RUN["branches_accepted"],
          "| cached", report["usage_all"].get("cached_pct"), "%", flush=True)
    e12_write_submission()
    _pack = WORKING / "Q27X_E12_DUCK_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in (SUBMISSION_PATH, OUT / "e12_run_report.json", OUT / "e12_generated_attempts.json",
                   OUT / "sglang_decode_summary.json", OUT / "sglang_requests.jsonl"):
            if fp.exists():
                z.write(fp, arcname=fp.name)
        for fp in sorted(E12_OUT.rglob("*")):     # transcripts (full thinking) + branch results
            if fp.is_file() and (fp.name in ("transcript.txt", "branch_result.json") or fp.name.endswith("requests.jsonl")):
                z.write(fp, arcname="duck/" + fp.relative_to(E12_OUT).as_posix())
        if E11_SERVER is not None and E11_SERVER.log_path.exists():
            z.write(E11_SERVER.log_path, arcname=E11_SERVER.log_path.name)
    print("E12 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("E12 reporting error (submission.json still exists):", traceback.format_exc()[-3000:], flush=True)
try:
    e11_kill_server(wait=False)
except Exception:
    pass
'''

if VLLM:
    # E12-V rev 2 (GPT audits, validated). Applied to the vLLM build only, so the SGLang notebooks stay byte-identical.
    # CELL 6: the vLLM server serves the name `qwen`; the SGLang model path is wrong as a fallback model id.
    c_solve = once(c_solve, '''            E12_RUN["model_id"] = str(SGL_MODEL_PATH)''',
                   '''            E12_RUN["model_id"] = VLLM_SERVED_NAME             # E12-V: the name the vLLM server serves''',
                   "vllm model-id fallback")
    # CELL 7: vLLM omits prompt_tokens_details when 0 tokens were cached (and may omit it entirely), so a missing value
    # is "not reported", not a measured 0: keep the coverage and both rates apart.
    c_rep = once(c_rep, '''    def _usage_summary(u):
        p = sum(int(x.get("prompt") or 0) for x in u); c = sum(int(x.get("cached") or 0) for x in u)
        return {"requests": len(u), "prompt_tokens": p, "cached_tokens": c, "cached_pct": round(100 * c / p, 1) if p else None,''',
                 '''    def _usage_summary(u):
        p = sum(int(x.get("prompt") or 0) for x in u); c = sum(int(x.get("cached") or 0) for x in u)
        known = [x for x in u if x.get("cached") is not None and int(x.get("prompt") or 0) > 0]
        kp = sum(int(x["prompt"]) for x in known)
        return {"requests": len(u), "prompt_tokens": p, "cached_tokens": c,
                # E12-V: vLLM 0.19 omits the details when nothing was cached, so "absent" counts as 0 here ...
                "cached_pct": round(100 * c / p, 1) if p else None,
                # ... and these show how many requests actually reported it
                "cache_detail_requests": len(known), "cache_detail_coverage": round(len(known) / len(u), 3) if u else None,
                "cached_pct_reported_only": round(100 * c / kp, 1) if kp else None,''',
                 "vllm cache summary")
    c_rep = once(c_rep, '''                "cached_pct_by_request": [round(100 * int(x.get("cached") or 0) / int(x["prompt"]), 1) if x.get("prompt") else None''',
                 '''                "cached_pct_by_request": [round(100 * int(x["cached"]) / int(x["prompt"]), 1)
                                          if x.get("prompt") and x.get("cached") is not None else None''',
                 "vllm cache per request")

MD = ("# E12 · Duck REPL agent for ARC-AGI-2 on "
      + ("vLLM 0.19 FP8 + MTP-3 (the S1-60K engine) as an OpenAI server" if VLLM else "SGLang 0.5.21 + RedHat INT4 + DFlash2")
      + f" ({VARIANT})\n\n"
      "**What runs:** Tufa Labs' Duck harness (Daniel Franzen's milestone-2 fork, vendored byte-identical, Apache-2.0) with "
      "a thin ARC-2 adapter: the task is in the system prompt; the model works in a Python REPL with `train`, `test_inputs`, "
      "`check(transform)` and `submit(transform)`; the host re-runs each submitted function in a clean sandbox and accepts "
      "it only when every train pair is exact. Duck settings = the milestone notebook (12,288 output per request, yield at "
      "2,048 tokens, preserved thinking, 128 Ki context with drain to 58 Ki, T 0.7 / top-p 0.95 / top-k 20). 2 independent "
      "branches per task; selection: host-verified > agreement > distinct second grid > best partial > test input.\n\n"
      + ("**Engine:** Duck's own Kaggle vLLM launcher (vLLM 0.19.0 OpenAI server: qwen3_coder tool parser, qwen3 reasoning "
         "parser, preserve_thinking default, prefix caching) with the S1-60K engine settings (TP4, fp8_e4m3 KV, MTP-3, "
         "FlashInfer, 4096 batched tokens, 16 seqs, 0.90 memory, libcuda shim), max-model-len 135,168, 12 active streams; "
         "history key `reasoning` (Duck's default; vLLM 0.19 reads only that).\n\n" if VLLM else
         "**Engine:** the E11-06 W1 launch unchanged (TP4, INT4 + DFlash2, server 13 / client 12, context 155,648) + Duck's "
         "serving flags (qwen3 reasoning parser, qwen3_coder tool parser, preserve_thinking default, lpm, cache report).\n\n")
      +
      "**Modes:** commit = " + ("SMOKE1 (1 task, 2 branches, 15 min; step 3.5)" if VARIANT == "SMOKE1" else
                                "EVAL30 (the E11-05 EVAL30 tasks, 4 h; step 4)")
      + " with score; competition rerun = every hidden task, 12 h clock. Inputs = the E11-06 notebook's.\n")


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)},
                code(cells[1]), code(c1), code(cells[3]), code(c_pkg), code(cells[4]), code(c_eng), code(c_solve), code(c_rep)],
      "metadata": src.get("metadata", {}), "nbformat": src.get("nbformat", 4), "nbformat_minor": src.get("nbformat_minor", 5)}
out = HERE.parent / NAME
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest(), round(out.stat().st_size / 1e3), "kB | package",
      len(MANIFEST), "files,", round(len(ZIP_B64) / 1e3), "kB b64")
