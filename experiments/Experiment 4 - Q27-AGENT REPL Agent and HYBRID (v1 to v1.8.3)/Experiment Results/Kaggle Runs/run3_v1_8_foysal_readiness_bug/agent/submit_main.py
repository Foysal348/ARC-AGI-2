"""Everything the competition rerun does after the fallback submission is written:
install the vLLM wheelhouse, start the server, run the agent. It NEVER raises --
an exception in a rerun fails the submission even when submission.json exists --
so every stage is caught and logged, and the fallback stands if a stage fails.

    python submit_main.py '<json config>'
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_PID = Path("/kaggle/working/vllm_server.pid")     # v1.1: lets the launcher kill a leftover server group
REPORT_PATH = Path("/kaggle/working/agent_server_report.json")
LINK_DIR = Path("/tmp/q27a_cuda_link")
CUDA_LINK = {}
KERNEL_ISO = {}                                          # v1.7: unshare probe result (informational)
CURRENT = {"proc": None}


def ensure_libcuda():
    """v1.1 (Q27-XHIGH run 2: FlashInfer JIT failed with `ld: cannot find -lcuda`; Kaggle ships only libcuda.so.1).
    Make -lcuda linkable through a private symlink directory and prove it with a tiny link."""
    import shutil as _sh
    info = {"needed": True}
    cands = []
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64", "/usr/lib64"):
        cands += sorted(glob.glob(f"{d}/libcuda.so.1")) + sorted(glob.glob(f"{d}/libcuda.so.*[0-9]"))
    real = next((Path(c) for c in cands if Path(c).is_file()), None)
    if real is None:
        info.update(probe="SKIP", reason="libcuda.so.1 not found")
    else:
        LINK_DIR.mkdir(parents=True, exist_ok=True)
        link = LINK_DIR / "libcuda.so"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(real.resolve())
        info.update(real=str(real), shim=str(link))
        cxx = _sh.which("c++") or _sh.which("g++") or _sh.which("gcc")
        if cxx:
            src = Path("/tmp/q27a_cuda_probe.c")
            src.write_text("int cuInit(unsigned int);\nint main(void){return 0;}\nvoid *p = (void *)cuInit;\n")
            env = dict(os.environ, LIBRARY_PATH=f"{LINK_DIR}:" + os.environ.get("LIBRARY_PATH", ""))
            r = subprocess.run([cxx, "-x", "c", str(src), "-o", "/tmp/q27a_cuda_probe", "-lcuda"],
                               capture_output=True, text=True, env=env, timeout=120)
            info.update(probe="PASS" if r.returncode == 0 else "FAIL", probe_out=(r.stdout + r.stderr)[-600:])
        else:
            info.update(probe="SKIP", reason="no C compiler")
    CUDA_LINK.clear()
    CUDA_LINK.update(info)
    print("libcuda link:", {k: v for k, v in info.items() if k != "probe_out"}, flush=True)
    return info


def _on_sigterm(signum, frame):
    """v1.1: a SIGTERM to this launcher also ends the vLLM server's process group (it runs in its own session)."""
    if CURRENT["proc"] is not None:
        try:
            os.killpg(CURRENT["proc"].pid, 9)
        except Exception:
            pass
    sys.exit(0)


def classify_startup_failure(path=None):
    """v1.3: why did a rung fail to come up? (read from the tail of the server log)"""
    try:
        tail = Path(path or LOG).read_text(errors="replace")[-20000:].lower()
    except Exception:
        return "NO_LOG"
    if "cannot find -lcuda" in tail or "cannot find libcuda" in tail:
        return "CUDA_LINKER"
    if "unrecognized arguments" in tail or "error: argument" in tail:
        return "UNSUPPORTED_FLAGS"
    if "out of memory" in tail or "outofmemory" in tail:
        return "OOM"
    return "UNKNOWN"


def _smoke_task():
    """v1.3: a fresh random identity task per smoke, so a hard-coded print cannot pass the python check."""
    import random
    h, w = random.choice([(2, 3), (3, 2), (2, 4), (4, 3), (3, 5), (5, 2)])
    grid = [[random.randint(1, 9) for _ in range(w)] for _ in range(h)]
    test = [[random.randint(1, 9) for _ in range(w)] for _ in range(h)]
    return {"train": [{"input": grid, "output": grid}], "test": [{"input": test}]}, f"({h}, {w})"


def smoke(cfg, smoke_deadline):
    """The server answering /v1/models does not prove tool calling. v1.3 checks, all inside ONE deadline:
      forced    a named `submit` call (thinking off)
      auto      an automatic `python` call (thinking off)
      thinking  an automatic `python` call with thinking on and the agent's REAL thinking_token_budget
      roundtrip the agent's real loop: python call -> tool result sent back (role "tool") -> second request ->
                a `submit` that verifies (no tool_choice, so the automatic submit path is tested too)
    A call only counts if its code really runs on a fresh random task (python must print the right shape, submit
    must verify and return the test grid). The thinking test fails if the client had to drop the budget.
    ok = (auto or thinking) and roundtrip."""
    sys.path.insert(0, HERE)
    from client import ChatClient
    from repl_agent import make_tools, verify
    from repl import run_snippet
    task, shape = _smoke_task()
    tools = make_tools(cfg.get("retain_functions", False))
    out = {"forced": [], "auto": [], "thinking": [], "roundtrip": [], "deadline_hit": False}

    def left():
        return smoke_deadline - time.time()

    def semantic(name, code):
        try:
            compile(code, "<smoke>", "exec")
        except Exception as e:
            return False, f"does not compile: {e}"[:200], None
        if left() <= 2:
            return False, "smoke deadline", None
        if name == "submit":
            n_ok, test_out, fb = verify(code, task, timeout=min(8.0, left()))
            return (n_ok == 1 and test_out == [task["test"][0]["input"]]), fb[:200], None
        o = run_snippet(code, task, timeout=min(8.0, left()))
        return (shape in o and "Traceback" not in o and "[snippet crashed]" not in o), o[:200], o

    def check(r, name):
        calls = r.get("tool_calls") or []
        if "error" in r or not calls or calls[0].get("name") != name:
            return False, "no tool call, or the wrong tool", None
        args = calls[0].get("arguments")
        code = args.get("code") if isinstance(args, dict) else None
        if not isinstance(code, str) or not code.strip():
            return False, "no code string", None
        return semantic(name, code)

    def rec(r, ok, why, t):
        return {"ok": ok, "why": why, "sec": round(time.time() - t, 1), "error": r.get("error"),
                "finish": r.get("finish_reason"), "completion_tokens": r.get("completion_tokens"),
                "calls": [(c.get("name"), str(c.get("arguments"))[:200]) for c in r.get("tool_calls") or []],
                "content": (r.get("content") or "")[:200]}

    def ask(cl, msgs, choice):
        rem = left()
        if rem <= 5:
            out["deadline_hit"] = True
            return None
        return cl.chat(msgs, tools=tools, tool_choice=choice, timeout=min(75.0, rem))

    sysmsg = {"role": "system", "content": "You are testing tool calling. Answer only with the requested tool call."}
    ask_submit = {"role": "user", "content": "Call the `submit` tool with a Python program that defines "
                                             "transform(grid) and returns grid unchanged."}
    ask_python = {"role": "user", "content": "Call the `python` tool with code that prints train_inputs[0].shape."}
    budget = cfg.get("thinking_token_budget") or 6144           # v1.3: the agent's real budget, not 1024
    plain = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=0.0, max_tokens=1024, thinking=False)
    # effort "low" keeps the smoke short; what is tested is that the server ACCEPTS this budget value with tools
    think = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=0.6,
                       max_tokens=max(int(cfg.get("turn_max_tokens") or 8192), budget + 1024), thinking=True,
                       thinking_token_budget=budget, template_kwargs={"reasoning_effort": "low"})
    first_python = {}
    for key, cl, msgs, name, choice in (("forced", plain, [sysmsg, ask_submit], "submit", "submit"),
                                        ("auto", plain, [sysmsg, ask_python], "python", None),
                                        ("thinking", think, [sysmsg, ask_python], "python", None)):
        for _ in range(2):
            t = time.time()
            r = ask(cl, msgs, choice)
            if r is None:
                break
            ok, why, tool_out = check(r, name)
            out[key].append(rec(r, ok, why, t))
            if ok:
                if name == "python":
                    first_python.setdefault(key, (cl, msgs, r, tool_out))
                break
    out["forced_ok"] = any(x["ok"] for x in out["forced"])
    out["auto_ok"] = any(x["ok"] for x in out["auto"])
    out["thinking_budget_requested"] = budget
    out["thinking_budget_retained"] = think.thinking_token_budget == budget
    out["thinking_ok"] = any(x["ok"] for x in out["thinking"]) and out["thinking_budget_retained"]
    # v1.5: the EXACT request shape the agent will send (its template_kwargs, sampling, budget, max_tokens, tools),
    # cut to 32 output tokens -- the server only has to ACCEPT it (run 1: the template rejected effort "high")
    agent_cl = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=cfg.get("temperature", 0.6),
                          top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20), max_tokens=32,
                          thinking=cfg.get("thinking", True) and out["thinking_ok"],
                          thinking_token_budget=cfg.get("thinking_token_budget") if out["thinking_ok"] else None,
                          template_kwargs=cfg.get("template_kwargs"))   # same fallback the ladder will apply
    out["agent_params"] = []
    rungs_eff = [None] + [{"reasoning_effort": e} for e in (cfg.get("effort_ladder") or [])]
    for extra_kw in rungs_eff:                      # the agent's own effort and every effort-ladder step
        if left() <= 5:
            out["deadline_hit"] = True
            break
        t = time.time()
        r = agent_cl.chat([sysmsg, ask_python], tools=tools, timeout=min(75.0, left()), template_kwargs=extra_kw)
        okp = "error" not in r and not getattr(agent_cl, "budget_dropped", False)
        out["agent_params"].append({"ok": okp, "effort": (extra_kw or cfg.get("template_kwargs") or {}),
                                    "sec": round(time.time() - t, 1), "error": (r.get("error") or "")[:300],
                                    "finish": r.get("finish_reason")})
    out["agent_params_ok"] = bool(out["agent_params"]) and all(x["ok"] for x in out["agent_params"])
    # v1.6: the DIRECT rescue's two request shapes (its effort + 32k budget; the no-thinking commit call), 32 output
    # tokens each. Informational: a rejection is reported, it does not fail the smoke (the agent can still run)
    out["rescue_params"] = []
    if cfg.get("rescue"):
        from rescue import direct_client
        rcl = direct_client(cfg)
        for extra_kw in (None, {"enable_thinking": False, "reasoning_effort": "low"}):
            if left() <= 5:
                out["deadline_hit"] = True
                break
            t = time.time()
            r = rcl.chat([sysmsg, {"role": "user", "content": "Reply with the digit 7."}], max_tokens=32,
                         timeout=min(75.0, left()), template_kwargs=extra_kw)
            out["rescue_params"].append({"ok": "error" not in r and not getattr(rcl, "budget_dropped", False),
                                         "commit_call": extra_kw is not None, "sec": round(time.time() - t, 1),
                                         "error": (r.get("error") or "")[:300], "finish": r.get("finish_reason")})
    out["rescue_params_ok"] = (all(x["ok"] for x in out["rescue_params"]) if out["rescue_params"] else None)
    # roundtrip on the mode the agent will actually use (thinking if it passed, else plain)
    mode = "thinking" if out["thinking_ok"] and "thinking" in first_python else "auto"
    out["roundtrip_mode"] = mode
    if mode in first_python:
        cl, msgs, r1, tool_out = first_python[mode]
        c = r1["tool_calls"][0]
        follow = msgs + [
            {"role": "assistant", "content": r1.get("content") or "",
             "tool_calls": [{"id": c["id"], "type": "function",
                             "function": {"name": "python", "arguments": json.dumps(c["arguments"])}}]},
            {"role": "tool", "tool_call_id": c["id"], "content": tool_out},
            {"role": "user", "content": "Good. Now call the `submit` tool with a Python program that defines "
                                        "transform(grid) and returns grid unchanged."}]
        for _ in range(2):
            t = time.time()
            r2 = ask(cl, follow, None)
            if r2 is None:
                break
            ok, why, _ = check(r2, "submit")
            out["roundtrip"].append(rec(r2, ok, why, t))
            if ok:
                break
    out["roundtrip_ok"] = any(x["ok"] for x in out["roundtrip"])
    out["ok"] = (out["auto_ok"] or out["thinking_ok"]) and out["roundtrip_ok"] and out["agent_params_ok"]
    print("tool smoke:", {k: out[k] for k in ("forced_ok", "auto_ok", "thinking_ok", "thinking_budget_retained",
                                              "roundtrip_ok", "roundtrip_mode", "agent_params_ok",
                                              "rescue_params_ok", "deadline_hit", "ok")}, flush=True)
    return out
WORK = Path("/kaggle/working")
SITE = WORK / "vllm-site-packages"
LOG = WORK / "vllm-server.log"


def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def install(cfg):
    if cfg.get("dryrun"):                                 # v1.8 DRYRUN: no vLLM on a CPU run
        print("DRYRUN: install skipped", flush=True)
        return
    wh = one("/kaggle/input/**/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock").parent
    t = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wh),
                    "--requirement", str(wh / "requirements.lock"), "--target", str(SITE),
                    "--upgrade", "--ignore-installed", "--only-binary", ":all:", "--no-compile",
                    "--disable-pip-version-check", "--no-warn-conflicts", "-q"], check=True,
                   timeout=float(cfg.get("install_timeout_s", 1800)))   # v1.4
    print(f"wheelhouse installed in {time.time() - t:.0f}s", flush=True)
    if cfg.get("overlay"):
        wheels = sorted(glob.glob(cfg["overlay"], recursive=True))
        if not wheels:
            raise FileNotFoundError(cfg["overlay"])
        for pat in ("transformers", "transformers-*.dist-info", "huggingface_hub",
                    "huggingface_hub-*.dist-info"):
            for p in SITE.glob(pat):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--target", str(SITE), "--upgrade", "--force-reinstall", "--no-compile",
                        "--disable-pip-version-check", "-q", *wheels], check=True, timeout=600)   # v1.4
        print("overlay installed:", [Path(w).name for w in wheels], flush=True)


def agent_e2e(cfg, e2e_deadline):
    """v1.7 (GPT final audit, Patch B): the agent's REAL request -- its effort, sampling, thinking budget and full
    per-turn max_tokens, with the agent's tools -- on a fresh random task: a python call whose code really runs, the
    output sent back as a role "tool" message, then a submit that verifies. Tried with the configured effort first,
    then each effort-ladder step; returns the first effort that passes (ok False if none does). Each request is bounded
    (<= 300 s) and the whole probe by e2e_deadline."""
    sys.path.insert(0, HERE)
    from client import ChatClient
    from repl_agent import make_tools, verify
    from repl import run_snippet
    tools = make_tools(cfg.get("retain_functions", False))
    base_tk = dict(cfg.get("template_kwargs") or {})
    efforts = [base_tk.get("reasoning_effort")] + [e for e in (cfg.get("effort_ladder") or [])
                                                   if e != base_tk.get("reasoning_effort")]
    sysmsg = {"role": "system", "content": "You solve grid puzzles with the tools. Answer only with the requested "
                                           "tool call."}
    ask_py = {"role": "user", "content": "Call the `python` tool with code that prints train_inputs[0].shape."}
    out = {"ok": False, "effort": None, "tries": []}

    def first_call(r, name):
        calls = r.get("tool_calls") or []
        if "error" in r or not calls or calls[0].get("name") != name or not isinstance(calls[0].get("arguments"), dict):
            return None, None
        code = calls[0]["arguments"].get("code")
        return (calls[0], code) if isinstance(code, str) and code.strip() else (None, None)

    for eff in efforts:
        left = e2e_deadline - time.time()
        if left < 60:
            out["deadline_hit"] = True
            break
        task, shape = _smoke_task()
        cl = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=cfg.get("temperature", 0.6),
                        top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20),
                        max_tokens=int(cfg.get("turn_max_tokens") or cfg.get("max_tokens") or 8192),
                        thinking=cfg.get("thinking", True), thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=dict(base_tk, **({"reasoning_effort": eff} if eff else {})))
        t = time.time()
        r1 = cl.chat([sysmsg, ask_py], tools=tools, timeout=min(300.0, left))
        rec = {"effort": eff, "python_s": round(time.time() - t, 1), "python_finish": r1.get("finish_reason"),
               "python_tokens": r1.get("completion_tokens"), "error": (r1.get("error") or "")[:300], "ok": False}
        c1, code1 = first_call(r1, "python")
        o = run_snippet(code1, task, timeout=8.0) if code1 else ""
        rec["python_ok"] = bool(code1) and shape in o and "Traceback" not in o and "[snippet crashed]" not in o
        if rec["python_ok"] and e2e_deadline - time.time() > 30:
            follow = [sysmsg, ask_py,
                      {"role": "assistant", "content": r1.get("content") or "",
                       "tool_calls": [{"id": c1["id"], "type": "function",
                                       "function": {"name": "python", "arguments": json.dumps(c1["arguments"])}}]},
                      {"role": "tool", "tool_call_id": c1["id"], "content": o},
                      {"role": "user", "content": "Good. Now call the `submit` tool with a Python program that "
                                                  "defines transform(grid) and returns grid unchanged."}]
            t = time.time()
            r2 = cl.chat(follow, tools=tools, timeout=min(300.0, e2e_deadline - time.time()))
            rec.update(submit_s=round(time.time() - t, 1), submit_finish=r2.get("finish_reason"),
                       submit_tokens=r2.get("completion_tokens"), error2=(r2.get("error") or "")[:300])
            _, code2 = first_call(r2, "submit")
            if code2:
                n_ok, test_out, _fb = verify(code2, task, timeout=8.0)
                rec["ok"] = n_ok == 1 and test_out == [task["test"][0]["input"]]
        rec["budget_dropped"] = bool(getattr(cl, "budget_dropped", False))
        rec["ok"] = rec["ok"] and not rec["budget_dropped"]
        out["tries"].append(rec)
        print("agent e2e:", rec, flush=True)
        if rec["ok"]:
            out.update(ok=True, effort=eff)
            break
    return out


def start_server(cfg, extra, timeout=1500):
    if cfg.get("dryrun"):                                 # v1.8 DRYRUN: fake_vllm.py, same port and API
        import urllib.request
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_vllm.py"), "1234"],
                                stdout=open(LOG, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        CURRENT["proc"] = proc
        SERVER_PID.write_text(str(proc.pid))
        t_f = time.time()
        while time.time() - t_f < 60:
            try:
                urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2).read()
                print(f"DRYRUN: fake server ready after {time.time() - t_f:.1f}s", flush=True)
                return proc
            except Exception:
                time.sleep(0.5)
        kill(proc)
        return None
    model = one(cfg["model_glob"]).parent
    env = dict(os.environ, PYTHONPATH=str(SITE), USE_TF="0", TRANSFORMERS_NO_TF="1",
               TRANSFORMERS_NO_TORCHVISION="1", VLLM_NO_USAGE_STATS="1",
               LIBRARY_PATH=f"{LINK_DIR}:/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", ""),
               VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1")   # v1.1: libcuda shim dir first; graph memory in budget
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
           "--served-model-name", "qwen", "--host", "127.0.0.1", "--port", "1234",
           "--tensor-parallel-size", "4", "--enable-auto-tool-choice", *cfg["parsers"],
           "--generation-config", "vllm", "--enable-prefix-caching",
           "--max-model-len", str(cfg.get("max_model_len", 65536)), "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", str(cfg.get("max_num_seqs", 32)),
           # v1.1: 4096 = the batch step that fixed Q27-XHIGH run 2's MTP-drafter OOM (default prefill step was larger)
           "--max-num-batched-tokens", str(cfg.get("max_num_batched_tokens", 4096)), *extra]
    print(" ".join(cmd), flush=True)
    t = time.time()
    # "w": a retry must not see the previous attempt's "EngineCore failed".
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    CURRENT["proc"] = proc
    SERVER_PID.write_text(str(proc.pid))                 # v1.1: the notebook kills this group if we are killed
    while time.time() - t < timeout:
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        if proc.poll() is not None or "EngineCore failed" in text:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
            print(f"server ready after {time.time() - t:.0f}s", flush=True)
            return proc
        except Exception:
            time.sleep(10)
    print("server FAILED to start; tail:\n" + "\n".join(
        LOG.read_text(errors="replace").splitlines()[-30:]), flush=True)
    kill(proc)
    return None


def kill(proc):
    """Kill the server's whole process group -- engine workers hold GPU memory -- and wait
    for the GPUs to drain, so a restart does not hit out-of-memory."""
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=60)
    except Exception:
        pass
    if shutil.which("nvidia-smi") is None:           # v1.8: no GPU (DRYRUN) -> nothing to drain
        return
    t_end = time.time() + 240                       # v1.4: the whole drain wait is bounded
    while time.time() < t_end:
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=20).stdout.split()
        except Exception:
            used = []
        if used and all(int(u) < 1500 for u in used if u.strip().isdigit()):
            return
        time.sleep(5)


def server_ladder(cfg):
    """Option sets to try in order. An explicit cfg["server_ladder"] wins (plain serving is
    appended if missing); otherwise server_extra + server_try -> server_extra -> plain."""
    if cfg.get("server_ladder"):
        ladder = [list(x) for x in cfg["server_ladder"]]
        return ladder if [] in ladder else ladder + [[]]
    base = cfg.get("server_extra", [])
    ladder = [base + cfg["server_try"]] if cfg.get("server_try") else []
    return ladder + [base] + ([[]] if base else [])


def main(cfg):
    proc = None
    try:
        import signal as _signal
        _signal.signal(_signal.SIGTERM, _on_sigterm)
        t_install = time.time()
        install(cfg)
        install_s = round(time.time() - t_install)
        try:                                 # v1.7: informational -- can a rootless namespace sandbox work here?
            _ns = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--fork", "true"],
                                 capture_output=True, text=True, timeout=10)
            KERNEL_ISO["unshare_rc"], KERNEL_ISO["stderr"] = _ns.returncode, _ns.stderr[-200:]
        except Exception as _e:
            KERNEL_ISO["unshare_rc"], KERNEL_ISO["stderr"] = None, repr(_e)[:200]
        link = {"probe": "SKIP (dryrun)"} if cfg.get("dryrun") else ensure_libcuda()   # v1.8
        if link.get("probe") == "FAIL":
            # v1.2: logged loudly, not fatal -- only FlashInfer's JIT needs -lcuda; other attention backends on the
            # ladder load without it, and every rung must still pass the tool smoke before the agent starts
            print("WARNING: -lcuda link probe FAILED; rungs that need FlashInfer JIT will fail", flush=True)
        # A ladder of server option sets, most ambitious first; the last is plain serving.
        # Untested-on-L4 options (MTP, --reasoning-config) must never cost the whole run.
        # v1.1: each rung must also pass a real tool-calling smoke test, and the ladder never eats the time the
        # agent needs (min_agent_s before the deadline).
        ladder = [[]] if cfg.get("dryrun") else server_ladder(cfg)      # v1.8 DRYRUN: one fake rung
        report = {"cuda_link": CUDA_LINK, "install_s": install_s, "rungs": [], "kernel_isolation": KERNEL_ISO}
        reserve = float(cfg.get("min_agent_s", 1800))
        for i, extra in enumerate(ladder):
            left = cfg["deadline_epoch"] - time.time() - reserve
            if left < 300:
                print(f"no time for server option set {i} ({left:.0f}s before the agent reserve) -> stop", flush=True)
                break
            t_rung = time.time()
            proc = start_server(cfg, extra, timeout=min(1500.0, left))
            rung = {"index": i, "extra": extra, "server_up": proc is not None, "start_s": round(time.time() - t_rung)}
            if proc is not None:
                # v1.2: the smoke has its own hard deadline inside the time left before the agent reserve
                slack = cfg["deadline_epoch"] - time.time() - reserve
                if slack <= 60:
                    print(f"option set {i}: server up but no time left for the tool smoke -> stop", flush=True)
                    rung["note"] = "no time for the smoke"
                    report["rungs"].append(rung)
                    kill(proc)
                    proc = None
                    break
                t_smoke = time.time()
                sm = smoke(cfg, time.time() + min(240.0, slack - 30.0))   # v1.3: 240 s (adds the roundtrip)
                rung["smoke_s"] = round(time.time() - t_smoke)
                rung["smoke"] = sm
                if sm["ok"]:
                    if not sm["forced_ok"]:
                        cfg["force_submit_after"] = 0      # named tool_choice does not work on this rung
                    if not sm["thinking_ok"]:
                        cfg["thinking"] = False            # explicit fallback: no thinking, so no unbounded thinking
                        cfg["thinking_token_budget"] = None
                    report["rungs"].append(rung)
                    report["chosen"] = {"index": i, "extra": extra, "force_submit_after": cfg.get("force_submit_after"),
                                        "thinking": cfg.get("thinking", True),
                                        "thinking_budget_retained": sm.get("thinking_budget_retained", False)}
                    report["remaining_agent_s"] = round(cfg["deadline_epoch"] - time.time())
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"server up with option set {i}: {extra}", flush=True)
                    break
                kill(proc)
                proc = None
                if sm.get("roundtrip_ok") and not sm.get("agent_params_ok"):
                    # v1.5: the server works, the AGENT's request is rejected (a config error, the same on every rung)
                    rung["failure"] = "AGENT_PARAMS"
                    report["rungs"].append(rung)
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"option set {i}: agent request parameters rejected -> stop the ladder", flush=True)
                    break
                print(f"option set {i}: server up but tool smoke FAILED -> next", flush=True)
            else:
                # v1.3: classify the startup failure; no rung changes the attention backend, so a CUDA-linker failure
                # would repeat on every rung -> stop instead of reloading the 27B model again and again
                rung["failure"] = classify_startup_failure()
                if rung["failure"] == "CUDA_LINKER":
                    report["rungs"].append(rung)
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"option set {i}: CUDA linker failure ({CUDA_LINK.get('probe')}) -> stop the ladder",
                          flush=True)
                    break
            report["rungs"].append(rung)
            REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
            print(f"option set {i} failed ({rung.get('failure', 'smoke')}) -> next", flush=True)
            time.sleep(20)
        REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
        if proc is None:
            return
        # v1.7 (GPT final audit): a rejected DIRECT-rescue request shape disables the rescue (the agent keeps the
        # whole time; the eval report marks the three-way comparison incomplete) instead of failing task by task
        chosen_sm = (report["rungs"][-1].get("smoke") or {}) if report["rungs"] else {}
        cfg["rescue_smoke_ok"] = chosen_sm.get("rescue_params_ok")
        if cfg.get("rescue") and chosen_sm.get("rescue_params_ok") is False:
            cfg["rescue"] = False
            report["rescue_disabled"] = "rescue request shape rejected by the server"
            print("WARNING: DIRECT rescue requests rejected -> rescue disabled", flush=True)
        # v1.7 (GPT Patch B): the agent's REAL request end to end before any task (bounded, never fatal)
        e2e_budget = min(float(cfg.get("e2e_max_s", 600)),
                         cfg["deadline_epoch"] - time.time() - float(cfg.get("e2e_reserve_s", 1800)))
        if e2e_budget >= float(cfg.get("e2e_min_s", 120)):
            e2e = agent_e2e(cfg, time.time() + e2e_budget)
            report["agent_e2e"] = e2e
            cfg["agent_e2e_ok"] = e2e["ok"]
            if e2e["ok"] and e2e["effort"] and e2e["effort"] != (cfg.get("template_kwargs") or {}).get("reasoning_effort"):
                lad = list(cfg.get("effort_ladder") or [])
                cfg["template_kwargs"] = dict(cfg.get("template_kwargs") or {}, reasoning_effort=e2e["effort"])
                cfg["effort_ladder"] = lad[lad.index(e2e["effort"]) + 1:] if e2e["effort"] in lad else lad
                print(f"agent effort switched to {e2e['effort']} (the configured effort failed end to end)", flush=True)
            elif not e2e["ok"]:
                print("WARNING: agent end-to-end probe failed on every effort; running with the configured effort",
                      flush=True)
        else:
            report["agent_e2e"] = {"ok": None, "skipped": f"only {e2e_budget:.0f}s available"}
        report["agent_config"] = {k: cfg.get(k) for k in ("dryrun", "template_kwargs", "effort_ladder", "thinking",
                                                          "thinking_token_budget", "rescue", "rescue_smoke_ok")}
        REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
        sys.path.insert(0, HERE)
        if cfg.get("eval"):                  # Q27-AGENT: commit run = evaluation subset, scored
            import run_eval
            run_eval.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))
        else:
            import run_submit
            run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))
    except BaseException:
        print("submit_main caught:\n" + traceback.format_exc(), flush=True)
    finally:
        if proc is not None:
            kill(proc)
        shutil.rmtree(SITE, ignore_errors=True)


if __name__ == "__main__":
    try:
        main(json.loads(sys.argv[1]))
    except BaseException:
        print(traceback.format_exc(), flush=True)
    sys.exit(0)
