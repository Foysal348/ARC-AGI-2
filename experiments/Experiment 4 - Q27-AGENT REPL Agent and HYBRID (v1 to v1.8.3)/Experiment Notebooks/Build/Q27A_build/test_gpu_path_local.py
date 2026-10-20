"""v1.8.1/v1.8.2: rehearse the GPU path with the REAL code. submit_main.main() runs with the launcher's own cfg
(parsed from the notebook's launch cell, not dryrun): real server ladder, real start_server readiness loop, real smoke,
real e2e probe, real run_eval / run_submit. Only three things are substituted:
  * install()        -- pip from the offline wheelhouse (proven in GPU runs 1-3)
  * ensure_libcuda() -- the libcuda shim (probe PASS in GPU runs 1-3)
  * the vLLM command  -- subprocess.Popen launches fake_vllm.py instead of `python -m vllm.entrypoints...`
Cases: the normal commit run; GPU run 3's bug re-inserted (must fail fast); v1.8.2 gates -- e2e NO-GO in the commit
run (no evaluation), e2e NO-GO in the hidden run (agent never called, DIRECT on every task, rescue log written), a
rejected rescue probe with recorded EVAL answers (comparison still complete), the agent's thinking fallback leaving
the DIRECT rescue's thinking + 49k budget untouched.

usage: python test_gpu_path_local.py <notebook> <competition data dir>
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NB, DATA = sys.argv[1], Path(sys.argv[2])
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
c12 = next(s for s in cells if s.startswith("# ==================================================================== launch"))
SERVER = json.loads(c12.split('SERVER = json.loads(r"""', 1)[1].split('""")', 1)[0])
AGENT = json.loads(c12.split('AGENT = json.loads(r"""', 1)[1].split('""")', 1)[0])
EVAL12 = json.loads(c12.split("EVAL12 = ", 1)[1].splitlines()[0])
MODS = ("submit_main", "run_eval", "repl_agent", "rescue", "client", "repl", "sandbox", "agent", "grids", "backend",
        "run_direct", "run_submit")


def run_case(inject_bug=False, e2e_fail=False, rerun=False, rescue_probe_fail=False, thinking_fallback=False):
    root = Path(tempfile.mkdtemp(prefix="q27gpu_"))
    agent_dir = root / "agent"
    agent_dir.mkdir()
    for s in cells:
        if s.startswith("%%writefile /kaggle/working/agent/"):
            (agent_dir / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n",
                                                                      encoding="utf-8")
    if inject_bug:                                   # GPU run 3's exact line, back in the dryrun branch
        p = agent_dir / "submit_main.py"
        src = p.read_text(encoding="utf-8")
        marker = '    if cfg.get("dryrun"):                                 # v1.8 DRYRUN: fake_vllm.py, same port and API\n'
        assert src.count(marker) == 1
        p.write_text(src.replace(marker, marker + "        import urllib.request\n"), encoding="utf-8")
    model = root / "model" / "qwen3-8-27b-fp8-repacked-v1" / "hf-fp8"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    out = root / "agent_eval"
    for k in [m for m in list(sys.modules) if m in MODS]:
        del sys.modules[k]
    sys.path.insert(0, str(agent_dir))
    import submit_main
    import run_eval
    import run_submit
    calls = {"run_eval": 0, "solve_repl_submit": 0, "direct_clients": []}
    real_eval_main, real_solve = run_eval.main, run_submit.solve_repl

    def eval_main(c):
        calls["run_eval"] += 1
        return real_eval_main(c)

    def solve_submit(*a, **k):
        calls["solve_repl_submit"] += 1
        return real_solve(*a, **k)

    run_eval.main, run_submit.solve_repl = eval_main, solve_submit
    import rescue
    real_dc = rescue.direct_client

    def dc(c):
        cl = real_dc(c)
        calls["direct_clients"].append({"thinking": cl.thinking, "budget": cl.thinking_token_budget})
        return cl

    rescue.direct_client = dc
    submit_main.install = lambda cfg: print("[test] install substituted", flush=True)
    submit_main.ensure_libcuda = lambda: {"probe": "PASS (substituted)"}
    submit_main.LOG = root / "vllm-server.log"
    submit_main.SERVER_PID = root / "vllm_server.pid"
    submit_main.REPORT_PATH = root / "agent_server_report.json"
    submit_main.SITE = root / "site"
    if e2e_fail:
        submit_main.agent_e2e = lambda cfg, dl, **kw: {"ok": False, "effort": None, "tries": [{"effort": "xhigh", "ok": False}]}
    if rescue_probe_fail or thinking_fallback:
        real_smoke = submit_main.smoke

        def smoke(cfg, dl):
            sm = real_smoke(cfg, dl)
            if rescue_probe_fail:
                sm["rescue_params_ok"] = False
            if thinking_fallback:
                sm["thinking_ok"] = False
            return sm

        submit_main.smoke = smoke
    real_popen = subprocess.Popen
    launched = []

    def popen(cmd, *a, **k):
        if isinstance(cmd, list) and "vllm.entrypoints.openai.api_server" in cmd:
            launched.append(cmd)
            cmd = [sys.executable, str(agent_dir / "fake_vllm.py"), "1234"]
        return real_popen(cmd, *a, **k)

    submit_main.subprocess.Popen = popen
    cfg = dict(SERVER, **AGENT)
    cfg.update(deadline_epoch=time.time() + 40 * 60, min_agent_s=60, data_dir=str(DATA),
               model_glob=str(root / "model" / "**" / "config.json"), e2e_reserve_s=60)
    if rerun:
        cfg.update(eval=False, sub_path=str(root / "submission.json"), interactive_tasks=3,
                   rescue_log=str(root / "rescue_attempts.jsonl"), challenges="arc-agi_test_challenges.json")
    else:
        cfg.update(eval=True, eval_task_ids=EVAL12[:3], eval_out=str(out), rescue_all=True)
    t = time.time()
    try:
        submit_main.main(cfg)
    finally:
        submit_main.subprocess.Popen = real_popen
        sys.path.remove(str(agent_dir))
        try:                                           # like the launcher's kill_leftover_server()
            pid = int(submit_main.SERVER_PID.read_text().strip())
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)] if os.name == "nt" else ["kill", "-9", str(pid)],
                           capture_output=True, text=True)
        except Exception:
            pass
    took = time.time() - t
    rep = json.loads(submit_main.REPORT_PATH.read_text()) if submit_main.REPORT_PATH.exists() else {}
    ev = json.loads((out / "eval_report.json").read_text()) if (out / "eval_report.json").exists() else None
    return dict(took=took, launched=launched, rep=rep, ev=ev, calls=calls, root=root)


# 1. the normal commit run
r = run_case()
rep, ev = r["rep"], r["ev"]
assert r["launched"] and "--tensor-parallel-size" in r["launched"][0], r["launched"]
assert '"num_speculative_tokens": 3' in " ".join(r["launched"][0]), r["launched"][0]   # v1.8.3: rung 0 = mtp x3
assert "kv_tokens" in (r["ev"] or {}).get("server_metrics", {}), (r["ev"] or {}).get("server_metrics")
assert rep.get("chosen", {}).get("index") == 0, rep.get("rungs")            # rung 0 detected as up (run 3: never)
sm = rep["rungs"][0]["smoke"]
assert sm["ok"] and sm["rescue_params_ok"] is True and rep["agent_e2e"]["ok"] is True, (sm, rep.get("agent_e2e"))
cm = ev["complementarity"]
assert cm["three_way_complete"] is True and "26.94" in cm["direct_source"], cm["direct_source"]
ng = cm["net_gain"]
assert ng["direct_tokens_per_query"] and ng["displaced_direct_queries_equiv"] is not None, ng
print(f"1 commit run PASS in {r['took']:.0f}s: rung 0 up, smoke + rescue probe + e2e OK, eval written, net_gain {ng}")

# 2. GPU run 3's bug must fail fast
r = run_case(inject_bug=True)
assert r["ev"] is None and not r["rep"].get("chosen") and r["took"] < 120, r["took"]
print(f"2 negative control PASS: run 3's bug fails in {r['took']:.0f}s")

# 3. e2e NO-GO in the commit run -> no evaluation
r = run_case(e2e_fail=True)
assert r["rep"].get("agent_gate") == "AGENT_E2E_NO_GO" and r["calls"]["run_eval"] == 0 and r["ev"] is None, r["calls"]
print("3 e2e NO-GO (commit) PASS: evaluation skipped, gate recorded")

# 4. e2e NO-GO in the hidden run -> agent never called, DIRECT on every task, valid submission, rescue log
r = run_case(e2e_fail=True, rerun=True)
assert r["calls"]["solve_repl_submit"] == 0, r["calls"]
sub = json.loads((r["root"] / "submission.json").read_text())
ch = json.loads((DATA / "arc-agi_test_challenges.json").read_text())
assert len(sub) == 3 and all(len(rows) == len(ch[t]["test"]) for t, rows in sub.items())
log = (r["root"] / "rescue_attempts.jsonl").read_text().splitlines()
assert len(log) == sum(len(ch[t]["test"]) for t in sub), (len(log), list(sub))
assert r["rep"]["agent_config"]["agent_disabled"] is True
print(f"4 e2e NO-GO (hidden run) PASS: 0 agent calls, DIRECT answered every test input, {len(log)} rescue log lines")

# 5. rejected rescue probe + recorded EVAL answers -> comparison still complete
r = run_case(rescue_probe_fail=True)
assert r["ev"] is not None and r["ev"]["complementarity"]["three_way_complete"] is True, r["ev"] and r["ev"]["complementarity"]
assert not r["rep"].get("rescue_disabled")
print("5 rescue probe rejected (ref eval) PASS: recorded DIRECT still merged")

# 6. agent thinking fallback must not change the DIRECT rescue (hidden run so the rescue runs live)
r = run_case(thinking_fallback=True, rerun=True)
assert r["rep"]["agent_config"]["thinking"] is False, r["rep"]["agent_config"]
live = r["calls"]["direct_clients"]
assert live and all(d == {"thinking": True, "budget": 49152} for d in live), live
print(f"6 thinking fallback PASS: agent thinking off, DIRECT clients {live[-1]}")
print("GPU PATH LOCAL TEST PASS")
