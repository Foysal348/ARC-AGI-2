"""E14 CPU probe driver (2026-10-23): real arc2_duck BranchSessions (rev 5 + rev 6 paths, Duck's own loop, trimmer,
openers, yields) against fake_prefix_server.py, first with E12_RAW_REASONING_HISTORY=0 (as run on Kaggle), then =1 (the
E14-P fix). Prints per mode how every follow-up request's rendered prompt relates to the previous prompt + generated reply.
usage: python run_prefix_probe.py <port>"""
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30999
E12 = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/duck_arc2")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
sys.path[:0] = [str(E12), str(E12 / "vendor"), str(FIX)]
os.environ.update({"E12_ARC2_REV5": "1", "E12_ARC2_REV6": "1", "E12_TOKENIZER_JSON": str(FIX / "tok_int4" / "tokenizer.json"),
                   "E12_DUCK_TEMPERATURE": "1.0"})

if os.name == "nt":      # the Windows shims of the E12 harness (Duck passes a 65 KB bootstrap with -c; no killpg)
    _real = subprocess.Popen
    _bdir = Path(tempfile.gettempdir()) / "e14_probe_bootstrap"
    _bdir.mkdir(exist_ok=True)

    class _Shim(_real):
        def __init__(self, args, *a, **kw):
            if isinstance(args, list) and len(args) >= 5 and args[1:4] == ["-I", "-S", "-c"]:
                f = _bdir / "sandbox_bootstrap.py"
                if not f.exists() or f.read_text(encoding="utf-8") != args[4]:
                    f.write_text(args[4], encoding="utf-8")
                args = [args[0], "-I", "-S", str(f)] + args[5:]
            super().__init__(args, *a, **kw)
    subprocess.Popen = _Shim
    if not hasattr(os, "killpg"):
        import signal as _sig
        os.killpg = lambda pid, sig: os.kill(pid, 9)
        if not hasattr(_sig, "SIGKILL"):
            _sig.SIGKILL = 9

import arc2_duck as A  # noqa: E402
import e12_fixtures as B  # noqa: E402

BASE = f"http://127.0.0.1:{PORT}/v1"
os.environ.update(A.duck_env(BASE, "fake-int4", 4))
A._import_duck()
A.install_sandbox_memory_limit(4096)
CH, SOL, multi, single = B.load()
TASKS = (multi[:1] + single[:1])


for _ in range(600):                      # the server imports transformers first (slow on this machine)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
        break
    except Exception:
        time.sleep(1)
else:
    sys.exit("fake prefix server did not come up")


def probe_log():
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/probe_log", timeout=10).read())


def run_mode(raw: str):
    os.environ["E12_RAW_REASONING_HISTORY"] = raw
    agent_cls = A.make_agent_class()
    out = Path(tempfile.mkdtemp(prefix=f"e14probe_raw{raw}_"))
    n0 = len(probe_log())
    counters = []
    for tid in TASKS:
        for b, view in ((0, "facts"), (1, "grids")):
            s = A.BranchSession(agent_cls, task_id=tid, task=CH[tid], branch=b, view=view, queue_position=b * 100,
                                out_dir=out, deadline=time.monotonic() + 600, base_url=BASE, model_id="fake-int4",
                                save_request_logs=False, max_generated_tokens=7200)
            s.run()
            counters.append((tid, b, dict(getattr(s.agent, "arc2_raw_reasoning", {})),
                             len(s.agent.arc2_usage), s.agent.arc2_rev6_events.get("trim_reasoning_only_blocks", 0)))
    return probe_log()[n0:], counters


for raw in ("0", "1"):
    log, counters = run_mode(raw)
    print(f"\n===== E12_RAW_REASONING_HISTORY={raw}: {len(log)} requests")
    for c in counters:
        print("  branch", c[0], "b", c[1], "requests", c[3], "raw_reasoning counters", c[2], "trims", c[4])
    errs = [r for r in log if "error" in r]
    if errs:
        print("  SERVER ERRORS:", errs[:3])
    tab = collections.Counter((r.get("cls"), r.get("reply_prev")) for r in log if r.get("cls"))
    follow = [r for r in log if r.get("cls")]
    print("  follow-up requests by class:", dict(collections.Counter(r["cls"] for r in follow)))
    for cls in ("in_reply", "before_reply"):
        ex = [r for r in follow if r["cls"] == cls]
        if ex:
            r = ex[0]
            print(f"  first {cls}: conv {r['conv']} i {r['i']}",
                  {k: r.get(k) for k in ("offset_in_reply", "in_reasoning", "chars_before_prev_end", "at_message",
                                         "expected", "got")})
    print("  detail:", [(r["conv"], r["i"], r["cls"], r.get("offset_in_reply", r.get("chars_before_prev_end",
                                                                                    r.get("lcp_minus_prev_prompt"))))
                        for r in follow])
