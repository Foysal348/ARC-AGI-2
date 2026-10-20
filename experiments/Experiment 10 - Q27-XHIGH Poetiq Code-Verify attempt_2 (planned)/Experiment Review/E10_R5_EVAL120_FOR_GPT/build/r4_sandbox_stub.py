def poetiq_sandbox_run(code, grid, timeout_s=5.0):   # HARNESS-ONLY plain runner (Windows has no setuid/seccomp)
    import subprocess, json as _j, sys as _s
    runner = ("import sys,json,numpy as np\n"
              "raw=json.load(sys.stdin)\n"
              "ns={}\n"
              "exec(raw['code'],ns)\n"
              "r=ns['transform'](np.array(raw['input'],dtype=int))\n"
              "r=r.tolist() if hasattr(r,'tolist') else r\n"
              "print('__E10_PROGRAM_RESULT__'+json.dumps({'ok':True,'result':r}))\n")
    cp = subprocess.run([_s.executable, "-c", runner], input=_j.dumps({"code": code, "input": grid}),
                        capture_output=True, text=True, timeout=timeout_s)
    p = cp.stdout.rfind("__E10_PROGRAM_RESULT__")
    if cp.returncode or p < 0:
        return False, (cp.stderr or cp.stdout)[-600:]
    return True, _j.dumps(_j.loads(cp.stdout[p + len("__E10_PROGRAM_RESULT__"):])["result"])
E10_SANDBOX_SELFTEST = {"ok": True, "mode": "HARNESS STUB"}
