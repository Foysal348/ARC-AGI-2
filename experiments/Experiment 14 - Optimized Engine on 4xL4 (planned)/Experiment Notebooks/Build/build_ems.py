"""EMS = Engine Matrix on SGLang (user 2026-10-25: "the best engine config, 600-800 tok/s"; one notebook per engine, every
config of that engine in it; a separate vLLM 0.32.0 notebook follows). Engine only: no solver.

Base cells (read only): the E15P rev 2 notebook (56a22f7f, the Kaggle-CPU-kit-verified production engine: SGLang 0.5.21,
RedHat INT4, DFlash2 width 6, delayer off, mrr 17, graphs 1..17, /dev/shm staging, 16 / 12 / W1 launch ladder):
  CELL 0 mode, CELL 1 config (+ the EMS profile), CELL 1b E14 facts / staging / modules (e14_measure = E14M),
  CELL 2 install, the task-order cell (TASK_IDS = the 12 G4B tasks, used only for real ARC prompts), the engine cell.
Dropped: the Poetiq donor / scheduler / report and the vLLM fallback (nothing to measure without SGLang). A one-line
cell replaces the donor's sandbox self-test (the engine cell requires the flag).
New: CELL EMS (the arms) and CELL EMS-R (report + bundle). The arm helpers are the E15 engine notebook's (build_e14.py
--e15: e14_launch / e14_phase / gate / windows), extended with 2 x TP2.

Arms (each its own launch; the correctness gate first, then 8k and 40k steady windows at the level the server really
runs; a 1,024-token greedy probe vs the reference where the weights are the same):
  ref (production) + its A/A | target_only (diagnostic) | MTP = EAGLE 3/1/4 (the 65.97 team's algorithm, INT4) |
  NTN-FP8 = NguyenThanhNhan's exact flags (65.97, discussion 745787) on our FP8 checkpoint | NTN-INT4 = the same flags
  on INT4 | DFlash width 6 at 20 and 24 streams | the best MTP at 20 | MTP 2/1/3, mem 0.86, ReplaySSM fp16 / fp32 |
  DFlash width 4 / 8 | LAST: 2 x TP2 (GPU pairs from `nvidia-smi topo -m`, DFlash W6, then the best MTP if time) with a
  reserved time slot so the earlier arms can never eat it.
rev 2 (GPT pre-GPU serving audit 2026-10-25, each claim checked in the SGLang 0.5.21 source; Notes in Experiment 14
"Bug ang fix"): target-only = the identity oracle (gate reference of every later arm; the DFlash reference judged against
it from its own rows); a gate FAIL is recorded but the speed is still measured; the two linear-attention backend arms
dropped (FlashInfer GDN dies on SM 8.9, CuTe DSL changes no kernel of a speculative run); a CAPPED flag when the server
ran fewer streams than asked; 2 x TP2 with GPU pairs from the topology, per-replica cache dirs (no driver cache) and the
custom all-reduce state recorded.
rev 2b (4-agent audit 2026-10-26): own gate (oracle verdict + a verdict vs the DFlash reference rows); dirty-server flags
from every A/A and probe pass; one oracle flag (target-only only when its gate A/A AND probe A/A are complete); `_gated`
needs complete probe evidence, a live server, both bands, >= 8 decode lines and no cap; MTP family tag (ntn_int4 joins
the best-MTP pick); slots-aware stream sizing (extra_buffer = 5); NTN mamba 85 = 5 x 17 so 16 streams fit; ntn_fp8 30-min
launch + FP8 page-cache prefetch; reference name from the real launch flags, fallback-aware skips; 2 x TP2 gated on both
replicas, ok only with all four windows, KV-for-production flag, custom all-reduce off (P2P 5.5x slower here), 30-min
wall, start loop guarded; the whole cell in try / except so the report cell always runs.
rev 2c (deep re-read of the GPT audit, its FINAL priority 1): SGLang PR #45151 (issue #44145, reproduced on 0.5.21: the
MTP draft's duplicate BF16 embed / lm_head, ~1.19 GiB per GPU at TP4, is resident when the target KV pool is sized) as an
env-gated patch of the installed eagle_worker_v2.py + the arm mtp_3_1_4_pr45151; #43029 not applicable (grammar only);
`nvidia-smi topo -p2p p` recorded and NCCL_DEBUG=INFO on the TP2 replicas (Gate D).
rev 2d (user 2026-10-26, "take 1 and 2"): an XL band (~85k-token prompts, at the streams the KV holds) on every
single-server arm; FlashInfer 0.7 PCIe-IPC all-reduce arms (dflash_w6_pcie, mtp_3_1_4_pcie) through a per-arm FlashInfer
0.7.0.post1 overlay from the vLLM 0.32.0 wheelhouse (attach foysalemonshanto/vllm-0-32-0-cu13-wheelhouse); budget 260 min.
rev 2e (independent GPT audit of rev 2c, 2026-10-26): adoptable only against the target-only oracle (fail closed);
a post-divergence tail check (a near-tie first divergence followed by degenerate / repeated output fails the gate);
the #45151 arm skipped unless the patch really applied; an explicit adoptable column; ntn_fp8 (a cold FP8 load from
/kaggle/input) moved after the 20 / 24-stream arms.
rev 2f (user 2026-10-26): two notebooks, SAFE and RISKY (a crash in one cannot touch the other); EMS_PART=ALL runs all
(CPU harness); every arm call in its own try/except.
usage: python build_ems.py   -> ../Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb + ../Q27-XHIGH-EMS_RISKY_ENGINE_MATRIX_SGLANG.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
E15 = NBDIR.parents[1] / "Experiment 15 - Duck 2x35k independent attempts (planned)" / "Experiment Notebooks"
SRC = E15 / "Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"
SRC_SHA = "56a22f7f"
OUT = NBDIR / "Q27-XHIGH-EMS_ENGINE_MATRIX_SGLANG.ipynb"
PUSHED: set = set()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


assert sha(SRC).startswith(SRC_SHA), sha(SRC)
S = ["".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
assert len(S) == 11 and "E15P: the E15B engine cell" in S[6] and "CELL 1b" in S[2] and "task-order" in S[4]

G4B = ["136b0064", "898e7135", "135a2760", "b5ca7ac4", "4c3d4a41", "62593bfd", "247ef758", "bf45cf4b", "f931b4a8",
       "65b59efc", "9385bd28", "cbebaa4b"]
c1 = rep(S[1], 'os.environ.get("Q27X_PROFILE", "E15PEVAL120")', 'os.environ.get("Q27X_PROFILE", "EMS")')
c1 = rep(c1, '''    "E14SEVAL120G4B": {''', '''    # EMS (user 2026-10-25): the SGLang engine-config matrix; the 12 G4B tasks only feed real ARC prompts; 200 min
    # rev 2d: 260 min (user 2026-10-26: the budget may grow; the XL band costs ~2-3 min per arm)
    "EMS":       {"source": "eval", "task_list": ''' + json.dumps(G4B) + ''',
                  "time_budget_sec": 260 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
                  "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "ems": True},
    "E14SEVAL120G4B": {''')
c1 += '''

# ---- EMS (user 2026-10-25): the engine-config matrix. The reference = the Duck production engine at 16 streams
# (KV per request 50k, as every E15B run); the rerun of this notebook only writes the cell-0 fallback.
EMS_ON = bool(P.get("ems")) and not RERUN
if EMS_ON:
    SGL_KV_PER_REQUEST = 50_000
    CONFIG["e15p"]["kv_per_request"] = SGL_KV_PER_REQUEST
CONFIG["ems"] = {"on": EMS_ON, "kv_per_request": SGL_KV_PER_REQUEST}
print("EMS:", json.dumps(CONFIG["ems"]))
'''
c1b = rep(S[2], '"variant": "E15P"', '"variant": "EMS"')
stub = '''# ==================================================================== EMS: no solver in this notebook
# The engine cell starts SGLang only after the solver's sandbox self-test; the engine matrix runs no generated code.
E10_SANDBOX_SELFTEST = {"ok": True, "mode": "EMS engine matrix: no solver, no sandbox"}
'''
eng = rep(S[6], "# E15P: the E15B engine cell (Poetiq uses /generate through the shim;",
          "# EMS: the production engine cell, unchanged (it starts the reference arm). E15P: the E15B engine cell (Poetiq uses "
          "/generate through the shim;")

ARMS = r'''# ==================================================================== CELL EMS — the SGLang engine-config matrix
# Each arm = its own launch of the production command with that arm's flag changes; the correctness gate first (greedy
# rows vs the reference's own A/A, flips decide), then an 8k and a 40k steady window at the level the server really runs
# (decode-line tok/s, accept length, step time), KV / mamba from the log, retract / OOM signs. Arms that change the
# weights (FP8) have no greedy parity with INT4: their gate is diagnostic. Every phase has its own deadline; whatever
# does not fit is skipped and recorded, never squeezed. 2 x TP2 runs LAST from a reserved time slot.
import re, shlex, shutil, statistics, subprocess, threading


def e14_edit_flags(flags, drop=(), set_=None, add=()):
    """drop: flag names removed with their value(s); set_: {flag: value | [values] | None (bare flag)}; add: extra args."""
    out, i, set_ = [], 0, dict(set_ or {})
    names = set(drop) | set(set_)
    while i < len(flags):
        f = flags[i]
        if f in names:
            i += 1
            while i < len(flags) and not str(flags[i]).startswith("--"):
                i += 1
            continue
        out.append(f)
        i += 1
    for k, v in set_.items():
        out += [k] + ([str(x) for x in v] if isinstance(v, (list, tuple)) else ([str(v)] if v is not None else []))
    return out + list(add)


E14_ARM = {"name": "ref", "set": {}, "drop": (), "env": {}}
E14_RESERVE = 0.0               # seconds kept for the 2 x TP2 slot (set below)
E14_DIRTY = False
E14_LOG_SIGNS = ("runs out of memory", "Not enough space", "Retract", "out of memory", "Traceback", "Exception")


def e14_launch(name, set_=None, drop=(), env=None, force=False, timeout=None):
    """Kill the current server, launch the production command + this arm's changes, wait, record caps."""
    global E11_SERVER, E14_ARM, E14_DIRTY
    rec = {"arm": name, "set": set_ or {}, "drop": list(drop), "env": env or {}, "force": force}
    if not E14M.launch_allowed(time.perf_counter(), HARD_STOP, reserve=0.0 if force else E14_RESERVE, force=force):
        rec["skipped"] = "NOT_TESTED_BUDGET: " + ("would eat the 2 x TP2 slot" if (not force and E14_RESERVE) else "less than ~17 min before the hard stop")
        E14_RESULTS.setdefault("launches", []).append(rec)
        e14_save()
        print(f"EMS launch {name}: skipped ({rec['skipped']})", flush=True)
        return False, rec
    E14_STATE.enter("LAUNCH:" + name, deadline=HARD_STOP)
    t0 = time.perf_counter()
    try:
        e11_kill_server(wait=True)
    except Exception:
        pass
    rec["gpu_free"] = _e11_gpu_free()
    flags = e14_edit_flags(e11_flags(E14_VENV["cfg"]), drop=drop, set_=set_)
    envx = dict(E14_VENV["env"])
    envx.update(env or {})
    srv = E11Server(E14_VENV["launch"] + flags, envx, "ems_" + name)
    srv.start()
    try:          # rev 2g: a crash while this server starts never leaves it holding the GPUs
        _room = HARD_STOP - time.perf_counter() - (10 * 60 if force else E14_RESERVE + 8 * 60)
        ok, why = srv.wait_ready(min(timeout or SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, _room)))
    except BaseException:
        try:
            srv.kill()
        except Exception:
            pass
        raise
    rec.update(ok=ok, why=why, sec=round(time.perf_counter() - t0, 1), flags=flags)
    if ok:
        E11_SERVER = srv
        rec["caps"] = e11_caps(srv.log_text())
        E14_ARM = {"name": name, "set": set_ or {}, "drop": tuple(drop), "env": env or {}}
        E14_DIRTY = False
    else:
        rec["log_tail"] = srv.log_text()[-3000:]
        srv.kill()
        E11_SERVER = None
    E14_RESULTS.setdefault("launches", []).append(rec)
    e14_save()
    print(f"EMS launch {name}: {why} ({rec['sec']} s)", flush=True)
    return ok, rec


def _e14_dirty(obj):
    if isinstance(obj, dict):
        if obj.get("server_idle") is False or obj.get("server_not_idle") is True:
            return True
        return any(_e14_dirty(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_e14_dirty(v) for v in obj[:20])
    return False


def e14_phase(name, fn, cap_sec, ctx=None):
    """One phase with its own deadline: min(cap, time left - 8 min - the reserved slot); skipped when < 2 min left."""
    global E14_DIRTY
    ctx = ctx or E14_CTX
    cap = E14M.phase_budget(time.perf_counter(), HARD_STOP, cap_sec, reserve=E14_RESERVE, margin=8 * 60)
    if cap is not None and E14_DIRTY and E11_SERVER is not None:
        e14_launch(E14_ARM["name"].split("_clean")[0] + "_clean", set_=E14_ARM["set"], drop=E14_ARM["drop"], env=E14_ARM["env"])
        cap = E14M.phase_budget(time.perf_counter(), HARD_STOP, cap_sec, reserve=E14_RESERVE, margin=8 * 60)
    if cap is None or E14_DIRTY:
        E14_RESULTS["phases"][name] = {"skipped": "dirty server" if E14_DIRTY else "reserved slot / hard stop",
                                       "left_sec": round(HARD_STOP - time.perf_counter())}
        e14_save()
        print(f"EMS phase {name}: skipped {E14_RESULTS['phases'][name]}", flush=True)
        return None
    ctx.deadline = time.perf_counter() + cap
    E14_STATE.enter("PHASE:" + name, deadline=ctx.deadline, arm=E14_ARM["name"])
    t0 = time.perf_counter()
    _srv = E11_SERVER
    _off = len(_srv.log_text()) if _srv is not None else 0
    try:
        res = fn()
    except BaseException as _e:
        res = {"error": f"{type(_e).__name__}: {_e}"[:500], "traceback": traceback.format_exc()[-1500:]}
    if isinstance(res, dict):
        res.setdefault("arm", E14_ARM["name"])
        res["phase_sec"] = round(time.perf_counter() - t0, 1)
        if _srv is not None:
            _new = _srv.log_text()[_off:]
            res["server_log_signs"] = {s: _new.count(s) for s in E14_LOG_SIGNS if s in _new}
        if _e14_dirty(res):
            E14_DIRTY = True
            res["server_not_idle_after"] = True
    E14_RESULTS["phases"][name] = res
    e14_save()
    print(f"EMS phase {name}: {json.dumps(res, default=str)[:900]}", flush=True)
    return res


def _e14_model_id(url=None):
    try:
        return json.loads(_urlreq.urlopen((url or SGL_URL) + "/v1/models", timeout=30).read())["data"][0]["id"]
    except Exception:
        return str(SGL_MODEL_PATH)


def _e14_rows(g):
    return "\n".join(" ".join(str(v) for v in r) for r in g)


def _e14_task_text(t):
    s = "".join(f"\nExample {i + 1}\nInput:\n{_e14_rows(p['input'])}\nOutput:\n{_e14_rows(p['output'])}\n"
                for i, p in enumerate(t["train"]))
    return s + "".join(f"\nTest input {j + 1}:\n{_e14_rows(q['input'])}\n" for j, q in enumerate(t["test"]))


def _e14_chat_ids(messages, think=True):
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think,
                                        reasoning_effort=REASONING_EFFORT)
    return tokenizer.encode(txt, add_special_tokens=False)


def ems_prompts():
    """short: one real task per stream (~6k context); long: a shared ~36k-token block of other eval tasks + the task."""
    ask = "Find the transformation rule of this ARC-AGI-2 task and describe it, then give the test output grids."
    short = [_e14_chat_ids([{"role": "user", "content": ask + "\n" + _e14_task_text(CHALLENGES[t])}]) for t in TASK_IDS]
    try:
        _all = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    except Exception:
        _all = dict(CHALLENGES)
    bg, n = [], 0
    for t in sorted(_all):
        if t in CHALLENGES:
            continue
        txt = _e14_task_text(_all[t])
        bg.append(txt)
        n += len(tokenizer.encode(txt, add_special_tokens=False))
        if n >= int(os.environ.get("E14_LONG_TOKENS", 36000)):
            break
    block = "Reference tasks (not to be solved):\n" + "\n".join(bg)
    long_ = [_e14_chat_ids([{"role": "system", "content": block},
                            {"role": "user", "content": ask + "\n" + _e14_task_text(CHALLENGES[t])}]) for t in TASK_IDS]
    # rev 2d: the XL band, a shared ~85k-token block (production thinks to 60k on top of a 5-10k prompt, so decode spends
    # most of its time at 60-100k context; the 40k band never reached it)
    bgx, nx = list(bg), n
    for t in sorted(_all):
        if nx >= int(os.environ.get("E14_XL_TOKENS", 85000)):
            break
        if t in CHALLENGES or _e14_task_text(_all[t]) in bgx:
            continue
        bgx.append(_e14_task_text(_all[t]))
        nx += len(tokenizer.encode(bgx[-1], add_special_tokens=False))
    blockx = "Reference tasks (not to be solved):\n" + "\n".join(bgx)
    xl = [_e14_chat_ids([{"role": "system", "content": blockx},
                         {"role": "user", "content": ask + "\n" + _e14_task_text(CHALLENGES[t])}]) for t in TASK_IDS]
    return {"short": short, "long": long_, "xl": xl,
            "tokens": {"short_p50": statistics.median(len(x) for x in short) if short else None,
                       "long_p50": statistics.median(len(x) for x in long_) if long_ else None,
                       "xl_p50": statistics.median(len(x) for x in xl) if xl else None}}


EMSR = {"arms": {}, "launches": {}, "order": []}
E14_RESULTS["ems"] = EMSR
# rev 2f: which part of the matrix this notebook runs (baked in at build time; EMS_PART=ALL = every arm, CPU harness)
EMS_PART = os.environ.get("EMS_PART") or "__EMS_PART__"
EMS_PART_ARMS = {"SAFE": ("mtp_3_1_4", "dflash_w6_s20", "dflash_w6_s24", "mtp_2_1_3", "mtp_3_1_4_mem86", "dflash_w4", "dflash_w8"),
                 "RISKY": ("dflash_w6_pcie", "mtp_3_1_4", "mtp_3_1_4_pr45151", "mtp_3_1_4_pcie", "ntn_int4", "ntn_fp8",
                           "mtp_replay_fp16", "mtp_replay_fp32")}
assert EMS_PART in ("ALL", "SAFE", "RISKY"), EMS_PART      # rev 2g: a typo must not silently skip every arm
EMS_TP2_ON = EMS_PART in ("ALL", "RISKY")
EMSR["part"] = EMS_PART


class _E14GenWindow(E14M.Gen):
    """rev 2g-c (agent 3 A1): window streams (max_new >= 16384) ignore EOS, so a stream that ends early never lowers the
    running share (a false reject of the fastest arms); greedy / probe / canary rows (<= 1,024 tokens) keep EOS."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.max_new >= 16384:
            self.sp["ignore_eos"] = True


if not getattr(E14M.Gen, "_ems_window", False):
    _E14GenWindow._ems_window = True
    E14M.Gen = _E14GenWindow
try:
    if EMS_ON and ENGINE_READY:
        E14_CTX = E14M.Ctx(SGL_URL, lambda: E11_SERVER.log_text() if E11_SERVER is not None else "", E14_OUT,
                           model_id=_e14_model_id(), deadline=HARD_STOP - 8 * 60, log=print, encode=None,
                           normalize=lambda x: x)
        E14_CTX.scale = E14_TS
        try:
            EMS_PROMPTS = ems_prompts()
        except Exception as _e:
            EMS_PROMPTS = {"short": [], "long": [], "xl": [], "tokens": {"error": repr(_e)[:300]}}
        EMSR["prompts"] = EMS_PROMPTS["tokens"]
        _S, _L, _X = EMS_PROMPTS["short"], EMS_PROMPTS["long"], EMS_PROMPTS.get("xl") or []
        if not _S or not _L:       # rev 2b (agent 1 F12): every window would divide by zero
            raise RuntimeError(f"EMS: no prompts ({EMSR['prompts']}); nothing can be measured")
        # rev 2b (agent 1 F8, agent 2 F7): the reference name / width / streams from the flags the server really got
        _RF = [str(x) for x in (E11_SGL.get("launch_flags") or e11_flags(E14_VENV["cfg"]))]
        _RW = int(_RF[len(_RF) - 1 - _RF[::-1].index("--speculative-num-draft-tokens") + 1]) if "--speculative-num-draft-tokens" in _RF else 0
        REF_NAME = f"ref_dflash_w{_RW}_s{SGL_MAX_RUNNING}"
        EMSR["ref_name"], EMSR["base_cfg"] = REF_NAME, dict(E14_VENV["cfg"])

        def ems_patch_pr45151():
            """SGLang PR #45151 (open; fixes #44145, reproduced on v0.5.21), env-gated: with EMS_PR45151=1 the EAGLE/MTP draft
            binds the target's embed_tokens / lm_head in EagleDraftWorker.__init__, before the scheduler sizes the target KV
            pool; alloc_memory_pool then skips the second binding. Without the env var the file behaves as shipped."""
            out = {"applied": False}
            try:
                if E11_HARNESS:
                    out["why"] = "harness: no SGLang install"
                    return out
                fp = subprocess.run([E14_VENV["launch"][0], "-c", "import sglang.srt.speculative.eagle_worker_v2 as m; print(m.__file__)"],
                                    capture_output=True, text=True, timeout=300, env=E14_VENV["env"]).stdout.strip().splitlines()[-1]
                src = Path(fp).read_text(encoding="utf-8")
                a = "        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)\n\n    def alloc_memory_pool(\n"
                b = "        self.init_token_map()\n        self.init_lm_head()\n\n        if get_spec().speculative_use_rejection_sampling:"
                _m1 = 'self._ems_shared = _ems_os.environ.get("EMS_PR45151") == "1"'
                _m2 = 'if not getattr(self, "_ems_shared", False):'
                if _m1 in src and _m2 in src:
                    out.update(applied=True, why="already patched (both snippets present)", file=fp,
                               sha256=hashlib.sha256(src.encode("utf-8")).hexdigest())
                    return out
                if "EMS_PR45151" in src:
                    out["why"] = "partial patch found (one snippet only): not used"
                    return out
                if src.count(a) != 1 or src.count(b) != 1:
                    out["why"] = f"anchors not unique / missing ({src.count(a)}, {src.count(b)})"
                    return out
                src = src.replace(a, "        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)\n"
                                     "        # EMS: SGLang PR #45151 (issue #44145), env-gated\n"
                                     "        import os as _ems_os\n"
                                     "        self._ems_shared = _ems_os.environ.get(\"EMS_PR45151\") == \"1\"\n"
                                     "        if self._ems_shared:\n"
                                     "            self.init_token_map()\n"
                                     "            self.init_lm_head()\n"
                                     "            torch.cuda.empty_cache()\n"
                                     "\n    def alloc_memory_pool(\n")
                src = src.replace(b, "        if not getattr(self, \"_ems_shared\", False):\n"
                                     "            self.init_token_map()\n            self.init_lm_head()\n\n"
                                     "        if get_spec().speculative_use_rejection_sampling:")
                compile(src, fp, "exec")
                assert src.count('self._ems_shared = _ems_os.environ.get("EMS_PR45151") == "1"') == 1
                assert src.count('if not getattr(self, "_ems_shared", False):') == 1
                import tempfile as _tf
                shutil.copy2(fp, fp + ".ems_orig")             # the shipped file, kept next to it
                _fd, _tmp = _tf.mkstemp(prefix=Path(fp).name + ".", suffix=".tmp", dir=str(Path(fp).parent))
                with os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                    _fh.write(src)
                    _fh.flush()
                    os.fsync(_fh.fileno())
                os.replace(_tmp, fp)                            # atomic: a killed process never leaves half a module
                assert Path(fp).read_text(encoding="utf-8") == src
                for pyc in Path(fp).parent.glob("__pycache__/eagle_worker_v2*.pyc"):
                    pyc.unlink()
                out.update(applied=True, file=fp, sha256=hashlib.sha256(src.encode("utf-8")).hexdigest())     # rev 2e: proof
            except Exception as e:
                out["why"] = f"{type(e).__name__}: {e}"[:300]
            return out
        try:          # rev 2g-c (checklist 9): the exact packages and prompts of this run
            EMSR["pip_freeze"] = subprocess.run([E14_VENV["launch"][0], "-m", "pip", "freeze"], capture_output=True, text=True,
                                                timeout=180, env=E14_VENV["env"]).stdout[-20000:]
        except Exception as _e:
            EMSR["pip_freeze"] = repr(_e)[:200]
        EMSR["prompt_sha"] = {k_: [__import__("hashlib").sha256(json.dumps(list(x_)).encode()).hexdigest()[:16] for x_ in v_]
                              for k_, v_ in (globals().get("EMS_PROMPTS") or {}).items() if k_ in ("short", "long", "xl")}
        EMSR["pr45151_patch"] = ems_patch_pr45151() if EMS_PART in ("ALL", "RISKY") else {"applied": False, "why": "SAFE part: not needed"}

        EMS_FI07 = "/tmp/ems-fi07"

        def ems_fi07_overlay():
            """rev 2d: FlashInfer 0.7.0.post1 (+ jit-cache cu130 + nccl-extensions) into an overlay dir for the PCIe-IPC arms
            only (PYTHONPATH). Source: the vLLM 0.32.0 wheelhouse dataset; Kaggle drops '+' from file names, so the jit-cache
            wheel is copied back under its real name. Verified by importing flashinfer.comm.PcieIpcAllReduceWorkspace with the
            SGLang venv's python."""
            out = {"ok": False, "dir": EMS_FI07}
            try:
                if E11_HARNESS:
                    out["why"] = "harness: no SGLang install"
                    return out
                roots = [Path(x) for x in os.environ.get("EMS_FI07_WHEELHOUSE", "").split(os.pathsep) if x] + [
                    Path("/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"), Path("/kaggle/input")]
                want = {"flashinfer_python-0.7.0.post1-": None, "flashinfer_jit_cache-0.7.0.post1": None, "nccl_extensions-0.1.0-": None}
                for r_ in roots:
                    if not r_.is_dir():
                        continue
                    for f in r_.rglob("*.whl"):
                        for k in want:
                            if want[k] is None and f.name.startswith(k):
                                want[k] = f
                    if all(want.values()):
                        break
                if not all(want.values()):
                    out["why"] = "wheels not found (attach foysalemonshanto/vllm-0-32-0-cu13-wheelhouse): " + str({k: str(v) for k, v in want.items()})
                    return out
                tmp = Path("/tmp/ems-fi07-wheels")
                tmp.mkdir(parents=True, exist_ok=True)
                wheels = []
                for f in want.values():
                    name = f.name.replace("0.7.0.post1cu130", "0.7.0.post1+cu130")
                    shutil.copy(f, tmp / name)
                    wheels.append(str(tmp / name))
                r_ = subprocess.run([E14_VENV["launch"][0], "-m", "pip", "install", "--no-deps", "--no-index", "--target", EMS_FI07,
                                     "--upgrade"] + wheels, capture_output=True, text=True, timeout=1200)
                out["pip_rc"], out["pip_tail"] = r_.returncode, (r_.stdout + r_.stderr)[-600:]
                if r_.returncode != 0:
                    return out
                chk = subprocess.run([E14_VENV["launch"][0], "-c", "import flashinfer, sys; from flashinfer.comm import "
                                      "PcieIpcAllReduceWorkspace; print(flashinfer.__version__, flashinfer.__file__)"],
                                     capture_output=True, text=True, timeout=600, env=dict(E14_VENV["env"], PYTHONPATH=EMS_FI07))
                out["import"] = (chk.stdout + chk.stderr).strip()[-400:]
                out["ok"] = chk.returncode == 0 and "0.7.0" in chk.stdout and EMS_FI07 in chk.stdout
            except Exception as e:
                out["why"] = f"{type(e).__name__}: {e}"[:300]
            return out
        EMSR["fi07_overlay"] = ems_fi07_overlay() if EMS_PART in ("ALL", "RISKY") else {"ok": False, "why": "SAFE part: not needed"}
        print("EMS FlashInfer 0.7 overlay:", json.dumps(EMSR["fi07_overlay"])[:500], flush=True)
        # the env of a PCIe-IPC arm: the overlay first on the path, the SGLang 0.5.21 switch on (pcie_ipc_ar.py; it sits after
        # custom all-reduce, which is off on 4 PCIe GPUs, and before NCCL; bf16 decode reductions only)
        _PCIE = {"PYTHONPATH": EMS_FI07, "SGLANG_ENABLE_PCIE_IPC_ALLREDUCE": "1", "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
        print("EMS PR #45151 patch:", EMSR["pr45151_patch"], flush=True)
        try:      # rev 2b (agent 2 F4): FP8 is not staged; read it into the page cache while the INT4 arms run (best effort)
            _fp8 = sorted(str(x) for x in Path(MODEL_PATH).glob("*.safetensors"))
            if _fp8 and not E11_HARNESS and EMS_PART in ("ALL", "RISKY"):      # rev 2g: SAFE has no FP8 arm
                subprocess.Popen(["bash", "-c", "cat " + " ".join(shlex.quote(x) for x in _fp8) + " > /dev/null"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            EMSR["fp8_prefetch"] = len(_fp8)
        except Exception as _e:
            EMSR["fp8_prefetch"] = repr(_e)[:200]
        _SPEC = ("--speculative-algorithm", "--speculative-draft-model-path", "--speculative-draft-model-quantization",
                 "--speculative-num-draft-tokens", "--speculative-num-steps", "--speculative-eagle-topk")
        LEVEL = SGL_MAX_RUNNING                       # 16 (12 if the production launch fell back)
        EMS_REF = None
        EMS_PROBE_REF = None

        def _mtp(steps=3, draft=4, path=None, extra=None):
            d = {"--speculative-algorithm": "EAGLE", "--speculative-draft-model-path": str(path or SGL_MODEL_PATH),
                 "--speculative-num-steps": steps, "--speculative-eagle-topk": 1, "--speculative-num-draft-tokens": draft}
            d.update(extra or {})
            return d

        def _slots(s):            # kv_cache_configurator.py:2232-2261, overlap on: extra_buffer 5, extra_buffer_lazy 4
            return 5 if str((s or {}).get("--mamba-radix-cache-strategy", E14_VENV["cfg"].get("strategy"))) == "extra_buffer" else 4

        def _streams(n, slots=4):  # n client streams: server cap n + 1, graphs 1..n+1, `slots` mamba slots per request
            return {"--max-running-requests": n + 1, "--cuda-graph-bs-decode": list(range(1, n + 2)),
                    "--max-mamba-cache-size": slots * (n + 1)}

        EMS_PROBE_TGT = None
        EMS_TGT = None
        EMS_TGT_XL = None       # rev 2e: the target-only oracle's own XL (~85k) greedy A/A

        def _canary(ctx, level, tag):
            """rev 2e (GPT audit MEDIUM-9, SGLang #36548 cross-request bleed): `level` concurrent greedy requests, each asked
            to repeat its own random sentinel (thinking off); a sentinel in ANOTHER request's output is a leak."""
            import random as _rnd
            rng = _rnd.Random(SEED + level)
            sent = ["Z%04d-%s" % (i, "".join(rng.choice("QWXJVK") for _ in range(6))) for i in range(level)]
            ids = [_e14_chat_ids([{"role": "user", "content": f"Repeat this code exactly once and nothing else: {s_}"}], think=False)
                   for s_ in sent]
            rows = E14M.parity_pass(ctx, ids, max_new=32, concurrent=True, tag=tag + "_canary")
            texts = [tokenizer.decode(r_.get("ids") or [], skip_special_tokens=True) for r_ in rows]
            c_ = {"level": level, "own_hits": sum(1 for s_, x in zip(sent, texts) if s_ in x),
                  "foreign_leaks": sum(1 for i, x in enumerate(texts) for j, s_ in enumerate(sent) if j != i and s_ in x),
                  "errors": sum(1 for r_ in rows if r_.get("error")) + max(0, level - len(rows)), "server_not_idle": _dirty_rows(rows)}
            # rev 2g (GPT R2F-07): 0 own hits = the detector showed nothing (a paraphrasing model): INCONCLUSIVE, not proof
            c_["informative"] = c_["own_hits"] > 0 and c_["errors"] == 0
            c_["classification"] = ("FAIL" if c_["foreign_leaks"] else "INCOMPLETE" if c_["errors"] else
                                    "INCONCLUSIVE" if not c_["informative"] else "PASS_NO_OBSERVED_LEAK")
            return c_

        def _prefill_hits(text, start):
            """rev 2e-d (GPT audit E9): radix-cache hits of the prefills logged after `start` (incl. the warm request)."""
            n_, new_, cached_ = 0, 0, 0
            for line in text[start:].splitlines():
                if "TP0]" in line and "Prefill batch" in line:
                    m_ = re.search(r"#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)", line)
                    if m_:
                        n_, new_, cached_ = n_ + 1, new_ + int(m_.group(1)), cached_ + int(m_.group(2))
            return {"prefill_lines": n_, "prefill_new_tokens": new_, "prefill_cached_tokens": cached_,
                    "prefix_hit_rate": round(cached_ / (new_ + cached_), 3) if new_ + cached_ else None}

        def _xl_rows(ctx, tag):
            return E14M.parity_pass(ctx, _X[:4], max_new=256, concurrent=True, tag=tag + "_xl") if _X else None

        def _xl_verdict(x, base_a, base_b):
            """rev 2e (GPT audits MEDIUM-7 / E11): ~85k-context greedy rows vs the oracle's XL A/A (flips and tails decide)."""
            if not x or not base_a or not base_b:
                return None
            cx = E14M.compare_passes(base_a, x[:len(base_a)], "xl vs oracle", n=256)
            aa = E14M.compare_passes(base_a, base_b, "oracle xl A/A", n=256)
            if not (_cmp_complete(cx) and _cmp_complete(aa)):
                return "INCOMPLETE"
            ok = cx["large_margin_flips"] <= aa["large_margin_flips"] and _tail_count(x) <= max(_tail_count(base_a), _tail_count(base_b))
            return "PASS" if ok else "FAIL"
        _DFL = None

        def _cmp_complete(c):
            """rev 2b (agent 1 F1): a compare is evidence only when every planned row was compared, none errored and every
            divergence has both margins."""
            return bool(c) and bool(c.get("expected")) and c.get("compared") == c.get("expected") and not c.get("errors") \
                and not c.get("missing_margins")

        def _dirty_rows(*passes):
            return any(r.get("server_not_idle") for p_ in passes for r in (p_ or [])[:1])

        def _aa_rows(tag):
            a = E14M.parity_pass(E14_CTX, _S[:6], max_new=256, tag=tag + "_serial_a")
            b = E14M.parity_pass(E14_CTX, _S[:6], max_new=256, tag=tag + "_serial_b")
            c = E14M.parity_pass(E14_CTX, _S, max_new=256, concurrent=True, tag=tag + "_conc_a")
            c2 = E14M.parity_pass(E14_CTX, _S, max_new=256, concurrent=True, tag=tag + "_conc_b")
            s_aa, c_aa = E14M.compare_passes(a, b, tag + " serial A/A"), E14M.compare_passes(c, c2, tag + " all-at-once A/A")
            dirty = _dirty_rows(a, b, c, c2)     # rev 2b (agent 1 F2): surfaces to e14_phase -> a clean relaunch
            return E14M.gate_reference(a, b, c, c2), {"serial_aa": s_aa, "conc_aa": c_aa, "server_not_idle": dirty,
                                                      "complete": _cmp_complete(s_aa) and _cmp_complete(c_aa) and not dirty}

        def _ref_gate():
            ref, summ = _aa_rows("ems_ref")
            globals()["EMS_REF"] = ref
            EMSR["ref_aa_complete"] = summ["complete"]
            # rev 2e: the production reference gets the same canary and XL rows (judged against the oracle later)
            summ["canary"] = EMSR["canary_ref"] = _canary(E14_CTX, LEVEL, "ems_ref")
            EMSR["_ref_xl_rows"] = _xl_rows(E14_CTX, "ems_ref") if _X else None
            return summ

        def _probe(tag, n=None, ctx=None):
            _ps = [_S[i % len(_S)] for i in range(n or len(_S))] if _S else []
            return E14M.parity_pass(ctx or E14_CTX, _ps, max_new=1024, concurrent=True, tag="probe_" + tag)

        def _probe_vs_ref(rows, name):
            if not EMS_PROBE_REF:
                return {"skipped": "no reference probe"}
            return E14M.compare_passes(EMS_PROBE_REF[0], rows[:len(EMS_PROBE_REF[0])], name)

        def _probe_vs_tgt(rows, name):
            # EMS rev 2 (GPT pre-GPU audit 2026-10-25): the same-checkpoint target-only probe is the identity oracle
            if not EMS_PROBE_TGT:
                return {"skipped": "no target-only probe"}
            return E14M.compare_passes(EMS_PROBE_TGT[0], rows[:len(EMS_PROBE_TGT[0])], name)

        def _gate2(ctx, tag, level=None):
            """rev 2b: E14M.arm_gate's rule (identity=False: flips decide; serial allowance 1, all at once 2) on our own passes,
            so the rows are kept: the verdict vs the gate reference (the oracle once switched) and, for every arm, a second
            verdict vs the DFlash reference rows (agent 1 F5: after the switch a DFlash width / stream arm is otherwise
            indistinguishable from "DFlash differs from plain decoding")."""
            if not EMS_REF or not EMS_REF.get("serial_a") or not EMS_REF.get("conc_a"):
                return {"ok": False, "verdict": "INCOMPLETE", "why": "no baseline reference"}
            if ctx.left() < ctx.m(120):
                return {"ok": False, "verdict": "INCOMPLETE", "why": "deadline"}
            s = E14M.parity_pass(ctx, _S[:6], max_new=256, tag=tag + "_serial")
            c = E14M.parity_pass(ctx, _S[:len(EMS_REF["conc_a"])], max_new=256, concurrent=True, tag=tag + "_conc")
            out = _gate_from_rows({"serial_a": s, "conc_a": c}, EMS_REF, tag)
            out["server_not_idle"] = _dirty_rows(s, c)
            if out["server_not_idle"]:
                out.update(ok=False, verdict="INCOMPLETE", why="dirty server")
            if _DFL is not None and _DFL is not EMS_REF:
                out["vs_dflash_ref"] = _gate_from_rows({"serial_a": s, "conc_a": c}, _DFL, tag + " vs DFlash reference")["verdict"]
            if EMS_TGT_XL and ctx.left() > ctx.m(240):      # rev 2e: XL parity and the canary (adoption needs both)
                out["xl_gate"] = _xl_verdict(_xl_rows(ctx, tag), *EMS_TGT_XL)
            out["canary"] = _canary(ctx, max(len(EMS_REF["conc_a"]), level or 0), tag)      # rev 2g: at the arm's level
            if out["canary"].get("server_not_idle"):
                out["server_not_idle"] = True
            return out

        def _dead(ctx=None):
            return E11_SERVER is None or not E11_SERVER.alive()

        def _xl_level(caps, level):
            """rev 2d: streams the KV holds at ~100k each (85k prompt + up to 4k generated + slack), never above `level`."""
            kv = (caps or {}).get("kv_tokens") or 0
            return max(1, min(level, kv // 100_000)) if kv else min(level, 6)

        def _windows(tag, level, gate=True, probe=True, ctx=None, xl_level=None):
            ctx = ctx or E14_CTX

            def run():
                out = {}
                if gate:
                    out["gate"] = _gate2(ctx, "gate_" + tag, level)
                    # EMS rev 2: a FAIL is recorded (no adoption), the speed is still measured; only a dirty server stops
                    if out["gate"].get("server_not_idle"):
                        out["windows_skipped"] = "the server was not idle after the gate"
                        return out
                    out["flush_after_gate"] = E14M.flush_cache(ctx)
                for band, ps in (("short", _S), ("long", _L)):
                    if _dead():         # rev 2b (agent 1 F3): a dead server is not "waiting for the level"
                        out[band] = {"error": "the server died"}
                        out["server_alive_after"] = False
                        return out
                    _off = len(ctx.log_text())
                    out[band] = E14M.steady_window(ctx, ps, level=level, secs=_ts(75), tag=tag + band, max_new=16384)
                    if isinstance(out[band], dict):
                        out[band].update(_prefill_hits(ctx.log_text(), _off))      # rev 2e-d (E9)
                if probe and not _dead():         # rev 2g-c (agent 3 A4): the mandatory probe before the XL window
                    out["flush_before_probe"] = E14M.flush_cache(ctx)
                    _rows = _probe(tag, ctx=ctx)
                    out["probe_dirty"] = _dirty_rows(_rows)
                    if out["probe_dirty"]:
                        out["server_not_idle"] = True
                    out["probe"] = _probe_vs_ref(_rows, f"{tag} probe vs reference")
                    out["probe_vs_target"] = _probe_vs_tgt(_rows, f"{tag} probe vs target-only")
                if xl_level and _X and not _dead():       # rev 2d: the XL band (flush first: the 40k block is not reused)
                    out["flush_before_xl"] = E14M.flush_cache(ctx)
                    out["xl"] = E14M.steady_window(ctx, _X, level=xl_level, secs=_ts(75), tag=tag + "xl", max_new=16384)
                out["server_alive_after"] = not _dead()
                return out
            return run

        def _w(r, band, key):
            return ((r or {}).get(band) or {}).get(key)

        def _record(name, ok, rec, r, gate, extra=None):
            caps = (rec or {}).get("caps") or {}
            a = {"ok": bool(ok and r and not r.get("error")),
                 "gate": ((r or {}).get("gate") or {}).get("verdict", ("reference" if r else "no data") if gate is None else ("diagnostic" if not gate else None)),
                 "gate_vs_dflash_ref": ((r or {}).get("gate") or {}).get("vs_dflash_ref"),
                 "probe_complete": _cmp_complete((r or {}).get("probe")), "probe_complete_vs_target": _cmp_complete((r or {}).get("probe_vs_target")),
                 "server_alive_after": (r or {}).get("server_alive_after"), "lines_short": _w(r, "short", "lines"),
                 "short_error": _w(r, "short", "error"), "long_error": _w(r, "long", "error"),
             "window_errors_short": _w(r, "short", "errors") or [], "window_errors_long": _w(r, "long", "errors") or [],
             "running_share_short": _w(r, "short", "real_running_share"), "running_share_long": _w(r, "long", "real_running_share"),
             "gate_class": ((r or {}).get("gate") or {}).get("gate_class"),
             "prefix_hit_short": _w(r, "short", "prefix_hit_rate"), "prefix_hit_long": _w(r, "long", "prefix_hit_rate"),
             "xl_gate": ((r or {}).get("gate") or {}).get("xl_gate"),
             "canary_leaks": (((r or {}).get("gate") or {}).get("canary") or {}).get("foreign_leaks"),
             "canary_errors": (((r or {}).get("gate") or {}).get("canary") or {}).get("errors"),
                 "short_tok_s": _w(r, "short", "tok_s_p50"), "long_tok_s": _w(r, "long", "tok_s_p50"),
                 "accept_short": _w(r, "short", "accept_p50"), "accept_long": _w(r, "long", "accept_p50"),
                 "step_ms_short": _w(r, "short", "step_ms_p50"), "step_ms_long": _w(r, "long", "step_ms_p50"),
                 "level_short": _w(r, "short", "measured_level"), "level_long": _w(r, "long", "measured_level"),
                 "xl_tok_s": _w(r, "xl", "tok_s_p50"), "accept_xl": _w(r, "xl", "accept_p50"), "step_ms_xl": _w(r, "xl", "step_ms_p50"),
                 "level_xl": _w(r, "xl", "measured_level"), "xl_error": _w(r, "xl", "error"), "ctx_xl": _w(r, "xl", "ctx_per_req_p50"),
                 "kv_tokens": caps.get("kv_tokens"), "mamba_cache": caps.get("mamba_cache_size"), "capped_to": caps.get("capped_to"),
                 "log_errors": E14M.log_errors((r or {}).get("server_log_signs")),
                 "probe_flips": ((r or {}).get("probe") or {}).get("large_margin_flips"),
                 "probe_identical": ((r or {}).get("probe") or {}).get("identical"),
                 "probe_flips_vs_target": ((r or {}).get("probe_vs_target") or {}).get("large_margin_flips"),
                 "probe_identical_vs_target": ((r or {}).get("probe_vs_target") or {}).get("identical"),
                 "launch_sec": (rec or {}).get("sec"), "skipped": (rec or {}).get("skipped")}
            a.update(extra or {})
            # EMS rev 2 (GPT Gate C, fail loud): the server ran fewer streams than the arm asked for (mamba / KV cap)
            _want = a.get("per_replica_streams") or a.get("level_wanted")
            a["capped"] = bool(_want and a.get("level_short") is not None and a["level_short"] < _want)
            if a.get("layout") != "2xTP2" and _want and (a.get("level_long") is None or a["level_long"] < _want):
                a["capped"] = a["capped"] or bool(a.get("ok"))      # rev 2b (agent 1 F3): the 40k band counts too
            EMSR["arms"][name] = a
            EMSR["order"].append(name)
            e14_save()
            print("EMS arm", name, json.dumps(a, default=str)[:700], flush=True)
            return a

        def _level(rec, want):
            caps = (rec or {}).get("caps") or {}
            lv = [want]
            if caps.get("capped_to"):
                lv.append(int(caps["capped_to"]) - 1)
            if caps.get("effective_running"):
                lv.append(int(caps["effective_running"]))
            return max(1, min(lv))

        def _arm(name, set_=None, drop=(), streams=None, gate=True, probe=True, cap=14 * 60, env=None, timeout=None):      # rev 2g: 12 -> 14 min
            s = dict(set_ or {})
            if streams:
                s.update(_streams(streams, _slots(s)))
            drop = tuple(drop) + (("--cuda-graph-max-bs-decode",) if "--cuda-graph-bs-decode" in s else ())   # agent 1 F8
            _ok, _rec = e14_launch(name, set_=s, drop=drop, env=env, timeout=timeout)
            EMSR["launches"][name] = {k: _rec.get(k) for k in ("ok", "why", "sec", "caps", "skipped")}
            _lv = _level(_rec, streams or LEVEL)
            _r = (e14_phase("EMS_" + name, _windows(name, _lv, gate=gate, probe=probe, xl_level=_xl_level(_rec.get("caps"), _lv)),
                            _ts(cap + 3 * 60)) if _ok else None)
            fam = "mtp" if str(s.get("--speculative-algorithm", "")).upper() == "EAGLE" else ("target" if "--speculative-algorithm" in drop else "dflash")
            _px = {}
            if (env or {}).get("SGLANG_ENABLE_PCIE_IPC_ALLREDUCE") and E11_SERVER is not None:
                _lt = E11_SERVER.log_text()
                _px = {"pcie_ipc": {"enabled": "FlashInfer PCIe-IPC all-reduce enabled" in _lt,
                                    "workspace_built": "FlashInfer PCIe-IPC workspace built" in _lt,
                                    "no_pcie_ipc_comm": "has no pcie_ipc_comm" in _lt,
                                    "workspace_failed": "PCIe-IPC workspace" in _lt and "failed" in _lt,
                                    "flashinfer_version_lines": [l[-160:] for l in _lt.splitlines() if "flashinfer" in l.lower() and "version" in l.lower()][:3]}}
            return _record(name, _ok, _rec, _r, gate, {"set": s, "drop": list(drop), "level_wanted": streams or LEVEL, "family": fam,
                                                       "env": dict(env or {}), **_px})

        def _gated(a):
            """rev 2b: adoptable = gate PASS, complete probe evidence vs the SAME oracle as the gate (one flag), a live server,
            both bands measured with >= 8 decode lines, not capped, no log errors."""
            if not a or not a.get("ok") or a.get("gate") not in ("PASS", "reference") or a.get("log_errors"):
                return False
            if a.get("server_alive_after") is False or a.get("capped") or a.get("short_error") or a.get("long_error"):
                return False
            if (a.get("lines_short") or 0) < (8 if _ts(75) >= 60 else 1) or not a.get("short_tok_s") or not a.get("long_tok_s"):
                return False
            # rev 2e (2nd audit HIGH-3): a failed stream or a window that did not really hold its level is no evidence
            if a.get("window_errors_short") or a.get("window_errors_long"):
                return False
            if any((a.get(f"running_share_{b_}") is None or a[f"running_share_{b_}"] < 0.80) for b_ in ("short", "long")):
                return False
            # rev 2e (GPT audit HIGH-3): no independent target-only oracle = nothing is adoptable (diagnostics still run)
            if not (EMSR.get("oracle") == "target_only" and EMS_TGT is not None and _cmp_complete(EMSR.get("probe_tgt_aa"))):
                return False
            # rev 2e (GPT audits MEDIUM-7 / MEDIUM-9): no cross-request leak, and the XL parity (when measured) passed
            if a.get("canary_leaks") or a.get("canary_errors") or a.get("canary_leaks") is None:
                return False
            if a.get("xl_gate") not in (None, "PASS"):
                return False
            # rev 2g (GPT R2F-02 / R6): XL parity required whenever XL prompts exist (skipped / missing = no evidence);
            # the oracle's own canary must exist and be clean; a PCIe-IPC arm must show its workspace was really built
            if _X and a.get("xl_gate") != "PASS":
                return False
            _oc = EMSR.get("canary_oracle")
            if not isinstance(_oc, dict) or _oc.get("foreign_leaks") != 0 or _oc.get("errors") != 0 or _oc.get("server_not_idle"):
                return False
            if (a.get("env") or {}).get("SGLANG_ENABLE_PCIE_IPC_ALLREDUCE") and not (a.get("pcie_ipc") or {}).get("workspace_built"):
                return False
            if EMSR.get("oracle") == "target_only":
                aa, ok_aa, pf, ok_pf = EMSR.get("probe_tgt_aa"), True, a.get("probe_flips_vs_target"), a.get("probe_complete_vs_target")
            else:
                aa, ok_aa, pf, ok_pf = EMSR.get("probe_ref_aa"), True, a.get("probe_flips"), a.get("probe_complete")
            if not _cmp_complete(aa) or not ok_pf or pf is None:
                return False
            return pf <= (aa.get("large_margin_flips") or 0) + 1

        def _text_bad(x):
            """rev 2e-c (shared with VMS): degenerate = empty, or (not grid-like: < 30 % digits among non-space chars) a
            32-character run of one character, < 8 distinct characters over 200, or < 10 % letters / digits. An ARC grid
            printed in the reasoning is never degenerate."""
            s = (x or "").strip()
            if not s:
                return True
            # rev 2g (GPT R2F-06): an ARC grid by structure (only digits / brackets / commas / whitespace, every row
            # <= 30 cells, the ARC maximum), not by a digit share: "0" * 250 or "0 0 0 ..." x 200 is degenerate
            if re.fullmatch(r"[0-9,\[\]\s]+", s) and len(re.findall(r"\d", s)) >= 2 \
                    and all(len(re.findall(r"\d", r_)) <= 30 for r_ in re.split(r"\]|\n", s)):
                return False
            if re.search(r"(.)\1{31,}", s):
                return True
            return len(s) >= 200 and (len(set(s)) < 8 or sum(ch.isalnum() for ch in s) / len(s) < 0.10)

        def _tail_bad(ids):
            """rev 2e (GPT audit HIGH-2), rev 2e-c: the decoded continuation is degenerate by the shared text rule."""
            ids = list(ids or [])
            if len(ids) < 16:
                return False
            try:
                return _text_bad(tokenizer.decode(ids, skip_special_tokens=True))
            except Exception:
                return False

        def _tail_count(rows):
            return sum(1 for r_ in rows or [] if _tail_bad(r_.get("ids")))

        def _post_div_tails(rows_c, rows_b):
            """rev 2g-c (agent 3 B5): degenerate continuations AFTER the first divergence from the matching base row
            (a whole-row check let a clean head mask a degenerate tail)."""
            n_ = 0
            for rc, rb in zip(rows_c or [], rows_b or []):
                ic, ib = list(rc.get("ids") or []), list(rb.get("ids") or [])
                k = next((j for j, (u, v) in enumerate(zip(ic, ib)) if u != v), None)
                if k is not None and _tail_bad(ic[k:]):
                    n_ += 1
            return n_

        def _gate_from_rows(cand, base, name):
            """arm_gate's rule on rows that already exist (no GPU time): cand's serial / all-at-once passes vs base's,
            judged against base's own A/A; flips decide. rev 2e: and no more degenerate continuations than base's A/A
            (a near-tie first divergence must not hide corruption after it)."""
            ps = E14M.parity_ok(E14M.compare_passes(base["serial_a"], cand["serial_a"][:len(base["serial_a"])], name + ": serial", n=256),
                                E14M.compare_passes(base["serial_a"], base["serial_b"], "oracle serial A/A", n=256),
                                allowance=1, identity="ignore")
            pc = E14M.parity_ok(E14M.compare_passes(base["conc_a"], cand["conc_a"][:len(base["conc_a"])], name + ": all-at-once", n=256),
                                E14M.compare_passes(base["conc_a"], base["conc_b"], "oracle all-at-once A/A", n=256),
                                allowance=2, identity="ignore")
            ok = ps["ok"] and pc["ok"]
            v = "PASS" if ok else ("FAIL" if "FAIL" in (ps["verdict"], pc["verdict"]) else "INCOMPLETE")
            if _dirty_rows(base.get("serial_a"), base.get("serial_b"), base.get("conc_a"), base.get("conc_b")):
                ok, v = False, "INCOMPLETE"       # rev 2b: an A/A taken on a busy server is no reference
            tail_c = _post_div_tails(cand.get("serial_a"), base.get("serial_a")) + _post_div_tails(cand.get("conc_a"), base.get("conc_a"))
            tail_b = _post_div_tails(base.get("serial_b"), base.get("serial_a")) + _post_div_tails(base.get("conc_b"), base.get("conc_a"))
            if v == "PASS" and tail_c > tail_b:
                ok, v = False, "FAIL"             # rev 2e: degenerate continuations beyond the oracle's own
            _same = (ps.get("arm_identical") == ps.get("compared")) and (pc.get("arm_identical") == pc.get("compared"))
            gc = ("PASS_EXACT" if _same else "BOUNDARY") if v == "PASS" else v      # rev 2e (2nd audit HIGH-6)
            return {"ok": ok, "verdict": v, "gate_class": gc, "serial": ps, "all_at_once": pc, "tail_degenerate": tail_c,
                    "tail_degenerate_aa": tail_b}

        # ---- the reference: the production launch the engine cell already made (A/A gate reference, windows, probe A/A)
        E14_STATE.enter("EMS_REF")
        e14_phase("EMS_ref_aa", _ref_gate, _ts(8 * 60))
        _r0 = e14_phase("EMS_ref_windows", _windows("ref", LEVEL, gate=False, probe=False,
                                                    xl_level=_xl_level(E11_SGL.get("caps"), LEVEL)), _ts(14 * 60))

        def _ref_probe():
            a, b = _probe("ref_a"), _probe("ref_b")
            globals()["EMS_PROBE_REF"] = (a, b)
            EMSR["probe_ref_aa"] = E14M.compare_passes(a, b, "reference probe A/A")
            return {"aa": EMSR["probe_ref_aa"], "server_not_idle": _dirty_rows(a, b)}
        e14_phase("EMS_ref_probe", _ref_probe, _ts(8 * 60))
        _record(REF_NAME, True, {"caps": (E11_SGL.get("caps") or {}), "sec": ENGINE_INFO.get("load_sec")}, _r0, None,
                {"set": {}, "drop": [], "level_wanted": LEVEL, "family": "dflash", "flags": _RF})

        # ---- the 2 x TP2 slot is reserved now: no earlier arm may launch or measure into it
        E14_RESERVE = _ts(30 * 60) if EMS_TP2_ON else 0.0       # rev 2f: the TP2 slot only where TP2 runs
        _MAMBA_RATIO = {"--mamba-full-memory-ratio": 0.5}
        _NTN = {"--attention-backend": "flashinfer", "--linear-attn-backend": "triton", "--chunked-prefill-size": 2048,
                "--max-prefill-tokens": 4096, "--mamba-radix-cache-strategy": "extra_buffer", "--mamba-ssm-dtype": "float32"}
        _NTN.update(_MAMBA_RATIO)
        # rev 2b (agent 2 F1, kv_cache_configurator.py:2292-2330 / 2505-2545): with the auto size (ratio 0.5, fp32, 5 slots per
        # request) the server caps at ~11 (INT4) / ~9 (FP8) running; an explicit 5 x 17 = 85 slots makes it a 16-stream arm
        # (the ratio is then ignored, :2462-2484). --attention-backend flashinfer / --linear-attn-backend triton equal the
        # reference's resolved values and --max-prefill-tokens 4096 is moot under chunk 2048 (no-ops, kept as the bundle).
        _NTN["--max-mamba-cache-size"] = 5 * SGL_SERVER_MAX_RUNNING
        _ntn_drop = _SPEC + ("--min-free-slots-delay",)
        EMSR["notes"] = {"ntn": "NTN flag bundle + --max-mamba-cache-size 85 (5 x 17) so 16 streams fit; attention flashinfer / "
                                "linear triton = the reference's values; max-prefill 4096 moot under chunk 2048"}

        # ---- EMS rev 2 (GPT pre-GPU audit 2026-10-25, ACC-2 / ACC-3: SGLang #38009 and #36548 report DFlash2 greedy
        # output diverging from plain decoding, worse with concurrent requests; not reproduced on our stack yet): target-only
        # on the same INT4 checkpoint = the identity oracle. Its own A/A becomes the gate reference of every later arm, and
        # the DFlash reference is judged against it from the rows it already has (no extra GPU time). If target-only gives
        # no complete A/A, the gate stays on the DFlash reference (the rev 1 rule) and the report says so.
        EMSR["gate_reference"] = REF_NAME + " (target-only A/A unavailable)"
        _DFL = EMS_REF
        EMSR["oracle"] = "dflash_ref"
        _tdrop = _SPEC + ("--min-free-slots-delay",)
        _ok, _rec = e14_launch("target_only", drop=_tdrop)
        EMSR["launches"]["target_only"] = {k: _rec.get(k) for k in ("ok", "why", "sec", "caps", "skipped")}
        _rt = None
        if _ok:
            def _tgt_aa():
                ref, summ = _aa_rows("ems_tgt")
                # rev 2g-c (agent 3 B6): the oracle's own rows must not be degenerate
                summ["degenerate_rows"] = sum(1 for k_ in ("serial_a", "serial_b", "conc_a", "conc_b") for r_ in (ref.get(k_) or [])
                                              if _text_bad(tokenizer.decode(r_.get("ids") or [], skip_special_tokens=True)))
                if summ["complete"] and not summ["degenerate_rows"]:      # rev 2b: no errors, no missing margins, not dirty
                    globals()["EMS_TGT"] = ref
                if _X:                    # rev 2e: the oracle's own XL A/A (rev 2g-c: only a complete one counts)
                    xa, xb = _xl_rows(E14_CTX, "ems_tgt_a"), _xl_rows(E14_CTX, "ems_tgt_b")
                    summ["xl_aa"] = E14M.compare_passes(xa or [], xb or [], "target-only XL A/A", n=256)
                    if xa and xb and not _dirty_rows(xa, xb) and _cmp_complete(summ["xl_aa"]):
                        globals()["EMS_TGT_XL"] = (xa, xb)
                    EMSR["oracle_xl"] = "complete" if globals().get("EMS_TGT_XL") else "NO_ORACLE_XL (nothing adoptable)"
                summ["canary"] = EMSR["canary_oracle"] = _canary(E14_CTX, LEVEL, "ems_tgt")
                return summ
            e14_phase("EMS_target_aa", _tgt_aa, _ts(12 * 60))
            _rt = e14_phase("EMS_target_only", _windows("target_only", _level(_rec, LEVEL), gate=False, probe=False,
                                                        xl_level=_xl_level(_rec.get("caps"), _level(_rec, LEVEL))), _ts(14 * 60))

            def _tgt_probe():
                a, b = _probe("tgt_a"), _probe("tgt_b")
                globals()["EMS_PROBE_TGT"] = (a, b)
                EMSR["probe_tgt_aa"] = E14M.compare_passes(a, b, "target-only probe A/A")
                EMSR["dflash_vs_target_probe"] = (E14M.compare_passes(a, EMS_PROBE_REF[0][:len(a)], "DFlash reference probe vs target-only")
                                                  if EMS_PROBE_REF else {"skipped": "no reference probe"})
                return {"aa": EMSR["probe_tgt_aa"], "dflash_vs_target": EMSR["dflash_vs_target_probe"], "server_not_idle": _dirty_rows(a, b)}
            e14_phase("EMS_target_probe", _tgt_probe, _ts(10 * 60))
        _dvt = EMSR.get("dflash_vs_target_probe") or {}
        _record("target_only", _ok, _rec, _rt, None, {"set": {}, "drop": list(_tdrop), "level_wanted": LEVEL, "role": "identity oracle",
                                                      "probe_flips": _dvt.get("large_margin_flips"), "probe_identical": _dvt.get("identical")})
        if EMS_TGT and not _cmp_complete(EMSR.get("probe_tgt_aa")):
            EMSR["gate_reference"] = "ref_dflash_w6 (target-only gate A/A complete, but its probe A/A was not)"
            EMS_TGT = None              # rev 2b: gate and probe always use the same oracle
        if EMS_TGT:
            EMSR["oracle"] = "target_only"
            EMSR["gate_reference"] = "target_only (same INT4 checkpoint, no speculation)"
            EMSR["dflash_vs_target_gate"] = (_gate_from_rows(_DFL, EMS_TGT, REF_NAME + " vs target-only") if _DFL
                                             else {"ok": False, "verdict": "INCOMPLETE", "why": "no DFlash reference rows"})
            EMS_REF = EMS_TGT
            _ra = EMSR["arms"].get(REF_NAME)
            if _ra is not None:
                _ra.update(gate=EMSR["dflash_vs_target_gate"]["verdict"], probe_flips_vs_target=_dvt.get("large_margin_flips"),
                           probe_complete_vs_target=_cmp_complete(_dvt),        # rev 2e-e: _gated needs it
                           probe_identical_vs_target=_dvt.get("identical"),
                           gate_class=EMSR["dflash_vs_target_gate"].get("gate_class"),
                           canary_leaks=(EMSR.get("canary_ref") or {}).get("foreign_leaks"),
                           canary_errors=(EMSR.get("canary_ref") or {}).get("errors"),
                           xl_gate=_xl_verdict(EMSR.pop("_ref_xl_rows", None), *EMS_TGT_XL) if EMS_TGT_XL else None)
            e14_save()
        print("EMS gate reference:", EMSR["gate_reference"], "| DFlash vs target-only:",
              (EMSR.get("dflash_vs_target_gate") or {}).get("verdict"), flush=True)

        PLAN = [
            # rev 2d: PCIe-IPC all-reduce on the reference config (communication = 51 % of a step, Notes/30), first after
            # the oracle; skipped (recorded) when the FlashInfer 0.7 overlay could not be built
            ("dflash_w6_pcie", dict(env=_PCIE, timeout=30 * 60)),
            ("mtp_3_1_4", dict(set_=_mtp(), drop=_SPEC + ("--min-free-slots-delay",))),
            # rev 2c: the same arm with PR #45151 (env-gated patch): compare KV tokens and tok/s with mtp_3_1_4
            ("mtp_3_1_4_pr45151", dict(set_=_mtp(), drop=_SPEC + ("--min-free-slots-delay",), env={"EMS_PR45151": "1"})),
            ("mtp_3_1_4_pcie", dict(set_=_mtp(), drop=_SPEC + ("--min-free-slots-delay",), env=_PCIE, timeout=30 * 60)),
            ("ntn_int4", dict(set_=dict(_NTN, **_mtp()), drop=_ntn_drop)),
            ("dflash_w6_s20", dict(streams=20)),
            ("dflash_w6_s24", dict(streams=24)),
            # rev 2e (GPT audit E12): the cold FP8 load from /kaggle/input after the admission-critical 20 / 24 arms
            ("ntn_fp8", dict(set_=dict(_NTN, **_mtp(path=MODEL_PATH), **{"--model-path": str(MODEL_PATH)}), drop=_ntn_drop,
                             gate=False, probe=False, timeout=30 * 60)),
            ("mtp_2_1_3", dict(set_=_mtp(steps=2, draft=3), drop=_SPEC + ("--min-free-slots-delay",))),
            ("mtp_3_1_4_mem86", dict(set_=_mtp(extra={"--mem-fraction-static": "0.86"}), drop=_SPEC + ("--min-free-slots-delay",))),
            ("mtp_replay_fp16", dict(set_=_mtp(extra={"--enable-linear-replayssm-spec": None}), drop=_SPEC + ("--min-free-slots-delay",))),
            ("mtp_replay_fp32", dict(set_=_mtp(extra={"--enable-linear-replayssm-spec": None, "--mamba-ssm-dtype": "float32"}),
                                     drop=_SPEC + ("--min-free-slots-delay",))),
            ("dflash_w4", dict(set_={"--speculative-num-draft-tokens": 4})),
            ("dflash_w8", dict(set_={"--speculative-num-draft-tokens": 8})),
        ]
        # EMS rev 2 (GPT pre-GPU audit S8, checked in the SGLang 0.5.21 source): no other linear-attention kernel exists for
        # a speculative run on the L4 (SM 8.9), so both backend arms are dropped instead of spending two launches.
        EMSR["dropped"] = {
            "linattn_flashinfer": "FlashInfer GDN needs SM90+: kernels/gdn_flashinfer.py:135-136 sets available = capability "
                                  ">= 9 and FlashInferGDNKernel() raises 'FlashInfer GDN kernels are not available' (:184-187), "
                                  "so the server dies at startup on the L4",
            "linattn_cutedsl": "CuTe DSL GDN prefill is SM100+ only and falls back to Triton (gdn_backend.py:304-324); GDN "
                               "verify stays Triton unless FlashInfer (gdn_backend.py:347-360); under DFlash / MTP every target "
                               "step is a verify, so the arm would run the reference's kernels"}
        def _safe(name, fn, *args, **kw):
            """rev 2g: run one arm; its own crash is recorded on it, its server killed, the results saved, the next arm runs."""
            try:
                return fn(*args, **kw)
            except Exception as _e:
                EMSR["arms"][name] = dict(EMSR["arms"].get(name) or {}, ok=False,
                                          error=f"arm crashed: {type(_e).__name__}: {_e}"[:500], adoptable=False)
                if name not in EMSR["order"]:
                    EMSR["order"].append(name)
                print(f"EMS arm {name} crashed:", traceback.format_exc()[-1500:], flush=True)
                try:
                    e11_kill_server(wait=True)
                except Exception:
                    pass
                e14_save()
                return None

        def _ems_adopt_all():
            """rev 2e / 2g: speed, correctness and adoption kept apart; production_approved needs the 60k+ generation and
            ARC task gates this matrix does not run."""
            for _k, _a in EMSR["arms"].items():
                _a["adoptable"] = bool(_gated(_a)) and _a.get("family") in ("mtp", "dflash")
                _a["candidate_for_arc_validation"] = _a["adoptable"]
                _a["certified_exact"] = _a["adoptable"] and _a.get("gate_class") == "PASS_EXACT"
                _a["production_approved"] = False

        for _name, _kw in PLAN:
            _why = None
            if _name in ("dflash_w4", "dflash_w8") and int(_kw["set_"]["--speculative-num-draft-tokens"]) == _RW:
                _why = f"= the reference (it runs width {_RW})"
            if _name.endswith("_pr45151") and not (EMSR["pr45151_patch"].get("applied") or E11_HARNESS):
                _why = "PR #45151 patch not applied: " + str(EMSR["pr45151_patch"].get("why"))[:300]     # rev 2e (HIGH-4)
            if _name.endswith("_pcie") and not (EMSR["fi07_overlay"].get("ok") or E11_HARNESS):
                _why = "FlashInfer 0.7 overlay: " + str(EMSR["fi07_overlay"].get("why") or EMSR["fi07_overlay"])[:300]
            if _name in ("dflash_w6_s20", "dflash_w6_s24") and E14_VENV["cfg"].get("streams12"):
                _why = "the reference fell back to 12 streams: 20 / 24 cannot fit the same memory"
            if EMS_PART != "ALL" and _name not in EMS_PART_ARMS.get(EMS_PART, ()):
                _why = "OTHER_PART: runs in the EMS " + ("SAFE" if EMS_PART == "RISKY" else "RISKY") + " notebook"
            if _why:
                EMSR["arms"][_name] = {"ok": False, "skipped": _why}
                EMSR["order"].append(_name)
                continue
            _safe(_name, _arm, _name, **_kw)      # rev 2f/2g: one arm's own crash never ends the remaining arms

        # rev 2g (agents N3): the fastest gated MTP of THIS part at 20 streams, after the whole plan (in RISKY the #45151 /
        # PCIe / NTN arms are candidates: the KV #45151 frees is the reason to run it); rev 2b: by family
        _mt = {k: v for k, v in EMSR["arms"].items() if v.get("family") == "mtp" and not k.endswith("_s20") and _gated(v)
               and v.get("short_tok_s")}
        if _mt and not E14_VENV["cfg"].get("streams12"):
            _best = max(_mt, key=lambda k: _mt[k]["short_tok_s"])
            _safe(_best + "_s20", _arm, _best + "_s20", set_=EMSR["arms"][_best]["set"], drop=tuple(EMSR["arms"][_best]["drop"]),
                  streams=20, env=EMSR["arms"][_best].get("env"))

        # ---- LAST: 2 x TP2 (two servers, GPUs 0-1 and 2-3, each TP2, own ports; 8 + 8 streams for the 8k band, the 40k
        # band at the level each replica's KV holds). DFlash W6 first, then the best gated MTP if the slot has time.
        E14_RESERVE = 0.0

        def ems_topo():
            """EMS rev 2 (GPT pre-GPU audit S9 / section 7): `nvidia-smi topo -m` recorded, and the GPU pairs chosen from it
            (the pairing whose weaker pair has the best link; ties keep 0-1 / 2-3)."""
            rank = {"X": 0, "SYS": 1, "NODE": 2, "PHB": 3, "PXB": 4, "PIX": 5}
            out = {"pairs": [[0, 1], [2, 3]]}
            try:
                txt = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
                out["matrix"] = txt[-3000:]
                try:      # rev 2c (GPT audit Gate D): the P2P read capability matrix as well
                    out["p2p_prop"] = subprocess.run(["nvidia-smi", "topo", "-p2p", "p"], capture_output=True, text=True,
                                                     timeout=30).stdout[-2000:]      # rev 2g-b: the audit asked for p
                    out["p2p_read"] = subprocess.run(["nvidia-smi", "topo", "-p2p", "r"], capture_output=True, text=True,
                                                     timeout=30).stdout[-2000:]
                except Exception as _e:
                    out["p2p_read"] = repr(_e)[:200]
                m = {}
                for ln in txt.splitlines():
                    p = ln.split()
                    if len(p) >= 5 and re.fullmatch(r"GPU\d", p[0]):
                        m[int(p[0][3:])] = p[1:5]
                if all(i in m for i in range(4)):
                    def link(i, j):
                        v = m[i][j]
                        return 10 if v.startswith("NV") else rank.get(v, 0)
                    cands = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
                    best = max(cands, key=lambda c: (min(link(*p) for p in c), c == cands[0]))
                    out.update(pairs=[list(p) for p in best], links={f"{i}-{j}": m[i][j] for i in range(4) for j in range(i + 1, 4)})
            except Exception as e:
                out["error"] = repr(e)[:300]
            EMSR["topology"] = out
            e14_save()
            return out["pairs"]

        def ems_replica_env(r):
            """EMS rev 2 (GPT pre-GPU audit S9): each replica gets its own cache dirs, seeded with a copy of the warm caches
            (never the CUDA driver cache: SGLang #42917, the E14 _NEVER_SHIP rule), so two servers never compile into one dir."""
            base = f"/tmp/e11-cache-r{r}"
            env = {}
            for k, v in E14L.jit_cache_env().items():
                dst = base + v[len("/tmp/e11-cache"):] if v.startswith("/tmp/e11-cache") else v + f"-r{r}"
                try:
                    if k != "CUDA_CACHE_PATH" and Path(v).is_dir() and not Path(dst).exists():
                        shutil.copytree(v, dst, ignore=shutil.ignore_patterns("nv"))
                    Path(dst).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"EMS replica {r} cache {k}: {e!r}"[:300], flush=True)
                env[k] = dst
            return env

        def ems_tp2(name, set_=None, drop=(), per=8, env=None):
            global E11_SERVER
            rec = {"arm": name, "replicas": []}
            if not E14M.launch_allowed(time.perf_counter(), HARD_STOP, reserve=0.0, force=False):
                rec["skipped"] = "NOT_TESTED_BUDGET: less than ~17 min before the hard stop"
                return _record(name, False, rec, None, False, {"layout": "2xTP2"})
            try:
                e11_kill_server(wait=True)
            except Exception:
                pass
            E11_SERVER = None
            rec["gpu_free"] = _e11_gpu_free()
            servers = []
            _pairs = (EMSR.get("topology") or {}).get("pairs") or ems_topo()
            rec["pairs"] = _pairs
            try:          # rev 2b (agent 1 F13): a failed start never leaves the other replica running
                for r, (devs, port, nccl) in enumerate(((",".join(map(str, _pairs[0])), SGL_PORT, 41231),
                                                        (",".join(map(str, _pairs[1])), SGL_PORT + 1, 41331))):
                    s = dict(set_ or {})
                    # ports: SGLang 0.5.21 without DP attention uses IPC temp files + the nccl port only (server_args.py:708-719)
                    s.update({"--tp-size": 2, "--port": port, "--nccl-port": nccl, **_streams(per, _slots(s)),
                              # rev 2b (agent 2 F3): at world size 2 the legacy CustomAllreduce turns on when the P2P test passes
                              # (custom_all_reduce_utils.py:442-512); here P2P measured 5.5x slower than NCCL (E14A p2p_sys)
                              "--disable-custom-all-reduce": None})
                    flags = e14_edit_flags(e11_flags(E14_VENV["cfg"]), drop=tuple(drop) + ("--cuda-graph-max-bs-decode",), set_=s)
                    env_r = dict(E14_VENV["env"], CUDA_VISIBLE_DEVICES=devs, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT")   # rev 2c: Gate D
                    env_r.update(env or {})
                    env_r.update(ems_replica_env(r))
                    rec.setdefault("cache_dirs", []).append({k: env_r[k] for k in E14L.jit_cache_env()})
                    rec.setdefault("replica_flags", []).append([str(x) for x in flags])
                    srv = E11Server(E14_VENV["launch"] + flags, env_r, f"ems_{name}_r{r}")
                    srv.start()
                    servers.append((srv, port))
            except Exception as e:
                rec["start_error"] = repr(e)[:300]
                for srv, _ in servers:
                    srv.kill()
                _e11_gpu_free()
                return _record(name, False, rec, None, False, {"layout": "2xTP2", "error": rec["start_error"]})
            t0, ready = time.perf_counter(), [False, False]
            # rev 2b (agent 2 F5): two cold TP2 compiles at once (new per-GPU shapes): a 30-min wall, never past the hard stop
            limit = min(30 * 60, max(120.0, HARD_STOP - time.perf_counter() - 12 * 60))
            while time.perf_counter() - t0 < limit and not all(ready):
                for i, (srv, port) in enumerate(servers):
                    if not ready[i] and srv.alive():
                        try:
                            _urlreq.urlopen(f"http://127.0.0.1:{port}/health", timeout=10).read()
                            ready[i] = True
                        except Exception:
                            pass
                if any(not s.alive() for s, _ in servers):
                    break
                time.sleep(5)
            rec["ready"], rec["sec"] = ready, round(time.perf_counter() - t0, 1)
            for srv, port in servers:
                _lt = srv.log_text()
                # TP2 is 2 GPUs: SGLang enables its custom all-reduce if P2P works (custom_all_reduce_utils.py:492-511)
                rec["replicas"].append({"port": port, "alive": srv.alive(), "caps": e11_caps(_lt),
                                        # rev 2g (agents N8): the flag disables it before the log line can appear
                                        "custom_allreduce_disabled": "CustomAllreduce is disabled" in _lt
                                        or any("--disable-custom-all-reduce" in f for f in rec.get("replica_flags") or []),
                                        "log_tail": _lt[-1500:] if not all(ready) else None})
            res = None
            if all(ready):
                ctxs = []
                for srv, port in servers:
                    c = E14M.Ctx(f"http://127.0.0.1:{port}", srv.log_text, E14_OUT / f"{name}_{port}",
                                 model_id=_e14_model_id(f"http://127.0.0.1:{port}"), deadline=HARD_STOP - 8 * 60, log=print,
                                 encode=None, normalize=lambda x: x)
                    c.scale = E14_TS
                    ctxs.append(c)
                kv = [((x["caps"] or {}).get("kv_tokens") or 0) for x in rec["replicas"]]
                long_per = max(1, min(per, min(kv) // 45000)) if min(kv) else 1

                def _both(fn):
                    outs = [None, None]

                    def go(i):
                        try:
                            outs[i] = fn(ctxs[i], i)
                        except BaseException as e:
                            outs[i] = {"error": repr(e)[:300]}
                    ths = [threading.Thread(target=go, args=(i,)) for i in range(2)]
                    for t in ths:
                        t.start()
                    _dl = min(getattr(c, "deadline", HARD_STOP) or HARD_STOP for c in ctxs)
                    for t in ths:          # rev 2g (GPT R2F-08): bounded by the arm's deadline
                        t.join(timeout=max(1.0, _dl - time.perf_counter() + 60))
                    if any(t.is_alive() for t in ths):
                        for srv, _ in servers:
                            try:
                                srv.kill()
                            except Exception:
                                pass
                        return [{"error": "TP2_WORKER_DEADLINE"}, {"error": "TP2_WORKER_DEADLINE"}]
                    return outs

                def run():
                    # rev 2b (agent 1 F4): both replicas gated (in parallel), both logs scanned
                    _off = [len(srv.log_text()) for srv, _ in servers]
                    gates = _both(lambda c, i: _gate2(c, f"gate_{name}_r{i}"))
                    out = {"gate_replicas": gates}
                    for c in ctxs:
                        E14M.flush_cache(c)
                    sw = _both(lambda c, i: E14M.steady_window(c, _S, level=per, secs=_ts(75), tag=f"{name}r{i}short"))
                    lw = _both(lambda c, i: E14M.steady_window(c, _L, level=long_per, secs=_ts(75), tag=f"{name}r{i}long"))
                    good = lambda x: bool(x) and bool(x.get("tok_s_p50")) and not x.get("error") and not x.get("skipped")
                    out.update(short=sw, long=lw, long_per_replica=long_per, all_windows_ok=all(good(x) for x in sw + lw),
                               short_tok_s_sum=sum((x or {}).get("tok_s_p50") or 0 for x in sw),
                               long_tok_s_sum=sum((x or {}).get("tok_s_p50") or 0 for x in lw),
                               server_log_signs=[{s_: srv.log_text()[o:].count(s_) for s_ in E14_LOG_SIGNS if s_ in srv.log_text()[o:]}
                                                 for (srv, _), o in zip(servers, _off)])
                    vs = [(g or {}).get("verdict") for g in gates]
                    out["gate_verdict"] = "PASS" if vs == ["PASS", "PASS"] else ("FAIL" if "FAIL" in vs else "INCOMPLETE")
                    return out
                for c in ctxs:
                    c.deadline = time.perf_counter() + min(_ts(14 * 60), max(60.0, HARD_STOP - time.perf_counter() - 8 * 60))
                try:
                    res = run()
                except BaseException as e:
                    res = {"error": repr(e)[:500]}
            for srv, _ in servers:
                try:
                    srv.kill()
                except Exception:
                    pass
            _e11_gpu_free()
            E14_RESULTS["phases"]["EMS_" + name] = {"launch": rec, "result": res}
            return _record(name, all(ready), {"caps": (rec["replicas"][0] if rec["replicas"] else {}).get("caps"),
                                              "sec": rec["sec"]}, None, False,
                           {"ok": bool(all(ready) and res and not res.get("error") and res.get("all_windows_ok")),
                            "kv_ok_for_production": bool(rec["replicas"]) and all(((x["caps"] or {}).get("kv_tokens") or 0) >= per * SGL_KV_PER_REQUEST
                                                                                 for x in rec["replicas"]),
                            "log_errors": {k: v for x in (res or {}).get("server_log_signs") or [] for k, v in E14M.log_errors(x).items()},
                            "layout": "2xTP2", "replicas_ready": ready, "per_replica_streams": per, "pairs": rec.get("pairs"), "cache_dirs": rec.get("cache_dirs"),
                            "custom_allreduce_disabled": [x.get("custom_allreduce_disabled") for x in rec["replicas"]],
                            "level_short": min([(x or {}).get("measured_level") or 0 for x in (res or {}).get("short") or [None]]),
                            "short_tok_s": (res or {}).get("short_tok_s_sum"), "long_tok_s": (res or {}).get("long_tok_s_sum"),
                            "long_per_replica": (res or {}).get("long_per_replica"),
                            "gate": (res or {}).get("gate_verdict"),
                            "kv_tokens_per_replica": [((x["caps"] or {}).get("kv_tokens")) for x in rec["replicas"]],
                            "error": (res or {}).get("error"), "set": set_ or {}, "drop": list(drop)})

        if not EMSR.get("topology"):      # rev 2g (agents N5): recorded in every part
            try:
                ems_topo()
            except Exception as _e:
                EMSR["topology"] = {"error": repr(_e)[:200]}
        if EMS_TP2_ON:
            _safe("tp2x2_dflash_w6", ems_tp2, "tp2x2_dflash_w6")
        else:
            for _k in ("tp2x2_dflash_w6", "tp2x2_mtp_best"):
                EMSR["arms"][_k] = {"ok": False, "skipped": "OTHER_PART: 2 x TP2 runs in the EMS RISKY notebook"}
                EMSR["order"].append(_k)
        _mt = {k: v for k, v in EMSR["arms"].items() if v.get("family") == "mtp" and not k.endswith("_s20") and _gated(v)
               and v.get("short_tok_s")}
        if _mt and EMS_TP2_ON:
            _best = max(_mt, key=lambda k: _mt[k]["short_tok_s"])
            _safe("tp2x2_" + _best, ems_tp2, "tp2x2_" + _best, set_=EMSR["arms"][_best]["set"],
                  drop=tuple(EMSR["arms"][_best]["drop"]), env=EMSR["arms"][_best].get("env"))
        _ems_adopt_all()
        E14_STATE.enter("EMS_DONE")
        e14_save()
    else:
        EMSR["skipped"] = "rerun" if RERUN else ("engine not ready" if not ENGINE_READY else "EMS profile off")
        print("EMS skipped:", EMSR["skipped"], flush=True)
except BaseException:            # rev 2d-b (agent 1 F13): record, never abort the notebook; the report + bundle cell runs
    EMSR["error"] = traceback.format_exc()[-3000:]
    print("EMS cell error:", EMSR["error"][-1500:], flush=True)
    try:          # rev 2g: the adoption columns are written even after a cell-level error
        globals().get("_ems_adopt_all", lambda: None)()
    except Exception:
        pass
    try:
        E14_STATE.enter("EMS_DONE")
        e14_save()
    except Exception:
        pass
'''

REPORT = r'''# ==================================================================== CELL EMS-R — report, bundle
try:
    E14_STATE.enter("REPORT")
    _ref = (EMSR.get("arms") or {}).get(EMSR.get("ref_name") or "ref_dflash_w6_s16") or {}
    _rs, _rl = _ref.get("short_tok_s") or 0, _ref.get("long_tok_s") or 0
    L = [f"# EMS {globals().get('EMS_PART', '')}: SGLang engine-config matrix", "", f"profile {PROFILE}, rerun {RERUN}, engine {ENGINE_INFO.get('backend')}",
         f"prompts {EMSR.get('prompts')}", f"reference A/A probe flips: {(EMSR.get('probe_ref_aa') or {}).get('large_margin_flips')}",
         # EMS rev 2 (GPT pre-GPU audit): the identity oracle, DFlash vs target-only, dropped arms, topology
         f"gate reference: {EMSR.get('gate_reference')} | oracle flag: {EMSR.get('oracle')} | reference: {EMSR.get('ref_name')} "
         f"{EMSR.get('base_cfg')}",
         f"EMS cell error: {(EMSR.get('error') or 'none')[-600:]}",
         "tok/s = SGLang decode-line rate in a steady window (decode only, not end-to-end); adoptable = gate PASS vs the "
         "target-only oracle (PASS_EXACT, or BOUNDARY = only near-tie divergences, accepted under the documented tolerance "
         "but not exact: confirm on ARC tasks before production), complete probe, no stream error, real running share >= "
         "0.80, no canary leak, XL parity PASS; certified_exact = adoptable and PASS_EXACT",
         f"oracle canary: {EMSR.get('canary_oracle')}",
         f"notes: {json.dumps(EMSR.get('notes') or {})}",
         f"FlashInfer 0.7 overlay (PCIe-IPC arms): {json.dumps(EMSR.get('fi07_overlay'))[:400]}",
         f"PR #45151 patch: {EMSR.get('pr45151_patch')}",
         f"DFlash reference vs target-only (same INT4, greedy 256 tok): {(EMSR.get('dflash_vs_target_gate') or {}).get('verdict')}; "
         f"1,024-tok probe: identical {(EMSR.get('dflash_vs_target_probe') or {}).get('identical')} of "
         f"{(EMSR.get('dflash_vs_target_probe') or {}).get('compared')}, large-margin flips "
         f"{(EMSR.get('dflash_vs_target_probe') or {}).get('large_margin_flips')} (target-only probe A/A flips "
         f"{(EMSR.get('probe_tgt_aa') or {}).get('large_margin_flips')})",
         f"dropped arms: {json.dumps(EMSR.get('dropped') or {})}",
         f"2 x TP2 GPU pairs: {(EMSR.get('topology') or {}).get('pairs')} links {(EMSR.get('topology') or {}).get('links')}",
         "", "| arm | ok | gate | streams (8k / 40k) | capped | 8k tok/s | x ref | 40k tok/s | x ref | accept 8k / 40k | step ms 8k / 40k | KV tokens | log errors | probe flips vs ref / vs target | gate vs DFlash ref / skip / error | XL tok/s (streams, ctx/req) | x ref | accept / step ms XL | PCIe-IPC workspace built | adoptable / gate class | XL gate | canary leaks | prefix-cache hit 8k / 40k (incl. warm) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    _rx = _ref.get("xl_tok_s") or 0
    for k in EMSR.get("order") or []:
        a = EMSR["arms"][k]
        f = lambda v, d=0: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
        xr = lambda v, r: "—" if not v or not r else f"{v / r:.2f}"
        L.append(f"| {k} | {a.get('ok')} | {a.get('gate')} | {f(a.get('level_short'))} / {f(a.get('level_long') or a.get('long_per_replica'))} | "
                 f"{'CAPPED' if a.get('capped') else ''} | "
                 f"{f(a.get('short_tok_s'))} | {xr(a.get('short_tok_s'), _rs)} | {f(a.get('long_tok_s'))} | {xr(a.get('long_tok_s'), _rl)} | "
                 f"{f(a.get('accept_short'), 2)} / {f(a.get('accept_long'), 2)} | {f(a.get('step_ms_short'), 1)} / {f(a.get('step_ms_long'), 1)} | "
                 f"{a.get('kv_tokens') or a.get('kv_tokens_per_replica')} | {a.get('log_errors') or ''} | "
                 f"{f(a.get('probe_flips'))} / {f(a.get('probe_flips_vs_target'))} | "
                 f"{a.get('gate_vs_dflash_ref') or a.get('skipped') or a.get('error') or ''} | "
                 f"{f(a.get('xl_tok_s'))} ({f(a.get('level_xl'))}, {f(a.get('ctx_xl'))}){' ' + str(a.get('xl_error')) if a.get('xl_error') else ''} | "
                 f"{xr(a.get('xl_tok_s'), _rx)} | {f(a.get('accept_xl'), 2)} / {f(a.get('step_ms_xl'), 1)} | "
                 f"{(a.get('pcie_ipc') or {}).get('workspace_built', '')} | {a.get('adoptable')} {a.get('gate_class') or ''}{' certified_exact' if a.get('certified_exact') else ''} | "
                 f"{a.get('xl_gate') or ''} | {a.get('canary_leaks') if a.get('canary_leaks') is not None else ''} | "
                 f"{f(a.get('prefix_hit_short'), 2)} / {f(a.get('prefix_hit_long'), 2)} |")
    (E14_OUT / "ems_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    E14_STATE.end("done")
    e14_save()
    _pack = WORKING / f"Q27X_EMS_{globals().get('EMS_PART', 'ALL')}_BUNDLE.zip"      # rev 2g: one name per part
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(E14_OUT.rglob("*")):
            if fp.is_file() and fp.stat().st_size < 300e6:
                z.write(fp, arcname="e14/" + fp.relative_to(E14_OUT).as_posix())
        for fp in sorted(OUT.glob("sglang_server_*.log")):
            z.write(fp, arcname="logs/" + fp.name)
        for fp in (WORKING / "e14_state.json", WORKING / "e14_heartbeat.log"):
            if fp.exists():
                z.write(fp, arcname=fp.name)
    print("EMS bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("EMS report error:", traceback.format_exc()[-2000:], flush=True)
try:
    e11_kill_server(wait=False)
except Exception:
    pass
'''

cells = [S[0], c1, c1b, S[3], S[4], stub, eng, ARMS, REPORT]
md = """# EMS — SGLang engine-config matrix (Qwen3.8-27B on 4x L4), engine only

Every SGLang configuration worth testing for 600-800 tok/s, each its own launch, in one notebook (the vLLM 0.32.0 matrix
is a separate notebook). Reference = the Duck production engine (INT4, DFlash2 width 6, delayer off, mrr 17, graphs
1..17, 16 streams). Arms: target-only; MTP = EAGLE 3/1/4; NguyenThanhNhan's 65.97 flags on FP8 and on INT4; DFlash at
20 / 24 streams; the best MTP at 20; MTP 2/1/3, mem 0.86, ReplaySSM fp16 / fp32; DFlash width 4 / 8; last, 2 x TP2 (two
servers, GPU pairs from `nvidia-smi topo -m`) from a reserved time slot. rev 2: target-only (same INT4 checkpoint, no
speculation) is the correctness oracle every arm is gated against; a FAIL is recorded, never adopted.
Each arm: correctness gate first, then 8k and 40k steady windows at the level the server really runs; report table
`e14/ems_report.md` and the bundle `Q27X_EMS_BUNDLE.zip`. Commit only (the rerun writes the cell-0 fallback).

**Inputs:** competition data; RedHat INT4, DFlash2 bf16, FP8 repack (the NTN-FP8 arm and the tokenizer); the SGLang 0.5.21
wheelhouse; the vLLM wheelhouse (the tokenizer pins); the payload dataset (E14 modules).
"""
for PART in ("SAFE", "RISKY"):
    cells_p = [c.replace('"__EMS_PART__"', f'"{PART}"') for c in cells]
    assert sum(c.count(f'or "{PART}"') for c in cells_p) == 1
    md_p = md.replace("# EMS — SGLang engine-config matrix", f"# EMS {PART} — SGLang engine-config matrix", 1) + (
        "\n**rev 2f part " + PART + ":** " + ("oracle + reference + MTP 3/1/4, DFlash 20 / 24 (+ the best MTP at 20), MTP 2/1/3, "
        "mem 0.86, DFlash width 4 / 8; no patch, no overlay, no TP2." if PART == "SAFE" else
        "oracle + reference + MTP 3/1/4 (A/B partner), #45151, PCIe-IPC (DFlash / MTP), NTN INT4 / FP8, ReplaySSM fp16 / "
        "fp32, then 2 x TP2.") + " The other part is its own notebook (a crash in one cannot touch the other).\n")
    nb = json.load(open(SRC, encoding="utf-8"))
    nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": md_p.splitlines(keepends=True)}] + [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)}
        for c in cells_p]
    for i, c in enumerate(cells_p):
        compile(c, f"<cell{i}>", "exec")
    out = NBDIR / f"Q27-XHIGH-EMS_{PART}_ENGINE_MATRIX_SGLANG.ipynb"
    if out.exists() and sha(out) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
    data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    out.write_bytes(data.encode("utf-8"))
    print(out.name, sha(out)[:8], f"{len(data.encode('utf-8')) / 1024:.0f} kB", len(cells_p), "code cells")
