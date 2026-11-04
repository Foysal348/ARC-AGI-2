"""E25 NB4 TICK (spec: Experiment Review/NB4_TICK_SPEC_2026-11-01.md + the lead's addenda of 2026-11-01 and the research notes
NB4_RESEARCH_graphs_2026-11-01.md (agent D) / NB4_RESEARCH_folders_2026-11-01.md): the stage-0 gap is a ~21 ms stall INSIDE
the stage-0 FULL graph, released by the PP sampled-token broadcast of the other micro-batch (agents C / D, the folder sweep:
three independent readings of the NB1 R2 B1 traces). Decode already runs FULL graphs; a 9-request micro-batch runs eager
with CAPS16. Qwen3.8-Flash-Next W4 on Merlin v3.0.0, 4xL4 TP2xPP2+EP, hard stop 150 min.

Base (every arm = base + exactly one change, validated at build and read back at run time; the flag set is identical on
every arm, T0b included): NB1 R2 B3 = NB5 QUALITY's FP8 boot (13/15 tasks, = BF16): W4 Merlin, FP8 E4M3 QSA KV, pin
3,553,580,524 B (214 blocks), MTP K3, --language-model-only, vLLM #60490, seqs 9, CAPS16; NB1's read-only worker extension
(+ NB4 read-only methods) + dev mode, --cudagraph-metrics, VLLM_CUSTOM_SCOPES_FOR_PROFILING=1, the idle torch profiler
config (CPU + CUDA, stack / shapes / memory / flops off).

Order (lead): T0 (+ the C runtime A/B inside its boot) -> S -> S2 -> G -> P -> T0b, then K4 / INT8 only if the budget allows.
  T0   base + the host timeline (profiled windows at 8k@8 and 8k@9, the iteration log, the engine-core step timer); booted
       with VLLM_RECOVERSSM_COMMIT_GRAPH=graph and flipped to eager before any measurement, so its windows are the base;
       then C = the same-boot eager vs graph A/B (NB1 commit_ab pattern: counters + a greedy bit-identity row)
  S    CUDA_DEVICE_MAX_CONNECTIONS=32 (hardware work-queue false-dependency test; env read back in every worker)
  S2   Q-pack patch (one packed PP broadcast; agent's nb4_ppack.py hook: applied before the boot, reverted after)
  G    capture list [1..9, 12, 16, 20, 24, 28, 32, 36] (FULL for every n at seqs 9; agent D); decides on the level-9 windows
  P    P5 in the engine's TP all-reduce (agent B's nb4_p5_engine.py: .so rebuilt in the Merlin venv, 4-process gate with the
       GPUs free, strict patch, <= 184,320 B, revert after; NB4_P5_INIT_FAILED -> P5_NOT_INTEGRATED)
  T0b  the base again, 2 replicates of every window (A/A spread = max(cross-boot, within-boot))
Per arm: read-back (fail closed: CONFIG_MISMATCH / NOT_ENGAGED stop the body), probes, sampled, greedy oracle vs base, TF
2k / 8k / 40k vs base (NB0 screening), 8k@8 / 40k@8 / 8k@9 / 40k@9 steady windows (90 s; tok/s, tick, cycle, micro-batches
per s, wall clock, acceptance per position, running == level, 0 preemptions, the --cudagraph-metrics rows = n histogram,
FULL / NONE steps and the micro-batch split BALANCED / IMBALANCED, CPU per process), RecoverSSM commit counters, a short
profiled window at the compare band analysed by agent C's nb4_timeline.analyze (the lab / NB1 graph-mode parsers are not
used for verdicts). Labels: WIN_SPEED / WIN_DIAGNOSTIC (P) / NO_GAIN / SLOWER / CORRECTNESS_FAIL / CORRECTNESS_INCONCLUSIVE /
NOT_INTEGRATED / P5_NOT_INTEGRATED / NOT_ENGAGED / CONFIG_MISMATCH / BOOT_FAILED / SKIPPED_BUDGET / SKIPPED_NO_BASE /
INCONCLUSIVE_SPEED / INCONCLUSIVE_NO_AA (nb4_lib.final_label). The measured never-do list fails the build.

Reuse (byte-checked): the NB0 and NB1 builders are executed up to their notebook-writing part (`\nsubs = {`), their cell
assembly is replayed and compared byte for byte with the Kaggle-run notebooks (NB0 37a427a2, NB1 R2 62670fce); RC2's CELL
1 / 3, NB1's edited CELL 2 / 4 / 5 and named functions of NB0 CELL 6 / NB1 CELL 6 (AST picks, sha of every pick recorded)
are used verbatim with marked edits. The klee100 audit lessons (Experiment 26 Review KLEE100_LOAD_AUDIT_FIXES_2026-10-31.md)
are marked edits (K-1, R1-1, R1-6, R1-7, R1-13, R2-3) and new code (due() on every item, one OOM policy with the pin ladder, a
fresh port + launch name per try, the whole-log OOM search, try/finally verdict, live records mid-boot, the sha thread under
the lock, min(nvidia-smi, worker) free memory, full stream traces). Embedded (lzma, sha-checked): nb4_lib.py, agent C's
nb4_timeline.py, agent B's nb4_p5_engine.py + the P5 sources (NB3's p5_src), the S2 module when built.

Run: python build_e25_nb4.py  ->  ../Q27-XHIGH-E25_NB4_TICK.ipynb + nb4_code_export.py + the push folder (no push)
"""
import ast
import base64
import hashlib
import json
import lzma
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
REVIEW = HERE.parents[1] / "Experiment Review"
NB0_BUILDER = HERE / "build_e25_nb0.py"
NB1_BUILDER = HERE / "build_e25_nb1.py"
NB0_NB, NB0_NB_SHA = NBDIR / "Q27-XHIGH-E25_NB0_FP8_QUALITY.ipynb", "37a427a235bf665ee51c7a01767427a299720b4d80d6676f0ff2accb4510bf2c"
NB1_NB, NB1_NB_SHA = NBDIR / "Q27-XHIGH-E25_NB1_CAPACITY_R2.ipynb", "62670fce0e5ee950e1d32027522c5b4feffa55c43b061e4c7394668142b2bd3a"
NB1R2_OUT = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1")      # NB1 R2's real GPU output
OUT_NB = NBDIR / "Q27-XHIGH-E25_NB4_TICK.ipynb"
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb4-tick__foysalemonshanto")
PAYLOAD = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb4_payload")
PAYLOAD_SLUG = "foysalemonshanto/e25-nb4-payload"
SLUG = "q27-xhigh-e25-nb4-tick"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"   # = NB1 R2
MODELS = ["foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/Transformers/default/1",                              # = NB1 R2
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p2/Transformers/default/1",
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p3/Transformers/default/1",
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p4/Transformers/default/1",
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p5/Transformers/default/1",
          "foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
          "foysalemonshanto/e23-merlin-env/Other/v1/1"]
PUSHED = set()        # shas of NB4 versions that ran on Kaggle (never overwrite one; rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")
NB_LIMIT = 400_000
NB_TARGET = 395_000                    # inline the B / C modules only while the notebook stays below this; else payload

# ---------------------------------------------------------------- HOOKS (agents B / C / D / the Q-pack agent)
G_OVERRIDE = None                       # None = agent D's capture list (nb4_lib.G_CAPS); a dict overrides compile / decisive
DISABLED_ARMS = ()                      # data-driven arm switch (lead: NB4S may drop arms), e.g. ("S3", "K4"); the notebook env
                                        # NB4_DISABLE="S3,K4" adds to it at run time; a disabled arm is labelled DISABLED
HOOK_FILES = {"nb4_timeline.py": HERE / "nb4_timeline.py",       # agent C: analyze(src, tp) (the in-graph stall classifier)
              "nb4_p5_engine.py": HERE / "nb4_p5_engine.py",     # agent B: apply / verify / revert_p5_patch, server_env, gate
              "nb4_ppack.py": HERE / "nb4_ppack.py"}             # Q-pack agent: apply / verify / revert_ppack_patch, server_env
P5_SRC_DIR = HERE / "nb3_export" / "p5_src"                      # NB3's P5 sources (build_host_ar.py builds the .so in-notebook)
RESEARCH_NOTES = [REVIEW / "NB4_RESEARCH_graphs_2026-11-01.md", REVIEW / "NB4_RESEARCH_folders_2026-11-01.md"]

def sha(b):
    return hashlib.sha256(b).hexdigest()


def pack(b):
    return base64.b85encode(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


def builder_ns(path):
    """Execute a builder up to its notebook-writing part (`\\nsubs = {`): definitions only, nothing is written."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("\nsubs = {") == 1, path.name
    prefix = text[:text.index("\nsubs = {")]
    ns = {"__file__": str(path), "__name__": "prefix_" + path.stem}
    exec(compile(prefix, str(path), "exec"), ns)
    return ns, text


def replay_cells(ns, text, nb_path, nb_sha, label):
    """Replay the builder's own cell assembly (its `subs` + `cells_src` loop, without the write) and require the result to
    equal the code cells of the Kaggle-run notebook byte for byte: the strings NB4 picks from are what ran on GPU."""
    assert sha(nb_path.read_bytes()) == nb_sha, f"{label}: {nb_path.name} is not the Kaggle-run version"
    tail = text[text.index("\nsubs = {"):text.index("\nnb = {")]
    assert "write_bytes(" not in tail and "write_text(" not in tail, f"{label}: the replayed tail must not write"
    exec(compile(tail, f"<{label} cell assembly>", "exec"), ns)
    ran = [norm("".join(c["source"])) for c in json.loads(nb_path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    assert ran == ns["cells_src"], f"{label}: replayed cells differ from the Kaggle-run notebook"
    return len(ran)


def norm(t):
    return t.replace("\r\n", "\n")


def _targets(node):
    """Names a top-level statement defines: def / class names, assignment targets (Name ids and dotted Attribute targets
    such as L.CATS)."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        assert not node.decorator_list, node.name
        return {node.name}
    if isinstance(node, ast.Assign):
        out = set()
        for t in node.targets:
            if isinstance(t, ast.Attribute):
                out.add(ast.unparse(t))
            else:
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        return out
    return set()


def pick(src, names, label):
    """Top-level statements of `src` that define one of `names`, verbatim, in source order; every name exactly once."""
    tree = ast.parse(src)
    want, got, out, shas = set(names), {}, [], {}
    for node in tree.body:
        hit = _targets(node) & want
        if hit:
            seg = ast.get_source_segment(src, node)
            out.append(seg)
            for h in hit:
                got[h] = got.get(h, 0) + 1
            shas[",".join(sorted(hit))] = sha(seg.encode())[:16]
    missing, dup = sorted(want - set(got)), sorted(k for k, v in got.items() if v > 1)
    assert not missing and not dup, f"{label}: missing {missing} duplicate {dup}"
    return "\n\n\n".join(out), shas


# ---------------------------------------------------------------- NB0 / NB1 builder prefixes, replayed vs the Kaggle-run notebooks
N0, N0_TEXT = builder_ns(NB0_BUILDER)
N1, N1_TEXT = builder_ns(NB1_BUILDER)
N0_CELLS_SRC = replay_cells(N0, N0_TEXT, NB0_NB, NB0_NB_SHA, "NB0")
N1_CELLS_SRC = replay_cells(N1, N1_TEXT, NB1_NB, NB1_NB_SHA, "NB1 R2")
RC2 = N1["RC2"]                                  # the RC2 builder namespace (its cells checked vs the Kaggle-run RC2 notebook)
RC2_REF = N0["RC2_REF"]                          # NB0's RC2 facts: argv tails / env / pins / oracle rows / prompt shas
assert RC2_REF["fp8"]["kv_bytes"] == 2479838700 and len(RC2_REF["oracle_rows"]["rc2_fp8"]) == 4
assert N1["CAPTURE_HI"] if "CAPTURE_HI" in N1 else True
LIB = (HERE / "nb4_lib.py").read_bytes().replace(b"\r\n", b"\n")
compile(LIB, "nb4_lib.py", "exec")
sys.path.insert(0, str(HERE))
import nb4_lib as NLB                            # noqa: E402  (build-time checks only)
assert NLB.CAPS16 == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64] and "CAPTURE_HI = " + json.dumps(NLB.CAPS16) in N1_TEXT
assert NLB.B3_PIN == NLB.KV_FP8_RC2 + (1 << 30) == 3553580524
_nb5_lib = (HERE / "nb5_lib.py").read_text(encoding="utf-8")
assert 'SEQS_TABLE = {"fp8": {1.0: 9,' in _nb5_lib and NLB.SEQS == 9, "NB5 QUALITY's FP8 seqs rule changed"
_ARMS_BUILD = NLB.make_arms(None, G_OVERRIDE, None, None, DISABLED_ARMS)
assert not NLB.validate_arms(_ARMS_BUILD), NLB.validate_arms(_ARMS_BUILD)
for _a in _ARMS_BUILD.values():                 # the measured never-do list fails the build (lead 2026-11-01)
    assert not NLB.never_do(_a), (_a["key"], NLB.never_do(_a))

# NB1 R2 B3's recorded greedy oracle rows (port 8003, T 0, 128 tokens, logprobs 2): T0 vs them is a RECORDED diagnostic (seqs
# 8 -> 9 and the captures differ), never a gate; references only for identical prompts (sha checked at run time)
NB1R2_B3_ROWS = []
for _l in (NB1R2_OUT / "traces" / "requests.jsonl").read_text(encoding="utf-8").splitlines():
    _r = json.loads(_l)
    _s = _r.get("sampling") or {}
    if _r.get("port") == 8003 and _s.get("temperature") == 0.0 and _s.get("max_tokens") == 128 and _s.get("logprobs") == 2:
        NB1R2_B3_ROWS.append({k: _r[k] for k in ("text", "tokens", "margins", "error", "prompt_sha", "prompt_len", "finish_reason")})
assert [r["prompt_len"] for r in NB1R2_B3_ROWS] == [2048, 2048, 8192, 40000], "NB1 R2 B3 oracle rows"
_B3 = json.loads((NB1R2_OUT / "nb1_results.json").read_text(encoding="utf-8"))["sections"]["B3"]["nb1_b3_fp8_pin100"]
NB1R2_B3_WINDOWS = {k: {x: v.get(x) for x in ("status", "tok_s", "step_ms", "mean_acceptance_length", "running_min", "running_max")}
                    for k, v in (_B3.get("steady") or {}).items() if isinstance(v, dict) and "@" in k}
assert NB1R2_B3_WINDOWS["8k@8"]["tok_s"] == 254.6 and NB1R2_B3_WINDOWS["40k@8"]["tok_s"] == 249.5

# ---------------------------------------------------------------- hook modules, verbatim + sha-checked when present
HOOKS = {n: (p.read_bytes().replace(b"\r\n", b"\n") if p.is_file() else None) for n, p in HOOK_FILES.items()}
for _n, _b in HOOKS.items():
    if _b is not None:
        compile(_b, _n, "exec")
P5_SRC = {}
if HOOKS["nb4_p5_engine.py"] is not None:       # the P5 sources must be the ones agent B pinned (P5_SRC_SHA)
    _p5ns = {"__name__": "nb4_p5_engine_build"}
    exec(compile(HOOKS["nb4_p5_engine.py"], "nb4_p5_engine.py", "exec"), _p5ns)
    for _n, _want in _p5ns["P5_SRC_SHA"].items():
        _b = (P5_SRC_DIR / _n).read_bytes()
        assert sha(_b) == _want, f"P5 source {_n}: {sha(_b)} != agent B's pin {_want}"
        P5_SRC["p5_src/" + _n] = _b
    assert _p5ns["DEFAULT_MAX_BYTES"] >= NLB.P5_MAX_BYTES and _p5ns["server_env"]("x", "0" * 64, max_bytes=NLB.P5_MAX_BYTES)["NB4_P5_MAX_BYTES"] == str(NLB.P5_MAX_BYTES)
RESEARCH = {p.name: (sha(p.read_bytes()) if p.is_file() else None) for p in RESEARCH_NOTES}

# ---------------------------------------------------------------- RC2 cells + marked edits (NB1's edited CELL 2 / 4 / 5 verbatim)
CELL1 = RC2["CELL1"]
CELL1 = once(CELL1, '# ==================================================================== CELL 1 — config, inputs, lab module',
             '# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 NB4 marked edits)', "c1 title")
CELL1 = once(CELL1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows',
             'BUDGET = int(os.environ.get("NB4_BUDGET_SEC") or 150 * 60)    # E25 NB4: hard stop 150 min (install + T0 / G / C / P / T0b)', "c1 budget")
CELL1 = once(CELL1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))', 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4"))   # E25 NB4', "c1 out")
CELL1 = once(CELL1, 'R = {"what": "E13 RC2 Flash-Next on Merlin",', 'R = {"what": "E25 NB4 TICK: stage-0 gap timeline + FULL graph / commit graph / P5 A/B",   # E25 NB4\n     ', "c1 what")
CELL1 = once(CELL1, '        tmp = OUT / "rc2_results.json.tmp"\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "rc2_results.json")',
             '''        for _i in range(5):                                      # E25 NB4 (klee100 R1-13): another thread may mutate R mid-dump
            try:
                _txt = json.dumps(R, indent=1, default=str)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            return
        tmp = OUT / "nb4_results.json.tmp"                       # E25 NB4
        tmp.write_text(_txt)
        tmp.replace(OUT / "nb4_results.json")                    # E25 NB4''', "c1 save")
CELL1 = once(CELL1, '_SAVE_LOCK = threading.Lock()\n',
             '_SAVE_LOCK = threading.Lock()\nNB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 NB4 (NB1 hooks): per-launch trailing vllm flags, env and KV pin\n'
             'NB4_PREBUILD = {}                         # E25 NB4: per-launch env applied BEFORE Merlin build() (DRAFT_HEAD)\n', "c1 hooks")
CELL1 = once(CELL1, 'print("E13 RC2 diagnostic: competition rerun -> fallback submission.json only")',
             'print("E25 NB4 tick diagnostic: competition rerun -> fallback submission.json only")', "c1 rerun")
CELL1 = once(CELL1, 'log("RC2 | budget"', 'log("NB4 | budget"', "c1 log")


def packed_expr(text, label):
    """A verbatim module as an lzma + b85 literal whose sha256 is checked at load (NB5 / E13 pattern; keeps the commit
    notebook < 400 kB). The decompressed text is byte-identical to the source."""
    h = sha(text.encode("utf-8"))
    return (f'(lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "{h}" else '
            f'(_ for _ in ()).throw(RuntimeError("{label}: embedded source sha256 mismatch")))('
            f'__import__("lzma").decompress(__import__("base64").b85decode("{pack(text.encode("utf-8"))}")).decode("utf-8"))')


LAB_SHA = sha(RC2["LAB"].encode("utf-8"))
CELL1 = once(CELL1, '    LAB_SRC = __LAB__\n', '    LAB_SRC = ' + packed_expr(RC2["LAB"], "e23_lab") +
             '   # E25 NB4: the E23 lab verbatim (lzma, sha ' + LAB_SHA[:16] + ' checked)\n', "c1 lab packed")
CELL2 = N1["CELL2"]                              # RC2 CELL 2 + NB1's marked per-boot flag / env hooks (env after build())
CELL2 = once(CELL2, 'timeout=5400)', 'timeout=int(max(600, left() - 100 * 60)))   # E25 NB4 (klee100 R1-7): never outlives the arms', "R1-7 install timeout")
CELL2 = once(CELL2, '''        SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
        R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
        save()''', '''        with _SAVE_LOCK:                   # E25 NB4 (klee100 R1-13): never mutate R while save() dumps it
            SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
            R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
        save()''', "R1-13 sha thread under the lock")
CELL2 = once(CELL2, '                CUDA_HOME=eng["cuda_home"])\n    if kv == "fp8":',
             '                CUDA_HOME=eng["cuda_home"])\n    base.update(NB4_PREBUILD.get(name) or {})          # E25 NB4: build() knobs (DRAFT_HEAD) of this launch\n    if kv == "fp8":',
             "NB4 pre-build env hook")
CELL3 = RC2["CELL3"]
CELL4 = N1["CELL4"]                              # RC2 CELL 4 + NB1 R2 (a post-processing error never discards the boot record)
CELL4 = once(CELL4, """            if not s.alive():
                out["status"] = "SERVER_DIED"
""", """            if not s.alive():
                out["server_status"] = "SERVER_DIED"         # E25 NB4 (klee100 R2-3): a specific body status is kept
                if out.get("status") in (None, "DONE", "NOT_RUN"):
                    out["status"] = "SERVER_DIED"
""", "R2-3 with_servers status")
CELL4 = once(CELL4, """        rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
        if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
            for s in servers:
                s.kill()
            rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
""", """        out["engine_procs_before_release"] = clear_engine_procs()      # E25 NB4 (klee100 R1-6)
        rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
        if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
            for s in servers:
                s.kill()
            out["engine_procs_before_fail"] = clear_engine_procs()     # E25 NB4 (klee100 R1-6): before FAIL_GPU_RELEASE
            rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
""", "R1-6 with_servers release")
CELL4 = once(CELL4, '''        if ABORT["why"] or left() < need + 120:
            out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
''', '''        if ABORT["why"] or left() < need + 120:
            out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
        elif not due(need + 120):                         # E25 NB4 (klee100 R1-1): the boot body cap, not only the run budget
            out[k] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
''', "R1-1 levels")
CELL5 = N1["CELL5"]                              # RC2 CELL 5 definitions + NB1's per-boot KV pin hook (merlin_boot rebound in CELL 7)


# ---------------------------------------------------------------- CELL 6: reused tools, picked verbatim by AST (+ marked edits)
# NB1's pp2_trace / microbatch_costs and the lab's trace_summary verdicts are NOT picked: their graph-mode rule reads FULL
# decode forwards as PIECEWISE (agent D §0.1, the folder sweep §1.10); every trace verdict comes from nb4_timeline.analyze
NB0_PICK = ["ENV_KEYS", "flags", "_set_value", "nb0_argv", "config_diff", "cg_snapshot", "_pids", "_stat", "_io", "session_procs",
            "_roles", "load_monitor", "_post", "_req_log", "ids_sha", "_cached", "pl_arrays", "e23_rows", "tf_request",
            "cmp_last256", "chunk_check", "cmp_full", "shift_check", "screen", "TF_EST", "tf_suite", "scales_facts"]
NB1_PICK = ["psha", "P60490", "P60490_REL", "P60490_PRE", "P60490_MARKER", "NB1_EXT_SRC", "apply_unified", "apply_60490",
            "cgroup_snapshot", "ENGINE_RX", "engine_procs", "clear_engine_procs", "drop_page_cache", "host_ram_gate",
            "_boot_gate_rc2", "BOOT", "boot_gate", "shm_cleanup", "boot_left", "due", "_pick", "server_caps", "pool_facts",
            "graph_facts", "jit_kernels", "nccl_transport", "rpc_cache_info", "rpc", "commit_mode", "commit_counter_delta", "kvu",
            "nb1_complete", "lp_compare", "warmup", "accept_pos", "_prefill_done", "_req_lines", "_ttfts", "_counters", "window",
            "profiler_tools"]
_p0, PICK_SHA0 = pick(N0["CELL6"], NB0_PICK, "NB0 CELL6")
_p1, PICK_SHA1 = pick(N1["CELL6"], NB1_PICK, "NB1 CELL6")
# klee100 K-1 / R1-15 / R2-4 (the same marked edits as the klee100 load builder)
_p0 = once(_p0, '''        for run in ("fresh", "repeat"):
            if not s.alive():
                o[run] = {"status": "SKIPPED_SERVER_DEAD"}
            elif key in WD or ABORT["why"] or left() < TF_EST[band] + 240:
                o[run] = {"status": "SKIPPED_BUDGET"}
''', '''        for run in runs:                            # E25 NB4 (klee100 R2-4): non-base arms run the fresh requests only
            if not s.alive():
                o[run] = {"status": "SKIPPED_SERVER_DEAD"}
            elif ABORT["why"] or left() < TF_EST[band] + 240:        # E25 NB4 (klee100 R1-15): WD (never set) removed
                o[run] = {"status": "SKIPPED_BUDGET"}
            elif not due(TF_EST[band] + 300):          # E25 NB4 (klee100 K-1): the boot body cap, not only the run budget
                o[run] = {"status": "SKIPPED_BOOT_BUDGET"}
''', "K-1 / R1-15 / R2-4 tf_suite")
_p0 = once(_p0, 'def tf_suite(key, s, bands):\n', 'def tf_suite(key, s, bands, runs=("fresh", "repeat")):    # E25 NB4 (klee100 R2-4)\n', "R2-4 tf_suite runs")
_p0 = once(_p0, '''        if band == "90k" and f"{key}_90k" in DROP:
            o["status"] = "SKIPPED_BUDGET_PRIORITY"
            o["why"] = "lead rule: under budget pressure the MTP-off BF16 boot's 90k band is dropped first"
            out[band] = o
            continue
''', '', "R1-15 tf_suite DROP (never filled) removed")
_p1 = once(_p1, '''        return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
''', '''        if not due((secs or WIN) + 210 + n * BANDS[band] / 1500):        # E25 NB4 (klee100 R1-1)
            return {"status": "SKIPPED_BOOT_BUDGET", "boot_left_sec": round(boot_left())}
        return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
''', "R1-1 window steady")
_p1 = once(_p1, '''    g = _boot_gate_rc2()
    if g.get("ok") is not True:
        return g
''', '''    pre = clear_engine_procs()                     # E25 NB4 (klee100 R1-6): before the release wait, not only after it
    g = _boot_gate_rc2()
    g["engine_procs_before_wait"] = pre
    if g.get("ok") is not True:
        return g
''', "R1-6 boot_gate")
NB4_EDITS = {"K-1 / R1-15 / R2-4": "tf_suite: + due(TF_EST[band] + 300) -> SKIPPED_BOOT_BUDGET; WD / DROP removed; runs=",
             "R1-1": "CELL 4 levels() + window() steady path: + due() -> SKIPPED_BOOT_BUDGET",
             "R1-6": "CELL 4 with_servers + NB1 boot_gate: clear_engine_procs() before every GPU-release wait",
             "R1-7": "CELL 2 merlin_install timeout = max(600, left - 100 min)",
             "R1-13": "CELL 1 save() retries a dump that races a mutation; CELL 2 the sha thread mutates R under _SAVE_LOCK",
             "R2-3": "CELL 4 with_servers: a specific body status is kept (server_status SERVER_DIED)",
             "NB4 pre-build env": "CELL 2 merlin_fn_server: base.update(NB4_PREBUILD[name]) before Merlin build() (DRAFT_HEAD)",
             "worker extension": "NB1WorkerExt + nb4_env (os.environ read-back per worker) + nb4_p5_stats (agent B's backend)",
             "rebound in CELL 7": "merlin_fn_server, merlin_boot, watched, L.template_probe (the RC2 / NB1 definitions stay verbatim before them)"}

# NB1's read-only worker extension + two NB4 read-only methods (the NB1 notebooks / builder are not touched)
NB4_EXT_ADD = r'''


# ==================================================================== E25 NB4: read-only additions
def nb4_env(self, keys=""):
    """os.environ of this worker for the given comma-separated keys (CUDA reads CUDA_DEVICE_MAX_CONNECTIONS at context
    creation: arm S's change must be visible in every worker) + this worker's pid."""
    import os as _os
    out = {"pid": _os.getpid()}
    try:
        import torch.distributed as dist
        out["rank"] = dist.get_rank() if dist.is_initialized() else None
    except Exception as e:
        out["rank_error"] = repr(e)[:120]
    ks = [k for k in str(keys).split(",") if k] or [k for k in _os.environ if k.startswith(("NCCL_", "CUDA_", "NB4_", "OMP_"))]
    out["env"] = {k: _os.environ.get(k) for k in ks}
    return out


def nb4_p5_stats(self, read_device="1"):
    """Agent B's P5 engaged counters (nb4_p5_backend.stats_json); call between windows only (a synchronous device read)."""
    import json as _json
    try:
        import nb4_p5_backend as b
        return b.stats_json(read_device=str(read_device) == "1")
    except Exception as e:
        return _json.dumps({"error": f"{type(e).__name__}: {e}"})


NB1WorkerExt.nb4_env = nb4_env
NB1WorkerExt.nb4_p5_stats = nb4_p5_stats
'''
NB4_EXT = N1["NB1_EXT"] + NB4_EXT_ADD
compile(NB4_EXT, "<nb1_worker_ext + nb4>", "exec")

CELL6 = r'''# ==================================================================== CELL 6 — reused tools, verbatim (NB0 CELL 6, NB1 CELL 6) + marked edits
import statistics
import numpy as np
import urllib.request as UR
import urllib.error as UE
from collections import Counter
R.pop("capacity_plan", None)          # RC2-only records written by RC2's CELL 5 definitions
R.pop("correctness_meaning", None)
RC2_REF = json.loads(__RC2REF__)      # NB0's build-time RC2 facts: argv tails / env / pins / oracle rows / prompt shas
# the globals the picked NB0 / NB1 functions read (NB0 CELL 6 values; NB1 CELL 6 values)
NB0 = {"topk": int(os.environ.get("NB0_TF_TOPK") or 5), "sc_tok": 256, "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6},
       "tf_bands": ["2k", "8k", "40k"], "sc_bands": [], "last": 256, "boot_watchdog_sec": 1800.0, "ready_sec": 1500.0,
       "host_floor_mib": (69 + 4) * 1024}
SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # NB0 (external audit 2): conservative screening, not a published limit
TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # NB0 (external audit 3): severe-discrepancy triage
RAW_DIR = OUT / "tf_raw"
RAW_DIR.mkdir(exist_ok=True)
NB0_REQ = TRACES / "nb4_requests.jsonl"        # every non-streaming request (teacher-forced) with its meta + generated text
TFR, SCR, LOADMON, SIDS = {}, {}, {}, set()
PROC = Path(os.environ.get("NB0_PROC", "/proc"))
CG = Path(os.environ.get("NB0_CGROUP", "/sys/fs/cgroup"))
HOSTMEM = OUT / "hostmem"
HOSTMEM.mkdir(exist_ok=True)
BLOCK, PP, GEN_RATE = 4096, 2, 45.0           # NB1: block size, pipeline stages (one engine step = one micro-batch), window growth
RAM_NEED_MIB = (69 + 4) * 1024                 # Merlin's ~69 GiB resident floor + 4 GiB margin (NB1 host_ram_gate)
EXT_FLAG = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
DIAG_ENV = {"VLLM_SERVER_DEV_MODE": "1"}      # /collective_rpc (read-only extension) + /reset_prefix_cache; 127.0.0.1 only


''' + _p0 + "\n\n\n" + _p1 + "\nsave()\n"

# ---------------------------------------------------------------- CELL 7: NB4 (new code)
CELL7 = r'''# ==================================================================== CELL 7 — E25 NB4: arms, hooks, launches, bodies, analysis (NEW)
import lzma as _lzma
import base64 as _b64
import gzip as _gzip
NB4 = R.setdefault("nb4", {})
NB4_EMB = json.loads(__NB4EMB__)          # build-time files [sha256, lzma + b85]: nb4_lib.py, the hook modules, the P5 sources
NB4_PIN = json.loads(__NB4PIN__)          # build-time facts: shas of the picks / hooks / research notes, NB1 R2 B3 references
NB4_EMB_DIR = TMP / "nb4_emb"
_emb_ok = {}
for _n, (_want, _blob) in NB4_EMB.items():
    _raw = _lzma.decompress(_b64.b85decode(_blob))
    _emb_ok[_n] = hashlib.sha256(_raw).hexdigest() == _want
    (NB4_EMB_DIR / _n).parent.mkdir(parents=True, exist_ok=True)
    (NB4_EMB_DIR / _n).write_bytes(_raw)
NB4["embedded_sha_ok"] = _emb_ok
if not all(_emb_ok.values()):
    raise RuntimeError("NB4_EMBEDDED_SHA_MISMATCH: " + json.dumps(_emb_ok))
sys.path.insert(0, str(NB4_EMB_DIR))
import nb4_lib as NL                      # E25 NB4 pure helpers (unit-tested): arms, labels, winner rule, engaged checks, planner


def nb4_hook_module(name):
    """A hook module: the embedded copy (sha-checked above), else the payload dataset copy whose sha256 the build pinned,
    else None (the arm / analysis is recorded as pending, never faked)."""
    want = (NB4_PIN.get("hooks") or {}).get(name)
    if not want:
        return None, f"{name}: not built at build time (hook pending)"
    p = NB4_EMB_DIR / name
    if not p.is_file():
        p = next((d / name for d in (INP / "datasets" / "foysalemonshanto" / "e25-nb4-payload", INP / "e25-nb4-payload")
                  if (d / name).is_file()), None)
    if p is None or hashlib.sha256(p.read_bytes()).hexdigest() != want:
        return None, f"{name}: missing or sha256 mismatch ({p})"
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(name[:-3], str(p))
    mod = _iu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, f"{name}: import error {type(e).__name__}: {e}"[:300]
    return mod, str(p)


NB4_TL, NB4["timeline_module"] = nb4_hook_module("nb4_timeline.py")
NB4_P5, NB4["p5_module"] = nb4_hook_module("nb4_p5_engine.py")
NB4_PK, NB4["ppack_module"] = nb4_hook_module("nb4_ppack.py")
NB4_DISABLED = sorted(set(NB4_PIN.get("disabled") or []) | {x.strip() for x in os.environ.get("NB4_DISABLE", "").split(",") if x.strip()})
ARMS = NL.make_arms(None, NB4_PIN.get("g_override"), None, None, NB4_DISABLED)
NB4["disabled"] = NB4_DISABLED
for _k, _m in (("P", NB4_P5), ("S2", NB4_PK), ("S3", NB4_PK)):   # a patch arm with its module present: booted after its prepare
    if _m is not None and _k not in NB4_DISABLED:
        ARMS[_k].update(integrated=True, why_not=None, pending_prepare=True)
NB4["arm_problems"] = NL.validate_arms(ARMS)
NB4["base"] = NL.BASE
NB4["research"] = NB4_PIN.get("research")
NB4["nb1r2_b3_windows"] = NB4_PIN.get("nb1r2_b3_windows")
NB4_PROF_DIR = TMP / "nb4_profile"                     # one profiler dir for every arm (an identical flag value)
NB4_TIMER = TMP / "nb4_core_timer"                     # T0: the step timer appends to nb4_core_timer.<pid>.jsonl
(OUT / "timeline").mkdir(exist_ok=True)
DONE, RECS, GATES = {}, {}, {}
NB4["ref"] = None                                      # the reference arm (T0, or T0b when T0 is not a clean base)
NB4["plan_log"] = []
LAUNCH = {}                                            # launch name -> {"arm", "try", "port", "rung"}
TRACE_CTX = {"collect": None}
for _b in ("8k", "40k"):                               # level 9 needs 9 distinct prompts (RC2 prompt(), variant 8 = new)
    PROMPTS[_b] += [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), 9)]
R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
_osha = [psha(p)[:16] for p in ORACLE_P + LONG_P]     # RC2's own prompt_sha formula (16 hex)
NB4["oracle_prompt_identity"] = {"rc2_fp8": [r["prompt_sha"] for r in RC2_REF["oracle_rows"]["rc2_fp8"]] == _osha,
                                 "nb1r2_b3": [r["prompt_sha"] for r in NB4_PIN["nb1r2_b3_rows"]] == _osha}
if NB4["oracle_prompt_identity"]["rc2_fp8"] and not FAKE:      # recorded diagnostics only (never in need=)
    ROWS["rc2_fp8"] = [{k: r[k] for k in ("text", "tokens", "margins", "error")} for r in RC2_REF["oracle_rows"]["rc2_fp8"]]
if NB4["oracle_prompt_identity"]["nb1r2_b3"] and not FAKE:
    ROWS["nb1r2_b3"] = [{k: r[k] for k in ("text", "tokens", "margins", "error")} for r in NB4_PIN["nb1r2_b3_rows"]]


# ---- A0 additions: the engine-core step timer (a dormant vLLM general plugin)
def _tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python", eng


def core_timer_install():
    """Writes nb4_core_timer.py + its dist-info (entry point vllm.general_plugins) next to the Merlin tree's vllm/ (on the
    server's PYTHONPATH) and checks with the Merlin venv that the entry point resolves. Dormant unless NB4_CORE_TIMER is
    set (T0 only), and only in the EngineCore process. Not a gate: a failure leaves T0 with vLLM's iteration log only."""
    if FAKE:
        return {"ok": False, "why": "local harness: no Merlin tree"}
    tree, py, _ = _tree_py()
    (tree / (NL.CORE_TIMER_MODULE + ".py")).write_text(NL.CORE_TIMER_SRC)
    for rel, txt in NL.CORE_TIMER_DIST.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(txt)
    envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    envp["PYTHONPATH"] = str(tree)
    cp = subprocess.run([py, "-c", "import importlib.metadata as m, importlib.util as u; "
                         "print([e.value for e in m.entry_points(group='vllm.general_plugins')]); "
                         "print(u.find_spec('nb4_core_timer').origin)"], capture_output=True, text=True, timeout=300, env=envp)
    out = cp.stdout.strip().splitlines()
    ok = cp.returncode == 0 and len(out) >= 2 and "nb4_core_timer:register" in out[-2] and out[-1].startswith(str(tree))
    return {"ok": ok, "rc": cp.returncode, "stdout": out[-2:], "stderr": cp.stderr[-400:], "tree": str(tree),
            "sha256": hashlib.sha256(NL.CORE_TIMER_SRC.encode()).hexdigest()}


# ---- the patch arms: P (agent B's P5) and S2 (the Q-pack agent), applied before the boot, reverted after
def _build_env(eng, cc=True):
    kvv = eng["kv"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH")}
    env.update(CUDA_HOME=eng["cuda_home"], PATH=f"{kvv['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
               LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for k in ("CC", "CXX", "CUDAHOSTCXX"):
        env.pop(k, None)
    if cc and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
        env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
        lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
        if (lib13 / "libstdc++.so.6").exists():
            env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def p5_prepare():
    """Arm P (agent B): (1) the P5 .so rebuilt from the embedded NB3 sources with NB3's build_host_ar.py in the Merlin venv
    (Merlin's GCC 13 first, then the system compiler); (2) nb4_p5_gate.py: 4 processes, both TP pairs concurrently, eager +
    graph replays, bit-exact vs the ordered fp32 reference, while the GPUs are free; (3) the strict cuda_communicator.py
    patch (pre / post sha, marker, py_compile, origin). Any miss -> P5_NOT_INTEGRATED (the boot never starts)."""
    rec = {"ok": False}
    if NB4_P5 is None:
        return dict(rec, why=str(NB4.get("p5_module")))
    if FAKE:
        return dict(rec, why="local harness: no Merlin tree / GPUs")
    tree, py, eng = _tree_py()
    src = NB4_EMB_DIR / "p5_src"                      # embedded with the module, else next to the payload copy
    if not src.is_dir():
        src = Path(str(NB4.get("p5_module"))).parent / "p5_src"
    rec["src"] = str(src)
    bad = [n for n, h in NB4_P5.P5_SRC_SHA.items() if not (src / n).is_file() or hashlib.sha256((src / n).read_bytes()).hexdigest() != h]
    if bad:
        return dict(rec, why=f"P5 sources missing / changed: {bad}")
    outd = TMP / "nb4_p5"
    tries = []
    for mode in ("merlin_cc", "system_cc"):
        if not due(600) and left() < 30 * 60:
            break
        cp = subprocess.run([py, str(src / "build_host_ar.py"), "--out", str(outd), "--build-dir", str(TMP / f"nb4_p5_build_{mode}"),
                             "--src", str(src)], capture_output=True, text=True, timeout=900, env=_build_env(eng, mode == "merlin_cc"))
        txt = cp.stdout + cp.stderr
        (LOGS / f"p5_build_{mode}.log").write_text(txt)
        bj = next((json.loads(l[len("BUILD_JSON "):]) for l in txt.splitlines() if l.startswith("BUILD_JSON ")), {})
        tries.append({"mode": mode, "rc": cp.returncode, "ok": bj.get("ok"), "so": bj.get("so"), "so_sha256": bj.get("so_sha256"),
                      "error": (bj.get("error") or "")[-600:]})
        if bj.get("ok") and bj.get("so"):
            break
    rec["build"] = tries
    b = next((t for t in tries if t.get("ok") and t.get("so")), None)
    if not b:
        return dict(rec, why="P5 .so build failed (logs/p5_build_*.log)")
    so = b["so"]
    so_sha = hashlib.sha256(Path(so).read_bytes()).hexdigest()
    pay = NB4_P5.write_payload(TMP / "nb4_p5_pay")
    rec["payload"] = pay
    gate_out = OUT / "p5_gate.json"
    g = boot_gate()                                   # the GPUs must be free (the previous arm released them)
    rec["gate_boot_gate"] = {k: g.get(k) for k in ("ok", "why")}
    if g.get("ok") is not True:
        return dict(rec, why="GPUs not free for the P5 gate")
    cp = subprocess.run(NB4_P5.gate_argv(py, pay["nb4_p5_gate.py"]["path"], TMP / "nb4_p5_pay", so, so_sha, gate_out,
                                         max_bytes=NL.P5_MAX_BYTES), capture_output=True, text=True, timeout=1200,
                        env=_build_env(eng, False))
    (LOGS / "p5_gate.log").write_text(cp.stdout + cp.stderr)
    gj = next((json.loads(l[len("NB4_P5_GATE_JSON "):]) for l in (cp.stdout + cp.stderr).splitlines()
               if l.startswith("NB4_P5_GATE_JSON ")), None)
    rec["gate"] = {"rc": cp.returncode, "verdict": (gj or {}).get("verdict"), "summary": {k: v for k, v in (gj or {}).items()
                                                                                          if k not in ("rows", "sizes")}}
    if cp.returncode != 0:
        return dict(rec, why=f"P5 gate not PASS (rc {cp.returncode}, logs/p5_gate.log)")
    rec["patch"] = NB4_P5.apply_p5_patch(tree, py)
    rec.update(ok=True, env=NB4_P5.server_env(so, so_sha, max_bytes=NL.P5_MAX_BYTES), init_rx=r'NB4_P5 \{.*"event": "init"',
               counters_rpc="nb4_p5_stats", so=so, so_sha256=so_sha)
    return rec


def p5_finish():
    """Back to the exact Merlin file (agent B's reverse patch, sha == CC_PRE) for every later arm (T0b)."""
    if NB4_P5 is None or FAKE:
        return {"skipped": True}
    tree, py, _ = _tree_py()
    try:
        return NB4_P5.revert_p5_patch(tree, py)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


def ppack_prepare(variant="Q-pack"):
    """Arms S2 (Q-pack) / S3 (Q-defer) (nb4_ppack.py): apply_ppack_patch(tree, py) (pp_utils.py + model_runner.py, strict
    pre / post sha, marker, py_compile), server_env(variant) = {NB4_PPACK: 1} / {NB4_PPDEFER: 1}; the patched workers log
    'NB4_PPQ {json}' init / stats lines (readback_label, all 4 ranks must agree). Missing pieces / a refused patch ->
    NOT_INTEGRATED."""
    if NB4_PK is None:
        return {"ok": False, "why": str(NB4.get("ppack_module"))}
    if FAKE:
        return {"ok": False, "why": "local harness: no Merlin tree"}
    need = [n for n in ("apply_ppack_patch", "revert_ppack_patch", "server_env", "readback_label") if not hasattr(NB4_PK, n)]
    if need:
        return {"ok": False, "why": f"nb4_ppack.py lacks {need}"}
    tree, py, _ = _tree_py()
    rec = {"patch": NB4_PK.apply_ppack_patch(tree, py)}
    if not (rec["patch"] or {}).get("ok"):
        return dict(rec, ok=False, why="Q-pack patch not ok")
    rec.update(ok=True, env=NB4_PK.server_env(variant), init_rx=r'NB4_PPQ \{.*"event":\s*"init"', variant=variant)
    return rec


def ppack_finish():
    if NB4_PK is None or FAKE or not hasattr(NB4_PK, "revert_ppack_patch"):
        return {"skipped": True}
    tree, py, _ = _tree_py()
    try:
        return NB4_PK.revert_ppack_patch(tree, py)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


def patch_states():
    """Read-only: the P5 / Q-pack file states in the installed tree (every arm reads back 'not_applied' but its own)."""
    out = {}
    if FAKE or MERLIN_ENG is None or (MERLIN_ENG or {}).get("failed"):
        return out
    tree = Path(MERLIN_ENG["kv"]["MERLIN_TREE"])
    for k, m, fn in (("p5", NB4_P5, "verify_p5_patch"), ("ppack", NB4_PK, "verify_ppack_patch")):
        if m is not None and hasattr(m, fn):
            try:
                out[k] = getattr(m, fn)(tree).get("state")
            except Exception as e:
                out[k] = f"ERROR {type(e).__name__}: {e}"[:200]
    return out


# ---- launches: per-launch flags / env (NB1 hooks), NB0's in-place argv edits, a fresh port + name per try, the pin ladder
_merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 (launch.build + host_phase) with NB1's per-boot hooks


def nb4_plan(arm_key, try_i, rung=0, drop_commit=False):
    a = ARMS[arm_key]
    name, port = NL.launch_name(arm_key, try_i), NL.port_for(arm_key, try_i)
    env = NL.arm_env(a, DIAG_ENV, NB4_TIMER)
    if drop_commit:
        env.pop("VLLM_RECOVERSSM_COMMIT_GRAPH", None)
    LAUNCH[name] = {"arm": arm_key, "try": try_i, "port": port, "rung": rung, "drop_commit": drop_commit}
    NB1_EXTRA[name], NB1_ENV[name], NB1_KV[name] = NL.arm_extra(a, EXT_FLAG, NB4_PROF_DIR), env, NL.pin_for(a, rung)
    NB4_PREBUILD[name] = dict(a.get("prebuild_env") or {})
    return name, port


def merlin_fn_server(name, kv, port, kv_bytes):
    """E25 NB4: RC2's Merlin plan with NB1's per-launch flags / env, then NB0's in-place edits (seqs 9, this arm's
    compilation config, K4's speculative config: the existing flags' values, no duplicate flags), the help-flag gate on the
    FINAL argv (klee100 K-9), the plan (argv + env + D8 switches) recorded, NB0's load monitor keyed by the launch name."""
    a = ARMS[LAUNCH[name]["arm"]]
    s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
    if FAKE:                                   # local harness: RC2's recorded Kaggle argv into the fake (NB0 rule)
        tail, out = RC2_REF[kv]["argv_tail"], []
        for i, x in enumerate(tail):
            out.append(str(port) if i and tail[i - 1] == "--port" else str(kv_bytes) if i and tail[i - 1] == "--kv-cache-memory-bytes" else x)
        s.argv = [sys.executable, FAKE, str(CKPT), "--host"] + out + list(NB1_EXTRA.get(name) or [])
    s.argv = nb0_argv(s.argv, {"mtp": True, "seqs": NL.BASE["seqs"], "captures": NL.compile_json(a)})
    if a.get("spec"):
        _set_value(s.argv, "--speculative-config", a["spec"])
    if not FAKE:
        HELP[name] = L.help_check("merlin", s.argv[:s.argv.index("serve") + 1], s.env, s.argv)
        h = HELP[name]
        if isinstance(h, dict) and h.get("ran") is True and h.get("missing"):
            NB4.setdefault("cli_flags", {})[name] = h
            save()
            raise RuntimeError("FAIL_CLI_FLAGS: " + " ".join(h["missing"]))
    env_rec = {k: s.env[k] for k in sorted(s.env) if k.startswith(ENV_KEYS + ("MERLIN_", "NB4_", "TORCH_", "PYTORCH_", "CUDA_", "OMP_", "DRAFT_"))}
    R["setup"].setdefault("nb4_plans", {})[name] = {"arm": a["key"], "change": a["change"], "argv_tail": s.argv[-72:],
                                                    "argv": s.argv, "kv_bytes": kv_bytes, "env": env_rec, "d8": NL.d8_switches(s.env),
                                                    "prebuild_env": NB4_PREBUILD.get(name),
                                                    "diff_vs_rc2_fp8": config_diff(s.argv, s.env, kv_bytes, "fp8")}
    save()
    if not os.environ.get("RC2_CPU_KIT"):
        deadline = time.time() + max(120, min(NL.READY_SEC + 120, left() - 240))
        threading.Thread(target=load_monitor, args=(s, name, deadline), daemon=True).start()
    return s


def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
    """E25 NB4 (rebinds RC2 CELL 5's merlin_boot; same contract and gates). ONE OOM policy (nb4_lib.retry_decision): a
    launch OOM steps down the pin ladder on non-base arms (B3 -> -0.25 GiB -> the FP8 base pin), a base arm keeps its pin;
    a device fault stops; a RecoverSSM capture failure of T0 retries without the commit env (the C A/B is then dropped); a
    fast non-OOM launch failure is retried once. Every retry = its own launch (fresh name + port: klee100 R1-9 / R2-5); OOM
    / device-fault lines are searched in the WHOLE log (R1-8); a server death is named by what was running (R1-10)."""
    arm_key = LAUNCH[name]["arm"]
    tries, lname, lport, rung = [], name, port, LAUNCH[name]["rung"]
    while True:
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname}
        shm0 = shm_names()
        t0 = time.time()
        BOOT["tag"] = lname
        srv_obj = merlin_fn_server(lname, kv, lport, NB1_KV.get(lname) or NL.BASE["kv_bytes"])
        o = with_servers([srv_obj], timeout, body)
        try:
            full = srv_obj.log() or ""
        except Exception:
            full = ""
        o.update(kv_bytes=NB1_KV.get(lname), pin_rung=rung, boot_gate=gate, shm_cleanup=shm_cleanup(shm0), launch_name=lname,
                 port=lport, ready_sec=getattr(srv_obj, "ready_sec", None), oom_in_log=bool(OOM_ERR.search(full)),
                 xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in full.splitlines() if OOM_ERR.search(l)][:6],
                 commit_env_dropped=LAUNCH[lname].get("drop_commit"))
        tries.append({"launch_name": lname, "port": lport, "kv_bytes": NB1_KV.get(lname), "status": o.get("status"),
                      "oom": o["oom_in_log"], "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"], "sec": round(time.time() - t0, 1)})
        retry, rung2, drop, why = NL.retry_decision(dict(ARMS[arm_key], env=NB1_ENV.get(lname)), o.get("status"), o["oom_in_log"],
                                                    o["xid_in_log"], time.time() - t0, len(tries), rung, left(),
                                                    NL.EST_MIN[arm_key] * 60 + 300, full[-20000:])
        tries[-1]["retry"] = why
        if not retry:
            break
        rung = rung2
        lname, lport = nb4_plan(arm_key, len(tries), rung, drop or LAUNCH[lname].get("drop_commit"))
        log("NB4 launch retry:", name, "->", lname, "port", lport, "pin", NB1_KV[lname], "|", why)
    o["tries"] = tries
    if o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]:
        o["status"] = "LAUNCH_FAILED_OOM"
    if arm_key == "P" and str(o.get("status") or "").startswith("LAUNCH_FAILED") and NB4_P5 is not None:
        lab = NB4_P5.p5_arm_label(full)
        o["p5_label"] = lab
        if "init_failed" in " ".join(lab[1]):
            o["status"] = "P5_NOT_INTEGRATED_INIT_FAILED"       # agent B: NB4_P5_INIT_FAILED -> NOT_INTEGRATED
    if o.get("server_status") == "SERVER_DIED" or o.get("status") == "SERVER_DIED":
        o["death"] = NL.death_label(o.get("boot_watchdog"), o["oom_in_log"], o["xid_in_log"], o.get("died_in"))
        if o.get("status") in (None, "DONE", "NOT_RUN", "SERVER_DIED", "BODY_ERROR"):
            o["status"] = o["death"]
    if str(o.get("status", "")).startswith(("LAUNCH_FAILED", "P5_NOT_INTEGRATED")):
        return o
    return gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)


def watched(body, cap_sec, name):
    """E25 NB4 (rebinds NB1's watched; klee100 R1-2): the body cap is clipped at body start to left() - 480 s (teardown +
    verdict); at cap + 300 s the server is killed (hard backstop); the record carries body_cap_sec / boot_watchdog."""
    def b(ss):
        cap = max(60.0, min(float(cap_sec), left() - 480))
        BOOT.update(deadline=time.time() + cap, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        o["body_cap_sec"] = round(cap)
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
        return o
    return b


class GpuSampler(threading.Thread):
    """nvidia-smi every 5 s while a body runs (klee100 R1-14): the highest used MiB per card."""
    def __init__(self, every=5.0):
        super().__init__(daemon=True)
        self.every, self.ev, self.max, self.tot, self.n = every, threading.Event(), {}, {}, 0

    def run(self):
        while not self.ev.is_set():
            for i, u, t in (L.gpu_mem() or []):
                self.max[i], self.tot[i] = max(self.max.get(i, 0), u), t
            self.n += 1
            self.ev.wait(self.every)

    def rows(self):
        return [[i, self.max[i], self.tot[i]] for i in sorted(self.max)]


# ---- full traces (klee100 K-8 / R2-10): every stream of every window, the probes' full answers, the template ids
def nb4_trace_install():
    if getattr(L.Stream, "_nb4_traced", False):
        return
    init = L.Stream.__init__

    def _init(self, *a, **k):
        init(self, *a, **k)
        c = TRACE_CTX["collect"]
        if c is not None:
            c.append(self)
    L.Stream.__init__ = _init
    L.Stream._nb4_traced = True


def nb4_trace(label, streams):
    path = TRACES / f"{BOOT.get('tag') or 'boot'}_{label}.jsonl"
    try:
        with _SAVE_LOCK, open(path, "a") as fh:
            for i, g in enumerate(streams):
                row = g.row()
                body = getattr(g, "body", None) or {}
                fh.write(json.dumps({"boot": BOOT.get("tag"), "window": label, "i": i, "port": getattr(g, "port", None),
                                     "prompt_tokens": len(body.get("prompt") or []), "prompt_sha": psha(body.get("prompt") or [])[:16],
                                     "seed": body.get("seed"), "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"),
                                     "text": row["text"], "tokens": row["tokens"], "finish_reason": row["finish_reason"],
                                     "error": row["error"], "done": row["done"], "t": time.time()}) + "\n")
        NB4.setdefault("traces", {})[path.name] = NB4.get("traces", {}).get(path.name, 0) + len(streams)
    except Exception as e:
        NB4.setdefault("trace_errors", []).append(f"{label}: {type(e).__name__}: {e}"[:200])


def traced(label, fn):
    got = []
    TRACE_CTX["collect"] = got
    try:
        return fn()
    finally:
        TRACE_CTX["collect"] = None
        nb4_trace(label, got)


nb4_trace_install()


def chat_probes_full(port, tag="probes"):
    """The lab's chat_probes request (thinking off, T 0, 32 tokens, the same contains-match); the FULL content and
    reasoning of every case go to traces/chat_probes.jsonl (klee100 R2-10)."""
    cases = (("Reply with exactly one number: 7 * 6 =", "42"), ("Reply with exactly one number: 2 + 2 =", "4"),
             ("Reply with exactly one number: 9 - 3 =", "6"))
    out = []
    for q, want in cases:
        body = {"model": "q27", "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False}}
        rec, trace = {"q": q, "want": want}, {}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=300).read())
            msg = r["choices"][0]["message"]
            full = (msg.get("content") or "").strip()
            rec.update(content=full[:200], reasoning_chars=len(msg.get("reasoning") or ""), finish=r["choices"][0].get("finish_reason"),
                       ok=re.search(r"(?<!\d)" + re.escape(want) + r"(?!\d)", full) is not None, garbage=L.degenerate(full))
            trace = {"content_full": full, "reasoning_full": msg.get("reasoning")}
        except Exception as e:
            rec.update(error=repr(e)[:200], ok=False)
        out.append(rec)
        try:
            with _SAVE_LOCK, open(TRACES / "chat_probes.jsonl", "a") as fh:
                fh.write(json.dumps(dict(rec, **trace, boot=BOOT.get("tag"), tag=tag, t=time.time())) + "\n")
        except Exception:
            pass
    return {"ok": all(x["ok"] for x in out), "cases": out}


_template_probe_lab = L.template_probe


def nb4_template_probe(port, messages, want_ids, model="q27", timeout=120):
    """The lab's template_probe (unchanged result) + the server's own token ids saved in traces/ (klee100 R2-10)."""
    res = _template_probe_lab(port, messages, want_ids, model=model, timeout=timeout)
    try:
        body = {"model": model, "messages": messages, "add_generation_prompt": True}
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout).read())
        p = TRACES / f"template_probe_{BOOT.get('tag') or 'boot'}.json"
        p.write_text(json.dumps({"server_tokens": list(r.get("tokens") or []), "want_tokens": list(want_ids), "result": res}))
        res["server_tokens_file"] = p.name
    except Exception as e:
        res["server_tokens_error"] = repr(e)[:200]
    return res


L.template_probe = nb4_template_probe


# ---- the arm body
def run_item(o, s, label, fn, need_sec=None):
    """Every item: gated on the boot body cap (due(need_sec) -> SKIPPED_BOOT_BUDGET); o["items"] in order; o["died_in"] =
    the first item that began with the server alive and ended with it dead (it names a death, klee100 R1-10)."""
    if need_sec is not None and not due(need_sec):
        o.setdefault("items_skipped", []).append(label)
        return {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need_sec), "boot_left_sec": round(boot_left())}
    alive0 = s.alive()
    o.setdefault("items", []).append(label)
    try:
        return fn()
    finally:
        if alive0 and not s.alive() and "died_in" not in o:
            o["died_in"] = label
        if LIVE.get("rec") is not None:
            save()                                     # klee100 R1-12: the live record (o) is saved after every item


def session_env(s, keys):
    """/proc/<pid>/environ of every process of the server session (workers, EngineCore, the PLE offload process): the
    values of `keys` per process (S: CUDA_DEVICE_MAX_CONNECTIONS must reach every CUDA process)."""
    out = {}
    if s.proc is None or not Path("/proc/self/environ").exists():
        return out
    for p in session_procs({s.proc.pid}):
        try:
            env = dict(x.split("=", 1) for x in (PROC / str(p["pid"]) / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in x)
            out[f"{p['pid']}:{p.get('comm')}"] = {k: env.get(k) for k in keys}
        except Exception as e:
            out[f"{p['pid']}:{p.get('comm')}"] = {"error": repr(e)[:80]}
    return out


def nb4_facts(s, a, o):
    """The config read-back from the server's own log + the worker extension + the session processes' env (fail closed)
    and the "exactly one change" check vs the reference launch (the base arm's recorded plan)."""
    t = s.log() or ""
    o["caps"], o["pool"], o["graphs"], o["nccl"] = server_caps(t, s.argv), pool_facts(t), graph_facts(t), nccl_transport(t)
    o["workers"] = rpc_cache_info(s.port)
    o["scales"] = scales_facts(t)
    keys = ["CUDA_DEVICE_MAX_CONNECTIONS", "VLLM_RECOVERSSM_COMMIT_GRAPH", "NB4_P5", "NB4_PPACK", "OMP_NUM_THREADS",
            "VLLM_CUSTOM_SCOPES_FOR_PROFILING", "PYTORCH_CUDA_ALLOC_CONF"]
    wr = (rpc(s.port, "nb4_env", ",".join(keys)) or {}).get("results") or []
    o["worker_env"] = wr
    o["session_env"] = session_env(s, keys)
    nd_line = next((l for l in t.splitlines() if "non-default args:" in l), "")
    nd = NL.parse_nondefault(nd_line)
    plan = (R["setup"].get("nb4_plans") or {}).get(BOOT.get("tag")) or {}
    ck = NL.readback_checks(nd, plan.get("env"), a, NB1_KV.get(BOOT.get("tag")) or NL.BASE["kv_bytes"])
    if a.get("env", {}).get("VLLM_RECOVERSSM_COMMIT_GRAPH") and LAUNCH.get(BOOT.get("tag"), {}).get("drop_commit"):
        ck.pop("env VLLM_RECOVERSSM_COMMIT_GRAPH", None)              # T0 relaunched without it (the C A/B is dropped)
    ck["static_scales_applied"] = o["scales"].get("ok") is True
    ck["patch_60490"] = ((R["sections"].get("A0") or {}).get("patch_60490") or {}).get("ok") is True or bool(FAKE)
    ps = patch_states()
    o["patch_states"] = ps
    for pk, arm_k in (("p5", "P"), ("ppack", "S2")):
        if pk in ps:
            ck[f"{pk} patch {'applied' if a['key'] == arm_k else 'not applied'}"] = ps[pk] == ("applied" if a["key"] == arm_k else "not_applied")
    ref = NB4.get("ref")
    refrec = RECS.get(ref) if ref else None
    refplan = (R["setup"].get("nb4_plans") or {}).get((refrec or {}).get("launch_name")) if refrec else None
    if refplan and ref != a["key"]:
        d = NL.diff_vs_ref(refplan.get("argv"), refplan.get("env"), plan.get("argv"), plan.get("env"))
        oc = NL.one_change_ok(d, a, dict(ARMS[ref], env={k: v for k, v in (ARMS[ref].get("env") or {}).items()}),
                              pin_differs=(NB1_KV.get(BOOT.get("tag")) != refrec.get("kv_bytes")))
        o["diff_vs_ref"] = {"ref": ref, "diff": d, "check": oc}
        ck["one_change_vs_ref"] = oc["ok"]
    o["readback"] = {"nondefault": nd, "checks": ck, "env": plan.get("env"), "d8": plan.get("d8"), "line": nd_line[:6000],
                     "timer_lines": _pick(t, r"NB4 core timer", 4), "commit_lines": _pick(t, r"\[recoverssm-commit\]", 24),
                     "graph_mode_lines": _pick(t, r"resolved cudagraph modes|cudagraph_mode=|Capturing CUDA graphs|Graph capturing finished", 24),
                     "p5_lines": _pick(t, r"NB4_P5|P5_HOST", 12), "int8_lines": _pick(t, r"INT8 weight-only copy", 4)}
    o["config_ok"] = all(ck.values())
    o["memory_census"] = {"rank_load_gib": o["pool"].get("rank_load_gib"), "rank_graph_gib": o["pool"].get("rank_graph_gib"),
                          "kv_tokens": o["pool"].get("kv_tokens"), "workers_mem": [[w.get("rank"), w.get("mem")] for w in
                                                                                     ((o["workers"] or {}).get("ranks") or []) if isinstance(w, dict)]}
    return o


def _worker_env_map(o, key):
    out = {}
    for w in o.get("worker_env") or []:
        if isinstance(w, dict):
            out[str(w.get("rank"))] = (w.get("env") or {}).get(key)
    return out


def _start(s, a, o):
    if LIVE.get("rec") is not None:                       # klee100 R1-12: a mid-boot save() keeps the body's record
        LIVE["rec"]["live_body"] = o
    o["census"] = kv_census(s)
    run_item(o, s, "facts", lambda: nb4_facts(s, a, o))
    if a.get("commit_ab") and ((o.get("readback") or {}).get("env") or {}).get("VLLM_RECOVERSSM_COMMIT_GRAPH") == "graph":
        o["commit_flip_to_eager"] = commit_mode(s.port, "eager")     # T0's windows are the base: eager commits
        o["commit_mode_after_flip"] = [(x or {}).get("mode") for x in ((rpc(s.port, "get_recoverssm_commit_stats") or {}).get("results") or [])]
        if any(m != "eager" for m in o["commit_mode_after_flip"] or ["?"]):
            o["readback"]["checks"]["commit flipped to eager"] = False
            o["config_ok"] = False
    o["engaged_log"] = NL.engaged_log(a, s.log() or "", (o.get("readback") or {}).get("env"),
                                      _worker_env_map(o, "CUDA_DEVICE_MAX_CONNECTIONS") if a["key"] == "S" else None,
                                      (o.get("readback") or {}).get("checks"))
    o["probes"] = NL.probe_engine_view(run_item(o, s, "probes", lambda: chat_probes_full(s.port)))
    o["sampled"] = run_item(o, s, "sampled", lambda: sampled_check(s))
    o["gpu_after_ready"] = L.gpu_mem()
    if short_gate_failed(o):
        o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
        return False
    if o.get("config_ok") is not True:
        o.update(early_stop="CONFIG_MISMATCH", status="CONFIG_MISMATCH",
                 config_mismatch=[k for k, v in o["readback"]["checks"].items() if not v])
        return False
    if not o["engaged_log"]["ok"]:
        o.update(early_stop="NOT_ENGAGED", engaged_why=o["engaged_log"]["evidence"])   # the body stops: no GPU on a no-op arm
        return False
    return True


def _finish(s, o, smp):
    smp.ev.set()
    try:
        o["gpu_end"], o["gpu_peak_rows"], o["gpu_samples"] = L.gpu_mem(), smp.rows(), smp.n
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(s.port) if s.alive() else {"error": "server dead"}
        smi = NL.free_by_card([o.get("gpu_after_ready"), o.get("gpu_end"), o.get("gpu_peak_rows")])
        wf = NL.worker_free([o.get("workers"), o.get("workers_end")])
        o["free_mib_by_card_smi"], o["free_mib_by_card_worker"] = smi, wf
        o["free_mib_by_card_all"], o["free_source"] = NL.combine_free(smi, wf)        # klee100 R1-3
        o["jit_compiled_in_body"] = jit_kernels(s.log() or "")
    except Exception as e:
        o["finish_error"] = f"{type(e).__name__}: {e}"[:300]
    return o


def commit_stats(port):
    return (rpc(port, "get_recoverssm_commit_stats") or {}).get("results")


def p5_stats(port):
    """Agent B's per-rank stats via the NB4 worker-extension method (between windows only)."""
    r = rpc(port, "nb4_p5_stats", "1")
    out = []
    for x in (r or {}).get("results") or []:
        try:
            out.append(json.loads(x) if isinstance(x, str) else x)
        except Exception:
            out.append({"error": str(x)[:200]})
    return out


def p5_executed(stats):
    return sum(int(c.get("p5_executed") or 0) for s_ in stats or [] if isinstance(s_, dict) for c in (s_.get("comms") or []))


def nb4_window(s, band, n, label, k=3):
    """One steady window (NB1 window(): STEADY, real running == level, prefill done, tick / cycle / micro-batches per s /
    wall clock, acceptance per position) + the --cudagraph-metrics rows logged during it (n histogram, FULL / NONE decode
    steps, the micro-batch split) + CPU per server process (lead item 2). Every stream traced in full."""
    lg0 = len(s.log() or "")
    p0, t0 = session_procs({s.proc.pid}) if s.proc is not None else [], time.time()
    w = traced(label, lambda: window(s.port, s, band, n))
    p1, dt = session_procs({s.proc.pid}) if s.proc is not None else [], time.time() - t0
    txt = s.log() or ""
    w["cudagraph"] = NL.cudagraph_table(txt[lg0:], k=k)
    w["split"] = NL.split_label(w["cudagraph"], n)
    w["cpu"] = NL.cpu_per_process(p0, p1, dt, _roles(txt))
    return w


def windows(o, s, a, prefix=""):
    o.setdefault("steady", {})
    for rep in range(int(a.get("reps") or 1)):
        for band, n in (("8k", 8), ("40k", 8), ("8k", 9), ("40k", 9)):
            key = f"{band}@{n}" + ("" if rep == 0 else f"#{rep + 1}")
            o["steady"][prefix + key] = run_item(o, s, "window_" + prefix + key,
                                                 lambda: nb4_window(s, band, n, prefix + key, a.get("k") or NL.K_SPEC))


def wait_traces(pdir, n=4, timeout=180):
    """The profiler writes one gzip per worker after /stop_profile; wait until n files exist and their sizes are stable."""
    t0, last, stable = time.time(), None, 0
    while time.time() - t0 < timeout:
        fs = sorted(Path(pdir).rglob("*.pt.trace.json*"))
        sz = [f.stat().st_size for f in fs]
        if len(fs) >= n and sz == last:
            stable += 1
            if stable >= 3:
                return {"files": len(fs), "sec": round(time.time() - t0, 1)}
        else:
            stable = 0
        last = sz
        time.sleep(2)
    return {"files": len(sorted(Path(pdir).rglob("*.pt.trace.json*"))), "sec": round(time.time() - t0, 1), "timeout": True}


def timeline_analysis(trace_dir, tag):
    """Agent C's nb4_timeline.analyze on the 4 rank traces: the full report to timeline/<tag>.json, the compact parts in
    the record (clock, summary, stalls, graph modes, cpu activity, NB1 reference)."""
    if NB4_TL is None:
        return {"status": "TIMELINE_MODULE_PENDING", "why": NB4.get("timeline_module")}
    t0 = time.time()
    try:
        rep = NB4_TL.analyze(str(trace_dir), tp=NL.TP)
        rep = NB4_TL.strip_private(rep) if hasattr(NB4_TL, "strip_private") else json.loads(json.dumps(rep, default=lambda x: None))
    except Exception as e:
        return {"status": "TIMELINE_ANALYSIS_ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    try:
        (OUT / "timeline" / f"{tag}.json").write_text(json.dumps(rep, default=str))
    except Exception as e:
        rep["write_error"] = repr(e)[:200]
    out = {k: rep.get(k) for k in ("clock", "summary", "stalls", "graph_modes", "cpu_activity", "reference_rank", "forwards_ref_rank",
                                   "stages", "n_hist", "parity", "error")}
    out["nb1_reference"] = {k: v for k, v in (rep.get("nb1_reference") or {}).items() if k != "by_rank"}
    out["ranks"] = sorted((rep.get("cpu_activity") or {}).keys())
    out["file"] = f"timeline/{tag}.json"
    out["sec"] = round(time.time() - t0, 1)
    return out


def op_placement_all(trace_dir):
    """Lead item 3 (T0): per rank, what runs where (PLE, embeddings, LM head, draft passes, sampler, RecoverSSM commit, the
    broadcast origin), from the rank's own trace."""
    out = {}
    for f in sorted(Path(trace_dir).rglob("*.pt.trace.json*")):
        try:
            with (_gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
                ev = json.load(fh).get("traceEvents") or []
            out[f.name] = NL.op_placement(ev)
        except Exception as e:
            out[f.name] = {"error": repr(e)[:200]}
    return out


def nb4_profile(port, a, band, n):
    """One profiled window at the compare band (lead / folder sweep item 6): `n` decoding streams of PROMPTS[band] after
    settling (the lab's profiled_load: its own window, T 1.0 seeds 4321+, bounded /start_profile {20, 40}); traces kept in
    traces/profile_<arm>_<band>@<n>; analysed by nb4_timeline.analyze."""
    tag = f"{a['key']}_{band}@{n}"
    shutil.rmtree(NB4_PROF_DIR, ignore_errors=True)
    NB4_PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS[band], n, NB4_PROF_DIR, secs=6.0, max_new=MAX_NEW[band], spec=True,
                          prefill_tok_s=PREFILL["tok_s"])
    rec.update(band=band, level=n, delay_iters=20, max_iters=40)
    rec["wait_traces"] = wait_traces(NB4_PROF_DIR)
    rec.pop("trace", None)                                 # the lab's trace_summary verdict is not used (wrong graph-mode rule)
    kept = TRACES / f"profile_{tag}"
    kept.mkdir(exist_ok=True)
    rec["kept"] = []
    for f in sorted(NB4_PROF_DIR.rglob("*.pt.trace.json*")):
        shutil.copy2(f, kept / f.name)
        rec["kept"].append([f.name, f.stat().st_size])
    rec["timeline"] = timeline_analysis(kept, tag)
    if a["key"] == "T0":
        rec["op_placement"] = op_placement_all(kept)
    return rec


def core_timer_rows():
    rows, files = [], sorted(TMP.glob(NB4_TIMER.name + ".*.jsonl"))
    dst = OUT / "core_timer"
    dst.mkdir(exist_ok=True)
    for f in files:
        try:
            shutil.copy2(f, dst / f.name)
            for l in f.read_text().splitlines():
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
        except Exception:
            pass
    return rows, [f.name for f in files]


def t0_extras(s, o):
    """T0: the engine-core side (the step timer rows, vLLM's iteration log: n per step in order = the parity sequence)."""
    rows, files = core_timer_rows()
    return {"core_timer_files": files, "core_timer_rows": len(rows), "core_timer": NL.core_timer_summary(rows),
            "iteration": NL.iteration_summary(s.log() or "")}


def commit_ab(o, s):
    """Arm C inside T0's boot (lead item 4, NB1 commit_ab pattern): a greedy 64-token row in eager, the counters, mode ->
    graph, 8k@8 + 40k@8 windows (n = 4 per micro-batch, captured with CAPS16), the counters, the same greedy row in graph
    mode (bit-identity claim), mode -> eager. Stage 1 only (stage-0 commits are PP-deferred)."""
    ab = {"windows": {}}
    ids = PROMPTS["2k"][2]
    ab["greedy_eager"] = g_e = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
    ab["stats_before"] = commit_stats(s.port)
    ab["set_graph"] = commit_mode(s.port, "graph")
    for band in ("8k", "40k"):
        lab = f"{band}@8"
        ab["windows"][lab] = run_item(o, s, "C_window_" + lab, lambda: nb4_window(s, band, 8, "C_" + lab))
    ab["stats_after"] = commit_stats(s.port)
    ab["greedy_graph_vs_eager"] = lp_compare(g_e, nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2))
    steps = sum(int((w or {}).get("steps") or 0) for w in ab["windows"].values())
    ab["split"] = NL.commit_split(ab["stats_before"], ab["stats_after"], steps or None)
    ab["counter_delta"] = commit_counter_delta(ab["stats_before"], ab["stats_after"]) if ab["stats_before"] and ab["stats_after"] else None
    ab["reset_eager"] = commit_mode(s.port, "eager")
    return ab


def body_arm(a, is_ref, ref_key):
    """One arm (= one boot): short gates + read-back + the change visible (_start), the greedy oracle (vs the base), TF
    2k / 8k / 40k (T0: fresh + repeat = the in-boot A/A), warm-up, the commit / P5 counters around the 4 windows (x reps),
    the profiled windows at the compare bands, T0's extras + the C runtime A/B. A greedy FAIL_HIGH_MARGIN vs the base
    keeps TF (diagnosis) and stops."""
    k = a["key"]

    def body(ss):
        s, o, smp = ss[0], {"arm": k, "is_ref": is_ref, "ref": ref_key}, GpuSampler()
        smp.start()
        try:
            if _start(s, a, o):
                refs = ("rc2_fp8", "nb1r2_b3") + (() if is_ref else (ref_key.lower(),))
                o["oracle"] = run_item(o, s, "oracle", lambda: oracle(s.port, key=k.lower(), refs=refs), need_sec=300)
                vs = ((o.get("oracle") or {}).get("vs_" + ref_key.lower()) or {}).get("status") if ref_key and not is_ref else None
                runs = ("fresh", "repeat") if k == "T0" else ("fresh",)
                o["tf"] = run_item(o, s, "tf_2k-8k-40k", lambda: tf_suite(k, s, ["2k", "8k", "40k"], runs=runs))
                o["reference_stored"] = bool(ROWS.get(k.lower())) and all(
                    (((TFR.get(k) or {}).get(b) or {}).get("fresh") or {}).get("arr") is not None for b in ("2k", "8k", "40k"))
                if vs == "FAIL_HIGH_MARGIN":
                    o["early_stop"] = "ORACLE_FAIL_VS_BASE"
                    return o
                o["warmup"] = run_item(o, s, "warmup", lambda: warmup(s.port, 9), need_sec=180)
                o["commit_stats_before"] = commit_stats(s.port)
                o["p5_before"] = p5_stats(s.port) if k == "P" else None
                windows(o, s, a)
                o["commit_stats_after"] = commit_stats(s.port)
                steps = sum(int((w or {}).get("steps") or 0) for w in (o.get("steady") or {}).values() if isinstance(w, dict))
                o["commit_split"] = NL.commit_split(o["commit_stats_before"], o["commit_stats_after"], steps or None)
                if k == "P":
                    o["p5_after"] = p5_stats(s.port)
                    o["p5_delta"] = {"executed": p5_executed(o["p5_after"]) - p5_executed(o["p5_before"]),
                                     "per_step": NB4_P5.engaged_per_step(o["p5_before"], o["p5_after"], max(1, steps)) if NB4_P5 else None}
                    o["p5_label"] = NB4_P5.p5_arm_label(s.log() or "", o["p5_after"]) if NB4_P5 else None
                if k in ("S2", "S3") and NB4_PK is not None:
                    o["ppack_label"] = NB4_PK.readback_label(s.log() or "", a.get("variant"))
                o["profiles"] = {}
                for pb in a.get("profiles") or []:
                    band, n = pb.split("@")
                    o["profiles"][pb] = run_item(o, s, "profile_" + pb, lambda: traced("profile_" + pb, lambda: nb4_profile(
                        s.port, a, band, int(n))), need_sec=420)
                if k == "T0":
                    o["profiler_tools"] = profiler_tools()
                    o["t0"] = run_item(o, s, "t0_extras", lambda: t0_extras(s, o))
                    if a.get("commit_ab") and o.get("commit_flip_to_eager") is not None:
                        o["commit_ab"] = run_item(o, s, "commit_ab", lambda: commit_ab(o, s), need_sec=480)
                    else:
                        o["commit_ab"] = {"status": "NOT_INTEGRATED", "why": "T0 booted without the commit-graph env"}
        except Exception as e:
            o.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2500:])
        finally:
            _finish(s, o, smp)
        return o
    return body


def run_arm(k):
    """One arm through merlin_boot (boot gate: GPU empty, engine processes gone, page cache dropped, Merlin's host-RAM
    rule; evidence + oracle gates vs the base) with the per-boot body cap; the reference arm stores its rows."""
    a = ARMS[k]
    ref = NB4.get("ref")
    is_ref = ref is None
    name, port = nb4_plan(k, 0)
    BOOT["tag"] = name
    need = () if is_ref else (ref.lower(),)
    o = merlin_boot(name, "fp8", port, min(NL.READY_SEC, max(300, left() - 600)),
                    watched(body_arm(a, is_ref, ref), NL.CAP_MIN[k] * 60, name), need=need, expected_rows=4)
    LIVE["out"] = o
    if LIVE.get("rec") is not None:
        LIVE["rec"].pop("live_body", None)                 # klee100 R1-12: the full record replaces the live copy
    try:
        why = str((o.get("boot_gate") or {}).get("why") or "")
        if o.get("status") == "GPU_NOT_EMPTY" and why.startswith(("HOST_RAM", "ENGINE_PROCS")):
            o["status"] = "HOST_RAM_GATE_FAILED" if why.startswith("HOST_RAM") else "ENGINE_PROCS_LEFT"
        ln = o.get("launch_name") or name
        o.update(arm=k, change=a["change"], plan_tail=((R["setup"].get("nb4_plans") or {}).get(ln) or {}).get("argv_tail"),
                 load_monitor=LOADMON.get(ln))
        srv = (o.get("servers") or {}).get(ln) or {}
        o["device_fault"] = bool(o.get("xid_in_log")) or bool(srv.get("xid"))
        o["oom_in_body"] = not str(o.get("status", "")).startswith("LAUNCH_FAILED") and bool(o.get("oom_in_log"))
    except Exception as e:
        keep_after_error(o, e)
    return o


def nb4_arm_step(k):
    """step() around one arm. A patch arm (P, S2): its prepare step first (P: build + gate + patch; S2: patch) -> the arm
    is integrated only when it is ok; the revert always runs after the boot (finally). The first clean base becomes the
    reference."""
    a = ARMS[k]
    if a.get("patch"):
        prep_fn, fin_fn = (p5_prepare, p5_finish) if a["patch"] == "p5" else ((lambda: ppack_prepare(a["variant"])), ppack_finish)
        prep = step(k, f"{k.lower()}_prepare", 8 if k == "P" else 2, prep_fn)
        NB4[f"{k.lower()}_prepare"] = {x: prep.get(x) for x in ("status", "ok", "why", "build", "gate", "patch", "so", "so_sha256", "error")}
        new = NL.make_arms(prep if k == "S2" else None, NB4_PIN.get("g_override"), prep if k == "P" else None,
                           prep if k == "S3" else None, NB4_DISABLED)[k]
        if prep.get("status") != "DONE" or prep.get("ok") is not True:
            ARMS[k] = dict(new, integrated=False, why_not=(("P5_NOT_INTEGRATED: " if k == "P" else "NOT_INTEGRATED: ")
                                                          + str(prep.get("why") or prep.get("error") or prep.get("status")))[:400])
            try:
                NB4[f"{k.lower()}_finish"] = fin_fn()
            except Exception as e:
                NB4[f"{k.lower()}_finish"] = {"error": repr(e)[:200]}
            rec = {"status": ("P5_NOT_INTEGRATED" if k == "P" else "NOT_INTEGRATED"), "why": ARMS[k]["why_not"]}
            RECS[k] = rec
            save()
            return rec
        ARMS[k] = new
        NB4["arm_problems"] = NL.validate_arms(ARMS)
        if NB4["arm_problems"]:
            NB4[f"{k.lower()}_finish"] = fin_fn()
            rec = {"status": "CONFIG_MISMATCH", "why": NB4["arm_problems"]}
            RECS[k] = rec
            return rec
        try:
            rec = step(k, NL.launch_name(k, 0), NL.EST_MIN[k], lambda: run_arm(k))
        finally:
            NB4[f"{k.lower()}_finish"] = fin_fn()
    else:
        rec = step(k, NL.launch_name(k, 0), NL.EST_MIN[k], lambda: run_arm(k))
    RECS[k] = rec
    if NB4.get("ref") is None and k in NL.BASE_ARMS and NL.base_ok(rec):
        NB4["ref"] = k
    save()
    return rec


# ---- analysis: correctness vs the base, engaged + mechanism, speed vs base with the A/A spread, labels
def _tf(k, band, run="fresh"):
    x = ((TFR.get(k) or {}).get(band) or {}).get(run)
    return x if x and x.get("arr") is not None else None


def tf_vs(ref, k, bands=("2k", "8k", "40k")):
    out = {}
    for b in bands:
        r, c = _tf(ref, b), _tf(k, b)
        if not r or not c:
            out[b] = {"label": "INCONCLUSIVE_MISSING_TF"}
            continue
        ful, lst = cmp_full(r["arr"], c["arr"]), cmp_last256(r, c)
        out[b] = {"label": NL.tf_label(ful), "label_last256": NL.tf_label(lst),
                  "full": {x: ful.get(x) for x in ("status", "n", "expected", "mean_abs_dlogprob", "p99_abs_dlogprob", "max_abs_dlogprob",
                                                   "rank1_agreement", "first_window_over_screen_mean", "bins_4096", "chunk")},
                  "last256": {x: lst.get(x) for x in ("status", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")},
                  "shift": shift_check(r["arr"], c["arr"])}
    return out


def _windows_of(k):
    return {w: v for w, v in ((RECS.get(k) or {}).get("steady") or {}).items() if isinstance(v, dict)}


def _w_brief(v):
    v = v or {}
    return dict({x: v.get(x) for x in ("status", "tok_s", "wall_clock_tok_s", "tick_ms", "cycle_ms", "microbatches_per_s", "gen_tokens",
                                        "running_min", "running_max", "running_eq_level", "mean_acceptance_length",
                                        "accept_rate_per_position_incl_admission", "preemptions", "kv_usage_max",
                                        "prefill_done_before_window")},
                split=(v.get("split") or {}), cudagraph_by_mode=(v.get("cudagraph") or {}).get("decode_by_mode"),
                cpu_total_cores=(v.get("cpu") or {}).get("total_cores"))


def arm_brief(rec):
    rec = rec or {}
    return {"status": rec.get("status"), "correctness": rec.get("correctness"), "early_stop": rec.get("early_stop"),
            "launch_name": rec.get("launch_name"), "port": rec.get("port"), "kv_bytes": rec.get("kv_bytes"), "pin_rung": rec.get("pin_rung"),
            "ready_sec": rec.get("ready_sec"), "tries": rec.get("tries"), "death": rec.get("death"), "died_in": rec.get("died_in"),
            "boot_watchdog": rec.get("boot_watchdog"), "config_ok": rec.get("config_ok"), "config_mismatch": rec.get("config_mismatch"),
            "readback_checks": (rec.get("readback") or {}).get("checks"), "d8": (rec.get("readback") or {}).get("d8"),
            "diff_vs_ref": (rec.get("diff_vs_ref") or {}).get("check"), "engaged_log": rec.get("engaged_log"),
            "worker_env": rec.get("worker_env"), "windows": {w: _w_brief(v) for w, v in (rec.get("steady") or {}).items() if isinstance(v, dict)},
            "profiles": {b: {"status": (p or {}).get("status"), "timeline_summary": ((p or {}).get("timeline") or {}).get("summary"),
                             "stalls": ((p or {}).get("timeline") or {}).get("stalls"),
                             "graph_modes": ((p or {}).get("timeline") or {}).get("graph_modes")}
                         for b, p in (rec.get("profiles") or {}).items() if isinstance(p, dict)},
            "commit_split": rec.get("commit_split"), "p5_delta": rec.get("p5_delta"), "p5_label": rec.get("p5_label"),
            "graph_capture": {"captured": (rec.get("graphs") or {}).get("captured"),
                              "rank_graph_gib": (rec.get("pool") or {}).get("rank_graph_gib")},
            "free_mib_by_card": rec.get("free_mib_by_card_all"), "free_source": rec.get("free_source"),
            "oom_in_log": rec.get("oom_in_log"), "oom_in_body": rec.get("oom_in_body"), "device_fault": rec.get("device_fault"),
            "error": rec.get("error"), "body_error": rec.get("body_error"), "postprocess_error": rec.get("postprocess_error")}


def _tl(rec, band):
    return (((rec or {}).get("profiles") or {}).get(band) or {}).get("timeline")


def nb4_analysis():
    ref = NL.ref_arm(RECS)
    an = {"ref": ref, "arms": {}, "aa": {}, "winners": [], "nb1r2_b3_windows": NB4.get("nb1r2_b3_windows"),
          "rule": "WIN = correctness PASS vs base AND every decisive window gain > max(2 %, A/A spread), no window SLOWER, "
                  "equal micro-batch splits only AND the mechanism proven (trace / counters); " + NL.WIN_NOTE}
    t0w, t0bw = _windows_of("T0"), _windows_of("T0b")
    an["aa"]["speed"] = NL.aa_spread(t0w, t0bw)
    an["aa"]["tf_t0_fresh_vs_repeat"] = {b: (lambda r, c: {"label": NL.tf_label(cmp_full(r["arr"], c["arr"], bins=False)),
                                                            "last256": cmp_last256(r, c)} if r and c else None)(_tf("T0", b), _tf("T0", b, "repeat"))
                                         for b in ("2k", "8k", "40k")}
    an["aa"]["tf_t0b_vs_t0"] = tf_vs("T0", "T0b") if RECS.get("T0b") and RECS.get("T0") else None
    an["aa"]["oracle_t0b_vs_t0"] = (((RECS.get("T0b") or {}).get("oracle") or {}).get("vs_t0") or {}).get("status")
    an["t0_vs_nb1r2_b3_oracle"] = (((RECS.get("T0") or {}).get("oracle") or {}).get("vs_nb1r2_b3") or {}).get("status")
    an["t0_vs_rc2_fp8_oracle"] = (((RECS.get("T0") or {}).get("oracle") or {}).get("vs_rc2_fp8") or {}).get("status")
    t0 = RECS.get("T0") or {}
    an["timeline"] = {"profiles": {b: (p or {}).get("timeline") for b, p in (t0.get("profiles") or {}).items() if isinstance(p, dict)},
                      "op_placement": {b: (p or {}).get("op_placement") for b, p in (t0.get("profiles") or {}).items() if isinstance(p, dict)},
                      "engine_core": t0.get("t0")}
    ab = t0.get("commit_ab") or {}
    if ab.get("windows"):
        an["C"] = NL.c_verdict(ab.get("windows"), t0w, an["aa"]["speed"], ab.get("greedy_graph_vs_eager"), ab.get("split"),
                               t0.get("correctness"))
        an["C"]["counter_delta"] = ab.get("counter_delta")
        an["C"]["captured_n"] = NL.commit_graph_ns("\n".join((t0.get("readback") or {}).get("commit_lines") or []))
    else:
        an["C"] = {"label": "NOT_INTEGRATED" if t0 else "SKIPPED_BUDGET", "why": ab.get("why") or ab.get("status") or "T0 not run"}
    for k in NL.ARM_ORDER:
        a, rec = ARMS[k], RECS.get(k)
        e = {"change": a["change"], "integrated": a.get("integrated"), "why_not": a.get("why_not"), "diagnostic": a.get("diagnostic")}
        corr = speed = mech = None
        if rec is not None and not str(rec.get("status", "")).startswith(("SKIPPED", "P5_NOT", "NOT_INT")):
            if ref and k != ref:
                tfl = tf_vs(ref, k)
                ors = ((rec.get("oracle") or {}).get("vs_" + ref.lower()) or {}).get("status")
                corr = NL.correctness(rec.get("correctness"), ors, {b: v["label"] for b, v in tfl.items()})
                e["tf_vs_ref"] = tfl
            elif k == ref:
                corr = {"verdict": "REFERENCE"}
            if k not in NL.BASE_ARMS:
                speed = NL.speed_verdict(_windows_of(k), t0w, t0bw, decisive=tuple(a.get("decisive") or NL.WINDOWS))
                cb = (a.get("profiles") or ["8k@8"])[0]
                mech = NL.mechanism(a, {"tl": _tl(rec, cb), "tl_ref": _tl(t0, cb),
                                        "cg": {w: (v or {}).get("cudagraph") for w, v in _windows_of(k).items()},
                                        "log": "\n".join(((rec.get("readback") or {}).get("int8_lines") or [])),
                                        "p5_label": rec.get("p5_label"), "p5_delta": rec.get("p5_delta"),
                                        "ppack_label": rec.get("ppack_label"),
                                        "accept": ((_windows_of(k).get("8k@8") or {}).get("accept_rate_per_position_incl_admission") or {})})
            elif k == "T0":
                mech = NL.mechanism(a, {"tl": _tl(rec, "8k@8"), "iteration_lines": ((rec.get("t0") or {}).get("iteration") or {}).get("lines"),
                                        "core_timer_rows": (rec.get("t0") or {}).get("core_timer_rows")})
        e.update(correctness=corr, speed=speed, mechanism=mech, brief=arm_brief(rec) if rec else None,
                 label=NL.final_label(a, rec, corr, speed, (rec or {}).get("engaged_log") if rec else None, mech))
        an["arms"][k] = e
        if str(e["label"]["label"]).startswith("WIN"):
            an["winners"].append(k)
    if str(an["C"].get("label", "")).startswith("WIN"):
        an["winners"].append("C")
    return an
save()
'''

# ---------------------------------------------------------------- CELL 8: main (fail closed, try/finally verdict, fallback)
CELL8 = r'''# ==================================================================== CELL 8 — E25 NB4 main: gates, install, T0 -> S -> S2 -> S3 -> G -> P -> T0b, verdict
def nb4_main():
    """Gates before any GPU (inputs, merge, arms valid incl. the never-do list, embedded files), install, #60490 (fail
    closed), the core timer (not a gate), the sha256 gate, then the arms in the planner's order (nb4_lib.next_arm: lead
    order, T0b protected / moved earlier when the next arm + T0b do not both fit, T0b as the base when T0 is not clean,
    optional K4 / INT8 only with budget)."""
    m = R["sections"]["A0"]["merge"]
    GATES["INPUTS"] = (bool(MERLIN_OK), None if MERLIN_OK else json.dumps(R["inputs"])[:400])
    GATES["MERGE"] = (m.get("ok") is True, m.get("why") or m.get("bad"))
    GATES["ARMS_VALID"] = (not NB4["arm_problems"], NB4["arm_problems"] or None)
    GATES["EMBEDDED"] = (all(NB4["embedded_sha_ok"].values()), NB4["embedded_sha_ok"])
    if all(ok for ok, _ in GATES.values()):
        rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        GATES["MERLIN_INSTALL"] = (rec.get("status") == "DONE", rec.get("error"))
    if all(ok for ok, _ in GATES.values()):
        rec = step("A0", "patch_60490", 3, apply_60490)
        GATES["PATCH_60490"] = (rec.get("status") == "DONE" and rec.get("ok") is True, rec.get("error"))
    if all(ok for ok, _ in GATES.values()):
        NB4["core_timer"] = step("A0", "core_timer_install", 2, core_timer_install)
        if NB4["core_timer"].get("ok") is not True:              # T0 keeps vLLM's iteration log; the timer is recorded absent
            ARMS["T0"]["core_timer"] = False
        NB4["arms"] = {k: {x: a.get(x) for x in ("change", "integrated", "why_not", "compile", "extra", "env", "prebuild_env", "spec",
                                                 "profiles", "decisive", "ladder", "reps", "mechanism")} for k, a in ARMS.items()}
        save()
    if all(ok for ok, _ in GATES.values()):
        ok, why = sha_gate(max(60, min(1500, left() - 110 * 60)))      # klee100 R1-7: the wait never starves the arms
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        GATES["SHA256"] = (ok, why)
        save()
    if all(ok for ok, _ in GATES.values()):
        while True:
            k, why = NL.next_arm(ARMS, DONE, left())
            NB4["plan_log"].append({"t_min": round((time.time() - T_START) / 60, 1), "left_min": round(left() / 60, 1), "next": k, "why": why})
            save()
            if k is None:
                if "no clean base" in why:
                    for k2 in NL.ARM_ORDER:
                        if k2 not in DONE and ARMS[k2].get("integrated"):
                            RECS[k2] = DONE[k2] = {"status": "SKIPPED_NO_BASE", "why": why}
                break
            DONE[k] = nb4_arm_step(k)
            if ABORT["why"]:
                break


def nb4_verdict(main_error):
    try:
        an = nb4_analysis()
    except Exception as e:
        an = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2500:], "arms": {}}
    labels = {k: (v.get("label") or {}).get("label") for k, v in (an.get("arms") or {}).items()}
    labels["C"] = (an.get("C") or {}).get("label")
    return {"labels": labels, "winners": an.get("winners"), "main_error": main_error,
            "gates": {k: {"ok": v[0], "why": v[1]} for k, v in GATES.items()}, "analysis": an,
            "plan_log": NB4.get("plan_log"), "ref": NB4.get("ref"), "core_timer": NB4.get("core_timer"),
            "prepare": {k: NB4.get(k) for k in ("p_prepare", "p_finish", "s2_prepare", "s2_finish", "s3_prepare", "s3_finish")},
            "disabled": NB4.get("disabled"),
            "modules": {"timeline": NB4.get("timeline_module"), "p5": NB4.get("p5_module"), "ppack": NB4.get("ppack_module")},
            "research": NB4.get("research"), "minutes": round((time.time() - T_START) / 60, 1), "abort": ABORT["why"],
            "notes": ["speed numbers are engine end-to-end; microbenchmarks are never counted (PLAN section 0)",
                      "G and P run on the user's instruction as diagnostics (expected gain below the PLAN 5 % rule)",
                      "a WIN_SPEED needs paired ARC accuracy + PLAN §4.1-7 evidence before adoption"]}


if not RERUN and not os.environ.get("RC2_UNIT"):
    MAIN_ERROR, VERDICT = None, {"labels": {}, "label": "NOT_REACHED"}
    try:                                       # klee100 R1-11: whatever happens, the verdict, save() and the fallback are written
        nb4_main()
    except Exception as e:
        MAIN_ERROR = {"error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-2500:]}
        R["errors"].append("CELL 8 main: " + MAIN_ERROR["error"])
    finally:
        try:
            VERDICT = nb4_verdict(MAIN_ERROR)
        except Exception as e:
            VERDICT = {"label": "VERDICT_ERROR", "why": f"{type(e).__name__}: {e}"[:400], "main_error": MAIN_ERROR,
                       "tb": traceback.format_exc()[-2500:]}
        finally:
            R["verdict"] = VERDICT
            R["minutes"] = round((time.time() - T_START) / 60, 1)
            R["host_phase"] = HOST_PHASE
            save()
            try:
                (OUT / "nb4_verdict.json").write_text(json.dumps(VERDICT, indent=1, default=str))
            except Exception as e:
                print("nb4_verdict.json not written:", repr(e)[:200], flush=True)
            write_fallback_submission()
            print(json.dumps({k: VERDICT.get(k) for k in ("labels", "winners", "main_error", "gates", "ref", "minutes")}, default=str)[:8000])
            print("@@NB4 " + json.dumps(VERDICT.get("labels"), default=str) + "@@", flush=True)
'''

MD0 = """# E25 NB4 TICK: the stage-0 in-graph stall + S / S2 / G / C / P A/B (Qwen3.8-Flash-Next W4, Merlin, 4×L4)

Three offline readings of NB1 R2 B1's traces agree: stage 0's FULL graph stalls ~21.7 ms per tick at kernel 172 of 1,036
until the PP sampled-token broadcast of the **other** micro-batch ends (a false dependency); decode already runs FULL; the
host is 62-117 ms ahead. Hard stop 150 min; every arm is **base + exactly one change** with an identical flag set.

**Base** = NB1 R2 B3 = NB5 QUALITY's FP8 boot: W4 Merlin, FP8 E4M3 KV, pin 3,553,580,524 B (214 blocks), MTP K3,
`--language-model-only`, #60490, seqs 9, CAPS16, read-only worker extension, `--cudagraph-metrics`, idle torch profiler.

| Order | Arm | Change | Proof that it engaged |
|---|---|---|---|
| 1 | T0 | host timeline (profiles 8k@8 + 8k@9, iteration log, step timer) + **C** = commit graph runtime A/B in the same boot | 4 rank traces; C: PP1 replays, greedy EXACT |
| 2 | S | `CUDA_DEVICE_MAX_CONNECTIONS=32` | env in every worker; in-graph stall > 1 ms in ≤ 5 % of graphs |
| 3 | S2 | Q-pack patch (one packed PP broadcast) | patch + init line; the stall gone |
| 4 | G | capture list 1..9, 12..36 (FULL for every n at seqs 9) | 36-token steps FULL, 0 eager decode rows |
| 5 | P | P5 host-staged TP all-reduce ≤ 184 KB (diagnostic) | gate PASS, 4 inits, P5 calls > 0 |
| 6 | T0b | the base again, 2 replicates (A/A) | — |

Per arm: read-back (CONFIG_MISMATCH / NOT_ENGAGED stop the body), probes, sampled, greedy oracle and TF 2k / 8k / 40k vs
base, 8k@8 / 40k@8 / 8k@9 / 40k@9 windows (90 s, the cudagraph rows = n histogram and micro-batch split), a short profiled
window at the compare band (agent C's `nb4_timeline`). **WIN_SPEED** only with correctness PASS, every decisive window above
max(2 %, the measured A/A spread) on equal splits, and the mechanism proven. Commit run only; a competition rerun writes
the fallback submission. Output: `nb4/nb4_verdict.json`, `nb4/nb4_results.json`, `nb4/traces`, `nb4/timeline`, `nb4/logs`,
`nb4/tf_raw`, `nb4/core_timer`.
"""

# ---------------------------------------------------------------- assembly
EMB = {"nb4_lib.py": LIB}
PIN = {"lib_sha256": sha(LIB), "picks": {"nb0": PICK_SHA0, "nb1_cell6": PICK_SHA1}, "nb0_notebook_sha256": NB0_NB_SHA,
       "nb1_r2_notebook_sha256": NB1_NB_SHA, "nb0_builder_sha256": sha(N0_TEXT.encode()), "nb1_builder_sha256": sha(N1_TEXT.encode()),
       "research": RESEARCH, "g_override": G_OVERRIDE, "disabled": list(DISABLED_ARMS), "hooks": {}, "hook_source": {}, "worker_ext_sha256": sha(NB4_EXT.encode()),
       "nb1r2_b3_rows": NB1R2_B3_ROWS, "nb1r2_b3_windows": NB1R2_B3_WINDOWS, "edits": NB4_EDITS}


def assemble(emb, pin):
    subs = {"__LAB__": json.dumps(RC2["LAB"]), "__PLE__": json.dumps(RC2["PLE"]), "__PATCH__": json.dumps(N1["PATCH_TEXT"]),
            "__PPRE__": N1["P_PRE"], "__PPOST__": N1["P_POST"], "__NB1EXT__": json.dumps(NB4_EXT),
            "__RC2REF__": json.dumps(json.dumps(RC2_REF)),
            "__NB4EMB__": json.dumps(json.dumps({n: [sha(b), pack(b)] for n, b in emb.items()})),
            "__NB4PIN__": json.dumps(json.dumps(pin))}
    cells = []
    for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6, CELL7, CELL8)):
        for k, v in subs.items():
            c = c.replace(k, v)
        assert not [k for k in subs if k in c], f"cell {i + 1}: placeholder left"
        if i:                                      # a competition rerun skips cells 2-8 without an exception (RC2 / E23 rule)
            c = "if not RERUN:\n" + textwrap.indent(c, "    ")
        compile(c, f"<nb4 cell {i + 1}>", "exec")
        cells.append(c)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
                   [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    return cells, data


# hook modules (+ the P5 sources with the P5 module): inline (lzma, sha-checked) while the notebook stays < NB_TARGET, else
# the payload dataset (sha pinned)
PAYLOAD_FILES = {}
_groups = [("nb4_timeline.py", {"nb4_timeline.py": HOOKS["nb4_timeline.py"]}),
           ("nb4_p5_engine.py", dict({"nb4_p5_engine.py": HOOKS["nb4_p5_engine.py"]}, **P5_SRC)),
           ("nb4_ppack.py", {"nb4_ppack.py": HOOKS["nb4_ppack.py"]})]
for _n, _files in _groups:
    if HOOKS[_n] is None:
        continue
    PIN["hooks"][_n] = sha(HOOKS[_n])
    _try = dict(EMB, **_files)
    if len(assemble(_try, PIN)[1]) < NB_TARGET:
        EMB = _try
        PIN["hook_source"][_n] = "embedded"
    else:
        PAYLOAD_FILES.update(_files)
        PIN["hook_source"][_n] = "payload " + PAYLOAD_SLUG
cells_src, data = assemble(EMB, PIN)
assert len(data) < NB_LIMIT, len(data)
DATASETS = [PAYLOAD_SLUG] if PAYLOAD_FILES else []

if __name__ == "__main__":
    OUT_NB.write_bytes(data)
    (HERE / "nb4_code_export.py").write_bytes("\n\n".join(cells_src).encode("utf-8"))   # one file, for the kits / audits (LF)
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / f"{SLUG}.ipynb").write_bytes(data)
    meta = {"id": f"foysalemonshanto/{SLUG}", "title": "Q27 Xhigh E25 NB4 Tick", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "NvidiaL4"}
    (PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    if PAYLOAD_FILES:                          # only when a hook module does not fit inline: the dataset folder to upload
        PAYLOAD.mkdir(parents=True, exist_ok=True)
        for _n, _b in PAYLOAD_FILES.items():
            (PAYLOAD / _n).parent.mkdir(parents=True, exist_ok=True)
            (PAYLOAD / _n).write_bytes(_b)
        (PAYLOAD / "SHA256SUMS").write_bytes("".join(f"{sha(_b)}  {_n}\n" for _n, _b in sorted(PAYLOAD_FILES.items())).encode())
        (PAYLOAD / "dataset-metadata.json").write_bytes((json.dumps({
            "title": "E25 NB4 Payload", "id": PAYLOAD_SLUG, "licenses": [{"name": "CC0-1.0"}],
            "description": "E25 NB4 TICK hook modules (agents B / C / Q-pack) and the P5 sources, sha256 pinned in the notebook."},
            indent=2) + "\n").encode())
    (HERE / "nb4_build_facts.json").write_bytes((json.dumps({
        "notebook": OUT_NB.name, "notebook_sha256": sha(data), "bytes": len(data), "push": str(PUSH), "datasets": DATASETS,
        "payload_files": {n: sha(b) for n, b in PAYLOAD_FILES.items()}, "embedded": {n: sha(b) for n, b in EMB.items()},
        "replayed_cells": {"nb0": N0_CELLS_SRC, "nb1_r2": N1_CELLS_SRC}, "pin": {k: v for k, v in PIN.items() if k != "nb1r2_b3_rows"},
        "arms_at_build": {k: {x: a.get(x) for x in ("change", "integrated", "why_not")} for k, a in _ARMS_BUILD.items()}},
        indent=1, default=str) + "\n").encode("utf-8"))
    print("wrote", OUT_NB.name, "sha256", sha(data), "|", len(data), "bytes | embedded", sorted(EMB), "| payload", sorted(PAYLOAD_FILES),
          "| push", PUSH)
