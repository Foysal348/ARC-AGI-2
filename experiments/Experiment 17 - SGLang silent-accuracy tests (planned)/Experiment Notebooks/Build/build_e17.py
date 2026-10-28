"""E17 = SGLang silent-accuracy tests (PLAN_E17 rev 2, Experiment 14 "Bug ang fix", 2026-10-26): the #43567 stale
lazy-checkpoint chain (upstream test_mamba_cache_kl.py Class 1 ported to our stack, the DFlash variant, natural
pressure), DFlash vs target-only parity, fp16 vs fp32 SSM. Engine only: no solver.

Base cells (read only), as EMS (Experiment 14 build_ems.py): the E15P rev 2 notebook (56a22f7f; the Kaggle-CPU-kit
verified production engine: SGLang 0.5.21, RedHat INT4, DFlash2 width 6, 16 streams, /dev/shm staging):
  CELL 0 mode, CELL 1 config (+ the E17 profile), CELL 1b E14 facts / staging / modules (E14M, E14P = the patch module
  with #43567), CELL 2 install, the task-order cell, a sandbox stub, the engine cell (the production launch = the
  first, JIT-warming launch).
New: the hook cell (src/e17_hook.py as text, sha-checked), CELL E17 (Build/e17_cell.py), CELL E17-R
(Build/e17_report_cell.py).
usage: python build_e17.py   -> ../Q27-XHIGH-E17_SGLANG_SILENT_ACCURACY.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
EXP = NBDIR.parent
E15 = EXP.parent / "Experiment 15 - Duck 2x35k independent attempts (planned)" / "Experiment Notebooks"
SRC = E15 / "Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"
SRC_SHA = "56a22f7f"
OUT = NBDIR / "Q27-XHIGH-E17_SGLANG_SILENT_ACCURACY.ipynb"
PUSHED: set = {"9793b41e1067940bf19cb77ef6883551b2b1fd2ed505ccbd8342178250a0992e"}   # foysalemonshanto v1 (2026-10-26, E17SHORT)


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
c1 = rep(S[1], 'os.environ.get("Q27X_PROFILE", "E15PEVAL120")', 'os.environ.get("Q27X_PROFILE", "E17SHORT")')
c1 = rep(c1, '''    "E14SEVAL120G4B": {''', '''    # E17 (PLAN_E17 rev 2, 2026-10-26): engine-only silent-accuracy tests; the 12 G4B tasks only feed the task cells;
    # 170 min = first launch ~9 min + ~12 relaunches ~1.5 min + the T1 / T2 / T3 blocks (~85 min) + reserve
    "E17":       {"source": "eval", "task_list": ''' + json.dumps(G4B) + ''',
                  "time_budget_sec": 170 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
                  "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "e17": True},
    # E17SHORT (user 2026-10-26, the default): the first session, <= 60 min, only the decisive #43567 pairs L1 / L7 and
    # L5 / L9; the full E17 plan runs later only if the bug shows
    "E17SHORT":  {"source": "eval", "task_list": ''' + json.dumps(G4B) + ''',
                  "time_budget_sec": 60 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
                  "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "e17": True,
                  "e17_short": True},
    "E14SEVAL120G4B": {''')
c1 += '''

# ---- E17: engine-only; the rerun of this notebook only writes the cell-0 fallback
CONFIG["e17"] = {"on": bool(P.get("e17")) and not RERUN}
print("E17:", json.dumps(CONFIG["e17"]))
'''
c1b = rep(S[2], '"variant": "E15P"', '"variant": "E17"')
stub = '''# ==================================================================== E17: no solver in this notebook
# The engine cell starts SGLang only after the solver's sandbox self-test; E17 runs no generated code.
E10_SANDBOX_SELFTEST = {"ok": True, "mode": "E17 engine tests: no solver, no sandbox"}
'''
eng = rep(S[6], "# E15P: the E15B engine cell (Poetiq uses /generate through the shim;",
          "# E17: the production engine cell, unchanged (its launch = the first, JIT-warming launch). E15P: the E15B engine "
          "cell (Poetiq uses /generate through the shim;")
HOOK = (EXP / "src" / "e17_hook.py").read_text(encoding="utf-8").replace("\r\n", "\n")
HOOK_SHA = hashlib.sha256(HOOK.encode("utf-8")).hexdigest()
hook_cell = ('''# ==================================================================== E17 hook source (src/e17_hook.py)
# Written to /tmp/e17_hook/sitecustomize.py and put on PYTHONPATH of the E17 server launches only (inert without
# E17_TRACE_DIR); it wraps four SGLang methods in place and never edits a source file.
E17_HOOK_SHA = "''' + HOOK_SHA + '''"
E17_HOOK_SRC = ''' + json.dumps(HOOK) + '''
assert hashlib.sha256(E17_HOOK_SRC.encode("utf-8")).hexdigest() == E17_HOOK_SHA
''')
CELL = (HERE / "e17_cell.py").read_text(encoding="utf-8").replace("\r\n", "\n")
REPORT = (HERE / "e17_report_cell.py").read_text(encoding="utf-8").replace("\r\n", "\n")

cells = [S[0], c1, c1b, S[3], S[4], stub, eng, hook_cell, CELL, REPORT]
md = """# E17 — SGLang silent-accuracy tests (Qwen3.8-27B on 4x L4), engine only

PLAN_E17 rev 2 (Experiment 14 `Bug ang fix/`, 2026-10-26; three GPT audits merged). One session, every server TP4 with
the production numerics (RedHat INT4 W4A16, FP8 e4m3 KV, fp16 SSM, extra_buffer_lazy, overlap, decode CUDA graphs).

- **T1** (#43567, the stale lazy Mamba checkpoint): upstream `test_mamba_cache_kl.py` Class 1 ported. A 4-slot,
  1-running server; warm `[44]*128`, `[42]*128`; the 255-token prompt scored before and after a 32-token seed;
  full-vocabulary KL. Arms: interval 128 (hook on, hook off), interval 256 (+ a 511-token prompt), the forced-fail env,
  DFlash W6 at interval 64 (prompts 246 / 249 / 252). Then the same arms with the #43567 backport. Patch-safety cases
  (EOS, abort, max_new, 5x reuse, cold) with a slot-leak check.
- **L6**: upstream Class 2 (no_buffer + MTP), a known stale-restore path in 0.5.21 (#43387): it shows that the harness
  can detect a stale restore.
- **T2** (only if T1 shows the bug, patched then unpatched): production flags, 16 streams, 4 shared ~30k prompts; failure
  and stale-donation rates; warm vs cold probes.
- **T3b**: target-only vs DFlash, radix cache off, serial and 16 concurrent, greedy 2,048 tokens. **T3a**: fp16 vs fp32
  SSM on 8k-48k prefixes.

**Default profile `E17SHORT` (user 2026-10-26: at most 60 min):** only L1 / L7 (ordinary decode + the patch-safety cases)
and L5 / L9 (DFlash W6), the decisive #43567 pairs. The full plan above (`Q27X_PROFILE=E17`, 170 min) runs later, and
only if the bug shows.

Instrumentation: `e17_hook` as `sitecustomize` on PYTHONPATH of the server only; it wraps four SGLang methods in place
and never edits the source. Verdict ceiling: ENGINE_CORRECTNESS_VALIDATED. Report `e14/e17_report.md`, bundle
`Q27X_E17_BUNDLE.zip`. Commit only (the rerun writes the cell-0 fallback).

**Inputs:** competition data; RedHat INT4, DFlash2 bf16, FP8 repack (the tokenizer); the SGLang 0.5.21 wheelhouse; the
vLLM wheelhouse (the tokenizer pins); the payload dataset (E14 modules).
"""
nb = json.load(open(SRC, encoding="utf-8"))
nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)}] + [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)}
    for c in cells]
for i, c in enumerate(cells):
    compile(c, f"<cell{i}>", "exec")
out = OUT
if out.exists() and sha(out) in PUSHED:
    out = out.with_name(out.stem + "_REBUILD.ipynb")
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
out.write_bytes(data.encode("utf-8"))
print(out.name, sha(out)[:8], f"{len(data.encode('utf-8')) / 1024:.0f} kB", len(cells), "code cells, hook", HOOK_SHA[:8])
