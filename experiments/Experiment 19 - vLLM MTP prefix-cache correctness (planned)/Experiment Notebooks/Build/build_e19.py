"""E19 builder: three GPU diagnostic notebooks (one per Kaggle account) that test whether vLLM 0.19's MTP + prefix
caching (and fp8 KV) silently corrupt Qwen3.8-27B greedy outputs on the exact S1-60K production stack.

Base (read only): S1-60K, Experiment 10, Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb, sha ed530071.
Cells:
  CELL 0      S1 verbatim (mode + identity fallback submission)
  CELL 1      S1 + the E19 profiles + the E19 config block (the default commit profile differs per notebook)
  CELL 2      S1 verbatim (vLLM 0.19.0 + transformers 4.57.6 install from the wheelhouse, tokenizer)
  task order  S1 verbatim (TASK_IDS = the profile's 12 tasks)
  Poetiq core S1 verbatim (poetiq_build_message etc.; its sandbox self-test runs as in S1)
  E19-J       src/e19_judge.py (pure verdict rules; src/e19_compare.py imports the same file offline)
  E19-1       src/e19_core.py with S1 CELL 5's ensure_libcuda_linkable + MAX_JOBS line inserted verbatim
  E19-2       src/e19_run.py with S1's e10_render_request inserted verbatim
Every source is normalised to LF. The builder asserts the verbatim cells and inserted functions byte for byte.
usage: python build_e19.py"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
EXP = NBDIR.parent
SRC_DIR = EXP / "src"
S1 = (EXP.parent / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)" / "Experiment Notebooks" /
      "Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb")
S1_SHA = "ed530071"
PUSHED: set = set()          # shas of notebooks pushed to Kaggle; a rebuild of one goes to *_REBUILD.ipynb
G4B = ["136b0064", "898e7135", "135a2760", "b5ca7ac4", "4c3d4a41", "62593bfd", "247ef758", "bf45cf4b", "f931b4a8",
       "65b59efc", "9385bd28", "cbebaa4b"]          # the 12 Experiment 14 gate tasks (real ARC eval prompts, 3.3-9.4k tokens)
NOTEBOOKS = {          # default commit profile -> notebook (one per account)
    "E19P1": "Q27-XHIGH-E19_MTPPC_P1_D_C.ipynb",
    "E19P2": "Q27-XHIGH-E19_MTPPC_P2_E_A.ipynb",
    "E19P3": "Q27-XHIGH-E19_MTPPC_P3_B_K.ipynb",
}
MAX_BYTES = 400 * 1024


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lf(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:120])
    return text.replace(a, b)


def between(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


assert sha(S1).startswith(S1_SHA), sha(S1)
_nb = json.load(open(S1, encoding="utf-8"))
S = [lf("".join(c["source"])) for c in _nb["cells"] if c["cell_type"] == "code"]
assert len(S) == 8 and "CELL 0" in S[0] and "CELL 1" in S[1] and "CELL 2" in S[2] and "task-order" in S[3] \
    and "poetiq_build_message" in S[4] and "CELL 5" in S[5] and "e10_render_request" in S[6], [s[:60] for s in S]

# ---- CELL 1: S1 + E19 profiles + E19 config
PROFILE_LINE = 'PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "S1COMMIT4").strip().upper()'
_prof = lambda arms, budget: ('{"source": "eval", "task_list": ' + json.dumps(G4B) + ', "time_budget_sec": ' + budget +
                              ',\n                  "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 3 * 60, '
                              '"verify_margin_sec": 1 * 60,\n                  "drain_margin_sec": 8 * 60, '
                              '"admit_margin_sec": 2 * 60, "e19_arms": ' + json.dumps(arms) + '},\n')
E19_PROFILES = (
    "    # E19 (2026-10-27): vLLM 0.19 MTP / prefix-cache / async / KV-dtype correctness probes, 60 min per account;\n"
    "    # e19_arms = [arm, minutes reserved for it while an earlier arm runs]; arm priority D, E, B, K, then C, A (each\n"
    "    # account runs one high-priority MTP arm first); the 12 G4B eval tasks only feed real Poetiq prompts\n"
    '    "E19P1":     ' + _prof([["D", 0], ["C", 22]], "60 * 60") +
    '    "E19P2":     ' + _prof([["E", 0], ["A", 22]], "60 * 60") +
    '    "E19P3":     ' + _prof([["B", 0], ["K", 18]], "60 * 60") +
    "    # all six arms on one account (local harness; about 2.5 h on a GPU)\n"
    '    "E19ALL":    ' + _prof([["A", 0], ["B", 0], ["C", 0], ["D", 0], ["E", 0], ["K", 0]], "240 * 60"))
E19_CONFIG = '''

# ---- E19 (2026-10-27): does vLLM 0.19's MTP + prefix caching (and fp8 KV) silently corrupt greedy outputs on the S1
# stack? Engine only, no solver. Commit mode only: the competition rerun keeps CELL 0's identity submission.
E19_ON = bool(P.get("e19_arms")) and not RERUN
if E19_ON:
    assert SPECULATIVE_CONFIG == {"method": "mtp", "num_speculative_tokens": 3} and KV_CACHE_DTYPE == "fp8_e4m3" \\
        and ENABLE_PREFIX_CACHING and MAX_NUM_SEQS == 16 and MAX_MODEL_LEN == 106496, "E19 needs S1's engine settings"
# each arm changes only its own switches; every other flag = S1's ENGINE_CONFIG (CELL 5)
E19_ARMS = {
    "D": {"spec": 3, "prefix": True, "kv": "fp8_e4m3", "what": "S1 exactly: MTP 3 + prefix caching ON, fp8 KV"},
    "E": {"spec": 3, "prefix": True, "kv": "fp8_e4m3", "async": False, "what": "S1 + --no-async-scheduling"},
    "B": {"spec": 3, "prefix": False, "kv": "fp8_e4m3", "what": "MTP 3, prefix caching OFF"},
    "C": {"spec": None, "prefix": True, "kv": "fp8_e4m3", "what": "MTP OFF, prefix caching ON"},
    "A": {"spec": None, "prefix": False, "kv": "fp8_e4m3", "what": "reference: MTP OFF, prefix caching OFF"},
    "K": {"spec": 3, "prefix": True, "kv": "auto", "what": "S1 with bf16 KV (--kv-cache-dtype auto)"},
}
E19_PLAN = {"n_serial": 6, "max_new": 1024, "g_pre": 512, "cmp": 512, "block": 1600, "backoff": 256,
            # churn: the 6 serial prompts x 4, up to 14 at once, max_tokens staggered, arrivals staggered + refilled
            "churn_reps": 4, "churn_conc": 14, "churn_max_tokens": [256, 512, 768, 1024, 1280, 1536, 1792, 2048],
            "churn_stagger_sec": float(os.environ.get("E19_CHURN_STAGGER_SEC", 0.5)),
            "long_conc": 6, "long": [["long8k", 8192, "other"], ["long30k", 30720, "serial"]]}
E19_LAUNCH_MIN_SEC = float(os.environ.get("E19_LAUNCH_MIN_SEC", 14 * 60))   # no launch with less time before the arm deadline
E19_MIN_SERIAL_SEC = float(os.environ.get("E19_MIN_SERIAL_SEC", 150))       # one serial prompt (3 x 1,024 greedy tokens)
E19_MIN_CHURN_SEC = float(os.environ.get("E19_MIN_CHURN_SEC", 150))         # the churn stage (24 requests, 14 at once)
E19_MIN_LONG_SEC = float(os.environ.get("E19_MIN_LONG_SEC", 240))           # one long-context stage
E19_REPORT_RESERVE_SEC = float(os.environ.get("E19_REPORT_RESERVE_SEC", 90))
E19_SERIAL_IDX, E19_LONG, E19_CHURN_JOBS, E19_CUR = [], [], [], None
CONFIG["e19"] = {"on": E19_ON, "arms": P.get("e19_arms"), "plan": E19_PLAN}
print("E19:", json.dumps(CONFIG["e19"]))
'''


def cell1(default_profile):
    c = rep(S[1], PROFILE_LINE, PROFILE_LINE.replace('"S1COMMIT4"', json.dumps(default_profile)))
    c = rep(c, "    # hidden test: every task;", E19_PROFILES + "    # hidden test: every task;")
    return c + E19_CONFIG


# ---- inserted S1 functions (verbatim)
LIBCUDA = between(S[5], "def ensure_libcuda_linkable(", "\nCUDA_LINK = ensure_libcuda_linkable()")
assert LIBCUDA.rstrip().endswith('os.environ.setdefault("MAX_JOBS", "4")              # bound FlashInfer JIT build parallelism')
RENDER = between(S[6], "def e10_render_request(", "\ndef e10_seed(")
assert "poetiq_build_message(" in RENDER and "reasoning_effort=REASONING_EFFORT" in RENDER
JUDGE = lf((SRC_DIR / "e19_judge.py").read_text(encoding="utf-8"))
CORE = rep(lf((SRC_DIR / "e19_core.py").read_text(encoding="utf-8")), "# @@S1_LIBCUDA@@\n",
           "# S1 CELL 5, verbatim:\n" + LIBCUDA.rstrip() + "\n")
RUN = rep(lf((SRC_DIR / "e19_run.py").read_text(encoding="utf-8")), "# @@S1_RENDER@@\n",
          "# S1 solver cell, verbatim:\n" + RENDER.rstrip() + "\n")
assert LIBCUDA in CORE and RENDER.rstrip() in RUN


def markdown(default_profile):
    arms = {"E19P1": "D (S1 exactly), then C (MTP off, prefix on)",
            "E19P2": "E (S1 + --no-async-scheduling), then A (MTP off, prefix off: the pure-noise calibrator)",
            "E19P3": "B (MTP on, prefix off: the MTP-noise calibrator), then K (S1 with bf16 KV)"}[default_profile]
    return f"""# E19: vLLM 0.19 MTP + prefix-cache correctness on the S1-60K stack ({default_profile})

Does vLLM 0.19's MTP speculative decoding with prefix caching, async scheduling or the fp8 KV cache silently corrupt Qwen3.8-27B
outputs on our exact production stack (S1-60K, LB 34.58)? Engine only, no solver; greedy decoding; about 60 min.

**This notebook's arms:** {arms}. The three notebooks (P1 / P2 / P3) run in parallel on three accounts (arm priority
D, E, B, K, then C, A). `src/e19_compare.py` judges all six arms together from the three bundles.

**Stack (S1 CELL 2 / CELL 5):** vLLM 0.19.0 + transformers 4.57.6 from `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`;
FP8 repack, TP4, fp8_e4m3 KV, MTP 3, FLASHINFER when `-lcuda` links, prefix caching, 4,096 batched tokens, memory 0.90,
custom all-reduce off, language-model-only, max_model_len 106,496, 16 seqs, seed 20261005. Here as an OpenAI server from
the kernel's Python, one per arm, killed between arms.

**Probes (identical token ids on every arm):** 12 real Poetiq prompts (S1's render code, xhigh). Serial cold / warm /
cold-repeat greedy 1,024 tokens on 6 of them; a churn pass (24 requests, up to 14 at once, max_tokens 256-2,048,
staggered and refilled arrivals: the batch reorders continuously) compared with the same prompts run alone;
phase-2-style continuations at about 8k and 30k context over the server's own generated tokens; MTP acceptance per
/metrics window.

**Main discriminator: within the arm** (serial A/A noise vs cache hit, vs churn, vs continuation); only large-margin
divergences (top-1 minus top-2 > 1 nat at the first divergent token) and corruption flags count; churn is calibrated
by arms A / B; the token comparison against arm A is informational.

**Verdicts:** PASS / FAIL / NOT_TESTED_BUDGET / INCOMPLETE / INCONCLUSIVE / LAUNCH_FAILED per arm, in
`q27x/e19/e19_report.md`; every raw row in `q27x/e19/e19_rows.jsonl`; bundle `Q27X_E19_BUNDLE.zip`.

**Inputs:** the competition data, the vLLM 0.19 wheelhouse and the FP8 repack model (the same inputs as S1-60K).
"""


def build():
    out = {}
    for prof, fname in NOTEBOOKS.items():
        cells = [S[0], cell1(prof), S[2], S[3], S[4], JUDGE, CORE, RUN]
        for i, c in enumerate(cells):
            compile(c, f"<{fname} cell {i}>", "exec")
        nb = json.loads(json.dumps(_nb))
        nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": markdown(prof).splitlines(keepends=True)}] + [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)}
            for c in cells]
        data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        assert len(data) < MAX_BYTES, (fname, len(data))
        assert b"\r" not in data
        path = NBDIR / fname
        if path.exists() and sha(path) in PUSHED:
            path = path.with_name(path.stem + "_REBUILD.ipynb")
        path.write_bytes(data)
        # the verbatim cells really are S1's
        back = ["".join(c["source"]) for c in json.loads(data)["cells"] if c["cell_type"] == "code"]
        assert back[0] == S[0] and back[2] == S[2] and back[3] == S[3] and back[4] == S[4]
        out[prof] = (path.name, sha(path), len(data))
        print(f"{path.name}  sha {sha(path)[:8]}  {len(data) / 1024:.0f} kB  {len(cells)} code cells")
    return out


if __name__ == "__main__":
    build()
