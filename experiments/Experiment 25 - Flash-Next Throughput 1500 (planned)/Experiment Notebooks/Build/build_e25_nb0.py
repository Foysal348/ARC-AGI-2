"""E25 NB0 FP8 quality isolation (lead spec 2026-10-31 + addendum: boot E = Merlin's sm_86 byte decoder on sm_89,
forced-rejection parity). Qwen3.8-Flash-Next on Kaggle 4xL4 with the Merlin engine (vLLM 0.32.0 fork, profile
fn-quad-24g-sm89, TP2xPP2+EP). Teacher-forced (TF) only, no throughput ladders.

Why: RC2 (E13 RC2 GPU v1) measured TF 40k FP8-KV vs BF16-KV mean |dlogprob| 1.88, p99 5.46, rank-1 agreement 0.367 with a
BF16 A/A of exactly 0, and its one long FP8 answer was wrong. Prefix-cache hits and differing prompt ids are ruled out.
Open: prompt-logprob path artefact vs a prefill defect of the never-run-on-Ada sm_89 native-E4M3 QSA path vs an
MTP / PP / RecoverSSM interaction.

Boots (order D, C, E, B, A; prefix caching ON in every boot; each records its config diff vs RC2 FP8):
  D  FP8 KV, MTP K3                     = RC2 FP8 config
  C  BF16 KV, MTP K3                    = RC2 BF16 config (seqs 3, captures to 12)
  E  FP8 KV, MTP K3, byte decoder       = D with qsa_e4m3_uses_native_cast() -> False (Merlin's sm_86-qualified path),
                                          env-gated patch of the installed tree, on-device decoder check first
  B  FP8 KV, MTP off                    = D without --speculative-config (Merlin: "the same command without it")
  A  BF16 KV, MTP off                   = C without --speculative-config, seqs 8 + FP8 capture list (A-B = KV dtype only)
Per boot: probes + sampled + RC2 greedy oracle rows (vs RC2's recorded rows and earlier NB0 boots); TF at 2k / 8k / 40k
/ 90k (RC2 prompt(band, 5); 40k v5 = RC2's TF_P) fresh then repeat (the per-boot A/A; prompt_logprobs never read the
prefix cache, so cached tokens are recorded and expected 0); full per-position raw arrays (actual-token logprob, rank,
rank-1 id, rank-1 logprob, top-5) on disk for every request; decode-vs-TF self-consistency + cold-vs-cached decode at
2k / 8k / 40k (v6, 256 greedy tokens); static scales file + matched layer names + the MTP layer's default-scale line.
Offline: A-B, C-D, A-C, B-D, C-E, D-E per band (last 256 = RC2's metric, full arrays: context bins, chunk edges, +-2
shift), cross-boot greedy parity (MTP on vs off: first divergence + margin), decision table, screening + triage gates.

Reuse (verbatim): the RC2 builder (Experiment 13 build_e13_rc2.py) is executed up to its notebook-writing part, so its
cells come out exactly as RC2 composed them from E23; they are checked byte-for-byte against the Kaggle-run RC2 notebook
(sha 12eaa96c). NB0 applies marked one-line edits to CELL 1 (budget, output names), cuts CELL 5 before RC2's arms, and
adds two new cells (NB0 library, NB0 main). RC2 functions are extended only by rebinding (RC2's own `_gated_e23` pattern).
Nothing in Experiment 13 or Experiment 23 is written.

Run: python build_e25_nb0.py  ->  ../Q27-XHIGH-E25_NB0_FP8_QUALITY.ipynb (+ nb0_code_export.py for the local harness)
"""
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
RC2_BUILDER = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/"
                   r"Experiment Notebooks/Build/build_e13_rc2.py")
RC2_NB = RC2_BUILDER.parents[1] / "Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb"
RC2_NB_SHA = "12eaa96cea2ced510ef54bca83586a3f545cb0609ccdb5634cf32ecfb9fcafa5"     # the RC2 GPU commit (S0 PASSED)
RC2_OUT = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")                             # RC2's real output
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")                           # Merlin ad42e37 (= the payload)
OUT_NB = HERE.parent / "Q27-XHIGH-E25_NB0_FP8_QUALITY.ipynb"
PUSHED = set()        # shas of NB0 versions that ran on Kaggle (never overwrite one; rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def cut(text, start, end, label):
    assert text.count(start) == 1, f"{label}: start x{text.count(start)}"
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


# ---------------------------------------------------------------- RC2 cells, exactly as RC2 composed them
RC2_TEXT = RC2_BUILDER.read_text(encoding="utf-8").replace("\r\n", "\n")
_prefix = RC2_TEXT[:RC2_TEXT.index('\nsubs = {"__LAB__": json.dumps(LAB), "__PLE__": json.dumps(PLE)}')]
assert "OUT_NB.write" not in _prefix and "write_text(\"\\n\\n\".join" not in _prefix, "RC2 prefix must not write at build time"
RC2 = {"__file__": str(RC2_BUILDER), "__name__": "rc2_builder_prefix"}
exec(compile(_prefix, str(RC2_BUILDER), "exec"), RC2)            # definitions only: reads E23 / LAB / parts.json
rc2_subs = {"__LAB__": json.dumps(RC2["LAB"]), "__PLE__": json.dumps(RC2["PLE"])}
rc2_raw = []
for c in (RC2[k] for k in ("CELL1", "CELL2", "CELL3", "CELL4", "CELL5", "CELL6")):
    for k, v in rc2_subs.items():
        c = c.replace(k, v)
    rc2_raw.append(c)
rc2_cells = [c if i == 0 else "if not RERUN:\n" + textwrap.indent(c, "    ") for i, c in enumerate(rc2_raw)]
assert hashlib.sha256(RC2_NB.read_bytes()).hexdigest() == RC2_NB_SHA, "RC2 notebook is not the Kaggle-run version"
_rc2_nb = json.loads(RC2_NB.read_text(encoding="utf-8"))
assert ["".join(c["source"]) for c in _rc2_nb["cells"] if c["cell_type"] == "code"] == rc2_cells, \
    "RC2 cells rebuilt from build_e13_rc2.py differ from the Kaggle-run RC2 notebook"

# ---------------------------------------------------------------- NB0 edits of RC2 CELL 1 (marked) and the CELL 5 cut
CELL1 = rc2_raw[0]
CELL1 = once(CELL1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows',
             'BUDGET = int(os.environ.get("NB0_BUDGET_SEC") or 100 * 60)     # NB0: 5 short TF boots (lead: ~100 min)', "budget")
CELL1 = once(CELL1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))',
             'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb0"))                 # NB0', "out")
CELL1 = once(CELL1, '"what": "E13 RC2 Flash-Next on Merlin"', '"what": "E25 NB0 FP8 quality isolation (Flash-Next on Merlin)"', "what")
CELL1 = once(CELL1, 'tmp = OUT / "rc2_results.json.tmp"', 'tmp = OUT / "nb0_results.json.tmp"                 # NB0', "save tmp")
CELL1 = once(CELL1, 'tmp.replace(OUT / "rc2_results.json")', 'tmp.replace(OUT / "nb0_results.json")', "save file")
CELL1 = once(CELL1, 'log("RC2 | budget"', 'log("NB0 | budget"', "log")
CELL1 = CELL1.replace("# ==================================================================== CELL 1 — config, inputs, lab module",
                      "# ==================================================================== CELL 1 — config, inputs, lab module (RC2; NB0 edits marked)", 1)
CELL2, CELL3, CELL4 = rc2_raw[1], rc2_raw[2], rc2_raw[3]
_main5 = '\nif not RERUN and not os.environ.get("RC2_UNIT"):\n    m = R["sections"]["A0"]["merge"]'
assert rc2_raw[4].count(_main5) == 1
CELL5 = rc2_raw[4][:rc2_raw[4].index(_main5)] + "\n# NB0: RC2's arms (A1 / A3 / F) and its main block are not used; NB0's boots follow in CELL 7.\n"

# ---------------------------------------------------------------- RC2 real facts: config reference, TF anchor, oracle rows, prompt shas
_R = json.loads((RC2_OUT / "rc2_results.json").read_text(encoding="utf-8"))
_plans = _R["setup"]["merlin_plans"]
_f, _b = _R["sections"]["A3"]["merlin_fp8"], _R["sections"]["A1"]["merlin_bf16"]
RC2_REF = {
    "fp8": {"argv_tail": _plans["merlin_fp8"]["argv_tail"], "env": _plans["merlin_fp8"]["env_keys"], "kv_bytes": 2479838700,
            "kv_tokens": _f["servers"]["merlin_fp8"]["kv_tokens"], "seqs": 8},
    "bf16": {"argv_tail": _plans["merlin_bf16"]["argv_tail"], "env": _plans["merlin_bf16"]["env_keys"], "kv_bytes": 2464415744,
             "kv_tokens": _b["servers"]["merlin_bf16"]["kv_tokens"], "seqs": 3},
    "tf40k": {"fp8_vs_bf16": _f["tf"]["vs_bf16"]["40k"], "bf16_aa": _b["tf"]["aa"]["40k"]},
    "oracle_rows": {"rc2_bf16": [], "rc2_fp8": []}, "prompt_shas": {},
}
assert RC2_REF["fp8"]["kv_tokens"] == 513940 and RC2_REF["bf16"]["kv_tokens"] == 269042, "unexpected RC2 results file"
assert RC2_REF["tf40k"]["fp8_vs_bf16"]["mean_abs_dlogprob"] == 1.8764 and RC2_REF["tf40k"]["fp8_vs_bf16"]["rank1_agreement"] == 0.3672
assert RC2_REF["fp8"]["argv_tail"][0] == "127.0.0.1" and "--speculative-config" in RC2_REF["fp8"]["argv_tail"]
for _l in (RC2_OUT / "traces" / "requests.jsonl").read_text(encoding="utf-8").splitlines():
    _r = json.loads(_l)
    RC2_REF["prompt_shas"][_r["prompt_sha"]] = _r["prompt_len"]
    _s = _r.get("sampling") or {}
    if _s.get("temperature") == 0.0 and _s.get("max_tokens") == 128 and _s.get("logprobs") == 2:
        RC2_REF["oracle_rows"][{8000: "rc2_bf16", 8001: "rc2_fp8"}[_r["port"]]].append(
            {k: _r[k] for k in ("text", "tokens", "margins", "error", "prompt_sha", "prompt_len", "finish_reason")})
assert [len(v) for v in RC2_REF["oracle_rows"].values()] == [4, 4], "RC2 recorded oracle rows: expected 4 + 4"
assert len(RC2_REF["prompt_shas"]) == 25, len(RC2_REF["prompt_shas"])   # 47 requests, 25 distinct prompts

# ---------------------------------------------------------------- boot E: the audited Merlin source (ad42e37) and the env-gated patch
OPS_REL, QSA_REL = "vllm/models/qwen4_exp/nvidia/ops/qsa.py", "vllm/models/qwen4_exp/nvidia/qsa.py"
_ops = (MERLIN_SRC / OPS_REL).read_bytes()
_qsa = (MERLIN_SRC / QSA_REL).read_bytes()
GATE = '__import__("os").environ.get("NB0_QSA_FORCE_BYTE_DECODER") == "1"'
QSA_PATCH = {
    "ops_rel": OPS_REL, "qsa_rel": QSA_REL,
    "ops_sha": hashlib.sha256(_ops).hexdigest(), "qsa_sha": hashlib.sha256(_qsa).hexdigest(),
    # the one function both files consult (ops/qsa.py:28-36, lru_cache): the env gate goes first in its body
    "ops_old": "    capability = current_platform.get_device_capability()\n"
               "    return capability is not None and capability.to_int() >= 89\n",
    "ops_new": f"    if {GATE}:  # NB0-E25 boot E: Merlin's sm_86 byte decoder on sm_89\n"
               "        print(f\"NB0 boot E: QSA E4M3 byte decoder forced (qsa_e4m3_uses_native_cast -> False) "
               "pid={__import__('os').getpid()}\", flush=True)\n"
               "        return False\n"
               "    capability = current_platform.get_device_capability()\n"
               "    return capability is not None and capability.to_int() >= 89\n",
    # qsa.py:231 (_validated_qsa_fp8_dtype): `!= 86 and not native` would RAISE on sm_89 once native is False
    "qsa_old": "        capability.to_int() != 86 and not qsa_e4m3_uses_native_cast()\n    ):\n",
    "qsa_new": "        capability.to_int() != 86 and not qsa_e4m3_uses_native_cast()\n"
               f"        and not ({GATE} and capability.to_int() >= 89)  # NB0-E25 boot E\n    ):\n",
    "marker": "NB0-E25 boot E",
    # every other consumer, audited (must each occur exactly once; the function name 3 times per file)
    "audit": {
        "ops": ["@lru_cache(maxsize=1)\ndef qsa_e4m3_uses_native_cast() -> bool:\n",
                "        if qsa_e4m3_uses_native_cast():\n            # SM89+: upstream's native E4M3 path (vllm#55557).\n            kv_dtype = 3\n",
                "        else:\n            kv_dtype = 1\n",
                "        if qsa_e4m3_uses_native_cast():\n            warm_kv_dtype = 3\n",
                "            keys = _decode_e4m3_to_bf16(key_bytes)\n", "            values = _decode_e4m3_to_bf16(value_bytes)\n"],
        "qsa": ["from .ops.qsa import qsa_e4m3_uses_native_cast\n",
                "            and (not fp8_layer or qsa_e4m3_uses_native_cast())\n",
                "        if not self.use_fused_qsa_prepare:\n",
                "            impl.do_kv_cache_update(\n"],
        "name_count": {"ops": 3, "qsa": 3},
    },
}
for _k, _t in (("ops", _ops.decode()), ("qsa", _qsa.decode())):
    assert "\r" not in _t, f"{_k}: CRLF in the audited source"
    assert _t.count(QSA_PATCH[_k + "_old"]) == 1, f"{_k}: patch anchor"
    for _a in QSA_PATCH["audit"][_k]:
        assert _t.count(_a) == 1, f"{_k}: audit anchor x{_t.count(_a)}: {_a[:60]!r}"
    assert _t.count("qsa_e4m3_uses_native_cast") == QSA_PATCH["audit"]["name_count"][_k], f"{_k}: name count"
    _p = _t.replace(QSA_PATCH[_k + "_old"], QSA_PATCH[_k + "_new"])
    compile(_p, _k, "exec")                                          # the patched file still parses
    assert _p.count(QSA_PATCH["marker"]) == 1

MD0 = """# E25 NB0: FP8 quality isolation (Qwen3.8-Flash-Next, Merlin, 4×L4), teacher-forced only

RC2 measured teacher-forced 40k FP8-KV vs BF16-KV: mean |Δlogprob| **1.88**, p99 5.46, rank-1 agreement **0.367**
(BF16 A/A exactly 0), and its one long FP8 answer was wrong. Prefix-cache hits and prompt-id differences are ruled out.
NB0 separates a prompt-logprob path artefact, a prefill defect of the never-run-on-Ada **sm_89 native-E4M3 QSA read
path**, and an **MTP / PP / RecoverSSM** interaction. Five short boots, no throughput ladders, prefix caching on in all:

| Boot | KV | MTP | Config vs RC2 FP8 |
|---|---|---|---|
| D | FP8 | K3 | none (= RC2 FP8) |
| C | BF16 | K3 | = RC2 BF16 (`--kv-cache-dtype bfloat16`, seqs 3, captures to 12, KV pin 2,464,415,744) |
| E | FP8 | K3 | = D + `NB0_QSA_FORCE_BYTE_DECODER=1` (Merlin's sm_86 byte decoder, unfused prepare), own compile cache |
| B | FP8 | off | = D without `--speculative-config` (Merlin: RecoverSSM falls back to the stock GDN path) |
| A | BF16 | off | = C without `--speculative-config`, seqs 8, FP8 capture list (so A vs B differs in KV dtype only) |

Per boot: teacher-forced 2k / 8k / 40k / 90k (RC2 `prompt(band, 5)`), fresh + repeat, full raw per-position arrays;
decode-vs-teacher-forced self-consistency and cold-vs-cached decode at 2k / 8k / 40k; RC2 greedy oracle rows; static
scales. Comparisons A–B, C–D, A–C, B–D, C–E, D–E with a decision table, a screening gate (HOLD if rank-1 < 0.90 or
mean > 0.15) and a triage gate (NO_GO if mean > 1.0 or rank-1 < 0.5, else PENDING_LONG_ARC_ACCURACY; never GO).
Commit run only; a competition rerun writes a fallback submission.
"""

CELL6 = r'''# ==================================================================== CELL 6 — NB0 library (new code)
import textwrap
import numpy as np
import urllib.request as UR
import urllib.error as UE
R.pop("capacity_plan", None)          # RC2-only records written by RC2's CELL 5 definitions
R.pop("correctness_meaning", None)
RC2_REF = json.loads(__RC2REF__)       # RC2 GPU v1 facts (argv / env / TF anchor / oracle rows / prompt shas), build-time
QSA_PATCH = json.loads(__QSAPATCH__)   # boot E: audited Merlin ad42e37 source shas, anchors and the env-gated patch
NB0 = {"topk": int(os.environ.get("NB0_TF_TOPK") or 5), "sc_tok": int(os.environ.get("NB0_SC_TOKENS") or 256),
       "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6}, "tf_bands": ["2k", "8k", "40k", "90k"], "sc_bands": ["2k", "8k", "40k"],
       "last": 256, "boot_watchdog_sec": float(os.environ.get("NB0_BOOT_WATCHDOG_SEC") or 1800),
       "ready_sec": float(os.environ.get("NB0_READY_SEC") or 1500), "host_floor_mib": (69 + 4) * 1024}
SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # external audit 2 (verified): conservative screening, not a published limit
TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # external audit 3 (verified): severe-discrepancy triage
_FP8_FLAGS = None


def flags(argv):
    """--flag -> value, the LAST occurrence wins (argparse); a flag followed by another flag is a switch (True)."""
    out = {}
    for i, a in enumerate(argv):
        if isinstance(a, str) and a.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            out[a] = nxt if (nxt is not None and not str(nxt).startswith("--")) else True
    return out


PORT_BASE = int(os.environ.get("NB0_PORT_BASE") or 8010)       # Kaggle 8010-8014 (Merlin's 8000-8355 port scheme)
FP8_CAPTURES = flags(RC2_REF["fp8"]["argv_tail"])["--compilation-config"]     # RC2 FP8's exact JSON (captures to 32)
BOOTS = {
    "D": {"name": "nb0_D_fp8_mtp3", "kv": "fp8", "mtp": True, "port": PORT_BASE + 0, "seqs": None, "captures": None, "byte": False,
          "title": "FP8 KV, MTP K3 (= RC2 FP8)"},
    "C": {"name": "nb0_C_bf16_mtp3", "kv": "bf16", "mtp": True, "port": PORT_BASE + 1, "seqs": None, "captures": None, "byte": False,
          "title": "BF16 KV, MTP K3 (= RC2 BF16)"},
    "E": {"name": "nb0_E_fp8_mtp3_bytedec", "kv": "fp8", "mtp": True, "port": PORT_BASE + 2, "seqs": None, "captures": None,
          "byte": True, "title": "FP8 KV, MTP K3, Merlin sm_86 byte decoder forced on sm_89"},
    "B": {"name": "nb0_B_fp8_mtp0", "kv": "fp8", "mtp": False, "port": PORT_BASE + 3, "seqs": None, "captures": None, "byte": False,
          "title": "FP8 KV, MTP off (target only)"},
    "A": {"name": "nb0_A_bf16_mtp0", "kv": "bf16", "mtp": False, "port": PORT_BASE + 4, "seqs": 8, "captures": FP8_CAPTURES,
          "byte": False, "title": "BF16 KV, MTP off (target only), seqs 8 + FP8 captures"},
}
for _k, _b in BOOTS.items():
    _b["key"] = _k
    _b["step"] = f"{_k}_{_b['name']}"
    _b["want_seqs"] = _b["seqs"] or RC2_REF[_b["kv"]]["seqs"]
BOOT_BY_NAME = {b["name"]: b for b in BOOTS.values()}
# Order (decided, see the report): D first = the production candidate on the RC2-proven config (its own A/A,
# self-consistency, shift and chunk tests settle "path artefact?" alone); C next = the RC2 pair with raw arrays; E third
# = the decisive read-path test (lead: priority over A's 90k band); the two never-run MTP-off boots last, B before A
# (B-D is the MTP effect on the production KV dtype; if B cannot launch, A is skipped: same MTP-off path).
ORDER = ["D", "C", "E", "B", "A"]
EST_MIN = {"D": 15, "C": 13, "E": 15, "B": 12, "A": 12}     # step() needs est*60 + 300 s left
PAIRS = {"AB": ("A", "B", "KV dtype only, target only (MTP off)"),
         "CD": ("C", "D", "KV dtype under MTP K3 (the RC2 pair)"),
         "AC": ("A", "C", "MTP effect, BF16 KV"),
         "BD": ("B", "D", "MTP effect, FP8 KV"),
         "CE": ("C", "E", "KV dtype under MTP K3, FP8 read through Merlin's byte decoder"),
         "DE": ("D", "E", "FP8 read path: sm_89 native cast vs byte decoder")}
ORACLE_REFS = {"D": ("rc2_fp8", "rc2_bf16"), "C": ("rc2_bf16", "rc2_fp8", "D"), "E": ("rc2_fp8", "D", "C"),
               "B": ("rc2_fp8", "D", "C"), "A": ("rc2_bf16", "C", "B", "D")}
for _k, _rows in RC2_REF["oracle_rows"].items():
    ROWS[_k] = _rows                    # RC2's recorded greedy rows (2k v0, 2k v1, 8k v2, 40k v3; 128 tokens)
RAW_DIR = OUT / "tf_raw"
RAW_DIR.mkdir(exist_ok=True)
NB0_REQ = TRACES / "nb0_requests.jsonl"         # every non-streaming request (teacher-forced, decode) with its full text
TFR, SCR, LOADMON, WD, SIDS, DROP = {}, {}, {}, {}, set(), set()
PROC = Path(os.environ.get("NB0_PROC", "/proc"))
CG = Path(os.environ.get("NB0_CGROUP", "/sys/fs/cgroup"))
SHM0 = shm_names()                               # /dev/shm at NB0 start: anything later and psm_* is ours
ENV_KEYS = ("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "NB0_")


def _set_value(a, flag, value):
    idx = [i for i, x in enumerate(a) if x == flag]
    if len(idx) != 1:
        raise ValueError(f"{flag} occurs {len(idx)} times in the Merlin argv (expected 1)")
    a[idx[0] + 1] = value


def nb0_argv(argv, boot):
    """Merlin's argv with this boot's edits. MTP off = the same command without --speculative-config (Merlin
    models/flash-next/CHANGELOG.md: "the no-MTP line is the same command without --speculative-config"; launch.py:441/452
    always emits it, R.MTP_K = 3 has no env knob). seqs / captures: the value of the existing flag (no new flags)."""
    a = [str(x) for x in argv]
    if not boot["mtp"]:
        idx = [i for i, x in enumerate(a) if x == "--speculative-config"]
        if len(idx) != 1:
            raise ValueError(f"--speculative-config occurs {len(idx)} times (expected 1)")
        del a[idx[0]:idx[0] + 2]
    if boot.get("seqs"):
        _set_value(a, "--max-num-seqs", str(boot["seqs"]))
    if boot.get("captures"):
        _set_value(a, "--compilation-config", boot["captures"])
    if "--enable-prefix-caching" not in a:
        raise ValueError("prefix caching must stay on in every NB0 boot")
    return a


def config_diff(argv, env, kv_bytes, ref="fp8"):
    """Flag / env / KV-pin diff of a boot vs RC2's recorded plan (last occurrence wins on both sides)."""
    rf, cf = flags(RC2_REF[ref]["argv_tail"]), flags(argv)      # RC2's tail starts at --host's value (127.0.0.1)
    head = set(flags(argv[:-len(RC2_REF[ref]["argv_tail"]) + 2])) - set(rf)   # Merlin flags before RC2's recorded window
    fl = {k: [rf.get(k), cf.get(k)] for k in sorted(set(rf) | set(cf))
          if k not in ("--port", "--host") and k not in head and rf.get(k) != cf.get(k)}
    renv = RC2_REF[ref]["env"]
    cenv = {k: v for k, v in env.items() if k.startswith(ENV_KEYS)}
    skip = ("VLLM_PORT", "MASTER_PORT", "NCCL_VERSION")
    ev = {k: [renv.get(k), cenv.get(k)] for k in sorted(set(renv) | set(cenv)) if k not in skip and renv.get(k) != cenv.get(k)}
    out = {"flags": fl, "env": ev}
    if kv_bytes != RC2_REF[ref]["kv_bytes"]:
        out["kv_bytes"] = [RC2_REF[ref]["kv_bytes"], kv_bytes]
    return out


_merlin_fn_server_rc2 = merlin_fn_server


def merlin_fn_server(name, kv, port, kv_bytes):
    """RC2's server (Merlin launch.build + host_phase, verbatim) + NB0's per-boot argv / env edits, the load monitor and
    the per-boot watchdog. FAKE: RC2's recorded Kaggle argv for this KV dtype (port / pin substituted) into the fake."""
    b = BOOT_BY_NAME[name]
    s = _merlin_fn_server_rc2(name, kv, port, kv_bytes)
    if FAKE:
        tail, out = RC2_REF[kv]["argv_tail"], []
        for i, x in enumerate(tail):
            out.append(str(port) if i and tail[i - 1] == "--port" else str(kv_bytes) if i and tail[i - 1] == "--kv-cache-memory-bytes" else x)
        s.argv = [sys.executable, FAKE, str(CKPT), "--host"] + out
        env_basis = dict(RC2_REF[kv]["env"])
        if kv == "fp8":
            s.env["VLLM_QSA_KV_SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
    s.argv = nb0_argv(s.argv, b)
    if b["byte"]:
        s.env.update(NB0_QSA_FORCE_BYTE_DECODER="1", VLLM_CACHE_ROOT=f"{TMP}/merlin-cache-bytedec")   # never D's AOT graphs
    else:
        s.env.pop("NB0_QSA_FORCE_BYTE_DECODER", None)
    if FAKE:
        env_basis.update({k: s.env[k] for k in ("NB0_QSA_FORCE_BYTE_DECODER", "VLLM_CACHE_ROOT") if b["byte"] and k in s.env})
    envd = env_basis if FAKE else s.env
    R["setup"].setdefault("nb0_plans", {})[name] = {
        "boot": b["key"], "title": b["title"], "argv_tail": s.argv[-64:], "kv_bytes": kv_bytes,
        "env": {k: envd[k] for k in sorted(envd) if k.startswith(ENV_KEYS)},
        "diff_vs_rc2_fp8": config_diff(s.argv, envd, kv_bytes, "fp8"), "diff_vs_rc2_bf16": config_diff(s.argv, envd, kv_bytes, "bf16")}
    save()
    deadline = time.time() + max(120, min(NB0["boot_watchdog_sec"], left() - 240))
    threading.Thread(target=load_monitor, args=(s, b["key"], deadline), daemon=True).start()
    threading.Thread(target=boot_watchdog, args=(s, b["key"], deadline), daemon=True).start()
    return s


# ---------------------------------------------------------------- host memory, processes, page cache (new)
def cg_snapshot(full=True):
    """cgroup v2 memory files (MiB for sizes). avail_mib = max - (current - inactive_file), Merlin's own rule."""
    o, kv = {}, {}
    for n in ("memory.current", "memory.max", "memory.peak"):
        try:
            o[n] = (CG / n).read_text().strip()
        except Exception:
            pass
    try:
        st = (CG / "memory.stat").read_text()
        kv = dict(l.split() for l in st.splitlines() if len(l.split()) == 2)
        o["stat_mib"] = {k: int(kv[k]) >> 20 for k in ("anon", "file", "active_file", "inactive_file", "shmem", "file_mapped",
                                                       "file_dirty", "file_writeback") if k in kv}
        o["stat_count"] = {k: int(kv[k]) for k in ("pgmajfault", "workingset_refault_file", "pgscan", "pgsteal") if k in kv}
        if full:
            o["memory.stat"] = st[:6000]
    except Exception:
        pass
    for n in ("memory.events", "memory.pressure"):
        try:
            o[n] = (CG / n).read_text().strip()[:800]
        except Exception:
            pass
    try:
        mx = o.get("memory.max")
        if mx and mx != "max":
            o["avail_mib"] = (int(mx) - (int(o["memory.current"]) - int(kv.get("inactive_file", 0)))) >> 20
    except Exception:
        pass
    return o


def _pids():
    try:
        return [int(p.name) for p in PROC.iterdir() if p.name.isdigit()]
    except Exception:
        return None


def _stat(pid):
    try:
        t = (PROC / str(pid) / "stat").read_text()
    except Exception:
        return None
    i = t.rfind(")")
    f = t[i + 2:].split()
    try:
        return {"pid": pid, "comm": t[t.find("(") + 1:i], "state": f[0], "ppid": int(f[1]), "session": int(f[3]),
                "minflt": int(f[7]), "majflt": int(f[9]), "utime": int(f[11]), "stime": int(f[12]), "rss_pages": int(f[21])}
    except Exception:
        return None


def _io(pid):
    try:
        return {k: int(v) for k, v in (l.split(": ") for l in (PROC / str(pid) / "io").read_text().splitlines())
                if k in ("rchar", "read_bytes", "syscr")}
    except Exception:
        return {}


def session_procs(sids):
    out = []
    for pid in _pids() or []:
        st = _stat(pid)
        if st and st["session"] in sids and st["state"] != "Z":
            st.update(_io(pid))
            out.append(st)
    return out


_KILL = os.kill                                  # unit tests replace it (never signal a real pid)


def reap_leftovers(wait=30):
    """Every process of an NB0 server session (we start each server with start_new_session, so sid = server pid) must
    be gone before the next boot: SIGKILL the leftovers (PLE offload, workers), wait, re-check; fail closed."""
    if _pids() is None:
        return {"ok": bool(FAKE), "why": "no /proc" + (" (local harness)" if FAKE else "")}
    found = session_procs(SIDS)
    killed = []
    for p in found:
        try:
            _KILL(p["pid"], 9)
            killed.append([p["pid"], p["comm"]])
        except Exception as e:
            killed.append([p["pid"], p["comm"], repr(e)[:80]])
    t = time.time()
    while session_procs(SIDS) and time.time() - t < wait:
        time.sleep(1)
    still = session_procs(SIDS)
    sus = []
    for pid in _pids() or []:
        if pid == os.getpid():
            continue
        try:
            cl = (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if re.search(r"vllm|ple_offload|EngineCore|fn_merlin", cl) and not re.search(r"ipykernel|jupyter", cl):
            sus.append([pid, cl[:160]])
    return {"ok": not still, "found": [[p["pid"], p["comm"]] for p in found][:20], "killed": killed[:20],
            "still": [[p["pid"], p["comm"]] for p in still][:20], "other_engine_processes": sus[:10], "sessions": sorted(SIDS)}


def unlink_own_psm():
    out = {"removed": [], "kept": []}
    try:
        for p in sorted(SHM.iterdir()):
            if p.name.startswith("psm_") and p.name not in SHM0:
                try:
                    p.unlink()
                    out["removed"].append(p.name)
                except Exception as e:
                    out["kept"].append([p.name, repr(e)[:80]])
    except Exception as e:
        out["error"] = repr(e)[:200]
    return out


def ckpt_files():
    fs = set()
    for d in [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else []):
        try:
            for p in Path(d).rglob("*"):
                try:
                    rp = p.resolve()
                    if rp.is_file():
                        fs.add(rp)
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(fs)


def drop_ckpt_cache():
    """posix_fadvise(DONTNEED) on every merged file's target and every file of the Model mounts (a successful call is not
    proof of reclaim: the cgroup file / inactive_file before and after are recorded by the caller)."""
    fad = getattr(os, "posix_fadvise", None)
    if fad is None:
        return {"supported": False}
    t, n, b, errs = time.time(), 0, 0, []
    for p in ckpt_files():
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                fad(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
                b += os.fstat(fd).st_size
            finally:
                os.close(fd)
        except Exception as e:
            errs.append([str(p)[-80:], repr(e)[:80]])
    return {"supported": True, "files": n, "gib": round(b / 2 ** 30, 2), "errors": errs[:5], "sec": round(time.time() - t, 1)}


def host_prep():
    """Before every boot: previous server + PLE workers gone, our leaked psm_* unlinked, checkpoint page cache dropped,
    Merlin's host rule max - (current - inactive_file) >= 69 + 4 GiB. Fail closed."""
    t = time.time()
    out = {"cg_before": cg_snapshot(False), "host_mem_before": host_mem()}
    out["procs"] = reap_leftovers()
    out["shm"] = unlink_own_psm() if out["procs"]["ok"] else {"skipped": "server processes still alive"}
    out["fadvise"] = drop_ckpt_cache()
    if not FAKE:
        time.sleep(3)
    out["cg_after"] = cg_snapshot(True)
    floor = NB0["host_floor_mib"]

    def avail():
        a = cg_snapshot(False).get("avail_mib")
        return (a, "cgroup") if a is not None else ((host_mem() or {}).get("MemAvailable_mib"), "MemAvailable")
    a, src = avail()
    if a is not None and a < floor:
        time.sleep(10)
        out["fadvise_2"] = drop_ckpt_cache()
        out["cg_after_2"] = cg_snapshot(False)
        a, src = avail()
    gate_ok = (a >= floor) if a is not None else False
    out["host_gate"] = {"avail_mib": a, "source": src, "floor_mib": floor, "ok": gate_ok,
                        "rule": "cgroup memory.max - (memory.current - inactive_file) >= 69 GiB (Merlin) + 4 GiB"}
    if FAKE and not gate_ok:            # local / CPU-kit harness: the fake server needs no host RAM; recorded, not applied
        out["host_gate"]["fake_not_applied"] = True
        gate_ok = True
    out["ok"] = bool(out["procs"]["ok"] and gate_ok)
    out["why"] = None if out["ok"] else ("server processes still alive" if not out["procs"]["ok"]
                                         else f"host memory {a} MiB < {floor} MiB ({src})")
    out["sec"] = round(time.time() - t, 1)
    return out


_boot_gate_rc2 = boot_gate


def boot_gate():
    """NB0 host preparation first, then RC2's gate (GPUs < 150 MiB each); `ok` is True only when both pass."""
    prep = host_prep()
    g = _boot_gate_rc2()
    g["host_prep"] = prep
    if prep.get("ok") is not True:
        g["ok"] = False
    return g


def _roles(text):
    return {int(p): n for n, p in re.findall(r"\((\w+) pid=(\d+)\)", text or "")}


def load_monitor(s, key, deadline):
    """Every 5 s from launch to ready: cgroup current / avail / stat / pressure and every process of the server session
    (majflt, minflt, rchar, read_bytes, rss, cpu) -> logs/load_monitor_<boot>.jsonl + a per-process summary."""
    path = LOGS / f"load_monitor_{key}.jsonl"
    while s.proc is None and time.time() < deadline:
        time.sleep(0.5)
    if s.proc is None:
        return
    sid = s.proc.pid
    SIDS.add(sid)
    tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    t0, first, last, n = time.time(), {}, {}, 0
    peak = {"cg_current_mib_max": None, "avail_mib_min": None, "pressure_full_avg10_max": None}
    while s.alive() and s.ready_sec is None and time.time() < deadline:
        cg = cg_snapshot(False)
        ps = session_procs({sid})
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps({"t": round(time.time() - t0, 1), "current": cg.get("memory.current"),
                                     "avail_mib": cg.get("avail_mib"), "stat_mib": cg.get("stat_mib"),
                                     "stat_count": cg.get("stat_count"), "pressure": cg.get("memory.pressure"),
                                     "procs": ps}) + "\n")
        except Exception:
            pass
        for p in ps:
            first.setdefault(p["pid"], p)
            last[p["pid"]] = p
        try:
            c = int(cg["memory.current"]) >> 20
            peak["cg_current_mib_max"] = max(peak["cg_current_mib_max"] or 0, c)
        except Exception:
            pass
        if cg.get("avail_mib") is not None:
            peak["avail_mib_min"] = min(x for x in (peak["avail_mib_min"], cg["avail_mib"]) if x is not None)
        m = re.search(r"full avg10=([\d.]+)", cg.get("memory.pressure") or "")
        if m:
            peak["pressure_full_avg10_max"] = max(peak["pressure_full_avg10_max"] or 0.0, float(m.group(1)))
        n += 1
        time.sleep(0.5 if FAKE else 5)
    roles = _roles(s.log())
    per = {}
    for pid, a in first.items():
        z = last[pid]
        per[str(pid)] = {"role": roles.get(pid, a["comm"]), "majflt": z["majflt"] - a["majflt"],
                         "minflt": z["minflt"] - a["minflt"],
                         "read_gib": round((z.get("read_bytes", 0) - a.get("read_bytes", 0)) / 2 ** 30, 2),
                         "rchar_gib": round((z.get("rchar", 0) - a.get("rchar", 0)) / 2 ** 30, 2),
                         "rss_gib_end": round(z["rss_pages"] * 4096 / 2 ** 30, 2),
                         "cpu_s": round((z["utime"] + z["stime"] - a["utime"] - a["stime"]) / tck, 1)}
    loads = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", s.log() or "")
    LOADMON[key] = {"samples": n, "ready_sec": s.ready_sec, "alive": s.alive(), "per_process": per, "peak": peak,
                    "model_loading": [[w, int(p), float(g), float(sec)] for w, p, g, sec in loads], "file": path.name}


def boot_watchdog(s, key, deadline):
    """Per-boot hard bound (<= 30 min from launch, and always before the global hard stop): kill the server."""
    while time.time() < deadline:
        if s.proc is not None and not s.alive():
            return
        time.sleep(2)
    if s.alive():
        WD[key] = {"fired_at_min": round((time.time() - T_START) / 60, 1), "bound_sec": NB0["boot_watchdog_sec"]}
        log("BOOT WATCHDOG:", key, "killing the server")
        try:
            s.kill()
        except Exception as e:
            WD[key]["kill_error"] = repr(e)[:200]


# ---------------------------------------------------------------- requests and raw arrays (new)
def _post(port, body, timeout=1800):
    t = time.perf_counter()
    try:
        r = json.loads(UR.urlopen(UR.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"}), timeout=timeout).read())
        return r, None, round(time.perf_counter() - t, 2)
    except UE.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            msg = ""
        return None, f"HTTP {e.code}: {msg}", round(time.perf_counter() - t, 2)
    except Exception as e:
        return None, repr(e)[:300], round(time.perf_counter() - t, 2)


def _req_log(rec):
    try:
        with _SAVE_LOCK, open(NB0_REQ, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def ids_sha(ids):
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


def _cached(r):
    return (((r or {}).get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def pl_arrays(pl, ids, k):
    """Every prompt position p >= 1 of an OpenAI prompt_logprobs list: the actual token's logprob and rank, the unique
    rank-1 id (E23's rule: an ambiguous rank 1 is -1), the best logprob, the top-k ids / logprobs by rank. NaN is kept
    (a non-finite value is never data); a missing position is valid=False."""
    n = len(ids)
    a = {"ids": np.asarray(ids, np.int32), "lp": np.full(n, np.nan, np.float32), "rank": np.full(n, -1, np.int32),
         "top1": np.full(n, -1, np.int32), "top1_lp": np.full(n, np.nan, np.float32),
         "topk_ids": np.full((n, k), -1, np.int32), "topk_lp": np.full((n, k), np.nan, np.float32), "valid": np.zeros(n, bool)}
    amb = 0
    for pos in range(1, n):
        d = pl[pos] if pos < len(pl) else None
        if not isinstance(d, dict):
            continue
        e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
        if not isinstance(e, dict):
            continue
        a["valid"][pos] = True
        v = e.get("logprob")
        a["lp"][pos] = float(v) if isinstance(v, (int, float)) else np.nan
        a["rank"][pos] = int(e.get("rank") or -1)
        ent = []
        for kk, vv in d.items():
            if isinstance(vv, dict) and isinstance(vv.get("logprob"), (int, float)):
                t = L._tok_id(kk)
                if isinstance(t, int):
                    ent.append((vv.get("rank") or 10 ** 9, -vv["logprob"], t, vv["logprob"]))
        ent.sort()
        r1 = [x for x in ent if x[0] == 1]
        if len(r1) == 1:
            a["top1"][pos] = r1[0][2]
        else:
            amb += 1
        if ent:
            a["top1_lp"][pos] = max(x[3] for x in ent)
        for j, x in enumerate([x for x in ent if x[0] <= k][:k]):
            a["topk_ids"][pos, j], a["topk_lp"][pos, j] = x[2], x[3]
    a["top1_ambiguous"] = amb
    return a


def e23_rows(a, last=256):
    """The same per-position rows E23's teacher_forced() keeps ([logprob, rank, rank-1 id]) for the last `last` positions,
    so RC2's compare_tf (with its non-finite guard) computes the RC2-comparable metric."""
    n, out = len(a["ids"]), []
    for pos in range(max(1, n - last), n):
        if not a["valid"][pos]:
            out.append(None)
            continue
        out.append([float(a["lp"][pos]), int(a["rank"][pos]), int(a["top1"][pos]) if a["top1"][pos] >= 0 else None])
    return {"positions": out, "n": len(out)}


def tf_request(key, port, ids, band, run, k=None):
    """One teacher-forced request (max_tokens 1, T 0, prompt_logprobs k, ids as ids). k > 1 refused by Merlin's prompt-
    scoring budget (prompt_logprobs.py validate) -> retried with k 1 = RC2's value, recorded. Raw arrays -> tf_raw/*.npz."""
    k = k or NB0["topk"]
    body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
            "return_tokens_as_token_ids": True}
    r, err, sec = _post(port, body)
    note = None
    if err and k > 1 and "prompt-scoring tensor bound" in err:
        note = f"prompt_logprobs {k} refused ({err[:160]}); retried with 1 (RC2's value)"
        k = body["prompt_logprobs"] = 1
        r, err, sec = _post(port, body)
    meta = {"boot": key, "band": band, "run": run, "k": k, "note": note, "sec": sec, "error": err, "prompt_len": len(ids),
            "prompt_sha256": ids_sha(ids), "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
    arr = None
    if r is not None:
        try:
            ch = r["choices"][0]
            pl = ch.get("prompt_logprobs") or []
            meta.update(prompt_logprobs_len=len(pl), gen_token_ids=ch.get("token_ids"), gen_text=ch.get("text"))
            arr = pl_arrays(pl, ids, k)
            vv = arr["valid"]
            meta.update(valid=int(vv.sum()), expected=len(ids) - 1, nonfinite=int((~np.isfinite(arr["lp"][vv])).sum()),
                        top1_ambiguous=arr["top1_ambiguous"])
            f = RAW_DIR / f"{key}_{band}_{run}.npz"
            np.savez_compressed(f, **{x: arr[x] for x in ("ids", "lp", "rank", "top1", "top1_lp", "topk_ids", "topk_lp", "valid")})
            meta["raw"] = f.name
        except Exception as e:
            meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
            arr = None
    _req_log({**meta, "kind": "teacher_forced", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
    return {"meta": meta, "arr": arr}


def decode(key, port, ids, n, band, run):
    """Greedy decode (T 0, ignore_eos, logprobs k): tokens, chosen-token logprobs, rank-1 id / logprob, rank-2 logprob."""
    body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": n, "temperature": 0, "logprobs": NB0["topk"],
            "return_tokens_as_token_ids": True, "return_token_ids": True, "ignore_eos": True}
    r, err, sec = _post(port, body, timeout=900)
    meta = {"boot": key, "band": band, "run": run, "sec": sec, "error": err, "prompt_len": len(ids), "prompt_sha256": ids_sha(ids),
            "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
    toks, lps, t1, t1lp, t2lp = [], [], [], [], []
    if r is not None:
        try:
            ch = r["choices"][0]
            lp = ch.get("logprobs") or {}
            toks = ch.get("token_ids") or [L._tok_id(t) for t in (lp.get("tokens") or [])]
            lps = [x if isinstance(x, (int, float)) else float("nan") for x in (lp.get("token_logprobs") or [])]
            for top in lp.get("top_logprobs") or []:
                it = sorted(((v, L._tok_id(kk)) for kk, v in (top or {}).items() if isinstance(v, (int, float))), reverse=True)
                t1.append(it[0][1] if it and isinstance(it[0][1], int) else -1)
                t1lp.append(it[0][0] if it else float("nan"))
                t2lp.append(it[1][0] if len(it) > 1 else float("nan"))
            meta.update(tokens=len(toks), finish=ch.get("finish_reason"), text=ch.get("text"))
            if any(not isinstance(x, int) or isinstance(x, bool) for x in toks):
                meta["error"] = "INVALID_TOKEN_IDS"
            elif len(toks) != n or len(lps) != n or len(t1) != n:
                meta["error"] = f"INCOMPLETE_DECODE tokens {len(toks)} logprobs {len(lps)} top {len(t1)} of {n}"
        except Exception as e:
            meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
    out = {"meta": meta, "tok": [int(x) for x in toks if isinstance(x, int)], "lp": np.asarray(lps, np.float64),
           "top1": np.asarray(t1, np.int64), "top1_lp": np.asarray(t1lp, np.float64), "top2_lp": np.asarray(t2lp, np.float64)}
    if not meta["error"]:
        f = RAW_DIR / f"{key}_dec{band}_{run}.npz"
        np.savez_compressed(f, ids=np.asarray(ids, np.int32), tok=np.asarray(out["tok"], np.int32), lp=out["lp"],
                            top1=out["top1"], top1_lp=out["top1_lp"], top2_lp=out["top2_lp"])
        meta["raw"] = f.name
    _req_log({**meta, "kind": "decode", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
    return out


# ---------------------------------------------------------------- comparisons (new; the last-256 metric is RC2's compare_tf)
def cmp_last256(ra, rb):
    if not ra or not rb or ra.get("arr") is None or rb.get("arr") is None:
        return {"status": "INCONCLUSIVE_MISSING_TF"}
    if ra["meta"]["prompt_sha256"] != rb["meta"]["prompt_sha256"]:
        return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
    return L.compare_tf(e23_rows(ra["arr"], NB0["last"]), e23_rows(rb["arr"], NB0["last"]))


def chunk_check(pos, d, edge=8):
    """|dlogprob| at the first / last `edge` positions of each 1024-token scheduler chunk (--max-num-batched-tokens) and
    each 256-row prompt-logprob chunk (VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE) vs mid-chunk."""
    out = {}
    for unit in (1024, 256):
        off = pos % unit
        e = (off < edge) | (off >= unit - edge)
        if e.any() and (~e).any():
            me, mm = float(d[e].mean()), float(d[~e].mean())
            out[str(unit)] = {"edge_mean": round(me, 4), "mid_mean": round(mm, 4), "edge_n": int(e.sum()),
                              "localised": bool(me > 3 * mm + 0.05)}
    return out


def cmp_full(a, b, bins=True):
    """Every prompt position: mean / p99 / max |dlogprob|, rank-1 agreement, 4096-token bins (does it grow with context?),
    the first 256-position window over the screening mean, chunk-edge localisation."""
    if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
        return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
    va, vb = a["valid"] & np.isfinite(a["lp"]), b["valid"] & np.isfinite(b["lp"])
    if (a["valid"] & ~np.isfinite(a["lp"])).any() or (b["valid"] & ~np.isfinite(b["lp"])).any():
        return {"status": "INCONCLUSIVE_NONFINITE"}
    pos = np.nonzero(va & vb)[0]
    n = len(a["ids"])
    if not len(pos):
        return {"status": "INCONCLUSIVE_NO_VALID"}
    d = np.abs(a["lp"][pos].astype(np.float64) - b["lp"][pos].astype(np.float64))
    t1 = (a["top1"][pos] == b["top1"][pos]) & (a["top1"][pos] >= 0)
    ds = np.sort(d)
    out = {"status": "MEASURED" if len(pos) == n - 1 else "PARTIAL", "n": int(len(pos)), "expected": n - 1,
           "mean_abs_dlogprob": round(float(d.mean()), 4), "p99_abs_dlogprob": round(float(ds[min(len(ds) - 1, int(round(0.99 * (len(ds) - 1))))]), 3),
           "max_abs_dlogprob": round(float(d.max()), 3), "rank1_agreement": round(float(t1.mean()), 4)}
    if bins:
        bl = []
        for lo in range(0, n, 4096):
            m = (pos >= lo) & (pos < lo + 4096)
            if m.any():
                bl.append([lo, int(m.sum()), round(float(d[m].mean()), 4), round(float(t1[m].mean()), 4)])
        out["bins_4096"] = bl                        # [start, n, mean |dlogprob|, rank-1 agreement]
        if len(d) >= 256:
            cs = np.convolve(d, np.ones(256) / 256, "valid")
            idx = np.nonzero(cs > SCREEN["mean_max"])[0]
            out["first_window_over_screen_mean"] = int(pos[idx[0]]) if len(idx) else None
        out["chunk"] = chunk_check(pos, d)
    return out


def shift_check(a, b, window=2048, shifts=(-2, -1, 0, 1, 2)):
    """Alignment test on the last `window` positions: reference rank-1 id at p vs candidate rank-1 id at p + s. A shifted
    agreement >= 0.9 that beats s = 0 by > 0.2 = a position-alignment artefact."""
    n = min(len(a["top1"]), len(b["top1"]))
    i0 = np.arange(max(1, n - window), n)
    res = {}
    for s in shifts:
        j = i0 + s
        m = (j >= 1) & (j < n)
        i, j = i0[m], j[m]
        v = a["valid"][i] & b["valid"][j] & (a["top1"][i] >= 0) & (b["top1"][j] >= 0)
        if not v.any():
            res[str(s)] = None
            continue
        dl = np.abs(a["top1_lp"][i][v].astype(np.float64) - b["top1_lp"][j][v].astype(np.float64))
        res[str(s)] = {"n": int(v.sum()), "top1_agree": round(float((a["top1"][i][v] == b["top1"][j][v]).mean()), 4),
                       "top1_lp_mean_abs_d": round(float(np.nanmean(dl)), 4) if np.isfinite(dl).any() else None}
    ok = {k: x for k, x in res.items() if x}
    best = max(ok, key=lambda k: (ok[k]["top1_agree"], k == "0")) if ok else None
    a0 = (ok.get("0") or {}).get("top1_agree")
    art = bool(best and best != "0" and ok[best]["top1_agree"] >= 0.9 and (a0 is None or ok[best]["top1_agree"] > a0 + 0.2))
    return {"shifts": res, "best_shift": int(best) if best is not None else None, "alignment_artefact": art}


def screen(c):
    if not isinstance(c, dict) or c.get("status") != "MEASURED":
        return "INCONCLUSIVE"
    m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
    if not (_finite(m) and _finite(r1)):
        return "INCONCLUSIVE_NONFINITE"
    return "DIVERGENT" if (r1 < SCREEN["rank1_min"] or m > SCREEN["mean_max"]) else "CLEAN"


def screening_gate(ref_aa, cand):
    """External audit 2 (verified): conservative screening; 0.90 / 0.15 are not published acceptance limits."""
    if (ref_aa or {}).get("status") != "MEASURED":
        return "INCONCLUSIVE_REFERENCE"
    if (cand or {}).get("status") != "MEASURED":
        return "INCONCLUSIVE_FP8"
    s = screen(cand)
    if s == "INCONCLUSIVE_NONFINITE":
        return s
    return "HOLD_FP8_NUMERICAL_DIVERGENCE" if s == "DIVERGENT" else "SCREENING_PASS"


def triage_gate(cand, short_ok=True):
    """External audit 3 (verified): severe-discrepancy triage. It can never give GO: TF similarity alone cannot certify
    ARC correctness, so the best outcome is PENDING_LONG_ARC_ACCURACY."""
    if short_ok is False:
        return "NO_GO_SHORT_ORACLE"
    if short_ok is not True:
        return "INCONCLUSIVE_SHORT_ORACLE"
    if (cand or {}).get("status") != "MEASURED":
        return "INCONCLUSIVE_TF"
    m, r1 = cand.get("mean_abs_dlogprob"), cand.get("rank1_agreement")
    if not (_finite(m) and _finite(r1)):
        return "INCONCLUSIVE_NONFINITE"
    if m > TRIAGE["mean_max"] or r1 < TRIAGE["rank1_min"]:
        return "NO_GO_FP8_TF_DIVERGENCE"
    return "PENDING_LONG_ARC_ACCURACY"


def _gate_rank(g):
    g = str(g)
    return 3 if g.startswith(("HOLD", "NO_GO")) else 2 if g.endswith("NONFINITE") else 1 if g.startswith("INCONCLUSIVE") else 0


def worst(gs):
    gs = list(gs)
    return max(gs, key=_gate_rank) if gs else "INCONCLUSIVE_NO_DATA"


def decode_compare(ref, cand):
    """Greedy rows token for token: first divergence, the reference's top-1 - top-2 margin there (E23's 0.5 rule:
    FAIL_HIGH_MARGIN at or above, NEAR_TIE below), mean |dlogprob| over the common prefix."""
    if not ref or not cand or ref["meta"].get("error") or cand["meta"].get("error") or not ref["tok"] or not cand["tok"]:
        return {"status": "INCONCLUSIVE_MISSING"}
    x, y = ref["tok"], cand["tok"]
    d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
    c = len(x) if d is None else d
    dl = np.abs(ref["lp"][:c] - cand["lp"][:c]) if c else np.zeros(0)
    if not np.isfinite(dl).all() or (d is not None and not math.isfinite(float(ref["top1_lp"][d] - ref["top2_lp"][d]))):
        return {"status": "INCONCLUSIVE_NONFINITE", "first_diff": d}
    out = {"n": len(x), "first_diff": d, "common_prefix_mean_abs_dlogprob": round(float(dl.mean()), 4) if c else None,
           "common_prefix_max_abs_dlogprob": round(float(dl.max()), 4) if c else None}
    if d is None:
        out["status"] = "EXACT" if len(x) == len(y) else "INCONCLUSIVE_LENGTH"
    else:
        m = float(ref["top1_lp"][d] - ref["top2_lp"][d])
        out.update(ref_margin_at_diff=round(m, 4), status="FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
    return out


def sc_compare(dec, tf, P):
    """Decode-vs-prompt-logprob self-consistency on ONE boot: the greedy decode's chosen-token logprob at step i vs the
    teacher-forced logprob of the same token at position P + i (prompt + those tokens), rank-1 ids, +-2 shift."""
    if not dec or dec["meta"].get("error") or not tf or tf.get("arr") is None:
        return {"status": "INCONCLUSIVE_MISSING"}
    a, n = tf["arr"], len(dec["tok"])
    if n == 0:                                             # A-4: nothing decoded, nothing to compare (no IndexError)
        return {"status": "INCONCLUSIVE_EMPTY_DECODE"}
    pos = np.arange(P, P + n)
    if pos[-1] >= len(a["ids"]):                           # the TF prompt does not cover every decoded position
        return {"status": "INCONCLUSIVE_TF_TOO_SHORT", "need_len": int(P + n), "tf_len": int(len(a["ids"]))}
    if not np.array_equal(a["ids"][pos], np.asarray(dec["tok"], np.int32)):
        return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
    dl, tl, v = dec["lp"], a["lp"][pos].astype(np.float64), a["valid"][pos]
    if not np.isfinite(dl).all() or not np.isfinite(tl[v]).all():
        return {"status": "INCONCLUSIVE_NONFINITE"}
    if not v.all():
        return {"status": "INCONCLUSIVE_INCOMPLETE_TF", "valid": int(v.sum()), "expected": n}
    d = np.sort(np.abs(dl - tl))
    r1 = float((dec["top1"] == a["top1"][pos]).mean())
    sh = {}
    for s in (-2, -1, 0, 1, 2):
        j = pos + s
        m = (j >= 1) & (j < len(a["ids"]))
        if m.any():
            sh[str(s)] = round(float((dec["top1"][m] == a["top1"][j[m]]).mean()), 4)
    out = {"status": "MEASURED", "n": n, "mean_abs_dlogprob": round(float(d.mean()), 4),
           "p99_abs_dlogprob": round(float(d[min(n - 1, int(round(0.99 * (n - 1))))]), 3), "max_abs_dlogprob": round(float(d[-1]), 3),
           "rank1_agreement": round(r1, 4), "greedy_self_top1": round(float((np.asarray(dec["tok"]) == dec["top1"]).mean()), 4),
           "shift_top1_agree": sh}
    out["screen"] = screen(out)
    return out


def prefix_consistency(key):
    """One boot, two bands: the shared prompt prefix (RC2's prompt() starts every band of a variant with the same tokens)
    must score the same (cross-request determinism within a boot)."""
    runs = {b: ((TFR.get(key) or {}).get(b) or {}).get("fresh") for b in NB0["tf_bands"]}
    runs = {b: r for b, r in runs.items() if r and r.get("arr") is not None}
    out, bl = {}, list(runs)
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            a, c = runs[bl[i]]["arr"], runs[bl[j]]["arr"]
            n = min(len(a["ids"]), len(c["ids"]))
            eq = a["ids"][:n] == c["ids"][:n]
            common = n if eq.all() else int(np.argmin(eq))
            p = np.arange(1, common)
            p = p[a["valid"][p] & c["valid"][p] & np.isfinite(a["lp"][p]) & np.isfinite(c["lp"][p])]
            if len(p):
                d = np.abs(a["lp"][p].astype(np.float64) - c["lp"][p].astype(np.float64))
                out[f"{bl[i]}~{bl[j]}"] = {"shared": common, "n": int(len(p)), "mean_abs_dlogprob": round(float(d.mean()), 5),
                                           "max_abs_dlogprob": round(float(d.max()), 4),
                                           "top1_agree": round(float((a["top1"][p] == c["top1"][p]).mean()), 4)}
    vals = [x for x in out.values() if isinstance(x, dict)]
    out["consistent"] = all(x["mean_abs_dlogprob"] <= 0.01 and x["top1_agree"] >= 0.99 for x in vals) if vals else None
    return out


# ---------------------------------------------------------------- per-boot suites (new)
TF_EST = {"2k": 15, "8k": 25, "40k": 60, "90k": 150}      # seconds per teacher-forced request, upper estimates
SC_EST = {"2k": 40, "8k": 60, "40k": 120}
# SC variants: RC2's prompt() starts every band of a variant with the same tokens, so 8k v6 would hand 40k v6 a 4096-token
# prefix hit; 2k v6 / 8k v7 / 40k v6 share < 4096 tokens with each other and with every other prompt of a boot (cold = cold).


def tf_suite(key, s, bands):
    out = {}
    for band in bands:
        ids = PROMPTS[band][NB0["tf_variant"]]
        rec = TFR.setdefault(key, {}).setdefault(band, {})
        o = {"prompt_len": len(ids), "prompt_sha256": ids_sha(ids), "variant": NB0["tf_variant"]}
        if band == "90k" and f"{key}_90k" in DROP:
            o["status"] = "SKIPPED_BUDGET_PRIORITY"
            o["why"] = "lead rule: under budget pressure the MTP-off BF16 boot's 90k band is dropped first"
            out[band] = o
            continue
        for run in ("fresh", "repeat"):
            if not s.alive():
                o[run] = {"status": "SKIPPED_SERVER_DEAD"}
            elif key in WD or ABORT["why"] or left() < TF_EST[band] + 240:
                o[run] = {"status": "SKIPPED_BUDGET"}
            else:
                rec[run] = tf_request(key, s.port, ids, band, run)
                o[run] = rec[run]["meta"]
        o["aa"] = cmp_last256(rec.get("fresh"), rec.get("repeat"))
        out[band] = o
        save()
    return out


def sc_suite(key, s):
    out = {}
    for band in NB0["sc_bands"]:
        ids = PROMPTS[band][NB0["sc_variant"][band]]
        if not s.alive():
            out[band] = {"status": "SKIPPED_SERVER_DEAD"}
            continue
        if key in WD or ABORT["why"] or left() < SC_EST[band] + 240:
            out[band] = {"status": "SKIPPED_BUDGET"}
            continue
        cold = decode(key, s.port, ids, NB0["sc_tok"], band, "cold")
        cached = decode(key, s.port, ids, NB0["sc_tok"], band, "cached")
        tf = tf_request(key, s.port, list(ids) + list(cold["tok"]), "sc" + band, "tf") if not cold["meta"].get("error") else None
        SCR.setdefault(key, {})[band] = {"cold": cold, "cached": cached, "tf": tf, "P": len(ids)}
        out[band] = {"cold": cold["meta"], "cached": cached["meta"], "tf": (tf or {}).get("meta"),
                     "decode_vs_tf": sc_compare(cold, tf, len(ids)), "cold_vs_cached": decode_compare(cold, cached),
                     "cold_was_cold": cold["meta"].get("cached_tokens") in (0, None)}
        save()
    return out


def scales_facts(t):
    """The static scales sidecar's values, the per-stage 'applied ... to N of 12' lines, every 'E4M3 KV active on <layer>'
    line matched to the file (logged %.6g), and the MTP layer's DEFAULT 1.0 line."""
    path = CKPT / "qsa_kv_scales_262k.json"
    try:
        raw = path.read_bytes()
        file, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    except Exception as e:
        file, sha = {}, f"unreadable: {e!r}"[:120]
    applied = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*applied static K/V scales from (.+?) to (\d+) of (\d+) layer\(s\) "
                         r"\(pipeline layers (\d+)-(\d+)\)", t)
    act = {}
    for name, k, v, dflt in re.findall(r"QSA E4M3 KV active on (\S+): k_scale=([-\d.eE+]+) v_scale=([-\d.eE+]+)(\s+\(DEFAULT 1\.0)?", t):
        act.setdefault(name, set()).add((float(k), float(v), bool(dflt)))
    tol = lambda a, b: abs(a - b) <= 1e-5 * max(1e-12, abs(b))
    matched, mismatched = [], []
    for name, sc in sorted(file.items()):
        got = act.get(name)
        if got:
            (matched if all(tol(k, sc["k_scale"]) and tol(v, sc["v_scale"]) and not d for k, v, d in got) else mismatched).append(name)
    stages = sorted({f"{a}-{b}" for *_, a, b in applied})
    return {"file": str(path), "sha256": sha, "values": file, "applied_lines": len(applied), "applied_stages": stages,
            "applied_counts": sorted({f"{n} of {m}" for _, _, _, n, m, _, _ in applied}),
            "applied_paths": sorted({p for _, _, p, *_ in applied}), "active_layers": sorted(act), "matched": len(matched),
            "mismatched": mismatched, "missing": sorted(set(file) - set(act)), "not_in_file": sorted(set(act) - set(file)),
            "mtp_default_scale": sorted(n for n, g in act.items() if any(d for _, _, d in g)),
            "ok": bool(file) and not mismatched and not (set(file) - set(act)) and stages == ["0-25", "25-48"]}


def boot_facts(s, key):
    """What the server itself logged for this boot (non-default args, engine config line, MTP-off warning, RecoverSSM
    path, byte-decoder marker, scales): config_ok only when every expected property holds (else CONFIG_MISMATCH)."""
    b, t = BOOTS[key], s.log() or ""
    lines = t.splitlines()
    nd = next((l for l in lines if "non-default args:" in l), "")
    eng = next((l for l in lines if "Initializing a V1 LLM engine" in l), "")
    g = lambda rx, txt: (re.search(rx, txt) or [None, None])[1]
    want_kv = "fp8_e4m3" if b["kv"] == "fp8" else "bfloat16"
    f = {"kv_cache_dtype": g(r"'kv_cache_dtype': '(\w+)'", nd), "max_num_seqs": g(r"'max_num_seqs': (\d+)", nd),
         "spec_in_args": "'speculative_config':" in nd, "spec_k_args": g(r"'num_speculative_tokens': (\d+)", nd),
         "captures": g(r"'cudagraph_capture_sizes': \[([\d, ]*)\]", nd), "prefix_caching": "'enable_prefix_caching': True" in nd,
         "engine_spec": g(r"speculative_config=(None|SpeculativeConfig\([^)]*\))", eng), "engine_kv": g(r"kv_cache_dtype=(\w+)", eng),
         "mtp_off_warning": bool(re.search(r"needs speculative decoding; num_speculative_tokens=0, so the stock GDN path is used", t)),
         "recoverssm_path_lines": len(re.findall(r"GDN RecoverSSM path taken", t)),
         "spec_metric_lines": len(re.findall(r"SpecDecoding metrics", t)),
         "byte_marker_pids": sorted(set(re.findall(r"NB0 boot E: QSA E4M3 byte decoder forced[^\n]*?pid=(\d+)", t))),
         "e4m3_active_lines": len(re.findall(r"QSA E4M3 KV active on", t))}
    f["scales"] = scales_facts(t) if b["kv"] == "fp8" else None
    es = f["engine_spec"] or ""
    ck = {"kv_args": f["kv_cache_dtype"] == want_kv, "kv_engine": f["engine_kv"] == want_kv,
          "seqs": f["max_num_seqs"] == str(b["want_seqs"]), "prefix_caching": f["prefix_caching"],
          "mtp": (f["spec_in_args"] and f["spec_k_args"] == "3" and es.startswith("SpeculativeConfig") and "num_spec_tokens=3" in es)
          if b["mtp"] else (not f["spec_in_args"] and es == "None" and f["mtp_off_warning"]),
          "byte_decoder": (len(f["byte_marker_pids"]) >= 4) if b["byte"] else not f["byte_marker_pids"],
          "bf16_no_e4m3": True if b["kv"] == "fp8" else f["e4m3_active_lines"] == 0}
    if b["captures"]:
        ck["captures"] = f["captures"] == "1, 2, 3, 4, 6, 8, 12, 16, 24, 32"
    f["checks"] = ck
    f["config_ok"] = all(ck.values())
    return f


def nb0_body(key):
    def body(ss):
        s = ss[0]
        p = s.port
        o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
        o["boot_facts"] = boot_facts(s, key)
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
            return o
        if o["boot_facts"]["config_ok"] is not True:
            bad = [k for k, v in o["boot_facts"]["checks"].items() if not v]
            o.update(early_stop="CONFIG_MISMATCH", status="CONFIG_MISMATCH", config_mismatch=bad, census_end=kv_census(s))
            return o
        o["oracle"] = oracle(p, key=key, refs=ORACLE_REFS[key])
        o["tf"] = tf_suite(key, s, ["2k", "8k", "40k"])
        o["sc"] = sc_suite(key, s)
        o["tf"].update(tf_suite(key, s, ["90k"]))                # 90k last: the largest prompt-logits request
        o["census_end"] = kv_census(s)
        o["boot_facts_end"] = boot_facts(s, key)
        o["tf_died"] = not s.alive()
        return o
    return body


# ---------------------------------------------------------------- boot E: byte decoder (new; audited, fail closed)
DECODER_TEST = r"""
import json, torch, triton
import triton.language as tl
__DECODER_SRC__

@triton.jit
def _nb0_decode_all(x_ptr, y_ptr):
    o = tl.arange(0, 256)
    tl.store(y_ptr + o, _decode_e4m3_to_bf16(tl.load(x_ptr + o)))


x = torch.arange(256, dtype=torch.uint8, device="cuda")
y = torch.empty(256, dtype=torch.bfloat16, device="cuda")
_nb0_decode_all[(1,)](x, y)
torch.cuda.synchronize()
ref = x.view(torch.float8_e4m3fn).to(torch.bfloat16)
yn, rn = torch.isnan(y), torch.isnan(ref)
same = (y.view(torch.int16) == ref.view(torch.int16)) | (yn & rn)
bad = [int(i) for i in torch.nonzero(~same).flatten().tolist()]
sub = [i for i in range(256) if (i & 0x78) == 0 and (i & 0x7) != 0]
print("NB0_DECODER_RESULT " + json.dumps({"ok": not bad, "n_bad": len(bad), "bad_codepoints": bad[:32],
      "subnormals_ok": all(i not in bad for i in sub), "n_subnormals": len(sub), "nan_ok": bool((yn == rn).all()),
      "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "triton": triton.__version__}))
"""
BYTE = {}


def decoder_script(src):
    """The device-check script (dedented: the cell wrapper indents this string literal)."""
    return textwrap.dedent(DECODER_TEST).replace("__DECODER_SRC__", src)


def byte_decoder_device_check(ops_txt):
    """The installed `_decode_e4m3_to_bf16` (Triton) vs torch's native E4M3 -> BF16 cast on all 256 codepoints, on
    GPU 0 of the L4s, bit for bit (NaN as NaN). Merlin verified the byte decoder's FP16-subnormal trick on sm_86 only."""
    m = re.search(r"@triton\.jit\ndef _decode_e4m3_to_bf16\(b\):\n(?:(?:    [^\n]*)?\n)+", ops_txt)
    if not m:
        return {"ok": False, "why": "_decode_e4m3_to_bf16 not found in the installed ops/qsa.py"}
    d = TMP / "nb0_decoder_test"
    d.mkdir(exist_ok=True)
    f = d / "nb0_decoder_test.py"
    f.write_text(decoder_script(m.group(0)), encoding="utf-8")
    eng = MERLIN_ENG or {}
    venv = (eng.get("kv") or {}).get("MERLIN_VENV")
    if not venv:
        return {"ok": False, "why": "Merlin venv unknown"}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    env.update(CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=str(d / "triton"), CUDA_HOME=eng.get("cuda_home", ""),
               PATH=f"{venv}/bin:{eng.get('cuda_home', '')}/bin:{os.environ.get('PATH', '')}")
    try:
        r = subprocess.run([f"{venv}/bin/python", str(f)], capture_output=True, text=True, timeout=600, env=env, cwd=str(d))
    except Exception as e:
        return {"ok": False, "why": f"device check did not run: {e!r}"[:300]}
    line = next((l for l in r.stdout.splitlines() if l.startswith("NB0_DECODER_RESULT ")), None)
    if r.returncode or not line:
        return {"ok": False, "why": f"device check rc {r.returncode}", "tail": (r.stdout + r.stderr)[-1500:]}
    res = json.loads(line.split(" ", 1)[1])
    res["source_sha256"] = hashlib.sha256(m.group(0).encode()).hexdigest()
    return res


def byte_decoder_prepare():
    """Boot E (lead addendum): qsa_e4m3_uses_native_cast() -> False on sm_89, i.e. Merlin's sm_86-qualified path (Triton
    byte decoder `_decode_e4m3_to_bf16` for K and V, unfused QSA prepare + reshape_and_cache_flash), and the qsa.py:231
    hardware gate still admitting E4M3 then. The source of the installed tree is patched (qsa.py binds the function by
    name at import, ops/qsa.py calls its module global: one body patch covers ops 639 / 840 and qsa 231 / 710); the patch
    is env-gated (NB0_QSA_FORCE_BYTE_DECODER=1, set in boot E's server env only); restored after the boot. Fail closed."""
    out = {"ok": False}
    if FAKE:
        out.update(ok=True, fake=True)
        return out
    if BYTE.get("applied"):                # a previous patch is still pending restore: never stack a second one
        out["why"] = f"a previous boot-E patch is still pending restore: {BYTE['applied']}"
        return out
    eng = MERLIN_ENG or {}
    tree = Path(eng["kv"]["MERLIN_TREE"]) if eng.get("kv") else None
    if tree is None:
        out["why"] = "Merlin tree unknown (install failed?)"
        return out
    files, audit = {}, {}
    for k in ("ops", "qsa"):
        p = tree / QSA_PATCH[k + "_rel"]
        try:
            raw = p.read_bytes()
            txt = raw.decode("utf-8")
        except Exception as e:
            out["why"] = f"{QSA_PATCH[k + '_rel']}: {e!r}"[:300]
            return out
        sha = hashlib.sha256(raw).hexdigest()
        anchors = [a for a in QSA_PATCH["audit"][k] + [QSA_PATCH[k + "_old"]] if txt.count(a) != 1]
        names = txt.count("qsa_e4m3_uses_native_cast")
        audit[k] = {"sha256": sha, "sha_matches_audited": sha == QSA_PATCH[k + "_sha"], "anchors_failed": len(anchors),
                    "name_count": names, "marker_already": QSA_PATCH["marker"] in txt}
        if anchors or names != QSA_PATCH["audit"]["name_count"][k] or QSA_PATCH["marker"] in txt:
            out.update(audit=audit, why=f"{QSA_PATCH[k + '_rel']}: the audited call sites differ in the installed tree")
            return out
        files[k] = (p, raw, txt)
    out["audit"] = audit
    unreviewed = [QSA_PATCH[k + "_rel"] for k in files if not audit[k]["sha_matches_audited"]]
    if unreviewed:                         # A-1: exact audited bytes only (anchors alone are not a review)
        out.update(status="UNREVIEWED_SOURCE_SHA", why=f"installed source sha256 != the audited Merlin ad42e37 sha: {unreviewed}")
        return out
    try:
        prod = (tree / ".build-products.sha256").read_text()
    except Exception:
        prod = ""
    listed = [QSA_PATCH[k + "_rel"] for k in files if QSA_PATCH[k + "_rel"] in prod]
    if listed:
        out["why"] = f"patched files are verified build products (host_phase would refuse the tree): {listed}"
        return out
    out["device_check"] = byte_decoder_device_check(files["ops"][2])
    if out["device_check"].get("ok") is not True:
        out["why"] = "on-device decoder check failed on the L4: " + json.dumps(out["device_check"])[:300]
        return out
    venv = (eng.get("kv") or {}).get("MERLIN_VENV")
    try:                                   # any failure from here on restores the tree before returning (fail closed)
        for k, (p, raw, txt) in files.items():
            bak = p.with_name(p.name + ".nb0_orig")
            # A-3: backup (read back = the audited sha) -> register -> write. A file is touched only after it is
            # registered, so a write that fails half-way is restored from its verified backup too.
            bak.write_bytes(raw)
            if hashlib.sha256(bak.read_bytes()).hexdigest() != QSA_PATCH[k + "_sha"]:
                raise OSError(f"backup {bak.name} does not read back as the audited source")
            BYTE.setdefault("applied", []).append(str(p))
            BYTE.setdefault("want", {})[str(p)] = QSA_PATCH[k + "_sha"]
            p.write_bytes(txt.replace(QSA_PATCH[k + "_old"], QSA_PATCH[k + "_new"]).encode("utf-8"))
            cp = subprocess.run([f"{venv}/bin/python", "-m", "py_compile", str(p)], capture_output=True, text=True, timeout=120)
            if cp.returncode or p.read_text(encoding="utf-8").count(QSA_PATCH["marker"]) != 1:
                out["why"] = f"patched {QSA_PATCH[k + '_rel']} does not compile / marker missing: {cp.stderr[-300:]}"
                out["restore"] = byte_decoder_restore()
                return out
    except Exception as e:
        out["why"] = f"patch step failed: {type(e).__name__}: {e}"[:300]
        out["restore"] = byte_decoder_restore()
        return out
    out.update(ok=True, patched=[QSA_PATCH["ops_rel"], QSA_PATCH["qsa_rel"]])
    return out


def byte_decoder_restore():
    """A-2: every pending file is restored from its backup, read back and compared with the audited original sha; it
    leaves the pending list only after that check, so a failed restore stays pending (a second call fails again)."""
    out, failed, want = {"files": {}}, [], BYTE.get("want") or {}
    for x in list(BYTE.get("applied") or []):
        p = Path(x)
        bak = p.with_name(p.name + ".nb0_orig")
        try:
            if not want.get(x):
                raise ValueError("no audited original sha recorded for this file")
            raw = bak.read_bytes()
            if hashlib.sha256(raw).hexdigest() != want[x]:
                raise ValueError("backup sha256 != the audited original")
            p.write_bytes(raw)
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want[x]:
                raise ValueError(f"restored file reads back as {got[:16]}, not the audited original")
            rec = {"restored": True, "sha256": got}
            try:
                bak.unlink()
            except Exception as e:
                rec["backup_left"] = repr(e)[:200]
            out["files"][x] = rec
        except Exception as e:
            failed.append(x)
            out["files"][x] = {"restored": False, "error": repr(e)[:200]}
    BYTE["applied"] = failed
    BYTE["want"] = {x: want[x] for x in failed if x in want}
    out.update(ok=not failed, pending=failed)
    return out


# ---------------------------------------------------------------- one boot (RC2's merlin_boot) + its NB0 records
def boot_record(key):
    return (R["sections"].get("NB0") or {}).get(BOOTS[key]["step"]) or {}


def qsa_restore_gate(key, rs):
    """A-3: a restore that did not verify (or any file still pending) leaves the Merlin tree patched: ABORT, so no later
    Merlin boot (B, A) runs on patched source."""
    if FAKE or ((rs or {}).get("ok") is not False and not BYTE.get("applied")):
        return
    ABORT["why"] = ABORT["why"] or f"QSA_PATCH_RESTORE_FAILED: boot {key}: still pending {BYTE.get('applied')}"
    save()


def run_boot(key):
    b = BOOTS[key]
    pre = None
    if b["byte"]:
        pre = byte_decoder_prepare()
        if pre.get("ok") is not True:
            qsa_restore_gate(key, pre.get("restore"))
            return {"status": pre.get("status") or "SKIPPED_BYTE_DECODER_UNSAFE", "byte_decoder": pre}
    try:
        o = merlin_boot(b["name"], b["kv"], b["port"], NB0["ready_sec"], nb0_body(key), need=(), expected_rows=4)
    finally:
        if b["byte"]:
            pre["restore"] = byte_decoder_restore() if not FAKE else {"fake": True}
            qsa_restore_gate(key, pre["restore"])
    gate = o.get("boot_gate") or {}
    if (gate.get("host_prep") or {}).get("ok") is False:
        o["status"] = "HOST_GATE_FAILED"
        ABORT["why"] = ABORT["why"] or f"host preparation failed before boot {key}: {gate['host_prep'].get('why')}"
    if key in WD:
        o["status"] = "BOOT_WATCHDOG"
    o.update(boot=key, title=b["title"], plan=(R["setup"].get("nb0_plans") or {}).get(b["name"]), load_monitor=LOADMON.get(key),
             watchdog=WD.get(key), byte_decoder=pre)
    return o


def boot_short_ok(key):
    r = boot_record(key)
    if not r:
        return None
    return r.get("correctness") == "CLEAN" and r.get("status") in ("DONE", "TF_OOM", "PARTIAL_HARD_STOP")


def prompt_check():
    """NB0's prompt ids = RC2's: every prompt RC2 sent on Kaggle (47 requests, sha256[:16] of the id list) must come out
    of RC2's prompt() unchanged here; plus the full sha256 of every NB0 teacher-forced / decode prompt."""
    have = {}
    for band, ps in PROMPTS.items():
        for v, ids in enumerate(ps):
            have[hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]] = [band, v, len(ids)]
    have[hashlib.sha256(json.dumps(TPL_IDS).encode()).hexdigest()[:16]] = ["TPL_IDS", None, len(TPL_IDS)]
    rec = RC2_REF["prompt_shas"]
    miss = [s for s, n in rec.items() if s not in have or have[s][2] != n]
    return {"ok": not miss and bool(MERLIN_PROMPTS), "rc2_recorded": len(rec), "reproduced": len(rec) - len(miss),
            "missing": miss[:10], "matched_as": {s: have[s][:2] for s in rec if s in have},
            "tf_prompts": {b: {"variant": NB0["tf_variant"], "len": len(PROMPTS[b][NB0["tf_variant"]]),
                               "sha256": ids_sha(PROMPTS[b][NB0["tf_variant"]])} for b in NB0["tf_bands"]},
            "sc_prompts": {b: {"variant": NB0["sc_variant"][b], "len": len(PROMPTS[b][NB0["sc_variant"][b]]),
                               "sha256": ids_sha(PROMPTS[b][NB0["sc_variant"][b]])} for b in NB0["sc_bands"]},
            "rc2_tf_prompt_is_40k_v5": PROMPTS["40k"][5] == TF_P["40k"], "source": R["prompt_render"].get("source")}


# ---------------------------------------------------------------- analysis, gates, decision table (new)
DECISION_TABLE = [
    ["KV pair (A-B, C-D or C-E) divergent already at 2k", "quantisation / scales / kernel"],
    ["KV pair clean at 2k (and 8k), divergent only at 40k / 90k", "long-context accumulation / recurrent or cache state"],
    ["Only a repeat differs (per-boot fresh vs repeat A/A, or decode cold vs cached)", "cache / state"],
    ["A-B clean but C-D divergent; or A-C / B-D divergent (target logits change with MTP)", "speculative decoding / PP / RecoverSSM"],
    ["Decode logprobs disagree with prompt logprobs on one boot", "prompt-logprob (prefill-output) path artefact or a prefill defect on that boot"],
    ["A +-1 / +-2 position shift restores the rank-1 agreement", "position-alignment artefact"],
    ["Divergence concentrated at 1024-token scheduler or 256-row logits chunk edges", "chunking artefact"],
    ["C-D divergent while C-E is clean", "the sm_89 native E4M3 read path is the defect; keep FP8 capacity via Merlin's byte decoder"],
    ["C-D and C-E both divergent", "not the native cast: FP8 KV itself (scales / quantisation) or the shared prefill path"],
    ["MTP-on vs MTP-off greedy rows diverge at a high margin (cf. upstream #63593)", "GDN state on rejected drafts (RecoverSSM)"],
]


def analyze():
    an = {"per_boot": {}, "pairs": {}, "rc2_anchor": RC2_REF["tf40k"], "decision_table": DECISION_TABLE,
          "thresholds": {"screening": SCREEN, "triage": TRIAGE, "decode_fail_margin": L.FAIL_MARGIN}}
    for key in ORDER:
        if key not in TFR and key not in SCR:
            continue
        pb = {"aa": {}, "aa_full": {}, "cached_tokens": {}, "prefix_consistency": prefix_consistency(key), "sc": {}}
        for band, runs in (TFR.get(key) or {}).items():
            f, r = runs.get("fresh"), runs.get("repeat")
            pb["aa"][band] = cmp_last256(f, r)
            pb["aa_full"][band] = (cmp_full(f["arr"], r["arr"], bins=False) if f and r and f.get("arr") is not None
                                   and r.get("arr") is not None else {"status": "INCONCLUSIVE_MISSING_TF"})
            pb["cached_tokens"][band] = [(runs.get(x) or {}).get("meta", {}).get("cached_tokens") for x in ("fresh", "repeat")]
        for band, sc in (SCR.get(key) or {}).items():
            pb["sc"][band] = {"decode_vs_tf": sc_compare(sc["cold"], sc["tf"], sc["P"]),
                              "cold_vs_cached": decode_compare(sc["cold"], sc["cached"]),
                              "cached_tokens": [sc["cold"]["meta"].get("cached_tokens"), sc["cached"]["meta"].get("cached_tokens")]}
        pb["scales"] = ((boot_record(key).get("boot_facts_end") or boot_record(key).get("boot_facts") or {}).get("scales"))
        an["per_boot"][key] = pb
    for pk, (ra, rb, what) in PAIRS.items():
        pr = {"what": what, "ref": ra, "cand": rb, "bands": {}, "parity": {}}
        for band in NB0["tf_bands"]:
            fa = ((TFR.get(ra) or {}).get(band) or {}).get("fresh")
            fb = ((TFR.get(rb) or {}).get(band) or {}).get("fresh")
            c = cmp_last256(fa, fb)
            e = {"last256": c, "screen": screen(c)}
            if fa and fb and fa.get("arr") is not None and fb.get("arr") is not None:
                e["full"] = cmp_full(fa["arr"], fb["arr"])
                e["shift"] = shift_check(fa["arr"], fb["arr"])
            ref_aa = ((an["per_boot"].get(ra) or {}).get("aa") or {}).get(band)
            e["screening_gate"] = screening_gate(ref_aa, c)
            e["triage_gate"] = triage_gate(c, short_ok=boot_short_ok(rb))
            pr["bands"][band] = e
        for band in NB0["sc_bands"]:
            xa, xb = ((SCR.get(ra) or {}).get(band) or {}).get("cold"), ((SCR.get(rb) or {}).get(band) or {}).get("cold")
            pr["parity"][band] = decode_compare(xa, xb) if xa and xb else {"status": "INCONCLUSIVE_MISSING"}
        orc = (boot_record(rb).get("oracle") or {}).get("vs_" + ra)
        pr["oracle_128"] = orc
        an["pairs"][pk] = pr
    an["gates"] = fp8_verdict(an)
    an["findings"] = decide(an)
    return an


def fp8_verdict(an):
    out = {}
    for pk, label in (("CD", "FP8 native sm_89 read path, MTP K3 (production default) vs BF16 MTP K3"),
                      ("AB", "FP8 vs BF16, target only (MTP off)"),
                      ("CE", "FP8 through Merlin's byte decoder, MTP K3 vs BF16 MTP K3")):
        bands = ((an["pairs"].get(pk) or {}).get("bands")) or {}
        sc = {b: (bands.get(b) or {}).get("screening_gate", "INCONCLUSIVE_FP8") for b in NB0["tf_bands"]}
        tr = {b: (bands.get(b) or {}).get("triage_gate", "INCONCLUSIVE_TF") for b in NB0["tf_bands"]}
        out[pk] = {"what": label, "screening": worst(sc.values()), "triage": worst(tr.values()), "per_band": {b: [sc[b], tr[b]] for b in sc}}
    out["fp8_native_production"] = out["CD"]["triage"]
    out["fp8_byte_decoder"] = out["CE"]["triage"]
    out["note"] = ("No gate here gives GO: TF similarity cannot certify ARC correctness; the best outcome is "
                   "PENDING_LONG_ARC_ACCURACY (a paired ARC accuracy run on the same tasks decides).")
    return out


def decide(an):
    """The decision table applied to the measured comparisons; every finding names its evidence."""
    F, P, B = [], an.get("pairs") or {}, an.get("per_boot") or {}

    def st(pk, band):
        return (((P.get(pk) or {}).get("bands") or {}).get(band) or {}).get("screen", "INCONCLUSIVE")

    def bands_with(pk, s):
        return [b for b in NB0["tf_bands"] if st(pk, b) == s]

    for key, pb in B.items():
        for band, c in (pb.get("aa") or {}).items():
            s = screen(c)
            if s == "DIVERGENT":
                F.append({"finding": "REPEAT_DIVERGENCE", "boot": key, "band": band, "evidence": c,
                          "meaning": "the same teacher-forced request twice on one boot differs: cache / state"})
            elif s != "CLEAN":
                F.append({"finding": "AA_INCONCLUSIVE", "boot": key, "band": band, "evidence": c.get("status")})
        for band, sc in (pb.get("sc") or {}).items():
            if (sc.get("decode_vs_tf") or {}).get("screen") == "DIVERGENT":
                F.append({"finding": "DECODE_VS_PROMPT_LOGPROB_MISMATCH", "boot": key, "band": band, "evidence": sc["decode_vs_tf"],
                          "meaning": "prompt-logprob (prefill-output) path artefact or a prefill-vs-decode defect on this boot"})
            cc = sc.get("cold_vs_cached") or {}
            if cc.get("status") == "FAIL_HIGH_MARGIN" or (cc.get("common_prefix_mean_abs_dlogprob") or 0) > SCREEN["mean_max"]:
                F.append({"finding": "CACHED_PREFIX_DIVERGENCE", "boot": key, "band": band, "evidence": cc,
                          "meaning": "decode on a cached prefix differs from the cold decode: cache / state"})
        pc = pb.get("prefix_consistency") or {}
        if pc.get("consistent") is False:
            F.append({"finding": "CROSS_REQUEST_NONDETERMINISM", "boot": key, "evidence": pc,
                      "meaning": "the shared prompt prefix scores differently in two requests of one boot"})
        scl = pb.get("scales")
        if isinstance(scl, dict) and not scl.get("ok"):
            F.append({"finding": "STATIC_SCALES_NOT_FULLY_APPLIED", "boot": key,
                      "evidence": {k: scl.get(k) for k in ("applied_stages", "matched", "mismatched", "missing")}})
    for pk, pr in P.items():
        for band, e in (pr.get("bands") or {}).items():
            if e.get("screen") != "DIVERGENT":
                continue
            if (e.get("shift") or {}).get("alignment_artefact"):
                F.append({"finding": "POSITION_SHIFT_ARTEFACT", "pair": pk, "band": band, "evidence": e["shift"]})
            if any(v.get("localised") for v in ((e.get("full") or {}).get("chunk") or {}).values()):
                F.append({"finding": "CHUNK_BOUNDARY_LOCALISED", "pair": pk, "band": band, "evidence": e["full"]["chunk"]})
    for pk in ("AB", "CD", "CE"):
        meas = [b for b in NB0["tf_bands"] if st(pk, b) in ("CLEAN", "DIVERGENT")]
        div = bands_with(pk, "DIVERGENT")
        if not meas:
            continue
        if "2k" in div:
            f, m = "KV_DIVERGENCE_FROM_2K", "quantisation / scales / kernel"
        elif div and set(div) <= {"40k", "90k"}:
            f, m = "KV_DIVERGENCE_LONG_ONLY", "long-context accumulation / recurrent or cache state"
        elif div:
            f, m = "KV_DIVERGENCE_FROM_8K", "grows with context: accumulation / state"
        else:
            f, m = "KV_CLEAN", "no KV-dtype divergence in the measured bands"
        F.append({"finding": f, "pair": pk, "divergent_bands": div, "measured_bands": meas, "meaning": m,
                  "what": P[pk]["what"]})
    ab_clean = all(st("AB", b) == "CLEAN" for b in NB0["tf_bands"] if st("AB", b) != "INCONCLUSIVE") and any(
        st("AB", b) == "CLEAN" for b in NB0["tf_bands"])
    if ab_clean and bands_with("CD", "DIVERGENT"):
        F.append({"finding": "MTP_ONLY", "evidence": {"AB": "clean", "CD": bands_with("CD", "DIVERGENT")},
                  "meaning": "speculative decoding / PP / RecoverSSM"})
    for pk, f in (("AC", "MTP_CHANGES_TARGET_LOGITS_BF16"), ("BD", "MTP_CHANGES_TARGET_LOGITS_FP8")):
        if bands_with(pk, "DIVERGENT"):
            F.append({"finding": f, "pair": pk, "divergent_bands": bands_with(pk, "DIVERGENT"),
                      "meaning": "the MTP path (RecoverSSM / spec verify / PP) changes the target's prompt logits"})
    cd, ce = bands_with("CD", "DIVERGENT"), bands_with("CE", "DIVERGENT")
    ce_meas = [b for b in NB0["tf_bands"] if st("CE", b) in ("CLEAN", "DIVERGENT")]
    if cd and ce_meas and not ce:
        F.append({"finding": "NATIVE_SM89_E4M3_PATH_DEFECT", "evidence": {"CD_divergent": cd, "CE_clean": ce_meas},
                  "meaning": "the sm_89 native float8 read path is the defect; keep FP8 capacity via Merlin's byte decoder (boot E config)"})
    elif cd and ce:
        F.append({"finding": "FP8_DIVERGENCE_NOT_THE_NATIVE_CAST", "evidence": {"CD": cd, "CE": ce},
                  "meaning": "FP8 KV itself (scales / quantisation) or the shared prefill path"})
    if bands_with("DE", "DIVERGENT"):
        F.append({"finding": "READ_PATH_CHANGES_LOGITS", "pair": "DE", "divergent_bands": bands_with("DE", "DIVERGENT")})
    for pk in ("BD", "AC"):
        for band, x in ((P.get(pk) or {}).get("parity") or {}).items():
            if x.get("status") == "FAIL_HIGH_MARGIN":
                F.append({"finding": "MTP_GREEDY_DIVERGENCE_HIGH_MARGIN", "pair": pk, "band": band, "evidence": x,
                          "meaning": "MTP-on greedy output leaves the MTP-off greedy output at a confident token (GDN state on rejected drafts, cf. #63593)"})
    c40 = (((P.get("CD") or {}).get("bands") or {}).get("40k") or {}).get("last256") or {}
    F.append({"finding": "RC2_ANCHOR", "rc2_40k": RC2_REF["tf40k"]["fp8_vs_bf16"],
              "nb0_CD_40k": {k: c40.get(k) for k in ("status", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")}})
    return F
'''

CELL7 = r'''# ==================================================================== CELL 7 — NB0 main: boots, analysis, report
if not RERUN and not os.environ.get("NB0_UNIT"):
    R["nb0_design"] = {"boots": {k: {x: v for x, v in b.items() if x != "captures"} | {"captures": bool(b["captures"])}
                                 for k, b in BOOTS.items()}, "order": ORDER, "pairs": PAIRS, "params": NB0,
                       "screening": SCREEN, "triage": TRIAGE, "decision_table": DECISION_TABLE}
    R["prompt_check"] = prompt_check()
    log("NB0 prompt check", json.dumps({k: R["prompt_check"][k] for k in ("ok", "rc2_recorded", "reproduced", "missing")}))
    if not R["prompt_check"]["ok"]:
        ABORT["why"] = "PROMPT_MISMATCH: NB0's prompt ids are not RC2's"
    save()
    if R["sections"]["A0"]["merge"].get("ok") and not ABORT["why"]:
        rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        if rec.get("status") != "DONE":
            ABORT["why"] = ABORT["why"] or "Merlin install failed"
        ok, why = sha_gate(min(1500, max(60, left() - 900)))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            ABORT["why"] = ABORT["why"] or f"SHA_GATE: {why}"
        save()
        for key in ORDER:
            b = BOOTS[key]
            if key == "A":
                rb = boot_record("B")
                if rb.get("status") not in ("DONE", "TF_OOM", "PARTIAL_HARD_STOP"):
                    R["sections"].setdefault("NB0", {})[b["step"]] = {"status": "SKIPPED_DEPENDENT",
                        "why": f"boot B (the MTP-off path) ended {rb.get('status')}: A uses the same path"}
                    save()
                    continue
                if left() < (EST_MIN["A"] + 3) * 60 + 300:
                    DROP.add("A_90k")
            r = step("NB0", b["step"], EST_MIN[key] - (2 if f"{key}_90k" in DROP else 0), lambda key=key: run_boot(key))
            if key == "D" and r.get("status") in ("LAUNCH_FAILED", "SERVER_DIED", "ERROR", "BODY_ERROR", "GPU_NOT_EMPTY"):
                ABORT["why"] = ABORT["why"] or f"boot D (the RC2-proven FP8 config) ended {r.get('status')}: environment suspect"
            if r.get("status") in ("HOST_GATE_FAILED", "GPU_NOT_EMPTY", "FAIL_GPU_RELEASE"):
                ABORT["why"] = ABORT["why"] or f"boot {key}: {r.get('status')}"
    else:
        R["errors"].append("NB0 did not boot: " + (ABORT["why"] or "Merlin inputs missing / merge failed"))
    try:
        R["analysis"] = analyze()
    except Exception as e:
        R["analysis"] = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2500:]}
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    R["host_phase"] = HOST_PHASE
    save()
    an = R["analysis"]
    verdict = {"gates": an.get("gates"), "findings": [{k: v for k, v in f.items() if k != "evidence"} for f in an.get("findings") or []],
               "boots": {k: boot_record(k).get("status") for k in ORDER}, "abort": ABORT["why"], "minutes": R["minutes"]}
    (OUT / "nb0_verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    write_fallback_submission()
    print(json.dumps(verdict, default=str)[:8000])
'''

subs = {"__LAB__": json.dumps(RC2["LAB"]), "__PLE__": json.dumps(RC2["PLE"]),
        "__RC2REF__": json.dumps(json.dumps(RC2_REF)), "__QSAPATCH__": json.dumps(json.dumps(QSA_PATCH))}
cells_src = []
for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6, CELL7)):
    for k, v in subs.items():
        c = c.replace(k, v)
    if i:                                      # E23 / RC2: a competition rerun skips cells 2-7 without an exception
        c = "if not RERUN:\n" + textwrap.indent(c, "    ")
    compile(c, f"<nb0 cell {i + 1}>", "exec")
    cells_src.append(c)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, len(data)
OUT_NB.write_bytes(data)
(HERE / "nb0_code_export.py").write_bytes("\n\n".join(cells_src).encode("utf-8"))   # one file, for the local harness (LF)
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "|", len(data), "bytes")
