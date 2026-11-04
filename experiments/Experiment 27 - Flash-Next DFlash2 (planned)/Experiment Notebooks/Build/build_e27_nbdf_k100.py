"""E27 NBDF-K100 (user 2026-11-01): the NBDF DFlash2 experiment on the klee100 3bpw-MTP target.

  python build_e27_nbdf_k100.py [--kv-gate-report FILE] [--test-out DIR]
      -> ../Q27-XHIGH-E27_NBDF_K100.ipynb, nbdf_k100_code_export.py, nbdf_k100_cell7_body.py, nbdf_k100_build_facts.json and the
         GPU push folder D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e27-nbdf-k100__foysalemonshanto/ (nothing is pushed)

Built by the NBDF builder itself (build_e27_nbdf.py, run here with NBDF_VARIANT=k100): every NBDF cell, the CELL 7 body, the
gates (real failures only), the seam (dflash_seam.py 7356d473), nbdf_lib.py, the kit gate and the assembly are shared. This
module holds ONLY the marked "E27 K100" edits, applied as exact-once text replacements, and the klee100 boot path reused
verbatim from the proven klee100 notebooks:
  - the K100Q builder (Exp 25 Build/build_e26_k100_quality.py, imported read-only, as E28 K100 does): klee100 LOAD rev 3's
    pieces cut from the K3_ONLY notebook (sha 7773e63f, asserted there): the input edit (klee100 + FP8 PLE overlay + the
    Merlin env + the tf-ref dataset), the R1-18 lab loader (the lab = k100load_lab.py of the tf-ref dataset, sha ef393fbb =
    NBDF's packed literal), CELL 2b (fixkeys overlay merge, model dir, K6, the K7 / K3 streaming digests behind sha_gate, the
    static QSA scales), the klee100 worker extension (K-7 method map, V-4 kv scales); the payload loader k100q_payload()
    (K100_EMB + the worker extension = ONE sha-pinned file of the tf-ref dataset, fd92a814, as in the GPU-run K100Q / E28);
  - klee100 LOAD rev 3 CELL 7's merlin_fn_server additions (K6 before every launch, VLLM_QSA_KV_SCALES on FP8, the --help
    flag gate) and its boot gates (GPU_METHOD_MAP_PASS, STATIC_SCALES_APPLIED), re-expressed around NBDF's own launcher;
  - CELL 2 / CELL 3: klee100 LOAD's two edits (install timeout R1-7; the klee100 tokenizer source + its pinned sha).
The W4 NBDF build is unchanged by this variant (the NBDF builder asserts its own outputs; build it alone to check the sha).

FINAL design (coordinator 2026-11-01, from research A engine / memory / boot and B acceptance / gain / design; K100_CFG):
arms Q > D3s > D4s at 24 sequences, every arm at 7,933,526,016 B, D OOM -> EMBED_HOST at the same pin then 0.25-GiB rungs,
windows 8k@16 x 2, 20k@24, 40k@16, 90k@8, 116k@6, Q's greedy oracle A/A + ladder, D token-id gates relative to Q's A/A,
the teacher-forced 2k / 8k / 40k envelope gate, real-failure gates only, no cross-arm skips.
"""
import ast
import hashlib
import json
import os
import re
import runpy
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the FINAL config (coordinator 2026-11-01, from research A + B)
K100_CFG = {
    "arms": ["Q", "D3s", "D4s"],                       # D3s first (default K), D4s always runs; no D4 / D7 / D4b / D4t
    "seqs": 24, "conc_ladder": [2, 8, 23],              # the E28 production level (A's B2)
    "q_pin": 7933526016,                               # EVERY arm starts here (A's B3): Q pool 479 / 480, D-SW ~540
    "q_ladder": [7933526016, 7665090560],              # Q: a launch OOM -> one 0.25-GiB rung
    "d_retry": [[7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}],
                [7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}]],
    "pin_ladder": [7933526016, 7665090560, 7396655104, 6859784192],
    "memory_gate": "flag",                             # the 512 MiB reading is recorded, never a step-down
    "q_ref": "k100_none",                              # no recorded klee100 oracle rows: Q's gate vs a reference = recorded
    "method_map_need_mtp": "q_only",                   # GPU_METHOD_MAP_PASS: MTP experts required on MTP arms only
    "q_i8": True,                                      # Q = E28 K100 production: MTP K3 + I8 (DRAFT_HEAD=int8, I8_ENGAGED required,
                                                       # else the same pin with the BF16 head) + Q-defer + MERLIN_SHARE_DRAFT_WORKSPACE=1
    "q_captures": None,                                # = Merlin registry capture_sizes_for(24, 3) (filled below)
    "wkeys": ["8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"],
    "windows": [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]],
    "profile_level": 16,
    "tf_bands": ["2k", "8k", "40k"], "tf_mean_margin": 0.05, "tf_rank1_margin": 0.03,
    "long_thr_floor": {"90k": 0.037, "116k": 0.054},   # one long window per arm: W4's measured long A/A (research B)
    "shorten_arm": "D4s", "shorten_min_left_min": 25,  # budget short: D4s starts anyway and drops its last items (due())
    "tap_mode_alt": "pre_combine", "tap_est_min": 15, "tap_cap_min": 10,
    "prefill_admit_floor_tok_s": 3000,                 # the admit limit of a window is computed at <= 3,000 tok/s (klee100 3.3-3.5k)
}

# ---------------------------------------------------------------- the proven klee100 pieces (K100Q builder, read-only import)
_REF27 = E25B / "k100q_ref27b.json"
_REF27_SHA0 = sha(_REF27.read_bytes())
sys.dont_write_bytecode = True
sys.path.insert(0, str(E25B))
import build_e26_k100_quality as KQ   # noqa: E402  (module import only; it rewrites k100q_ref27b.json with the same bytes)
assert sha(_REF27.read_bytes()) == _REF27_SHA0, "the K100Q builder import changed k100q_ref27b.json"
K3ONLY_SHA = KQ.K3ONLY_SHA
assert K3ONLY_SHA == "7773e63f96f3ae468a21ad3fdff4376124f919eb17996e7ebe11c22d8312ce22"
PAYLOAD_NAME, PAYLOAD_SHA = KQ.PAYLOAD_NAME, KQ.PAYLOAD_SHA
assert (PAYLOAD_NAME, PAYLOAD_SHA) == ("k100q_payload_emb.json", "fd92a814040aea6dda12855badf801675580dcd78b2f2135eb0b86121dd80880")
assert sha((KQ.DS_STAGE / PAYLOAD_NAME).read_bytes()) == PAYLOAD_SHA, "the staged tf-ref payload != the K100Q / E28 one"
assert KQ.K100_PINS[0] == K100_CFG["q_pin"] == K100_CFG["pin_ladder"][0] and K100_CFG["pin_ladder"][-1] == KQ.K100_PINS[1]
assert [a - b for a, b in zip(K100_CFG["pin_ladder"], K100_CFG["pin_ladder"][1:])] == [268435456, 268435456, 536870912]
K100_CFG["q_captures"] = KQ.capture_sizes_for(K100_CFG["seqs"], 3)
assert K100_CFG["q_captures"][-1] == 96 and set(KQ.K100_CAPS) <= set(K100_CFG["q_captures"]) | {84}, K100_CFG["q_captures"]
_E28L_SRC = (Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
             / "e28k100_lib.py").read_text(encoding="utf-8").replace("\r\n", "\n")
_t = ast.parse(_E28L_SRC)
_want = ("last_value", "spec_from_argv", "_WTAG", "_HEAD_RX", "_SPEC_RX", "_literal", "i8_readback")
_seg = []
for _n in _t.body:
    _nm = getattr(_n, "name", None) or (_n.targets[0].id if isinstance(_n, ast.Assign) and isinstance(_n.targets[0], ast.Name) else None)
    if _nm in _want:
        _seg.append(ast.get_source_segment(_E28L_SRC, _n))
I8RB_SRC = "\n\n\n".join(_seg) + "\n"
assert len(_seg) == len(_want), len(_seg)
I8RB_SHA = sha(I8RB_SRC)
K100_EXT_BYTES = KQ.EXT_BYTES                         # klee100 LOAD rev 3's nb1_worker_ext.py (NB1 + K-7 method map + V-4 scales)
assert sha(K100_EXT_BYTES) == json.loads((KQ.DS_STAGE / PAYLOAD_NAME).read_text(encoding="utf-8"))["nb1_worker_ext.py"][0]
PAYLOAD_LOADER = KQ.K100_PAYLOAD_CELL1.split("NB5_LIB_SRC = ")[0].rstrip("\n") + "\n"
PAYLOAD_LOADER = PAYLOAD_LOADER.replace("__PAYLOADNAME__", PAYLOAD_NAME).replace("__PAYLOADSHA__", PAYLOAD_SHA)
assert "def k100q_payload():" in PAYLOAD_LOADER and "K100_EMB = k100q_payload()" in PAYLOAD_LOADER and "__" not in PAYLOAD_LOADER.replace("__import__", "")
CELL2B_SRC = KQ.CELL2B.replace(KQ._emb_line, "K100_EMB = dict(K100_EMB)   # E27 K100: k100q_payload() (klee100 LOAD rev 3's blobs, tf-ref dataset)")
assert CELL2B_SRC.count("K100_EMB = dict(K100_EMB)") == 1 and len(CELL2B_SRC) < 20000, len(CELL2B_SRC)
K3_LAB = KQ.K3_LAB                                    # dedented (4-space body in NBDF CELL 1: re-indented below)
K3_INPUTS = KQ.K3_INPUTS                              # 4-space indented, as NBDF CELL 1's else-block
LAB_SHA = KQ.LAB_SHA
assert LAB_SHA == "ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e"

OUT_NB_NAME = "Q27-XHIGH-E27_NBDF_K100.ipynb"
SLUG = "q27-xhigh-e27-nbdf-k100"
EXPORT_NAME, BODY_NAME, FACTS_NAME = "nbdf_k100_code_export.py", "nbdf_k100_cell7_body.py", "nbdf_k100_build_facts.json"
TITLE = "Q27 Xhigh E27 NBDF K100"
DATASETS = ["foysalemonshanto/q27-e25-nb0-tf-ref"]       # the lab, the QSA scales sidecar, the K100 payload (klee100 LOAD / K100Q)


def models(drafter_model):
    return list(KQ.MODELS) + [drafter_model]            # klee100 3bpw-MTP, the FP8 PLE overlay, the Merlin env + the drafter


def once(text, old, new, label):
    assert text.count(old) == 1, f"E27 K100 {label}: x{text.count(old)}"
    return text.replace(old, new)


# ---------------------------------------------------------------- cell edits
EXT_ADD_MARK = "\n# ==================================================================== E25 klee100 (K-7): quant method map (read-only)\n"


def combined_ext(nbdf_ext_src):
    """NBDF's worker extension (NB1 + NB4S nb4s_env, the env read-back) + klee100 LOAD rev 3's K-7 / V-4 additions (the
    method map and kv scales RPCs), the NB1 part asserted identical in both."""
    k = K100_EXT_BYTES.decode("utf-8")
    assert k.count(EXT_ADD_MARK) == 1
    head, add = k.split(EXT_ADD_MARK)
    n_head = nbdf_ext_src.split("    def nb4s_env(self")[0]
    assert head.rstrip("\n") == n_head.rstrip("\n"), "the NB1 part of the two worker extensions differs"
    assert "def nb4s_env(self" in nbdf_ext_src and "def k100_method_map(self):" in add and "def k100_kv_scales(self):" in add
    return nbdf_ext_src.rstrip("\n") + "\n\n" + EXT_ADD_MARK + add


def edit_cells(cell1, cell2, cell3_6, b85x):
    """NBDF CELL 1 / 2 / 3 / 6 -> the klee100 boot path; returns (cell1, cell2, cell2b, cell3_6, facts)."""
    c1 = cell1
    lab_line = next(l for l in c1.splitlines() if l.startswith("    LAB_SRC = "))
    c1 = once(c1, lab_line + "\n", textwrap.indent(K3_LAB, "    ").rstrip("\n") + "   # E27 K100: klee100 LOAD R1-18 (the lab of the tf-ref dataset)\n",
              "c1 lab")
    i0 = c1.index("    MERLIN_OK = all(PARTS.values()) and PLE_DIR is not None and MERLIN_PAY is not None\n")
    i1 = c1.index("    ABORT = {\"why\": None}\n", i0)
    c1 = c1[:i0] + K3_INPUTS + c1[i1:]
    c1 = once(c1, "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF marked edits)",
              "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF / E27 K100 marked edits)", "c1 title")
    c1 = once(c1, '"/kaggle/working/nbdf"))   # E27 NBDF', '"/kaggle/working/nbdf"))   # E27 NBDF (E27 K100: the same folder name)', "c1 out")
    c1 = once(c1, 'R = {"what": "E27 NBDF: DFlash2 K4 / K3 / K7 vs MTP K3 on the Q-defer base, 8k-116k (Flash-Next W4 on Merlin)",   # E27 NBDF',
              'R = {"what": "E27 NBDF-K100: DFlash2 vs MTP K3 on the Q-defer base, 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E27 K100', "c1 what")
    c2 = once(cell2, 'timeout=5400)', 'timeout=int(max(600, left() - 90 * 60)))   # E25 klee100 R1-7 (E27 K100)', "c2 install timeout")
    c3 = once(cell3_6[0], "    _src = CKPT if MERLIN_PROMPTS else INTEL\n",
              "    _src = CKPT if MERLIN_PROMPTS else (K100_DIR or INTEL)   # E25 klee100: the mount's own tokenizer when the merge failed (E27 K100)\n", "c3 tok src")
    c3 = once(c3, '    TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"   # = the 27B Merlin tokenizer.json',
              '    TOK_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"   # E25 klee100: klee100@600715b3 tokenizer.json (E27 K100)', "c3 tok sha")
    c6 = cell3_6[3]
    ext_line = next(l for l in c6.splitlines() if l.startswith("    NB1_EXT_SRC = "))
    import ast
    nbdf_ext = ast.literal_eval(ext_line.split("=", 1)[1].strip())
    ext = combined_ext(nbdf_ext)
    ext_sha = sha(ext)
    c6 = once(c6, ext_line + "\n",
              f'    NB1_EXT_SRC = (lambda _t: _t if hashlib.sha256(_t.encode()).hexdigest() == "{ext_sha}" else (_ for _ in ()).throw(\n'
              f'        RuntimeError("NB1_EXT_SRC sha mismatch")))(__import__("lzma").decompress(__import__("base64").b85decode("{b85x(ext)}")).decode("utf-8"))'
              "   # E27 K100: NBDF's extension (NB1 + nb4s_env) + klee100 LOAD rev 3's method map / kv scales\n", "c6 ext")
    c2b_add = K100_CELL2B_ADD.replace("__I8RB__", I8RB_SRC.rstrip("\n"))
    c2b = ("# ==================================================================== CELL 2b — E27 K100: the klee100 model path (klee100 LOAD rev 3 / K100Q, verbatim)\n"
           "import struct                                    # klee100 LOAD CELL 1 (\"import struct   # E25 klee100\"): st_header below\n"
           + PAYLOAD_LOADER + "R[\"setup\"][\"k100q_payload\"] = K100Q_PAYLOAD_USED\n\n\n" + CELL2B_SRC.rstrip("\n") + "\n\n\n" + c2b_add.replace("__K100CFG__", json.dumps(K100_CFG)))
    compile(c2b, "cell2b", "exec")
    facts = {"k3only_sha256": K3ONLY_SHA, "payload": {"file": PAYLOAD_NAME, "sha256": PAYLOAD_SHA}, "lab_sha256": LAB_SHA,
             "worker_ext_sha256": ext_sha, "k100_ext_add_sha256": sha(K100_EXT_BYTES), "cell2b_sha256": sha(CELL2B_SRC),
             "datasets": DATASETS, "cfg": K100_CFG, "i8_readback_sha256": I8RB_SHA}
    return c1, c2, c2b, [c3] + list(cell3_6[1:3]) + [c6], facts


# the klee100 LOAD rev 3 merlin_fn_server additions + boot gates + setup, around NBDF's launcher (CELL 2's merlin_fn_server)
K100_CELL2B_ADD = r'''# ---- E27 K100: klee100 LOAD rev 3 CELL 7's launch additions around NBDF's merlin_fn_server (wrapped once)
K100_CFG = json.loads(r"""__K100CFG__""")


def _k100_wrap_server(prev):
    def merlin_fn_server_k100(name, kv, port, kv_bytes):
        """K6 before every launch (one missing indexed shard -> no server), NBDF's launcher, VLLM_QSA_KV_SCALES on FP8 (Merlin
        also sets it from SCALES), the --help flag gate (KL.cli_flags_check: a missing flag -> FAIL_CLI_FLAGS, no launch)."""
        if not FAKE:
            k6 = KL.k6_check(CKPT)
            K1.setdefault("k6_before_launch", {})[name] = {k: k6.get(k) for k in ("ok", "why", "index_files", "missing")}
            if not k6["ok"]:
                save()
                raise RuntimeError("K6_INDEXED_SHARD_MISSING before launch: " + str(k6["why"]))
        s = prev(name, kv, port, kv_bytes)
        if kv == "fp8" and not FAKE:
            s.env["VLLM_QSA_KV_SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        if not FAKE and HELP.get(name) is not None:
            cli = K1.setdefault("cli_flags", {})[name] = KL.cli_flags_check(HELP[name])
            if cli.get("ok") is False:
                save()
                raise RuntimeError("FAIL_CLI_FLAGS: " + str(cli.get("why")))
        return s
    merlin_fn_server_k100._k100_orig = prev
    return merlin_fn_server_k100


if not hasattr(merlin_fn_server, "_k100_orig"):
    merlin_fn_server = _k100_wrap_server(merlin_fn_server)


def k100_setup():
    """Before the first boot (K100Q's k100_setup, NBDF's install order): the overlay merge + K6 (CELL 3's merge record), the
    FP8 QSA static scales sidecar (the installed Merlin tree first, then the tf-ref dataset; sha 5cbe6ae8) copied into the
    klee100 dir. NBDF's apply_60490 writes the worker extension; nb4_ppack (Q-defer) is NBDF's own sequence."""
    m = (R["sections"].get("A0") or {}).get("merge") or {}
    out = {"overlay_merge_k6": {"ok": m.get("ok") is True, "why": m.get("why") or m.get("bad")}}
    if FAKE:
        return dict(out, scales={"ok": None, "fake": True}, ok=out["overlay_merge_k6"]["ok"])
    K1["scales"] = out["scales"] = k100_scales(((MERLIN_ENG or {}).get("kv") or {}).get("MERLIN_TREE"))
    out["ok"] = out["overlay_merge_k6"]["ok"] and out["scales"].get("ok") is True
    out["why"] = None if out["ok"] else ("K100_OVERLAY_MERGE_K6_FAILED" if not out["overlay_merge_k6"]["ok"] else "K100_NO_SCALES")
    return out


def k100_boot_gates(p, s, arm, t):
    """klee100 LOAD rev 3 K-7 / V-4 on every boot: GPU_METHOD_MAP_PASS (the 48 routed layers' humming method + bits / group
    per PP stage; MTP experts required only when the arm speculates with MTP, K100_CFG method_map_need_mtp) and
    STATIC_SCALES_APPLIED (every rank applied its stage's QSA K/V scales; an MTP layer at DEFAULT 1.0 is recorded only).
    Returns {"method_map", "scales", "real_failure", "why"}; the caller decides (PROVISIONAL gate set)."""
    part = KL.parse_partition((getattr(s, "env", None) or {}).get("VLLM_PP_LAYER_PARTITION"))
    need_mtp = arm.get("method") != "dflash" if K100_CFG.get("method_map_need_mtp") == "q_only" else True
    raw = rpc(p, "k100_method_map")
    mm = KL.method_map_check((raw or {}).get("results"), need_mtp=need_mtp, partition=part)
    try:
        side = json.loads((CKPT / "qsa_kv_scales_262k.json").read_text())
    except Exception as e:
        side = {"_error": repr(e)[:200]}
    sc = KL.scales_check(t, {k: v for k, v in side.items() if not k.startswith("_")}, part)
    sc["calibration"] = SCALES_CALIBRATION
    why = ([] if mm.get("gate_ok") is True else ["GPU_METHOD_MAP: " + str(mm.get("why") or mm.get("verdict"))]) + \
          ([] if sc.get("ok") is True else ["STATIC_SCALES: " + str(sc.get("why"))])
    return {"method_map": mm, "method_map_raw": raw, "scales": sc, "scales_rpc": rpc(p, "k100_kv_scales"), "need_mtp": need_mtp,
            "real_failure": bool(why), "why": "; ".join(why)[:800] or None}


# ---- E28 K100 / NB4T: the I8 draft-head read-back (e28k100_lib.py, verbatim functions)
__I8RB__
'''


K100_BODY_HEAD = r'''
def _k100_next_pin(kvb):
    """The next lower rung of the K100 pin ladder (None below the last one)."""
    low = [x for x in [int(x) for x in K100_CFG["pin_ladder"]] if x < int(kvb)]
    return max(low) if low else None


def _k100_retry(arm, is_d, n_oom):
    """A's B3 (coordinator decision 4): the n-th launch OOM of an arm -> (pin, env added) or None. D: DFLASH_DRAFT_EMBED_HOST=1
    at the SAME pin first, then 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192 (never the -1 GiB step first); Q: one rung."""
    if is_d:
        plan = K100_CFG["d_retry"]
        return (int(plan[n_oom - 1][0]), dict(plan[n_oom - 1][1])) if 1 <= n_oom <= len(plan) else None
    ql = K100_CFG["q_ladder"]
    return (int(ql[n_oom]), {}) if 1 <= n_oom < len(ql) else None


# ---- E27 K100 (coordinator 2026-11-01, final design): Q = E28 production (I8), Q's A/A, the TF envelope, deferred reruns
Q_I8 = {"want": bool(K100_CFG["q_i8"]), "on": bool(K100_CFG["q_i8"]), "dropped": None}
NB4T_BUILD_ENV = {}                         # per launch name: DRAFT_HEAD=int8 into Merlin's build() (Q only, while Q_I8 is on)
Q_AA = {}                                   # Q's own A/A: greedy oracle twice (cache reset between) + its ladder
Q_TF_AA = {}                                # Q's teacher-forced A/A per band (fresh vs repeat on the same boot)
PENDING = []                                # (kind, arm) reruns deferred to the end: nometa / iso / tap (D4s always runs first)
NBDF.update(k100_cfg=K100_CFG, q_i8=Q_I8, q_aa=Q_AA, q_tf_aa=Q_TF_AA, pending=PENDING)
if "20k" not in BANDS:                      # the production-level window 20k@24 (K3_ONLY baseline 562 tok/s)
    BANDS["20k"] = 20000
    PROMPTS["20k"] = [prompt(BANDS["20k"], v) for v in range(int(K100_CFG["seqs"]))]
    MAX_NEW["20k"] = MAX_LEN - BANDS["20k"] - 512
TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2's TF variant 5)


def _k100_residents(o):
    """Capacity facts (coordinator decision 8): the boot's resident bound at 40k / 64k / 90k / 120k with the target block 4096
    (the MIXED path when the engine block is 1024), F at the boot's own block, the pool; Q's reference 29 / 21 / 17 / 13."""
    cap = o.get("capacity_after_ready") or {}
    pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
    blk = o.get("engine_block") or (((o.get("kv_layout") or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [N.BLOCK_TOKENS])[0]
    out = {"pool": pool, "engine_block": blk, "mc_131072": cap.get("max_concurrency_131072"), "q_reference": {"40k": 29, "64k": 21, "90k": 17, "120k": 13}}
    for nm, tok, band in (("40k", 40000, "40k"), ("64k", 65536, "90k"), ("90k", 90000, "90k"), ("120k", 120000, "116k")):
        try:
            lv, inf = N.level_for_band(band, tok, pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                                       1e9, 1.0, seqs=64)
            out[nm] = {"level": lv, "F": inf.get("F"), "block": inf.get("block"), "source": inf.get("source"), "mixed": inf.get("mixed")}
        except Exception as e:
            out[nm] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def _k100_tap_arm(a):
    """The tap-mode diagnostic (research B): the D arm with DFLASH_TAP_MODE=<alt>, smoke + oracle + 8k only (end of run)."""
    return N.variant_arm(a, "_TAP", "_tap", 9, env_add={"DFLASH_TAP_MODE": K100_CFG["tap_mode_alt"]},
                         title_add=f"DFLASH_TAP_MODE={K100_CFG['tap_mode_alt']} (diagnostic smoke, budget permitting)",
                         short_only=True, est_min=int(K100_CFG["tap_est_min"]), cap_min=int(K100_CFG["tap_cap_min"]))


def _k100_run_pending(runner):
    """Coordinator decision 1: no rerun may starve D4s; the nometa / Q-defer-off / tap-mode reruns run after every arm, each
    only when its full minimum still fits (else SKIPPED_BUDGET, recorded)."""
    isos = 0
    for kind, k in list(PENDING):
        a = ARMS[k]
        if kind == "iso":
            if isos >= N.ISO_MAX:
                continue
            isos += 1
            na = N.iso_arm(a)
        elif kind == "nometa":
            na = N.nometa_arm(a)
        else:
            na = _k100_tap_arm(a)
        if na["key"] in ISO:
            continue
        ISO[na["key"]] = na
        NBDF.setdefault("variants", {})[na["key"]] = {x: na.get(x) for x in _AKEYS}
        if N.arm_fits(left(), na):
            step(na["key"], na["name"], na["est_min"], lambda kk=na["key"]: runner(kk))
        else:
            skip_arm(na["key"], f"left {left() / 60:.1f} min < {na['est_min']} + 5 min", "SKIPPED_BUDGET")
'''


def edit_body(body):
    """NBDF's CELL 7 body -> K100 (the final design, K100_CFG)."""
    b = body
    b = once(b, '''B3_PIN = KV_FP8 + (1 << 30)                 # NB1 R2 B3 = NB4S B = NB4T Q: 3,553,580,524 B (FP8 KV)
if B3_PIN != 3553580524:
    raise RuntimeError(f"B3_PIN {B3_PIN} != 3553580524")''', '''B3_PIN = int(K100_CFG["q_pin"])             # E27 K100: every arm starts at the K100Q / E28 top K3 rung (Q pool 479, F 6)
if B3_PIN != 7933526016:
    raise RuntimeError(f"K100 Q pin {B3_PIN} != 7933526016")''', "b pin")
    b = once(b, '''Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4T Q_BASE_ENV''',
             '''Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4T Q_BASE_ENV
N.configure(SEQS=int(K100_CFG["seqs"]), CONC_LADDER=tuple(K100_CFG["conc_ladder"]), Q_CAPTURES=list(K100_CFG["q_captures"]),
            Q_TAIL=["--compilation-config", N.compilation_json(K100_CFG["q_captures"])], WKEYS=tuple(K100_CFG["wkeys"]),
            PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(), LONG_THR_FLOOR=dict(K100_CFG["long_thr_floor"]),
            TF_MEAN_MARGIN=float(K100_CFG["tf_mean_margin"]), TF_RANK1_MARGIN=float(K100_CFG["tf_rank1_margin"]), TF_REQUIRED=True)   # E27 K100
Q_EXTRA = Q_EXTRA + ["--max-num-seqs", str(K100_CFG["seqs"])]               # E27 K100: every arm at 24 sequences (E28)
Q_BASE_ENV["MERLIN_SHARE_DRAFT_WORKSPACE"] = "1"                              # E27 K100: = E28 / K100Q (inert with DFlash)''', "b configure")
    b = once(b, '''ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, DRAFTER_DIR, q_env=P.server_env("Q-defer"), seam_envs=SEAM_ENVS)''',
             '''ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, DRAFTER_DIR, q_env=P.server_env("Q-defer"), seam_envs=SEAM_ENVS)
N.restrict_arms(K100_CFG["arms"])                # E27 K100: Q > D3s > D4s
ARMS = {k: ARMS[k] for k in N.ORDER}''', "b arms")
    i = b.index('''_AKEYS = ("name", "port", "env",''')
    j = b.index("\n", b.index('''"title", "optional")''', i)) + 1
    b = b[:j] + K100_BODY_HEAD + b[j:]
    b = once(b, 'PIN = {"kv_bytes": B3_PIN, "source": "NB1 R2 B3 = NB4T Q"}',
             'PIN = {"kv_bytes": B3_PIN, "source": "E27 K100: K100Q / E28 top K3 rung"}', "b pin src")
    # pins: every arm starts at the klee100 pin; the retry plan (A's B3); the I8 drop (E28)
    b = once(b, '''    kvb = ((SW_PIN["kv_bytes"] or PIN["kv_bytes"] - N.SW_START_RUNGS * OOM_STEP) if is_sw else (D_PIN["kv_bytes"] or arm["kv_bytes"])) if is_d else PIN["kv_bytes"]''',
             '''    kvb = int(K100_CFG["q_pin"])              # E27 K100: every arm starts at 7,933,526,016 B''', "b kvb")
    b = once(b, '''    max_tries = 1 + ((N.SW_OOM_RETRIES if is_sw else N.D_OOM_RETRIES) if is_d else 1)''',
             '''    max_tries = 1 + (len(K100_CFG["d_retry"]) if is_d else len(K100_CFG["q_ladder"]))   # E27 K100: D 4 OOM retries; Q 1 OOM + 1 I8 drop
    n_oom = 0''', "b tries")
    b = once(b, '''        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb''',
             '''        if arm["key"] == "Q":                     # E27 K100: Q = E28 production, the INT8 draft head while Q_I8 is on
            arm["draft_head"] = "int8" if Q_I8["on"] else None
            arm["expect_cfg"]["draft_head"] = arm["draft_head"]
            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": "int8"} if Q_I8["on"] else {}
        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb''', "b i8 env")
    b = once(b, '''        if not (oom or (is_d and memstep)) or o["xid_in_log"] or len(tries) >= max_tries or left() < NBDF_READY_SEC + arm["cap_min"] * 60:
            break
        kvb -= OOM_STEP
        port += 1
        log("boot", "OOM" if oom else "memory gate", lname, "-> one rung lower, KV_BYTES", kvb, "port", port)''',
             '''        i8_drop = arm["key"] == "Q" and Q_I8["on"] and (o.get("status") == "I8_READBACK_UNVERIFIED"
                                                        or (o.get("status") == "LAUNCH_FAILED" and not o["xid_in_log"]))
        if i8_drop:                               # E28 K100 I8: the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100)
            Q_I8.update(on=False, dropped={"try": lname, "status": o.get("status"), "readback": (o.get("i8_readback") or {}).get("label")})
            tries[-1]["i8_drop"] = True
        n_oom += 1 if (oom and not i8_drop) else 0
        if not (oom or i8_drop) or o["xid_in_log"] or len(tries) >= max_tries or left() < NBDF_READY_SEC + arm["cap_min"] * 60:
            break
        if not i8_drop:                           # A's B3: D -> EMBED_HOST at the same pin, then 0.25-GiB rungs; Q -> one rung
            nxt = _k100_retry(arm, is_d, n_oom)
            if nxt is None:
                break
            kvb = nxt[0]
            if nxt[1]:
                arm["env"].update(nxt[1])
                arm["expect"] = {w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
                arm.setdefault("retry_env", {}).update(nxt[1])
        port += 1
        log("boot", "I8 drop" if i8_drop else "OOM", lname, "-> KV_BYTES", kvb, "env", arm.get("retry_env"), "port", port)''', "b retry")
    b = once(b, '''    return (arm.get("method") == "dflash" and (CUR.get("tries_left") or 0) > 0
            and left() >= NBDF_READY_SEC + arm["cap_min"] * 60 + 180)''', '''    return False                              # E27 K100: the memory gate is a flag (coordinator decision 4)''', "b can_step")
    b = once(b, '''        fr = (q.get("free_after_ready_min") or {}).get("min") or []
        pp1 = min(fr[2:4]) if len(fr) >= 4 and all(isinstance(x, (int, float)) for x in fr[2:4]) else None
        plan = N.pin_plan(PIN["kv_bytes"], pp1, 4, N.host_embed_on(SEAM_ENVS.get("D4") or {}))   # host embed off: embed counted
        D_PIN.update(kv_bytes=plan["pin"], plan=plan, source="pin_plan from Q's PP1 free memory")''',
             '''        D_PIN.update(kv_bytes=int(K100_CFG["q_pin"]), plan=None, source="E27 K100: every arm starts at 7,933,526,016 B")''', "b d pin")
    b = once(b, '''    ref = "rc2_fp8" if arm["key"] == "Q" else "q"''', '''    ref = K100_CFG["q_ref"] if arm["key"] == "Q" else "q"   # E27 K100''', "b ref1")
    b = once(b, '''    o = arm_boot(arm, arm_body(arm, "rc2_fp8" if key == "Q" else "q", "q" if key == "Q" else None))''',
             '''    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == "Q" else "q", "q" if key == "Q" else None))   # E27 K100''', "b ref2")
    b = once(b, '''        if "rc2_fp8" not in ROWS and not FAKE:''', '''        ks = step("A0", "k100_setup", 3, k100_setup)          # E27 K100: overlay merge + K6, the QSA static scales
        if ks.get("status") != "DONE" or ks.get("ok") is not True:
            for k in ARMS:
                skip_arm(k, f"K100 setup: {ks.get('why') or ks.get('error') or ks.get('status')}", "NOT_INTEGRATED", not_integrated=True)
            return
        if "rc2_fp8" not in ROWS and not FAKE:''', "b setup")
    # the I8 gate (Q) + the klee100 boot gates, after the read-backs and before the smoke
    b = once(b, '''        if due(240):
            CUR["item"] = "smoke"''', '''        if is_q:                                  # E28 K100 I8: INT8 requested -> I8_ENGAGED required, else the same pin + BF16 head
            o["i8_readback"] = (i8_readback(t, list(getattr(s, "argv", None) or []), arm.get("draft_head")) if not FAKE
                                else {"label": "FAKE_SERVER"})
            if arm.get("draft_head") == "int8" and o["i8_readback"].get("label") not in ("I8_ENGAGED", "FAKE_SERVER"):
                o.update(status="I8_READBACK_UNVERIFIED", early_stop="I8: " + json.dumps(o["i8_readback"], default=str)[:300])
                return o
        _live(o, "k100_gates", k100_boot_gates(p, s, arm, t) if not FAKE else {"real_failure": False, "fake": True})
        if o["k100_gates"].get("real_failure"):    # E27 K100: wrong quant kernels / FP8 KV without its static scales
            o.update(status="NO_GO_K100_BOOT_GATES", early_stop=str(o["k100_gates"].get("why"))[:400])
            return o
        if due(240):
            CUR["item"] = "smoke"''', "b gates")
    b = once(b, '''            o["smoke_gate"] = N.smoke_gate(o["smoke"], is_q)''',
             '''            o["smoke_gate"] = N.smoke_gate(o["smoke"], is_q, q_nondet=Q_AA.get("clean") is not True)   # E27 K100: ladder vs Q's A/A
            o["smoke_pos0"] = N.pos0_label((o["smoke"] or {}).get("acceptance"))                        # STRUCTURAL / TAP_SUSPECT (recorded)''', "b smoke")
    b = once(b, '''        o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")''', '''        if is_q:                                  # E27 K100: Q's A/A = the greedy oracle again after a cache reset + Q's own ladder
            CUR["item"] = "oracle_aa"
            o["oracle_aa_reset"] = reset_prefix(p)
            _live(o, "oracle_aa", oracle(p, key=None, refs=("q",)))
            Q_AA.update(oracle=(o["oracle_aa"].get("vs_q") or {}).get("status"),
                        ladder=N.contamination_gate((o.get("smoke") or {}).get("ladder")))
            Q_AA["clean"] = N.q_aa_clean(Q_AA)
            o["q_aa"] = dict(Q_AA)
        elif isinstance(o["oracle"].get("vs_" + ref), dict):   # D: token ids judged against Q's A/A (k100_rel)
            _vs = o["oracle"]["vs_" + ref]
            _vs["status_raw"] = _vs.get("status")
            _vs["status"] = N.k100_rel(_vs.get("status"), Q_AA.get("clean") is True)
        o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")''', "b oracle aa")
    b = once(b, '''            o["long_gate"] = N.long_oracle_gate(o, is_q)''', '''            if not is_q and isinstance((o.get("long_oracle") or {}).get("vs_q_long"), dict):   # E27 K100: vs Q's A/A
                _lv = o["long_oracle"]["vs_q_long"]
                _lv["status_raw"] = _lv.get("status")
                _lv["status"] = N.k100_rel(_lv.get("status"), Q_AA.get("clean") is True)
            o["long_gate"] = N.long_oracle_gate(o, is_q)''', "b long rel")
    b = once(b, '''        _live(o, "capacity_after_ready", T.capacity_facts(dict(o, pin=CUR["kv_bytes"])))''',
             '''        _live(o, "capacity_after_ready", T.capacity_facts(dict(o, pin=CUR["kv_bytes"])))
        o["k100_residents"] = _k100_residents(o)              # E27 K100: residents at 40k / 64k / 90k / 120k (block 4096)''', "b residents")
    # the windows: 8k@16 x 2, 20k@24, 40k@16, 90k@8, 116k@6 (fixed levels; lower only when the boot holds fewer)
    i = b.index("        long_ok = True\n")
    i = b.rindex('        run_window(o, s, arm, "8k", n, "#1")\n', 0, i)
    j = b.index("                run_window(o, s, arm, band, lv, tag)\n", i) + len("                run_window(o, s, arm, band, lv, tag)\n")
    b = b[:i] + '''        long_ok = True                            # E27 K100: the fixed window list (no short exit)
        for band, lvw, tag in K100_CFG["windows"]:
            lvw = int(lvw)
            if band in ("40k",) + N.LONG_BANDS:
                lv, why = band_level(o, band)
                use = min(lvw, lv) if isinstance(lv, int) and lv > 0 else lvw
                o["long_levels"][band] = dict(why, level=use, fixed=lvw, resident=lv)
                if band in N.LONG_BANDS:
                    CUR["item"] = f"reset_{band}"
                    o["long_levels"][band]["reset"] = reset_prefix(p) if due(win_need(band, use) + 60) else "SKIPPED"
                lvw = use
            _pf = PREFILL.get("tok_s")
            PREFILL["tok_s"] = min(float(_pf or 3000.0), float(K100_CFG["prefill_admit_floor_tok_s"]))   # the admit limit scales up
            try:
                run_window(o, s, arm, band, lvw, tag)
            finally:
                PREFILL["tok_s"] = _pf
''' + b[j:]
    b = once(b, '''        for band, lvb in (("8k", n),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''',
             '''        for band, lvb in (("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''', "b profile")
    # the teacher-forced envelope gate, LAST in the boot (prompt logits can OOM the last PP rank)
    b = once(b, '''        _live(o, "steady", o["steady"])
        CUR["item"] = "end"''', '''        _live(o, "steady", o["steady"])
        o["measured_before_tf"] = True            # E27 K100: a server death in the TF below keeps the measured verdict (flag)
        if due(240) and s.alive():
            CUR["item"] = "tf_k100"
            try:
                if is_q:
                    _live(o, "tf_ref", tforced(p, key="q_tf"))
                    _live(o, "tf_aa", tforced(p, ref="q_tf") if s.alive() else {"error": "server died"})
                    Q_TF_AA.clear()
                    Q_TF_AA.update({b_: v_ for b_, v_ in (o["tf_aa"] or {}).items() if isinstance(v_, dict)})
                else:
                    _live(o, "tf_vs_q", tforced(p, ref="q_tf"))
                    o["tf_gate"] = N.tf_gate(o["tf_vs_q"], Q_TF_AA, tuple(K100_CFG["tf_bands"]))
            except Exception as e:
                o["tf_error"] = f"{type(e).__name__}: {e}"[:300]
            o["tf_server_died"] = not s.alive()
        else:
            o["items_skipped"].append("tf_k100")
        CUR["item"] = "end"''', "b tf")
    # no cross-arm skips: Q not usable / a broken tree before one arm are recorded; the reruns go to the end
    b = once(b, '''        if not bq[0]:
            for k in N.D_ARMS:
                skip_arm(k, f"Q not a usable base (status {q.get('status')}, correctness {q.get('correctness')}, gate "
                            f"{q.get('quality_gate')}, base {bq[1]}: {bq[2]})")
            return''', '''        if not bq[0]:                              # E27 K100: recorded only; the D arms run anyway (no cross-arm skips)
            NBDF["base_not_usable"] = f"{bq[1]}: {bq[2]}"
            log("NBDF K100: Q not a usable base (recorded, the D arms run):", NBDF["base_not_usable"])''', "b base")
    b = once(b, '''            if stop_d:
                skip_arm(k, stop_d, "TREE_INVALID_MIXED", not_integrated=True)
                continue''', '''            if stop_d:                             # E27 K100: recorded; each arm's own tree check decides
                NBDF.setdefault("tree_notes", []).append(stop_d)
                stop_d = None''', "b stop_d")
    b = once(b, '''            if not N.arm_fits(left(), a):''',
             '''            if not N.arm_fits(left(), a) and not (k == K100_CFG["shorten_arm"] and left() >= K100_CFG["shorten_min_left_min"] * 60):''', "b fits")
    i = b.index("            if N.needs_nometa(r):")
    j = b.index('''                    skip_arm(ia["key"], f"left {left() / 60:.1f} min < {ia['est_min']} + 5 min", "SKIPPED_BUDGET")\n''', i)
    j = b.index("\n", j) + 1
    b = b[:i] + '''            if N.needs_nometa(r):                  # E27 K100: every rerun is deferred to the end (D4s always runs first)
                PENDING.append(("nometa", k))
            if N.needs_isolation(r) and not str(r.get("smoke_gate", "")).endswith("ZERO_ACCEPT"):
                PENDING.append(("iso", k))
            if (r.get("smoke_pos0") or {}).get("label") in ("STRUCTURAL", "TAP_SUSPECT"):
                PENDING.append(("tap", k))
        _k100_run_pending(runner)
''' + b[j:]
    return b


K100_KIT_KEYS = ("k100_setup_scales_sidecar_ok", "k100_r9_q480_dsw_kv_configs_ok", "k100_dsw_F_10_14",
                 "k100_block_drop_level_for_band_ok", "k100_mtp_vs_drafter_bytes_found", "k100_r12_klee100_draft_unquantized_67",
                 "k100_scan_empty_method_map_pass", "k100_tokenizer_mask_row_ok", "k100_captures_argv_packviews_seqs24_ok",
                 "k100_seam_ppack_60490_apply_verify_revert_ok")


def k100_gate_extra(gate, rep):
    """The K100 gated rebuild opens only when the K100 real kit itself passed: its ALL_OK, every K100 check above True, and
    an explicit per-K success for D3s / D4s from the real get_kv_cache_configs runs (nbdf_lib.kv_gate_from_report's per_k
    booleans: num_blocks > 0, no error, an int group count) AND the kit's own K100_R9 per-run ok (drafter page, PP0 page,
    no error; the ~540-block prediction is record-only). Never sets passed / per_k True by hand: it can only close the gate."""
    g = json.loads(json.dumps(gate))
    v = (rep or {}).get("verdict") or {}
    chk = v.get("checks") or {}
    why = list(g.get("why") or []) if not g.get("passed") else []
    if v.get("ALL_OK") is not True:
        why.append("the K100 real kit's ALL_OK is not True")
    miss = [k for k in K100_KIT_KEYS if chk.get(k) is not True]
    if miss:
        why.append(f"K100 real-kit checks not True: {miss}")
    r9 = ((rep or {}).get("checks") or {}).get("K100_R9") or {}
    succ = {}
    for arm, run in (("D3s", "K3_swblock1024"), ("D4s", "K4_swblock1024")):
        succ[arm] = {"kv_configs_ok": (g.get("per_k") or {}).get(arm) is True, "k100_r9_run_ok": (r9.get(run) or {}).get("ok") is True,
                     "run": run, "blocks": (r9.get(run) or {}).get("blocks"), "predicted": (r9.get(run) or {}).get("predicted")}
        succ[arm]["success"] = succ[arm]["kv_configs_ok"] and succ[arm]["k100_r9_run_ok"]
        if not succ[arm]["success"]:
            why.append(f"no explicit per-K KV-config success for {arm} ({run}): {succ[arm]}")
    g["per_k_success"] = succ
    g["k100_kit_checks"] = {k: chk.get(k) for k in K100_KIT_KEYS}
    if why:
        g["passed"] = False
        g["why"] = why
    return g


def edit_meta(md0, flags):
    md = ("# E27 NBDF-K100: DFlash2 vs MTP K3 (Qwen3.8-Flash-Next klee100 3bpw-MTP, Merlin v3.0.0, 4 x L4), 8k to 116k\n\n"
          "Arms: " + " > ".join(K100_CFG["arms"]) + " (seqs 24; Q = E28 K100 production: MTP K3 + I8 + Q-defer). The klee100 boot = klee100 LOAD rev 3 / K100Q (overlay merge, K6, K7 / K3 "
          "digests, method map + static QSA scales gates), pin 7,933,526,016. Readable code: nbdf_k100_cell7_body.py. Commit run only; a "
          "competition rerun writes CELL 0's identity submission (E18 R2 format).\n")
    f = dict(flags)
    f["K100"] = {"boot": "klee100 LOAD rev 3 (K3_ONLY 7773e63f) / K100Q / E28 K100", "cfg": K100_CFG,
                 "method_map": "need_mtp only on MTP arms: the DFlash2 drafter is dense (5 layers, no MoE) -> no MTP / drafter expert rows",
                 "scales": "the sidecar has the 12 target QSA layers only; the MTP attention layer is DEFAULT 1.0 and recorded (never an error)"}
    return md, f


if __name__ == "__main__":
    os.environ["NBDF_VARIANT"] = "k100"
    sys.argv = [str(HERE / "build_e27_nbdf.py")] + sys.argv[1:]
    runpy.run_path(str(HERE / "build_e27_nbdf.py"), run_name="__main__")
