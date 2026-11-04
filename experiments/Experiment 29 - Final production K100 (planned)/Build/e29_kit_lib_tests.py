"""E29 kit unit tests (quick; the Kaggle CPU units kit runs them on Py3.12): e29_lib's pure helpers and the notebook's OWN
nb5_boot / e29_env / e29_extra / e29_ntries / e29_drop defs (exec of the built notebook's engine cell) with E29 levers on,
a scripted nb5_boot_try (no server, no GPU). The real-mount versions of the same paths run in the real kit.

usage: python e29_kit_lib_tests.py   env E29LT_NB (notebook) / E29LT_FACTS (<variant>_build_facts.json) / E29LT_KF (kit facts:
       variants_cfg, nbdf_k100 d_retry / q_pin / q_captures) / E29LT_NB5LIB (nb5_lib.py). e29_lib.py, e28k100_lib.py next to it.
Prints one PASS / FAIL line per check, then "<n>/<m> PASS"; exit 1 on any FAIL.
"""
import ast
import json
import os
import sys
import time
import traceback
import importlib.util
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_lib as L          # noqa: E402
import e28k100_lib as E28L   # noqa: E402

RESULTS = []


def check(label, fn):
    try:
        ok, why = bool(fn()), ""
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-700:]}"
    RESULTS.append((label, ok))
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <- " + why[:1200]), flush=True)
    return ok


NB = Path(os.environ["E29LT_NB"])
FACTS = json.loads(Path(os.environ["E29LT_FACTS"]).read_text(encoding="utf-8"))
KF = json.loads(Path(os.environ["E29LT_KF"]).read_text(encoding="utf-8"))
_s = importlib.util.spec_from_file_location("nb5_lib_e29lt", os.environ["E29LT_NB5LIB"])
NB5L = importlib.util.module_from_spec(_s)
_s.loader.exec_module(NB5L)
CODE = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def unpack_engine(c):
    """rev 2: the DFlash notebook packs its E29 engine block (= build_e29_final.unpack_engine): the readable engine cell."""
    head = "\n\n\n# ---- E29 FINAL rev 2 (size: the 400-kB commit limit): the E29 engine block"
    while head in c:                # rev 2d: every packed block
        h = c.index(head)
        x = c.index("exec(compile(", h)
        z = c.index("\n", x) + 1
        call = next(n for n in ast.walk(ast.parse(c[x:z])) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        expr = ast.get_source_segment(c[x:z], call.args[0].args[0])
        c = c[:h] + eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e29 packed>", "eval")) + c[z:]
    return c


C1, ENG = CODE[1], unpack_engine(CODE[4])
VAR = FACTS["e29"]["variant"]
REV2 = L.LIB_VERSION.startswith("e29_lib 1.1")          # the DFlash notebook (rev 2); DEFAULT / MTP_V64 keep e29_lib 1.0


def raises(fn):
    try:
        fn()
        return False
    except (AssertionError, ValueError, TypeError, KeyError):
        return True


def modules():
    line = next(l for l in C1.splitlines() if l.startswith("E28_SRC = json.loads("))
    expr = ast.get_source_segment(line, ast.parse(line).body[0].value.args[0])
    return json.loads(eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<packed>", "eval")))


MODS = modules()
NBDF = {}
exec(compile(MODS["e29_nbdf.py"], "e29_nbdf.py", "exec"), NBDF)


# ------------------------------------------------------------------ L: e29_lib
def _l1():
    pre = 'E29_CFG = E29L.check_cfg(json.loads(r"""'
    line = next(l for l in C1.splitlines() if l.startswith(pre))
    cfg = json.loads(line[len(pre):line.rindex('"""')])
    var = next(l for l in C1.splitlines() if l.startswith("E29_VARIANT = "))
    return (cfg == FACTS["e29"]["cfg"] == L.check_cfg(cfg) and var == f'E29_VARIANT = "{VAR}"'
            and MODS["e29_lib.py"] == (HERE / "e29_lib.py").read_text(encoding="utf-8"))


check("L1 the notebook's E29_CFG literal == the facts' cfg == check_cfg(it); E29_VARIANT == the facts' variant; the packed "
      "e29_lib.py == this e29_lib.py", _l1)
check("L2 check_cfg: every builder variant (E29_DEFAULT / E29_DFLASH_D3S / E29_MTP_V64 / ALL_OFF) passes; ALL_OFF == default_cfg; "
      "bad values raise (unknown key, drafter, vocab with dflash, embed_host with mtp, workers 20, a non-bool flag)",
      lambda: all(L.check_cfg(c) == dict(L.default_cfg(), **c) for c in KF["variants_cfg"].values())
      and L.is_default(KF["variants_cfg"]["ALL_OFF"]) and not L.is_default(KF["variants_cfg"]["E29_DEFAULT"])
      and raises(lambda: L.check_cfg({"x": 1})) and raises(lambda: L.check_cfg({"drafter": "eagle"}))
      and raises(lambda: L.check_cfg({"drafter": "dflash_d3s", "draft_vocab": "64k"}))
      and raises(lambda: L.check_cfg({"embed_host": True})) and raises(lambda: L.check_cfg({"workers": 20}))
      and raises(lambda: L.check_cfg({"indexer_fp8": 1})))


def _l3():
    t0, t1 = L.df_tries(False), L.df_tries(True)
    want = [[KF["nbdf_q_pin"], False]] + [[p, bool(e.get("DFLASH_DRAFT_EMBED_HOST") == "1")] for p, e in KF["nbdf_d_retry"]]
    got0 = [[t["kv_bytes"], t["df"]["embed_host"]] for t in t0]
    assert got0 == want, (got0, want)
    assert [t["df"]["rung"] for t in t0] == [0, 1, 2, 3, 4] and all(abs(t["gib"] - L.gib_of(t["kv_bytes"])) < 1e-12 for t in t0)
    assert L.D_RETRY == [[p, True] for p, _ in KF["nbdf_d_retry"]] and L.D_PIN0 == KF["nbdf_q_pin"] == FACTS["pins"][0]
    # rev 2f: EMBED_HOST first starts at L.D_START_PIN (the GPU smoke: the top rung always stepped on the F2 probe)
    return ([[t["kv_bytes"], t["df"]["embed_host"]] for t in t1] == [w for w in want[1:] if w[0] <= L.D_START_PIN]
            and L.D_START_PIN == 7665090560 and t1[0]["kv_bytes"] == L.D_START_PIN)


check("L3 df_tries == the NBDF-K100 D ladder: try 0 = the K3 top pin 7,933,526,016 B (no EMBED_HOST), then NBDF's d_retry rungs "
      "(EMBED_HOST=1); embed_host first skips the duplicate pin-0 rung; gib = (pin - FP8 pin) / 2^30", _l3)
check("L4 lever_to_drop: fixed order df > idx > vocab > gdn; a lever the log names goes first (indexer_kv_dtype / draft_vocab / "
      "gdn switches; 1.1: only in an error line); df never by name; None with no E29 lever on",
      lambda: L.lever_to_drop(["vocab", "idx", "df"], "OOM") == "df" and L.lever_to_drop(["idx", "vocab"], "") == "idx"
      and L.lever_to_drop(["df", "idx"], "FAIL_CLI_FLAGS: missing flags: --attention-config.indexer_kv_dtype") == "idx"
      and L.lever_to_drop(["idx", "vocab"], "ValueError: bad draft_vocab file") == "vocab"
      and L.lever_to_drop(["gdn_align", "gdn_fastpath"], "RuntimeError: VLLM_GDN_SHARED_ALIGN_INDEX ...") == "gdn_align"
      and (not REV2 or (L.lever_to_drop(["idx", "vocab"], "INFO [qsa_indexer.py:797] QSA prefill logits; sparse_indexer_topk_backend") == "idx"
                        and L.lever_to_drop(["vocab", "idx"], "INFO draft_vocab loaded") == "idx"))
      and L.lever_to_drop([], "x") is None and L.lever_to_drop(None) is None)
_TAIL = L.df_spec_argv('{"method":"dflash"}', '{"cudagraph_capture_sizes":[4,8]}')
check("L5 df_spec_argv = the 4 trailing args; strip_df_argv round trip (base + idx + tail -> base + idx; no tail -> unchanged; "
      "an earlier identical block stays: only the LAST exact occurrence is removed)",
      lambda: _TAIL == ["--speculative-config", '{"method":"dflash"}', "--compilation-config", '{"cudagraph_capture_sizes":[4,8]}']
      and L.strip_df_argv(["--a", "1"] + L.IDX_ARGS + _TAIL, _TAIL) == ["--a", "1"] + L.IDX_ARGS
      and L.strip_df_argv(["--a", "1"], _TAIL) == ["--a", "1"] and L.strip_df_argv(_TAIL + ["--b"] + _TAIL, _TAIL) == _TAIL + ["--b"]
      and L.strip_df_argv(None, _TAIL) == [] and L.strip_df_argv(["--x"], []) == ["--x"])


def _c(d, t, a):
    return {"drafts": d, "draft_tokens": t, "accepted": a}


check("L6 canary (nbdf smoke rule): 0 / 300 draft tokens accepted -> hit DFLASH_ZERO_ACCEPT; 4 / 300 (1.3 %) -> hit; 150 tokens -> "
      "TOO_FEW_DRAFTS (no hit); 30 % -> OK with AL 1 + accepted / drafts; None counters -> no hit",
      lambda: L.canary(_c(0, 0, 0), _c(100, 300, 0), 3)["label"] == "DFLASH_ZERO_ACCEPT"
      and L.canary(_c(0, 0, 0), _c(100, 300, 4), 3)["hit"] is True
      and L.canary(_c(0, 0, 0), _c(50, 150, 0), 3)["label"] == "TOO_FEW_DRAFTS" and L.canary(_c(0, 0, 0), _c(50, 150, 0), 3)["hit"] is False
      and (lambda r: r["label"] == "OK" and r["al"] == 1.9 and r["accept_rate"] == 0.3)(L.canary(_c(10, 30, 0), _c(110, 330, 90), 3))
      and L.canary(None, None, 3)["hit"] is False)


def _rows(vals):
    return [{"t": 1000.0 + 15 * i, "sd_drafts": d, "sd_draft_tok": t, "sd_acc": a} for i, (d, t, a) in enumerate(vals)]


check("L7 inrun_zero on sampler rows (hit at 0 % over >= 200 draft tokens, TOO_FEW_ROWS with < 2 rows); hourly_al per hour + the "
      "0.7 x first-hour flag; next_try_index (OOM -> next D rung, else the first non-D try); is_oom",
      lambda: L.inrun_zero(_rows([(0, 0, 0), (100, 300, 0)]), 3)["hit"] is True
      and L.inrun_zero(_rows([(0, 0, 0)]), 3)["label"] == "TOO_FEW_ROWS"
      and L.inrun_zero(_rows([(0, 0, 0), (100, 300, 200)]), 3)["hit"] is False
      and (lambda h: h["first_al"] == 3.0 and h["flag"] == "AL_DROP_BELOW_0.7_FIRST_HOUR" and h["low_hours"] == [1])(
          L.hourly_al([{"t": 0, "sd_drafts": 0, "sd_draft_tok": 0, "sd_acc": 0}, {"t": 3500, "sd_drafts": 100, "sd_draft_tok": 300, "sd_acc": 200},
                       {"t": 7000, "sd_drafts": 200, "sd_draft_tok": 600, "sd_acc": 300}]))
      and L.next_try_index([{"df": 1}, {"df": 1}, {}], 0, True, False) == 1 and L.next_try_index([{"df": 1}, {"df": 1}, {}], 0, False, False) == 2
      and L.next_try_index([{"df": 1}, {}], 0, True, False) == 1 and L.next_try_index([{"df": 1}], 0, True, False) == 1
      and L.is_oom("torch.OutOfMemoryError: CUDA out of memory") and not L.is_oom("ValueError"))
_CR = {"ok": False, "checks": {"kv_dtype": True, "max_num_seqs": True, "speculative_config": False, "engine_spec": False},
       "log": {"speculative_config": {"method": "dflash", "num_speculative_tokens": 3}, "engine_spec": {"method": "dflash", "num_spec_tokens": 3}}}
check("L8 df_config_check: C-11's MTP checks dropped, the DFlash arch line + method dflash K3 (log spec + engine spec) required; "
      "a wrong K or a missing arch line -> INCONCLUSIVE_CONFIG_MISMATCH",
      lambda: (lambda r: r["ok"] and r["verdict"] == "PASS" and r["mtp_checks_dropped"] == ["speculative_config", "engine_spec"])(
          L.df_config_check(_CR, "Resolved architecture: DFlash2DraftModel", 3))
      and L.df_config_check(_CR, "Resolved architecture: DFlash2DraftModel", 4)["ok"] is False
      and L.df_config_check(_CR, "", 3)["verdict"] == "INCONCLUSIVE_CONFIG_MISMATCH")


def _l9():
    sp = NBDF["dflash_spec"]("/d", 3, "auto", "FLASH_ATTN")
    caps = NBDF["capture_sizes"](24, 3)
    js, cj = json.loads(NBDF["spec_json"](sp)), json.loads(NBDF["compilation_json"](caps))
    assert caps[0] == 4 and caps[-1] == 96 and all(c % 4 == 0 for c in caps), caps
    assert js["method"] == "dflash" and js["num_speculative_tokens"] == 3 and js["model"] == "/d", js
    assert js.get("kv_cache_dtype") == "auto" and js.get("attention_backend") == "FLASH_ATTN" and js.get("draft_sample_method") == "probabilistic", js
    assert cj.get("cudagraph_capture_sizes") == caps, cj
    assert NBDF["SW_BLOCK"] == 1024 and NBDF["DRAFTER_MODEL"] == FACTS["e29"]["drafter_model"] and L.K_OF["dflash_d3s"] == 3
    return KF["nbdf_q_captures"][-1] == caps[-1] == 96


check("L9 the DFlash spec the notebook builds (its e29_nbdf literal = nbdf_lib verbatim): method dflash, K3, the drafter dir, BF16 "
      "drafter KV auto, FLASH_ATTN, probabilistic; captures(24, K3) = the multiples of 4 up to 96 (= NBDF-K100 q_captures max); the "
      "compilation JSON carries them; SW_BLOCK 1024; the drafter Model pin", _l9)


# ------------------------------------------------------------------ B: the notebook's own nb5_boot with E29 levers on
def _defs(names):
    t = ast.parse(ENG)
    fns = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in fns) == sorted(names), [f.name for f in fns]
    return ast.Module(fns, [])


class DSStub:
    ALL_ENVS = ("DFLASH_SEAM", "DFLASH_MASK_ROW_SHA", "DFLASH_DRAFT_SW_BLOCK", "DFLASH_DRAFT_EMBED_HOST")

    @staticmethod
    def server_env(mask_row_sha=None, embed_host=False, draft_sw_block=None, **_):
        e = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": mask_row_sha, "DFLASH_DRAFT_SW_BLOCK": str(draft_sw_block)}
        if embed_host:
            e["DFLASH_DRAFT_EMBED_HOST"] = "1"
        return e


TAIL = L.df_spec_argv(NBDF["spec_json"](NBDF["dflash_spec"]("/drafter", 3, "auto", "FLASH_ATTN")),
                      NBDF["compilation_json"](NBDF["capture_sizes"](24, 3)))
STEPS = [float(x) for x in FACTS["pin_steps_gib"]]
SEQS_T = {round(float(k), 2): int(v) for k, v in FACTS["seqs_table"].items()}


def ladder(on, cfg, script, rerun=False, i8_on=True, vocab=None, ds=None, start_gib=None, qsa=None):
    g = {"json": json, "time": time, "NB5L": NB5L, "NB5_KV_DTYPE": "fp8", "NB5_KV_BYTES": {"fp8": L.FP8_PIN}, "NB5_BOOTS": [],
         "NB5_EXTRA_ARGS": [], "NB5_MTP_OFF_ARGS": ["--speculative-config", "None"], "FAKE": None,
         "P": dict({"pin_ladder": True, "patch_60490": True, "k100": True, "i8": True, "top_rung_seqs": 24}, **({"e18_ladder": True} if rerun else {})),
         "NB5_PIN": {"start_gib": start_gib, "levers": None, "patch_60490": {"ok": True}}, "K100_PIN_STEPS_GIB": STEPS,
         "E29_QSA": qsa if qsa is not None else {},
         "NB5_LEVER_ENV": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, "NB5_SEQS_RULE": "k100", "K100_SEQS_TABLE": SEQS_T,
         "left": lambda: 1e9, "E28_I8": {"want": True, "on": i8_on, "static": None, "dropped": None, "readback": None},
         "E28_I8_ENV": {"DRAFT_HEAD": "int8"}, "E28L": E28L, "log": lambda *a: None, "nb5_write_report": lambda: None, "ABORT": {},
         "NB5_READY_TIMEOUT": 1800, "NB5_BOOT_LEVERS": {}, "NB5_BOOT_EXTRA": [], "NB5_BOOT_KV": None,
         "E29L": L, "E29_CFG": L.check_cfg(cfg), "DS": ds or DSStub, "SSEQ": None, "NOTEBOOK_T0": time.perf_counter(),
         "E29_RT": {"cfg": cfg, "on": list(on), "dropped": [], "tries": [], "canary": [], "errors": [], "df_try": False, "df_live": False,
                    "df": {"k": 3, "tail": TAIL, "mask_row_sha": "ab" * 32, "sw_block": 1024} if "df" in on else None,
                    "vocab": vocab}}
    if not i8_on:
        g["E28_I8"]["suspended"] = "N/A_DFLASH"
    seen, it = [], iter(script)

    def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
        st = next(it)
        st, st_log = (st if isinstance(st, tuple) else (st, None))      # rev 2: (status, log text) = a scripted real failure
        av = list(g["NB5_BOOT_EXTRA"])
        seen.append({"name": name, "kv_bytes": kv_bytes, "levers": dict(g["NB5_BOOT_LEVERS"]), "extra": av, "df_try": g["E29_RT"]["df_try"],
                     "seqs": av[av.index("--max-num-seqs") + 1] if "--max-num-seqs" in av else None, "e29_on": list(trec.get("e29_on") or []),
                     "e28_i8": trec.get("e28_i8")})
        if st_log is not None:
            trec.update(status=st, log_text_for_ladder=st_log, **({"phases_started": True} if st == "DONE" else {}))
        elif st == "OOM":
            trec.update(status="SERVER_DIED", log_text_for_ladder="torch.OutOfMemoryError: CUDA out of memory")
        elif st == "NOT_ENGAGED":
            trec.update(status="E29_DFLASH_NOT_ENGAGED", log_text_for_ladder="")
        elif st == "CLI":
            trec.update(status="E29_LAUNCH_PREP_FAILED", log_text_for_ladder="RuntimeError: FAIL_CLI_FLAGS: missing flags: --attention-config.indexer_kv_dtype")
        else:
            trec.update(status="DONE", phases_started=True)
        return trec
    g["nb5_boot_try"] = nb5_boot_try
    exec(compile(_defs(["nb5_boot", "e29_env", "e29_extra", "e29_ntries", "e29_drop"]), "<nb5_boot e29>", "exec"), g)
    b = g["nb5_boot"]("merlin_fp8_mtp", [["Q", "all", None, "on", "A", "fp8"]], True, "fp8")
    return b, seen, g


DCFG = dict(L.default_cfg(), drafter="dflash_d3s", indexer_fp8=True, attempts_supply=True)
VCFG = dict(L.default_cfg(), draft_vocab="64k", indexer_fp8=True, attempts_supply=True)
TOP = FACTS["pins"][0]
DPINS = [KF["nbdf_q_pin"]] + [p for p, _ in KF["nbdf_d_retry"]]


def _b1():
    for rr in (False, True):
        b, seen, g = ladder(["df", "idx"], DCFG, ["OOM"] * 5 + ["OOM", "DONE"], rerun=rr, i8_on=False)
        assert [s["kv_bytes"] for s in seen] == DPINS + [TOP, TOP], [s["kv_bytes"] for s in seen]
        d, m = seen[:5], seen[5:]
        assert all(s["df_try"] and s["extra"][-4:] == TAIL and s["seqs"] == "24" and s["levers"]["DFLASH_SEAM"] == "1"
                   and s["levers"]["DFLASH_DRAFT_SW_BLOCK"] == "1024" and "DRAFT_HEAD" not in s["levers"] for s in d), d
        assert [s["levers"].get("DFLASH_DRAFT_EMBED_HOST") for s in d] == [None, "1", "1", "1", "1"], d
        assert all(" ".join(L.IDX_ARGS) in " ".join(s["extra"]) for s in d + m[:1]) and " ".join(L.IDX_ARGS) not in " ".join(m[1]["extra"])
        assert [s["name"].split("_pin")[1] for s in d][0].endswith("_df3r0_idx"), [s["name"] for s in seen]
        assert all(not s["df_try"] and s["levers"].get("DRAFT_HEAD") == "int8" and not any(k.startswith("DFLASH_") for k in s["levers"])
                   and s["seqs"] == "24" and "dflash" not in " ".join(s["extra"]) for s in m), m
        assert [x["lever"] for x in g["E29_RT"]["dropped"]] == ["df", "idx"] and b["status"] == "DONE" and g["E28_I8"]["on"] is True, (
            g["E29_RT"]["dropped"], b["status"], g["E28_I8"])
        assert m[1]["name"].endswith("_i8") and m[0]["name"].endswith("_i8_idx"), [s["name"] for s in m]
    return True


check("B1 DFLASH (df + idx on): the notebook's nb5_boot runs the 5 NBDF-K100 D tries first (pins = q_pin + d_retry; --max-num-seqs "
      "24; the trailing dflash spec + captures LAST; seam env DFLASH_SEAM=1 / SW_BLOCK 1024 / EMBED_HOST from rung 1; no DRAFT_HEAD), "
      "an OOM on every D rung -> df dropped -> the MTP K3 top rung with I8 restored + idx, an OOM there -> idx dropped -> the same pin "
      "plain E28 (DONE); the commit and the rerun (E18 3-try cap counts non-E29 tries only)", _b1)


def _b2():
    b, seen, g = ladder(["df", "idx"], DCFG, ["NOT_ENGAGED", "DONE"], i8_on=False)
    assert [s["kv_bytes"] for s in seen] == [DPINS[0], TOP] and seen[0]["df_try"] and not seen[1]["df_try"], seen
    assert seen[1]["levers"].get("DRAFT_HEAD") == "int8" and seen[1]["e29_on"] == ["idx"] and b["status"] == "DONE"
    b2, seen2, g2 = ladder(["df", "idx"], DCFG, ["CLI", "NOT_ENGAGED", "DONE"], i8_on=False)
    assert [s["kv_bytes"] for s in seen2] == [DPINS[0], DPINS[0], TOP], seen2
    assert seen2[0]["e29_on"] == ["df", "idx"] and seen2[1]["e29_on"] == ["df"] and seen2[2]["e29_on"] == []
    return [x["lever"] for x in g2["E29_RT"]["dropped"]] == ["idx", "df"] and b2["status"] == "DONE"


check("B2 DFLASH: a non-OOM D failure (NOT_ENGAGED) ends DFlash at once -> the MTP top rung (I8 + idx); a FAIL_CLI_FLAGS naming the "
      "indexer flag drops idx first, the same D pin retried without it", _b2)


def _b3():
    voc = {"want": "64k", "file": "/m/draft_vocab_k100_arc_n65536.pt", "sha256": L.VOCAB_FILES["64k"][1]}
    b, seen, g = ladder(["idx", "vocab"], VCFG, ["OOM", "OOM", "DONE"], vocab=voc)
    assert [s["kv_bytes"] for s in seen] == [TOP] * 3 and all(s["seqs"] == "24" and s["levers"].get("DRAFT_HEAD") == "int8" for s in seen), seen
    assert [s["levers"].get("VLLM_DRAFT_VOCAB_FILE") for s in seen] == [voc["file"], voc["file"], None]
    assert [s["levers"].get("VLLM_DRAFT_VOCAB_SHA256") for s in seen] == [voc["sha256"], voc["sha256"], None]
    assert [" ".join(L.IDX_ARGS) in " ".join(s["extra"]) for s in seen] == [True, False, False]
    return [x["lever"] for x in g["E29_RT"]["dropped"]] == ["idx", "vocab"] and b["status"] == "DONE" and g["E28_I8"]["on"] is True


check("B3 MTP_V64 (idx + vocab on): the E28 top rung (pin 7,933,526,016 B, 24 seqs, I8) with VLLM_DRAFT_VOCAB_FILE / _SHA256 and "
      "the indexer flag; an OOM -> idx dropped (order), an OOM -> vocab dropped -> the same pin plain E28 (DONE)", _b3)


def _b4():
    b, seen, g = ladder([], dict(L.default_cfg(), attempts_supply=True), ["DONE"])
    return (len(seen) == 1 and seen[0]["kv_bytes"] == TOP and seen[0]["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1", "DRAFT_HEAD": "int8"}
            and seen[0]["extra"] == ["--max-num-seqs", "24"] and seen[0]["name"].endswith("_i8") and not g["E29_RT"]["dropped"])


check("B4 DEFAULT (no E29 lever): nb5_boot == E28 K100 (one try: top pin, 24 seqs, I8 + the lever env, no E29 argv / env)", _b4)
check("B5 the E29 edits in place: launch-prep raise wrapper in nb5_boot_try (E29_LAUNCH_PREP_FAILED), the D tries prepended only "
      "with df on and MTP, e29_setup() in CELL 8 after the I8 static check, the report key e29",
      lambda: 'trec.update(status="E29_LAUNCH_PREP_FAILED"' in ENG and "tries = E29L.df_tries(bool(E29_CFG[\"embed_host\"])) + list(tries)" in ENG
      and CODE[6].index("e28_i8_static()") < CODE[6].index("    e29_setup()") and 'E28_OUT["e29"] = dict(E29L.summary(E29_RT)' in CODE[7])

# ------------------------------------------------------------------ R: rev 2 (the DFlash notebook only; E25 NB-S1 GPU facts + DIRECT)
def payload_file(name, sha_):
    """A payload-dataset file by name + sha256: next to this file, E29LT_PAYLOAD_DIR, or the Kaggle mounts (<= 4 levels)."""
    import hashlib
    roots = [HERE] + [Path(x) for x in os.environ.get("E29LT_PAYLOAD_DIR", "").split(os.pathsep) if x] + [Path("/kaggle/input")]
    for d in roots:
        for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
            for f in (sorted(d.glob(pat)) if d.is_dir() else []):
                if hashlib.sha256(f.read_bytes()).hexdigest() == sha_:
                    return f
    return None


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


if REV2:
    FX_P = payload_file("e29_nbs1_log_fixture.json", FACTS["e29"]["nbs1_fixture_sha256"])   # the payload dataset (kit size)
    FX = json.loads(FX_P.read_text(encoding="utf-8"))
    LOG_OK = "\n".join(FX["logs"]["b_t1_embed_host_ran"]["kept"])
    LOG_OOM = "\n".join(FX["logs"]["b_no_embed_host_oom"]["kept"])
    XID_LINE = "NVRM: Xid (PCI:0000:00:04): 79, pid=2451, GPU has fallen off the bus."
    SEAM_P = payload_file(*FACTS["e29"]["seam"])
    DSM = _mod("dflash_seam_e29lt", SEAM_P) if SEAM_P else None
    RCFG = L.check_cfg(FACTS["e29"]["cfg"])
    RUNGS = [t_["kv_bytes"] for t_ in L.df_tries(True)]
    NBX_P = payload_file(*FACTS["e29"]["direct"]["files"]["nbx"])
    NBX = NBX_P.read_text(encoding="utf-8") if NBX_P else ""
    QSA = L.qsa_cfg(ast.literal_eval(next(l for l in C1.splitlines() if l.startswith("E29_QSA = {"))[len("E29_QSA = "):].split("   #")[0]))
    QENV = L.qsa_env(QSA)

    def _ndefs(names):
        tr = ast.parse(NBX)
        fns = [n for n in tr.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert sorted(f.name for f in fns) == sorted(names), [f.name for f in fns]
        return ast.Module(fns, [])

    def _r1():
        assert FACTS["e29"]["seam"] == ["dflash_seam.py", "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b"] == list(L.SEAM_FILE)
        assert L.SEAM_PREV_SHA.startswith("7356d473") and SEAM_P is not None, ("seam file (payload dataset) not found", SEAM_P)
        return (L.GRD_REL in DSM.CORE_RELS and DSM.PATCHES[L.GRD_REL][1] == L.GRD_POST == DSM.GRD_POST
                and "query_cudagraph_manager" in DSM.GRD_PATCH and FACTS["e29"]["grd"] == {"rel": L.GRD_REL, "post": L.GRD_POST})

    check("R1 the seam pin = E27's dflash_seam.py 341d859b (the facts, e29_lib 1.1, the payload-dataset file by sha); its core files "
          "include the GRD edit of vllm/v1/e2_guards.py (post sha d50d754e, query_cudagraph_manager); the old pin 7356d473 recorded", _r1)

    class _DSG:                      # the real seam module's read-back + a scripted tree verify (the GRD file state)
        def __init__(self, grd_state="applied", grd_sha=None):
            self.readback_label, self.grd = DSM.readback_label, {"state": grd_state, "sha": grd_sha or L.GRD_POST}

        def verify_dflash_patch(self, tree, ring=True):
            return {"files": {L.GRD_REL: dict(self.grd)}}

    class _SSEQ:
        def __init__(self, ok=True):
            self.ring, self.ok = True, ok

        def check(self, tag):
            return self.ok, "FULLY_PATCHED" if self.ok else "SEAM_TREE_INVALID_MIXED"

    class _Srv:
        def __init__(self, text, host=True):
            self.text, self.env, self.name, self.port = text, ({"DFLASH_DRAFT_EMBED_HOST": "1"} if host else {}), "srv", 8011

        def log(self):
            return self.text

    def _rb(ds, sseq, srv):
        g = {"FAKE": None, "DS": ds, "SSEQ": sseq, "E29L": L, "E29_RT": {"df": {"mask_row_sha": FX["mask_row_sha"], "tree": "/t"}},
             "e29_qsa_rb": lambda s_, want=None: dict(QRB), "e29_ledger": lambda port: {"stub": True}, "E29_QSA": QSA}
        exec(compile(_defs(["e29_df_readback"]), "<rb>", "exec"), g)
        return g["e29_df_readback"](srv)

    QRB = {"env_ok": True, "label": "QSA_VERIFIED", "workers": 4}

    def _r2():
        QRB.update(env_ok=False, label="QSA_ENV_MISMATCH")
        assert _rb(_DSG(), _SSEQ(), _Srv(LOG_OK))["label"] == "QSA_ENV_MISMATCH"        # rev 2d F3: required before the 1st prefill
        QRB.update(env_ok=True, label="QSA_VERIFIED")
        ok = _rb(_DSG(), _SSEQ(), _Srv(LOG_OK))
        assert ok["label"] == "ENGAGED" and ok["grd"] is True and ok["grd_log_line"] is True and ok["arch_line"] is True, ok
        assert _rb(_DSG("not_applied", "aa25"), _SSEQ(), _Srv(LOG_OK))["label"] == "NOT_INTEGRATED_GRD"
        assert _rb(_DSG(), _SSEQ(False), _Srv(LOG_OK))["label"] == "NOT_INTEGRATED_GRD"
        assert _rb(_DSG(), _SSEQ(), _Srv(LOG_OOM))["label"] == "NOT_INTEGRATED"    # EMBED_HOST requested, the boot had host False
        return _rb(_DSG(), _SSEQ(), _Srv(LOG_OOM, host=False))["label"] == "ENGAGED"

    check("R2 the notebook's e29_df_readback on the REAL NB-S1 boot logs: the EMBED_HOST boot (b_t1) = ENGAGED with the GRD edit "
          "applied (+ the e2-guard query_cudagraph_manager line, recorded); GRD file not applied / SeamSeq not FULLY_PATCHED -> "
          "NOT_INTEGRATED_GRD (a real failure); EMBED_HOST requested on the no-EMBED_HOST boot -> NOT_INTEGRATED", _r2)

    def _r3():
        cr = NB5L.config_readback(LOG_OK, FX["argv_tail_b_t1"], "fp8", True)
        d = L.df_config_check(cr, LOG_OK, 3)
        return (cr["ok"] is False and d["ok"] is True and d["checks"].get("dflash_arch_line") is True
                and d["checks"].get("dflash_spec_method") is True and d["checks"].get("dflash_engine_spec") is True
                and L.df_config_check(cr, LOG_OK, 4)["ok"] is False)

    check("R3 C-11 on the REAL NB-S1 DFlash boot log + its argv: the MTP C-11 is False (spec dflash), the DFlash C-11 "
          "(df_config_check) passes (arch line, method dflash K3 in the log + engine spec); K4 vs a K3 boot = a real mismatch", _r3)

    def _r4():
        assert RCFG["embed_host"] is True and RCFG["drafter"] == "dflash_d3s" and RCFG["indexer_fp8"] is False
        assert all(t_["df"]["embed_host"] for t_ in L.df_tries(True)) and RUNGS[0] == L.D_START_PIN   # rev 2f
        b, seen, g = ladder(["df"], RCFG, [("DONE", "")], i8_on=False, ds=DSM, qsa=QSA)
        s0 = seen[0]
        assert QENV and all(s0["levers"].get(k) == v for k, v in QENV.items()), (s0["levers"], QENV)
        return (s0["df_try"] and s0["kv_bytes"] == L.D_START_PIN and s0["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1"   # rev 2f
                and s0["levers"].get("DFLASH_SEAM") == "1" and s0["levers"].get("DFLASH_DRAFT_SW_BLOCK") == "1024"
                and " ".join(L.IDX_ARGS) not in " ".join(s0["extra"]) and b["status"] == "DONE" and len(seen) == 1)

    check("R4 EMBED_HOST from the FIRST D try: the variant cfg (embed_host on, indexer fp8 off); the notebook's nb5_boot + e29_env with "
          "the REAL dflash_seam.server_env: try 1 = pin 7,665,090,560 B (rev 2f D_START_PIN) with DFLASH_DRAFT_EMBED_HOST=1, the seam env, no indexer flag", _r4)

    def _r5():
        ok = True
        for sc in ([("NO_GO_SHORT_CORRECTNESS_GATE_FAILED", LOG_OOM), ("E29_DFLASH_WARMUP_FAILED", LOG_OOM), ("DONE", "")],
                   [("E29_DFLASH_DIED_BEFORE_PHASES", LOG_OOM), ("E29_DFLASH_WARMUP_FAILED", LOG_OOM), ("DONE", "")]):   # rev 2f: 3 rungs
            b, seen, g = ladder(["df"], RCFG, sc, i8_on=False, ds=DSM)
            ok = ok and ([s["kv_bytes"] for s in seen] == RUNGS and all(s["df_try"] for s in seen) and not g["E29_RT"]["dropped"]
                         and b["status"] == "DONE" and all(s["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" for s in seen))
        return ok and RUNGS == [7665090560, 7396655104, 6859784192]

    check("R5 an OOM AFTER ready (the first request / probes, the warm-up, before the phases; the REAL NB-S1 OOM log) steps the D "
          "ladder (rev 2f: from D_START_PIN): 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192 B, EMBED_HOST on every rung, DFlash kept", _r5)

    def _r6():
        b, seen, g = ladder(["df"], RCFG, [("NO_GO_SHORT_CORRECTNESS_GATE_FAILED", LOG_OOM + "\n" + XID_LINE), "DONE"], i8_on=False, ds=DSM)
        dr = g["E29_RT"]["dropped"]
        return ([s["kv_bytes"] for s in seen] == [L.D_START_PIN, FACTS["pins"][0]] and seen[0]["df_try"] and not seen[1]["df_try"]   # rev 2f
                and [x["lever"] for x in dr] == ["df"] and "oom False" in dr[0]["why"] and seen[1]["levers"].get("DRAFT_HEAD") == "int8")

    check("R6 a device fault (XID) with the OOM is NO retry on the D ladder: DFlash dropped at once -> the proven MTP K3 top rung "
          "(I8 restored)", _r6)

    def _r7():
        b, seen, g = ladder(["df"], RCFG, [("SERVER_DIED", LOG_OOM)] * len(RUNGS) + ["DONE"], i8_on=False, ds=DSM, qsa=QSA)   # rev 2f: 3 rungs
        m = seen[-1]
        assert all(k not in m["levers"] for k in L.QSA_KEYS) and m["seqs"] == "24", m
        return ([s["kv_bytes"] for s in seen] == RUNGS + [FACTS["pins"][0]] and not m["df_try"] and m["levers"].get("DRAFT_HEAD") == "int8"
                and not any(k.startswith("DFLASH_") for k in m["levers"]) and "dflash" not in " ".join(m["extra"])
                and [x["lever"] for x in g["E29_RT"]["dropped"]] == ["df"] and b["status"] == "DONE" and g["E28_I8"]["on"] is True)

    check("R7 the D ladder end (an OOM on all 4 rungs) -> the existing drop rule -> the proven MTP K3 config at the top rung (I8 "
          "restored, no seam env, no dflash argv) -> DONE", _r7)

    def _r8():
        return (L.lever_to_drop(["df", "idx"], LOG_OOM, "SERVER_DIED") == "df" and L.df_oom_step("NO_GO_X", LOG_OOM)
                and not L.df_oom_step("NO_GO_X", LOG_OOM + "\n" + XID_LINE) and not L.df_oom_step("E29_DFLASH_NOT_ENGAGED", LOG_OOM)
                and not L.df_oom_step("NO_GO_X", LOG_OK) and "indexer" in LOG_OOM.lower())

    check("R8 1.1 lever naming on the REAL OOM log (its traceback names qsa_indexer.py): df, never idx (1.0 named idx in every "
          "Merlin log); df_oom_step = OOM line and not XID and not a DFlash gate", _r8)

    def _r9():
        W = lambda rows: {"rows": rows}
        assert L.df_warmup_failure(W([{"tokens": [256], "errors": []}]), False) == "SERVER_DIED"
        assert L.df_warmup_failure(W([{"tokens": [3, 256], "errors": ["HTTPError 500"]}]), True).startswith("REQUEST_ERRORS")
        assert L.df_warmup_failure(W([{"tokens": [256, 0], "errors": []}]), True) == "EMPTY_STREAM"
        assert L.df_warmup_failure(W([{"tokens": [1, 2], "errors": []}, {"tokens": [384] * 24, "errors": []}]), True) is None
        c = lambda d, t_, a: {"drafts": d, "draft_tokens": t_, "accepted": a}
        assert L.canary(c(0, 0, 0), c(300, 900, 90), 3)["hit"] is False      # AL 1.3 (low) = not gated
        assert L.canary(c(0, 0, 0), c(300, 900, 10), 3)["hit"] is True       # 1.1 % accepted = the structural zero-accept rule
        h = L.hourly_al([{"t": 0, "sd_drafts": 0, "sd_draft_tok": 0, "sd_acc": 0}, {"t": 3500, "sd_drafts": 100, "sd_draft_tok": 300, "sd_acc": 200},
                         {"t": 7000, "sd_drafts": 200, "sd_draft_tok": 600, "sd_acc": 210}])
        assert h["flag"] == "AL_DROP_BELOW_0.7_FIRST_HOUR"                    # flagged ...
        eng = ENG + "\n".join(CODE[i] for i in (0, 1, 2, 3, 5, 6))
        assert "hourly_al(" not in eng and "AL_DROP" not in eng and "LOW_VARIETY" not in eng   # ... and read only in the report
        assert CODE[7].count("E29L.hourly_al(") == 1
        for s_ in ('"E29_DFLASH_WARMUP_FAILED"', '"E29_DFLASH_DIED_BEFORE_PHASES"', '"NOT_INTEGRATED_GRD"', "E29L.df_oom_step(_e29s, _e28_log)",
                   'if trec["e29_canary"]["hit"] and not FAKE:', "        e29_relaunch_plan(old)   #", '"SEAM_GRD_NOT_APPLIED: "'):
            assert ENG.count(s_) == 1, s_
        return 'if not c.get("hit") or FAKE:' in ENG

    check("R9 every DFlash gate acts on a real failure only: the warm-up gate (dead / request errors / an empty stream; slow = fine), "
          "the zero-accept canary (< 2 % accepted; a low AL is not gated), hourly AL flagged but read only by the report, no label "
          "gate in the boot / run cells; the gates in place once each (warm-up, pre-phase death, GRD, OOM step, canary, in-run, relaunch)", _r9)

    def _r10():
        T = L.df_tries(True)
        assert L.df_relaunch_plan(T[0]["df"], LOG_OOM, T)["kv_bytes"] == RUNGS[1]
        assert L.df_relaunch_plan(T[-1]["df"], LOG_OOM, T) == "mtp"
        assert L.df_relaunch_plan(T[0]["df"], LOG_OOM + "\n" + XID_LINE, T) == "mtp" and L.df_relaunch_plan(T[0]["df"], "died", T) is None   # rev 2d F10
        drops = []
        g = {"E29_RT": {"df_live": True, "on": ["df"], "df_cur": T[0]["df"], "df": {"tail": TAIL, "mask_row_sha": "ab" * 32, "sw_block": 1024},
                        "errors": []},
             "NB5_ACTIVE_BOOT": {"name": "b", "kv_bytes": RUNGS[0], "extra": ["--max-num-seqs", "24"] + TAIL,
                                 "levers": dict(DSM.server_env(mask_row_sha="ab" * 32, embed_host=True, draft_sw_block=1024), MERLIN_SHARE_DRAFT_WORKSPACE="1")},
             "FAKE": None, "E29L": L, "E29_CFG": RCFG, "DS": DSM, "E28_I8": {"on": False}, "json": json, "log": lambda *a: None,
             "E29_QSA": QSA, "E28_I8_ENV": {"DRAFT_HEAD": "int8"}, "K100_PIN_STEPS_GIB": STEPS,
             "e29_drop": lambda lv, why, nm: (drops.append((lv, why)), g["E28_I8"].update(on=True))}
        g["NB5_ACTIVE_BOOT"]["levers"].update(QENV)
        exec(compile(_defs(["e29_relaunch_plan", "e29_env", "e29_mtp_boot"]), "<rp>", "exec"), g)

        class _Old:
            def log(self):
                return LOG_OOM
        g["e29_relaunch_plan"](_Old())
        ab = g["NB5_ACTIVE_BOOT"]
        assert ab["kv_bytes"] == RUNGS[1] and ab["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and ab["extra"][-4:] == TAIL and not drops
        assert all(ab["levers"].get(k) == v for k, v in QENV.items())
        g["E29_RT"]["df_cur"] = T[-1]["df"]
        g["e29_relaunch_plan"](_Old())
        return (ab["kv_bytes"] == L.D_PIN0 and ab["extra"] == ["--max-num-seqs", "24"] and not any(k.startswith("DFLASH_") for k in ab["levers"])
                and ab["levers"].get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1" and [d[0] for d in drops] == ["df"] and g["E28_I8"]["on"] is True
                and ab["levers"].get("DRAFT_HEAD") == "int8" and ab["pin_gib"] == STEPS[0] and not any(k in ab["levers"] for k in L.QSA_KEYS))

    check("R10 an in-run OOM death of the DFlash server (the E15B watchdog relaunch, e29_relaunch_plan): the next D rung "
          "(EMBED_HOST kept, same argv); no rung left -> the proven MTP K3 config at the top rung (dflash argv / seam env removed, "
          "DFlash dropped, BF16 head); a device fault / any other death = E15B's relaunch of the same config", _r10)

    def _r11():
        first = next(l for l in C1.splitlines() if l.strip() and not l.lstrip().startswith("#"))
        assert first.startswith("E29_DIRECT2 = {") and C1.splitlines()[0] == first
        lit = ast.literal_eval(first[len("E29_DIRECT2 = "):first.index("}") + 1])
        assert lit == {"on": True, "think": 104000, "answer": 8192, "temperature": 1.0, "drain_min": 26, "phase": "C1"}, lit   # rev 2f
        assert L.direct2_cfg(dict(lit, phase="C"))["phase"] == "C" and L.direct2_cfg(dict(lit, phase="C2")).get("invalid")
        assert L.direct2_cfg({k: v for k, v in lit.items() if k != "phase"})["phase"] == "C1"      # rev 2f: absent = C1
        assert L.direct2_cfg(lit)["on"] is True and L.direct2_cfg(dict(lit, on=False))["on"] is False
        assert L.direct2_cfg(dict(lit, think=50000)).get("invalid") and L.direct2_cfg(dict(lit, x=1)).get("invalid")
        assert L.direct2_cfg(dict(lit, think="104000")).get("invalid") and L.direct2_cfg(lit, '{"think": 6000}')["think"] == 6000
        return "E29_DIRECT2 = E29L.direct2_cfg(E29_DIRECT2," in C1 and "drain_min" in lit

    check("R11 the DIRECT switch: the FIRST line of CELL 1 is the plain literal E29_DIRECT2 = {'on': True, 'think': 104000, "
          "'answer': 8192, 'temperature': 1.0, 'drain_min': 26, 'phase': 'C1'} (outside any packed / sha-locked text); direct2_cfg: a bad value turns "
          "DIRECT off (never stops the run), the 60k floor on the switch, a kit override only in the harness", _r11)

    def _r12():
        out = {}
        for on, sched in ((False, {"ok": True}), (True, {"ok": False, "error": "x"})):
            os.environ["E29_DIRECT"] = "1"
            os.environ["E29_DIRECT_PHASE"] = "C"   # rev 2f: a stale scope must not survive an OFF path

            class _E12:
                E29D_RUN = E29D_CAN_START = None
            g = {"E29_DIRECT2": {"on": on}, "E29_RT": {}, "E29D_SCHED_REC": sched, "E12A": _E12, "os": os, "json": json,
                 "log": lambda *a: None, "E29_DIRECT_RT": {"cfg": None, "state": None, "hook": False, "errors": []}, "E29D": None,
                 "TMP": Path("."), "E29D_FILES": {}, "q27x_payload": lambda *a: (_ for _ in ()).throw(AssertionError("no payload read"))}
            exec(compile(_ndefs(["e29_direct_setup"]), "<ds>", "exec"), g)
            g["e29_direct_setup"]()
            out[on] = (g["E29_DIRECT_RT"]["state"], _E12.E29D_RUN, g["E29_RT"].get("direct") is g["E29_DIRECT_RT"], os.environ.get("E29_DIRECT"))
            assert os.environ.get("E29_DIRECT_PHASE") is None, (on, os.environ.get("E29_DIRECT_PHASE"))   # rev 2f
        os.environ.pop("E29_DIRECT", None)
        os.environ.pop("E29_DIRECT_PHASE", None)
        return (out[False][0] == "OFF_SWITCH" and out[False][1] is None and out[False][2] and out[True][0] == "OFF_SETUP"
                and out[True][1] is None and out[True][3] is None)

    check("R12 DIRECT off: 'on' False -> OFF_SWITCH (no hook, nothing loaded); switch on but CELL 3's DIRECT diff not applied -> "
          "OFF_SETUP, the E29_DIRECT env removed, no hook (phase C = the E28 Duck attempt 2 = today's path)", _r12)

    def _r13():
        files = FACTS["e29"]["direct"]["files"]
        line = next(l for l in C1.splitlines() if l.startswith("E29D_FILES = "))
        assert json.loads(line[len("E29D_FILES = "):line.index("   # ")]) == files
        pd, ps = payload_file(*files["direct"]), payload_file(*files["sched"])
        assert pd is not None and ps is not None and NBX_P is not None, ("DIRECT / nbx files (payload dataset) not found", files)
        compile(NBX, "e29_nbx.py", "exec")
        d = (pd.read_text(encoding="utf-8"))
        assert d.count("# E29") >= 4 and 'LIB_VERSION = "e29_direct 1.0 (2026-11-02) = e31_direct 1.0 (9967a6e0)' in d
        s = ps.read_text(encoding="utf-8")
        return (f'PRE = "{FACTS["e29"]["sched_sha"]["post"]}"' in s and f'POST = "{FACTS["e29"]["direct"]["duck_post_sha256"]}"' in s
                and FACTS["e29"]["direct"]["e31_direct_sha256"].startswith("9967a6e0"))

    check("R13 the DIRECT files: CELL 1's E29D_FILES == the facts; e29_direct.py (= e31_direct 9967a6e0 + the marked E29 edits) and "
          "e29d_sched_patch.py (pre = the E29 scheduler post sha, post = the facts' DIRECT post) found in the payload dataset by sha", _r13)

    def _r14():
        c3 = CODE[2]
        i, j = c3.index('CONFIG["e28_sched"] = E28_SCHED_REC\n'), c3.index("E29D_SCHED_REC = {")
        k = c3.index("import arc2_duck as E12A")
        s = payload_file(*FACTS["e29"]["direct"]["files"]["sched"]).read_text(encoding="utf-8")
        cs = s[s.index("def cell3("):]
        return (i < j < k and c3.count('os.environ.pop("E29_DIRECT", None)') == 1 and "E29DS.cell3(E12_DIR, E28_SCHED_REC, E29D_SCHED_REC)" in c3
                and 'os.environ["E29_DIRECT"]' not in c3 and cs.index("verify_direct(e12_dir)") < cs.index("os.environ[ENV_DIRECT] = \"1\"")
                and 'sched_rec.get("path") == "E29"' in cs and "revert_direct(e12_dir)" in cs)

    # ---------------------------------------------------------------- rev 2b: LIVE lines (print only)
    def _lv_run(sol):
        tasks = {"t1": {"test": [{"input": [[0]]}]}, "t2": {"test": [{"input": [[0]]}, {"input": [[1]]}]}}
        ck = [0.0]
        _ns = {}
        exec(compile(_ndefs(["LiveLog"]), "<lv>", "exec"), _ns)
        lv = _ns["LiveLog"](tasks, sol, lambda tid, task, rs: [{"attempt_1": (r_["submissions"][0]["test_outputs"][qi]), "attempt_2": [[9]]}
                                                         for qi in range(len(task["test"])) for r_ in [sorted(rs, key=lambda x: x["branch"])[0]]],
                       lambda rec, qi: {"grid": rec["submissions"][0]["test_outputs"][qi]}, t0=0.0, clock=lambda: ck[0])
        A1 = {"task_id": "t1", "branch": 0, "accepted": True, "e28": {"phase": "A", "best_train": {"k": 3, "n": 3}},
              "submissions": [{"test_outputs": [[[5]]], "accepted": True, "train_exact": [True] * 3}]}
        A2 = {"task_id": "t2", "branch": 0, "accepted": False, "e28": {"phase": "A", "best_train": {"k": 1, "n": 2}},
              "submissions": [{"test_outputs": [[[7]], [[8]]], "accepted": False, "train_exact": [True, False]}]}
        D2 = {"task_id": "t2", "branch": 1, "accepted": False, "e29_direct": True, "generated_tokens": 58210,
              "e31": {"mode": "natural"}, "submissions": [{"test_outputs": [[[7]], None], "accepted": False, "train_exact": []}]}
        out = []
        for m_, r_ in ((182.4, A1), (190.0, A2), (301.0, D2)):
            ck[0] = m_ * 60
            out.append(lv.on_rec(r_))
        out.append(lv.summary(612.3, 24))
        return out

    def _r15():
        g = _lv_run({"t1": [[[5]]], "t2": [[[7]], [[9]]]})
        assert g[0] == ("[LIVE 182.4 min] t1 A  accepted(train 3/3) exact 1/1 | A 1/2 done, accepted 1, exact 1 | "
                        "queries 1/3 correct, task-norm 100.0 % (1 tasks)"), g[0]
        assert g[1].startswith("[LIVE 190.0 min] t2 A  not accepted(train 1/2) exact 1/2 | A 2/2 done, accepted 1, exact 1 | queries 3/3"), g[1]
        assert g[2].startswith("[LIVE 301.0 min] t2 C-DIRECT  natural 58,210 tok  grid 1/2 exact 1/2 | A 2/2 done") and g[2].endswith("attempt_2 hits 1"), g[2]
        assert "C-DIRECT 1 done, accepted 0, exact 0" in g[2] and g[3].startswith("[LIVE 15-min 301.0 min] tok/s 612.3 | running 24 | A 2/2"), g[3]
        h = _lv_run(None)                                                   # the hidden rerun: no gold fields
        assert all("exact" not in x and "queries" not in x and "hits" not in x for x in h), h
        return h[0] == "[LIVE 182.4 min] t1 A  accepted(train 3/3) | A 1/2 done, accepted 1" and "natural 58,210 tok  grid 1/2 |" in h[2]

    check("R15 LIVE lines (e29_lib.LiveLog, print only): per finished attempt (minutes, task, phase, train k/n, exact vs gold, "
          "per-phase totals, queries correct + task-norm from the CURRENT selection; DIRECT: mode, tokens, grids, attempt_2 hits) "
          "and the 15-min line; with solutions = gold fields, without (hidden rerun) = none; totals exact on a fake run", _r15)

    def _r16():
        prints, cbs = [], []

        class _E12:
            def solve_all_r6(tasks, order, *, on_branch_done=None, **k):
                for r_ in ({"task_id": "t1", "branch": 0, "accepted": True, "submissions": []}, {"task_id": "zz", "branch": 0}):
                    on_branch_done(r_)
                return {"ret": 1}

            def build_submission(tasks, recs):
                raise RuntimeError("boom")                                 # LIVE's selection fails: never propagates
            _e28_cand = None
        import builtins
        g = {"E12A": _E12, "E29L": L, "E29_RT": {}, "time": time, "json": json, "NOTEBOOK_T0": 0.0, "threading": __import__("threading"), "time": time,
             "SOLUTIONS": {"t1": [[[1]]], "zz": [[[1]]]}, "e21_prom": lambda: [], "print": lambda *a, **k: prints.append(a)}
        exec(compile(_ndefs(["e29_live_setup", "LiveLog"]), "<live>", "exec"), g)
        g["e29_live_setup"]()
        g["e29_live_setup"]()                                                # once (never wraps twice)
        ret = _E12.solve_all_r6({"t1": {"test": [{"input": [[0]]}]}}, ["t1"], on_branch_done=cbs.append, out_dir="x")
        return (ret == {"ret": 1} and [c["task_id"] for c in cbs] == ["t1", "zz"] and g["E29_RT"]["live"]["errors"] == 2
                and _E12._e29_live_orig is not None and [a[0] for a in prints] == ["[LIVE start] algo"])

    check("R16 an exception inside LIVE never propagates: the original on_branch_done gets every record first (same order, same "
          "objects), the solver's return value is unchanged, LIVE errors are counted, the wrapper installs once", _r16)

    def _r17():
        blk = NBX[NBX.index("def e29_live_setup():"):NBX.index("E12A.solve_all_r6 = live_solve")]
        assert 'exec(compile(q27x_payload(*E29D_FILES["nbx"]).decode("utf-8"), "e29_nbx.py", "exec"), globals())' in ENG
        return ('LiveLog(tasks, globals().get("SOLUTIONS")' in blk and "E12A.build_submission({t_: task_}, {t_: rs_})[t_]" in blk
                and blk.index("on_branch_done(rec)") < blk.index("lv.on_rec(rec)") and "stop.wait(900)" in blk
                and ENG.count("    e29_extras()   # E29 rev 2c") == 1 and "records" not in blk.replace("E29_RT", ""))

    check("R17 the LIVE glue: SOLUTIONS only when loaded, the selection = the notebook's build_submission rule, the original "
          "callback first, a 15-min summary thread, installed from e29_setup", _r17)

    def _r18():
        assert abs(L.gib_of(L.D_PIN0) - STEPS[0]) < 1e-9 and FACTS["pins"][0] == L.D_PIN0
        sc = [("SERVER_DIED", LOG_OOM)] * 6 + ["DONE"]
        b, seen, g = ladder(["df"], RCFG, sc, i8_on=False, ds=DSM, start_gib=0.0, qsa=QSA)   # E28's own start = the RC2 pin
        m = [s for s in seen if not s["df_try"]]
        assert m[0]["kv_bytes"] == L.D_PIN0 and m[0]["seqs"] == "24" and m[0]["levers"].get("DRAFT_HEAD") == "int8", m[0]
        assert m[1]["kv_bytes"] == L.D_PIN0 and "DRAFT_HEAD" not in m[1]["levers"], m          # E28's own I8 rule (same pin, BF16)
        assert m[2]["kv_bytes"] == L.FP8_PIN and b["status"] == "DONE", m                     # then E28's own ladder (its start)
        b2, seen2, _ = ladder(["df"], RCFG, [("NO_GO_SHORT_CORRECTNESS_GATE_FAILED", LOG_OOM + "\n" + XID_LINE), "DONE"], i8_on=False, ds=DSM,
                              start_gib=0.0)
        m2 = [s for s in seen2 if not s["df_try"]]
        return (m2[0]["kv_bytes"] == L.D_PIN0 and m2[0]["seqs"] == "24" and m2[0]["levers"].get("DRAFT_HEAD") == "int8"
                and L.mtp_top([{"gib": STEPS[0], "kv_bytes": L.D_PIN0}], 0, STEPS[0]) == 0)

    check("R18 (rev 2c, the real-kit blocker) after DFlash is dropped (ladder end / XID) the NEXT try is the E28 production top "
          "rung: MTP K3, 7,933,526,016 B, --max-num-seqs 24, the INT8 head restored; only then E28's own ladder (here from its "
          "fail-safe start, the RC2 pin), also when E28's list does not start at the top", _r18)

    def _r19():
        assert C1.splitlines()[1].startswith("E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}") and QSA == {"max_logits_mb": 128, "granule_mb": 64}
        assert QENV == {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "128", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "64"}
        assert L.qsa_cfg({"max_logits_mb": 0, "granule_mb": 64}).get("invalid") and L.qsa_env(L.qsa_cfg({"x": 1})) == {}
        # Merlin qsa_indexer.py:510 _prefill_backing_numel (+ :776 rows_per_chunk): the backing stays <= the cap
        cap = 128 * 2 ** 20
        for width in (64, 512, 2048, 8192, 32768):
            rpc_ = max(1, cap // (width * 4))
            for rows in (1, 7, 1024, 4096, 131072):
                need = min(rows, rpc_) * width
                gran = 64 * 2 ** 20 // 4
                numel = max(need, min(-(-need // gran) * gran, cap // 4))
                assert numel * 4 <= cap, (width, rows, numel * 4)
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="e29proc_"))
        tree = {100: (1, "bash"), 101: (100, "vllm"), 102: (101, "VLLM::Worker_PP0"), 103: (101, "VLLM::Worker_PP1"), 104: (101, "python"),
                105: (101, "VLLM::Worker_PP0"), 106: (101, "VLLM::Worker_PP1"), 200: (1, "VLLM::Worker_PP0")}
        for pid, (pp, comm) in tree.items():
            (d / str(pid)).mkdir()
            (d / str(pid) / "stat").write_text(f"{pid} ({comm}) S {pp} 0 0", encoding="utf-8")
            (d / str(pid) / "comm").write_text(comm + "\n", encoding="utf-8")
            env = {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "128", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "64" if pid != 106 else "256"}
            (d / str(pid) / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()))
        w = L.proc_env_readback(100, L.QSA_KEYS, proc=str(d))
        assert [x["pid"] for x in w] == [102, 103, 105, 106], w                               # the 4 workers under the server only
        r = L.qsa_readback(w, QENV, "INFO qsa_indexer.py:797] QSA prefill logits backing: granule 64 MiB (VLLM_E2_QSA_LOGITS_GRANULE_MB)")
        assert r["env_ok"] is False and r["granule_logged_mib"] == [64] and r["backing_bound_mib"] == 128   # 106 kept Merlin's 256
        (d / "106" / "environ").write_bytes(b"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128\0VLLM_E2_QSA_LOGITS_GRANULE_MB=64")
        assert L.qsa_readback(L.proc_env_readback(100, L.QSA_KEYS, proc=str(d)), QENV)["env_ok"] is True
        return (ENG.count('        env.update({} if E29_RT.get("qsa_off") else E29L.qsa_env(E29_QSA))   # E29 rev 2c / 2d') == 1 and "e29_qsa_rb(s" in ENG
                and 'if k_ not in DS.ALL_ENVS + E29L.QSA_KEYS}' in ENG)

    check("R19 (rev 2c) the QSA switch (CELL 1 line 2: max_logits 128 MiB, granule 64 MiB) -> the env of the DFlash tries only "
          "(after Merlin's build(): NB5_BOOT_LEVERS), stripped for the MTP fallback / relaunch; the read-back of the 4 workers' "
          "spawn env (/proc environ under the server pid) + the logged granule (recorded); Merlin's backing formula <= 128 MiB", _r19)
    # ---------------------------------------------------------------- rev 2d (the rev 2c audit fixes)
    class _Srv2:
        def __init__(self, text="", alive=True):
            self.text, self.alive_, self.port, self.name, self.argv, self.env = text, alive, 8011, "srv", ["--x"], {}

        def log(self):
            return self.text

        def alive(self):
            return self.alive_

        def kill(self):
            self.alive_ = False

    def _probe(log_text="", free=(900, 900, 700, 650), alive=True, tokens=8, rpc_res=True):
        sent = {}

        class _St:
            def __init__(self, port, ids, n, temperature=1.0, seed=0):
                sent["ids"], self.tokens, self.err = ids, [1] * tokens, None

            def start(self):
                pass

            def join(self, t_=None):
                pass

            def is_alive(self):
                return False
        h_, tl_ = 61, 7
        p40 = [[1] * h_ + [1000 + v] * (40000 - h_ - tl_) + [2] * tl_ for v in range(8)]
        g = {"E29L": L, "R": {"prompt_render": {"head_tokens": h_, "tail_tokens": tl_}}, "PROMPTS": {"40k": p40}, "time": time,
             "L": type("L", (), {"Stream": _St}), "left": lambda: 1e9, "log": lambda *a: None, "json": json,
             "rpc": lambda port, m, *a: ({"results": [{"mem": {"free_mib": f}} for f in free]} if rpc_res else {"error": "no rpc"}),
             "e29_ledger": lambda port: {"host": {"cgroup_cur_mib": 1}}}
        exec(compile(_defs(["e29_prefill_probe"]), "<pp>", "exec"), g)
        trec = {}
        return g["e29_prefill_probe"](_Srv2(log_text, alive), trec), trec, sent

    def _r20():
        r, tr, sent = _probe()
        assert r is None and len(sent["ids"]) == L.PROBE_TOKENS == 130000 and tr["e29_probe"]["workers_free_mib"] == [900, 900, 700, 650]
        assert sent["ids"][:61] == [1] * 61 and sent["ids"][-7:] == [2] * 7 and tr["e29_probe"]["ledger"] == {"host": {"cgroup_cur_mib": 1}}
        assert _probe(log_text=LOG_OOM)[0] == "E29_DFLASH_PROBE_FAILED" and _probe(alive=False)[0] == "E29_DFLASH_PROBE_FAILED"
        assert _probe(tokens=0)[0] == "E29_DFLASH_PROBE_FAILED" and _probe(free=(900, 900, 400, 900))[0] == "E29_DFLASH_LOW_FREE"
        assert _probe(rpc_res=False)[0] is None                                              # no worker RPC = recorded, not gated
        b, seen, g = ladder(["df"], RCFG, [("E29_DFLASH_LOW_FREE", ""), ("E29_DFLASH_PROBE_FAILED", LOG_OOM), ("DONE", "")], i8_on=False, ds=DSM, qsa=QSA)
        assert [s["kv_bytes"] for s in seen] == RUNGS[:3] and all(s["df_try"] for s in seen) and not g["E29_RT"]["dropped"], seen
        b, seen, g = ladder(["df"], RCFG, [("E29_DFLASH_PROBE_FAILED", "request error 500"), "DONE"], i8_on=False, ds=DSM, start_gib=0.0)
        return ([s["kv_bytes"] for s in seen] == [L.D_START_PIN, L.D_PIN0] and not seen[1]["df_try"] and seen[1]["seqs"] == "24"   # rev 2f
                and ENG.count("_e29p = None if FAKE else e29_prefill_probe(s, trec)") == 1
                and ENG.index("e29_prefill_probe(s, trec)") < ENG.index('        trec["phases_started"] = True'))

    check("R20 (rev 2d F2) the forced max first prefill on every DFlash try before the phases (130,000 tokens, the batched-token "
          "chunks) + every worker's own free memory (nb1_cache_info RPC) + the ledger: OOM / dead / error / no token -> "
          "E29_DFLASH_PROBE_FAILED (an OOM steps the D ladder; another error drops DFlash -> the E28 top rung), a worker < 512 MiB -> "
          "E29_DFLASH_LOW_FREE = one ladder step; no RPC = recorded only", _r20)

    def _r21():
        b, seen, g = ladder(["df"], RCFG, [("E29_QSA_ENV_MISMATCH", ""), ("DONE", "")], i8_on=False, ds=DSM, qsa=QSA)
        s0, s1 = seen
        assert s0["kv_bytes"] == s1["kv_bytes"] == L.D_START_PIN and s0["df_try"] and s1["df_try"] and not g["E29_RT"]["dropped"]   # rev 2f
        assert all(s0["levers"].get(k) == v for k, v in QENV.items()) and not any(k in s1["levers"] for k in L.QSA_KEYS)
        assert s1["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and g["E29_RT"]["qsa_off"] is True
        w4 = [{"env": dict(QENV)}] * 4
        assert L.qsa_readback(w4, QENV)["label"] == "QSA_VERIFIED" and L.qsa_readback(w4[:3], QENV)["label"] == "UNVERIFIED_QSA"
        assert L.qsa_readback(w4[:3] + [{"env": {L.QSA_KEYS[0]: "256", L.QSA_KEYS[1]: "256"}}], QENV)["label"] == "QSA_ENV_MISMATCH"
        return ('trec["status"] = "E29_QSA_ENV_MISMATCH" if trec["e29_df_readback"]["label"] == "QSA_ENV_MISMATCH"' in ENG
                and 'r_ = rpc(s.port, "nb4s_env", ",".join(E29L.QSA_KEYS))' in ENG and ENG.index("q_ = e29_qsa_rb(s, want)") < ENG.index("ok, pr = nb5_probes(s)"))

    check("R21 (rev 2d F3 / F14) the 4-rank QSA env read-back is REQUIRED before the first prefill (/proc environ, the worker RPC "
          "when /proc gives < 4): a mismatch / a missing rank -> the SAME rung relaunched WITHOUT the QSA env (Merlin 256/256 + "
          "EMBED_HOST), DFlash kept, recorded; labels QSA_VERIFIED / UNVERIFIED_QSA / QSA_ENV_MISMATCH", _r21)

    def _r22():
        T = L.df_tries(True)
        assert L.df_relaunch_plan(T[0]["df"], XID_LINE, T) == "mtp" and L.df_relaunch_plan(T[0]["df"], LOG_OOM + "\n" + XID_LINE, T) == "mtp"
        assert L.df_relaunch_plan(T[0]["df"], LOG_OOM, T)["kv_bytes"] == RUNGS[1] and L.df_relaunch_plan(T[-1]["df"], LOG_OOM, T) == "mtp"
        assert L.df_relaunch_plan(T[0]["df"], "Engine core died (no OOM)", T) is None
        drops = []
        g = {"E29_RT": {"df_live": True, "on": ["df"], "df_cur": T[0]["df"], "df": {"tail": TAIL, "mask_row_sha": "ab" * 32, "sw_block": 1024},
                        "errors": []},
             "NB5_ACTIVE_BOOT": {"name": "b", "kv_bytes": RUNGS[0], "extra": ["--max-num-seqs", "24"] + TAIL, "levers": dict(QENV, DFLASH_SEAM="1")},
             "FAKE": None, "E29L": L, "E29_CFG": RCFG, "DS": DSM, "E28_I8": {"on": False}, "json": json, "log": lambda *a: None, "time": time,
             "NOTEBOOK_T0": 0.0, "E29_QSA": QSA, "E28_I8_ENV": {"DRAFT_HEAD": "int8"}, "K100_PIN_STEPS_GIB": STEPS,
             "e29_drop": lambda lv, why, nm: (drops.append((lv, why)), g["E28_I8"].update(on=True))}
        exec(compile(_defs(["e29_relaunch_plan", "e29_env", "e29_mtp_boot"]), "<rp>", "exec"), g)
        g["e29_relaunch_plan"](_Srv2(LOG_OOM + "\n" + XID_LINE))
        ab = g["NB5_ACTIVE_BOOT"]
        return (drops == [("df", "IN_RUN GPU_DEVICE_FAULT")] and ab["kv_bytes"] == L.D_PIN0 and ab["levers"] == {"DRAFT_HEAD": "int8"}
                and ab["extra"] == ["--max-num-seqs", "24"] and g["E29_RT"]["gpu_device_fault"][0]["lines"] and g["E29_RT"]["mtp_relaunch"] is True)

    check("R22 (rev 2d F10) in-run death: a device fault FIRST -> MTP (XID alone and XID + OOM; recorded GPU_DEVICE_FAULT, the "
          "E28 top rung / 24 / I8 / no DFlash or QSA env, the F4 gates armed); plain OOM -> the next rung; last rung -> MTP; a "
          "healthy non-OOM death -> the same config", _r22)

    def _r23():
        out = {}
        for case in ("ok", "i8", "c11"):
            g = {"E29_RT": {"mtp_relaunch": True, "errors": []}, "NB5_ACTIVE_BOOT": {"levers": {"DRAFT_HEAD": "int8", "X": "1"}, "kv": "fp8",
                                                                                        "kv_bytes": L.D_PIN0},
                 "NB5_BOOT_KV": "fp8", "E28_I8": {"on": True}, "E28_I8_ENV": {"DRAFT_HEAD": "int8"},
                 "NB5L": type("N", (), {"config_readback": staticmethod(lambda lg, av, kv, m: {"ok": case != "c11"})}),
                 "k100_boot_gates": lambda s_, m: {"ok": True, "fatal": None},
                 "E28L": type("E", (), {"i8_readback": staticmethod(lambda lg, av, h: {"label": "I8_ENGAGED" if case == "ok" else "BF16_HEAD"})})}
            exec(compile(_defs(["e29_relaunch_gates"]), "<rg>", "exec"), g)
            s_ = _Srv2()
            ok, why = g["e29_relaunch_gates"](s_, "ready")
            out[case] = (ok, s_.alive_, g["NB5_ACTIVE_BOOT"]["levers"], g["E28_I8"], g["E29_RT"]["mtp_relaunch"])
        assert out["ok"][0] is True and out["ok"][1] is True and out["ok"][4] is False
        assert out["i8"][0] is False and out["i8"][1] is False and out["i8"][2] == {"X": "1"} and out["i8"][3]["on"] is False and out["i8"][4] is True
        assert out["c11"][0] is False and out["c11"][1] is False and out["c11"][2] == {"DRAFT_HEAD": "int8", "X": "1"}
        return ("        if ok and E29_RT.get(\"mtp_relaunch\"):   # E29 rev 2d F4\n            ok, why = e29_relaunch_gates(srv, why)\n" in ENG
                and "NB5_SERVERS[NB5_SERVERS.index(srv)] = old" in ENG)

    check("R23 (rev 2d F4) the in-run DFlash -> MTP relaunch runs the E28 boot gates on the new server: C-11 + method map + the I8 "
          "read-back; INT8 not engaged -> E28's own I8 rule (the same pin, BF16 head: the watchdog's retry boots it); C-11 / method "
          "map failed -> this relaunch failed (server killed, retried once)", _r23)

    def _r24():
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="e29seam_"))
        rels = DSM._rels(True, True)
        for r_ in rels:
            (d / r_).parent.mkdir(parents=True, exist_ok=True)
            (d / r_).write_text("pristine " + r_, encoding="utf-8")
        g = {"E29_RT": {}, "DS": DSM, "E29L": L, "Path": Path, "hashlib": __import__("hashlib"), "E29_SEAM_BYTES": {}}
        exec(compile(_defs(["e29_seam_snapshot", "e29_seam_restore"]), "<sr>", "exec"), g)
        g["e29_seam_snapshot"](d)
        g["e29_seam_restore"]()
        assert g["E29_RT"]["seam_restore"]["label"] == "SEAM_PRISTINE" and not g["E29_RT"]["seam_unverified"]
        (d / rels[0]).write_text("PATCHED", encoding="utf-8")                                # finish() raised / partial revert
        g["e29_seam_restore"]()
        sr = g["E29_RT"]["seam_restore"]
        assert sr["dirty_after_finish"] == [rels[0]] and sr["dirty_after_restore"] == [] and (d / rels[0]).read_text(encoding="utf-8") == "pristine " + rels[0]
        g["E29_SEAM_BYTES"][rels[1]] = b"different from the snapshot sha"                     # an unrestorable file
        (d / rels[1]).write_text("PATCHED", encoding="utf-8")
        g["e29_seam_restore"]()
        assert g["E29_RT"]["seam_unverified"] is True and g["E29_RT"]["seam_restore"]["label"] == "SEAM_REVERT_UNVERIFIED"
        assert L.seam_inert(LOG_OK)["ok"] is False and L.seam_inert("INFO Resolved architecture: Qwen4ExpForCausalLM")["label"] == "SEAM_REVERT_UNVERIFIED_INERT"
        return (ENG.count("e29_seam_restore()   # E29 rev 2d F11") == 2 and "e29_seam_snapshot(tree)   # E29 rev 2d F11\n            ok, why = SSEQ.apply()" in ENG
                and 'trec["status"] = "E29_SEAM_NOT_INERT"' in ENG and 'fail["failed"] = fail["failed"] or str(trec.get("status")) == "E29_SEAM_NOT_INERT"' in ENG)

    check("R24 (rev 2d F11) the seam revert is verified against a pre-apply snapshot of every seam-touched file: clean -> "
          "SEAM_PRISTINE; a dirty file (finish raised / partial) -> rewritten from the snapshot; still dirty -> SEAM_REVERT_UNVERIFIED "
          "and the MTP boot needs q_seam_inert (inert -> continue, recorded; not inert -> a real failure of that try, E28 ladder); "
          "never a stop of the run", _r24)

    def _r25():
        rt = {"on": ["df"], "cfg": RCFG, "direct": {"state": "ON"}, "qsa_state": "QSA_VERIFIED"}
        e = L.effective(rt, False)
        assert e["drafter"] == "DFlash2 D3S" and e["direct"] == "ON" and e["experiment_validity"] == "OK" and e["embed_host"] is True
        e2 = L.effective({"on": [], "cfg": RCFG, "direct": {"state": "OFF_SETUP", "error": "payload e29_nbx.py not found"},
                          "qsa_state": "UNVERIFIED_QSA"}, False)
        assert (e2["drafter"], e2["direct"], e2["experiment_validity"], e2["qsa"]) == ("MTP K3", "OFF_SETUP", "DIRECT_OFF", "UNVERIFIED_QSA")
        assert L.effective({"extras_error": "x"}, True)["experiment_validity"] == "OK"                 # the rerun: never a flag
        return ('E29_RT["effective_algorithm"] = E29L.effective(E29_RT, RERUN)   # E29 rev 2d F5' in ENG
                and "effective_algorithm=E29L.effective(E29_RT, RERUN)" in CODE[7] and 'print("[LIVE start] algo"' in NBX)

    check("R25 (rev 2d F5 / F14) effective_algorithm (drafter, DIRECT + why, QSA label, EMBED_HOST) printed at the start, in the "
          "report and the LIVE lines; EVAL with DIRECT off -> experiment_validity DIRECT_OFF (the run continues)", _r25)

    def _r26():
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="e29sub_"))
        tasks = {"a": {"test": [{"input": [[0]]}]}, "b": {"test": [{"input": [[1]]}, {"input": [[2]]}]}}
        f = d / "submission.json"
        ok = {"a": [{"attempt_1": [[1]], "attempt_2": [[0]]}], "b": [{"attempt_1": [[1, 2]], "attempt_2": [[3]]}] * 2}
        f.write_text(json.dumps(ok), encoding="utf-8")
        assert L.submission_check(f, tasks)["ok"] is True
        f.write_text(json.dumps({"a": ok["a"]}), encoding="utf-8")
        assert L.submission_check(f, tasks)["missing"] == ["b"]
        f.write_text(json.dumps(dict(ok, b=ok["b"][:1])), encoding="utf-8")
        assert L.submission_check(f, tasks)["bad_rows"] == ["b"]
        f.write_text(json.dumps(dict(ok, a=[{"attempt_1": [[10]], "attempt_2": [[0]]}])), encoding="utf-8")
        assert L.submission_check(f, tasks)["bad_grids"] == ["a"] and L.submission_check(d / "none.json", tasks)["ok"] is False
        return ("submission_check=E29L.submission_check(SUBMISSION_PATH" in CODE[7]
                and 'E29_RT["submission_check"] = E29L.submission_check(globals().get("SUBMISSION_PATH"), globals().get("test_challenges") or tasks)' in NBX
                and 'globals().get("test_challenges") or globals().get("CHALLENGES")' in CODE[7])   # rev 2f: the file's own task set

    check("R26 (rev 2d F12) submission_check (record only) after every writer callback and at the end: every task id, one row per "
          "test input, valid 0-9 grids for attempt_1 / attempt_2; a missing task / a short row list / a bad grid / no file are flagged", _r26)

    def _r27():
        class _E12:
            E29D_RUN = E29D_CAN_START = None
        g = {"E29_DIRECT2": {"on": True, "drain_min": 26}, "time": time, "E12A": _E12}
        line = next(l for l in NBX.splitlines() if "E12A.E29D_CAN_START = lambda deadline:" in l)
        exec(line.strip().split("   #")[0], g)
        cs = _E12.E29D_CAN_START
        now = time.monotonic()
        assert cs(now + 31 * 60) is True and cs(now + 20 * 60) is False and cs(now + 60) is False and cs(now + 27 * 60 + 5) is True
        return (NBX.count("E12A.E29D_RUN = E12A.E29D_CAN_START = None") >= 2 and ENG.count("E12A.E29D_RUN = E12A.E29D_CAN_START = None") == 1)

    check("R27 (rev 2d F1, the notebook hook) e29_direct_setup sets E12A.E29D_CAN_START(deadline) = now + 60 s + drain_min x 60 < "
          "deadline (31 / 27.1 min left -> True; 20 / 1 min -> False); cleared on OFF / setup failure / a missing payload", _r27)

    def _r28():
        o = {}

        class _Res:
            used, cap, held, waiting = 30, 540, {("t", 0): 30}, 2
        g = {"host_mem": lambda: {"cgroup_cur_mib": 150000, "cgroup_max_mib": 177152, "avail_mib": 20000, "anon_mib": 1},
             "e21_prom": lambda: [type("R", (), {"name": "vllm:num_preemptions_total", "value": 3.0})()],
             "NB5_RES": _Res(), "NB5_BLOCKS": {"f": 12}, "NB5_POOL_CAP": {"blocks": 540}, "E29_RT": {"host_peak_mib": 155000}}
        exec(compile(_defs(["e29_ledger"]), "<lg>", "exec"), g)
        o = g["e29_ledger"](8011)
        assert g["E29_RT"]["host_peak_mib"] == 155000 and g["E29_RT"]["host_max_mib"] == 177152      # rev 2e: the peak is kept
        return (o["host"] == {"cgroup_cur_mib": 150000, "cgroup_max_mib": 177152, "avail_mib": 20000} and o["preemptions"] == 3
                and (o["res_held_blocks"], o["res_reclaimable"], o["res_attempts"], o["res_waiting"], o["f"], o["pool_blocks"]) == (30, 510, 1, 2, 12, 540)
                and '" | ledger " + json.dumps(e29_ledger(int(NB5_PORT)))' in NBX and '"ledger": e29_ledger(s.port)' in ENG)

    check("R28 (rev 2d F6 / F13, record only) the ledger at ready, after the F2 probe and in every LIVE 15-min line: host cgroup "
          "current / max / avail, KV usage, running / waiting, preemptions, the NB5 reservation held / reclaimable / attempts / "
          "waiting, F, the pool", _r28)
    # ---------------------------------------------------------------- rev 2e: the GPU smoke profile (E29SMOKE)
    SMOKE12 = ["1818057f", "2ba387bc", "45a5af55", "bf45cf4b", "446ef5d2", "b6f77b65", "7b5033c1", "800d221b", "31f7f899",
               "d59b0160", "409aa875", "a47bf94d"]

    def _r29():
        i, j = C1.index("_NB5_T = {"), C1.index('if E11_HARNESS and os.environ.get("NB5H_TASKS"):')
        k, m = C1.index("# ---- time (E15B / E21 CELL 1 block, verbatim)"), C1.index("# ---- NB5 engine + workload parameters.")
        sm = ast.literal_eval(next(l for l in C1.splitlines() if l.startswith("E29_SMOKE = "))[len("E29_SMOKE = "):].split("   #")[0])
        assert sm == {"tasks": sorted(SMOKE12), "window_min": 40, "budget_min": 70}, sm
        out = {}
        for prof in ("E29SMOKE", "E28EVAL120"):
            g = {"os": os, "json": json, "E11_HARNESS": False, "E29_SMOKE": sm, "RERUN": False}
            exec(compile(C1[i:j] + f"\nPROFILE = '{prof}'\nP = PROFILE_TABLE[PROFILE]\nNOTEBOOK_T0 = 1000.0\n" + C1[k:m], "<smoke>", "exec"), g)
            out[prof] = g
        S_, E_ = out["E29SMOKE"]["PROFILE_TABLE"]["E29SMOKE"], out["E29SMOKE"]["PROFILE_TABLE"]["E28EVAL120"]
        diff = {k_ for k_ in set(S_) | set(E_) if S_.get(k_) != E_.get(k_)}
        assert diff == {"time_budget_sec", "n_tasks", "task_list", "fixed_window_sec"}, diff          # all else = production
        assert (S_["time_budget_sec"], S_["n_tasks"], S_["task_list"], S_["fixed_window_sec"]) == (4200, 12, sorted(SMOKE12), 2400)
        assert out["E29SMOKE"]["HARD_STOP"] - 1000.0 == 65 * 60 and out["E28EVAL120"]["HARD_STOP"] - 1000.0 == 410 * 60
        dl = "\n".join(l for l in C1.splitlines() if l.startswith("K100_DEADMAN_SEC = "))
        for prof, want in (("E29SMOKE", 4200), ("E28EVAL120", 24900)):
            gd = {"E29_SMOKE": sm, "E29_PROFILE": prof}
            exec(dl, gd)
            assert gd["K100_DEADMAN_SEC"] == want, (prof, gd["K100_DEADMAN_SEC"])
        lines = C1.splitlines()
        assert lines[2].startswith('E29_PROFILE = "E28EVAL120"') and C1.count("E29_PROFILE = ") == 1 and "NB5_DEFAULT_PROFILE = E29_PROFILE" in C1
        sk = FACTS["e29"]["smoke"]
        a_, b_ = (json.dumps(x + "\n")[1:-1].encode("utf-8") for x in sk["diff"][1:])
        raw = NB.read_bytes()
        import hashlib
        return (raw.count(a_) == 1 and hashlib.sha256(raw.replace(a_, b_)).hexdigest() == sk["sha256"] and "E29SMOKE" in sk["diff"][2]
                and sk["tasks"] == sorted(SMOKE12) and "E29SMOKE" in out["E29SMOKE"]["K100_PROFILES"])

    check("R29 (rev 2e) the GPU smoke: CELL 1 line 3 E29_PROFILE (production E28EVAL120); the E29SMOKE row = the production row "
          "except budget 70 min / 12 tasks (task_list, plain sorted) / the fixed 40-min window; HARD_STOP 65 min, deadman 70 min "
          "(production 410 / 415 unchanged); the smoke notebook = these bytes with ONLY that one line changed (its sha = the facts)", _r29)

    def _r30():
        rt = {"on": ["df"], "cfg": RCFG, "direct": {"state": "ON", "tasks": {"a": {"mode": "forced_close"}, "b": {"mode": "natural"}}},
              "qsa_state": "QSA_VERIFIED", "submission_check": {"ok": True}, "host_peak_mib": 160000, "host_max_mib": 177152,
              "seam_restore": None, "dropped": []}
        boots = [{"tries": [{"name": "d0", "status": "E29_DFLASH_LOW_FREE", "e29_df": {"rung": 0}, "e29_probe": {"workers_free_mib": [900, 400]}},
                            {"name": "d2", "status": "DONE", "phases_started": True, "e29_df": {"rung": 2},
                             "e29_df_readback": {"label": "ENGAGED", "grd": True}, "server_env_levers": {"DFLASH_DRAFT_EMBED_HOST": "1"},
                             "e29_probe": {"workers_free_mib": [900, 700, 650, 640], "sec": 31.0}}]}]
        samples = [{"t": 0, "gen": 0, "sd_drafts": 0, "sd_acc": 0}, {"t": 900, "gen": 540000, "sd_drafts": 100, "sd_acc": 210},
                   {"t": 1800, "gen": 1080000, "sd_drafts": 200, "sd_acc": 420}]
        sched = {"e28": {"started": {"A": 12, "C": 5}, "done": {"A": 12, "C": 5}, "slots": {"a": [{"attempt_2": "D"}], "b": [{"attempt_2": "C"}]},
                         "direct": {"started": 3, "records": 2, "no_record": 1, "errors": 0, "duck_c": 2, "duck_c_why": {"late": 2}}}}
        e28o = {"engine": {"solver_tok_s": 590.1, "preemptions": 4, "prefix_hit_pct": 70.2},
                "score": {"generated": {"eval_query_hits": 9, "eval_query_total": 15, "eval_task_norm_pct": 58.3}}}
        s = L.run_summary(rt, e28o, sched, samples, boots, "E29SMOKE")
        assert s["dflash_engaged"] and s["embed_host"] and s["qsa"] == "QSA_VERIFIED" and s["booted_try"] == "d2"
        assert [p_["min_free_mib"] for p_ in s["probe"]] == [400, 640] and s["tok_s_per_15min"] == [{"min": 0, "tok_s": None}, {"min": 15, "tok_s": 600.0}, {"min": 30, "tok_s": 600.0}]
        assert s["tok_s_window"] == 600.0 and s["al"] == 3.1 and s["attempt_2_from_direct"] == 1 and s["direct_modes"] == {"forced_close": 1, "natural": 1}
        assert s["direct"]["duck_c_why"] == {"late": 2} and s["submission_ok"] is True and s["eval"]["eval_task_norm_pct"] == 58.3
        txt = L.summary_text(s)
        assert "DFlash engaged True" in txt and "QSA QSA_VERIFIED" in txt and "min free 640 MiB" in txt and "attempt_2 from DIRECT 1" in txt
        return ('print("[SMOKE SUMMARY]" if PROFILE == "E29SMOKE" else "[E29 RUN SUMMARY]"' in CODE[7]
                and 'E29L.run_summary(E29_RT, E28_OUT, getattr(E12A, "SCHED_LAST", None)' in CODE[7] and "host_peak_mib=max(" in ENG)

    check("R30 (rev 2e, record only) the [SMOKE SUMMARY] block (+ JSON) from the run records: DFlash engaged (seam + GRD), "
          "EMBED_HOST, QSA, the F2 probe per try + min free, tries / drops, tok/s per 15 min + window, AL, preemptions, prefix hit, "
          "A / B / C / DIRECT counts, Duck-C late routes + why, DIRECT modes, attempt_2 from DIRECT, submission ok, eval, host peak", _r30)
    check("R14 CELL 3: the DIRECT diff right after the scheduler apply and before `import arc2_duck` (only on the E29 scheduler; "
          "E29_DIRECT=1 only after apply + verify; a failure reverts to the E29 post file)", _r14)

n_bad = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_bad}/{len(RESULTS)} PASS", flush=True)
sys.exit(1 if n_bad else 0)
