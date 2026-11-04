"""E32 checks shared by the local unit tests (e32_tests.py: inputs from the builder) and the Kaggle CPU units kit
(e32_kit_tests.py: inputs from the notebooks + the payload files). Stdlib + the notebook's own modules only.

ctx(...) builds the context from two notebook files (E32 and the E29_DFLASH_D3S base) and the module texts; run(ctx, check)
registers every check:
  P   switch parse: every value parses; a bad / missing value = that lever OFF, never an exception
  A   argv / env / the DFlash tail for EVERY lever combination (3 x 3 x 2^5 = 288), through the notebook's OWN e29_extra / e29_env
      defs (+ e32_call and e32_nbx); OFF = the E29 tail exactly
  O   levers OFF (and the E32 payload missing) = exactly the E29_DFLASH_D3S nb5_boot tries (names, pins, seqs, argv, env, drops)
  D   the drop order on injected real failures through the notebook's OWN nb5_boot / e29_drop / e29_env / e29_extra
  R   in-run: the relaunch rebuild = what nb5_boot builds fresh; e29_mtp_boot leaves no E32 argv / env / seqs
  S   the speed pass (stub L.steady), the read-back, the fingerprint
  K   identity vs the E29 notebook: CELL 0 / CELL 3 / cell 4 / CELL 6/7 byte-identical, the E29D_FILES pins, < 400,000 B, no CR
"""
import ast
import copy
import itertools
import json
import os
import re
import time
from pathlib import Path

ENGINE_PACK_HEAD = "\n\n\n# ---- E29 FINAL rev 2 (size: the 400-kB commit limit): the E29 engine block"


def unpack_engine(c):
    """= build_e29_final.unpack_engine: every packed E29 block of the engine cell back in place."""
    while ENGINE_PACK_HEAD in c:
        h = c.index(ENGINE_PACK_HEAD)
        x = c.index("exec(compile(", h)
        e = c.index("\n", x) + 1
        call = next(n for n in ast.walk(ast.parse(c[x:e])) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        expr = ast.get_source_segment(c[x:e], call.args[0].args[0])
        c = c[:h] + eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e32 packed>", "eval")) + c[e:]
    return c


def code_cells(nb_bytes):
    return ["".join(c["source"]) for c in json.loads(nb_bytes.decode("utf-8"))["cells"] if c["cell_type"] == "code"]


def all_cells(nb_bytes):
    return ["".join(c["source"]) for c in json.loads(nb_bytes.decode("utf-8"))["cells"]]


def modules_literal(c1):
    """The E28_SRC literal of CELL 1 (e28k100_lib / e28_sched_patch / e29_lib / e29_sched_patch / e29_nbdf texts)."""
    line = next(l for l in c1.splitlines() if l.startswith("E28_SRC = json.loads("))
    expr = ast.get_source_segment(line, ast.parse(line).body[0].value.args[0])
    return json.loads(eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<packed>", "eval")))


def _lit(c1, name):
    line = next(l for l in c1.splitlines() if l.startswith(name + " = "))
    return ast.literal_eval(ast.parse(line).body[0].value)


def ctx(e32_nb, e29_nb, E29L, E28L, NB5L, E32L, nbx_text):
    """The check context from the two notebook files' bytes and the modules (imported by the caller)."""
    c32, c29 = code_cells(e32_nb), code_cells(e29_nb)
    mods = modules_literal(c32[1])
    N = {}
    exec(compile(mods["e29_nbdf.py"], "e29_nbdf.py", "exec"), N)
    pins = _lit(c32[1], "K100_PINS")
    cfg = json.loads(re.search(r'E29_CFG = E29L\.check_cfg\(json\.loads\(r"""(\{.*?\})"""\)\)', c32[1]).group(1))
    return {"E29L": E29L, "E28L": E28L, "NB5L": NB5L, "E32L": E32L, "N": N, "NBX": nbx_text, "MODS": mods,
            "E32_NB": e32_nb, "E29_NB": e29_nb, "C32": c32, "C29": c29, "E32_ENG": unpack_engine(c32[4]), "E29_ENG": unpack_engine(c29[4]),
            "STEPS": [(b - 2479838700) / 2 ** 30 for b in pins], "SEQS_T": _lit(c32[1], "K100_SEQS_TABLE"), "TOP": pins[0],
            "BASE_CFG": E29L.check_cfg(cfg), "E32_FILES": json.loads(re.search(r"^E32_FILES = (\{.*\})$", unpack_engine(c32[4]), re.M).group(1))}


# ------------------------------------------------------------------ the harness (the notebook's own defs, stubs around them)
class DSStub:
    ALL_ENVS = ("DFLASH_SEAM", "DFLASH_MASK_ROW_SHA", "DFLASH_DRAFT_SW_BLOCK", "DFLASH_DRAFT_EMBED_HOST")

    @staticmethod
    def server_env(mask_row_sha=None, embed_host=False, draft_sw_block=None, **_):
        e = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": mask_row_sha, "DFLASH_DRAFT_SW_BLOCK": str(draft_sw_block)}
        if embed_host:
            e["DFLASH_DRAFT_EMBED_HOST"] = "1"
        return e


def defs(eng, names):
    t = ast.parse(eng)
    fns = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in fns) == sorted(names), (sorted(f.name for f in fns), names)
    return ast.Module(fns, [])


class Old:
    def __init__(self, text):
        self.text = text

    def log(self):
        return self.text


class Srv:
    port, argv, env = 8001, ["vllm", "serve", "--max-num-seqs", "28"], {"VLLM_GDN_DECODE_FASTPATH": "1", "HOME": "/root"}

    def __init__(self, alive=True):
        self._a = alive

    def alive(self):
        return self._a

    def log(self):
        return ""


def run(X, check):
    E29L, E28L, NB5L, E32L, N = X["E29L"], X["E28L"], X["NB5L"], X["E32L"], X["N"]
    E32_ENG, E29_ENG, NBX = X["E32_ENG"], X["E29_ENG"], X["NBX"]
    STEPS, SEQS_T, TOP = X["STEPS"], X["SEQS_T"], X["TOP"]
    SPEC = N["dflash_spec"]("/drafter", 3, "auto", "FLASH_ATTN")
    CAPS = N["capture_sizes"](24, 3)
    TAIL = E29L.df_spec_argv(N["spec_json"](SPEC), N["compilation_json"](CAPS))
    DEF, OFF = dict(E32L.DEFAULT), dict(E32L.OFF)
    OOM = ("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB")
    DPINS = [t["kv_bytes"] for t in E29L.df_tries(True)]

    def ns(eng, switches=None, e32=True, payload=True):
        cfg = E29L.check_cfg(X["BASE_CFG"])
        g = {"json": json, "time": time, "os": os, "copy": copy, "NB5L": NB5L, "NB5_KV_DTYPE": "fp8", "NB5_KV_BYTES": {"fp8": E29L.FP8_PIN},
             "NB5_BOOTS": [], "NB5_EXTRA_ARGS": [], "NB5_MTP_OFF_ARGS": ["--speculative-config", "None"], "FAKE": None,
             "P": {"pin_ladder": True, "patch_60490": True, "k100": True, "i8": True, "top_rung_seqs": 24},
             "NB5_PIN": {"start_gib": None, "levers": None, "patch_60490": {"ok": True}}, "K100_PIN_STEPS_GIB": STEPS,
             "E29_QSA": {"max_logits_mb": 128, "granule_mb": 64}, "NB5_LEVER_ENV": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
             "NB5_SEQS_RULE": "k100", "K100_SEQS_TABLE": SEQS_T, "left": lambda: 1e9,
             "E28_I8": {"want": True, "on": False, "static": None, "dropped": None, "readback": None, "suspended": "N/A_DFLASH"},
             "E28_I8_ENV": {"DRAFT_HEAD": "int8"}, "E28L": E28L, "log": lambda *a: None, "nb5_write_report": lambda: None, "ABORT": {},
             "NB5_READY_TIMEOUT": 1800, "NB5_BOOT_LEVERS": {}, "NB5_BOOT_EXTRA": [], "NB5_BOOT_KV": None, "E29L": E29L,
             "N": type("N", (), N), "E29_CFG": cfg, "DS": DSStub, "SSEQ": None, "NOTEBOOK_T0": time.perf_counter(), "CONFIG": {},
             "E11_HARNESS": False, "RERUN": False, "NB5_ACTIVE_BOOT": {}, "E11_SERVER": None,
             "E29_RT": {"cfg": cfg, "on": ["df"], "dropped": [], "tries": [], "canary": [], "errors": [], "df_try": False, "df_live": False,
                        "df": {"ok": True, "k": 3, "spec": copy.deepcopy(SPEC), "captures": list(CAPS), "tail": list(TAIL),
                               "mask_row_sha": "ab" * 32, "sw_block": 1024}}}
        names = ["nb5_boot", "e29_env", "e29_extra", "e29_ntries", "e29_drop"] + (["e32_call", "e32_report"] if e32 else [])
        exec(compile(defs(eng, names), "<engine defs>", "exec"), g)
        if e32:
            g["E32_RT"] = {"state": None, "errors": [], "files": X["E32_FILES"]}
            g["E32L"] = E32L
            if payload:
                exec(compile(NBX, "e32_nbx.py", "exec"), g)
                for k, v in (switches or {}).items():
                    g["E32_" + k] = v
                g["E32_SPEED"] = (switches or {}).get("_SPEED", E32L.SPEED_DEFAULT)
                g["E32_RT"]["state"] = "ON"
                g["_e32x_setup"]()
                if not g["E32_RT"].get("active"):
                    g["E32_RT"]["state"] = "OFF_NO_DFLASH"
            else:
                g["E32_RT"].update(state="OFF_SETUP", error="FileNotFoundError: payload e32_lib.py not found")
        return g

    def ladder(eng, script, switches=None, e32=True, payload=True):
        g = ns(eng, switches, e32, payload)
        seen, it = [], iter(script)

        def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
            st = next(it)
            st, lg = st if isinstance(st, tuple) else (st, "")
            av = list(g["NB5_BOOT_EXTRA"])
            seen.append({"name": name, "kv_bytes": kv_bytes, "levers": dict(g["NB5_BOOT_LEVERS"]), "extra": av, "df_try": g["E29_RT"]["df_try"],
                         "seqs": av[av.index("--max-num-seqs") + 1] if "--max-num-seqs" in av else None, "e29_on": list(trec.get("e29_on") or []),
                         "e28_i8": trec.get("e28_i8"), "k": (g["E29_RT"].get("df") or {}).get("k")})
            if st == "DONE":
                trec.update(status="DONE", phases_started=True)
            else:
                trec.update(status=st, log_text_for_ladder=lg)
            if st == "E29_DFLASH_CONFIG_MISMATCH":
                trec["config_readback"] = {"checks": {"kv_dtype": True, "max_num_seqs": False}}
            return trec
        g["nb5_boot_try"] = nb5_boot_try
        b = g["nb5_boot"]("merlin_fp8_mtp", [["Q", "all", None, "on", "A", "fp8"]], True, "fp8")
        assert not (g.get("E32_RT") or {}).get("errors"), ("an E32 hook raised (swallowed by e32_call)", g["E32_RT"]["errors"])
        return b, seen, g

    VALS = {"SEQS": (24, 28, 32), "SPEC": ("K3", "D4s", "DYN"), "LP": (True, False), "SPC": (True, False), "IDX": (True, False),
            "G1": (True, False), "G2": (True, False)}

    # ------------------------------------------------------------------ P
    def _p1():
        n = 0
        for combo in itertools.product(*VALS.values()):
            sw = dict(zip(VALS, combo))
            p = E32L.parse(sw)
            assert p["cfg"] == sw and not p["invalid"], (sw, p)
            assert p["on"] == [k for k in E32L.LEVERS if sw[k] != OFF[k]]
            n += 1
        return n == 288 and E32L.parse(DEF)["on"] == list(E32L.LEVERS) and E32L.parse(OFF)["on"] == []
    check("P1 every switch value parses (288 combinations): cfg = the values, on = the non-OFF levers; DEFAULT = all 7 on; OFF = none", _p1)

    def _p2():
        bad = {"SEQS": [26, "28", 28.0, True, None, [28]], "SPEC": ["dyn", "K4", None, 3, True], "LP": ["yes", 1, 0, None, "True"],
               "SPC": [1], "IDX": ["fp8"], "G1": [None], "G2": [2]}
        for k, vs in bad.items():
            for v in vs:
                p = E32L.parse(dict(DEF, **{k: v}))
                assert p["cfg"][k] == OFF[k] and k in p["invalid"] and k not in p["on"], (k, v, p)
                assert all(p["cfg"][x] == DEF[x] for x in E32L.LEVERS if x != k)
        for junk in (None, "garbage", 5, [], {"SEQS": object()}):
            try:
                p = E32L.parse(junk if isinstance(junk, dict) or junk is None else {})
            except Exception as e:
                raise AssertionError(("parse raised", junk, e))
        p = E32L.parse({})
        assert p["on"] == [] and set(p["invalid"]) == set(E32L.LEVERS)
        p = E32L.parse(DEF, harness_json='{"SPEC": "K3", "LP": false, "nope": 1}')
        assert p["cfg"]["SPEC"] == "K3" and p["cfg"]["LP"] is False and p["harness_override"] == ["LP", "SPEC"]
        assert E32L.parse(DEF, harness_json="{not json")["cfg"] == DEF
        sp = E32L.speed_cfg({"on": "yes"})
        assert sp["on"] is False and sp["invalid"]
        sp = E32L.speed_cfg({"windows": [["8k", 16]]})
        assert sp["windows"] == E32L.SPEED_DEFAULT["windows"] and sp["invalid"]
        sp = E32L.speed_cfg("x")
        return sp["on"] is True and sp["invalid"] and E32L.speed_cfg(E32L.SPEED_DEFAULT)["invalid"] is None
    check("P2 a bad / missing value = that lever OFF (recorded invalid), the others unchanged, never an exception (ints as str / float / "
          "bool, unknown SPEC names, non-bool flags, a missing switch, junk input, a bad harness JSON); the speed switch: a bad 'on' = "
          "off, windows without 90k / 116k = the default set", _p2)

    def _p3():
        g = ns(E32_ENG, dict(DEF, SEQS="28", SPEC="dyn", LP=1))
        st = g["E32_RT"]["parsed"]
        assert st["cfg"]["SEQS"] == 24 and st["cfg"]["SPEC"] == "K3" and st["cfg"]["LP"] is False and g["E32_RT"]["active"]
        assert g["E32_RT"]["cur"] == dict(DEF, SEQS=24, SPEC="K3", LP=False) and g["E29_RT"]["df"]["tail"] == TAIL
        g2 = ns(E32_ENG, {})
        return g2["E32_RT"]["cur"] == OFF and g2["E29_RT"]["df"]["tail"] == TAIL and set(g2["E32_RT"]["parsed"]["invalid"]) == set(E32L.LEVERS)
    check("P3 the notebook's own setup (_e32x_setup on the CELL 1 globals): bad values -> those levers OFF in the booted plan; every "
          "switch line missing -> all OFF and the E29 DFlash tail exactly", _p3)

    # ------------------------------------------------------------------ A
    def _a1():
        base_env = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": "ab" * 32, "DFLASH_DRAFT_SW_BLOCK": "1024", "DFLASH_DRAFT_EMBED_HOST": "1",
                    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "128", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "64"}
        n = 0
        for combo in itertools.product(*VALS.values()):
            sw = dict(zip(VALS, combo))
            g = ns(E32_ENG, sw)
            df = {"embed_host": True, "rung": 2}
            ex, env = g["e29_extra"](df), g["e29_env"](df)
            want_argv = ((E32L.LP_ARGS if sw["LP"] else []) + (E32L.SPC_ARGS if sw["SPC"] else []) + (E32L.IDX_ARGS if sw["IDX"] else []))
            k = 3 if sw["SPEC"] == "K3" else 4
            tail = ex[len(want_argv):]
            assert ex[:len(want_argv)] == want_argv, (sw, ex)
            assert tail[0] == "--speculative-config" and tail[2] == "--compilation-config" and len(tail) == 4, tail
            sp, comp = json.loads(tail[1]), json.loads(tail[3])
            assert sp["method"] == "dflash" and sp["num_speculative_tokens"] == k and sp["draft_sample_method"] == "probabilistic"
            assert sp["kv_cache_dtype"] == "auto" and sp["attention_backend"] == "FLASH_ATTN" and sp["model"] == "/drafter"
            assert ("num_speculative_tokens_per_batch_size" in sp) == (sw["SPEC"] == "DYN")
            if sw["SPEC"] == "DYN":
                d = E32L.dense_schedule(sp["num_speculative_tokens_per_batch_size"], sw["SEQS"], 4)
                assert d[1:9] == [4] * 8 and set(d[9:]) == {3} and d == E32L.dense_schedule(E32L.DYN_SCHEDULE, sw["SEQS"], 4)
                assert comp["cudagraph_capture_sizes"] == E32L.dyn_captures(sw["SEQS"])
            else:
                assert comp["cudagraph_capture_sizes"] == E32L.dflash_captures(sw["SEQS"], k) == N["capture_sizes"](sw["SEQS"], k)
            assert comp["cudagraph_mode"] == "FULL_AND_PIECEWISE" and g["E32_RT"]["plan"]["coverage"]["covers_all"]
            assert max(comp["cudagraph_capture_sizes"]) <= N["MAX_NUM_BATCHED"]
            if sw["SEQS"] == 24 and sw["SPEC"] == "K3":
                assert tail == TAIL
            assert env == dict(base_env, **{E32L.GDN_ENV[x]: "1" for x in ("G1", "G2") if sw[x]}), (sw, env)
            assert g["e29_extra"](None) == [] and "VLLM_GDN_DECODE_FASTPATH" not in g["e29_env"](None)
            assert g["e32_call"]("seqs", 24) == sw["SEQS"] and g["E29_RT"]["df"]["k"] == k
            assert g["e32_call"]("tag", "", df) == E32L.tag(sw) and (E32L.tag(sw) == "") == (not E32L.on_list(sw))
            assert not g["E32_RT"]["errors"], g["E32_RT"]["errors"]
            n += 1
        return n == 288
    check("A1 every lever combination (288) through the notebook's own e29_extra / e29_env (+ e32_call, e32_nbx): the E32 argv in the "
          "fixed order LP, SPC, IDX before the DFlash tail; the tail = --speculative-config (dflash, K3 / K4, DYN schedule only on DYN, "
          "its dense table = NB-D1's) + --compilation-config (captures = nbdf capture_sizes / NB-D1 dyn_captures, every decode batch "
          "covered, <= max batched 1024); the env = E29's seam + EMBED_HOST + QSA 128/64 + exactly the on G1 / G2 keys; an MTP try gets "
          "nothing; the try tag", _a1)

    # ------------------------------------------------------------------ O
    scripts = [["DONE"], [OOM, "DONE"], [OOM, OOM, OOM, "DONE"], ["E29_DFLASH_NOT_ENGAGED", "DONE"], ["E29_DFLASH_LOW_FREE", "DONE"],
               [("E29_LAUNCH_PREP_FAILED", "RuntimeError: FAIL_CLI_FLAGS: missing flags: --enable-mamba-shared-prefix-checkpoint"), "DONE"],
               ["E29_DFLASH_ZERO_ACCEPT", "DONE"], [OOM, OOM, OOM, OOM, "DONE"]]

    def _o1():
        for sc in scripts:
            b29, s29, g29 = ladder(E29_ENG, sc, e32=False)
            for sw, pay in ((OFF, True), (DEF, False), (OFF, False)):
                b32, s32, g32 = ladder(E32_ENG, sc, sw, payload=pay)
                assert s32 == s29, (sc, sw, pay, s29, s32)
                assert g32["E29_RT"]["dropped"] == g29["E29_RT"]["dropped"] and b32["status"] == b29["status"], (sc, sw, pay)
                assert g32["E28_I8"] == g29["E28_I8"]
        return True
    check("O1 levers OFF = exactly E29_DFLASH_D3S: the notebook's nb5_boot on 8 scripted ladders (DONE, OOMs down the D ladder to the "
          "MTP top rung, NOT_ENGAGED, LOW_FREE, a rejected flag, zero-accept, every D rung failed) gives the same tries (names, pins, "
          "seqs, argv, env, e29_on), drops, status and I8 as the E29 engine; also with the E32 payload missing (state OFF_SETUP) and "
          "every switch on", _o1)

    # ------------------------------------------------------------------ D
    def _d1():
        b, seen, g = ladder(E32_ENG, [OOM] * 5 + ["DONE"], DEF)
        drops = [(d["lever"], d["class"]) for d in g["E32_RT"]["drops"]]
        assert drops == [("LP", "MEMORY"), ("SEQS", "MEMORY"), ("SPEC", "MEMORY"), ("SPC", "MEMORY")], drops
        assert [s["kv_bytes"] for s in seen] == [DPINS[0]] * 5 + [DPINS[1]], [s["kv_bytes"] for s in seen]
        assert [s["seqs"] for s in seen] == ["28", "28", "24", "24", "24", "24"] and [s["k"] for s in seen] == [4, 4, 4, 3, 3, 3]
        assert seen[0]["extra"][:10] == ["--max-num-seqs", "28"] + E32L.LP_ARGS + E32L.SPC_ARGS + E32L.IDX_ARGS
        assert seen[0]["name"].endswith("_e32s8ylpifa") and seen[-1]["name"].endswith("_e32ifa")
        assert all("--long-prefill-token-threshold" not in s["extra"] for s in seen[1:])
        last = seen[-1]
        assert last["extra"][-6:-4] == E32L.IDX_ARGS and last["levers"].get("VLLM_GDN_DECODE_FASTPATH") == "1" and last["extra"][-4:] == TAIL
        assert b["status"] == "DONE" and [x["lever"] for x in g["E29_RT"]["dropped"]] == ["E32:LP", "E32:SEQS", "E32:SPEC", "E32:SPC"]
        return g["E32_RT"]["cur"] == dict(OFF, IDX=True, G1=True, G2=True)
    check("D1 OOM at boot (every lever on): the memory order LP -> SEQS -> SPEC -> SPC, one lever per failed try at the SAME D rung "
          "(7,665,090,560); the 5th OOM with only IDX / G1 / G2 left = E29's rule (the next D rung 7,396,655,104, IDX / G1 / G2 kept, "
          "the E29 K3 tail at 24 seqs); every drop recorded (E32 + E29_RT dropped)", _d1)

    def _d2():
        b, seen, g = ladder(E32_ENG, ["E29_DFLASH_LOW_FREE", "DONE"], DEF)
        a = g["E32_RT"]["drops"][0]
        b2, seen2, g2 = ladder(E32_ENG, ["E29_DFLASH_LOW_FREE", "DONE"], dict(OFF, IDX=True, G1=True))
        return (a["lever"] == "LP" and a["class"] == "MEMORY" and [s["kv_bytes"] for s in seen] == [DPINS[0]] * 2
                and not g2["E32_RT"]["drops"] and [s["kv_bytes"] for s in seen2] == DPINS[:2])
    check("D2 the F2 probe's E29_DFLASH_LOW_FREE: LP dropped (same rung); with no memory lever on (IDX / G1 only) = E29's one D-rung "
          "step", _d2)

    def _d3():
        out = {}
        for lev, line in (("SPC", "RuntimeError: FAIL_CLI_FLAGS: missing flags: --enable-mamba-shared-prefix-checkpoint"),
                          ("LP", "vllm serve: error: unrecognized arguments: --long-prefill-token-threshold-adaptive"),
                          ("IDX", "RuntimeError: FAIL_CLI_FLAGS: missing flags: --attention-config"),
                          ("SPEC", "ValueError: num_speculative_tokens_per_batch_size must cover 1..max_num_seqs"),
                          ("G2", "AttributeError: VLLM_GDN_SHARED_ALIGN_INDEX unknown")):
            b, seen, g = ladder(E32_ENG, [("E29_LAUNCH_PREP_FAILED", line), "DONE"], DEF)
            d = g["E32_RT"]["drops"]
            out[lev] = (d[0]["lever"], d[0]["class"], seen[1]["kv_bytes"] == seen[0]["kv_bytes"])
        return all(v == (k, "NAMED_IN_ERROR_LINE", True) for k, v in out.items()) or out
    check("D3 the server / help gate rejects a flag (FAIL_CLI_FLAGS, argparse 'unrecognized arguments', a config ValueError): the "
          "lever whose own flag text is in the error line is dropped first (SPC, LP, IDX, SPEC, G2), the same rung again", _d3)

    def _d4():
        lg = "INFO ... WARNING config/vllm.py: Overriding cudagraph_mode to PIECEWISE for num_speculative_tokens_per_batch_size"
        b, seen, g = ladder(E32_ENG, [("E32_DYN_FALLBACK", lg), "DONE"], DEF)
        d = g["E32_RT"]["drops"][0]
        sp = json.loads(seen[1]["extra"][-3])
        b2, seen2, g2 = ladder(E32_ENG, [("E32_DYN_FALLBACK", lg), "DONE"], dict(DEF, SPEC="D4s"))
        return (d["lever"] == "SPEC" and d["class"] == "DYN_FALLBACK_LINE" and sp["num_speculative_tokens"] == 3 and seen[1]["seqs"] == "28"
                and "num_speculative_tokens_per_batch_size" not in sp and g2["E32_RT"]["drops"][0]["class"] != "DYN_FALLBACK_LINE"
                and json.loads(seen[1]["extra"][-1])["cudagraph_capture_sizes"] == E32L.dflash_captures(28, 3))
    check("D4 the DYN fail regex ('Overriding cudagraph_mode' / 'Disabling num_speculative_tokens_per_batch_size'): SPEC dropped -> K3 "
          "(the D3s spec, 28 seqs kept, K3 captures for 28); with D4s it is not a DYN fallback", _d4)

    def _d5():
        b, seen, g = ladder(E32_ENG, ["E29_DFLASH_ZERO_ACCEPT"] * 8 + ["DONE"], DEF)
        order = [d["lever"] for d in g["E32_RT"]["drops"] if d["class"] != "WITH_DFLASH"]
        assert order == ["SPEC", "SPC", "IDX", "G1", "G2", "LP", "SEQS"], order
        m = seen[-1]
        assert [s["df_try"] for s in seen] == [True] * 8 + [False] and m["kv_bytes"] == TOP and m["seqs"] == "24"
        assert m["levers"].get("DRAFT_HEAD") == "int8" and not any(k.startswith(("DFLASH_", "VLLM_GDN")) for k in m["levers"])
        assert not any(x in m["extra"] for x in E32L.LP_ARGS + E32L.SPC_ARGS + E32L.IDX_ARGS) and "dflash" not in " ".join(m["extra"])
        return [x["lever"] for x in g["E29_RT"]["dropped"]][-1] == "df" and b["status"] == "DONE" and g["E28_I8"]["on"] is True
    check("D5 a zero-accept canary on every D try: the general order SPEC -> SPC -> IDX -> G1 -> G2 -> LP -> SEQS (one per try, the "
          "same rung), then E29 drops DFlash -> the E28 MTP top rung (7,933,526,016, 24 seqs, I8 restored, no E32 argv / env)", _d5)

    def _d6():
        xid = ("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory\nRuntimeError: CUDA error: an illegal memory access was encountered")
        b, seen, g = ladder(E32_ENG, [xid, "DONE"], DEF)
        b2, seen2, g2 = ladder(E32_ENG, ["E29_DFLASH_CONFIG_MISMATCH", "DONE"], DEF)
        b3, seen3, g3 = ladder(E32_ENG, ["NOT_READY", "DONE"], DEF)
        return (g["E32_RT"]["drops"][0]["lever"] == "SPEC" and g["E32_RT"]["drops"][0]["class"] == "DEVICE_FAULT"
                and g2["E32_RT"]["drops"][0]["lever"] == "SEQS" and g2["E32_RT"]["drops"][0]["class"] == "CONFIG_MISMATCH_SEQS"
                and g3["E32_RT"]["drops"][0]["lever"] == "SPEC" and g3["E32_RT"]["drops"][0]["class"] == "GENERAL")
    check("D6 a device fault (XID with an OOM line) = the general order (SPEC first, not the memory order); CONFIG_MISMATCH on the seqs "
          "check -> SEQS; any other pre-phase failure (not ready) -> the general order", _d6)

    def _d7():
        r1 = E32L.pick(OFF, "SERVER_DIED", OOM[1], True, False)
        r2 = E32L.pick(dict(OFF, G1=True), "E29_DFLASH_LOW_FREE", "", False, False)
        r3 = E32L.pick(dict(OFF, G1=True), "X", "", False, False)
        r4 = E32L.pick(DEF, "NO_GO_INSUFFICIENT_KV", "", False, False)
        return (r1 == (None, "NO_E32_LEVER_ON") and r2 == (None, "MEMORY_NO_MEMORY_LEVER") and r3 == ("G1", "GENERAL")
                and r4 == ("SEQS", "CAPACITY_GATE"))
    check("D7 pick: no lever on -> None (E29's rule); a memory failure with no memory lever -> None (the D ladder); G1 alone -> G1; "
          "the capacity gate (NO_GO_INSUFFICIENT_KV) -> SEQS (back to 24)", _d7)

    # ------------------------------------------------------------------ R
    def boot_ns(sw):
        b, seen, g = ladder(E32_ENG, ["DONE"], sw)
        exec(compile(defs(E32_ENG, ["e29_relaunch_plan", "e29_mtp_boot"]), "<rp>", "exec"), g)
        g["NB5_ACTIVE_BOOT"].update(name=seen[0]["name"], kv="fp8", kv_bytes=seen[0]["kv_bytes"], extra=list(seen[0]["extra"]),
                                    levers=dict(seen[0]["levers"]), max_num_seqs=int(seen[0]["seqs"]), pin_gib=E29L.gib_of(seen[0]["kv_bytes"]))
        g["E29_RT"].update(df_live=True, df_cur={"embed_host": True, "rung": 2})
        return g, seen

    def _r1():
        g, seen = boot_ns(DEF)
        ok = g["e32_call"]("relaunch", False, Old(OOM[1])) or g["e29_relaunch_plan"](Old(OOM[1]))
        ab = g["NB5_ACTIVE_BOOT"]
        b, fresh, g2 = ladder(E32_ENG, ["DONE"], dict(DEF, LP=False))
        assert ok is True and g["E32_RT"]["drops"][0]["lever"] == "LP" and g["E32_RT"]["drops"][0]["stage"] == "in-run"
        assert ab["extra"] == fresh[0]["extra"] and ab["levers"] == fresh[0]["levers"] and ab["kv_bytes"] == DPINS[0], (ab["extra"], fresh[0]["extra"])
        g3, _ = boot_ns(dict(OFF, IDX=True))
        ok3 = g3["e32_call"]("relaunch", False, Old(OOM[1])) or g3["e29_relaunch_plan"](Old(OOM[1]))
        assert ok3 is None and g3["NB5_ACTIVE_BOOT"]["kv_bytes"] == DPINS[1] and g3["NB5_ACTIVE_BOOT"]["extra"][-6:-4] == E32L.IDX_ARGS
        g4, _ = boot_ns(DEF)
        g4["e32_call"]("relaunch", False, Old("RuntimeError: worker died"))
        b5, fresh5, _ = ladder(E32_ENG, ["DONE"], dict(DEF, SPEC="K3"))
        return (g4["E32_RT"]["drops"][0]["lever"] == "SPEC" and g4["NB5_ACTIVE_BOOT"]["extra"] == fresh5[0]["extra"]
                and not g["E32_RT"]["errors"] and not g4["E32_RT"]["errors"])
    check("R1 in-run death with E32 levers on: one lever dropped (OOM -> LP; other -> SPEC), NB5_ACTIVE_BOOT rebuilt = exactly what "
          "nb5_boot builds fresh for the new lever set (argv, tail, seqs, env); a memory death with no memory lever = E29's in-run plan "
          "(the next D rung, IDX kept)", _r1)

    def _r2():
        g, seen = boot_ns(DEF)
        g["e29_mtp_boot"]("IN_RUN DFLASH_ZERO_ACCEPT test")
        ab = g["NB5_ACTIVE_BOOT"]
        assert ab["extra"] == ["--max-num-seqs", "24"] and ab["max_num_seqs"] == 24 and ab["kv_bytes"] == E29L.D_PIN0, ab
        assert not any(k.startswith(("DFLASH_", "VLLM_GDN", "VLLM_SPARSE", "VLLM_E2_QSA")) for k in ab["levers"]), ab["levers"]
        assert g["E32_RT"]["active"] is False and {d["lever"] for d in g["E32_RT"]["drops"]} == set(E32L.LEVERS)
        st = g["_e32x_states"]()
        return (all(v["state"] == "dropped" for v in st.values()) and g["e29_extra"](None) == [] and g["E29_RT"]["mtp_relaunch"] is True
                and not g["E32_RT"]["errors"])
    check("R2 in-run DFlash -> MTP (e29_mtp_boot, the zero-accept / XID / no-rung path): the booted config loses every E32 argv, the "
          "G1 / G2 env and the 28 seqs (24 = the E28 top rung) before E29 strips the tail; every lever recorded dropped WITH_DFLASH", _r2)

    # ------------------------------------------------------------------ S
    def _s1():
        g = ns(E32_ENG, DEF)
        calls = []

        def steady(port, prompts, n, secs=90, max_new=0, temperature=1.0, prefill_tok_s=700):
            calls.append((len(prompts[0]), n, max_new, round(prefill_tok_s)))
            return {"status": "STEADY", "tok_s": 500.0 + n, "mean_acceptance_length": 2.7, "preemptions": 0, "running_max": n}
        stub = dict(L=type("L", (), {"steady": staticmethod(steady)}), WIN=90.0, MAX_LEN=131072, PREFILL={"tok_s": 700.0},
                    prompt=lambda tok, v: [v] * tok, e29_counters=lambda: {"drafts": 0, "draft_tokens": 0, "accepted": 0})
        g.update(stub)
        trec = {"name": "t", "e29_probe": {"prompt": 130000, "sec": 21.2}}
        g["e32_call"]("speed", None, Srv(), trec)
        w = g["E32_RT"]["speed"]["windows"]
        assert [k for k in w] == ["8k@16", "20k@24", "90k@8", "116k@6"] and all(v["status"] == "STEADY" for v in w.values())
        assert calls == [(8192, 16, 131072 - 8192 - 512, 6132), (20000, 24, 131072 - 20000 - 512, 6132),
                         (90000, 8, 131072 - 90000 - 512, 6132), (116000, 6, 131072 - 116000 - 512, 6132)], calls
        g["e32_call"]("speed", None, Srv(), trec)
        assert len(calls) == 4
        g2 = ns(E32_ENG, dict(DEF, _SPEED={"on": True, "windows": [["8k", 16], ["20k", 24], ["90k", 8], ["116k", 6]], "cap_min": 1}))
        g2.update(stub)
        calls.clear()
        g2["e32_call"]("speed", None, Srv(), trec)
        w2 = g2["E32_RT"]["speed"]["windows"]
        assert w2["8k@16"]["status"] == "SKIPPED_CAP" and w2["20k@24"]["status"] == "SKIPPED_CAP" and w2["90k@8"]["status"] == "STEADY"
        assert w2["116k@6"]["status"] == "STEADY" and [c[0] for c in calls] == [90000, 116000]
        g3 = ns(E32_ENG, DEF)
        g3["RERUN"] = True
        g3["e32_call"]("speed", None, Srv(), trec)
        return g3["E32_RT"]["speed"]["status"] == "SKIPPED_RERUN" and not g["E32_RT"]["errors"] and not g2["E32_RT"]["errors"]
    check("S1 the speed pass (stub L.steady): 8k@16, 20k@24, 90k@8, 116k@6 in order, prompts of the band size, max_new = 131,072 - "
          "band - 512, the measured F2 prefill rate; once per run; a 1-min cap skips 8k / 20k but never 90k / 116k (user rule); the "
          "hidden rerun skips it", _s1)

    def _s2():
        t = ("non-default args: {'max_num_batched_tokens': 2048, 'long_prefill_token_threshold': 1024, 'long_prefill_token_threshold_adaptive': "
             "True, 'enable_mamba_shared_prefix_checkpoint': True, 'attention_config': {'indexer_kv_dtype': 'fp8'}}")
        rb = E32L.readback(DEF, t, {"max_num_seqs": 28, "speculative_config": {"num_speculative_tokens": 4,
                                                                              "num_speculative_tokens_per_batch_size": [[1, 8, 4]]}},
                           [{"env": {"VLLM_GDN_DECODE_FASTPATH": "1", "VLLM_GDN_SHARED_ALIGN_INDEX": "1"}}] * 4, None)
        assert rb["applied"] == {k: True for k in E32L.LEVERS} and not rb["real_failure"], rb
        rb2 = E32L.readback(DEF, t + "\nWARNING Disabling num_speculative_tokens_per_batch_size", {}, [], None)
        assert rb2["real_failure"] == "E32_DYN_FALLBACK" and rb2["applied"]["G1"] is None
        rb3 = E32L.readback(OFF, "", {}, [], None)
        assert rb3["real_failure"] is None and rb3["applied"]["LP"] is None
        g = ns(E32_ENG, DEF)

        class S(Srv):
            proc = type("p", (), {"pid": -1})

            def log(self):
                return "Overriding cudagraph_mode to PIECEWISE"
        g.update(rpc=lambda *a: None, MERLIN_ENG={"kv": {"MERLIN_TREE": "/nonexistent"}}, hashlib=__import__("hashlib"), Path=Path)
        trec = {"name": "t", "config_readback": {"log": {}}}
        return (g["e32_call"]("readback", False, S(), trec) is True and trec["status"] == "E32_DYN_FALLBACK" and g["E32_RT"]["fingerprints"]
                and not g["E32_RT"]["errors"])
    check("S2 the read-back: every lever's own logged / worker value (record: applied True / False / None); only the DYN fallback lines "
          "are a real failure (the notebook hook sets E32_DYN_FALLBACK and stops the try); a fingerprint per booted try", _s2)

    def _s3():
        st = {"SEQS": "28 on"}
        a = E32L.fingerprint(["a", "--x", "1"], {"VLLM_A": "1", "HOME": "/r", "DFLASH_SEAM": "1"}, 7665090560, {"commit_pin": "c"}, st)
        b = E32L.fingerprint(["a", "--x", "1"], {"DFLASH_SEAM": "1", "VLLM_A": "1", "PATH": "/x"}, 7665090560, {"commit_pin": "c"}, st)
        c = E32L.fingerprint(["a", "--x", "2"], {"VLLM_A": "1"}, 7665090560, {"commit_pin": "c"}, st)
        g = ns(E32_ENG, DEF)
        g.update(MERLIN_ENG={"kv": {"MERLIN_TREE": "/nonexistent"}}, hashlib=__import__("hashlib"), Path=Path, E11_SERVER=Srv())
        g["_e32x_patch_effective"]()
        try:
            eff = g["E29L"].effective({"on": ["df"], "cfg": {"embed_host": True}})
        finally:
            g["_e32x_rollback"]()
        return (a["sha256"] == b["sha256"] != c["sha256"] and set(a["env"]) == {"VLLM_A", "DFLASH_SEAM"} and len(a["short"]) == 12
                and eff["e32"]["levers"]["SPEC"] == "DYN on" and eff["e32"]["fp"] and E29L.effective is getattr(E32L, "_e32_effective_orig"))
    check("S3 the engine fingerprint: deterministic (env key order / non-engine keys do not matter), changes with the argv; the "
          "effective() line (LIVE start / 15-min / report) carries the E32 lever states + the short fingerprint; the rollback restores it",
          _s3)

    # ------------------------------------------------------------------ K
    def _k1():
        a32, a29 = all_cells(X["E32_NB"]), all_cells(X["E29_NB"])
        same = [i for i in range(len(a29)) if a29[i] == a32[i]]
        assert len(a32) == len(a29) == 9 and same == [1, 3, 4, 6], same
        line = next(l for l in a32[2].splitlines() if l.startswith("E29D_FILES = "))
        assert line in a29[2].splitlines()
        assert X["MODS"] == modules_literal(X["C29"][1]), "the E28 / E29 modules literal differs from E29's"
        nbx_ok = __import__("hashlib").sha256(NBX.encode("utf-8")).hexdigest() == X["E32_FILES"]["nbx"][1]
        return len(X["E32_NB"]) < 400_000 and b"\r" not in X["E32_NB"].replace(b"\\r", b"") and nbx_ok
    check("K1 identity vs E29_DFLASH_D3S: 9 cells; CELL 0 (dual mode + fallback), CELL 3 (scheduler apply + DIRECT diff), cell 4, CELL "
          "6/7 (Duck run, the E15B writer, the clock) byte-identical; the E29D_FILES pins and the E28 / E29 modules literal = E29's; the "
          "e32_nbx text = the notebook's pin; < 400,000 B; no CR", _k1)
