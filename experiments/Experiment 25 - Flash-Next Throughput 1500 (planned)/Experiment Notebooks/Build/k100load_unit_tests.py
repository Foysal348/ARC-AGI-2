"""E25 klee100 load test: quick unit tests (seconds; local and in the Kaggle units kit, Py3.12).

  T1 model dir + K6           build_model_dir / k6_check: complete dir OK; one FP8 PLE link removed -> K6 fails; a dangling
                              link, the BF16 PLE shard in the index, a config without the FP8 PLE dtype -> fail
  T2 K6 before launch         the notebook's own merlin_fn_server (picked from the export) raises K6 before the RC2 / NB1
                              server function is called when one indexed shard is missing; calls it when complete
  T3 digests                  digest_plan on the real v2 manifest (23 files, the BF16 PLE shard excluded); verify_digests:
                              OK / one flipped byte / size / missing / threads
  T4 pin ladder + K2 pin      pin_ladder arithmetic (safety, PP1 reserve, fracs, align, min gain, fail closed), free_by_card,
                              k2_pin_from_k (same / lowered / floor / TF_OOM lower step)
  T5 TF comparison            NB0's own cmp_last256 / cmp_full / shift_check (picked from the export) + the lab's compare_tf on
                              NB0's REAL arrays: A/A = 0; A vs B (FP8 KV) reproduces NB0's recorded numbers when the local
                              NB0 output is present; synthetic shifts; prompt mismatch / NaN -> INCONCLUSIVE
  T6 labels                   tf_label / yardstick / tf_reading / final_label (never an accuracy PASS)
  T7 capacity + argv          the export's k100_capacity (RC2 FP8 numbers: P 149 MATCH, F 6), NB0's nb0_argv on RC2's argv
  T8 build facts              the embedded PIN = the dataset SHA256SUMS + the v2 overlay shas; notebook < 400 kB; RERUN guard
  T9  K-1 boot budget          tf_suite / sc_suite with boot_left 0 and left 4000 -> zero TF / decode HTTP calls
  T10 K-4 / K-6 / K-9 main     CELL 8 run on stubs: missing / corrupt reference, prompt identity, oracle identity -> zero
                              installs and boots; K / K2 launch OOM -> next lower pin; Xid -> stop; no budget -> SKIPPED_BUDGET;
                              unverified K headroom -> K2 on the base pin; --help missing a flag -> FAIL_CLI_FLAGS
  T11 K-3 capacity gate       body_k2 on a MISMATCH pool: 8k@8 only, no 40k@N, no admission; MATCH: all three
  T12 K-7 method map          method_map_check (48 / 47 layers, bits, conflicts, MTP), the embedded worker extension's
                              k100_method_map on a fake model, k100_facts: 47/48 -> config not clean on K, recorded on K0
  T13 K-8 traces              traced() writes every stream's full text / ids / finish after the window; LAB_SRC unchanged
  T14 K-9 --help              merlin_fn_server: a missing flag -> FAIL_CLI_FLAGS before launch; help error -> launch continues
  T15 K-10 K2 reserve         body_k: the 120k row and the 90k TF -> SKIPPED_K2_RESERVE when left - reserve is short
  T16 K-12 arithmetic         8 exact-answer cases, exact-match scoring, errors recorded (never gated)
  T25 rev 3 K3_ONLY GPU v1    the v1 failure rows (every layer HummingMoEMethod / None / None, MTP {}) replayed: bits no longer
                              fail (MTP missing alone does), + the drafter rows -> METHOD_CLASS_ONLY (not CONFIG_MISMATCH); real
                              class / bit / partial / MTP mismatches stay fatal; k100_facts keeps the raw rows; labels, predicates
                              (T12 rev 3: the extension on Merlin-shaped objects, the V2 runner's drafter, bit-source conflicts)
Fixtures (K-13): resolved relative to the repo root (this file's folder, 4 levels up) or env K100_TEST_*; a missing
fixture is a SKIP with its reason (printed and counted), never a FAIL. The lab is the export's own LAB_SRC.
Run: python k100load_unit_tests.py   (prints 'N/M PASS' and 'K skipped')
"""
import ast
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import statistics
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k100load_lib as KL  # noqa: E402

REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE     # K-13: <repo>/experiments/<exp>/Experiment Notebooks/Build


def fixture(env, rel):
    return Path(os.environ.get(env) or (REPO / rel))


EXPORT = Path(os.environ.get("K100_TEST_EXPORT") or HERE / "k100load_code_export.py")
REF = fixture("K100_TEST_REF", "_scratch/push_e25/ds_e25_nb0_tf_ref")
LAB_FILE = fixture("K100_TEST_LAB", "experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/src/e23_lab.py")   # T13 only
DIGESTS = fixture("K100_TEST_DIGESTS", "_scratch/e25_kits_out_klee_v2/q27-e25-klee100-cpu-kit/klee100_shard_digests.json")
NB0_RAW = fixture("K100_TEST_NB0_RAW", "_scratch/e25_gpu_out/q27-xhigh-e25-nb0-fp8-quality/nb0/tf_raw")
NB_PATH = Path(os.environ.get("K100_TEST_NB") or HERE.parent / "Q27-XHIGH-E25_KLEE100_LOAD.ipynb")
TMP = Path(tempfile.mkdtemp(prefix="k100u_"))
RES, SKIPS = [], []


def check(name, cond, info=""):
    RES.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  | {info}"), flush=True)


def skip(name, why):
    SKIPS.append((name, why))
    print(f"SKIP {name}: {why}", flush=True)


def have(path, name):
    """K-13: True when the fixture exists, else a SKIP with the reason."""
    if Path(path).exists():
        return True
    skip(name, f"fixture not present: {path}")
    return False


class SkipTest(Exception):
    pass


def run(name, fn):
    try:
        fn()
    except SkipTest as e:
        skip(name, str(e))
    except Exception as e:
        RES.append((name, False))
        print(f"FAIL {name} raised {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- export helpers: definitions inside `if not RERUN:` blocks
EXPORT_SRC = EXPORT.read_text(encoding="utf-8")
_TREE = ast.parse(EXPORT_SRC)


def defs(name):
    """Every (def / class / assignment) node defining `name` at module level or one level inside an `if` block."""
    out = []
    for node in _TREE.body:
        for n in ([node] + (node.body if isinstance(node, ast.If) else [])):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
                out.append(n)
            elif isinstance(n, ast.Assign) and name in {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}:
                out.append(n)
    return out


def src_of(name, which=-1):
    d = defs(name)
    assert d, f"{name} not in the export"
    return textwrap_dedent(ast.get_source_segment(EXPORT_SRC, d[which]))


def textwrap_dedent(s):
    import textwrap
    first = s.splitlines()[0]
    ind = len(first) - len(first.lstrip())
    return textwrap.dedent((" " * ind) + s) if ind == 0 else textwrap.dedent(s)


def load(names, ns, which=None):
    for n in names:
        if n == "K100":                    # CELL 7: the dict, then K100["k2_reserve_sec"] = ... (every statement, in order)
            for d in defs("K100"):
                exec(compile(textwrap_dedent(ast.get_source_segment(EXPORT_SRC, d)), "<export K100>", "exec"), ns)
            continue
        code = src_of(n, (which or {}).get(n, -1))
        exec(compile(code, f"<export {n}>", "exec"), ns)
    return ns


PAYLOAD = fixture("K100_TEST_PAYLOAD", "_scratch/push_e25/ds_e25_nb0_tf_ref")      # V-13a: repo-relative (the dataset copy)


def export_value(name):
    """The value of an assignment in the export, evaluated (packed lzma / b85 constants included: RC2_REF, NB1_EXT_SRC)."""
    node = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)][-1]
    return eval(compile(ast.Expression(node.value), f"<export {name}>", "eval"), {"json": json, "__import__": __import__})


def export_const(name):
    """A string constant assigned in the export (any depth), e.g. K100_LAB_SHA."""
    node = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                                          for t in n.targets)][-1]
    return ast.literal_eval(node.value)


def lab_src():
    """R1-18: the lab the notebook loads = k100load_lab.py of the payload (the dataset copy in the kit), sha-checked
    against the export's K100_LAB_SHA."""
    raw = (PAYLOAD / "k100load_lab.py").read_bytes()
    assert sha(raw) == export_const("K100_LAB_SHA"), "payload lab sha != the export's K100_LAB_SHA"
    return raw.decode("utf-8")


def lab():
    p = TMP / "e23_lab_from_export.py"
    if not p.exists():
        p.write_text(lab_src(), encoding="utf-8")
    return types.SimpleNamespace(**runpy.run_path(str(p), run_name="e23_lab_test"))


# ======================================================================================== T1 model dir + K6
def mk_fake():
    base = TMP / "t1"
    shutil.rmtree(base, ignore_errors=True)
    k, p, o = base / "k100", base / "ple", base / "ovl"
    for d in (k, p, o):
        d.mkdir(parents=True)
    for n in ("model-00002-of-00011.safetensors", "mtp-model-00001-of-00002.safetensors", "tokenizer.json", "KAGGLE_MIRROR_MANIFEST.json"):
        (k / n).write_bytes(n.encode() * 3)
    (k / "config.json").write_text(json.dumps({"text_config": {}}))                      # the published config (replaced)
    (k / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-00001-of-00011.safetensors"}}))
    for n in ("model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"):
        (p / n).write_bytes(n.encode())
    (o / "config.json").write_text(json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}))
    (o / "quantization_config.json").write_text("{}")
    (o / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00002-of-00011.safetensors", "b": "mtp-model-00001-of-00002.safetensors",
        "c": "model-plefp8-00000.safetensors", "d": "model-plefp8-00001.safetensors"}}))
    return base, k, p, o


def _link():
    """os.symlink when this host allows it (Linux / Kaggle), else a copy (a Windows host without the privilege)."""
    t = TMP / "linkprobe"
    try:
        (TMP / "linksrc").write_text("x")
        os.symlink(str(TMP / "linksrc"), str(t))
        return os.symlink, True
    except OSError:
        return (lambda a, b: shutil.copyfile(a, b)), False


LINK, SYMLINKS = _link()


def t1():
    base, k, p, o = mk_fake()
    d = base / "ckpt"
    md = KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
    check("T1 build_model_dir ok", md["ok"] and len(md["ple_links"]) == 2 and len(md["klee100_links"]) == 4, md)
    check("T1 overlay replaces the published config / index",
          json.loads((d / "config.json").read_text())["text_config"].get("ple_embedding_dtype") == "float8_e4m3fn"
          and "c" in json.loads((d / "model.safetensors.index.json").read_text())["weight_map"])
    k6 = KL.k6_check(d)
    check("T1 K6 complete dir ok", k6["ok"] and k6["index_files"] == 4 and not k6["missing"], k6)
    (d / "model-plefp8-00001.safetensors").unlink()
    k6 = KL.k6_check(d)
    check("T1 K6 one PLE link removed -> fail", not k6["ok"] and k6["missing"] == ["model-plefp8-00001.safetensors"]
          and k6["why"].startswith("K6_INDEXED_SHARD_MISSING"), k6)
    if SYMLINKS:
        md = KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
        (p / "model-plefp8-00000.safetensors").rename(p / "gone")
        k6 = KL.k6_check(d)
        check("T1 K6 dangling symlink -> missing", not k6["ok"] and "model-plefp8-00000.safetensors" in k6["missing"], k6)
        (p / "gone").rename(p / "model-plefp8-00000.safetensors")
    else:
        check("T1 K6 dangling symlink (host has no symlink privilege: copy mode)", True)
    md = KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    idx["weight_map"]["e"] = "model-00001-of-00011.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    (d / "model-00001-of-00011.safetensors").write_bytes(b"x")
    k6 = KL.k6_check(d)
    check("T1 K6 index naming the BF16 PLE shard -> fail", not k6["ok"] and k6["bf16_ple_referenced"], k6)
    md = KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
    (d / "config.json").write_text(json.dumps({"text_config": {}}))
    k6 = KL.k6_check(d)
    check("T1 K6 config without the FP8 PLE dtype -> fail", not k6["ok"] and "ple_embedding_dtype" in k6["why"], k6)
    md = KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00009.safetensors"], link=LINK)
    check("T1 build_model_dir missing PLE source -> not ok", not md["ok"] and any("00009" in e for e in md["errors"]), md)
    check("T1 rebuild removes stale entries", not (d / "model-00001-of-00011.safetensors").exists())


# ======================================================================================== T2 K6 before launch (the export's code)
def t2():
    base, k, p, o = mk_fake()
    d = base / "ckpt2"
    KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
    calls = []

    class S:
        argv = ["vllm", "serve", "--speculative-config", "{}", "--max-num-seqs", "3", "--compilation-config", "{}",
                "--enable-prefix-caching", "--kv-cache-memory-bytes", "123"]
        env = {"VLLM_X": "1"}
    ns = {"KL": KL, "CKPT": d, "K1": {}, "K1BOOTS": {"n": {"key": "K0", "name": "n", "mtp": False, "seqs": 8, "captures": "{\"c\":1}",
                                                          "title": "t"}},
          "save": lambda: None, "R": {"setup": {}}, "os": os, "time": __import__("time"), "threading": __import__("threading"),
          "json": json, "ENV_KEYS": ("VLLM_",), "K100": {"ready_sec": 10}, "left": lambda: 1e6,
          "config_diff": lambda *a: {"stub": True}, "load_monitor": lambda *a: None, "HELP": {}, "LIVE_PIN": {},
          "L": types.SimpleNamespace(help_check=lambda key, pre, env, argv: {"ran": True, "missing": []}),
          "_merlin_fn_server_nb1": lambda *a: (calls.append(a), S())[1]}
    os.environ["RC2_CPU_KIT"] = "1"
    load(["flags", "_set_value", "nb0_argv"], ns)
    load(["merlin_fn_server"], ns)                     # the LAST definition = CELL 7's K6-guarded rebinding
    s = ns["merlin_fn_server"]("n", "bf16", 8020, 123)
    check("T2 complete dir: server function called once, MTP off removes --speculative-config, seqs 8",
          len(calls) == 1 and "--speculative-config" not in s.argv and s.argv[s.argv.index("--max-num-seqs") + 1] == "8"
          and s.argv[s.argv.index("--compilation-config") + 1] == "{\"c\":1}", s.argv)
    (d / "model-plefp8-00000.safetensors").unlink()
    try:
        ns["merlin_fn_server"]("n", "bf16", 8020, 123)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    check("T2 one PLE link removed: K6 raises before launch, server function NOT called",
          raised is not None and raised.startswith("K6_INDEXED_SHARD_MISSING") and len(calls) == 1, raised)
    check("T2 K6 recorded per boot", ns["K1"]["k6_before_launch"]["n"]["ok"] is False)


# ======================================================================================== T3 digests
def t3():
    if not have(DIGESTS, "T3 real digest manifest"):
        return
    man = json.loads(DIGESTS.read_text())
    plan = KL.digest_plan(man, "/k", "/p")
    names = [x[0] for x in plan]
    check("T3 real manifest plan: 12 klee100 shards + tokenizer.json + 10 FP8 PLE, BF16 PLE excluded",
          len(plan) == 23 and KL.BF16_PLE_SHARD not in names and "tokenizer.json" in names
          and sum(n.startswith("model-plefp8-") for n in names) == 10, names)
    check("T3 plan bytes ~101.8 GB (50.6 klee100 + 51.2 FP8 PLE)", 101e9 < sum(x[2] for x in plan) < 103e9, sum(x[2] for x in plan))
    d = TMP / "t3"
    d.mkdir(exist_ok=True)
    files = {}
    for i in range(5):
        b = os.urandom(3_000_000 + i)
        (d / f"f{i}").write_bytes(b)
        files[f"f{i}"] = b
    fake = [(n, str(d / n), len(b), sha(b), "t") for n, b in files.items()]
    r = KL.verify_digests(fake, threads=3)
    check("T3 verify all OK", r["ok"] and r["n_checked"] == 5 and not r["bad"], r)
    b = bytearray(files["f2"])
    b[12345] ^= 1
    (d / "f2").write_bytes(bytes(b))
    r = KL.verify_digests(fake, threads=3)
    check("T3 one flipped byte -> mismatch", not r["ok"] and r["bad"] == ["f2"] and "sha256" in r["files"]["f2"]["why"], r["files"]["f2"])
    (d / "f2").write_bytes(files["f2"] + b"x")
    r = KL.verify_digests(fake, threads=3)
    check("T3 size difference -> fail before hashing", not r["ok"] and r["files"]["f2"]["why"].startswith("size"), r["files"]["f2"])
    (d / "f2").unlink()
    r = KL.verify_digests(fake, threads=3)
    check("T3 missing file -> fail", not r["ok"] and r["files"]["f2"]["why"] == "missing")
    check("T3 empty plan -> not ok", KL.verify_digests([], threads=2)["ok"] is False)


# ======================================================================================== T4 pin ladder + K2 pin
def t4():
    MiB, base = 2 ** 20, 2464415744
    free = {0: 7000.0, 1: 7010.0, 2: 6800.0, 3: 6810.0}
    lad = KL.pin_ladder(free, base, (0.9, 0.8, 0.7), 1024, 1192)
    head = {0: 7000 - 1024, 1: 7010 - 1024, 2: 6800 - 1024 - 1192, 3: 6810 - 1024 - 1192}
    want = [int((base + int(f * min(head.values()) * MiB)) // (2 * MiB) * (2 * MiB)) for f in (0.9, 0.8, 0.7)]
    check("T4 ladder: min headroom on PP1 (reserve) card 2", lad["ok"] and lad["binding_card"] == 2
          and lad["headroom_min_mib"] == min(head.values()), lad)
    check("T4 ladder steps exact + descending + 2 MiB aligned", [s["kv_bytes"] for s in lad["steps"]] == want
          and all(s["kv_bytes"] % (2 * MiB) == 0 for s in lad["steps"]) and want[0] > want[1] > want[2] > base, lad["steps"])
    check("T4 ladder: missing card -> fail closed", not KL.pin_ladder({0: 7000, 1: 7000, 2: 7000}, base)["ok"])
    check("T4 ladder: NaN -> fail closed", not KL.pin_ladder({0: 7000, 1: float("nan"), 2: 7000, 3: 7000}, base)["ok"])
    small = KL.pin_ladder({0: 1300, 1: 1300, 2: 2600, 3: 2600}, base, safety_mib=1024, reserve_mib_pp1=1192)
    check("T4 ladder: headroom < min gain -> no steps", not small["ok"] and small["steps"] == [], small)
    check("T4 stage_of_card", [KL.stage_of_card(i) for i in range(4)] == [0, 0, 1, 1])
    fb = KL.free_by_card([[[0, 100, 23034], [1, 200, 23034]], None, [[0, 300, 23034], [1, 50, 23034]]])
    check("T4 free_by_card takes the highest use", fb == {0: 23034 - 300, 1: 23034 - 200}, fb)
    pin = want[0]
    ok = KL.k2_pin_from_k({0: 2000, 1: 2000, 2: 2000, 3: 2000}, pin, 1192, 512, base)
    check("T4 k2 pin: K free covers the reserve -> same pin", ok["kv_bytes"] == pin and ok["rule"] == "SAME_PIN_AS_K", ok)
    lo = KL.k2_pin_from_k({0: 2000, 1: 2000, 2: 1500, 3: 1600}, pin, 1192, 512, base)
    check("T4 k2 pin: deficit 204 MiB on card 2 -> lowered by >= 204 MiB, aligned",
          lo["rule"].startswith("LOWERED") and pin - lo["kv_bytes"] >= 204 * MiB and pin - lo["kv_bytes"] < 207 * MiB
          and lo["kv_bytes"] % (2 * MiB) == 0, lo)
    fl = KL.k2_pin_from_k({0: 0, 1: 0, 2: 0, 3: 0}, base + 10 * MiB, 1192, 512, base)
    check("T4 k2 pin: never below the floor", fl["kv_bytes"] == base, fl)
    tf = KL.k2_pin_from_k({0: 2000, 1: 2000, 2: 2000, 3: 2000}, pin, 1192, 512, base, lower_pin=want[1])
    check("T4 k2 pin: K TF_OOM -> the next lower step", tf["kv_bytes"] == want[1] and tf["rule"].startswith("K_TF_OOM"), tf)
    check("T4 k2 pin: valid reading -> headroom_verified True", ok["headroom_verified"] is True and lo["headroom_verified"] is True)
    # K-2: missing / partial / NaN / negative / wrong card ids -> the base BF16 pin, never K's pin
    for lab_, fm in (("None", None), ("empty", {}), ("partial 3 cards", {0: 9000, 1: 9000, 2: 9000}),
                     ("NaN", {0: 9000, 1: float("nan"), 2: 9000, 3: 9000}), ("inf", {0: 9000, 1: float("inf"), 2: 9000, 3: 9000}),
                     ("negative", {0: 9000, 1: -5, 2: 9000, 3: 9000}), ("cards 1-4", {1: 9000, 2: 9000, 3: 9000, 4: 9000}),
                     ("None value", {0: 9000, 1: None, 2: 9000, 3: 9000}), ("list", [9000, 9000, 9000, 9000])):
        r = KL.k2_pin_from_k(fm, pin, 1192, 512, base)
        check(f"T4 K-2 k2 pin: free {lab_} -> base pin, FALLBACK rule, headroom_verified False",
              r["kv_bytes"] == base and r["kv_bytes"] != pin and r["rule"] == "FALLBACK_BASE_PIN_UNVERIFIED_HEADROOM"
              and r["headroom_verified"] is False, r)
    r = KL.k2_pin_from_k({0: 9000, 1: 9000, 2: 9000, 3: 9000}, None, 1192, 512, base)
    check("T4 K-2 k2 pin: no K pin -> base pin", r["kv_bytes"] == base and r["headroom_verified"] is False, r)
    r = KL.k2_pin_from_k({"0": 2000, "1": 2000, "2": 2000, "3": 2000}, pin, 1192, 512, base)
    check("T4 K-2 k2 pin: string card keys (a JSON round trip) still valid", r["rule"] == "SAME_PIN_AS_K" and r["kv_bytes"] == pin, r)
    # K-6: K2's pins and the ladder
    p2 = base + 2048 * MiB
    check("T4 K-6 k2_pins = [pin2, pin2 - 512 MiB, base]", KL.k2_pins(p2, base) == [p2, p2 - 512 * MiB, base])
    check("T4 K-6 k2_pins: pin2 - 512 MiB below base -> clamped, deduped", KL.k2_pins(base + 100 * MiB, base) == [base + 100 * MiB, base]
          and KL.k2_pins(base, base) == [base])
    seen = []
    oom = {"status": "LAUNCH_FAILED", "oom_at_launch": True}
    rec, tries = KL.ladder_run([3, 2, 1], lambda p, i: (seen.append(p), oom if p == 3 else {"status": "DONE", "boot_name": f"b{p}"})[1])
    check("T4 K-6 ladder: launch OOM -> next lower pin, then stop", seen == [3, 2] and rec["status"] == "DONE"
          and [t["stop"] for t in tries] == ["NEXT_PIN", "NOT_A_LAUNCH_OOM"] and tries[1]["name"] == "b2", tries)
    seen = []
    rec, tries = KL.ladder_run([3, 2, 1], lambda p, i: (seen.append(p), dict(oom, device_fault=True))[1])
    check("T4 K-6 ladder: Xid / device fault -> stop at once", seen == [3] and tries[0]["stop"] == "DEVICE_FAULT", tries)
    seen = []
    rec, tries = KL.ladder_run([3, 2, 1], lambda p, i: (seen.append(p), oom if p == 3 else {"status": "SKIPPED_BUDGET"})[1])
    check("T4 K-6 ladder: no budget for the next try -> SKIPPED_BUDGET, stop", seen == [3, 2] and rec["status"] == "SKIPPED_BUDGET"
          and tries[-1]["stop"] == "SKIPPED_BUDGET", tries)
    seen = []
    rec, tries = KL.ladder_run([3, 2], lambda p, i: (seen.append(p), oom)[1])
    check("T4 K-6 ladder: every pin OOMs -> LADDER_EXHAUSTED", seen == [3, 2] and tries[-1]["stop"] == "LADDER_EXHAUSTED", tries)
    rec, tries = KL.ladder_run([3, 2], lambda p, i: {"status": "LAUNCH_FAILED", "oom_at_launch": False})
    check("T4 K-6 ladder: launch failure without OOM -> stop", len(tries) == 1 and tries[0]["stop"] == "NOT_A_LAUNCH_OOM", tries)
    rec, tries = KL.ladder_run([3], lambda p, i: {"status": "TF_OOM", "oom_in_body": True})
    check("T4 K-6 ladder: body OOM recorded apart from launch OOM", tries[0]["oom_in_body"] is True and tries[0]["oom_at_launch"] is False, tries)


# ======================================================================================== T5 TF comparison (NB0's own code)
def tf_ns():
    if not (PAYLOAD / "k100load_lab.py").is_file():
        raise SkipTest(f"lab payload not present: {PAYLOAD / 'k100load_lab.py'}")
    ns = {"np": np, "math": math, "json": json, "hashlib": hashlib, "statistics": statistics, "L": lab(),
          "NB0": {"last": 256}, "SCREEN": KL.SCREEN}
    load(["_finite", "ids_sha", "e23_rows", "cmp_last256", "chunk_check", "cmp_full", "shift_check", "screen"], ns)
    return ns


def rec(path, ns):
    z = np.load(path)
    a = {k: z[k] for k in z.files}
    return {"meta": {"prompt_sha256": ns["ids_sha"](a["ids"])}, "arr": a}


def t5():
    if not have(REF / "nb0_reference.json", "T5 NB0 reference arrays"):
        return
    ns = tf_ns()
    meta = json.loads((REF / "nb0_reference.json").read_text())
    for band in ("2k", "8k", "40k", "90k"):
        f, r = rec(REF / f"A_{band}_fresh.npz", ns), rec(REF / f"A_{band}_repeat.npz", ns)
        full, last = ns["cmp_full"](f["arr"], r["arr"]), ns["cmp_last256"](f, r)
        check(f"T5 A/A {band} (SELF-IDENTITY: A_fresh = A_repeat byte-identical): 0 / 1.0 -> LOWBIT_TF_CLOSE",
              full["mean_abs_dlogprob"] == 0 and full["rank1_agreement"] == 1.0 and last["status"] == "MEASURED"
              and KL.tf_label(full) == "LOWBIT_TF_CLOSE" and f["meta"]["prompt_sha256"] == meta["tf"]["A"][band]["prompt_sha256"],
              (full.get("mean_abs_dlogprob"), full.get("rank1_agreement"), last.get("status")))
    a40 = rec(REF / "A_40k_fresh.npz", ns)
    c40 = rec(REF / "C_40k_fresh.npz", ns)
    ac = ns["cmp_full"](a40["arr"], c40["arr"])
    check("T5 A vs C at 40k = 0 (SELF-IDENTITY: C_fresh = A_fresh byte-identical)", ac["mean_abs_dlogprob"] == 0 and ac["rank1_agreement"] == 1.0, ac)
    yAB = meta["yardstick"]["pairs"]["AB"]["bands"]
    if (NB0_RAW / "B_40k_fresh.npz").is_file():
        for band in ("2k", "8k", "40k", "90k"):
            b = rec(NB0_RAW / f"B_{band}_fresh.npz", ns)
            a = rec(REF / f"A_{band}_fresh.npz", ns)
            full, last = ns["cmp_full"](a["arr"], b["arr"]), ns["cmp_last256"](a, b)
            yf, yl = yAB[band]["full"], yAB[band]["last256"]
            check(f"T5 A vs B (FP8 KV) {band} reproduces NB0's recorded full + last256",
                  all(full[k] == yf[k] for k in ("mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement"))
                  and all(last[k] == yl[k] for k in ("mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")),
                  ({k: full[k] for k in ("mean_abs_dlogprob", "rank1_agreement")}, yf))
        b40 = rec(NB0_RAW / "B_40k_fresh.npz", ns)
        full = ns["cmp_full"](a40["arr"], b40["arr"])
        check("T5 the FP8-KV 40k divergence labels DIVERGENT (full) / SEVERE (last256 rank-1 0.367)",
              KL.tf_label(full) == "LOWBIT_TF_DIVERGENT" and KL.tf_label(ns["cmp_last256"](a40, b40)) == "LOWBIT_TF_DIVERGENT_SEVERE")
    else:
        skip("T5 A vs B reproduction", f"local NB0 output not present: {NB0_RAW}")
    syn = {k: v.copy() for k, v in a40["arr"].items()}
    syn["lp"] = syn["lp"] + np.float32(0.05)
    full = ns["cmp_full"](a40["arr"], syn)
    check("T5 synthetic +0.05 logprob -> mean 0.05, CLOSE", abs(full["mean_abs_dlogprob"] - 0.05) < 1e-3 and KL.tf_label(full) == "LOWBIT_TF_CLOSE", full)
    syn2 = {k: v.copy() for k, v in a40["arr"].items()}
    idx = np.arange(1, len(syn2["top1"]))[::5]
    syn2["top1"][idx] = syn2["top1"][idx] + 1
    full = ns["cmp_full"](a40["arr"], syn2)
    check("T5 synthetic rank-1 changed on 20 % -> DIVERGENT", 0.79 < full["rank1_agreement"] < 0.81 and KL.tf_label(full) == "LOWBIT_TF_DIVERGENT", full)
    syn3 = {k: v.copy() for k, v in a40["arr"].items()}
    syn3["ids"][100] += 1
    check("T5 different prompt ids -> INCONCLUSIVE_PROMPT_MISMATCH", ns["cmp_full"](a40["arr"], syn3)["status"] == "INCONCLUSIVE_PROMPT_MISMATCH"
          and KL.tf_label(ns["cmp_full"](a40["arr"], syn3)) == "INCONCLUSIVE_PROMPT_MISMATCH")
    syn4 = {k: v.copy() for k, v in a40["arr"].items()}
    syn4["lp"][500] = np.nan
    check("T5 NaN in a valid position -> INCONCLUSIVE_NONFINITE", ns["cmp_full"](a40["arr"], syn4)["status"] == "INCONCLUSIVE_NONFINITE")
    sh = ns["shift_check"](a40["arr"], a40["arr"])
    check("T5 shift check: identical arrays best shift 0, no artefact", sh["best_shift"] == 0 and not sh["alignment_artefact"], sh)


# ======================================================================================== T6 labels
def t6():
    L_ = KL.tf_label
    check("T6 tf_label edges", L_({"status": "MEASURED", "mean_abs_dlogprob": 0.15, "rank1_agreement": 0.90}) == "LOWBIT_TF_CLOSE"
          and L_({"status": "MEASURED", "mean_abs_dlogprob": 0.151, "rank1_agreement": 0.95}) == "LOWBIT_TF_DIVERGENT"
          and L_({"status": "MEASURED", "mean_abs_dlogprob": 0.10, "rank1_agreement": 0.899}) == "LOWBIT_TF_DIVERGENT"
          and L_({"status": "MEASURED", "mean_abs_dlogprob": 1.01, "rank1_agreement": 0.95}) == "LOWBIT_TF_DIVERGENT_SEVERE"
          and L_({"status": "MEASURED", "mean_abs_dlogprob": 0.2, "rank1_agreement": 0.49}) == "LOWBIT_TF_DIVERGENT_SEVERE"
          and L_({"status": "MEASURED", "mean_abs_dlogprob": float("nan"), "rank1_agreement": 0.9}) == "INCONCLUSIVE_NONFINITE"
          and L_(None) == "INCONCLUSIVE_MISSING" and L_({"status": "INCONCLUSIVE_MISSING_TF"}) == "INCONCLUSIVE_MISSING_TF")
    # K-5: PARTIAL / n != expected are never a reading
    good = {"mean_abs_dlogprob": 0.01, "rank1_agreement": 0.99}
    check("T6 K-5 tf_label: PARTIAL -> INCONCLUSIVE_PARTIAL (was CLOSE)", L_(dict(good, status="PARTIAL", n=900, expected=1000)) == "INCONCLUSIVE_PARTIAL")
    check("T6 K-5 tf_label: MEASURED with n != expected -> INCONCLUSIVE_INCOMPLETE_POSITIONS",
          L_(dict(good, status="MEASURED", n=999, expected=1000)) == "INCONCLUSIVE_INCOMPLETE_POSITIONS")
    check("T6 K-5 tf_label: n == expected -> CLOSE; expected unknown (last256) -> CLOSE",
          L_(dict(good, status="MEASURED", n=1000, expected=1000)) == "LOWBIT_TF_CLOSE" and L_(dict(good, status="MEASURED", n=256)) == "LOWBIT_TF_CLOSE")
    y = {"mean_abs_dlogprob": 0.93, "rank1_agreement": 0.75, "p99_abs_dlogprob": 6.2}
    check("T6 yardstick BELOW / ABOVE / MIXED / INCONCLUSIVE",
          KL.yardstick({"mean_abs_dlogprob": 0.3, "rank1_agreement": 0.9}, y)["vs_nb0_fp8_kv"] == "BELOW_NB0_FP8_KV"
          and KL.yardstick({"mean_abs_dlogprob": 1.3, "rank1_agreement": 0.6}, y)["vs_nb0_fp8_kv"] == "ABOVE_NB0_FP8_KV"
          and KL.yardstick({"mean_abs_dlogprob": 0.3, "rank1_agreement": 0.6}, y)["vs_nb0_fp8_kv"] == "MIXED"
          and KL.yardstick({"status": "INCONCLUSIVE"}, y)["vs_nb0_fp8_kv"] == "INCONCLUSIVE")
    three = {"2k": {"label_full": "LOWBIT_TF_CLOSE"}, "8k": {"label_full": "LOWBIT_TF_DIVERGENT"}, "40k": {"label_full": "LOWBIT_TF_CLOSE"}}
    r = KL.tf_reading(dict(three, **{"90k": {"label_full": "INCONCLUSIVE_MISSING"}}))
    check("T6 K-5 tf_reading: missing 90k -> TF_INCOMPLETE (+ worst_measured, missing)", r["reading"] == "TF_INCOMPLETE"
          and r["worst_measured"] == "LOWBIT_TF_DIVERGENT" and r["worst_band"] == "8k" and r["missing"] == ["90k"] and r["complete"] is False, r)
    r = KL.tf_reading(three)
    check("T6 K-5 tf_reading: 90k absent from the records -> TF_INCOMPLETE", r["reading"] == "TF_INCOMPLETE" and r["missing"] == ["90k"], r)
    for st in ("SKIPPED_BUDGET_PRIORITY", "SKIPPED_K2_RESERVE", "SKIPPED_BOOT_BUDGET", "SKIPPED_BUDGET"):
        r = KL.tf_reading(three, {"90k": st})
        check(f"T6 K-5 tf_reading: 90k dropped under budget ({st}) -> TF_PARTIAL_BUDGET", r["reading"] == "TF_PARTIAL_BUDGET", r)
    r = KL.tf_reading(three, {"90k": "SKIPPED_SERVER_DEAD"})
    check("T6 K-5 tf_reading: 90k lost to a dead server -> TF_INCOMPLETE", r["reading"] == "TF_INCOMPLETE", r)
    r = KL.tf_reading({"2k": {"label_full": "LOWBIT_TF_CLOSE"}, "8k": {"label_full": "LOWBIT_TF_CLOSE"}}, {"90k": "SKIPPED_K2_RESERVE"})
    check("T6 K-5 tf_reading: 40k missing too -> TF_INCOMPLETE even with a budget 90k drop", r["reading"] == "TF_INCOMPLETE"
          and r["missing"] == ["40k", "90k"], r)
    r = KL.tf_reading(dict(three, **{"90k": {"label_full": "LOWBIT_TF_CLOSE"}}))
    check("T6 K-5 tf_reading: all four measured -> the worst label", r["reading"] == "LOWBIT_TF_DIVERGENT" and r["complete"] is True, r)
    check("T6 tf_reading nothing measured", KL.tf_reading({"2k": {"label_full": "INCONCLUSIVE_MISSING"}})["reading"] == "TF_NOT_MEASURED")
    clean = {"status": "DONE", "correctness": "CLEAN", "config_ok": True}
    for rd in ("TF_INCOMPLETE", "TF_PARTIAL_BUDGET", "TF_NOT_MEASURED"):
        f = KL.final_label({"INPUTS": (True, None)}, clean, clean, {"reading": rd})
        g = KL.final_label({"INPUTS": (True, None)}, clean, {"status": "BODY_ERROR"}, {"reading": rd})
        check(f"T6 K-5 final label with {rd}: GPU_LOAD_INCOMPLETE_TF (and the MTP-off-only variant), never PASS",
              f["label"] == f"GPU_LOAD_INCOMPLETE_TF + {rd}" and g["label"] == f"GPU_LOAD_INCOMPLETE_TF_MTP_OFF_ONLY + {rd}"
              and "PASS" not in f["label"] and "PASS" not in g["label"] and f["tf_complete"] is False, (f, g))
    tfr = {"reading": "LOWBIT_TF_DIVERGENT"}
    f1 = KL.final_label({"INPUTS": (True, None), "DIGESTS_K7_K3": (False, "sha256 mismatch")}, clean, clean, tfr)
    f2 = KL.final_label({"INPUTS": (True, None)}, {"status": "TF_OOM", "correctness": "CLEAN", "config_ok": True}, clean, tfr)
    f3 = KL.final_label({"INPUTS": (True, None)}, clean, {"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE"}, tfr)
    f4 = KL.final_label({"INPUTS": (True, None)}, clean, clean, tfr)
    f5 = KL.final_label({"INPUTS": (True, None)}, {"status": "CONFIG_MISMATCH", "correctness": "INCONCLUSIVE_NO_ORACLE"}, None, tfr)
    check("T6 final label: the first failing gate", f1["label"] == "FAIL_DIGESTS_K7_K3", f1)
    check("T6 final label: K not clean (TF_OOM before its core / config mismatch)", f2["label"] == "FAIL_BOOT_K_TF_OOM"
          and f5["label"] == "FAIL_BOOT_K_CONFIG_MISMATCH", (f2, f5))
    k_oom = {"status": "TF_OOM", "correctness": "CLEAN", "config_ok": True, "core_done": True, "kv_bytes": 3 * 2 ** 30}
    f2b = KL.final_label({"INPUTS": (True, None)}, k_oom, clean, tfr)
    check("T6 R2-3 K TF_OOM after its core + K2 clean -> K2's result kept", f2b["label"] ==
          "GPU_LOAD_PASS_K_TF_OOM_AT_PIN3.0 + K2_CLEAN + LOWBIT_TF_DIVERGENT", f2b)
    f2c = KL.final_label({"INPUTS": (True, None)}, dict(k_oom, status="BODY_OOM_long_120k"), clean, {"reading": "TF_INCOMPLETE"})
    check("T6 R2-3 K BODY_OOM after its core -> GPU_LOAD_INCOMPLETE_TF_K_BODY_OOM_long_120k_AT_PIN3.0 + K2_CLEAN",
          f2c["label"] == "GPU_LOAD_INCOMPLETE_TF_K_BODY_OOM_long_120k_AT_PIN3.0 + K2_CLEAN + TF_INCOMPLETE", f2c)
    f2d = KL.final_label({"INPUTS": (True, None)}, k_oom, {"status": "NO_GO_ORACLE"}, tfr)
    check("T6 R2-3 K TF_OOM after its core but K2 not clean -> FAIL_BOOT_K_TF_OOM", f2d["label"] == "FAIL_BOOT_K_TF_OOM", f2d)
    g = KL.final_label({"INPUTS": (True, None)}, clean, {"status": "NO_GO_GARBAGE_UNDER_LOAD", "correctness": "NO_GO_GARBAGE_UNDER_LOAD"}, tfr)
    check("T6 R2-12 K2 garbage under load -> GPU_LOAD_PASS_K2_GARBAGE_UNDER_LOAD (not MTP_OFF_ONLY)",
          g["label"] == "GPU_LOAD_PASS_K2_GARBAGE_UNDER_LOAD + LOWBIT_TF_DIVERGENT", g)
    lg = KL.final_label({"INPUTS": (True, None)}, dict(clean, long_120k={"verdict": "FAIL_LONG_GARBAGE"}), clean, tfr)
    check("T6 R2-1 the long-row verdict is appended, never a gate", lg["label"] == "GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT + LONG_FAIL_LONG_GARBAGE", lg)
    # K3 heads the label (R3-1 + the user's order)
    k3r = {"reading": "LOWBIT_TF_CLOSE", "complete": True}
    for k3, want in ((clean, "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"status": "SKIPPED_BUDGET"}, "K3_SKIPPED_BUDGET | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"status": "SKIPPED_NO_SCALES"}, "K3_SKIPPED_NO_SCALES | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"status": "LAUNCH_FAILED"}, "K3_FP8_LAUNCH_FAILED | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"status": "NO_GO_GARBAGE_UNDER_LOAD", "correctness": "NO_GO_GARBAGE_UNDER_LOAD"},
                      "K3_FP8_GARBAGE_UNDER_LOAD + LOWBIT_TF_CLOSE | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"status": "NOT_RUN"}, "GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT")):
        r_ = KL.final_label({"INPUTS": (True, None)}, clean, clean, tfr, k3, k3r)
        check(f"T6 K3 label for {k3.get('status')}: {want}", r_["label"] == want, r_)
    r_ = KL.final_label({"INPUTS": (True, None)}, {"status": "CONFIG_MISMATCH"}, None, tfr, {"status": "SKIPPED_BUDGET"}, k3r)
    check("T6 K3 never changes the K verdict", r_["label"] == "K3_SKIPPED_BUDGET | FAIL_BOOT_K_CONFIG_MISMATCH", r_)
    r_ = KL.final_label({"INPUTS": (False, "x")}, clean, clean, tfr, {"status": "NOT_RUN"}, k3r)
    check("T6 a gate failing before K3 -> FAIL_<gate> alone", r_["label"] == "FAIL_INPUTS", r_)
    # V-2: LOAD_PASS only with a complete K3 reading over its designated bands (2k / 8k)
    for rd, want in (({"reading": "TF_INCOMPLETE", "complete": False, "missing": ["8k"]}, "K3_FP8_INCOMPLETE_TF + TF_INCOMPLETE"),
                     ({"reading": "TF_NOT_MEASURED", "complete": False}, "K3_FP8_INCOMPLETE_TF + TF_NOT_MEASURED"),
                     ({"reading": "LOWBIT_TF_DIVERGENT", "complete": True}, "K3_FP8_LOAD_PASS + LOWBIT_TF_DIVERGENT"),
                     ({"reading": "LOWBIT_TF_CLOSE", "complete": True}, "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE"),
                     ({"reading": "LOWBIT_TF_CLOSE"}, "K3_FP8_INCOMPLETE_TF + LOWBIT_TF_CLOSE")):
        check(f"T6 V-2 k3_label {rd}: {want}", KL.k3_label(clean, rd) == want, KL.k3_label(clean, rd))
    r2 = KL.tf_reading({"2k": {"label_full": "LOWBIT_TF_CLOSE"}}, None, bands=("2k", "8k"))
    check("T6 V-2 one designated band missing -> INCOMPLETE", KL.k3_label(clean, r2) == "K3_FP8_INCOMPLETE_TF + TF_INCOMPLETE", r2)
    check("T6 V-3 K3 body OOM label carries the item and the pin",
          KL.k3_label({"status": "BODY_OOM_window_sw40k@24", "kv_bytes": 5270142976}, k3r) == "K3_FP8_BODY_OOM_window_sw40k@24_AT_PIN4.91")
    # V-9 / V-11
    r_ = KL.final_label({"INPUTS": (True, None)}, None, None, tfr, clean, k3r, profile="K3_ONLY")
    check("T6 V-9 K3_ONLY: the BF16 part is BF16_NOT_IN_PROFILE", r_["label"] == "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE | BF16_NOT_IN_PROFILE", r_)
    r_ = KL.final_label({"INPUTS": (True, None)}, clean, clean, tfr, {"status": "NOT_RUN", "why": "profile BF16_ONLY"}, k3r, profile="BF16_ONLY")
    check("T6 V-9 BF16_ONLY: no K3 part", r_["label"] == "GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT", r_)
    r_ = KL.final_label({"INPUTS": (True, None), "ABORT_DEVICE_FAULT": (False, {"after": "K3"})}, None, None, tfr,
                        {"status": "LAUNCH_FAILED", "device_fault": True}, k3r)
    check("T6 V-11 an ABORT_DEVICE_FAULT gate is its own label", r_["label"] == "K3_FP8_LAUNCH_FAILED | ABORT_DEVICE_FAULT", r_)
    # V-12
    pr = KL.boot_predicates(dict(clean, method_map={"ok": True}, scales={"ok": True},
                                 readback={"checks": {"STATIC_SCALES_APPLIED": True}}), k3r, fp8=True)
    check("T6 V-12 machine-readable predicates", pr == {"boot_clean": True, "tf_complete": True, "numerical_screen": "CLOSE",
                                                         "method_map_pass": True, "method_map_verdict": "GPU_METHOD_MAP_PASS",
                                                         "scales_applied_all_ranks": True, "accuracy_certified": False}, pr)
    pr = KL.boot_predicates({"status": "CONFIG_MISMATCH"}, {"reading": "TF_INCOMPLETE", "complete": False})
    check("T6 V-12 predicates of a failed BF16 boot", pr["boot_clean"] is False and pr["tf_complete"] is False
          and pr["numerical_screen"] == "INCONCLUSIVE" and pr["scales_applied_all_ranks"] is None and pr["accuracy_certified"] is False, pr)
    check("T6 final label: K2 failing -> MTP off only", f3["label"] == "GPU_LOAD_PASS_MTP_OFF_ONLY + LOWBIT_TF_DIVERGENT", f3)
    check("T6 final label: all clean", f4["label"] == "GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT", f4)
    check("T6 never an accuracy PASS", all("PASS" not in x["accuracy"] and "ACCURACY" not in x["label"] for x in (f1, f2, f3, f4, f5)))


# ======================================================================================== T7 capacity + argv (the export's code)
def cap_ns():
    ns = {"KL": KL, "math": math, "re": re, "json": json, "statistics": statistics, "BLOCK": 4096, "F_MAX_BLOCKS": 40, "WIN": 90.0,
          "GEN_RATE": 45.0, "PREFILL": {"tok_s": 2000.0}, "BANDS": {"40k": 40000, "60k": 60000}, "K100": {"admission_max_new": 4096},
          "PROMPTS": {"40k": [0] * 16, "60k": [0] * 16}}
    load(["FP8_BLOCK_BYTES", "BF16_BLOCK_BYTES", "_f_valid", "f_from_log", "pool_blocks_from_usage", "n_fit", "residency_bound",
          "k100_capacity"], ns)
    return ns


RC2_O = {"pool": {"kv_tokens": 513940, "max_concurrency": 3.92, "max_concurrency_tokens": 131072},
         "caps": {"block_size": 4096, "max_num_seqs": 16}, "workers": {"num_gpu_blocks": [150, 150, 150, 150]}}
RC2_US = [k / 149 for k in (3, 17, 40, 41, 77)]
MISMATCH_US = [k / 120 for k in (7, 13, 29, 41, 77)]          # the gauge reads a 120-block pool, the workers 149


def t7():
    ns = cap_ns()
    o, us = RC2_O, RC2_US
    cap = ns["k100_capacity"](o, us, 2479838700)
    check("T7 capacity on RC2 FP8 numbers: P 149 MATCH, F 6 (NB1's B1 / RC2 rule)",
          cap["pool_blocks"] == 149 and cap["pool_check"] == "MATCH" and cap["F"] == 6 and cap["capacity_ok"] is True, cap)
    check("T7 capacity: n40 / n64 / fp8 equivalence", isinstance(cap["n40"], int) and cap["n64"] == 149 // (16 + 6)
          and abs(cap["fp8_equivalent_blocks_of_pin"] - 150.0) < 0.01, cap)
    cap2 = ns["k100_capacity"](dict(o, workers={"error": "x"}), [], 2479838700)
    check("T7 capacity: no pool reading -> F None, no stream counts", cap2["pool_blocks"] is None and cap2["F"] is None
          and cap2["n40"] is None and cap2["capacity_ok"] is False, cap2)
    # K-3: n40 / n64 / resident_bound only on a MATCH pool with F measured
    for lab_, oo, uu, why in (("MISMATCH", o, MISMATCH_US, "KV_POOL_MISMATCH"), ("WORKER_ONLY", o, [], "KV_POOL_WORKER_ONLY"),
                              ("GAUGE_ONLY", dict(o, workers={"error": "x"}), us, "KV_POOL_GAUGE_ONLY"),
                              ("UNKNOWN", dict(o, workers={"num_gpu_blocks": [150, 151]}), [], "KV_POOL_UNKNOWN"),
                              ("F not measured", dict(o, pool={}), us, "F_NOT_MEASURED")):
        c = ns["k100_capacity"](oo, uu, 2479838700)
        check(f"T7 K-3 capacity {lab_}: capacity_ok False, n40 / n64 / resident_bound None, why {why}",
              c["capacity_ok"] is False and c["n40"] is None and c["n64"] is None and c["resident_bound"] is None and c["why"] == why, c)
    # K-11: both estimates; 463 FP8-equivalent blocks = ~244 BF16 usable blocks; the BF16 pin = 79 BF16 blocks
    cb = ns["k100_capacity"](o, us, 2464415744)
    check("T7 K-11 capacity: BF16 estimate listed next to the FP8 equivalent (463 FP8 -> 244 BF16; RC2 BF16 pin = 79)",
          cb["kit_estimate"]["bf16_usable_blocks_est"] == 244 and abs(cb["bf16_estimate_blocks_of_pin"] - 79.0) < 0.01
          and abs(ns["BF16_BLOCK_BYTES"] / 2 ** 20 - 29.75) < 0.01 and "authoritative" in cb, cb)
    # R2-8: one usable-block basis; pin / total and pin / usable separately; the measured FP8 equivalent from the pool
    check("T7 R2-8 RC2 FP8 pool: total 150, usable 149, bytes per total block = RC2's FP8 block",
          cap["num_gpu_blocks_total"] == 150 and cap["bytes_per_block_total"] == round(2479838700 / 150)
          and cap["bytes_per_block_usable"] == round(2479838700 / 149) and abs(cap["pin_fp8_equivalent_usable"] - 149.0) < 0.01, cap)
    o80 = dict(o, workers={"num_gpu_blocks": [80] * 4})
    c80 = ns["k100_capacity"](o80, [k / 79 for k in (3, 17, 40, 41, 77)], 2464415744)
    check("T7 R2-8 RC2 BF16 pool (80 total / 79 usable): measured FP8 equivalent = 80 x 29.75 / 15.77 - 1 = 149.9 usable",
          c80["pool_blocks"] == 79 and abs(c80["measured_fp8_equivalent_usable"] - (80 * ns["BF16_BLOCK_BYTES"] / ns["FP8_BLOCK_BYTES"] - 1)) < 0.11
          and abs(c80["pin_bf16_estimate_usable"] - 78.0) < 0.01, c80)
    c3 = ns["k100_capacity"](o, us, 2479838700, kv="fp8")
    check("T7 R3-1 an FP8 boot's measured FP8 equivalent = its own pool", c3["measured_fp8_equivalent_usable"] == 149 and c3["kv_dtype"] == "fp8", c3)
    # R2-7: >= 4 distinct samples, the worker's P first
    check("T7 R2-7 pool_from_usage: 3 distinct samples -> None (WORKER_ONLY)", KL.pool_from_usage([3 / 149, 17 / 149, 40 / 149], 149) is None
          and ns["k100_capacity"](o, [3 / 149, 17 / 149, 40 / 149], 2479838700)["pool_check"] == "WORKER_ONLY")
    check("T7 R2-7 pool_from_usage: 4 multiples of 1/38 also fit 76: the worker's 76 is tested first",
          KL.pool_from_usage([2 / 76, 4 / 76, 6 / 76, 10 / 76], 76) == 76 and KL.pool_from_usage([2 / 76, 4 / 76, 6 / 76, 10 / 76]) == 38)
    check("T7 R2-7 pool_from_usage: NaN / out of range ignored", KL.pool_from_usage([float("nan"), 0, 1, 3 / 149, 17 / 149, 40 / 149, 41 / 149], 149) == 149)
    ref = export_value("RC2_REF")
    ns2 = {}
    load(["flags", "_set_value", "nb0_argv"], ns2)
    a = list(ref["bf16"]["argv_tail"])
    off = ns2["nb0_argv"](a, {"mtp": False, "seqs": 8, "captures": "CAPS"})
    on = ns2["nb0_argv"](a, {"mtp": True, "seqs": 16, "captures": "HI"})
    check("T7 nb0_argv on RC2's real BF16 argv: MTP off / seqs / captures",
          "--speculative-config" not in off and ns2["flags"](off)["--max-num-seqs"] == "8" and ns2["flags"](off)["--compilation-config"] == "CAPS"
          and "--speculative-config" in on and ns2["flags"](on)["--max-num-seqs"] == "16" and ns2["flags"](on)["--kv-cache-dtype"] == "bfloat16")
    try:
        ns2["nb0_argv"](a + ["--max-num-seqs", "4"], {"mtp": True, "seqs": 16, "captures": None})
        dup = False
    except ValueError:
        dup = True
    check("T7 nb0_argv refuses a duplicated flag", dup)


# ======================================================================================== T8 build facts
def t8():
    pin = json.loads(json.loads(re.search(r'K100_PIN = json\.loads\((".*?")\)\s', EXPORT_SRC).group(1)))
    if have(REF / "SHA256SUMS.json", "T8 reference dataset SHA256SUMS"):
        sums = json.loads((REF / "SHA256SUMS.json").read_text())
        check("T8 embedded reference shas == the dataset SHA256SUMS", pin["ref_sha256"] == sums)
        check("T8 dataset files match SHA256SUMS", all(sha((REF / n).read_bytes()) == s for n, s in sums.items()))
    check("T8 embedded overlay shas = the v2 artifact", pin["overlay_sha256"]["config.json"].startswith("c78d48f3")
          and pin["overlay_sha256"]["model.safetensors.index.json"].startswith("c21d8c98")
          and pin["overlay_sha256"]["quantization_config.json"].startswith("e55659bc"))
    check("T8 klee100 input shas", pin["klee100_config_sha256"].startswith("fb17316d") and pin["klee100_index_sha256"].startswith("873191b5"))
    if have(DIGESTS, "T8 digest manifest sha"):
        check("T8 digests sha = the v2 kit artifact", pin["digests_sha256"] == sha(DIGESTS.read_bytes()))
    if NB_PATH.is_file():
        nb = NB_PATH.read_bytes()
        check("T8 notebook < 400 kB, export == cells", len(nb) < 400_000 and "\n\n".join(
            "".join(c["source"]) for c in json.loads(nb)["cells"] if c["cell_type"] == "code") == EXPORT_SRC, len(nb))
        cells = ["".join(c["source"]) for c in json.loads(nb)["cells"] if c["cell_type"] == "code"]
    else:                                     # the Kaggle units kit carries the export only (the cells joined by blank lines)
        check("T8 notebook file not in this kit: the export's cells are checked", True)
        cells = [c if i == 0 else "if not RERUN:\n" + c for i, c in enumerate(EXPORT_SRC.split("\n\nif not RERUN:\n"))]
    check("T8 RERUN guard: cells 2-9 under `if not RERUN:`, cell 1 writes the fallback", len(cells) == 9 and all(
        c.startswith("if not RERUN:\n") for c in cells[1:]) and "write_fallback_submission()" in cells[0], len(cells))
    check("T8 main only without RC2_UNIT", 'if not RERUN and not os.environ.get("RC2_UNIT"):' in cells[-1])
    check("T8 R3-1 FP8 KV only on K3 (kv='fp8' in one k1_boot call: K3), the BF16 boots default to bf16",
          EXPORT_SRC.count('ladder=True, kv="fp8")') == 1 and 'kv="bf16"' in src_of("k1_boot") and 'run_k(b0' in EXPORT_SRC
          and EXPORT_SRC.count('k1_boot("K3"') == 1)
    check("T8 R1-18 LAB_SRC is not inline (loaded from the sha-checked dataset file)", 'LAB_SRC = "' not in EXPORT_SRC
          and "K100_LAB_FILE" in EXPORT_SRC and len(export_const("K100_LAB_SHA")) == 64)


# ======================================================================================== T9 K-1 TF / SC on the boot cap
def t9():
    import time as _t
    calls = []
    ns = {"time": _t, "json": json, "PROMPTS": {b: [[1, 2, 3]] * 8 for b in ("2k", "8k", "40k", "90k")},
          "NB0": {"tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6}, "sc_bands": ["2k", "8k", "40k"], "sc_tok": 8},
          "TFR": {}, "SCR": {}, "DROP": set(), "WD": {}, "ABORT": {"why": None}, "BOOT": {"deadline": None},
          "ids_sha": lambda ids: "s", "cmp_last256": lambda a, b: {"status": "x"}, "save": lambda: None,
          "tf_request": lambda *a: (calls.append(("tf",) + a[:1]), {"meta": {"error": None}})[1],
          "decode": lambda *a: (calls.append(("decode",) + a[:1]), {"meta": {"error": None}, "tok": [1]})[1],
          "sc_compare": lambda *a: {}, "decode_compare": lambda *a: {}}
    load(["TF_EST", "SC_EST", "boot_left", "due", "tf_suite", "sc_suite"], ns)
    ns["left"] = lambda: 4000.0
    s = types.SimpleNamespace(alive=lambda: True, port=1)
    ns["BOOT"]["deadline"] = _t.time() - 1                 # boot_left 0 (the body cap is reached), the run has 4000 s left
    tf = ns["tf_suite"]("K", s, ["2k", "8k", "40k", "90k"])
    sc = ns["sc_suite"]("K", s)
    check("T9 K-1 boot_left 0 with left 4000: zero TF / decode HTTP calls, every request SKIPPED_BOOT_BUDGET",
          calls == [] and all(tf[b][r]["status"] == "SKIPPED_BOOT_BUDGET" for b in tf for r in ("fresh", "repeat"))
          and all(sc[b]["status"] == "SKIPPED_BOOT_BUDGET" for b in sc), (calls, tf, sc))
    ns["BOOT"]["deadline"] = _t.time() + 400               # 400 s of body left: 2k / 8k / 40k fit (EST + 300), 90k (150 + 300) not
    tf = ns["tf_suite"]("K", s, ["2k", "90k"])
    check("T9 K-1 400 s of body left: the 2k request runs, the 90k request does not start",
          [c[0] for c in calls] == ["tf", "tf"] and tf["90k"]["fresh"]["status"] == "SKIPPED_BOOT_BUDGET", (calls, tf))
    calls.clear()
    ns["BOOT"]["deadline"] = None                          # outside a boot: the run budget only (boot_left = left)
    ns["sc_suite"]("K", s)
    check("T9 K-1 a full budget: SC decodes run (cold + cached + TF per band)", len(calls) == 9, calls)


# ======================================================================================== T10 CELL 8 main on stubs (K-4 / K-6 / K-9)
def cell8_node():
    for n in ast.walk(_TREE):
        if isinstance(n, ast.If) and "RC2_UNIT" in (ast.get_source_segment(EXPORT_SRC, n.test) or "") and \
                any("GATES" in (ast.get_source_segment(EXPORT_SRC, x) or "") for x in n.body):
            return n
    raise AssertionError("CELL 8 main block not found")


GOOD_PID = {"tf": {b: {"used": "rendered"} for b in ("2k", "8k", "40k", "90k")},
            "sc": {b: {"used": "nb0_ids"} for b in ("2k", "8k", "40k")}, "rc2_oracle_prompts_match": {"rc2_bf16": True, "rc2_fp8": True}}
FREE4 = {0: 7000.0, 1: 7000.0, 2: 6800.0, 3: 6800.0}
CLEAN = {"status": "DONE", "correctness": "CLEAN", "config_ok": True}


def cell8(run, ref=None, pid=None, left=7200.0, profile="ALL"):
    """CELL 8 on stubs: run(b, ns) -> the boot record; returns (ns, calls)."""
    import threading as _th
    import time as _t
    calls = {"install": 0, "patch": 0, "boots": [], "printed": []}
    clock = {"left": left}
    ns = {"json": json, "time": _t, "traceback": traceback, "os": os, "KL": KL, "RERUN": False, "MERLIN_OK": True,
          "R": {"sections": {"A0": {"merge": {"ok": True}}}, "inputs": {}, "setup": {}, "errors": []}, "GATES": {}, "RECS": {},
          "LOADMON": {}, "BOOT": {"tag": None}, "MERLIN_ENG": {"kv": {"MERLIN_TREE": "/tree"}},
          "k100_scales": lambda tree: dict(SCALES_STUB[0], tree=tree),
          "K1": {"reference": {"ok": True} if ref is None else ref, "prompt_identity": GOOD_PID if pid is None else pid},
          "NB0": {"tf_bands": ["2k", "8k", "40k", "90k"], "sc_variant": {"2k": 6, "8k": 7, "40k": 6}},
          "K1BOOTS": {}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {}, "EXT_FLAG": ["--x"], "DIAG_ENV": {},
          "KV_DEFAULT": 2464415744, "FP8_CAPTURES": "caps", "COMPILE_HI": "hi", "T_START": _t.time(), "HOST_PHASE": {},
          "ABORT": {"why": None}, "OUT": TMP, "save": lambda: None, "log": lambda *a: None,
          "print": lambda *a, **k: calls["printed"].append(" ".join(str(x) for x in a)),
          "write_fallback_submission": lambda: FALLBACK.__setitem__(0, FALLBACK[0] + 1), "left": lambda: clock["left"], "_th": _th,
          "merlin_install": lambda: (calls.__setitem__("install", calls["install"] + 1), {"kv": {"MERLIN_TREE": "t"}})[1],
          "apply_60490": lambda: (calls.__setitem__("patch", calls["patch"] + 1), {"ok": True})[1],
          "sha_gate": lambda t: (True, "ok"), "body_k0": lambda b: None, "body_k": lambda b: None, "body_k2": lambda b: None,
          "k100_analysis": lambda: {"tf_reading": {"reading": "LOWBIT_TF_CLOSE"},
                                    "tf_k3_vs_A_reading": {"reading": "LOWBIT_TF_CLOSE", "complete": True}},
          "method_map_summary": lambda o: None, "body_k3": lambda b: None, "NO_INNER_RETRY": set(),
          "K100_PROFILE": profile, "BUDGET": 9000, "SCALES_CALIBRATION": "EXPERIMENTAL_REUSED_CALIBRATION",
          "gpu_health_gate": lambda: (calls.setdefault("health", []).append(1), dict(HEALTH[0]))[1]}
    load(["K100", "PORTS", "CAP_MIN", "EST_MIN", "BODY_MIN", "K3_GATE_D_GIB", "K3_PINS", "k1_boot", "boot_summary", "boot_brief",
          "tf_brief", "k_ready_sec", "k2_reserve_sec", "k3_ready_sec", "later_reserve_sec", "health_after"], ns)

    def step(sec, name, est, fn):
        if clock["left"] < est * 60 + 300:
            return {"status": "SKIPPED_BUDGET", "est_min": est}
        out = fn() or {}
        if out.get("status") in (None, "NOT_RUN"):
            out["status"] = "DONE"
        return out

    def run_k(b, body):
        calls["boots"].append((b["key"], b["kv_bytes"]))
        return dict(run(b, ns, clock), boot_name=b["name"])
    ns.update(step=step, run_k=run_k)
    node = cell8_node()
    exec(compile(ast.Module(body=node.body, type_ignores=[]), "<export CELL 8 main>", "exec"), ns)
    return ns, calls


SCALES_STUB = [{"ok": True, "layers": 12}]
HEALTH = [{"ok": True, "probe_ok": True, "ecc_ok": True, "released": True}]
K3FREE = {0: 1800.0, 1: 1800.0, 2: 1600.0, 3: 1600.0}
FALLBACK = [0]


def clean_run(b, ns, clock):
    return dict(CLEAN, free_mib_by_card_all=dict(FREE4), kv_bytes=b["kv_bytes"], ready_sec=480.0)


def t10():
    base = 2464415744
    # K-4: no / corrupt reference, prompt identity not ready, RC2 oracle prompts differ -> zero installs and zero boots
    for lab_, ref, pid, gate in (("reference missing", {"ok": False, "why": "the dataset q27-e25-nb0-tf-ref is not attached"}, None, "REFERENCE_DATASET"),
                                 ("reference corrupt", {"ok": False, "bad": ["sha256 nb0_reference.json"], "why": "1 reference files missing / changed"}, None, "REFERENCE_DATASET"),
                                 ("reference load crashed", {"ok": False, "why": "REF_LOAD_ERROR ValueError"}, None, "REFERENCE_DATASET"),
                                 ("prompt identity: no reference ids", None, dict(GOOD_PID, tf=dict(GOOD_PID["tf"], **{"90k": {"used": "rendered_no_reference"}})), "PROMPT_IDENTITY_READY"),
                                 ("prompt identity: a band missing", None, dict(GOOD_PID, sc={"2k": {"used": "rendered"}}), "PROMPT_IDENTITY_READY"),
                                 ("prompt identity crashed", None, {"error": "KeyError"}, "PROMPT_IDENTITY_READY"),
                                 ):
        ns, calls = cell8(clean_run, ref=ref, pid=pid)
        check(f"T10 K-4 {lab_}: zero installs, zero boots, FAIL_{gate}", calls["install"] == 0 and calls["boots"] == []
              and ns["FINAL"]["label"] == f"FAIL_{gate}" and ns["GATES"][gate][0] is False, (calls, ns["FINAL"]))
    # the notebook's own ref_load on a corrupt / incomplete / absent reference -> ok False -> CELL 8 boots nothing
    rd = TMP / "ref10"
    rd.mkdir(exist_ok=True)
    (rd / "nb0_reference.json").write_bytes(b"{\"tf\": {}}  corrupted")
    rns = {"hashlib": hashlib, "json": json, "np": np, "TFR": {}, "SCR": {}, "REF": {}, "REF_DIR": rd,
           "K100_PIN": {"ref_sha256": {"nb0_reference.json": sha(b"{\"tf\": {}}"), "A_2k_fresh.npz": sha(b"x")}}}
    load(["ref_load"], rns)
    bad_ref = rns["ref_load"]()
    rns["REF_DIR"] = None
    no_ref = rns["ref_load"]()
    check("T10 K-4 ref_load: a changed file + a missing file -> ok False; no dataset -> ok False",
          bad_ref["ok"] is False and "sha256 nb0_reference.json" in bad_ref["bad"] and "missing A_2k_fresh.npz" in bad_ref["bad"]
          and no_ref["ok"] is False and "not attached" in no_ref["why"], (bad_ref, no_ref))
    for lab_, r_ in (("corrupt", bad_ref), ("absent", no_ref)):
        ns, calls = cell8(clean_run, ref=r_)
        check(f"T10 K-4 ref_load {lab_} reference -> zero installs, zero boots", calls["install"] == 0 and calls["boots"] == []
              and ns["FINAL"]["label"] == "FAIL_REFERENCE_DATASET", (calls, ns["FINAL"]))
    ns, calls = cell8(clean_run)
    keys = lambda c: [k for k, _ in c["boots"]]
    bf = lambda n: n["FINAL"].get("bf16", n["FINAL"]["label"])
    check("T10 all gates hold: K3 -> K0 -> K -> K2 (the user's order), K3 heads the label",
          keys(calls) == ["K3", "K0", "K", "K2"] and ns["FINAL"]["label"] == "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE | GPU_LOAD_PASS + LOWBIT_TF_CLOSE"
          and all(ns["GATES"][g][0] for g in ("REFERENCE_DATASET", "PROMPT_IDENTITY_READY")), (calls, ns["FINAL"]))
    check("T10 K-4 the later gates do not overwrite the reference gates; R2-11 no RC2_ORACLE_IDENTITY gate", list(ns["GATES"])[:4] ==
          ["INPUTS", "OVERLAY_MERGE_K6", "REFERENCE_DATASET", "PROMPT_IDENTITY_READY"] and "RC2_ORACLE_IDENTITY" not in ns["GATES"], list(ns["GATES"]))
    for lab_, om in (("differ", {"rc2_bf16": True, "rc2_fp8": False}), ("absent", {})):
        ns2, calls2 = cell8(clean_run, pid=dict(GOOD_PID, rc2_oracle_prompts_match=om))
        check(f"T10 R2-11 RC2 oracle prompts {lab_}: recorded only, every boot still runs",
              keys(calls2) == ["K3", "K0", "K", "K2"] and ns2["K1"]["rc2_oracle_identity"]["all_match"] is False
              and ns2["K1"]["rc2_oracle_identity"]["record_only"] is True, (calls2["boots"], ns2["FINAL"]["label"]))
    kpin = calls["boots"][2][1]
    check("T10 K-2 valid K reading: K2 on K's pin, headroom verified", calls["boots"][3][1] == kpin
          and ns["K1"]["k2_pin"]["headroom_verified"] is True, ns["K1"]["k2_pin"])
    b3 = ns["K1BOOTS"][ns["K1"]["k3_tries"][0]["name"]]
    check("T10 rev 3 K3's first rung is step 0 of its ladder (the raised 7,933,526,016 B), port 8023, kv fp8; V-14 seqs 32",
          calls["boots"][0][1] == ns["K3_PINS"]["steps"][0] == 7933526016 and b3["kv"] == "fp8" and b3["port"] == 8023 and b3["seqs"] == 32
          and ns["K1"]["k3_booted_pin"] == ns["K3_PINS"]["steps"][0] and ns["K1"]["k3_first_rung"] == 7933526016, ns["K1"].get("k3_tries"))
    check("T10 R1-17 every boot passes --language-model-only", all("--language-model-only" in ns["NB1_EXTRA"][n] for n in ns["K1BOOTS"]))
    check("T10 R1-4 ladder boots get no inner retry", all(n in ns["NO_INNER_RETRY"] for n in ns["K1BOOTS"] if not n.startswith("k100_K0")))

    # K-2 in the flow: K's free memory partial -> K2 on the base pin
    def partial(b, ns, clock):
        r = clean_run(b, ns, clock)
        if b["key"] == "K":
            r["free_mib_by_card_all"] = {0: 9000.0, 1: 9000.0}
        return r
    ns, calls = cell8(partial)
    check("T10 K-2 K's free memory partial: K2 boots on the base pin (not K's), headroom_verified False in the verdict",
          calls["boots"][3][1] == base and calls["boots"][2][1] != base and ns["R"]["verdict"]["k2_headroom_verified"] is False
          and ns["K1"]["k2_pin"]["rule"] == "FALLBACK_BASE_PIN_UNVERIFIED_HEADROOM", (calls, ns["K1"]["k2_pin"]))

    # K-6: K2 launch OOM -> the next lower pin (fresh name / port, R1-9)
    k2_seen = []

    def k2_oom_once(b, ns, clock):
        if b["key"] == "K2":
            k2_seen.append((b["kv_bytes"], b["port"], b["name"]))
            if len(k2_seen) == 1:
                return {"status": "LAUNCH_FAILED", "oom_at_launch": True, "kv_bytes": b["kv_bytes"]}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k2_oom_once)
    k2 = [p for k, p in calls["boots"] if k == "K2"]
    pins = ns["K1"]["k2_pins"]
    check("T10 K-6 K2 launch OOM -> the next lower pin (pin2 - 512 MiB), then clean -> PASS",
          len(k2) == 2 and k2 == pins[:2] and k2[0] - k2[1] == 512 * 2 ** 20 and bf(ns) == "GPU_LOAD_PASS + LOWBIT_TF_CLOSE"
          and [t["stop"] for t in ns["K1"]["k2_tries"]] == ["NEXT_PIN", "NOT_A_LAUNCH_OOM"], (k2, pins, ns["K1"].get("k2_tries")))
    check("T10 R1-9 / R2-5 a K2 retry has a fresh port (+10) and its own name", k2_seen[0][1] == 8022 and k2_seen[1][1] == 8032
          and k2_seen[0][2] != k2_seen[1][2], k2_seen)

    def k2_xid(b, ns, clock):
        if b["key"] == "K2":
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True, "device_fault": True}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k2_xid)
    check("T10 K-6 K2 Xid / illegal memory access -> stop at once (one K2 try), MTP off only",
          keys(calls).count("K2") == 1 and ns["K1"]["k2_tries"][0]["stop"] == "DEVICE_FAULT"
          and bf(ns).startswith("GPU_LOAD_PASS_MTP_OFF_ONLY"), (calls, ns["K1"].get("k2_tries")))

    def k2_no_budget(b, ns, clock):
        r = clean_run(b, ns, clock)
        if b["key"] == "K":
            clock["left"] = 1500.0                     # < K2's minimum (16 min body + 1.5 x ready + teardown + margin)
        return r
    ns, calls = cell8(k2_no_budget)
    check("T10 K-6 / V-10 budget drop order: K2's minimum does not fit -> SKIPPED_BUDGET_K2_MINIMUM, no launch; K3 / K0 / K kept",
          keys(calls) == ["K3", "K0", "K"] and ns["K1"]["k2_tries"][0]["stop"] == "SKIPPED_BUDGET_K2_MINIMUM"
          and ns["RECS"]["K2"]["status"] == "SKIPPED_BUDGET_K2_MINIMUM" and ns["R"]["verdict"]["k2_budget"][0]["fits"] is False,
          (calls, ns["K1"].get("k2_tries")))

    def k2_oom_then_budget(b, ns, clock):
        if b["key"] == "K2":
            clock["left"] = 1500.0
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k2_oom_then_budget)
    check("T10 K-6 K2 launch OOM, then no budget for the next pin -> SKIPPED_BUDGET (each try budget-checked)",
          keys(calls).count("K2") == 1 and [t["stop"] for t in ns["K1"]["k2_tries"]] == ["NEXT_PIN", "SKIPPED_BUDGET_K2_MINIMUM"],
          ns["K1"].get("k2_tries"))

    def k_oom_first(b, ns, clock):
        if b["key"] == "K" and not [x for x in calls_k if x]:
            calls_k.append(1)
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True}
        return clean_run(b, ns, clock)
    calls_k = []
    ns, calls = cell8(k_oom_first)
    kp = [p for k, p in calls["boots"] if k == "K"]
    check("T10 K-6 K launch OOM -> the next ladder step", len(kp) == 2 and kp[0] > kp[1] and bf(ns).startswith("GPU_LOAD_PASS"),
          (kp, ns["K1"].get("k_tries")))

    def k_oom_short(b, ns, clock):
        if b["key"] == "K":
            clock["left"] = 3000.0                     # < EST_K + K2's minimum: no extra K rung
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k_oom_short)
    check("T10 R1-4 an extra K rung only if K + K2's minimum still fit -> SKIPPED_K2_RESERVE",
          keys(calls).count("K") == 1 and [t["stop"] for t in ns["K1"]["k_tries"]] == ["NEXT_PIN", "SKIPPED_K2_RESERVE"], ns["K1"].get("k_tries"))

    # K-9: the first BF16 boot's --help lacks a flag -> FAIL_CLI_FLAGS, nothing after K0 (K3 reported on its own)
    def cli_fail(b, ns, clock):
        ns["K1"].setdefault("cli_flags", {})[b["name"]] = {"ok": False, "missing": ["--kv-cache-memory-bytes"], "why": "missing flags: --kv-cache-memory-bytes"}
        return {"status": "ERROR", "error": "RuntimeError: FAIL_CLI_FLAGS: missing flags: --kv-cache-memory-bytes"}
    ns, calls = cell8(cli_fail)
    check("T10 K-9 --help missing a required flag on K0 -> FAIL_CLI_FLAGS, no K / K2 (K3_FP8_ERROR in front)",
          ns["FINAL"]["label"] == "K3_FP8_ERROR | FAIL_CLI_FLAGS" and keys(calls) == ["K3", "K0"], (calls, ns["FINAL"]))

    # K3 (R3-1 + the user's order)
    def k3_oom_once(b, ns, clock):
        if b["key"] == "K3" and b["name"].endswith(("_r0", "_r1")):  # rev 3: both raised rungs launch-OOM
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True, "kv_bytes": b["kv_bytes"]}
        if b["key"] == "K3":
            return dict(clean_run(b, ns, clock), free_mib_by_card_all=dict(K3FREE))
        return clean_run(b, ns, clock)
    ns, calls = cell8(k3_oom_once, left=12000.0)
    k3 = [p for k, p in calls["boots"] if k == "K3"]
    st3 = ns["K3_PINS"]["steps"]
    check("T10 K3 launch OOM -> the next lower step of its ladder (fresh ports 8033 / 8043), booted step recorded (rev 3: the "
          "two raised rungs OOM -> v1's 5,270,142,976)",
          k3 == st3[0:3] and ns["K1BOOTS"][ns["K1"]["k3_tries"][1]["name"]]["port"] == 8033
          and ns["K1BOOTS"][ns["K1"]["k3_tries"][2]["name"]]["port"] == 8043 and ns["K1"]["k3_booted_pin"] == st3[2] == 5270142976
          and keys(calls)[3:] == ["K0", "K", "K2"], (k3, ns["K1"].get("k3_tries")))
    lad = ns["K1"]["pin_ladder"]
    w3 = KL.pin_ladder(dict(K3FREE), st3[2], ns["K100"]["fracs"], ns["K100"]["safety_mib"], 0)
    w0 = KL.pin_ladder(dict(FREE4), base, ns["K100"]["fracs"], ns["K100"]["safety_mib"], ns["K100"]["mtp_reserve_mib"])
    want = [min(x["kv_bytes"], y["kv_bytes"]) for x, y in zip(w0["steps"], w3["steps"])]
    check("T10 R1-3 / K3: K3's reading feeds K's ladder (per frac the lower of K0 / K3; here K3 binds at 0.9 / 0.8)",
          lad["sources"] == ["K0", "K3"] and [x["kv_bytes"] for x in lad["steps"]] == want
          and "K3" in [x["source"] for x in lad["steps"]] and calls["boots"][4] == ("K", want[0]), lad.get("steps"))

    def k3_xid(b, ns, clock):
        if b["key"] == "K3":
            return {"status": "LAUNCH_FAILED", "oom_at_launch": True, "device_fault": True}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k3_xid)
    check("T10 V-11 K3 Xid + the health gate passes -> K0 / K / K2 run, device_fault_recovered recorded",
          keys(calls) == ["K3", "K0", "K", "K2"] and ns["FINAL"]["label"] == "K3_FP8_LAUNCH_FAILED | GPU_LOAD_PASS + LOWBIT_TF_CLOSE"
          and calls.get("health") == [1] and ns["K1"]["device_fault_recovered"] == ["K3"]
          and ns["R"]["verdict"]["device_fault_recovered"] == ["K3"], ns["FINAL"]["label"])
    HEALTH[0] = {"ok": False, "probe_ok": False, "ecc_ok": True, "released": True}
    ns, calls = cell8(k3_xid)
    HEALTH[0] = {"ok": True, "probe_ok": True, "ecc_ok": True, "released": True}
    check("T10 V-11 K3 Xid + the health probe FAILS -> ABORT_DEVICE_FAULT: no K0 / K / K2, verdict + fallback written",
          keys(calls) == ["K3"] and ns["FINAL"]["label"] == "K3_FP8_LAUNCH_FAILED | ABORT_DEVICE_FAULT" and ns["ABORT"]["why"]
          and ns["R"]["verdict"]["gates"]["ABORT_DEVICE_FAULT"]["ok"] is False, (calls, ns["FINAL"]))
    ns, calls = cell8(clean_run)
    check("T10 V-11 no device fault -> no health probe", "health" not in calls)

    # V-3: one controlled body-OOM retry before the core
    def k3_body_oom(b, ns, clock, also_second=False, fault=False, core=False):
        if b["key"] == "K3" and (b["name"].endswith("_r0") or also_second):
            return {"status": "TF_OOM", "correctness": "CLEAN", "config_ok": True, "core_done": core, "kv_bytes": b["kv_bytes"],
                    "device_fault": fault, "gpu_released": True, "oom_in_body": True}
        return clean_run(b, ns, clock)
    ns, calls = cell8(k3_body_oom, left=12000.0)
    k3p = [p for k, p in calls["boots"] if k == "K3"]
    check("T10 V-3 K3 body OOM before its core -> ONE retry at the next lower rung (clean there)", k3p == st3[0:2]
          and ns["K1"]["k3_tries"][0]["stop"] == "BODY_OOM_RETRY_NEXT_PIN" and ns["FINAL"]["label"].startswith("K3_FP8_LOAD_PASS"), ns["K1"]["k3_tries"])
    ns, calls = cell8(lambda b, n_, c_: k3_body_oom(b, n_, c_, also_second=True), left=12000.0)
    check("T10 V-3 a second body OOM -> no second retry; K3_FP8_TF_OOM_AT_PIN..., K0 continues", keys(calls).count("K3") == 2
          and ns["FINAL"]["label"].startswith("K3_FP8_TF_OOM_AT_PIN6.39 | ") and "K0" in keys(calls), (calls["boots"], ns["FINAL"]["label"]))
    for lab_, kw in (("with a device fault", {"fault": True}), ("after the core", {"core": True})):
        ns, calls = cell8(lambda b, n_, c_, kw=kw: k3_body_oom(b, n_, c_, **kw), left=12000.0)
        check(f"T10 V-3 K3 body OOM {lab_} -> no retry", keys(calls).count("K3") == 1, ns["K1"]["k3_tries"])
    ns, calls = cell8(k3_body_oom, left=9000.0)
    check("T10 V-3 body-OOM retry only if a K3 core + the K0 + K + K2 reserves fit (9,000 s left < 9,390 s: no retry)",
          keys(calls).count("K3") == 1, (calls["boots"], ns["K1"]["k3_tries"]))
    # V-9 profiles
    ns, calls = cell8(clean_run, profile="K3_ONLY")
    check("T10 V-9 K3_ONLY: only K3 boots; the label says BF16_NOT_IN_PROFILE", keys(calls) == ["K3"]
          and ns["FINAL"]["label"] == "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE | BF16_NOT_IN_PROFILE", (calls, ns["FINAL"]))
    ns, calls = cell8(clean_run, profile="BF16_ONLY")
    check("T10 V-9 BF16_ONLY: K0 -> K -> K2, no K3 part", keys(calls) == ["K0", "K", "K2"]
          and ns["FINAL"]["label"] == "GPU_LOAD_PASS + LOWBIT_TF_CLOSE", (calls, ns["FINAL"]))

    # V-10: K2's minimum with K ready 600 s: 2,340 s left -> no launch, 2,400 s -> launch
    for lft, want_n in ((2340.0, 0), (2400.0, 1)):
        def k_then(b, ns, clock, lft=lft):
            r = clean_run(b, ns, clock)
            if b["key"] == "K":
                r["ready_sec"] = 600.0
                clock["left"] = lft
            return r
        ns, calls = cell8(k_then)
        check(f"T10 V-10 K2 with {lft:.0f} s left (K ready 600 s): {want_n} launch(es)", keys(calls).count("K2") == want_n
              and (want_n or ns["RECS"]["K2"]["status"] == "SKIPPED_BUDGET_K2_MINIMUM"), (calls["boots"], ns["K1"].get("k2_budget")))
    ns, calls = cell8(clean_run, left=2000.0)
    check("T10 K3 budget skip: left < K3's step -> K3_SKIPPED_BUDGET, the BF16 boots that fit still run",
          keys(calls)[:1] == ["K0"] and ns["FINAL"]["label"].startswith("K3_SKIPPED_BUDGET | "), (calls, ns["FINAL"]))
    SCALES_STUB[0] = {"ok": False, "why": "qsa_kv_scales_262k.json in neither the Merlin tree nor the reference dataset"}
    ns, calls = cell8(clean_run)
    SCALES_STUB[0] = {"ok": True, "layers": 12}
    check("T10 K3 without the scales sidecar -> K3_SKIPPED_NO_SCALES, no K3 boot, the BF16 boots run",
          keys(calls) == ["K0", "K", "K2"] and ns["FINAL"]["label"].startswith("K3_SKIPPED_NO_SCALES | GPU_LOAD_PASS"), (calls, ns["FINAL"]))
    check("T10 K3 scales looked up in the installed Merlin tree first", ns["K1"]["scales"]["tree"] == "/tree")

    # R2-6 c: K TF_OOM after its core -> K2 runs (lower pin); long FAIL -> K2 still runs
    def k_tf_oom(b, ns, clock):
        r = clean_run(b, ns, clock)
        if b["key"] == "K":
            r.update(status="TF_OOM", core_done=True)
        return r
    ns, calls = cell8(k_tf_oom)
    k2p = [p for k, p in calls["boots"] if k == "K2"]
    kpn = [p for k, p in calls["boots"] if k == "K"][0]
    check("T10 R2-6c K TF_OOM after its core -> K2 runs at the next lower step; label keeps K2",
          len(k2p) == 1 and k2p[0] < kpn and bf(ns).startswith("GPU_LOAD_PASS_K_TF_OOM_AT_PIN") and "K2_CLEAN" in bf(ns), (calls, ns["FINAL"]))

    def k_long_fail(b, ns, clock):
        r = clean_run(b, ns, clock)
        if b["key"] == "K":
            r.update(long_120k={"verdict": "FAIL_LONG_GARBAGE"}, quality=["LONG_FAIL_LONG_GARBAGE"])
        return r
    ns, calls = cell8(k_long_fail)
    check("T10 R2-6c / R2-1 long FAIL -> K2 still runs; + LONG_FAIL_LONG_GARBAGE appended", keys(calls) == ["K3", "K0", "K", "K2"]
          and bf(ns) == "GPU_LOAD_PASS + LOWBIT_TF_CLOSE + LONG_FAIL_LONG_GARBAGE", ns["FINAL"])

    # R1-11: an unexpected exception -> the verdict, save() and the fallback still happen
    def boom(b, ns, clock):
        if b["key"] == "K":
            raise KeyError("boom")
        return clean_run(b, ns, clock)
    vf = TMP / "k100_verdict.json"
    if vf.exists():
        vf.unlink()
    ns, calls = cell8(boom)
    check("T10 R1-11 an exception in the main flow -> MAIN_ERROR label, k100_verdict.json written, fallback called",
          ns["FINAL"]["label"].startswith("MAIN_ERROR + ") and vf.is_file() and json.loads(vf.read_text())["main_error"]["error"].startswith("KeyError")
          and ns["R"]["verdict"]["main_error"] and FALLBACK[0] > 0, ns["FINAL"])


# ======================================================================================== T11 K-3 body_k2 capacity gate
def sweep_ns(calls, window=None):
    ns = cap_ns()
    ns.update(k_body=lambda b, fn: fn, _oracle_gate=lambda o, s, k, r: True, arith_probe=lambda p: {}, tf_suite=lambda *a, **k: {},
              _tf_stop=lambda o, tf: False, traced=lambda label, fn: fn(), live_pin=lambda b: b["kv_bytes"], due=lambda sec: True,
              boot_left=lambda: 1e6, prompt=lambda L_, v: [v] * 3, MAX_NEW={}, MAX_LEN=131072,
              window=window or (lambda port, s, band, n: (calls.append(("window", band, n)),
                                                       {"status": "STEADY", "running_min": n, "running_max": n, "running_eq_level": True,
                                                        "preemptions": 0, "admission": {"sec": 12.0}})[1]),
              admission_test=lambda port, s, band, n, max_new: (calls.append(("admission", band, n)), {"status": "ALL_RESIDENT"})[1])
    ns["BANDS"].update({"2k": 2000, "8k": 8000, "90k": 90000})
    ns["PREFILL"]["tok_s"] = 4000.0
    ns["K100"].update(n_cap=16, admission_max_new=4096)
    ns["PROMPTS"]["60k"] = [0] * 16
    load(["SWEEP_BANDS"], ns)
    for k_, (L_, v_) in ns["SWEEP_BANDS"].items():
        ns["BANDS"][k_], ns["MAX_NEW"][k_] = L_, 131072 - L_ - 512
    load(["run_item", "sweep_prompts", "gen_during_window", "level_need_sec", "kv_sweep", "admission_at_best", "body_k2"], ns)
    return ns


BIG_O = {"pool": {"kv_tokens": 1586661, "max_concurrency_tokens": 131072}, "caps": {"block_size": 4096, "max_num_seqs": 32},
         "workers": {"num_gpu_blocks": [461] * 4}}                  # P 460, F 6 (= the lead's FP8 sanity numbers)
BIG_US = [k / 460 for k in (3, 17, 41, 77, 101)]


def t11():
    calls = []
    ns = sweep_ns(calls)
    s = types.SimpleNamespace(port=1, alive=lambda: True)
    for lab_, us in (("MISMATCH", MISMATCH_US), ("WORKER_ONLY", [])):
        calls.clear()
        ns["warmup_gauge"] = lambda port, n, us=us: ({}, us)
        o = dict(RC2_O)
        ns["body_k2"]({"kv_bytes": 2479838700})(s, o)
        check(f"T11 K-3 / V-14 pool {lab_}: no sweep level, no 64k admission (capacity not verified)",
              calls == [] and o["admission_64k"]["status"] == "SKIPPED_CAPACITY_NOT_VERIFIED"
              and o["sweep"]["sw20k@16"]["status"] == "SKIPPED_CAPACITY_NOT_VERIFIED", (calls, o.get("sweep")))
    calls.clear()
    ns["warmup_gauge"] = lambda port, n: ({}, RC2_US)
    o = dict(RC2_O)
    ns["body_k2"]({"kv_bytes": 2479838700})(s, o)
    check("T11 V-14 K2 on RC2's 149-block pool: every band KV-bound at 16 (SKIPPED_KV_BOUND with the numbers), no admission",
          calls == [] and all(o["sweep"][f"{b_}@16"]["status"] == "SKIPPED_KV_BOUND" for b_ in ("sw20k", "sw40k", "sw60k", "sw100k"))
          and o["sweep"]["sw20k@16"]["resident_bound_095"] < 16 and o["admission_64k"]["status"] == "SKIPPED_NO_RESIDENT_40K_LEVEL", o.get("sweep"))
    calls.clear()
    ns["warmup_gauge"] = lambda port, n: ({}, BIG_US)
    o = json.loads(json.dumps(BIG_O))
    ns["body_k2"]({"kv_bytes": 7 * 2 ** 30})(s, o)
    check("T11 V-14 K2 (seqs 16) on a 460-block pool: 16 per band where resident (20k / 40k / 60k), 100k KV-bound; admission at 16",
          calls == [("window", "sw20k", 16), ("window", "sw40k", 16), ("window", "sw60k", 16), ("admission", "60k", 16)]
          and o["sweep"]["sw100k@16"]["status"] == "SKIPPED_KV_BOUND" and o["sweep_best"] == {"sw20k": 16, "sw40k": 16, "sw60k": 16}
          and "sw20k@24" not in o["sweep"], (calls, o.get("sweep")))


# ======================================================================================== T12 K-7 method map
def full_map(rank_split=True, tp_dup=True, drop=None, change=None, mtp=True):
    """Fake per-rank k100_method_map results: PP stage 0 = layers 0-24 (ranks 0, 1), stage 1 = 25-47 + MTP (ranks 2, 3)."""
    exp = KL.k100_expected_map()
    res = []
    for r in range(4):
        rows = []
        for L_, (b, g) in exp.items():
            if (L_ <= 24) != (r < 2) or L_ == drop:
                continue
            m = "HummingMoEMethod"
            if change and L_ == change[0]:
                m, b, g = change[1], change[2], change[3]
            rows.append({"name": f"language_model.model.layers.{L_}.mlp.experts", "layer": L_, "mtp": False, "method": m, "bits": b, "group": g})
        if mtp and r >= 2:
            rows.append({"name": "model.layers.48.mlp.experts", "layer": 48, "mtp": True, "method": "AutoGPTQMoEMethod", "bits": 4, "group": 128})
        res.append({"rank": r, "experts": rows, "errors": []})
    return res


def ext_ns():
    """The worker extension the notebook embeds (NB1's + the klee100 K-7 method), exec'd without torch / vLLM."""
    ns = {}
    exec(compile(export_value("NB1_EXT_SRC"), "<export nb1_worker_ext>", "exec"), ns)
    return ns


def merlin_fakes():
    """rev 3: fakes with the attribute structure of Merlin v3.0.0 (paths in the tree / the humming 0.1.17 wheel):
    HummingMoEMethod.quant_config = HummingLayerQuantizationConfig (weight_schema, NO config_parser; humming.py:343-360),
    .weight_schema = humming GPTQWeightSchema(bits, group_size) (humming.py:646, inc_wna16_scheme.py:299-306,
    humming/schema/gptq.py:12-17); RoutedExperts .layer_name / .quant_config = the INCConfig (routed_experts.py:112 / :114,
    INCConfig.weight_bits / group_size = the GLOBAL default 2 / 64, inc.py:111-112) with .config_parser.resolve(layer, name)
    -> INCLayerConfig(bits, group_size, sym) (config_parser.py:18-21 / :64); .weight_schemas {w13, w2} = converted
    HummingWeightSchema (b_dtype.num_bits, weight_scale_group_size; utils/humming/moe.py:620, humming/schema/humming.py:16-20);
    AutoGPTQMoEMethod.quant_config = AutoGPTQConfig(weight_bits, group_size) (auto_gptq.py:143 / :147); the V2 runner's
    get_model() / get_draft_model() / speculator (v1/worker/gpu/model_runner.py:378-384, :690-697), the speculator only on
    the last PP rank. ranks(...) -> the four per-rank k100_method_map results (PP 25/23, TP2)."""
    exp = KL.k100_expected_map()
    W = ext_ns()["NB1WorkerExt"]

    class INCLayerConfig:
        def __init__(self, bits, group_size, sym=True):
            self.bits, self.group_size, self.sym = bits, group_size, sym

    class INCConfigParser:
        def __init__(self, override=None):
            self.override = override or {}

        def resolve(self, layer, layer_name):
            L_ = int(re.search(r"layers\.(\d+)", layer_name).group(1))
            return INCLayerConfig(*(self.override.get(L_) or (exp[L_] if L_ < 48 else (4, 128))))

    class INCConfig:
        def __init__(self, override=None):
            self.weight_bits, self.group_size, self.sym = 2, 64, True       # the checkpoint's global default
            self.config_parser = INCConfigParser(override)

    class GPTQWeightSchema:
        def __init__(self, bits, group_size):
            self.quant_method, self.bits, self.group_size, self.desc_act, self.sym = "gptq", bits, group_size, False, True

    class DataType:
        def __init__(self, n):
            self.num_bits = n

    class HummingWeightSchema:
        def __init__(self, bits, group):
            self.quant_method, self.b_dtype, self.weight_scale_group_size = "humming", DataType(bits), group

    class HummingLayerQuantizationConfig:
        def __init__(self, ws):
            self.weight_schema, self.input_schema, self.force_weight_schema, self.force_input_schema = ws, None, None, None

    class AutoGPTQConfig:
        def __init__(self, b, g):
            self.weight_bits, self.group_size = b, g

    HummingMoEMethod = type("HummingMoEMethod", (), {})
    AutoGPTQMoEMethod = type("AutoGPTQMoEMethod", (), {})
    LinM = type("INCLinearMethod", (), {})

    class RoutedExperts:
        def __init__(self, layer_name, quant_config, quant_method, weight_schemas=None):
            self.layer_name, self.quant_config, self.quant_method = layer_name, quant_config, quant_method
            if weight_schemas is not None:
                self.weight_schemas = weight_schemas

    class Model:
        def __init__(self, items):
            self.items = items

        def named_modules(self):
            return iter(self.items)

    class DraftModelSpeculator:
        def __init__(self, model):
            self.model = model

    def ranks(runner="v2", v1_probe_shapes=False, conflict_layer=None, kernel_bits=None, drop=None):
        inc = INCConfig({conflict_layer: (2, 64)} if conflict_layer is not None else None)
        dinc = INCConfig()

        def routed(L_):
            b, g = exp[L_]
            name = f"language_model.model.layers.{L_}.mlp.experts"
            if v1_probe_shapes:             # what the v1 probe could see: no schema, no config_parser, no layer config
                qm = HummingMoEMethod()
                qm.quant_config = types.SimpleNamespace()
                return RoutedExperts(name, None, qm)
            qm = HummingMoEMethod()
            qm.quant_config = HummingLayerQuantizationConfig(GPTQWeightSchema(b, g))
            qm.weight_schema = qm.quant_config.weight_schema
            kb = kernel_bits[1] if kernel_bits and kernel_bits[0] == L_ else b
            return RoutedExperts(name, inc, qm, {"w13": HummingWeightSchema(kb, g), "w2": HummingWeightSchema(kb, g)})

        def worker(layers, last):
            items = [(f"model.layers.{L_}.mlp.experts", routed(L_)) for L_ in layers if L_ != drop]
            items += [(f"model.layers.{L_}.self_attn.qkv", types.SimpleNamespace(quant_method=LinM(), layer_name=f"x.layers.{L_}.qkv"))
                      for L_ in layers[:2]]
            mq = AutoGPTQMoEMethod()
            mq.quant_config = AutoGPTQConfig(4, 128)
            draft = Model([("model.layers.48.mlp.experts", RoutedExperts("mtp.layers.48.mlp.experts", dinc, mq))])
            spec = DraftModelSpeculator(draft) if last and runner != "none" else None
            main = Model(items)
            if runner == "v2":
                mr = types.SimpleNamespace(get_model=lambda: main, speculator=spec,
                                           get_draft_model=lambda: spec.model if spec is not None else None)
            elif runner == "speculator":
                mr = types.SimpleNamespace(get_model=lambda: main, speculator=spec)
            elif runner == "v1":
                mr = types.SimpleNamespace(get_model=lambda: main, drafter=spec)
            else:
                mr = types.SimpleNamespace(get_model=lambda: main, speculator=None, get_draft_model=lambda: None)
            w = W()
            w.model_runner = mr
            return w.k100_method_map()

        out = []
        for r_ in range(4):
            o_ = dict(worker(list(range(25)) if r_ < 2 else list(range(25, 48)), r_ >= 2), rank=r_)
            o_.pop("rank_error", None)
            out.append(o_)
        return out
    return {"ranks": ranks, "W": W}


def t12():
    exp = KL.k100_expected_map()
    check("T12 K-7 expected map: 0-11 / 36-47 W3 g128, 12-35 W2 g64", len(exp) == 48 and exp[0] == exp[11] == exp[36] == exp[47] == (3, 128)
          and exp[12] == exp[35] == (2, 64))
    r = KL.method_map_check(full_map(), need_mtp=True)
    check("T12 K-7 48/48 + MTP W4 -> PASS", r["ok"] is True and r["layers_matching"] == 48 and r["mtp_ok"] is True, r)
    r = KL.method_map_check(full_map(drop=30), need_mtp=True)
    check("T12 K-7 47/48 -> not clean (layer 30 not seen)", r["ok"] is False and r["layers_matching"] == 47 and r["missing"] == [30], r)
    r = KL.method_map_check(full_map(change=(5, "HummingMoEMethod", 2, 64)), need_mtp=False)
    check("T12 K-7 layer 5 resolved W2 g64 (checkpoint W3 g128) -> not clean", r["ok"] is False and r["bad"][0]["layer"] == 5, r)
    r = KL.method_map_check(full_map(change=(20, "UnquantizedFusedMoEMethod", None, None)), need_mtp=False)
    check("T12 K-7 a routed layer on UnquantizedFusedMoEMethod -> not clean", r["ok"] is False and r["bad"][0]["layer"] == 20, r)
    r = KL.method_map_check(full_map(mtp=False), need_mtp=True)
    check("T12 K-7 MTP on but no MTP experts -> not clean", r["ok"] is False and r["mtp_ok"] is False, r)
    r = KL.method_map_check(full_map(mtp=False), need_mtp=False)
    check("T12 K-7 MTP off, no MTP experts -> PASS", r["ok"] is True, r)
    bad_mtp = full_map()
    bad_mtp[2]["experts"][-1].update(method="HummingMoEMethod", bits=3)
    r = KL.method_map_check(bad_mtp, need_mtp=True)
    check("T12 K-7 MTP experts not W4 / AutoGPTQ -> not clean", r["ok"] is False and r["mtp_ok"] is False, r)
    conf = full_map()
    conf[1]["experts"][0].update(bits=2, group=64)            # rank 1 disagrees with rank 0 on layer 0
    r = KL.method_map_check(conf, need_mtp=True)
    check("T12 K-7 two TP ranks disagree on a layer -> not clean", r["ok"] is False and r["conflicts"], r)
    check("T12 K-7 rpc error / no results -> not clean", KL.method_map_check(None, True)["ok"] is False
          and KL.method_map_check([], False)["ok"] is False)
    # V-1: exactly the ranks 0-3, no rank error, PP-local layer ownership, MTP only on the last stage, no layer outside 0-47
    m3 = full_map()
    del m3[1]
    r = KL.method_map_check(m3, need_mtp=True)
    check("T12 V-1 rank 1 missing (3 of 4 ranks) -> FAIL", r["ok"] is False and "RANKS" in r["why"], r["why"])
    dup = full_map()
    dup[1] = dict(dup[0])
    r = KL.method_map_check(dup, need_mtp=True)
    check("T12 V-1 a duplicated rank (0, 0, 2, 3) -> FAIL", r["ok"] is False and "RANKS" in r["why"], r["why"])
    er = full_map()
    er[2] = dict(er[2], errors=["get_model: RuntimeError"])
    r = KL.method_map_check(er, need_mtp=True)
    check("T12 V-1 a rank reporting an error -> FAIL", r["ok"] is False and "RANK_ERRORS" in r["why"], r["why"])
    er = full_map()
    er[3] = dict(er[3], rank_error="RuntimeError('Default process group has not been initialized')")
    check("T12 V-1 a rank-id error -> FAIL", KL.method_map_check(er, need_mtp=True)["ok"] is False)
    wp = full_map()
    wp[0]["experts"].append({"name": "x.layers.30.mlp.experts", "layer": 30, "mtp": False, "method": "HummingMoEMethod", "bits": 2, "group": 64})
    r = KL.method_map_check(wp, need_mtp=True)
    check("T12 V-1 a PP1 layer reported by a PP0 rank -> WRONG_PP_LAYERS FAIL", r["ok"] is False and "WRONG_PP_LAYERS" in r["why"], r["why"])
    mp = full_map()
    mp[0]["experts"].append({"name": "model.layers.48.mlp.experts", "layer": 48, "mtp": True, "method": "AutoGPTQMoEMethod", "bits": 4, "group": 128})
    r = KL.method_map_check(mp, need_mtp=True)
    check("T12 V-1 MTP experts on a PP0 rank -> MTP_ON_PP0 FAIL", r["ok"] is False and "MTP_ON_PP0" in r["why"], r["why"])
    ul = full_map()
    ul[2]["experts"].append({"name": "model.layers.48.mlp.experts", "layer": 48, "mtp": False, "method": "AutoGPTQMoEMethod", "bits": 4, "group": 128})
    r = KL.method_map_check(ul, need_mtp=True)
    check("T12 V-1 the MTP layer 48 counted as routed -> UNEXPECTED_ROUTED_LAYERS FAIL", r["ok"] is False and "UNEXPECTED_ROUTED_LAYERS" in r["why"], r["why"])
    r = KL.method_map_check(full_map(), need_mtp=True, partition=(24, 24))
    check("T12 V-1 the layer ownership follows the boot's partition (24/24 vs a 25/23 map -> FAIL)", r["ok"] is False and "WRONG_PP_LAYERS" in r["why"])
    r = KL.method_map_check(full_map(), need_mtp=True)
    check("T12 V-1 a valid four-rank 25/23 map -> PASS", r["ok"] is True and r["ranks_ok"] is True, r["why"])
    check("T12 V-1 parse_partition", KL.parse_partition("25,23") == (25, 23) and KL.parse_partition(None) == (25, 23)
          and KL.parse_partition("24/24") == (24, 24) and KL.parse_partition("x") == (25, 23))
    # the embedded worker extension on a fake model (no torch / vLLM here: rank_error recorded, classes by name)
    ens = ext_ns()
    W = ens["NB1WorkerExt"]
    check("T12 K-7 the embedded extension = NB1WorkerExt + k100_method_map (NB1's methods kept)",
          hasattr(W, "k100_method_map") and hasattr(W, "nb1_cache_info") and hasattr(W, "nb1_routing"))

    # rev 3: fakes with the REAL Merlin v3.0.0 object structure (the v1 fakes put an INC config_parser on the method's
    # quant_config, which a HummingMoEMethod does not have; the K3_ONLY GPU v1 boot then read bits None on every layer)
    fk = merlin_fakes()
    ranks = fk["ranks"]()
    r = KL.method_map_check(ranks, need_mtp=True)
    rows0, mrow = ranks[0]["experts"], ranks[2]["experts"][-1]
    check("T12 rev 3 extension on Merlin-shaped objects (HummingMoEMethod.weight_schema = GPTQWeightSchema, the module's INCConfig, "
          "layer.weight_schemas, AutoGPTQConfig for MTP, V2 runner get_draft_model) -> GPU_METHOD_MAP_PASS",
          r["ok"] is True and r["verdict"] == "GPU_METHOD_MAP_PASS" and r["layers_matching"] == 48 and r["mtp_ok"] is True
          and all(x["source"] == "method.weight_schema" and not x["bits_conflict"] for x in rows0)
          and set(rows0[0]["sources"]) == {"method.weight_schema", "inc.config_parser.resolve", "layer.weight_schemas.w13", "layer.weight_schemas.w2"}
          and mrow["mtp"] is True and mrow["source"] == "method.quant_config" and [mrow["bits"], mrow["group"]] == [4, 128]
          and r["drafter_paths"] == {"0": "none", "1": "none", "2": "model_runner.get_draft_model()", "3": "model_runner.get_draft_model()"},
          (r.get("why"), rows0[:1], mrow))
    check("T12 rev 3 the INC global default (weight_bits 2 / group 64 on the INCConfig) is never read as a layer's bits",
          all(x["bits"] == exp[x["layer"]][0] and x["group"] == exp[x["layer"]][1] for w in ranks for x in w["experts"] if not x["mtp"])
          and all("method.quant_config" not in x["sources"] for x in rows0), rows0[:2])
    for runner, want in (("speculator", "model_runner.speculator.model"), ("v1", "model_runner.drafter.model")):
        rr = fk["ranks"](runner=runner)
        r = KL.method_map_check(rr, need_mtp=True)
        check(f"T12 rev 3 the MTP drafter is found through {want}", r["verdict"] == "GPU_METHOD_MAP_PASS" and r["mtp_ok"] is True
              and r["drafter_paths"]["2"] == want, (r.get("why"), r.get("drafter_paths")))
    rr = fk["ranks"](runner="none")
    r = KL.method_map_check(rr, need_mtp=True)
    check("T12 rev 3 no drafter anywhere (MTP on) -> GPU_METHOD_MAP_FAIL 'MTP experts missing'", r["verdict"] == "GPU_METHOD_MAP_FAIL"
          and r["gate_ok"] is False and "MTP experts missing" in r["why"], r.get("why"))
    rr = fk["ranks"](v1_probe_shapes=True)
    r = KL.method_map_check(rr, need_mtp=True)
    check("T12 rev 3 the v1 failure shape (method.quant_config without config_parser, no weight_schema, module without "
          "quant_config / weight_schemas) with the MTP drafter found -> METHOD_CLASS_ONLY (gate_ok, not ok), MTP read",
          r["verdict"] == "METHOD_CLASS_ONLY" and r["gate_ok"] is True and r["ok"] is False and r["layers_class_only"] == 48
          and r["bits_readable"] == {"routed": "UNREADABLE", "mtp": "READ"} and r["class_only_families"] == ["routed"]
          and all(x.get("bits") is None and x.get("sources") == {} for x in rr[0]["experts"]), (r.get("why"), r.get("bits_readable")))
    rr = fk["ranks"](conflict_layer=7)
    r = KL.method_map_check(rr, need_mtp=True)
    check("T12 rev 3 the method's schema says W3 but the module's INC parser says W2 (layer 7) -> bits_conflict -> "
          "GPU_METHOD_MAP_FAIL BITS_SOURCES_DISAGREE", r["verdict"] == "GPU_METHOD_MAP_FAIL" and "BITS_SOURCES_DISAGREE" in r["why"]
          and r["source_conflicts"] and r["source_conflicts"][0]["name"].endswith("layers.7.mlp.experts"), r.get("why"))
    rr = fk["ranks"](kernel_bits=(12, 2))
    r = KL.method_map_check(rr, need_mtp=True)
    check("T12 rev 3 the converted kernel schema runs W2 where the checkpoint is W2 (layer 12, all sources 2 / 64) -> PASS; "
          "a kernel schema at W4 on layer 0 -> FAIL", r["verdict"] == "GPU_METHOD_MAP_PASS"
          and KL.method_map_check(fk["ranks"](kernel_bits=(0, 4)), need_mtp=True)["verdict"] == "GPU_METHOD_MAP_FAIL", r.get("why"))
    ranks = fk["ranks"](drop=24)      # layer 24 lost on both stage-0 ranks
    r = KL.method_map_check(ranks, need_mtp=True)
    check("T12 K-7 extension: 47/48 layers -> not clean", r["ok"] is False and r["missing"] == [24] and r["gate_ok"] is False, r)
    # k100_facts: GPU_METHOD_MAP_PASS gates K and K2 (config not clean), recorded on K0
    log_ = ("INFO non-default args: {'model': '/ck', 'kv_cache_dtype': 'bfloat16', 'max_num_seqs': 8, 'kv_cache_memory_bytes': 123, "
            "'enable_prefix_caching': True}\nINFO Initializing a V1 LLM engine with config: model='/ck', speculative_config=None, "
            "dtype=torch.bfloat16, quantization=inc, kv_cache_dtype=bfloat16\nWARNING needs speculative decoding; num_speculative_tokens=0\n")
    fns = {"re": re, "json": json, "KL": KL, "CKPT": "/ck", "CAPS_LIST": {"C": [1, 2]}, "KV_FP8_DTYPE": "fp8_e4m3",
           "server_caps": lambda t, argv: {"capture_sizes": [1, 2]}, "pool_facts": lambda t: {}, "graph_facts": lambda t: {},
           "rpc_cache_info": lambda p: {}}
    load(["flags", "_pick", "k100_facts"], fns)
    s = types.SimpleNamespace(log=lambda: log_, argv=["vllm", "serve", "/ck", "--kv-cache-memory-bytes", "123"], port=1)
    for key, res, want in (("K", full_map(mtp=False), True), ("K", full_map(drop=7, mtp=False), False), ("K0", full_map(drop=7, mtp=False), True)):
        fns["rpc"] = lambda port, method, res=res: {"results": res} if method == "k100_method_map" else {"error": "x"}
        o = fns["k100_facts"](s, {"key": key, "seqs": 8, "captures": "C", "mtp": False, "kv": "bf16"}, {})
        check(f"T12 K-7 k100_facts {key} with {o['method_map']['layers_matching']}/48: config_ok {want}"
              + (" (recorded only on K0)" if key == "K0" else ""),
              o["config_ok"] is want and (key == "K0" or o["readback"]["checks"]["GPU_METHOD_MAP_PASS"] is want)
              and ("GPU_METHOD_MAP_PASS" not in o["readback"]["checks"] if key == "K0" else True), o["readback"]["checks"])
    clean = KL.boot_clean({"status": "CONFIG_MISMATCH", "correctness": "CLEAN", "config_ok": False})
    check("T12 K-7 a boot with config_ok False is not clean", clean is False)


# ======================================================================================== T13 K-8 traces
def t13():
    import threading as _th
    import time as _t

    class FakeStream:
        def __init__(self, port, ids, max_new, temperature=1.0, seed=None, **k):
            self.port, self.body = port, {"prompt": list(ids), "max_tokens": max_new, "temperature": temperature, "seed": seed}
            self.text, self.tokens = [f"text-{seed}-" + "x" * 5000], list(range(3000))

        def row(self):
            return {"text": "".join(self.text), "tokens": list(self.tokens), "margins": [], "error": None, "done": True, "finish_reason": "length"}
    tr = TMP / "traces13"
    tr.mkdir(exist_ok=True)
    ns = {"json": json, "time": _t, "hashlib": hashlib, "_SAVE_LOCK": _th.Lock(), "K1": {}, "TRACES": tr,
          "BOOT": {"tag": "k100_K2_bf16_mtp3_pin300"}, "L": types.SimpleNamespace(Stream=FakeStream)}
    ns["shutil"] = shutil
    load(["psha", "TRACE_CTX", "k100_trace_install", "k100_trace_one", "k100_trace", "trace_disk_ok", "traced"], ns)
    ns["k100_trace_install"]()
    ns["k100_trace_install"]()                                 # idempotent: one wrap only
    path = tr / "k100_K2_bf16_mtp3_pin300_8k@8.jsonl"
    seen_inside = []

    def window():
        ss = [ns["L"].Stream(1, [1, 2, 3], 8192, seed=1234 + i) for i in range(3)]
        seen_inside.append(path.exists())                       # nothing is written while the window runs
        return {"status": "STEADY", "n": len(ss)}
    r = ns["traced"]("8k@8", window)
    ns["L"].Stream(1, [9], 1, seed=7)                            # outside traced(): not collected
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    check("T13 K-8 every stream of the window in full (text, token ids, finish), written after the window",
          r["status"] == "STEADY" and seen_inside == [False] and len(rows) == 3 and all(len(x["tokens"]) == 3000 and len(x["text"]) > 5000
          and x["finish_reason"] == "length" and x["window"] == "8k@8" and x["boot"] == "k100_K2_bf16_mtp3_pin300" for x in rows)
          and [x["seed"] for x in rows] == [1234, 1235, 1236] and ns["K1"]["traces"][path.name] == 3, (seen_inside, len(rows)))

    def boom():
        ns["L"].Stream(1, [1], 4, seed=99)
        raise RuntimeError("window crashed")
    try:
        ns["traced"]("admission_60k@4", boom)
    except RuntimeError:
        pass
    p2 = tr / "k100_K2_bf16_mtp3_pin300_admission_60k@4.jsonl"
    check("T13 K-8 a crashing window still writes its streams", p2.is_file() and len(p2.read_text().splitlines()) == 1)
    check("T13 K-8 every sweep level and the admission go through traced()",
          src_of("kv_sweep").count("traced(") == 1 and src_of("admission_at_best").count("traced(") == 1
          and "kv_sweep(" in src_of("body_k2") and "kv_sweep(" in src_of("body_k3"))

    class RunStream:                                          # its own class (FakeStream is already wrapped above)
        def __init__(self, port, ids, max_new, temperature=1.0, seed=None, **k):
            self.port, self.body = port, {"prompt": list(ids), "max_tokens": max_new, "temperature": temperature, "seed": seed}
            self.text, self.tokens = ["t"], [1, 2]

        def row(self):
            return {"text": "".join(self.text), "tokens": list(self.tokens), "margins": [], "error": None, "done": True, "finish_reason": "length"}

        def run(self):
            self.ran = True
    ns2 = {"json": json, "time": _t, "hashlib": hashlib, "_SAVE_LOCK": _th.Lock(), "K1": {}, "TRACES": tr, "shutil": shutil,
           "BOOT": {"tag": "k100_K3_x"}, "L": types.SimpleNamespace(Stream=RunStream)}
    load(["psha", "TRACE_CTX", "k100_trace_install", "k100_trace_one", "k100_trace", "trace_disk_ok", "traced"], ns2)
    ns2["k100_trace_install"]()
    p3 = tr / "k100_K3_x_sw20k@16.jsonl"
    mid = []

    def win():
        ss = [ns2["L"].Stream(1, [1], 4, seed=i) for i in range(3)]
        ss[0].run()                                           # stream 0 finishes inside the window
        mid.append(len(p3.read_text().splitlines()) if p3.exists() else 0)
        return {}
    ns2["traced"]("sw20k@16", win)
    rows = [json.loads(x) for x in p3.read_text().splitlines()]
    check("T13 V-8 a finished stream is written at once (1 line mid-window), the rest at the end, each once, key (boot, window, i)",
          mid == [1] and len(rows) == 3 and sorted(r["i"] for r in rows) == [0, 1, 2] and all(r["window"] == "sw20k@16" for r in rows), (mid, len(rows)))
    ns2["trace_disk_ok"] = lambda min_gb=2.0: (False, 1.2)
    ns2["traced"]("sw40k@16", win)
    check("T13 V-8 < 2 GB free -> the window's traces skipped with a note", not (tr / "k100_K3_x_sw40k@16.jsonl").exists()
          and "traces skipped" in ns2["K1"]["trace_notes"][0] and ns2["K1"]["trace_disk_gb"]["sw40k@16"] == 1.2, ns2["K1"].get("trace_notes"))
    if have(LAB_FILE, "T13 LAB_SRC == the shared E23 lab file"):
        check("T13 K-8 / R1-18 the payload lab is the shared E23 lab, unchanged (traces wrap it on the klee100 side)",
              lab_src() == LAB_FILE.read_text(encoding="utf-8").replace("\r\n", "\n"))


# ======================================================================================== T14 K-9 --help flags at launch
def t14():
    base, k, p, o = mk_fake()
    d = base / "ckpt14"
    KL.build_model_dir(d, k, p, o, ["model-plefp8-00000.safetensors", "model-plefp8-00001.safetensors"], link=LINK)
    calls = []

    class S:
        argv = ["vllm", "serve", "--speculative-config", "{}", "--max-num-seqs", "3", "--compilation-config", "{}",
                "--enable-prefix-caching", "--kv-cache-memory-bytes", "123"]
        env = {"VLLM_X": "1"}
    helpres = {}
    ns = {"KL": KL, "CKPT": d, "K1": {}, "K1BOOTS": {"n": {"key": "K0", "name": "n", "mtp": False, "seqs": 8, "captures": "{}", "title": "t"}},
          "save": lambda: None, "R": {"setup": {}}, "os": os, "time": __import__("time"), "threading": __import__("threading"),
          "json": json, "ENV_KEYS": ("VLLM_",), "K100": {"ready_sec": 10}, "left": lambda: 1e6, "HELP": {},
          "config_diff": lambda *a: {}, "load_monitor": lambda *a: None, "LIVE_PIN": {},
          "L": types.SimpleNamespace(help_check=lambda key, pre, env, argv: dict(helpres["v"], argv=list(argv))),
          "_merlin_fn_server_nb1": lambda *a: (calls.append(a), S())[1]}
    os.environ["RC2_CPU_KIT"] = "1"
    load(["flags", "_set_value", "nb0_argv"], ns)
    load(["merlin_fn_server"], ns)
    helpres["v"] = {"ran": True, "missing": ["--kv-cache-memory-bytes"]}
    try:
        ns["merlin_fn_server"]("n", "bf16", 8020, 123)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    check("T14 K-9 --help ran and a required flag is missing -> FAIL_CLI_FLAGS before launch, recorded",
          raised is not None and raised.startswith("FAIL_CLI_FLAGS") and ns["K1"]["cli_flags"]["n"]["ok"] is False
          and "n" not in (ns["R"]["setup"].get("k100_plans") or {}), raised)
    helpres["v"] = {"ran": False, "error": "help output too short or failed", "tail": "Traceback"}
    s = ns["merlin_fn_server"]("n", "bf16", 8020, 123)
    check("T14 K-9 --help itself fails -> recorded (ok None), the launch plan continues",
          s is not None and ns["K1"]["cli_flags"]["n"]["ok"] is None and "HELP_ERROR" in ns["K1"]["cli_flags"]["n"]["why"]
          and "n" in ns["R"]["setup"]["k100_plans"], ns["K1"]["cli_flags"])
    helpres["v"] = {"ran": True, "missing": []}
    s = ns["merlin_fn_server"]("n", "bf16", 8020, 123)
    check("T14 K-9 all flags listed -> ok True; checked against the final argv (after nb0_argv)",
          ns["K1"]["cli_flags"]["n"]["ok"] is True and ns["HELP"]["n"]["argv"] == s.argv, ns["HELP"]["n"])
    check("T14 K-9 cli_flags_check: not a dict -> None", KL.cli_flags_check(None)["ok"] is None)


# ======================================================================================== T15 K-10 K2 reserve before K's optional items
def t15():
    calls = []
    ns = {"KL": KL, "LOADMON": {}, "BOOT": {"tag": None}, "RECS": {}, "live_pin": lambda b: b["kv_bytes"],
          "k_body": lambda b, fn: fn, "_oracle_gate": lambda o, s, k, r: True, "arith_probe": lambda p: {},
          "warmup_gauge": lambda p, n: ({}, []), "k100_capacity": lambda o, us, kv: {},
          "tf_suite": lambda key, s, bands: (calls.append(("tf", tuple(bands))), {b: {"fresh": {}, "repeat": {}} for b in bands})[1],
          "_tf_stop": lambda o, tf: False, "sc_suite": lambda key, s: {}, "due": lambda sec: True,
          "long_prefill": lambda port: (calls.append(("long",)), {"verdict": LONGV[0]})[1]}
    LONGV = ["FINISHED_NATURALLY_CLEAN"]
    ns["os"] = os
    load(["K100", "TF_EST", "run_item", "k_ready_sec", "k2_reserve_sec", "k2_reserve_ok", "body_k"], ns)
    s = types.SimpleNamespace(port=1, alive=lambda: True)
    res = ns["k2_reserve_sec"]()
    check("T15 K-10 / R1-2 reserve = 16 min body + 1.5 x 780 s default ready + 180 + 300 s", res == 960 + 1170 + 480, res)
    ns["LOADMON"]["k100_K_x"] = {"ready_sec": 400.0}
    ns["BOOT"]["tag"] = "k100_K_x"
    check("T15 R1-2 K's measured ready time (load monitor) sets the reserve", ns["k2_reserve_sec"]() == 960 + 600 + 480, ns["k2_reserve_sec"]())
    ns["LOADMON"].clear()
    ns["left"] = lambda: res + 500.0                          # 500 s beyond the reserve: neither 120k (1140 s) nor 90k TF (600 s) fits
    o = {}
    ns["body_k"]({"kv_bytes": 1})(s, o)
    check("T15 K-10 short of the reserve: the 120k row and the 90k TF -> SKIPPED_K2_RESERVE, the core TF still runs",
          calls == [("tf", ("2k", "8k", "40k"))] and o["long_120k"]["verdict"] == "SKIPPED_K2_RESERVE"
          and o["tf"]["90k"]["status"] == "SKIPPED_K2_RESERVE", (calls, o.get("long_120k")))
    calls.clear()
    ns["left"] = lambda: res + 650.0                          # the 90k TF (2 x 150 + 300 = 600 s) fits, the 120k row does not
    o = {}
    ns["body_k"]({"kv_bytes": 1})(s, o)
    check("T15 K-10 room for the 90k TF only: 90k runs, 120k SKIPPED_K2_RESERVE",
          calls == [("tf", ("2k", "8k", "40k")), ("tf", ("90k",))] and o["long_120k"]["verdict"] == "SKIPPED_K2_RESERVE", calls)
    calls.clear()
    ns["left"] = lambda: res + 3600.0
    o = {}
    ns["body_k"]({"kv_bytes": 1})(s, o)
    check("T15 K-10 enough budget: the 120k row and the 90k TF run", calls == [("tf", ("2k", "8k", "40k")), ("long",), ("tf", ("90k",))], calls)
    LONGV[0] = "FAIL_LONG_GARBAGE"
    o = {}
    ns["body_k"]({"kv_bytes": 1})(s, o)
    check("T15 R2-1 a FAIL long verdict -> o['quality'], the status untouched (K2 still depends on transport only)",
          o.get("quality") == ["LONG_FAIL_LONG_GARBAGE"] and "status" not in o and o["core_done"] is True, o.get("quality"))
    ns2 = {"KL": KL}
    load(["tf_skips"], ns2)
    sk = ns2["tf_skips"]({"2k": {"fresh": {"x": 1}}, "90k": {"status": "SKIPPED_K2_RESERVE"}, "40k": {"fresh": {"status": "SKIPPED_BOOT_BUDGET"}}})
    check("T15 K-10 tf_skips -> the 90k reserve drop reads TF_PARTIAL_BUDGET", sk == {"90k": "SKIPPED_K2_RESERVE", "40k": "SKIPPED_BOOT_BUDGET"}
          and KL.tf_reading({b: {"label_full": "LOWBIT_TF_CLOSE"} for b in ("2k", "8k", "40k")}, sk)["reading"] == "TF_PARTIAL_BUDGET", sk)


# ======================================================================================== T16 K-12 arithmetic probe
def t16():
    cases = KL.arith_cases()
    check("T16 K-12 8 fixed exact-answer cases (add / sub / mul with carries), answers exact",
          len(cases) == 8 and dict(cases)["Reply with exactly one number: 1234 * 5678 ="] == "7006652"
          and dict(cases)["Reply with exactly one number: 4827 + 3594 ="] == "8421" and dict(cases)["Reply with exactly one number: 90071 - 4568 ="] == "85503")
    res = {"cases": [{"q": q, "want": w, "content": (w + "." if i == 0 else "The answer is " + w if i == 1 else w if i < 6 else "123"),
                      "ok": i < 6} for i, (q, w) in enumerate(cases)]}
    sc = KL.arith_score(res)
    check("T16 K-12 exact-match: '8421.' exact, 'The answer is ...' contains-only, wrong -> neither", sc["n"] == 8 and sc["exact"] == 5
          and sc["contains"] == 6 and sc["gated"] is False and sc["cases"][1]["exact"] is False, sc)
    ns = {"KL": KL, "L": types.SimpleNamespace(chat_probes=lambda port, cases: (_ for _ in ()).throw(OSError("down")))}
    load(["arith_probe"], ns)
    r = ns["arith_probe"](1)
    check("T16 K-12 a failing probe is recorded, never raises, never gated", "error" in r and r["gated"] is False, r)
    check("T16 K-12 K and K2 bodies call the probe; no gate reads it", src_of("body_k").count("arith_probe(") == 1
          and src_of("body_k2").count("arith_probe(") == 1 and '"arith"' not in src_of("_start"))


# ======================================================================================== T17 R1 budget / cap / memory helpers
def t17():
    import time as _t
    # R1-1: levels() gated on the boot cap
    calls = []
    ns = {"time": _t, "json": json, "ABORT": {"why": None}, "LIVE": {"rec": None}, "save": lambda: None, "WIN": 90.0,
          "BANDS": {"8k": 8000}, "PROMPTS": {"8k": [[1]] * 8}, "MAX_NEW": {"8k": 100}, "BOOT": {"deadline": _t.time() + 100},
          "L": types.SimpleNamespace(steady=lambda *a, **k: (calls.append(a), {"status": "STEADY"})[1])}
    ns["left"] = lambda: 5000.0
    load(["boot_left", "due", "levels"], ns)
    out = ns["levels"](1, "8k", [8])
    check("T17 R1-1 levels(): boot body cap short (left 5000) -> SKIPPED_BOOT_BUDGET, no steady window",
          calls == [] and out["8k@8"]["status"] == "SKIPPED_BOOT_BUDGET", out)
    ns["BOOT"]["deadline"] = _t.time() + 3000
    out = ns["levels"](1, "8k", [8])
    check("T17 R1-1 levels(): enough body left -> the window runs", len(calls) == 1 and out["8k@8"]["status"] == "STEADY", out)
    check("T17 R1-1 window()'s direct steady path is gated too", "SKIPPED_BOOT_BUDGET" in src_of("window"))
    # R1-2: watched() clips the cap at body start; k2_plan
    ns = {"time": _t, "threading": __import__("threading"), "BOOT": {"deadline": None, "fired": None}, "LIVE": {}, "left": lambda: 1000.0}
    load(["watched"], ns)
    seen = {}
    o = ns["watched"](lambda ss: (seen.update(dl=ns["BOOT"]["deadline"] - _t.time()), {})[1], 3000, "n")([])
    check("T17 R1-2 watched(): cap 3000 s with left 1000 s -> body cap clipped to left - 480 = 520 s", o["body_cap_sec"] == 520
          and 515 < seen["dl"] <= 520, (o, seen))
    p = KL.k2_plan(5400, 600)
    check("T17 R1-2 k2_plan: cap from K's ready time (5400 left, ready 600): cap = 34 min max, est = cap + 900 + 180",
          p["body_cap_sec"] == 2040 and p["est_min"] == 52 and p["fits"], p)
    p = KL.k2_plan(3000, 600)
    check("T17 R1-2 k2_plan: 3000 left -> cap 1560 s, step() still fits (est x 60 + 300 <= left)",
          p["body_cap_sec"] == 1560 and p["est_min"] * 60 + 300 <= 3000 and p["fits"], p)
    p = KL.k2_plan(2000, 777)
    check("T17 R1-2 k2_plan: too little left -> fits False, est above the budget (step() skips)",
          not p["fits"] and p["est_min"] * 60 + 300 > 2000, p)
    # R1-3: worker vs nvidia-smi (NB1 R2 B3: smi 1629 / 1629 / 939 / 925, worker 639 / 639 / 205 / 191)
    smi = {0: 1629.0, 1: 1629.0, 2: 939.0, 3: 925.0}
    rd = {"ranks": [{"rank": r, "mem": {"free_mib": f}} for r, f in enumerate((639, 639, 205, 191))]}
    wf = KL.worker_free([rd, {"ranks": [{"rank": 3, "mem": {"free_mib": 180}}, {"rank": True, "mem": {"free_mib": 1}}]}])
    fr, src = KL.combine_free(smi, wf)
    check("T17 R1-3 free per card = min(nvidia-smi, worker) (the worker's lowest reading)", fr == {0: 639, 1: 639, 2: 205, 3: 180}
          and src[0] == "min(nvidia-smi, worker)", (fr, src))
    fr, src = KL.combine_free(smi, {0: 639}, {1: 990.0, "2": 734.0})
    check("T17 R1-3 no worker reading: nvidia-smi - K0's gap, else - 1024 MiB", fr == {0: 639, 1: 639.0, 2: 205.0, 3: 925 - 1024}
          and "K0 gap" in src[1] and "1024" in src[3], (fr, src))
    lad_smi, lad_w = KL.pin_ladder(smi, 2479838700, safety_mib=0), KL.pin_ladder(KL.combine_free(smi, wf)[0], 2479838700, safety_mib=0)
    check("T17 R1-3 the ladder from the combined reading is lower than from nvidia-smi alone",
          lad_w["steps"] == [] or lad_w["steps"][0]["kv_bytes"] < lad_smi["steps"][0]["kv_bytes"], (lad_smi.get("steps"), lad_w.get("steps")))
    check("T17 R1-3 run_k records both readings and the combination", all(x in src_of("run_k") for x in
          ("free_mib_by_card_smi", "free_mib_by_card_worker", "combine_free")))
    # R1-4: rungs >= an OOMed pin skipped; known OOM pins; live pin
    seen = []
    rec, tries = KL.ladder_run([5, 6, 4], lambda p, i: (seen.append(p), {"status": "LAUNCH_FAILED", "oom_at_launch": True} if p == 5 else {"status": "DONE"})[1])
    check("T17 R1-4 a rung >= the lowest OOMed pin is skipped (6 after 5 OOMed), the next lower one runs", seen == [5, 4]
          and [t["stop"] for t in tries] == ["NEXT_PIN", "SKIPPED_ABOVE_OOM_PIN", "NOT_A_LAUNCH_OOM"], tries)
    rec, tries = KL.ladder_run([5, 4], lambda p, i: {"status": "DONE"}, known_oom=[5])
    check("T17 R1-4 known OOM pins (an earlier ladder) skip equal / higher rungs", tries[0]["status"] == "SKIPPED_ABOVE_OOM_PIN" and rec["status"] == "DONE", tries)
    rec, tries = KL.ladder_run([5], lambda p, i: {"status": "DONE"}, known_oom=[3])
    check("T17 R1-4 every rung above an OOM pin -> NOT launched, a record anyway", rec["status"] == "SKIPPED_ABOVE_OOM_PIN" and len(tries) == 1, rec)
    ns = {"LIVE_PIN": {"name": "k100_K0_x_t1", "kv_bytes": 100}}
    load(["live_pin"], ns)
    check("T17 R1-4 live_pin: the inner retry's pin (name _t1) is the one the body uses",
          ns["live_pin"]({"name": "k100_K0_x", "kv_bytes": 368}) == 100 and ns["live_pin"]({"name": "other", "kv_bytes": 368}) == 368)
    # R1-5
    v = KL.probe_engine_view({"ok": False, "cases": [{"q": "a", "content": "41", "ok": False}, {"q": "b", "content": "4", "ok": True},
                                                     {"q": "c", "content": "6", "ok": True}]})
    check("T17 R1-5 a wrong answer is recorded (2/3), not a gate", v["ok"] is True and v["probes_correct"] == "2/3" and v["arith_all_correct"] is False, v)
    for lab_, c in (("request error", {"q": "a", "error": "x", "ok": False}), ("empty", {"q": "a", "content": " ", "ok": False}),
                    ("garbage", {"q": "a", "content": "!!!!", "garbage": "BANG_GARBAGE", "ok": False})):
        v = KL.probe_engine_view({"cases": [c, {"q": "b", "content": "4", "ok": True}]})
        check(f"T17 R1-5 engine fault ({lab_}) -> probes not ok", v["ok"] is False and v["engine_faults"] == ["a"], v)
    check("T17 R1-5 no probe rows -> not ok", KL.probe_engine_view({"cases": []})["ok"] is False)
    check("T17 R1-5 _start gates on the engine view", "probe_engine_view" in src_of("_start"))
    # R1-6
    order = []
    ns = {"_boot_gate_rc2": lambda: (order.append("release_wait"), {"ok": True})[1],
          "clear_engine_procs": lambda: (order.append("clear"), {"ok": True})[1], "host_ram_gate": lambda tag: {"ok": True, "why": "x"},
          "BOOT": {"tag": "t"}, "json": json}
    load(["boot_gate"], ns)
    g = ns["boot_gate"]()
    check("T17 R1-6 boot_gate: engine processes cleared BEFORE the release wait", order[:2] == ["clear", "release_wait"] and g["ok"], order)
    check("T17 R1-6 with_servers clears engine processes before the release wait and before FAIL_GPU_RELEASE",
          src_of("with_servers").count("clear_engine_procs()") == 2)
    # R1-7
    check("T17 R1-7 the sha wait never starves K / K2; the install timeout follows the budget",
          "sha_gate(max(60, min(2400, left() - 86 * 60)))" in EXPORT_SRC and "timeout=int(max(600, left() - 90 * 60))" in EXPORT_SRC
          and "timeout=5400" not in EXPORT_SRC)
    # R1-12 / R1-13 / R1-14 / R1-15 / R1-16 / R1-17
    check("T17 R1-12 _start attaches the body record to the live step record; run_k removes it",
          'LIVE["rec"]["live_body"] = o' in src_of("_start") and 'pop("live_body", None)' in src_of("run_k"))

    class Flaky:
        n = 0

        def __str__(self):
            Flaky.n += 1
            if Flaky.n == 1:
                raise RuntimeError("dictionary changed size during iteration")
            return "ok"
    out = TMP / "t17save"
    out.mkdir(exist_ok=True)
    ns = {"json": json, "time": _t, "_SAVE_LOCK": __import__("threading").Lock(), "OUT": out, "R": {"x": Flaky()}}
    load(["save"], ns)
    ns["save"]()
    check("T17 R1-13 save() retries a dump that raced another thread", json.loads((out / "k100_results.json").read_text()) == {"x": "ok"})
    check("T17 R1-13 hash_all mutates R / K1 under _SAVE_LOCK", "with _SAVE_LOCK:" in src_of("merge_ckpt").split("def hash_all")[1])
    check("T17 R1-14 the GPU sampler polls every 5 s", "every=5.0" in src_of("GpuSampler"))
    check("T17 R1-15 DROP / WD are gone (no such name in the code)",
          not [n for n in ast.walk(_TREE) if isinstance(n, ast.Name) and n.id in ("DROP", "WD")])
    got = []
    ns = {"due": lambda sec: (got.append(sec), False)[1], "BANDS": {"60k": 60000}, "PREFILL": {"tok_s": 2000.0}}
    load(["admission_test"], ns)
    r = ns["admission_test"](1, None, band="60k", n=8)
    check("T17 R1-16 / R2-2 admission due() = max(240 + 150 + n x band / prefill, its wait limit + 60) (n 8 x 60k: 720 s)",
          r["status"] == "SKIPPED_BOOT_BUDGET" and got == [max(240 + 150 + 8 * 60000 / 2000, max(420, 180 + 8 * 60000 / 2000) + 60)], got)
    check("T17 R1-17 --language-model-only on every klee100 boot, recorded in the read-back",
          '"--language-model-only"' in src_of("k1_boot") and '"language_model_only"' in src_of("k100_facts"))


# ======================================================================================== T18 merlin_boot (R1-8 / R1-9 / R1-10 / R2-2 / R2-3 / R2-5)
def t18():
    def harness(results, logs, no_retry=()):
        launched = []

        class Srv:
            def __init__(self, name, port, log):
                self.name, self.port, self._log, self.ready_sec = name, port, log, 321.0

            def log(self):
                return self._log
        it = iter(zip(results, logs))
        ns = {"KL": KL, "NB1_KV": {"b": 2000}, "NB1_EXTRA": {"b": ["--x"]}, "NB1_ENV": {"b": {}}, "K1BOOTS": {"b": {"name": "b", "port": 8020}},
              "KV_BF16": 1, "OOM_STEP": 256, "NO_INNER_RETRY": set(no_retry), "left": lambda: 1e6, "log": lambda *a: None,
              "boot_gate": lambda: {"ok": True}, "shm_names": lambda: set(), "shm_cleanup": lambda x: {},
              "OOM_ERR": re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory"), "XID": re.compile(r"NVRM: Xid|illegal memory access"),
              "gated": lambda o, **k: dict(o, correctness=o.get("correctness_stub", "CLEAN")), "long_gate": lambda g: None,
              "BOOT": {"fired": None}}
        cur = {}

        def mfs(name, kv, port, kvb):
            r, lg = next(it)
            cur["r"] = r
            launched.append((name, port, kvb))
            return Srv(name, port, lg)
        ns["merlin_fn_server"] = mfs
        ns["with_servers"] = lambda servers, timeout, body: dict(cur["r"])
        load(["merlin_boot"], ns)
        return ns, launched
    big = "x" * 9000 + "\ntorch.OutOfMemoryError: CUDA out of memory\n" + "y" * 9000      # the OOM line 9 kB before the end
    ns, launched = harness([{"status": "LAUNCH_FAILED"}, {"status": "DONE"}], [big, "ok"])
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 R1-8 a launch OOM far from the log tail is seen (whole log); R2-5 / R1-9 the retry is b_t1 on port 8030, 0.25 GiB lower",
          launched == [("b", 8020, 2000), ("b_t1", 8030, 1744)] and g["launch_name"] == "b_t1" and g["tries"][0]["oom"] is True
          and "b_t1" in ns["K1BOOTS"] and ns["NB1_EXTRA"]["b_t1"] == ["--x"], (launched, g.get("tries")))
    ns, launched = harness([{"status": "LAUNCH_FAILED"}], [big], no_retry=("b",))
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 R1-4 a ladder boot (NO_INNER_RETRY) gets no inner retry", len(launched) == 1 and g["status"] == "LAUNCH_FAILED", launched)
    ns, launched = harness([{"status": "LAUNCH_FAILED"}], [big + "\nNVRM: Xid 79"])
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 K-6 a device fault at launch is never retried", len(launched) == 1 and g["xid_in_log"] is True, g.get("tries"))
    for lab_, body, lg, want in (
            ("watchdog fired", {"status": "SERVER_DIED", "boot_watchdog": "FIRED after 900 s", "died_in": "tf_90k"}, big, "BOOT_CAP_KILL"),
            ("OOM in the 90k TF", {"status": "SERVER_DIED", "died_in": "tf_90k"}, big, "TF_OOM"),
            ("OOM in the 120k row", {"status": "SERVER_DIED", "died_in": "long_120k"}, big, "BODY_OOM_long_120k"),
            ("death without OOM in a window", {"status": "SERVER_DIED", "died_in": "window_8k@8"}, "no oom", "SERVER_DIED_IN_window_8k@8"),
            ("device fault in a TF", {"status": "SERVER_DIED", "died_in": "tf_2k-8k-40k"}, "NVRM: Xid 13", "DEVICE_FAULT_IN_tf_2k-8k-40k")):
        ns, _ = harness([body], [lg])
        g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
        check(f"T18 R1-10 / R2-2 {lab_} -> {want}", g["status"] == want and g["death"] == want, g)
    ns, _ = harness([{"status": "BODY_ERROR", "body_error": "KeyError"}], ["no oom"])
    ns["BOOT"]["fired"] = "b"
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 V-13b an exception in the body while the watchdog fired -> BOOT_CAP_KILL kept (status before kept too)",
          g["status"] == "BOOT_CAP_KILL" and g["status_before_kill"] == "BODY_ERROR", g)
    ns, _ = harness([{"status": "NO_GO_TF_REQUEST_ERROR", "server_status": "SERVER_DIED", "died_in": "tf_90k"}], [big])
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 R2-3 a specific body status is kept; the death is recorded next to it",
          g["status"] == "NO_GO_TF_REQUEST_ERROR" and g["death"] == "TF_OOM" and g["server_status"] == "SERVER_DIED", g)
    ns, _ = harness([{"status": "SERVER_DIED", "died_in": "long_120k", "correctness_stub": "NO_GO_ORACLE_GARBAGE"}], [big])
    g = ns["merlin_boot"]("b", "bf16", 8020, 100, None)
    check("T18 a BODY_OOM with unclean evidence takes the evidence status", g["status"] == "NO_GO_ORACLE_GARBAGE", g)
    s = types.SimpleNamespace(alive=lambda: ALIVE[0])
    ALIVE = [True]
    o = {}
    ns = {}
    load(["run_item"], ns)
    ns["run_item"](o, s, "oracle", lambda: None)
    ns["run_item"](o, s, "tf_90k", lambda: ALIVE.__setitem__(0, False))
    ns["run_item"](o, s, "long_120k", lambda: None)
    check("T18 R1-10 run_item: died_in = the item during which the server died", o["died_in"] == "tf_90k" and o["items"] == ["oracle", "tf_90k", "long_120k"], o)
    check("T18 R2-5 the load monitor is keyed by the launch name", "args=(s, name, deadline)" in src_of("merlin_fn_server")
          and "LOADMON.get(ln)" in src_of("run_k"))


# ======================================================================================== T19 K3 (R3-1 + the user's order)
def t19():
    kit = {"PP0": 11.973, "PP1": 12.026}
    r = KL.k3_pin_steps(kit)
    exp = min(f + (16.89 - kit["PP0" if i < 2 else "PP1"]) * 1024 - 1024 - 102.4 - 768 for i, f in KL.B3_WORKER_FREE_MIB.items())
    check("T19 K3 ladder: est_extra = min over cards (B3 worker free + W4 - klee100 Gate D - 1 GiB - 0.1 GiB graphs - 768 MiB)",
          abs(r["est_extra_mib"] - round(exp, 1)) < 0.11 and r["binding_card"] == 3, r)
    check("T19 K3 ladder: [B3 + 0.8 x est, B3 + 0.5 x est, B3, FP8 base], descending, 2 MiB aligned",
          r["steps"][2:] == [3553580524, 2479838700] and r["steps"][0] > r["steps"][1] > r["steps"][2]
          and all(p % (2 * 2 ** 20) == 0 for p in r["steps"][:2])
          and abs(r["steps"][0] - (3553580524 + 0.8 * exp * 2 ** 20)) < 2 * 2 ** 20, r["steps"])
    r2 = KL.k3_pin_steps(kit, w4_gib=12.0)
    check("T19 K3 ladder: no estimated headroom -> [B3, base] only", r2["steps"] == [3553580524, 2479838700], r2)
    check("T19 K3 the export builds K3_PINS from the kit's Gate D numbers", export_const_json("K3_GATE_D_GIB") == kit)
    m = KL.k3_measured_rungs()
    check("T19 rev 3 raised K3 rungs from the v1 GPU reading (free 4583 / 4579 / 3257 / 3257 MiB at 5,270,142,976): pin + (3257 - "
          "0.7 / 1.7 GiB) MiB, 2 MiB aligned -> 7,933,526,016 and 6,859,784,192",
          m["rungs"] == [7933526016, 6859784192] and m["free_min_mib"] == 3257.0 and m["binding_card"] in (2, 3)
          and all(p % (2 * 2 ** 20) == 0 for p in m["rungs"])
          and all(0 <= 3257.0 - (p - 5270142976) / 2 ** 20 - lv < 2 for p, lv in zip(m["rungs"], (716.8, 1740.8))), m)
    lad = KL.k3_ladder(KL.k3_pin_steps(kit))
    check("T19 rev 3 the K3 ladder, every step tried: 7.39 -> 6.39 -> 4.91 (v1) -> 3.31 (B3) -> 2.31 GiB (FP8 base)",
          lad["steps"] == [7933526016, 6859784192, 5270142976, 3553580524, 2479838700] and lad["tried_from"] == 0, lad["steps"])
    check("T19 rev 3 the export builds K3_PINS = k3_ladder(k3_pin_steps(K3_GATE_D_GIB)) and tries K3_PINS['steps'] from 0",
          "K3_PINS = KL.k3_ladder(KL.k3_pin_steps(K3_GATE_D_GIB))" in EXPORT_SRC and 'KL.ladder_run(K3_PINS["steps"], k3_launch' in EXPORT_SRC
          and 'K1["k3_first_rung"] = K3_PINS["steps"][0]' in EXPORT_SRC and 'K3_PINS["steps"][1:]' not in EXPORT_SRC)
    check("T19 rev 3 a free reading with no room above the pin adds no rung", KL.k3_measured_rungs({0: 500.0, 1: 500.0, 2: 500.0, 3: 500.0})["rungs"] == [])
    a = {"ok": True, "steps": [{"frac": 0.9, "kv_bytes": 10}, {"frac": 0.8, "kv_bytes": 8}]}
    b = {"ok": True, "steps": [{"frac": 0.9, "kv_bytes": 9}, {"frac": 0.8, "kv_bytes": 9}]}
    m = KL.ladder_min(a, b)
    check("T19 ladder_min: per frac the lower pin, its source", [(x["kv_bytes"], x["source"]) for x in m["steps"]] == [(9, "K3"), (8, "K0")], m)
    check("T19 ladder_min: one side not ok -> the other", KL.ladder_min(a, {"ok": False})["steps"] == a["steps"]
          and KL.ladder_min(None, b)["sources"] == ["K3"])
    # k100_facts: FP8 read-back + static scales gate
    log_ = ("INFO non-default args: {'model': '/ck', 'kv_cache_dtype': 'fp8_e4m3', 'max_num_seqs': 16, 'kv_cache_memory_bytes': 123, "
            "'enable_prefix_caching': True, 'speculative_config': {'num_speculative_tokens': 3}, 'num_speculative_tokens': 3}\n"
            "INFO Initializing a V1 LLM engine with config: model='/ck', speculative_config=SpeculativeConfig(method=mtp, num_spec_tokens=3), "
            "dtype=torch.bfloat16, quantization=inc, kv_cache_dtype=fp8_e4m3\n")
    side = {f"language_model.model.layers.{L_}.self_attn.attn": {"k_scale": 0.02 + L_ / 1000, "v_scale": 0.04 + L_ / 1000}
            for L_ in range(3, 48, 4)}
    ck_dir = TMP / "t19ck"
    ck_dir.mkdir(exist_ok=True)
    (ck_dir / "qsa_kv_scales_262k.json").write_text(json.dumps(side))

    def scale_lines(ranks=(0, 1, 2, 3), drop=None, wrong=None, extra_pid=False, swap=False):
        out = []
        for r in ranks:
            pp, tp_ = r // 2, r % 2
            lay = [L_ for L_ in range(3, 48, 4) if (L_ < 25) == (pp == 0)]
            if swap and r == 0:
                lay = [L_ for L_ in range(3, 48, 4) if L_ >= 25]
            for L_ in lay:
                if (r, L_) == drop:
                    continue
                k, v = side[f"language_model.model.layers.{L_}.self_attn.attn"].values()
                if (r, L_) == wrong:
                    k = k * 1.5
                out.append(f"(Worker_PP{pp}_TP{tp_} pid={100 + r}) INFO QSA E4M3 KV active on language_model.model.layers.{L_}.self_attn.attn: "
                           f"k_scale={k:.6g} v_scale={v:.6g}")
            if pp == 1:
                out.append(f"(Worker_PP{pp}_TP{tp_} pid={100 + r}) INFO QSA E4M3 KV active on mtp.layers.48.self_attn.attn: k_scale=1 v_scale=1 (DEFAULT 1.0)")
            out.append(f"(Worker_PP{pp}_TP{tp_} pid={100 + r}) INFO applied static K/V scales from /ck/qsa_kv_scales_262k.json to 6 of 6 layer(s)")
        if extra_pid:
            out.append("(Worker_PP0_TP0 pid=999) INFO QSA E4M3 KV active on language_model.model.layers.3.self_attn.attn: k_scale=0.023 v_scale=0.043")
        return "\n".join(out)
    r = KL.scales_check(scale_lines(), side)
    check("T19 V-4 four ranks, each its stage's 6 QSA layers with the sidecar values, MTP at DEFAULT 1.0 recorded -> PASS",
          r["ok"] is True and r["per_rank"][0] == [3, 7, 11, 15, 19, 23] and r["per_rank"][3] == [27, 31, 35, 39, 43, 47]
          and set(r["mtp_default_scale"]) == {2, 3}, r)
    for lab_, kw in (("a missing rank", {"ranks": (0, 1, 2)}), ("a duplicated rank (two pids)", {"extra_pid": True}),
                     ("a wrong subset (a layer missing)", {"drop": (1, 11)}), ("a wrong subset (PP1 layers on a PP0 rank)", {"swap": True}),
                     ("a wrong value", {"wrong": (2, 31)})):
        r = KL.scales_check(scale_lines(**kw), side)
        check(f"T19 V-4 {lab_} -> STATIC_SCALES FAIL", r["ok"] is False and r["why"], r.get("why"))
    check("T19 V-4 no worker tag on the lines -> every rank missing -> FAIL",
          KL.scales_check(re.sub(r"\(Worker_PP\d_TP\d ", "(EngineCore ", scale_lines()), side)["ok"] is False)
    for ok_sc, want in ((True, True), (False, False)):
        lg = log_ + (scale_lines() if ok_sc else scale_lines(ranks=(0, 1, 2)))
        class CkPath(str):                                    # str() = '/ck' (the read-back's model dir); / -> the real sidecar dir
            def __truediv__(self, x):
                return ck_dir / x
        fns = {"re": re, "json": json, "KL": KL, "CKPT": CkPath("/ck"), "CAPS_LIST": {"C": [1, 2]}, "KV_FP8_DTYPE": "fp8_e4m3",
               "SCALES_CALIBRATION": "EXPERIMENTAL_REUSED_CALIBRATION (the W4-calibrated QSA sidecar reused on klee100)",
               "server_caps": lambda t, argv: {"capture_sizes": [1, 2]}, "pool_facts": lambda t: {}, "graph_facts": lambda t: {},
               "rpc_cache_info": lambda p: {}, "rpc": lambda port, method: {"results": full_map()},
               "scales_facts": lambda t: {"ok": True}}
        load(["flags", "_pick", "k100_facts"], fns)
        s = types.SimpleNamespace(log=lambda lg=lg: lg, argv=["vllm", "serve", "/ck", "--kv-cache-memory-bytes", "123"], port=1,
                                  env={"VLLM_PP_LAYER_PARTITION": "25,23"})
        o = fns["k100_facts"](s, {"key": "K3", "seqs": 16, "captures": "C", "mtp": True, "kv": "fp8"}, {})
        check(f"T19 K3 read-back: FP8 E4M3 KV, MTP K3, method map with MTP, per-rank scales {'all 4' if ok_sc else '3 of 4'} -> config_ok {want}",
              o["config_ok"] is want and o["readback"]["checks"]["kv_dtype_args"] and o["readback"]["checks"]["GPU_METHOD_MAP_PASS"]
              and o["readback"]["checks"]["STATIC_SCALES_APPLIED"] is ok_sc and o["readback"]["scales_calibration"].startswith("EXPERIMENTAL_REUSED")
              and o["scales"]["calibration"].startswith("EXPERIMENTAL_REUSED"), o["readback"]["checks"])
    # k100_scales: the Merlin tree first, the dataset second, sha-checked, copied into the model dir
    d = TMP / "t19"
    shutil.rmtree(d, ignore_errors=True)
    (d / "tree/models/flash-next/scales").mkdir(parents=True)
    (d / "ref").mkdir()
    (d / "ck").mkdir()
    good = b'{"a": {"k_scale": 1, "v_scale": 2}}'
    (d / "tree/models/flash-next/scales/qsa_kv_scales_262k.json").write_bytes(good)
    (d / "ref/qsa_kv_scales_262k.json").write_bytes(b"{}")
    ns = {"hashlib": hashlib, "json": json, "Path": Path, "REF_DIR": d / "ref", "CKPT": d / "ck", "K100_SCALES_SHA": sha(good),
          "SCALES_CALIBRATION": "EXPERIMENTAL_REUSED_CALIBRATION"}
    load(["k100_scales"], ns)
    r = ns["k100_scales"](str(d / "tree"))
    check("T19 k100_scales: the installed Merlin tree first; sha ok; copied into the model dir", r["ok"] and "tree" in r["src"]
          and (d / "ck/qsa_kv_scales_262k.json").read_bytes() == good, r)
    r = ns["k100_scales"](None)
    check("T19 k100_scales: no tree -> the dataset file; a changed file -> ok False (K3 skipped, not a gate)", r["ok"] is False
          and r["looked"][0]["ok"] is False and r["looked"][0]["sha256"], r)
    (d / "tree/models/flash-next/scales/qsa_kv_scales_262k.json").write_bytes(b"{corrupt")
    (d / "ref/qsa_kv_scales_262k.json").write_bytes(good)
    r = ns["k100_scales"](str(d / "tree"))
    check("T19 V-5 a corrupt Merlin-tree copy does not block the valid dataset copy (both candidates recorded with their sha)",
          r["ok"] and "ref" in r["src"] and [x["ok"] for x in r["looked"]] == [False, True] and all(x.get("sha256") for x in r["looked"]), r)
    (d / "ref/qsa_kv_scales_262k.json").write_bytes(b"{}")
    r = ns["k100_scales"](str(d / "tree"))
    check("T19 V-5 both copies bad -> ok False, both recorded", r["ok"] is False and len(r["looked"]) == 2, r)
    ns["REF_DIR"] = None
    check("T19 k100_scales: neither -> ok False", ns["k100_scales"](None)["ok"] is False)
    b3s = src_of("body_k3")
    order = ["_oracle_gate", 'tf_suite("K3", s, ["2k", "8k"])', "k100_capacity(", '("sw20k", "sw40k")', '("sw60k", "sw100k")',
             'tf_suite("K3", s, ["40k"])', "admission_at_best(", "arith_probe("]
    check("T19 V-3 / V-14 K3 order: core (oracle, TF 2k + 8k, pool / F), the 20k + 40k sweeps, 60k + 100k, TF 40k, admission, arithmetic",
          [b3s.index(x) for x in order] == sorted(b3s.index(x) for x in order) and b3s.count("gate=k3_optional_ok") == 3
          and "admission_at_best(o, s, cap, gate=k3_optional_ok" in b3s, [b3s.find(x) for x in order])
    check("T19 _start: read-back + method map before the probes (the K3 order)", src_of("_start").index("k100_facts") < src_of("_start").index("chat_probes_full"))
    stops = export_const_json("K100_STOPS_MIN")
    check("T19 V-3 / V-9 the per-profile hard stops are in the notebook (ALL <= 190 min)", set(stops) == {"ALL", "K3_ONLY", "BF16_ONLY"}
          and stops["ALL"] <= 190 and 'K100_PROFILE = (os.environ.get("K100_PROFILE") or "ALL")' in EXPORT_SRC, stops)


def export_const_json(name):
    """A dict literal assigned in the export (e.g. K3_GATE_D_GIB = {...})."""
    node = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)][-1]
    return ast.literal_eval(node.value)


# ======================================================================================== T21 incomplete TF responses (agent C's finding)
def t21():
    import time as _t
    ids = [11, 12, 13, 14, 15]
    good = [None] + [{f"token_id:{t}": {"logprob": -0.1, "rank": 1}} for t in ids[1:]]
    cases = (("prompt_logprobs null", None), ("absent", "ABSENT"), ("all None", [None] * 5), ("cut short", good[:3]),
             ("forced token missing at a position", good[:2] + [{"token_id:99": {"logprob": -1.0, "rank": 1}}] + good[3:]))
    for lab_, pl in cases:
        ch = {"token_ids": [1], "text": "x"}
        if pl != "ABSENT":
            ch["prompt_logprobs"] = pl
        ns = {"np": np, "json": json, "hashlib": hashlib, "time": _t, "T_START": 0.0, "NB0": {"topk": 1}, "RAW_DIR": TMP,
              "_post": lambda port, body, timeout=1800, ch=ch: ({"choices": [ch]}, None, 0.1), "_req_log": lambda rec: None,
              "L": types.SimpleNamespace(_tok_id=lambda k: int(str(k).split(":")[-1]) if str(k).split(":")[-1].isdigit() else None)}
        load(["ids_sha", "_cached", "pl_arrays", "tf_request"], ns)
        m = ns["tf_request"]("K", 1, ids, "2k", "fresh")["meta"]
        g = {"KL": KL}
        load(["tf_gate"], g)
        check(f"T21 R4 HTTP 200 with {lab_}: INCOMPLETE_TF error, tf_gate stops the boot, core not done",
              str(m["error"]).startswith("INCOMPLETE_TF") and g["tf_gate"]({"2k": {"fresh": m}}).startswith("NO_GO_TF_REQUEST_ERROR")
              and KL.tf_core_done({b: {"fresh": m} for b in ("2k", "8k", "40k")}) is False, m)
    ns["_post"] = lambda port, body, timeout=1800: ({"choices": [{"token_ids": [1], "prompt_logprobs": good}]}, None, 0.1)
    m = ns["tf_request"]("K", 1, ids, "2k", "fresh")["meta"]
    check("T21 R4 a complete response: valid == expected (p >= 1), no error", m["error"] is None and m["valid"] == m["expected"] == 4, m)
    ns["_post"] = lambda port, body, timeout=1800: ({"choices": [{"token_ids": [1], "prompt_logprobs": [None]}]}, None, 0.1)
    m = ns["tf_request"]("K", 1, [7], "2k", "fresh")["meta"]
    check("T21 R4 the 1-position edge: expected 0, valid 0 -> no error", m["error"] is None and m["valid"] == m["expected"] == 0, m)
    g = {"KL": KL}
    load(["tf_gate"], g)
    check("T21 R4 belt and braces: valid < expected without an error string -> NO_GO_TF_INCOMPLETE; core not done",
          g["tf_gate"]({"8k": {"fresh": {"error": None, "valid": 3, "expected": 9}}}).startswith("NO_GO_TF_INCOMPLETE")
          and KL.tf_core_done({b: {"fresh": {"error": None, "valid": 3, "expected": 9}} for b in ("2k", "8k", "40k")}) is False
          and KL.tf_core_done({b: {"fresh": {"error": None, "valid": 9, "expected": 9}} for b in ("2k", "8k", "40k")}) is True)


# ======================================================================================== T20 R2-4 / R2-9 / R2-10 analysis, references, traces
def t20():
    import threading as _th
    import time as _t
    # R2-4: headline = K0 vs A when K0 has the band, else K vs A; K0 vs K = the pin only
    rng = np.random.default_rng(1)
    n = 400

    def arr(shift=0.0):
        ids = np.arange(n, dtype=np.int32)
        lp = (-rng.random(n)).astype(np.float32) if shift == 0 else None
        return {"ids": ids, "lp": lp, "rank": np.ones(n, np.int32), "top1": ids.copy(), "top1_lp": np.zeros(n, np.float32),
                "valid": np.r_[False, np.ones(n - 1, bool)]}
    A = arr()
    K0 = dict(A, lp=(A["lp"] + np.float32(0.02)))
    K = dict(A, lp=(A["lp"] + np.float32(0.05)))
    meta = {"prompt_sha256": "p"}
    ns = tf_ns()
    ns.update(KL=KL, REF={"meta": {"yardstick": {"pairs": {}, "per_boot": {}}}}, K1={"reference": {}}, SCR={}, RECS={}, RC2_WINDOWS={},
              NB0={"tf_bands": ["2k", "8k", "40k", "90k"], "sc_bands": [], "last": 256}, sc_compare=lambda *a: {}, decode_compare=lambda *a: {})
    ns["TFR"] = {"A": {b: {"fresh": {"meta": meta, "arr": A}} for b in ("2k", "8k", "40k", "90k")},
                 "K0": {b: {"fresh": {"meta": meta, "arr": K0}} for b in ("2k", "8k", "40k")},
                 "K": {b: {"fresh": {"meta": meta, "arr": K}} for b in ("2k", "8k", "40k", "90k")}}
    load(["_arr", "cmp_pair", "tf_skips", "k100_analysis"], ns)
    an = ns["k100_analysis"]()
    hl = an["tf_headline"]
    check("T20 R2-4 headline: K0 vs A (weights only) where K0 has the band, K vs A at 90k (K0 has no 90k here)",
          [hl[b]["headline_source"].split(" ")[0] for b in ("2k", "8k", "40k", "90k")] == ["K0_vs_A", "K0_vs_A", "K0_vs_A", "K_vs_A"]
          and abs(hl["2k"]["full"]["mean_abs_dlogprob"] - 0.02) < 1e-3 and abs(hl["90k"]["full"]["mean_abs_dlogprob"] - 0.05) < 1e-3, hl["2k"]["full"])
    check("T20 R2-4 K0 vs K = the pin only (0.03); K vs A = weights + pin (0.05); sources recorded",
          abs(an["tf_k0_vs_k"]["2k"]["full"]["mean_abs_dlogprob"] - 0.03) < 1e-3 and abs(an["tf_k_vs_A"]["2k"]["full"]["mean_abs_dlogprob"] - 0.05) < 1e-3
          and an["tf_reading"]["sources"]["2k"].startswith("K0_vs_A") and "nb0_prefix_consistency" in hl["2k"]["noise_floor"], an["tf_reading"])
    check("T20 R3-1 the K3 pairs INCONCLUSIVE when K3 did not run", an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"]["2k"]["label_full"].startswith("INCONCLUSIVE")
          and an["tf_k3_vs_k2_reading"]["reading"] == "TF_NOT_MEASURED" and an["fp8_headline"]["verdict"] == "INCONCLUSIVE_NO_K2")
    ns["TFR"]["K3"] = {b: {"fresh": {"meta": meta, "arr": dict(A, lp=(A["lp"] + np.float32(0.08)))}} for b in ("2k", "8k", "40k")}
    an = ns["k100_analysis"]()
    check("T20 V-6 K3 vs K is labelled FP8 + MTP vs BF16 no-MTP (0.03, not 'the FP8 effect alone'); no K2 -> headline INCONCLUSIVE_NO_K2",
          abs(an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"]["8k"]["full"]["mean_abs_dlogprob"] - 0.03) < 1e-3
          and "not the KV effect alone" in an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"]["8k"]["meaning"] and "tf_k3_vs_k" not in an
          and an["fp8_headline"]["verdict"] == "INCONCLUSIVE_NO_K2" and an["tf_k3_vs_A_reading"]["complete"] is True, an["fp8_headline"])
    ns["TFR"]["K2"] = {b: {"fresh": {"meta": meta, "arr": dict(A, lp=(A["lp"] + np.float32(0.07)))}} for b in ("2k", "8k", "40k")}
    ns["RECS"]["K2"] = {"status": "DONE"}
    an = ns["k100_analysis"]()
    check("T20 V-6 K3 vs K2 = the FP8 KV effect under MTP (0.01) -> the FP8 headline (complete over 2k / 8k); MTP parity reported",
          abs(an["tf_k3_vs_k2"]["2k"]["full"]["mean_abs_dlogprob"] - 0.01) < 1e-3 and an["fp8_headline"]["verdict"] == "LOWBIT_TF_CLOSE"
          and an["fp8_headline"]["source"].startswith("K3 vs K2") and "parity" in an["tf_k2_vs_k_40k_mtp_effect"]["meaning"], an["fp8_headline"])
    # R2-9
    w = export_const_json_eval("RC2_WINDOWS")
    check("T20 R2-9 references: NB1 R1 rows relabelled, NB1 R2 B1 / B3 rows added",
          w["nb1_r1_b1_fp8_8k@8"]["tok_s"] == 280.0 and w["nb1_r2_b1_fp8_8k@8"] == {"tok_s": 261.0, "tick_ms": 48.2, "mal": 3.148}
          and w["nb1_r2_b1_fp8_40k@5"]["tok_s"] == 196.3 and w["nb1_r2_b1_fp8_40k@7"]["tick_ms"] == 46.7 and w["nb1_r2_b3_fp8_40k@8"]["tok_s"] == 249.5)
    # R2-10: decode token ids on error, full probe content, the server's template ids
    logged = []
    ns = {"np": np, "json": json, "hashlib": hashlib, "time": _t, "T_START": 0.0, "NB0": {"topk": 5}, "RAW_DIR": TMP,
          "_post": lambda port, body, timeout=900: ({"choices": [{"token_ids": [1, 2], "logprobs": {"token_logprobs": [-0.1, -0.2], "top_logprobs": [{}, {}]},
                                                                 "finish_reason": "length", "text": "ab"}]}, None, 0.1),
          "_req_log": lambda rec: logged.append(rec), "L": types.SimpleNamespace(_tok_id=lambda k: None)}
    load(["ids_sha", "_cached", "decode"], ns)
    ns["decode"]("K", 1, [5, 6], 4, "2k", "cold")
    check("T20 R2-10 decode(): an INCOMPLETE decode keeps its token ids in the request log", logged[-1]["error"].startswith("INCOMPLETE_DECODE")
          and logged[-1]["token_ids"] == [1, 2], logged[-1])
    tr = TMP / "traces20"
    tr.mkdir(exist_ok=True)

    class Resp:
        def __init__(self, b):
            self.b = b

        def read(self):
            return self.b
    long_ans = "42" + " because" * 100
    U = types.SimpleNamespace(Request=lambda *a, **k: None,
                              urlopen=lambda req, timeout=0: Resp(json.dumps({"choices": [{"message": {"content": long_ans, "reasoning": "r" * 500},
                                                                                           "finish_reason": "stop"}], "tokens": [7, 8, 9]}).encode()))
    ns = {"json": json, "re": re, "time": _t, "_SAVE_LOCK": _th.Lock(), "TRACES": tr, "BOOT": {"tag": "k100_K3_x"},
          "L": types.SimpleNamespace(U=U, degenerate=lambda t: None, template_probe=lambda port, m, w, model="q27", timeout=120: {"status": "MISMATCH"})}
    load(["chat_probes_full", "_template_probe_lab", "k100_template_probe"], ns)
    r = ns["chat_probes_full"](1, (("q", "42"),))
    row = json.loads((tr / "chat_probes.jsonl").read_text().splitlines()[-1])
    check("T20 R2-10 chat probes: the FULL content + reasoning in the trace (the record keeps 200 chars)",
          row["content_full"] == long_ans and len(row["reasoning_full"]) == 500 and len(r["cases"][0]["content"]) == 200 and r["ok"], row.keys())
    t_ = ns["k100_template_probe"](1, [{"role": "user", "content": "x"}], [7, 8])
    saved = json.loads((tr / "template_probe_k100_K3_x.json").read_text())
    check("T20 R2-10 template probe: the lab's verdict kept + the server's token ids saved", t_["status"] == "MISMATCH"
          and saved["server_tokens"] == [7, 8, 9] and saved["want_tokens"] == [7, 8], saved)


def export_const_json_eval(name):
    node = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)][-1]
    return eval(compile(ast.Expression(node.value), "<const>", "eval"), {})


# ======================================================================================== T22 schedule replay (V-3 / V-9 / V-14)
REPLAY_DUR = {"tf": {"2k": 8, "8k": 15, "40k": 45, "90k": 110}, "decode": 25, "oracle": 60, "start": 60, "warmup": 50, "prefill": 20,
              "arith": 30, "teardown": 150, "prefill_tok_s": 4000.0}


def replay(profile, stop_min, setup_min, ready_min, k3_body_oom=False, k3_launch_oom=False, k3_launch_ooms=None):
    """The REAL CELL 8 + the REAL boot bodies (tf_suite / sc_suite / kv_sweep / admission_at_best / the reserves / k2_plan /
    the ladders) on a virtual clock: every server call is a stub that advances the clock by a modelled duration (TF per band,
    decode 25 s, a window = level x L / 4,000 tok/s + 90 s + 30 s, the 64k admission = n x 60k / 4,000 + 150 s, the 120k row up
    to its 900 s cap, ready = `ready_min` per boot, teardown 150 s, A0 = `setup_min`). Returns what ran and what was skipped."""
    T = [0.0]
    adv = lambda sec: T.__setitem__(0, T[0] + float(sec))
    ft = types.SimpleNamespace(time=lambda: T[0], perf_counter=lambda: T[0], sleep=adv, strftime=__import__("time").strftime,
                               gmtime=__import__("time").gmtime)
    budget = stop_min * 60.0
    tries = {"K3": 0}
    ns = {"json": json, "time": ft, "traceback": traceback, "os": os, "KL": KL, "math": math, "RERUN": False, "MERLIN_OK": True,
          "R": {"sections": {"A0": {"merge": {"ok": True}}}, "inputs": {}, "setup": {}, "errors": []}, "GATES": {}, "RECS": {},
          "LOADMON": {}, "BOOT": {"tag": None, "deadline": None, "fired": None}, "MERLIN_ENG": {"kv": {"MERLIN_TREE": "/tree"}},
          "k100_scales": lambda tree: {"ok": True, "layers": 12},
          "K1": {"reference": {"ok": True}, "prompt_identity": GOOD_PID}, "K1BOOTS": {}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {},
          "EXT_FLAG": ["--x"], "DIAG_ENV": {}, "KV_DEFAULT": 2464415744, "FP8_CAPTURES": "caps", "COMPILE_HI": "hi", "T_START": 0.0,
          "HOST_PHASE": {}, "ABORT": {"why": None}, "OUT": TMP, "save": lambda: None, "log": lambda *a: None, "print": lambda *a, **k: None,
          "write_fallback_submission": lambda: None, "left": lambda: budget - T[0], "NO_INNER_RETRY": set(),
          "K100_PROFILE": profile, "BUDGET": budget, "SCALES_CALIBRATION": "x", "gpu_health_gate": lambda: {"ok": True},
          "method_map_summary": lambda o: None, "k100_analysis": lambda: {"tf_reading": {"reading": "LOWBIT_TF_CLOSE"},
                                                                          "tf_k3_vs_A_reading": {"reading": "LOWBIT_TF_CLOSE", "complete": True}},
          "merlin_install": lambda: (adv(setup_min * 60 * 0.7), {"kv": {"MERLIN_TREE": "t"}})[1], "apply_60490": lambda: {"ok": True},
          "sha_gate": lambda t: (adv(setup_min * 60 * 0.3), (True, "ok"))[1],
          # the boot-body world
          "NB0": {"topk": 5, "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6}, "sc_bands": ["2k", "8k", "40k"], "sc_tok": 256,
                  "tf_bands": ["2k", "8k", "40k", "90k"]},
          "PROMPTS": {b: [[1]] * 16 for b in ("2k", "8k", "40k", "60k", "90k", "120k")}, "TFR": {}, "SCR": {}, "SIDS": set(),
          "BANDS": {"2k": 2000, "8k": 8000, "40k": 40000, "60k": 60000, "90k": 90000, "120k": 120000}, "MAX_NEW": {}, "MAX_LEN": 131072,
          "PREFILL": {"tok_s": REPLAY_DUR["prefill_tok_s"]}, "WIN": 90.0, "GEN_RATE": 45.0, "BLOCK": 4096,
          "ids_sha": lambda ids: "s", "cmp_last256": lambda a, b: {}, "sc_compare": lambda *a: {}, "decode_compare": lambda *a: {},
          "prompt": lambda L_, v: [v], "traced": lambda label, fn: fn(), "live_pin": lambda b: b["kv_bytes"],
          "k_body": lambda b, fn: fn, "oracle_ok": lambda orc: True, "cmp_pair": lambda a, c, y=None: {"label_full": "LOWBIT_TF_CLOSE"},
          "_arr": lambda *a: None}
    ns["tf_request"] = lambda key, port, ids, band, run, k=None: (adv(REPLAY_DUR["tf"].get(band, 30)),
                                                                  {"meta": {"error": None, "valid": 9, "expected": 9}, "arr": None})[1]
    ns["decode"] = lambda key, port, ids, n, band, run: (adv(REPLAY_DUR["decode"]), {"meta": {"error": None, "cached_tokens": 0}, "tok": [1]})[1]
    ns["_oracle_gate"] = lambda o, s, key, refs: (adv(REPLAY_DUR["oracle"]), True)[1]
    ns["warmup_gauge"] = lambda port, n: (adv(REPLAY_DUR["warmup"]), ({}, []))[1]
    ns["prefill_probe"] = lambda port: (adv(REPLAY_DUR["prefill"]), {"ok": True})[1]
    ns["arith_probe"] = lambda port: (adv(REPLAY_DUR["arith"]), {"exact": 8})[1]

    def capacity(o, us, kv_bytes, kv="bf16"):
        key = ns["BOOT"]["tag"] or ""
        P, seqs = (460, 32) if kv == "fp8" else ((210, 16) if "_K2_" in key else (210, 8))
        return {"capacity_ok": True, "pool_blocks": P, "F": 6, "max_num_seqs": seqs, "n64": P // (16 + 6)}
    ns["k100_capacity"] = capacity
    ns["window"] = lambda port, s, band, n: (adv(n * ns["BANDS"][band] / REPLAY_DUR["prefill_tok_s"] + 90 + 30),
                                             {"status": "STEADY", "running_min": n, "running_max": n, "running_eq_level": True,
                                              "preemptions": 0, "admission": {"sec": n * ns["BANDS"][band] / REPLAY_DUR["prefill_tok_s"]}})[1]
    ns["admission_test"] = lambda port, s, band, n, max_new: (adv(n * 60000 / REPLAY_DUR["prefill_tok_s"] + 150), {"status": "ALL_RESIDENT"})[1]

    def long_row(port):
        adv(min(900, ns["boot_left"]() - 240) + 60)
        return {"verdict": "CAPPED_BY_TIME_CLEAN"}
    ns["long_prefill"] = long_row
    load(["K100", "PORTS", "CAP_MIN", "EST_MIN", "BODY_MIN", "K3_GATE_D_GIB", "K3_PINS", "TF_EST", "SC_EST", "k1_boot", "boot_summary",
          "k3_core_marker",
          "boot_brief", "tf_brief", "boot_left", "due", "run_item", "tf_gate", "_tf_stop", "tf_suite", "sc_suite", "k_ready_sec",
          "k2_reserve_sec", "k2_reserve_ok", "k3_ready_sec", "later_reserve_sec", "k3_optional_ok", "health_after", "SWEEP_BANDS",
          "sweep_prompts", "gen_during_window", "level_need_sec", "kv_sweep", "admission_at_best", "body_k0", "body_k", "body_k2", "body_k3"], ns)
    for k_, (L_, v_) in ns["SWEEP_BANDS"].items():
        ns["BANDS"][k_], ns["MAX_NEW"][k_] = L_, 131072 - L_ - 512

    def step(sec, name, est, fn):
        if ns["ABORT"]["why"]:
            return {"status": "SKIPPED_ABORT"}
        if ns["left"]() < est * 60 + 300:
            return {"status": "SKIPPED_BUDGET", "est_min": est, "left_min": round(ns["left"]() / 60, 1)}
        out = fn() or {}
        if out.get("status") in (None, "NOT_RUN"):
            out["status"] = "DONE"
        return out

    def run_k(b, body):
        key, r = b["key"], ready_min * 60.0
        if key == "K3":
            tries["K3"] += 1
            if tries["K3"] <= (k3_launch_ooms if k3_launch_ooms is not None else int(bool(k3_launch_oom))):   # rev 3: N launch OOMs
                adv(0.8 * r + REPLAY_DUR["teardown"])
                return {"status": "LAUNCH_FAILED", "oom_at_launch": True, "kv_bytes": b["kv_bytes"], "boot_name": b["name"]}
        adv(r)
        ns["LOADMON"][b["name"]] = {"ready_sec": r}
        ns["BOOT"]["tag"] = b["name"]
        ns["BOOT"]["deadline"] = T[0] + max(60.0, min(ns["CAP_MIN"][key] * 60, ns["left"]() - 480))
        if key == "K3" and k3_body_oom and tries["K3"] == 1 + (k3_launch_ooms or 0):
            adv(REPLAY_DUR["start"] + 120)
            ns["BOOT"]["deadline"] = None
            adv(REPLAY_DUR["teardown"])
            return dict(CLEAN, status="TF_OOM", core_done=False, gpu_released=True, kv_bytes=b["kv_bytes"], ready_sec=r, boot_name=b["name"])
        adv(REPLAY_DUR["start"])
        o = {}
        body(types.SimpleNamespace(alive=lambda: True, port=1), o)
        ns["BOOT"]["deadline"] = None
        adv(REPLAY_DUR["teardown"])
        return dict(CLEAN, **o, kv_bytes=b["kv_bytes"], ready_sec=r, boot_name=b["name"],
                    free_mib_by_card_all=dict(K3FREE if key == "K3" else FREE4))
    ns.update(step=step, run_k=run_k)
    exec(compile(ast.Module(body=cell8_node().body, type_ignores=[]), "<export CELL 8 main>", "exec"), ns)
    rc = ns["RECS"]

    def tfst(rec, bands):
        out = {}
        for b_ in bands:
            x = ((rec or {}).get("tf") or {}).get(b_) or {}
            f_ = x.get("fresh") if isinstance(x.get("fresh"), dict) else {}
            out[b_] = x.get("status") or f_.get("status") or ("ok" if f_ else "absent")
        return out
    k3, k0, k, k2 = rc.get("K3") or {}, rc.get("K0") or {}, rc.get("K") or {}, rc.get("K2") or {}
    sw3 = {kk: v.get("status") for kk, v in (k3.get("sweep") or {}).items()}
    return {"profile": profile, "stop": stop_min, "setup": setup_min, "ready": ready_min, "k3_body_oom": k3_body_oom, "k3_launch_oom": k3_launch_oom,
            "used_min": round(T[0] / 60, 1), "label": ns["FINAL"]["label"],
            "K3": {"status": k3.get("status"), "tries": tries["K3"], "tf": tfst(k3, ("2k", "8k", "40k")), "sweep_run": sorted(x for x, v in sw3.items() if v == "STEADY"),
                   "sweep_best": k3.get("sweep_best"), "skipped": k3.get("k3_optional_skipped"), "admission": (k3.get("admission_64k") or {}).get("status")},
            "K0": {"status": k0.get("status"), "tf": tfst(k0, ("2k", "8k", "40k", "90k"))},
            "K": {"status": k.get("status"), "tf": tfst(k, ("2k", "8k", "40k", "90k")), "long": (k.get("long_120k") or {}).get("verdict"),
                  "sc": {b_: (v or {}).get("status", "ok") for b_, v in (k.get("sc") or {}).items()}},
            "K2": {"status": k2.get("status"), "sweep_best": k2.get("sweep_best"), "admission": (k2.get("admission_64k") or {}).get("status")}}


def bf16_essentials(r):
    """K0's TF bands and K's core never skipped (the K3 optional items must not eat them)."""
    return (all(v == "ok" for v in r["K0"]["tf"].values()) and all(r["K"]["tf"][b_] == "ok" for b_ in ("2k", "8k", "40k")))


def t22():
    stops = export_const_json("K100_STOPS_MIN")
    rows = []
    scen = [(8, 8, False, False), (15, 13, False, False), (8, 8, True, False), (8, 8, False, True), (15, 13, True, False), (15, 13, False, True)]
    for profile in ("ALL", "K3_ONLY", "BF16_ONLY"):
        for setup, ready, bo, lo in scen:
            if profile == "BF16_ONLY" and (bo or lo):
                continue
            r = replay(profile, stops[profile], setup, ready, k3_body_oom=bo, k3_launch_oom=lo)
            rows.append(r)
            if profile in ("ALL", "BF16_ONLY"):
                check(f"T22 {profile} stop {stops[profile]} setup {setup} / ready {ready}{' + K3 body OOM' if bo else ''}{' + K3 launch OOM' if lo else ''}: "
                      f"K0's TF bands and K's core never skipped", bf16_essentials(r), r)
                check(f"T22 {profile} stop {stops[profile]} setup {setup} / ready {ready}{' + K3 body OOM' if bo else ''}{' + K3 launch OOM' if lo else ''}: "
                      f"K2 runs (reserved)", r["K2"]["status"] == "DONE", r["K2"])
            if profile in ("ALL", "K3_ONLY"):
                check(f"T22 {profile} setup {setup} / ready {ready}{' + K3 body OOM' if bo else ''}{' + K3 launch OOM' if lo else ''}: "
                      f"K3's core ran (TF 2k + 8k)", r["K3"]["tf"]["2k"] == "ok" and r["K3"]["tf"]["8k"] == "ok", r["K3"])
            if profile == "K3_ONLY" and not (bo or lo) and setup == 8:
                check("T22 K3_ONLY nominal: the 20k + 40k sweeps run", any(x.startswith("sw20k") for x in r["K3"]["sweep_run"])
                      and any(x.startswith("sw40k") for x in r["K3"]["sweep_run"]), r["K3"])
    # rev 3: the two raised K3 rungs both launch-OOM (the worst case of the higher ladder), with and without a body OOM at
    # the third rung (the v1 rung 5,270,142,976): K3_ONLY still runs its core, and its sweeps in the nominal setup
    for setup, ready, n_oom, bo in ((8, 8, 2, False), (15, 13, 2, False), (8, 8, 2, True), (15, 13, 2, True)):
        r = replay("K3_ONLY", stops["K3_ONLY"], setup, ready, k3_body_oom=bo, k3_launch_ooms=n_oom)
        r["k3_launch_ooms"] = n_oom
        rows.append(r)
        check(f"T22 rev 3 K3_ONLY stop {stops['K3_ONLY']} setup {setup} / ready {ready} + {n_oom} K3 launch OOMs{' + a body OOM' if bo else ''}: "
              f"K3's core ran (TF 2k + 8k)", r["K3"]["tf"]["2k"] == "ok" and r["K3"]["tf"]["8k"] == "ok", r["K3"])
        if setup == 8 and not bo:
            check(f"T22 rev 3 K3_ONLY + {n_oom} launch OOMs, setup 8 / ready 8: the 20k + 40k sweeps run",
                  any(x.startswith("sw20k") for x in r["K3"]["sweep_run"]) and any(x.startswith("sw40k") for x in r["K3"]["sweep_run"]), r["K3"])
    REPLAY_ROWS.extend(rows)
    print("REPLAY " + json.dumps([{k: r[k] for k in ("profile", "stop", "setup", "ready", "k3_body_oom", "k3_launch_oom", "used_min")} |
                                  {"k3_sweep": r["K3"]["sweep_run"], "k3_skipped": r["K3"]["skipped"], "k0_tf": r["K0"]["tf"],
                                   "k_tf": r["K"]["tf"], "k_long": r["K"]["long"], "k2": r["K2"]["status"], "k2_best": r["K2"]["sweep_best"]}
                                  for r in rows]), flush=True)


REPLAY_ROWS = []


# ======================================================================================== T24 V-15 the K3 load smoke
def t24():
    printed = []
    ns = {"json": json, "K1": {}, "save": lambda: None, "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
          "live_pin": lambda b: 5270142976, "oracle_ok": lambda orc: bool(orc), "k3_ready_sec": lambda: 512.0,
          "cmp_pair": lambda a, c, y=None: {"label_full": "LOWBIT_TF_CLOSE"}, "_arr": lambda *a: None}
    load(["k3_core_marker"], ns)
    o = {"method_map": {"ok": True}, "scales": {"ok": True}, "oracle": {"complete": True}, "core_done": True}
    mk = ns["k3_core_marker"](o, {"kv_bytes": 1}, {"pool_blocks": 460, "F": 6})
    line = [x for x in printed if x.startswith("@@K3_CORE_DONE ")]
    got = json.loads(line[0].split(" ", 1)[1]) if line else {}
    check("T24 V-15 Rule 2: one @@K3_CORE_DONE line with pin / pool / F / method map / scales / oracle / TF 2k + 8k / ready, saved in K1",
          len(line) == 1 and got == mk == ns["K1"]["k3_core_done"] and got["pin"] == 5270142976 and got["pool_blocks"] == 460
          and got["F"] == 6 and got["method_map_pass"] and got["scales_all_ranks"] and got["oracle"] and got["tf_2k"] == got["tf_8k"] == "LOWBIT_TF_CLOSE"
          and got["ready_sec"] == 512.0, got)
    check("T24 V-15 the marker is printed right after the core, before any sweep",
          src_of("body_k3").index("k3_core_marker(") < src_of("body_k3").index("kv_sweep("))
    r = replay("K3_ONLY", 100, 8, 8)
    check("T24 V-15 the replayed K3_ONLY run prints the core marker (core done)", (r["K3"]["status"] == "DONE"), r["K3"])
    # Rule 1: K3_ONLY, a core gate fails -> no further boot or attempt; the verdict and the fallback are written
    for lab_, rec in (("a core gate (CONFIG_MISMATCH: method map / scales)", {"status": "CONFIG_MISMATCH", "correctness": "INCONCLUSIVE_NO_ORACLE"}),
                      ("an oracle error", {"status": "NO_GO_ORACLE_REQUEST_ERROR", "correctness": "NO_GO_ORACLE_REQUEST_ERROR"}),
                      ("a TF 2k / 8k request error", {"status": "NO_GO_TF_REQUEST_ERROR", "correctness": "CLEAN"})):
        FALLBACK[0] = 0
        ns2, calls = cell8(lambda b, n_, c_, rec=rec: dict(rec, kv_bytes=b["kv_bytes"]), profile="K3_ONLY", left=12000.0)
        check(f"T24 V-15 Rule 1 K3_ONLY + {lab_}: one K3 attempt, no other boot, early exit recorded, verdict + fallback written",
              [k for k, _ in calls["boots"]] == ["K3"] and ns2["K1"].get("k3_early_exit") and ns2["K1"]["k3_core_failed"]["status"] == rec["status"]
              and FALLBACK[0] == 1 and ns2["R"]["verdict"]["k3_early_exit"], (calls["boots"], ns2["K1"].get("k3_core_failed")))
    for lab_, run in (("rungs exhausted (every rung a launch OOM)", lambda b, n_, c_: {"status": "LAUNCH_FAILED", "oom_at_launch": True, "kv_bytes": b["kv_bytes"]}),
                      ("a device fault", lambda b, n_, c_: {"status": "LAUNCH_FAILED", "oom_at_launch": True, "device_fault": True, "kv_bytes": b["kv_bytes"]})):
        ns2, calls = cell8(run, profile="K3_ONLY", left=12000.0)
        n3 = [k for k, _ in calls["boots"]].count("K3")
        check(f"T24 V-15 Rule 1 K3_ONLY + {lab_}: stop, no BF16 boot, early exit recorded", n3 == (5 if "exhausted" in lab_ else 1)
              and set(k for k, _ in calls["boots"]) == {"K3"} and ns2["K1"].get("k3_early_exit"), calls["boots"])
    ns2, calls = cell8(lambda b, n_, c_: dict(CLEAN, kv_bytes=b["kv_bytes"], core_done=True), profile="K3_ONLY")
    check("T24 V-15 a K3 core done -> no early exit", not ns2["K1"].get("k3_early_exit"), ns2["K1"].get("k3_core_failed"))
    # V-16: CORE_DONE needs TF 2k + 8k AND a verified pool (MATCH) AND F
    tf_ok = {b_: {"fresh": {"error": None, "valid": 9, "expected": 9}} for b_ in ("2k", "8k")}
    for lab_, tf_, cap, want in (("TF PASS + WORKER_ONLY + F None", tf_ok, {"capacity_ok": False, "pool_check": "WORKER_ONLY", "pool_blocks": 460, "F": None}, False),
                                 ("TF PASS + MISMATCH + F 6", tf_ok, {"capacity_ok": False, "pool_check": "MISMATCH", "pool_blocks": 460, "F": 6}, False),
                                 ("TF PASS + MATCH + F 6", tf_ok, {"capacity_ok": True, "pool_check": "MATCH", "pool_blocks": 460, "F": 6}, True),
                                 ("a TF request error + MATCH + F 6", dict(tf_ok, **{"8k": {"fresh": {"error": "HTTP 500", "valid": 0, "expected": 9}}}),
                                  {"capacity_ok": True, "pool_check": "MATCH", "pool_blocks": 460, "F": 6}, False),
                                 ("TF PASS + MATCH + no pool blocks", tf_ok, {"capacity_ok": True, "pool_blocks": None, "F": 6}, False)):
        check(f"T24 V-16 k3_core_done: {lab_} -> {want}", KL.k3_core_done(tf_, cap) is want)
    check("T24 V-16 body_k3 sets core_done through k3_core_done", "o[\"core_done\"] = KL.k3_core_done(o[\"tf\"], cap)" in src_of("body_k3"))
    for lab_, cap in (("WORKER_ONLY", {"capacity_ok": False, "pool_check": "WORKER_ONLY", "pool_blocks": 460, "F": None}),
                      ("MISMATCH", {"capacity_ok": False, "pool_check": "MISMATCH", "pool_blocks": 460, "F": 6})):
        FALLBACK[0] = 0
        rec = dict(CLEAN, core_done=KL.k3_core_done(tf_ok, cap), capacity=cap)
        ns2, calls = cell8(lambda b, n_, c_, rec=rec: dict(rec, kv_bytes=b["kv_bytes"]), profile="K3_ONLY", left=12000.0)
        printed = [x for x in calls["printed"] if x.startswith("@@K3_CORE_FAILED ")]
        check(f"T24 V-16 K3_ONLY, TF fine but pool {lab_}: @@K3_CORE_FAILED printed, early exit, verdict + fallback written, no other boot",
              len(printed) == 1 and ns2["K1"].get("k3_early_exit") and FALLBACK[0] == 1 and ns2["R"]["verdict"]["k3_core_failed"]
              and [k for k, _ in calls["boots"]] == ["K3"] and any(x.startswith("@@K100 ") for x in calls["printed"]), (calls["boots"], printed))


# ======================================================================================== T23 V-14 the stream sweep
def t23():
    g = lambda L_, pf=4000.0: (L_ * 8 / pf + 90 + 60) * 45
    rows = {L_: KL.sweep_fit(460, 6, L_, 16, g(L_)) for L_ in (20000, 40000, 60000, 100000)}
    check("T23 V-14 sweep_fit on the lead's FP8 sanity pool (P 460, F 6, prefill 4,000): 20k up to 32, 40k 16 (24 at the edge: 23),"
          " 60k 16, 100k KV-bound at 16", rows[20000]["resident_bound_095"] >= 32 and rows[40000]["resident_bound_095"] == 23
          and rows[60000]["resident_bound_095"] >= 16 and rows[100000]["fits"] is False, {k: v["resident_bound_095"] for k, v in rows.items()})
    check("T23 V-14 sweep_fit: pool / F unknown -> never fits", KL.sweep_fit(None, 6, 20000, 16, 0)["fits"] is False
          and KL.sweep_fit(460, None, 20000, 16, 0)["fits"] is False)
    check("T23 V-14 sweep_stop: preemption / CAPACITY_LIMITED / running below the level stop the band",
          KL.sweep_stop({"preemptions": 2, "running_max": 24}, 24) == "PREEMPTION"
          and KL.sweep_stop({"status": "CAPACITY_LIMITED", "running_max": 20}, 24) == "CAPACITY_LIMITED"
          and KL.sweep_stop({"status": "INCONCLUSIVE_X", "running_max": 20}, 24) == "RUNNING_BELOW_LEVEL"
          and KL.sweep_stop({"status": "STEADY", "running_max": 24, "preemptions": 0}, 24) is None)
    calls = []

    def win(port, s, band, n):
        calls.append((band, n))
        pre = 1 if (band, n) == ("sw20k", 24) else 0
        return {"status": "STEADY", "running_min": n, "running_max": n, "running_eq_level": True, "preemptions": pre, "admission": {"sec": 30.0}}
    ns = sweep_ns(calls, window=win)
    s = types.SimpleNamespace(port=1, alive=lambda: True)
    cap = {"capacity_ok": True, "pool_blocks": 460, "F": 6, "max_num_seqs": 32, "n64": 20}
    o = {}
    ns["kv_sweep"](o, s, cap, ("sw20k", "sw40k", "sw60k", "sw100k"))
    check("T23 V-14 sweep order 16 -> 24 -> 32 per band; 20k stops at the first preempting level (24), 40k 16 then KV-bound at 24,"
          " 60k 16, 100k KV-bound", calls == [("sw20k", 16), ("sw20k", 24), ("sw40k", 16), ("sw60k", 16)]
          and o["sweep"]["sw20k@24"]["stop"] == "PREEMPTION" and "sw20k@32" not in o["sweep"]
          and o["sweep"]["sw40k@24"]["status"] == "SKIPPED_KV_BOUND" and o["sweep"]["sw100k@16"]["status"] == "SKIPPED_KV_BOUND"
          and o["sweep_best"] == {"sw20k": 16, "sw40k": 16, "sw60k": 16}, (calls, o["sweep"]))
    r16 = o["sweep"]["sw20k@16"]
    check("T23 V-14 per level: tok/s, tick, cycle, acceptance per position, running == level, preemptions, kv usage, label, prefill s, fit",
          all(k in r16 for k in ("tok_s", "tick_ms", "cycle_ms", "accept_rate_per_position_incl_admission", "running_eq_level",
                                 "preemptions", "kv_usage_max", "label", "prefill_sec", "fit", "need_sec")) and r16["prefill_sec"] == 30.0, r16)
    check("T23 V-14 a band's distinct prompts are built lazily (no shared prefix) and released after the band",
          "sw20k" not in ns["PROMPTS"] and ns["sweep_prompts"]("sw20k", 3) == [[100] * 3, [101] * 3, [102] * 3])
    calls.clear()
    o = {}
    ns["kv_sweep"](o, s, dict(cap, max_num_seqs=16), ("sw20k",))
    check("T23 V-14 levels above max_num_seqs are not run (K2 at seqs 16 sweeps 16 only)", calls == [("sw20k", 16)]
          and o["sweep"]["sw20k@24"]["status"] == "SKIPPED_ABOVE_MAX_NUM_SEQS", o["sweep"])
    calls.clear()
    o, sk = {}, []
    ns["kv_sweep"](o, s, cap, ("sw20k",), gate=lambda need: need < 200, skipped=sk)
    check("T23 V-14 the K3 reserve gate (prefill-aware need: 16 x 20k / 4,000 + 90 + 180 = 350 s) -> SKIPPED_K3_RESERVE, recorded",
          calls == [] and o["sweep"]["sw20k@16"]["status"] == "SKIPPED_K3_RESERVE" and o["sweep"]["sw20k@16"]["need_sec"] == 350
          and sk == ["sw20k@16"], o["sweep"])
    ns["due"] = lambda sec: False
    o = {}
    ns["kv_sweep"](o, s, cap, ("sw100k",))
    check("T23 V-14 the boot body cap: a level never starts after it", o["sweep"]["sw100k@16"]["status"] in ("SKIPPED_BOOT_BUDGET", "SKIPPED_KV_BOUND"))
    check("T23 V-14 K3 launches with --max-num-seqs 32, captures to 64 (COMPILE_HI)", '"seqs_k3": 32,' in EXPORT_SRC
          and 'k1_boot("K3", f"k100_K3_fp8_mtp3_pin{int(round(kvb / 2 ** 30 * 100))}_r{i}", True, K100["seqs_k3"], COMPILE_HI,' in EXPORT_SRC)


# ======================================================================================== T25 rev 3: the K3_ONLY GPU v1 failure rows
# The K3_ONLY GPU run v1 (2026-11-01, foysalemonshanto; _scratch/e25_gpu_out/q27-xhigh-e25-klee100-load-k3only/k100/
# k100_results.json sections.K3.k100_K3_fp8_mtp3_pin491_r0.method_map): ranks [0, 1, 2, 3], partition 25/23, no rank error,
# every layer got ["HummingMoEMethod", None, None], the MTP map {} (the drafter was not found), verdict CONFIG_MISMATCH
# (config_mismatch ["GPU_METHOD_MAP_PASS"]). The raw rows were not saved; they are rebuilt here from that record.
K3V1_RESULTS = fixture("K100_TEST_K3V1", "_scratch/e25_gpu_out/q27-xhigh-e25-klee100-load-k3only/k100/k100_results.json")
K3V1_SUMMARY = {"ok": False, "layers_expected": 48, "layers_matching": 0, "missing": [], "unexpected": [], "wrong_pp": [],
                "conflicts": [], "mtp": {}, "mtp_ok": False, "need_mtp": True, "ranks": [0, 1, 2, 3], "ranks_ok": True,
                "errors": [], "partition": [25, 23]}


def k3v1_rows(mtp=None, change=None):
    """The v1 rows: per rank its PP stage's layers, each HummingMoEMethod with bits / group None (the layer names the
    worker extension reports for RoutedExperts); `mtp` = (method, bits, group) rows added on ranks 2-3; change = {L: v}."""
    res = []
    for r_ in range(4):
        rows = [{"name": f"language_model.model.layers.{L_}.mlp.experts", "module": "RoutedExperts", "method": "HummingMoEMethod",
                 "src": "model", "layer": L_, "mtp": False, "bits": None, "group": None}
                for L_ in (range(25) if r_ < 2 else range(25, 48))]
        for L_, v in (change or {}).items():
            for x in rows:
                if x["layer"] == L_:
                    x.update(method=v[0], bits=v[1], group=v[2])
        if mtp and r_ >= 2:
            rows.append({"name": "mtp.layers.48.mlp.experts", "module": "RoutedExperts", "method": mtp[0], "src": "drafter",
                         "layer": 48, "mtp": True, "bits": mtp[1], "group": mtp[2]})
        res.append({"rank": r_, "experts": rows, "errors": []})
    return res


def t25():
    exp = KL.k100_expected_map()
    v1 = k3v1_rows()
    # the rebuilt rows reproduce the v1 run's recorded verdict under the v1 rule (class + bits compared, MTP required)
    want_bad = [{"layer": L_, "got": ["HummingMoEMethod", None, None], "want": ["HummingMoEMethod", *exp[L_]]} for L_ in range(12)]
    if K3V1_RESULTS.is_file():
        rec = json.loads(K3V1_RESULTS.read_text(encoding="utf-8"))["sections"]["K3"]["k100_K3_fp8_mtp3_pin491_r0"]
        mm = rec["method_map"]
        check("T25 rev 3 fixture = the v1 run's recorded method map (ranks, partition, no errors, MTP {}, the first 12 bad rows)",
              {k: mm.get(k) for k in K3V1_SUMMARY} == K3V1_SUMMARY and mm["bad"] == want_bad
              and rec["config_mismatch"] == ["GPU_METHOD_MAP_PASS"] and rec["status"] == "CONFIG_MISMATCH", mm.get("why"))
    else:
        skip("T25 fixture vs the v1 run's k100_results.json", f"fixture not present: {K3V1_RESULTS}")
    r = KL.method_map_check(v1, need_mtp=True)
    check("T25 rev 3 the v1 rows replayed: the bits no longer fail (routed UNREADABLE, all 48 classes right); the missing MTP "
          "drafter alone keeps it GPU_METHOD_MAP_FAIL",
          r["verdict"] == "GPU_METHOD_MAP_FAIL" and r["gate_ok"] is False and r["why"] == "MTP experts missing"
          and r["bits_readable"]["routed"] == "UNREADABLE" and r["layers_class_only"] == 48 and r["bad"] == []
          and {k: r[k] for k in ("missing", "unexpected", "wrong_pp", "conflicts", "mtp", "ranks", "ranks_ok", "errors", "partition")}
          == {k: K3V1_SUMMARY[k] for k in ("missing", "unexpected", "wrong_pp", "conflicts", "mtp", "ranks", "ranks_ok", "errors", "partition")},
          r.get("why"))
    for mtp, fams in ((("AutoGPTQMoEMethod", 4, 128), ["routed"]), (("AutoGPTQMoEMethod", None, None), ["routed", "mtp"])):
        r = KL.method_map_check(k3v1_rows(mtp=mtp), need_mtp=True)
        check(f"T25 rev 3 the v1 rows + the MTP drafter rows {list(mtp)} -> METHOD_CLASS_ONLY (gate_ok True, ok False), not CONFIG_MISMATCH",
              r["verdict"] == "METHOD_CLASS_ONLY" and r["gate_ok"] is True and r["ok"] is False and r["why"] is None
              and r["class_only_families"] == fams and r["mtp_ok"] is True and "METHOD_CLASS_ONLY" in r["note"], (r.get("why"), r.get("bits_readable")))
    r = KL.method_map_check(k3v1_rows(), need_mtp=False)
    check("T25 rev 3 the v1 routed rows with MTP off -> METHOD_CLASS_ONLY", r["verdict"] == "METHOD_CLASS_ONLY" and r["gate_ok"] is True, r.get("why"))
    gm = ("AutoGPTQMoEMethod", 4, 128)
    for lab_, ch, mtp, key in (("a routed layer on UnquantizedFusedMoEMethod (bits unreadable)", {20: ("UnquantizedFusedMoEMethod", None, None)}, gm, "differ"),
                               ("the MTP experts on HummingMoEMethod (bits unreadable)", None, ("HummingMoEMethod", None, None), "MTP experts"),
                               ("the MTP experts read W3 g128", None, ("AutoGPTQMoEMethod", 3, 128), "MTP experts"),
                               ("bits read on layer 0 only (3 / 128), None elsewhere", {0: ("HummingMoEMethod", 3, 128)}, gm, "BITS_PARTIALLY_READ"),
                               ("a read layer 5 at W2 g64 on otherwise unreadable rows", {5: ("HummingMoEMethod", 2, 64)}, gm, "BITS_PARTIALLY_READ")):
        r = KL.method_map_check(k3v1_rows(mtp=mtp, change=ch), need_mtp=True)
        check(f"T25 rev 3 a real mismatch stays fatal: {lab_} -> GPU_METHOD_MAP_FAIL", r["verdict"] == "GPU_METHOD_MAP_FAIL"
              and r["gate_ok"] is False and key in (r["why"] or ""), r.get("why"))
    full = k3v1_rows(mtp=gm, change={L_: ("HummingMoEMethod", *exp[L_]) for L_ in range(48)})
    r = KL.method_map_check(full, need_mtp=True)
    check("T25 rev 3 the same rows with bits read and right -> GPU_METHOD_MAP_PASS", r["verdict"] == "GPU_METHOD_MAP_PASS"
          and r["ok"] is True and r["layers_matching"] == 48 and r["bits_readable"] == {"routed": "READ", "mtp": "READ"}, r.get("why"))
    wrong = k3v1_rows(mtp=gm, change={L_: ("HummingMoEMethod", *(exp[L_] if L_ != 30 else (3, 128))) for L_ in range(48)})
    r = KL.method_map_check(wrong, need_mtp=True)
    check("T25 rev 3 bits read, layer 30 W3 g128 (checkpoint W2 g64) -> GPU_METHOD_MAP_FAIL", r["verdict"] == "GPU_METHOD_MAP_FAIL"
          and r["bad"] == [{"layer": 30, "got": ["HummingMoEMethod", 3, 128], "want": ["HummingMoEMethod", 2, 64]}], r.get("why"))
    # k100_facts on the K3 boot key: CLASS_ONLY -> config_ok True (no CONFIG_MISMATCH), the verdict in the read-back, raw rows kept
    log_ = ("INFO non-default args: {'model': '/ck', 'kv_cache_dtype': 'bfloat16', 'max_num_seqs': 8, 'kv_cache_memory_bytes': 123, "
            "'enable_prefix_caching': True}\nINFO Initializing a V1 LLM engine with config: model='/ck', speculative_config=None, "
            "dtype=torch.bfloat16, quantization=inc, kv_cache_dtype=bfloat16\nWARNING needs speculative decoding; num_speculative_tokens=0\n")
    fns = {"re": re, "json": json, "KL": KL, "CKPT": "/ck", "CAPS_LIST": {"C": [1, 2]}, "KV_FP8_DTYPE": "fp8_e4m3",
           "server_caps": lambda t, argv: {"capture_sizes": [1, 2]}, "pool_facts": lambda t: {}, "graph_facts": lambda t: {},
           "rpc_cache_info": lambda p: {}}
    load(["flags", "_pick", "k100_facts"], fns)
    s = types.SimpleNamespace(log=lambda: log_, argv=["vllm", "serve", "/ck", "--kv-cache-memory-bytes", "123"], port=1)
    for rows, want_ok, want_v in ((k3v1_rows(), True, "METHOD_CLASS_ONLY"),
                                  (k3v1_rows(change={3: ("UnquantizedFusedMoEMethod", None, None)}), False, "GPU_METHOD_MAP_FAIL")):
        raw = {"results": rows}
        fns["rpc"] = lambda port, method, raw=raw: raw if method == "k100_method_map" else {"error": "x"}
        o = fns["k100_facts"](s, {"key": "K3", "seqs": 8, "captures": "C", "mtp": False, "kv": "bf16"}, {})
        check(f"T25 rev 3 k100_facts K3 on the v1 rows ({want_v}): config_ok {want_ok}, read-back verdict, raw per-rank rows kept",
              o["config_ok"] is want_ok and o["readback"]["checks"]["GPU_METHOD_MAP_PASS"] is want_ok
              and o["readback"]["method_map_verdict"] == want_v and o["method_map_raw"] is raw
              and len(o["method_map_raw"]["results"]) == 4 and o["method_map_raw"]["results"][0]["experts"][0]["layer"] == 0,
              (o["readback"]["checks"], o["method_map"].get("why")))
    # labels and predicates: METHOD_CLASS_ONLY is a tag, never CONFIG_MISMATCH
    co = KL.method_map_check(k3v1_rows(mtp=gm), need_mtp=True)
    clean = {"status": "DONE", "correctness": "CLEAN", "config_ok": True}
    k3c = dict(clean, method_map=co)
    k3r = {"reading": "LOWBIT_TF_CLOSE", "complete": True}
    r_ = KL.final_label({"INPUTS": (True, None)}, None, None, {"reading": "TF_NOT_MEASURED"}, k3c, k3r, profile="K3_ONLY")
    check("T25 rev 3 K3_ONLY label with a class-only map: K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE + METHOD_CLASS_ONLY | BF16_NOT_IN_PROFILE",
          r_["label"] == "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE + METHOD_CLASS_ONLY | BF16_NOT_IN_PROFILE" and "CONFIG_MISMATCH" not in r_["label"], r_)
    r_ = KL.final_label({"INPUTS": (True, None)}, dict(clean, method_map=co), clean, {"reading": "LOWBIT_TF_DIVERGENT"})
    check("T25 rev 3 BF16 label with K class-only: GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT + METHOD_CLASS_ONLY",
          r_["label"] == "GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT + METHOD_CLASS_ONLY", r_)
    r_ = KL.final_label({"INPUTS": (True, None)}, clean, clean, {"reading": "LOWBIT_TF_DIVERGENT"}, dict(clean, method_map=KL.method_map_check(full, True)), k3r)
    check("T25 rev 3 a PASS map adds no tag", r_["label"] == "K3_FP8_LOAD_PASS + LOWBIT_TF_CLOSE | GPU_LOAD_PASS + LOWBIT_TF_DIVERGENT", r_)
    pr = KL.boot_predicates(k3c, k3r, fp8=True)
    check("T25 rev 3 predicates: method_map_pass False, method_map_verdict METHOD_CLASS_ONLY",
          pr["method_map_pass"] is False and pr["method_map_verdict"] == "METHOD_CLASS_ONLY", pr)
    ns = {}
    load(["method_map_summary"], ns)
    sm = ns["method_map_summary"]({"method_map": co})
    check("T25 rev 3 the verdict summary carries verdict / gate_ok / bits_readable / class-only layers",
          sm["verdict"] == "METHOD_CLASS_ONLY" and sm["gate_ok"] is True and sm["bits_readable"] == {"routed": "UNREADABLE", "mtp": "READ"}
          and sm["layers_class_only"] == 48, sm)
    # the worker extension source: every attribute path the probe reads (the Merlin source names) and no v1 drafter-only path
    ext = export_value("NB1_EXT_SRC")
    check("T25 rev 3 the embedded extension reads weight_schema / b_dtype.num_bits / weight_scale_group_size / weight_schemas / "
          "the module's quant_config.config_parser / get_draft_model / speculator; method_map_raw saved by k100_facts",
          all(x in ext for x in ('getattr(qm, "weight_schema"', '"b_dtype"', '"weight_scale_group_size"', '"weight_schemas"',
                                 'getattr(getattr(mod, "quant_config", None), "config_parser"', '"get_draft_model"', '"speculator"'))
          and 'o["method_map_raw"] = raw_mm = rpc(s.port, "k100_method_map")' in EXPORT_SRC)


for n, f in (("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5), ("T6", t6), ("T7", t7), ("T8", t8), ("T9", t9),
             ("T10", t10), ("T11", t11), ("T12", t12), ("T13", t13), ("T14", t14), ("T15", t15), ("T16", t16), ("T17", t17),
             ("T18", t18), ("T19", t19), ("T20", t20), ("T21", t21), ("T22", t22), ("T23", t23), ("T24", t24), ("T25", t25)):
    run(n, f)
shutil.rmtree(TMP, ignore_errors=True)
ok = sum(1 for _, v in RES if v)
print(f"{ok}/{len(RES)} PASS", flush=True)
print(f"{len(SKIPS)} skipped" + (": " + "; ".join(f"{n} ({w})" for n, w in SKIPS) if SKIPS else ""), flush=True)
sys.exit(0 if ok == len(RES) else 1)
